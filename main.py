#!/usr/bin/env python3
"""Five-quantile, four-hour forecaster for the "Next CEO of NVIDIA?" market.

The program reads every *.in case of $MAKEFUN_INPUT_DIR (default /input) and
writes $MAKEFUN_OUTPUT_DIR/<case>.out (default /output) holding, for each
market of the case, 240 rows of [q05, q25, q50, q75, q95] for the minutes
at+1 .. at+240.  Standard library only; no clock, no network, no randomness,
so two runs on the same bytes produce the same bytes.

Coordinate
----------
A price is a probability, so it is modelled in log-odds

    L = logit(p) = log(p / (1 - p)),        p = sigmoid(L),

with p clamped to [2e-4, 1-2e-4] before the transform.  In L a prediction
market looks close to a driftless random walk: increments are nearly
unpredictable and their cumulative variance grows about linearly in time.
That is what licenses the sqrt(h) fan below.  Mapping quantiles back through
the logistic also lands them inside [0, 1] automatically, so the grader's
range check never depends on a clamp at the edges.

Volatility of one minute of L
-----------------------------
Two estimators answer two different questions, and the blend keeps both:

  flat  -- RMS and scaled MAD of the increments of the last 240 minutes,
           0.5 * (RMS + 1.4826 * median|d|).  The MAD half stops one jump
           from setting the scale for the next four hours; the RMS half keeps
           a market that genuinely woke up from being read as asleep.
  ewma  -- exponentially weighted RMS over the whole series, half-life 60
           minutes, newest increment at weight 1.  This is the "now" estimate:
           it forgets old regimes, at the price of being noisy.

vol_m = 0.75 * flat + 0.25 * ewma.  The flat window is the stable backbone;
the EWMA is a small, fast correction.  Both are per market because how much a
name moves is a property of the name.

Why the level of the step is cross-sectional
--------------------------------------------
The 24 candidates of a case are one event: the news that reprices the field
reprices all of it.  A market that has been quiet for four hours is therefore
not safe -- it is a market whose turn has not come.  So the case-wide median
volatility sets the level, and the market's own estimate only modulates it:

    ev      = median(vol over the case's markets)
    step_m  = ev * (vol_m / ev) ** 0.65 * 0.70

The exponent 0.65 says own-history is real information but only partly
trustworthy (1 would be pure per-market, 0 pure pooled).  The 0.70 is the
calibration of the fan's overall width: measured, not derived.

Centre and spread at horizon h
------------------------------
Spikes retrace, so part of the last four hours' move is given back, more so
further out:

    recent    = L[-1] - L[-241]        (0 when the history is shorter)
    centre(h) = L[-1] - 0.10 * recent * (h / 240) ** 0.85
    spread(h) = step_m * sqrt(h)

The index matters: 241 points delimit 240 increments ending at the checkpoint.
sqrt(h) is the diffusion law, the same shape the house baseline uses, only
with a measured step instead of a fixed 0.0025.

Why the multipliers are not the normal ones
-------------------------------------------
Standardising the realised outcome, (logit(y) - centre) / spread, gives a
residual with a long left tail and a short right one: a dropped candidate
collapses to a cent within minutes, while the upside is capped by the field
being finite and the event eventually resolving.  The five multipliers are the
empirical 5/25/50/75/95 percentiles of that residual, per horizon band, and
the asymmetry is the point -- the 75th percentile sits at 0.20-0.50 where a
Gaussian would put 0.67, while the 95th sits at 1.50-2.10 where a Gaussian
would put 1.64.  The upside lives in rare jumps, not in a fat middle.  The
bands let the left tail shorten as the horizon grows.

Why the cross-section is renormalised
-------------------------------------
The candidates are mutually exclusive and nearly exhaustive, so their prices
sum to about 1 at every minute, and that sum mean-reverts.  The fans are built
market by market and do not know this, and the fade in the centre moves each
market's median by a different amount.  So at every horizon the medians are
pulled toward

    T = S + 0.5 * (1 - S),            S = the sum of the checkpoint prices,

by scaling each median by T / (sum of the medians) and shifting that market's
whole row by the same amount, which preserves the fan's shape while moving
its centre.  The half-step toward 1 (rather than all the way) reflects that
the sum mean-reverts but does not sit exactly at 1.  Degenerate rows are left
alone: a sum of medians at or below 1e-6, or a scale outside [0.25, 4.0], is a
sign the inputs are pathological, not a signal to trust.
"""

import json
import math
import os
import statistics

# --- tuning constants ------------------------------------------------------
LOGIT_EPS = 2e-4       # keeps logit finite for prices quoted at 0 or 1
WIN = 240              # minutes of history behind the flat-window estimate
EWMA_HALFLIFE = 60.0   # minutes; half-life of the fast estimator
W_EWMA = 0.25          # weight of the EWMA in the blend
GAMMA = 0.65           # how far a market's own volatility is believed
SCALE = 0.70           # overall width of the fan
FADE = 0.10            # fraction of the four-hour move retraced at h = 240
FADE_EXP = 0.85        # how the retracement grows with the horizon
KAPPA = 0.50           # pull of the cross-sectional sum toward 1
H_REF = 240.0          # horizon the fade is calibrated at
NDIGITS = 6            # decimals kept in the output
MIN_VOL = 1e-3         # last-resort volatility for a series with no increments
VOL_FLOOR = 1e-9       # keeps (vol/ev)**GAMMA finite
SUM_EPS = 1e-6         # below this a row's medians carry no information
FACTOR_LO, FACTOR_HI = 0.25, 4.0   # sanity bounds on the renormalisation

# Empirical 5/25/50/75/95 percentiles of the standardised residual, by horizon
# band: (first horizon, last horizon, multipliers).
Z_BANDS = (
    (1, 15, (-2.70, -1.25, 0.00, 0.50, 2.10)),
    (16, 60, (-2.70, -1.25, 0.00, 0.20, 1.70)),
    (61, 120, (-1.90, -0.85, 0.00, 0.20, 1.50)),
    (121, 240, (-1.90, -0.85, 0.00, 0.20, 1.70)),
)


# --- numerics --------------------------------------------------------------

def clamp(x, lo=0.0, hi=1.0):
    """Clamp x to [lo, hi]."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def logit(p):
    """log(p/(1-p)) with p held away from the two absorbing points."""
    p = clamp(p, LOGIT_EPS, 1.0 - LOGIT_EPS)
    return math.log(p / (1.0 - p))


def sigmoid(x):
    """1/(1+exp(-x)), split so neither branch can overflow."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def flat_vol(series):
    """Robust per-minute vol over the last WIN increments; None if too few."""
    seg = series[-WIN - 1:] if len(series) > WIN else series
    d = [seg[i + 1] - seg[i] for i in range(len(seg) - 1)]
    if len(d) < 5:
        return None
    rms = math.sqrt(sum(x * x for x in d) / len(d))
    mad = 1.4826 * statistics.median([abs(x) for x in d])
    return 0.5 * (rms + mad)


def ewma_vol(series):
    """Exponentially weighted RMS of all increments, newest at weight 1."""
    d = [series[i + 1] - series[i] for i in range(len(series) - 1)]
    if not d:
        return None
    lam = 0.5 ** (1.0 / EWMA_HALFLIFE)
    num = 0.0
    den = 0.0
    w = 1.0
    for x in reversed(d):
        num += w * x * x
        den += w
        w *= lam
    return math.sqrt(num / den)


def z_row(h):
    """The five residual multipliers for horizon h."""
    for lo, hi, zs in Z_BANDS:
        if lo <= h <= hi:
            return zs
    return Z_BANDS[-1][2]


def monotone(row):
    """Raise any value below its predecessor (rounding can break the order)."""
    for i in range(1, len(row)):
        if row[i] < row[i - 1]:
            row[i] = row[i - 1]


# --- model -----------------------------------------------------------------

def forecast_case(markets, hmax):
    """Return one list of hmax rows of five quantiles per market of a case."""
    paths = [[logit(p) for p in (m.get('prices') or [])] for m in markets]
    last = [(m.get('prices') or [0.5])[-1] for m in markets]

    # Per-market volatility, then the case-wide level of the step.
    vols = []
    for path in paths:
        flat = flat_vol(path)
        fast = ewma_vol(path)
        if flat is None:
            flat = fast if fast is not None else MIN_VOL
        if fast is None:
            fast = flat
        vols.append(max((1.0 - W_EWMA) * flat + W_EWMA * fast, VOL_FLOOR))

    ev = statistics.median(vols) if vols else MIN_VOL
    if ev <= 0.0:
        ev = MIN_VOL
    steps = [ev * (v / ev) ** GAMMA * SCALE for v in vols]

    # Fan each market: faded centre, sqrt(h) spread, asymmetric multipliers.
    fans = []
    for path, step in zip(paths, steps):
        base = path[-1] if path else 0.0
        recent = base - path[-1 - WIN] if len(path) >= WIN + 1 else 0.0
        rows = []
        for h in range(1, hmax + 1):
            centre = base - FADE * recent * (h / H_REF) ** FADE_EXP
            spread = step * math.sqrt(h)
            row = [round(clamp(sigmoid(centre + z * spread)), NDIGITS)
                   for z in z_row(h)]
            monotone(row)
            rows.append(row)
        fans.append(rows)

    renormalise(fans, last)
    return fans


def renormalise(fans, last):
    """Pull each horizon's cross-section of medians toward T = S + KAPPA*(1-S)."""
    if not fans or not fans[0]:
        return
    s_total = sum(last)
    target = s_total + KAPPA * (1.0 - s_total)

    for h in range(len(fans[0])):
        current = sum(fan[h][2] for fan in fans)
        if current <= SUM_EPS:
            continue
        factor = clamp(target / current, FACTOR_LO, FACTOR_HI)
        if abs(factor - 1.0) < 1e-12:
            continue
        for fan in fans:
            row = fan[h]
            shift = row[2] * factor - row[2]
            if shift == 0.0:
                continue
            for k in range(5):
                row[k] = round(clamp(row[k] + shift), NDIGITS)
            monotone(row)


# --- io --------------------------------------------------------------------

def run(indir, outdir):
    """Forecast every *.in case of indir into outdir, in name order."""
    os.makedirs(outdir, exist_ok=True)
    for name in sorted(os.listdir(indir)):
        if not name.endswith('.in'):
            continue
        with open(os.path.join(indir, name), 'r') as fh:
            case = json.load(fh)
        hmax = int(case.get('horizon_minutes') or 240)
        markets = case.get('markets') or []
        fans = forecast_case(markets, hmax)
        forecasts = [{'id': m['id'], 'q': fan} for m, fan in zip(markets, fans)]
        with open(os.path.join(outdir, name[:-3] + '.out'), 'w') as fh:
            json.dump({'forecasts': forecasts}, fh, separators=(',', ':'))


def main():
    run(os.environ.get('MAKEFUN_INPUT_DIR', '/input'),
        os.environ.get('MAKEFUN_OUTPUT_DIR', '/output'))


if __name__ == '__main__':
    main()
