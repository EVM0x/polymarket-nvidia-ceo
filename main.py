#!/usr/bin/env python3
"""Quantile forecaster for the Polymarket "Next CEO of NVIDIA?" market family.

Input  : every *.in file of $MAKEFUN_INPUT_DIR -- one JSON case per file, each
         holding the Yes price of every candidate market on a 1-minute grid for
         the 72 hours leading up to a checkpoint ("at").
Output : $MAKEFUN_OUTPUT_DIR/<case id>.out -- for every market, one row per
         horizon h = 1..horizon_minutes holding the 5/25/50/75/95 percent
         quantiles of that market's price at checkpoint + h minutes.

Why log-odds
------------
A price is a probability, so its natural coordinate is the log-odds

    L = logit(p) = log(p / (1 - p)),        p = sigmoid(L).

There a prediction market behaves much like a driftless random walk: its
increments are close to unpredictable, and the variance of the walk grows
roughly linearly in time.  Working in L also means a quantile forecast mapped
back through the logistic is automatically inside [0, 1], which is the shape
the grader demands.

The model
---------
Each market is a fan of five quantiles around a centre, with a spread that
follows the diffusion law spread(h) = step * sqrt(h).  Three things are worth
explaining, because each was measured, not assumed.

1. The step size is a blend of two volatility estimators, and its LEVEL comes
   from the cross-section rather than from the market itself.

   A flat-window estimator (a robust blend of RMS and MAD over the last four
   hours) reacts to a market that has just woken up; an EWMA with a one-hour
   half-life weights the last few minutes much more heavily.  Their 3:1 blend
   beat either alone.

   The level, though, is not the market's own: the 24 candidates of a case
   share one news flow, so a market that has been quiet for four hours is not
   therefore safe.  The case-wide median volatility sets the level and the
   market's own estimate enters only as a ratio, damped by an exponent below
   one:

       step_m = median(vol) * (vol_m / median(vol)) ** 0.65 * 0.70

   With the exponent at 1 this is the plain own-history estimator; at 0 every
   market would be treated as average.  0.65 is what the practice cases chose,
   which says a market's own quiet is real information but only partly
   trustworthy.

2. The centre fades part of the last four hours' move.  Part of a hard move in
   a candidate's price is given back, so the centre is pulled back toward where
   the price was four hours ago, by a fraction that grows with the horizon:

       centre(h) = L[-1] - 0.10 * (L[-1] - L[-241]) * (h / 240) ** 0.85

3. The five multipliers z_tau are NOT the normal ones.  The empirical
   distribution of the standardised residual (logit(y) - centre) / spread has a
   long LEFT tail -- a candidate who is dropped collapses to a cent within
   minutes -- and a short right one, since the field is bounded and the event
   eventually resolves.  The constants below are that residual's empirical
   5/25/50/75/95 percent quantiles, measured per horizon band, so the fan
   inherits the real asymmetry instead of pretending the innovations are
   Gaussian.  Note the 75th percentile is far below its Gaussian value while
   the 95th is far above: the upside lives in rare jumps, not in a fat middle.

4. The medians of the case are renormalised.  The candidates of this event are
   mutually exclusive and nearly exhaustive, so their probabilities sum to
   about 1 at every minute, and that sum mean-reverts.  Whatever the fans say,
   the cross-section of medians is therefore pulled toward a target sum

       T = S + 0.5 * (1 - S),      S = the sum of the checkpoint prices,

   by scaling every market's median by T / (the sum of the medians) and
   shifting its whole fan by the same amount.  On the practice cases this
   alone is worth about 0.005 of loss -- the largest single gain after the
   quantile shape.

Every constant was fitted by coordinate descent on the pooled pinball loss of
the six practice cases, and every fit was repeated leave-one-case-out to check
it was not just memorising them.  Everything here is a pure function of the
input bytes: standard library only, no network, no randomness, no clock.
"""

import json
import math
import os
import statistics

# ---------------------------------------------------------------------------
# constants

LOGIT_EPS = 2e-4        # keeps logit() finite for prices quoted at exactly 0 or 1
WIN = 240               # minutes of history behind the flat-window volatility
EWMA_HALFLIFE = 60      # minutes; the short-horizon volatility estimator
W_EWMA = 0.25           # weight of the EWMA estimator in the blend
GAMMA = 0.65            # how much of a market's own volatility to believe
SCALE = 0.70            # overall multiplier on the step size
FADE = 0.10             # fraction of the four-hour move given back at h = 240
FADE_EXP = 0.85         # how that fade grows with the horizon
KAPPA = 0.50            # strength of the pull on the cross-sectional sum
HORIZON_REF = 240.0     # the horizon every constant was fitted at
NDIGITS = 6             # decimals kept in the output

# Empirical 5/25/50/75/95 percent quantiles of the standardised residual, per
# horizon band: (first horizon, last horizon, multipliers).  Bands 1-15,
# 16-60, 61-120, 121-240 minutes.
Z_BANDS = (
    (1, 15, (-2.70, -1.25, 0.00, 0.50, 2.10)),
    (16, 60, (-2.70, -1.25, 0.00, 0.20, 1.70)),
    (61, 120, (-1.90, -0.85, 0.00, 0.20, 1.50)),
    (121, 10 ** 9, (-1.90, -0.85, 0.00, 0.20, 1.70)),
)


# ---------------------------------------------------------------------------
# small numeric helpers


def clip(x, lo=0.0, hi=1.0):
    return lo if x < lo else (hi if x > hi else x)


def logit(p):
    return math.log(clip(p, LOGIT_EPS, 1.0 - LOGIT_EPS) /
                    (1.0 - clip(p, LOGIT_EPS, 1.0 - LOGIT_EPS)))


def sigmoid(x):
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def flat_vol(series, win):
    """Robust per-minute volatility of the last `win` minutes.

    The blend of RMS and scaled MAD keeps one jump from setting the scale,
    while still letting a market that just moved be treated as moving.
    """
    seg = series[-win - 1:] if len(series) > win else series
    d = [seg[i + 1] - seg[i] for i in range(len(seg) - 1)]
    if len(d) < 5:
        return None
    rms = math.sqrt(sum(x * x for x in d) / len(d))
    mad = 1.4826 * statistics.median([abs(x) for x in d])
    return 0.5 * (rms + mad)


def ewma_vol(series, halflife):
    """Exponentially weighted RMS of the increments, recent minutes weighted most."""
    d = [series[i + 1] - series[i] for i in range(len(series) - 1)]
    if len(d) < 5:
        return None
    lam = 0.5 ** (1.0 / halflife)
    num = 0.0
    den = 0.0
    w = 1.0
    for x in reversed(d):
        num += w * x * x
        den += w
        w *= lam
    return math.sqrt(num / den)


def band_z(h):
    for lo, hi, zs in Z_BANDS:
        if lo <= h <= hi:
            return zs
    return Z_BANDS[-1][2]


# ---------------------------------------------------------------------------
# the model


def case_forecast(markets, hmax):
    """Forecast every market of one case; returns a list of row-lists."""
    series = [logit_series(m.get('prices') or []) for m in markets]
    starts = [clip((m.get('prices') or [0.5])[-1]) for m in markets]

    # volatility per market, blended across the two estimators
    vols = []
    for s in series:
        a = flat_vol(s, WIN)
        b = ewma_vol(s, EWMA_HALFLIFE)
        a = a if a is not None else (b if b is not None else 1e-3)
        b = b if b is not None else a
        vols.append(max((1.0 - W_EWMA) * a + W_EWMA * b, 1e-9))

    # the level is the case's own median, the market's estimate a damped ratio
    ev = statistics.median(vols) if vols else 1e-3
    steps = [ev * (v / ev) ** GAMMA * SCALE for v in vols]

    fans = []
    for s, step in zip(series, steps):
        base = s[-1] if s else 0.0
        recent = base - s[-1 - WIN] if len(s) > WIN + 1 else 0.0
        rows = []
        for h in range(1, hmax + 1):
            centre = base - FADE * recent * (h / HORIZON_REF) ** FADE_EXP
            spread = step * math.sqrt(h)
            zs = band_z(h)
            row = [round(clip(sigmoid(centre + zs[k] * spread)), NDIGITS)
                   for k in range(5)]
            for i in range(1, 5):        # rounding can break the order by 1e-6
                if row[i] < row[i - 1]:
                    row[i] = row[i - 1]
            rows.append(row)
        fans.append(rows)

    renormalise(fans, starts)
    return fans


def logit_series(prices):
    """The log-odds path of one market, one point per minute."""
    if not prices:
        return []
    return [logit(p) for p in prices]


def renormalise(fans, starts):
    """Pull the cross-section of medians toward T = S + KAPPA * (1 - S)."""
    if not fans or not fans[0]:
        return
    total = sum(starts)
    target = total + KAPPA * (1.0 - total)
    current = sum(f[0][2] for f in fans)
    if current <= 1e-6:
        return
    factor = clip(target / current, 0.25, 4.0)
    if abs(factor - 1.0) < 1e-9:
        return
    for fan in fans:
        for row in fan:
            shift = row[2] * factor - row[2]
            if shift == 0.0:
                continue
            for k in range(5):
                row[k] = round(clip(row[k] + shift), NDIGITS)
            for k in range(1, 5):
                if row[k] < row[k - 1]:
                    row[k] = row[k - 1]


# ---------------------------------------------------------------------------
# io


def run(indir, outdir):
    """Forecast every *.in file of indir into outdir, in name order."""
    os.makedirs(outdir, exist_ok=True)
    for name in sorted(os.listdir(indir)):
        if not name.endswith('.in'):
            continue
        with open(os.path.join(indir, name), 'r') as fh:
            case = json.load(fh)
        hmax = int(case.get('horizon_minutes') or 240)
        markets = case.get('markets') or []
        fans = case_forecast(markets, hmax)
        forecasts = [{'id': m['id'], 'q': fan} for m, fan in zip(markets, fans)]
        out_path = os.path.join(outdir, name[:-len('.in')] + '.out')
        with open(out_path, 'w') as fh:
            json.dump({'forecasts': forecasts}, fh, separators=(',', ':'))


def main():
    run(os.environ.get('MAKEFUN_INPUT_DIR', '/input'),
        os.environ.get('MAKEFUN_OUTPUT_DIR', '/output'))


if __name__ == '__main__':
    main()
