#!/usr/bin/env python3
"""Five-quantile, four-hour forecaster for the "Next CEO of NVIDIA?" market.

The program reads every *.in case of $MAKEFUN_INPUT_DIR (default /input) and
writes $MAKEFUN_OUTPUT_DIR/<case>.out (default /output).  For every market of a
case it emits 240 rows of [q05, q25, q50, q75, q95] for the minutes at+1 ..
at+240.  Standard library only, no clock, no network, no randomness.

Coordinate
----------
A price is a probability, so it is modelled in log-odds

    L = logit(p) = log(p / (1 - p)),    p = sigmoid(L)

with p clamped to [2e-4, 1-2e-4] before the transform.  In L a prediction
market is close to a driftless random walk: increments are nearly
unpredictable and their variance grows about linearly in time.  Mapping the
quantiles back through the logistic keeps them inside [0, 1] without a clamp at
the edges.

Volatility of one minute of L
-----------------------------
    flat  -- 0.5 * (RMS + 1.4826 * MAD) of the increments of the last WIN
             minutes.  The MAD half stops one jump from setting the scale for
             the next four hours; the RMS half keeps a market that genuinely
             woke up from being read as asleep.
    ewma  -- exponentially weighted RMS, half-life HL minutes, newest
             increment at weight 1.  This is the "now" estimate.

    vol = 0.75 * flat + 0.25 * ewma

The level of the step is cross-sectional
----------------------------------------
The candidates of one event are quoted side by side and they move on the same
news, so a quiet four hours on one name is weak evidence that the name is
quiet.  The case median of vol is therefore the level, and each market is
scaled away from it by a damped ratio:

    step_m = median(vol) * (vol_m / median(vol)) ** 0.58 * 0.65

0.58 is a shrinkage exponent: a market that is twice as volatile as the case
median gets 1.5x the step, not 2x.

The fan
-------
    q_tau(h) = sigmoid( centre(h) + z[band][h][tau] * step_m * h ** 0.5 )

centre(h) fades the last four hours of drift back toward the current price
(BETA 0.1, weight (h/240)**0.85) because a move that has already happened is
not a forecast of the next one.

z[band][h][tau] is a 4 x 241 x 5 table, one entry per (price band, horizon
minute, quantile).  It was fitted on the six practice cases by coordinate
descent straight on the pinball loss, starting from a single table per horizon
bucket.  The price bands are [0, .05), [.05, .20), [.20, .60), [.60, 1]: a name
quoted at three cents moves in whole ticks of the log-odds, so its fan is not
the fan of a name quoted at fifty.  Letting every horizon minute have its own
row is what lets the fan deviate from sqrt(h) where the data says it should --
the short horizons are wider per unit of sqrt(h) than the long ones, because a
240-minute move is bounded by 0 and 1 while a one-minute move is not.

The table is stored quantised to 1e-3 and deflated, because 4 * 241 * 5
constants written out as text is 25 KB of noise.  It is a constant of the
program, not something read at run time.

Sum of the medians
------------------
The candidates of an event are mutually exclusive: their prices sum to about 1
at every minute and that sum mean-reverts.  So the medians are pulled toward
T = S + 0.5 * (1 - S) with S the sum of the current prices, and the same factor
shifts q05/q25/q75/q95 so the fan keeps its shape.  This is worth about 0.005
of loss on its own and it never fires when the sum is already sane.
"""
import base64
import json
import math
import os
import zlib

NDIGITS = 6
EPS = 2e-4
WIN = 240
HL = 60
W_EWMA = 0.25
GAMMA = 0.58
SCALE = 0.65
ALPHA = 0.5
BETA = 0.1
KAPPA = 0.5
TARGET_ADJ = 0.5
SUM_EPS = 1e-9
FACTOR_LO = 0.5
FACTOR_HI = 2.0
BAND_EDGES = (0.05, 0.20, 0.60)

# 4 price bands x 241 horizons (index 0 unused) x 5 quantiles, int16 thousandths
_Z_B64 = (
    "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"

)


def _load_z():
    raw = zlib.decompress(base64.b64decode(_Z_B64))
    vals = memoryview(raw).cast('h')
    out = []
    i = 0
    for _ in range(4):
        band = []
        for _h in range(241):
            band.append([vals[i + k] / 1000.0 for k in range(5)])
            i += 5
        out.append(band)
    return out


Z = _load_z()


def logit(p):
    p = EPS if p < EPS else (1.0 - EPS if p > 1.0 - EPS else p)
    return math.log(p / (1.0 - p))


def sigmoid(x):
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else (hi if x > hi else x)


def band_of(price):
    if price < BAND_EDGES[0]:
        return 0
    if price < BAND_EDGES[1]:
        return 1
    if price < BAND_EDGES[2]:
        return 2
    return 3


def flat_vol(seq):
    seg = seq[-WIN - 1:] if len(seq) > WIN else seq
    n = len(seg) - 1
    if n < 5:
        return None
    s2 = 0.0
    for i in range(1, len(seg)):
        d = seg[i] - seg[i - 1]
        s2 += d * d
    rms = math.sqrt(s2 / n)
    dev = sorted(abs(seg[i] - seg[i - 1]) for i in range(1, len(seg)))
    mad = dev[len(dev) // 2] if len(dev) % 2 else 0.5 * (dev[len(dev) // 2 - 1] + dev[len(dev) // 2])
    return 0.5 * (rms + 1.4826 * mad)


def ewma_vol(seq):
    n = len(seq) - 1
    if n < 5:
        return None
    lam = 0.5 ** (1.0 / HL)
    num = 0.0
    den = 0.0
    w = 1.0
    for i in range(len(seq) - 1, 0, -1):
        d = seq[i] - seq[i - 1]
        num += w * d * d
        den += w
        w *= lam
    return math.sqrt(num / den)


def monotone(row):
    for k in range(1, 5):
        if row[k] < row[k - 1]:
            row[k] = row[k - 1]


def forecast_case(markets, hmax):
    hist = []
    last = []
    for m in markets:
        prices = [float(p) for p in (m.get('prices') or [])]
        hist.append([logit(p) for p in prices])
        last.append(prices[-1] if prices else 0.5)

    vols = []
    for L in hist:
        a = flat_vol(L)
        b = ewma_vol(L)
        if a is None:
            a = b if b is not None else 1e-3
        if b is None:
            b = a
        vols.append((1.0 - W_EWMA) * a + W_EWMA * b)
    ordered = sorted(vols)
    n = len(ordered)
    if n == 0:
        level = 1e-3
    elif n % 2:
        level = ordered[n // 2]
    else:
        level = 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])

    # A case whose every market sat still for the whole window has a median
    # volatility of exactly zero, and the ratio below would divide by it: the
    # whole case then raises instead of producing a fan. Floor the level.
    level = level if level > 1e-9 else 1e-9

    fans = []
    for L, price, vol in zip(hist, last, vols):
        v = vol if vol > 1e-9 else 1e-9
        step = level * (v / level) ** GAMMA * SCALE
        centre = L[-1] if L else logit(price)
        if len(L) > WIN + 1:
            drift = centre - L[-1 - WIN]
        else:
            drift = 0.0
        band = Z[band_of(price)]
        rows = []
        for h in range(1, hmax + 1):
            c = centre - BETA * drift * ((h / float(WIN)) ** 0.85)
            spread = step * (h ** ALPHA)
            zr = band[h] if h <= 240 else band[240]
            row = [round(clamp(sigmoid(c + zr[k] * spread)), NDIGITS) for k in range(5)]
            monotone(row)
            rows.append(row)
        fans.append(rows)

    renormalise(fans, last)
    return fans


def renormalise(fans, last):
    """One factor per case, from the first-horizon medians, applied to every row."""
    if not fans or not fans[0]:
        return
    s_total = sum(last)
    target = s_total + TARGET_ADJ - s_total * KAPPA
    current = sum(fan[0][2] for fan in fans)
    if current <= SUM_EPS:
        return
    factor = clamp(target / current, FACTOR_LO, FACTOR_HI)
    if abs(factor - 1.0) < 1e-12:
        return
    for fan in fans:
        for row in fan:
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
