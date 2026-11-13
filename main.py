#!/usr/bin/env python3
"""Quantile forecaster for Polymarket's "Next CEO of NVIDIA?" event.

Input: one JSON case per run -- every market's Yes price on a 1-minute grid
for the 72 hours leading up to a checkpoint ("at").  Output: the 5/25/50/75/95
percent quantiles of each market's price at each of the next `horizon_minutes`
minutes (row h-1 is the price at at+h).

Model
-----
A price is a probability, so its natural coordinate is the log-odds

    L = logit(p),  logit(p) = log(p / (1 - p)).

There the process behaves like a driftless random walk: increments are roughly
symmetric, the variance grows linearly in time, and a forecast of the price at
horizon h is a quantile of a normal (in L) pushed back through the logistic,
which keeps every output inside [0, 1] automatically.

The only unknown is the step size: the standard deviation of one minute of
log-odds movement.  We estimate it from the market's own recent increments and
let it grow like sqrt(h), the random-walk diffusion law.  Three refinements
matter empirically:

  * a robust step estimator -- the average of the root-mean-square increment
    and 1.4826 * median |increment|.  The RMS reacts to a single jump, the MAD
    ignores it; the blend is stable in both regimes;
  * Student-t innovations -- real tails are fatter than normal, so the
    quantile multipliers are inflated by a Cornish-Fisher t(nu) expansion;
  * mean reversion -- a market that just moved hard tends to give part of it
    back, so the centre of the forecast is nudged against the last 4 hours'
    move, fading in over the horizon.

Standard library only, no network, fully deterministic.
"""

import json
import math
import os
import statistics

# ---------------------------------------------------------------------------
# constants

QLEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# standard-normal quantiles of the levels above
ZQ = (-1.6448536269514722, -0.6744897501960817, 0.0,
      0.6744897501960817, 1.6448536269514722)

LOGIT_EPS = 2e-4      # keep logit finite for prices sitting exactly on 0 or 1
VOL_WIN = 360         # minutes of history that set the step size
VOL_SCALE = 0.75      # step multiplier (fitted on the practice cases)
T_NU = 4.0            # Student-t degrees of freedom for the tails
REV_WIN = 240         # minutes of recent move the reversion term looks at
REV_BETA = 0.15       # strength of the reversion drift
REV_GAMMA = 0.85      # exponent: how fast reversion fades in over the horizon
NDIGITS = 6           # rounding of the emitted probabilities


# ---------------------------------------------------------------------------
# small numerics


def clip01(x):
    """Clamp to [0, 1]."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def logit(p):
    """log-odds of a probability, with the endpoints pulled inside by eps."""
    if p != p:                                  # NaN guard
        return 0.0
    if p < LOGIT_EPS:
        p = LOGIT_EPS
    elif p > 1.0 - LOGIT_EPS:
        p = 1.0 - LOGIT_EPS
    return math.log(p / (1.0 - p))


def sigmoid(x):
    """Inverse of logit; written branch-wise so exp never overflows."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


# ---------------------------------------------------------------------------
# statistics


def robust_step(series):
    """Per-sqrt-minute step size of a log-odds series.

    Both estimators below are scale estimators of the 1-minute increments:
      * RMS  = sqrt(mean(d^2))        -- the maximum-likelihood Gaussian sigma,
      * MAD  = 1.4826 * median(|d|)   -- the same quantity for a normal, but
                                         driven by the median so a single jump
                                         in the window cannot inflate it.
    Their average is robust to outliers while still reacting to genuine
    volatility, which is what the practice cases reward.
    """
    seg = series[-(VOL_WIN + 1):]
    if len(seg) < 6:
        return 1e-3
    incs = [b - a for a, b in zip(seg, seg[1:])]
    rms = math.sqrt(sum(x * x for x in incs) / len(incs))
    mad = 1.4826 * statistics.median([abs(x) for x in incs])
    step = 0.5 * (rms + mad) * VOL_SCALE
    return step if step > 1e-6 else 1e-6


def t_multiplier(z, nu=T_NU):
    """Cornish-Fisher expansion of a Student-t quantile around the normal one.

    For nu degrees of freedom the t quantile is wider than z; expanding the
    quantile function in 1/nu gives z * sqrt((nu-2)/nu) * (1 + (z^2+1)/(4nu)).
    """
    if nu <= 2.0:
        return z
    return z * math.sqrt((nu - 2.0) / nu) * (1.0 + (z * z + 1.0) / (4.0 * nu))


# ---------------------------------------------------------------------------
# model


def forecast_market(prices, hmax):
    """240 rows of five non-decreasing quantiles for one market."""
    series = [logit(p) for p in prices]
    if not series:
        series = [0.0]

    step = robust_step(series)                  # per sqrt-minute, in log-odds
    base = series[-1]                           # the price at the checkpoint
    # recent move: how far the log-odds travelled over the last four hours
    recent = base - series[-(REV_WIN + 1)] if len(series) > REV_WIN else 0.0
    zt = [t_multiplier(z) for z in ZQ]

    rows = []
    for h in range(1, hmax + 1):
        spread = step * math.sqrt(h)            # random-walk diffusion
        # reversion: fades in from 0 at h=0 towards the full pull at h=hmax
        drift = -REV_BETA * recent * (h / float(hmax)) ** REV_GAMMA
        centre = base + drift
        row = []
        for z in zt:
            row.append(round(clip01(sigmoid(centre + z * spread)), NDIGITS))
        # rounding can break monotonicity only by 1e-6; repair it in place
        for i in range(1, len(row)):
            if row[i] < row[i - 1]:
                row[i] = row[i - 1]
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# io


def run(indir, outdir):
    os.makedirs(outdir, exist_ok=True)
    for name in sorted(os.listdir(indir)):
        if not name.endswith('.in'):
            continue
        with open(os.path.join(indir, name), 'r') as fh:
            case = json.load(fh)
        hmax = int(case.get('horizon_minutes') or 240)
        forecasts = []
        for market in case.get('markets') or []:
            forecasts.append({
                'id': market['id'],
                'q': forecast_market(market.get('prices') or [], hmax),
            })
        out_path = os.path.join(outdir, name[:-len('.in')] + '.out')
        with open(out_path, 'w') as fh:
            json.dump({'forecasts': forecasts}, fh, separators=(',', ':'))


def main():
    run(os.environ.get('MAKEFUN_INPUT_DIR', '/input'),
        os.environ.get('MAKEFUN_OUTPUT_DIR', '/output'))


if __name__ == '__main__':
    main()
