# polymarket-nvidia-ceo

A quantile forecaster for Polymarket's "Next CEO of NVIDIA?" markets, entered in
[make.fun · polymarket-nvidia-ceo-forecast-01](https://make.fun/bounty/polymarket-nvidia-ceo-forecast-01).

Each case holds every trading market of the event with its Yes price on a
1-minute grid for the 72 hours up to a checkpoint. `main.py` writes, for each
market and each of the 240 minutes after the checkpoint, the 5%, 25%, 50%, 75%
and 95% quantiles of that minute's price.

## Model

Prices are bounded, so the forecast is built in log-odds space and mapped back
through the logistic:

- **Step size.** The market's own last four hours set the per-sqrt-minute step,
  as a blend of the RMS and the MAD of its 1-minute log-odds changes. Using both
  keeps a single jump from inflating the whole forecast, and it adapts when a
  market goes quiet.
- **Tails.** Student-t innovations (4 degrees of freedom) instead of normal
  ones: these markets jump, and a normal spread pays for the jumps it misses.
- **Mean reversion.** The path is pulled back against the last four hours' move,
  because a jump in a thin book tends to partly revert.

The step size is re-estimated per market on every case, so a market that has
been flat for hours gets a tight forecast and a market in play gets a wide one.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

In the grader it reads `/input/<id>.in` and writes `/output/<id>.out`:
`{"forecasts":[{"id":..., "q":[[5 numbers] x 240]}]}`.

Standard library only, no network, deterministic (two runs on a case are byte
for byte the same), well under the 8 s per-case cap.

## License

MIT
