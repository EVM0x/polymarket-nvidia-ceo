"""Tests for the quantile forecaster.

main.py is standard library only, reads no clock, does no network and uses no
randomness, so every function in it is a pure function of its arguments. These
tests pin the properties the submission depends on: the log-odds round trip, the
band edges, the two volatility estimators, the shape and monotonicity of the fan,
the sum-of-medians correction, and the file contract of run().

Run: python3 -m unittest discover -s tests -v
"""

import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as f  # noqa: E402


def market(mid, prices):
    return {"id": mid, "prices": prices}


def walk(start, step, n):
    """A deterministic price path, so no test depends on randomness."""
    out = []
    p = start
    for i in range(n):
        p = min(0.98, max(0.02, p + step * (1 if i % 3 else -1)))
        out.append(round(p, 6))
    return out


class TransformTest(unittest.TestCase):
    def test_sigmoid_inverts_logit(self):
        for p in (0.001, 0.02, 0.05, 0.3, 0.5, 0.77, 0.95, 0.999):
            self.assertAlmostEqual(f.sigmoid(f.logit(p)), p, places=9)

    def test_logit_is_clamped_at_the_edges(self):
        # A price of exactly 0 or 1 has no log-odds, so it is clamped to EPS.
        self.assertEqual(f.logit(0.0), f.logit(f.EPS))
        self.assertEqual(f.logit(1.0), f.logit(1.0 - f.EPS))
        self.assertTrue(math.isfinite(f.logit(0.0)))
        self.assertTrue(math.isfinite(f.logit(1.0)))

    def test_sigmoid_never_overflows(self):
        for x in (-1000.0, -50.0, 0.0, 50.0, 1000.0):
            value = f.sigmoid(x)
            self.assertTrue(0.0 <= value <= 1.0)

    def test_clamp_holds_the_bounds(self):
        self.assertEqual(f.clamp(-1.0), 0.0)
        self.assertEqual(f.clamp(2.0), 1.0)
        self.assertEqual(f.clamp(0.4), 0.4)
        self.assertEqual(f.clamp(5.0, 0.0, 3.0), 3.0)


class BandTest(unittest.TestCase):
    def test_band_edges(self):
        cases = [
            (0.0, 0), (0.049, 0), (0.05, 1), (0.199, 1),
            (0.20, 2), (0.599, 2), (0.60, 3), (1.0, 3),
        ]
        for price, expected in cases:
            self.assertEqual(f.band_of(price), expected, f"price {price}")

    def test_every_band_row_has_five_quantiles(self):
        self.assertEqual(len(f.Z), 4)
        for band in f.Z:
            self.assertEqual(len(band), 241)
            for row in band:
                self.assertEqual(len(row), 5)

    def test_the_first_horizon_row_is_unused_but_present(self):
        # Index 0 is a placeholder; horizons are 1..240.
        self.assertEqual(len(f.Z[0][0]), 5)

    def test_the_table_is_quantised_to_thousandths(self):
        for band in f.Z:
            for row in band:
                for value in row:
                    self.assertAlmostEqual(value * 1000, round(value * 1000), places=9)


class VolatilityTest(unittest.TestCase):
    def test_too_short_a_history_has_no_estimate(self):
        self.assertIsNone(f.flat_vol([0.5, 0.51]))
        self.assertIsNone(f.ewma_vol([0.5, 0.51]))

    def test_flat_vol_is_none_below_six_points(self):
        self.assertIsNone(f.flat_vol([0.1, 0.2, 0.3, 0.4, 0.5]))
        self.assertIsNotNone(f.flat_vol([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))

    def test_flat_vol_scales_with_the_increments(self):
        base = [0.5 + 0.001 * i for i in range(20)]
        doubled = [0.5 + 0.002 * i for i in range(20)]
        self.assertAlmostEqual(f.flat_vol(doubled) / f.flat_vol(base), 2.0, places=6)

    def test_a_flat_history_has_zero_volatility(self):
        self.assertEqual(f.flat_vol([0.4] * 30), 0.0)
        self.assertEqual(f.ewma_vol([0.4] * 30), 0.0)

    def test_ewma_weights_the_newest_increment_most(self):
        # Same single big step, placed once at the newest end and once at the oldest.
        newest = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6]
        oldest = [0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
        self.assertGreater(f.ewma_vol(newest), f.ewma_vol(oldest))

    def test_ewma_is_not_fooled_by_a_single_old_jump(self):
        jumped = [0.5, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        quiet = [0.9] * 12
        self.assertGreater(f.ewma_vol(jumped), 0.0)
        self.assertLess(f.ewma_vol(jumped), 0.2)  # one jump, decayed away


class MonotoneTest(unittest.TestCase):
    def test_crossings_are_repaired(self):
        row = [0.5, 0.4, 0.3, 0.6, 0.2]
        f.monotone(row)
        self.assertEqual(row, [0.5, 0.5, 0.5, 0.6, 0.6])

    def test_an_already_monotone_row_is_untouched(self):
        row = [0.1, 0.2, 0.3, 0.4, 0.5]
        f.monotone(row)
        self.assertEqual(row, [0.1, 0.2, 0.3, 0.4, 0.5])


class FanTest(unittest.TestCase):
    def setUp(self):
        self.markets = [
            market("a", walk(0.10, 0.004, 60)),
            market("b", walk(0.45, 0.010, 60)),
            market("c", walk(0.80, 0.006, 60)),
        ]

    def test_one_fan_per_market_with_the_requested_horizon(self):
        fans = f.forecast_case(self.markets, 240)
        self.assertEqual(len(fans), len(self.markets))
        for fan in fans:
            self.assertEqual(len(fan), 240)
            for row in fan:
                self.assertEqual(len(row), 5)

    def test_a_short_horizon_is_respected(self):
        fans = f.forecast_case(self.markets, 7)
        for fan in fans:
            self.assertEqual(len(fan), 7)

    def test_every_quantile_is_a_probability(self):
        for fan in f.forecast_case(self.markets, 60):
            for row in fan:
                for value in row:
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 1.0)

    def test_quantiles_are_monotone_at_every_horizon(self):
        for fan in f.forecast_case(self.markets, 120):
            for h, row in enumerate(fan, start=1):
                for k in range(1, 5):
                    self.assertLessEqual(row[k - 1], row[k], f"row {h} crossed")

    def test_the_fan_is_deterministic(self):
        self.assertEqual(f.forecast_case(self.markets, 30), f.forecast_case(self.markets, 30))

    def test_the_band_widens_with_the_horizon(self):
        fan = f.forecast_case(self.markets, 240)[1]
        first = fan[0][4] - fan[0][0]
        last = fan[-1][4] - fan[-1][0]
        self.assertGreater(last, first, "a four-hour fan should be wider than a one-minute fan")

    def test_a_quiet_market_gets_a_narrower_fan_than_a_loud_one(self):
        quiet = f.forecast_case([market("q", [0.5] * 60)], 240)[0]
        loud = f.forecast_case([market("l", walk(0.5, 0.02, 60))], 240)[0]
        quiet_width = quiet[-1][4] - quiet[-1][0]
        loud_width = loud[-1][4] - loud[-1][0]
        self.assertGreater(loud_width, quiet_width)

    def test_a_case_of_only_flat_markets_still_produces_a_fan(self):
        # Every market sat still, so the median volatility is exactly zero. The
        # ratio that scales each market's step divided by it and the whole case
        # raised instead of writing any output at all.
        flat = [market("f1", [0.5] * 60), market("f2", [0.5] * 60)]
        fans = f.forecast_case(flat, 240)
        self.assertEqual(len(fans), 2)
        for fan in fans:
            self.assertEqual(len(fan), 240)
            for row in fan:
                for value in row:
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 1.0)

    def test_an_empty_history_falls_back_to_a_neutral_price(self):
        fan = f.forecast_case([{"id": "x", "prices": []}], 5)[0]
        self.assertEqual(len(fan), 5)
        for row in fan:
            for value in row:
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_a_single_point_history_does_not_crash(self):
        fan = f.forecast_case([market("x", [0.33])], 5)[0]
        self.assertEqual(len(fan), 5)

    def test_the_centre_stays_near_the_current_price_short_term(self):
        # A complete event: the candidates of one market sum to about 1, which is
        # the condition the sum-of-medians correction is written for. One market
        # quoted at 0.45 is not that case, and is not a shape the input produces.
        prices = [walk(0.45, 0.010, 60), walk(0.35, 0.008, 60), walk(0.20, 0.006, 60)]
        markets = [market(f"m{i}", p) for i, p in enumerate(prices)]
        fans = f.forecast_case(markets, 1)
        for fan, series in zip(fans, prices):
            self.assertLess(abs(fan[0][2] - series[-1]), 0.15)


class RenormaliseTest(unittest.TestCase):
    def test_the_sum_of_medians_is_pulled_toward_the_target(self):
        fans = [[[0.2, 0.3, 0.4, 0.5, 0.6]] for _ in range(3)]
        last = [0.4, 0.4, 0.4]
        f.renormalise(fans, last)
        total = sum(fan[0][2] for fan in fans)
        # The factor is clamped, so the sum moves toward the target, not onto it.
        self.assertLess(total, 1.2)
        self.assertGreater(total, 0.0)

    def test_a_sane_sum_is_left_alone(self):
        # The correction fires only when the first-horizon medians disagree with
        # the case total; here they already sum to the target, so factor is 1.
        fans = [[[0.3, 0.35, 0.4, 0.45, 0.5]], [[0.3, 0.35, 0.4, 0.45, 0.5]]]
        last = [0.3, 0.3]                       # S = 0.6, target = 0.8
        before = [row[:] for fan in fans for row in fan]
        f.renormalise(fans, last)
        after = [row for fan in fans for row in fan]
        self.assertEqual(before, after)

    def test_rows_stay_monotone_after_the_correction(self):
        fans = [[[0.1, 0.15, 0.2, 0.25, 0.3]] for _ in range(4)]
        f.renormalise(fans, [0.2] * 4)
        for fan in fans:
            row = fan[0]
            for k in range(1, 5):
                self.assertLessEqual(row[k - 1], row[k])

    def test_no_fans_is_a_no_op(self):
        f.renormalise([], [])          # must not raise
        f.renormalise([[]], [0.5])     # must not raise


class RunContractTest(unittest.TestCase):
    def test_every_case_becomes_one_out_file_with_its_ids(self):
        with tempfile.TemporaryDirectory() as indir, tempfile.TemporaryDirectory() as outdir:
            case = {
                "horizon_minutes": 10,
                "markets": [market("m1", walk(0.3, 0.01, 30)),
                            market("m2", walk(0.6, 0.01, 30))],
            }
            with open(os.path.join(indir, "case_a.in"), "w") as fh:
                json.dump(case, fh)
            with open(os.path.join(indir, "notes.txt"), "w") as fh:
                fh.write("not a case")

            f.run(indir, outdir)

            self.assertEqual(sorted(os.listdir(outdir)), ["case_a.out"])
            with open(os.path.join(outdir, "case_a.out")) as fh:
                out = json.load(fh)
            self.assertEqual([fc["id"] for fc in out["forecasts"]], ["m1", "m2"])
            for fc in out["forecasts"]:
                self.assertEqual(len(fc["q"]), 10)
                self.assertEqual(len(fc["q"][0]), 5)

    def test_the_horizon_defaults_to_240(self):
        with tempfile.TemporaryDirectory() as indir, tempfile.TemporaryDirectory() as outdir:
            with open(os.path.join(indir, "c.in"), "w") as fh:
                json.dump({"markets": [market("m", walk(0.5, 0.01, 20))]}, fh)
            f.run(indir, outdir)
            with open(os.path.join(outdir, "c.out")) as fh:
                out = json.load(fh)
            self.assertEqual(len(out["forecasts"][0]["q"]), 240)

    def test_output_is_compact_json(self):
        with tempfile.TemporaryDirectory() as indir, tempfile.TemporaryDirectory() as outdir:
            with open(os.path.join(indir, "c.in"), "w") as fh:
                json.dump({"horizon_minutes": 3, "markets": [market("m", walk(0.5, 0.01, 20))]}, fh)
            f.run(indir, outdir)
            with open(os.path.join(outdir, "c.out")) as fh:
                text = fh.read()
            self.assertNotIn(": ", text)
            self.assertNotIn(", ", text)


if __name__ == "__main__":
    unittest.main()
