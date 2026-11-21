"""Tests for the I/O layer: what a case on disk turns into.

`forecast_case` was already covered. `run()` was not, and that is where a case
can be silently answered against the wrong horizon or lost entirely.
"""
import importlib.util
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("pm", os.path.join(ROOT, "main.py"))
pm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pm)


def history(n=60):
    return [{"t": i, "price": 0.4 + 0.01 * (i % 3)} for i in range(n)]


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.indir = os.path.join(self.dir, "in")
        self.outdir = os.path.join(self.dir, "out")
        os.makedirs(self.indir)

    def put(self, name, payload, raw=None):
        path = os.path.join(self.indir, name)
        with open(path, "w") as f:
            f.write(raw if raw is not None else json.dumps(payload))
        return path

    def run_quiet(self):
        err = io.StringIO()
        with redirect_stderr(err):
            skipped = pm.run(self.indir, self.outdir)
        return skipped, err.getvalue()

    def out(self, name):
        with open(os.path.join(self.outdir, name)) as f:
            return json.load(f)


class HorizonTest(unittest.TestCase):
    def test_a_missing_horizon_defaults_to_240(self):
        self.assertEqual(pm.horizon_of({}), 240)

    def test_a_horizon_of_zero_is_zero(self):
        # `int(case.get(k) or 240)` reads 0 as missing and forecasts 240 minutes
        # against a question that asked for none.
        self.assertEqual(pm.horizon_of({"horizon_minutes": 0}), 0)

    def test_an_explicit_null_defaults(self):
        self.assertEqual(pm.horizon_of({"horizon_minutes": None}), 240)

    def test_a_numeric_string_is_read(self):
        self.assertEqual(pm.horizon_of({"horizon_minutes": "30"}), 30)

    def test_a_negative_horizon_is_clamped_to_zero(self):
        self.assertEqual(pm.horizon_of({"horizon_minutes": -5}), 0)


class RunTest(Base):
    def test_one_output_per_input(self):
        self.put("a.in", {"markets": [{"id": "m", "history": history()}]})
        self.put("b.in", {"markets": [{"id": "m", "history": history()}]})
        skipped, err = self.run_quiet()
        self.assertEqual(skipped, [])
        self.assertEqual(err, "")
        self.assertEqual(sorted(os.listdir(self.outdir)), ["a.out", "b.out"])

    def test_the_output_is_shaped_the_way_the_case_asked(self):
        self.put("a.in", {"markets": [{"id": "m1", "history": history()}]})
        self.run_quiet()
        body = self.out("a.out")
        self.assertEqual(len(body["forecasts"]), 1)
        self.assertEqual(body["forecasts"][0]["id"], "m1")
        self.assertEqual(len(body["forecasts"][0]["q"]), 240)

    def test_a_horizon_of_zero_writes_no_rows(self):
        self.put("a.in", {"horizon_minutes": 0, "markets": [{"id": "m", "history": history()}]})
        self.run_quiet()
        self.assertEqual(self.out("a.out")["forecasts"][0]["q"], [])

    def test_a_short_horizon_writes_that_many_rows(self):
        self.put("a.in", {"horizon_minutes": 5, "markets": [{"id": "m", "history": history()}]})
        self.run_quiet()
        self.assertEqual(len(self.out("a.out")["forecasts"][0]["q"]), 5)

    def test_an_unreadable_case_does_not_take_the_batch_with_it(self):
        self.put("a.in", {"markets": [{"id": "a", "history": history()}]})
        self.put("b.in", None, raw="{not json")
        self.put("c.in", {"markets": [{"id": "c", "history": history()}]})
        skipped, err = self.run_quiet()
        self.assertEqual([name for name, _ in skipped], ["b.in"])
        self.assertEqual(sorted(os.listdir(self.outdir)), ["a.out", "c.out"])
        self.assertIn("b.in", err, "the skipped case was not reported anywhere")

    def test_a_case_that_is_not_an_object_is_skipped(self):
        self.put("a.in", [1, 2, 3])
        self.put("b.in", {"markets": [{"id": "b", "history": history()}]})
        skipped, err = self.run_quiet()
        self.assertEqual([name for name, _ in skipped], ["a.in"])
        self.assertEqual(os.listdir(self.outdir), ["b.out"])

    def test_a_case_with_no_markets_writes_an_empty_forecast(self):
        self.put("a.in", {})
        skipped, _ = self.run_quiet()
        self.assertEqual(skipped, [])
        self.assertEqual(self.out("a.out"), {"forecasts": []})

    def test_a_market_without_an_id_does_not_lose_the_case(self):
        self.put("a.in", {"markets": [{"history": history()}, {"id": "m2", "history": history()}]})
        skipped, _ = self.run_quiet()
        self.assertEqual(skipped, [])
        ids = [f["id"] for f in self.out("a.out")["forecasts"]]
        self.assertEqual(len(ids), 2, "a market without an id lost its forecast")
        self.assertIsNone(ids[0])

    def test_only_dot_in_files_are_read(self):
        self.put("a.in", {"markets": [{"id": "m", "history": history()}]})
        self.put("notes.txt", None, raw="ignore me")
        skipped, _ = self.run_quiet()
        self.assertEqual(skipped, [])
        self.assertEqual(os.listdir(self.outdir), ["a.out"])

    def test_the_output_directory_is_created(self):
        self.assertFalse(os.path.exists(self.outdir))
        self.put("a.in", {"markets": []})
        self.run_quiet()
        self.assertTrue(os.path.isdir(self.outdir))

    def test_every_market_gets_exactly_one_fan(self):
        markets = [{"id": "m%d" % i, "history": history()} for i in range(5)]
        self.put("a.in", {"markets": markets})
        self.run_quiet()
        self.assertEqual(len(self.out(self.outdir + "/a.out")["forecasts"]), 5)


class MainTest(unittest.TestCase):
    def capture_main(self, env):
        """Call main() with the given environment and record what it asked run()."""
        old = {k: os.environ.get(k) for k in ("MAKEFUN_INPUT_DIR", "MAKEFUN_OUTPUT_DIR")}
        seen = []
        real = pm.run
        pm.run = lambda indir, outdir: seen.append((indir, outdir))
        try:
            for k, v in env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            pm.main()
        finally:
            pm.run = real
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        return seen

    def test_an_empty_output_variable_falls_back_instead_of_crashing(self):
        # `os.environ.get(k, default)` only uses the default when the key is
        # absent. Exported-but-empty gave FileNotFoundError: '' — the platform
        # exports the variable, so an empty value is a normal thing to see.
        seen = self.capture_main({"MAKEFUN_INPUT_DIR": "/in", "MAKEFUN_OUTPUT_DIR": ""})
        self.assertEqual(seen, [("/in", "/output")])

    def test_an_empty_input_variable_falls_back_too(self):
        seen = self.capture_main({"MAKEFUN_INPUT_DIR": "", "MAKEFUN_OUTPUT_DIR": "/out"})
        self.assertEqual(seen, [("/input", "/out")])

    def test_the_variables_are_used_when_they_are_set(self):
        seen = self.capture_main({"MAKEFUN_INPUT_DIR": "/a", "MAKEFUN_OUTPUT_DIR": "/b"})
        self.assertEqual(seen, [("/a", "/b")])

    def test_the_defaults_apply_when_nothing_is_set(self):
        seen = self.capture_main({"MAKEFUN_INPUT_DIR": None, "MAKEFUN_OUTPUT_DIR": None})
        self.assertEqual(seen, [("/input", "/output")])


if __name__ == "__main__":
    unittest.main()
