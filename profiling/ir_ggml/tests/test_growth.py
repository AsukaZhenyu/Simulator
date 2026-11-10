"""The growth tranche: the window it fills and the rules it must not break.

A price tranche is the one place a campaign can quietly stop being a measurement:
a depth borrowed from a neighbour, a cell carried above its device's cv rule, a
`(kind, phase)` the frozen table never priced. Each of those produces a plausible
A1 and is refused here, so the refusals are what these tests pin. The report side
gets the same treatment: a depth gap and a device gap are different findings and
must not be counted as one.
"""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from profiling.ir_ggml import growth_prices as prices_subject
from profiling.ir_ggml import growth_report as report_subject


def cpu_entry(kind: str, phase: str, tokens: int, history: int, duration: int = 1000) -> dict:
    return {"kind": kind, "device": "cpu", "phase": phase, "batch": 1, "tokens": tokens,
            "visible_history": history, "duration_ns": duration,
            "physical_signature": {"config": {"hidden": 8, "kind": kind}}}


def gpu_entry(kind: str, phase: str, tokens: int, history: int) -> dict:
    return {"kind": kind, "device": "gpu", "phase": phase, "batch": 1, "tokens": tokens,
            "visible_history": history, "duration_ns": 500,
            "physical_signature": {"config": {"hidden": 8, "kind": kind}}}


class ExportFixture(unittest.TestCase):
    """A temporary model export whose tables declare only what a test needs."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        export = self.root / "m"
        export.mkdir()
        costs = {"entries": [cpu_entry("projection", "decode", 1, 0),
                             cpu_entry("attention", "decode", 1, 0),
                             cpu_entry("attention", "prefill", 64, 0)]}
        (export / "costs.json").write_text(json.dumps(costs), encoding="utf-8")
        gpu = {"prices": [gpu_entry("attention", "decode", 1, 0)]}
        (export / "gpu_costs.json").write_text(json.dumps(gpu), encoding="utf-8")
        self.export = export
        # `growth_prices` imports its siblings as top-level modules (it puts its own
        # directory on `sys.path`), so the export map to patch is the one its module
        # object holds -- not the package-level `instantiate` this file imported.
        patcher = mock.patch.dict(prices_subject.inst.EXPORTS, {"m": export})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.folder.cleanup)

    def run_dir(self, name: str, rows: list[dict]) -> Path:
        path = self.root / name
        path.mkdir()
        (path / "measurements.json").write_text(json.dumps(rows), encoding="utf-8")
        return path


def row(kind: str, phase: str, tokens: int, history: int, median: float = 1234.5,
        cv: float = 0.02, status: str = "measured", case_id: str | None = None) -> dict:
    case = {"id": case_id or f"{kind}-{phase}-{tokens}-{history}", "kind": kind,
            "phase": phase, "batch": 1, "tokens": tokens, "visible_history": history}
    return {"case": case, "status": status, "summary": {"median_ns": median, "cv": cv,
                                                        "samples": 40},
            "raw_output": "raw.json"}


class WindowTests(unittest.TestCase):
    def test_the_window_is_the_depths_the_chain_visits(self):
        # A round-r step is at visible_history = tokens + r, so the window of `rounds`
        # rounds starts at `tokens` and the last depth is tokens + rounds - 1.
        self.assertEqual(prices_subject.window(64, 8), [64, 65, 66, 67, 68, 69, 70, 71])
        self.assertEqual(prices_subject.window(1, 1), [1])

    def test_a_window_with_no_depth_or_no_round_is_refused(self):
        for tokens, rounds in ((0, 8), (64, 0), (-1, 4)):
            with self.assertRaises(SystemExit):
                prices_subject.window(tokens, rounds)

    def test_the_order_keeps_every_stage_a_generator_offers(self):
        # The regression this pins: `in` on a generator consumes it, so a suite built
        # from `_order(kind for kind in ...)` silently lost every stage after the
        # first when the generator was tested before being materialised.
        offered = (kind for kind in ("attention", "dense_ffn", "projection"))
        self.assertEqual(prices_subject._order(offered),
                         ["projection", "attention", "dense_ffn"])


class FrozenKindsTests(ExportFixture):
    def test_a_device_reports_the_pairs_its_own_table_prices(self):
        self.assertEqual(prices_subject.frozen_kinds("m", "cpu"),
                         {("projection", "decode"), ("attention", "decode"),
                          ("attention", "prefill")})
        self.assertEqual(prices_subject.frozen_kinds("m", "gpu"),
                         {("attention", "decode")})

    def test_a_model_with_no_accelerator_table_says_so(self):
        (self.export / "gpu_costs.json").unlink()
        with self.assertRaises(SystemExit):
            prices_subject.frozen_kinds("m", "gpu")


class ExportTests(ExportFixture):
    def test_a_cpu_cell_is_rounded_up_to_a_whole_nanosecond(self):
        run = self.run_dir("a", [row("attention", "decode", 1, 64, median=1000.5)])
        document = prices_subject.export("m", 64, 1, [run], [])
        self.assertEqual(document["prices"][0]["duration_ns"], 1001)
        self.assertEqual(document["prices"][0]["duration_conversion"],
                         "ceil_observed_median_ns")

    def test_an_accelerator_cell_keeps_the_observed_median(self):
        run = self.run_dir("g", [row("attention", "decode", 1, 64, median=1000.5)])
        document = prices_subject.export("m", 64, 1, [], [run])
        self.assertEqual(document["prices"][0]["device"], "gpu")
        self.assertEqual(document["prices"][0]["duration_ns"], 1000.5)

    def test_a_pair_the_frozen_table_never_priced_is_refused(self):
        # The tranche extends the depth axis only. Pricing dense_ffn/decode here would
        # give this campaign a placement the frozen corpus does not have, and A2 would
        # then compare two strategy spaces.
        run = self.run_dir("a", [row("dense_ffn", "decode", 1, 64)])
        with redirect_stderr(io.StringIO()) as errors:
            document = prices_subject.export("m", 64, 1, [run], [])
        self.assertEqual(document["prices"], [])
        self.assertIn("extends the depth axis only", errors.getvalue())

    def test_a_cpu_cell_above_the_rule_is_refused_unless_the_export_asks_for_it(self):
        run = self.run_dir("a", [row("attention", "decode", 1, 64, cv=0.148)])
        with redirect_stderr(io.StringIO()):
            refused = prices_subject.export("m", 64, 1, [run], [])
            carried = prices_subject.export("m", 64, 1, [run], [], allow_unstable=True)
        self.assertEqual(refused["prices"], [])
        flag = carried["prices"][0]["measurement"]
        self.assertEqual(flag["quality_flags"], [prices_subject.CPU_UNSTABLE_FLAG])
        self.assertTrue(flag["cv_gt_0_1"])

    def test_an_accelerator_cell_above_its_rule_is_refused_even_on_request(self):
        # The accelerator export has no validation mode: an unstable cell would be
        # indistinguishable from a tier change once the number is written down.
        run = self.run_dir("g", [row("attention", "decode", 1, 64, cv=0.2)])
        with redirect_stderr(io.StringIO()):
            document = prices_subject.export("m", 64, 1, [], [run], allow_unstable=True)
        self.assertEqual(document["prices"], [])

    def test_an_earlier_run_wins_a_cell_a_later_run_also_measured(self):
        first = self.run_dir("a", [row("attention", "decode", 1, 64, median=1000)])
        second = self.run_dir("b", [row("attention", "decode", 1, 64, median=2000)])
        with redirect_stderr(io.StringIO()):
            document = prices_subject.export("m", 64, 1, [first, second], [])
        self.assertEqual([p["duration_ns"] for p in document["prices"]], [1000])

    def test_the_reason_for_a_superseded_cell_is_the_earlier_run_not_its_cv(self):
        # A cell the first pass refused is re-measured alone; the earlier run's row is
        # still in the log and is now unstable as well. It lost to the earlier run, and
        # saying "cv above the rule" would name a reason the export never acted on --
        # and would read in the log as a second refusal of the cell that was rescued.
        first = self.run_dir("a", [row("attention", "decode", 1, 64, median=1000)])
        second = self.run_dir("b", [row("attention", "decode", 1, 64, median=2000,
                                        cv=0.5)])
        with redirect_stderr(io.StringIO()) as errors:
            document = prices_subject.export("m", 64, 1, [first, second], [])
        self.assertEqual([p["duration_ns"] for p in document["prices"]], [1000])
        self.assertIn("already priced by an earlier run", errors.getvalue())
        self.assertNotIn("cv 0.5000", errors.getvalue())

    def test_a_failed_row_is_refused_rather_than_priced(self):
        run = self.run_dir("a", [row("attention", "decode", 1, 64, status="failed")])
        with redirect_stderr(io.StringIO()):
            document = prices_subject.export("m", 64, 1, [run], [])
        self.assertEqual(document["prices"], [])

    def test_a_shape_the_campaign_does_not_use_is_refused(self):
        # A two-token decode cell is not a round of this chain, and pricing it would
        # put a cell in the window that the corpus never asks for.
        run = self.run_dir("a", [row("attention", "decode", 2, 64)])
        with redirect_stderr(io.StringIO()) as errors:
            document = prices_subject.export("m", 64, 1, [run], [])
        self.assertEqual(document["prices"], [])
        self.assertIn("one-token decode", errors.getvalue())

    def test_the_window_and_the_runs_are_written_into_the_table(self):
        run = self.run_dir("a", [row("attention", "decode", 1, 64)])
        document = prices_subject.export("m", 64, 8, [run], [])
        self.assertEqual(document["window"]["depths"], list(range(64, 72)))
        self.assertEqual(document["runs"], [str(run)])
        self.assertEqual(document["kv_dtype"], prices_subject.KV_DTYPE)


def unit(kind: str, phase: str, history, ratio, measured=1000.0, price=1000.0) -> dict:
    return {"unit": kind, "phase": phase, "history": history, "ratio": ratio,
            "measured_ns": measured, "price_ns": price}


def document(units_by_depth: dict[int, list[dict]], unpriceable=None) -> dict:
    rows = []
    for depth, units in units_by_depth.items():
        rows.append({"case": f"case-{depth}", "unstable": False,
                     "units": {f"{unit['unit']}-{index}": unit
                               for index, unit in enumerate(units)}})
    return {"model": "m", "tokens": 64, "replicas": 2,
            "rows": rows,
            "summary": {"cases": len(rows), "unstable_cases": 0,
                        "state": {"verdict": "in_band", "median": 1.05, "n": 13},
                        "unpriceable": unpriceable or {},
                        "makespan_resolved": {"n": len(rows), "median": 1.04,
                                              "p10": 0.95, "p90": 1.15}}}


class ReportTests(unittest.TestCase):
    def report(self, document: dict, prices: dict | None = None) -> str:
        stream = io.StringIO()
        with redirect_stdout(stream):
            report_subject.report(document, 64, len(
                {unit["history"] for units in document["rows"]
                 for unit in units["units"].values() if unit.get("history") is not None}),
                prices)
        return stream.getvalue()

    def test_a_round_out_of_the_band_fails_the_depth_gate(self):
        depths = {depth: [unit("attention", "decode", depth,
                              1.10 if depth < 66 else 1.30)] for depth in range(64, 72)}
        text = self.report(document(depths))
        self.assertIn("rounds outside: h=66, h=67, h=68, h=69, h=70, h=71", text)
        self.assertIn("FAIL  G4 depth", text)

    def test_every_round_in_the_band_passes_it(self):
        depths = {depth: [unit("attention", "decode", depth, 1.10)]
                  for depth in range(64, 72)}
        text = self.report(document(depths))
        self.assertIn("PASS  G4 depth", text)

    def test_a_device_gap_is_not_counted_as_a_missing_depth(self):
        # G3 is about the depth axis. An action on a device the tranche does not price
        # was declared out of scope in the prereg, so it is disclosed, not failed.
        depths = {depth: [unit("attention", "decode", depth, 1.10)]
                  for depth in range(64, 72)}
        text = self.report(document(depths, {"attention: no_price_for_device": 12}))
        self.assertIn("PASS  G3 coverage", text)
        self.assertIn("declared outside the campaign's scope", text)

    def test_a_missing_accelerator_depth_is_a_device_gap_not_a_depth_gap(self):
        # The MoE column's accelerator tranche refused five depths per stage, so a round
        # placed there has no price at its own depth -- but the device axis is the one the
        # prereg put out of scope, and `summarise` names it. Counting it as a missing
        # depth would fail G3 for a gap the report already declares elsewhere.
        depths = {depth: [unit("attention", "decode", depth, 1.10)]
                  for depth in range(64, 72)}
        text = self.report(document(depths, {"attention: no_price_at on gpu": 24}))
        self.assertIn("PASS  G3 coverage", text)
        self.assertIn("declared outside the campaign's scope", text)

    def test_a_missing_depth_fails_the_coverage_gate(self):
        depths = {depth: [unit("attention", "decode", depth, 1.10)]
                  for depth in range(64, 72)}
        text = self.report(document(depths, {"attention: no_price_at": 3}))
        self.assertIn("FAIL  G3 coverage", text)

    def test_units_are_split_by_the_depth_they_were_priced_at(self):
        depths = {64: [unit("projection", "decode", 64, 1.5)],
                  65: [unit("projection", "decode", 65, 1.6)]}
        split = report_subject.rounds_by_unit(document(depths))
        self.assertEqual(split[("projection", 64)], [1.5])
        self.assertEqual(split[("projection", 65)], [1.6])

    def test_a_prefill_unit_is_not_read_as_a_round(self):
        depths = {64: [unit("projection", "prefill", None, 1.2)]}
        self.assertEqual(report_subject.rounds_by_unit(document(depths)), {})
        self.assertEqual(report_subject.prefill_by_unit(document(depths)),
                         {"projection": [1.2]})

    def test_the_scatter_section_reads_the_tables_own_wander(self):
        depths = {depth: [unit("attention", "decode", depth, 1.0, measured=1000.0)]
                  for depth in range(64, 72)}
        table = {"prices": [{"kind": "attention", "phase": "decode",
                             "visible_history": depth,
                             "duration_ns": 200.0 if depth != 66 else 158.0}
                            for depth in range(64, 72)]}
        stream = io.StringIO()
        with redirect_stdout(stream):
            report_subject.scatter_section(document(depths), table, 64, 8)
        text = stream.getvalue()
        self.assertIn("price spread", text)
        self.assertIn("measured spread", text)


if __name__ == "__main__":
    unittest.main()
