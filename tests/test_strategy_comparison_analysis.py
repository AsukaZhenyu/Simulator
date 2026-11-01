"""Handwritten scalar oracles for finite, paired analytical comparisons."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.strategy_comparison_analysis import analyze, plots


def row(candidate, time, *, scenario="dense-short", kind="dense", **metrics):
    return dict(scenario_id=scenario, model_kind=kind, prefill_tokens=3, decode_steps=2,
                route_pattern="not_applicable" if kind == "dense" else "hot",
                candidate_id=candidate, success=time is not None,
                status="completed" if time is not None else "capacity_exceeded",
                error_code=None if time is not None else "CAPACITY",
                completed_makespan_ns=time, calibrated=False, **metrics)


def pair_by(rows, before, after):
    return next(r for r in rows if r["before_candidate"] == before and r["after_candidate"] == after)


def interaction_by(result, execution, storage):
    return next(r for r in result["interactions"] if r["execution"] == execution and r["storage"] == storage)


class StrategyComparisonAnalysisTests(unittest.TestCase):
    def test_exact_integer_ties_best_reference_and_no_mutation(self):
        large = 2**54
        records = [row("e0-s0", large+2), row("e0-s1", large+1),
                   row("e2-s3", large+1), row("e1-s0", large+3)]
        original = copy.deepcopy(records)
        result = analyze(records)
        self.assertEqual(result["scenarios"][0]["best_candidates"], ["e0-s1", "e2-s3"])
        self.assertEqual(result["scenarios"][0]["best_candidate"], "e0-s1")
        self.assertEqual(result["scenarios"][0]["best_makespan_ns"], large+1)
        self.assertEqual(result["scenarios"][0]["best_reference_speedup"], (large+2)/(large+1))
        self.assertEqual(records, original)
        json.dumps(result, allow_nan=False)

    def test_missing_and_failed_cells_do_not_get_zero_latency_or_ratios(self):
        records = [row("e0-s0", 100), row("e0-s1", None), row("e1-s0", 80)]
        records[1]["diagnostic_partial_makespan_ns"] = 7
        result = analyze(records)
        self.assertEqual(result["totals"]["complete_count"], 2)
        self.assertEqual(result["totals"]["failure_count"], 1)
        self.assertEqual(result["totals"]["missing_count"], 13)
        failed = next(r for r in result["comparisons"] if r["candidate_id"] == "e0-s1")
        missing = next(r for r in result["comparisons"] if r["candidate_id"] == "e0-s2")
        self.assertIsNone(failed["completed_makespan_ns"])
        self.assertIsNone(failed["reference_speedup"])
        self.assertEqual(missing["status"], "missing")
        self.assertIsNone(missing["completed_makespan_ns"])
        cell = interaction_by(result, "e1", "s1")
        self.assertFalse(cell["complete_four_cells"])
        self.assertIsNone(cell["joint_gain_over_better_single"])
        self.assertEqual(result["execution_rank_reversals"], [])

    def test_missing_reference_does_not_prevent_catalogue_best(self):
        result = analyze([row("e1-s2", 12), row("e3-s3", 6)])
        summary = result["scenarios"][0]
        self.assertEqual(summary["best_candidate"], "e3-s3")
        self.assertIsNone(summary["best_reference_speedup"])
        self.assertEqual(result["one_shot_selections"][0]["status"], "selectors_unavailable")

    def test_paired_prefetch_deltas_and_role_traffic_are_hand_counted(self):
        records = [row("e2-s0", 100, h2d_bytes=30, d2h_bytes=2,
                       immutable_weight_reload_bytes=8,
                       copy_bytes_by_direction_and_role={"dram->vram": {"weight": 30},
                                                         "vram->dram": {"activation": 2}}),
                   row("e2-s1", 80, h2d_bytes=50, d2h_bytes=0,
                       immutable_weight_reload_bytes=4,
                       copy_bytes_by_direction_and_role={"dram->vram": {"weight": 40, "kv_cache": 10}})]
        result = analyze(records)
        pair = pair_by(result["prefetch_pairs"], "e2-s0", "e2-s1")
        self.assertTrue(pair["complete_pair"])
        self.assertEqual(pair["latency_delta_ns"], -20)
        self.assertEqual(pair["latency_saving_ns"], 20)
        self.assertEqual(pair["latency_saving_fraction"], .2)
        self.assertEqual(pair["before_over_after_ratio"], 1.25)
        self.assertEqual(pair["metric_deltas"]["h2d_bytes"], 20)
        self.assertEqual(pair["metric_deltas"]["d2h_bytes"], -2)
        self.assertEqual(pair["metric_deltas"]["immutable_weight_reload_bytes"], -4)
        self.assertIsNone(pair["metric_deltas"]["peak_vram_bytes"])
        self.assertEqual(pair["copy_bytes_by_direction_and_role_delta"], {
            "dram->vram": {"weight": 10, "kv_cache": 10}, "vram->dram": {"activation": -2}})
        self.assertEqual(pair["before"]["completed_makespan_ns"], 100)

    def test_cache_pairs_hold_prefetch_fixed_and_preserve_unknown_traffic(self):
        result = analyze([row("e0-s0", 100), row("e0-s2", 110),
                          row("e0-s1", 80), row("e0-s3", 60)])
        disabled = pair_by(result["cache_pairs"], "e0-s0", "e0-s2")
        enabled = pair_by(result["cache_pairs"], "e0-s1", "e0-s3")
        self.assertFalse(disabled["prefetch_enabled"])
        self.assertEqual(disabled["latency_saving_fraction"], -.1)
        self.assertTrue(enabled["prefetch_enabled"])
        self.assertEqual(enabled["latency_saving_fraction"], .25)
        self.assertIsNone(enabled["copy_bytes_by_direction_and_role_delta"])

    def test_strict_reversal_is_detected_beyond_reference_axes_with_gap(self):
        result = analyze([row("e2-s2", 250), row("e3-s2", 200),
                          row("e2-s3", 100), row("e3-s3", 150)])
        self.assertEqual(len(result["execution_rank_reversals"]), 1)
        reversal = result["execution_rank_reversals"][0]
        self.assertTrue(reversal["strict_reversal"])
        self.assertFalse(reversal["tie_at_either_storage"])
        self.assertEqual(reversal["execution_a_minus_b_at_storage_a_ns"], 50)
        self.assertEqual(reversal["execution_a_minus_b_at_storage_b_ns"], -50)
        self.assertEqual(reversal["absolute_gap_at_storage_a_ns"], 50)
        self.assertEqual(reversal["relative_gap_at_storage_a"], .25)
        self.assertEqual(reversal["relative_gap_at_storage_b"], .5)
        self.assertEqual(result["totals"]["strict_execution_rank_reversals"], 1)

    def test_a_tie_is_not_a_strict_reversal(self):
        result = analyze([row("e0-s0", 100), row("e1-s0", 100),
                          row("e0-s1", 100), row("e1-s1", 90)])
        cell = result["execution_rank_reversals"][0]
        self.assertFalse(cell["strict_reversal"])
        self.assertTrue(cell["tie_at_either_storage"])
        self.assertFalse(interaction_by(result, "e1", "s1")["strict_execution_order_reversal"])

    def test_positive_joint_saving_and_nonadditivity(self):
        result = analyze([row("e0-s0", 100), row("e1-s0", 90),
                          row("e0-s1", 80), row("e1-s1", 50)])
        cell = interaction_by(result, "e1", "s1")
        self.assertEqual(cell["joint_gain_over_better_single"], 30/80)
        self.assertEqual(cell["nonadditive_time_saving_ns"], 20)
        self.assertFalse(cell["strict_execution_order_reversal"])

    def test_negative_joint_gain_and_one_shot_regret_are_not_clipped(self):
        # Independently choose e1 from s0 and s1 from e0. Their composition
        # takes 120, although the larger declared catalogue contains e2-s2=50.
        result = analyze([row("e0-s0", 100), row("e1-s0", 80),
                          row("e0-s1", 70), row("e1-s1", 120), row("e2-s2", 50)])
        cell = interaction_by(result, "e1", "s1")
        self.assertEqual(cell["joint_gain_over_better_single"], -50/70)
        self.assertEqual(cell["nonadditive_time_saving_ns"], -70)
        self.assertTrue(cell["strict_execution_order_reversal"])
        selection = result["one_shot_selections"][0]
        self.assertEqual(selection["composed_candidate"], "e1-s1")
        self.assertEqual(selection["completed_makespan_ns"], 120)
        self.assertEqual(selection["offline_catalogue_best_candidates"], ["e2-s2"])
        self.assertEqual(selection["finite_catalogue_regret_fraction"], 1.4)
        self.assertEqual(selection["composed_over_offline_best_ratio"], 2.4)

    def test_one_shot_ties_are_deterministic_and_failed_composition_unavailable(self):
        records = [row("e0-s0", 100), row("e1-s0", 80), row("e2-s0", 80),
                   row("e0-s1", 70), row("e0-s2", 70), row("e1-s1", None)]
        selection = analyze(records)["one_shot_selections"][0]
        self.assertEqual(selection["execution_selection_ties_at_s0"], ["e1-s0", "e2-s0"])
        self.assertEqual(selection["storage_selection_ties_at_e0"], ["e0-s1", "e0-s2"])
        self.assertEqual(selection["composed_candidate"], "e1-s1")
        self.assertEqual(selection["status"], "composed_candidate_failed")
        self.assertIsNone(selection["finite_catalogue_regret_fraction"])
        missing = analyze(records[:-1])["one_shot_selections"][0]
        self.assertEqual(missing["status"], "composed_candidate_missing")

    def test_dense_moe_catalogues_and_scenarios_are_not_mixed(self):
        result = analyze([row("e0-s0", 10), row("e5-s3", 20, scenario="moe-hot", kind="moe")])
        self.assertEqual(result["totals"]["scenario_count"], 2)
        self.assertEqual(result["totals"]["expected_count"], 40)
        self.assertEqual(result["totals"]["missing_count"], 38)
        self.assertEqual([s["best_candidate"] for s in result["scenarios"]], ["e0-s0", "e5-s3"])
        self.assertFalse(any(r["complete_four_cells"] for r in result["interactions"]))

    def test_invalid_completed_times_fail_loudly(self):
        for value in (0, -1, True, 2.5, "4", None):
            record = row("e0-s0", 10)
            record["completed_makespan_ns"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                analyze([record])
        failed = row("e0-s0", None)
        failed["completed_makespan_ns"] = 4
        with self.assertRaises(ValueError):
            analyze([failed])

    def test_duplicate_identity_unknown_candidate_and_changed_metadata_rejected(self):
        valid = row("e0-s0", 10)
        bad_rows = [
            [valid, dict(valid)],
            [dict(valid, candidate_id="e4-s0")],
            [dict(valid, candidate_id="e0-s9")],
            [dict(valid, calibrated=True)],
            [dict(valid, success=1)],
            [valid, dict(row("e1-s0", 12), prefill_tokens=8)],
            [valid, row("e1-s0", 12, kind="moe")],
        ]
        for records in bad_rows:
            with self.subTest(records=records), self.assertRaises(ValueError):
                analyze(records)

    def test_empty_analysis_is_json_safe_and_explicit(self):
        result = analyze([])
        self.assertEqual(result["totals"]["scenario_count"], 0)
        self.assertEqual(result["totals"]["expected_count"], 0)
        self.assertIn("not measurements", result["protocol"]["scope"])
        json.dumps(result, allow_nan=False)

    def test_plot_export_includes_failure_and_missing_cells(self):
        try:
            import matplotlib  # noqa: F401
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("plotting runtime unavailable")
        records = [row("e0-s0", 100), row("e1-s0", 80), row("e0-s1", None)]
        # Keep plots under the writable checkout on restricted Windows runners.
        scratch = Path(__file__).resolve().parents[1]/"build"/"analysis-test-temp"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            names = plots(records, analyze(records), directory)
            self.assertEqual(len(names), 6)
            self.assertEqual(len(set(names)), 6)
            for name in names:
                self.assertGreater((Path(directory)/name).stat().st_size, 100)
            svg = (Path(directory)/"latency_dense-short.svg").read_text(encoding="utf-8")
            self.assertIn("FAIL", svg)
            self.assertIn("MISS", svg)
            self.assertIn("not hardware measurements", svg)


if __name__ == "__main__":
    unittest.main()
