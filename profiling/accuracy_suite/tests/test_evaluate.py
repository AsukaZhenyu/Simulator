from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("accuracy_evaluate", Path(__file__).resolve().parents[1] / "evaluate.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def rows():
    result = []
    for split, seed, start in (("profile", 11, "2026-10-18T08:00:00+00:00"),
                               ("heldout", 23, "2026-10-18T10:00:00+00:00")):
        for repeat in range(3):
            for strategy, cost in (("cpu", 100), ("gpu", 120)):
                result.append(dict(id=f"{split}-{repeat}-{strategy}", case_id="case", strategy_id=strategy,
                    comparison_group="resident_input_to_dram_output", family="ffn", split=split,
                    process_id=f"process-{split}-{repeat}", seed=seed, status="measured",
                    samples_ns=[cost-1, cost, cost+1], numerical_valid=True, source_sha256="a"*64,
                    physical_signature_sha256=("b" if strategy=="cpu" else "c")*64,
                    timing_unit="isolated_completed_invocation_ns", started_utc=start))
    return result


def evaluate(values=None, predictions=None, **kwargs):
    return module.evaluate(rows() if values is None else values,
        {("case", "cpu"): 101, ("case", "gpu"): 119} if predictions is None else predictions,
        model_frozen_utc="2026-10-18T09:00:00+00:00", model_sha256="d"*64,
        bootstrap_repetitions=100, **kwargs)


class EvidenceTests(unittest.TestCase):
    def test_accepted_fresh_paths_and_process_cluster_intervals(self):
        source = rows()
        before = deepcopy(source)
        report = evaluate(source)
        self.assertEqual(source, before)
        self.assertEqual(report["fresh_validation_accepted"], 2)
        self.assertEqual(report["original_quality_accepted"], 2)
        self.assertEqual(report["results"][0]["heldout_process_median_interval_ns"], [100,100])
        self.assertTrue(report["rankings"][0]["selected_best"])
        self.assertTrue(report["rankings"][0]["observed_winner_interval_separated"])
        self.assertEqual(report["rankings"][0]["selection_regret_ns"], 0)
        self.assertFalse(report["physical_capacity_validated"])

    def test_freshness_noise_numeric_process_and_seed_gates_independent_of_low_error(self):
        mutations = [
            ("frozen_before_heldout", lambda values: [r.update(started_utc="2026-10-18T08:00:00+00:00") for r in values if r["split"]=="heldout"]),
            ("disjoint_profile_heldout_seeds", lambda values: [r.update(seed=11) for r in values]),
            ("independent_process_coverage", lambda values: [r.update(process_id="one-process") for r in values]),
            ("numerical_validity", lambda values: values[0].update(numerical_valid=False)),
            ("within_process_cv", lambda values: values[0].update(samples_ns=[1,100,199])),
        ]
        for gate, change in mutations:
            with self.subTest(gate=gate):
                source=rows();change(source);report=evaluate(source)
                self.assertLess(report["fresh_validation_accepted"],2)
                self.assertTrue(any(not r["gates"][gate] for r in report["results"]))
                self.assertTrue(all(r["gates"]["heldout_relative_error"] for r in report["results"]))

    def test_retrospective_cannot_become_fresh_acceptance(self):
        report=evaluate(historically_inspected=True)
        self.assertEqual(report["original_quality_accepted"],2)
        self.assertEqual(report["fresh_validation_accepted"],0)
        self.assertFalse(report["rankings"][0]["fresh_validation_accepted"])

    def test_failed_and_unknown_paths_remain_visible_and_block_rankings(self):
        source=rows();source[0].update(status="failed", samples_ns=None)
        report=evaluate(source,{("case","cpu"):101})
        self.assertEqual(report["paths"],2)
        self.assertEqual(report["unknown_predictions"],1)
        self.assertEqual(report["results"][0]["failed_rows"],[source[0]["id"]])
        self.assertEqual(report["fresh_validation_accepted"],0)
        self.assertFalse(report["rankings"][0]["known"])

    def test_near_crossing_wrong_choice_reports_regret_even_with_small_time_errors(self):
        report=evaluate(predictions={("case","cpu"):111,("case","gpu"):110})
        rank=report["rankings"][0]
        self.assertFalse(rank["selected_best"])
        self.assertEqual(rank["selection_regret_ns"],20)
        self.assertAlmostEqual(rank["selection_regret_relative"],.2)
        self.assertTrue(rank["latency_quality_accepted"])

    def test_different_completion_boundaries_and_timing_units_never_rank_together(self):
        for field,value in (("comparison_group","cold_to_dram"),("timing_unit","mean_within_16_invocations_ns")):
            source=rows()
            for r in source:
                if r["strategy_id"]=="gpu":r[field]=value
            self.assertEqual(evaluate(source)["rankings"],[])

    def test_mixed_protocol_duplicate_raw_rows_and_bad_samples_rejected(self):
        mutations=[lambda values:values[0].update(timing_unit="different"),
            lambda values:values[0].update(physical_signature_sha256="e"*64),
            lambda values:values[0].update(samples_ns=[True,100,101]),
            lambda values:values.append(deepcopy(values[0])),
            lambda values:values[0].update(started_utc="2026-10-18T08:00:00")]
        for change in mutations:
            with self.subTest(change=change):
                source=rows();change(source)
                with self.assertRaises(ValueError):evaluate(source)


if __name__=="__main__":unittest.main()
