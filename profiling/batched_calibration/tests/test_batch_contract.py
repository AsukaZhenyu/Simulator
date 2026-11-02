"""Synthetic batch-unit, evidence separation and failure tests; no native execution."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "run.py"
SPEC = importlib.util.spec_from_file_location("batch_calibration_runner_reference", SOURCE)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def suite(batch=16):
    return runner.validate_suite(dict(
        format_version=runner.SCHEMA, batch_repetitions=batch,
        cases=[dict(id="tiny_ffn", family="ffn", backend="hybrid", dtype="f32",
                    hidden=8, intermediate=16, cpu_intermediate=4, tokens=1,
                    history=0, heads=0, kv_heads=0, head_dim=0)],
        protocol=dict(threads=1, cpu_id=0, device=0, warmup=1, samples=3,
                      min_warmup_ms=0, cuda_graphs="disabled"),
        profile_seeds=[11], heldout_seeds=[22], repeats=2,
        quality=dict(max_within_process_cv=.15, max_between_process_cv=.20,
                     max_heldout_relative_error=.20),
        bootstrap=dict(iterations=100, seed=7)))


def fake_payload(declared):
    case = declared["cases"][0]
    row = dict(id="profile-tiny_ffn-s11-r0", case=case, split="profile", seed=11, repeat=0)
    payload = {key: value for key, value in case.items() if key != "id"}
    payload.update(declared["protocol"])
    payload.update(schema="execution-transform-native-batch-0.1",
        timing_unit="completed_work_batch_total", batch_repetitions=declared["batch_repetitions"],
        seed=11, timer="steady_clock_completed_work", matmul_precision="GGML_PREC_F32_explicit",
        nvidia_tf32_override="0", dispatch_cpu_id=2, affinity_variant="dual_logical_cpu_explicit",
        observed_affinity=dict(process_mask=5, main_thread_mask=1, main_thread_group=0,
            dispatch_worker_created=True, dispatch_thread_mask=4, dispatch_thread_group=0),
        profile_gpu_shard_thread="dispatch_worker_same_as_parallel_strategy",
        profile_gpu_shard_affinity_mask=4, profile_gpu_shard_affinity_group=0,
        gpu_capacity_proved=False, control_overhead_separately_identified=False,
        boundary=dict(output_location="dram", merge_device="cpu",
            input_location="all_required_immutable_inputs_already_resident_in_both_cpu_and_gpu",
            weights="generated_f32_not_model_weights"))
    payload["batch_wall_ns"] = {metric: [1600, 1600, 1600]
        for metric in runner.base.expected_metrics(case) | runner.WEIGHT_METRICS}
    checks = {"gpu/resident/output", "hybrid/resident/output", "hybrid/preload/output"}
    checks |= {f"{device}/shard/{stage}" for device in ("cpu", "gpu")
               for stage in ("gate", "up", "activated", "output")}
    payload["checks"] = {name: dict(passed=True, max_abs_error=0.0,
        max_relative_error=0.0, elements=32, atol=.00001, rtol=.0001) for name in checks}
    payload["all_checks_passed"] = True
    # Fixed hand-specified F32 geometry: H=8, I=16, CPU/GPU cuts 4/12.
    payload["ports"] = {}
    for prefix, matrix_bytes, intermediate_bytes in (
            ("cpu/shard/", 128, 16), ("gpu/shard/", 384, 48), ("gpu/full/", 512, 64)):
        for name, count in dict(input=32, gate_weight=matrix_bytes, up_weight=matrix_bytes,
                down_weight=matrix_bytes, gate=intermediate_bytes, up=intermediate_bytes,
                activated=intermediate_bytes, output=32).items():
            payload["ports"][prefix + name] = dict(bytes=count, input_fixture_fnv1a64="0123456789abcdef")
    payload["workspaces"] = {
        "cpu/shard/gate": dict(known_gallocr_plus_cplan_bytes=4, workspace_bytes=4,
                               source="gallocr_plus_explicit_cplan"),
        "gpu/shard/gate": dict(known_gallocr_plus_cplan_bytes=4, workspace_bytes=None,
                               source="unknown_private_cuda_scratch")}
    return row, payload


class BatchContractTests(unittest.TestCase):
    def test_raw_batch_totals_are_unchanged_by_explicit_normalization(self):
        declared = suite()
        row, payload = fake_payload(declared)
        payload["batch_wall_ns"]["cpu/sum"] = [1599, 1600, 1601]
        original = deepcopy(payload)
        result = runner.normalized_payload(payload, row, declared, 2)
        self.assertEqual(payload, original)
        self.assertEqual(result["cpu/sum"], [100, 100, 100])
        self.assertNotIn("wall_ns", payload)
        self.assertEqual(payload["batch_wall_ns"]["cpu/sum"], [1599, 1600, 1601])
        self.assertEqual(set(result), runner.base.expected_metrics(row["case"]) | runner.WEIGHT_METRICS)

    def test_schema_unit_repeat_count_and_old_raw_key_are_rejected(self):
        declared = suite()
        for mutation in (dict(schema="execution-transform-native-0.1"),
                         dict(timing_unit="single_invocation"), dict(batch_repetitions=8),
                         dict(wall_ns={})):
            with self.subTest(mutation=mutation):
                row, payload = fake_payload(declared); payload.update(mutation)
                with self.assertRaises(ValueError):
                    runner.normalized_payload(payload, row, declared, 2)

    def test_missing_independent_weight_metric_is_not_filled_from_aggregate(self):
        declared = suite(); row, payload = fake_payload(declared)
        del payload["batch_wall_ns"]["hybrid/preload_gate_weight_h2d"]
        with self.assertRaisesRegex(ValueError, "metric coverage"):
            runner.normalized_payload(payload, row, declared, 2)

    def test_boolean_repetition_metadata_is_rejected_even_when_declared_count_is_one(self):
        declared = suite(1); row, payload = fake_payload(declared)
        payload["batch_repetitions"] = True
        with self.assertRaisesRegex(ValueError, "repetition mismatch"):
            runner.normalized_payload(payload, row, declared, 2)

    def test_unrepresentable_batch_mean_is_rejected_instead_of_clamped(self):
        declared = suite(); row, payload = fake_payload(declared)
        payload["batch_wall_ns"]["cpu/sum"] = [1, 1, 1]
        with self.assertRaisesRegex(ValueError, "not representable"):
            runner.normalized_payload(payload, row, declared, 2)

    def test_raw_sample_type_positivity_and_count_are_strict(self):
        declared = suite()
        for values in ([True, 1600, 1600], [1600.0, 1600, 1600], [0, 1600, 1600],
                       [-1, 1600, 1600], [1600, 1600], [1600] * 4):
            with self.subTest(values=values):
                row, payload = fake_payload(declared)
                payload["batch_wall_ns"]["cpu/sum"] = values
                with self.assertRaises(ValueError):
                    runner.normalized_payload(payload, row, declared, 2)

    def test_numeric_failure_precision_and_affinity_cannot_be_normalized_away(self):
        declared = suite()
        for change in ("numeric", "scratch", "tf32", "dispatch"):
            with self.subTest(change=change):
                row, payload = fake_payload(declared)
                if change == "numeric":
                    payload["checks"]["hybrid/resident/output"]["passed"] = False
                elif change == "scratch":
                    payload["workspaces"]["gpu/shard/gate"]["workspace_bytes"] = 0
                elif change == "tf32":
                    payload["nvidia_tf32_override"] = "1"
                else:
                    payload["profile_gpu_shard_affinity_mask"] = 1
                with self.assertRaises(ValueError):
                    runner.normalized_payload(payload, row, declared, 2)

    def test_preload_recipe_uses_three_independent_transfer_prices(self):
        terms = runner.a.terms(runner.recipes(suite()["cases"][0])["hybrid/preload"])
        self.assertTrue(runner.WEIGHT_METRICS <= terms)
        self.assertNotIn("hybrid/preload_weights", terms)
        self.assertNotIn("hybrid/resident", terms)
        self.assertNotIn("hybrid/parallel_branches", terms)

    def test_heldout_samples_and_profile_strategy_totals_never_enter_frozen_prices(self):
        declared = suite(); case = declared["cases"][0]
        job, payload = fake_payload(declared)
        metrics = runner.normalized_payload(payload, job, declared, 2)
        rows = [dict(id="profile-0", case=case, split="profile", seed=11, repeat=0,
                     status="measured", metrics=metrics)]
        recipes = {case["id"]: runner.recipes(case)}
        aliases = {case["id"]: runner.base.observed_aliases(case)}
        before = runner.a.freeze_model(rows, recipes, aliases)
        rows[0]["metrics"] = deepcopy(metrics)
        for key in ("gpu/resident", "hybrid/resident", "hybrid/preload", "hybrid/parallel_branches"):
            rows[0]["metrics"][key] = [999999, 999999, 999999]
        rows.append(dict(id="heldout-0", case=case, split="heldout", seed=22, repeat=0,
            status="measured", metrics={key: [777777, 777777, 777777] for key in metrics}))
        self.assertEqual(before, runner.a.freeze_model(rows, recipes, aliases))

    def test_failed_process_blocks_calibration_even_with_exact_remaining_predictions(self):
        declared = suite(); case = declared["cases"][0]
        job, payload = fake_payload(declared)
        metrics = runner.normalized_payload(payload, job, declared, 2)
        rows = [dict(id=f"profile-{repeat}", case=case, split="profile", seed=11,
            repeat=repeat, status="measured", metrics=deepcopy(metrics)) for repeat in (0, 1)]
        targets = {"gpu/resident": 500, "hybrid/resident": 600, "hybrid/preload": 900}
        for repeat in (0, 1):
            truth = deepcopy(metrics)
            truth.update({name: [value] * 3 for name, value in targets.items()})
            rows.append(dict(id=f"heldout-{repeat}", case=case, split="heldout", seed=22,
                             repeat=repeat, status="measured", metrics=truth))
        recipes = {case["id"]: runner.recipes(case)}
        aliases = {case["id"]: runner.base.observed_aliases(case)}
        model = runner.a.freeze_model(rows, recipes, aliases)
        clean = runner.a.evaluate(rows, model, declared["quality"], declared["bootstrap"],
            scope="synthetic unit test", frozen_before_heldout=True)
        primary = [row for row in clean["results"] if "@" not in row["strategy"]]
        self.assertTrue(all(row["calibrated"] for row in primary))
        rows.append(dict(id="failed-extra-process", case=case, split="heldout", seed=22,
                         repeat=2, status="failed", error="synthetic child failure"))
        failed = runner.a.evaluate(rows, model, declared["quality"], declared["bootstrap"],
            scope="synthetic unit test", frozen_before_heldout=True)
        self.assertEqual(failed["failed_native_rows"], ["failed-extra-process"])
        self.assertFalse(any(row["calibrated"] for row in failed["results"]))
        self.assertTrue(all(not row["gates"]["numeric_and_process_coverage"] for row in failed["results"]))


if __name__ == "__main__":
    unittest.main()
