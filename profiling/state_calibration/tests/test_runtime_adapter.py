"""Measured-state cost to common-runtime bridge; no hardware execution."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import unittest

from profiling.state_calibration.model import UnknownCost, fit_model
from profiling.state_calibration.runtime_adapter import MeasuredTargetRuntime, load_measured_target


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def save(path, value):
    raw = canonical(value)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def campaign(root, *, values=None, unstable=None, numerical_failure=False, target_kind="attention_pv", device="gpu",
             primary_boundary="legacy_double_sync", native_boundary="legacy_double_sync", job_boundary=None):
    values = values or {"baseline": [100] * 4, "cpu_replay_load": [150] * 4, "gpu_replay_load": [300] * 4}
    action = dict(kind="COMPUTE", device=device, target_id="recorded-pv")
    binding = dict(operation_id="recorded-pv", kind=target_kind, shape=dict(tokens=16, visible_history=0))
    program = dict(actions=[action], bindings=[binding])
    program_path = root / "program.json"
    program_sha = save(program_path, program)
    rows = []
    manifest = dict(programs={"case-A": {}}, protocol=dict(primary_execution_boundary=primary_boundary))
    for condition, samples in values.items():
        for repeat in range(3):
            ident = f"profile-{condition}-{repeat}"
            program_key = f"profile-{repeat}"
            manifest["programs"]["case-A"][program_key] = dict(path=str(program_path), sha256=program_sha)
            sample_values = unstable if unstable is not None and condition == "gpu_replay_load" and repeat == 0 else samples
            job = dict(id=ident, split="profile", seed=1, repeat=repeat, case_id="case-A", mode="isolated",
                       instrumentation="stamped", telemetry=True, condition=condition, program_key=program_key)
            if job_boundary is not None:
                job["execution_boundary"] = job_boundary
            native = dict(actions=[action], target_action_index=0, mode="isolated", execution_boundary=native_boundary,
                target=dict(action=action,
                binding=binding, graph_nodes=[dict(op="MUL_MAT", type="f32", requested_precision="GGML_PREC_F32")],
                workspace_bytes=1234, workspace_scope="known allocator only; private scratch unknown"))
            native_sha = save(root / "jobs" / ident / "native.json", native)
            summary = dict(target_samples_ns=sample_values, iteration_samples_ns=sample_values,
                target_median_ns=statistics.median(sample_values), iteration_median_ns=statistics.median(sample_values),
                target_cv=statistics.pstdev(sample_values) / statistics.mean(sample_values),
                iteration_cv=statistics.pstdev(sample_values) / statistics.mean(sample_values),
                native_numeric_passed=not numerical_failure, condition_verified=True,
                alignment_model_ready=True, process_id=1234 + repeat, native_sha256=native_sha)
            rows.append(dict(job=job, summary=summary, status="measured", program_sha256=program_sha,
                command=["probe.exe", str(program_path), "isolated", "0"], artifacts_sha256={"native.json": native_sha}))
    manifest_sha = save(root / "manifest.json", manifest)
    model = fit_model(rows, {"manifest_sha256": manifest_sha}, "2026-10-18T00:00:00+00:00")
    save(root / "profile-rows.json", rows)
    save(root / "frozen-model.json", model)
    return model


class MeasuredTargetAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def target(self, **changes):
        return load_measured_target(self.root, case_id="case-A", mode="isolated", **changes)

    def test_exact_conditional_lookup_drives_the_actual_common_runtime(self):
        model = campaign(self.root)
        runtime = self.target().build_session(initial_condition="gpu_replay_load")
        result = runtime.run()
        self.assertEqual(result.evaluation.makespan_ns, 300)
        self.assertEqual(result.evaluation.events[0].target_id, "recorded-pv")
        self.assertEqual(result.nominal_ends[0], 100)
        evidence = runtime.provenance()
        self.assertEqual(evidence["model_sha256"], model["model_sha256"])
        self.assertEqual(evidence["regime_events"][0]["adapter_service_rate"], dict(numerator=1, denominator=3))
        self.assertEqual(len(evidence["target_evidence"]["source_processes"]), 9)
        self.assertTrue(evidence["all_used_profile_prices_accepted"])
        self.assertFalse(evidence["physical_capacity_claimed"])
        self.assertFalse(evidence["native_target_memory_modeled"])
        self.assertFalse(evidence["universal_device_multiplier"])
        self.assertIsNone(evidence["private_cuda_workspace_bytes"])

    def test_slowdown_and_restore_preserve_exact_remaining_reference_work(self):
        campaign(self.root)
        runtime = self.target().build_session(changes=((40, "gpu_replay_load"), (120, "baseline")))
        self.assertTrue(runtime.start().ok)
        self.assertEqual(runtime.advance().t_ns, 40)
        self.assertEqual(runtime.advance().t_ns, 120)
        progress, = runtime.result().compute_progress
        self.assertEqual(Fraction(progress.remaining_reference_ns_numerator,
                                  progress.remaining_reference_ns_denominator), Fraction(100, 3))
        self.assertEqual(runtime.run().evaluation.makespan_ns, 154)
        work = sum(Fraction(row.reference_service_ns_numerator, row.reference_service_ns_denominator)
                   for row in runtime.result().service_intervals)
        self.assertEqual(work, 100)

    def test_half_nanosecond_price_uses_exact_median_and_rounds_only_completion(self):
        campaign(self.root, values={"baseline": [100, 101], "gpu_replay_load": [300, 301]})
        runtime = self.target().build_session(initial_condition="gpu_replay_load")
        self.assertEqual(runtime.run().evaluation.makespan_ns, 301)
        evidence = runtime.provenance()
        self.assertEqual(evidence["reference_work_ns"], 101)
        self.assertEqual(evidence["regime_events"][0]["adapter_service_rate"], dict(numerator=202, denominator=601))
        self.assertEqual(evidence["regime_events"][0]["price"]["price_ns_exact"], dict(numerator=601, denominator=2))

    def test_stability_failure_requires_diagnostic_optin_and_keeps_gates(self):
        campaign(self.root, unstable=[1, 300, 300, 900])
        target = self.target()
        with self.assertRaisesRegex(UnknownCost, "gates failed"):
            target.build_session(initial_condition="gpu_replay_load")
        runtime = target.build_session(initial_condition="gpu_replay_load", allow_unaccepted_diagnostic=True)
        self.assertEqual(runtime.run().evaluation.makespan_ns, 300)
        self.assertTrue(runtime.provenance()["diagnostic_only"])
        self.assertIn("one_or_more_within_process_cv_gates_failed",
                      runtime.provenance()["regime_events"][0]["price"]["profile_gate_failures"])

    def test_unknown_or_numerically_failed_price_is_never_replaced(self):
        campaign(self.root)
        with self.assertRaises(UnknownCost):
            self.target().build_session(initial_condition="unseen", allow_unaccepted_diagnostic=True)
        campaign(self.root, numerical_failure=True)
        with self.assertRaises(UnknownCost):
            self.target()

    def test_iteration_and_cpu_metrics_cannot_become_gpu_target_rates(self):
        campaign(self.root)
        for changes in (dict(metric="iteration"), dict(device="cpu")):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "cannot become"):
                self.target(**changes)

    def test_wrong_target_kind_or_device_is_rejected_even_with_bound_files(self):
        for kwargs in (dict(target_kind="ffn_down"), dict(device="cpu")):
            campaign(self.root, **kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "attention_pv target only"):
                self.target()

    def test_execution_boundary_mismatch_or_unknown_is_rejected_with_bound_files(self):
        for kwargs in (dict(primary_boundary=None), dict(primary_boundary="async_one_sync"),
                       dict(native_boundary=None), dict(native_boundary="async_events")):
            campaign(self.root, **kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "execution boundary"):
                self.target()
        # Explicit nonlegacy profile jobs are rejected even earlier by the
        # cost-model contract. Do not bypass that guard to manufacture a model
        # just for testing the downstream adapter.
        with self.subTest(job_boundary="async_one_sync"), self.assertRaisesRegex(ValueError, "nonlegacy execution boundaries"):
            campaign(self.root, job_boundary="async_one_sync")

    def test_exact_context_cannot_be_substituted(self):
        campaign(self.root)
        with self.assertRaises(UnknownCost):
            load_measured_target(self.root, case_id="case-A", mode="mixed")
        with self.assertRaises(UnknownCost):
            self.target(instrumentation="bare")

    def test_native_program_and_profile_artifact_mutations_are_rejected(self):
        for filename in ("program.json", "jobs/profile-baseline-0/native.json", "profile-rows.json", "manifest.json"):
            campaign(self.root)
            path = self.root / filename
            raw = json.loads(path.read_bytes())
            if isinstance(raw, dict):
                raw["mutation"] = True
            else:
                raw[0]["mutation"] = True
            save(path, raw)
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                self.target()

    def test_no_arbitrary_runtime_injection_or_command_submission_api(self):
        with self.assertRaisesRegex(ValueError, "arbitrary runtime injection"):
            MeasuredTargetRuntime(object(), "anything", "{}")
        campaign(self.root)
        runtime = self.target().build_session()
        self.assertFalse(hasattr(runtime, "submit"))
        self.assertFalse(hasattr(runtime, "submit_transfer"))
        self.assertFalse(hasattr(runtime, "build_measured_target_event"))
        with self.assertRaisesRegex(ValueError, "before advancing"):
            runtime.advance()
        runtime.start()
        with self.assertRaisesRegex(ValueError, "already started"):
            runtime.start()

    def test_future_condition_times_are_not_in_current_public_observation(self):
        campaign(self.root)
        runtime = self.target().build_session(changes=((77, "gpu_replay_load"),))
        runtime.start()
        value = json.dumps(runtime.observe().__dict__, default=lambda x: x.__dict__)
        self.assertNotIn("77", value)
        self.assertNotIn("regime_events", value)
        self.assertEqual(runtime.observe().gpu_service_rate_numerator, 1)

    def test_bad_timeline_and_nonboolean_diagnostic_flag_are_rejected(self):
        campaign(self.root)
        target = self.target()
        for changes in (((0, "baseline"),), ((2, "baseline"), (2, "gpu_replay_load")),
                        ((True, "baseline"),), ([2, "baseline"],)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                target.build_session(changes=changes)
        with self.assertRaises(ValueError):
            target.build_session(allow_unaccepted_diagnostic=1)


if __name__ == "__main__":
    unittest.main()
