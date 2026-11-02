import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from profiling.physical_campaign_audit import (
    _background, _counter, _gate, _observer, _posttarget_lifecycle, _reference_audit, audit_campaign,
)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def background(end=1000):
    return dict(format_version="state-response-native-v1", role="cpu-worker", process_id=7,
        program_sha256="program", owned_Executor_destructor_returned=True,
        clock=dict(clock_id="windows_qpc", frequency_hz=1_000_000_000),
        go_schedule=dict(schema="state-response-go-v1", start_tick=1, end_tick=1000,
            clock=dict(clock_id="windows_qpc", frequency_hz=1_000_000_000),
            phases=[dict(begin_tick=1, end_tick=1000, cpu_replay=1)]),
        sampling_begin_tick=1, sampling_end_tick=end, end_reason="cooperative_stop",
        blocks=[dict(iterations=[dict(begin_tick=100, end_tick=180, enabled_check_tick=99),
                                 dict(begin_tick=220, end_tick=300, enabled_check_tick=219)])])


def physical():
    return dict(schema="physical-stage-native-1", instrumentation="bare", process_id=1,
        program_sha256="program", owned_Executor_destructor_returned=True,
        clock=dict(clock_id="windows_qpc_floor_ns", clock_origin="absolute_qpc_zero", frequency_hz=10_000_000),
        sampling_begin_ns=100, sampling_end_ns=400, samples_requested=2, samples_completed=2,
        iterations=[dict(begin_ns=110, end_ns=210), dict(begin_ns=220, end_ns=320)],
        cupti=dict(enabled=False, records=[]),
        native_repeat_consistency=dict(passed=True), warmup_repeat_consistency=dict(passed=True),
        post_replay_consistency=dict(passed=True),
        batch_cpu=dict(begin_ns=100, end_ns=400, main_thread_active_cpu_ns=312, counter_quantization_retained=True))


def campaign(root):
    job = dict(id="main", split="profile", program_key="profile-0", instrumentation="bare", references=False,
               cpu_background=0, gpu_background=0, samples=2, condition="idle")
    binding = dict(program_sha256="program")
    manifest = dict(schema="physical-service-campaign-1", sources={},
        source_inventory_sha256=hashlib.sha256(b"{}").hexdigest(), jobs=[job], programs={"profile-0": binding},
        protocol=dict(max_reference_hold_ns=1000, observer_relative_error_max=0.05, minimum_independent_runs=3))
    save(root/"manifest.json", manifest)
    raw_path = root/"jobs/main/main/result.json"
    sha = save(raw_path, physical())
    receipt = dict(name="main", pid=1, returncode=0, process_exited_verified=True, forced_termination=False)
    save(root/"jobs/main/main/process.json", receipt)
    execution = dict(job=job, program=binding, source_inventory_sha256=manifest["source_inventory_sha256"], status="completed",
        processes=[receipt], cleanup_verified=True, numerical=dict(main=dict(passed=True)),
        raw_files=dict(main=dict(path=str(raw_path), sha256=sha)))
    save(root/"jobs/main/execution.json", execution)
    return manifest, execution


class IntervalAuditTests(unittest.TestCase):
    def test_go_is_not_observed_call_occupancy(self):
        row = _background(background(), "cpu-worker", 7, "program", [("main", 100, 300)], 1_000_000_000)
        self.assertTrue(row["lifecycle_qualified"])
        observed = row["windows"][0]
        self.assertEqual(observed["requested_on_fraction"], 1)
        self.assertEqual(observed["worker_host_call_union_overlap_fraction"], 0.8)
        self.assertFalse(row["physical_service_or_utilization_inferred"])

    def test_requested_go_does_not_hide_early_sampling_end(self):
        raw = background(end=300)
        row = _background(raw, "cpu-worker", 7, "program", [("main", 100, 500)], 1_000_000_000)
        self.assertFalse(row["lifecycle_qualified"])
        self.assertEqual(row["windows"][0]["requested_on_fraction"], 1)
        self.assertFalse(row["windows"][0]["actual_sampling_lifecycle_covers_window"])

    def test_unenabled_worker_call_not_relabelled_busy(self):
        raw = background()
        raw["blocks"][0]["iterations"][0]["enabled_check_tick"] = 0
        row = _background(raw, "cpu-worker", 7, "program", [("main", 100, 300)], 1_000_000_000)
        self.assertFalse(row["lifecycle_qualified"])
        self.assertIn("ON phase", " ".join(row["errors"]))

    def test_cross_clock_rejected(self):
        row = _background(background(), "cpu-worker", 7, "program", [("main", 100, 300)], 1000)
        self.assertFalse(row["lifecycle_qualified"])

    def test_phase_outside_go_rejected(self):
        raw = background()
        raw["go_schedule"]["phases"][0]["end_tick"] = 1001
        row = _background(raw, "cpu-worker", 7, "program", [("main", 100, 300)], 1_000_000_000)
        self.assertFalse(row["lifecycle_qualified"])

    def test_active_counter_above_wall_is_retained(self):
        row = _counter(physical())
        self.assertTrue(row["active_exceeds_wall"])
        self.assertEqual(row["main_thread_active_wall_ratio"], 1.04)
        self.assertFalse(row["available_cpu_service_inferred"])

    def test_reference_windows_and_stale_hold(self):
        actors = {name: dict(begin_ns=begin, end_ns=begin+10) for name, begin in
                  [("before-gpu", 10), ("before-cpu", 30), ("main", 100), ("after-gpu", 300), ("after-cpu", 500)]}
        row = _reference_audit(actors, True, 75)
        self.assertTrue(row["all_measured_windows_disjoint"])
        self.assertEqual(row["per_resource"]["gpu"]["before_end_to_main_begin_ns"], 80)
        self.assertFalse(row["per_resource"]["gpu"]["before_reference_causal_and_within_declared_hold"])
        self.assertFalse(row["after_reference_used_for_forward_rate"])
        actors["before-cpu"]["end_ns"] = 105
        self.assertFalse(_reference_audit(actors, True, 1000)["all_measured_windows_disjoint"])


class CampaignAuditTests(unittest.TestCase):
    def test_empty_gate_is_unknown(self):
        self.assertEqual(_gate([]), "unknown")

    def test_integrity_and_observer_are_separate(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            campaign(root)
            report = audit_campaign(root)
            self.assertEqual(report["counts"]["diagnostic_included"], 1)
            self.assertEqual(report["observer"]["gates"]["cupti_vs_bare_without_references"]["status"], "unknown")
            self.assertFalse(report["qualifications"]["hardware_accuracy_qualified"])
            self.assertTrue(report["jobs"][0]["actors"]["main"]["cpu_counter"]["active_exceeds_wall"])

    def test_raw_hash_mismatch_excludes_without_timing_selection(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            campaign(root)
            raw_path = root/"jobs/main/main/result.json"
            raw_path.write_bytes(raw_path.read_bytes()+b"\n")
            row = audit_campaign(root)["jobs"][0]
            self.assertFalse(row["diagnostic_included"])
            self.assertIn("raw_hashes", row["exclusions"])

    def test_inventory_digest_failure_excludes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            manifest, _ = campaign(root)
            manifest["sources"]["unbound"] = "changed"
            save(root/"manifest.json", manifest)
            report = audit_campaign(root)
            self.assertFalse(report["source_inventory_digest_matches"])
            self.assertFalse(report["jobs"][0]["diagnostic_included"])

    def test_incomplete_execution_preserves_receipts(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            campaign(root)
            (root/"jobs/main/execution.json").write_bytes(b'{"incomplete":')
            row = audit_campaign(root)["jobs"][0]
            self.assertEqual(row["state"], "unreadable_execution")
            self.assertFalse(row["diagnostic_included"])
            self.assertEqual(len(row["process_receipts"]), 1)

    def test_constructor_failure_cannot_hide_in_execution_cleanup_claim(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            campaign(root)
            save(root/"jobs/main/after-gpu/process.json", dict(name="after-gpu", pid=2,
                returncode=1, forced_termination=True, process_exited_verified=True))
            row = audit_campaign(root)["jobs"][0]
            self.assertTrue(row["execution_cleanup_verified_claim"])
            self.assertFalse(row["diagnostic_included"])
            self.assertIn("all_process_receipts_clean", row["exclusions"])
            self.assertFalse(row["process_receipts"][0]["listed_in_execution"])

    def test_retained_readiness_failure_and_exact_retry(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            manifest, execution = campaign(root)
            execution.update(status="failed", error="TimeoutError owned child readiness deadline expired")
            failure_path = root/"jobs/main/execution.json"
            failure_sha = save(failure_path, execution)
            _, manifest_sha = read_saved(root/"manifest.json")
            save(root/"job-replacements.json", dict(schema="physical-job-replacements-1", maximum_retries=1,
                parent_manifest_sha256=manifest_sha, replacements=[dict(original_id="main",
                replacement_job=dict(manifest["jobs"][0], id="main-retry1"), failed_execution_path=str(failure_path),
                failed_execution_sha256=failure_sha, selection_reason="readiness infrastructure only")]))
            report = audit_campaign(root)
            self.assertEqual(report["jobs"][0]["state"], "pending")
            self.assertEqual(len(report["retained_failed_attempts"]), 1)
            self.assertFalse(report["retained_failed_attempts"][0]["diagnostic_included"])

    def test_observer_failure_not_corrected_or_filtered(self):
        rows = []
        for repeat in range(3):
            for instrument, refs, duration in [("bare", False, 100), ("cupti", False, 120), ("cupti", True, 121)]:
                rows.append(dict(split="observer", program_key=str(repeat), instrumentation=instrument, references=refs,
                    job_id=f"{repeat}-{instrument}-{refs}", diagnostic_included=True, main_invocation_median_ns=duration))
        row = _observer(rows, dict(observer_relative_error_max=0.05, minimum_independent_runs=3))
        self.assertEqual(row["gates"]["cupti_vs_bare_without_references"]["status"], "fail")
        self.assertEqual(row["gates"]["references_with_cupti_vs_cupti_without_references"]["status"], "pass")
        self.assertFalse(row["bare_accuracy_claim_permitted"])
        self.assertFalse(row["tracer_correction_fitted"])


def posttarget_fixture(root, *, stop=215, exited=220):
    directory = root/"jobs/amended"
    actors = dict(main=dict(process_id=1, begin_ns=100, end_ns=200, invocation_count=2,
        clock=dict(frequency_hz=1_000_000_000)))
    processes = [dict(name="cpu-worker", pid=7, stop_requested_tick=stop, exit_observed_tick=exited,
        launch_tick=1, returncode=0, forced_termination=False, process_exited_verified=True),
        dict(name="after-gpu", pid=8, launch_tick=250), dict(name="after-cpu", pid=9, launch_tick=350)]
    raw = background(end=215)
    raw["blocks"][0]["iterations"][1].update(begin_tick=190, end_tick=210, enabled_check_tick=189)
    windows = [("main", 100, 200), ("after-gpu", 300, 310), ("after-cpu", 400, 410)]
    bg = _background(raw, "cpu-worker", 7, "program", windows, 1_000_000_000, required_window_names={"main"})
    index_path = root/"posttarget-job-index.json"
    index_sha = save(index_path, {})
    sidecar = dict(schema="physical-posttarget-cleanup-1", job_id="amended", index=dict(path=str(index_path), sha256=index_sha),
        clock=dict(clock_id="windows_qpc", frequency_hz=1_000_000_000),
        main_sampling_complete=dict(schema="physical-stages-sampling-complete-1", process_id=1,
            sampling_begin_ns=100, sampling_end_ns=200, completed_samples=2),
        main_completion_observed_tick=205, background_stop_begin_tick=210, background_stop_end_tick=max(230, exited),
        background_processes=[processes[0]], background_count=1,
        posttarget_after_reference_scope="unloaded_drift_only", target_sampling_altered=False,
        before_reference_conditions_altered=False, numerical_or_prediction_selection=False)
    sidecar_path = directory/"posttarget-cleanup.json"
    sidecar_sha = save(sidecar_path, sidecar)
    execution = dict(posttarget_cleanup_applied=True, posttarget_after_reference_scope="unloaded_drift_only",
        posttarget_cleanup_receipt=dict(path=str(sidecar_path), sha256=sidecar_sha))
    return actors, processes, {"cpu-worker": bg}, execution, directory


class PosttargetAmendmentTests(unittest.TestCase):
    def test_after_reference_background_missing_is_declared_not_hidden(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            args = posttarget_fixture(Path(directory))
            background_row = args[2]["cpu-worker"]
            self.assertTrue(background_row["lifecycle_qualified"])
            after = [row for row in background_row["windows"] if row["actor"].startswith("after-")]
            self.assertTrue(all(row["actual_sampling_lifecycle_coverage_ns"] == 0 for row in after))
            self.assertTrue(all(row["requested_on_fraction"] == 1 for row in after))
            self.assertTrue(all(not row["required_for_constant_background_admission"] for row in after))
            audit = _posttarget_lifecycle(*args)
            self.assertTrue(audit["qualified"], audit)
            self.assertFalse(audit["after_reference_same_load_drift_claimed"])

    def test_stop_request_before_main_end_excluded_even_if_call_finishes_later(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            audit = _posttarget_lifecycle(*posttarget_fixture(Path(directory), stop=195))
            self.assertFalse(audit["qualified"])
            self.assertFalse(audit["backgrounds"][0]["checks"]["stop_requested_after_main"])

    def test_background_exit_after_reference_construction_excluded(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            audit = _posttarget_lifecycle(*posttarget_fixture(Path(directory), exited=260))
            self.assertFalse(audit["qualified"])
            self.assertFalse(audit["backgrounds"][0]["checks"]["exit_before_after_construction"])

    def test_receipt_main_completion_identity_must_match_observation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            args = posttarget_fixture(Path(directory))
            binding = args[3]["posttarget_cleanup_receipt"]
            path = Path(binding["path"])
            content, _ = read_saved(path)
            content["main_sampling_complete"]["process_id"] = 123
            binding["sha256"] = save(path, content)
            self.assertFalse(_posttarget_lifecycle(*args)["qualified"])

    def test_posttarget_index_preserves_all_jobs_and_frozen_binding(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            manifest, execution = campaign(root)
            execution["ended_at_utc"] = "2026-01-01T00:00:00+00:00"
            save(root/"jobs/main/execution.json", execution)
            heldout = dict(manifest["jobs"][0], id="heldout", split="heldout")
            manifest["jobs"].append(heldout)
            manifest_sha = save(root/"manifest.json", manifest)
            frozen_path = root/"frozen.json"
            frozen_sha = save(frozen_path, {})
            source_path = root/"operational_source.py"
            source_path.write_bytes(b"pass\n")
            source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
            save(root/"posttarget-job-index.json", dict(schema="physical-posttarget-job-index-1",
                parent_manifest_sha256=manifest_sha, frozen_file=dict(path=str(frozen_path), sha256=frozen_sha),
                jobs=manifest["jobs"], replacements=[], original_completed_job_ids=[], posttarget_cleanup_job_ids=["heldout"],
                posttarget_after_reference_scope="unloaded_drift_only", declared_at_utc="2026-01-02T00:00:00+00:00",
                source_receipts=[dict(path=str(source_path), sha256=source_sha)]))
            report = audit_campaign(root)
            self.assertEqual(report["counts"]["planned"], 2)
            self.assertEqual(report["posttarget_operational_amendment"]["changed_job_ids"], ["heldout"])
            self.assertEqual(report["jobs"][1]["state"], "pending")
            source_path.write_bytes(b"changed\n")
            with self.assertRaisesRegex(ValueError, "source receipt changed"):
                audit_campaign(root)


def read_saved(path):
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    unittest.main()
