"""Small synthetic evidence tests; no hardware, inference backend or NumPy."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
for folder in (ROOT, ROOT / "modeling"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from profiling.state_response import analysis as a

OFF = dict(cpu_replay=0, gpu_replay=0, host_memory_bytes=0, device_memory_bytes=0)
ON = dict(OFF, gpu_replay=1)
CONTEXT = dict(state_semantics="offered_load_controlled", backend="synthetic exact action", threadpool_poll=0)


class ResponseAnalysisTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "build").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="state-response-test-", dir=ROOT / "build")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.counter = 0

    def record(self, split="profile", *, durations=(10, 10, 20, 20), bare=False, events=None, seed=None,
               custom_context=None, go=None):
        self.counter += 1
        index = self.counter
        seed = index + (1000 if split == "heldout" else 0) if seed is None else seed
        program = dict(program_sha256=a.digest(dict(seed=seed)), seed=seed, initial_values={"x": [seed]},
            identity={"initial_values": a.digest(seed), "graph": "unchanged graph"},
            actions=[dict(kind="COMPUTE", target_id="A", device="gpu")], physical={"threads": 1})
        starts = [10, 40, 120, 170] if len(durations) == 4 else [10, 95, 150]
        iterations = [dict(iteration=i, begin_tick=begin, end_tick=begin + duration, elapsed_ns=duration)
                      for i, (begin, duration) in enumerate(zip(starts, durations))]
        actions = [] if bare else [dict(iteration=row["iteration"], action_index=0,
            begin_tick=row["begin_tick"], end_tick=row["end_tick"], elapsed_ns=row["elapsed_ns"])
            for row in iterations]
        native = dict(format_version="state-response-native-v1", role="main", program_sha256=program["program_sha256"],
            process_id=100 + index, actions=program["actions"], instrumentation="bare" if bare else "stamped",
            clock=dict(clock_id="synthetic-QPC", frequency_hz=1_000_000_000),
            blocks=[dict(iterations=iterations, action_intervals=actions)])
        if go is not None:
            native["go_schedule"] = go
        if events is None:
            events = [dict(begin_tick=0, end_tick=100, state=OFF), dict(begin_tick=100, end_tick=300, state=ON)]
        numerical = dict(passed=True, oracle="synthetic fixture only")
        directory = self.root / str(index)
        directory.mkdir()
        inputs = dict(program=program, native=native, events=events, numerical=numerical)
        bindings = {}
        for name, value in inputs.items():
            path = directory / (name + ".json")
            path.write_text(json.dumps(value), encoding="utf-8")
            bindings[name] = str(path.resolve())
        files = {name: a.file_sha(name) for name in bindings.values()}
        begin, end = ("2026-10-21T00:00:00Z", "2026-10-21T00:01:00Z") if split == "profile" else (
            "2026-10-21T02:00:00Z", "2026-10-21T02:01:00Z")
        record = a.make_run_record(program, native, events, context=CONTEXT if custom_context is None else custom_context,
            run_id=f"{split}-{index}", split=split,
            seed=seed, process_id=native["process_id"], started_at_utc=begin, ended_at_utc=end,
            source_files=files, numerical_verification=numerical, raw_bindings=bindings)
        return program, record

    def profile(self, **kwargs):
        rows = [self.record(**kwargs) for _ in range(3)]
        program, records = rows[0][0], [row[1] for row in rows]
        frozen = a.freeze_profile(program, records, context=CONTEXT, required_states=[OFF, ON],
            frozen_at_utc="2026-10-21T01:00:00Z", protocol={"planned_schedule_family": "synthetic-off-on"})
        return program, records, frozen

    def barrier(self, frozen):
        return dict(schema="state-response-heldout-barrier-1", frozen_artifact_sha256=frozen["artifact_sha256"],
            created_at_utc="2026-10-21T01:30:00Z",
            profile_record_sha256s=sorted(row["record_sha256"] for row in frozen["profile_runs"]))

    def test_numeric_state_is_not_utilization(self):
        self.assertEqual(a.validate_state(dict(OFF, cpu_replay=True))["cpu_replay"], 1)
        with self.assertRaises(ValueError):
            a.validate_state(dict(OFF, gpu_replay=50))

    def test_seed_initialization_does_not_change_contract(self):
        first, _ = self.record()
        second, _ = self.record()
        self.assertNotEqual(first["identity"]["initial_values"], second["identity"]["initial_values"])
        self.assertEqual(a.contract_identity(first), a.contract_identity(second))
        second["physical"]["threads"] = 2
        self.assertNotEqual(a.contract_identity(first), a.contract_identity(second))

    def test_full_trace_telescopes_including_zero_gaps(self):
        _, record = self.record()
        self.assertEqual([token["elapsed_ns"] for token in record["invocations"][0]["tokens"]], [0, 10, 0])
        for invocation in record["invocations"]:
            self.assertEqual(sum(token["elapsed_ns"] for token in invocation["tokens"]), invocation["elapsed_ns"])

    def test_bare_heldout_is_supported(self):
        _, record = self.record("heldout", bare=True)
        self.assertEqual(record["instrumentation"], "bare")
        self.assertTrue(all(row["tokens"] == [] for row in record["invocations"]))

    def test_bare_profile_cannot_estimate_internal_actions(self):
        with self.assertRaisesRegex(ValueError, "profile/trace"):
            self.record(bare=True)

    def test_freeze_exact_equal_process_median(self):
        program, _, frozen = self.profile()
        cells = [cell for cell in frozen["cells"] if cell["token"] == "action:0:duration"]
        self.assertEqual([a.exact(cell["median_ns"]) for cell in cells], [10, 20])
        self.assertTrue(frozen["source_qualified"])
        self.assertEqual(a.load_frozen(program, frozen), frozen)

    def test_unstable_numeric_value_is_retained_not_qualified(self):
        _, _, frozen = self.profile(durations=(1, 19, 2, 38))
        cells = [cell for cell in frozen["cells"] if cell["token"] == "action:0:duration"]
        self.assertFalse(frozen["source_qualified"])
        self.assertEqual(a.exact(cells[0]["median_ns"]), 10)
        self.assertIn("within_process_cv_gate_failed_or_missing", cells[0]["evidence_issues"])

    def test_unknown_cell_remains_unknown(self):
        program, records, _ = self.profile()
        frozen = a.freeze_profile(program, records, context=CONTEXT, required_states=[dict(OFF, host_memory_bytes=100)],
            frozen_at_utc="2026-10-21T01:00:00Z", protocol={"grid": "unsupported"})
        self.assertTrue(all(cell["median_ns"] is None for cell in frozen["cells"]))
        self.assertFalse(frozen["source_qualified"])

    def test_reused_profile_raw_is_rejected(self):
        program, record = self.record()
        renamed = deepcopy(record)
        renamed["run_id"] = "renamed"
        renamed["record_sha256"] = a.digest({key: value for key, value in renamed.items() if key != "record_sha256"})
        with self.assertRaises(ValueError):
            a.freeze_profile(program, [record, renamed], context=CONTEXT, required_states=[OFF],
                frozen_at_utc="2026-10-21T01:00:00Z", protocol={"synthetic": True})

    def test_raw_tamper_is_rejected(self):
        program, _, frozen = self.profile()
        Path(frozen["profile_runs"][0]["raw_bindings"]["native"]).write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source bytes changed"):
            a.load_frozen(program, frozen)

    def test_rehashed_fake_normalized_scalar_is_rejected(self):
        program, records, _ = self.profile()
        records[0]["invocations"][0]["tokens"][1]["elapsed_ns"] = 999
        records[0]["record_sha256"] = a.digest({key: value for key, value in records[0].items() if key != "record_sha256"})
        with self.assertRaisesRegex(ValueError, "normalization does not reconstruct"):
            a.freeze_profile(program, records, context=CONTEXT, required_states=[OFF],
                frozen_at_utc="2026-10-21T01:00:00Z", protocol={"synthetic": True})

    def test_rehashed_fake_frozen_statistic_is_rejected(self):
        program, _, frozen = self.profile()
        frozen["cells"][0]["median_ns"] = a.encoded(999)
        frozen["artifact_sha256"] = a.digest({key: value for key, value in frozen.items() if key != "artifact_sha256"})
        with self.assertRaisesRegex(ValueError, "statistics do not reconstruct"):
            a.load_frozen(program, frozen)

    def test_new_temporal_order_is_predicted_by_model(self):
        program, _, frozen = self.profile()
        heldout = [self.record("heldout", bare=True, durations=(10, 15, 20))[1] for _ in range(3)]
        result = a.evaluate_heldout(program, frozen, heldout, barrier=self.barrier(frozen))
        self.assertEqual([row["predicted_ns"] for row in result["predictions"][:3]], [10, 15, 20])
        self.assertTrue(result["statistical_protocol_accepted"])
        self.assertFalse(result["full_accuracy_qualified"])
        self.assertFalse(result["accepted"])
        self.assertEqual(result["coverage"]["crossing_independent_runs"], 3)
        self.assertTrue(all(row["model_result"] is None for row in result["predictions"]))

    def test_no_barrier_allows_diagnostic_only(self):
        program, _, frozen = self.profile()
        heldout = [self.record("heldout", bare=True, durations=(10, 15, 20))[1] for _ in range(3)]
        result = a.evaluate_heldout(program, frozen, heldout)
        self.assertTrue(result["median_error_within_threshold"])
        self.assertFalse(result["accepted"])

    def test_barrier_inventory_cannot_drop_profile(self):
        program, _, frozen = self.profile()
        heldout = [self.record("heldout", bare=True, durations=(10, 15, 20))[1] for _ in range(3)]
        barrier = self.barrier(frozen)
        barrier["profile_record_sha256s"].pop()
        with self.assertRaisesRegex(ValueError, "inventory"):
            a.evaluate_heldout(program, frozen, heldout, barrier=barrier)

    def test_heldout_seed_cannot_be_training_seed(self):
        program, records, frozen = self.profile()
        heldout = [self.record("heldout", bare=True, seed=records[0]["seed"])[1]]
        with self.assertRaisesRegex(ValueError, "seed leaked"):
            a.evaluate_heldout(program, frozen, heldout)

    def test_transition_unknown_is_retained_not_treated_as_zero_service(self):
        program, _, frozen = self.profile()
        events = [dict(begin_tick=0, end_tick=100, state=OFF), dict(begin_tick=130, end_tick=300, state=ON)]
        heldout = [self.record("heldout", bare=True, durations=(10, 50, 20), events=events)[1] for _ in range(3)]
        result = a.evaluate_heldout(program, frozen, heldout, barrier=self.barrier(frozen))
        self.assertEqual(result["buckets"]["unknown_transition_or_outside"]["sample_count"], 3)
        self.assertEqual(result["buckets"]["unknown_transition_or_outside"]["predictable_count"], 0)
        self.assertFalse(result["accepted"])

    def test_dynamic_requests_and_profile_settle_windows_are_distinct(self):
        events = dict(known_segments=[dict(phase_index=0, begin_tick=0, end_tick=100,
                         calibration_begin_tick=30, state=OFF), dict(phase_index=1,
                         begin_tick=100, end_tick=300, calibration_begin_tick=140, state=ON)],
            events=[dict(steady_begin_tick=30, steady_end_tick=100, requested_resources=OFF),
                    dict(steady_begin_tick=140, steady_end_tick=300, requested_resources=ON)])
        _, record = self.record(events=events)
        token = record["invocations"][0]["tokens"][1]
        self.assertEqual(token["steady_state"], OFF)
        self.assertIsNone(token["calibration_state"])
        self.assertEqual(record["invocations"][1]["tokens"][1]["calibration_state"], OFF)

    def test_microgap_cv_is_not_silently_original_source_criterion(self):
        _, _, frozen = self.profile()
        gaps = [cell for cell in frozen["cells"] if cell["token"] != "action:0:duration"]
        self.assertTrue(all(not cell["original_cv_criterion"] for cell in gaps))
        self.assertTrue(all("not_original" in cell["criterion_scope"] for cell in gaps))

    def test_fixed_comparisons_use_same_frozen_table_without_refit(self):
        program, _, frozen = self.profile()
        heldout = [self.record("heldout", bare=True, durations=(10, 15, 20))[1] for _ in range(3)]
        result = a.evaluate_heldout(program, frozen, heldout)
        crossing = result["predictions"][1]
        self.assertEqual(crossing["predicted_ns"], 15)
        self.assertEqual(crossing["comparisons"]["no_owned_cpu_gpu_load"]["predicted_ns"], 10)
        self.assertEqual(crossing["comparisons"]["entry_state_constant"]["predicted_ns"], 10)
        on = result["predictions"][2]
        self.assertEqual(on["comparisons"]["entry_state_constant"]["predicted_ns"], 20)
        self.assertTrue(all(not row["refitted"] for row in result["comparison_models"].values()))

    def test_zero_interval_at_state_boundary_uses_new_request(self):
        segments = [dict(begin_ns=0, end_ns=10, state=OFF), dict(begin_ns=10, end_ns=20, state=ON)]
        self.assertEqual(a._state_for_interval(segments, 10, 10), ON)

    def test_recovery_labels_use_predeclared_state_age_without_excluding_samples(self):
        program, records, _ = self.profile()
        frozen = a.freeze_profile(program, records, context=CONTEXT, required_states=[OFF, ON],
            frozen_at_utc="2026-10-21T01:00:00Z", protocol={"profile_settle_ms": 150})
        heldout = [self.record("heldout", bare=True, durations=(10, 15, 20))[1] for _ in range(3)]
        result = a.evaluate_heldout(program, frozen, heldout)
        self.assertEqual(result["buckets"]["recovery_before_declared_profile_settle_age"]["sample_count"], 6)
        self.assertEqual(result["overall"]["sample_count"], 9)

    def test_future_freeze_is_not_heldout_chronology(self):
        program, records, _ = self.profile()
        frozen = a.freeze_profile(program, records, context=CONTEXT, required_states=[OFF, ON],
            frozen_at_utc="2026-10-21T03:00:00Z", protocol={"predeclared": True})
        heldout = [self.record("heldout", bare=True)[1]]
        with self.assertRaisesRegex(ValueError, "before parameter freeze"):
            a.evaluate_heldout(program, frozen, heldout)

    def test_memory_groups_are_not_missing_replicates_of_other_groups(self):
        memory_off, memory_on = dict(OFF, host_memory_bytes=64), dict(ON, host_memory_bytes=64)
        rows = [self.record() for _ in range(3)] + [self.record(events=[dict(begin_tick=0, end_tick=100, state=memory_off),
            dict(begin_tick=100, end_tick=300, state=memory_on)]) for _ in range(3)]
        frozen = a.freeze_profile(rows[0][0], [row[1] for row in rows], context=CONTEXT,
            required_states=[OFF, ON, memory_off, memory_on], frozen_at_utc="2026-10-21T01:00:00Z",
            protocol={"predeclared_memory_grid": True})
        self.assertTrue(frozen["source_qualified"])
        self.assertTrue(all(len(cell["processes"]) == 3 for cell in frozen["cells"]))

    def test_declared_calibration_state_without_complete_samples_still_fails(self):
        events = dict(known_segments=[dict(begin_tick=0, end_tick=100, state=OFF),
                                      dict(begin_tick=100, end_tick=300, state=ON)],
            events=[dict(steady_begin_tick=65, steady_end_tick=75, requested_resources=OFF),
                    dict(steady_begin_tick=100, steady_end_tick=300, requested_resources=ON)])
        _, _, frozen = self.profile(events=events)
        cell = next(cell for cell in frozen["cells"] if cell["token"] == "action:0:duration" and cell["state"] == OFF)
        self.assertEqual(len(cell["processes"]), 3)
        self.assertTrue(all(process["sample_count"] == 0 for process in cell["processes"]))
        self.assertFalse(cell["qualified"])
        self.assertIsNone(cell["median_ns"])

    def test_requested_only_grace_state_is_not_declared_calibration_eligibility(self):
        events = dict(known_segments=[dict(begin_tick=0, end_tick=100, state=OFF),
                                      dict(begin_tick=100, end_tick=300, state=ON)],
            events=[dict(steady_begin_tick=30, steady_end_tick=100, requested_resources=OFF)])
        _, _, frozen = self.profile(events=events)
        on_cells = [cell for cell in frozen["cells"] if cell["state"] == ON]
        self.assertTrue(all(cell["processes"] == [] for cell in on_cells))

    def test_predeclared_minimum_samples_gate_does_not_erase_numeric_price(self):
        program, records, _ = self.profile()
        frozen = a.freeze_profile(program, records, context=CONTEXT, required_states=[OFF, ON],
            frozen_at_utc="2026-10-21T01:00:00Z", protocol={"minimum_cell_samples_per_process": 10})
        cell = next(cell for cell in frozen["cells"] if cell["token"] == "action:0:duration" and cell["state"] == OFF)
        self.assertEqual(a.exact(cell["median_ns"]), 10)
        self.assertFalse(cell["qualified"])
        self.assertIn("minimum_samples_per_process_gate_failed", cell["evidence_issues"])

    def test_go_time_origin_precedes_first_observed_foreground_call(self):
        _, record = self.record(custom_context=dict(CONTEXT, time_origin="planned_go_start_v1"), go={"start_tick": 0})
        self.assertEqual(record["origin_tick"], 0)
        self.assertEqual(record["invocations"][0]["begin_ns"], 10)

    def test_lag_candidate_freezes_profile_workers_but_never_parses_heldout_workers(self):
        go = dict(start_tick=0, end_tick=300, phases=[
            dict(phase_index=0, begin_tick=0, end_tick=100, cpu_replay=0, gpu_replay=0),
            dict(phase_index=1, begin_tick=100, end_tick=300, cpu_replay=0, gpu_replay=1)])
        context = dict(CONTEXT, time_origin="planned_go_start_v1")
        events = dict(go=go, known_segments=[dict(begin_tick=0, end_tick=100, state=OFF),
                                              dict(begin_tick=100, end_tick=310, state=ON)])
        def attach_workers(pair):
            program, record = pair
            bindings, sources = dict(record["raw_bindings"]), dict(record["source_files"])
            directory = Path(bindings["native"]).parent
            for resource in ("cpu", "gpu"):
                path = directory / ("worker_" + resource + ".json")
                intervals = [] if resource == "cpu" else [
                    dict(iteration=0, enabled_check_tick=102, begin_tick=105, end_tick=150, elapsed_ns=45),
                    dict(iteration=1, enabled_check_tick=200, begin_tick=200, end_tick=305, elapsed_ns=105)]
                worker = dict(role=resource + "-worker", process_id=record["process_id"] + 1000,
                    program_sha256=record["program_sha256"], actions=program["actions"], clock=record["clock"],
                    go_schedule=go, ready_tick=0, sampling_begin_tick=0, sampling_end_tick=310,
                    blocks=[dict(iterations=intervals, iteration_count=len(intervals))])
                # Invalid heldout worker JSON is deliberate: only bytes are
                # provenance-checked; predictions must not parse these targets.
                path.write_text(json.dumps(worker) if record["split"] == "profile" else "not predictor input", encoding="utf-8")
                bindings["worker_" + resource] = str(path)
                sources[str(path)] = a.file_sha(path)
            native = json.loads(Path(bindings["native"]).read_text(encoding="utf-8"))
            numerical = json.loads(Path(bindings["numerical"]).read_text(encoding="utf-8"))
            return program, a.make_run_record(program, native, events, context=context,
                run_id=record["run_id"], split=record["split"], seed=record["seed"], process_id=record["process_id"],
                started_at_utc=record["started_at_utc"], ended_at_utc=record["ended_at_utc"], source_files=sources,
                numerical_verification=numerical, raw_bindings=bindings)
        profiles = [attach_workers(self.record(custom_context=context, go=go, events=events)) for _ in range(3)]
        program = profiles[0][0]
        frozen = a.freeze_profile(program, [row[1] for row in profiles], context=context, required_states=[OFF, ON],
            frozen_at_utc="2026-10-21T01:00:00Z", protocol=dict(request_response_lags={"enabled": True},
            primary_prediction_model="lagged_dynamic_state_integration"))
        self.assertEqual(frozen["request_response_delays"]["gpu_start_ns"], 5)
        heldout = [attach_workers(self.record("heldout", bare=True, durations=(10, 10, 20),
                    custom_context=context, go=go, events=events))[1] for _ in range(3)]
        result = a.evaluate_heldout(program, frozen, heldout, barrier=self.barrier(frozen))
        self.assertEqual(result["predictions"][1]["predicted_ns"], 15)
        self.assertEqual(result["predictions"][1]["comparisons"]["lagged_dynamic_state_integration"]["predicted_ns"], 10)
        self.assertEqual(result["predeclared_primary_model"], "lagged_dynamic_state_integration")
        self.assertEqual(result["predeclared_primary_summary"]["process_balanced_median_absolute_relative_error"], 0)
        self.assertFalse(result["comparison_models"]["lagged_dynamic_state_integration"]["heldout_worker_intervals_used"])

    def test_lag_cannot_reconstruct_actuator_state_across_unknown_past(self):
        record = dict(segments=[dict(begin_ns=0, end_ns=10, state=OFF), dict(begin_ns=20, end_ns=50, state=ON)])
        self.assertTrue(a._known_request_history(record, 5))
        self.assertFalse(a._known_request_history(record, 30))

    def test_unknown_primary_model_is_rejected(self):
        program, records, _ = self.profile()
        frozen = a.freeze_profile(program, records, context=CONTEXT, required_states=[OFF, ON],
            frozen_at_utc="2026-10-21T01:00:00Z", protocol={"primary_prediction_model": "posthoc_best"})
        heldout = [self.record("heldout", bare=True)[1]]
        with self.assertRaisesRegex(ValueError, "unknown predeclared primary"):
            a.evaluate_heldout(program, frozen, heldout)

    def test_worker_activity_binding_is_hashed_but_never_parsed_as_predictor_input(self):
        program, record = self.record()
        bindings, sources = dict(record["raw_bindings"]), dict(record["source_files"])
        activity = Path(bindings["native"]).parent / "worker-activity.jsonl"
        activity.write_text("opaque diagnostic fixture, deliberately not JSON", encoding="utf-8")
        bindings["worker_activity"] = str(activity)
        sources[str(activity)] = a.file_sha(activity)
        native = json.loads(Path(bindings["native"]).read_text(encoding="utf-8"))
        events = json.loads(Path(bindings["events"]).read_text(encoding="utf-8"))
        numerical = json.loads(Path(bindings["numerical"]).read_text(encoding="utf-8"))
        normalized = a.make_run_record(program, native, events, context=CONTEXT,
            run_id=record["run_id"], split=record["split"], seed=record["seed"], process_id=record["process_id"],
            started_at_utc=record["started_at_utc"], ended_at_utc=record["ended_at_utc"], source_files=sources,
            numerical_verification=numerical, raw_bindings=bindings)
        a._validate_record(normalized, program, CONTEXT, "profile", True)
        self.assertEqual(normalized["invocations"], record["invocations"])


if __name__ == "__main__":
    unittest.main()
