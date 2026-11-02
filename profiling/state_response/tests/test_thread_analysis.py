"""Synthetic scope/identity/containment guards; no OS or native program calls."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

SOURCE = Path(__file__).with_name("thread_analysis.py")
if not SOURCE.is_file():
    SOURCE = Path(__file__).resolve().parents[1] / "thread_analysis.py"
SPEC = importlib.util.spec_from_file_location("thread_analysis_candidate", SOURCE)
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)

CLOCK = dict(clock_id="windows_qpc", frequency_hz=1000)
IDENTITIES = {"main": dict(actor_id="main", process_id=1, creation_time_100ns=77, main_thread_id=42)}
GO = dict(clock=CLOCK, start_tick=0, end_tick=500, phases=[
    dict(phase_index=0, begin_tick=0, end_tick=100, cpu_replay=0, gpu_replay=0),
    dict(phase_index=1, begin_tick=100, end_tick=200, cpu_replay=1, gpu_replay=0),
    dict(phase_index=2, begin_tick=200, end_tick=500, cpu_replay=0, gpu_replay=0)])


def snapshot(index, begin, end, *, cycles=10, state="Ready"):
    def thread(tid, thread_state):
        return dict(thread_id=tid, thread_creation_time_100ns=90 + tid, cycle_total=cycles,
            kernel_time_100ns=0, user_time_100ns=0, handle_owner_pid=1,
            state_name=thread_state, wait_reason_name="UserRequest" if thread_state == "Wait" else None,
            unknown=[], query_begin_tick=begin, query_end_tick=end,
            state_snapshot_begin_tick=begin, state_snapshot_end_tick=end)
    return dict(sample_index=index, query_begin_tick=begin, query_end_tick=end,
        actors=[dict(actor_id="main", process_id=1, creation_time_100ns=77,
            query_begin_tick=begin, query_end_tick=end, unknown=[],
            threads=[thread(43, "Wait"), thread(42, state)])])


def document(rows):
    return dict(schema="owned-thread-snapshots-1", clock=CLOCK, records=rows,
        exact_thread_state_durations_claimed=False, used_as_predictor=False, missed_periods=0)


def fixture():
    action = dict(kind="COMPUTE", device="cpu", target_id="projection")
    program = dict(actions=[action], physical=dict(threads=2))
    row = dict(iteration=0, action_index=0, begin_tick=10, end_tick=30, elapsed_ns=20_000_000)
    cpu = dict(iteration=0, action_index=0, action_begin_tick=10, action_end_tick=30,
        action_elapsed_ns=20_000_000, before_query_ns=1_000_000, after_query_ns=2_000_000,
        outer_elapsed_ns=23_000_000,
        cpu_counter_bracket=dict(process_cpu_ns=0, main_thread_cpu_ns=0, main_thread_cycles=999))
    native = dict(clock=CLOCK, go_schedule=GO, actions=[action], main_thread_id=42,
        process_id=1, instrumentation="thread-diagnostic", blocks=[dict(
            iterations=[dict(iteration=0, begin_tick=9, end_tick=32, elapsed_ns=23_000_000)],
            action_intervals=[row], thread_action_diagnostics=[cpu], target_diagnostics=[])])
    job = dict(id="synthetic", condition="solo", repeat=0)
    observations = document([snapshot(0, 9, 11), snapshot(1, 12, 14), snapshot(2, 50, 51)])
    return job, program, native, GO, observations, deepcopy(IDENTITIES)


class ThreadAnalysisTests(unittest.TestCase):
    def test_native_foreground_pid_must_match_the_observed_lifetime_identity(self):
        parts = fixture()
        parts[2]["process_id"] = 99
        with self.assertRaisesRegex(ValueError, "native main PID"):
            A.summarize_job(*parts)

    def test_initial_and_recovery_00_are_different_matched_strata(self):
        initial = A.interval_stratum(GO, 10, 30)
        recovery = A.interval_stratum(GO, 210, 230)
        self.assertEqual(initial["phase_kind"], "initial_00")
        self.assertIsNone(initial["predecessor_request_state"])
        self.assertEqual(recovery["phase_kind"], "post_load_00")
        self.assertEqual(recovery["predecessor_request_state"], [1, 0])

    def test_crossing_phase_cannot_be_assigned_to_entry_state(self):
        self.assertEqual(A.interval_stratum(GO, 90, 110)["category"], "crossing_or_outside")

    def test_crossing_age_boundary_is_retained_not_force_binned(self):
        self.assertEqual(A.interval_stratum(GO, 40, 60)["age_bin"], "crosses_age_boundary")
        self.assertEqual(A.interval_stratum(GO, 40, 50)["age_bin"], 0)

    def test_outer_query_envelope_not_thread_property_midpoint_controls_join(self):
        index = A.SnapshotIndex(document([snapshot(0, 9, 11), snapshot(1, 12, 14)]), IDENTITIES, CLOCK)
        self.assertEqual([r["sample_index"] for r in index.contained(10, 30)], [1])

    def test_tight_state_query_join_excludes_later_counter_enumeration(self):
        row = snapshot(0, 9, 40)
        for thread in row["actors"][0]["threads"]:
            thread.update(state_snapshot_begin_tick=12, state_snapshot_end_tick=14,
                          query_begin_tick=35, query_end_tick=38)
        index = A.SnapshotIndex(document([row]), IDENTITIES, CLOCK)
        self.assertEqual(list(index.contained(10, 30)), [])
        self.assertEqual(len(list(index.contained_states(10, 30))), 2)
        parts = list(fixture()); parts[4] = document([row])
        result = A.summarize_job(*parts)
        self.assertEqual(result["focus_legacy_whole_outer_snapshot_counts"]["0"], 0)
        self.assertEqual(result["focus_contained_snapshot_counts"]["0"], 1)

    def test_known_main_tid_overrides_thread_enumeration_order(self):
        counts = A.ThreadCounts()
        counts.add(snapshot(0, 10, 12), IDENTITIES)
        row = counts.report()["actors"]["main"]["roles"]
        self.assertEqual(row["main"]["state_counts"], {"Ready": 1})
        self.assertEqual(row["auxiliary"]["state_counts"], {"Wait": 1})

    def test_unknown_state_and_wait_reason_stay_unknown(self):
        row = snapshot(0, 10, 12, state=None)
        row["actors"][0]["threads"][0]["wait_reason_name"] = None
        counts = A.ThreadCounts(); counts.add(row, IDENTITIES)
        roles = counts.report()["actors"]["main"]["roles"]
        self.assertEqual(roles["main"]["state_counts"], {"unknown": 1})
        self.assertEqual(roles["auxiliary"]["wait_reason_counts"], {"unknown": 1})

    def test_native_wait_code_is_not_cast_to_managed_wait_reason(self):
        row = snapshot(0, 10, 12)
        thread = row["actors"][0]["threads"][0]
        thread.update(thread_state_provider="NtQuerySystemInformation:SystemProcessInformation",
            state_code=5, native_wait_reason_code=13, wait_reason_code=None, wait_reason_name=None)
        counts = A.ThreadCounts(); counts.add(row, IDENTITIES)
        reason = counts.report()["wait_code_observations"]["main"]["auxiliary"][0]
        self.assertEqual(reason["code"], 13)
        self.assertTrue(reason["namespace"].endswith(":native_wait_reason"))
        self.assertIsNone(reason["name"])

    def test_actor_lifetime_or_thread_owner_mismatch_is_rejected(self):
        for key, value in (("creation_time_100ns", 88), ("process_id", 2)):
            row = snapshot(0, 10, 12); row["actors"][0][key] = value
            with self.assertRaises(ValueError):
                A.SnapshotIndex(document([row]), IDENTITIES, CLOCK)
        row = snapshot(0, 10, 12); row["actors"][0]["threads"][0]["handle_owner_pid"] = 2
        with self.assertRaises(ValueError):
            A.SnapshotIndex(document([row]), IDENTITIES, CLOCK)

    def test_nested_snapshot_query_cannot_escape_outer_envelope(self):
        row = snapshot(0, 10, 12)
        row["actors"][0]["threads"][0]["state_snapshot_begin_tick"] = 9
        with self.assertRaises(ValueError):
            A.SnapshotIndex(document([row]), IDENTITIES, CLOCK)

    def test_counter_overhead_stays_disjoint_from_action_timer(self):
        result = A.summarize_job(*fixture())
        self.assertEqual(result["overall_invocation_wall_ns"]["p50"], 23_000_000)
        self.assertEqual(result["overall_sum_counter_query_wall_ns"]["p50"], 3_000_000)
        self.assertEqual(result["focus_action_calls"]["0"], dict(observed=1, with_contained_snapshot=1))
        self.assertEqual(result["focus_contained_snapshot_counts"]["0"], 1)
        self.assertTrue(any(group["observations"].get("main_thread_cpu_ns", {}).get("p50") == 0
                            for group in result["groups"]))

    def test_raw_cycles_keep_own_unit_and_backwards_counter_is_unresolved(self):
        parts = list(fixture())
        parts[4] = document([snapshot(0, 12, 14, cycles=100), snapshot(1, 15, 16, cycles=50)])
        result = A.summarize_job(*parts)
        self.assertEqual(result["thread_cycle_window_problems"]["counter_moved_backwards"], 2)
        self.assertFalse(any(group["stratum"]["scope"] == "thread_cycle_window" for group in result["groups"]))

    def test_missing_cycles_do_not_become_zero_increments(self):
        parts = list(fixture())
        parts[4] = document([snapshot(0, 12, 14, cycles=None), snapshot(1, 15, 16, cycles=50)])
        result = A.summarize_job(*parts)
        self.assertEqual(result["thread_cycle_window_problems"]["unknown_lifetime_or_cycles"], 2)

    def test_declared_absent_observer_keeps_bare_invocations_without_synthetic_trace(self):
        parts = list(fixture()); parts[4] = None
        parts[2]["instrumentation"] = "bare"
        parts[2]["blocks"][0]["action_intervals"] = []
        parts[2]["blocks"][0]["thread_action_diagnostics"] = []
        result = A.summarize_job(*parts)
        self.assertEqual(result["thread_observer_status"], "declared_absent")
        self.assertEqual(result["invocation_count"], 1)
        self.assertEqual(result["snapshot_count"], 0)
        self.assertIsNone(result["overall_sum_counter_query_wall_ns"])

    def test_source_hash_is_verified_and_unbound_input_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=SOURCE.parent) as directory:
            path = Path(directory) / "input.json"; path.write_bytes(b"{}")
            audit = A.SourceAudit(); digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(audit.bound_json(path, {str(path.resolve()): digest}), {})
            with self.assertRaises(ValueError): audit.bound_json(path, {})
            path.write_bytes(b"{\"mutated\":true}")
            with self.assertRaises(ValueError): audit.bound_json(path, {str(path.resolve()): digest})


if __name__ == "__main__":
    unittest.main()
