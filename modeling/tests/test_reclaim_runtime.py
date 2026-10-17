"""Hand-calculated cooperative reclaim, causal timers and control accounting."""
from dataclasses import asdict, replace
import unittest

from llm_infer_model.tensor.engine import Action, CODE_TENSOR_IN_USE
from llm_infer_model.tensor.environment import (
    CODE_EXTERNAL_MEMORY_CONFLICT, CODE_RECLAIM_ADMISSION, ControlCosts,
    EnvironmentEvent, EnvironmentReclaimRequest, SimulationSession,
)
from llm_infer_model.tensor.runtime import ControlCommand, ExecutionCommand, dispatch_command, create_runtime
from llm_infer_model.tensor.services import ServiceSession, parse_services
from llm_infer_model.tensor.spec import COPY_ABSENT, COPY_READY, InvalidInput, build_scenario


def scene(*, capacity=96, initial_weight=("dram", "vram"), workspace=8, duration=10):
    def tensor(tid, role, ne, locations):
        size = ne[0] * ne[1] * 4
        return dict(id=tid, name=tid, role=role, dtype="GGML_TYPE_F32", ne=ne + [1, 1],
                    nb=[4, ne[0]*4, size, size], storage_bytes=size, initial_locations=list(locations))
    tensors = [tensor("w", "weight", [4, 2], initial_weight),
               tensor("x", "input", [4, 1], ["dram", "vram"]),
               tensor("y0", "output", [2, 1], []), tensor("y1", "output", [2, 1], [])]
    workload = dict(schema_version="0.2", id="reclaim-graph", origin=dict(
        kind="synthetic_fixture", ggml_version=None, sample=None), tensors=tensors,
        operations=[dict(id=f"op{i}", semantic_op="MUL_MAT", ggml_op="GGML_OP_MUL_MAT",
            op_params=[], inputs=["w", "x"], output=f"y{i}") for i in range(2)], outputs=["y0", "y1"])
    ids = [tensor["id"] for tensor in tensors]
    document = dict(schema_version="0.2", id="reclaim-scene", architecture=dict(
        dram_capacity_bytes=256, vram_capacity_bytes=capacity,
        dram_runtime_reserved_bytes=0, vram_runtime_reserved_bytes=0,
        dram_alignment_bytes=1, vram_alignment_bytes=1, cpu_compute_slots=1,
        gpu_compute_slots=1, pcie_copy_slots=1, ssd_io_slots=1), costs=dict(
        source="synthetic_reclaim_contract", compute={f"op{i}": {
            "cpu":dict(duration_ns=duration, workspace_bytes=workspace),
            "gpu":dict(duration_ns=duration, workspace_bytes=workspace)} for i in range(2)},
        transfer={route: {tid:dict(duration_ns=4) for tid in ids}
                  for route in ("dram_to_vram", "vram_to_dram")}), mapspace=dict(
        allowed_devices={f"op{i}":["cpu", "gpu"] for i in range(2)},
        allowed_transfers={route:ids for route in ("dram_to_vram", "vram_to_dram")},
        allow_eviction=True, allow_cpu_gpu_overlap=True, allow_copy_compute_overlap=True,
        allow_recomputation=False), goal=dict(output_locations={"y0":"vram", "y1":"vram"}),
        mapper=dict(algorithm="uniform_cost", max_expanded_states=0, wall_time_limit_s=0))
    return build_scenario(workload, document)


def costs(ack=2, decline=1, decision=3):
    return ControlCosts(ack, decline, decision, "synthetic_reclaim_contract")


def request(notice=0, deadline=20, target=64):
    return EnvironmentReclaimRequest("app", notice, target, deadline)


def usage(session):
    return next(row for row in session.observe().memory if row.domain == "vram")


def services(slots=()):
    return parse_services(dict(format_version="tensor-services-0.1", pcie=dict(
        bandwidth_bytes_per_second=10**9, ce_bytes_per_second=10**9,
        sm_bytes_per_second=10**9, arbitration="max_min_fair",
        timing_scope="payload_only_no_launch_latency"), slots=list(slots)))


class ReclaimRuntimeTests(unittest.TestCase):
    def session(self, **kwargs):
        return SimulationSession(scene(), reclaim_requests=(request(),), control_costs=costs(), **kwargs)

    def test_notice_does_not_allocate_and_ack_is_paid_at_completion(self):
        session = self.session()
        self.assertEqual((usage(session).external_bytes, usage(session).used_bytes), (0, 48))
        self.assertEqual(session.observe().safe_weight_release_candidates, ("w",))
        self.assertTrue(session.submit(Action("EVICT", "w", location="vram")).ok)
        self.assertTrue(session.submit_control(ControlCommand("ACK_RECLAIM", "app"), 7).ok)
        self.assertEqual(usage(session).external_bytes, 0)
        self.assertEqual(session.advance().t_ns, 2)
        self.assertEqual((usage(session).external_bytes, usage(session).used_bytes), (64, 80))
        self.assertEqual(session.result().reclaim_records[0].status, "accepted")
        self.assertEqual(session.result().control_events[0].source_index, 7)

    def test_wait_for_read_lock_then_reload_is_charged_after_recovery(self):
        session = SimulationSession(scene(), reclaim_requests=(request(3, 20),), control_costs=costs(),
                                    environment_events=(EnvironmentEvent(20, vram_external_bytes=0),))
        session.submit(Action("COMPUTE", "op0", device="gpu")); session.advance()
        self.assertEqual(session.observe().safe_weight_release_candidates, ())
        self.assertEqual(session.advance().t_ns, 10)
        self.assertEqual(session.observe().safe_weight_release_candidates, ("w",))
        session.submit(Action("EVICT", "w", location="vram"))
        session.submit_control(ControlCommand("ACK_RECLAIM", "app")); session.advance()
        self.assertEqual(session.advance().t_ns, 20)
        session.submit(Action("COPY", "w", src="dram", dst="vram")); session.advance()
        session.submit(Action("COMPUTE", "op1", device="gpu")); session.advance()
        self.assertTrue(session.finished)
        self.assertEqual((session.result().evaluation.h2d_bytes, session.state.t), (32, 34))

    def test_locked_weight_evict_rejected_without_mutation(self):
        session = SimulationSession(scene(), reclaim_requests=(request(3),), control_costs=costs())
        session.submit(Action("COMPUTE", "op0", device="gpu")); session.advance()
        result = session.submit(Action("EVICT", "w", location="vram"))
        self.assertEqual(result.error_code, CODE_TENSOR_IN_USE)
        self.assertEqual(session.state.copy_at("w", "vram"), COPY_READY)

    def test_single_vram_weight_is_not_safe_backed_candidate(self):
        session = SimulationSession(scene(initial_weight=("vram",)),
                                    reclaim_requests=(request(),), control_costs=costs())
        self.assertEqual(session.observe().safe_weight_release_candidates, ())
        session.submit_control(ControlCommand("ACK_RECLAIM", "app")); session.advance()
        self.assertEqual(session.result().reclaim_records[0].status, "capacity_rejected")
        self.assertIsNone(session.error_code)
        self.assertEqual(usage(session).external_bytes, 0)

    def test_fixed_slot_owner_eviction_does_not_reclaim_reserve(self):
        slot = dict(id="weights", domain="vram", capacity_bytes=32, members=["w"])
        session = ServiceSession(scene(), services((slot,)), reclaim_requests=(request(),), control_costs=costs())
        session.submit(Action("EVICT", "w", location="vram"))
        self.assertEqual((usage(session).runtime_reserved_bytes, usage(session).used_bytes), (32, 48))
        session.submit_control(ControlCommand("ACK_RECLAIM", "app")); session.advance()
        self.assertEqual(session.result().reclaim_records[0].status, "capacity_rejected")

    def test_deadline_rejects_no_response_but_keeps_inference_live(self):
        session = self.session()
        self.assertEqual(session.advance().t_ns, 20)
        # Last response window exists until another advance seals it.
        self.assertIsNotNone(session.observe().pending_reclaim)
        session.submit(Action("EVICT", "w", location="vram"))
        session.submit_control(ControlCommand("ACK_RECLAIM", "app")); session.advance()
        self.assertEqual(session.result().reclaim_records[0].status, "deadline_rejected")
        self.assertIsNone(session.error_code)
        self.assertEqual(usage(session).external_bytes, 0)

    def test_same_time_kernel_completion_deadline_zero_cost_ack(self):
        session = SimulationSession(scene(), reclaim_requests=(request(3, 10),), control_costs=costs(ack=0))
        session.submit(Action("COMPUTE", "op0", device="gpu")); session.advance(); session.advance()
        self.assertEqual(session.state.t, 10)
        self.assertEqual(session.observe().safe_weight_release_candidates, ("w",))
        session.submit(Action("EVICT", "w", location="vram"))
        session.submit_control(ControlCommand("ACK_RECLAIM", "app"))
        self.assertEqual(session.result().reclaim_records[0].status, "accepted")

    def test_ack_exact_completion_deadline_accepted(self):
        session = SimulationSession(scene(), reclaim_requests=(request(0, 2),), control_costs=costs())
        session.submit(Action("EVICT", "w", location="vram"))
        session.submit_control(ControlCommand("ACK_RECLAIM", "app")); session.advance()
        self.assertEqual(session.result().reclaim_records[0].status, "accepted")

    def test_pending_blocks_new_gpu_output_and_workspace(self):
        session = self.session()
        step = session.submit(Action("COMPUTE", "op0", device="gpu"))
        self.assertEqual(step.error_code, CODE_RECLAIM_ADMISSION)
        self.assertEqual(session.state.copy_at("y0", "vram"), COPY_ABSENT)

    def test_pending_blocks_service_transfer_even_with_slot_reserve(self):
        session = ServiceSession(scene(initial_weight=("dram",)), services(),
                                 reclaim_requests=(request(),), control_costs=costs())
        step = session.submit_transfer(Action("COPY", "w", src="dram", dst="vram"), engine="ce")
        self.assertEqual(step.error_code, CODE_RECLAIM_ADMISSION)
        self.assertEqual(session.state.copy_at("w", "vram"), COPY_ABSENT)

    def test_inflight_transfer_continues_and_reservation_cannot_be_evicted(self):
        session = ServiceSession(scene(initial_weight=("dram",)), services(),
                                 reclaim_requests=(request(3, 40),), control_costs=costs())
        session.submit_transfer(Action("COPY", "w", src="dram", dst="vram"), engine="ce")
        self.assertEqual(session.advance().t_ns, 3)
        self.assertEqual(session.observe().safe_weight_release_candidates, ())
        self.assertEqual(session.advance().t_ns, 32)
        self.assertEqual(session.result().evaluation.h2d_bytes, 32)
        self.assertEqual(session.observe().safe_weight_release_candidates, ("w",))

    def test_future_prefix_is_identical_and_no_private_queue_in_observation(self):
        sessions = [self.session(environment_events=(EnvironmentEvent(t, vram_external_bytes=0),))
                    for t in (30, 90)]
        self.assertEqual(asdict(sessions[0].observe()), asdict(sessions[1].observe()))
        for session in sessions:
            session.submit(Action("EVICT", "w", location="vram"))
            session.submit_control(ControlCommand("ACK_RECLAIM", "app")); session.advance()
        self.assertEqual(asdict(sessions[0].observe()), asdict(sessions[1].observe()))
        self.assertNotIn("environment_events", asdict(sessions[0].observe()))

    def test_wait_until_public_timer_interrupts_for_notice(self):
        session = SimulationSession(scene(), reclaim_requests=(request(3),), control_costs=costs())
        session.submit_control(ControlCommand("WAIT_UNTIL", until_ns=100))
        self.assertEqual(session.advance().t_ns, 3)
        self.assertIsNone(session.observe().control_running)
        self.assertEqual(session.result().control_events[0].cost_ns, 3)
        self.assertEqual(session.result().control_events[0].status, "interrupted")

    def test_control_cost_service_continues_during_cpu_blackout(self):
        session = self.session(environment_events=(EnvironmentEvent(1, cpu_available=False),))
        session.submit(Action("EVICT", "w", location="vram"))
        dispatch_command(session, ExecutionCommand(ControlCommand("DECIDE", "app")))
        self.assertEqual(session.advance().t_ns, 1)
        self.assertEqual(session.advance().t_ns, 3)
        self.assertEqual(session.result().control_events[0].cost_ns, 3)
        self.assertFalse(session.observe().cpu_available)

    def test_duplicate_and_unknown_response_audited_not_applied(self):
        session = self.session()
        session.submit(Action("EVICT", "w", location="vram"))
        session.submit_control(ControlCommand("ACK_RECLAIM", "app")); session.advance()
        for rid in ("app", "private-future"):
            self.assertFalse(session.submit_control(ControlCommand("ACK_RECLAIM", rid)).ok)
        self.assertEqual(usage(session).external_bytes, 64)
        self.assertEqual([event.status for event in session.result().control_events],
                         ["completed", "rejected", "rejected"])

    def test_decline_is_charged_unmet_and_does_not_stop_inference(self):
        session = self.session()
        session.submit_control(ControlCommand("DECLINE_RECLAIM", "app", reason="insufficient reusable weights"))
        session.advance()
        self.assertEqual(session.result().reclaim_records[0].status, "declined")
        self.assertEqual(session.result().control_events[0].cost_ns, 1)
        self.assertIsNone(session.error_code)

    def test_strict_inputs_unknown_costs_and_overlapping_requests_rejected(self):
        for value in (True, -1, None):
            with self.subTest(value=value), self.assertRaises(InvalidInput):
                EnvironmentReclaimRequest("app", 0, value, 10)
        with self.assertRaises(InvalidInput):
            SimulationSession(scene(), reclaim_requests=(request(),))
        with self.assertRaises(InvalidInput):
            SimulationSession(scene(), reclaim_requests=(request(), EnvironmentReclaimRequest("other", 2, 70, 30)),
                              control_costs=costs())

    def test_old_forced_overcommit_still_terminal_and_no_automatic_eviction(self):
        session = SimulationSession(scene(), environment_events=(EnvironmentEvent(0, vram_external_bytes=64),))
        self.assertEqual(session.error_code, CODE_EXTERNAL_MEMORY_CONFLICT)
        self.assertEqual(session.state.copy_at("w", "vram"), COPY_READY)
        self.assertEqual(usage(session).external_bytes, 0)

    def test_create_runtime_factory_propagates_explicit_contract(self):
        session = create_runtime(scene(), services=services(), reclaim_requests=(request(),), control_costs=costs())
        self.assertEqual(session.observe().pending_reclaim.id, "app")
        self.assertEqual(session.observe().control_costs.source, "synthetic_reclaim_contract")

    def test_nonweight_copy_never_in_clean_weight_candidates(self):
        session = self.session()
        self.assertNotIn("x", session.observe().safe_weight_release_candidates)
        step = session.submit(Action("EVICT", "x", location="vram"))
        self.assertEqual(step.error_code, CODE_RECLAIM_ADMISSION)
        self.assertEqual(session.state.copy_at("x", "vram"), COPY_READY)

    def test_goal_does_not_silently_drop_published_request(self):
        session = SimulationSession(scene(capacity=128), reclaim_requests=(request(10, 40, target=80),),
                                    control_costs=costs(ack=0))
        session.submit(Action("COMPUTE", "op0", device="gpu")); session.advance()
        # Second output would be new VRAM allocation while pending. Decline is
        # an explicit unmet request, after which ordinary inference continues.
        session.submit_control(ControlCommand("DECLINE_RECLAIM", "app", reason="keep example fixed"))
        session.advance()
        session.submit(Action("COMPUTE", "op1", device="gpu")); session.advance()
        self.assertTrue(session.finished)
        self.assertEqual(session.result().reclaim_records[0].status, "declined")

    def test_request_notice_at_goal_time_must_resolve_before_finished(self):
        session = SimulationSession(scene(capacity=128), reclaim_requests=(request(20, 25, target=80),),
                                    control_costs=costs(ack=0))
        for index in (0, 1):
            session.submit(Action("COMPUTE", f"op{index}", device="gpu")); session.advance()
        self.assertEqual(session.state.t, 20)
        self.assertFalse(session.finished)
        session.submit(Action("EVICT", "w", location="vram"))
        session.submit_control(ControlCommand("ACK_RECLAIM", "app"))
        self.assertTrue(session.finished)
        self.assertEqual(session.result().evaluation.makespan_ns, 20)

    def test_future_unpublished_request_after_goal_is_not_drained(self):
        session = SimulationSession(scene(), reclaim_requests=(request(100, 120),), control_costs=costs())
        for index in (0, 1):
            session.submit(Action("COMPUTE", f"op{index}", device="gpu")); session.advance()
        self.assertTrue(session.finished)
        self.assertEqual(session.result().reclaim_records, ())

    def test_multiple_disjoint_requests_use_absolute_targets(self):
        session = SimulationSession(scene(capacity=160), control_costs=costs(ack=0),
            reclaim_requests=(request(0, 5, 32), EnvironmentReclaimRequest("second", 10, 64, 20)))
        session.submit_control(ControlCommand("ACK_RECLAIM", "app"))
        self.assertEqual(usage(session).external_bytes, 32)
        session.advance()
        self.assertEqual(session.state.t, 10)
        session.submit_control(ControlCommand("ACK_RECLAIM", "second"))
        self.assertEqual(usage(session).external_bytes, 64)
        self.assertEqual([r.status for r in session.result().reclaim_records], ["accepted", "accepted"])

    def test_replay_control_and_tensor_sequence_preserves_records_exactly(self):
        def replay():
            session = self.session(environment_events=(EnvironmentEvent(20, vram_external_bytes=0),))
            commands = [ExecutionCommand(Action("EVICT", "w", location="vram")),
                        ExecutionCommand(ControlCommand("DECIDE", "app")),
                        ExecutionCommand(Action("ADVANCE")),
                        ExecutionCommand(ControlCommand("ACK_RECLAIM", "app")),
                        ExecutionCommand(Action("ADVANCE")),
                        ExecutionCommand(ControlCommand("WAIT_UNTIL", until_ns=20)),
                        ExecutionCommand(Action("ADVANCE")),
                        ExecutionCommand(Action("COPY", "w", src="dram", dst="vram")),
                        ExecutionCommand(Action("ADVANCE"))]
            for index, command in enumerate(commands):
                self.assertTrue(dispatch_command(session, command, source_index=index).ok)
            return asdict(session.result())
        self.assertEqual(replay(), replay())

    def test_cpu_blackout_keeps_dram_reader_locked_and_workspace_reserved(self):
        session = SimulationSession(scene(), environment_events=(EnvironmentEvent(3, cpu_available=False),
            EnvironmentEvent(7, cpu_available=True)), reclaim_requests=(request(3, 20),), control_costs=costs())
        session.submit(Action("COMPUTE", "op0", device="cpu")); session.advance()
        self.assertEqual(session.state.task_on("cpu_compute").remaining_ns, 7)
        self.assertEqual(next(m.workspace_bytes for m in session.observe().memory if m.domain == "dram"), 8)
        # The untouched GPU copy is safe even though its DRAM backing is read.
        self.assertEqual(session.observe().safe_weight_release_candidates, ("w",))
        session.submit(Action("EVICT", "w", location="vram"))
        session.submit_control(ControlCommand("ACK_RECLAIM", "app")); session.advance()
        session.advance(); session.advance()
        self.assertEqual(session.state.t, 14)
        self.assertEqual(session.state.op("op0"), "DONE")


if __name__ == "__main__":
    unittest.main()
