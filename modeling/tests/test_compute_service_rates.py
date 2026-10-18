"""Hand-calculated, exact compute service under causal device-state changes.

These tests establish execution arithmetic and contracts, not a physical
utilization-to-throughput relationship.
"""
from dataclasses import asdict
from fractions import Fraction
import json
import unittest

from llm_infer_model.tensor.engine import Action, CODE_RESOURCE_BUSY, CODE_TENSOR_IN_USE
from llm_infer_model.tensor.environment import (
    CODE_ENVIRONMENT_DEADLOCK, CODE_EXTERNAL_MEMORY_CONFLICT, EnvironmentEvent,
    SimulationSession,
)
from llm_infer_model.tensor.services import ServiceSession, parse_services
from llm_infer_model.tensor.spec import COPY_RESERVED_OUTPUT, InvalidInput, OP_DONE

from test_environment import CPU, GPU, scene
from test_services import copy, document, frac, scene as service_scene


def rate(t, device, numerator, denominator=1, **extra):
    return EnvironmentEvent(t, **{f"{device}_service_rate_numerator": numerator,
                                 f"{device}_service_rate_denominator": denominator}, **extra)


def drain(session):
    while session.state.running:
        step = session.advance()
        if not step.ok:
            raise AssertionError(step)


def served(session, target):
    return sum((Fraction(row.reference_service_ns_numerator, row.reference_service_ns_denominator)
                for row in session.result().service_intervals
                if row.target_id == target and row.reference_service_ns_numerator is not None), Fraction())


class ComputeServiceRateTests(unittest.TestCase):
    def test_constant_slow_cpu_and_fast_gpu_have_independent_completion_times(self):
        session = SimulationSession(scene(), environment_events=(rate(0, "cpu", 1, 2), rate(0, "gpu", 2)))
        session.submit(CPU); session.submit(GPU)
        self.assertEqual(session.advance().t_ns, 3)
        self.assertEqual(session.state.op("gpu-op"), OP_DONE)
        self.assertEqual(session.advance().t_ns, 20)
        self.assertTrue(session.finished)
        self.assertEqual(served(session, "cpu-op"), 10)
        self.assertEqual(served(session, "gpu-op"), 6)
        # Nominal ends retain the supplied cost table's reference-rate timing.
        self.assertEqual(session.result().nominal_ends[:2], (10, 6))

    def test_mid_task_slowdown_consumes_only_remaining_work(self):
        session = SimulationSession(scene(), environment_events=(rate(3, "cpu", 1, 2),))
        session.submit(CPU); session.submit(GPU)
        self.assertEqual(session.advance().t_ns, 3)
        self.assertEqual(session.state.task_on("cpu_compute").remaining_ns, 14)
        self.assertEqual(session.advance().t_ns, 6)
        self.assertEqual(session.advance().t_ns, 17)
        self.assertEqual(served(session, "cpu-op"), 10)

    def test_mid_task_speedup_and_final_tick_do_not_invent_extra_work(self):
        session = SimulationSession(scene(), environment_events=(rate(3, "cpu", 2),))
        session.submit(CPU)
        self.assertEqual(session.advance().t_ns, 3)
        self.assertEqual(session.advance().t_ns, 7)  # 3 + ceil(7 / 2).
        last = session.result().service_intervals[-1]
        self.assertEqual((last.start_ns, last.end_ns), (3, 7))
        self.assertEqual((last.service_rate_numerator, last.service_rate_denominator), (2, 1))
        self.assertEqual(last.reference_service_ns_numerator, 7)
        self.assertEqual(served(session, "cpu-op"), 10)
        self.assertEqual(session.result().compute_progress, ())

    def test_fractional_residue_survives_unrelated_boundaries_and_rate_change(self):
        events = (rate(0, "cpu", 2, 3), EnvironmentEvent(1, dram_external_bytes=0),
                  EnvironmentEvent(2, dram_external_bytes=1), rate(3, "cpu", 5, 4))
        session = SimulationSession(scene(), environment_events=events)
        session.submit(CPU); session.submit(GPU)
        self.assertEqual(session.advance().t_ns, 1)
        row = next(row for row in session.result().compute_progress if row.target_id == "cpu-op")
        self.assertEqual(Fraction(row.remaining_reference_ns_numerator,
                                  row.remaining_reference_ns_denominator), Fraction(28, 3))
        self.assertEqual(session.advance().t_ns, 2)
        self.assertEqual(session.advance().t_ns, 3)
        drain(session)
        self.assertEqual(session.state.t, 10)  # 2 served before t=3, then ceil(8 / (5/4)).
        self.assertEqual(served(session, "cpu-op"), 10)

    def test_cpu_availability_blackout_preserves_changed_declared_rate(self):
        events = (rate(0, "cpu", 1, 2), EnvironmentEvent(2, cpu_available=False),
                  rate(6, "cpu", 3, 2), EnvironmentEvent(8, cpu_available=True))
        session = SimulationSession(scene(), environment_events=events)
        session.submit(CPU)
        self.assertEqual(session.advance().t_ns, 2)
        self.assertEqual(session.observe().cpu_service_rate_numerator, 0)
        self.assertEqual(session.advance().t_ns, 6)
        self.assertEqual(session.observe().cpu_service_rate_numerator, 0)
        attempted = session.result().environment_events[-1]
        self.assertEqual((attempted.cpu_service_rate_numerator, attempted.cpu_service_rate_denominator), (3, 2))
        self.assertEqual(session.advance().t_ns, 8)
        self.assertEqual((session.observe().cpu_service_rate_numerator,
                          session.observe().cpu_service_rate_denominator), (3, 2))
        self.assertEqual(session.advance().t_ns, 14)
        self.assertEqual(served(session, "cpu-op"), 10)
        self.assertEqual([(row.start_ns, row.end_ns) for row in session.result().service_intervals],
                         [(0, 2), (8, 14)])

    def test_gpu_zero_rate_pauses_and_recovers_without_restarting(self):
        session = SimulationSession(scene(), environment_events=(rate(2, "gpu", 0), rate(10, "gpu", 1)))
        session.submit(GPU)
        self.assertEqual(session.advance().t_ns, 2)
        self.assertEqual(session.state.copy_at("y", "vram"), COPY_RESERVED_OUTPUT)
        self.assertEqual(session.advance().t_ns, 10)
        self.assertEqual(session.advance().t_ns, 14)
        self.assertEqual(served(session, "gpu-op"), 6)

    def test_zero_rates_reject_new_compute_without_committing_output(self):
        for device, action, output, domain in (("cpu", CPU, "h", "dram"), ("gpu", GPU, "y", "vram")):
            with self.subTest(device=device):
                session = SimulationSession(scene(), environment_events=(rate(0, device, 0),))
                before = session.state
                self.assertEqual(session.submit(action).error_code, CODE_RESOURCE_BUSY)
                self.assertEqual(session.state, before)
                self.assertEqual(session.result().compute_progress, ())

    def test_all_compute_paused_without_future_recovery_is_explicit_deadlock(self):
        session = SimulationSession(scene(), environment_events=(rate(1, "cpu", 0), rate(1, "gpu", 0)))
        session.submit(CPU); session.submit(GPU)
        self.assertEqual(session.advance().t_ns, 1)
        self.assertEqual(session.advance().error_code, CODE_ENVIRONMENT_DEADLOCK)
        self.assertEqual(served(session, "cpu-op"), 1)
        self.assertEqual(served(session, "gpu-op"), 1)

    def test_paused_compute_keeps_inputs_output_and_workspace_committed(self):
        session = SimulationSession(scene(), environment_events=(rate(2, "cpu", 0),))
        session.submit(CPU); session.advance()
        memory = next(row for row in session.observe().memory if row.domain == "dram")
        self.assertEqual((memory.tensor_bytes, memory.workspace_bytes, memory.used_bytes), (32, 16, 48))
        self.assertEqual(session.state.copy_at("h", "dram"), COPY_RESERVED_OUTPUT)
        self.assertEqual(session.submit(Action("EVICT", "x", location="dram")).error_code, CODE_TENSOR_IN_USE)

    def test_completion_precedes_same_time_zero_rate_and_memory_change(self):
        session = SimulationSession(scene(), environment_events=(rate(0, "gpu", 2),
            rate(3, "gpu", 0, vram_external_bytes=16)))
        session.submit(GPU)
        step = session.advance()
        self.assertEqual(step.completed_operation_ids, ("gpu-op",))
        self.assertEqual(session.state.t, 3)
        self.assertFalse(session.state.running)
        self.assertEqual(session.observe().gpu_service_rate_numerator, 0)
        self.assertEqual(served(session, "gpu-op"), 6)

    def test_rejected_atomic_memory_group_does_not_commit_its_rate(self):
        session = SimulationSession(scene(), environment_events=(
            rate(3, "cpu", 1, 2, dram_external_bytes=200),))
        session.submit(CPU)
        self.assertEqual(session.advance().error_code, CODE_EXTERNAL_MEMORY_CONFLICT)
        self.assertEqual(session.observe().cpu_service_rate_numerator, 1)
        self.assertEqual(session.state.task_on("cpu_compute").remaining_ns, 7)
        self.assertFalse(session.result().environment_events[-1].accepted)
        self.assertEqual(session.result().environment_events[-1].cpu_service_rate_denominator, 2)

    def test_observation_exposes_only_current_normalized_effective_rate(self):
        session = SimulationSession(scene(), environment_events=(rate(0, "cpu", 2, 4), rate(999, "gpu", 777)))
        value = asdict(session.observe())
        self.assertEqual((value["cpu_service_rate_numerator"], value["cpu_service_rate_denominator"]), (1, 2))
        self.assertEqual(value["gpu_service_rate_numerator"], 1)
        self.assertNotIn("999", json.dumps(value))
        self.assertNotIn("777", json.dumps(value))
        self.assertNotIn("compute_progress", value)

    def test_exact_progress_and_service_intervals_are_json_serializable(self):
        session = SimulationSession(scene(), environment_events=(rate(0, "cpu", 2, 3),
            EnvironmentEvent(1, dram_external_bytes=0)))
        session.submit(CPU); session.advance()
        result = json.loads(json.dumps(asdict(session.result())))
        self.assertEqual(result["compute_progress"][0]["remaining_reference_ns_numerator"], 28)
        self.assertEqual(result["compute_progress"][0]["remaining_reference_ns_denominator"], 3)
        self.assertEqual(result["service_intervals"][0]["reference_service_ns_denominator"], 3)

    def test_rates_require_strict_atomic_integer_pairs(self):
        malformed = [dict(cpu_service_rate_numerator=1), dict(gpu_service_rate_denominator=1),
            dict(cpu_service_rate_numerator=True, cpu_service_rate_denominator=1),
            dict(cpu_service_rate_numerator=1, cpu_service_rate_denominator=False),
            dict(gpu_service_rate_numerator=-1, gpu_service_rate_denominator=1),
            dict(gpu_service_rate_numerator=1, gpu_service_rate_denominator=0),
            dict(gpu_service_rate_numerator=1.0, gpu_service_rate_denominator=2),
            dict(gpu_service_rate_numerator=1, gpu_service_rate_denominator=1.5)]
        for kwargs in malformed:
            with self.subTest(kwargs=kwargs), self.assertRaises(InvalidInput):
                EnvironmentEvent(0, **kwargs)
        with self.assertRaisesRegex(InvalidInput, "repeats cpu_service_rate"):
            SimulationSession(scene(), environment_events=(rate(0, "cpu", 1, 2), rate(0, "cpu", 2, 4)))


class SharedLinkRateTests(unittest.TestCase):
    def session(self, events):
        return ServiceSession(service_scene((64, 16), initial=[["dram"], ["dram", "vram"]]),
                              parse_services(document()), environment_events=events)

    def test_compute_rate_changes_do_not_change_copy_byte_rate(self):
        session = self.session((rate(2, "gpu", 1, 2), rate(8, "gpu", 2)))
        session.submit_transfer(copy(0), engine="ce")
        session.submit(Action("COMPUTE", "op1", device="gpu"))
        self.assertEqual(session.advance().t_ns, 2)
        self.assertEqual(session.advance().t_ns, 8)
        self.assertEqual(session.advance().t_ns, 11)  # 2 + 3 units served, then ceil(5 / 2).
        self.assertEqual(session.state.op("op1"), OP_DONE)
        self.assertEqual(session.advance().t_ns, 64)
        records = session.service_document()["link_service_intervals"]
        self.assertEqual(sum(frac(flow["served_bytes"]) for row in records for flow in row["flows"]), 64)
        self.assertTrue(all(frac(flow["rate_bytes_per_ns"]) == 1 for row in records for flow in row["flows"]))
        self.assertEqual(served(session, "op1"), 10)

    def test_zero_gpu_rate_pauses_compute_but_ce_copy_continues(self):
        session = self.session((rate(2, "gpu", 0), rate(80, "gpu", 1)))
        session.submit_transfer(copy(0), engine="ce")
        session.submit(Action("COMPUTE", "op1", device="gpu"))
        self.assertEqual(session.advance().t_ns, 2)
        self.assertEqual(session.advance().t_ns, 64)
        self.assertEqual(session.advance().t_ns, 80)
        self.assertEqual(session.advance().t_ns, 88)
        self.assertEqual(served(session, "op1"), 10)
        self.assertEqual(session.result().evaluation.h2d_bytes, 64)


if __name__ == "__main__":
    unittest.main()
