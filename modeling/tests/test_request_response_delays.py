"""Actuator edge hand calculations, independent of native timing observations."""
from dataclasses import replace
import json
import unittest
from unittest import mock

from llm_infer_model.performance import InvalidModel, UnknownParameter, evaluate
from llm_infer_model.performance import conditioned_response as response


BASE = response.ResponseState(0, 0, 0, 0)
CPU = response.ResponseState(1, 0, 0, 0)
GPU = response.ResponseState(0, 1, 0, 0)
BOTH = response.ResponseState(1, 1, 0, 0)


def delays(cpu_start=5, cpu_stop=2, gpu_start=7, gpu_stop=3, *, qualified=True):
    return response.RequestResponseDelays(cpu_start, cpu_stop, gpu_start, gpu_stop,
        ("synthetic independent worker observation",), qualified)


def prices():
    return response.ResponseTable("b" * 64, ("a",), tuple(response.ResponseCell(
        "a", state, duration, ("synthetic scoped action observation",), True)
        for state, duration in ((BASE, 10), (CPU, 20), (GPU, 30), (BOTH, 40))))


def transform(trajectory, lag=None):
    return response.apply_request_response_delays(trajectory, delays() if lag is None else lag,
                                                 initial_effective_state=BASE)


class RequestResponseDelayTests(unittest.TestCase):
    def test_initial_request_has_latency_from_external_origin_and_retains_progress(self):
        requested = response.ResponseTrajectory(CPU)
        effective = transform(requested)
        self.assertEqual(effective.initial_state, BASE)
        self.assertEqual(effective.points, (response.ResponsePoint(5, CPU),))
        with mock.patch.object(response, "evaluate", wraps=evaluate) as engine:
            result = response.evaluate_response(prices(), requested, request_delays=delays(),
                                               initial_effective_state=BASE)
        engine.assert_called_once()
        # Half the reference work at the original rate, then half at half-rate.
        self.assertEqual(result.duration_ns, 15)
        self.assertEqual(result.effective_trajectory, effective)

    def test_delayed_on_and_off_use_independent_edges_not_added_time_offsets(self):
        requested = response.ResponseTrajectory(BASE, (
            response.ResponsePoint(2, CPU), response.ResponsePoint(8, BASE)))
        effective = transform(requested, delays(cpu_start=3, cpu_stop=2))
        self.assertEqual(effective.points, (response.ResponsePoint(5, CPU), response.ResponsePoint(10, BASE)))
        result = response.evaluate_response(prices(), requested,
            request_delays=delays(cpu_start=3, cpu_stop=2), initial_effective_state=BASE)
        # 0..5: 1/2 work, 5..10: 1/4, remaining 1/4 takes 2.5 ns rounded to 3.
        self.assertEqual(result.duration_ns, 13)

    def test_cpu_and_gpu_actuators_are_independent(self):
        requested = response.ResponseTrajectory(BOTH)
        effective = transform(requested, delays(cpu_start=2, gpu_start=5))
        self.assertEqual(effective.points, (response.ResponsePoint(2, CPU), response.ResponsePoint(5, BOTH)))
        result = response.evaluate_response(prices(), requested,
            request_delays=delays(cpu_start=2, gpu_start=5), initial_effective_state=BASE)
        # 2/10 + 3/20 of reference work done; the rest takes .65*40 = 26 ns.
        self.assertEqual(result.duration_ns, 31)

    def test_opposite_request_cancels_pending_start_including_exact_due_timestamp(self):
        for stop in (4, 5):
            with self.subTest(stop=stop):
                requested = response.ResponseTrajectory(CPU, (response.ResponsePoint(stop, BASE),))
                self.assertEqual(transform(requested), response.ResponseTrajectory(BASE))

    def test_restart_cancels_pending_stop_without_restarting_active_worker(self):
        requested = response.ResponseTrajectory(CPU, (
            response.ResponsePoint(3, BASE), response.ResponsePoint(6, CPU)))
        effective = transform(requested, delays(cpu_start=2, cpu_stop=10))
        self.assertEqual(effective.points, (response.ResponsePoint(2, CPU),))

    def test_repeated_same_request_does_not_postpone_pending_activation(self):
        requested = response.ResponseTrajectory(CPU, (
            response.ResponsePoint(1, BASE), response.ResponsePoint(2, CPU), response.ResponsePoint(7, CPU)))
        self.assertEqual(transform(requested, delays(cpu_start=10)).points,
                         (response.ResponsePoint(12, CPU),))

    def test_zero_delay_and_zero_time_point_use_final_request_at_that_instant(self):
        requested = response.ResponseTrajectory(CPU, (
            response.ResponsePoint(0, GPU), response.ResponsePoint(5, BOTH)))
        effective = transform(requested, delays(0, 0, 0, 0))
        self.assertEqual(effective.initial_state, GPU)
        self.assertEqual(effective.points, (response.ResponsePoint(5, BOTH),))

    def test_memory_holds_and_availability_change_immediately(self):
        held = response.ResponseState(1, 0, 64, 128)
        requested = response.ResponseTrajectory(CPU, (
            response.ResponsePoint(1, held), response.ResponsePoint(2, held, False),
            response.ResponsePoint(4, held, True)))
        effective = transform(requested, delays(cpu_start=3))
        self.assertEqual(effective.points, (
            response.ResponsePoint(1, response.ResponseState(0, 0, 64, 128)),
            response.ResponsePoint(2, response.ResponseState(0, 0, 64, 128), False),
            response.ResponsePoint(3, held, False), response.ResponsePoint(4, held, True)))

    def test_invocation_entry_does_not_restart_actuator_delays(self):
        requested = response.ResponseTrajectory(CPU)
        entered = response.evaluate_response(prices(), requested, start_ns=10,
            request_delays=delays(), initial_effective_state=BASE)
        self.assertEqual((entered.duration_ns, entered.end_ns), (20, 30))
        crossing = response.evaluate_response(prices(), requested, start_ns=3,
            request_delays=delays(), initial_effective_state=BASE)
        self.assertEqual((crossing.duration_ns, crossing.end_ns), (18, 21))

    def test_unknown_needed_delay_rejected_without_zero_or_foreground_fallback(self):
        unknown = delays(cpu_start=None, cpu_stop=None, gpu_start=None, gpu_stop=None)
        self.assertEqual(transform(response.ResponseTrajectory(BASE), unknown), response.ResponseTrajectory(BASE))
        with self.assertRaisesRegex(UnknownParameter, "cpu_start_ns"):
            transform(response.ResponseTrajectory(CPU), unknown)
        with self.assertRaisesRegex(UnknownParameter, "cpu_stop_ns"):
            transform(response.ResponseTrajectory(CPU, (response.ResponsePoint(2, BASE),)),
                      delays(cpu_start=0, cpu_stop=None))

    def test_initial_effective_idle_state_is_required_explicit_and_memory_matched(self):
        for initial in (None, CPU, response.ResponseState(0, 0, 1, 0)):
            with self.subTest(initial=initial), self.assertRaises(InvalidModel):
                response.evaluate_response(prices(), response.ResponseTrajectory(CPU),
                    request_delays=delays(), initial_effective_state=initial)
        with self.assertRaises(InvalidModel):
            response.evaluate_response(prices(), response.ResponseTrajectory(BASE), initial_effective_state=BASE)

    def test_source_qualification_is_not_silently_promoted(self):
        result = response.evaluate_response(prices(), response.ResponseTrajectory(CPU),
            request_delays=delays(qualified=False), initial_effective_state=BASE).to_dict()
        self.assertTrue(result["response_support_qualified"])
        self.assertFalse(result["request_response_support_qualified"])
        self.assertFalse(result["all_model_parameter_sources_qualified"])
        self.assertFalse(result["independently_validated_accuracy_claimed"])
        self.assertFalse(result["request_response_lag"]["worker_evidence_estimated_here"])
        original = response.evaluate_response(prices(), response.ResponseTrajectory(BASE)).to_dict()
        self.assertNotIn("request_response_lag", original)
        self.assertNotIn("all_model_parameter_sources_qualified", original)

    def test_delay_serialization_and_invalid_values_are_strict(self):
        value = delays(cpu_stop=None)
        self.assertEqual(response.RequestResponseDelays.from_dict(json.loads(json.dumps(value.to_dict()))), value)
        for bad in (-1, True, 0.5):
            with self.subTest(bad=bad), self.assertRaises(InvalidModel):
                replace(value, cpu_start_ns=bad)
        with self.assertRaises(InvalidModel):
            replace(value, evidence=())
        with self.assertRaises(InvalidModel):
            replace(value, qualified=1)
        doc = value.to_dict()
        doc["foreground_elapsed_ns"] = 10
        with self.assertRaises(InvalidModel):
            response.RequestResponseDelays.from_dict(doc)


if __name__ == "__main__":
    unittest.main()
