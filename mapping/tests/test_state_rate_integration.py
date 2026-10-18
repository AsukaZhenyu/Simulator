"""Public rate parsing, current estimates and deterministic replay contracts."""
from dataclasses import asdict, replace
from types import SimpleNamespace
import unittest

from llm_infer_model.m2.causal import (
    compute_service_available, current_compute_duration_ns,
)
from llm_infer_model.tensor.environment import ComputeProgress, EnvironmentEvent
from llm_infer_model.tensor.spec import InvalidInput
from tensor_mapping.causal import Decision, SerialDemandPolicy
from tensor_mapping.causal_cli import parse_environment
from tensor_mapping.policy_executor import DeclarativePolicy
from tensor_mapping.reconfiguration import parse_environment_trace
from tensor_mapping.unified_policy import UnifiedPolicy
from tensor_mapping.unified_strategy import _runtime_signature

from .test_policy_executor import small_observation


class StateRateIntegrationTests(unittest.TestCase):
    def test_trace_extension_roundtrips_and_control_trace_reuses_same_parser(self):
        row = dict(t_ns=7, cpu_service_rate_numerator=2, cpu_service_rate_denominator=3,
                   gpu_service_rate_numerator=0, gpu_service_rate_denominator=1)
        base = dict(format_version="environment-trace-0.1", events=[row])
        event, = parse_environment(base)
        self.assertEqual(event, EnvironmentEvent(**row))
        self.assertEqual({key: value for key, value in asdict(event).items() if value is not None}, row)
        control = dict(format_version="environment-trace-0.2", events=[row], requests=[],
            control_costs=dict(ack_ns=0, decline_ns=0, decision_ns=0,
                              source="explicit-test", timing_scope="dedicated_serial_control_resource"))
        self.assertEqual(parse_environment_trace(control).events, (event,))

    def test_missing_or_null_rate_parts_are_rejected_without_guessing(self):
        for row in (dict(t_ns=0, cpu_service_rate_numerator=1),
                    dict(t_ns=0, cpu_service_rate_numerator=1, cpu_service_rate_denominator=None),
                    dict(t_ns=0, gpu_service_rate=0.5)):
            with self.subTest(row=row), self.assertRaises(InvalidInput):
                parse_environment(dict(format_version="environment-trace-0.1", events=[row]))

    def test_current_duration_is_a_separate_exact_ceil_estimate(self):
        base = replace(small_observation(), cpu_service_rate_numerator=2, cpu_service_rate_denominator=3,
                       gpu_service_rate_numerator=0)
        self.assertEqual(current_compute_duration_ns(11, base, "cpu"), 17)
        self.assertIsNone(current_compute_duration_ns(11, base, "gpu"))
        self.assertFalse(compute_service_available(base, "gpu"))
        self.assertIsNone(current_compute_duration_ns(11, replace(base, cpu_available=False), "cpu"))

    def test_both_builtin_resource_checks_skip_zero_rate_but_not_the_link(self):
        base = replace(small_observation(), cpu_service_rate_numerator=0, gpu_service_rate_numerator=0)
        for policy in (DeclarativePolicy, UnifiedPolicy):
            with self.subTest(policy=policy.__name__):
                self.assertFalse(policy._resource_available(base, "cpu_compute", compute=True))
                self.assertFalse(policy._resource_available(base, "gpu_compute", compute=True))
                self.assertTrue(policy._resource_available(base, "pcie_copy"))

    def test_serial_demand_waits_instead_of_submitting_zero_rate_compute(self):
        base = small_observation()
        # Make every selected input resident so the next decision is compute or wait.
        tensors = tuple(replace(tensor, copies=(("dram", "READY"), ("vram", "READY"), ("ssd", "ABSENT")))
                        for tensor in base.tensors)
        for device in ("cpu", "gpu"):
            unavailable = replace(base, tensors=tensors, **{f"{device}_service_rate_numerator": 0})
            policy = SerialDemandPolicy(device)
            self.assertEqual(policy(unavailable), Decision("ADVANCE"))

    def test_replay_signature_binds_fractional_work_even_when_integer_state_matches(self):
        progress = ComputeProgress("cpu_compute", "a", 28, 3, 2, 3)
        from llm_infer_model.tensor.engine import State
        state = State(t=1, op_status=(), copy_status=(), running=(), dram_status=(), ssd_status=())
        common = dict(final_state=state, environment_events=(), service_intervals=(),
                      nominal_ends=(), action_sources=(), reclaim_records=(), control_events=())
        observation = SimpleNamespace(pending_reclaim=None, control_running=None, services=None)
        def session(value):
            return SimpleNamespace(result=lambda: SimpleNamespace(**common, compute_progress=(value,)),
                                   observe=lambda: observation)
        first = _runtime_signature(session(progress))
        second = _runtime_signature(session(replace(progress, remaining_reference_ns_numerator=27)))
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
