"""Policy dead ends are unsuccessful; future resource restoration still works."""
import tempfile
import unittest

from llm_simulator import (AnalyticalCosts, Decision, Device, EnvironmentEvent,
    Hardware, Model, RuntimePolicy, Scheduling, Simulator, Workload)
from llm_infer_model.tensor.environment import CODE_NO_PROGRESS


def hardware(*, vram=65536, gpu=True):
    return Hardware(Device(1e9, 1e9), Device(2e9, 2e9) if gpu else None,
                    1 << 20, vram, 1e9, 1e9, 1e9)


def simulator(machine):
    return Simulator(model=Model.dense(layers=1, hidden_size=8, intermediate_size=16,
                                      heads=2, kv_heads=1), hardware=machine,
                     workload=Workload(1), cost_model=AnalyticalCosts())


class IdleEvenWithReadyWork(RuntimePolicy):
    def decide(self, observation):
        return Decision("ADVANCE")


class ComputeBeforeInputs(RuntimePolicy):
    def decide(self, observation):
        operation = next(op for op in observation.operations if op.kind == "projection")
        return Decision("COMPUTE", operation.id, device="gpu")


class PolicyNoProgressTests(unittest.TestCase):
    def run_case(self, machine, *, device="gpu", events=(), configure=None):
        with tempfile.TemporaryDirectory() as directory:
            with simulator(machine).compile(directory=directory,
                    scheduling=Scheduling(default_device=device, release="last_use"),
                    environment_events=events, configure=configure) as compiled:
                return compiled.run()

    def test_impossible_fixed_gpu_plan_is_unsuccessful_policy_dead_end_no_fallback(self):
        result = self.run_case(hardware(vram=1))
        self.assertFalse(result.success)
        self.assertEqual(result.raw[0].termination_reason, "policy_dead_end")
        self.assertEqual(result.evaluation.error_code, CODE_NO_PROGRESS)
        self.assertIsNone(result.makespan_ns)
        self.assertNotEqual(result.status, "infeasible")  # No global impossibility proof.
        self.assertFalse(any(event.kind == "COMPUTE" and event.device == "cpu" for event in result.events))
        self.assertEqual(result.evaluation.peak_vram_bytes, 0)

    def test_idle_bad_policy_is_not_mistaken_for_hardware_infeasibility(self):
        machine = hardware(gpu=False)
        baseline = self.run_case(machine, device="cpu")
        self.assertTrue(baseline.success)
        result = self.run_case(machine, device="cpu",
                               configure=lambda plan: plan.runtime_policy(IdleEvenWithReadyWork))
        self.assertFalse(result.success)
        self.assertEqual(result.raw[0].termination_reason, "policy_dead_end")
        self.assertEqual(result.evaluation.error_code, CODE_NO_PROGRESS)
        self.assertNotEqual(result.status, "infeasible")

    def test_empty_running_list_can_wait_for_future_cpu_restoration(self):
        machine = hardware(gpu=False)
        baseline = self.run_case(machine, device="cpu")
        restored = self.run_case(machine, device="cpu", events=(
            EnvironmentEvent(0, cpu_available=False), EnvironmentEvent(10000, cpu_available=True)))
        self.assertTrue(restored.success, restored.summary())
        self.assertEqual(restored.makespan_ns, baseline.makespan_ns + 10000)
        self.assertEqual(restored.raw[0].decisions[0].action.kind, "ADVANCE")
        self.assertFalse(restored.raw[0].decisions[0].observation.running)

    def test_empty_running_list_can_wait_for_future_memory_restoration(self):
        machine = hardware()
        baseline = self.run_case(machine)
        restored = self.run_case(machine, events=(
            EnvironmentEvent(0, vram_external_bytes=machine.vram_capacity_bytes),
            EnvironmentEvent(10000, vram_external_bytes=0)))
        self.assertTrue(restored.success, restored.summary())
        self.assertEqual(restored.makespan_ns, baseline.makespan_ns + 10000)
        self.assertEqual(restored.raw[0].decisions[0].action.kind, "ADVANCE")
        self.assertFalse(restored.raw[0].decisions[0].observation.running)

    def test_malformed_early_compute_keeps_existing_readiness_failure(self):
        result = self.run_case(hardware(), configure=lambda plan: plan.runtime_policy(ComputeBeforeInputs))
        self.assertFalse(result.success)
        self.assertEqual(result.raw[0].termination_reason, "runtime_failure")
        self.assertEqual(result.evaluation.error_code, "INPUT_NOT_READY")
        self.assertNotEqual(result.evaluation.error_code, CODE_NO_PROGRESS)


if __name__ == "__main__":
    unittest.main()
