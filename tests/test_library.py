"""Public integration checks, independent of repository example/profile data."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llm_simulator import (
    AnalyticalCosts, Cache, DataPreparation, Device, EnvironmentEvent, Hardware,
    InvalidInput, Model, PhasePlacement, Route, Scheduling, Simulator,
    Unsupported, Workload,
)
from llm_simulator.demo import make_demo


class LibraryTests(unittest.TestCase):
    def test_dense_and_moe_use_the_real_multilayer_runtime(self):
        for kind in ("dense", "moe"):
            with self.subTest(kind=kind):
                simulator, scheduling = make_demo(kind)
                with simulator.compile(scheduling=scheduling) as compiled:
                    result = compiled.run()
                    self.assertTrue(result.success, result.summary())
                    self.assertGreater(result.makespan_ns, 0)
                    self.assertFalse(result.summary()["calibrated"])
                    self.assertEqual(result.summary()["cost_model"], "analytical_roofline")
                    model = compiled.plan._compiled.model
                    self.assertEqual(len(model.invocation_scopes), 6)
                    self.assertEqual({scope.layer_id for scope in model.invocation_scopes}, {"layer0", "layer1"})
                    self.assertTrue(model.aggregate.physical_boundary is not None)

    def test_no_saved_json_is_read_outside_generated_project(self):
        simulator, scheduling = make_demo("moe")
        read_bytes = Path.read_bytes
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder) / "project"
            def bounded_read(path):
                if path.suffix.lower() == ".json":
                    self.assertTrue(path.resolve().is_relative_to(project.resolve()), str(path))
                return read_bytes(path)
            with patch.object(Path, "read_bytes", bounded_read):
                with simulator.compile(scheduling=scheduling, directory=project) as compiled:
                    self.assertTrue(compiled.run().success)

    def test_phase_placement_and_preparation_are_compiled_and_executed(self):
        simulator, scheduling = make_demo("dense")
        def configure(plan):
            PhasePlacement(prefill={"projection": "gpu", "attention": "gpu"},
                           decode={"projection": "cpu", "attention": "cpu"}).apply(plan)
            plan.prepare(DataPreparation(layer_id="layer1", destination="vram", trigger="spare"))
        with simulator.compile(scheduling=scheduling, configure=configure) as compiled:
            result = compiled.run()
            self.assertTrue(result.success, result.summary())
            contexts = {task.operation_id: scope.phase
                        for scope in compiled.plan._compiled.model.invocation_scopes
                        for task in compiled.plan._compiled.model.aggregate.tasks
                        if task.operation_id in scope.operation_ids}
            graph = result.raw[1].execution_document()
            kinds = {task["operation_id"]: task["kind"] for task in graph["tasks"]}
            for event in result.events:
                if event.kind == "COMPUTE" and kinds.get(event.target_id) in {"projection", "attention"}:
                    self.assertEqual(event.device, "gpu" if contexts[event.target_id] == "prefill" else "cpu")
            self.assertTrue(compiled.ir_document()["data_preparations"])

    def test_cpu_only_needs_no_gpu_or_link_profile(self):
        simulator, _ = make_demo("dense")
        hardware = replace(simulator.hardware, gpu=None, vram_capacity_bytes=1,
                           transfer_bandwidth_bytes_per_second=None)
        simulator = Simulator(model=simulator.model, hardware=hardware,
            workload=simulator.workload, cost_model=AnalyticalCosts(workspace_bytes=16))
        result = simulator.run(scheduling=Scheduling(default_device="cpu", release="last_use"))
        self.assertTrue(result.success, result.summary())
        self.assertEqual(result.summary()["macro_contract"], "m2-layer-0.3")
        self.assertEqual(result.summary()["h2d_bytes"], 0)
        self.assertTrue(all(event.workspace_bytes == 16 for event in result.events if event.kind == "COMPUTE"))

    def test_gpu_only_runs_with_declared_link(self):
        simulator, scheduling = make_demo("dense")
        simulator = Simulator(model=simulator.model, hardware=replace(simulator.hardware, cpu=None),
            workload=simulator.workload, cost_model=AnalyticalCosts())
        self.assertTrue(simulator.run(scheduling=scheduling).success)

    def test_moe_batch_is_explicit_group_accounting(self):
        simulator, scheduling = make_demo("moe")
        routes = tuple(Route(f"request{request}", f"layer{layer}", position, (0, 1))
                       for request in range(2) for layer in range(2) for position in range(4))
        simulator = Simulator(model=simulator.model, hardware=simulator.hardware,
            workload=Workload(prefill_tokens=2, decode_steps=2, batch_size=2,
                              routes=routes, route_source="explicit-test-routes"),
            cost_model=AnalyticalCosts())
        result = simulator.run(scheduling=scheduling)
        self.assertTrue(result.success, result.summary())
        experts = [task for task in result.compiled.plan._compiled.model.aggregate.tasks if task.kind == "expert_ffn"]
        self.assertEqual(len(experts), 12)  # 2 experts x 2 layers x 3 rounds, not per-request duplicates.
        self.assertEqual(sorted({task.shape.tokens for task in experts}), [2, 4])

    def test_moe_missing_routes_is_rejected_instead_of_invented(self):
        simulator, scheduling = make_demo("moe")
        simulator = Simulator(model=simulator.model, hardware=simulator.hardware,
            workload=Workload(prefill_tokens=3), cost_model=AnalyticalCosts())
        with self.assertRaisesRegex(ValueError, "explicit routes"):
            simulator.run(scheduling=scheduling)

    def test_missing_moe_ssd_bandwidth_is_unknown_not_free(self):
        simulator, scheduling = make_demo("moe")
        simulator = Simulator(model=simulator.model,
            hardware=replace(simulator.hardware, ssd_write_bandwidth_bytes_per_second=None),
            workload=simulator.workload, cost_model=AnalyticalCosts())
        with self.assertRaisesRegex(Unsupported, "ssd_write"):
            simulator.run(scheduling=scheduling)

    def test_dense_batch_is_explicitly_unsupported(self):
        simulator, scheduling = make_demo("dense")
        simulator = Simulator(model=simulator.model, hardware=simulator.hardware,
            workload=Workload(prefill_tokens=3, batch_size=2), cost_model=AnalyticalCosts())
        with self.assertRaisesRegex(Unsupported, "batch_size=1"):
            simulator.run(scheduling=scheduling)

    def test_replay_and_repeated_execution_are_deterministic(self):
        simulator, scheduling = make_demo("moe")
        with simulator.compile(scheduling=scheduling) as compiled:
            partial = compiled.run(max_decisions=2)
            self.assertFalse(partial.success)
            resumed = compiled.run(checkpoint=partial.checkpoint())
            full = compiled.run()
            self.assertTrue(resumed.success)
            self.assertEqual(resumed.makespan_ns, full.makespan_ns)
            self.assertEqual(resumed.events, full.events)

    def test_mutated_sources_are_rejected(self):
        simulator, scheduling = make_demo("dense")
        with simulator.compile(scheduling=scheduling) as compiled:
            path = compiled.directory / "layer0-prefill.costs.json"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaises(InvalidInput):
                compiled.run()

    def test_device_blackout_delays_real_cpu_service(self):
        simulator, scheduling = make_demo("dense", device="cpu")
        baseline = simulator.run(scheduling=scheduling)
        delayed = simulator.run(scheduling=scheduling, environment_events=(
            EnvironmentEvent(t_ns=0, cpu_available=False),
            EnvironmentEvent(t_ns=100000, cpu_available=True),
        ))
        self.assertTrue(delayed.success, delayed.summary())
        self.assertEqual(delayed.makespan_ns, baseline.makespan_ns + 100000)

    def test_explicit_cpu_rate_scales_the_public_generated_multiround_workload(self):
        simulator, scheduling = make_demo("dense", device="cpu")
        baseline = simulator.run(scheduling=scheduling)
        delayed = simulator.run(scheduling=scheduling, environment_events=(EnvironmentEvent(
            0, cpu_service_rate_numerator=1, cpu_service_rate_denominator=2),))
        self.assertTrue(delayed.success, delayed.summary())
        self.assertEqual(delayed.makespan_ns, baseline.makespan_ns * 2)
        self.assertFalse(delayed.summary()["calibrated"])
        observations = [row.observation for row in delayed.raw[0].decisions]
        self.assertTrue(all((row.cpu_service_rate_numerator, row.cpu_service_rate_denominator) == (1, 2)
                            for row in observations))

    def test_mid_compute_rate_checkpoint_replays_fractional_work_exactly(self):
        simulator, scheduling = make_demo("dense", device="cpu")
        events = (EnvironmentEvent(0, cpu_service_rate_numerator=2, cpu_service_rate_denominator=3),
                  EnvironmentEvent(1, cpu_service_rate_numerator=5, cpu_service_rate_denominator=4))
        with simulator.compile(scheduling=scheduling, environment_events=events) as compiled:
            partial = compiled.run(max_decisions=2)  # Launch first CPU op, stop at its t=1 rate boundary.
            self.assertFalse(partial.success)
            self.assertEqual(partial.raw[1].observe().t_ns, 1)
            progress = partial.raw[0].runtime_result.compute_progress
            self.assertTrue(progress)
            self.assertEqual(progress[0].remaining_reference_ns_denominator, 3)
            resumed = compiled.run(checkpoint=partial.checkpoint())
            reference = compiled.run()
            self.assertTrue(resumed.success, resumed.summary())
            self.assertEqual(resumed.raw[0].runtime_result, reference.raw[0].runtime_result)
            self.assertEqual(resumed.events, reference.events)

    def test_zero_gpu_rate_waits_for_recovery_in_the_builtin_policy(self):
        simulator, scheduling = make_demo("dense", device="gpu")
        recovery = 100000
        delayed = simulator.run(scheduling=scheduling, environment_events=(
            EnvironmentEvent(0, gpu_service_rate_numerator=0, gpu_service_rate_denominator=1),
            EnvironmentEvent(recovery, gpu_service_rate_numerator=1, gpu_service_rate_denominator=1)))
        self.assertTrue(delayed.success, delayed.summary())
        computes = [event for event in delayed.events if event.kind == "COMPUTE"]
        self.assertTrue(computes)
        self.assertTrue(all(event.start_ns >= recovery for event in computes))
        self.assertTrue(any(row.observation.gpu_service_rate_numerator == 0
                            for row in delayed.raw[0].decisions))

    def test_impossible_vram_is_failure_not_cpu_fallback(self):
        simulator, scheduling = make_demo("dense")
        simulator = Simulator(model=simulator.model, hardware=replace(simulator.hardware, vram_capacity_bytes=1),
            workload=simulator.workload, cost_model=AnalyticalCosts())
        result = simulator.run(scheduling=scheduling)
        self.assertFalse(result.success)
        self.assertTrue(result.summary()["compilation_error"] or result.summary()["error_code"]
                        or result.summary()["termination_reason"] != "goal_reached")
        self.assertFalse(any(event.kind == "COMPUTE" and event.device == "cpu"
                             for event in result.events))

    def test_save_contains_inputs_and_truthful_cost_source(self):
        simulator, scheduling = make_demo("dense")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with simulator.compile(scheduling=scheduling, directory=root / "project") as compiled:
                result = compiled.run()
                result.save(root / "result")
                summary = json.loads((root / "result/library-summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["status"], "valid")
                self.assertFalse(summary["hardware_measurement"])
                self.assertTrue((root / "result/input-snapshot/model.global.json").is_file())
                self.assertTrue((root / "result/events.json").is_file())
                with self.assertRaises(InvalidInput):
                    result.save(root / "result")
            self.assertTrue((root / "project/model.global.json").is_file())

    def test_close_releases_temporary_files_and_rejects_reuse(self):
        simulator, scheduling = make_demo("dense")
        compiled = simulator.compile(scheduling=scheduling)
        path = compiled.directory
        self.assertTrue(path.is_dir())
        compiled.close()
        self.assertFalse(path.exists())
        with self.assertRaisesRegex(InvalidInput, "closed"):
            compiled.run()

    def test_existing_project_is_not_overwritten(self):
        simulator, scheduling = make_demo("dense")
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "keep").write_text("keep", encoding="utf-8")
            with self.assertRaises(InvalidInput):
                simulator.compile(scheduling=scheduling, directory=folder)
            self.assertEqual((Path(folder) / "keep").read_text(), "keep")

    def test_analytical_units_and_rounding_against_hand_calculation(self):
        costs = AnalyticalCosts()
        device = Device(compute_flops_per_second=1e9, memory_bandwidth_bytes_per_second=2e9,
                        launch_overhead_ns=7)
        self.assertEqual(costs.compute_duration_ns(device, flops=100, traffic_bytes=40), 107)
        self.assertEqual(costs.compute_duration_ns(device, flops=2, traffic_bytes=100), 57)
        hardware = Hardware(cpu=device, gpu=None, dram_capacity_bytes=1024,
            vram_capacity_bytes=1, transfer_bandwidth_bytes_per_second=2e9, transfer_latency_ns=3)
        self.assertEqual(costs.transfer_duration_ns(hardware, route="dram_to_vram", storage_bytes=5), 6)

    def test_invalid_geometry_and_nonfinite_rates_are_rejected(self):
        for rate in (0, -1, float("inf"), float("nan"), True):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                Device(rate, 1e9)
        with self.assertRaises(ValueError):
            Model.dense(hidden_size=15)
        with self.assertRaises(ValueError):
            Model.moe(experts=2, top_k=3)


if __name__ == "__main__":
    unittest.main()
