"""Finite request arrivals, private wakeups, per-request tokens and P/D state."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest

from llm_simulator import AnalyticalCosts, Device, Hardware, Model, Workload
from llm_infer_model.m2.global_execution import compile_global_execution_file, GlobalM2Session
from llm_infer_model.m2.request_flow import parse_request_flow, RequestArrivalBoundary
from llm_infer_model.tensor.engine import Action
from llm_infer_model.tensor.spec import InvalidInput, Unsupported, COPY_READY, OP_NOT_STARTED
from tensor_mapping.causal import SerialDemandPolicy, run_causal, observation_document


def write(path, document):
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")


def fixture(directory, *, arrivals=(0, 1000), lengths=(2, 2), decode=(1, 1)):
    directory = Path(directory)
    hardware = Hardware(Device(1e9, 1e9), Device(2e9, 2e9), 1 << 24, 1 << 24, 1e9)
    requests = []
    for number, (arrival, length, count) in enumerate(zip(arrivals, lengths, decode)):
        root = directory / f"source{number}"
        AnalyticalCosts().build_inputs(root, Model.dense(layers=2), hardware, Workload(length, decode_steps=count))
        for file in root.glob("*.json"):
            file.write_text(file.read_text(encoding="utf-8").replace("request0", f"r{number}"),
                            encoding="utf-8", newline="\n")
        raw = json.loads((root / "model.global.json").read_text(encoding="utf-8"))
        steps = [dict(id=step["id"], input_mode=step["input_mode"],
            gate_costs={device: dict(duration_ns=10, workspace_bytes=0) for device in ("cpu", "gpu")},
            invocations=[dict(row, id=f"r{number}-{row['id']}",
                invocation_file=f"source{number}/{row['invocation_file']}") for row in step["invocations"]])
            for step in raw["rounds"]]
        requests.append(dict(request_id=f"r{number}", arrival_ns=arrival, steps=steps))
    document = dict(format_version="global-execution-0.2", id="finite-flow", requests=requests)
    path = directory / "model.json"
    write(path, document)
    return path, document


class RequestFlowTests(unittest.TestCase):
    def test_gate_cost_key_order_does_not_select_execution_devices(self):
        with tempfile.TemporaryDirectory() as folder:
            path, raw = fixture(folder, decode=(2, 0))
            first = compile_global_execution_file(path)
            for request in raw["requests"]:
                for step in request["steps"]:
                    step["gate_costs"] = dict(reversed(list(step["gate_costs"].items())))
            write(path, raw)
            reversed_ = compile_global_execution_file(path)
            self.assertEqual(first.aggregate.workload_document, reversed_.aggregate.workload_document)
            self.assertEqual(first.aggregate.scenario_document, reversed_.aggregate.scenario_document)
            self.assertEqual(first.aggregate.tasks, reversed_.aggregate.tasks)
            for task in reversed_.aggregate.tasks:
                if task.kind == "state_update":
                    self.assertEqual(task.device, "cpu")

    def test_default_gate_requires_cpu_price_but_gpu_handoff_needs_only_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            path, raw = fixture(folder, decode=(2, 0))
            for step_index in (0, 2):
                wrong = deepcopy(raw)
                wrong["requests"][0]["steps"][step_index]["gate_costs"].pop("cpu")
                write(path, wrong)
                with self.subTest(step=step_index), self.assertRaisesRegex(Unsupported, "missing exact cpu"):
                    compile_global_execution_file(path, handoffs=(("r0", "vram"),))
            handoff_only_gpu = deepcopy(raw)
            handoff_only_gpu["requests"][0]["steps"][1]["gate_costs"].pop("cpu")
            write(path, handoff_only_gpu)
            compiled = compile_global_execution_file(path, handoffs=(("r0", "vram"),))
            handoff = compiled.aggregate.physical_boundary["request_flow"]["handoffs"][0]
            self.assertEqual(compiled.scenario.mapspace.devices_for(handoff["operation_id"]), ("gpu",))

    def test_explicit_arrival_cost_and_request_declarations_are_strict(self):
        with tempfile.TemporaryDirectory() as folder:
            _, raw = fixture(folder)
            for field, value in (("arrival_ns", True), ("arrival_ns", -1), ("steps", [])):
                wrong = deepcopy(raw); wrong["requests"][0][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(InvalidInput):
                    parse_request_flow(wrong)
            wrong = deepcopy(raw); wrong["requests"][0]["steps"][0]["gate_costs"] = None
            with self.assertRaisesRegex(InvalidInput, "input materialization"):
                parse_request_flow(wrong)
            with self.assertRaises(InvalidInput):
                RequestArrivalBoundary(t_ns=1, cpu_available=True)

    def test_future_prompt_size_does_not_change_public_initial_memory(self):
        observed = []
        for future_tokens in (2, 30):
            with tempfile.TemporaryDirectory() as folder:
                path, _ = fixture(folder, arrivals=(0, 100000), lengths=(2, future_tokens))
                session = GlobalM2Session(compile_global_execution_file(path))
                observation = session.observe()
                self.assertEqual(observation.arrived_requests, ("r0",))
                self.assertFalse(any(op.request_id == "r1" for op in observation.operations))
                self.assertFalse(any(":r1:" in tensor.id for tensor in observation.tensors))
                observed.append(observation_document(observation))
        self.assertEqual(observed[0], observed[1])

    def test_arrival_is_an_exact_private_boundary_during_running_compute(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = fixture(folder, arrivals=(0, 1000))
            session = GlobalM2Session(compile_global_execution_file(path))
            first = session.observe().operations[0]
            self.assertEqual(first.id, "g:r0:request_arrival")
            self.assertTrue(session.submit(Action("COMPUTE", first.id, device="cpu")).ok)
            session.advance()
            projection = next(op for op in session.observe().operations if op.kind == "projection")
            self.assertTrue(session.submit(Action("COMPUTE", projection.id, device="cpu")).ok)
            step = session.advance()
            self.assertEqual(step.t_ns, 1000)
            self.assertIn("r1", session.observe().arrived_requests)
            self.assertTrue(session.observe().running)
            self.assertTrue(all(op.request_id != "r1" or op.id == "g:r1:request_arrival"
                                for op in session.observe().operations))

    def test_all_future_arrivals_wake_idle_runtime_without_public_timer(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = fixture(folder, arrivals=(100, 300))
            session = GlobalM2Session(compile_global_execution_file(path))
            self.assertEqual(session.observe().operations, ())
            self.assertFalse(hasattr(session.observe(), "next_request_arrival_ns"))
            self.assertTrue(session.advance().ok)
            self.assertEqual(session.observe().t_ns, 100)
            self.assertEqual(session.observe().arrived_requests, ("r0",))

    def test_different_lifetimes_keep_per_request_token_dependencies(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = fixture(folder, arrivals=(0, 1000), decode=(2, 0))
            compiled = compile_global_execution_file(path, visibility_mode="data_dependencies")
            gates = dict(compiled.gate_operations)
            token = "g:r0-decode0:trace_next:r0"
            self.assertTrue(gates[token])
            self.assertTrue(all(":r0:" in oid for oid in gates[token]))
            session = GlobalM2Session(compiled)
            run = run_causal(SerialDemandPolicy("cpu"), session, max_decisions=1000)
            self.assertEqual(run.termination_reason, "goal_reached")
            self.assertEqual(session.observe().retired_requests, ("r0", "r1"))

    def test_handoff_has_all_layer_prefill_state_and_one_explicit_device(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = fixture(folder)
            compiled = compile_global_execution_file(path, handoffs=(("r0", "dram"),))
            boundary = compiled.aggregate.physical_boundary["request_flow"]
            handoff = boundary["handoffs"][0]
            self.assertEqual(len(handoff["kv_tensor_ids"]), 4)
            op = compiled.scenario.workload.operation_by_id[handoff["operation_id"]]
            self.assertEqual(set(op.inputs), {handoff["hidden_tensor_id"], *handoff["kv_tensor_ids"]})
            self.assertEqual(compiled.scenario.mapspace.devices_for(op.id), ("cpu",))
            self.assertTrue(all(compiled.scenario.workload.tensor_by_id[tid].storage_bytes == 64
                                for tid in handoff["kv_tensor_ids"]))

    def test_handoff_fixes_actual_first_decode_kv_readers_only(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = fixture(folder, decode=(2, 0))
            compiled = compile_global_execution_file(path, handoffs=(("r0", "dram"),))
            handoff = compiled.aggregate.physical_boundary["request_flow"]["handoffs"][0]
            self.assertEqual(len(handoff["kv_reader_operation_ids"]), 2)
            for oid in handoff["kv_reader_operation_ids"]:
                self.assertEqual(compiled.scenario.mapspace.devices_for(oid), ("cpu",))
                self.assertEqual(set(compiled.aggregate.scenario_document["costs"]["compute"][oid]), {"cpu"})
            later = [task.operation_id for task in compiled.aggregate.tasks
                     if "r0-layer" in task.operation_id and "decode1" in task.operation_id and task.kind == "attention"]
            self.assertEqual(len(later), 2)
            self.assertTrue(all(set(compiled.scenario.mapspace.devices_for(oid)) == {"cpu", "gpu"} for oid in later))

    def test_runtime_override_cannot_move_first_decode_kv_reader_elsewhere(self):
        from tensor_mapping.causal import compile_command
        from llm_infer_model.tensor.runtime import dispatch_command
        with tempfile.TemporaryDirectory() as folder:
            path, _ = fixture(folder, arrivals=(0, 1000000))
            compiled = compile_global_execution_file(path, handoffs=(("r0", "dram"),))
            handoff = compiled.aggregate.physical_boundary["request_flow"]["handoffs"][0]
            readers = set(handoff["kv_reader_operation_ids"])
            session, policy = GlobalM2Session(compiled), SerialDemandPolicy("cpu")
            for index in range(100):
                observation = session.observe()
                target = next((op for op in observation.operations if op.id in readers), None)
                if target is not None:
                    break
                dispatch_command(session, compile_command(policy(observation), observation), source_index=index)
            self.assertIsNotNone(target)
            self.assertEqual([price.device for price in target.prices], ["cpu"])
            rejected = session.submit(Action("COMPUTE", target.id, device="gpu"))
            self.assertFalse(rejected.ok)
            self.assertIn("not an allowed device", rejected.error_message)

    def test_handoff_requires_decode_and_matching_exact_gate_cost(self):
        with tempfile.TemporaryDirectory() as folder:
            path, raw = fixture(folder, decode=(0, 0))
            with self.assertRaisesRegex(InvalidInput, "at least one"):
                compile_global_execution_file(path, handoffs=(("r0", "vram"),))

    def test_first_decode_gate_rejects_one_missing_kv_after_completed_prefill(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = fixture(folder, arrivals=(0, 1000000), decode=(1, 0))
            compiled = compile_global_execution_file(path, handoffs=(("r0", "dram"),))
            session = GlobalM2Session(compiled)
            handoff = compiled.aggregate.physical_boundary["request_flow"]["handoffs"][0]
            target = handoff["operation_id"]
            for _ in range(200):
                observation = session.observe()
                if any(op.id == target for op in observation.operations):
                    break
                if observation.running:
                    self.assertTrue(session.advance().ok); continue
                tensors = {tensor.id: tensor for tensor in observation.tensors}
                candidates = [op for op in observation.operations if op.status == OP_NOT_STARTED
                              and all(any(status == COPY_READY for _, status in tensors[tid].copies) for tid in op.inputs)]
                op = candidates[0]
                device = "cpu" if op.kind == "state_update" else "gpu"
                destination = "dram" if device == "cpu" else "vram"
                missing = next((tid for tid in op.inputs if tensors[tid].status_at(destination) != COPY_READY), None)
                action = (Action("COPY", missing, src="dram" if destination == "vram" else "vram", dst=destination)
                          if missing else Action("COMPUTE", op.id, device=device))
                self.assertTrue(session.submit(action).ok)
            self.assertTrue(all(scope.id in session.observe().completed_invocations
                                for scope in compiled.invocation_scopes if scope.round_id == "r0-prefill"))
            withheld = handoff["kv_tensor_ids"][-1]
            for tid in (handoff["hidden_tensor_id"], *handoff["kv_tensor_ids"][:-1]):
                self.assertTrue(session.submit(Action("COPY", tid, src="vram", dst="dram")).ok)
                self.assertTrue(session.advance().ok)
            self.assertNotEqual(session._runtime.state.copy_at(withheld, "dram"), COPY_READY)
            rejected = session.submit(Action("COMPUTE", target, device="cpu"))
            self.assertFalse(rejected.ok)
            self.assertIn(withheld, rejected.error_message)
        with tempfile.TemporaryDirectory() as folder:
            path, raw = fixture(folder)
            raw["requests"][0]["steps"][1]["gate_costs"].pop("gpu")
            write(path, raw)
            with self.assertRaisesRegex(Unsupported, "missing exact gpu"):
                compile_global_execution_file(path, handoffs=(("r0", "vram"),))

    def test_retired_kv_is_releasable_without_deleting_its_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = fixture(folder, arrivals=(0, 1000000), decode=(0, 0))
            session = GlobalM2Session(compile_global_execution_file(path))
            policy = SerialDemandPolicy("cpu")
            from tensor_mapping.causal import compile_command
            from llm_infer_model.tensor.runtime import dispatch_command
            for index in range(100):
                observation = session.observe()
                if "r0" in observation.retired_requests:
                    break
                dispatch_command(session, compile_command(policy(observation), observation), source_index=index)
            self.assertIn("r0", session.observe().retired_requests)
            self.assertNotIn("r1", session.observe().arrived_requests)
            tid, location = next((tid, location) for tid, location in session.observe().release_candidates if tid.startswith("kv:"))
            self.assertNotIn(tid, session._runtime.scenario.goal.output_locations)
            before = next(item.used_bytes for item in session.observe().memory if item.domain == location)
            size = session._runtime.scenario.workload.tensor_by_id[tid].storage_bytes
            self.assertTrue(session.submit(Action("EVICT", tid, location=location)).ok)
            after = next(item.used_bytes for item in session.observe().memory if item.domain == location)
            self.assertEqual(before - after, size)
            self.assertIn(tid, session._runtime.scenario.workload.tensor_by_id)

    def test_single_request_moe_groups_keep_real_request_for_round_robin(self):
        from llm_simulator import Route
        from llm_simulator.request_flow import Request, RequestFlow, build_request_flow_inputs
        from mapping.tensor_mapping.request_scheduling import RequestPipeline, order_request_operations
        with tempfile.TemporaryDirectory() as folder:
            device = Device(1e9, 1e9)
            hardware = Hardware(device, device, 1 << 24, 1 << 24, 1e9, 1e9, 1e9)
            workload = RequestFlow((Request("r0", 1), Request("r1", 1)),
                routes=(Route("r0", "layer0", 0, (0,)), Route("r1", "layer0", 0, (1,))),
                route_source="explicit independent test routes")
            inputs = build_request_flow_inputs(AnalyticalCosts(), folder,
                Model.moe(layers=1, experts=2, top_k=1), hardware, workload)
            compiled = compile_global_execution_file(inputs["model_file"])
            groups = [task for task in compiled.aggregate.tasks if task.kind == "expert_ffn"]
            self.assertEqual({task.request_id for task in groups}, {"r0", "r1"})
            self.assertTrue(all({request for request, _ in task.work_items} == {task.request_id} for task in groups))
            session = GlobalM2Session(compiled)
            run = run_causal(SerialDemandPolicy("cpu"), session, max_decisions=1000)
            self.assertEqual(run.termination_reason, "goal_reached")
            public_groups = [op for op in session.observe().operations if op.kind == "expert_ffn"]
            self.assertEqual({op.request_id for op in public_groups}, {"r0", "r1"})
            ordered = order_request_operations(public_groups, RequestPipeline("round_robin"), last_request="r0")
            self.assertEqual(ordered[0].request_id, "r1")


if __name__ == "__main__":
    unittest.main()
