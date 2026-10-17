"""Synthetic fixed-plan IR/compiler/runtime integration, without any search."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llm_infer_model.tensor.engine import (
    ACTION_ADVANCE, ACTION_COMPUTE, ACTION_COPY, ACTION_READ_SSD, ACTION_WRITE_SSD,
    CODE_CAPACITY_EXCEEDED, CODE_INPUT_NOT_READY, CODE_RESOURCE_BUSY,
    STATUS_INVALID_MAPPING, STATUS_VALID, evaluate_mapping, load_mapping,
)
from llm_infer_model.tensor.program import ActionStep, DeviceStateStep, WaitStep
from llm_infer_model.tensor.spec import COPY_READY, InvalidInput, Unsupported, build_scenario
from tensor_mapping.strategy import main, write_run
from tensor_mapping.strategy_compiler import (
    compile_strategy, compile_strategy_file, compiled_document, run_strategy,
)
from tensor_mapping.strategy_ir import parse_strategy_ir


_SIMULATOR = Path(__file__).resolve().parents[2]
_FIXED = Path(__file__).resolve().parent / "fixtures/fixed_plans"


def documents(case="prefetch-overlap"):
    base = _FIXED / case
    return (json.loads((base / "graph.json").read_text(encoding="utf-8")),
            json.loads((base / "scenario.json").read_text(encoding="utf-8")))


def strategy_document(*, serial=False):
    compute0 = {"kind": "compute", "operation_id": "layer0"}
    compute1 = {"kind": "compute", "operation_id": "layer1"}
    transfer = {"kind": "transfer", "tensor_id": "W:L1", "src": "dram", "dst": "vram"}
    release = {"kind": "release", "tensor_id": "W:L0", "location": "vram"}
    if serial:
        program = [compute0, {"kind": "wait", "operations": ["layer0"], "copies": []},
                   release, transfer,
                   {"kind": "wait", "operations": [],
                    "copies": [{"tensor_id": "W:L1", "location": "vram"}]},
                   compute1, {"kind": "drain"}]
    else:
        program = [compute0, transfer,
                   {"kind": "wait", "operations": ["layer0"],
                    "copies": [{"tensor_id": "W:L1", "location": "vram"}]},
                   release, compute1, {"kind": "drain"}]
    return {"format_version": "strategy-ir-0.1", "id": "declared-prefetch-plan",
            "scenario_file": "scene.json", "information_mode": "offline_fixed",
            "regions": [{"id": "L0", "kind": "layer", "members": ["layer0"]},
                        {"id": "L1", "kind": "layer", "members": ["layer1"]}],
            "placements": {"layer0": "gpu", "layer1": "gpu"}, "program": program,
            "device_states": {}}


def compile_memory(raw=None, *, case="prefetch-overlap"):
    return compile_strategy(parse_strategy_ir(raw or strategy_document()),
                            build_scenario(*documents(case)))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def materialize(directory, raw=None, workload=None, scene=None):
    """Write fresh copies in a temporary directory; historical fixtures are read-only."""
    directory = Path(directory)
    if workload is None or scene is None:
        workload, scene = documents()
    scene = deepcopy(scene)
    scene["workload_file"] = "graph.json"
    write_json(directory / "graph.json", workload)
    write_json(directory / "scene.json", scene)
    write_json(directory / "strategy.json", raw or strategy_document())
    return directory / "strategy.json"


class StrategyCompilerTests(unittest.TestCase):
    def test_symbolic_wait_expands_two_events_and_matches_historical_7ms_replay(self):
        compiled = compile_memory()
        actual = run_strategy(compiled)
        previous = load_mapping(_FIXED / "prefetch-overlap/mapping.json", compiled.scenario)
        self.assertEqual(actual.evaluation, evaluate_mapping(compiled.scenario, previous.actions))
        self.assertEqual(actual.evaluation.status, STATUS_VALID)
        self.assertEqual(actual.evaluation.makespan_ns, 7_000_000)
        advances = [(action.kind, source) for action, source in
                    zip(actual.evaluation.actions, actual.action_sources) if source == 2]
        self.assertEqual(advances, [(ACTION_ADVANCE, 2), (ACTION_ADVANCE, 2)])
        self.assertEqual(actual.action_sources, (0, 1, 2, 2, 3, 4, 5))
        self.assertEqual([event.t_ns for event in actual.evaluation.events if event.kind == ACTION_ADVANCE],
                         [3_000_000, 5_000_000, 7_000_000])
        self.assertTrue(all(profile == actual.action_profiles[0] for profile in actual.action_profiles))

    def test_regions_are_annotations_and_create_no_implicit_barrier(self):
        grouped = compile_memory()
        raw = strategy_document()
        raw["regions"] = []
        ungrouped = compile_memory(raw)
        self.assertEqual(grouped.steps, ungrouped.steps)
        self.assertEqual(run_strategy(grouped).evaluation, run_strategy(ungrouped).evaluation)
        report = compiled_document(grouped)
        self.assertTrue(all(region["execution_barrier"] is False for region in report["regions"]))
        self.assertEqual([step["kind"] for step in report["program"]],
                         ["action", "action", "wait", "action", "action", "wait"])

    def test_10ms_serial_plan_and_capacity_refusal_use_same_runtime(self):
        serial = compile_memory(strategy_document(serial=True), case="prefetch-small-serial")
        self.assertEqual(run_strategy(serial).evaluation.makespan_ns, 10_000_000)
        overlap = compile_memory(case="prefetch-small-overlap-refused")
        result = run_strategy(overlap)
        self.assertEqual(result.evaluation.status, STATUS_INVALID_MAPPING)
        self.assertEqual(result.evaluation.error_code, CODE_CAPACITY_EXCEEDED)
        self.assertEqual(result.error_source_index, 1)
        self.assertEqual(result.evaluation.action_count, 1)
        self.assertEqual(result.evaluation.error_t_ns, 0)
        self.assertIsNone(result.evaluation.makespan_ns)

    def test_placement_inventory_is_exact_and_device_must_be_permitted(self):
        for placements in ({"layer0": "gpu"}, {"layer0": "gpu", "layer1": "gpu", "extra": "gpu"}):
            raw = strategy_document()
            raw["placements"] = placements
            with self.subTest(placements=placements):
                with self.assertRaisesRegex(InvalidInput, "cover the graph exactly"):
                    compile_memory(raw)
        raw = strategy_document()
        raw["placements"]["layer1"] = "cpu"
        with self.assertRaisesRegex(Unsupported, "does not permit"):
            compile_memory(raw)

    def test_unknown_region_member_is_rejected_without_renaming_or_expansion(self):
        raw = strategy_document()
        raw["regions"][1]["members"].append("not-a-node")
        with self.assertRaisesRegex(InvalidInput, "unknown operations"):
            compile_memory(raw)

    def test_missing_selected_compute_or_transfer_cost_is_rejected_without_guessing(self):
        scenario = build_scenario(*documents())
        compute = deepcopy(scenario.costs.compute)
        del compute["layer1"]["gpu"]
        incomplete = replace(scenario, costs=replace(scenario.costs, compute=compute))
        with self.assertRaisesRegex(InvalidInput, "missing a duration.*gpu"):
            compile_strategy(parse_strategy_ir(strategy_document()), incomplete)
        transfer = deepcopy(scenario.costs.transfer)
        del transfer["dram_to_vram"]["W:L1"]
        incomplete = replace(scenario, costs=replace(scenario.costs, transfer=transfer))
        with self.assertRaisesRegex(InvalidInput, "missing a duration.*W:L1"):
            compile_strategy(parse_strategy_ir(strategy_document()), incomplete)

    def test_compute_is_required_exactly_once_per_declared_node(self):
        raw = strategy_document()
        raw["program"].insert(1, {"kind": "compute", "operation_id": "layer0"})
        with self.assertRaisesRegex(InvalidInput, "duplicate compute"):
            compile_memory(raw)
        raw = strategy_document()
        raw["program"] = [item for item in raw["program"] if item.get("operation_id") != "layer1"]
        with self.assertRaisesRegex(InvalidInput, "omits compute"):
            compile_memory(raw)
        raw = strategy_document()
        raw["program"][0]["operation_id"] = "unknown"
        with self.assertRaisesRegex(InvalidInput, "unknown operation"):
            compile_memory(raw)

    def test_wait_targets_and_tensor_references_are_checked_against_graph(self):
        for instruction, message in (
            ({"kind": "wait", "operations": ["unknown"], "copies": []}, "unknown operation"),
            ({"kind": "wait", "operations": [], "copies": [{"tensor_id": "unknown", "location": "dram"}]}, "unknown tensor"),
            ({"kind": "transfer", "tensor_id": "unknown", "src": "dram", "dst": "vram"}, "unknown tensor"),
            ({"kind": "release", "tensor_id": "unknown", "location": "dram"}, "unknown tensor"),
        ):
            raw = strategy_document()
            raw["program"].insert(0, instruction)
            with self.subTest(instruction=instruction):
                with self.assertRaisesRegex(InvalidInput, message):
                    compile_memory(raw)

    def test_ssd_to_vram_expands_both_priced_hops_and_visible_middle_wait(self):
        workload, scene = documents()
        next(t for t in workload["tensors"] if t["id"] == "W:L1")["initial_locations"] = ["ssd"]
        scene["costs"]["transfer"]["ssd_to_dram"]["W:L1"] = {"duration_ns": 1_000_000}
        scene["mapspace"]["allowed_transfers"]["ssd_to_dram"] = ["W:L1"]
        raw = strategy_document()
        raw["program"][1]["src"] = "ssd"
        compiled = compile_strategy(parse_strategy_ir(raw), build_scenario(workload, scene))
        self.assertEqual(len(compiled.steps), len(raw["program"]) + 2)
        first_hop, middle_wait, second_hop = compiled.steps[1:4]
        self.assertIsInstance(first_hop, ActionStep)
        self.assertEqual(first_hop.action.kind, ACTION_READ_SSD)
        self.assertIsInstance(middle_wait, WaitStep)
        self.assertEqual(middle_wait.copies, (("W:L1", "dram"),))
        self.assertIsInstance(second_hop, ActionStep)
        self.assertEqual((second_hop.action.kind, second_hop.action.src, second_hop.action.dst),
                         (ACTION_COPY, "dram", "vram"))
        self.assertEqual([step.source_index for step in compiled.steps[1:4]], [1, 1, 1])
        result = run_strategy(compiled)
        self.assertEqual(result.evaluation.status, STATUS_VALID)
        self.assertEqual(result.evaluation.makespan_ns, 7_000_000)
        self.assertEqual((result.evaluation.ssd_read_bytes, result.evaluation.h2d_bytes), (64, 64))
        transfer_events = [event for event in result.evaluation.events
                           if event.kind in (ACTION_READ_SSD, ACTION_COPY)]
        self.assertEqual([(event.start_ns, event.end_ns) for event in transfer_events],
                         [(0, 1_000_000), (1_000_000, 4_000_000)])
        self.assertEqual([step["kind"] for step in compiled_document(compiled)["program"][1:4]],
                         ["action", "wait", "action"])

    def test_vram_to_ssd_expands_d2h_wait_then_write(self):
        workload, scene = documents()
        # Delivery to an explicit persistent goal makes both staging and write
        # useful; an already-consumed non-output weight would correctly refuse.
        workload["outputs"].append("W:L0")
        scene["goal"]["output_locations"]["W:L0"] = "ssd"
        scene["costs"]["transfer"]["vram_to_dram"]["W:L0"] = {"duration_ns": 1_000_000}
        scene["costs"]["transfer"]["dram_to_ssd"]["W:L0"] = {"duration_ns": 2_000_000}
        scene["mapspace"]["allowed_transfers"]["vram_to_dram"] = ["W:L0"]
        scene["mapspace"]["allowed_transfers"]["dram_to_ssd"] = ["W:L0"]
        raw = strategy_document()
        raw["program"][3:4] = [
            {"kind": "release", "tensor_id": "W:L0", "location": "dram"},
            {"kind": "transfer", "tensor_id": "W:L0", "src": "vram", "dst": "ssd"},
            {"kind": "drain"},
            {"kind": "release", "tensor_id": "W:L0", "location": "vram"},
        ]
        compiled = compile_strategy(parse_strategy_ir(raw), build_scenario(workload, scene))
        hop_steps = [step for step in compiled.steps if step.source_index == 4]
        self.assertEqual([type(step) for step in hop_steps], [ActionStep, WaitStep, ActionStep])
        self.assertEqual(hop_steps[0].action.kind, ACTION_COPY)
        self.assertEqual((hop_steps[0].action.src, hop_steps[0].action.dst), ("vram", "dram"))
        self.assertEqual(hop_steps[1].copies, (("W:L0", "dram"),))
        self.assertEqual(hop_steps[2].action.kind, ACTION_WRITE_SSD)
        result = run_strategy(compiled)
        self.assertEqual(result.evaluation.status, STATUS_VALID, result.evaluation.reason)
        self.assertEqual(result.evaluation.makespan_ns, 10_000_000)
        self.assertEqual((result.evaluation.d2h_bytes, result.evaluation.ssd_write_bytes), (64, 64))
        self.assertEqual(result.final_state.copy_at("W:L0", "ssd"), COPY_READY)

    def test_two_hop_request_requires_each_actual_route_and_cost(self):
        original = build_scenario(*documents())
        raw = strategy_document()
        raw["program"][1]["src"] = "ssd"
        with self.assertRaisesRegex(Unsupported, "ssd_to_dram.*not permitted"):
            compile_strategy(parse_strategy_ir(raw), original)
        # The compiler also revalidates programmatic scenarios against the same
        # spec: an allowed physical route cannot exist without its exact price.
        workload, scene = documents()
        next(t for t in workload["tensors"] if t["id"] == "W:L1")["initial_locations"] = ["ssd"]
        scene["costs"]["transfer"]["ssd_to_dram"]["W:L1"] = {"duration_ns": 1_000_000}
        scene["mapspace"]["allowed_transfers"]["ssd_to_dram"] = ["W:L1"]
        priced = build_scenario(workload, scene)
        transfer = deepcopy(priced.costs.transfer)
        del transfer["ssd_to_dram"]["W:L1"]
        missing_cost = replace(priced, costs=replace(priced.costs, transfer=transfer))
        with self.assertRaisesRegex(InvalidInput, "ssd_to_dram.*missing a duration"):
            compile_strategy(parse_strategy_ir(raw), missing_cost)

    def test_runtime_missing_dependency_or_unscheduled_wait_is_not_repaired(self):
        raw = strategy_document()
        raw["program"] = [
            {"kind": "compute", "operation_id": "layer0"},
            {"kind": "wait", "operations": ["layer0"], "copies": []},
            {"kind": "compute", "operation_id": "layer1"},
            {"kind": "drain"},
        ]
        result = run_strategy(compile_memory(raw))
        self.assertEqual(result.evaluation.status, STATUS_INVALID_MAPPING)
        self.assertEqual(result.evaluation.error_code, CODE_INPUT_NOT_READY)
        self.assertEqual(result.error_source_index, 2)
        self.assertEqual([action.kind for action in result.evaluation.actions],
                         [ACTION_COMPUTE, ACTION_ADVANCE, ACTION_COMPUTE])
        self.assertEqual(result.action_sources, (0, 1, 2))
        self.assertEqual(result.evaluation.action_count, 2)
        raw = strategy_document()
        raw["program"].insert(0, {"kind": "wait", "operations": ["layer0"], "copies": []})
        result = run_strategy(compile_memory(raw))
        self.assertEqual(result.error_source_index, 0)
        self.assertEqual(result.evaluation.error_code, CODE_INPUT_NOT_READY)
        self.assertEqual(result.evaluation.actions, ())

    def test_device_state_is_explicit_and_cannot_switch_during_running_compute(self):
        workload, scene = documents()
        base = build_scenario(workload, scene)
        updated = deepcopy(scene)
        updated["costs"]["source"] = "synthetic_later_state"
        updated["costs"]["compute"]["layer1"]["gpu"]["duration_ns"] = 4_000_000
        profile = build_scenario(workload, updated)
        raw = strategy_document()
        raw["device_states"] = {"later": "later.scene.json"}
        raw["program"].insert(3, {"kind": "device_state", "name": "later"})
        compiled = compile_strategy(parse_strategy_ir(raw), base, device_states={"later": profile})
        self.assertIsInstance(compiled.steps[3], DeviceStateStep)
        result = run_strategy(compiled)
        self.assertEqual(result.evaluation.status, STATUS_VALID)
        self.assertEqual(result.evaluation.makespan_ns, 9_000_000)
        self.assertEqual(len(result.environment_events), 1)
        self.assertEqual(result.environment_events[0].t_ns, 5_000_000)
        self.assertEqual(result.final_state.copy_at("W:L1", "vram"), COPY_READY)
        # A profile transition cannot revise the cost of the already-launched layer0.
        raw = strategy_document()
        raw["device_states"] = {"later": "later.scene.json"}
        raw["program"].insert(1, {"kind": "device_state", "name": "later"})
        result = run_strategy(compile_strategy(parse_strategy_ir(raw), base, device_states={"later": profile}))
        self.assertEqual(result.evaluation.error_code, CODE_RESOURCE_BUSY)
        self.assertEqual(result.error_source_index, 1)
        self.assertEqual(result.environment_events, ())

    def test_profile_inventory_must_resolve_every_and_only_declared_name(self):
        ir = parse_strategy_ir(strategy_document())
        scenario = build_scenario(*documents())
        with self.assertRaisesRegex(InvalidInput, "every and only"):
            compile_strategy(ir, scenario, device_states={"extra": scenario})
        raw = strategy_document()
        raw["device_states"] = {"later": "later.json"}
        with self.assertRaisesRegex(InvalidInput, "every and only"):
            compile_strategy(parse_strategy_ir(raw), scenario)

    def test_file_compilation_resolves_local_paths_and_pins_all_direct_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            strategy_path = materialize(temporary)
            compiled = compile_strategy_file(strategy_path)
            self.assertEqual(run_strategy(compiled).evaluation.makespan_ns, 7_000_000)
            self.assertEqual({Path(source["path"]).name for source in compiled.sources},
                             {"strategy.json", "scene.json", "graph.json"})
            for source in compiled.sources:
                self.assertEqual(source["sha256"], hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest())
            # Input pinning is checked again before output is written.
            (Path(temporary) / "graph.json").write_text("{}\n", encoding="utf-8", newline="\n")
            with self.assertRaisesRegex(ValueError, "input changed"):
                write_run(Path(temporary) / "rejected", compiled, run_strategy(compiled))

    def test_cli_refuses_nonempty_output_before_compiling_and_preserves_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "occupied"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("preserve\n", encoding="utf-8", newline="\n")
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                code = main([str(Path(temporary) / "missing-strategy.json"), "--out", str(output)])
            self.assertEqual(code, 2)
            self.assertIn("new or empty directory", stderr.getvalue())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve\n")
            self.assertEqual(list(output.iterdir()), [sentinel])

    def test_cli_exports_lf_artifacts_and_rejected_action_original_ir_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            raw = strategy_document()
            raw["program"] = [
                {"kind": "compute", "operation_id": "layer0"},
                {"kind": "wait", "operations": ["layer0"], "copies": []},
                {"kind": "compute", "operation_id": "layer1"},
                {"kind": "drain"},
            ]
            path = materialize(directory, raw)
            output = directory / "invalid-output"
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = main([str(path), "--out", str(output)])
            self.assertEqual(code, 3)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], STATUS_INVALID_MAPPING)
            self.assertEqual(summary["error_code"], CODE_INPUT_NOT_READY)
            self.assertEqual(summary["error_action_index"], 2)
            self.assertEqual(summary["error_ir_instruction_index"], 2)
            self.assertFalse(summary["search_performed"])
            execution = json.loads((output / "execution.json").read_text(encoding="utf-8"))
            self.assertEqual(execution["error_ir_instruction_index"], 2)
            self.assertEqual([row["ir_instruction_index"] for row in execution["actions"]], [0, 1, 2])
            self.assertEqual(execution["actions"][-1]["action"]["operation_id"], "layer1")
            self.assertEqual(execution["actions"][-1]["index"], 2)
            for artifact in output.glob("*.json"):
                with self.subTest(artifact=artifact.name):
                    payload = artifact.read_bytes()
                    self.assertNotIn(b"\r", payload)
                    self.assertTrue(payload.endswith(b"\n"))

    def test_cli_valid_output_matches_actual_runtime_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = materialize(directory)
            output = directory / "valid-output"
            output.mkdir()  # Explicit empty directories are accepted.
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = main([str(path), "--out", str(output)])
            self.assertEqual(code, 0)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual((summary["status"], summary["makespan_ns"], summary["h2d_bytes"]),
                             (STATUS_VALID, 7_000_000, 64))
            self.assertFalse(summary["online_policy_claimed"])
            self.assertFalse(summary["physical_calibration_claimed"])
            self.assertFalse(summary["search_performed"])
            self.assertFalse((output / "state_out.json").exists())
            for artifact in output.glob("*.json"):
                self.assertNotIn(b"\r", artifact.read_bytes())

    def test_ir_changed_during_parse_refuses_compilation_and_creates_no_output(self):
        from tensor_mapping import strategy_compiler
        real_parse = strategy_compiler.parse_strategy_ir
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = materialize(directory)
            output = directory / "race-output"

            def parse_and_mutate(document, where="strategy"):
                parsed = real_parse(document, where=where)
                path.write_bytes(path.read_bytes() + b"\n")
                return parsed

            with patch.object(strategy_compiler, "parse_strategy_ir", side_effect=parse_and_mutate):
                with self.assertRaisesRegex(InvalidInput, "input changed during compilation.*strategy.json"):
                    compile_strategy_file(path)
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    code = main([str(path), "--out", str(output)])
            self.assertEqual(code, 2)
            self.assertFalse(output.exists())

    def test_m2_cost_file_changed_during_parse_refuses_and_creates_no_output(self):
        from llm_infer_model.m2 import costs
        from llm_infer_model.m2.expand import load_and_compile
        real_parse = costs.parse_cost_table
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ("fcd-prefill-3.json", "fcd.layer.json", "fcd.costs.json"):
                value = json.loads((_SIMULATOR / "mapping/examples/m2" / name).read_text(encoding="utf-8"))
                write_json(directory / name, value)
            invocation = load_and_compile(directory / "fcd-prefill-3.json")
            operations = tuple(invocation.scenario.workload.operation_by_id)
            raw = strategy_document()
            raw.update(scenario_file="fcd-prefill-3.json", regions=[], device_states={},
                       placements={operation: "cpu" for operation in operations},
                       program=[instruction for operation in operations for instruction in
                                ({"kind": "compute", "operation_id": operation},
                                 {"kind": "wait", "operations": [operation], "copies": []})])
            path = directory / "strategy.json"
            write_json(path, raw)
            # Establish that the fresh fixture is usable before inducing a race.
            self.assertEqual(run_strategy(compile_strategy_file(path)).evaluation.status, STATUS_VALID)
            cost_path = directory / "fcd.costs.json"
            output = directory / "race-output"

            def parse_and_mutate(document, **kwargs):
                table = real_parse(document, **kwargs)
                cost_path.write_bytes(cost_path.read_bytes() + b"\n")
                return table

            with patch.object(costs, "parse_cost_table", side_effect=parse_and_mutate):
                with self.assertRaisesRegex(InvalidInput, "input changed during compilation.*fcd.costs.json"):
                    compile_strategy_file(path)
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    code = main([str(path), "--out", str(output)])
            self.assertEqual(code, 2)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
