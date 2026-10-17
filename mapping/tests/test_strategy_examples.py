"""Fixed-plan evidence: multi-step ledger and the existing M2 state bridge."""
from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
import tempfile
import unittest

from llm_infer_model.m2.expand import compile_invocation, plan_for
from llm_infer_model.m2.runner import state_out_from_result
from llm_infer_model.m2.spec import (
    parse_invocation_document, parse_state_document, state_in_from_state_out,
)
from llm_infer_model.tensor.engine import evaluate_mapping
from llm_infer_model.tensor.spec import build_scenario, scenario_fingerprint

from tensor_mapping.strategy import write_run
from tensor_mapping.strategy_compiler import compile_strategy, compile_strategy_file, run_strategy
from tensor_mapping.strategy_ir import parse_strategy_ir, strategy_to_document

SIMULATOR = Path(__file__).resolve().parents[2]
EXAMPLES = SIMULATOR / "mapping/examples/strategy"


def declared_ir(binding, placements):
    """Render a supplied serial baseline; never enumerate/select placements."""
    program = []
    for action in plan_for(binding, placements=placements):
        if action.kind == "COMPUTE":
            row = dict(kind="compute", operation_id=action.target_id)
        elif action.kind == "ADVANCE":
            row = dict(kind="drain")
        elif action.kind == "COPY":
            row = dict(kind="transfer", tensor_id=action.target_id, src=action.src, dst=action.dst)
        elif action.kind == "READ_SSD":
            row = dict(kind="transfer", tensor_id=action.target_id, src="ssd", dst="dram")
        elif action.kind == "WRITE_SSD":
            row = dict(kind="transfer", tensor_id=action.target_id, src="dram", dst="ssd")
        else:
            raise AssertionError(action)
        program.append(row)
    return parse_strategy_ir(dict(
        format_version="strategy-ir-0.1", id="explicit-state-bridge", scenario_file="unused.json",
        information_mode="offline_fixed", regions=[], placements=placements,
        program=program, device_states={},
    ))


class ExampleTests(unittest.TestCase):
    def load(self, name):
        return compile_strategy_file(EXAMPLES / f"{name}.strategy.json")

    def test_hand_calculated_examples(self):
        expected = {
            "prefetch": (7_000_000, 128, 176),
            "ssd-prefetch": (7_000_000, 128, 176),
            "router-experts": (6_000_000, 128, 128),
            "stateful-three-step-static": (24_000_000, 256, 176),
            "stateful-three-step": (32_000_000, 256, 256),
            "m2-hybrid-prefill": (3100, 11756, 2432),
            "m2-moe-prefill": (34_000_000, 496, 0),
        }
        for name, metrics in expected.items():
            with self.subTest(name=name):
                result = run_strategy(self.load(name))
                value = result.evaluation
                self.assertEqual(value.status, "valid", value.reason)
                self.assertEqual((value.makespan_ns, value.peak_dram_bytes, value.peak_vram_bytes), metrics)
                self.assertEqual(len(result.action_sources), len(value.actions))
                self.assertEqual(len(result.action_profiles), len(value.actions))

    def test_environment_preserves_three_step_kv_and_time(self):
        compiled = self.load("stateful-three-step")
        result = run_strategy(compiled)
        self.assertEqual(result.environment_events[0].t_ns, 8_000_000)
        self.assertEqual(result.environment_events[0].used_vram_bytes, 144)
        final = result.final_state
        for step in range(3):
            for layer in range(2):
                self.assertEqual(final.copy_at(f"kv:L{layer}:s{step}:v{step+1}", "dram"), "READY")
                self.assertEqual(final.op(f"L{layer}:kv:s{step}"), "DONE")
        before = scenario_fingerprint(compiled.scenario)
        after = scenario_fingerprint(compiled.device_states["gpu_busy"])
        self.assertEqual(set(result.action_profiles), {before, after})
        gpu_hidden = [e for e in result.evaluation.events if (e.target_id or "").startswith("L1:hidden:") and e.start_ns is not None]
        self.assertEqual([e.end_ns - e.start_ns for e in gpu_hidden], [3_000_000, 6_000_000, 6_000_000])
        self.assertEqual(result.evaluation.h2d_bytes, 48)
        self.assertEqual(result.evaluation.d2h_bytes, 96)

    def test_capacity_shrink_and_next_compute_are_distinct_failures(self):
        original = self.load("stateful-three-step")
        raw = json.loads((EXAMPLES / "stateful-three-step.gpu-busy.json").read_text(encoding="utf-8"))
        graph = json.loads((EXAMPLES / "stateful-three-step.graph.json").read_text(encoding="utf-8"))
        for capacity, source_index, successful_updates in ((143, 12, 0), (255, 32, 1)):
            with self.subTest(capacity=capacity):
                candidate = copy.deepcopy(raw)
                del candidate["workload_file"]
                candidate["architecture"]["vram_capacity_bytes"] = capacity
                profile = build_scenario(graph, candidate)
                compiled = compile_strategy(original.ir, original.scenario, device_states={"gpu_busy": profile})
                result = run_strategy(compiled)
                self.assertEqual(result.evaluation.status, "invalid_mapping")
                self.assertEqual(result.evaluation.error_code, "CAPACITY_EXCEEDED")
                self.assertEqual(result.error_source_index, source_index)
                self.assertEqual(len(result.environment_events), successful_updates)

    def test_busy_profile_change_does_not_auto_wait(self):
        original = self.load("stateful-three-step")
        document = strategy_to_document(original.ir)
        change = document["program"].pop(12)
        document["program"].insert(1, change)
        compiled = compile_strategy(parse_strategy_ir(document), original.scenario,
                                    device_states=original.device_states)
        result = run_strategy(compiled)
        self.assertEqual(result.evaluation.error_code, "RESOURCE_BUSY")
        self.assertEqual(result.error_source_index, 1)
        self.assertEqual(result.final_state.t, 0)
        self.assertEqual(len(result.environment_events), 0)

    def test_failed_wait_artifact_keeps_ir_error_index(self):
        original = self.load("prefetch")
        document = strategy_to_document(original.ir)
        document["program"].insert(0, dict(kind="wait", operations=["layer0"], copies=[]))
        compiled = compile_strategy(parse_strategy_ir(document), original.scenario)
        result = run_strategy(compiled)
        with tempfile.TemporaryDirectory() as folder:
            write_run(Path(folder), compiled, result)
            summary = json.loads((Path(folder) / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["error_ir_instruction_index"], 0)
            self.assertEqual(summary["error_code"], "INPUT_NOT_READY")
            self.assertTrue(summary["error_message"])
            self.assertFalse((Path(folder) / "state_out.json").exists())


class M2StateBridgeTests(unittest.TestCase):
    def test_dense_prefill_state_drives_decode_without_changing_contract(self):
        compiled = compile_strategy_file(EXAMPLES / "m2-hybrid-prefill.strategy.json")
        result = run_strategy(compiled)
        # Equality with the independently supplied original fixed baseline.
        baseline = evaluate_mapping(compiled.scenario, plan_for(compiled.m2_binding, placements=compiled.ir.placements))
        self.assertEqual(result.evaluation, baseline)
        state = state_out_from_result(compiled.m2_binding, result.evaluation)
        self.assertEqual(state.requests[0].next_position, 3)
        self.assertEqual(state.requests[0].version, 2)
        raw = json.loads((SIMULATOR / "mapping/examples/m2/hybrid-interface-synthetic-decode-1.json").read_text(encoding="utf-8"))
        invocation = dataclasses.replace(parse_invocation_document(raw), state_in=state_in_from_state_out(state))
        binding = compile_invocation(compiled.m2_binding.layer, invocation, compiled.m2_binding.cost_table)
        placements = {task.operation_id: device for task, device in zip(binding.tasks, ("cpu", "gpu", "cpu"))}
        decode = compile_strategy(declared_ir(binding, placements), binding.scenario, m2_binding=binding)
        output = run_strategy(decode)
        self.assertEqual(output.evaluation.status, "valid", output.evaluation.reason)
        state_out = state_out_from_result(binding, output.evaluation)
        self.assertEqual(state_out.requests[0].next_position, 4)
        self.assertEqual(state_out.requests[0].version, 3)
        self.assertEqual(len(state_out.requests[0].blocks), 4)
        self.assertEqual(state_out.requests[0].blocks[:3], state.requests[0].blocks)

    def test_moe_export_uses_ready_expert_copies_and_real_kv_for_decode(self):
        compiled = compile_strategy_file(EXAMPLES / "m2-moe-prefill.strategy.json")
        result = run_strategy(compiled)
        with tempfile.TemporaryDirectory() as folder:
            write_run(Path(folder), compiled, result)
            raw_state = json.loads((Path(folder) / "state_out.json").read_text(encoding="utf-8"))
            parsed = parse_state_document(raw_state)
            self.assertEqual(parsed.requests[0].next_position, 2)
            self.assertEqual(parsed.requests[0].version, 2)
            expert = next(w for w in parsed.weight_copies if w.id == "E0")
            self.assertEqual(expert.locations, ("dram", "ssd"))
            self.assertEqual(expert.version, 1)
        state = state_out_from_result(compiled.m2_binding, result.evaluation)
        raw = json.loads((SIMULATOR / "mapping/examples/m2/moe-interface-synthetic-decode-1.json").read_text(encoding="utf-8"))
        invocation = dataclasses.replace(parse_invocation_document(raw), state_in=state_in_from_state_out(state))
        binding = compile_invocation(compiled.m2_binding.layer, invocation, compiled.m2_binding.cost_table)
        placements = {task.operation_id: "cpu" for task in binding.tasks}
        decode = compile_strategy(declared_ir(binding, placements), binding.scenario, m2_binding=binding)
        output = run_strategy(decode)
        self.assertEqual(output.evaluation.status, "valid", output.evaluation.reason)
        self.assertEqual(output.evaluation.ssd_read_bytes, 0)
        final = state_out_from_result(binding, output.evaluation)
        self.assertEqual(final.requests[0].next_position, 3)
        self.assertEqual(final.requests[0].version, 3)
        self.assertEqual(len(final.requests[0].blocks), 3)


if __name__ == "__main__":
    unittest.main()
