"""Independent event/ledger acceptance checks, using synthetic costs only.

The timeline and byte expectations are hand calculations, not predictions from
a mapper. This suite does not execute hardware kernels or claim calibrated time.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

from llm_infer_model.tensor.engine import (
    ACTION_ADVANCE, ACTION_COMPUTE, ACTION_COPY, ACTION_EVICT, ACTION_READ_SSD,
    ACTION_WRITE_SSD, CODE_CAPACITY_EXCEEDED, CODE_ILLEGAL_ACTION,
    CODE_INPUT_NOT_READY, CODE_TENSOR_IN_USE, Action, MappingError,
    STATUS_INVALID_MAPPING, STATUS_VALID, evaluate_mapping, initial_state,
    transition, used_bytes,
)
from llm_infer_model.tensor.spec import (
    COPY_READY, COPY_RESERVED_COPY, COPY_RESERVED_OUTPUT, DTYPE_F32, ComputeCost, H2DCost,
    InvalidInput, TransferCost, build_scenario,
)
from llm_infer_model.m2.expand import compile_invocation, load_and_compile
from llm_infer_model.m2.runner import run_invocation
from llm_infer_model.m2.spec import state_in_from_state_out


SIMULATOR = Path(__file__).resolve().parents[2]
EXAMPLES = SIMULATOR / "mapping" / "examples" / "m2"


def tensor(name, nbytes, locations=(), role="intermediate"):
    result = dict(id=name, name=name, role=role, dtype=DTYPE_F32,
                  ne=[nbytes // 4, 1, 1, 1], nb=[4, nbytes, nbytes, nbytes],
                  storage_bytes=nbytes)
    if locations:
        result["initial_locations"] = list(locations)
    return result


def four_resource_documents():
    """CPU 0..5, GPU 0..7, SSD/PCIe 0..3, then CPU 5..7 and GPU 7..9.

    DRAM: reserve 8 + X padded 8 + V padded 16 + W 16 + C 8 + KV
    padded 8 + Y 8 + running CPU scratch padded 8 = peak 80.
    VRAM: reserve 4 + G 8 + V 12 + A 8 + H 8 = peak 40.
    """
    def operation(name, semantic, inputs, output, extra=()):
        value = dict(id=name, semantic_op=semantic, ggml_op="M2_" + semantic.upper(),
                     unary_op=None, op_params=[], inputs=list(inputs), output=output)
        if extra:
            value["extra_outputs"] = list(extra)
        return value
    workload = dict(schema_version="0.2", id="readiness-four-resource",
        origin=dict(kind="synthetic_layer"),
        tensors=[tensor("X", 4, ("dram",), "input"),
                 tensor("V", 12, ("dram",), "input"),
                 tensor("G", 8, ("vram",), "input"),
                 tensor("W", 16, ("ssd",), "weight"),
                 tensor("C", 8, role="output"), tensor("KV", 4, role="output"),
                 tensor("A", 8), tensor("Y", 8, role="output"),
                 tensor("H", 8, role="output")],
        operations=[operation("cpu", "attention", ("X",), "C", ("KV",)),
                    operation("gpu", "projection", ("G",), "A"),
                    operation("cpu_tail", "dense_ffn", ("W",), "Y"),
                    operation("gpu_tail", "dense_ffn", ("A", "V"), "H")],
        outputs=["C", "KV", "Y", "H"])
    scenario = dict(schema_version="0.2", id="readiness-four-resource",
        architecture=dict(dram_capacity_bytes=80, vram_capacity_bytes=40,
            dram_runtime_reserved_bytes=8, vram_runtime_reserved_bytes=4,
            dram_alignment_bytes=8, vram_alignment_bytes=4,
            cpu_compute_slots=1, gpu_compute_slots=1, pcie_copy_slots=1, ssd_io_slots=1),
        goal=dict(output_locations={"C": "dram", "KV": "dram", "Y": "dram", "H": "vram"}),
        costs=dict(source="synthetic_hand_calculated_readiness",
            compute={"cpu": {"cpu": dict(duration_ns=5, workspace_bytes=7)},
                     "gpu": {"gpu": dict(duration_ns=7, workspace_bytes=3)},
                     "cpu_tail": {"cpu": dict(duration_ns=2, workspace_bytes=4)},
                     "gpu_tail": {"gpu": dict(duration_ns=2, workspace_bytes=0)}},
            transfer={"dram_to_vram": {"V": dict(duration_ns=3)},
                      "vram_to_dram": {}, "ssd_to_dram": {"W": dict(duration_ns=3)},
                      "dram_to_ssd": {}}),
        mapspace=dict(allowed_devices={"cpu": ["cpu"], "gpu": ["gpu"],
            "cpu_tail": ["cpu"], "gpu_tail": ["gpu"]},
            allowed_transfers={"dram_to_vram": ["V"], "vram_to_dram": [],
                               "ssd_to_dram": ["W"], "dram_to_ssd": []},
            allow_eviction=True, allow_copy_compute_overlap=True,
            allow_cpu_gpu_overlap=True, allow_recomputation=False),
        mapper=dict(algorithm="uniform_cost", max_expanded_states=1, wall_time_limit_s=0))
    return workload, scenario


def four_resource_actions():
    return (Action(ACTION_COMPUTE, "cpu", device="cpu"),
            Action(ACTION_COMPUTE, "gpu", device="gpu"),
            Action(ACTION_READ_SSD, "W"),
            Action(ACTION_COPY, "V", src="dram", dst="vram"),
            Action(ACTION_ADVANCE), Action(ACTION_ADVANCE),
            Action(ACTION_COMPUTE, "cpu_tail", device="cpu"),
            Action(ACTION_ADVANCE),
            Action(ACTION_COMPUTE, "gpu_tail", device="gpu"),
            Action(ACTION_ADVANCE))


class EventAndMemoryAudit(unittest.TestCase):
    def test_four_resources_follow_hand_timeline_and_exact_peaks(self):
        scenario = build_scenario(*four_resource_documents())
        result = evaluate_mapping(scenario, four_resource_actions())
        self.assertEqual(result.status, STATUS_VALID)
        self.assertEqual(result.makespan_ns, 9)
        self.assertEqual((result.peak_dram_bytes, result.peak_vram_bytes), (80, 40))
        intervals = [(e.kind, e.target_id, e.start_ns, e.end_ns)
                     for e in result.events if e.start_ns is not None]
        self.assertEqual(intervals, [("COMPUTE", "cpu", 0, 5), ("COMPUTE", "gpu", 0, 7),
            ("READ_SSD", "W", 0, 3), ("COPY", "V", 0, 3),
            ("COMPUTE", "cpu_tail", 5, 7), ("COMPUTE", "gpu_tail", 7, 9)])
        self.assertEqual([e.t_ns for e in result.events if e.kind == ACTION_ADVANCE], [3, 5, 7, 9])
        self.assertEqual((result.ssd_read_bytes, result.h2d_bytes), (16, 12))
        # Transferred payloads exclude padding; destination allocations include it.
        self.assertEqual(result.states[4].used_dram_bytes, 72)
        self.assertEqual(result.states[4].used_vram_bytes, 36)

    def test_one_byte_short_refuses_both_inflight_and_later_peak(self):
        for domain, capacity, expected_index, expected_target in (
            ("dram", 71, 2, "W"), ("vram", 35, 3, "V"),
            ("dram", 79, 6, "cpu_tail"), ("vram", 39, 8, "gpu_tail"),
        ):
            with self.subTest(domain=domain, capacity=capacity):
                workload, raw = four_resource_documents()
                raw["architecture"][domain + "_capacity_bytes"] = capacity
                result = evaluate_mapping(build_scenario(workload, raw), four_resource_actions())
                self.assertEqual(result.status, STATUS_INVALID_MAPPING)
                self.assertEqual(result.error_code, CODE_CAPACITY_EXCEEDED)
                self.assertEqual(result.error_action_index, expected_index)
                self.assertIsNone(result.makespan_ns)
                self.assertEqual(four_resource_actions()[expected_index].target_id, expected_target)

    def test_read_locks_cover_sources_but_not_other_ready_copies(self):
        scenario = build_scenario(*four_resource_documents())
        state = initial_state(scenario)
        for action in four_resource_actions()[:4]:
            state = transition(scenario, state, action)
        for name, domain, code in (("X", "dram", CODE_TENSOR_IN_USE),
                                   ("V", "dram", CODE_TENSOR_IN_USE),
                                   ("V", "vram", CODE_ILLEGAL_ACTION)):
            with self.subTest(tensor=name, domain=domain), self.assertRaises(MappingError) as caught:
                transition(scenario, state, Action(ACTION_EVICT, name, location=domain))
            self.assertEqual(caught.exception.code, code)
        at_three = transition(scenario, state, Action(ACTION_ADVANCE))
        self.assertEqual(at_three.copy_at("W", "dram"), COPY_READY)
        self.assertEqual(at_three.copy_at("V", "vram"), COPY_READY)
        released = transition(scenario, at_three, Action(ACTION_EVICT, "V", location="dram"))
        self.assertEqual(used_bytes(scenario, released, "dram"), 56)
        # The PCIe lock ended at 3; the independent CPU input remains locked until 5.
        with self.assertRaises(MappingError) as caught:
            transition(scenario, released, Action(ACTION_EVICT, "X", location="dram"))
        self.assertEqual(caught.exception.code, CODE_TENSOR_IN_USE)

    def test_extra_output_cannot_be_read_or_copied_before_producer_finishes(self):
        workload, raw = four_resource_documents()
        workload["operations"][-1]["inputs"] = ["KV"]
        workload["outputs"].append("A")
        raw["goal"]["output_locations"]["A"] = "vram"
        raw["mapspace"]["allowed_transfers"]["dram_to_vram"].append("KV")
        raw["costs"]["transfer"]["dram_to_vram"]["KV"] = dict(duration_ns=3)
        scenario = build_scenario(workload, raw)
        state = transition(scenario, initial_state(scenario), four_resource_actions()[0])
        self.assertEqual(state.copy_at("KV", "dram"), COPY_RESERVED_OUTPUT)
        with self.assertRaises(MappingError) as caught:
            transition(scenario, state, Action(ACTION_COMPUTE, "gpu_tail", device="gpu"))
        self.assertEqual(caught.exception.code, CODE_INPUT_NOT_READY)
        with self.assertRaises(MappingError) as caught:
            transition(scenario, state, Action(ACTION_COPY, "KV", src="dram", dst="vram"))
        self.assertEqual(caught.exception.code, CODE_INPUT_NOT_READY)
        done = transition(scenario, state, Action(ACTION_ADVANCE))
        self.assertEqual(done.copy_at("KV", "dram"), COPY_READY)
        self.assertEqual(done.copy_at("C", "dram"), COPY_READY)
        self.assertEqual(used_bytes(scenario, done, "dram"), 48)
        copied = transition(scenario, done, Action(ACTION_COPY, "KV", src="dram", dst="vram"))
        self.assertEqual(copied.copy_at("KV", "vram"), COPY_RESERVED_COPY)

    def test_ssd_snapshot_requires_finished_write_and_locks_its_source(self):
        workload, raw = four_resource_documents()
        raw["goal"]["output_locations"]["KV"] = "ssd"
        raw["mapspace"]["allowed_transfers"]["dram_to_ssd"] = ["KV"]
        raw["mapspace"]["allowed_transfers"]["ssd_to_dram"].append("KV")
        raw["costs"]["transfer"]["dram_to_ssd"]["KV"] = dict(duration_ns=2)
        raw["costs"]["transfer"]["ssd_to_dram"]["KV"] = dict(duration_ns=2)
        scenario = build_scenario(workload, raw)
        state = initial_state(scenario)
        for action in four_resource_actions()[:6]:
            state = transition(scenario, state, action)
        before = used_bytes(scenario, state, "dram")
        writing = transition(scenario, state, Action(ACTION_WRITE_SSD, "KV"))
        self.assertEqual(used_bytes(scenario, writing, "dram"), before)
        self.assertEqual(writing.copy_at("KV", "ssd"), COPY_RESERVED_COPY)
        for action, code in ((Action(ACTION_EVICT, "KV", location="dram"), CODE_TENSOR_IN_USE),
                             (Action(ACTION_READ_SSD, "KV"), CODE_INPUT_NOT_READY)):
            with self.assertRaises(MappingError) as caught:
                transition(scenario, writing, action)
            self.assertEqual(caught.exception.code, code)
        done = transition(scenario, writing, Action(ACTION_ADVANCE))
        self.assertEqual(done.copy_at("KV", "ssd"), COPY_READY)
        dropped = transition(scenario, done, Action(ACTION_EVICT, "KV", location="dram"))
        restored = transition(scenario, dropped, Action(ACTION_READ_SSD, "KV"))
        self.assertEqual(restored.copy_at("KV", "dram"), COPY_RESERVED_COPY)


class DirectApiGuardAudit(unittest.TestCase):
    def test_all_duration_dataclasses_require_positive_integer_nanoseconds(self):
        for cost_type in (ComputeCost, TransferCost, H2DCost):
            for invalid in (True, False, 0, -1, 1.5, None):
                with self.subTest(cost=cost_type.__name__, value=invalid), self.assertRaises(InvalidInput):
                    cost_type(duration_ns=invalid)
            self.assertEqual(cost_type(duration_ns=1).duration_ns, 1)

    def test_compute_workspace_requires_explicit_nonnegative_integer(self):
        for invalid in (True, False, -1, 0.5, None):
            with self.subTest(value=invalid), self.assertRaises(InvalidInput):
                ComputeCost(duration_ns=1, workspace_bytes=invalid)
        self.assertEqual(ComputeCost(duration_ns=1, workspace_bytes=0).workspace_bytes, 0)
        self.assertEqual(ComputeCost(duration_ns=1, workspace_bytes=8).workspace_bytes, 8)

    def test_duplicate_dense_request_or_state_versions_are_refused(self):
        compiled = load_and_compile(EXAMPLES / "fcd-prefill-3.json")
        state = compiled.state_in.requests[0]
        history = replace(compiled.state_in, requests=(state, replace(state, version=99)))
        with self.assertRaises(InvalidInput):
            compile_invocation(compiled.layer, compiled.invocation, compiled.cost_table, state_in=history)
        repeated = replace(compiled.invocation, requests=compiled.invocation.requests * 2)
        with self.assertRaises(InvalidInput):
            compile_invocation(compiled.layer, repeated, compiled.cost_table)

    def test_dense_and_moe_request_versions_must_be_positive_integers(self):
        for filename in ("fcd-prefill-3.json", "moe-interface-synthetic-prefill-2.json"):
            compiled = load_and_compile(EXAMPLES / filename)
            for invalid in (-1, 0, True, 1.5, None):
                state = replace(compiled.state_in.requests[0], version=invalid)
                history = replace(compiled.state_in, requests=(state,))
                with self.subTest(layer=filename, version=invalid), self.assertRaises(InvalidInput):
                    compile_invocation(compiled.layer, compiled.invocation, compiled.cost_table, state_in=history)


class PersistentM2StateAudit(unittest.TestCase):
    def test_dense_actual_kv_chain_preserves_identity_and_increments_version_once(self):
        prefill = load_and_compile(EXAMPLES / "fcd-prefill-3.json")
        state = replace(prefill.state_in.requests[0], version=7)
        prefill = compile_invocation(prefill.layer, prefill.invocation, prefill.cost_table,
                                     state_in=replace(prefill.state_in, requests=(state,)))
        first = run_invocation(prefill)
        self.assertEqual(first.status, STATUS_VALID)
        self.assertEqual((first.makespan_ns, first.evaluation.peak_dram_bytes), (10_000_000, 368))
        actual = state_in_from_state_out(first.state_out)
        self.assertEqual(actual.requests[0].version, 8)
        decode = load_and_compile(EXAMPLES / "fcd-decode-1.json")
        invocation = replace(decode.invocation, hardware=replace(decode.invocation.hardware,
                             dram_capacity_bytes=256))
        rebound = compile_invocation(decode.layer, invocation, decode.cost_table, state_in=actual)
        attention = rebound.scenario.workload.operation_by_id[rebound.attention_task("r0").operation_id]
        for old in actual.requests[0].blocks:
            self.assertIn(old.id, attention.inputs)
            self.assertNotIn(old.id, rebound.scenario.workload.producer_of)
        second = run_invocation(rebound)
        self.assertEqual(second.status, STATUS_VALID)
        self.assertEqual((second.makespan_ns, second.evaluation.peak_dram_bytes), (10_000_000, 256))
        out = second.state_out.requests[0]
        self.assertEqual((out.next_position, out.version), (4, 9))
        self.assertEqual(out.blocks[:3], actual.requests[0].blocks)
        self.assertEqual(sum(block.storage_bytes for block in out.blocks), 64)
        too_small = replace(invocation, hardware=replace(invocation.hardware, dram_capacity_bytes=255))
        refused = run_invocation(compile_invocation(decode.layer, too_small, decode.cost_table, state_in=actual))
        self.assertEqual(refused.status, STATUS_INVALID_MAPPING)
        self.assertEqual(refused.evaluation.error_code, CODE_CAPACITY_EXCEEDED)
        self.assertIsNone(refused.state_out)

    def test_moe_actual_cache_and_kv_chain_preserves_versions_and_inactive_source(self):
        prefill = load_and_compile(EXAMPLES / "moe-interface-synthetic-prefill-2.json")
        copies = tuple(replace(weight, version=17) for weight in prefill.state_in.weight_copies)
        prefill = compile_invocation(prefill.layer, prefill.invocation, prefill.cost_table,
                                     state_in=replace(prefill.state_in, weight_copies=copies))
        first = run_invocation(prefill)
        self.assertEqual(first.status, STATUS_VALID)
        self.assertEqual((first.makespan_ns, first.evaluation.peak_dram_bytes), (34_000_000, 496))
        self.assertEqual(first.evaluation.ssd_read_bytes, 3 * 64)
        self.assertEqual([len(prefill.expert_work_items[str(i)]) for i in range(3)], [1, 2, 1])
        actual = state_in_from_state_out(first.state_out)
        by_weight = {weight.id: weight for weight in actual.weight_copies}
        self.assertEqual(by_weight["E3"].locations, ("ssd",))
        for i in range(3):
            self.assertEqual(by_weight[f"E{i}"].locations, ("dram", "ssd"))
        self.assertEqual({weight.version for weight in actual.weight_copies}, {17})
        decode = load_and_compile(EXAMPLES / "moe-interface-synthetic-decode-1.json")
        second = run_invocation(compile_invocation(decode.layer, decode.invocation,
                                decode.cost_table, state_in=actual))
        self.assertEqual(second.status, STATUS_VALID)
        self.assertEqual((second.makespan_ns, second.evaluation.peak_dram_bytes), (15_000_000, 408))
        self.assertEqual(second.evaluation.ssd_read_bytes, 0)
        self.assertEqual(second.state_out.requests[0].blocks[:2], actual.requests[0].blocks)
        self.assertEqual((second.state_out.requests[0].next_position,
                          second.state_out.requests[0].version), (3, 3))
        self.assertEqual({weight.version for weight in second.state_out.weight_copies}, {17})


if __name__ == "__main__":
    unittest.main()
