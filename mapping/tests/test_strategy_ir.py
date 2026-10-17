"""Structural contract checks for declarative fixed-plan Strategy IR."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from llm_infer_model.tensor.spec import InvalidInput, Unsupported
from tensor_mapping.strategy_ir import (
    CopyRef, Instruction, Region, StrategyIR, load_strategy_ir,
    parse_strategy_ir, strategy_to_document,
)


def document():
    return {
        "format_version": "strategy-ir-0.1", "id": "fixed-two-regions",
        "scenario_file": "../inputs/tiny.scenario.json", "information_mode": "offline_fixed",
        "regions": [{"id": "first", "kind": "layer", "members": ["alpha"]},
                    {"id": "second", "kind": "layer", "members": ["beta", "omega"]}],
        "placements": {"alpha": "cpu", "beta": "gpu", "omega": "cpu"},
        "program": [
            {"kind": "transfer", "tensor_id": "a", "src": "ssd", "dst": "dram"},
            {"kind": "wait", "operations": [], "copies": [{"tensor_id": "a", "location": "dram"}]},
            {"kind": "compute", "operation_id": "alpha"},
            {"kind": "transfer", "tensor_id": "b", "src": "ssd", "dst": "vram"},
            {"kind": "wait", "operations": ["alpha"], "copies": [{"tensor_id": "b", "location": "vram"}]},
            {"kind": "compute", "operation_id": "beta"},
            {"kind": "drain"},
            {"kind": "release", "tensor_id": "a", "location": "dram"},
            {"kind": "device_state", "name": "limited"},
            {"kind": "compute", "operation_id": "omega"},
            {"kind": "transfer", "tensor_id": "b", "src": "vram", "dst": "ssd"},
            {"kind": "drain"},
        ],
        "device_states": {"limited": "tiny.limited.scenario.json"},
    }


class StrategyIRTests(unittest.TestCase):
    def test_roundtrip_preserves_every_explicit_field_and_instruction(self):
        raw = document()
        ir = parse_strategy_ir(raw)
        self.assertEqual(strategy_to_document(ir), raw)
        self.assertEqual(parse_strategy_ir(strategy_to_document(ir)), ir)
        self.assertIsInstance(ir.regions[0], Region)
        self.assertIsInstance(ir.program[1].copies[0], CopyRef)
        self.assertEqual(ir.program[3].src, "ssd")
        self.assertEqual(ir.program[3].dst, "vram")
        self.assertEqual(ir.program[-2].src, "vram")
        self.assertEqual(ir.program[-2].dst, "ssd")

    def test_references_are_preserved_without_loading_scenario(self):
        raw = document()
        raw["scenario_file"] = "not-created/scenario.json"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "strategy.json"
            path.write_text(json.dumps(raw), encoding="utf-8", newline="\n")
            ir = load_strategy_ir(path)
        self.assertEqual(ir.scenario_file, raw["scenario_file"])
        self.assertEqual(ir.device_states, raw["device_states"])

    def test_every_top_level_field_is_required_and_no_unknown_fields(self):
        for field in document():
            with self.subTest(missing=field):
                raw = document()
                del raw[field]
                with self.assertRaisesRegex(InvalidInput, "missing fields"):
                    parse_strategy_ir(raw)
        raw = document()
        raw["automatic_prefetch"] = True
        with self.assertRaisesRegex(InvalidInput, "unknown fields"):
            parse_strategy_ir(raw)

    def test_arrays_are_explicit_and_may_not_be_omitted_or_coerced(self):
        for field, wrong in (("regions", {}), ("regions", ()), ("program", None), ("program", ())):
            with self.subTest(field=field, wrong=wrong):
                raw = document()
                raw[field] = wrong
                with self.assertRaises(InvalidInput):
                    parse_strategy_ir(raw)
        raw = document()
        raw.update(regions=[], placements={}, program=[], device_states={})
        self.assertEqual(strategy_to_document(parse_strategy_ir(raw)), raw)

    def test_flat_regions_have_unique_ids_and_disjoint_nonempty_members(self):
        cases = [
            [{"id": "x", "kind": "layer", "members": []}],
            [{"id": "x", "kind": "layer", "members": ["alpha", "alpha"]}],
            [{"id": "x", "kind": "layer", "members": ["alpha"]},
             {"id": "x", "kind": "layer", "members": ["beta"]}],
            [{"id": "x", "kind": "layer", "members": ["alpha"]},
             {"id": "y", "kind": "layer", "members": ["alpha", "beta"]}],
            [{"id": "x", "kind": "layer", "members": [{"region": "nested"}]}],
            [{"id": "x", "kind": "layer", "members": ["alpha"], "barrier": True}],
        ]
        for regions in cases:
            with self.subTest(regions=regions):
                raw = document()
                raw["regions"] = regions
                with self.assertRaises(InvalidInput):
                    parse_strategy_ir(raw)
        raw = document()
        raw["regions"][0]["kind"] = "annotation-only"
        self.assertEqual(parse_strategy_ir(raw).regions[0].kind, "annotation-only")

    def test_placements_require_explicit_cpu_or_gpu_strings(self):
        for wrong in (True, 1, None, "CPU", "cuda0", "dram", "", float("nan")):
            with self.subTest(wrong=wrong):
                raw = document()
                raw["placements"]["alpha"] = wrong
                with self.assertRaises(InvalidInput):
                    parse_strategy_ir(raw)
        for wrong in ([], True, None, {True: "cpu"}, {" ": "cpu"}):
            with self.subTest(placements=wrong):
                raw = document()
                raw["placements"] = wrong
                with self.assertRaises(InvalidInput):
                    parse_strategy_ir(raw)

    def test_instruction_schemas_are_exact_for_every_supported_kind(self):
        examples = [document()["program"][index] for index in (0, 1, 2, 6, 7, 8)]
        for example in examples:
            for field in example:
                with self.subTest(kind=example["kind"], missing=field):
                    raw = document()
                    instruction = deepcopy(example)
                    del instruction[field]
                    raw["program"] = [instruction]
                    with self.assertRaises(InvalidInput):
                        parse_strategy_ir(raw)
            raw = document()
            raw["program"] = [dict(example, optional_hint="ignored?")]
            with self.subTest(kind=example["kind"], extra=True):
                with self.assertRaisesRegex(InvalidInput, "unknown fields"):
                    parse_strategy_ir(raw)
        raw = document()
        raw["program"] = [{"kind": "drain", "operations": []}]
        with self.assertRaises(InvalidInput):
            parse_strategy_ir(raw)

    def test_transfers_allow_all_distinct_domains_without_implying_direct_link(self):
        for src in ("dram", "vram", "ssd"):
            for dst in ("dram", "vram", "ssd"):
                raw = document()
                raw["program"] = [{"kind": "transfer", "tensor_id": "t", "src": src, "dst": dst}]
                with self.subTest(src=src, dst=dst):
                    if src == dst:
                        with self.assertRaisesRegex(InvalidInput, "must differ"):
                            parse_strategy_ir(raw)
                    else:
                        self.assertEqual(strategy_to_document(parse_strategy_ir(raw)), raw)
        for domain in ("CPU", "gpu", True, None):
            raw = document()
            raw["program"][0]["src"] = domain
            with self.subTest(domain=domain):
                with self.assertRaises(InvalidInput):
                    parse_strategy_ir(raw)

    def test_release_is_limited_to_dram_and_vram(self):
        for location in ("dram", "vram"):
            raw = document()
            raw["program"] = [{"kind": "release", "tensor_id": "t", "location": location}]
            self.assertEqual(parse_strategy_ir(raw).program[0].location, location)
        raw = document()
        raw["program"] = [{"kind": "release", "tensor_id": "t", "location": "ssd"}]
        with self.assertRaises(InvalidInput):
            parse_strategy_ir(raw)

    def test_wait_requires_unique_operation_or_copy_dependencies(self):
        invalid = [
            {"kind": "wait", "operations": [], "copies": []},
            {"kind": "wait", "operations": ["alpha", "alpha"], "copies": []},
            {"kind": "wait", "operations": [], "copies": [
                {"tensor_id": "t", "location": "dram"}, {"tensor_id": "t", "location": "dram"}]},
            {"kind": "wait", "operations": [], "copies": [{"tensor_id": "t"}]},
            {"kind": "wait", "operations": [], "copies": [{"tensor_id": "t", "location": "dram", "version": 1}]},
            {"kind": "wait", "operations": True, "copies": []},
            {"kind": "wait", "operations": [False], "copies": []},
        ]
        for instruction in invalid:
            raw = document()
            raw["program"] = [instruction]
            with self.subTest(instruction=instruction):
                with self.assertRaises(InvalidInput):
                    parse_strategy_ir(raw)
        raw = document()
        raw["program"] = [{"kind": "wait", "operations": [], "copies": [
            {"tensor_id": "t", "location": "dram"}, {"tensor_id": "t", "location": "ssd"}]}]
        self.assertEqual(len(parse_strategy_ir(raw).program[0].copies), 2)

    def test_device_state_names_must_reference_explicit_file_bindings(self):
        raw = document()
        raw["program"] = [{"kind": "device_state", "name": "missing"}]
        with self.assertRaisesRegex(InvalidInput, "undeclared device state"):
            parse_strategy_ir(raw)
        for states in ({"limited": ""}, {"limited": None}, {"": "x.json"}, []):
            raw = document()
            raw["device_states"] = states
            with self.subTest(states=states):
                with self.assertRaises(InvalidInput):
                    parse_strategy_ir(raw)

    def test_unsupported_version_information_contract_and_instruction_are_explicit(self):
        for field, value in (("format_version", "strategy-ir-2.0"),
                             ("information_mode", "online_causal"),
                             ("information_mode", "offline_oracle")):
            raw = document()
            raw[field] = value
            with self.subTest(field=field, value=value):
                with self.assertRaises(Unsupported):
                    parse_strategy_ir(raw)
        for kind in ("predict", "slice", "nested", "choose_device"):
            raw = document()
            raw["program"] = [{"kind": kind}]
            with self.subTest(kind=kind):
                with self.assertRaises(Unsupported):
                    parse_strategy_ir(raw)

    def test_non_string_identifiers_bool_and_nonfinite_values_are_never_coerced(self):
        for wrong in (False, 0, 1.0, float("inf"), float("nan"), None, " "):
            for target in ("id", "scenario_file", "information_mode"):
                raw = document()
                raw[target] = wrong
                with self.subTest(target=target, wrong=wrong):
                    with self.assertRaises(InvalidInput):
                        parse_strategy_ir(raw)
            raw = document()
            raw["program"] = [{"kind": "compute", "operation_id": wrong}]
            with self.subTest(operation_id=wrong):
                with self.assertRaises(InvalidInput):
                    parse_strategy_ir(raw)

    def test_loader_rejects_duplicate_keys_at_any_depth_and_nonfinite_json(self):
        raw = json.dumps(document())
        invalid = [
            raw.replace('"id": "fixed-two-regions"', '"id": "x", "id": "fixed-two-regions"', 1),
            raw.replace('"alpha": "cpu"', '"alpha": "cpu", "alpha": "gpu"', 1),
            raw.replace('"kind": "compute"', '"kind": "compute", "kind": "drain"', 1),
            raw.replace('"id": "fixed-two-regions"', '"id": NaN', 1),
            raw.replace('"id": "fixed-two-regions"', '"id": Infinity', 1),
            raw.replace('"id": "fixed-two-regions"', '"id": -Infinity', 1),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.json"
            for text in invalid:
                with self.subTest(text=text[:80]):
                    path.write_text(text, encoding="utf-8", newline="\n")
                    with self.assertRaises(InvalidInput):
                        load_strategy_ir(path)

    def test_loader_reports_file_encoding_and_json_errors_as_invalid_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.json"
            with self.assertRaises(InvalidInput):
                load_strategy_ir(path)
            path.write_bytes(b"\xff")
            with self.assertRaises(InvalidInput):
                load_strategy_ir(path)
            path.write_text('{"id":', encoding="utf-8", newline="\n")
            with self.assertRaises(InvalidInput):
                load_strategy_ir(path)

    def test_typed_ir_rejects_inapplicable_fields_and_invalid_construction(self):
        with self.assertRaises(InvalidInput):
            Instruction(kind="compute", operation_id="alpha", tensor_id="hidden")
        with self.assertRaises(InvalidInput):
            Instruction(kind="drain", operations=["alpha"])
        with self.assertRaises(InvalidInput):
            Instruction(kind="wait", operations=("alpha",), copies=[])
        with self.assertRaises(InvalidInput):
            Region(id="nested", kind="layer", members=["alpha"])
        with self.assertRaises(InvalidInput):
            replace(parse_strategy_ir(document()), program=("compute alpha",))
        with self.assertRaises(InvalidInput):
            strategy_to_document({"id": "not-typed"})

    def test_mutable_mapping_aliases_are_removed_and_serialization_rechecks(self):
        raw = document()
        ir = parse_strategy_ir(raw)
        raw["placements"]["alpha"] = "gpu"
        raw["device_states"]["limited"] = "other.json"
        self.assertEqual(ir.placements["alpha"], "cpu")
        self.assertEqual(ir.device_states["limited"], "tiny.limited.scenario.json")
        ir.placements["alpha"] = True
        with self.assertRaises(InvalidInput):
            strategy_to_document(ir)


if __name__ == "__main__":
    unittest.main()
