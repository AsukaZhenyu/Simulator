"""Exact synthetic transfer costs and independent CPU/GPU macro signatures.

No hardware measurements run here. Unknown measured CUDA scratch is kept
unknown and cannot enter an executable hybrid capacity model.
"""
from __future__ import annotations

import copy
import unittest

from llm_infer_model.m2.costs import (
    MacroShape, canonical_contract_hash, expand_costs_document,
    expand_transfer_costs, parse_cost_table,
)
from llm_infer_model.tensor.spec import InvalidInput, Unsupported


SHAPE = MacroShape("prefill", 1, 3, 0)


def macro_entry(device="cpu", **updates):
    signature = {"contract": "test_macro_boundary", "device": device, "tokens": 3}
    raw = {
        "kind": "projection", "device": device, "phase": SHAPE.phase,
        "batch": SHAPE.batch, "tokens": SHAPE.tokens,
        "visible_history": SHAPE.visible_history,
        "duration_ns": 100 if device == "cpu" else 60,
        "workspace_bytes": 0, "provenance": "synthetic_layer",
        "physical_signature": signature,
        "contract_hash": canonical_contract_hash(signature),
    }
    raw.update(updates)
    return raw


def transfer_entry(route="dram_to_vram", storage_bytes=128, **updates):
    raw = {
        "route": route, "storage_bytes": storage_bytes,
        "duration_ns": 20, "provenance": "synthetic_transfer",
    }
    raw.update(updates)
    return raw


def cost_document(**updates):
    raw = {
        "format_version": "m2-layer-0.5", "source": "synthetic-hybrid-test",
        "entries": [macro_entry("cpu"), macro_entry("gpu")],
        "transfer_entries": [
            transfer_entry(), transfer_entry("vram_to_dram", duration_ns=35),
            transfer_entry(storage_bytes=256, duration_ns=41),
        ],
    }
    raw.update(updates)
    return raw


class HybridTransferParsing(unittest.TestCase):
    def test_direction_and_storage_bytes_are_independent_exact_keys(self):
        table = parse_cost_table(cost_document())
        self.assertEqual(table.lookup_transfer("dram_to_vram", 128).duration_ns, 20)
        self.assertEqual(table.lookup_transfer("vram_to_dram", 128).duration_ns, 35)
        self.assertEqual(table.lookup_transfer("dram_to_vram", 256).duration_ns, 41)

    def test_hybrid_requires_explicit_transfer_list(self):
        raw = cost_document()
        del raw["transfer_entries"]
        with self.assertRaisesRegex(InvalidInput, "transfer_entries"):
            parse_cost_table(raw)

    def test_explicit_empty_list_permits_compute_only_table(self):
        table = parse_cost_table(cost_document(transfer_entries=[]))
        self.assertEqual(table.transfers, {})
        self.assertEqual(table.provenances, frozenset({"synthetic_layer"}))

    def test_legacy_versions_refuse_new_transfer_field_even_when_empty(self):
        for version in [None, "m2-layer-0.1", "m2-layer-0.2", "m2-layer-0.3", "m2-layer-0.4"]:
            with self.subTest(version=version):
                raw = cost_document(format_version=version, transfer_entries=[])
                if version is None:
                    del raw["format_version"]
                with self.assertRaisesRegex(Unsupported, "transfer_entries"):
                    parse_cost_table(raw)

    def test_legacy_table_keeps_default_empty_transfers(self):
        raw = cost_document(format_version="m2-layer-0.4")
        del raw["transfer_entries"]
        self.assertEqual(parse_cost_table(raw).transfers, {})

    def test_list_type_is_not_guessed(self):
        for value in [None, {}, "", True]:
            with self.subTest(value=value), self.assertRaises(InvalidInput):
                parse_cost_table(cost_document(transfer_entries=value))

    def test_transfer_entry_requires_exact_fields(self):
        for field in transfer_entry():
            raw = transfer_entry()
            del raw[field]
            with self.subTest(missing=field), self.assertRaises(InvalidInput):
                parse_cost_table(cost_document(transfer_entries=[raw]))
        with self.assertRaises(InvalidInput):
            parse_cost_table(cost_document(transfer_entries=[transfer_entry(bandwidth=10)]))

    def test_transfer_entry_requires_object(self):
        for value in [None, [], 20]:
            with self.subTest(value=value), self.assertRaises(InvalidInput):
                parse_cost_table(cost_document(transfer_entries=[value]))

    def test_synthetic_provenance_is_explicit_and_nonempty(self):
        for value in [None, "", " ", 0]:
            with self.subTest(value=value), self.assertRaises(InvalidInput):
                parse_cost_table(cost_document(transfer_entries=[transfer_entry(provenance=value)]))

    def test_measured_or_ambiguous_transfer_provenance_is_unsupported(self):
        for value in ["measured_transfer", "synthetic", "synthetic_transfer "]:
            with self.subTest(value=value), self.assertRaises(Unsupported):
                parse_cost_table(cost_document(transfer_entries=[transfer_entry(provenance=value)]))

    def test_table_provenance_cannot_fill_missing_transfer_provenance(self):
        entry = transfer_entry()
        del entry["provenance"]
        with self.assertRaises(InvalidInput):
            parse_cost_table(cost_document(provenance="synthetic_transfer", transfer_entries=[entry]))

    def test_duration_is_strict_positive_integer(self):
        for value in [0, -1, True, 2.0, "20", None]:
            with self.subTest(value=value), self.assertRaises(InvalidInput):
                parse_cost_table(cost_document(transfer_entries=[transfer_entry(duration_ns=value)]))

    def test_size_is_strict_positive_integer(self):
        for value in [0, -1, True, 128.0, "128", None]:
            with self.subTest(value=value), self.assertRaises(InvalidInput):
                parse_cost_table(cost_document(transfer_entries=[transfer_entry(storage_bytes=value)]))

    def test_duplicate_route_and_size_is_not_overwritten(self):
        with self.assertRaisesRegex(InvalidInput, "重复"):
            parse_cost_table(cost_document(transfer_entries=[transfer_entry(), transfer_entry(duration_ns=99)]))

    def test_transfer_routes_are_limited_to_dram_vram(self):
        for route in ["dram_to_ssd", "ssd_to_dram", "ssd_to_vram", "gpu", ""]:
            exception = InvalidInput if not route else Unsupported
            with self.subTest(route=route), self.assertRaises(exception):
                parse_cost_table(cost_document(transfer_entries=[transfer_entry(route)]))

    def test_lookup_does_not_interpolate_a_nearby_size(self):
        table = parse_cost_table(cost_document())
        with self.assertRaisesRegex(Unsupported, "storage_bytes=129"):
            table.lookup_transfer("dram_to_vram", 129)

    def test_lookup_does_not_borrow_other_direction(self):
        table = parse_cost_table(cost_document(transfer_entries=[transfer_entry()]))
        with self.assertRaises(Unsupported):
            table.lookup_transfer("vram_to_dram", 128)


class HybridTransferExpansion(unittest.TestCase):
    def setUp(self):
        self.table = parse_cost_table(cost_document())

    def test_only_allowed_tensors_are_costed_using_storage_bytes(self):
        provenance, document = expand_transfer_costs(
            self.table, {"hidden": 128, "weight": 256, "unused": 129},
            {"dram_to_vram": ["hidden", "weight"], "vram_to_dram": ["hidden"]},
        )
        self.assertEqual(provenance, {"synthetic_transfer"})
        self.assertEqual(document, {
            "dram_to_vram": {"hidden": {"duration_ns": 20}, "weight": {"duration_ns": 41}},
            "vram_to_dram": {"hidden": {"duration_ns": 35}},
        })

    def test_missing_tensor_storage_size_is_rejected(self):
        with self.assertRaisesRegex(InvalidInput, "storage_bytes"):
            expand_transfer_costs(self.table, {}, {"dram_to_vram": ["hidden"]})

    def test_missing_exact_allowed_tensor_size_is_rejected(self):
        with self.assertRaises(Unsupported):
            expand_transfer_costs(self.table, {"hidden": 129}, {"dram_to_vram": ["hidden"]})

    def test_duplicate_allowed_tensor_does_not_silently_collapse(self):
        with self.assertRaisesRegex(InvalidInput, "重复"):
            expand_transfer_costs(self.table, {"hidden": 128}, {"dram_to_vram": ["hidden", "hidden"]})

    def test_empty_ssd_routes_can_match_core_schema(self):
        provenance, document = expand_transfer_costs(
            self.table, {}, {"dram_to_vram": [], "vram_to_dram": [], "dram_to_ssd": [], "ssd_to_dram": []},
        )
        self.assertEqual(provenance, set())
        self.assertEqual(document, {"dram_to_vram": {}, "vram_to_dram": {}, "dram_to_ssd": {}, "ssd_to_dram": {}})

    def test_ssd_data_transfer_is_not_silently_accepted(self):
        with self.assertRaises(Unsupported):
            expand_transfer_costs(self.table, {"hidden": 128}, {"dram_to_ssd": ["hidden"]})

    def test_unknown_route_is_rejected_even_if_empty(self):
        with self.assertRaises(Unsupported):
            expand_transfer_costs(self.table, {}, {"ssd_to_vram": []})

    def test_allowed_tensor_ids_require_real_sequence(self):
        for value in ["hidden", {"hidden"}, None]:
            with self.subTest(value=value), self.assertRaises(InvalidInput):
                expand_transfer_costs(self.table, {"hidden": 128}, {"dram_to_vram": value})

    def test_allowed_tensor_ids_are_nonempty_strings(self):
        for value in [None, "", 1]:
            with self.subTest(value=value), self.assertRaises(InvalidInput):
                expand_transfer_costs(self.table, {"hidden": 128}, {"dram_to_vram": [value]})

    def test_storage_size_cannot_be_zero_unknown_or_boolean(self):
        for value in [None, 0, True]:
            with self.subTest(value=value), self.assertRaises(InvalidInput):
                expand_transfer_costs(self.table, {"hidden": value}, {"dram_to_vram": ["hidden"]})


class HybridComputeSignatures(unittest.TestCase):
    def setUp(self):
        self.raw = cost_document()
        self.table = parse_cost_table(self.raw)
        self.requirements = [("r0.projection", "projection", device, SHAPE) for device in ["cpu", "gpu"]]
        self.signatures = {("r0.projection", entry["device"]): entry["physical_signature"] for entry in self.raw["entries"]}

    def test_same_operation_binds_each_device_signature_independently(self):
        source, document = expand_costs_document(self.table, self.requirements, physical_signatures=self.signatures)
        self.assertEqual(source, "m2:synthetic_layer")
        self.assertEqual(document["compute"]["r0.projection"], {
            "cpu": {"duration_ns": 100, "workspace_bytes": 0},
            "gpu": {"duration_ns": 60, "workspace_bytes": 0},
        })

    def test_device_key_takes_precedence_over_legacy_operation_key(self):
        signatures = dict(self.signatures)
        signatures["r0.projection"] = {"irrelevant_legacy_fallback": True}
        _, document = expand_costs_document(self.table, self.requirements, physical_signatures=signatures)
        self.assertEqual(set(document["compute"]["r0.projection"]), {"cpu", "gpu"})

    def test_legacy_operation_signature_still_works_for_single_device(self):
        source, _ = expand_costs_document(
            self.table, self.requirements[:1],
            physical_signatures={"r0.projection": self.signatures[("r0.projection", "cpu")]},
        )
        self.assertEqual(source, "m2:synthetic_layer")

    def test_cpu_signature_cannot_be_used_for_gpu_entry(self):
        signatures = dict(self.signatures)
        signatures[("r0.projection", "gpu")] = signatures[("r0.projection", "cpu")]
        with self.assertRaisesRegex(Unsupported, "不匹配"):
            expand_costs_document(self.table, self.requirements, physical_signatures=signatures)

    def test_duplicate_operation_device_requirement_is_rejected(self):
        with self.assertRaisesRegex(InvalidInput, "两条成本"):
            expand_costs_document(self.table, self.requirements[:1] * 2, physical_signatures=self.signatures)

    def test_source_combines_compute_and_actually_used_transfer_provenance(self):
        used_provenance, transfer = expand_transfer_costs(
            self.table, {"hidden": 128}, {"dram_to_vram": ["hidden"]},
        )
        source, document = expand_costs_document(
            self.table, self.requirements, physical_signatures=self.signatures,
            transfer_provenances=used_provenance,
        )
        document["transfer"] = transfer
        self.assertEqual(source, "m2:synthetic_layer+synthetic_transfer")
        self.assertEqual(document["source"], source)
        self.assertEqual(document["transfer"]["dram_to_vram"]["hidden"], {"duration_ns": 20})

    def test_unused_transfer_table_does_not_label_compute_as_transfer(self):
        used_provenance, _ = expand_transfer_costs(self.table, {}, {"dram_to_vram": []})
        source, _ = expand_costs_document(
            self.table, self.requirements, physical_signatures=self.signatures,
            transfer_provenances=used_provenance,
        )
        self.assertEqual(source, "m2:synthetic_layer")

    def test_unknown_measured_workspace_is_preserved_and_execution_refused(self):
        raw = copy.deepcopy(self.raw)
        raw["validation_only"] = True
        gpu = raw["entries"][1]
        gpu.update(
            provenance="measured_macro", workspace_bytes=None,
            accounting="per_request", workspace_scope="macro_internal_scratch",
            workspace_source="test-evidence-unknown-CUDA-scratch",
        )
        table = parse_cost_table(raw)
        self.assertIsNone(table.entries[("projection", "gpu", SHAPE)].workspace_bytes)
        with self.assertRaisesRegex(Unsupported, "workspace_bytes 未知"):
            expand_costs_document(table, self.requirements, physical_signatures=self.signatures)


if __name__ == "__main__":
    unittest.main()
