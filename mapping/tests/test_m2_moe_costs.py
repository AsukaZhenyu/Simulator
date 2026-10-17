"""MoE group accounting and exact synthetic transfer contracts."""
from __future__ import annotations

import copy
import dataclasses
import unittest

from llm_infer_model.m2.costs import (
    MacroCost, MacroShape, M2CostTable, TransferEntry,
    expand_costs_document, expand_transfer_costs, parse_cost_table,
)
from llm_infer_model.tensor.spec import InvalidInput, Unsupported

VERSION = "m2-layer-0.6"


def entry(kind="expert_ffn", **changes):
    raw = dict(kind=kind, device="cpu", phase="prefill", batch=2,
               tokens=4, visible_history=0, duration_ns=700,
               workspace_bytes=64, provenance="synthetic_layer",
               accounting="per_expert_group" if kind in {"expert_ffn", "shared_expert_ffn"} else "per_request")
    raw.update(changes)
    return raw


def transfer(route="ssd_to_dram", **changes):
    raw = dict(route=route, storage_bytes=128, duration_ns=19, provenance="synthetic_transfer")
    raw.update(changes)
    return raw


def document(**changes):
    raw = dict(format_version=VERSION, source="moe-semantic-fixture",
               entries=[entry()], transfer_entries=[transfer()])
    raw.update(changes)
    return raw


class MoeAccountingTests(unittest.TestCase):
    def test_aggregated_expert_group_cost_is_used_once_and_not_multiplied_by_batch(self):
        table = parse_cost_table(document())
        shape = MacroShape("prefill", 2, 4, 0)
        source, costs = expand_costs_document(table, [("group:expert-0", "expert_ffn", "cpu", shape)])
        self.assertEqual(source, "m2:synthetic_layer")
        self.assertEqual(costs["compute"], {"group:expert-0": {"cpu": {"duration_ns": 700, "workspace_bytes": 64}}})
        self.assertEqual(table.lookup("expert_ffn", "cpu", shape).accounting, "per_expert_group")

    def test_shared_expert_uses_same_whole_group_accounting(self):
        table = parse_cost_table(document(entries=[entry("shared_expert_ffn", tokens=6)]))
        self.assertEqual(table.lookup("shared_expert_ffn", "cpu", MacroShape("prefill", 2, 6, 0)).duration_ns, 700)

    def test_router_attention_and_combine_keep_per_request_history_context(self):
        for kind in ("projection", "attention", "router", "combine"):
            with self.subTest(kind=kind):
                table = parse_cost_table(document(entries=[entry(kind, tokens=3, visible_history=9)]))
                cost = table.lookup(kind, "cpu", MacroShape("prefill", 2, 3, 9))
                self.assertEqual(cost.accounting, "per_request")

    def test_expert_accounting_must_be_declared_explicitly(self):
        raw = document()
        del raw["entries"][0]["accounting"]
        with self.assertRaisesRegex(InvalidInput, "per_expert_group"):
            parse_cost_table(raw)

    def test_per_request_cannot_silently_charge_an_aggregated_expert(self):
        for kind in ("expert_ffn", "shared_expert_ffn"):
            with self.subTest(kind=kind), self.assertRaises(Unsupported):
                parse_cost_table(document(entries=[entry(kind, accounting="per_request")]))

    def test_group_accounting_is_not_accepted_for_other_tasks_or_older_versions(self):
        with self.assertRaises(Unsupported):
            parse_cost_table(document(entries=[entry("attention", accounting="per_expert_group")]))
        for version in (None, "m2-layer-0.1", "m2-layer-0.5"):
            raw = document(format_version=version)
            if version is None:
                del raw["format_version"]
            if version != "m2-layer-0.5":
                del raw["transfer_entries"]
            else:
                raw["transfer_entries"] = []
            with self.subTest(version=version), self.assertRaises(Unsupported):
                parse_cost_table(raw)

    def test_history_zero_is_an_explicit_expert_group_nonapplicability_contract(self):
        with self.assertRaisesRegex(Unsupported, "visible_history"):
            parse_cost_table(document(entries=[entry(visible_history=5)]))
        table = parse_cost_table(document())
        # Programmatic mutation cannot bypass the shape/accounting contract.
        wrong_shape = MacroShape("prefill", 2, 4, 5)
        table = dataclasses.replace(table, entries={("expert_ffn", "cpu", wrong_shape): MacroCost(700, "synthetic_layer", 64, "per_expert_group")})
        with self.assertRaisesRegex(Unsupported, "visible_history"):
            table.lookup("expert_ffn", "cpu", wrong_shape)

    def test_missing_exact_group_token_count_is_not_interpolated(self):
        with self.assertRaises(Unsupported):
            parse_cost_table(document()).lookup("expert_ffn", "cpu", MacroShape("prefill", 2, 3, 0))

    def test_duplicate_operation_device_is_rejected_instead_of_hidden_double_charge(self):
        requirement = ("group:expert-0", "expert_ffn", "cpu", MacroShape("prefill", 2, 4, 0))
        with self.assertRaises(InvalidInput):
            expand_costs_document(parse_cost_table(document()), [requirement, requirement])

    def test_measured_or_unverified_compute_provenance_is_refused(self):
        for tag in ("measured_macro", "synthetic", "synthetic_fixed", "measured", "unknown"):
            with self.subTest(tag=tag), self.assertRaises(Unsupported):
                parse_cost_table(document(entries=[entry(provenance=tag)]))

    def test_physical_envelope_fields_are_rejected_even_when_null(self):
        for field in ("physical_signature", "contract_hash"):
            with self.subTest(field=field), self.assertRaises(Unsupported):
                parse_cost_table(document(entries=[entry(**{field: None})]))

    def test_unknown_scratch_is_not_inferred_zero(self):
        with self.assertRaises(InvalidInput):
            parse_cost_table(document(entries=[entry(workspace_bytes=None)]))
        raw = document()
        del raw["entries"][0]["workspace_bytes"]
        with self.assertRaises(InvalidInput):
            parse_cost_table(raw)
        table = parse_cost_table(document())
        key = next(iter(table.entries))
        table = dataclasses.replace(table, entries={key: dataclasses.replace(table.entries[key], workspace_bytes=None)})
        with self.assertRaisesRegex(Unsupported, "workspace_bytes"):
            table.lookup(*key)

    def test_manual_moe_cost_cannot_bypass_synthetic_or_group_accounting(self):
        shape = MacroShape("prefill", 2, 4, 0)
        for cost in (MacroCost(700, "measured_macro", 64, "per_expert_group"),
                     MacroCost(700, "synthetic_layer", 64, "per_request")):
            table = M2CostTable("manual", {("expert_ffn", "cpu", shape): cost}, format_version=VERSION)
            with self.subTest(cost=cost), self.assertRaises(Unsupported):
                table.lookup("expert_ffn", "cpu", shape)


class MoeTransferTests(unittest.TestCase):
    def test_all_four_routes_bind_exact_size_independent_costs(self):
        routes = ("dram_to_vram", "vram_to_dram", "dram_to_ssd", "ssd_to_dram")
        raw = document(transfer_entries=[transfer(route, duration_ns=i+1) for i, route in enumerate(routes)])
        table = parse_cost_table(raw)
        labels, costs = expand_transfer_costs(table, {"weight": 128}, {route: ["weight"] for route in routes})
        self.assertEqual(labels, {"synthetic_transfer"})
        for i, route in enumerate(routes):
            self.assertEqual(costs[route], {"weight": {"duration_ns": i+1}})
        self.assertEqual(table.format_version, VERSION)

    def test_ssd_route_does_not_borrow_pcie_or_reverse_direction(self):
        table = parse_cost_table(document(transfer_entries=[transfer("dram_to_vram")]))
        for route in ("ssd_to_dram", "dram_to_ssd", "vram_to_dram"):
            with self.subTest(route=route), self.assertRaises(Unsupported):
                table.lookup_transfer(route, 128)

    def test_exact_storage_size_is_required_no_bandwidth_default(self):
        table = parse_cost_table(document())
        with self.assertRaises(Unsupported):
            table.lookup_transfer("ssd_to_dram", 127)
        with self.assertRaises(Unsupported):
            expand_transfer_costs(table, {"weight": 256}, {"ssd_to_dram": ["weight"]})

    def test_legacy_hybrid_remains_closed_to_ssd(self):
        for route in ("ssd_to_dram", "dram_to_ssd"):
            with self.subTest(route=route), self.assertRaises(Unsupported):
                parse_cost_table(document(format_version="m2-layer-0.5", entries=[entry("attention")], transfer_entries=[transfer(route)]))
        table = parse_cost_table(document(format_version="m2-layer-0.5", entries=[entry("attention")], transfer_entries=[]))
        table = dataclasses.replace(table, transfers={("ssd_to_dram", 128): TransferEntry(19, "synthetic_transfer")})
        with self.assertRaises(Unsupported):
            table.lookup_transfer("ssd_to_dram", 128)
        with self.assertRaises(Unsupported):
            expand_transfer_costs(table, {"weight": 128}, {"ssd_to_dram": ["weight"]})

    def test_moe_requires_explicit_transfer_table_even_when_empty(self):
        raw = document()
        del raw["transfer_entries"]
        with self.assertRaises(InvalidInput):
            parse_cost_table(raw)
        self.assertEqual(parse_cost_table(document(transfer_entries=[])).transfers, {})

    def test_transfer_synthetic_provenance_and_positive_duration_remain_mandatory(self):
        with self.assertRaises(Unsupported):
            parse_cost_table(document(transfer_entries=[transfer(provenance="measured_transfer")]))
        with self.assertRaises(InvalidInput):
            parse_cost_table(document(transfer_entries=[transfer(duration_ns=0)]))

    def test_unsupported_direct_ssd_vram_route_is_not_invented(self):
        for route in ("ssd_to_vram", "vram_to_ssd"):
            with self.subTest(route=route), self.assertRaises(Unsupported):
                parse_cost_table(document(transfer_entries=[transfer(route)]))


if __name__ == "__main__":
    unittest.main()
