"""Measured macro costs bind exact physical boundaries and per-request accounting."""
from __future__ import annotations

import copy
import dataclasses
import unittest

from llm_infer_model.m2.costs import (
    MacroShape, canonical_contract_hash, expand_costs_document, parse_cost_table,
)
from llm_infer_model.tensor.spec import InvalidInput, Unsupported


def signature():
    # The physical producer owns its domain schema. The cost table binds every
    # selected JSON field, including port layouts, operation lists and variants.
    return {
        "contract": "test-physical-macro-1", "kind": "dense_ffn",
        "geometry": {"hidden": 8, "intermediate": 16},
        "operations": ["gate", "up", "silu", "multiply", "down", "copy"],
        "outputs": [{"name": "hidden", "dtype": "f32", "shape": [8, 3]}],
        "weight_dtypes": {"gate": "f16", "up": "f16", "down": "f32"},
        "implementation_variant": "explicit_ggml_swiglu",
    }


def entry(*, measured=True):
    raw = {
        "kind": "dense_ffn", "device": "cpu", "phase": "prefill",
        "batch": 1, "tokens": 3, "visible_history": 0,
        "duration_ns": 1200, "workspace_bytes": 96,
        "provenance": "measured_macro" if measured else "synthetic_layer",
    }
    if measured:
        raw.update({
            "accounting": "per_request", "workspace_scope": "macro_internal_scratch",
            "workspace_source": "allocator_capacity_plus_cpu_plan",
            "physical_signature": signature(),
            "contract_hash": canonical_contract_hash(signature()),
        })
    return raw


def table(raw):
    return parse_cost_table({"source": "unit-test", "entries": [raw], "validation_only": False})


class M2CostContract(unittest.TestCase):
    shape = MacroShape("prefill", 1, 3, 0)

    def test_measured_tables_cannot_omit_machine_readable_quality_status(self):
        with self.assertRaisesRegex(InvalidInput, "validation_only"):
            parse_cost_table({"source": "unmarked-export", "entries": [entry()]})

    def lookup(self, cost_table, *, physical=None, shape=None):
        return cost_table.lookup(
            "dense_ffn", "cpu", shape or self.shape, physical_signature=physical,
        )

    def test_cost_table_versions_are_explicit_and_supported(self):
        for version in ("m2-layer-0.1", "m2-layer-0.2", "m2-layer-0.3"):
            with self.subTest(version=version):
                parsed = parse_cost_table({"source": "test", "format_version": version,
                                          "entries": [entry(measured=False)]})
                self.assertFalse(parsed.validation_only)
        with self.assertRaisesRegex(Unsupported, "format_version"):
            parse_cost_table({"source": "test", "format_version": "m2-layer-0.99",
                              "entries": [entry(measured=False)]})
        for version in (None, True, 2, ""):
            with self.subTest(version=version), self.assertRaisesRegex(InvalidInput, "format_version"):
                parse_cost_table({"source": "test", "format_version": version,
                                  "entries": [entry(measured=False)]})
        self.assertFalse(table(entry(measured=False)).validation_only)

    def test_validation_only_flag_preserves_explicit_boolean(self):
        for flag in (True, False):
            with self.subTest(flag=flag):
                parsed = parse_cost_table({"source": "test", "validation_only": flag,
                                          "entries": [entry(measured=False)]})
                self.assertIs(parsed.validation_only, flag)
        for flag in (None, 0, 1, "true", []):
            with self.subTest(flag=flag), self.assertRaisesRegex(InvalidInput, "validation_only"):
                parse_cost_table({"source": "test", "validation_only": flag,
                                  "entries": [entry(measured=False)]})

    def test_old_synthetic_entry_keeps_per_request_context_batch(self):
        raw = entry(measured=False)
        raw["batch"] = 2
        cost_table = table(raw)
        cost = self.lookup(cost_table, shape=MacroShape("prefill", 2, 3, 0))
        self.assertEqual(cost.accounting, "per_request")
        self.assertEqual(cost.duration_ns, 1200)
        # Two requests are two charges; the legacy B2 key is invocation context.
        source, doc = expand_costs_document(cost_table, [
            ("r0", "dense_ffn", "cpu", MacroShape("prefill", 2, 3, 0)),
            ("r1", "dense_ffn", "cpu", MacroShape("prefill", 2, 3, 0)),
        ])
        self.assertEqual(source, "m2:synthetic_layer")
        self.assertEqual(sum(v["cpu"]["duration_ns"] for v in doc["compute"].values()), 2400)

    def test_measured_batch_two_is_refused_even_when_declared_per_request(self):
        raw = entry()
        raw["batch"] = 2
        with self.assertRaisesRegex(Unsupported, "batch=1"):
            table(raw)
        self.lookup(table(entry()), physical=signature())

    def test_whole_batch_accounting_cannot_be_charged_per_request(self):
        for measured in (True, False):
            with self.subTest(measured=measured):
                raw = entry(measured=measured)
                raw["accounting"] = "whole_batch"
                with self.assertRaisesRegex(Unsupported, "per_request"):
                    table(raw)

    def test_measured_requires_explicit_accounting(self):
        raw = entry()
        del raw["accounting"]
        with self.assertRaisesRegex(InvalidInput, "accounting"):
            table(raw)
        self.lookup(table(entry()), physical=signature())

    def test_unknown_measured_workspace_is_retained_and_cannot_execute(self):
        raw = entry()
        raw["workspace_bytes"] = None
        cost_table = table(raw)
        self.assertIsNone(next(iter(cost_table.entries.values())).workspace_bytes)
        with self.assertRaisesRegex(Unsupported, "workspace_bytes 未知"):
            self.lookup(cost_table, physical=signature())
        with self.assertRaisesRegex(Unsupported, "workspace_bytes 未知"):
            expand_costs_document(cost_table, [("op", "dense_ffn", "cpu", self.shape)],
                                  physical_signatures={"op": signature()})

    def test_explicit_zero_measured_workspace_needs_evidence_and_remains_zero(self):
        raw = entry()
        raw["workspace_bytes"] = 0
        self.assertEqual(self.lookup(table(raw), physical=signature()).workspace_bytes, 0)
        del raw["workspace_source"]
        with self.assertRaisesRegex(InvalidInput, "workspace_source"):
            table(raw)

    def test_workspace_scope_is_not_silently_reinterpreted(self):
        raw = entry()
        raw["workspace_scope"] = "whole_process_peak"
        with self.assertRaisesRegex(Unsupported, "workspace_scope"):
            table(raw)
        del raw["workspace_scope"]
        with self.assertRaisesRegex(InvalidInput, "workspace_scope"):
            table(raw)

    def test_empty_workspace_evidence_is_invalid(self):
        for value in ("", "   ", 0):
            raw = entry()
            raw["workspace_source"] = value
            with self.subTest(value=value), self.assertRaisesRegex(InvalidInput, "workspace_source"):
                table(raw)

    def test_synthetic_unknown_workspace_still_requires_explicit_number(self):
        raw = entry(measured=False)
        raw["workspace_bytes"] = None
        with self.assertRaisesRegex(InvalidInput, "workspace_bytes"):
            table(raw)

    def test_signature_is_required_for_measured_entries(self):
        raw = entry()
        del raw["physical_signature"]
        del raw["contract_hash"]
        with self.assertRaisesRegex(InvalidInput, "physical_signature"):
            table(raw)

    def test_measured_lookup_requires_expansion_boundary(self):
        with self.assertRaisesRegex(Unsupported, "展开器提供 physical_signature"):
            self.lookup(table(entry()))
        self.lookup(table(entry()), physical=signature())

    def test_hash_binds_complete_signature(self):
        raw = entry()
        raw["physical_signature"]["outputs"][0]["dtype"] = "f16"
        with self.assertRaisesRegex(InvalidInput, "contract_hash"):
            table(raw)
        raw["contract_hash"] = canonical_contract_hash(raw["physical_signature"])
        table(raw)

    def test_boundary_dtype_geometry_and_variant_mismatches_are_refused(self):
        cost_table = table(entry())
        for name, mutate in (
            ("boundary", lambda s: s["operations"].remove("copy")),
            ("dtype", lambda s: s["weight_dtypes"].update(down="f16")),
            ("geometry", lambda s: s["geometry"].update(hidden=16)),
            ("variant", lambda s: s.update(implementation_variant="fused")),
        ):
            expected = signature()
            mutate(expected)
            with self.subTest(name=name), self.assertRaisesRegex(Unsupported, "不匹配"):
                self.lookup(cost_table, physical=expected)
        self.lookup(cost_table, physical=signature())

    def test_hash_is_independent_of_object_key_order(self):
        original = signature()
        reversed_keys = dict(reversed(list(original.items())))
        self.assertEqual(canonical_contract_hash(original), canonical_contract_hash(reversed_keys))

    def test_signature_requires_finite_json_and_string_keys(self):
        for value in ({}, [], {1: "not a JSON key"}, {"x": float("nan")}, {"x": object()}):
            with self.subTest(value=repr(value)), self.assertRaises(InvalidInput):
                canonical_contract_hash(value)

    def test_distinct_json_bool_and_integer_do_not_compare_equal(self):
        raw = entry()
        raw["physical_signature"]["switch"] = True
        raw["contract_hash"] = canonical_contract_hash(raw["physical_signature"])
        expected = copy.deepcopy(raw["physical_signature"])
        expected["switch"] = 1
        with self.assertRaisesRegex(Unsupported, "不匹配"):
            self.lookup(table(raw), physical=expected)

    def test_mutating_a_parsed_signature_does_not_bypass_hash_validation(self):
        cost_table = table(entry())
        next(iter(cost_table.entries.values())).physical_signature["geometry"]["hidden"] = 16
        with self.assertRaisesRegex(InvalidInput, "contract_hash"):
            self.lookup(cost_table, physical=signature())

    def test_physical_synthetic_cost_requires_binding_when_expansion_provides_one(self):
        with self.assertRaisesRegex(Unsupported, "没有 physical_signature"):
            self.lookup(table(entry(measured=False)), physical=signature())

    def test_exact_bound_cost_expands_without_double_counting_workspace(self):
        source, doc = expand_costs_document(
            table(entry()), [("op", "dense_ffn", "cpu", self.shape)],
            physical_signatures={"op": signature()},
        )
        self.assertEqual(source, "m2:measured_macro")
        self.assertEqual(doc["compute"], {"op": {"cpu": {"duration_ns": 1200, "workspace_bytes": 96}}})

    def test_manual_cost_cannot_bypass_unknown_workspace_or_batch_gate(self):
        cost_table = table(entry())
        key = next(iter(cost_table.entries))
        cost_table.entries[key] = dataclasses.replace(cost_table.entries[key], workspace_bytes=None)
        with self.assertRaisesRegex(Unsupported, "workspace_bytes 未知"):
            self.lookup(cost_table, physical=signature())


if __name__ == "__main__":
    unittest.main()
