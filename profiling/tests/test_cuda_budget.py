"""Offline explicit CUDA budget checks. No model or hardware is executed."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bind_cuda_budget as binder


def invocation():
    return {
        "format_version": "m2-layer-0.4", "id": "synthetic-budget-fixture",
        "layer": "synthetic-layer", "devices": ["gpu"],
        "phase": "prefill", "requests": [], "state_in": {"requests": []},
        "hardware": {"dram_capacity_bytes": 1, "dram_alignment_bytes": 1,
            "dram_runtime_reserved_bytes": 0, "vram_capacity_bytes": 512,
            "vram_runtime_reserved_bytes": 0, "vram_alignment_bytes": 1},
        "costs": {"format_version": "m2-layer-0.4", "validation_only": True,
            "entries": [{"workspace_bytes": None, "duration_ns": 37}]},
        "allow_validation_only_costs": True,
    }


def device():
    return {"index": 0, "name": "Synthetic GPU", "uuid": "synthetic-uuid",
            "total_memory_bytes": 4096, "compute_capability_major": 8,
            "compute_capability_minor": 9}


def arguments():
    return {"observed_device": device(), "capacity_bytes": 2048,
        "runtime_reserved_bytes": 256, "alignment_bytes": 32,
        "reserve_source": "Explicit synthetic policy, excludes macro workspace",
        "device_uuid": "synthetic-uuid",
        "source_hashes": {"manifest_sha256": "a"*64, "memory_report_sha256": "b"*64}}


class PureBudgetTests(unittest.TestCase):
    def test_only_three_vram_fields_change_and_inputs_are_immutable(self):
        original = invocation()
        params = arguments()
        saved = copy.deepcopy((original, params))
        result, report = binder.bind_budget(original, **params)
        expected = invocation()
        expected["hardware"].update(vram_capacity_bytes=2048,
            vram_runtime_reserved_bytes=256, vram_alignment_bytes=32)
        self.assertEqual(result, expected)
        self.assertEqual((original, params), saved)
        self.assertIsNone(result["costs"]["entries"][0]["workspace_bytes"])
        self.assertTrue(result["costs"]["validation_only"])
        self.assertEqual(result["hardware"]["dram_capacity_bytes"], 1)
        result["costs"]["entries"][0]["duration_ns"] = 1
        self.assertEqual(original["costs"]["entries"][0]["duration_ns"], 37)
        self.assertIsNone(report["model_capacity_under_declared_budget"])
        for key in ("workspace_bound_known", "physical_capacity_verified",
                    "execution_performed", "costs_modified",
                    "reserve_inferred_from_device_free_memory"):
            self.assertIs(report[key], False)

    def test_free_memory_does_not_supply_or_override_declared_reserve(self):
        params = arguments()
        params["observed_device"].update(free_bytes=1, used_bytes=4095)
        bound, report = binder.bind_budget(invocation(), **params)
        self.assertEqual(bound["hardware"]["vram_runtime_reserved_bytes"], 256)
        self.assertEqual(report["declared_budget"]["available_for_modeled_allocations_bytes"], 1792)

    def test_explicit_zero_reserve_is_retained(self):
        params = arguments()
        params["runtime_reserved_bytes"] = 0
        bound, report = binder.bind_budget(invocation(), **params)
        self.assertEqual(bound["hardware"]["vram_runtime_reserved_bytes"], 0)
        self.assertEqual(report["declared_budget"]["runtime_reserved_bytes"], 0)
        self.assertIs(report["physical_capacity_verified"], False)

    def test_every_declaration_is_required_by_function(self):
        for key in ("capacity_bytes", "runtime_reserved_bytes", "alignment_bytes",
                    "reserve_source", "device_uuid", "observed_device", "source_hashes"):
            params = arguments()
            params.pop(key)
            with self.subTest(field=key), self.assertRaises(TypeError):
                binder.bind_budget(invocation(), **params)

    def test_bool_negative_zero_and_noninteger_budgets_rejected(self):
        for field, values in (("capacity_bytes", [True, False, 0, -1, 2.0, "2048", None]),
                ("runtime_reserved_bytes", [True, -1, 1.0, "256", None]),
                ("alignment_bytes", [True, False, 0, -1, 1.0, "32", None])):
            for value in values:
                params = arguments()
                params[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(binder.BudgetError):
                    binder.bind_budget(invocation(), **params)

    def test_reservation_must_leave_positive_budget(self):
        for reserved in (2048, 2049, 99999):
            params = arguments()
            params["runtime_reserved_bytes"] = reserved
            with self.subTest(reserved=reserved), self.assertRaises(binder.BudgetError):
                binder.bind_budget(invocation(), **params)

    def test_declared_capacity_cannot_exceed_queried_total(self):
        params = arguments()
        params["capacity_bytes"] = 4097
        with self.assertRaises(binder.BudgetError):
            binder.bind_budget(invocation(), **params)

    def test_alignment_cannot_exceed_declared_capacity(self):
        params = arguments()
        params["alignment_bytes"] = 2049
        with self.assertRaises(binder.BudgetError):
            binder.bind_budget(invocation(), **params)

    def test_uuid_is_exact_and_explicit(self):
        for value in (None, "", " ", "synthetic-uuid ", "another-device"):
            params = arguments()
            params["device_uuid"] = value
            with self.subTest(value=value), self.assertRaises(binder.BudgetError):
                binder.bind_budget(invocation(), **params)

    def test_reserve_source_cannot_be_missing_or_blank(self):
        for value in (None, "", " ", 0):
            params = arguments()
            params["reserve_source"] = value
            with self.subTest(value=value), self.assertRaises(binder.BudgetError):
                binder.bind_budget(invocation(), **params)

    def test_source_hashes_are_explicit_and_strict(self):
        for value in ({}, None, [], {"x": "ABC"}, {"x": "A"*64}, {"": "a"*64}):
            params = arguments()
            params["source_hashes"] = value
            with self.subTest(value=value), self.assertRaises(binder.BudgetError):
                binder.bind_budget(invocation(), **params)

    def test_cpu_multi_device_and_other_versions_rejected(self):
        for change in ({"devices": ["cpu"]}, {"devices": ["gpu", "cpu"]},
                       {"devices": []}, {"format_version": "m2-layer-0.3"}):
            source = invocation()
            source.update(change)
            with self.subTest(change=change), self.assertRaises(binder.BudgetError):
                binder.bind_budget(source, **arguments())

    def test_missing_or_invalid_existing_hardware_does_not_get_guessed(self):
        for field in ("vram_capacity_bytes", "vram_runtime_reserved_bytes", "vram_alignment_bytes"):
            source = invocation()
            source["hardware"].pop(field)
            with self.subTest(field=field), self.assertRaises(binder.BudgetError):
                binder.bind_budget(source, **arguments())
        source = invocation()
        source["hardware"]["vram_alignment_bytes"] = True
        with self.assertRaises(binder.BudgetError):
            binder.bind_budget(source, **arguments())

    def test_invalid_device_identity_rejected(self):
        for field, value in (("index", True), ("index", -1), ("uuid", ""),
                ("name", None), ("total_memory_bytes", True), ("total_memory_bytes", 0)):
            params = arguments()
            params["observed_device"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(binder.BudgetError):
                binder.bind_budget(invocation(), **params)

    def test_nonfinite_invocation_values_rejected(self):
        source = invocation()
        source["comment"] = float("nan")
        with self.assertRaises(binder.BudgetError):
            binder.bind_budget(source, **arguments())


class ReferencedBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source_dir = self.root/"original"
        self.source_dir.mkdir()
        self.source = self.source_dir/"invoke.json"
        self.layer = self.source_dir/"layer.json"
        self.costs = self.source_dir/"costs.json"
        self.layer.write_text(json.dumps({"layer": {"physical": {"cuda_device": 0}}}), encoding="utf-8")
        self.costs.write_text(json.dumps(invocation()["costs"]), encoding="utf-8")
        document = invocation()
        document.pop("costs")
        document.update(layer_file="layer.json", costs_file="costs.json")
        self.source.write_text(json.dumps(document), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_relative_references_keep_original_targets_after_output_move(self):
        costs_before = self.costs.read_bytes()
        layer_before = self.layer.read_bytes()
        source_before = self.source.read_bytes()
        bound, report = binder.prepare_invocation(self.source, **arguments())
        relocated = self.root/"different"/"invoke.json"
        relocated.parent.mkdir()
        relocated.write_text(json.dumps(bound), encoding="utf-8")
        loaded = json.loads(relocated.read_text(encoding="utf-8"))
        self.assertEqual(Path(loaded["layer_file"]), self.layer.resolve())
        self.assertEqual(Path(loaded["costs_file"]), self.costs.resolve())
        self.assertEqual(self.source.read_bytes(), source_before)
        self.assertEqual(self.layer.read_bytes(), layer_before)
        self.assertEqual(self.costs.read_bytes(), costs_before)
        self.assertIsNone(json.loads(self.costs.read_bytes())["entries"][0]["workspace_bytes"])
        self.assertEqual(report["reference_bindings"]["layer_file"]["original_reference"], "layer.json")
        self.assertEqual(report["source_hashes"][str(self.costs.resolve())], hashlib.sha256(costs_before).hexdigest())

    def test_absolute_references_are_retained_and_hashed(self):
        source = json.loads(self.source.read_text(encoding="utf-8"))
        source["layer_file"] = str(self.layer.resolve())
        self.source.write_text(json.dumps(source), encoding="utf-8")
        bound, report = binder.prepare_invocation(self.source, **arguments())
        self.assertEqual(bound["layer_file"], str(self.layer.resolve()))
        self.assertIn(str(self.layer.resolve()), report["source_hashes"])

    def test_layer_cuda_index_must_match_profile(self):
        for index in (1, True, None, 0.0):
            self.layer.write_text(json.dumps({"layer": {"physical": {"cuda_device": index}}}), encoding="utf-8")
            with self.subTest(index=index), self.assertRaises(binder.BudgetError):
                binder.prepare_invocation(self.source, **arguments())

    def test_conflicting_source_hash_is_rejected(self):
        params = arguments()
        params["source_hashes"][str(self.costs.resolve())] = "f"*64
        with self.assertRaises(binder.BudgetError):
            binder.prepare_invocation(self.source, **params)

    def test_missing_or_invalid_referenced_source_is_rejected(self):
        self.costs.unlink()
        with self.assertRaises(binder.BudgetError):
            binder.prepare_invocation(self.source, **arguments())
        self.costs.write_text("not-json", encoding="utf-8")
        with self.assertRaises(binder.BudgetError):
            binder.prepare_invocation(self.source, **arguments())

    def test_inline_costs_are_kept_and_unknown_workspace_stays_unknown(self):
        self.source.write_text(json.dumps(invocation()), encoding="utf-8")
        bound, report = binder.prepare_invocation(self.source, **arguments())
        self.assertEqual(bound["costs"], invocation()["costs"])
        self.assertIsNone(bound["costs"]["entries"][0]["workspace_bytes"])
        self.assertEqual(report["reference_bindings"], {})


if __name__ == "__main__":
    unittest.main()
