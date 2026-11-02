"""Independent synthetic accounting and provenance tests; no native timings."""
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import unittest

from test_model import known_fit_rows, model

SOURCE = Path(__file__).resolve().parents[1] / "bind_ir.py"
SPEC = importlib.util.spec_from_file_location("composition_binding_test", SOURCE)
binding_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(binding_module)


def binding(fitted, family="ffn", strategy="hybrid/preload"):
    case = fitted["profile_cases"][family]
    evidence = [dict(row_id=r["row_id"], raw_sha256=r["raw_sha256"])
                for r in fitted["training"]["rows"] if r["case_id"] == family]
    signature = dict(case=case, profile_evidence=evidence,
        protocol=dict(measurement_contract=dict(kind="completed_batch_mean", batch_repetitions=16,
                                                isolated_latency_validated=False)))
    prices = fitted["profile_component_prices_us"][family]
    def terms(node):
        return {node} if isinstance(node, str) else set().union(*(terms(v) for v in next(iter(node.values()))))
    components = [dict(metric=metric, measured_median_ns=prices[metric] * 1000,
        runtime_duration_ns=math.ceil(prices[metric] * 1000),
        sources=[dict(**r, split="profile", metric=metric) for r in evidence])
        for metric in sorted(terms(model.recipe(case, strategy)))]
    baseline = {"gpu/resident": 5000, "hybrid/resident": 6000, "hybrid/preload": 9000}[strategy]
    return dict(schema="exact-measured-fixed-ir-binding-0.1", case_id=family, strategy=strategy,
        measured_physical_signature=signature, physical_signature_sha256=binding_module.canonical_hash(signature),
        timing_only=True, physical_capacity_validated=False, capacity_fields_are_nonphysical=True,
        unknown_cuda_workspace_preserved=True, high_level_ModelPlan_transform_lowering_exercised=False,
        full_Dense_MoE_inference_validated=False, component_bindings=components,
        independently_replayed_makespan_ns=baseline, formula_with_integer_prices_ns=baseline)


class ExactIRAccountingTests(unittest.TestCase):
    def setUp(self):
        self.fitted = model.fit(known_fit_rows())

    def test_hand_derived_totals_keep_replay_and_sources_unchanged(self):
        for family, strategy, expected in (("ffn", "gpu/resident", 13000),
                ("ffn", "hybrid/resident", 17000), ("ffn", "hybrid/preload", 33000),
                ("attention", "gpu/resident", 13000), ("attention", "hybrid/resident", 16000)):
            with self.subTest(family=family, strategy=strategy):
                value = binding(self.fitted, family, strategy)
                before, fitted_before = deepcopy(value), deepcopy(self.fitted)
                result = binding_module.apply(value, self.fitted, applicability_signature=before["measured_physical_signature"])
                self.assertEqual(value, before)
                self.assertEqual(self.fitted, fitted_before)
                self.assertEqual(result["independently_replayed_makespan_ns"], value["independently_replayed_makespan_ns"])
                self.assertAlmostEqual(result["corrected_complete_chain_latency_ns"], expected)
                self.assertFalse(result["timeline_modified"])
                self.assertFalse(result["per_event_correction_identified"])
                self.assertFalse(result["physical_capacity_validated"])
                self.assertTrue(result["unknown_cuda_workspace_preserved"])
                self.assertFalse(result["heldout_used_for_calibration"])

    def test_mismatched_geometry_protocol_raw_identity_and_component_prices_fail_closed(self):
        def change_geometry(b):
            b["measured_physical_signature"]["case"]["hidden"] *= 2
        def change_repetitions(b):
            b["measured_physical_signature"]["protocol"]["measurement_contract"]["batch_repetitions"] = 8
        def change_raw(b):
            b["measured_physical_signature"]["profile_evidence"][0]["raw_sha256"] = "a" * 64
        mutations = [change_geometry, change_repetitions, change_raw,
            lambda b: b["measured_physical_signature"]["protocol"].update(threads=99),
            lambda b: b["measured_physical_signature"].update(boundary={"output_location": "vram"}),
            lambda b: b["measured_physical_signature"]["protocol"]["measurement_contract"].update(normalization="other"),
            lambda b: b["component_bindings"][0].update(measured_median_ns=1001),
            lambda b: b["component_bindings"][0].update(runtime_duration_ns=999),
            lambda b: b["component_bindings"][0]["sources"][0].update(split="heldout"),
            lambda b: b["component_bindings"][0]["sources"][0].update(metric="other"),
            lambda b: b["component_bindings"].append(deepcopy(b["component_bindings"][0])),
            lambda b: b.update(formula_with_integer_prices_ns=1),
            lambda b: b.update(independently_replayed_makespan_ns=9001)]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                value = deepcopy(binding(self.fitted))
                signature = deepcopy(value["measured_physical_signature"])
                mutation(value)
                value["physical_signature_sha256"] = binding_module.canonical_hash(value["measured_physical_signature"])
                with self.assertRaises(ValueError):
                    binding_module.apply(value, self.fitted, applicability_signature=signature)

    def test_signature_hash_and_capacity_claims_cannot_be_silently_accepted(self):
        for key, bad in (("physical_signature_sha256", "a" * 64),
                         ("physical_capacity_validated", True),
                         ("unknown_cuda_workspace_preserved", False),
                         ("high_level_ModelPlan_transform_lowering_exercised", True)):
            with self.subTest(key=key):
                value = binding(self.fitted)
                signature = deepcopy(value["measured_physical_signature"])
                value[key] = bad
                with self.assertRaises(ValueError):
                    binding_module.apply(value, self.fitted, applicability_signature=signature)
        for key, bad in (("heldout_used_for_calibration", True),
                         ("gpu_capacity_proved", True),
                         ("timing_unit", "nanoseconds_per_isolated_request"),
                         ("feature_units", {"composition_boundaries": "seconds"}),
                         ("coefficient_units", {"composition_boundaries": "seconds"}),
                         ("profile_complete_strategy_totals_used_for_calibration", False)):
            with self.subTest(model_key=key):
                fitted = deepcopy(self.fitted)
                fitted[key] = bad
                with self.assertRaises(ValueError):
                    binding_module.apply(binding(self.fitted), fitted, applicability_signature=binding(self.fitted)["measured_physical_signature"])


if __name__ == "__main__":
    unittest.main()
