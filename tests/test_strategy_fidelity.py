"""Independent geometry, causality and rejection references for strategy lowering."""
import importlib.util
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "experiments/strategy_fidelity.py"
SPEC = importlib.util.spec_from_file_location("strategy_fidelity_reference", SOURCE)
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


class StrategyFidelityTests(unittest.TestCase):
    def check_case(self, name):
        with tempfile.TemporaryDirectory() as directory:
            reference.CASES[name](Path(directory))

    def test_attention_stage_geometry_and_charged_transfer(self):
        self.check_case("attention_boundary")

    def test_ffn_partitions_conserve_weights_and_merge_both_partials(self):
        self.check_case("ffn_partition")

    def test_composed_transforms_keep_startup_preparation_and_cache_budget(self):
        self.check_case("composition")

    def test_phase_placement_preserves_multilayer_multiround_kv(self):
        self.check_case("phase_kv_persistence")

    def test_scoped_placement_cross_layer_prefetch_and_cache_share_runtime(self):
        self.check_case("prefetch_cache_placement")

    def test_private_future_routes_do_not_change_public_prefix(self):
        self.check_case("moe_causality")

    def test_invalid_or_unpriced_transforms_are_rejected(self):
        self.check_case("transform_rejections")

    def test_infeasible_gpu_capacity_does_not_silently_fall_back(self):
        self.check_case("unavailable_capacity")

    def test_initial_cpu_blackout_has_exact_hand_derived_delay(self):
        self.check_case("cpu_service_blackout")

    def test_same_size_wrong_attention_and_ffn_operands_fail_independent_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            with reference.split_plan(directory) as compiled:
                result = compiled.run()
                self.assertTrue(result.success)
                operations, _, _ = reference.observed(result)
                reference.check_expected_wiring(operations, reference.attention_wiring())
                reference.check_expected_wiring(operations, reference.ffn_wiring())
                up = next(op for op in operations.values() if op.kind == "ffn_up" and "host.up" in op.id)
                wrong = dict(operations)
                wrong[up.id] = replace(up, inputs=("w:layer0:W_gate:ffn:host", up.inputs[1]))
                with self.assertRaisesRegex(AssertionError, "exact operand identities"):
                    reference.check_expected_wiring(wrong, reference.ffn_wiring())
                qk = next(op for op in operations.values() if op.kind == "attention_qk")
                wrong = dict(operations)
                wrong[qk.id] = replace(qk, inputs=(qk.inputs[0], reference.tensor_id("proj.v")))
                with self.assertRaisesRegex(AssertionError, "exact operand identities"):
                    reference.check_expected_wiring(wrong, reference.attention_wiring())
                output = next(op for op in operations.values() if op.kind == "attention_output")
                wrong = dict(operations)
                wrong[output.id] = replace(output, inputs=(*output.inputs[:-1], reference.tensor_id("attn.context")))
                with self.assertRaisesRegex(AssertionError, "exact operand identities"):
                    reference.check_expected_wiring(wrong, reference.attention_wiring())

    def test_imported_modules_must_resolve_to_the_expected_checkout(self):
        injected = ModuleType("tensor_mapping.foreign_copy")
        injected.__file__ = str(Path(tempfile.gettempdir()) / "foreign-tensor-mapping.py")
        with patch.dict(sys.modules, {injected.__name__: injected}):
            with self.assertRaisesRegex(ValueError, "not the expected checkout"):
                reference.imported_sources()

    def test_source_drift_and_unbound_import_hash_rejected(self):
        before = reference.source_snapshot()
        changed = dict(before)
        changed["experiments/strategy_fidelity.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source bytes changed"):
            reference.verify_source_integrity(before, changed, [])
        modules = reference.imported_sources()
        self.assertTrue(modules)
        modules[0] = dict(modules[0], sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "not bound to frozen"):
            reference.verify_source_integrity(before, before, modules)

    def test_input_lifetime_lock_applies_to_the_executing_storage_domain(self):
        operation = SimpleNamespace(id="op", inputs=("input",), outputs=("output",))
        computation = SimpleNamespace(kind="COMPUTE", target_id="op", device="gpu", start_ns=0, end_ns=10)
        eviction = SimpleNamespace(kind="EVICT", target_id="input", location="dram", start_ns=5)
        session = SimpleNamespace(observe=lambda: SimpleNamespace(operations=(operation,), tensors=()))
        result = SimpleNamespace(raw=(None, session), events=[computation, eviction])
        reference.check_dependency_lifetimes(result)  # Dropping an unused CPU backup is legal.
        eviction.location = "vram"
        with self.assertRaisesRegex(AssertionError, "in-flight compute input"):
            reference.check_dependency_lifetimes(result)


if __name__ == "__main__":
    unittest.main()
