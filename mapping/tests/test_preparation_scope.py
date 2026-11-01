"""Static selector intersections preserve shared physical weight identities."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
import unittest

from llm_infer_model.tensor.spec import InvalidInput, Unsupported
from tensor_mapping.phase_scheduling import DataPreparation


class PreparationScopeTests(unittest.TestCase):
    def test_module_layer_intersects_existing_stage_and_expert(self):
        rule = DataPreparation(stage="gate", expert_id=1)
        bound = rule.with_scope(layer_id="L1")
        observation = SimpleNamespace(weight_catalogue=(
            ("w:L0:E1:gate", "L0", "gate", 1),
            ("w:L1:E0:gate", "L1", "gate", 0),
            ("w:L1:E1:up", "L1", "up", 1),
            ("w:L1:E1:gate", "L1", "gate", 1),
        ))
        self.assertEqual(bound.matching_tensor_ids(observation), ("w:L1:E1:gate",))
        self.assertIsNone(rule.layer_id)
        self.assertEqual(bound.with_scope(layer_id="L1"), bound)
        with self.assertRaises(FrozenInstanceError):
            bound.layer_id = "L0"

    def test_explicit_tensor_sets_intersect_with_original_physical_identities(self):
        rule = DataPreparation(tensor_ids=("w:z", "w:b", "w:a"))
        bound = rule.with_scope(tensor_ids=("w:x", "w:a", "w:b"))
        observation = SimpleNamespace(weight_catalogue=(
            ("w:a", "L0", "gate", 0), ("w:b", "L0", "up", 0),
            ("w:z", "L1", "gate", 1), ("w:x", "L2", "gate", 2),
        ))
        self.assertEqual(bound.tensor_ids, ("w:a", "w:b"))
        self.assertEqual(bound.matching_tensor_ids(observation), ("w:a", "w:b"))
        # Binding labels neither prefixes weight IDs nor constructs new copies.
        other = DataPreparation(tensor_ids=("w:b",)).with_scope(layer_id="L0")
        self.assertEqual(other.matching_tensor_ids(observation), ("w:b",))
        self.assertEqual(set(bound.matching_tensor_ids(observation)) |
                         set(other.matching_tensor_ids(observation)), {"w:a", "w:b"})
        self.assertEqual(rule.tensor_ids, ("w:z", "w:b", "w:a"))

    def test_unrestricted_scope_is_a_semantic_noop(self):
        rule = DataPreparation(layer_id="L0", stage="gate", expert_id="shared",
                               tensor_ids=("w:b", "w:a"), trigger="startup",
                               destination="dram", max_count=2)
        self.assertEqual(rule.with_scope(), rule)
        self.assertEqual(rule.with_scope(layer_id=None, stage=None,
                                         tensor_ids=(), expert_id=None), rule)

    def test_binding_preserves_destination_trigger_and_selection_limit(self):
        rule = DataPreparation(stage="gate", destination="dram", trigger="startup", max_count=1)
        bound = rule.with_scope(layer_id="L0", tensor_ids=("w:b", "w:a"))
        self.assertEqual((bound.destination, bound.trigger, bound.max_count), ("dram", "startup", 1))
        observation = SimpleNamespace(weight_catalogue=(
            ("w:b", "L0", "gate", None), ("w:a", "L0", "gate", None),
        ))
        self.assertEqual(bound.matching_tensor_ids(observation), ("w:a",))
        self.assertEqual(DataPreparation.from_document(bound.document()), bound)

    def test_contradictory_scalar_or_tensor_scope_rejects_without_mutation(self):
        rule = DataPreparation(layer_id="L0", stage="gate", expert_id=0, tensor_ids=("w:a",))
        before = rule.document()
        for field, value in (("layer_id", "L1"), ("stage", "up"),
                             ("expert_id", 1), ("expert_id", "shared"),
                             ("tensor_ids", ("w:b",))):
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(InvalidInput, field):
                    rule.with_scope(**{field: value})
                self.assertEqual(rule.document(), before)

    def test_unrepresentable_scope_fields_never_silently_disappear(self):
        rule = DataPreparation(layer_id="L0")
        for field, value in (("phase", "decode"), ("request_id", "r0"),
                             ("invocation_id", "L0-decode"), ("kind", "ffn_gate"),
                             ("destination", "dram"), ("unknown", "value")):
            for supplied in (value, None):
                with self.subTest(field=field, supplied=supplied):
                    with self.assertRaisesRegex(Unsupported, field):
                        rule.with_scope(**{field: supplied})

    def test_scope_values_use_same_strict_types_as_declarations(self):
        rule = DataPreparation(expert_id=0)
        for scope in ({"expert_id": False}, {"expert_id": -1},
                      {"expert_id": "0"}, {"layer_id": ""},
                      {"layer_id": "bad\nname"}, {"stage": ""},
                      {"tensor_ids": ["w:a"]}, {"tensor_ids": ("w:a", "w:a")}):
            with self.subTest(scope=scope):
                with self.assertRaises(InvalidInput):
                    rule.with_scope(**scope)

    def test_cross_label_conflict_requires_public_catalogue_resolution(self):
        # A tensor ID is an identity, not a string grammar for guessing its layer.
        bound = DataPreparation(tensor_ids=("arbitrary-weight-name",)).with_scope(layer_id="L0")
        observation = SimpleNamespace(weight_catalogue=(
            ("arbitrary-weight-name", "L1", "gate", None),
        ))
        with self.assertRaisesRegex(InvalidInput, "contradicts"):
            bound.matching_tensor_ids(observation)

    def test_scope_resolution_does_not_observe_future_routes(self):
        class StaticCatalogueOnly:
            weight_catalogue = (("w:E7", "L0", "gate", 7),)

            @property
            def route_history(self):
                raise AssertionError("static scope must not consult router outcomes")

        selected = DataPreparation(expert_id=7).with_scope(layer_id="L0", stage="gate")
        self.assertEqual(selected.matching_tensor_ids(StaticCatalogueOnly()), ("w:E7",))

    def test_overlapping_scopes_share_one_charged_physical_copy(self):
        from .test_phase_scheduling import dense_plan

        target = "w:denseL0:W_down"
        # The fixture's F32 down matrix is [32,16]: 32 * 16 * 4 = 2048 B.
        # All computation stays on CPU, so this is the only VRAM allocation.
        plan = dense_plan(device="cpu", cache_budget=2048)
        plan.prepare(DataPreparation(tensor_ids=(target,), trigger="startup").with_scope(layer_id="denseL0"))
        plan.prepare(DataPreparation(layer_id="denseL0", trigger="startup").with_scope(tensor_ids=(target,)))
        run, _, policy, _ = plan.compile().run()
        self.assertEqual(run.termination_reason, "goal_reached", run.compilation_error)
        copies = [event for event in run.evaluation.events
                  if event.kind == "COPY" and event.target_id == target]
        self.assertEqual(len(copies), 1)
        self.assertEqual(copies[0].bytes_transferred, 2048)
        self.assertGreater(copies[0].end_ns, copies[0].start_ns)
        self.assertEqual(run.evaluation.h2d_bytes, 2048)
        self.assertEqual(run.evaluation.peak_vram_bytes, 2048)
        self.assertEqual(len(policy.audit_document()["explicit_data_preparations"]), 1)


if __name__ == "__main__":
    unittest.main()
