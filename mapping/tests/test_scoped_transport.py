"""Scoped engines use physical weight ownership and one shared controller."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import unittest

from llm_infer_model.tensor.spec import COPY_ABSENT, InvalidInput, Unsupported
from tensor_mapping.composition_ir import compile_composition_document
from tensor_mapping.composition_policy import CompositionPolicy, parse_composition_policy
from tensor_mapping.storage_rules import (
    TensorSelect, ScopedTransportRule, normalize_scoped_transport,
    resolve_scoped_transport, validate_scoped_transport_bindings,
)
from tensor_mapping.unified_strategy import (
    checkpoint_document, run_unified_strategy, strategy_fingerprint,
)
from .test_composition_policy import inline_policy, region_observation, with_services


STRATEGY = Path(__file__).resolve().parents[1] / "examples/composition/block.strategy.json"


def rule(engine="sm", *, priority=0, module_path=(), **selection):
    return ScopedTransportRule("weight", "dram", "vram", engine,
                               TensorSelect(**selection), priority, module_path)


def compile_block(rules=(), *, budget=96):
    document = json.loads(STRATEGY.read_text(encoding="utf-8"))
    document["cache"]["budget_bytes"] = budget
    return compile_composition_document(document, base_dir=STRATEGY.parent,
                                        scoped_transport=rules)


class ScopedTransportContractTests(unittest.TestCase):
    def test_typed_contract_roundtrip_and_intersection(self):
        selected = TensorSelect(layer_id="L1").intersect(TensorSelect(stage="gate", expert_id=0))
        original = ScopedTransportRule("weight", "dram", "vram", "sm", selected, 3, ("decoder", "layer1"))
        self.assertEqual(ScopedTransportRule.from_document(original.document()), original)
        self.assertEqual(normalize_scoped_transport([original]), (original,))
        self.assertTrue(selected.specializes(TensorSelect(layer_id="L1")))
        self.assertFalse(TensorSelect(layer_id="L1").specializes(selected))
        with self.assertRaisesRegex(InvalidInput, "contradictory"):
            selected.intersect(TensorSelect(layer_id="L0"))

    def test_only_static_weight_ownership_is_accepted(self):
        for role in ("input", "intermediate", "output", "kv"):
            with self.subTest(role=role), self.assertRaisesRegex(Unsupported, "static weight ownership"):
                ScopedTransportRule(role, "dram", "vram", "ce")
        for invalid in (True, -1, "future", 1.0):
            with self.subTest(expert=invalid), self.assertRaises(InvalidInput):
                TensorSelect(expert_id=invalid)
        serialized = rule(layer_id="L0").document()
        serialized["selector"]["phase"] = "decode"
        with self.assertRaisesRegex(InvalidInput, "exactly"):
            ScopedTransportRule.from_document(serialized)
        with self.assertRaises(InvalidInput):
            normalize_scoped_transport([serialized])

    def test_priority_then_true_specialization_then_global_fallback(self):
        broad = rule("ce", layer_id="L1")
        specific = rule("sm", layer_id="L1", expert_id=0)
        context = dict(tensor_id="wg", role="weight", src="dram", dst="vram",
                       layer_id="L1", stage="gate", expert_id=0)
        self.assertEqual(resolve_scoped_transport((broad, specific), **context), specific)
        self.assertEqual(resolve_scoped_transport((specific, broad), **context), specific)
        explicit = replace(broad, priority=1)
        self.assertEqual(resolve_scoped_transport((specific, explicit), **context), explicit)
        self.assertIsNone(resolve_scoped_transport((broad,), **(context | {"layer_id": "L0"})))
        self.assertIsNone(resolve_scoped_transport((broad,), **(context | {"role": "input"})))

    def test_incomparable_equal_priority_conflicts_are_rejected(self):
        # Three labels are not a reason to silently defeat an independent layer rule.
        rules = (rule("ce", layer_id="L1"), rule("sm", stage="gate", expert_id=0, tensor_id="wg"))
        context = dict(tensor_id="wg", role="weight", src="dram", dst="vram",
                       layer_id="L1", stage="gate", expert_id=0)
        for ordered in (rules, tuple(reversed(rules))):
            with self.subTest(order=ordered), self.assertRaisesRegex(InvalidInput, "ambiguous"):
                resolve_scoped_transport(ordered, **context)
        equivalent = tuple(replace(item, engine="ce") for item in rules)
        self.assertEqual(resolve_scoped_transport(equivalent, **context).engine, "ce")

    def test_explicit_none_is_a_rule_and_not_a_missing_rule(self):
        selected = rule(None, layer_id="L0")
        result = resolve_scoped_transport((selected,), tensor_id="wg", role="weight",
            src="dram", dst="vram", layer_id="L0", stage="gate", expert_id=0)
        self.assertIs(result, selected)
        self.assertIsNone(result.engine)

    def test_unmatched_selector_is_rejected_without_exposing_future_routes(self):
        catalogue = (("wg", "L0", "gate", 0), ("wu", "L0", "up", 1))
        validate_scoped_transport_bindings((rule(expert_id=1),), catalogue)
        with self.assertRaisesRegex(InvalidInput, "matches no public model weight"):
            validate_scoped_transport_bindings((rule(layer_id="missing"),), catalogue)


class ScopedTransportPolicyTests(unittest.TestCase):
    def test_estimation_and_copy_use_the_identical_scoped_engine(self):
        document = inline_policy(services=True)
        document["placements"].update(expert_gate="gpu", expert_up="gpu", expert_down="gpu")
        selected = rule(stage="gate", layer_id="L0", expert_id=0)
        policy = CompositionPolicy(replace(parse_composition_policy(document), scoped_transport=(selected,)))
        observed = with_services(region_observation(future=True))
        estimated = policy._estimate_observation(observed)
        prices = {t.id: {p.route: p.duration_ns for p in t.transfers} for t in estimated.tensors}
        self.assertEqual(prices["wg"]["dram_to_vram"], 64)
        self.assertEqual(prices["wg"]["vram_to_dram"], 32)
        self.assertEqual(prices["wu"]["dram_to_vram"], 32)
        self.assertEqual(prices["L1wg"]["dram_to_vram"], 32)
        decision = policy(observed)
        self.assertEqual((decision.kind, decision.target_id, decision.transfer_engine), ("COPY", "wg", "sm"))
        audit = policy.audit_document()
        self.assertEqual(audit["scoped_transport"], [selected.document()])
        self.assertEqual(audit["decisions"][-1]["evidence"]["scoped_transport_rule"], selected.document())

    def test_scope_does_not_change_the_shared_budget_or_global_nonweight_rules(self):
        spec = parse_composition_policy(inline_policy(cache="lru", unit="tensor", budget=64, services=True))
        selected = rule(layer_id="L1")
        policy = CompositionPolicy(replace(spec, scoped_transport=(selected,)))
        observed = with_services(region_observation(future=True))
        policy._estimate_observation(observed)
        tensors = {item.id: item for item in observed.tensors}
        self.assertEqual(policy._budget_for_additions(tensors, ("wg", "L1wg")), 64)
        self.assertEqual(policy._budget_for_additions(tensors, ("wg", "wu", "L1wg")), 96)
        self.assertEqual(policy._engine(tensors["compact"], "dram", "vram"), "ce")
        self.assertEqual(policy.spec.cache_budget_bytes, 64)
        self.assertEqual(policy.spec.scoped_transport, (selected,))


class ScopedTransportIntegrationTests(unittest.TestCase):
    def test_two_layers_dispatch_ce_and_sm_with_one_96_byte_weight_budget(self):
        selected = rule(layer_id="L1", module_path=("decoder", "layer1"))
        compiled = compile_block((selected,))
        run, session, policy, _ = run_unified_strategy(compiled)
        self.assertEqual(run.evaluation.status, "valid", (run.termination_reason, run.compilation_error))
        copies = [row.decision for row in run.decisions if row.decision.kind == "COPY"
                  and row.decision.target_id in ("w:L0", "w:L1") and row.decision.dst == "vram"]
        self.assertEqual({(row.target_id, row.transfer_engine) for row in copies},
                         {("w:L0", "ce"), ("w:L1", "sm")})
        # Includes in-flight/reserved copies: scopes cannot reserve a budget each.
        observed = [row.observation for row in run.decisions] + [session.observe()]
        weights = [sum(t.storage_bytes for t in row.tensors if t.role == "weight"
                       and t.status_at("vram") != COPY_ABSENT) for row in observed]
        self.assertGreater(max(weights), 0)
        self.assertLessEqual(max(weights), 96)
        self.assertTrue(any(row.decision.kind == "EVICT" and row.decision.target_id == "w:L0"
                            for row in run.decisions))
        audit = policy.audit_document()
        self.assertEqual(audit["composition"]["cache"]["budget_bytes"], 96)
        matching = [row["evidence"]["scoped_transport_rule"] for row in audit["decisions"]
                    if "scoped_transport_rule" in row.get("evidence", {})]
        self.assertTrue(matching)
        self.assertEqual(matching, [selected.document()] * len(matching))
        self.assertFalse(audit["claims"]["paper_strategy_reproduced"])

    def test_compile_rejects_real_conflict_and_unknown_owners(self):
        with self.assertRaisesRegex(InvalidInput, "ambiguous"):
            compile_block((rule("ce", layer_id="L1"), rule("sm", stage="dense_ffn")))
        with self.assertRaisesRegex(InvalidInput, "matches no public model weight"):
            compile_block((rule(layer_id="absent"),))
        compiled = compile_block((rule("ce", layer_id="L1"), rule("sm", stage="dense_ffn", priority=1)))
        run, _, _, _ = run_unified_strategy(compiled)
        self.assertEqual(run.evaluation.status, "valid")

    def test_checkpoint_replay_keeps_scopes_and_rejects_changed_rule(self):
        compiled = compile_block((rule(layer_id="L1", module_path=("layer1",)),))
        partial = run_unified_strategy(compiled, max_decisions=25)
        checkpoint = checkpoint_document(compiled, partial)
        resumed = run_unified_strategy(compiled, checkpoint=deepcopy(checkpoint))
        full = run_unified_strategy(compiled)
        self.assertEqual(resumed[0].evaluation, full[0].evaluation)
        self.assertEqual(resumed[1].service_document(), full[1].service_document())
        self.assertEqual(resumed[2].audit_document(), full[2].audit_document())
        self.assertEqual(resumed[3], 25)
        changed = compile_block((rule("ce", layer_id="L1", module_path=("layer1",)),))
        self.assertNotEqual(strategy_fingerprint(changed), strategy_fingerprint(compiled))
        with self.assertRaisesRegex(InvalidInput, "sources or runtime changed"):
            run_unified_strategy(changed, checkpoint=checkpoint)


if __name__ == "__main__":
    unittest.main()
