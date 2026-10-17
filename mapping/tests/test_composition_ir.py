"""Inline mechanism composition, honest accounting and deterministic replay."""
from collections import Counter
from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import tempfile
import unittest

from llm_infer_model.tensor.spec import COPY_ABSENT, OP_DONE, SpecError
from tensor_mapping.composition_observation import CompositionSession
from tensor_mapping.composition_policy import CompositionPolicy, composition_policy_document
from tensor_mapping.unified_strategy import (compile_unified_strategy_file, create_session,
    run_unified_strategy, checkpoint_document, _runtime_signature)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "mapping/examples/composition"


class ComposedEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiled = {name: compile_unified_strategy_file(EXAMPLES / f"{name}.strategy.json")
                        for name in ("dense", "moe")}
        cls.results = {name: run_unified_strategy(model) for name, model in cls.compiled.items()}

    def test_decomposition_and_four_rules_share_one_runtime(self):
        for name, model in self.compiled.items():
            run, session, policy, _ = self.results[name]
            self.assertEqual(run.evaluation.status, "valid", run.compilation_error)
            self.assertEqual(run.termination_reason, "goal_reached")
            self.assertIsNotNone(model.composition)
            self.assertEqual(len(session.observe().completed_invocations), 4)
            self.assertFalse(session.observe().running)
            self.assertEqual(policy.audit_document()["composition"], dict(model.document["policy"],
                cache=model.document["cache"], prefetch=model.document["prefetch"], transport=model.document["transport"]))
            self.assertEqual(run.runtime_result.action_sources, tuple(range(len(run.decisions))))
        sources = {Path(row["path"]).name for row in self.compiled["moe"].sources}
        self.assertNotIn("fragmentL0.execution.json", sources)
        self.assertNotIn("fragmentL1.execution.json", sources)

    def test_pure_attention_explicit_binding_changes_by_phase(self):
        run, session, _, _ = self.results["dense"]
        operations = {op.id: op for op in session.observe().operations}
        cores = [record for record in run.decisions if record.action.kind == "COMPUTE" and
                 operations[record.action.target_id].kind == "attention_core"]
        self.assertEqual([record.action.device for record in cores], ["gpu", "gpu", "cpu", "cpu"])
        self.assertTrue(all(record.observation.operations for record in cores))
        self.assertTrue(all(op.status == OP_DONE for op in operations.values()))
        self.assertEqual(len(session.persistent_state_document()["kv_ids"]), 8)

    def test_staged_regions_choose_once_and_actual_engine_override(self):
        run, session, policy, _ = self.results["moe"]
        operations = {op.id: op for op in session.observe().operations}
        by_region = {}
        for record in run.decisions:
            if record.action.kind != "COMPUTE":
                continue
            op = operations[record.action.target_id]
            if op.kind in ("expert_gate", "expert_up", "expert_swiglu", "expert_down"):
                by_region.setdefault((op.invocation_id, op.region_id), []).append((op.kind, record.action.device))
        self.assertTrue(by_region)
        for rows in by_region.values():
            self.assertEqual(Counter(kind for kind, _ in rows),
                             Counter({"expert_gate": 1, "expert_up": 1, "expert_swiglu": 1, "expert_down": 1}))
            self.assertEqual(len({device for _, device in rows}), 1)
        engines = session.service_document()["transfer_engines"]
        down = [row for row in engines if row["tensor_id"].endswith(":down")]
        self.assertTrue(down)
        self.assertTrue(all(row["engine"] == "sm" for row in down))
        self.assertTrue(any(row["engine"] == "ce" for row in engines))
        self.assertFalse(policy.audit_document()["claims"]["search"])

    def test_group_budget_accounts_partial_and_in_flight_children(self):
        run, _, _, _ = self.results["moe"]
        budget = self.compiled["moe"].composition.cache_budget_bytes
        for record in run.decisions:
            observation = record.observation
            tensors = {tensor.id: tensor for tensor in observation.tensors}
            groups = {}
            for tid, layer, _, expert in observation.weight_catalogue:
                key = (layer, expert) if expert is not None else (tid,)
                groups.setdefault(key, []).append(tid)
            used = sum(sum(tensors[tid].storage_bytes for tid in members)
                for members in groups.values()
                if any(tensors[tid].status_at("vram") != COPY_ABSENT for tid in members))
            self.assertLessEqual(used, budget)

    def test_replay_preserves_in_flight_inventory_and_policy(self):
        for name, model in self.compiled.items():
            partial = run_unified_strategy(model, max_decisions=1)
            self.assertTrue(partial[1].observe().running)
            resumed = run_unified_strategy(model, checkpoint=checkpoint_document(model, partial))
            full = self.results[name]
            self.assertEqual(_runtime_signature(resumed[1]), _runtime_signature(full[1]))
            self.assertEqual(resumed[0].runtime_result.evaluation.actions, full[0].runtime_result.evaluation.actions)
            self.assertEqual(resumed[2].audit_document(), full[2].audit_document())
            self.assertEqual(resumed[2].history_document(), full[2].history_document())


class ComposedInputTests(unittest.TestCase):
    def test_unknown_or_malformed_choices_are_not_ignored(self):
        original = json.loads((EXAMPLES / "dense.strategy.json").read_text(encoding="utf-8"))
        # Preserve relative paths by placing the disposable manifests beside
        # the inputs, and remove them with TemporaryDirectory afterwards.
        with tempfile.TemporaryDirectory(dir=EXAMPLES) as directory:
            folder = Path(directory)
            for mutate in (
                lambda doc: doc.update(unknown_mapping=True),
                lambda doc: doc["execution"][0].update(scope_id="missing"),
                lambda doc: doc["execution"][0]["spec"].update(invocation_files=["wrong-source.json"]),
                lambda doc: doc["cache"].update(budget_bytes=-1),
                lambda doc: doc["model"].update(kind="unknown"),
            ):
                document = deepcopy(original)
                for key in ("services_file", "environment_file"):
                    document[key] = str((EXAMPLES / document[key]).resolve())
                document["model"]["file"] = str((EXAMPLES / document["model"]["file"]).resolve())
                for row in document["execution"]:
                    row["spec"]["invocation_files"] = [str((EXAMPLES / value).resolve())
                                                         for value in row["spec"]["invocation_files"]]
                mutate(document)
                path = folder / "strategy.json"
                path.write_text(json.dumps(document) + "\n", encoding="utf-8", newline="\n")
                with self.subTest(mutate=mutate), self.assertRaises(SpecError):
                    compile_unified_strategy_file(path)

    def test_public_context_frozen_and_committed_tail_not_free(self):
        model = compile_unified_strategy_file(EXAMPLES / "dense.strategy.json")
        session = create_session(model)
        observation = session.observe()
        with self.assertRaises(FrozenInstanceError):
            observation.layer_order = ()
        # Isolate the envelope's completion contract: no private task or route
        # must be opened while previously committed work is drained.
        from dataclasses import replace
        class Adapter:
            def observe(self):
                return replace(observation, complete=True, running=("committed_transfer",))
        session._session = Adapter()
        drained = session.observe()
        self.assertFalse(drained.complete)
        self.assertEqual(drained.active_invocations, ())
        self.assertEqual(drained.running, ("committed_transfer",))


if __name__ == "__main__":
    unittest.main()
