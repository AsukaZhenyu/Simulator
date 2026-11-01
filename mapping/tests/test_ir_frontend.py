"""Public declaration inspection and honest module ownership contracts."""
import json
from pathlib import Path
import tempfile
import unittest

from llm_infer_model.tensor.spec import InvalidInput, Unsupported
from tensor_mapping.phase_scheduling import DataPreparation
from tensor_mapping.python_frontend import ModelPlan
from tensor_mapping.storage_rules import TensorSelect

from .test_strategy_inspection import document


ROOT = Path(__file__).resolve().parents[2]
BLOCK = ROOT / "mapping/examples/composition/block.strategy.json"


class FrontendInspectionTests(unittest.TestCase):
    def unpriced_plan(self, folder):
        path = Path(folder) / "unpriced.strategy.json"
        path.write_text(json.dumps(document(Path(folder))), encoding="utf-8", newline="\n")
        return ModelPlan.from_strategy(path)

    def test_missing_prices_remain_inspectable_but_cannot_execute(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = self.unpriced_plan(folder)
            with plan.module("ffn", kind="dense_ffn"):
                plan.place("dense_ffn", "cpu")
            report = plan.inspect()
            self.assertEqual(report.declarations.status, "valid")
            self.assertEqual(report.static_bindings.status, "valid")
            self.assertEqual(report.cost_coverage.status, "missing")
            self.assertFalse(report.executable)
            self.assertIn("Cost coverage: missing", plan.explain())
            self.assertIn("dense_ffn", plan.explain())
            with self.assertRaises(FileNotFoundError):
                plan.compile()

    def test_declaration_snapshot_is_nonexecutable_and_isolated(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = self.unpriced_plan(folder)
            snapshot = plan.declaration_document()
            self.assertIs(snapshot["executable"], False)
            snapshot["backend"]["policy"]["placements"]["attention"] = "cpu"
            self.assertEqual(plan.declaration_document()["backend"]["policy"]["placements"]["attention"], "gpu")
            json.dumps(plan.declaration_document(), allow_nan=False)

    def test_open_module_and_changed_import_are_not_silently_inspected(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = self.unpriced_plan(folder)
            with plan.module("unfinished"):
                with self.assertRaisesRegex(InvalidInput, "close all module"):
                    plan.inspect()
            path = Path(folder) / "unpriced.strategy.json"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(InvalidInput, "source changed"):
                plan.inspect()


class FrontendOwnershipTests(unittest.TestCase):
    def test_prepare_inherits_layer_and_matches_the_explicit_flat_plan(self):
        flat, scoped = ModelPlan.from_strategy(BLOCK), ModelPlan.from_strategy(BLOCK)
        flat.prepare(DataPreparation(layer_id="L0", destination="dram", trigger="startup"))
        with scoped.module("first_layer", layer_id="L0"):
            scoped.prepare(destination="dram", trigger="startup")
        a, b = flat.compile(), scoped.compile()
        self.assertEqual(a.ir_document()["data_preparations"], b.ir_document()["data_preparations"])
        ar, br = a.run(), b.run()
        self.assertEqual(br[0].termination_reason, "goal_reached", br[0].compilation_error)
        self.assertEqual(ar[0].evaluation.actions, br[0].evaluation.actions)
        self.assertEqual(ar[0].evaluation.events, br[0].evaluation.events)

    def test_prepare_rejects_conflicting_and_unimplemented_module_scopes(self):
        plan = ModelPlan.from_strategy(BLOCK)
        with plan.module("first", layer_id="L0"):
            with self.assertRaisesRegex(InvalidInput, "contradictory"):
                plan.prepare(DataPreparation(layer_id="L1"))
            with self.assertRaisesRegex(InvalidInput, "not both"):
                plan.prepare(DataPreparation(layer_id="L0"), trigger="startup")
        with plan.module("phase", phase="decode"):
            with self.assertRaisesRegex(Unsupported, "static weight ownership"):
                plan.prepare(DataPreparation(layer_id="L0"))

    def test_refinement_does_not_ignore_a_module_region_selector(self):
        plan = ModelPlan.from_strategy(BLOCK)
        for selector in ({"layer_id": "L0"}, {"kind": "dense_ffn"}, {"phase": "decode"}):
            with self.subTest(selector=selector), plan.module("scope" + next(iter(selector)), **selector):
                with self.assertRaisesRegex(Unsupported, "whole invocation"):
                    plan.refine("invocation", {})
                with self.assertRaisesRegex(Unsupported, "whole invocation"):
                    plan.transform("invocation", {})
                with self.assertRaisesRegex(Unsupported, "whole invocation"):
                    plan.transform_file("invocation", "not-read.json")
        with plan.module("invocation", invocation_id="first"):
            with self.assertRaisesRegex(InvalidInput, "contradicts"):
                plan.refine("different", {})

    def test_module_transfer_is_in_ir_and_drives_real_copy_actions(self):
        plan = ModelPlan.from_strategy(BLOCK)
        with plan.module("second_layer", layer_id="L1"):
            plan.transfer(engine="sm")
        declaration = plan.declaration_document()
        selected = declaration["scoped_transport"][0]
        self.assertEqual(selected["selector"]["layer_id"], "L1")
        self.assertEqual(selected["module_path"], ["second_layer"])
        self.assertEqual(plan.inspect().semantic_validity, "unknown")
        self.assertIn("engine=sm", plan.explain())
        compiled = plan.compile()
        self.assertEqual(compiled.ir_document()["scoped_transport"], [selected])
        run, _, policy, _ = compiled.run()
        self.assertEqual(run.termination_reason, "goal_reached", run.compilation_error)
        copies = {(row.decision.target_id, row.decision.transfer_engine)
                  for row in run.decisions if row.decision.kind == "COPY" and row.decision.dst == "vram"
                  and row.decision.target_id in {"w:L0", "w:L1"}}
        self.assertEqual(copies, {("w:L0", "ce"), ("w:L1", "sm")})
        self.assertEqual(policy.audit_document()["composition"]["cache"]["budget_bytes"], 128)

    def test_transport_does_not_guess_ownership_or_ignore_scope_conflicts(self):
        plan = ModelPlan.from_strategy(BLOCK)
        with plan.module("first", layer_id="L0"):
            with self.assertRaisesRegex(InvalidInput, "contradictory"):
                plan.transfer(engine="sm", select=TensorSelect(layer_id="L1"))
            with self.assertRaisesRegex(Unsupported, "weight ownership"):
                plan.transfer(engine="sm", role="intermediate")
        with plan.module("request", request_id="r0"):
            with self.assertRaisesRegex(Unsupported, "static weight ownership"):
                plan.transfer(engine="sm")

    def test_checkpoint_is_bound_to_the_frontend_transport_choices(self):
        plan = ModelPlan.from_strategy(BLOCK)
        with plan.module("second_layer", layer_id="L1"):
            plan.transfer(engine="sm")
        compiled = plan.compile()
        checkpoint = compiled.checkpoint(compiled.run(max_decisions=2))
        resumed = compiled.run(checkpoint=checkpoint)
        full = compiled.run()
        self.assertEqual(resumed[0].evaluation.events, full[0].evaluation.events)
        changed = ModelPlan.from_strategy(BLOCK)
        with changed.module("second_layer", layer_id="L1"):
            changed.transfer(engine="ce")
        with self.assertRaisesRegex(InvalidInput, "sources or runtime changed"):
            changed.compile().run(checkpoint=checkpoint)


if __name__ == "__main__":
    unittest.main()
