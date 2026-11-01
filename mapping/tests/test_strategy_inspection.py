"""Price absence must not become either illegal semantics or executable zeros."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tensor_mapping.composition_ir import compile_composition_document
from tensor_mapping.strategy_inspection import inspect_strategy_document


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "modeling/tests/fixtures/global_execution"


def document(directory):
    layer = json.loads((FIXTURE / "denseL0.layer.json").read_text(encoding="utf-8"))
    invocation = json.loads((FIXTURE / "denseL0.prefill.json").read_text(encoding="utf-8"))
    invocation["costs_file"] = "not-measured.costs.json"
    for name, data in (("denseL0.layer.json", layer), ("invocation.json", invocation)):
        (directory / name).write_text(json.dumps(data), encoding="utf-8", newline="\n")
    return dict(format_version="unified-strategy-0.2", id="unpriced-plan",
        model=dict(kind="m2", file="invocation.json"), execution=[],
        policy=dict(placements=dict(projection="gpu", attention="gpu", dense_ffn="gpu"),
                    request_placements={}, experts=dict(device_rule="fixed", selection_unit="expert_region"),
                    release=dict(rule="none")),
        cache=dict(rule="none", unit="tensor", budget_bytes=None, alpha=None, top_p=None),
        prefetch=dict(enabled=False, scope="none", lookahead_layers=0, max_count=0,
                      ranking="history_frequency", include_dense_weights=False),
        transport=dict(rules=[dict(role="weight", stage=None, src="dram", dst="vram", engine=None)],
                       slot_reuse="disabled", boundary_release="disabled"),
        services_file=None, environment_file=str(ROOT / "mapping/examples/causal/no-disturbance.environment.json"),
        history_file=None, routing_scores_file=None, max_decisions=1000)


def binding(device, *, selector=None, priority=0):
    return dict(selector=dict(dict(layer_id=None, invocation_id=None, request_id=None, kind="dense_ffn", phase=None),
                              **(selector or {})), device=device, priority=priority, module_path=[])


class StrategyInspectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.strategy = document(self.directory)

    def inspect(self, **options):
        return inspect_strategy_document(self.strategy, base_dir=self.directory, **options)

    def test_unpriced_declarations_and_bindings_checked_without_compiler_or_runtime(self):
        with patch("llm_infer_model.m2.expand.load_and_compile", side_effect=AssertionError("strict compiler called")):
            report = self.inspect()
        self.assertEqual(report.declarations.status, "valid")
        self.assertEqual(report.static_bindings.status, "valid")
        self.assertEqual(report.cost_coverage.status, "missing")
        self.assertEqual(report.lowering.status, "not_checked")
        self.assertEqual(report.runtime_feasibility.status, "not_run")
        self.assertEqual(report.semantic_validity, "unknown")
        self.assertFalse(report.executable)
        self.assertFalse(hasattr(report, "run"))
        self.assertEqual(len(report.operations), 3)
        self.assertEqual(len(report.cost_inputs), 1)
        self.assertEqual(report.cost_inputs[0]["status"], "missing")
        self.assertNotIn("duration_ns", json.dumps(report.document()))
        with self.assertRaises(FileNotFoundError):
            compile_composition_document(self.strategy, base_dir=self.directory)

    def test_readable_explanation_does_not_turn_declaration_pass_into_semantic_pass(self):
        report = self.inspect()
        text = report.render_text()
        self.assertEqual(text, report.explain())
        self.assertIn("Semantic validity: unknown", text)
        self.assertIn("Declarations: valid", text)
        self.assertIn("Cost coverage: missing", text)
        self.assertIn("Lowering: not checked", text)
        self.assertIn("Executable: no", text)
        self.assertEqual(report.document()["semantic_validity"], "unknown")
        self.assertIs(report.document()["executable"], False)

    def test_absent_execution_scope_is_reported_as_invalid_semantics(self):
        self.strategy["execution"] = [dict(scope_id="absent", spec={})]
        report = self.inspect()
        self.assertEqual(report.semantic_validity, "invalid")
        self.assertIn("Semantic validity: invalid", report.render_text())

    def test_scoped_transport_is_deferred_without_claiming_it_was_checked(self):
        report = self.inspect(scoped_transport=(dict(unexamined_rule="extension"),))
        self.assertEqual(report.declarations.status, "valid")
        self.assertEqual(report.static_bindings.status, "valid")
        self.assertEqual(report.semantic_validity, "unknown")
        self.assertTrue(any("scoped transport" in issue.message for issue in report.lowering.issues))
        self.assertFalse(any("transport" in item for item in report.static_bindings.checked))

    def test_block_inspection_does_not_claim_unperformed_layer_checks(self):
        (self.directory / "block.json").write_text("{}", encoding="utf-8", newline="\n")
        self.strategy["model"] = dict(kind="block", file="block.json", override_model_policy=True)
        report = self.inspect(visibility_mode="scope_barriers")
        self.assertEqual(report.semantic_validity, "unknown")
        self.assertEqual(report.static_bindings.status, "unknown")
        self.assertFalse(any("layer dependency" in item for item in report.declarations.checked))
        self.assertTrue(any("block schema remains unchecked" in item for item in report.declarations.checked))

    def test_inline_transform_does_not_read_replaced_inherited_execution_source(self):
        model = dict(format_version="global-execution-0.1", id="global", rounds=[
            dict(id="initial", input_mode="initial_supplied_hidden", gate_cost=None,
                 invocations=[dict(id="scope", invocation_file="invocation.json",
                                   execution_file="obsolete.execution.json")])])
        (self.directory / "global.json").write_text(json.dumps(model), encoding="utf-8", newline="\n")
        self.strategy["model"] = dict(kind="global_m2", file="global.json")
        transform = json.loads((ROOT / "modeling/tests/fixtures/ffn_execution/dense.execution.json").read_text(encoding="utf-8"))
        self.strategy["execution"] = [dict(scope_id="scope", spec=transform)]
        report = self.inspect()
        self.assertEqual(report.declarations.status, "valid")
        self.assertEqual(report.semantic_validity, "unknown")
        self.assertFalse(any("obsolete.execution" in row["path"] for row in report.sources))

    def test_presence_is_not_coverage_and_cost_files_are_not_opened(self):
        (self.directory / "not-measured.costs.json").write_bytes(b"not even JSON\xff")
        report = self.inspect()
        self.assertEqual(report.declarations.status, "valid")
        self.assertEqual(report.cost_coverage.status, "unknown")
        self.assertEqual(report.cost_inputs[0]["status"], "present_unverified")
        self.assertFalse(any(row["path"].endswith("not-measured.costs.json") for row in report.sources))

    def test_broken_layer_dependency_is_invalid_even_without_prices(self):
        p = self.directory / "denseL0.layer.json"
        layer = json.loads(p.read_text(encoding="utf-8"))
        layer["tasks"][2]["depends_on"] = ["unknown-producer"]
        p.write_text(json.dumps(layer), encoding="utf-8", newline="\n")
        report = self.inspect()
        self.assertEqual(report.declarations.status, "invalid")
        self.assertEqual(report.declarations.issues[0].code, "invalid_declaration")
        self.assertFalse(report.executable)

    def test_ambiguous_scoped_devices_are_rejected_without_cost_lookup(self):
        report = self.inspect(placements=(binding("cpu"), binding("gpu")))
        self.assertEqual(report.declarations.status, "invalid")
        self.assertIn("ambiguous", report.declarations.issues[0].message)
        self.assertEqual(report.cost_coverage.status, "missing")

    def test_priority_resolves_ambiguity_without_claiming_cost_availability(self):
        report = self.inspect(placements=(binding("cpu"), binding("gpu", priority=1)))
        self.assertEqual(report.declarations.status, "valid")
        self.assertEqual(report.static_bindings.status, "valid")
        self.assertEqual(report.cost_coverage.status, "missing")

    def test_absent_selector_is_rejected_without_prices(self):
        row = binding("cpu")
        row["selector"]["layer_id"] = "absent-layer"
        report = self.inspect(placements=(row,))
        self.assertEqual(report.declarations.status, "invalid")
        self.assertIn("matches no", report.declarations.issues[0].message)

    def test_transform_declarations_do_not_prove_lowered_ports(self):
        transform = json.loads((ROOT / "modeling/tests/fixtures/ffn_execution/dense.execution.json").read_text(encoding="utf-8"))
        # Unusable inline prices must not poison a separate declaration check.
        transform["entries"] = [dict(duration_ns="unmeasured")]
        transform["transfer_entries"] = []
        self.strategy["execution"] = [dict(scope_id="single", spec=transform)]
        report = self.inspect()
        self.assertEqual(report.declarations.status, "valid")
        self.assertEqual(report.static_bindings.status, "unknown")
        self.assertEqual(report.lowering.status, "not_checked")
        self.assertTrue(report.lowering.issues)
        self.assertFalse(report.executable)

    def test_invalid_partition_is_rejected_independent_of_inline_prices(self):
        transform = json.loads((ROOT / "modeling/tests/fixtures/ffn_execution/dense.execution.json").read_text(encoding="utf-8"))
        transform["entries"] = []
        transform["shards"][0]["start"] = 1
        self.strategy["execution"] = [dict(scope_id="single", spec=transform)]
        report = self.inspect()
        self.assertEqual(report.declarations.status, "invalid")
        self.assertIn("partition", report.declarations.issues[0].message)

    def test_missing_model_is_unknown_not_valid_or_a_cost_failure(self):
        self.strategy["model"]["file"] = "not-provided.invocation.json"
        report = self.inspect()
        self.assertEqual(report.declarations.status, "unknown")
        self.assertEqual(report.declarations.issues[0].code, "missing_model_input")
        self.assertEqual(report.cost_coverage.status, "unknown")

    def test_inspection_is_read_only_and_records_actual_model_bytes(self):
        before = {p.name: p.read_bytes() for p in self.directory.iterdir()}
        report = self.inspect()
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.directory.iterdir()})
        self.assertEqual(len(report.sources), 2)
        self.assertTrue(all(len(row["sha256"]) == 64 for row in report.sources))
        self.assertFalse(any("not-measured" in row["path"] for row in report.sources))


if __name__ == "__main__":
    unittest.main()
