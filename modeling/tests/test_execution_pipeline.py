"""Ordered execution composition and model-owned runtime integration.

Cost declarations and source workloads are reused from their transform suites.
These tests exercise common parsing/lowering/visibility/state, not a new model.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from llm_infer_model.m2.attention_execution import parse_attention_execution
from llm_infer_model.m2.attention_stages import parse_attention_stages
from llm_infer_model.m2.causal import CausalM2Session
from llm_infer_model.m2.costs import TransferEntry
from llm_infer_model.m2.execution_pipeline import (
    ExecutionPipelineSpec, apply_execution_transform, execution_metadata,
    materialized_weight_members, parse_execution_pipeline, parse_transform,
    transform_document, weight_children,
)
from llm_infer_model.m2.expand import compile_invocation, load_and_compile
from llm_infer_model.m2.ffn_execution import parse_ffn_execution
from llm_infer_model.m2.global_execution import compile_global_execution_file, GlobalM2Session
from llm_infer_model.m2.moe import plan_moe
from llm_infer_model.m2.row_execution import parse_execution_spec
from llm_infer_model.tensor.engine import ACTION_ADVANCE, STATUS_VALID, evaluate_mapping
from llm_infer_model.tensor.spec import COPY_READY, InvalidInput, Unsupported

from test_attention_stages import (dense_source, dense_ffn_spec, document as attention_document,
                                  moe_source, row_spec)
from test_attention_execution import document as legacy_document
from test_ffn_execution import moe_source as row_materialized_source, document as ffn_document

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures"
SCRATCH = Path(__file__).parent / "_tmp_execution_pipeline"


def pipeline(*specs):
    return parse_execution_pipeline(dict(format_version="execution-pipeline-0.1",
                                         transforms=[transform_document(spec) for spec in specs]))


def execute(session, compiled):
    for action in plan_moe(compiled):
        result = session.advance() if action.kind == ACTION_ADVANCE else session.submit(action)
        if not result.ok:
            raise AssertionError(result.error_message)


def legacy_spec():
    raw = legacy_document()
    raw["invocation_files"] = ["source.json"]
    raw["subbatches"] = raw["subbatches"][:1]
    return parse_attention_execution(raw)


class ExecutionPipelineTests(unittest.TestCase):
    def test_parse_serialize_preserves_explicit_order(self):
        source = dense_source()
        attention = parse_attention_stages(attention_document(source))
        ffn = dense_ffn_spec(source)
        a, b = pipeline(attention, ffn), pipeline(ffn, attention)
        raw = transform_document(a)
        self.assertEqual(transform_document(parse_transform(raw)), raw)
        self.assertEqual([row["format_version"] for row in raw["transforms"]],
                         ["attention-stages-0.1", "ffn-execution-0.1"])
        self.assertNotEqual(transform_document(a), transform_document(b))
        self.assertEqual(a.variant, "execution_pipeline")

    def test_empty_nested_unknown_and_duplicate_passes_rejected(self):
        source = dense_source()
        spec = parse_attention_stages(attention_document(source))
        raw = transform_document(pipeline(spec))
        for transforms in ([], [raw], [transform_document(spec), transform_document(spec)]):
            with self.assertRaises(InvalidInput):
                parse_execution_pipeline(dict(format_version="execution-pipeline-0.1", transforms=transforms))
        with self.assertRaises(Unsupported):
            parse_transform(dict(format_version="not-an-execution-transform"))

    def test_malformed_transform_versions_fail_with_a_contract_error(self):
        for version in ([], {}, None, True, 0):
            with self.subTest(version=version):
                with self.assertRaises((InvalidInput, Unsupported)):
                    parse_transform(dict(format_version=version))

    def test_same_row_family_in_two_versions_is_rejected_before_lowering(self):
        staged = json.loads((FIXTURES / "subweights/execution.json").read_text(encoding="utf-8"))
        with self.assertRaises(InvalidInput):
            parse_execution_pipeline(dict(format_version="execution-pipeline-0.1",
                transforms=[transform_document(row_spec()), staged]))

    def test_legacy_pure_attention_only_at_start_and_moe_ffn_requires_row_materialization(self):
        source = dense_source()
        with self.assertRaises(Unsupported): pipeline(dense_ffn_spec(source), legacy_spec())
        materialized = row_materialized_source()
        expert_ffn = parse_ffn_execution(ffn_document(materialized))
        unmaterialized = load_and_compile(FIXTURES / "subweights/invocation.json")
        rows = parse_execution_spec(json.loads((FIXTURES / "subweights/execution.json").read_text(encoding="utf-8")))
        with self.assertRaisesRegex(Unsupported, "requires selected_rows"):
            apply_execution_transform(unmaterialized, pipeline(expert_ffn, rows))

    def test_dense_attention_ffn_orders_lower_to_same_graph_and_costs(self):
        source = dense_source(); snapshot = deepcopy(source.workload_document)
        attention = parse_attention_stages(attention_document(source)); ffn = dense_ffn_spec(source)
        a = apply_execution_transform(source, pipeline(attention, ffn))
        b = apply_execution_transform(source, pipeline(ffn, attention))
        for name in ("tensors", "operations"):
            self.assertEqual(sorted(a.workload_document[name], key=lambda row: row["id"]),
                             sorted(b.workload_document[name], key=lambda row: row["id"]))
        self.assertEqual(a.scenario_document["costs"]["compute"], b.scenario_document["costs"]["compute"])
        self.assertEqual(a.hidden_out_ids, source.hidden_out_ids)
        self.assertEqual(a.block_ids, source.block_ids)
        self.assertEqual(source.workload_document, snapshot)
        for result in (a, b):
            self.assertEqual(evaluate_mapping(result.scenario, plan_moe(result)).status, STATUS_VALID)

    def test_moe_attention_row_orders_preserve_static_causal_prefix_and_complete(self):
        source = moe_source()
        attention = parse_attention_stages(attention_document(source))
        results = [apply_execution_transform(source, spec) for spec in
                   (pipeline(attention, row_spec()), pipeline(row_spec(), attention))]
        for name in ("tensors", "operations"):
            self.assertEqual(sorted(results[0].workload_document[name], key=lambda row: row["id"]),
                             sorted(results[1].workload_document[name], key=lambda row: row["id"]))
        observations = []
        for compiled in results:
            session = CausalM2Session(compiled)
            obs = session.observe(); observations.append(obs)
            kinds = {op.kind for op in obs.operations}
            self.assertTrue({"attention_projection", "attention_qk", "attention_softmax", "attention_pv", "attention_output"} <= kinds)
            self.assertFalse({"gather_rows", "expert_gate", "scatter_rows"} & kinds)
            self.assertEqual(obs.route_history, ())
            execute(session, compiled)
            self.assertEqual(session.result().evaluation.status, STATUS_VALID)
        self.assertEqual(observations[0], observations[1])

    def test_merged_metadata_describes_actual_attention_and_ffn_tasks(self):
        source = dense_source()
        compiled = apply_execution_transform(source, pipeline(
            parse_attention_stages(attention_document(source)), dense_ffn_spec(source)))
        stages, rows = execution_metadata(compiled.physical_boundary)
        obs = CausalM2Session(compiled).observe()
        for task in compiled.tasks:
            if task.kind.startswith(("attention_", "ffn_")):
                self.assertEqual(stages[task.operation_id]["execution_stage"], task.kind)
                public = next(op for op in obs.operations if op.id == task.operation_id)
                self.assertEqual(public.execution_stage, task.kind)
                self.assertTrue(public.region_id)
        self.assertTrue(rows)

    def test_nested_moe_weight_catalogue_and_partial_inventory_contain_only_real_leaves(self):
        source = row_materialized_source()
        compiled = apply_execution_transform(source, pipeline(parse_ffn_execution(ffn_document(source))))
        session = CausalM2Session(compiled)
        available = {t.id for t in compiled.scenario.workload.tensors if t.role == "weight"}
        children = weight_children(compiled.physical_boundary)
        self.assertTrue(children)
        for expert in session.observe().experts:
            parent = f"w:{compiled.layer.id}:E{expert.expert_id}"
            self.assertEqual(expert.weight_tensor_ids, materialized_weight_members(compiled.physical_boundary, parent, available))
            self.assertEqual(len(expert.weight_tensor_ids), 6)
            self.assertTrue(set(expert.weight_tensor_ids) <= available)
            self.assertNotIn(parent, available)
        inventory = session.persistent_state_document()
        self.assertFalse(inventory["resumable_checkpoint"])
        self.assertIsNone(inventory["kv_state"])
        self.assertTrue(inventory["weight_fragments"])
        self.assertTrue(all(row["tensor_id"] in available for row in inventory["weight_fragments"]))
        self.assertTrue(all(row["locations"] for row in inventory["weight_fragments"]))
        self.assertTrue(all(row["version"] == 1 for row in inventory["weight_fragments"]))

    def test_moe_row_attention_and_partitioned_ffn_compose_in_three_valid_orders(self):
        original = load_and_compile(FIXTURES / "subweights/invocation.json")
        layer = replace(original.layer, weights=tuple(
            replace(weight, storage_bytes=192) if weight.id == "W_proj" else weight for weight in original.layer.weights))
        state = replace(original.state_in, weight_copies=tuple(
            replace(weight, storage_bytes=192) if weight.id == "W_proj" else weight for weight in original.state_in.weight_copies))
        transfers = dict(original.cost_table.transfers)
        transfers.update({key: value for key, value in moe_source().cost_table.transfers.items() if key[1] == 192})
        source = compile_invocation(layer, replace(original.invocation, state_in=state),
                                    replace(original.cost_table, transfers=transfers))
        rows = parse_execution_spec(json.loads((FIXTURES / "subweights/execution.json").read_text(encoding="utf-8")))
        selected = apply_execution_transform(source, rows)
        ffn = parse_ffn_execution(ffn_document(selected))
        # The independent suites choose different arbitrary synthetic transfer
        # numbers. A composed workload uses one shared exact transfer catalogue.
        transfers = dict(selected.cost_table.transfers)
        transfers.update({(entry.route, entry.storage_bytes): TransferEntry(entry.duration_ns, "synthetic_transfer")
                          for entry in ffn.transfer_entries})
        source = replace(source, cost_table=replace(source.cost_table, transfers=transfers))
        attention = parse_attention_stages(attention_document(source))
        results = []
        for specs in ((rows, attention, ffn), (attention, rows, ffn), (rows, ffn, attention)):
            compiled = apply_execution_transform(source, pipeline(*specs))
            results.append(compiled)
            session = CausalM2Session(compiled)
            self.assertFalse(any(op.kind.startswith("ffn_") for op in session.observe().operations))
            self.assertTrue(all(len(expert.weight_tensor_ids) == 6 for expert in session.observe().experts))
            execute(session, compiled)
            self.assertEqual(session.result().evaluation.status, STATUS_VALID)
            inventory = session.persistent_state_document()
            self.assertTrue(inventory["kv_state"])
            self.assertTrue(all(row["version"] == 1 for row in inventory["weight_fragments"]))
        for name in ("tensors", "operations"):
            expected = sorted(results[0].workload_document[name], key=lambda row: row["id"])
            for actual in results[1:]:
                self.assertEqual(sorted(actual.workload_document[name], key=lambda row: row["id"]), expected)

    def test_legacy_plus_new_passes_export_the_actual_pipeline(self):
        source = dense_source()
        spec = pipeline(legacy_spec(), dense_ffn_spec(source))
        compiled = apply_execution_transform(source, spec)
        session = CausalM2Session(compiled)
        exported = session.execution_document()
        self.assertEqual(exported["execution_spec"], transform_document(spec))
        self.assertEqual(exported["execution_variant"], "execution_pipeline")

    def test_typed_pipeline_version_is_revalidated(self):
        source = dense_source()
        invalid = ExecutionPipelineSpec((parse_attention_stages(attention_document(source)),), format_version="bad-version")
        with self.assertRaises((InvalidInput, Unsupported)):
            apply_execution_transform(source, invalid)

    def test_causal_session_accepts_single_stage_spec_or_a_pipeline_directly(self):
        source = dense_source()
        attention = parse_attention_stages(attention_document(source))
        for spec in (attention, pipeline(attention, dense_ffn_spec(source))):
            with self.subTest(variant=spec.variant):
                session = CausalM2Session(source, execution_spec=spec)
                self.assertEqual(session.execution_document()["execution_spec"], transform_document(spec))
                self.assertTrue(any(op.kind == "attention_qk" for op in session.observe().operations))
                execute(session, session._compiled)
                self.assertEqual(session.result().evaluation.status, STATUS_VALID)

    def test_one_scope_global_file_compiles_pipeline_and_runs_same_runtime(self):
        SCRATCH.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=SCRATCH) as directory:
            directory = Path(directory)
            dense = dense_source()
            dense_spec = pipeline(parse_attention_stages(attention_document(dense)), dense_ffn_spec(dense))
            moe = moe_source()
            moe_spec = pipeline(row_spec(), parse_attention_stages(attention_document(moe)))
            for name, source_path, spec in (
                ("dense", ROOT / "mapping/examples/m2/hybrid-interface-synthetic-decode-1.json", dense_spec),
                ("moe", FIXTURES / "attention_stages/moe-prefill.json", moe_spec),
            ):
                with self.subTest(model=name):
                    execution_path = directory / (name + ".execution.json")
                    execution_path.write_text(json.dumps(transform_document(spec)) + "\n", encoding="utf-8", newline="\n")
                    global_path = directory / (name + ".global.json")
                    global_path.write_text(json.dumps(dict(format_version="global-execution-0.1", id=name + "-pipeline",
                        rounds=[dict(id="round0", input_mode="initial_supplied_hidden", gate_cost=None,
                                     invocations=[dict(id="scope", invocation_file=str(source_path.resolve()),
                                                       execution_file=execution_path.name)])])) + "\n", encoding="utf-8", newline="\n")
                    compiled = compile_global_execution_file(global_path, visibility_mode="data_dependencies")
                    self.assertEqual(len(compiled.invocation_scopes), 1)
                    self.assertEqual(compiled.invocation_scopes[0].execution_variant, "execution_pipeline")
                    session = GlobalM2Session(compiled)
                    before = session.observe()
                    self.assertTrue(any(op.kind == "attention_qk" and op.execution_stage == "attention_qk" for op in before.operations))
                    execute(session, compiled.aggregate)
                    result = session.result().evaluation
                    self.assertEqual(result.status, STATUS_VALID)
                    self.assertEqual(session.observe().t_ns, result.makespan_ns)
                    inventory = session.persistent_state_document()
                    self.assertFalse(inventory["resumable_checkpoint"])
                    self.assertEqual(inventory["completed_invocations"], ["scope"])
                    for tid in inventory["kv_ids"]:
                        self.assertEqual(session._runtime.state.copy_at(tid, "dram"), COPY_READY)


if __name__ == "__main__":
    unittest.main()
