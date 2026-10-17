"""Portable numerical and selected-row accounting checks; no hardware timings."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_moe_rows import (compare_outputs, main, materialize, run_verification,
    selected_row_graph, validate_case, validate_row_execution_document, verify_case,
    whole_token_reference)


CASE_PATH = Path(__file__).resolve().parents[1] / "cases" / "moe_rows_reference.json"


def case():
    return json.loads(CASE_PATH.read_bytes())


def accounting_fixture():
    rows = [{"request_id": "r0", "absolute_position": 3},
            {"request_id": "r1", "absolute_position": 8}]
    scopes = {name: deepcopy(rows) for name in ("compact", "gate", "up", "activation", "group")}
    scopes["r0contribution"] = [rows[0]]
    scopes["r1contribution"] = [rows[1]]
    sizes = {"hidden": 80, "weight": 420, "compact": 40, "gate": 56, "up": 56,
             "activation": 56, "group": 40, "r0contribution": 20, "r1contribution": 20}
    tensors = [{"id": name, "storage_bytes": value} for name, value in sizes.items()]
    operations = []
    def op(name, kind, inputs, output):
        operations.append({"id": name, "semantic_op": kind, "inputs": inputs, "output": output})
    op("gather", "gather_rows", ["hidden"], "compact")
    op("gate", "expert_gate", ["weight", "compact"], "gate")
    op("up", "expert_up", ["weight", "compact"], "up")
    op("swiglu", "expert_swiglu", ["gate", "up"], "activation")
    op("down", "expert_down", ["weight", "activation"], "group")
    op("scatter0", "scatter_rows", ["group"], "r0contribution")
    op("scatter1", "scatter_rows", ["group"], "r1contribution")
    region = {"expert_id": 0, "work_items": rows, "grouped_input_tensor": "compact",
              "grouped_output_tensor": "group", "gather_operation_id": "gather",
              "expert_operation_ids": ["gate", "up", "swiglu", "down"],
              "scatter_operation_ids": ["scatter0", "scatter1"]}
    return {"tensors": tensors, "operations": operations}, {"row_execution": {
        "format_version": "moe-execution-0.1", "expert_variant": "swiglu_staged",
        "hidden_bytes_per_token": 20, "intermediate_bytes_per_token": 28,
        "regions": [region], "tensor_row_scopes": scopes}}, {"0": [("r0", 3), ("r1", 8)]}


class MoERowNumericalTests(unittest.TestCase):
    def test_batched_repeated_groups_shared_scores_and_residual_equivalent(self):
        report, raw = verify_case(case())
        self.assertTrue(report["numeric_check"]["passed"])
        self.assertLess(report["numeric_check"]["max_abs_error"], 2e-5)
        # A nonzero error establishes these are independent reduction paths.
        self.assertGreater(report["numeric_check"]["max_abs_error"], 0)
        self.assertEqual(set(raw["group_ports"]), {"0", "1", "2", "shared"})
        self.assertEqual(report["row_accounting"]["0"]["row_count"], 5)
        self.assertEqual(report["row_accounting"]["1"]["row_count"], 4)
        self.assertEqual(report["row_accounting"]["2"]["row_count"], 3)
        self.assertEqual(report["row_accounting"]["shared"]["row_count"], 6)
        self.assertEqual(raw["group_ports"]["1"]["route_slots"], [1, 0, 1, 0])

    def test_selected_rows_reduce_whole_request_byte_scope(self):
        report, _ = verify_case(case())
        self.assertEqual(report["row_accounting"]["2"]["selected_input_bytes"], 60)
        self.assertEqual(report["row_accounting"]["2"]["whole_request_input_bytes"], 100)
        self.assertEqual(report["row_accounting"]["2"]["gate_bytes"], 84)
        self.assertIsNone(report["row_accounting"]["2"]["workspace_bytes"])
        for field in ("performance_measured", "cost_table_exported", "GGML_execution_verified",
                      "GPU_execution_verified", "workspace_measured"):
            self.assertFalse(report[field])

    def test_no_shared_expert(self):
        c = case(); c["shared_expert"] = False
        report, raw = verify_case(c)
        self.assertTrue(report["numeric_check"]["passed"])
        self.assertNotIn("shared", raw["group_ports"])

    def test_inactive_expert_is_never_executed(self):
        c = case(); data = materialize(c)
        data["weights"]["3"] = None
        whole = whole_token_reference(c, data)
        selected, ports = selected_row_graph(c, data)
        self.assertTrue(compare_outputs(whole, selected, atol=c["atol"], rtol=c["rtol"])["passed"])
        self.assertNotIn("3", ports)

    def test_zero_scores_preserve_exactly_one_residual(self):
        c = case(); c["shared_expert"] = False
        for item in c["work_items"]:
            item["scores"] = [0, 0]
        data = materialize(c)
        whole = whole_token_reference(c, data)
        selected, _ = selected_row_graph(c, data)
        self.assertEqual(whole, data["hidden"])
        self.assertEqual(selected, data["hidden"])

    def test_scores_are_explicit_not_renormalized(self):
        c = case(); c["work_items"][0]["scores"] = [0.1, 0.2]
        report, raw = verify_case(c)
        self.assertTrue(report["numeric_check"]["passed"])
        self.assertEqual(raw["case"]["work_items"][0]["scores"], [0.1, 0.2])
        self.assertFalse(report["scores_normalized_by_verifier"])

    def test_expert_completion_order_does_not_change_equivalence(self):
        c = case(); data = materialize(c)
        whole = whole_token_reference(c, data)
        selected, _ = selected_row_graph(c, data, group_order=["shared", "2", "1", "0"])
        self.assertTrue(compare_outputs(whole, selected, atol=c["atol"], rtol=c["rtol"])["passed"])

    def test_wrong_score_slot_is_detected(self):
        c = case(); data = materialize(c)
        whole = whole_token_reference(c, data)
        changed = deepcopy(c)
        changed["work_items"][0]["scores"].reverse()
        selected, _ = selected_row_graph(changed, data)
        self.assertFalse(compare_outputs(whole, selected, atol=c["atol"], rtol=c["rtol"])["passed"])

    def test_wrong_gather_row_is_detected(self):
        c = case(); data = materialize(c)
        whole = whole_token_reference(c, data)
        bad = deepcopy(data)
        bad["hidden"][0], bad["hidden"][1] = bad["hidden"][1], bad["hidden"][0]
        selected, _ = selected_row_graph(c, bad)
        self.assertFalse(compare_outputs(whole, selected, atol=c["atol"], rtol=c["rtol"])["passed"])

    def test_omitted_shared_contribution_is_detected(self):
        c = case(); data = materialize(c)
        whole = whole_token_reference(c, data)
        omitted = deepcopy(c); omitted["shared_expert"] = False
        selected, _ = selected_row_graph(omitted, data)
        self.assertFalse(compare_outputs(whole, selected, atol=c["atol"], rtol=c["rtol"])["passed"])

    def test_nan_or_inf_cannot_pass(self):
        for value in (float("inf"), float("nan")):
            check = compare_outputs([[0.0]], [[value]], atol=1, rtol=1)
            self.assertFalse(check["passed"])
            self.assertIsNone(check["max_abs_error"])

    def test_deterministic_same_seed_and_fixture(self):
        self.assertEqual(verify_case(case()), verify_case(case()))

    def test_invalid_inputs_fail_without_inference(self):
        mutations = [lambda c: c.update(dtype="f16"),
                     lambda c: c.update(hidden_size=True),
                     lambda c: c.update(atol=float("nan")),
                     lambda c: c.update(backend="CPU"),
                     lambda c: c["work_items"].append(deepcopy(c["work_items"][0])),
                     lambda c: c["work_items"][0].update(expert_ids=[0, 0]),
                     lambda c: c["work_items"][0].update(expert_ids=[0, 4]),
                     lambda c: c["work_items"][0].update(scores=[0.5]),
                     lambda c: c["work_items"][0].update(scores=[-0.1, 1.1]),
                     lambda c: c["work_items"][0].pop("scores")]
        for mutation in mutations:
            c = case(); mutation(c)
            with self.subTest(case=c), self.assertRaises(ValueError):
                validate_case(c)


class ModelingRowAccountingTests(unittest.TestCase):
    def test_actual_expanded_modeling_graph_when_package_is_available(self):
        try:
            from llm_infer_model.m2.expand import load_and_compile
            from llm_infer_model.m2.row_execution import parse_execution_spec, transform_moe_execution
        except ModuleNotFoundError as exc:
            if exc.name is not None and exc.name.startswith("llm_infer_model"):
                self.skipTest("optional Modeling package not installed; numerical oracle uses only stdlib")
            raise
        invocation = Path(__file__).resolve().parents[2] / "mapping/examples/m2/moe-interface-synthetic-prefill-2.json"
        original = load_and_compile(invocation)
        for staged in (False, True):
            spec = {"format_version": "moe-execution-0.1", "variant": "selected_rows",
                    "expert_variant": "swiglu_staged" if staged else "opaque",
                    "provenance": "synthetic_execution", "entries": [], "transfer_entries": [],
                    "index_storage_contract": "stage_workspace_includes_row_index_temporaries"}
            if staged:
                spec["intermediate_bytes_per_token"] = 32
            else:
                spec["opaque_cost_source"] = "existing_synthetic_per_expert_group"
            kinds = ["gather_rows", "scatter_rows", "combine_rows"] + (
                ["expert_gate", "expert_up", "expert_swiglu", "expert_down"] if staged else [])
            for kind in kinds:
                for device in ("cpu", "gpu"):
                    for count in (1, 2):
                        spec["entries"].append({"kind": kind, "device": device, "phase": "prefill",
                            "batch": 1, "token_count": count, "hidden_bytes_per_token": 16,
                            "intermediate_bytes_per_token": 32 if staged else 0,
                            "duration_ns": 100, "workspace_bytes": 4})
            compiled = transform_moe_execution(original, parse_execution_spec(spec))
            checked = validate_row_execution_document(compiled.workload_document, compiled.physical_boundary,
                                                      expected_groups=compiled.expert_work_items)
            self.assertTrue(checked["row_accounting_verified"])
            self.assertTrue(checked["full_request_combine_checked"])
            self.assertEqual(checked["region_count"], 3)
            self.assertFalse(compiled.physical_boundary["numeric_execution_verified"])

    def test_staged_graph_scope_and_bytes_match_true_group(self):
        workload, boundary, expected = accounting_fixture()
        result = validate_row_execution_document(workload, boundary, expected_groups=expected)
        self.assertTrue(result["row_accounting_verified"])
        self.assertFalse(result["numeric_execution_verified"])
        self.assertFalse(result["performance_calibrated"])
        self.assertEqual(result["regions"][0]["selected_input_bytes"], 40)

    def test_shared_catalogue_marker(self):
        workload, boundary, expected = accounting_fixture()
        boundary["row_execution"]["regions"][0]["expert_id"] = "shared"
        validate_row_execution_document(workload, boundary, expected_groups={"shared": expected["0"]})

    def test_whole_request_bytes_disguised_as_selected_are_rejected(self):
        workload, boundary, _ = accounting_fixture()
        next(t for t in workload["tensors"] if t["id"] == "compact")["storage_bytes"] = 80
        with self.assertRaisesRegex(ValueError, "selected row count"):
            validate_row_execution_document(workload, boundary)

    def test_intermediate_shape_mismatch_is_rejected(self):
        workload, boundary, _ = accounting_fixture()
        next(t for t in workload["tensors"] if t["id"] == "gate")["storage_bytes"] = 40
        with self.assertRaisesRegex(ValueError, "stage logical bytes"):
            validate_row_execution_document(workload, boundary)

    def test_group_row_reordering_is_rejected(self):
        workload, boundary, _ = accounting_fixture()
        boundary["row_execution"]["tensor_row_scopes"]["compact"].reverse()
        with self.assertRaisesRegex(ValueError, "row order"):
            validate_row_execution_document(workload, boundary)

    def test_scatter_missing_or_duplicate_row_is_rejected(self):
        workload, boundary, _ = accounting_fixture()
        boundary["row_execution"]["regions"][0]["scatter_operation_ids"] = ["scatter0", "scatter0"]
        with self.assertRaisesRegex(ValueError, "cover each selected row"):
            validate_row_execution_document(workload, boundary)

    def test_future_or_wrong_route_rows_do_not_pass(self):
        workload, boundary, _ = accounting_fixture()
        with self.assertRaisesRegex(ValueError, "true explicit routing rows"):
            validate_row_execution_document(workload, boundary, expected_groups={"0": [("r0", 4), ("r1", 8)]})

    def test_unknown_execution_variant_is_rejected(self):
        workload, boundary, _ = accounting_fixture()
        boundary["row_execution"]["expert_variant"] = "auto"
        with self.assertRaises(ValueError):
            validate_row_execution_document(workload, boundary)

    def test_dropped_swiglu_stage_is_rejected(self):
        workload, boundary, _ = accounting_fixture()
        boundary["row_execution"]["regions"][0]["expert_operation_ids"].remove("swiglu")
        with self.assertRaisesRegex(ValueError, "dropped or duplicated"):
            validate_row_execution_document(workload, boundary)

    def test_wrong_swiglu_dependency_is_rejected(self):
        workload, boundary, _ = accounting_fixture()
        next(op for op in workload["operations"] if op["id"] == "swiglu")["inputs"] = ["gate", "gate"]
        with self.assertRaisesRegex(ValueError, "dependencies"):
            validate_row_execution_document(workload, boundary)

    def test_wrong_scatter_source_is_rejected(self):
        workload, boundary, _ = accounting_fixture()
        next(op for op in workload["operations"] if op["id"] == "scatter0")["inputs"] = ["compact"]
        with self.assertRaisesRegex(ValueError, "declared expert output"):
            validate_row_execution_document(workload, boundary)

    def test_opaque_expert_uses_same_compact_scopes(self):
        workload, boundary, _ = accounting_fixture()
        metadata = boundary["row_execution"]; metadata["expert_variant"] = "opaque"
        metadata["intermediate_bytes_per_token"] = 0
        metadata["regions"][0]["expert_operation_ids"] = ["down"]
        op = next(op for op in workload["operations"] if op["id"] == "down")
        op["semantic_op"] = "expert_ffn"; op["inputs"] = ["weight", "compact"]
        validate_row_execution_document(workload, boundary)

    def test_complete_combines_consume_only_own_request_contributions(self):
        workload, boundary, _ = accounting_fixture()
        for request, position in (("r0", 3), ("r1", 8)):
            output = request + "out"
            workload["tensors"].append({"id": output, "storage_bytes": 20})
            workload["operations"].append({"id": request + "combine", "semantic_op": "combine_rows",
                "inputs": ["hidden", request + "contribution"], "output": output})
            boundary["row_execution"]["tensor_row_scopes"][output] = [
                {"request_id": request, "absolute_position": position}]
        result = validate_row_execution_document(workload, boundary)
        self.assertTrue(result["full_request_combine_checked"])
        workload["operations"][-1]["inputs"].append("r0contribution")
        with self.assertRaisesRegex(ValueError, "own-request contribution"):
            validate_row_execution_document(workload, boundary)

    def test_malformed_workload_fails_cleanly(self):
        _, boundary, _ = accounting_fixture()
        with self.assertRaises(ValueError):
            validate_row_execution_document({}, boundary)


class MoERowEvidenceTests(unittest.TestCase):
    def test_cli_writes_lf_evidence_and_bound_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results"
            with redirect_stdout(io.StringIO()):
                code = main(["--case", str(CASE_PATH), "--out", str(output)])
            self.assertEqual(code, 0)
            manifest = json.loads((output / "manifest.json").read_bytes())
            for name, digest in manifest["artifacts_sha256"].items():
                content = (output / name).read_bytes()
                self.assertEqual(hashlib.sha256(content).hexdigest(), digest)
                self.assertNotIn(b"\r\n", content)
            report = json.loads((output / "report.json").read_bytes())
            self.assertEqual(report["raw_sha256"], manifest["artifacts_sha256"]["raw.json"])
            self.assertFalse(manifest["physical_cost_export"])

    def test_no_overwrite_of_existing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            sentinel = output / "keep.txt"; sentinel.write_bytes(b"unchanged")
            with self.assertRaisesRegex(ValueError, "never overwritten"):
                run_verification(CASE_PATH, output)
            self.assertEqual(sentinel.read_bytes(), b"unchanged")

    def test_modeling_workload_and_boundary_checked_when_explicit(self):
        workload, boundary, _ = accounting_fixture()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            wp, bp = directory / "workload.json", directory / "boundary.json"
            wp.write_text(json.dumps(workload), encoding="utf-8")
            bp.write_text(json.dumps(boundary), encoding="utf-8")
            report = run_verification(CASE_PATH, directory / "result", workload_path=wp, boundary_path=bp)
            self.assertTrue(report["modeling_row_accounting"]["row_accounting_verified"])

    def test_missing_pair_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            with self.assertRaisesRegex(ValueError, "supplied together"):
                run_verification(CASE_PATH, output, workload_path=Path(directory) / "missing")
            self.assertFalse(output.exists())

    def test_source_change_during_verification_is_rejected(self):
        original = verify_case
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "case.json"; source.write_bytes(CASE_PATH.read_bytes())
            output = directory / "out"
            def changed(document):
                result = original(document)
                source.write_bytes(b"{}")
                return result
            with patch("verify_moe_rows.verify_case", changed), self.assertRaisesRegex(ValueError, "changed"):
                run_verification(source, output)
            self.assertFalse(output.exists())

    def test_numeric_failure_has_distinct_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory); c = case(); c["atol"] = c["rtol"] = 0
            source = directory / "case.json"; source.write_text(json.dumps(c), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                code = main(["--case", str(source), "--out", str(directory / "out")])
            self.assertEqual(code, 3)

    def test_malformed_case_has_distinct_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad.json"; source.write_text("{}", encoding="utf-8")
            with redirect_stderr(io.StringIO()):
                code = main(["--case", str(source), "--out", str(Path(directory) / "out")])
            self.assertEqual(code, 2)
            self.assertFalse((Path(directory) / "out").exists())


if __name__ == "__main__":
    unittest.main()
