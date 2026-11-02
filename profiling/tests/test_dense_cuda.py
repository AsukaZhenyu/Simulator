"""Offline CUDA Dense protocol/evidence checks; no GPU tasks are executed.

The synthetic records assert specific device and buffer evidence. They do not
claim a real CUDA measurement or estimate an unknown backend workspace.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_dense_cuda_measurements as runner
import export_dense_cuda as exporter
from test_dense_layer import case as cpu_case, layer_document, native as cpu_native, suite
from llm_infer_model.m2.spec import parse_layer_document
from llm_infer_model.m2.costs import MacroShape, parse_cost_table
from llm_infer_model.tensor.spec import Unsupported
from profile_contract import summarize_samples

from llm_infer_model.m2 import dense_cuda_contract as contract


def cuda_case(kind="attention", **changes):
    selected = cpu_case(kind)
    selected.pop("threads")
    selected.update(changes)
    return contract.build_plan(suite([selected]), 0, "disabled")[0]


def cuda_native(selected, **changes):
    # Reuse the mathematical port inventory, then replace the entire CPU envelope.
    cpu_selected = {**selected, "threads": 2}
    result = cpu_native(cpu_selected)
    for key in ("threads", "threads_requested", "threads_applied", "thread_management", "cpu_cplan_work_size_bytes"):
        result.pop(key, None)
    result.update(schema="ggml_dense_layer_cuda_probe/1", contract=contract.CONTRACT,
        backend="CUDA0", cuda_device=0, cuda_graphs="disabled",
        cuda_graphs_requested="disabled", cuda_graphs_applied="disabled", cuda_graph_disable_environment="1",
        matmul_precision="ggml_cuda_backend_default",
        implementation_variant=contract.VARIANTS[selected["kind"]],
        timing="steady_clock_cuda_graph_compute_including_sync", workspace_bytes=None,
        workspace_source="unavailable_cuda_backend_private_scratch_and_runtime_overheads",
        nodes=[dict(name="synthetic_matmul", op="MUL_MAT", bytes=96, view=False, supported_by_requested_backend=True,
                    buffer_type="CUDA0")],
        backend_execution_check=dict(all_nodes_supported=True,
            all_tensors_on_requested_cuda_buffer=True, cpu_fallback_used=False, node_count=1),
        device_info=dict(index=0, name="Synthetic GPU", uuid="synthetic-device-uuid",
            total_memory_bytes=1024 * 1024, compute_capability_major=8, compute_capability_minor=9))
    result["memory_accounting"] = dict(resident_logical_bytes=result["resident_logical_bytes"],
        resident_allocated_bytes=result["resident_logical_bytes"]+32,
        internal_allocated_buffer_bytes=96, internal_logical_tensor_bytes_sum=96,
        known_tensor_allocation_bytes=result["resident_logical_bytes"]+128,
        workspace_bytes=None, backend_scratch_bytes=None, runtime_overhead_bytes=None,
        workspace_bound="unknown",
        workspace_scope="known_cuda_gallocr_capacity_excludes_private_backend_scratch_and_runtime_overheads")
    result["internal_tensor_bytes_upper_bound"] = 96
    native_names = {"q":"query", "k":"new_key", "v":"new_value", "gate":"attention_gate",
                    "hidden_out":"hidden_output", "attention_hidden":"attention_hidden"}
    checks = {}
    for port, size in result["output_ports_bytes"].items():
        count = size // 4
        checks[native_names[port]] = dict(passed=True, samples=count, output_elements=count,
            coverage="all_output_elements", max_abs_error=0.0, max_relative_error=0.0,
            rmse=0.0, reference_rms=1.0, normalized_rmse=0.0, atol=0.0002, rtol=0.002,
            reference="independent_double_declared_block_with_dequantized_stored_weights_and_f32_operator_boundaries",
            criterion="each_port_normalized_rmse_le_rtol_or_rmse_le_atol_and_exact_kv_validation")
    result["port_reference_checks"] = checks
    total = sum(check["output_elements"] for check in checks.values())
    result["reference_check"] = {**next(iter(checks.values())), "samples":total, "output_elements":total}
    result.update(changes)
    return result


def protocol_args():
    return SimpleNamespace(warmup=2, samples=3, seed=17, cuda_device=0, cuda_graphs="disabled")


class CudaNativeProtocolTests(unittest.TestCase):
    def reject(self, edit, kind="attention"):
        selected = cuda_case(kind)
        payload = cuda_native(selected)
        edit(payload)
        with self.assertRaises(ValueError):
            runner.validate_native_binding(selected, payload, protocol_args())

    def test_all_four_stages_bind_their_cuda_protocol(self):
        for kind in ("projection", "attention", "dense_ffn", "block"):
            with self.subTest(kind=kind):
                selected = cuda_case(kind)
                runner.validate_native_binding(selected, cuda_native(selected), protocol_args())

    def test_command_has_explicit_device_graph_mode_and_no_cpu_threads(self):
        command = runner.native_command(cuda_case(), Path("probe-dir"), protocol_args())
        self.assertEqual(command[command.index("--cuda-device")+1], "0")
        self.assertEqual(command[command.index("--cuda-graphs")+1], "disabled")
        self.assertNotIn("--threads", command)
        self.assertIn("ggml-dense-layer-cuda-probe", command[0])

    def test_cpu_case_and_native_envelopes_cannot_be_relabeled(self):
        with self.assertRaises(Unsupported):
            contract.build_plan(suite([cpu_case()]), 0, "disabled")
        self.reject(lambda payload: payload.update(threads=2))
        self.reject(lambda payload: payload.update(cpu_cplan_work_size_bytes=0))

    def test_wrong_backend_or_device_fails(self):
        for field, value in (("backend", "CPU"), ("backend", "CUDA1"),
                             ("cuda_device", 1), ("cuda_device", True)):
            with self.subTest(field=field, value=value):
                self.reject(lambda payload: payload.update({field:value}))
        self.reject(lambda payload: payload["device_info"].update(index=1))

    def test_cuda_graph_modes_must_match_explicit_disabled_protocol(self):
        for field in ("cuda_graphs", "cuda_graphs_requested", "cuda_graphs_applied"):
            for value in (None, "enabled", "auto"):
                with self.subTest(field=field, value=value):
                    self.reject(lambda payload: payload.update({field:value}))

    def test_fallback_and_global_truthy_flags_are_rejected(self):
        for field, value in (("cpu_fallback_used", True), ("cpu_fallback_used", 0),
                             ("all_nodes_supported", False), ("all_nodes_supported", 1),
                             ("all_tensors_on_requested_cuda_buffer", False)):
            with self.subTest(field=field, value=value):
                self.reject(lambda payload: payload["backend_execution_check"].update({field:value}))

    def test_global_flags_cannot_hide_wrong_per_node_buffer_or_support(self):
        for field, value in (("buffer_type", "CPU"), ("buffer_type", "CUDA1"),
                             ("buffer_type", None), ("supported_by_requested_backend", False),
                             ("supported_by_requested_backend", 1)):
            with self.subTest(field=field, value=value):
                self.reject(lambda payload: payload["nodes"][0].update({field:value}))

    def test_node_count_and_inventory_must_agree(self):
        self.reject(lambda payload: payload["backend_execution_check"].update(node_count=2))
        self.reject(lambda payload: payload.update(nodes=[]))

    def test_queried_device_identity_cannot_be_missing_or_guessed(self):
        for field, value in (("uuid", ""), ("name", None), ("total_memory_bytes", 0),
                             ("compute_capability_major", True), ("compute_capability_minor", -1)):
            with self.subTest(field=field, value=value):
                self.reject(lambda payload: payload["device_info"].update({field:value}))

    def test_unknown_workspace_and_backend_scratch_cannot_be_filled_with_zero(self):
        self.reject(lambda payload: payload.update(workspace_bytes=0))
        for field in ("workspace_bytes", "backend_scratch_bytes", "runtime_overhead_bytes"):
            with self.subTest(field=field):
                self.reject(lambda payload: payload["memory_accounting"].update({field:0}))
        self.reject(lambda payload: payload["memory_accounting"].pop("backend_scratch_bytes"))

    def test_known_memory_cannot_understate_resident_tensor_capacity(self):
        self.reject(lambda payload: payload["memory_accounting"].update(resident_allocated_bytes=0))
        self.reject(lambda payload: payload["memory_accounting"].update(internal_allocated_buffer_bytes=True))
        self.reject(lambda payload: payload["memory_accounting"].update(resident_logical_bytes=1))

    def test_numerical_failure_or_nonfinite_output_is_never_accepted(self):
        self.reject(lambda payload: payload.update(all_outputs_finite=False))
        self.reject(lambda payload: payload.update(reference_check={"passed":False}))
        self.reject(lambda payload: payload.update(reference_check={"passed":1}))

    def test_kv_history_and_append_must_pass_independently(self):
        for kind in ("attention", "block"):
            for field in ("passed", "append_exact", "history_unchanged"):
                with self.subTest(kind=kind, field=field):
                    self.reject(lambda payload: payload["kv_append_check"].update({field:False}), kind)

    def test_ports_and_digests_bind_exact_physical_layout(self):
        self.reject(lambda payload: payload["port_accounting"]["hidden"].update(dtype="f16"))
        self.reject(lambda payload: payload["input_port_digests"].update(guessed="a"*16))
        self.reject(lambda payload: payload.update(port_digest_algorithm="sha256"))

    def test_shape_history_seed_and_precision_mismatch_reject(self):
        for field, value in (("q", 4), ("history", 1), ("seed", 18),
                             ("matmul_precision", "f32"), ("samples", 4)):
            with self.subTest(field=field):
                self.reject(lambda payload: payload.update({field:value}))

    def test_known_allocation_and_upper_bound_must_match_component_inventory(self):
        self.reject(lambda payload: payload["memory_accounting"].pop("known_tensor_allocation_bytes"))
        self.reject(lambda payload: payload["memory_accounting"].update(known_tensor_allocation_bytes=0))
        self.reject(lambda payload: payload.update(internal_tensor_bytes_upper_bound=0))

    def test_each_output_port_needs_its_own_reference_evidence(self):
        self.reject(lambda payload: payload.pop("port_reference_checks"))
        self.reject(lambda payload: payload["port_reference_checks"].pop("attention_hidden"))
        self.reject(lambda payload: payload["port_reference_checks"].update(guessed={"passed":True}))
        self.reject(lambda payload: payload["port_reference_checks"]["attention_hidden"].update(passed=False))

    def test_reference_counts_coverage_and_fixed_tolerance_are_not_mutable(self):
        for field,value in (("samples",1),("output_elements",1),("coverage","sampled"),
                            ("atol",0),("rtol",1),("criterion","approximate")):
            with self.subTest(field=field):
                self.reject(lambda payload: payload["port_reference_checks"]["attention_hidden"].update({field:value}))

    def test_numerical_pass_flag_cannot_hide_rmse_failure_or_nonfinite_metrics(self):
        self.reject(lambda payload: payload["port_reference_checks"]["attention_hidden"].update(rmse=0.1,normalized_rmse=0.1))
        self.reject(lambda payload: payload["port_reference_checks"]["attention_hidden"].update(rmse=float("nan")))
        self.reject(lambda payload: payload["port_reference_checks"]["attention_hidden"].update(reference_rms=-1))

    def test_aggregate_reference_cannot_disagree_with_per_port_checks(self):
        for field,value in (("rmse",0.1),("samples",1),("normalized_rmse",0.1),("max_abs_error",0.1)):
            with self.subTest(field=field):
                self.reject(lambda payload: payload["reference_check"].update({field:value}))


class CudaRunnerEvidenceTests(unittest.TestCase):
    def run_fake(self, payload, *, selected=None, exitcode=0, failure=None):
        selected = selected or cuda_case()
        with tempfile.TemporaryDirectory(prefix="cuda-dense-runner-test-") as folder:
            root = Path(folder)
            binary = root / ("ggml-dense-layer-cuda-probe.exe" if os.name == "nt" else "ggml-dense-layer-cuda-probe")
            binary.write_bytes(b"synthetic CUDA probe never executed")
            source = root / "suite.json"
            source.write_text(json.dumps(suite([selected])), encoding="utf-8", newline="\n")
            output = root / "output"
            stdout = payload if isinstance(payload,str) else json.dumps(payload)
            child = subprocess.CompletedProcess([],exitcode,stdout,"synthetic diagnostic")
            arguments = ["--suite",str(source),"--native-bin",str(root),"--output",str(output),
                         "--cuda-device","0","--cuda-graphs","disabled","--samples","3","--warmup","2"]
            with (mock.patch.object(runner,"ROOT",root),
                  mock.patch.object(runner,"environment",return_value={"synthetic":True}),
                  mock.patch.object(runner.subprocess,"run",return_value=child,side_effect=failure) as probe,
                  contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO())):
                code = runner.main(arguments)
            self.assertEqual(probe.call_count,1)
            self.assertFalse((root/".measurement.lock").exists())
            return dict(code=code, rows=json.loads((output/"measurements.json").read_text()),
                        raw=json.loads((output/"raw/001.json").read_text()),
                        manifest=json.loads((output/"manifest.json").read_text()),
                        subprocess_kwargs=probe.call_args.kwargs,
                        files={str(p.relative_to(output)):p.read_bytes() for p in output.rglob("*") if p.is_file()})

    def test_unknown_workspace_valid_measurement_stays_capacity_unverified(self):
        result = self.run_fake(cuda_native(cuda_case()))
        self.assertEqual(result["code"],0)
        row=result["rows"][0]
        self.assertIsNone(row["native"]["workspace_bytes"])
        self.assertFalse(row["eligibility"]["workspace_known"])
        self.assertFalse(row["eligibility"]["capacity_verified"])
        self.assertFalse(row["eligibility"]["automatic_M2_export"])
        self.assertFalse(row["eligibility"]["CPU_fallback_permitted"])
        self.assertEqual(row["native"]["wall_ns"], json.loads(result["raw"]["stdout"])["wall_ns"])
        self.assertEqual(result["manifest"]["protocol"]["backend"], "CUDA0")
        self.assertEqual(result["subprocess_kwargs"]["env"]["GGML_CUDA_DISABLE_GRAPHS"], "1")
        for name,contents in result["files"].items():
            if name.endswith((".json",".csv")):
                self.assertNotIn(b"\r\n", contents)

    def test_failed_gpu_record_preserves_stderr_and_native_output(self):
        payload=cuda_native(cuda_case(),backend="CPU")
        result=self.run_fake(payload)
        self.assertEqual(result["code"],1)
        self.assertEqual(result["rows"][0]["status"],"failed")
        self.assertEqual(json.loads(result["raw"]["stdout"]),payload)
        self.assertEqual(result["raw"]["stderr"],"synthetic diagnostic")
        self.assertEqual(result["manifest"]["measured_cases"],0)

    def test_timeout_preserves_partial_raw_evidence(self):
        failure=subprocess.TimeoutExpired(["synthetic"],1,output=b"partial output",stderr=b"partial stderr")
        result=self.run_fake("",failure=failure)
        self.assertEqual(result["code"],1)
        self.assertTrue(result["raw"]["timed_out"])
        self.assertEqual(result["raw"]["stdout"],"partial output")
        self.assertEqual(result["raw"]["stderr"],"partial stderr")

    def test_missing_or_nonpositive_timer_samples_fail_in_runner(self):
        for samples in ([1000,1001],[1000,0,1001],[1000,True,1001]):
            with self.subTest(samples=samples):
                result=self.run_fake(cuda_native(cuda_case(),wall_ns=samples))
                self.assertEqual(result["code"],1)



def cuda_layer_document():
    document = layer_document()
    document["format_version"] = "m2-layer-0.4"
    document["id"] = "synthetic-full-cuda-dense-fixture"
    physical = document["physical"]
    physical.pop("threads")
    physical.update(contract=contract.CONTRACT, cuda_device=0, cuda_graphs="disabled")
    for weight in document["weights"]:
        weight["initial_location"] = "vram"
    return document


class CudaExportEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory(prefix="cuda-dense-export-test-")
        self.addCleanup(self.folder.cleanup)
        self.root=Path(self.folder.name)
        (self.root/"raw").mkdir()
        self.binary=self.root/("ggml-dense-layer-cuda-probe.exe" if os.name=="nt" else "ggml-dense-layer-cuda-probe")
        self.binary.write_bytes(b"synthetic CUDA executable never executed")
        self.digest=hashlib.sha256(self.binary.read_bytes()).hexdigest()
        self.layer=parse_layer_document(cuda_layer_document())
        self.cases=[cuda_case(kind) for kind in ("projection","attention","dense_ffn","block")]
        self.manifest=dict(schema_version="dense_cuda_measurement_profile_v1",contract=contract.CONTRACT,
            profile_id="synthetic-cuda-evidence",suite=suite(self.cases),
            protocol=dict(backend="CUDA0",batch=1,cuda_device=0,cuda_graphs="disabled",warmup=2,
                samples=3,seed=17,sequential_execution=True,graphs_per_sample=1,setup_compute_repetitions=1,
                measured_graph_untimed_compute_repetitions=3,
                upstream_setup_compute_repetitions_by_kind={kind:0 if kind=="block" else 1 for kind in ("projection","attention","dense_ffn","block")},
                total_untimed_compute_repetitions_by_kind={kind:3 if kind=="block" else 4 for kind in ("projection","attention","dense_ffn","block")}),
            software=dict(runtime_sha256={str(self.binary):self.digest},child_flags={"GGML_CUDA_DISABLE_GRAPHS":"1"}))
        self.rows=[]
        for index,selected in enumerate(self.cases):
            payload=cuda_native(selected)
            summary=summarize_samples(payload["wall_ns"])
            self.rows.append(dict(case_id=selected["id"],case=selected,status="measured",
                signature=runner.measurement_signature(selected,self.digest),
                native={**payload,"expected_samples":3,"summary":summary},summary=summary,
                quality_flags=[],raw_output=f"raw/{index+1:03d}.json"))
            self.write_raw(index)

    def write_raw(self,index):
        row=self.rows[index]
        native={key:value for key,value in row["native"].items() if key not in ("summary","expected_samples")}
        raw=dict(case=row["case"],command=runner.native_command(row["case"],self.binary.parent,protocol_args()),
                 returncode=0,stdout=json.dumps(native),stderr="synthetic not executed")
        (self.root/row["raw_output"]).write_text(json.dumps(raw),encoding="utf-8",newline="\n")

    def modify(self,index,**changes):
        row=self.rows[index]
        row["native"].update(changes)
        row["summary"]=row["native"]["summary"]=summarize_samples(row["native"]["wall_ns"])
        self.write_raw(index)

    def export(self,**kwargs):
        return exporter.export_cost_table(self.layer,self.rows,raw_directory=self.root,manifest=self.manifest,**kwargs)

    def test_complete_four_stage_export_keeps_unknown_workspace_non_executable(self):
        costs,report=self.export()
        self.assertEqual(costs["format_version"],"m2-layer-0.4")
        self.assertEqual(len(costs["entries"]),3)
        self.assertNotIn("block",{entry["kind"] for entry in costs["entries"]})
        self.assertTrue(costs["validation_only"])
        self.assertTrue(report["validation_only"])
        self.assertFalse(report["capacity_verified"])
        self.assertFalse(report["execution_ready"])
        self.assertFalse(report["calibration_claimed"])
        self.assertEqual(report["composition"][0]["absolute_relative_error"],0)
        self.assertIsNone(report["composition"][0]["within_explicit_tolerance"])
        parsed=parse_cost_table(costs)
        for entry in costs["entries"]:
            self.assertEqual(entry["device"],"gpu")
            self.assertIsNone(entry["workspace_bytes"])
            shape=MacroShape(entry["phase"],entry["batch"],entry["tokens"],entry["visible_history"])
            with self.assertRaisesRegex(Unsupported,"workspace_bytes"):
                parsed.lookup(entry["kind"],"gpu",shape,physical_signature=entry["physical_signature"])

    def test_export_never_mutates_source_evidence(self):
        before=copy.deepcopy((self.rows,self.manifest))
        self.export()
        self.assertEqual((self.rows,self.manifest),before)

    def test_incomplete_shape_group_and_duplicate_records_reject(self):
        original=copy.deepcopy(self.rows)
        self.rows.pop()
        with self.assertRaisesRegex(ValueError,"complete exact shape group"):
            self.export()
        self.rows=original+[copy.deepcopy(original[0])]
        with self.assertRaisesRegex(ValueError,"duplicate"):
            self.export()

    def test_missing_history_cannot_be_supplied_from_another_shape(self):
        self.rows[0]["case"]["visible_history"]=3
        with self.assertRaises(ValueError):
            self.export()

    def test_complete_ports_must_match_joint_bytes_and_upstream_lineage(self):
        for index,field,port in ((0,"stage_port_digests","q"),(1,"input_port_digests","q"),
                                 (1,"input_port_digests","hidden"),(2,"input_port_digests","attention_hidden")):
            with self.subTest(index=index,field=field,port=port):
                before=copy.deepcopy(self.rows)
                digests={**self.rows[index]["native"][field],port:"fedcba9876543210"}
                self.modify(index,**{field:digests})
                with self.assertRaisesRegex(ValueError,"materialized port|input"):
                    self.export()
                self.rows=before
                self.write_raw(index)

    def test_actual_cuda_device_identity_must_be_identical_across_stages(self):
        observed={**self.rows[1]["native"]["device_info"],"uuid":"different-physical-device"}
        self.modify(1,device_info=observed)
        with self.assertRaisesRegex(ValueError,"observed CUDA device"):
            self.export()

    def test_fallback_or_wrong_node_buffer_cannot_be_waived(self):
        for field,value in (("cpu_fallback_used",True),("all_nodes_supported",False)):
            with self.subTest(field=field):
                before=copy.deepcopy(self.rows)
                checks={**self.rows[0]["native"]["backend_execution_check"],field:value}
                self.modify(0,backend_execution_check=checks)
                with self.assertRaises(ValueError):
                    self.export(allow_unstable=True)
                self.rows=before
                self.write_raw(0)
        nodes=copy.deepcopy(self.rows[0]["native"]["nodes"])
        nodes[0]["buffer_type"]="CPU"
        self.modify(0,nodes=nodes)
        with self.assertRaises(ValueError):
            self.export(allow_unstable=True)

    def test_unknown_workspace_cannot_become_zero_at_export(self):
        self.modify(0,workspace_bytes=0)
        with self.assertRaisesRegex(ValueError,"workspace"):
            self.export(allow_unstable=True)

    def test_unstable_observations_require_explicit_allow_unstable(self):
        self.modify(0,wall_ns=[100,1000,10000])
        self.rows[0]["quality_flags"]=["high_observed_variability_cv_gt_0.1"]
        with self.assertRaisesRegex(ValueError,"unstable"):
            self.export()
        costs,report=self.export(allow_unstable=True)
        self.assertTrue(costs["validation_only"])
        self.assertFalse(report["execution_ready"])
        self.assertTrue(report["entries"][0]["cv_gt_0_1"])

    def test_composition_error_report_does_not_claim_calibration(self):
        self.modify(3,wall_ns=[1000,1001,999])
        costs,report=self.export()
        self.assertEqual(report["composition"][0]["absolute_relative_error"],2)
        self.assertFalse(report["calibration_claimed"])
        with self.assertRaisesRegex(ValueError,"exceeds explicit tolerance"):
            self.export(max_relative_error=0.2)
        costs,report=self.export(max_relative_error=0.2,allow_composition_error=True)
        self.assertTrue(costs["validation_only"])
        self.assertFalse(report["composition"][0]["within_explicit_tolerance"])

    def test_signature_raw_summary_and_runtime_hash_tampering_reject(self):
        for mutation in ("signature","raw","summary","native_summary","runtime","raw_command"):
            with self.subTest(mutation=mutation):
                before=copy.deepcopy((self.rows,self.manifest))
                if mutation=="signature":
                    self.rows[0]["signature"]["physical_contract_sha256"]="0"*64
                elif mutation=="raw":
                    path=self.root/self.rows[0]["raw_output"]
                    raw=json.loads(path.read_text());raw["stdout"]="{}"
                    path.write_text(json.dumps(raw),encoding="utf-8",newline="\n")
                elif mutation=="summary":
                    self.rows[0]["summary"]={**self.rows[0]["summary"],"median_ns":1}
                elif mutation=="native_summary":
                    self.rows[0]["native"]["summary"]={**self.rows[0]["native"]["summary"],"cv":0.9}
                elif mutation=="runtime":
                    self.manifest["software"]["runtime_sha256"][str(self.binary)]="0"*64
                else:
                    path=self.root/self.rows[0]["raw_output"]
                    raw=json.loads(path.read_text());i=raw["command"].index("--cuda-device")
                    raw["command"][i+1]="1";path.write_text(json.dumps(raw),encoding="utf-8",newline="\n")
                with self.assertRaises(ValueError):
                    self.export()
                self.rows,self.manifest=before
                self.write_raw(0)

    def test_manifest_cuda_protocol_and_environment_are_binding(self):
        for location,field,value in (("protocol","cuda_device",1),("protocol","cuda_graphs","auto"),
                                      ("protocol","backend","CPU"),("protocol","warmup",3),
                                      ("protocol","threads",2),("flags","GGML_CUDA_DISABLE_GRAPHS",None)):
            with self.subTest(location=location,field=field):
                before=copy.deepcopy(self.manifest)
                target=self.manifest["protocol"] if location=="protocol" else self.manifest["software"]["child_flags"]
                target[field]=value
                with self.assertRaises(ValueError):
                    self.export()
                self.manifest=before

    def test_failed_records_and_raw_path_escape_are_not_exportable(self):
        original=copy.deepcopy(self.rows)
        self.rows[0]["status"]="failed"
        with self.assertRaisesRegex(ValueError,"failed"):
            self.export(allow_unstable=True)
        self.rows=original
        self.rows[0]["raw_output"]="../other.json"
        with self.assertRaisesRegex(ValueError,"escapes"):
            self.export()

    def test_invalid_acceptance_tolerance_cannot_be_waived(self):
        for value in (True,float("nan"),float("inf"),-1):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,"tolerance"):
                self.export(max_relative_error=value,allow_composition_error=True)

    def test_cli_writes_observations_lf_and_reports_explicit_phase_omissions(self):
        measurements=self.root/"measurements.json"
        manifest=self.root/"manifest.json"
        layer=self.root/"layer.json"
        omitted=dict(case_id="failed_decode",case=cuda_case("block",id="failed_decode",phase="decode",tokens=1,visible_history=3),
                     status="failed",quality_flags=[])
        measurements.write_text(json.dumps(self.rows+[omitted]),encoding="utf-8",newline="\n")
        manifest.write_text(json.dumps(self.manifest),encoding="utf-8",newline="\n")
        layer.write_text(json.dumps(cuda_layer_document()),encoding="utf-8",newline="\n")
        output=self.root/"export"
        with contextlib.redirect_stdout(io.StringIO()):
            code=exporter.main(["--measurements",str(measurements),"--layer",str(layer),"--output",str(output),"--phase","prefill"])
        self.assertEqual(code,0)
        report=json.loads((output/"export_report.json").read_text())
        self.assertFalse(report["record_selection"]["silent_quality_filtering"])
        self.assertEqual(report["record_selection"]["omitted_case_ids"],["failed_decode"])
        for filename in ("costs.json","export_report.json"):
            self.assertNotIn(b"\r\n",(output/filename).read_bytes())
        with self.assertRaisesRegex(ValueError,"empty"):
            exporter.main(["--measurements",str(measurements),"--layer",str(layer),"--output",str(output),"--phase","prefill"])


if __name__ == "__main__":
    unittest.main()
