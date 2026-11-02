"""Offline CUDA memory evidence checks; no CUDA or process measurements run here.

A device-wide free-memory snapshot cannot establish process workspace or a peak.
These tests preserve that distinction, including changes caused by other users
of the device that make an observed used-byte delta negative.
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
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cuda_memory_contract as contract
import run_dense_cuda_memory as runner
from test_dense_cuda import cuda_case, cuda_native, protocol_args
from test_dense_layer import suite


SNAPSHOT_LABELS = (
    "after_context_initialization",
    "after_upstream_preparation",
    "after_stage_allocation",
    "after_setup",
    "after_warmup",
    "after_samples",
    "after_stage_release",
)


def memory_native(selected=None, **changes):
    selected = selected or cuda_case()
    dense = cuda_native(selected)
    total = dense["device_info"]["total_memory_bytes"]
    # Other processes are allowed to free memory at release; never clamp it.
    used = [1000, 1200, 1600, 1800, 1900, 2100, 900]
    snapshots = [dict(label=label, elapsed_ns=(index+1)*100,
        device_index=0, free_bytes=total-current, total_bytes=total,
        used_bytes=current) for index,(label,current) in enumerate(zip(SNAPSHOT_LABELS,used))]
    result = dict(schema="ggml_dense_layer_cuda_memory_probe/1",
        purpose="memory_diagnostics_only", timing_usable_as_cost=False,
        hardware_compute_performed=True, workspace_bytes=None,
        workspace_bound="unknown", snapshot_scope="device_wide_cuda_memory_availability",
        source="cudaMemGetInfo_after_cudaDeviceSynchronize", process_peak_bytes=None,
        capacity_verified=False, peak_measurement_claimed=False,
        process_attribution_verified=False, backend="CUDA0", cuda_device=0,
        cuda_graphs="disabled", device_info=copy.deepcopy(dense["device_info"]),
        snapshot_count=len(snapshots), max_observed_used_bytes=max(used),
        max_observed_used_bytes_scope="maximum_of_seven_device_wide_snapshots_not_process_peak",
        upstream_setup_compute_repetitions=0 if selected["kind"]=="block" else 1,
        dense_validation_exit_code=0, snapshots=snapshots, dense_validation=dense)
    result.update(changes)
    return result


class CudaMemoryObservationTests(unittest.TestCase):
    def validate(self,payload=None,selected=None):
        selected = selected or cuda_case()
        return contract.validate_memory_probe(selected, payload or memory_native(selected), protocol_args())

    def reject(self,edit,selected=None):
        selected = selected or cuda_case()
        payload=memory_native(selected)
        edit(payload)
        with self.assertRaises(ValueError):
            self.validate(payload,selected)

    def test_complete_four_stages_remain_non_cost_observations(self):
        for kind in ("projection","attention","dense_ffn","block"):
            with self.subTest(kind=kind):
                selected=cuda_case(kind)
                result=self.validate(selected=selected)
                self.assertEqual([p["label"] for p in result["checkpoints"]],list(SNAPSHOT_LABELS))
                self.assertEqual(result["max_observed_used_bytes"],2100)
                self.assertEqual(result["max_observed_increase_from_context_bytes"],1100)
                self.assertEqual(result["known_allocations"],memory_native(selected)["dense_validation"]["memory_accounting"])
                self.assertIsNone(result["workspace_bytes"])
                self.assertIsNone(result["process_peak_bytes"])
                self.assertIsNone(result["model_capacity_under_declared_budget"])
                self.assertFalse(result["physical_capacity_verified"])
                self.assertFalse(result["workspace_bound_known"])
                self.assertFalse(result["timing_usable_as_cost"])

    def test_negative_device_wide_change_is_preserved(self):
        result=self.validate()
        self.assertEqual(result["checkpoints"][0]["delta_used_from_context_bytes"],0)
        self.assertEqual(result["checkpoints"][-1]["delta_used_from_context_bytes"],-100)

    def test_runtime_total_memory_is_kept_as_its_own_api_observation(self):
        payload=memory_native()
        for checkpoint in payload["snapshots"]:
            checkpoint["total_bytes"]-=1024
            checkpoint["free_bytes"]-=1024
        result=self.validate(payload)
        self.assertEqual(result["device_info"]["total_memory_bytes"],1048576)
        self.assertEqual(result["checkpoints"][0]["total_bytes"],1047552)
        self.assertFalse(result["physical_capacity_verified"])

    def test_validation_never_mutates_original_raw_evidence(self):
        payload=memory_native()
        before=copy.deepcopy(payload)
        self.validate(payload)
        self.assertEqual(payload,before)

    def test_missing_extra_duplicate_or_reordered_snapshot_rejects(self):
        mutations=(lambda p:p["snapshots"].pop(),
            lambda p:p["snapshots"].append(copy.deepcopy(p["snapshots"][-1])),
            lambda p:p["snapshots"][1].update(label=SNAPSHOT_LABELS[0]),
            lambda p:p["snapshots"].reverse(),
            lambda p:p.update(snapshot_count=6))
        for index,edit in enumerate(mutations):
            with self.subTest(index=index):self.reject(edit)

    def test_each_snapshot_checks_free_total_and_used_arithmetic(self):
        for field,value in (("free_bytes",1048577),("total_bytes",0),
                             ("total_bytes",1048575),("used_bytes",1),
                             ("free_bytes",-1),("used_bytes",-1)):
            with self.subTest(field=field,value=value):
                self.reject(lambda p:p["snapshots"][2].update({field:value}))

    def test_bool_is_not_a_memory_count_elapsed_time_or_device_index(self):
        for field in ("elapsed_ns","device_index","free_bytes","total_bytes","used_bytes"):
            with self.subTest(field=field):
                self.reject(lambda p:p["snapshots"][2].update({field:True}))
        for field in ("snapshot_count","max_observed_used_bytes","cuda_device","dense_validation_exit_code"):
            with self.subTest(field=field):self.reject(lambda p:p.update({field:True}))

    def test_checkpoint_elapsed_time_is_nonnegative_and_strictly_increasing(self):
        for value in (-1,100,200):
            with self.subTest(value=value):
                self.reject(lambda p:p["snapshots"][2].update(elapsed_ns=value))

    def test_observation_identity_must_bind_dense_validation_and_all_snapshots(self):
        self.reject(lambda p:p["device_info"].update(uuid="different-device"))
        self.reject(lambda p:p["device_info"].update(index=1))
        self.reject(lambda p:p["snapshots"][2].update(device_index=1))
        self.reject(lambda p:p.update(backend="CUDA1"))
        self.reject(lambda p:p.update(cuda_device=1))

    def test_maximum_is_recomputed_from_all_seven_checkpoints(self):
        self.reject(lambda p:p.update(max_observed_used_bytes=2000))
        self.reject(lambda p:p.update(max_observed_used_bytes=1048576))

    def test_workspace_and_peak_claims_cannot_be_derived_from_free_memory(self):
        for field,value in (("workspace_bytes",0),("workspace_bound","measured"),
            ("process_peak_bytes",2100),("capacity_verified",True),
            ("peak_measurement_claimed",True),("process_attribution_verified",True),
            ("timing_usable_as_cost",True),("timing_usable_as_cost",0),
            ("max_observed_used_bytes_scope","process_peak")):
            with self.subTest(field=field):self.reject(lambda p:p.update({field:value}))

    def test_context_scope_source_and_graph_mode_are_required(self):
        for field,value in (("schema","ggml_dense_layer_cuda_probe/1"),
            ("purpose","performance_measurement"),("snapshot_scope","process_allocations"),
            ("source","NVML"),("cuda_graphs","enabled"),("hardware_compute_performed",False)):
            with self.subTest(field=field):self.reject(lambda p:p.update({field:value}))

    def test_failed_or_tampered_dense_validation_is_not_memory_evidence(self):
        self.reject(lambda p:p.update(dense_validation_exit_code=1))
        self.reject(lambda p:p["dense_validation"].update(q=4))
        self.reject(lambda p:p["dense_validation"].update(seed=18))
        self.reject(lambda p:p["dense_validation"]["backend_execution_check"].update(cpu_fallback_used=True))
        self.reject(lambda p:p["dense_validation"].update(workspace_bytes=0))

    def test_missing_fields_are_unknown_never_defaulted(self):
        for field in ("workspace_bytes","process_peak_bytes","device_info","snapshot_count",
                      "max_observed_used_bytes","snapshots","dense_validation"):
            with self.subTest(field=field):self.reject(lambda p:p.pop(field))


class CudaMemoryProfileEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory(prefix="cuda-memory-evidence-test-")
        self.addCleanup(self.folder.cleanup)
        self.root=Path(self.folder.name)
        (self.root/"raw").mkdir()
        self.cases=[cuda_case(id="synthetic_prefill"),
            cuda_case(id="synthetic_decode",phase="decode",tokens=1,visible_history=3)]
        self.binary=str(self.root/("ggml-dense-layer-cuda-memory-probe.exe" if os.name=="nt" else "ggml-dense-layer-cuda-memory-probe"))
        self.binary_hash=hashlib.sha256(b"synthetic memory binary identity not executed").hexdigest()
        self.manifest=dict(schema="dense_cuda_memory_manifest/1",profile_id="synthetic-memory-profile",
            suite=suite(self.cases),protocol=dict(cuda_device=0,cuda_graphs="disabled",
                sequential_execution=True,warmup=2,samples=3,seed=17),
            software=dict(runtime_sha256={self.binary:self.binary_hash},
                child_flags={"GGML_CUDA_DISABLE_GRAPHS":"1"}))
        self.records=[]
        self.raws=[]
        for index,selected in enumerate(self.cases):
            payload=memory_native(selected)
            raw=dict(case=selected,command=[self.binary]+contract.native_arguments(selected,2,3,17),
                returncode=0,timed_out=False,stdout=json.dumps(payload),stderr="synthetic not executed")
            self.raws.append(raw)
            self.records.append(dict(case_id=selected["id"],case=selected,status="observed",
                signature=contract.measurement_signature(selected,self.binary_hash),native=payload,
                observation=contract.validate_memory_probe(selected,payload,protocol_args()),
                raw_output=f"raw/{index+1:03d}.json",raw_sha256="pending"))
            self.write_raw(index)
        self.report=dict(schema="dense_cuda_memory_profile/1",purpose="memory_diagnostics_only",
            profile_id=self.manifest["profile_id"],complete=True,workspace_bytes=None,
            workspace_bound_known=False,physical_capacity_verified=False,
            model_capacity_under_declared_budget=None,timing_usable_as_cost=False,
            device_info=copy.deepcopy(self.records[0]["native"]["device_info"]),records=self.records)
        self.save()

    def dump(self,path,value):
        path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")

    def write_raw(self,index):
        path=self.root/self.records[index]["raw_output"]
        self.dump(path,self.raws[index])
        self.records[index]["raw_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()

    def save(self):
        self.dump(self.root/"manifest.json",self.manifest)
        self.dump(self.root/"memory_report.json",self.report)
        self.dump(self.root/"plan.json",self.cases)

    def load(self):
        self.save()
        return contract.load_memory_profile(self.root)

    def test_portable_profile_revalidates_native_math_and_raw_source_hashes(self):
        result=self.load()
        self.assertEqual(result["device_info"],self.report["device_info"])
        self.assertEqual(result["records"],self.records)
        self.assertEqual(result["source_hashes"],{
            "manifest_sha256":hashlib.sha256((self.root/"manifest.json").read_bytes()).hexdigest(),
            "memory_report_sha256":hashlib.sha256((self.root/"memory_report.json").read_bytes()).hexdigest()})
        self.assertFalse(result["workspace_bound_known"])
        self.assertFalse(result["physical_capacity_verified"])
        self.assertFalse(Path(self.binary).exists())

    def test_failed_or_incomplete_profiles_cannot_bind_a_budget(self):
        for field,value in (("complete",False),("workspace_bytes",0),
            ("workspace_bound_known",True),("physical_capacity_verified",True),
            ("model_capacity_under_declared_budget",True),("timing_usable_as_cost",True)):
            with self.subTest(field=field):
                original=copy.deepcopy(self.report[field]);self.report[field]=value
                with self.assertRaises(ValueError):self.load()
                self.report[field]=original
        self.records[0]["status"]="failed"
        with self.assertRaises(ValueError):self.load()

    def test_report_profile_and_device_identity_are_bound_to_sources(self):
        for field,value in (("profile_id","different-profile"),
            ("device_info",{**self.report["device_info"],"uuid":"different-device"})):
            with self.subTest(field=field):
                original=copy.deepcopy(self.report[field]);self.report[field]=value
                with self.assertRaises(ValueError):self.load()
                self.report[field]=original

    def test_every_declared_case_has_one_successful_record(self):
        original=copy.deepcopy(self.records)
        self.records.pop()
        with self.assertRaises(ValueError):self.load()
        self.records[:]=original
        self.records[1]=copy.deepcopy(self.records[0])
        with self.assertRaises(ValueError):self.load()

    def test_raw_relative_path_cannot_escape_profile(self):
        for name in ("../outside.json",str(self.root.parent/"absolute.json")):
            with self.subTest(name=name):
                self.records[0]["raw_output"]=name
                with self.assertRaises(ValueError):self.load()

    def test_raw_hash_and_embedded_output_must_match(self):
        path=self.root/self.records[0]["raw_output"]
        path.write_bytes(path.read_bytes()+b" ")
        with self.assertRaises(ValueError):self.load()
        self.write_raw(0)
        self.raws[0]["stdout"]="{}"
        self.write_raw(0)
        with self.assertRaises(ValueError):self.load()

    def test_forged_observation_cannot_turn_delta_into_workspace(self):
        self.records[0]["observation"]["workspace_bytes"]=1100
        with self.assertRaises(ValueError):self.load()

    def test_signature_protocol_and_binary_identity_are_not_decorative(self):
        original=copy.deepcopy(self.records[0]["signature"])
        self.records[0]["signature"]["physical_contract_sha256"]="0"*64
        with self.assertRaises(ValueError):self.load()
        self.records[0]["signature"]=original
        self.manifest["software"]["runtime_sha256"][self.binary]="0"*64
        with self.assertRaises(ValueError):self.load()

    def test_graph_disable_environment_and_sequential_protocol_are_binding(self):
        self.manifest["software"]["child_flags"]["GGML_CUDA_DISABLE_GRAPHS"]="0"
        with self.assertRaises(ValueError):self.load()
        self.manifest["software"]["child_flags"]["GGML_CUDA_DISABLE_GRAPHS"]="1"
        self.manifest["protocol"]["sequential_execution"]=False
        with self.assertRaises(ValueError):self.load()

    def test_raw_command_must_match_memory_binary_math_device_and_protocol(self):
        original=copy.deepcopy(self.raws[0])
        for change in ("device","binary","seed"):
            with self.subTest(change=change):
                self.raws[0]=copy.deepcopy(original)
                command=self.raws[0]["command"]
                if change=="binary":command[0]=str(self.root/"ggml-dense-layer-cuda-probe.exe")
                else:
                    key="--cuda-device" if change=="device" else "--seed"
                    command[command.index(key)+1]="1" if change=="device" else "18"
                self.write_raw(0)
                with self.assertRaises(ValueError):self.load()
        self.raws[0]=original;self.write_raw(0)

    def test_native_nonzero_exit_or_timeout_is_preserved_but_not_accepted(self):
        for field,value in (("returncode",1),("timed_out",True)):
            with self.subTest(field=field):
                original=copy.deepcopy(self.raws[0]);self.raws[0][field]=value
                self.write_raw(0)
                with self.assertRaises(ValueError):self.load()
                self.raws[0]=original;self.write_raw(0)

    def test_saved_plan_cannot_change_declared_history(self):
        self.save()
        modified=copy.deepcopy(self.cases);modified[1]["visible_history"]=4
        self.dump(self.root/"plan.json",modified)
        with self.assertRaises(ValueError):contract.load_memory_profile(self.root)


class CudaMemoryRunnerEvidenceTests(unittest.TestCase):
    def run_fake(self,payload,*,exitcode=0,failure=None):
        selected=cuda_case()
        with tempfile.TemporaryDirectory(prefix="cuda-memory-runner-test-") as folder:
            root=Path(folder)
            binary=root/("ggml-dense-layer-cuda-memory-probe.exe" if os.name=="nt" else "ggml-dense-layer-cuda-memory-probe")
            binary.write_bytes(b"synthetic memory probe never executed")
            for source_name in ("run_dense_cuda_memory.py","cuda_memory_contract.py","run_dense_cuda_measurements.py",
                "native/dense_layer_cuda_probe.cpp","native/dense_layer_cuda_memory_probe.cpp","native/CMakeLists.txt"):
                path=root/source_name;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(b"synthetic source identity")
            source=root/"suite.json"
            source.write_text(json.dumps(suite([selected])),encoding="utf-8",newline="\n")
            output=root/"output"
            stdout=payload if isinstance(payload,str) else json.dumps(payload)
            child=subprocess.CompletedProcess([],exitcode,stdout,"synthetic diagnostic")
            arguments=["--suite",str(source),"--native-bin",str(root),"--output",str(output),
                "--cuda-device","0","--cuda-graphs","disabled","--samples","3","--warmup","2"]
            with (mock.patch.object(runner,"ROOT",root),
                mock.patch.object(runner,"environment",return_value={"synthetic":True}),
                mock.patch.object(runner.subprocess,"run",return_value=child,side_effect=failure) as probe,
                contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO())):
                code=runner.main(arguments)
            self.assertEqual(probe.call_count,1)
            self.assertFalse((root/".measurement.lock").exists())
            return dict(code=code,report=json.loads((output/"memory_report.json").read_text()),
                raw=json.loads((output/"raw/001.json").read_text()),
                manifest=json.loads((output/"manifest.json").read_text()),
                kwargs=probe.call_args.kwargs,
                files={p.relative_to(output).as_posix():p.read_bytes() for p in output.rglob("*") if p.is_file()})

    def test_memory_run_has_separate_evidence_and_never_a_cost_or_capacity_claim(self):
        result=self.run_fake(memory_native())
        self.assertEqual(result["code"],0)
        report=result["report"]
        self.assertTrue(report["complete"])
        self.assertFalse(report["timing_usable_as_cost"])
        self.assertFalse(report["physical_capacity_verified"])
        self.assertIsNone(report["workspace_bytes"])
        self.assertIsNone(report["model_capacity_under_declared_budget"])
        row=report["records"][0]
        self.assertEqual(row["status"],"observed")
        self.assertEqual(row["observation"]["checkpoints"][-1]["delta_used_from_context_bytes"],-100)
        self.assertEqual(row["raw_sha256"],hashlib.sha256(result["files"]["raw/001.json"]).hexdigest())
        self.assertEqual(json.loads(result["raw"]["stdout"]),row["native"])
        self.assertNotIn("measurements.json",result["files"])
        self.assertEqual(result["kwargs"]["env"]["GGML_CUDA_DISABLE_GRAPHS"],"1")
        self.assertNotIn("LLAMA_VIZ_TIMING",result["kwargs"]["env"])
        for name,contents in result["files"].items():
            if name.endswith((".json",".csv")):self.assertNotIn(b"\r\n",contents)

    def test_native_failure_preserves_raw_stdout_stderr_and_is_incomplete(self):
        payload=memory_native()
        result=self.run_fake(payload,exitcode=1)
        self.assertEqual(result["code"],1)
        self.assertFalse(result["report"]["complete"])
        self.assertEqual(result["report"]["records"][0]["status"],"failed")
        self.assertEqual(json.loads(result["raw"]["stdout"]),payload)
        self.assertEqual(result["raw"]["stderr"],"synthetic diagnostic")

    def test_memory_timeout_preserves_partial_output_and_releases_measurement_lock(self):
        failure=subprocess.TimeoutExpired(["synthetic"],1,output=b"partial output",stderr=b"partial stderr")
        result=self.run_fake("",failure=failure)
        self.assertEqual(result["code"],1)
        self.assertEqual(result["report"]["records"][0]["status"],"timed_out")
        self.assertFalse(result["report"]["complete"])
        self.assertTrue(result["raw"]["timed_out"])
        self.assertEqual(result["raw"]["stdout"],"partial output")
        self.assertEqual(result["raw"]["stderr"],"partial stderr")

    def test_invalid_claim_keeps_original_native_output_and_fails_record(self):
        payload=memory_native(workspace_bytes=0)
        result=self.run_fake(payload)
        self.assertEqual(result["code"],1)
        self.assertEqual(result["report"]["records"][0]["status"],"failed")
        self.assertEqual(json.loads(result["raw"]["stdout"]),payload)

    def test_cuda_device_has_no_silent_default(self):
        arguments=["--suite","does-not-exist.json","--native-bin","absent", "--output","absent-output",
            "--cuda-graphs","disabled"]
        with (mock.patch.object(runner.subprocess,"run") as probe,
            contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as failure):
            runner.main(arguments)
        self.assertEqual(failure.exception.code,2)
        probe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
