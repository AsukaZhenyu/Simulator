"""Exact public ModelPlan bridge contracts. Timing fixtures here are synthetic.

Optional HIGHLEVEL_PROBE / HIGHLEVEL_RUNTIME_DIR enable untimed CPU numerical
checks. These tests never start a hardware timing campaign or claim accuracy.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "profiling", ROOT / "modeling", ROOT / "mapping"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from highlevel_accuracy.bridge import (BOUNDARY, BridgeError, export_plan, reference_ports,
                                      sha, validate_program, verify_native)
from highlevel_accuracy.cases import build_case
from highlevel_accuracy.profiles import (compare_heldout, freeze_profile, measured_scenario,
    run_plan_with_profile, simulate_frozen, validate_ledger)
from highlevel_accuracy.run import case_specs, native_run


def reseal(program):
    program["identity"]={key:sha(program[key]) for key in program["identity"]}
    program.pop("program_sha256",None);program["program_sha256"]=sha(program)
    return program


def fake_native(program,mode="components"):
    tensors={t["id"]:t for t in program["graph"]["workload"]["tensors"]}
    result=dict(format_version="highlevel-accuracy-native-v1",program_sha256=program["program_sha256"],
        boundary_contract=BOUNDARY,actions=program["actions"],ports=reference_ports(program),copy_values_equal=True,
        observed_runtime_flags=program["required_runtime_flags"],
        final_execution_ports_captured_outside_timing=True,workspace_scope="unit_test_not_native_memory",
        copies=[dict(index=i,tensor_id=a["target_id"],src=a["src"],dst=a["dst"],storage_bytes=tensors[a["target_id"]]["storage_bytes"])
                for i,a in enumerate(program["actions"]) if a["kind"]=="COPY"],
        mode=mode,samples=3,warmup=0,timing_scope="unit_test_synthetic_not_measured",
        components=[dict(action_index=i,action=a,samples_ns=[100,110,120],workspace_bytes=512)
                    for i,a in enumerate(program["actions"]) if a["kind"] in {"COPY","COMPUTE"}] if mode=="components" else [],
        total_samples_ns=[] if mode=="components" else [1000,1100,1200])
    if mode=="components":
        result["independent_processes"]=[dict(process_index=i,components=deepcopy(result["components"])) for i in range(3)]
        for row in result["components"]:row["samples_ns"]*=3
        result["samples"]*=3
    return result


class HighLevelAccuracyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/"highlevel_accuracy");root=Path(cls.temp.name)
        kwargs=dict(tokens=4,history=3,devices=("cpu",)*3)
        cls.macro_plan=build_case(root/"macro",**kwargs).compile()
        cls.split_plan=build_case(root/"split",**kwargs,attention_stages=True,attention_devices=("cpu",)*4,
            ffn_cut=8,ffn_devices=("cpu",)*2,norm_device="cpu",merge_device="cpu").compile()
        cls.mixed_plan=build_case(root/"mixed",tokens=1,history=3,attention_stages=True,ffn_cut=12).compile()
        cls.macro=export_plan(cls.macro_plan)
        cls.split=export_plan(cls.split_plan)
        cls.mixed=export_plan(cls.mixed_plan)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_actual_highlevel_frontend_and_entire_graph_exported(self):
        p=self.split
        self.assertEqual(p["frontend_ir"]["format_version"],"hierarchical-strategy-0.2")
        self.assertEqual(len(p["bindings"]),len(p["graph"]["workload"]["operations"]))
        self.assertFalse(p["original_simulation"]["measured_accuracy_claimed"])
        kinds=[b["kind"] for b in p["bindings"]]
        self.assertEqual(kinds.count("ffn_norm"),1);self.assertEqual(kinds.count("ffn_residual"),1)
        self.assertEqual(kinds.count("ffn_sum"),1);self.assertEqual(kinds.count("ffn_gate"),2)
        self.assertTrue({"attention_qk","attention_softmax","attention_pv","attention_output"}<=set(kinds))

    def test_full_macro_matches_partitioned_norm_residual_and_kv(self):
        macro,split=reference_ports(self.macro),reference_ports(self.split)
        for tid in set(macro)&set(split):
            np.testing.assert_allclose(macro[tid],split[tid],atol=2e-6,rtol=2e-6,err_msg=tid)
        residual=next(b for b in self.split["bindings"] if b["kind"]=="ffn_residual")
        expected=np.asarray(split[residual["operands"]["x0"]])+np.asarray(split[residual["operands"]["x1"]])
        np.testing.assert_allclose(split[residual["outputs"]["result"]],expected,atol=1e-7)

    def test_changed_operand_rejected_even_if_resigned(self):
        p=deepcopy(self.split);b=next(b for b in p["bindings"] if b["kind"]=="ffn_gate")
        b["operands"]["x1"]=next(b for b in p["bindings"] if b["kind"]=="ffn_up")["outputs"]["result"]
        with self.assertRaisesRegex(BridgeError,"operands differ"):validate_program(reseal(p))

    def test_changed_shape_rejected_even_if_resigned(self):
        p=deepcopy(self.macro);tid=next(b for b in p["bindings"] if b["kind"]=="projection")["outputs"]["q"]
        tensor=next(t for t in p["graph"]["workload"]["tensors"] if t["id"]==tid)
        tensor["ne"]=[8,4,2,1];tensor["nb"]=[4,32,128,256]
        with self.assertRaisesRegex(BridgeError,"physical operation boundary"):validate_program(reseal(p))

    def test_changed_boundary_rejected(self):
        p=deepcopy(self.macro);p["boundary_contract"]="compact_ffn_without_norm"
        with self.assertRaisesRegex(BridgeError,"boundary contract"):validate_program(reseal(p))

    def test_unsupported_transform_is_rejected(self):
        p=deepcopy(self.macro);p["bindings"][0]["kind"]="attention_core"
        with self.assertRaisesRegex(BridgeError,"unsupported"):validate_program(reseal(p))

    def test_public_compiled_pure_attention_transform_is_rejected(self):
        folder=Path(self.temp.name)/"unsupported-pure-attention"
        plan=build_case(folder,tokens=1,history=3,devices=("cpu",)*3)
        layer=json.loads((folder/"layer.json").read_text());costs=json.loads((folder/"costs.json").read_text())
        plan.transform("dense",dict(format_version="dense-attention-execution-0.1",variant="pure_attention",
            provenance="synthetic_execution",invocation_files=[str(folder/"invocation.json")],frontend_config=layer["physical"]["config"],
            batch_contract="independent_requests_not_fused_batch_kernel",subbatches=[dict(id="one",request_ids=["r0"],attention_device="cpu")],
            entries=[dict(kind=kind,device=device,phase="decode",batch=1,tokens=1,visible_history=3,duration_ns=1,workspace_bytes=0)
                     for kind in ("attention_core","attention_output") for device in ("cpu","gpu")],
            transfer_entries=[{k:v for k,v in row.items() if k!="provenance"} for row in costs["transfer_entries"]]))
        plan.place("attention_core","cpu");plan.place("attention_output","cpu")
        compiled=plan.compile()
        with self.assertRaisesRegex(BridgeError,"unsupported execution transform"):export_plan(compiled)

    def test_native_port_copy_and_trace_changes_are_rejected(self):
        p=self.mixed;n=fake_native(p)
        verify_native(p,n)
        missing=deepcopy(n);missing["ports"].pop(next(iter(missing["ports"])))
        with self.assertRaisesRegex(BridgeError,"port inventory"):verify_native(p,missing)
        copied=deepcopy(n);copied["copies"][0]["storage_bytes"]+=4
        with self.assertRaisesRegex(BridgeError,"copy identity"):verify_native(p,copied)
        changed=deepcopy(n);changed["actions"]=deepcopy(n["actions"]);changed["actions"][0]["kind"]="ADVANCE"
        with self.assertRaisesRegex(BridgeError,"action trace"):verify_native(p,changed)

    def test_profile_reprices_actual_modelplan_without_synthetic_fallback(self):
        p=self.mixed;n=fake_native(p);ledger=freeze_profile(p,n,executable_sha256="0"*64,evidence={"test_only":True})
        direct=simulate_frozen(p,ledger);highlevel=run_plan_with_profile(self.mixed_plan,p,ledger)
        self.assertEqual(direct["makespan_ns"],highlevel["makespan_ns"])
        self.assertTrue(highlevel["actual_ModelPlan_policy_rerun"])
        self.assertFalse(highlevel["synthetic_costs_used"])
        incomplete=deepcopy(n);incomplete["components"].pop()
        with self.assertRaisesRegex(BridgeError,"missing exact component"):freeze_profile(p,incomplete,executable_sha256="0"*64,evidence={"test_only":True})

    def test_profile_and_heldout_use_disjoint_data_after_freeze(self):
        p=self.macro;ledger=freeze_profile(p,fake_native(p),executable_sha256="0"*64,evidence={"test_only":True},frozen_at="2026-01-01T00:00:00+00:00")
        heldout=export_plan(self.macro_plan,seed=99999)
        result=compare_heldout(heldout,ledger,fake_native(heldout,"replay"),heldout_started_at="2026-01-02T00:00:00+00:00",executable_sha256="0"*64)
        self.assertTrue(result["baseline_frozen_before_heldout"])
        with self.assertRaisesRegex(BridgeError,"seed/data"):compare_heldout(p,ledger,fake_native(p,"replay"),heldout_started_at="2026-01-02T00:00:00+00:00",executable_sha256="0"*64)
        with self.assertRaisesRegex(BridgeError,"after baseline"):compare_heldout(heldout,ledger,fake_native(heldout,"replay"),heldout_started_at="2025-01-02T00:00:00+00:00",executable_sha256="0"*64)

    def test_changed_action_cannot_reuse_matching_shape_profile(self):
        p=self.mixed;ledger=freeze_profile(p,fake_native(p),executable_sha256="0"*64,evidence={"test_only":True})
        changed=deepcopy(p);a=next(a for a in changed["actions"] if a["kind"]=="COPY");a["src"],a["dst"]=a["dst"],a["src"]
        with self.assertRaisesRegex(BridgeError,"exactly match"):validate_ledger(reseal(changed),ledger)

    def test_representative_suite_has_close_cuts_and_macro_paths(self):
        rows=case_specs("representative")
        self.assertTrue(any(row["kwargs"].get("devices")==["cpu"]*3 for row in rows))
        self.assertTrue(any(row["kwargs"].get("devices")==["gpu"]*3 for row in rows))
        for hidden in (64,128,256):
            cuts=[row["kwargs"]["ffn_cut"] for row in rows if row["kwargs"]["hidden"]==hidden and "ffn_cut" in row["kwargs"]]
            self.assertEqual(cuts,[hidden-4,hidden,hidden+4])

    def test_unstable_source_process_is_preserved_and_fails_fixed_gate(self):
        p=self.macro;n=fake_native(p);n["independent_processes"][0]["components"][0]["samples_ns"]=[1,100,1000]
        index=n["independent_processes"][0]["components"][0]["action_index"]
        next(row for row in n["components"] if row["action_index"]==index)["samples_ns"]=sum([next(row for row in process["components"] if row["action_index"]==index)["samples_ns"] for process in n["independent_processes"]],[])
        ledger=freeze_profile(p,n,executable_sha256="0"*64,evidence={"test_only":True})
        self.assertFalse(ledger["source_quality_passed"])
        self.assertEqual(ledger["fixed_quality_gates"]["within_process_cv_max"],0.15)
        self.assertEqual(ledger["fixed_quality_gates"]["between_process_median_cv_max"],0.20)
        self.assertEqual(len(ledger["independent_process_statistics"]),3)

    def test_timeout_and_launch_failure_preserve_raw_execution(self):
        for label,error in (("timeout",subprocess.TimeoutExpired(["probe"],600,output=b"partial",stderr=b"failure")),("launch",FileNotFoundError("missing binary"))):
            directory=Path(self.temp.name)/("failed-"+label)
            with patch("highlevel_accuracy.run.subprocess.run",side_effect=error):
                with self.assertRaisesRegex(BridgeError,"raw evidence preserved"):
                    native_run(Path("missing.exe"),Path("program.json"),"components",0,3,directory,[])
            record=json.loads((directory/"execution.json").read_text())
            self.assertIsNone(record["returncode"])
            self.assertEqual(record["timed_out"],label=="timeout")
            self.assertTrue(record["stderr"])

    @unittest.skipUnless(os.environ.get("HIGHLEVEL_PROBE"),"optional untimed real GGML CPU correctness check")
    def test_untimed_native_cpu_full_and_partitioned(self):
        env=os.environ.copy();env["PATH"]=env.get("HIGHLEVEL_RUNTIME_DIR","")+os.pathsep+env.get("PATH","")
        for label,program in (("macro",self.macro),("partition",self.split)):
            directory=Path(self.temp.name)/("native-"+label);directory.mkdir()
            source=directory/"program.json";source.write_text(json.dumps(program),encoding="utf-8")
            output=directory/"native.json"
            result=subprocess.run([env["HIGHLEVEL_PROBE"],str(source),"validate","0","1",str(output)],env=env,capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stderr)
            native=json.loads(output.read_text());self.assertEqual(native["total_samples_ns"],[])
            self.assertGreater(verify_native(program,native)["independently_checked_ports"],0)


if __name__=="__main__":unittest.main()
