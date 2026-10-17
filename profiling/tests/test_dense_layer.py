"""Full Dense runner/export boundaries; all subprocess fixtures are synthetic."""
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
import run_dense_layer_measurements as runner
import export_dense_layer as exporter
from llm_infer_model.m2.dense_contract import (CONTRACT, EXCLUDED, VARIANTS, build_plan,
    operations, ports, resident_accounting, stage_weight_roles, validate_case, validate_config, weight_shapes, DTYPES)
from llm_infer_model.m2.costs import MacroShape, parse_cost_table
from llm_infer_model.m2.spec import parse_layer_document
from llm_infer_model.tensor.spec import ggml_contiguous_layout, SpecError
from profile_contract import summarize_samples


def config():
    return validate_config({"hidden":16,"intermediate":32,"heads":4,"kv_heads":2,"head_dim":4,
        "attention_gate":"sigmoid","qk_norm":"rms","rope_mode":"normal","rope_dims":4,
        "rope_base":10000,"norm_epsilon":1e-6,"weight_dtypes":{role:"f32" for role in
            ("input_norm","q","k","v","gate_proj","q_norm","k_norm","output","post_attention_norm","gate","up","down")}})


def case(kind="attention", **changes):
    raw = {"id":kind,"kind":kind,"phase":"prefill","batch":1,"tokens":3,"visible_history":0,
           "config":config(),"threads":2}
    raw.update(changes)
    return validate_case(raw)


def suite(cases):
    return {"schema_version":"dense_layer_measurement_suite_v1","cases":cases}


def layer_document():
    c = config()
    shapes = weight_shapes(c)
    weights = []
    for role, shape in shapes.items():
        _, size = ggml_contiguous_layout(DTYPES[c["weight_dtypes"][role]], (*shape, *([1]*(4-len(shape)))))
        weights.append({"id":"W_"+role,"dtype":c["weight_dtypes"][role],"shape":list(shape),
                        "storage_bytes":size,"initial_location":"dram"})
    return {"format_version":"m2-layer-0.3","id":"synthetic-full-dense-fixture",
        "provenance":"synthetic_layer","hidden_size":16,"hidden_bytes_per_token":64,
        "kv_bytes_per_token":64,"attention":"full_causal","ffn":"dense","weights":weights,
        "physical":{"contract":CONTRACT,"scope":"complete_declared_decoder_block",
            "input_boundary":"hidden_activation","port_layout":"materialized_stage_ports",
            "kv_allocation":"logical_blocks","activation_dtype":"f32","kv_dtype":"f32",
            "config":c,"threads":2,"excluded_operations":[],
            "weight_groups":{kind:{role:"W_"+role for role in stage_weight_roles(c,kind)}
                             for kind in ("projection","attention","dense_ffn")}},
        "tasks":[{"id":"projection","kind":"projection","depends_on":[]},
                 {"id":"attention","kind":"attention","depends_on":["projection"]},
                 {"id":"dense_ffn","kind":"dense_ffn","depends_on":["attention"]}]}


def native(case, **changes):
    mem = resident_accounting(case["config"],case["kind"],case["tokens"],case["visible_history"])
    inputs, outputs = ports(case["config"], case["kind"], case["tokens"])
    port_accounting = {name:{"dtype":port["dtype"],"shape":list(port["ne"][:3]),"ne":list(port["ne"]),
        "bytes":__import__("math").prod(port["ne"])*4,"category":category}
        for category,declared in (("input",inputs),("output",outputs)) for name,port in declared.items()}
    if case["kind"] in ("attention","block"):
        for name in ("key_cache","value_cache"):
            port_accounting[name] = {"dtype":"f32","shape":[case["config"]["head_dim"],
                case["visible_history"]+case["tokens"],case["config"]["kv_heads"]],
                "bytes":mem["kv_cache_bytes"]//2,"category":"kv_cache",
                "ne":[case["config"]["head_dim"],case["visible_history"]+case["tokens"],case["config"]["kv_heads"],1]}
    timing = 3000 if case["kind"] == "block" else 1000
    result = {"schema":"ggml_dense_layer_probe/1","kind":"dense_layer","contract":CONTRACT,
        "stage":case["kind"],"backend":"CPU","batch":1,"config":case["config"],
        "q":case["tokens"],"history":case["visible_history"],"activation_dtype":"f32","kv_dtype":"f32",
        "implementation_variant":VARIANTS[case["kind"]],"included":operations(case["config"],case["kind"]),
        "excluded":list(EXCLUDED),"timing":"steady_clock_graph_compute_including_sync",
        "scope":"resident_repeated_complete_declared_decoder_block","setup_compute_repetitions":1,
        "threads":2,"threads_requested":2,"threads_applied":2,"warmup":2,"samples":3,"seed":17,
        "upstream_setup_compute_repetitions":0 if case["kind"] == "block" else 1,
        "thread_management":"persistent_ggml_threadpool","input_lineage":"same_seed_complete_upstream_preparation_outside_timing",
        "position_dtype":"i32","port_accounting":port_accounting,
        "all_outputs_finite":True,"reference_check":{"passed":True},
        "kv_append_check":{"passed":True,"append_exact":True,"history_unchanged":True}
            if case["kind"] in ("attention","block") else None,
        "stage_port_digests":{port:"0123456789abcdef" for port in mem["output_ports_bytes"]},
        "port_digest_algorithm":"fnv1a64_native_tensor_bytes",
        "input_port_digests":{port:"0123456789abcdef" for port in mem["input_ports_bytes"]},
        "wall_ns":[timing,timing+1,timing-1],"workspace_bytes":128,
        "workspace_source":"ggml_gallocr_internal_buffer_plus_cpu_graph_plan",
        "cpu_cplan_work_size_bytes":32,
        "memory_accounting":{"resident_logical_bytes":mem["resident_logical_bytes"],
            "resident_allocated_bytes":mem["resident_logical_bytes"]+32,
            "internal_allocated_buffer_bytes":96,"cpu_cplan_work_size_bytes":32,"workspace_bytes":128,
            "workspace_bound":"conservative_allocated_capacity",
            "workspace_scope":"internal_reusable_allocator_capacity_including_alignment_plus_cpu_cplan_scratch_excludes_resident_tensors_host_reference_and_runtime_overheads"},**mem}
    result.update(changes)
    return result


class DenseRunnerTests(unittest.TestCase):
    def run_fake(self, payload, *, selected=None, exitcode=0, failure=None):
        selected = selected or case()
        with tempfile.TemporaryDirectory(prefix="dense-runner-test-") as folder:
            root = Path(folder)
            binary = root / ("ggml-dense-layer-probe.exe" if os.name == "nt" else "ggml-dense-layer-probe")
            binary.write_bytes(b"synthetic probe never executed")
            path = root / "suite.json"
            path.write_text(json.dumps(suite([selected])),encoding="utf-8",newline="\n")
            out = root / "output"
            text = payload if isinstance(payload,str) else json.dumps(payload)
            child = subprocess.CompletedProcess([],exitcode,text,"synthetic diagnostic")
            arguments = ["--suite",str(path),"--native-bin",str(root),"--output",str(out),
                         "--threads","2","--samples","3","--warmup","2"]
            with (mock.patch.object(runner,"ROOT",root),
                  mock.patch.object(runner,"environment",return_value={"mock":True}),
                  mock.patch.object(runner.subprocess,"run",return_value=child,side_effect=failure) as probe,
                  contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO())):
                code = runner.main(arguments)
            self.assertEqual(probe.call_count,1)
            self.assertFalse((root/".measurement.lock").exists())
            return {"code":code,"rows":json.loads((out/"measurements.json").read_text()),
                    "raw":json.loads((out/"raw/001.json").read_text()),
                    "files":{str(p.relative_to(out)):p.read_bytes() for p in out.rglob("*") if p.is_file()}}

    def test_complete_config_and_raw_samples_survive(self):
        for kind in ("projection","attention","dense_ffn","block"):
            with self.subTest(kind=kind):
                c = case(kind)
                result = self.run_fake(native(c),selected=c)
                self.assertEqual(result["code"],0)
                row = result["rows"][0]
                self.assertEqual(json.loads(result["raw"]["stdout"])["wall_ns"],row["native"]["wall_ns"])
                self.assertTrue(row["signature"]["native_binary_sha256"])
                self.assertTrue(row["eligibility"]["complete_declared_block_variant"])

    def test_materialized_port_shape_dtype_and_category_must_match(self):
        for key,value in (("dtype","f16"),("shape",[4,3,4]),("category","output"),("bytes",1)):
            c = case("attention")
            bad = copy.deepcopy(native(c)["port_accounting"])
            bad["hidden"][key] = value
            with self.subTest(key=key):
                self.assertEqual(self.run_fake(native(c,port_accounting=bad))["code"],1)

    def test_unknown_workspace_stays_unknown(self):
        result = self.run_fake(native(case(),workspace_bytes=None))
        self.assertEqual(result["code"],0)
        self.assertIsNone(result["rows"][0]["native"]["workspace_bytes"])
        self.assertFalse(result["rows"][0]["eligibility"]["workspace_known"])

    def test_native_binding_rejects_different_graph_or_protocol(self):
        for key,value in (("stage","block"),("history",2),("q",1),("batch",True),
                          ("contract","unknown"),("scope","partial"),("setup_compute_repetitions",0),
                          ("threads_applied",1),("samples",4),("included",[]),("all_outputs_finite",False)):
            with self.subTest(key=key):
                result = self.run_fake(native(case(),**{key:value}))
                self.assertEqual(result["code"],1)
                self.assertEqual(json.loads(result["raw"]["stdout"])[key],value)

    def test_native_rope_and_dtype_mismatch_rejected(self):
        for key,value in (("rope_base",20000),("norm_epsilon",1e-4),("qk_norm","none")):
            with self.subTest(key=key):
                changed = config(); changed[key] = value
                self.assertEqual(self.run_fake(native(case(),config=changed))["code"],1)

    def test_numeric_workspace_requires_allocator_evidence(self):
        self.assertEqual(self.run_fake(native(case(),workspace_bytes=0))["code"],1)
        self.assertEqual(self.run_fake(native(case(),workspace_source="unknown"))["code"],1)

    def test_missing_port_digest_and_kv_history_check_rejected(self):
        for changes in ({"stage_port_digests":{}},{"kv_append_check":{"passed":True,"append_exact":True,"history_unchanged":False}}):
            with self.subTest(changes=changes):
                self.assertEqual(self.run_fake(native(case(),**changes))["code"],1)

    def test_missing_or_malformed_port_digests_rejected(self):
        c = case("projection")
        for key in ("input_port_digests", "stage_port_digests"):
            for bad in ({}, {"undeclared":"0123456789abcdef"},
                        {port:"NOT_A_DIGEST" for port in native(c)[key]},
                        {port:"0123456789ABCDEF" for port in native(c)[key]}):
                with self.subTest(key=key,bad=bad):
                    self.assertEqual(self.run_fake(native(c,**{key:bad}),selected=c)["code"],1)

    def test_failures_and_timeouts_preserve_stdout(self):
        result = self.run_fake("partial output",exitcode=7)
        self.assertEqual(result["code"],1)
        self.assertEqual(result["raw"]["stdout"],"partial output")
        result = self.run_fake({},failure=subprocess.TimeoutExpired([],1,output=b"partial",stderr=b"diagnostic"))
        self.assertEqual(result["raw"]["stdout"],"partial")
        self.assertTrue(result["raw"]["timed_out"])

    def test_lf_files_and_no_output_overwrite(self):
        result = self.run_fake(native(case()))
        for contents in result["files"].values():
            self.assertNotIn(b"\r\n",contents)
            self.assertTrue(contents.endswith(b"\n"))

    def test_shared_lock_is_not_removed_by_rejected_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / ("ggml-dense-layer-probe.exe" if os.name == "nt" else "ggml-dense-layer-probe")
            binary.write_bytes(b"mock")
            path = root / "suite.json"; path.write_text(json.dumps(suite([case()])))
            lock = root / ".measurement.lock"; lock.write_text("another process")
            with (mock.patch.object(runner,"ROOT",root),mock.patch.object(runner.subprocess,"run") as child,
                  contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit)):
                runner.main(["--suite",str(path),"--native-bin",str(root),"--threads","2"])
            child.assert_not_called()
            self.assertEqual(lock.read_text(),"another process")


class DenseExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dense-export-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); (self.root/"raw").mkdir()
        self.layer = parse_layer_document(layer_document())
        self.cases = [case(kind) for kind in ("projection","attention","dense_ffn","block")]
        self.binary = self.root / ("ggml-dense-layer-probe.exe" if os.name == "nt" else "ggml-dense-layer-probe")
        self.digest = hashlib.sha256(b"fake executable never run").hexdigest()
        self.manifest = {"schema_version":"dense_layer_measurement_profile_v1","contract":CONTRACT,
            "profile_id":"synthetic-evidence","suite":suite(self.cases),
            "protocol":{"backend":"CPU","batch":1,"threads":2,"warmup":2,"samples":3,"seed":17,
                "sequential_execution":True,"graphs_per_sample":1,"setup_compute_repetitions":1,"measured_graph_untimed_compute_repetitions":3,
                "upstream_setup_compute_repetitions_by_kind":{kind:0 if kind=="block" else 1 for kind in ("projection","attention","dense_ffn","block")},
                "total_untimed_compute_repetitions_by_kind":{kind:3 if kind=="block" else 4 for kind in ("projection","attention","dense_ffn","block")}},
            "software":{"runtime_sha256":{str(self.binary):self.digest}}}
        self.rows = []
        for index,c in enumerate(self.cases):
            n = native(c); summary = summarize_samples(n["wall_ns"])
            self.rows.append({"case_id":c["id"],"case":c,"status":"measured",
                "signature":runner.measurement_signature(c,self.digest),"native":{**n,"expected_samples":3,"summary":summary},
                "summary":summary,"quality_flags":[],"raw_output":f"raw/{index+1:03d}.json"})
            self.write_raw(index)

    def write_raw(self,index):
        row = self.rows[index]
        n = {key:value for key,value in row["native"].items() if key not in ("summary","expected_samples")}
        raw = {"case":row["case"],"command":runner.native_command(row["case"],self.binary.parent,SimpleNamespace(warmup=2,samples=3,seed=17)),
               "returncode":0,"stdout":json.dumps(n),"stderr":"synthetic not executed"}
        (self.root/row["raw_output"]).write_text(json.dumps(raw),encoding="utf-8",newline="\n")

    def modify(self,index,**changes):
        row = self.rows[index]; row["native"].update(changes)
        row["summary"] = row["native"]["summary"] = summarize_samples(row["native"]["wall_ns"])
        self.write_raw(index)

    def export(self,**kwargs):
        return exporter.export_cost_table(self.layer,self.rows,raw_directory=self.root,manifest=self.manifest,**kwargs)

    def test_exports_three_stages_joint_block_only_for_comparison(self):
        costs,report = self.export()
        self.assertEqual(len(costs["entries"]),3)
        self.assertNotIn("block",{entry["kind"] for entry in costs["entries"]})
        self.assertEqual(report["composition"][0]["absolute_relative_error"],0)
        self.assertIsNone(report["composition"][0]["within_explicit_tolerance"])
        self.assertFalse(report["calibration_claimed"])
        self.assertTrue(report["complete_declared_block_variant"])
        self.assertFalse(costs["validation_only"])

    def test_export_does_not_change_evidence(self):
        original = copy.deepcopy((self.rows,self.manifest))
        self.export()
        self.assertEqual(original,(self.rows,self.manifest))

    def test_missing_stage_or_unobserved_history_never_filled(self):
        self.rows.pop()
        with self.assertRaisesRegex(ValueError,"complete exact shape group"):
            self.export()

    def test_stage_output_digest_must_equal_joint_port(self):
        self.modify(0,stage_port_digests={key:"fedcba9876543210" for key in self.rows[0]["native"]["stage_port_digests"]})
        with self.assertRaisesRegex(ValueError,"materialized port"):
            self.export()

    def test_stage_input_lineage_must_match_predecessor_materialized_ports(self):
        for index,port in ((0,"hidden"),(1,"q"),(1,"hidden"),(2,"attention_hidden")):
            with self.subTest(index=index,port=port):
                original = copy.deepcopy(self.rows)
                digests = dict(self.rows[index]["native"]["input_port_digests"])
                digests[port] = "fedcba9876543210"
                self.modify(index,input_port_digests=digests)
                with self.assertRaisesRegex(ValueError,"input"):
                    self.export()
                self.rows = original
                self.write_raw(index)

    def test_error_is_reported_and_only_explicit_tolerance_gates(self):
        self.modify(3,wall_ns=[1000,1001,999])
        costs,report = self.export()
        self.assertEqual(report["composition"][0]["absolute_relative_error"],2)
        self.assertFalse(costs["validation_only"])
        with self.assertRaisesRegex(ValueError,"exceeds explicit tolerance"):
            self.export(max_relative_error=0.2)
        costs,report = self.export(max_relative_error=0.2,allow_composition_error=True)
        self.assertTrue(costs["validation_only"])
        self.assertFalse(report["composition"][0]["within_explicit_tolerance"])

    def test_unstable_evidence_is_validation_only_if_explicitly_allowed(self):
        self.modify(0,wall_ns=[100,1000,10000])
        self.rows[0]["quality_flags"]=["high_observed_variability_cv_gt_0.1"]
        with self.assertRaisesRegex(ValueError,"unstable"):
            self.export()
        costs,report = self.export(allow_unstable=True)
        self.assertTrue(costs["validation_only"])
        self.assertTrue(report["validation_only"])

    def test_unknown_workspace_and_failed_records_cannot_be_waived(self):
        self.modify(0,workspace_bytes=None)
        with self.assertRaisesRegex(ValueError,"workspace is unknown"):
            self.export(allow_unstable=True)
        self.rows[0]["status"]="failed"
        with self.assertRaisesRegex(ValueError,"failed"):
            self.export(allow_unstable=True)

    def test_tampered_signature_raw_summary_and_runtime_hash_rejected(self):
        mutations = ("signature","raw","summary","runtime")
        for what in mutations:
            with self.subTest(what=what):
                original = copy.deepcopy((self.rows,self.manifest))
                if what == "signature": self.rows[0]["signature"]["physical_contract_sha256"]="0"*64
                elif what == "raw":
                    p = self.root/self.rows[0]["raw_output"]; raw=json.loads(p.read_text()); raw["stdout"]="{}";p.write_text(json.dumps(raw))
                elif what == "summary": self.rows[0]["summary"]["median_ns"]=1
                else: self.manifest["software"]["runtime_sha256"][str(self.binary)]="0"*64
                with self.assertRaises(ValueError): self.export()
                self.rows,self.manifest=original
                self.write_raw(0)

    def test_raw_path_escape_and_duplicate_shapes_rejected(self):
        self.rows[0]["raw_output"]="../other.json"
        with self.assertRaisesRegex(ValueError,"escapes"):
            self.export()
        self.rows[0]["raw_output"]="raw/001.json"
        self.rows.append(copy.deepcopy(self.rows[0]))
        with self.assertRaisesRegex(ValueError,"duplicate"):
            self.export()

    def test_invalid_tolerance_cannot_be_waived(self):
        for value in (True,float("nan"),float("inf"),-1):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,"tolerance"):
                self.export(max_relative_error=value,allow_composition_error=True)


if __name__ == "__main__":
    unittest.main()
