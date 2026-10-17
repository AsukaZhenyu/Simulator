from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from moe_contract import (CONTRACT,NATIVE_SCHEMA,TIMING_SCOPE,build_plan,contract_hash,groups,
    physical_signature,stage_memory,stage_names,stage_ports,statistics_ns,validate_case,validate_native,
    INTERLEAVED_CONTRACT,INTERLEAVED_TIMING_SCOPE,contract_for)
from run_moe_measurements import dump,load_measurements,native_command,run_suite,summarize_native,validate_result


def case():
    return {"id":"small","config":{"hidden":16,"intermediate":32,"experts":4,"top_k":2,"tokens":3},
            "routes":[[0,1],[1,2],[0,2]],"threads":1,"backend":"CPU","phase":"prefill",
            "work_items":[{"request_id":"r0","absolute_position":token} for token in range(3)]}


def result(c=None):
    c=c or case()
    stages={}
    for stage in stage_names(c):
        ports={name:{**port,"digest":"0123456789abcdef"} for name,port in stage_ports(c,stage).items()}
        stages[stage]={"samples_ns":[100,101,102],"workspace_bytes":96,
            "workspace_source":"ggml_gallocr_internal_buffer_plus_cpu_graph_plan",
            "memory_accounting":{"resident_logical_bytes":stage_memory(c,stage)["resident_logical_bytes"],
                "resident_allocated_bytes":stage_memory(c,stage)["resident_logical_bytes"]+64,
                "internal_buffer_bytes":64,"cpu_graph_plan_work_bytes":32,"metadata_bytes_included":False},
            "all_outputs_finite":True,"reference_check":{"passed":True,"max_abs_error":0.000001,
                "max_rel_error":0.00001,"atol":0.0001,"rtol":0.0001},"ports":ports}
    return {"schema_version":NATIVE_SCHEMA,"contract":CONTRACT,"backend":"CPU",
        "routing_mode":"prescribed_trace_replay","scope":"resident_moe_ffn_subgraph",
        "random_weights_not_model_weights":True,"router_discovery_used_for_grouping":False,
        "dtype":"f32","config":c["config"],"routes":c["routes"],"discovered_routes":c["routes"],
        "threads_requested":1,"threads_applied":1,"warmup":5,"samples":3,"seed":123,
        "timing_scope":TIMING_SCOPE,"port_digest_algorithm":"fnv1a64_native_tensor_bytes","stages":stages}


class MoEPhysicalContractTests(unittest.TestCase):
    def test_group_rows_and_slots_are_actual_trace(self):
        self.assertEqual(groups(case()),{0:[(0,0),(2,0)],1:[(0,1),(1,0)],2:[(1,1),(2,1)]})
    def test_inactive_expert_has_no_physical_weights(self):
        self.assertFalse(any(name.endswith(".3") for name in stage_ports(case(),"joint")))
    def test_expert_includes_gather_input_indices_and_three_matrices(self):
        ports=stage_ports(case(),"expert.1")
        self.assertEqual(ports["hidden_input"]["ne"],[16,3,1,1])
        self.assertEqual(ports["expert_output.1"]["ne"],[16,2,1,1])
        self.assertEqual(ports["rows.1"]["dtype"],"i32")
        self.assertEqual({name for name,p in ports.items() if p["category"]=="weight"},{"gate.1","up.1","down.1"})
    def test_combine_declares_every_materialized_group(self):
        ports=stage_ports(case(),"combine")
        self.assertEqual(ports["selected_scores"]["ne"],[2,3,1,1])
        self.assertTrue(all("expert_output."+str(e) in ports and "slots."+str(e) in ports for e in (0,1,2)))
    def test_router_discovery_and_prescribed_ids_are_separate(self):
        ports=stage_ports(case(),"router")
        self.assertEqual(ports["route_ids"]["category"],"input")
        self.assertEqual(ports["discovered_topk"]["category"],"output")
    def test_joint_inventory_sum_matches_categories(self):
        memory=stage_memory(case(),"joint")
        self.assertEqual(memory["resident_logical_bytes"],sum(p["bytes"] for p in stage_ports(case(),"joint").values()))
    def test_duplicate_route_expert_rejected(self):
        c=case();c["routes"][0]=[0,0]
        with self.assertRaises(ValueError):validate_case(c)
    def test_missing_token_route_rejected(self):
        c=case();c["routes"].pop()
        with self.assertRaises(ValueError):validate_case(c)
    def test_unknown_geometry_not_inferred(self):
        c=case();del c["config"]["intermediate"]
        with self.assertRaises(ValueError):validate_case(c)
    def test_unknown_case_axis_rejected(self):
        c=case();c["dtype"]="f16"
        with self.assertRaises(ValueError):validate_case(c)
    def test_no_cuda_to_cpu_fallback(self):
        c=case();c["backend"]="CUDA0"
        with self.assertRaises(ValueError):validate_case(c)
    def test_duplicate_work_item_rejected(self):
        c=case();c["work_items"][1]=c["work_items"][0]
        with self.assertRaises(ValueError):validate_case(c)
    def test_inactive_stage_rejected(self):
        with self.assertRaises(ValueError):stage_ports(case(),"expert.3")
    def test_bool_not_integer(self):
        c=case();c["threads"]=True
        with self.assertRaises(ValueError):validate_case(c)
    def test_signature_binds_trace_and_binary(self):
        a=physical_signature(case(),"a"*64);c=case();c["routes"][0]=[0,2]
        self.assertNotEqual(contract_hash(a),contract_hash(physical_signature(c,"a"*64)))
        self.assertNotEqual(contract_hash(a),contract_hash(physical_signature(case(),"b"*64)))
    def test_suite_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):build_plan({"schema_version":"moe_measurement_suite_v1","cases":[case(),case()]})
    def test_native_valid_contract_passes(self):
        self.assertEqual(validate_native(result(),case())["contract"],CONTRACT)
    def test_native_workspace_null_rejected(self):
        n=result();n["stages"]["router"]["workspace_bytes"]=None
        with self.assertRaises(ValueError):validate_native(n,case())
    def test_native_workspace_forged_total_rejected(self):
        n=result();n["stages"]["router"]["workspace_bytes"]=95
        with self.assertRaises(ValueError):validate_native(n,case())
    def test_native_external_ports_not_hidden_in_workspace(self):
        n=result();n["stages"]["router"]["memory_accounting"]["resident_logical_bytes"]-=4
        with self.assertRaises(ValueError):validate_native(n,case())
    def test_native_changed_row_port_rejected(self):
        n=result();n["stages"]["expert.0"]["ports"]["rows.0"]["ne"]=[3,1,1,1]
        with self.assertRaises(ValueError):validate_native(n,case())
    def test_native_stage_lineage_mismatch_rejected(self):
        n=result();n["stages"]["combine"]["ports"]["expert_output.0"]["digest"]="f"*16
        with self.assertRaises(ValueError):validate_native(n,case())
    def test_native_numeric_failure_rejected(self):
        n=result();n["stages"]["joint"]["reference_check"]["passed"]=False
        with self.assertRaises(ValueError):validate_native(n,case())
    def test_native_tolerance_change_rejected(self):
        n=result();n["stages"]["joint"]["reference_check"]["atol"]=1
        with self.assertRaises(ValueError):validate_native(n,case())
    def test_native_missing_active_stage_rejected(self):
        n=result();del n["stages"]["expert.0"]
        with self.assertRaises(ValueError):validate_native(n,case())
    def test_native_nonpositive_timing_rejected(self):
        n=result();n["stages"]["joint"]["samples_ns"]=[0,1,2]
        with self.assertRaises(ValueError):validate_native(n,case())
    def test_native_bool_applied_threads_rejected(self):
        n=result();n["threads_applied"]=True
        with self.assertRaises(ValueError):validate_native(n,case())
    def test_native_seed_and_warmup_match_runner(self):
        with self.assertRaises(ValueError):validate_result(result(),case(),warmup=5,samples=3,seed=456)
    def test_unstable_samples_explicitly_validation_only(self):
        n=result();n["stages"]["router"]["samples_ns"]=[1,2,300]
        _,flags=summarize_native(n)
        self.assertIn("high_cv:router",flags)
    def test_low_variance_statistics(self):
        self.assertEqual(statistics_ns([100,101,102])["median_ns"],101)
    def test_json_writes_only_lf(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"x.json";dump(path,{"routes":case()["routes"]})
            self.assertNotIn(b"\r\n",path.read_bytes())
    def test_command_preserves_trace_slot_order(self):
        command=native_command(Path("probe"),case(),warmup=5,samples=3,seed=123)
        self.assertEqual(command[command.index("--routes")+1],"0,1;1,2;0,2")

    def test_native_boolean_trailing_dimension_rejected(self):
        n=result();n["stages"]["router"]["ports"]["route_ids"]["ne"][3]=True
        with self.assertRaises(ValueError):validate_native(n,case())
    def test_native_float_dimension_rejected(self):
        n=result();n["stages"]["router"]["ports"]["route_ids"]["ne"][3]=1.0
        with self.assertRaises(ValueError):validate_native(n,case())
    def test_native_float_port_bytes_rejected(self):
        n=result();n["stages"]["router"]["ports"]["route_ids"]["bytes"]=24.0
        with self.assertRaises(ValueError):validate_native(n,case())
    def test_native_bool_geometry_rejected(self):
        c=case();c["config"]["tokens"]=1;c["routes"]=[[0,1]];c.pop("work_items")
        n=result(c);n["config"]["tokens"]=True
        with self.assertRaises(ValueError):validate_native(n,c)
    def test_native_float_route_expert_rejected(self):
        n=result();n["routes"][0][0]=0.0
        with self.assertRaises(ValueError):validate_native(n,case())


class MoERunnerEvidenceTests(unittest.TestCase):
    def prepare(self, folder):
        base=Path(folder)
        suite=base/"suite.json"
        dump(suite,{"schema_version":"moe_measurement_suite_v1","cases":[case()]})
        binary=base/"probe.exe";binary.write_bytes(b"explicit test binary")
        runtime=base/"runtime";runtime.mkdir()
        (runtime/"ggml-base.dll").write_bytes(b"test-base")
        (runtime/"ggml-cpu.dll").write_bytes(b"test-cpu")
        return SimpleNamespace(suite=suite,native_bin=binary,runtime_dir=[runtime],output=base/"evidence",
            warmup=5,samples=3,seed=123,dry_run=False,timeout_seconds=3)
    def produce(self, args):
        process=SimpleNamespace(returncode=0,stdout=json.dumps(result()),stderr="")
        with patch("run_moe_measurements.subprocess.run",return_value=process):
            run_suite(args)
    def test_runner_portable_load_checks_complete_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            args=self.prepare(folder);self.produce(args)
            self.assertEqual(load_measurements(args.output,verify_live_sources=True)["records"][0]["case_id"],"small")
    def test_raw_modification_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            args=self.prepare(folder);self.produce(args)
            raw=args.output/"raw/000.json";raw.write_text("{}\n",encoding="utf-8",newline="\n")
            with self.assertRaisesRegex(ValueError,"source changed"):load_measurements(args.output)
    def test_summary_modification_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            args=self.prepare(folder);self.produce(args)
            path=args.output/"measurements.json";report=json.loads(path.read_text())
            report["records"][0]["summary"]["router"]["median_ns"]=1;dump(path,report)
            with self.assertRaisesRegex(ValueError,"statistics changed"):load_measurements(args.output)
    def test_changed_live_runtime_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            args=self.prepare(folder);self.produce(args)
            (args.runtime_dir[0]/"ggml-base.dll").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError,"live measurement source changed"):
                load_measurements(args.output,verify_live_sources=True)
    def test_raw_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            args=self.prepare(folder);self.produce(args)
            path=args.output/"measurements.json";report=json.loads(path.read_text())
            report["records"][0]["raw_output"]="../suite.json";dump(path,report)
            with self.assertRaisesRegex(ValueError,"path escapes"):load_measurements(args.output)
    def test_output_evidence_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            args=self.prepare(folder);self.produce(args)
            with self.assertRaisesRegex(ValueError,"never overwritten"):run_suite(args)
    def test_dry_run_does_not_execute_hardware(self):
        with tempfile.TemporaryDirectory() as folder:
            args=self.prepare(folder);args.dry_run=True
            with patch("run_moe_measurements.subprocess.run") as execute:
                run_suite(args)
            execute.assert_not_called()
            self.assertFalse((args.output/"measurements.json").exists())

    def test_signature_boolean_dimension_cannot_match_integer_by_equality(self):
        with tempfile.TemporaryDirectory() as folder:
            args=self.prepare(folder);self.produce(args)
            path=args.output/"measurements.json";report=json.loads(path.read_text())
            report["records"][0]["physical_signature"]["stage_ports"]["router"]["route_ids"]["ne"][3]=True
            dump(path,report)
            with self.assertRaisesRegex(ValueError,"physical contract source changed"):
                load_measurements(args.output)
    def test_embedded_native_boolean_dimension_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            args=self.prepare(folder);self.produce(args)
            path=args.output/"measurements.json";report=json.loads(path.read_text())
            report["records"][0]["native"]["stages"]["router"]["ports"]["route_ids"]["ne"][3]=True
            dump(path,report)
            with self.assertRaisesRegex(ValueError,"embedded native measurement changed"):
                load_measurements(args.output)


def shared_result():
    c=case();c["sampling_mode"]="interleaved_joint_context"
    n=result(c);n["schema_version"]="resident_moe_ffn_native_v2";n["contract"]=INTERLEAVED_CONTRACT
    n["sampling_mode"]="interleaved_joint_context";n["timing_scope"]=INTERLEAVED_TIMING_SCOPE
    total=stage_memory(c,"joint")["resident_logical_bytes"]
    for value in n["stages"].values():
        value["workspace_source"]="shared_ggml_gallocr_internal_buffer_plus_cpu_plan_arena"
        memory=value["memory_accounting"]
        memory.update(shared_resident_logical_bytes=total,shared_resident_allocated_bytes=total+64,
            resident_allocated_bytes=total+64,stage_cpu_graph_plan_required_bytes=16,shared_allocator=True)
    return c,n


class MoESharedContextTests(unittest.TestCase):
    def test_shared_context_contract_and_stage_lineage_pass(self):
        c,n=shared_result();self.assertEqual(validate_native(n,c)["contract"],INTERLEAVED_CONTRACT)
    def test_v1_historical_evidence_still_valid(self):
        self.assertEqual(validate_native(result(),case())["contract"],CONTRACT)
    def test_sampling_mode_changes_physical_signature(self):
        c,n=shared_result()
        self.assertNotEqual(contract_hash(physical_signature(c)),contract_hash(physical_signature(case())))
        self.assertEqual(physical_signature(c)["sampling_mode"],"interleaved_joint_context")
    def test_mixing_protocols_in_suite_rejected(self):
        c,n=shared_result();c["id"]="v2"
        with self.assertRaises(ValueError):build_plan({"schema_version":"moe_measurement_suite_v1","cases":[case(),c]})
    def test_v2_requires_explicit_native_mode(self):
        c,n=shared_result();n.pop("sampling_mode")
        with self.assertRaises(ValueError):validate_native(n,c)
    def test_shared_workspace_allocations_must_match_all_stages(self):
        c,n=shared_result();n["stages"]["router"]["workspace_bytes"]+=8
        n["stages"]["router"]["memory_accounting"]["internal_buffer_bytes"]+=8
        with self.assertRaises(ValueError):validate_native(n,c)
    def test_shared_external_bytes_cannot_be_a_stage_subset(self):
        c,n=shared_result();n["stages"]["router"]["memory_accounting"]["shared_resident_logical_bytes"]=stage_memory(c,"router")["resident_logical_bytes"]
        with self.assertRaises(ValueError):validate_native(n,c)
    def test_slice_plan_cannot_exceed_actual_shared_arena(self):
        c,n=shared_result();n["stages"]["router"]["memory_accounting"]["stage_cpu_graph_plan_required_bytes"]=64
        with self.assertRaises(ValueError):validate_native(n,c)
    def test_shared_allocator_false_rejected(self):
        c,n=shared_result();n["stages"]["router"]["memory_accounting"]["shared_allocator"]=False
        with self.assertRaises(ValueError):validate_native(n,c)
    def test_v2_isolated_workspace_provenance_rejected(self):
        c,n=shared_result();n["stages"]["router"]["workspace_source"]="ggml_gallocr_internal_buffer_plus_cpu_graph_plan"
        with self.assertRaises(ValueError):validate_native(n,c)


if __name__=="__main__":
    unittest.main()
