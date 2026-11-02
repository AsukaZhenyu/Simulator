"""Paired clock, path equivalence and memory protocol rejection tests."""
import copy
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from llm_infer_model.m2.dense_contract import build_plan, resident_accounting, ports
from run_dense_paired_measurements import main, paired_command, paired_quality_flags, summarize_paired, validate_paired_binding

ROOT = Path(__file__).resolve().parents[1]


def fixture(mode="graph"):
    case = build_plan(json.loads((ROOT/"cases/dense_error_grid.json").read_text(encoding="utf-8")), 2)[0]
    args = SimpleNamespace(compute_mode=mode, warmup=10, samples=4, seed=17)
    inputs, outputs = ports(case["config"], "block", case["tokens"])
    native = dict(schema="ggml_dense_layer_paired_probe/1", stage="paired", compute_mode=mode,
                  backend="CPU", batch=1, q=case["tokens"], history=case["visible_history"],
                  threads=2, warmup=10, samples=4, seed=17, config=case["config"],
                  implementation_variant=f"ggml_cpu_dense_materialized_paired_{mode}_v1",
                  timing="steady_clock_paired_same_resident_graph_partitions_including_sync",
                  paired_order=["AB", "BA", "AB", "BA"], all_outputs_finite=True,
                  reference_check={"passed": True}, port_digest_algorithm="fnv1a64_native_tensor_bytes",
                  stage_wall_ns={stage:[100, 101, 99, 100] for stage in ("projection", "attention", "dense_ffn")},
                  stage_sum_wall_ns=[300, 303, 297, 300], serial_wall_ns=[300, 313, 307, 310],
                  joint_wall_ns=[250, 251, 249, 250],
                  stage_node_ranges={stage:{"begin":i, "end":i+1, "ops":["CPY"]} for i,stage in enumerate(("projection", "attention", "dense_ffn"))})
    inventory = resident_accounting(case["config"], "block", case["tokens"], case["visible_history"])
    native.update(inventory)
    path = {"all_outputs_finite":True, "kv_append_check":dict(passed=True,append_exact=True,history_unchanged=True),
            "input_port_digests":{name:"a"*16 for name in inputs}, "stage_port_digests":{name:"b"*16 for name in outputs},
            "kv_cache_digests":{name:"c"*16 for name in ("key_cache","value_cache")}}
    native["paired_validation"] = dict(serial=copy.deepcopy(path), joint=copy.deepcopy(path), ports_exact=True,
                                        kv_caches_exact=True, input_ports_unchanged=True, checked_after_timing=True)
    work = dict(projection=100, attention=200, dense_ffn=300, joint=300)
    native["shared_memory_inventory"] = dict(timed_paths_share_tensor_addresses=True,
          resident_logical_bytes=inventory["resident_logical_bytes"], resident_allocated_bytes=inventory["resident_logical_bytes"]+32,
          internal_allocated_buffer_bytes=500, per_graph_cplan_work_size_bytes=work,
          backend_shared_cplan_capacity_bytes=300 if mode=="graph" else 0,
          cached_plans_total_work_bytes=900 if mode=="plan" else 0,
          physical_tensor_and_work_allocated_bytes=inventory["resident_logical_bytes"]+32+500+(300 if mode=="graph" else 900))
    return case,native,args


class PairedProtocol(unittest.TestCase):
    def test_modes_have_different_real_work_allocations(self):
        for mode in ("graph", "plan"):
            case,native,args=fixture(mode)
            self.assertIs(validate_paired_binding(case,native,args),native)
        self.assertGreater(fixture("plan")[1]["shared_memory_inventory"]["physical_tensor_and_work_allocated_bytes"],
                           fixture()[1]["shared_memory_inventory"]["physical_tensor_and_work_allocated_bytes"])

    def test_command_preserves_explicit_config_and_only_selects_paired_mode(self):
        case,_,args=fixture("plan")
        command=paired_command(case,Path("probe.exe"),args)
        self.assertEqual(command[command.index("--stage")+1],"paired")
        self.assertEqual(command[-2:],["--compute-mode","plan"])
        self.assertEqual(command[command.index("--history")+1],"0")

    def test_zero_clock_gap_is_preserved(self):
        summary=summarize_paired(fixture()[1])
        self.assertEqual(summary["clock_bookkeeping_ns"]["samples_ns"][0],0)
        self.assertFalse(summary["model_accuracy_claimed"])

    def test_stage_quality_does_not_disappear_under_stable_total(self):
        summary=summarize_paired(fixture()[1])
        summary["stages"]["projection"]["cv"]=0.2
        self.assertEqual(paired_quality_flags(summary),["stage:projection"])

    def test_order_is_part_of_the_protocol(self):
        case,native,args=fixture();native["paired_order"]=["AB"]*4
        with self.assertRaises(ValueError):validate_paired_binding(case,native,args)

    def test_missing_or_wrong_mode_cannot_be_substituted(self):
        for value in (None,"plan"):
            case,native,args=fixture();native["compute_mode"]=value
            with self.assertRaises(ValueError):validate_paired_binding(case,native,args)

    def test_changed_rope_or_history_rejects(self):
        for field in ("history","config"):
            case,native,args=fixture();native=copy.deepcopy(native)
            if field=="history":native[field]=1
            else:native[field]["rope_base"]=20000
            with self.assertRaises(ValueError):validate_paired_binding(case,native,args)

    def test_timer_counts_values_and_sum_are_strict(self):
        for field,value in (("serial_wall_ns",[300]*3),("joint_wall_ns",[250,True,250,250]),("stage_sum_wall_ns",[301,303,297,300]),("serial_wall_ns",[299,313,307,310])):
            case,native,args=fixture();native[field]=value
            with self.assertRaises(ValueError):validate_paired_binding(case,native,args)

    def test_partition_gap_or_missing_ops_rejects(self):
        for field,value in (("begin",2),("ops",[])):
            case,native,args=fixture();native["stage_node_ranges"]["attention"][field]=value
            with self.assertRaises(ValueError):validate_paired_binding(case,native,args)

    def test_digest_equality_flag_cannot_hide_mismatching_ports(self):
        case,native,args=fixture();native["paired_validation"]["joint"]["stage_port_digests"]["q"]="d"*16
        with self.assertRaises(ValueError):validate_paired_binding(case,native,args)

    def test_history_or_missing_kv_check_rejects(self):
        for key in ("history_unchanged","append_exact"):
            case,native,args=fixture();native["paired_validation"]["serial"]["kv_append_check"][key]=False
            with self.assertRaises(ValueError):validate_paired_binding(case,native,args)

    def test_extra_digest_port_and_unsupported_hash_reject(self):
        case,native,args=fixture();native["paired_validation"]["serial"]["input_port_digests"]["guessed"]="a"*16
        with self.assertRaises(ValueError):validate_paired_binding(case,native,args)
        case,native,args=fixture();native["port_digest_algorithm"]="sha256"
        with self.assertRaises(ValueError):validate_paired_binding(case,native,args)

    def test_plan_workspaces_must_include_all_four_allocations(self):
        for field in ("cached_plans_total_work_bytes","physical_tensor_and_work_allocated_bytes"):
            case,native,args=fixture("plan");native["shared_memory_inventory"][field]-=100
            with self.assertRaises(ValueError):validate_paired_binding(case,native,args)

    def test_unknown_memory_or_independent_addresses_cannot_pass(self):
        for field,value in (("internal_allocated_buffer_bytes",None),("timed_paths_share_tensor_addresses",False)):
            case,native,args=fixture();native["shared_memory_inventory"][field]=value
            with self.assertRaises(ValueError):validate_paired_binding(case,native,args)

    def test_duration_warmup_requires_both_time_and_rounds(self):
        case,native,args=fixture()
        args.min_warmup_ms=500;args.cpu_mask=None
        native.update(min_warmup_ms=500,warmup_rounds_requested=10,warmup_rounds_effective=20,
                      actual_warmup_elapsed_ns=500000000,
                      cpu_affinity=dict(requested_logical_ids=None,applied_logical_ids=None,verified=None))
        validate_paired_binding(case,native,args)
        for field,value in (("warmup_rounds_effective",9),("actual_warmup_elapsed_ns",499999999)):
            bad=copy.deepcopy(native);bad[field]=value
            with self.assertRaises(ValueError):validate_paired_binding(case,bad,args)

    def test_affinity_is_observed_not_only_requested(self):
        case,native,args=fixture()
        args.min_warmup_ms=500;args.cpu_mask=[0,2]
        native.update(min_warmup_ms=500,warmup_rounds_requested=10,warmup_rounds_effective=20,
                      actual_warmup_elapsed_ns=500000000,
                      cpu_affinity=dict(requested_logical_ids=[0,2],applied_logical_ids=[0,2],strict_cpu=True,verified=True,
                                        observed_threads=[dict(thread_id=10,group=0,mask_hex="1",logical_id=0),
                                                          dict(thread_id=11,group=0,mask_hex="4",logical_id=2)]))
        validate_paired_binding(case,native,args)
        for field,value in (("verified",False),("applied_logical_ids",[0,1])):
            bad=copy.deepcopy(native);bad["cpu_affinity"][field]=value
            with self.assertRaises(ValueError):validate_paired_binding(case,bad,args)
        for field,value in (("mask_hex","5"),("thread_id",10),("group",1),("logical_id",True)):
            bad=copy.deepcopy(native);bad["cpu_affinity"]["observed_threads"][1][field]=value
            with self.assertRaises(ValueError):validate_paired_binding(case,bad,args)
        command=paired_command(case,Path("probe.exe"),args)
        self.assertEqual(command[-4:],["--min-warmup-ms","500","--cpu-mask","0,2"])

    def test_unbound_affinity_stays_unknown(self):
        case,native,args=fixture();args.min_warmup_ms=0;args.cpu_mask=None
        native.update(min_warmup_ms=0,warmup_rounds_requested=10,warmup_rounds_effective=10,actual_warmup_elapsed_ns=1000,
                      cpu_affinity=dict(requested_logical_ids=None,applied_logical_ids=[0,2],verified=True))
        with self.assertRaises(ValueError):validate_paired_binding(case,native,args)

    def test_phase_label_cannot_turn_development_geometry_into_held_out(self):
        suite=json.loads((ROOT/"cases/dense_error_grid.json").read_text(encoding="utf-8"))
        case=copy.deepcopy(suite["cases"][1])
        duplicate=copy.deepcopy(case);duplicate.update(id="relabel",phase="prefill")
        duplicate["metadata"]["split"]="held_out"
        suite["cases"]=[case,duplicate]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"suite.json";path.write_text(json.dumps(suite),encoding="utf-8")
            arguments=["--suite",str(path),"--native-bin","unused.exe","--compute-mode","graph","--threads","2",
                       "--warmup","10","--min-warmup-ms","500","--samples","4","--seed","17","--repeats","2",
                       "--output",str(Path(folder)/"unused"),"--dry-run"]
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as caught:
                main(arguments)
            self.assertEqual(caught.exception.code,2)


if __name__=="__main__":unittest.main()
