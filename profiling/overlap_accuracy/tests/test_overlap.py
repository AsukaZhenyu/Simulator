"""Synthetic protocol/compiler tests; these are not hardware accuracy evidence."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from argparse import Namespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import contract as c
import bridge
SPEC=importlib.util.spec_from_file_location("overlap_run",HERE/"run.py")
r=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(r)


def profile_fixture():
    suite=c.suite();suite["cases"]=suite["cases"][:1]
    metrics={"copy/a":[100]*3,"copy/b":[120]*3,"compute/current_ffn":[200]*3,
             "serial/one_copy_compute":[300]*3,"overlap/one_copy_compute":[200]*3,
             "serial/two_copy_compute":[420]*3,"overlap/two_copy_compute":[220]*3}
    rows=[dict(id=f"profile-{n}",case=suite["cases"][0],split="profile",seed=2903,repeat=n,status="measured",raw_sha256=str(n)*64,
               metrics=deepcopy(metrics),known_workspace_bytes=16)for n in range(3)]
    return suite,rows


def payload_fixture(case,protocol,seed):
    metrics=profile_fixture()[1][0]["metrics"]
    checks={s+suffix:dict(passed=True,elements=1)for s in c.STRATEGIES for suffix in("/output","/diagnostic_output")}
    checks.update({"current_inputs_unchanged/"+name:dict(passed=True,elements=1)for name in("gate_weight","up_weight","down_weight","input")})
    traces={}
    for strategy in c.STRATEGIES:
        two="two_copy"in strategy
        trace=dict(total_ns=1000,copy_a_submit_begin_ns=0,copy_a_submit_end_ns=10,
            copy_b_submit_begin_ns=11 if two else-1,copy_b_submit_end_ns=20 if two else-1,
            compute_begin_ns=30,compute_end_ns=200,join_begin_ns=201,join_end_ns=999,
            copy_a_event_elapsed_ns=100,copy_b_event_elapsed_ns=120 if two else-1)
        traces[strategy]=[deepcopy(trace)for _ in range(protocol["samples"])]
    return dict(schema="overlap-accuracy-native-0.1",mode="measure",**{k:v for k,v in case.items()if k!="id"},**protocol,seed=seed,
        source_binding=dict(probe_sha256=c.sha(r.ROOT/"profiling/native/overlap_accuracy/compute.cpp"),context_sha256=c.CONTEXT_SHA,matched_sha256=c.MATCHED_SHA),
        wall_ns={k:[v[0]]*protocol["samples"]for k,v in metrics.items()},checks=checks,diagnostic_host_and_copy_events=traces,
        instrumentation_overhead_ns={"clock_pair":[1]*protocol["samples"],"empty_cuda_event_pair_completed":[10]*protocol["samples"]},
        prefetch_payload_checks_passed=True,prefetch_destinations_disjoint_from_each_other_and_all_current_ports=True,current_output_location="vram",
        prefetch_destination_location="vram",prefetch_source="preinitialized_page_locked_host_memory",completed_work_timer="steady_clock",
        matmul_precision="GGML_PREC_F32",nvidia_tf32_override="0",cuda_graphs="disabled",known_gallocr_workspace_bytes=16,
        complete_cuda_workspace_bytes=None,physical_capacity_verified=False,kernel_overlap_proved=False,copy_events_prove_kernel_overlap=False,
        instrumentation_overhead_subtracted=False,all_checks_passed=True)


class OverlapTests(unittest.TestCase):
    def test_full_grid_and_fresh_process_plan(self):
        suite=c.validate_suite(c.suite())
        self.assertEqual(18,len(suite["cases"]))
        self.assertEqual(108,len(r.jobs(suite)))

    def test_original_gates_cannot_be_relaxed(self):
        for key in c.QUALITY:
            suite=c.suite();suite["quality"]=dict(c.QUALITY,**{key:.99})
            with self.assertRaises(ValueError):c.validate_suite(suite)

    def test_seeds_must_be_disjoint(self):
        suite=c.suite();suite["heldout_seeds"]=suite["profile_seeds"]
        with self.assertRaises(ValueError):c.validate_suite(suite)

    def test_payload_and_case_bounds(self):
        for changes in(dict(payload_bytes=3),dict(hidden=2048),dict(id="../bad")):
            suite=c.suite();suite["cases"][0].update(changes)
            with self.assertRaises(ValueError):c.validate_suite(suite)

    def test_real_common_compiler_serial_and_overlap_laws(self):
        suite,rows=profile_fixture();bundle=r.frozen_bundle(rows,suite)
        expected=dict(zip(c.STRATEGIES,(300,200,420,220)))
        for program in bundle["programs"]:
            self.assertEqual(expected[program["strategy"]],program["predicted_ns"])
            self.assertTrue(program["common_fixed_tensor_compiler_runtime_exercised"])
            self.assertFalse(program["highlevel_ModelPlan_coverage"])
            self.assertEqual(1,program["scenario"]["architecture"]["pcie_copy_slots"])
        json.dumps(bundle,allow_nan=False)

    def test_copy_and_compute_tensors_do_not_share_data_dependencies(self):
        suite,rows=profile_fixture();program=r.frozen_bundle(rows,suite)["programs"][3]
        op=program["workload"]["operations"][0]
        self.assertNotIn("next_payload_a",op["inputs"])
        self.assertNotIn("next_payload_b",op["inputs"])
        self.assertEqual("vram",program["scenario"]["goal"]["output_locations"]["next_payload_a"])

    def test_heldout_is_forbidden_as_a_price_source(self):
        suite,rows=profile_fixture();rows[0]["split"]="heldout"
        with self.assertRaises(ValueError):c.freeze(rows,suite)

    def test_heldout_target_changes_cannot_change_costs_or_programs(self):
        suite,rows=profile_fixture();before=r.frozen_bundle(rows,suite)
        heldout=[dict(row,id="h"+row["id"],split="heldout",frozen_model_sha256_at_start="a"*64)for row in rows]
        result=r.evaluate(rows+heldout,before,suite)
        self.assertTrue(result["all_paths_accepted"])
        heldout[0]["metrics"]=deepcopy(heldout[0]["metrics"]);heldout[0]["metrics"]["overlap/one_copy_compute"]=[1000000]*3
        self.assertEqual(before,r.frozen_bundle(rows,suite))
        self.assertFalse(r.evaluate(rows+heldout,before,suite)["all_paths_accepted"])

    def test_noisy_profile_cost_fails_even_if_prediction_error_is_zero(self):
        suite,rows=profile_fixture();rows[0]["metrics"]["copy/a"]=[1,100,199]
        bundle=r.frozen_bundle(rows,suite)
        heldout=[dict(row,id="h"+row["id"],split="heldout",frozen_model_sha256_at_start="a"*64)for row in rows]
        output=r.evaluate(rows+heldout,bundle,suite)
        self.assertTrue(all(not row["gates"]["within_process_cv"]for row in output["results"]))

    def test_failure_kept_as_failed_coverage(self):
        suite,rows=profile_fixture();bundle=r.frozen_bundle(rows,suite)
        heldout=[dict(row,id="h"+row["id"],split="heldout",frozen_model_sha256_at_start="a"*64)for row in rows]
        heldout[0]=dict(heldout[0],status="failed",error="synthetic failure")
        output=r.evaluate(rows+heldout,bundle,suite)
        self.assertEqual([heldout[0]["id"]],output["failed_rows"])
        self.assertFalse(output["all_paths_accepted"])

    def test_native_contract_rejects_aliasing_or_overlap_claim(self):
        suite=c.suite();case=suite["cases"][0];p=suite["protocol"]
        payload=payload_fixture(case,p,23);probe=payload["source_binding"]["probe_sha256"]
        c.validate_native(case,p,23,payload,probe)
        for key,value in(("kernel_overlap_proved",True),("prefetch_destinations_disjoint_from_each_other_and_all_current_ports",False),("instrumentation_overhead_subtracted",True)):
            changed=deepcopy(payload);changed[key]=value
            with self.assertRaises(ValueError):c.validate_native(case,p,23,changed,probe)

    def test_native_contract_rejects_bad_host_order(self):
        suite=c.suite();case=suite["cases"][0];p=suite["protocol"]
        payload=payload_fixture(case,p,23);payload["diagnostic_host_and_copy_events"][c.STRATEGIES[0]][0]["compute_begin_ns"]=1
        with self.assertRaises(ValueError):c.validate_native(case,p,23,payload,payload["source_binding"]["probe_sha256"])

    def test_full_mocked_runner_and_offline_verifier(self):
        """All subprocesses are in-memory fixtures; no GPU or real lock used."""
        suite,unused=profile_fixture();suite["protocol"].update(samples=3,warmup=0,min_warmup_ms=0)
        class Process:
            pid=123
            returncode=0
            def __init__(self,command,**kwargs):
                seed=int(command[command.index("--seed")+1]);self.stdout=json.dumps(payload_fixture(suite["cases"][0],suite["protocol"],seed)).encode()
            def communicate(self,timeout=None):return self.stdout,b"SYNTHETIC ONLY; no hardware"
        with tempfile.TemporaryDirectory(prefix="synthetic-only-",dir=HERE/"tests")as directory:
            root=Path(directory).resolve();self.assertTrue(root.is_relative_to((HERE/"tests").resolve()))
            binary=root/"synthetic-probe.bin";binary.write_bytes(b"not an executable")
            generated=root/"context_generated.inc";generated.write_bytes(b"synthetic only")
            native=r.ROOT/"profiling/native/overlap_accuracy"
            binding=dict(probe_sha256=c.sha(native/"compute.cpp"),context_sha256=c.CONTEXT_SHA,matched_sha256=c.MATCHED_SHA,
                         generated_include_sha256=c.sha(generated),cmake_sha256=c.sha(native/"CMakeLists.txt"),build_script_sha256=c.sha(native/"build.py"))
            c.dump(root/"build-record.json",dict(returncode=0,binary_sha256=c.sha(binary),sources=binding,linked_libraries={},ggml_headers={}))
            c.dump(root/"suite.json",suite)
            args=Namespace(suite=root/"suite.json",binary=binary,output=root/"SYNTHETIC-NOT-HARDWARE",runtime_dir=[root],timeout_s=1,dry_run=False)
            with patch.object(r.subprocess,"Popen",Process),patch.object(r,"LOCK_PATH",root/"synthetic-only.lock"):
                self.assertEqual(0,r.run(args))
            with patch.dict(sys.modules,{"run":r}):
                spec=importlib.util.spec_from_file_location("overlap_verify",HERE/"verify.py")
                module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                self.assertTrue(module.verify(args.output)["verified"])
                saved_rows=r.load(args.output/"measurements.json")
                raw_path=Path(saved_rows[0]["raw_path"]);raw_bytes=raw_path.read_bytes()
                for field,value in(("split","heldout"),("case",dict(saved_rows[0]["case"],tokens=32)),("seed",999),("repeat",7)):
                    changed=deepcopy(saved_rows);changed[0][field]=value;c.dump(args.output/"measurements.json",changed)
                    with self.assertRaisesRegex(ValueError,"immutable split/case/seed/repeat"):
                        module.verify(args.output)
                for corruption in("command","nonzero_exit","timeout"):
                    changed=deepcopy(saved_rows);raw=json.loads(raw_bytes)
                    if corruption=="command":
                        changed[0]["command"]=list(changed[0]["command"]);changed[0]["command"][-1]="validate";raw["command"]=changed[0]["command"]
                    elif corruption=="nonzero_exit":raw["returncode"]=1
                    else:raw["timed_out"]=True
                    c.dump(raw_path,raw);changed[0]["raw_sha256"]=c.sha(raw_path);c.dump(args.output/"measurements.json",changed)
                    with self.assertRaisesRegex(ValueError,"regenerated planned command|unsuccessful raw process"):
                        module.verify(args.output)
                    raw_path.write_bytes(raw_bytes)
                c.dump(args.output/"measurements.json",saved_rows)
                self.assertTrue(module.verify(args.output)["verified"])


if __name__=="__main__":unittest.main()
