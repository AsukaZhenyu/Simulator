"""Synthetic-only protocol tests. These fixtures are not hardware evidence."""
import copy
import importlib.util
import json
from argparse import Namespace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import tempfile

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("chain_run", HERE / "run.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
from make_suite import make_suite
from trace_evaluate import evaluate


def make_trace(case, strategy):
    inventory = r.trace_inventory(case, strategy)
    spans, cursor = [], 1
    for name in inventory["main"]:
        spans.append(dict(name=name, thread="main", begin_ns=cursor, end_ns=cursor+10))
        cursor += 11
    if inventory["dispatch"]:
        launch = next(x for x in spans if x["name"] == "dispatch")
        worker_time = launch["begin_ns"] + 2
        for name in inventory["dispatch"]:
            spans.append(dict(name=name, thread="dispatch", begin_ns=worker_time, end_ns=worker_time+10))
            worker_time += 11
    trace = dict(begin_ns=0, end_ns=cursor+1, stages=spans,
                 branch_brackets=None, branch_start_skew_ns=None, join_wait_ns=None,
                 completed_work_bracket_overlap_ns=None)
    if inventory["dispatch"]:
        cpu = [x for x in spans if x["thread"] == "main" and x["name"] in ("gate", "up", "activation", "down", "cpu_joint")]
        gpu = [x for x in spans if x["thread"] == "dispatch"]
        join = next(x for x in spans if x["name"] == "join")
        cb, ce = min(x["begin_ns"] for x in cpu), max(x["end_ns"] for x in cpu)
        gb, ge = min(x["begin_ns"] for x in gpu), max(x["end_ns"] for x in gpu)
        trace.update(branch_brackets=dict(cpu=dict(begin_ns=cb,end_ns=ce), gpu_host=dict(begin_ns=gb,end_ns=ge),
                                          join=dict(begin_ns=join["begin_ns"],end_ns=join["end_ns"])),
                     branch_start_skew_ns=gb-cb, join_wait_ns=join["end_ns"]-join["begin_ns"],
                     completed_work_bracket_overlap_ns=max(0,min(ce,ge)-max(cb,gb)))
    return trace


def make_payload(case, suite):
    traces = {strategy: [make_trace(case, strategy) for _ in range(suite["protocol"]["samples"])] for strategy in r.recipes(case)}
    payload = {k:v for k,v in case.items() if k != "id"}
    payload.update(suite["protocol"])
    payload.update(schema="chain-calibration-native-0.1", seed=113, dispatch_cpu_id=1,
                   timer="steady_clock_completed_work", matmul_precision="GGML_PREC_F32_explicit", nvidia_tf32_override="0",
                   source_binding=dict(context_sha256=r.CONTEXT_SHA, matched_sha256=r.MATCHED_SHA,
                                       probe_sha256=r.a.sha(r.PROFILING / "native/chain_calibration/compute.cpp")),
                   wall_ns={key:[100]*suite["protocol"]["samples"] for key in r.expected_metrics(case)},
                   chain_traces=traces,
                   timestamp_overhead_ns={key:[3]*suite["protocol"]["samples"] for key in ["clock_pair", "noop_stage_pair", *("noop_trace/"+s for s in traces)]},
                   checks={key+"/output":dict(passed=True,elements=1,max_abs_error=0,max_relative_error=0,atol=.00001,rtol=.0001) for key in traces},
                   boundary=dict(input_location="all_required_immutable_inputs_already_resident_in_both_cpu_and_gpu",output_location="dram",overhead_subtracted=False),
                   kernel_overlap_proved=False,gpu_capacity_proved=False,context_switch_attribution_proved=False,all_checks_passed=True,
                   observed_affinity=dict(process_mask=3,main_thread_mask=1,main_thread_group=0,dispatch_thread_mask=2 if case["family"]=="ffn" else 0,dispatch_thread_group=0),
                   profile_gpu_shard_thread="persistent_dispatch_worker_same_as_parallel_path" if case["family"]=="ffn" else "not_applicable",
                   profile_gpu_shard_affinity_mask=2 if case["family"]=="ffn" else 0,profile_gpu_shard_affinity_group=0)
    ports = r.legacy.expected_port_bytes(case)
    if case["family"] == "ffn":
        ports.update({k.replace("gpu/full/","cpu/full/"):v for k,v in list(ports.items()) if k.startswith("gpu/full/")})
    payload["ports"]={k:dict(bytes=v,input_fixture_fnv1a64="0000000000000000") for k,v in ports.items()}
    if case["family"]=="ffn":
        extra={device+"/"+scope+"/"+stage for device in ("cpu","gpu") for scope in ("shard","full") for stage in ("gate","up","activated","output")}
    else:
        extra={device+"/"+stage for device in ("cpu","gpu") for stage in ("context","key_appended","value_appended","key_cache_unchanged","value_cache_unchanged","qk/scores","softmax/probabilities")}
    payload["checks"].update({key:dict(passed=True,elements=1,max_abs_error=0,max_relative_error=0,atol=.00001,rtol=.0001) for key in extra})
    return payload


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.suite=make_suite()
        self.suite["protocol"].update(samples=3,warmup=0,min_warmup_ms=0)
        self.case=self.suite["cases"][0]
        self.row=dict(id="synthetic",case=self.case,seed=113)

    def test_grid_is_explicit_and_broader(self):
        self.assertEqual(23,len(r.validate_suite(self.suite)["cases"]))
        self.assertEqual(138,sum(len(r.legacy.jobs(self.suite,s)) for s in ("profile","heldout")))

    def test_all_shape_protocols_validate(self):
        for case in self.suite["cases"]:
            r.validate_native(dict(self.row,case=case),make_payload(case,self.suite),self.suite,1)

    def test_cannot_relax_gate_or_reduce_processes(self):
        for key in r.GATES:
            changed=copy.deepcopy(self.suite)
            changed["quality"][key]=.99
            with self.assertRaises(ValueError):r.validate_suite(changed)
        changed=copy.deepcopy(self.suite);changed["repeats"]=2
        with self.assertRaises(ValueError):r.validate_suite(changed)

    def test_unknown_or_missing_stage_rejected(self):
        trace=make_trace(self.case,"hybrid/resident")
        trace["stages"][1]["name"]="fake"
        with self.assertRaises(ValueError):r.validate_trace(trace,hybrid_ffn=True,inventory=r.trace_inventory(self.case,"hybrid/resident"))

    def test_preload_requires_three_real_copy_spans(self):
        trace=make_trace(self.case,"hybrid/preload")
        trace["stages"][0]["name"]="preload_weights"
        with self.assertRaises(ValueError):r.validate_trace(trace,hybrid_ffn=True,inventory=r.trace_inventory(self.case,"hybrid/preload"))

    def test_join_cannot_finish_before_worker(self):
        trace=make_trace(self.case,"hybrid/resident")
        next(x for x in trace["stages"] if x["thread"]=="dispatch" and x["name"]=="down")["end_ns"]=trace["end_ns"]
        with self.assertRaises(ValueError):r.validate_trace(trace,hybrid_ffn=True)

    def test_skew_is_derived_from_raw_brackets(self):
        trace=make_trace(self.case,"hybrid/resident");trace["branch_start_skew_ns"]+=1
        with self.assertRaises(ValueError):r.validate_trace(trace,hybrid_ffn=True)

    def test_no_false_overlap_or_overhead_subtraction(self):
        for field in ("kernel_overlap_proved","context_switch_attribution_proved"):
            payload=make_payload(self.case,self.suite);payload[field]=True
            with self.assertRaises(ValueError):r.validate_native(self.row,payload,self.suite,1)
        payload=make_payload(self.case,self.suite);payload["boundary"]["overhead_subtracted"]=True
        with self.assertRaises(ValueError):r.validate_native(self.row,payload,self.suite,1)

    def test_heldout_cannot_enter_trace_model(self):
        payload=make_payload(self.case,self.suite)
        row=dict(self.row,split="heldout",status="measured",raw_sha256="a"*64,chain_traces=payload["chain_traces"])
        with self.assertRaises(ValueError):r.trace_model.freeze([row])

    def test_trace_model_matches_synthetic_timeline_and_freezes(self):
        payload=make_payload(self.case,self.suite)
        rows=[dict(self.row,id=f"p{n}",split="profile",status="measured",raw_sha256="a"*64,chain_traces=payload["chain_traces"],metrics=payload["wall_ns"]) for n in range(3)]
        model=r.trace_model.freeze(rows)
        for strategy,traces in payload["chain_traces"].items():
            self.assertEqual(traces[0]["end_ns"],r.trace_model.predict(model,self.case["id"],strategy)["predicted_ns"])
        heldout=[dict(row,id="h"+row["id"],split="heldout",frozen_trace_model_sha256_at_start="b"*64) for row in rows]
        before=copy.deepcopy(model)
        evaluate(rows+heldout,model,self.suite,r.trace_model)
        self.assertEqual(before,model)
        heldout[0]["metrics"]=dict(heldout[0]["metrics"],**{"hybrid/resident":[10**9]*3})
        self.assertEqual(before,r.trace_model.freeze(rows))

    def test_full_runner_and_offline_replay_with_synthetic_processes(self):
        """No executable is launched: Popen is a synthetic in-memory fixture."""
        suite=copy.deepcopy(self.suite)
        suite["cases"]=[suite["cases"][0],suite["cases"][-1]]
        suite["bootstrap"]["iterations"]=100
        suite["protocol"].update(samples=3,warmup=0,min_warmup_ms=0)
        class FakeProcess:
            pid=111
            returncode=0
            def __init__(inner,command,**kwargs):
                fields={command[n][2:].replace("-","_"):command[n+1] for n in range(1,len(command),2)}
                case=next(c for c in suite["cases"] if all(str(c[k])==fields[k] for k in c if k!="id"))
                payload=make_payload(case,suite);payload["seed"]=int(fields["seed"])
                inner.stdout=json.dumps(payload).encode()
            def communicate(inner,timeout):return inner.stdout,b"synthetic fixture; no hardware execution"
        with tempfile.TemporaryDirectory(prefix="synthetic-chain-only-",dir=HERE/"tests") as temporary:
            out=Path(temporary)
            self.assertTrue(out.resolve().is_relative_to((HERE/"tests").resolve()))
            binary=out/"synthetic-probe.bin";binary.write_bytes(b"synthetic-only-not-executable")
            generated=out/"context_generated.inc";generated.write_bytes(b"synthetic-only")
            r.a.dump(out/"source-binding.json",dict(context_sha256=r.CONTEXT_SHA,matched_sha256=r.MATCHED_SHA,
                probe_sha256=r.a.sha(r.PROFILING/"native/chain_calibration/compute.cpp"),
                cmake_sha256=r.a.sha(r.PROFILING/"native/chain_calibration/CMakeLists.txt"),
                generated_include_sha256=r.a.sha(generated),transformation="rename_terminal_main_only"))
            r.a.dump(out/"suite.json",suite)
            args=Namespace(suite=out/"suite.json",binary=binary,runtime_dir=[out],timeout_s=1,
                           output=out/"SYNTHETIC-NOT-HARDWARE",dry_run=False,preserve_file=[])
            with patch.object(r.subprocess,"Popen",FakeProcess),patch.object(r,"environment",lambda:{"synthetic_only":True}),patch.dict(r.os.environ,{"GGML_CALIBRATION_GPU_DISPATCH_CPU_ID":"1"}):
                self.assertEqual(0,r.run(args))
            # verify.py imports run by name; make it resolve the same bound module.
            with patch.dict(sys.modules,{"run":r}):
                verify_spec=importlib.util.spec_from_file_location("chain_verify",HERE/"verify.py")
                verifier=importlib.util.module_from_spec(verify_spec);verify_spec.loader.exec_module(verifier)
                self.assertTrue(verifier.verify(args.output)["verified"])
                saved_rows=r.legacy.document(args.output/"measurements.json")
                raw_path=Path(saved_rows[0]["raw_path"]);raw_bytes=raw_path.read_bytes()
                for field,value in(("split","heldout"),("case",dict(saved_rows[0]["case"],tokens=3)),("seed",999),("repeat",7)):
                    changed=copy.deepcopy(saved_rows);changed[0][field]=value;r.a.dump(args.output/"measurements.json",changed)
                    with self.assertRaisesRegex(ValueError,"immutable planned split/case/seed/repeat"):
                        verifier.verify(args.output)
                for corruption in("command","nonzero_exit","timeout"):
                    changed=copy.deepcopy(saved_rows);raw=json.loads(raw_bytes)
                    if corruption=="command":
                        changed[0]["command"]=list(changed[0]["command"]);changed[0]["command"][-1]="999";raw["command"]=changed[0]["command"]
                    elif corruption=="nonzero_exit":raw["returncode"]=1
                    else:raw["timed_out"]=True
                    r.a.dump(raw_path,raw);changed[0]["raw_sha256"]=r.a.sha(raw_path);r.a.dump(args.output/"measurements.json",changed)
                    with self.assertRaisesRegex(ValueError,"immutable planned|unsuccessful raw process"):
                        verifier.verify(args.output)
                    raw_path.write_bytes(raw_bytes)
                r.a.dump(args.output/"measurements.json",saved_rows)
                self.assertTrue(verifier.verify(args.output)["verified"])


if __name__ == "__main__":unittest.main()
