from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

SPEC=importlib.util.spec_from_file_location("trace_cost_test",Path(__file__).resolve().parents[1]/"trace_model.py")
module=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(module)


def span(name,begin,end,thread="main"):
    return dict(name=name,thread=thread,begin_ns=begin,end_ns=end)


def serial():
    return dict(begin_ns=0,end_ns=110,stages=[span("a",10,40),span("b",45,100)])


def parallel():
    return dict(begin_ns=0,end_ns=110,stages=[span("dispatch",0,5),span("cpu",10,30),
        span("join",35,90),span("copy",95,105),span("gpu",3,80,"dispatch")])


def record(trace):
    return dict(id="p1",split="profile",status="measured",case={"id":"shape","hidden":16},
                raw_sha256="a"*64,chain_traces={"path":[deepcopy(trace) for _ in range(3)]})


class TraceModelTests(unittest.TestCase):
    def test_serial_host_gaps_are_explicit_and_exact_for_constant_profile(self):
        source=record(serial());before=deepcopy(source)
        model=module.freeze([source]);result=module.predict(model,"shape","path")
        self.assertEqual(source,before)
        self.assertEqual(result["predicted_ns"],110)
        self.assertEqual([(x["begin_ns"],x["end_ns"]) for x in result["modeled_host_events"]],[(10,40),(45,100)])
        self.assertFalse(result["physical_kernel_timeline"])
        self.assertFalse(result["instrumentation_overhead_subtracted"])
        self.assertTrue(result["profile_joint_trace_intervals_used_for_calibration"])

    def test_fork_may_start_before_dispatch_returns_and_join_wait_is_not_double_counted(self):
        model=module.freeze([record(parallel())]);result=module.predict(model,"shape","path")
        self.assertEqual(result["predicted_ns"],110)
        byname={x["name"]:x for x in result["modeled_host_events"]}
        self.assertEqual((byname["gpu"]["begin_ns"],byname["gpu"]["end_ns"]),(3,80))
        self.assertEqual((byname["join"]["begin_ns"],byname["join"]["end_ns"]),(35,90))
        costs=model["paths"][0]["costs"]
        self.assertEqual(costs["main:join:completion_overhead"]["median_ns"],10)
        self.assertNotIn("main:join:duration",costs)

    def test_worker_completed_before_main_join_uses_only_actual_join_completion_cost(self):
        trace=parallel();trace["stages"][-1]["end_ns"]=20
        model=module.freeze([record(trace)]);result=module.predict(model,"shape","path")
        self.assertEqual(result["predicted_ns"],110)
        self.assertEqual(model["paths"][0]["costs"]["main:join:completion_overhead"]["median_ns"],55)

    def test_heldout_or_failed_rows_and_changed_programs_are_rejected(self):
        for key,value in (("split","heldout"),("status","failed"),("raw_sha256","unknown")):
            with self.subTest(key=key):
                source=record(serial());source[key]=value
                with self.assertRaises(ValueError):module.freeze([source])
        source=record(serial());source["chain_traces"]["path"][1]["stages"][0]["name"]="different"
        with self.assertRaises(ValueError):module.freeze([source])

    def test_bad_dependencies_thread_overlap_and_unknown_paths_fail_closed(self):
        malformed=[dict(begin_ns=0,end_ns=110,stages=[span("a",0,60),span("b",50,90)]),
            dict(begin_ns=0,end_ns=110,stages=[span("dispatch",5,10),span("join",90,100),span("gpu",0,80,"dispatch")]),
            dict(begin_ns=0,end_ns=110,stages=[span("dispatch",0,10),span("join",20,40),span("gpu",10,50,"dispatch")])]
        for trace in malformed:
            with self.assertRaises(ValueError):module.freeze([record(trace)])
        model=module.freeze([record(serial())])
        with self.assertRaises(ValueError):module.predict(model,"new_shape","path")
        with self.assertRaises(ValueError):module.predict(model,"shape","new_strategy")


if __name__=="__main__":unittest.main()
