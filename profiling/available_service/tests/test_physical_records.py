"""Synthetic attribution checks; no profiler, native executable or GPU runs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from profiling.available_service.physical_records import load_physical_records, matmul_work, parse_physical_records
from profiling.available_service.calibration import digest


def fixture():
    def timed(kind,left,right,**extra):
        return dict(kind=kind,begin_ns=left,end_ns=right,timestamp_valid=True,correlation_id=5,**extra)
    action=dict(kind="COMPUTE",target_id="op",device="gpu")
    capture=dict(schema="physical-stage-cupti-1",clock_id="windows_qpc_floor_ns",clock_origin="absolute_qpc_zero",
        qpc_frequency_hz=10_000_000,capture_begin_ns=90,capture_end_ns=210,
        timestamp_callback_registered_before_activity_enable=True,timestamp_callback_failed=False,
        cupti_dropped_records=0,local_record_overflow=0,callback_errors=[],all_requested_records_retained=True,
        external_correlation_kind=1,records=[
            dict(kind="external_correlation",correlation_id=5,external_id=1,external_kind=1),
            timed("runtime_api",112,172,name="cudaLaunchKernel",process_id=7,thread_id=8,return_value=0),
            timed("driver_api",115,168,name="cuLaunchKernel",process_id=7,thread_id=8,return_value=0),
            timed("kernel",120,140,name="gemm",device_id=0,context_id=1,stream_id=2),
            timed("memcpy",150,160,bytes=24,device_id=0,context_id=1,stream_id=2),
        ])
    return dict(schema="physical-stage-native-1",process_id=7,main_thread_id=8,
        clock=dict(clock_id="windows_qpc_floor_ns",clock_origin="absolute_qpc_zero",frequency_hz=10_000_000),
        sampling_begin_ns=100,sampling_end_ns=200,actions=[action],mode="full",instrumentation="cupti",
        samples_completed=1,samples_requested=1,program_sha256="f"*64,
        iterations=[dict(iteration=0,begin_ns=100,end_ns=200,actions=[
            dict(action_index=0,kind="COMPUTE",target_id="op",device="gpu",external_id=1,begin_ns=110,end_ns=190)])],
        cupti=capture,batch_cpu=dict(per_action_active_cpu_time_measured=False))


class PhysicalRecordTests(unittest.TestCase):
    def test_exact_correlation_and_descriptive_unions(self):
        d=parse_physical_records(fixture());a=d["actions"][0]
        self.assertTrue(d["capture_qualified"])
        self.assertEqual(a["host_span_ns"],80)
        self.assertEqual(a["kernel_union_ns"],20);self.assertEqual(a["memcpy_union_ns"],10)
        self.assertEqual(a["runtime_api_union_ns"],60);self.assertEqual(a["driver_api_union_ns"],53)
        self.assertEqual(a["runtime_and_driver_API_union_ns"],60)
        self.assertEqual(a["before_first_device_ns"],10)
        self.assertEqual(a["between_device_intervals_ns"],10)
        self.assertEqual(a["after_last_device_ns"],30)
        self.assertEqual(a["host_without_device_activity_ns"],50)
        self.assertTrue(d["iterations"][0]["all_actions_and_disjoint_gaps_telescope"])
        self.assertFalse(d["hardware_accuracy_qualified"])
        self.assertFalse(d["device_activity_is_SM_active_service"])

    def test_nested_overlapping_kernels_union_counted_once(self):
        n=fixture();r=deepcopy(n["cupti"]["records"][3]);r.update(begin_ns=130,end_ns=145)
        n["cupti"]["records"].append(r)
        d=parse_physical_records(n);self.assertEqual(d["actions"][0]["kernel_union_ns"],25)
        self.assertEqual(d["actions"][0]["device_activity_union_ns"],35)

    def test_missing_external_correlation_never_uses_containment_fallback(self):
        n=fixture();n["cupti"]["records"].pop(0)
        d=parse_physical_records(n)
        self.assertFalse(d["capture_qualified"]);self.assertEqual(len(d["orphan_records"]),4)
        self.assertFalse(d["timestamp_attribution_fallback_used"])
        self.assertEqual(d["actions"][0]["kernel_records"],[])

    def test_conflicting_correlation_is_ambiguous(self):
        n=fixture();n["cupti"]["records"].append(dict(kind="external_correlation",correlation_id=5,external_id=2,external_kind=1))
        d=parse_physical_records(n)
        self.assertFalse(d["capture_qualified"])
        self.assertEqual(d["conflicting_external_correlations"],{"5":[1,2]})

    def test_duplicate_identical_mapping_is_retained_not_ambiguous(self):
        n=fixture();n["cupti"]["records"].append(deepcopy(n["cupti"]["records"][0]))
        d=parse_physical_records(n)
        self.assertTrue(d["capture_qualified"])
        self.assertEqual(d["external_correlation_record_indices"]["5"],[0,5])

    def test_foreign_external_kind_has_no_attribution(self):
        n=fixture();n["cupti"]["records"][0]["external_kind"]=2
        d=parse_physical_records(n);self.assertFalse(d["capture_qualified"])
        self.assertEqual(len(d["orphan_records"]),5)

    def test_unknown_external_action_retained(self):
        n=fixture();n["cupti"]["records"][0]["external_id"]=42
        d=parse_physical_records(n);self.assertFalse(d["capture_qualified"])
        self.assertTrue(all(r["reason"]=="external_action_unknown" for r in d["orphan_records"]))

    def test_invalid_timestamp_retained_and_excluded_from_union(self):
        n=fixture();n["cupti"]["records"][3]["timestamp_valid"]=False
        d=parse_physical_records(n)
        self.assertFalse(d["capture_qualified"]);self.assertEqual(len(d["invalid_time_records"]),1)
        self.assertEqual(d["actions"][0]["kernel_union_ns"],0)
        self.assertIn(3,d["actions"][0]["record_indices"]["kernel"])

    def test_zero_or_reversed_activity_interval_retained(self):
        for begin,end in ((0,0),(140,130)):
            n=fixture();n["cupti"]["records"][3].update(begin_ns=begin,end_ns=end)
            d=parse_physical_records(n);self.assertEqual(len(d["invalid_time_records"]),1)

    def test_drop_overflow_callback_and_timestamp_failures_qualify_false(self):
        for key,value in (("cupti_dropped_records",2),("local_record_overflow",1),("callback_errors",["error"]),
                          ("timestamp_callback_failed",True),("all_requested_records_retained",False)):
            n=fixture();n["cupti"][key]=value
            with self.subTest(key=key):self.assertFalse(parse_physical_records(n)["capture_qualified"])

    def test_clock_mismatch_is_rejected(self):
        n=fixture();n["cupti"]["clock_id"]="different"
        with self.assertRaisesRegex(ValueError,"clocks differ"):parse_physical_records(n)

    def test_API_foreign_thread_retained_and_flagged(self):
        n=fixture();n["cupti"]["records"][1]["thread_id"]=9
        d=parse_physical_records(n);self.assertFalse(d["capture_qualified"])
        self.assertEqual(len(d["actions"][0]["runtime_api_records"]),1)

    def test_nonzero_query_return_is_diagnostic_not_capture_loss(self):
        n=fixture();n["cupti"]["records"][1].update(name="cudaEventQuery",return_value=600)
        d=parse_physical_records(n);self.assertTrue(d["capture_qualified"])
        self.assertEqual(len(d["nonzero_API_return_records"]),1)

    def test_record_outside_host_action_is_not_clipped(self):
        n=fixture();n["cupti"]["records"][3]["end_ns"]=195
        d=parse_physical_records(n);a=d["actions"][0]
        self.assertFalse(d["capture_qualified"])
        self.assertEqual(a["kernel_records"][0]["end_ns"],195)
        self.assertIsNone(a["after_last_device_ns"])

    def test_unscoped_setup_API_retained_outside_actions(self):
        n=fixture();r=deepcopy(n["cupti"]["records"][1]);r.update(correlation_id=99,begin_ns=95,end_ns=98)
        n["cupti"]["records"].append(r)
        d=parse_physical_records(n);self.assertTrue(d["capture_qualified"])
        self.assertEqual(len(d["orphan_records"]),1)

    def test_unattributed_API_inside_action_does_not_get_time_assigned(self):
        n=fixture();r=deepcopy(n["cupti"]["records"][1]);r.update(correlation_id=99)
        n["cupti"]["records"].append(r)
        d=parse_physical_records(n);self.assertFalse(d["capture_qualified"])
        self.assertEqual(len(d["actions"][0]["runtime_api_records"]),1)

    def test_unknown_kind_retained(self):
        n=fixture();n["cupti"]["records"].append(dict(kind="unknown",correlation_id=5))
        d=parse_physical_records(n);self.assertFalse(d["capture_qualified"])
        self.assertEqual(len(d["unknown_kind_records"]),1)

    def test_bare_capture_has_no_device_claim(self):
        n=fixture();n["instrumentation"]="bare";n["cupti"]=dict(schema="physical-stage-cupti-1",enabled=False,records=[])
        d=parse_physical_records(n);self.assertFalse(d["capture_qualified"])
        self.assertEqual(d["actions"][0]["host_span_ns"],80)

    def test_duplicate_external_action_and_missing_inventory_rejected(self):
        n=fixture();n["iterations"][0]["actions"].append(deepcopy(n["iterations"][0]["actions"][0]))
        with self.assertRaises(ValueError):parse_physical_records(n)

    def test_source_file_bytes_are_pinned(self):
        with tempfile.TemporaryDirectory(prefix="physical-record-",dir=ROOT/"build") as directory:
            p=Path(directory)/"native.json";p.write_text(json.dumps(fixture()),encoding="utf-8",newline="\n")
            d=load_physical_records(p,hashlib.sha256(p.read_bytes()).hexdigest())
            self.assertTrue(d["capture_qualified"])
            with self.assertRaisesRegex(ValueError,"source bytes"):
                load_physical_records(p,"0"*64)

    def program(self,batch=1):
        return dict(actions=[dict(kind="COMPUTE",target_id="op",device="gpu")],
            bindings=[dict(operation_id="op",kind="ffn_up",operands={"x0":"w","x1":"x"},outputs={"result":"y"})],
            graph=dict(workload=dict(tensors=[dict(id="w",dtype="GGML_TYPE_F32",ne=[4,3,1,1]),
                dict(id="x",dtype="GGML_TYPE_F32",ne=[4,2,batch,1]),dict(id="y",dtype="GGML_TYPE_F32",ne=[3,2,batch,1])])))

    def test_matmul_known_arithmetic_and_output_materialization_boundary(self):
        result=matmul_work(self.program(),"op")
        self.assertEqual(result["gemm_flops"],48);self.assertEqual(result["output_storage_bytes"],24)
        self.assertFalse(result["entire_action_physical_work_identified"])
        self.assertEqual(matmul_work(self.program(batch=2),"op")["gemm_flops"],96)

    def test_matmul_wrong_geometry_or_macro_not_guessed(self):
        p=self.program();p["bindings"][0]["kind"]="dense_ffn"
        with self.assertRaisesRegex(ValueError,"pure matmul"):matmul_work(p,"op")
        p=self.program();p["graph"]["workload"]["tensors"][1]["ne"][0]=5
        with self.assertRaisesRegex(ValueError,"geometry"):matmul_work(p,"op")

    def test_program_identity_and_actions_checked(self):
        p=self.program();p["program_sha256"]=digest(p);n=fixture();n["program_sha256"]=p["program_sha256"]
        self.assertTrue(parse_physical_records(n,program=p)["program_binding_checked"])
        p["actions"][0]["device"]="cpu"
        with self.assertRaisesRegex(ValueError,"identity differs"):parse_physical_records(n,program=p)


if __name__=="__main__":unittest.main()
