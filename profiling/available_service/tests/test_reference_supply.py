"""Causal estimate construction and honest scope, without hardware workloads."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
for directory in (ROOT, ROOT / "modeling"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from profiling.available_service.reference_supply import (
    build_reference_supply, materialize_reference_window, validate_reference_supply,
)
from profiling.available_service.calibration import digest
from llm_infer_model.performance import Graph, InvalidModel, ParameterBundle, ServiceBinding, Task, WorkDemand


class ReferenceSupplyTests(unittest.TestCase):
    def cpu(self, ident="cpu0", *, begin=0, end=10, work=5, role="before_target"):
        return dict(sample_id=ident, sample_role=role, resource="cpu", work=work,
            work_unit="thread_cpu_ns", begin_ns=begin, end_ns=end,
            timing_scope="thread_active_service", sensor_window=dict(begin_ns=begin, end_ns=end),
            context={"fixture": "reference"}, clock_id="qpc-ns", clock_origin="go",
            evidence=["fixture counter receipt"], cpu_active_ns=work,
            physical_interval_verified=True, numerical_verified=True)

    def gpu(self, ident="gpu0", *, begin=0, end=10, work=20, role="before_target"):
        return dict(sample_id=ident, sample_role=role, resource="gpu", work=work,
            work_unit="flop", begin_ns=begin, end_ns=end,
            timing_scope="device_compute_active", sensor_window=dict(begin_ns=begin, end_ns=end),
            context={"fixture": "reference"}, clock_id="qpc-ns", clock_origin="go",
            evidence=["fixture correlated kernel receipt"], kernel_intervals=[dict(begin_ns=begin, end_ns=end)],
            gpu_intervals_verified=True, numerical_verified=True)

    def build(self, rows, **kwargs):
        arguments=dict(context={"fixture": "reference"}, clock_id="qpc-ns", clock_origin="go",
                       horizon_ns=100, max_hold_ns=100)
        arguments.update(kwargs)
        return build_reference_supply(rows, **arguments)

    def segment(self, artifact, at):
        return next(s for s in artifact["segments"] if s["begin_ns"] <= at < s["end_ns"])

    def test_cpu_active_service_over_wall_window(self):
        d=self.build([self.cpu()])
        self.assertIsNone(self.segment(d, 9)["rates"]["cpu"])
        self.assertEqual(self.segment(d, 10)["rates"]["cpu"], {"numerator":500_000_000,"denominator":1})
        self.assertEqual(d["estimate_kind"], "reference_workload_effective_service_estimate")
        self.assertEqual(d["source_kind"], "observed_diagnostic")
        self.assertFalse(d["universal_capacity_identified"])
        self.assertFalse(d["hardware_accuracy_qualified"])

    def test_gpu_uses_correlated_union_not_stream_envelope(self):
        row=self.gpu(work=9)
        row["kernel_intervals"]=[dict(begin_ns=2,end_ns=4),dict(begin_ns=3,end_ns=5)]
        d=self.build([row])
        self.assertEqual(d["sample_estimates"][0]["rate_denominator_ns"],3)
        self.assertEqual(self.segment(d,10)["rates"]["gpu"],{"numerator":3_000_000_000,"denominator":1})

    def test_host_sensor_finish_controls_causal_availability(self):
        row=self.gpu(begin=2,end=4,work=4)
        row["sensor_window"]=dict(begin_ns=0,end_ns=10)
        d=self.build([row])
        self.assertIsNone(self.segment(d,8)["rates"]["gpu"])
        self.assertIsNotNone(self.segment(d,10)["rates"]["gpu"])

    def test_after_reference_drift_never_updates_forward_rate(self):
        before=self.gpu()
        after=self.gpu("after",begin=30,end=40,work=40,role="after_target")
        d=self.build([before,after],target_windows=[dict(begin_ns=10,end_ns=20)])
        self.assertEqual(self.segment(d,20)["rates"]["gpu"],self.segment(d,50)["rates"]["gpu"])
        self.assertEqual(d["after_reference_drift"][0]["after_to_before_rate_ratio"],{"numerator":2,"denominator":1})
        self.assertFalse(d["after_reference_drift"][0]["used_for_forward_supply"])

    def test_only_after_reference_cannot_create_forward_coverage(self):
        d=self.build([self.gpu(role="after_target")])
        self.assertTrue(all(s["rates"]["gpu"] is None for s in d["segments"]))

    def test_cadence_expiry_is_unknown_not_last_rate_forever(self):
        d=self.build([self.cpu()],max_hold_ns=10)
        self.assertIsNotNone(self.segment(d,19)["rates"]["cpu"])
        self.assertIsNone(self.segment(d,20)["rates"]["cpu"])
        with self.assertRaisesRegex(ValueError,"stale"):
            materialize_reference_window(d,10,21)

    def test_coverage_before_first_completed_sample_is_unknown(self):
        d=self.build([self.cpu()])
        with self.assertRaisesRegex(ValueError,"unknown reference coverage"):
            materialize_reference_window(d,9,20)

    def test_new_failed_before_reference_does_not_keep_previous_good_value(self):
        first=self.gpu()
        second=self.gpu("failed",begin=30,end=40)
        second["gpu_intervals_verified"]=False
        d=self.build([first,second])
        self.assertIsNotNone(self.segment(d,20)["rates"]["gpu"])
        self.assertIsNone(self.segment(d,40)["rates"]["gpu"])

    def test_sensor_window_overlapping_target_is_rejected(self):
        row=self.gpu(begin=0,end=5)
        row["sensor_window"]["end_ns"]=20
        with self.assertRaisesRegex(ValueError,"overlaps a target"):
            self.build([row],target_windows=[dict(begin_ns=10,end_ns=25)])

    def test_disjoint_half_open_sensor_target_boundary_allowed(self):
        d=self.build([self.gpu()],target_windows=[dict(begin_ns=10,end_ns=20)])
        self.assertIsNotNone(self.segment(d,10)["rates"]["gpu"])

    def test_overlapping_sensor_windows_rejected(self):
        with self.assertRaisesRegex(ValueError,"sensor windows overlap"):
            self.build([self.cpu(),self.gpu(begin=5,end=15)])

    def test_duplicate_sample_identity_rejected(self):
        with self.assertRaisesRegex(ValueError,"duplicate"):
            self.build([self.gpu(),self.gpu(begin=20,end=30)])

    def test_wrong_clock_and_context_rejected(self):
        for key,value in (("clock_id","different"),("clock_origin","different"),("context",{"fixture":"different"})):
            row=self.gpu();row[key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,"context or clock"):
                self.build([row])

    def test_inclusive_wall_scope_refused(self):
        row=self.gpu();row["timing_scope"]="whole_action_completed"
        with self.assertRaisesRegex(ValueError,"inclusive host"):
            self.build([row])

    def test_unverified_gpu_identity_retains_unknown(self):
        row=self.gpu();row.pop("gpu_intervals_verified")
        d=self.build([row])
        self.assertIsNotNone(d["sample_estimates"][0]["candidate_rate"])
        self.assertIsNone(d["sample_estimates"][0]["rate"])

    def test_empty_or_outside_gpu_activity_not_fabricated(self):
        row=self.gpu();row["kernel_intervals"]=[]
        self.assertIsNone(self.build([row])["sample_estimates"][0]["rate"])
        row["kernel_intervals"]=[dict(begin_ns=0,end_ns=11)]
        with self.assertRaisesRegex(ValueError,"outside physical"):
            self.build([row])

    def test_cpu_work_must_match_counter(self):
        row=self.cpu();row["cpu_active_ns"]=4
        with self.assertRaisesRegex(ValueError,"differs from CPU"):
            self.build([row])

    def test_cpu_service_above_one_thread_not_clipped(self):
        row=self.cpu(work=11)
        d=self.build([row])
        self.assertIsNone(d["sample_estimates"][0]["rate"])
        self.assertEqual(d["sample_estimates"][0]["candidate_rate"],{"numerator":1_100_000_000,"denominator":1})

    def test_unknown_work_not_zero(self):
        row=self.gpu();row["work"]=None
        d=self.build([row]);self.assertIsNone(d["sample_estimates"][0]["candidate_rate"])
        self.assertIn("reference_work_unknown",d["sample_estimates"][0]["evidence_issues"])

    def test_float_work_and_boolean_time_rejected(self):
        row=self.gpu();row["work"]=2.0
        with self.assertRaises(ValueError):self.build([row])
        row=self.gpu();row["begin_ns"]=False
        with self.assertRaises(ValueError):self.build([row])

    def test_target_durations_do_not_change_estimated_rates(self):
        left=self.build([self.gpu()],target_windows=[dict(begin_ns=10,end_ns=20)])
        right=self.build([self.gpu()],target_windows=[dict(begin_ns=15,end_ns=50)])
        self.assertEqual(left["segments"],right["segments"])
        self.assertEqual(left["sample_estimates"],right["sample_estimates"])
        self.assertFalse(left["target_timing_used_to_compute_rates"])

    def test_materialized_window_is_explicit_diagnostic_only(self):
        d=self.build([self.cpu()]);available=materialize_reference_window(d,10,40)
        graph=Graph((Task("cpu","lane",(WorkDemand("work",10,"thread_cpu_ns"),)),))
        bindings=(ServiceBinding("work","cpu",1,"declared",(),"fixture"),)
        with self.assertRaisesRegex(InvalidModel,"diagnostic"):
            available.evaluate(graph,ParameterBundle(()),bindings,start_ns=10)
        result=available.evaluate(graph,ParameterBundle(()),bindings,start_ns=10,allow_observed_diagnostic=True)
        self.assertEqual(result.duration_ns,20)
        self.assertFalse(result.accuracy_qualified)

    def test_materialized_window_rejects_unknown_resource_or_horizon(self):
        d=self.build([self.cpu()])
        with self.assertRaises(ValueError):materialize_reference_window(d,10,101)
        with self.assertRaises(ValueError):materialize_reference_window(d,10,20,resources=("missing",))

    def test_semantic_tamper_cannot_be_hidden_by_rehash(self):
        d=self.build([self.gpu()]);d=deepcopy(d)
        d["segments"][-1]["rates"]["gpu"]={"numerator":1,"denominator":1}
        d["artifact_sha256"]=digest({k:v for k,v in d.items() if k!="artifact_sha256"})
        with self.assertRaisesRegex(ValueError,"does not reconstruct"):
            validate_reference_supply(d)

    def test_exact_serialization_roundtrip(self):
        d=self.build([self.cpu(),self.gpu(begin=20,end=30)])
        loaded=json.loads(json.dumps(d));self.assertEqual(validate_reference_supply(loaded),d)
        available=materialize_reference_window(d,30,50)
        self.assertEqual(available.value_at("cpu",30),500_000_000)
        self.assertEqual(available.value_at("gpu",30),2_000_000_000)


if __name__ == "__main__":
    unittest.main()
