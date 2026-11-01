from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest

from profiling.event_stage_evaluate import (
    CANDIDATES, _write_new, build_response, make_observers, metrics,
    phase_age_diagnostic, predict_call,
)
from llm_infer_model.performance.host_pause_observer import CompletedHostAction


def frozen():
    return dict(artifact_sha256="frozen-profile-only", profile_scalar_wall_ns=220,
        baseline_gpu_reference_wall_ns=100,
        actions={"0":dict(kind="COMPUTE", cpu_work_affine=dict(identified=True,intercept=100,slope=0),
            completion_response_affine=dict(identified=True,intercept=100,slope=0), fast_wall_ns=200)},
        gap_cycles={}, head_wall_bounds=dict(minimum_statistics=dict(median=10)),
        tail_wall_bounds=dict(minimum_statistics=dict(median=10)),
        clock=dict(clock_id="test-clock",clock_origin="test-zero"),
        declared_protocol=dict(minimum_excess_ns=15),
        evaluation_protocol=dict(jitter_ns=2,calendar_horizon_ns=10_000),
        CPU_blackout_hypothesis=dict(period_candidate_ns=1000,pause_width_candidate_ns=100,
            artifact_sha256="profile-pause-hypothesis"))


def before(phase=False):
    return dict(gpu_reference_fast_action_wall_ns=100, cpu_cycle_rate_per_s=1_000_000_000,
        evidence=dict(cpu=["independent BEFORE CPU wall"],gpu=["independent BEFORE GPU wall"]),
        before_phase=dict(known=phase,pause_end_window_ns=dict(minimum=100,maximum=100)))


def prediction(f, b, begin=300):
    compiled,bundle,_=build_response(f,b)
    observers=make_observers(f,b,cpu_reference_cutoff_ns=200)
    snapshots={mode:observer.predict(begin,call_id="call-0") for mode,observer in observers.items()}
    return predict_call(f,compiled,bundle,snapshots,begin_ns=begin,call_id="call-0"),observers


class EventStageEvaluateTests(unittest.TestCase):
    def test_declares_all_variants_without_best_selection(self):
        self.assertEqual(len(CANDIDATES),7)
        row,_=prediction(frozen(),before())
        self.assertEqual(set(row["predictions"]),set(CANDIDATES))

    def test_unknown_phase_does_not_erase_no_calendar_prediction(self):
        row,_=prediction(frozen(),before())
        values=row["predictions"]
        self.assertEqual(values["cycle_after_host_no_calendar"]["predicted_ns"],220)
        self.assertEqual(values["cycle_overlap_host_no_calendar"]["predicted_ns"],120)
        self.assertFalse(values["cycle_after_host_before_only"]["known"])
        self.assertIsNone(values["cycle_after_host_before_only"]["predicted_ns"])

    def test_calendars_use_common_graph_without_target_truth(self):
        row,_=prediction(frozen(),before(True))
        self.assertEqual(row["predictions"]["cycle_after_host_before_only"]["predicted_ns"],220)
        self.assertFalse(row["current_target_end_read"])
        self.assertFalse(row["current_target_actions_read"])
        self.assertFalse(row["current_target_cycles_read"])

    def test_entry_blackout_conflict_is_retained(self):
        row,_=prediction(frozen(),before(True),begin=1000)
        candidate=row["predictions"]["cycle_after_host_before_only"]
        self.assertTrue(candidate["call_begin_inside_blackout_conflict"])
        self.assertTrue(candidate["known"])
        self.assertEqual(candidate["predicted_ns"],310)

    def test_unidentified_fit_is_unknown_not_median_fallback(self):
        f=frozen();f["actions"]["0"]["cpu_work_affine"]={"identified":False,"intercept":None,"slope":None}
        f["actions"]["0"]["cpu_cycles_ceiling"]=100
        row,_=prediction(f,before(True))
        self.assertFalse(row["predictions"]["cycle_after_host_no_calendar"]["known"])
        self.assertTrue(row["predictions"]["profile_scalar_median"]["known"])

    def test_missing_nonfirst_gap_remains_unknown(self):
        f=frozen();f["actions"]["1"]=deepcopy(f["actions"]["0"])
        compiled,_,costs=build_response(f,before())
        self.assertEqual(compiled,{})
        self.assertIsNone(costs["stages"][1]["gap_cycles"])
        self.assertEqual(costs["stages"][0]["gap_cycles"],0)

    def test_prediction_immutable_under_future_target_mutations(self):
        f,b=frozen(),before(True)
        old,_=prediction(f,b)
        unrelated_target=dict(end_ns=400,actions=[dict(begin_ns=300,end_ns=400,thread_cycles=dict(delta_cycles=50))])
        unrelated_target["end_ns"]=999_999
        unrelated_target["actions"][0]["thread_cycles"]=dict(delta_cycles=999_999_999)
        later,_=prediction(f,b)
        self.assertEqual(old,later)

    def test_completed_past_history_updates_only_causal_candidate(self):
        f,b=frozen(),before(False)
        compiled,bundle,_=build_response(f,b)
        first,observers=prediction(f,b)
        sealed=json.dumps(first,sort_keys=True)
        for observer in observers.values():
            observer.observe_completed("call-0",300,600,
                actions=(CompletedHostAction("0",310,580),),observed_at_ns=600,
                clock_id="test-clock",clock_origin="test-zero")
        phases={mode:observer.predict(601,call_id="call-1") for mode,observer in observers.items()}
        second=predict_call(f,compiled,bundle,phases,begin_ns=601,call_id="call-1")
        self.assertFalse(second["predictions"]["cycle_after_host_before_only"]["known"])
        self.assertTrue(second["predictions"]["cycle_after_host_causal_history"]["known"])
        self.assertEqual(json.dumps(first,sort_keys=True),sealed)
        self.assertLess(second["predictions"]["cycle_after_host_causal_history"]["past_target_cutoff_ns"],601)

    def test_long_range_phase_uncertainty_becomes_unknown(self):
        f=frozen();f["evaluation_protocol"]["jitter_ns"]=100
        row,_=prediction(f,before(True),begin=10_000)
        candidate=row["predictions"]["cycle_after_host_before_only"]
        self.assertFalse(candidate["known"])
        self.assertEqual(candidate["reason"],"extrapolated_phase_uncertainty_spans_period")
        self.assertGreater(candidate["phase_age_diagnostic"]["extrapolated_cycles"],1)
        self.assertTrue(row["predictions"]["cycle_after_host_no_calendar"]["known"])

    def test_before_pause_end_after_cutoff_is_rejected(self):
        b=before(True);b["before_phase"]["pause_end_window_ns"]["maximum"]=201
        with self.assertRaisesRegex(ValueError,"cutoff"):
            make_observers(frozen(),b,cpu_reference_cutoff_ns=200)

    def test_conditional_accuracy_cannot_pass_full_coverage(self):
        row,_=prediction(frozen(),before(False))
        row["observed_wall_ns"]=220
        result=metrics([row],expected_count=2)
        self.assertEqual(result["profile_scalar_median"]["known_subset_median_ape"],0)
        self.assertEqual(result["profile_scalar_median"]["coverage"],.5)
        self.assertFalse(result["profile_scalar_median"]["point_accuracy_gate"])

    def test_all_slow_calls_remain_in_metric_denominator(self):
        row,_=prediction(frozen(),before(False));row["observed_wall_ns"]=220
        slow=deepcopy(row);slow["observed_wall_ns"]=22000
        result=metrics([row,slow],expected_count=2)["profile_scalar_median"]
        self.assertEqual(result["total_calls"],2)
        self.assertEqual(result["known_calls"],2)
        self.assertEqual(result["within_20_percent"],1)
        self.assertGreater(result["known_subset_p90_ape"],.9)

    def test_write_once_and_lf(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path=Path(directory)/"frozen.json"
            _write_new(path,{"value":"explicit"})
            self.assertNotIn(b"\r",path.read_bytes())
            with self.assertRaises(FileExistsError):
                _write_new(path,{"value":"changed"})


if __name__=="__main__":
    unittest.main()
