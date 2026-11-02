"""Small causal/profile-only checks; no native process or hardware is used."""
from copy import deepcopy
import unittest

from profiling.host_supply_calibration import (
    ANCHOR_SCHEMA, OBSERVATION_SCHEMA, bind_before_phase_anchor,
    calibrate_blackout_hypothesis,
)


def observations():
    result=[]
    for run in range(3):
        cursor=run*1_000_000_000
        for iteration in range(12):
            pause=32_000_000 if iteration in {2,5,8} else 0
            left=cursor;middle=left+1_000_000+pause;right=middle+1_000_000
            result.append(dict(schema=OBSERVATION_SCHEMA,observation_id=f'r{run}:i{iteration}',
                run_id=f'r{run}',process_id=100+run,seed=1000+run,split='profile',signature='exact-executor-v1',
                resource_id='cpu:lp2:host',competitor_context=dict(external_actors=[dict(resource_id='cpu:lp2',
                    workload_signature='fixed-projection-f32-shape',placement=dict(logical_cpu_ids=[2]))]),
                clock_id='qpc_ns',clock_origin='qpc_zero',begin_ns=left,end_ns=right,
                actions=[dict(action_index=0,begin_ns=left,end_ns=middle,active_cycles=1000,
                              active_cycles_scope='main_thread_raw_cycles'),
                         dict(action_index=1,begin_ns=middle,end_ns=right)],
                numerical_verified=True,evidence=[f'raw:r{run}']))
            cursor=right
    return result


def freeze(rows=None):
    return calibrate_blackout_hypothesis(observations() if rows is None else rows,'exact-executor-v1',
        minimum_excess_ns=15_000_000)


def before(frozen):
    return dict(schema=ANCHOR_SCHEMA,sample_role='before_target',source_role='external_competitor',
        derived_from_target_timing=False,uses_future_observations=False,
        signature_sha256=frozen['signature_sha256'],resource_id='cpu:lp2:host',
        competitor_context=deepcopy(frozen['context']['competitor_context']),
        clock_id='qpc_ns',clock_origin='qpc_zero',sensor_begin_ns=10_000_000_000,
        sensor_end_ns=10_010_000_000,phase_start_window_ns=dict(minimum=10_001_000_000,maximum=10_001_100_000),
        observed_process_ids=[900],source_observation_ids=['external-900:observation'],evidence=['external-sensor-raw'])


class HostSupplyCalibrationTests(unittest.TestCase):
    def test_period_width_and_phase_unknown(self):
        result=freeze()
        self.assertEqual(result['period_candidate_ns'],38_000_000)
        self.assertEqual(result['pause_width_candidate_ns'],32_000_000)
        self.assertEqual(result['profile_event_candidates'],9)
        self.assertFalse(result['phase_identified'])
        self.assertIsNone(result['phase'])
        self.assertFalse(result['CPU_available_service_identified'])
        self.assertFalse(result['hardware_accuracy_qualified'])

    def test_cycles_are_raw_not_elapsed_time_or_zero_defaults(self):
        result=freeze()
        self.assertEqual(result['action_work_cycle_candidates']['0']['statistics']['median'],1000)
        missing=result['action_work_cycle_candidates']['1']
        self.assertEqual(missing['unknown_samples'],36)
        self.assertIsNone(missing['statistics']['median'])
        self.assertFalse(missing['cycle_to_time_conversion_performed'])

    def test_heldout_input_rejected_even_when_renamed_run(self):
        rows=observations();rows[0]['split']='heldout';rows[0]['run_id']='renamed'
        with self.assertRaisesRegex(ValueError,'only PROFILE'):freeze(rows)

    def test_minimum_independent_runs(self):
        with self.assertRaisesRegex(ValueError,'insufficient independent'):freeze(observations()[:24])

    def test_duplicate_native_identity_rejected(self):
        rows=observations()
        for row in rows[12:24]:row.update(process_id=100,seed=1000)
        with self.assertRaisesRegex(ValueError,'renamed duplicate'):freeze(rows)

    def test_mixed_physical_competitor_context_rejected(self):
        rows=observations();rows[0]['competitor_context']['external_actors'][0]['placement']['logical_cpu_ids']=[3]
        with self.assertRaisesRegex(ValueError,'mixed resource'):freeze(rows)

    def test_state_label_is_not_predictor_context(self):
        rows=observations();rows[0]['competitor_context']={'state':'cpu'}
        with self.assertRaisesRegex(ValueError,'state labels'):freeze(rows)

    def test_no_events_remains_unknown(self):
        rows=observations()
        for row in rows:
            left=row['begin_ns'];row['actions'][0]['end_ns']=left+1_000_000
            row['actions'][1].update(begin_ns=left+1_000_000,end_ns=left+2_000_000)
        result=freeze(rows)
        self.assertIsNone(result['period_candidate_ns'])
        self.assertIsNone(result['pause_width_candidate_ns'])
        self.assertFalse(result['coverage_qualified'])

    def test_before_anchor_is_scoped_diagnostic(self):
        frozen=freeze();result=bind_before_phase_anchor(frozen,before(frozen),
            prediction_begin_ns=10_020_000_000,target_process_id=901)
        self.assertTrue(result['phase_anchor_causality_checks_passed'])
        self.assertFalse(result['physical_CPU_supply_identified'])
        self.assertFalse(result['hardware_accuracy_qualified'])

    def test_after_target_anchor_rejected(self):
        frozen=freeze();anchor=before(frozen);anchor['sample_role']='after_target'
        with self.assertRaisesRegex(ValueError,'target/AFTER leakage'):
            bind_before_phase_anchor(frozen,anchor,prediction_begin_ns=10_020_000_000,target_process_id=901)

    def test_sensor_crossing_cutoff_rejected(self):
        frozen=freeze()
        with self.assertRaisesRegex(ValueError,'crosses target'):
            bind_before_phase_anchor(frozen,before(frozen),prediction_begin_ns=10_005_000_000,target_process_id=901)

    def test_target_process_counters_rejected(self):
        frozen=freeze()
        with self.assertRaisesRegex(ValueError,'own phase'):
            bind_before_phase_anchor(frozen,before(frozen),prediction_begin_ns=10_020_000_000,target_process_id=900)

    def test_target_observation_ids_rejected(self):
        frozen=freeze()
        with self.assertRaisesRegex(ValueError,'renamed into external'):
            bind_before_phase_anchor(frozen,before(frozen),prediction_begin_ns=10_020_000_000,
                target_process_id=901,target_observation_ids=['external-900:observation'])

    def test_target_derived_phase_flag_rejected(self):
        frozen=freeze();anchor=before(frozen);anchor['derived_from_target_timing']=True
        with self.assertRaisesRegex(ValueError,'target/AFTER leakage'):
            bind_before_phase_anchor(frozen,anchor,prediction_begin_ns=10_020_000_000,target_process_id=901)

    def test_frozen_mutation_rejected(self):
        frozen=freeze();anchor=before(frozen);frozen['pause_width_candidate_ns']=1
        with self.assertRaisesRegex(ValueError,'digest differs'):
            bind_before_phase_anchor(frozen,anchor,prediction_begin_ns=10_020_000_000,target_process_id=901)

    def test_unknown_period_cannot_acquire_fake_phase(self):
        rows=observations()
        for row in rows:
            for action in row['actions']:
                action['end_ns']=action['begin_ns']
        frozen=freeze(rows)
        with self.assertRaisesRegex(ValueError,'period/width unknown'):
            bind_before_phase_anchor(frozen,before(frozen),prediction_begin_ns=10_020_000_000,target_process_id=901)


if __name__=='__main__':unittest.main()
