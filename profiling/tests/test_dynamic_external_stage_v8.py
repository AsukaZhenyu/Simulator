"""Protocol regressions: causality, observer gating and heldout isolation."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from profiling.dynamic_validation import external_stage_v8 as v8


class StageProtocolTests(unittest.TestCase):
    def template(self):
        return {d+'_actor': {'device': d, 'component': {'kind': 'GEMM'},
                            'intervals_ms': []} for d in ('cpu', 'gpu')}

    def jobs(self):
        return v8.job_plans(self.template(), 'a'*64, 'b'*64, 'c'*64)

    def test_exact_inventory_and_constant_windows(self):
        jobs = self.jobs()
        self.assertEqual(len(jobs), 30)
        self.assertEqual(len({j['id'] for j in jobs}), 30)
        self.assertEqual([sum(j['stage'] == s for j in jobs) for s in
                          ('00_observer_gate', 'ON_profiles', 'heldout')], [6,18,6])
        for j in jobs[:24]:
            r = j['request']
            self.assertEqual((r['replays'], r['block_replays']), (1,1))
            for i,d in enumerate(('cpu','gpu')):
                self.assertEqual(r[d+'_actor']['intervals_ms'],
                                 [[0,30000]] if j['offered_state'][i] else [])
            self.assertEqual(r['target_entry_offset_ms'],
                             1000 if any(j['offered_state']) else 0)

    def test_counter_order_not_always_control_first(self):
        jobs = self.jobs()
        for offset in (0,6,12,18):
            self.assertEqual([j['request']['instrumentation'] for j in jobs[offset:offset+6]],
                             ['stamped','runtime_callbacks','runtime_callbacks',
                              'stamped','stamped','runtime_callbacks'])

    def test_heldout_independent_fixed_input_pulses(self):
        jobs = self.jobs()[24:]
        for j in jobs:
            r = j['request']
            self.assertEqual((r['replays'],r['block_replays']), (256,16))
            self.assertEqual(r['instrumentation'], 'stamped')
            self.assertIsNone(r['offered_state'])
            for d in ('cpu','gpu'):
                self.assertTrue(all(100 <= a < b <= 2000
                                    for a,b in r[d+'_actor']['intervals_ms']))
        # Mutating one request cannot rewrite the plan or another process.
        jobs[0]['request']['cpu_actor']['intervals_ms'][0][0] = 999
        self.assertEqual(v8.PULSES['early_joint']['cpu'][0][0], 150)
        self.assertEqual(jobs[1]['request']['cpu_actor']['intervals_ms'][0][0],150)

    def test_gate_retains_slow_pair_and_missing_denominator(self):
        jobs=self.jobs()[:6]
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)
            v8.write(out/'manifest.json',dict(jobs=jobs))
            for i,j in enumerate(jobs):
                folder=out/'jobs'/j['id'];folder.mkdir(parents=True)
                v8.write(folder/'result.json',dict(target_wall_ns=200 if i==5 else 100))
                v8.write(folder/'execution.json',dict(status='completed',native=v8.receipt(folder/'result.json')))
            result=v8.observer_gate(out,'00_observer_gate')
            self.assertFalse(result['passed'])
            self.assertEqual(result['declared_pairs'],3)
            self.assertEqual(len(result['pairs']),3)
            self.assertEqual(sum(p['passed'] for p in result['pairs']),2)

    def test_missing_pair_is_not_removed_from_denominator(self):
        jobs=self.jobs()[:6]
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);v8.write(out/'manifest.json',dict(jobs=jobs))
            for j in jobs[1:]:
                folder=out/'jobs'/j['id'];folder.mkdir(parents=True)
                v8.write(folder/'result.json',dict(target_wall_ns=100))
                v8.write(folder/'execution.json',dict(status='completed',native=v8.receipt(folder/'result.json')))
            result=v8.observer_gate(out,'00_observer_gate')
            self.assertFalse(result['passed'])
            self.assertEqual(result['declared_pairs'],3)
            self.assertEqual(result['missing_or_failed_jobs'],[jobs[0]['id']])

    def test_on_refuses_failed_off_gate(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);v8.write(out/'manifest.json',dict(jobs=[]))
            v8.write(out/'ON_profiles-authorization.json',dict(collection_authorized=True,
                stage='ON_profiles',manifest_sha256=v8.sha(out/'manifest.json')))
            v8.write(out/'00_observer_gate-observer-gate.json',dict(passed=False))
            with self.assertRaises(ValueError): v8.authorization(out,'ON_profiles')

    def test_no_unmeasured_transition_parameter_or_whole_model_claim(self):
        p=v8.protocol()
        self.assertFalse(p['whole_LLM_dynamic_accuracy_claimed'])
        self.assertFalse(p['macro_baseline_fitted_from_whole_target'])
        self.assertFalse(p['API_response_is_CPU_active'])
        self.assertFalse(p['internal_GPU_cost_added'])
        self.assertEqual(p['acceptance']['model_mean_APE_max'],.10)
        self.assertEqual(p['acceptance']['model_block_P90_APE_max'],.20)
        self.assertIn('no default',p['tau'])


if __name__=='__main__':unittest.main()
