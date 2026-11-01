from copy import deepcopy
import unittest

from profiling.h2d_history.independent_audit import (
    close, higher_p90, independent_coordinates, numeric_equal, recompute_case,
)


def program():
    def tensor(tid, size):
        return dict(id=tid, dtype='GGML_TYPE_F32', ne=[size//4, 1, 1, 1],
                    nb=[4, size, size, size], storage_bytes=size)
    return dict(boundary_contract='completed-action-test',
        graph=dict(workload=dict(tensors=[tensor('a', 8), tensor('b', 16)])),
        actions=[dict(kind='COPY', target_id='a', src='dram', dst='vram'),
                 dict(kind='COPY', target_id='b', src='dram', dst='vram'), dict(kind='ADVANCE')],
        physical=dict(config={}, threads=1), bindings=[])


def records(actual=100):
    return [dict(job_id=str(i)+'-'+instrument, program_key='case', instrumentation=instrument,
                 round=i, status='completed', values=[actual]*16, numeric=True)
            for i in range(4) for instrument in ('bare', 'stamped')]


class H2DHistoryIndependentAuditTests(unittest.TestCase):
    def test_independent_reuse_coordinates_need_no_timing_or_source_ids(self):
        original = program()
        other = deepcopy(original)
        for t in other['graph']['workload']['tensors']:
            t['id'] = 'renamed:'+t['id']
        for a in other['actions']:
            if 'target_id' in a:
                a['target_id'] = 'renamed:'+a['target_id']
        a = independent_coordinates(original, 'a'*64, repeats=2)
        b = independent_coordinates(other, 'a'*64, repeats=2)
        self.assertEqual(a, b)
        self.assertEqual(a[0]['conditional']['condition'], {'h2d_source_reuse_bucket':'16'})
        self.assertEqual(a[1]['conditional']['condition'], {'h2d_source_reuse_bucket':'8'})
        self.assertEqual(a[2]['conditional']['condition'], {})

    def test_failed_scientific_gate_returns_valid_negative_result(self):
        prediction = dict(mean_replay_ns='100', trial_mean_replay_ns=['100']*16)
        good = recompute_case(prediction, records())
        self.assertTrue(good['fully_qualified'])
        poor = recompute_case(prediction, records(200))
        self.assertFalse(poor['fully_qualified'])
        self.assertIn('each_primary_process_mean_ape', poor['failed_gates'])
        self.assertEqual(poor['maximum_primary_process_mean_ape'], .5)
        self.assertEqual(poor['maximum_primary_process_cv'], 0)

    def test_failed_process_remains_in_denominator(self):
        prediction = dict(mean_replay_ns='100', trial_mean_replay_ns=['100']*16)
        actual = records()
        actual[0].update(status='failed', values=[], numeric=False)
        result = recompute_case(prediction, actual)
        self.assertFalse(result['fully_qualified'])
        self.assertIn('all_declared_processes_completed', result['failed_gates'])
        self.assertIsNone(result['processes'][actual[0]['job_id']])

    def test_observer_and_primary_cv_gates_are_reported_separately(self):
        prediction = dict(mean_replay_ns='100', trial_mean_replay_ns=['100']*16)
        actual = records()
        for r in actual:
            if r['instrumentation'] == 'stamped':
                r['values'] = [50, 150]*8
        result = recompute_case(prediction, actual)
        self.assertTrue(result['qualified_original_primary_cv_gate'])
        self.assertFalse(result['qualified_conservative_all_process_cv_gate'])
        self.assertEqual(result['failed_conservative_gates'], ['each_observer_process_cv'])

    def test_p90_uses_higher_order_statistic_and_prediction_tampering_rejects(self):
        self.assertEqual(higher_p90(list(range(10))), 9)
        with self.assertRaises(ValueError):
            recompute_case(dict(mean_replay_ns='101', trial_mean_replay_ns=['100']*16), records())
        with self.assertRaises(ValueError):
            recompute_case(dict(mean_replay_ns='100', trial_mean_replay_ns=['100']*15), records())

    def test_numeric_oracle_is_finite_shape_sensitive_and_tolerance_bound(self):
        self.assertTrue(numeric_equal([[1, 2]], [[1, 2]]))
        self.assertFalse(numeric_equal([[1]], [[1, 2]]))
        self.assertFalse(numeric_equal([float('nan')], [0]))
        self.assertFalse(numeric_equal([True], [1]))
        self.assertFalse(numeric_equal([1.01], [1]))
        self.assertFalse(close(None, 1))
        self.assertTrue(close(None, None))


if __name__ == '__main__':
    unittest.main()
