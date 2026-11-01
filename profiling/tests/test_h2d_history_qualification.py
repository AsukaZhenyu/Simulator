import unittest

from profiling.h2d_history.qualification import case_metrics,numeric_equal,quantile


class HistoryIndependentGateTests(unittest.TestCase):
    def processes(self):
        rows=[]
        for round_id in range(4):
            for instrument in ('bare','stamped'):
                rows.append(dict(job_id=f'heldout-r{round_id}-test-{instrument}',program_key='test',round=round_id,
                    instrumentation=instrument,status='completed',mean_ns=100.,trial_cv=0.,
                    trial_mean_replay_ns=[100.]*16,early_mean_ns=100.,late_mean_ns=100.,late_over_early_minus_one=0.,
                    numeric_oracle_passed=True,native_repeat_consistency_passed=True))
        return rows

    def prediction(self):
        return dict(mean_replay_ns='100',trial_mean_replay_ns=['100']*16)

    def test_p90_preserves_previous_higher_order_statistic(self):
        self.assertEqual(quantile([0,10,20,30],.9),30)
        self.assertEqual(quantile(list(range(64)),.9),57)

    def test_all_original_and_secondary_gates_pass(self):
        result,blocks=case_metrics('history_conditional','test',self.prediction(),self.processes())
        self.assertTrue(result['fully_qualified'])
        self.assertTrue(result['qualified_conservative_all_process_cv_gate'])
        self.assertEqual(len(blocks),64)
        self.assertEqual(result['pooled_primary_block_ape_p90'],0.)

    def test_observer_cv_is_secondary_not_original_gate(self):
        rows=self.processes()
        for row in rows:
            if row['instrumentation']=='stamped':
                row['trial_cv']=.21
                row['trial_mean_replay_ns']=[80.,120.]*8
        result,_=case_metrics('context_static','test',self.prediction(),rows)
        self.assertTrue(result['fully_qualified'])
        self.assertFalse(result['qualified_conservative_all_process_cv_gate'])
        self.assertEqual(result['failed_gates'],[])
        self.assertEqual(result['failed_conservative_gates'],['each_observer_process_cv'])

    def test_failed_native_process_remains_in_denominator(self):
        rows=self.processes(); row=rows[0]
        row.update(status='failed',mean_ns=None,trial_cv=None,trial_mean_replay_ns=[],
                   numeric_oracle_passed=False,native_repeat_consistency_passed=False)
        result,blocks=case_metrics('old_hot','test',self.prediction(),rows)
        self.assertEqual(result['declared_primary_processes'],4)
        self.assertEqual(result['completed_primary_processes'],3)
        self.assertEqual(len(blocks),48)
        self.assertFalse(result['fully_qualified'])

    def test_block_accuracy_cannot_be_hidden_by_accurate_overall_mean(self):
        prediction=dict(mean_replay_ns='100',trial_mean_replay_ns=['60','140']*8)
        result,_=case_metrics('history_conditional','test',prediction,self.processes())
        self.assertEqual(result['maximum_primary_process_mean_ape'],0.)
        self.assertEqual(result['pooled_primary_block_ape_p90'],.4)
        self.assertFalse(result['fully_qualified'])

    def test_future_provider_mean_cannot_disagree_with_frozen_block_predictions(self):
        prediction=self.prediction(); prediction['mean_replay_ns']='101'
        with self.assertRaises(ValueError):
            case_metrics('context_static','test',prediction,self.processes())

    def test_numeric_oracle_checks_shape_and_finiteness(self):
        self.assertTrue(numeric_equal([[1.,2.]],[[1.,2.00001]]))
        self.assertFalse(numeric_equal([1.,2.],[[1.,2.]]))
        self.assertFalse(numeric_equal([float('nan')],[1.]))


if __name__=='__main__': unittest.main()
