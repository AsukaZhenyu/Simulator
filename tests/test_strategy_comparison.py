"""Handwritten experimental-protocol checks, not latency-model self tests."""
import unittest
from dataclasses import asdict

from experiments.strategy_comparison import CANDIDATES, candidate_spec, relative_rows, paired_interactions


class StrategyComparisonProtocolTests(unittest.TestCase):
    def test_storage_ablation_preserves_budget_and_execution(self):
        self.assertEqual(len(CANDIDATES),8)
        for kind in ('dense','moe'):
            for e in range(4):
                a=asdict(candidate_spec(kind,f'e{e}-s0'))
                b=asdict(candidate_spec(kind,f'e{e}-s1'))
                self.assertNotEqual(a.pop('prefetch'),b.pop('prefetch'))
                self.assertEqual(a,b)

    def test_failed_partial_run_has_no_speedup(self):
        rows=[dict(model_kind='dense',candidate_id='e0-s0',success=False,
                   completed_makespan_ns=None,makespan_ns=1),
              dict(model_kind='dense',candidate_id='e1-s0',success=True,
                   completed_makespan_ns=100)]
        self.assertTrue(all(r['reference_speedup'] is None for r in relative_rows(rows)))

    def test_interaction_and_reversal_have_independent_expected_values(self):
        # Baseline100, execution-alone80, storage-alone60, joint70:
        # execution flips from faster to slower and has negative synergy.
        times={'e0-s0':100,'e0-s1':60,'e1-s0':80,'e1-s1':70,
               'e2-s0':90,'e2-s1':40,'e3-s0':50,'e3-s1':None}
        rows=[dict(model_kind='dense',candidate_id=c,completed_makespan_ns=t) for c,t in times.items()]
        result={r['execution']:r for r in paired_interactions(rows)}
        self.assertTrue(result['e1']['strict_execution_order_reversal'])
        self.assertEqual(result['e1']['nonadditive_time_saving_ns'],-30)
        self.assertAlmostEqual(result['e1']['joint_gain_over_better_single'],-1/6)
        self.assertEqual(result['e2']['nonadditive_time_saving_ns'],10)
        self.assertFalse(result['e3']['complete_four_cells'])
        self.assertIsNone(result['e3']['nonadditive_time_saving_ns'])


if __name__=='__main__':
    unittest.main()
