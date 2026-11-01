"""Mutation tests for offline selection evidence, not reimplementations of runtime."""
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('paper_selection_study', Path(__file__).parents[1]/'experiments/paper_selection_study.py')
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


class SelectionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(strategy='a', predicted_ns=10, observed_ns=20, gates={'stable':True}),
                     dict(strategy='b', predicted_ns=11, observed_ns=10, gates={'stable':True})]
        self.processes={'a': [[20]*4]*3, 'b':[[10]*4]*3}

    def test_selected_by_prediction_not_actual_winner(self):
        r=study.selection('g',self.rows,self.processes)
        self.assertEqual(r['selected_strategy'],'a')
        self.assertEqual(r['observed_winner'],'b')
        self.assertEqual(r['selection_regret'],1)
        self.assertTrue(r['observed_winner_separated_from_all_alternatives'])
        self.assertFalse(r['qualified_decisive_winner_match'])

    def test_correct_winner_is_not_automatically_a_near_tie(self):
        self.rows[0]['predicted_ns']=12
        r=study.selection('g',self.rows,self.processes)
        self.assertTrue(r['qualified_decisive_winner_match'])
        self.assertFalse(r['mismatched_selection_inconclusive'])

    def test_sample_loss_rejected(self):
        self.processes['a']=[[20]*4,[20]*3,[20]*4]
        with self.assertRaises(ValueError):study.selection('g',self.rows,self.processes)

    def test_stability_failure_is_not_removed(self):
        self.rows[0]['gates']['stable']=False
        r=study.selection('g',self.rows,self.processes)
        self.assertFalse(r['all_original_gates_passed'])
        self.assertEqual(r['selected_strategy'],'a')

    def test_missing_candidate_fails_closed(self):
        del self.processes['b']
        with self.assertRaises(ValueError): study.selection('g',self.rows,self.processes)

    def test_duplicate_candidate_fails_closed(self):
        self.rows[1]['strategy']='a'
        with self.assertRaises(ValueError): study.selection('g',self.rows,self.processes)

    def test_duplicate_process_count_fails_closed(self):
        self.processes['a']*=2
        with self.assertRaises(ValueError): study.selection('g',self.rows,self.processes)

    def test_prediction_tie_is_deterministic(self):
        self.rows[1]['predicted_ns']=10
        self.assertEqual(study.selection('g',self.rows,self.processes)['selected_strategy'],'a')

    def test_uncertainty_uses_paired_whole_processes(self):
        # Large common drift must cancel; an unpaired call bootstrap would not.
        self.processes={'a':[[10]*4,[100]*4,[1000]*4], 'b':[[9]*4,[99]*4,[999]*4]}
        r=study.selection('g',self.rows,self.processes)
        self.assertEqual(r['selected_minus_winner_ns_95pct'],[1,1])
        self.assertEqual(r['cluster_draws'],27)

    def test_near_tie_is_inconclusive_not_pass(self):
        self.processes={'a':[[9]*4,[11]*4,[10]*4], 'b':[[11]*4,[9]*4,[10]*4]}
        self.rows[1]['gates']['stable']=False
        r=study.selection('g',self.rows,self.processes)
        self.assertTrue(r['selected_vs_winner_inconclusive'])
        self.assertFalse(r['all_original_gates_passed'])

    def test_missing_price_rejected(self):
        with self.assertRaises(ValueError): study.expression_cost({'sum':['unknown']},{})

    def test_posthoc_gate_relaxation_rejected(self):
        row={'predicted_ns':10,'observed_ns':10,'gates':dict(within_process_cv=True,between_process_cv=True,heldout_relative_error=True)}
        with self.assertRaises(ValueError):
            study.check_statistical_gates(row,[[1,100]]*3,[[10,10]]*3)

    def test_evidence_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'raw.json';p.write_text('original')
            protocol={'script_sha256':study.sha(study.__file__),'inputs':[{'path':'raw.json','sha256':study.sha(p)}]}
            study.verify_inventory(root,protocol)
            p.write_text('changed')
            with self.assertRaises(ValueError):study.verify_inventory(root,protocol)

    def test_study_source_mutation_rejected(self):
        with self.assertRaises(ValueError):study.verify_inventory(Path('.'),{'script_sha256':'0'*64,'inputs':[]})


if __name__=='__main__': unittest.main()
