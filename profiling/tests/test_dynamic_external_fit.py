"""Software tests for calibration support, identifiability and target barriers."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
import math

from llm_infer_model.performance.parameters import UnknownParameter
from llm_infer_model.performance.reusable_action_costs import digest
from profiling.dynamic_validation import external,external_fit_gate
from profiling.dynamic_validation.external_fit import identify_tau,price_map,no_heldout,TAUS_NS


class ExternalFitTests(TestCase):
    def losses(self,flat=False):
        losses={};counts={}
        for r in range(3):
            for process in range(6):
                key=f'profile-round{r}-process{process}'
                losses[key]={0:1. if flat else 10.,**{t:1. if flat else 1.+math.log(t/2000000)**2 for t in TAUS_NS}}
                counts[key]=dict(round=r,strata=100)
        return losses,counts

    def test_flat_recovery_objective_is_unsupported(self):
        result=identify_tau(*self.losses(flat=True))
        self.assertEqual(result['status'],'unsupported');self.assertIsNone(result.get('tau_ns'))
        self.assertTrue(any('plateau' in reason for reason in result['reasons']))

    def test_identified_shared_transition_is_interior_and_round_stable(self):
        result=identify_tau(*self.losses())
        self.assertEqual(result['status'],'identified');self.assertEqual(result['tau_ns'],2000000)
        self.assertFalse(result['independent_accuracy_claimed'])

    def test_missing_state_has_no_mean_or_adjacent_state_fallback(self):
        semantic=dict(kind='COPY',src='dram',dst='vram',tensor=dict(bytes=2097152));key=digest(semantic)
        entry=dict(semantic_key=key,semantic=semantic,states=[dict(cpu_offered=0,gpu_offered=0,mean_ns='100')])
        with self.assertRaisesRegex(UnknownParameter,'missing exact'):price_map([entry])

    def test_equal_process_cell_means_do_not_overweight_longer_exposure(self):
        semantic=dict(kind='COPY',src='dram',dst='vram');key=digest(semantic)
        records=[]
        for job,count,total in (('a',10,1000),('b',100,20000)):
            records.append(dict(job=dict(id=job),native=dict(path='receipt-'+job),cells=[dict(semantic_key=key,semantic=semantic,
                cpu_offered=0,gpu_offered=0,count=count,sum_ns=total,action_indices=[3,7])]))
        result=external_fit_gate.combine(records)[0]
        self.assertEqual(result['states'][0]['mean_ns'],'150')
        self.assertEqual(result['states'][0]['sample_count'],110)
        self.assertIsNone(result['states'][1]['mean_ns'])

    def test_fit_rejected_after_target_acquisition_marker(self):
        parent=(external.ROOT/'build').resolve();self.assertTrue(parent.is_relative_to(external.ROOT.resolve()))
        with TemporaryDirectory(dir=parent) as temporary:
            out=Path(temporary);(out/'heldout-started.json').write_text('{}',encoding='utf-8',newline='\n')
            with self.assertRaisesRegex(ValueError,'acquisition rejected'):no_heldout(out,dict(jobs=[]))
