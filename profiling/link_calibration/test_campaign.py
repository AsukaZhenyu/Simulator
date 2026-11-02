"""Causality, exact-state and preregistered sampling software tests only."""
import copy
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from .campaign import SIZES,FIELDS,balanced_orders,query_orders,state_key,query_extent,classify,coverage,validate_freeze,sha

def sample(gen=4,width=8,begin=10):
    values=dict(zip(FIELDS,(gen,4,width,8)))
    return dict(status='ok',values=values,queries=[dict(qpc_valid=True,qpc_begin=begin+i*2,qpc_end=begin+i*2+1) for i in range(4)])
def burst(gen=4,size=131072,after_gen=None):
    return dict(control='state_queries',bytes=size,burst_begin_tick=100,burst_end_tick=200,
                pcie_before=sample(gen),pcie_after=sample(gen if after_gen is None else after_gen,begin=210))

class ProtocolTests(unittest.TestCase):
    def test_size_sets_are_disjoint(self):
        self.assertFalse(set(SIZES['profile'])&set(SIZES['heldout']))
        self.assertEqual(SIZES['profile'],(131072,524288,2097152))
        self.assertEqual(SIZES['heldout'],(262144,1048576))
    def test_positions_balanced_for_every_process(self):
        for n in (2,3):
            for r in range(3):
                orders=balanced_orders(n,r)
                for slot in range(n):self.assertEqual(Counter(row[slot] for row in orders),Counter({i:12//n for i in range(n)}))
                q=query_orders(n,r);self.assertEqual(Counter(row[0] for row in q),Counter({'bare':6*n,'state_queries':6*n}))
                for size in range(n):
                    per_size=[q[t*n+slot][0] for t,row in enumerate(orders) for slot,index in enumerate(row) if index==size]
                    self.assertEqual(Counter(per_size),Counter({'bare':6,'state_queries':6}))
    def test_invalid_sampling_plan_rejected(self):
        for args in ((1,0),(4,0),(3,3),(3,-1)):
            with self.assertRaises(ValueError):balanced_orders(*args)
    def test_unknown_does_not_use_max_or_theory(self):
        for bad in (None,{},dict(status='partial',values=sample()['values']),dict(status='ok',values={'pcie_max_gen':4,'pcie_max_width':8})):
            self.assertIsNone(state_key(bad))
    def test_pre_state_is_only_key(self):
        row=burst(after_gen=3);c=classify(row)
        self.assertEqual(c['prior_key'],(4,4,8,8));self.assertFalse(c['qualified'])
    def test_bare_control_has_unknown_state(self):
        row=burst();row['control']='bare'
        self.assertIsNone(classify(row)['prior_key']);self.assertFalse(classify(row)['qualified'])
    def test_future_query_rejected(self):
        row=burst();row['pcie_before']=sample(begin=99)
        with self.assertRaises(ValueError):classify(row)
    def test_post_overlap_rejected(self):
        row=burst();row['pcie_after']=sample(begin=190)
        with self.assertRaises(ValueError):classify(row)
    def test_invalid_query_clock_is_unknown(self):
        row=burst();row['pcie_before']['queries'][0]['qpc_valid']=False
        self.assertIsNone(query_extent(row['pcie_before']));self.assertFalse(classify(row)['qualified'])
    def test_exact_width_not_pooled(self):
        a=burst();b=burst();b['pcie_before']=sample(width=4);b['pcie_after']=sample(width=4,begin=210)
        self.assertNotEqual(classify(a)['prior_key'],classify(b)['prior_key'])
    def test_same_context_many_trials_not_three_contexts(self):
        rows=[burst(g,size) for g in (3,4) for size in SIZES['profile']]*20
        report=coverage([dict(context_id='one',bursts=rows)])
        self.assertFalse(report['coverage_passed']);self.assertTrue(report['missing_cells_stop_without_extra_sampling'])
        self.assertTrue(all(row['independent_contexts']==1 for row in report['cells']))
    def test_both_states_all_sizes_three_contexts(self):
        rows=[burst(g,size) for g in (3,4) for size in SIZES['profile']]
        report=coverage([dict(context_id=str(i),bursts=copy.deepcopy(rows)) for i in range(3)])
        self.assertTrue(report['coverage_passed']);self.assertEqual(len(report['completed_exact_state_keys']),2)
        self.assertFalse(report['post_state_is_predictor_input'])
    def test_missing_size_cell_stops(self):
        rows=[burst(g,size) for g in (3,4) for size in SIZES['profile']]
        records=[dict(context_id=str(i),bursts=copy.deepcopy(rows)) for i in range(3)]
        records[2]['bursts'].pop()
        self.assertFalse(coverage(records)['coverage_passed'])
    def test_native_scope_is_exact_alias_Executor(self):
        source=(Path(__file__).parent/'native/link_probe.cpp').read_text(encoding='utf-8')
        self.assertIn('ggml_view_1d(ctx,source,bytes/sizeof(float),0)',source)
        self.assertIn('e.action(a.action)',source)
        self.assertNotIn('#include "cupti',source)
        self.assertNotIn('cudaEventCreate',source)
        hot=source.split('b.begin=steady::tick();',1)[1].split('b.end=steady::tick();',1)[0]
        self.assertNotIn('nvml',hot);self.assertNotIn('json',hot);self.assertNotIn('payload',hot)
    def test_empty_freeze_cannot_launch_unseen_sizes(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[2]/'build') as tmp:
            out=Path(tmp);(out/'manifest.json').write_text('{}',encoding='utf-8')
            with self.assertRaises(ValueError):validate_freeze(out,dict(schema='link-copy-cost-freeze-1',manifest={'sha256':sha(out/'manifest.json')},files={}))
    def test_freeze_requires_complete_profile_support(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[2]/'build') as tmp:
            out=Path(tmp);(out/'manifest.json').write_text('{}',encoding='utf-8');(out/'model.json').write_text('{}',encoding='utf-8')
            (out/'profile-summary.json').write_text(json.dumps(dict(declared_contexts=3,completed_contexts=3,exact_state_coverage={'coverage_passed':False})),encoding='utf-8')
            barrier=dict(schema='link-copy-cost-freeze-1',manifest={'sha256':sha(out/'manifest.json')},files={'model.json':sha(out/'model.json')},predictor_parameters_frozen=True,model_scope='conditional_known_prior_state',profile_summary={'sha256':sha(out/'profile-summary.json')})
            with self.assertRaises(ValueError):validate_freeze(out,barrier)
    def test_valid_freeze_binds_support_and_nonempty_model(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[2]/'build') as tmp:
            out=Path(tmp);(out/'manifest.json').write_text('{}',encoding='utf-8');(out/'model.json').write_text('{}',encoding='utf-8')
            (out/'profile-summary.json').write_text(json.dumps(dict(declared_contexts=3,completed_contexts=3,exact_state_coverage={'coverage_passed':True})),encoding='utf-8')
            barrier=dict(schema='link-copy-cost-freeze-1',manifest={'sha256':sha(out/'manifest.json')},files={'model.json':sha(out/'model.json')},predictor_parameters_frozen=True,model_scope='conditional_known_prior_state',profile_summary={'sha256':sha(out/'profile-summary.json')})
            validate_freeze(out,barrier)
            (out/'model.json').write_text('{"changed":true}',encoding='utf-8')
            with self.assertRaises(ValueError):validate_freeze(out,barrier)

if __name__=='__main__':unittest.main()
