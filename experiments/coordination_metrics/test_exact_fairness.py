"""Tiny exhaustive witnesses for the experiment's exact-oracle constraints."""
import itertools
import unittest
from experiments.coordination_metrics.exact_fairness import verify,lower_bound,strengthened_lower_bound,solve_exact,faithful_nodes
from experiments.ir_value import campaign as base
from experiments.dac_validation_e1_oracle import declared_dag

def node(identity,machine,duration,deps=(),alternatives=None):
    result=dict(id=identity,machine=machine,duration=duration,release=0,deps=list(deps))
    if alternatives is not None:result['dependency_alternatives']=alternatives
    return result

class ExactFairnessTests(unittest.TestCase):
    def test_OR_matches_integer_exhaustion_not_AND_or_ignored(self):
        nodes={'a':node('a','cpu',3),'b':node('b','gpu',5),
            'copy':node('copy','pcie',3,alternatives=[['a'],['b']])}
        legal=[]
        for starts in itertools.product(range(9),repeat=3):
            trace=[dict(id=k,machine=n['machine'],start_ns=start,end_ns=start+n['duration']) for (k,n),start in zip(nodes.items(),starts)]
            try:legal.append((verify(nodes,trace),trace))
            except AssertionError:pass
        optimum,trace=min(legal,key=lambda r:r[0]);self.assertEqual(optimum,6)
        result=solve_exact(nodes,dict(duration_ns=optimum,events=trace),2)
        self.assertEqual(result['status'],0);self.assertEqual(result['verified_makespan_ns'],6)
        self.assertEqual(result['OR_selector_variables'],2)
    def test_ignored_OR_witness_is_rejected(self):
        nodes={'a':node('a','cpu',3),'b':node('b','gpu',5),'copy':node('copy','pcie',3,alternatives=[['a'],['b']])}
        trace=[dict(id=k,machine=n['machine'],start_ns=0,end_ns=n['duration']) for k,n in nodes.items()]
        with self.assertRaisesRegex(AssertionError,'OR readiness'):verify(nodes,trace)
    def test_resource_collision_is_rejected(self):
        nodes={'a':node('a','pcie',3),'b':node('b','pcie',2)}
        trace=[dict(id=k,machine=n['machine'],start_ns=0,end_ns=n['duration']) for k,n in nodes.items()]
        with self.assertRaisesRegex(AssertionError,'resource collision'):verify(nodes,trace)
    def test_energetic_join_keeps_all_copy_work(self):
        nodes={'gate':node('gate','gpu',1),'a':node('a','pcie',2,['gate']),
            'b':node('b','pcie',3,['gate']),'finish':node('finish','cpu',2,['a','b'])}
        self.assertEqual(strengthened_lower_bound(nodes)['lower_bound_ns'],8)
        trace=[dict(id='gate',machine='gpu',start_ns=0,end_ns=1),dict(id='a',machine='pcie',start_ns=1,end_ns=3),
            dict(id='b',machine='pcie',start_ns=3,end_ns=6),dict(id='finish',machine='cpu',start_ns=6,end_ns=8)]
        self.assertEqual(verify(nodes,trace),8)
    def test_energetic_bound_does_not_force_both_OR_branches(self):
        nodes={'a':node('a','cpu',3),'b':node('b','gpu',5),'copy':node('copy','pcie',3,alternatives=[['a'],['b']])}
        self.assertEqual(strengthened_lower_bound(nodes)['lower_bound_ns'],6)
    def test_OR_relaxation_handles_shared_copy_later_consumer(self):
        nodes={'first':node('first','cpu',1),'copy':node('copy','pcie',1,alternatives=[['first'],['later']]),
            'later':node('later','cpu',1,deps=['copy'])}
        self.assertEqual(lower_bound(nodes)['lower_bound_ns'],3)
    def test_spare_only_relaxes_static_weight_COPY_gate(self):
        s=base.scenario('dense','A01',False,0,1,'compute');d=declared_dag(s)
        choices={k:'gpu' for k in base.groups_for(s,d)};devices=base.placement(s,d,choices)
        demand=faithful_nodes(s,d,devices,'demand');spare=faithful_nodes(s,d,devices,'spare')
        for key,n in spare.items():
            if n['kind']!='copy':continue
            if d['tensors'][n['tensor']]['role']!='weight':self.assertEqual(n['deps'],demand[key]['deps'])
        self.assertLess(len(spare['copy:w:layer0:W_q:cpu>gpu']['deps']),len(demand['copy:w:layer0:W_q:cpu>gpu']['deps']))

if __name__=='__main__':unittest.main()
