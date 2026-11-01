"""Closed-form checks on independent reference price and causality boundaries."""
import unittest
from .campaign import scenario,declared_dag,groups_for,placement,reference_nodes,schedule,PRIORITIES

class IndependentReferenceChecks(unittest.TestCase):
    def make(self,name,pattern,attention=False,decode=0,requests=1):
        s=scenario(name,pattern,attention,decode,requests,'compute')
        return s,declared_dag(s)

    def test_dense_host_chain_closed_form(self):
        s,d=self.make('dense','A01');p={g:'cpu' for g in groups_for(s,d)}
        nodes=reference_nodes(s,d,placement(s,d,p),'demand')
        # 36-byte input preparation plus 30k projection, 50k Attention,
        # 120k host SwiGLU; no transfers on an all-host chain.
        self.assertEqual(schedule(nodes,PRIORITIES[0])['duration_ns'],200036)
        self.assertFalse(any(n['kind']=='copy' for n in nodes.values()))

    def test_gpu_ffn_requires_host_return(self):
        s,d=self.make('dense','A01');p={g:'cpu' for g in groups_for(s,d)}
        p['r0:prefill:dense_ffn']='gpu'
        nodes=reference_nodes(s,d,placement(s,d,p),'demand')
        # 1536 bytes of gate/up/down plus32-byte postnorm weights,
        # a32-byte hidden input and32-byte returned hidden output.
        self.assertEqual(sum(n['bytes'] for n in nodes.values()),1632)
        self.assertEqual(schedule(nodes,PRIORITIES[0])['duration_ns'],141668)

    def test_expert_copy_cannot_precede_router(self):
        s,d=self.make('moe','A06');p={g:'gpu' for g in groups_for(s,d)}
        nodes=reference_nodes(s,d,placement(s,d,p),'spare')
        copies=[n for n in nodes.values() if n['kind']=='copy' and ':E' in n['tensor']]
        self.assertTrue(copies)
        for n in copies:
            self.assertTrue(any(d['operations'].get(x,{}).get('semantic_op')=='router' for x in n['deps']))
        self.assertEqual({n['machine'] for n in nodes.values() if n['kind']=='copy'},{'pcie'})

    def test_pd_gate_reads_prefill_kv(self):
        s,d=self.make('pd','A10',decode=1)
        gates=[o for o in d['operations'].values() if o['semantic_op']=='state_update' and o['inputs']]
        self.assertEqual(len(gates),1)
        self.assertIn('kv:layer0:request0:0',gates[0]['inputs'])

    def test_shared_weight_demand_can_use_any_ready_consumer(self):
        nodes={
            'producer0':dict(kind='compute',machine='cpu',duration=100,release=0,deps=[]),
            'producer1':dict(kind='compute',machine='gpu',duration=20,release=0,deps=[]),
            'shared_copy':dict(kind='copy',machine='pcie',duration=10,release=0,deps=[],
                dependency_alternatives=[['producer0'],['producer1']]),
        }
        trace=schedule(nodes,PRIORITIES[0])['events']
        copied=next(r for r in trace if r['id']=='shared_copy')
        self.assertEqual(copied['start_ns'],20)
        self.assertEqual(copied['end_ns'],30)

if __name__=='__main__':unittest.main()
