"""Independent representation equivalence and invalid-contract regression."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
import numpy as np
from profiling.moe_policy_validation import campaign as environment
from profiling.highlevel_accuracy.cases import build_case
from profiling.highlevel_accuracy.bridge import export_plan,reference_ports,validate_program,BridgeError,sha

def reseal(p):
    p['identity']={k:sha(p[k]) for k in p['identity']}
    p.pop('program_sha256',None);p['program_sha256']=sha(p);return p

class PhysicalViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(dir=environment.ROOT/'build')
        cls.programs={}
        for contract in ('materialized_partition_replaces_parent','shared_parent_slice_views'):
            plan=build_case(Path(cls.tmp.name)/contract,hidden=8,intermediate=16,heads=2,kv_heads=1,head_dim=4,
                tokens=2,history=0,threads=1,devices=('cpu',)*3,ffn_cut=12,ffn_devices=('cpu','gpu'),
                norm_device='cpu',merge_device='cpu',ffn_storage_contract=contract)
            cls.programs[contract]=export_plan(plan.compile(),seed=32130)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_same_full_parent_values_produce_same_request_outputs(self):
        a,b=(reference_ports(self.programs[c]) for c in self.programs)
        compared=0
        for tid in set(a)&set(b):
            np.testing.assert_allclose(a[tid],b[tid],atol=1e-6,rtol=1e-6);compared+=1
        self.assertGreater(compared,10)
    def test_shared_allocations_keep_full_bytes_and_cpu_gpu_views(self):
        p=self.programs['shared_parent_slice_views'];boundary=p['graph']['boundary']['ffn_execution']
        self.assertEqual(len(boundary['weight_views']),6)
        self.assertTrue(all(v['allocation'] is False and v['storage_bytes']==0 for v in boundary['weight_views']))
        weights={t['id']:t for t in p['graph']['workload']['tensors'] if t['role']=='weight'}
        parent_ids={v['source_tensor_id'] for v in boundary['weight_views']}
        self.assertEqual(sum(weights[x]['storage_bytes'] for x in parent_ids),1536)
        self.assertEqual(len([b for b in p['bindings'] if b['kind']=='ffn_residual']),1)
        self.assertTrue(all(b['shape']['batch']==1 for b in p['bindings']))
    def test_wrong_interval_rejected_after_resigning_graph_and_binding(self):
        p=deepcopy(self.programs['shared_parent_slice_views'])
        b=next(b for b in p['bindings'] if b['kind']=='ffn_down')
        b['op_params']=[12,17]
        op=next(o for o in p['graph']['workload']['operations'] if o['id']==b['operation_id'])
        op['op_params']=b['op_params']
        with self.assertRaisesRegex(BridgeError,'interval outside'):validate_program(reseal(p))
    def test_parent_without_view_contract_rejected(self):
        p=deepcopy(self.programs['shared_parent_slice_views'])
        p['graph']['boundary']['ffn_execution']['weight_views']=[]
        with self.assertRaisesRegex(BridgeError,'missing explicit physical view'):validate_program(reseal(p))

if __name__=='__main__':unittest.main()
