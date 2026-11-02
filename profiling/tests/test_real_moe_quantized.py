"""Pure software quantized contracts; no GGUF load or native graph execution."""
from __future__ import annotations
import copy
import unittest

from profiling.real_moe_validation.inventory import audit
from profiling.real_moe_validation.quantized import request,validate_native


def directory():
    rows=[]
    for layer in range(16):
        for role,shape,dtype in (
            ('ffn_gate_exps',[2048,1024,64],'Q4_K'),('ffn_up_exps',[2048,1024,64],'Q4_K'),
            ('ffn_down_exps',[1024,2048,64],'Q6_K' if layer%2==0 else 'Q4_K'),
            ('ffn_gate_inp',[2048,64],'F32'),('attn_q',[2048,2048],'Q4_K'),
            ('attn_k',[2048,2048],'Q4_K'),('attn_v',[2048,2048],'Q6_K'),('attn_output',[2048,2048],'Q4_K')):
            rows.append(dict(name=f'blk.{layer}.{role}.weight',shape=shape,dtype=dtype,bytes=100,data_offset=4096))
    return dict(model=dict(path='model.gguf',bytes=4213513024,sha256='a'*64),model_contract=dict(architecture='olmoe',layers=16,experts=64,top_k=8),tensors=rows)


def req(**kwargs):
    return request(directory(),operation='olmoe_cpu_ffn',layer=0,tokens=1,backend='CPU',source_sha256='b'*64,runtime_binding='c'*64,**kwargs)


def native(r):
    ops=['MUL_MAT','SOFT_MAX','ARGSORT','GET_ROWS']+['MUL_MAT_ID']*3+['GLU']+['ADD']*7
    return dict(schema='real-moe-quantized-subgraph-native-1',request=r,source_sha256=r['source_sha256'],process_id=123,
        observed_process_mask=r['expected_process_mask'],packed_weights_read_from_model=True,weights_dequantized_or_substituted=False,
        output_finite=True,whole_model_accuracy_certified=False,samples_ns=[100]*r['samples'],
        actual_weight_types={k:v['dtype'].lower() for k,v in r['weights'].items()},graph_nodes=[dict(op=o) for o in ops],
        cache_context=r['cache_context'],foreign_working_set_bytes=r['foreign_working_set_bytes'],cache_conditioning_in_target_interval=False,
        cpu_threadpool_policy=r['cpu_threadpool_policy'],
        observed_routes_flat=list(range(8)))


class QuantizedMoeContracts(unittest.TestCase):
    def test_actual_layer_specific_quantization_is_preserved(self):
        d=directory();r0=request(d,operation='olmoe_cpu_ffn',layer=0,tokens=1,backend='CPU',source_sha256='b'*64,runtime_binding='c'*64)
        r1=request(d,operation='olmoe_cpu_ffn',layer=1,tokens=1,backend='CPU',source_sha256='b'*64,runtime_binding='c'*64)
        self.assertEqual(r0['weights']['down']['dtype'],'Q6_K');self.assertEqual(r1['weights']['down']['dtype'],'Q4_K')

    def test_f32_expert_weight_is_rejected(self):
        d=directory();next(t for t in d['tensors'] if t['name']=='blk.0.ffn_gate_exps.weight')['dtype']='F32'
        with self.assertRaisesRegex(ValueError,'mixed-quantized'):
            request(d,operation='olmoe_cpu_ffn',layer=0,tokens=1,backend='CPU',source_sha256='b'*64,runtime_binding='c'*64)

    def test_hybrid_cannot_reuse_homogeneous_cpu_ffn(self):
        with self.assertRaisesRegex(ValueError,'heterogeneous FFN'):
            request(directory(),operation='olmoe_cpu_ffn',layer=0,tokens=1,backend='CUDA0',source_sha256='b'*64,runtime_binding='c'*64)

    def test_foreign_cache_context_is_separate_and_budgeted(self):
        warm=req();cold=req(cache_context='foreign_cpu_touch_512MiB')
        self.assertEqual(cold['minimum_free_host_bytes']-warm['minimum_free_host_bytes'],512*1024**2)
        self.assertNotEqual(warm['cache_context'],cold['cache_context'])

    def test_changed_packed_dtype_is_rejected(self):
        r=req();n=native(r);n['actual_weight_types']['down']='f32'
        with self.assertRaisesRegex(ValueError,'packed dtypes'):validate_native(n,r,123)

    def test_legacy_grouped_matmul_topology_is_rejected(self):
        r=req();n=native(r)
        for node in n['graph_nodes']:
            if node['op']=='MUL_MAT_ID':node['op']='MUL_MAT'
        with self.assertRaisesRegex(ValueError,'vectorized OLMoE'):validate_native(n,r,123)

    def test_changed_route_ids_are_rejected(self):
        r=req();n=native(r);n['observed_routes_flat']=[0]*8
        with self.assertRaisesRegex(ValueError,'expert selection'):validate_native(n,r,123)

    def test_finite_subgraph_does_not_certify_full_model(self):
        r=req();proof=validate_native(native(r),r,123)
        self.assertTrue(proof['passed']);self.assertFalse(proof['whole_model_accuracy_certified'])
        self.assertFalse(proof['actual_runtime_partition_equivalence'])

    def test_static_directory_does_not_promote_existing_f32_profiles(self):
        coverage=audit(directory())
        self.assertEqual(coverage['qualified_existing_families'],0)
        self.assertFalse(coverage['current_end_to_end_prediction_supported'])
        self.assertEqual(len(coverage['layers']),16)


if __name__=='__main__':unittest.main()
