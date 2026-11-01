"""Contracts that prevent unlike blocks or missing prices becoming comparisons."""
from copy import deepcopy
import math
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.dac_followup_external import (PLACEMENTS,compile_schedule,
    import_block,independent_serial,make_scenario,validate_native_case)


def case_fixture():
    h,q,n=128,16,2
    shape_data=[
        ('Q_proj','projection',[[1,q,h],[h,h]],[1,q,h]),
        ('K_proj','projection',[[1,q,h],[h,h]],[1,q,h]),
        ('V_proj','projection',[[1,q,h],[h,h]],[1,q,h]),
        ('Q_mul_K','attention',[[1,n,q,64],[1,n,64,q]],[1,n,q,q]),
        ('A_softmax','attention',[[1,n,q,q]],[1,n,q,q]),
        ('A_mul_V','attention',[[1,n,q,q],[1,n,q,64]],[1,n,q,64]),
        ('H_matmul0','attention',[[1,q,h],[h,h]],[1,q,h]),
        ('layer_norm0','attention',[[1,q,h]],[1,q,h]),
        ('H_matmul1','dense_ffn',[[1,q,h],[h,4*h]],[1,q,4*h]),
        ('H_gelu','dense_ffn',[[1,q,4*h]],[1,q,4*h]),
        ('H_matmul2','dense_ffn',[[1,q,4*h],[4*h,h]],[1,q,h]),
        ('layer_norm1','dense_ffn',[[1,q,h]],[1,q,h]),
    ]
    return dict(id='explicit-test-fixture',hidden=h,tokens=q,heads=n,
                intermediate=4*h,dtype='fp16',operations=[
                    dict(id=name,macro=macro,input_shapes=inputs,output_shape=output,
                         logical_input_bytes=[math.prod(s)*2 for s in inputs],
                         logical_output_bytes=math.prod(output)*2,
                         inclusive_seconds=(index+1)*1e-6)
                    for index,(name,macro,inputs,output) in enumerate(shape_data)])


class ExternalContracts(unittest.TestCase):
    def test_missing_operator_cannot_become_one_ns_price(self):
        case=case_fixture()
        case['operations'].pop()
        with self.assertRaisesRegex(ValueError,'Missing'):
            make_scenario(case)

    def test_wrong_mathematics_or_byte_contract_is_rejected(self):
        case=case_fixture()
        case['operations'][9]['output_shape']=[1,16,128]
        with self.assertRaisesRegex(ValueError,'shape'):
            validate_native_case(case)
        case=case_fixture()
        case['operations'][0]['logical_input_bytes'][0]*=2
        with self.assertRaisesRegex(ValueError,'byte'):
            validate_native_case(case)
        case=case_fixture()
        case['operations'][0]['inclusive_seconds']=None
        with self.assertRaisesRegex(ValueError,'cost'):
            validate_native_case(case)

    def test_three_qkv_outputs_and_exact_native_parent_weight_storage(self):
        workload=import_block(case_fixture())
        self.assertEqual(workload.operations[0].outputs,('Q','K','V'))
        h,q=128,16
        self.assertEqual(sum(t.storage_bytes for t in workload.tensors if t.role=='weight'),12*h*h*2)
        self.assertEqual(workload.tensor_by_id['Y'].storage_bytes,h*q*2)
        self.assertEqual(workload.tensor_by_id['W1'].ne,(4*h,h,1,1))
        self.assertEqual(workload.tensor_by_id['W2'].ne,(h,4*h,1,1))

    def test_four_ir_programs_match_independent_demand_ledger(self):
        case=case_fixture()
        expected_counts={'all_cpu':0,'all_gpu':8,'cpu_attention':11,'cpu_ffn':6}
        h,q=case['hidden'],case['tokens']
        expected_bytes={'all_cpu':0,'all_gpu':24*h*h+4*h*q,
                        'cpu_attention':22*h*h+12*h*q,
                        'cpu_ffn':8*h*h+4*h*q}
        for name,placement in PLACEMENTS.items():
            with self.subTest(name=name):
                _,_,result,_=compile_schedule(make_scenario(case),placement)
                expected,copies=independent_serial(case,placement)
                self.assertEqual(result.evaluation.makespan_ns,expected)
                self.assertEqual(len(copies),expected_counts[name])
                self.assertEqual(sum(r['bytes'] for r in copies),expected_bytes[name])


if __name__=='__main__':
    unittest.main()
