"""Pure software postcollection route coverage checks; no model or timing."""
from __future__ import annotations
import copy
import unittest
from profiling.real_moe_validation.postcollection import CONTRACT,audit_native,expected_forwards,effective_parameters

STDERR='llama_context: n_batch = 256\nllama_context: n_ubatch = 256\nllama_context: n_seq_max = 1\n'


def fixture(prompt=1029,outputs=8):
    f=expected_forwards(prompt,outputs);tokens=list(range(prompt));out=[17]*outputs;routes=[]
    for forward in f:
        for layer in range(16):
            positions=list(range(forward['position'],forward['position']+forward['tokens'])) if layer<15 else ([forward['position']+forward['tokens']-1] if forward['output_rows'] else [])
            if not positions:continue
            routes.append(dict(forward_index=forward['index'],layer=layer,phase=forward['phase'],absolute_positions=positions,expert_ids=[list(range(8))]*len(positions)))
    row=dict(id='request',prompt_token_ids=tokens,output_token_ids=out,teacher_forced=True,timing_usable_as_bare_reference=False,
        all_greedy_tokens_match=True,forward_inventory=f,forward_count=len(f),decode_forward_count=outputs-1,routes=routes)
    return dict(schema='real-moe-native-1',mode='trace',observed_model_contract=CONTRACT,threadpool_policy='explicit_persistent_owned',
        model_weights_loaded=True,trace_callback_enabled=True,trace_callback_asks_all_nodes=False,source_sha256='a'*64,model_sha256_verified_by_parent='b'*64,
        request=dict(mode='trace',batch_size=256,placement='cpu',model_contract=CONTRACT,requests=[dict(id='request',prompt_token_ids=tokens,output_token_ids=out)]),results=[row])


class PostcollectionMoeAudit(unittest.TestCase):
    def test_long_prompt_has_four_full_chunks_and_a_tail(self):
        report=audit_native(fixture(),STDERR)
        self.assertEqual(report['routing_shape_status'],'passed')
        self.assertEqual([f['tokens'] for f in report['cases'][0]['forwards'] if f['phase']=='prefill'],[256,256,256,256,5])
        self.assertEqual(report['required_nonempty_ffn_q_invocations'],{'1':113,'5':15,'256':60})
        self.assertEqual(report['pruned_final_layer_ffn_invocations'],4)
        self.assertEqual(report['q_shapes_without_prepared_profile'],[5])
        self.assertEqual(report['exact_timing_coverage'],'unknown')

    def test_missing_prefill_layer_is_not_silently_accepted(self):
        trace=fixture();trace['results'][0]['routes']=[r for r in trace['results'][0]['routes'] if not (r['forward_index']==0 and r['layer']==3)]
        report=audit_native(trace,STDERR)
        self.assertEqual(report['routing_shape_status'],'failed');self.assertTrue(any('missing forward 0/layer3' in p for p in report['problems']))

    def test_pruned_last_layer_rows_cannot_be_invented(self):
        trace=fixture();route=copy.deepcopy(trace['results'][0]['routes'][0]);route['layer']=15;trace['results'][0]['routes'].append(route)
        report=audit_native(trace,STDERR)
        self.assertEqual(report['routing_shape_status'],'failed');self.assertTrue(any('fabricated pruned' in p for p in report['problems']))

    def test_five_token_tail_is_not_a_one_token_task(self):
        trace=fixture();route=next(r for r in trace['results'][0]['routes'] if r['forward_index']==4 and r['layer']==0)
        route['absolute_positions']=[1028];route['expert_ids']=[list(range(8))]
        self.assertEqual(audit_native(trace,STDERR)['routing_shape_status'],'failed')

    def test_multiple_real_callbacks_are_not_merged(self):
        trace=fixture();trace['results'][0]['routes'].append(trace['results'][0]['routes'][0])
        report=audit_native(trace,STDERR)
        self.assertEqual(report['routing_shape_status'],'failed');self.assertTrue(any('not silently merged' in p for p in report['problems']))

    def test_wrong_ubatch_fails_and_missing_ubatch_is_unknown(self):
        self.assertEqual(effective_parameters(STDERR.replace('n_ubatch = 256','n_ubatch = 128'))['status'],'failed')
        missing=STDERR.replace('llama_context: n_ubatch = 256\n','')
        self.assertEqual(audit_native(fixture(),missing)['routing_shape_status'],'unknown')

    def test_decode_absolute_position_is_checked(self):
        trace=fixture();route=next(r for r in trace['results'][0]['routes'] if r['forward_index']==5 and r['layer']==0);route['absolute_positions']=[1030]
        self.assertEqual(audit_native(trace,STDERR)['routing_shape_status'],'failed')

    def test_normal_trace_pair_checks_actual_frozen_tokens(self):
        trace=fixture();bare=copy.deepcopy(trace);bare['mode']='generate';bare['request']['mode']='generate';bare['results'][0]['routes']=[];bare['results'][0]['timing_usable_as_bare_reference']=True
        self.assertEqual(audit_native(trace,STDERR,bare=bare)['routing_shape_status'],'passed')
        bare['results'][0]['output_token_ids'][0]=18
        self.assertEqual(audit_native(trace,STDERR,bare=bare)['routing_shape_status'],'failed')

    def test_eos_after_first_output_has_no_invented_decode(self):
        report=audit_native(fixture(prompt=137,outputs=1),STDERR)
        self.assertEqual(report['routing_shape_status'],'passed')
        self.assertEqual(len(report['cases'][0]['forwards']),1)
        self.assertEqual(report['required_nonempty_ffn_q_invocations'],{'1':1,'137':15})


if __name__=='__main__':unittest.main()
