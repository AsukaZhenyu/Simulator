"""Small software checks; never load a model or execute hardware."""
from __future__ import annotations
import copy
import unittest
from unittest.mock import patch

from profiling.real_moe_validation.dataset import assemble, chatml, render_official
from profiling.real_moe_validation.run import make_request, request, verify


def native_request(mode='replay'):
    return dict(mode=mode,source_sha256='a'*64,model_sha256='b'*64,
                threadpool_policy='explicit_persistent_owned',
                model_contract=dict(architecture='olmoe',layers=16,experts=64,top_k=8),
                expected_process_mask=4294967295,requests=[dict(id='q',prompt_token_ids=[1,2],output_token_ids=[3,4])])


def native_result(mode='replay'):
    req=native_request(mode)
    row=dict(id='q',prompt_token_ids=[1,2],output_token_ids=[3,4],begin_tick=100,prefill_end_tick=110,
             end_tick=120,whole_request_wall_ns=2000,output_count=2,decode_forward_count=1,
             core_decode_forward_ns=[1000],all_greedy_tokens_match=True,
             timing_usable_as_bare_reference=mode!='trace',routes=[],
             forward_inventory=[dict(index=0,phase='prefill',position=0,tokens=2),dict(index=1,phase='decode',position=2,tokens=1)])
    return dict(schema='real-moe-native-1',request=req,process_id=123,source_sha256='a'*64,
                model_contract_verified=True,observed_model_contract=req['model_contract'],model_layers=16,
                tokenization_add_special=False,
                threadpool_policy=req['threadpool_policy'],
                process_affinity=dict(observed_mask=4294967295,enforced_before_model_load=True),
                automatic_fit_used=False,all_prompt_tokens_marked_output=False,extra_final_decode=False,
                model_sha256_verified_by_parent='b'*64,clock=dict(frequency_hz=10000000),results=[row])


class RealMoeValidationTests(unittest.TestCase):
    def test_extra_final_decode_rejected(self):
        n=native_result();n['results'][0]['decode_forward_count']=2
        n['results'][0]['core_decode_forward_ns']=[1000,1000]
        with self.assertRaisesRegex(ValueError,'extra or missing final'):verify(n,n['request'],123)

    def test_changed_teacher_forced_tokens_rejected(self):
        n=native_result();n['results'][0]['output_token_ids']=[3,7]
        with self.assertRaisesRegex(ValueError,'teacher-forced'):verify(n,n['request'],123)

    def test_trace_time_cannot_be_promoted(self):
        n=native_result('trace');n['results'][0]['timing_usable_as_bare_reference']=True
        with self.assertRaisesRegex(ValueError,'trace promoted'):verify(n,n['request'],123)

    def test_normal_pruning_cannot_be_replaced_by_all_outputs(self):
        n=native_result();n['all_prompt_tokens_marked_output']=True
        with self.assertRaisesRegex(ValueError,'execution-boundary'):verify(n,n['request'],123)

    def test_route_positions_must_be_inside_forward(self):
        n=native_result('trace');n['results'][0]['routes']=[dict(forward_index=1,phase='decode',absolute_positions=[400],expert_ids=[[3,7]])]
        with self.assertRaisesRegex(ValueError,'route position outside'):verify(n,n['request'],123)

    def test_duplicate_experts_rejected(self):
        n=native_result('trace');n['results'][0]['routes']=[dict(forward_index=1,phase='decode',absolute_positions=[2],expert_ids=[[3,3]])]
        with self.assertRaisesRegex(ValueError,'expert ID/uniqueness'):verify(n,n['request'],123)

    def test_valid_replay_does_not_certify_simulator(self):
        n=native_result();proof=verify(n,n['request'],123)
        self.assertTrue(proof['passed']);self.assertFalse(proof['simulator_accuracy_certified'])

    def test_request_has_explicit_memory_and_threads(self):
        manifest=dict(native_source_sha256='a'*64,model=dict(path='x.gguf',sha256='b'*64,bytes=100),model_contract=native_request()['model_contract'])
        r=request(manifest,mode='generate',placement='cpu',requests=[],threads=16)
        self.assertEqual(r['threads'],16);self.assertGreater(r['minimum_free_host_bytes'],100)
        self.assertEqual(r['expected_process_mask'],4294967295)
        with self.assertRaisesRegex(ValueError,'placement'):request(manifest,mode='generate',placement='unknown',requests=[])

    def test_public_prompt_assembly_has_disjoint_target_examples(self):
        sources=dict(math=dict(records=[dict(question='q'+str(i),answer='a'+str(i)) for i in range(54)]),
                     code=dict(records=[dict(task_id=i,text='t'+str(i),code='pass',test_list=['assert True']) for i in range(54)]),
                     dialogue=dict(records=[dict(row_idx=0,messages=[dict(role='user',content='hello'),dict(role='assistant',content='answer'),dict(role='user',content='again')])]))
        rows=assemble(sources)
        for row in rows:
            if row['domain']!='dialogue':self.assertNotIn(row['source_target_id'],row['context_source_ids'])
        self.assertTrue(all(row['prompt'].endswith('<|im_start|>assistant\n') for row in rows))
        self.assertTrue(any(row['construction']=='original conversation prefix' for row in rows))

    def test_generic_contract_rejects_wrong_model(self):
        n=native_result();n['observed_model_contract']=dict(architecture='qwen35moe',layers=40,experts=256,top_k=8)
        with self.assertRaisesRegex(ValueError,'architecture/shape'):verify(n,n['request'],123)

    def test_token_suite_from_another_model_is_rejected(self):
        manifest=dict(model=dict(sha256='b'*64),official_tokenizer_config_sha256='c'*64)
        suite=dict(model_sha256='d'*64,requests=[])
        with patch('profiling.real_moe_validation.run.read',side_effect=[manifest,suite]):
            with self.assertRaisesRegex(ValueError,'different model'):
                make_request('unused','unused-suite','unused-output',mode='generate',placement='cpu')

    def test_implicit_special_token_insertion_is_rejected(self):
        n=native_result();n['tokenization_add_special']=True
        with self.assertRaisesRegex(ValueError,'implicit special tokens'):verify(n,n['request'],123)

    def test_trace_requires_all_declared_layers_without_40_layer_hardcode(self):
        n=native_result('trace')
        n['results'][0]['routes']=[dict(forward_index=1,phase='decode',absolute_positions=[2],expert_ids=[list(range(8))],layer=i) for i in range(16)]
        self.assertTrue(verify(n,n['request'],123)['passed'])
        n['results'][0]['routes'][0]['expert_ids'][0][-1]=64
        with self.assertRaisesRegex(ValueError,'expert ID'):verify(n,n['request'],123)

    def test_official_prompt_rendering_preserves_roles_and_end_marker(self):
        cfg=dict(bos_token='<B>',eos_token='<E>',chat_template="{{ bos_token }}{% for message in messages %}{{ '<'+message['role']+'>\\n'+message['content'] }}{% if message['role']=='assistant' %}{{ eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}")
        messages=[dict(role='user',content='Q'),dict(role='assistant',content='A'),dict(role='user',content='Q2')]
        self.assertEqual(render_official(messages,cfg),'<B><user>\nQ<assistant>\nA<E><user>\nQ2<assistant>')


if __name__=='__main__':unittest.main()
