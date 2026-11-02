"""Paired price weighting/scope tests; no measured targets or device calls."""
from fractions import Fraction
import unittest

from profiling.dynamic_validation import continuous_price_ablation as ablation
from profiling.dynamic_validation import bare_macro_fit as shared


class PairedEstimatorTests(unittest.TestCase):
    def records(self):
        records=[]
        for state_index,state in enumerate(ablation.primary.STATES):
            for index in range(3):
                number=state_index*3+index
                records.append(dict(state=state,process_identity=(100+number,1000+number),
                    native=dict(sha256=f'{number:064x}'),prepared_single_ns=400+100*index,
                    continuous_mean_ns=Fraction(100+100*index),continuous_invocations=(1,2,100)[index]))
        return records

    def test_both_estimators_use_same_three_contexts_with_equal_weight(self):
        cells=ablation.paired_cells(self.records())
        self.assertTrue(all(c['mean_ns']==500 for c in cells['prepared_single']))
        self.assertTrue(all(c['mean_ns']==200 for c in cells['continuous']))
        self.assertEqual([c['observations'] for c in cells['prepared_single']],
                         [c['observations'] for c in cells['continuous']])
        self.assertNotEqual(cells['continuous'][0]['mean_ns'],Fraction(100+2*200+100*300,103))

    def test_missing_state_context_or_reused_native_identity_cannot_be_filled(self):
        with self.assertRaises(ValueError):ablation.paired_cells(self.records()[:-1])
        records=self.records();records[3]['process_identity']=records[0]['process_identity']
        with self.assertRaises(ValueError):ablation.paired_cells(records)

    def test_unknown_or_nonpositive_response_is_not_a_zero_baseline(self):
        records=self.records();records[0]['prepared_single_ns']=0
        with self.assertRaises(ValueError):ablation.paired_cells(records)


class HonestScopeTests(unittest.TestCase):
    def inputs(self):
        actors={d:dict(device=d,component=dict(kind='fixture'),main_thread_mask=4 if d=='cpu' else 65536) for d in ('cpu','gpu')}
        context=dict(actors=actors,foreground_lifecycle='continuous_invocation')
        semantic=dict(kind='complete_Attention_fixture')
        scope=dict(context=context,context_sha256=shared.digest(context),semantic_coordinate=semantic,
                   semantic_sha256=shared.digest(semantic))
        request=dict(replays=4,block_replays=2,cpu_actor=dict(actors['cpu'],intervals_ms=[]),gpu_actor=dict(actors['gpu'],intervals_ms=[]))
        return scope,request

    def test_prepared_single_scope_is_explicit_and_does_not_mutate_primary_scope(self):
        scope,request=self.inputs();original_hash=scope['context_sha256']
        continuous=ablation.variant_scope(scope,'continuous');single=ablation.variant_scope(scope,'prepared_single')
        self.assertEqual(continuous,scope)
        self.assertEqual(scope['context']['foreground_lifecycle'],'continuous_invocation')
        self.assertEqual(scope['context_sha256'],original_hash)
        self.assertEqual(single['context']['foreground_lifecycle'],'prepared_single_invocation')
        self.assertNotEqual(single['context_sha256'],original_hash)
        self.assertEqual(shared.digest(single['context']),single['context_sha256'])

    def test_same_g_same_request_different_paired_prices_have_no_target_input(self):
        scope,request=self.inputs()
        single=[dict(state=s,mean_ns=2000) for s in ablation.primary.STATES]
        continuous=[dict(state=s,mean_ns=1000) for s in ablation.primary.STATES]
        a=shared.predict(single,ablation.variant_scope(scope,'prepared_single'),request,ablation.PROVIDER,('software fixture',))
        b=shared.predict(continuous,ablation.variant_scope(scope,'continuous'),request,ablation.PROVIDER,('software fixture',))
        self.assertEqual(a['total_ns'],8000)
        self.assertEqual(b['total_ns'],4000)
        self.assertTrue(a['own_predicted_completion_clock'] and b['own_predicted_completion_clock'])
        self.assertFalse(a['observed_target_boundaries_used'] or b['observed_target_boundaries_used'])


if __name__=='__main__':unittest.main()
