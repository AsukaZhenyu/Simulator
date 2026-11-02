"""Pure transform tests; generated parser fixtures are never empirical reports."""
from __future__ import annotations
import unittest
from profiling.tests.test_real_moe_postcollection import fixture
from profiling.real_moe_validation.route_analysis import statistics,decode_fragments,m2_invocation_patch


class ObservedMoeAnalysis(unittest.TestCase):
    def test_prefill_denominator_preserves_actual_last_layer_pruning(self):
        report=statistics(fixture())
        first=next(r for r in report['frequencies'] if r['phase']=='prefill' and r['layer']==0)
        last=next(r for r in report['frequencies'] if r['phase']=='prefill' and r['layer']==15)
        self.assertEqual(first['observed_token_rows'],1029);self.assertEqual(last['observed_token_rows'],1)
        self.assertEqual(first['counts'][0],1029);self.assertEqual(last['counts'][0],1)
        self.assertEqual(first['activation_fraction'][0],1);self.assertEqual(first['selection_share'][0],1/8)
        self.assertFalse(report['prefill_pruning_imputed'])

    def test_decode_reuse_has_a_defined_adjacent_pair_denominator(self):
        native=fixture(prompt=137,outputs=3)
        for route in native['results'][0]['routes']:
            if route['phase']=='decode' and route['layer']==0 and route['absolute_positions']==[138]:
                route['expert_ids']=[[4,5,6,7,8,9,10,11]]
        report=statistics(native);layer0=next(r for r in report['decode_adjacent_reuse'] if r['layer']==0)
        self.assertEqual(layer0['adjacent_pairs'],1);self.assertEqual(layer0['reuse_fraction'],.5)
        self.assertEqual(report['decode_reuse_by_category'][0]['reuse_fraction'],124/128)

    def test_a_single_decode_forward_has_unknown_adjacent_reuse(self):
        report=statistics(fixture(prompt=137,outputs=2))
        self.assertTrue(all(r['reuse_fraction'] is None and r['adjacent_pairs']==0 for r in report['decode_adjacent_reuse']))

    def test_decode_export_is_complete_per_step_and_never_fakes_prefill(self):
        native=fixture(prompt=137,outputs=3);private=decode_fragments(native,source_evidence={'software_fixture':True})
        self.assertEqual(len(private['events']),2);self.assertEqual(len(private['fragments']),32)
        self.assertTrue(all(len(e['route_trace'])==16 for e in private['events']))
        self.assertTrue(all(f['phase']=='decode' and f['request']['tokens']==1 and f['information_mode']=='causal_replay' for f in private['fragments']))
        self.assertTrue(all('scores' not in f['route_trace'][0] for f in private['fragments']))
        self.assertFalse(private['old_full_prefill_bundle_compatible']);self.assertFalse(private['prefill_missing_rows_imputed'])
        self.assertFalse(private['complete_quantized_model_binding_supported'])

    def test_missing_decode_layer_is_rejected_instead_of_filled(self):
        native=fixture(prompt=137,outputs=3)
        native['results'][0]['routes']=[r for r in native['results'][0]['routes'] if not (r['phase']=='decode' and r['forward_index']==1 and r['layer']==3)]
        with self.assertRaisesRegex(ValueError,'sixteen-layer'):decode_fragments(native,source_evidence={'software_fixture':True})

    def test_original_category_labels_are_preserved(self):
        native=fixture(prompt=137,outputs=3);native['request']['placement']='cpu'
        native['request']['requests'][0].update(domain='code',length_class='short',split='calibration')
        report=statistics(native)
        self.assertEqual(report['decode_reuse_by_category'][0]['domain'],'code')
        self.assertEqual(report['decode_reuse_by_category'][0]['length_class'],'short')
        self.assertEqual(report['decode_reuse_by_category'][0]['split'],'calibration')

    def test_invocation_patch_keeps_only_actual_routes_and_causal_mode(self):
        native=fixture(prompt=137,outputs=3);private=decode_fragments(native,source_evidence={'software_fixture':True})
        fragment=private['fragments'][0];patch=m2_invocation_patch(fragment)
        self.assertEqual(patch['route_trace'],fragment['route_trace'])
        self.assertEqual(set(patch),{'phase','requests','route_trace','route_source','information_mode'})
        self.assertEqual(patch['information_mode'],'causal_replay')
        fragment['information_mode']='offline_oracle'
        with self.assertRaisesRegex(ValueError,'causal Decode'):m2_invocation_patch(fragment)


if __name__=='__main__':unittest.main()
