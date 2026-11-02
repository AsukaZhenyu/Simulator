"""Small host-only counterexamples for held-out numerical controls."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from .mmid_numerical import actual_prefix, compare_rows, derived_replay_request, legacy_audit_view, request_for, select_requests, verified_pair_spec
from .numerical_vectors import FORMAT
from .tap_v4 import receipt


def suite_fixture():
    rows = []
    for domain in ('dialogue', 'code', 'math'):
        for length in ('short', 'long'):
            for name in ('z', 'a'):
                rows.append(dict(id=f'{domain}-{length}-{name}', domain=domain, length_class=length,
                    split='heldout', actual_prompt_tokens=3, prompt_token_ids=[1, 2, 3],
                    prompt_sha256='a' * 64, max_output_tokens=64))
    return dict(requests=list(reversed(rows)))


class HeldOutSelection(unittest.TestCase):
    def test_six_cells_selected_deterministically_without_outputs(self):
        source = suite_fixture()
        selected = select_requests(source, {'dialogue-short-a'})
        self.assertEqual(len(selected), 6)
        self.assertEqual(selected[0]['id'], 'dialogue-short-z')
        self.assertTrue(all(row['id'].endswith('-a') for row in selected[1:]))
        self.assertEqual(source, suite_fixture())

    def test_target_outputs_or_missing_metadata_cell_refused(self):
        source = suite_fixture()
        source['requests'][0]['output_token_ids'] = [9]
        # Make the poisoned row the selected one explicitly.
        source['requests'][0]['id'] = '0'
        with self.assertRaises(ValueError):
            select_requests(source, set())
        with self.assertRaises(ValueError):
            select_requests(dict(requests=[]), set())

    def test_request_metadata_remains_bound_but_ids_and_artifacts_are_distinct(self):
        selected = select_requests(suite_fixture(), set())
        template = dict(mode='generate', requests=[], threads=16, cuda_graph_policy='default')
        out = Path.cwd()
        normal = request_for(template, selected, 'fixed-generate', out)
        replay = request_for(template, selected, 'fixed-replay', out)
        self.assertEqual(normal['threads'], 16)
        self.assertTrue(all(r['max_output_tokens'] == 8 and 'output_token_ids' not in r for r in replay['requests']))
        self.assertEqual([r['origin_case_id'] for r in normal['requests']], [r['origin_case_id'] for r in replay['requests']])
        self.assertNotEqual(normal['requests'][0]['logit_artifact_path'], replay['requests'][0]['logit_artifact_path'])


class ActualInputCorrespondence(unittest.TestCase):
    def test_mutable_pair_plan_cannot_change_runtime_or_rehash_a_changed_request(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            def save(path, data):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data), encoding='utf-8', newline='\n')
            template = dict(threads=16, requests=[dict(origin_case_id='source', prompt_token_ids=[1], max_output_tokens=8)])
            reference = dict(results=[dict(origin_case_id='source', prompt_token_ids=[1], output_token_ids=[7])])
            tp, rp = out / 'template.json', out / 'jobs/original-generate/result.json'
            save(tp, template); save(rp, reference)
            save(out / 'jobs/original-generate/execution.json', dict(status='completed', native=receipt(rp)))
            request_path = out / 'requests/fixed-replay.json'
            save(request_path, derived_replay_request(template, reference))
            spec = dict(job_id='fixed-replay', binding='fixed', template=receipt(tp), request=None)
            plan = dict(spec, request=receipt(request_path), actual_reference=receipt(rp), teacher_forcing_used_as_policy_input=False)
            pp = out / 'pair-plans/fixed-replay.json'
            save(pp, plan)
            self.assertEqual(verified_pair_spec(out, spec)['binding'], 'fixed')
            save(pp, dict(plan, binding='original'))
            with self.assertRaisesRegex(ValueError, 'frozen job metadata'):
                verified_pair_spec(out, spec)
            changed = derived_replay_request(template, reference)
            changed['threads'] = 1
            save(request_path, changed)
            save(pp, dict(plan, request=receipt(request_path)))
            with self.assertRaisesRegex(ValueError, 'token-only'):
                verified_pair_spec(out, spec)

    def test_replay_derivation_changes_only_actual_prior_output_tokens(self):
        template = dict(threads=16, requests=[dict(origin_case_id='source', prompt_token_ids=[1, 2], max_output_tokens=8)])
        reference = dict(results=[dict(origin_case_id='source', prompt_token_ids=[1, 2], output_token_ids=[7, 8])])
        request = derived_replay_request(template, reference)
        self.assertEqual(request, dict(threads=16, requests=[dict(origin_case_id='source', prompt_token_ids=[1, 2], max_output_tokens=8, output_token_ids=[7, 8])]))
        self.assertNotIn('output_token_ids', template['requests'][0])
        for changed in (dict(results=reference['results'] * 2),
                        dict(results=[dict(origin_case_id='other', prompt_token_ids=[1, 2], output_token_ids=[7])]),
                        dict(results=[dict(origin_case_id='source', prompt_token_ids=[9], output_token_ids=[7])]),
                        dict(results=[dict(origin_case_id='source', prompt_token_ids=[1, 2], output_token_ids=[True])])):
            with self.assertRaises(ValueError):
                derived_replay_request(template, changed)

    def test_prefill_has_all_chunks_before_first_output(self):
        row = dict(prompt_token_ids=[1] * 513, output_count=2,
            actual_batch_inputs=[dict(actual_input_rows_sha256=str(i)) for i in range(4)])
        self.assertEqual(actual_prefix(row, 0), ['0', '1', '2'])
        self.assertEqual(actual_prefix(row, 1), ['0', '1', '2', '3'])
        with self.assertRaises(ValueError):
            actual_prefix(row, 2)

    def artifact(self, folder, name, values):
        raw = struct.pack('<4f', *values)
        path = Path(folder) / name
        path.write_bytes(raw)
        return dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(),
            bytes=len(raw), rows=2, columns=2, format=FORMAT)

    def test_unequal_actual_prefix_never_counts_as_numerical_agreement(self):
        with tempfile.TemporaryDirectory() as folder:
            base = dict(origin_case_id='same-source', prompt_token_ids=[1, 2], output_count=2,
                actual_batch_inputs=[dict(actual_input_rows_sha256='a'), dict(actual_input_rows_sha256='b')],
                sampled_greedy_token_ids=[10, 11], graph_observations=[dict(graph_partition_fingerprint_sha256='g')] * 2)
            left, right = copy.deepcopy(base), copy.deepcopy(base)
            left['full_logits_artifact'] = self.artifact(folder, 'a.f32', [1., 2., 3., 4.])
            right['full_logits_artifact'] = self.artifact(folder, 'b.f32', [1., 2., 3., 4.])
            right['actual_batch_inputs'][1]['actual_input_rows_sha256'] = 'different-token-input'
            result = compare_rows(left, right)
            self.assertTrue(result['rows'][0]['bitwise_equal'])
            self.assertIsNone(result['rows'][1]['bitwise_equal'])
            self.assertEqual(result['rows'][1]['comparison_status'], 'unsupported_unequal_actual_input_prefix')

    def test_logit_difference_on_same_inputs_is_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            base = dict(origin_case_id='same-source', prompt_token_ids=[1], output_count=2,
                actual_batch_inputs=[dict(actual_input_rows_sha256='a'), dict(actual_input_rows_sha256='b')],
                sampled_greedy_token_ids=[10, 11], graph_observations=[dict(graph_partition_fingerprint_sha256='g')] * 2)
            left, right = copy.deepcopy(base), copy.deepcopy(base)
            left['full_logits_artifact'] = self.artifact(folder, 'a.f32', [1., 2., 3., 4.])
            right['full_logits_artifact'] = self.artifact(folder, 'b.f32', [1., 2.5, 3., 4.])
            result = compare_rows(left, right)
            self.assertEqual(result['first_bitwise_divergence'], 0)
            self.assertEqual(result['rows'][0]['max_abs'], .5)
            self.assertTrue(result['rows'][1]['bitwise_equal'])

    def test_legacy_view_never_erases_execution_inputs_or_edits_raw_record(self):
        source = dict(schema='real-moe-numerical-native-6', source_sha256='a' * 64,
            request=dict(schema='real-moe-numerical-request-6', threads=16, cuda_graph_policy='default',
                requests=[dict(id='job-o00', origin_case_id='source', prompt_token_ids=[1],
                    output_token_ids=[2], logit_artifact_path='x.f32')]),
            results=[dict(id='job-o00', origin_case_id='source', output_token_ids=[2])])
        original = copy.deepcopy(source)
        view = legacy_audit_view(source)
        self.assertEqual(source, original)
        self.assertEqual(view['request']['threads'], 16)
        self.assertEqual(view['request']['requests'][0]['output_token_ids'], [2])
        self.assertEqual(view['request']['requests'][0]['id'], 'source')
        self.assertEqual(view['results'][0]['id'], 'source')
        self.assertNotIn('logit_artifact_path', view['request']['requests'][0])


if __name__ == '__main__':
    unittest.main()
