from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest

from llm_infer_model.performance.reusable_action_costs import digest
from profiling.h2d_history.fit import (
    FIELDS, action_samples, aggregate_samples, calibration_scope, coordinate_trace, freeze,
)


def program():
    def tensor(tid, size):
        return dict(id=tid, dtype='GGML_TYPE_F32', ne=[size//4, 1, 1, 1],
                    nb=[4, size, size, size], storage_bytes=size)
    return dict(boundary_contract='completed-action-test',
        graph=dict(workload=dict(tensors=[tensor('a', 8), tensor('b', 16)])),
        actions=[dict(kind='COPY', target_id='a', src='dram', dst='vram'),
                 dict(kind='COPY', target_id='b', src='dram', dst='vram'), dict(kind='ADVANCE')],
        physical=dict(config={}, threads=1), bindings=[])


def native(trace):
    intervals = [dict(trial=0, invocation=r['invocation'], action_index=r['action_index'], wall_ns=i+1)
                 for i, r in enumerate(trace)]
    return dict(mode='full', instrumentation='stamped', memory_mode='pageable',
        history_intervention='none', preparation_contract=dict(reset_before_batch=True,
            untimed_complete_prime_passes=1, timed_replays_per_batch=2, fixed_input_replay=True),
        trials=[dict(trial=0, repeats=2, actual_stamped=True, action_intervals=intervals,
                     wall_ns='POISONED complete target duration must never be read')])


class H2DHistoryFitTests(unittest.TestCase):
    def test_action_labels_ignore_complete_program_timing(self):
        trace = coordinate_trace(program(), 'a'*64, [5, 7, 0], repeats=2)
        labels = list(action_samples(native(trace), trace, trials=1, repeats=2))
        self.assertEqual([value for _, value in labels], [1, 2, 3, 4, 5, 6])
        self.assertEqual(labels[0][0]['conditional']['condition'], {'h2d_source_reuse_bucket': '16'})

    def test_measured_action_time_does_not_change_later_state_coordinate(self):
        a = coordinate_trace(program(), 'a'*64, [5, 7, 0], repeats=2)
        b = coordinate_trace(program(), 'a'*64, [50000, 70000, 100], repeats=2)
        self.assertEqual(a, b)

    def test_diagnostic_or_changed_sampling_cannot_supply_fitting_labels(self):
        trace = coordinate_trace(program(), 'a'*64, [5, 7, 0], repeats=2)
        for mutate in (
            lambda n: n.update(mode='target-hot'),
            lambda n: n.update(instrumentation='cupti'),
            lambda n: n.update(memory_mode='registered'),
            lambda n: n.update(history_intervention='source-read'),
            lambda n: n['trials'][0]['action_intervals'].reverse(),
            lambda n: n['trials'][0]['action_intervals'][0].update(wall_ns=None),
        ):
            value = native(trace)
            mutate(value)
            with self.subTest(native=value), self.assertRaises(ValueError):
                list(action_samples(value, trace, trials=1, repeats=2))

    def test_aggregation_is_action_mean_and_keeps_all_samples(self):
        coordinate = coordinate_trace(program(), 'a'*64, [5, 7, 0], repeats=2)[0]['conditional']
        key = digest(coordinate)
        rows = aggregate_samples({key: dict(coordinate=coordinate,
            values=[Fraction(1), Fraction(4), Fraction(10)])},
            {key: [dict(native=dict(path='profile.json', sha256='f'*64), split='profile')]})
        self.assertEqual(rows[0]['mean_ns'], '5')
        self.assertEqual(rows[0]['sample_count'], 3)
        self.assertEqual(rows[0]['calibration_spread']['max_ns'], '10')

    def test_scope_excludes_program_or_strategy_predictors(self):
        request = dict(sampling=dict(repeats=8, main_thread_mask=4, warmup_min_ms=500),
            memory_mode='pageable', mode='full', instrumentation='stamped',
            expected_process_mask=4294967295, source_sha256='c'*64)
        manifest = dict(jobs=[dict(split='profile', program_key='layer1-strategy1', request=request)],
            baseline_context=dict(hardware={'cpu':'test','gpu':'test'}),
            binary=dict(sha256='b'*64), runtime=[dict(path='runtime.dll',sha256='d'*64)])
        context, identity = calibration_scope(manifest)
        other = deepcopy(manifest)
        other['jobs'][0]['program_key'] = 'layer999-different-strategy'
        self.assertEqual(calibration_scope(other), (context, identity))
        self.assertEqual(context['state_features'], FIELDS)
        self.assertFalse(context['target_outcome_or_program_identity_used'])

    def test_freeze_rejects_any_heldout_start_before_reading_target_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out/'manifest.json').write_text(json.dumps(dict(schema='h2d-history-campaign-1')),encoding='utf-8')
            (out/'heldout-started.json').write_text('{}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'fit after heldout'):
                freeze(out)
            self.assertFalse((out/'model-frozen.json').exists())


if __name__ == '__main__':
    unittest.main()
