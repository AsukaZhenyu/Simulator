from copy import deepcopy
from fractions import Fraction
import unittest

from llm_infer_model.performance.execution_history import HistoryScope, SerialExecutionHistory
from llm_infer_model.performance.history_conditioned_costs import (
    BUCKETING, INITIALIZATION, SCHEMA, HistoryConditionedResponseTable,
    predict_blocks, response_coordinate, reuse_bucket,
)
from llm_infer_model.performance.parameters import InvalidModel, UnknownParameter
from llm_infer_model.performance.reusable_action_costs import digest

from test_execution_history import program


SCOPE = HistoryScope('b' * 64)
FIELDS = ['h2d_source_reuse_bucket']


def priced(coordinate, duration):
    return dict(coordinate=coordinate, key=digest(coordinate), mean_ns=str(duration),
                sample_count=20, evidence=[dict(path='independent-profile.json', sha256='f' * 64)])


def profile():
    p = program()
    h = SerialExecutionHistory(p, SCOPE)
    rows = {}
    for index in range(len(p['actions'])):
        coord = response_coordinate(h.snapshot(index), FIELDS, use_history=False)
        rows[digest(coord)] = priced(coord, 10 + index)
    # Explicit logical prime with static predicted prices, then independently
    # priced H2D contexts. No complete program measurement is a fitting input.
    for index in range(len(p['actions'])):
        h.commit(index, 10 + index)
    for _ in range(2):
        for index in range(len(p['actions'])):
            coord = response_coordinate(h.snapshot(index), FIELDS)
            if coord['condition']:
                rows[digest(coord)] = priced(coord, 20 + index)
            h.commit(index, 10 + index)
    return dict(schema=SCHEMA, scope_sha256=SCOPE.sha256,
        condition_fields=FIELDS, bucketing=BUCKETING, initialization=INITIALIZATION,
        unmodeled_reset_accesses=True, entries=list(rows.values()))


class HistoryConditionedCostTests(unittest.TestCase):
    def test_bins_are_explicit_floor_power_two_and_unknown_is_separate(self):
        self.assertEqual([reuse_bucket(v) for v in (None, 0, 1, 2, 3, 4, 7, 8)],
                         ['unknown', 'zero', '1', '2', '2', '4', '4', '8'])
        for invalid in (-1, 1.5, True):
            with self.assertRaises(InvalidModel):
                reuse_bucket(invalid)

    def test_conditional_comparison_uses_same_static_calibration_and_no_extra_gap(self):
        table = HistoryConditionedResponseTable(profile())
        conditional = table.predict_blocks(program(), blocks=3, repeats=8)
        static = table.predict_blocks(program(), blocks=3, repeats=8, use_history=False)
        self.assertEqual(conditional['duration_ns'], 80)
        self.assertEqual(static['duration_ns'], 60)
        self.assertEqual(conditional['block_ns'], (80, 80, 80))
        self.assertEqual(conditional['timed_batch_ns'], (640, 640, 640))
        self.assertEqual(len(conditional['action_rows']), 120)
        self.assertEqual(conditional['extra_transition_ns'], 0)
        self.assertFalse(conditional['physical_cache_reset_assumed'])
        self.assertTrue(conditional['unmodeled_reset_accesses'])
        self.assertFalse(conditional['hardware_accuracy_qualified'])
        self.assertFalse(conditional['causal_mechanism_identified'])

    def test_logical_prime_uses_static_cells_without_unknown_context_fallback(self):
        table = HistoryConditionedResponseTable(profile())
        h = SerialExecutionHistory(program(), SCOPE)
        with self.assertRaises(UnknownParameter):
            table.lookup(h.snapshot(0))
        self.assertEqual(table.lookup(h.snapshot(0), use_history=False), 10)
        # The missing cold context is not encountered because prediction models
        # the explicitly declared untimed prime through static cells.
        result = table.predict_blocks(program(), blocks=1)
        self.assertEqual(result['action_rows'][0]['coordinate']['condition'],
                         {'h2d_source_reuse_bucket': '8'})

    def test_missing_timed_condition_rejects_even_when_static_cell_exists(self):
        value = profile()
        value['entries'] = [row for row in value['entries']
                            if row['coordinate']['condition'] != {'h2d_source_reuse_bucket': '8'}]
        table = HistoryConditionedResponseTable(value)
        self.assertEqual(table.predict_blocks(program(), blocks=1, use_history=False)['duration_ns'], 60)
        with self.assertRaises(UnknownParameter):
            table.predict_blocks(program(), blocks=1)

    def test_only_h2d_is_conditioned(self):
        h = SerialExecutionHistory(program(), SCOPE)
        for index in (2, 3, 4):
            self.assertEqual(response_coordinate(h.snapshot(index), FIELDS)['condition'], {})

    def test_time_prediction_does_not_affect_logical_reuse_bin(self):
        a, b = SerialExecutionHistory(program(), SCOPE), SerialExecutionHistory(program(), SCOPE)
        for index in range(len(program()['actions'])):
            a.commit(index, 1)
            b.commit(index, 10000)
        self.assertEqual(response_coordinate(a.snapshot(0), FIELDS), response_coordinate(b.snapshot(0), FIELDS))

    def test_unknown_scope_or_geometry_is_not_interpolated(self):
        table = HistoryConditionedResponseTable(profile())
        other = SerialExecutionHistory(program(), HistoryScope('c' * 64))
        with self.assertRaises(UnknownParameter):
            table.lookup(other.snapshot(0), use_history=False)
        p = program()
        p['graph']['workload']['tensors'][0]['storage_bytes'] = 64
        with self.assertRaises(UnknownParameter):
            table.predict_blocks(p, blocks=1)

    def test_layer_strategy_ordinal_and_future_outcome_predictors_are_rejected(self):
        for name in ('layer_id', 'strategy', 'action_index', 'target_id', 'program_sha256', 'actual_target_ns'):
            value = profile()
            row = value['entries'][0]
            row['coordinate']['semantic'][name] = 'not-a-valid-predictor'
            row['key'] = digest(row['coordinate'])
            with self.subTest(name=name), self.assertRaises(InvalidModel):
                HistoryConditionedResponseTable(value)

    def test_extra_condition_or_reset_sensitive_age_is_rejected(self):
        value = profile()
        value['condition_fields'] = ['age_since_cpu_write_ns']
        with self.assertRaises(InvalidModel):
            HistoryConditionedResponseTable(value)
        value = profile()
        row = next(row for row in value['entries'] if row['coordinate']['condition'])
        row['coordinate']['condition']['another'] = 1
        row['key'] = digest(row['coordinate'])
        with self.assertRaises(InvalidModel):
            HistoryConditionedResponseTable(value)

    def test_missing_baseline_and_duplicate_cell_rejected(self):
        value = profile()
        row = next(row for row in value['entries'] if row['coordinate']['condition'])
        static = dict(row['coordinate'], condition={})
        value['entries'] = [r for r in value['entries'] if r['coordinate'] != static]
        with self.assertRaises(InvalidModel):
            HistoryConditionedResponseTable(value)
        value = profile()
        value['entries'].append(deepcopy(value['entries'][0]))
        with self.assertRaises(InvalidModel):
            HistoryConditionedResponseTable(value)

    def test_unknown_nonfinite_negative_or_unsupported_mean_rejected(self):
        for duration in (None, 'nan', 'inf', '-1', '1/0', True, 1.5):
            value = profile()
            value['entries'][0]['mean_ns'] = duration
            with self.subTest(duration=duration), self.assertRaises(InvalidModel):
                HistoryConditionedResponseTable(value)

    def test_changed_profile_cannot_mutate_frozen_provider(self):
        value = profile()
        table = HistoryConditionedResponseTable.from_profile(value)
        value['entries'][0]['mean_ns'] = '100000000'
        self.assertEqual(predict_blocks(program(), table, blocks=1)['duration_ns'], 80)

    def test_exact_fraction_price_is_not_rounded_per_action(self):
        value = profile()
        for row in value['entries']:
            row['mean_ns'] = '1/3'
        table = HistoryConditionedResponseTable(value)
        result = table.predict_blocks(program(), blocks=2, repeats=3)
        self.assertEqual(result['duration_ns'], Fraction(5, 3))
        self.assertEqual(result['timed_batch_ns'], (5, 5))

    def test_optional_predecessor_and_route_fields_are_explicit(self):
        h = SerialExecutionHistory(program(), SCOPE)
        fields = ['h2d_source_reuse_bucket', 'previous_service', 'previous_copy_route']
        initial = response_coordinate(h.snapshot(0), fields)['condition']
        self.assertIsNone(initial['previous_service'])
        h.commit(1, 10)
        later = response_coordinate(h.snapshot(0), fields)['condition']
        self.assertEqual(later['previous_copy_route'], ['dram', 'vram'])
        self.assertEqual(later['previous_service']['kind'], 'COPY')


if __name__ == '__main__':
    unittest.main()
