from copy import deepcopy
from fractions import Fraction
import json
import unittest

from llm_infer_model.performance.execution_history import (
    HistoryScope, SerialExecutionHistory, evaluate_with_history,
)
from llm_infer_model.performance.parameters import InvalidModel, UnknownParameter


SCOPE = HistoryScope('a' * 64)


def program():
    def tensor(tid, size):
        return dict(id=tid, dtype='GGML_TYPE_F32', ne=[size // 4, 1, 1, 1],
                    nb=[4, size, size, size], storage_bytes=size)
    return dict(boundary_contract='test-completed-action',
        graph=dict(workload=dict(tensors=[tensor('a', 8), tensor('b', 16), tensor('out', 8)])),
        physical=dict(config=dict(hidden=2), threads=1),
        actions=[dict(kind='COPY', target_id='a', src='dram', dst='vram'),
            dict(kind='COPY', target_id='b', src='dram', dst='vram'),
            dict(kind='COMPUTE', target_id='op', device='cpu'),
            dict(kind='COPY', target_id='out', src='vram', dst='dram'),
            dict(kind='ADVANCE')],
        bindings=[dict(operation_id='op', kind='test_compute', shape=dict(tokens=1), op_params=[],
            operands=dict(left='a', duplicate_role='a', right='b'),
            outputs=dict(result='out'), history=[], new_kv=[])])


class ExecutionHistoryTests(unittest.TestCase):
    def test_initial_cache_and_physical_readiness_remain_unknown(self):
        state = SerialExecutionHistory(program(), SCOPE).snapshot(0).to_dict()
        self.assertFalse(state['physical_cache_queue_or_clock_inferred'])
        self.assertFalse(state['target_outcome_used'])
        self.assertEqual(state['features']['logical_host_distinct_bytes'], 0)
        for access in state['features']['accesses']:
            self.assertIsNone(access['age_since_any_ns'])
            self.assertIsNone(access['distinct_other_host_bytes_since_last_access'])
        json.dumps(state, allow_nan=False)

    def test_reuse_footprint_is_distinct_not_accumulated_traffic(self):
        h = SerialExecutionHistory(program(), SCOPE)
        h.commit(0, 10)
        h.commit(1, 20)
        h.commit(1, 20)
        h.idle(5)
        state = h.snapshot(0).features
        source = next(a for a in state['accesses'] if a['location'] == 'dram')
        self.assertEqual(source['distinct_other_host_bytes_since_last_access'], 16)
        self.assertEqual(source['age_since_any_ns'], 45)
        self.assertEqual(source['age_since_copy_read_ns'], 45)
        self.assertIsNone(source['age_since_cpu_read_ns'])
        self.assertEqual(state['logical_host_distinct_bytes'], 24)
        self.assertEqual(state['logical_host_access_bytes']['copy_read'], 40)
        self.assertEqual(state['predicted_busy_ns']['copy'], 50)
        self.assertEqual(state['explicit_idle_ns'], 5)
        self.assertEqual(state['idle_since_last_service_ns'], 5)

    def test_cpu_read_write_and_copy_are_separate_mechanisms(self):
        h = SerialExecutionHistory(program(), SCOPE)
        h.commit(0, 10)
        h.commit(2, 30)
        state = h.snapshot(0).features
        source = next(a for a in state['accesses'] if a['location'] == 'dram')
        self.assertEqual(source['age_since_cpu_read_ns'], 0)
        self.assertEqual(source['age_since_copy_read_ns'], 30)
        self.assertEqual(state['logical_host_access_bytes']['cpu_read'], 24)
        self.assertEqual(state['logical_host_access_bytes']['cpu_write'], 8)
        self.assertEqual(state['predicted_busy_ns']['cpu'], 30)
        self.assertEqual(state['last_cpu_compute'], 'test_compute')
        self.assertEqual(state['previous_service'], dict(kind='COMPUTE', device='cpu', semantic_op='test_compute'))
        h.commit(3, 7)
        state = h.snapshot(3).features
        dest = next(a for a in state['accesses'] if a['location'] == 'dram')
        self.assertEqual(dest['age_since_copy_write_ns'], 0)
        self.assertEqual(dest['age_since_cpu_write_ns'], 7)

    def test_route_switch_and_logical_controls_keep_physical_predecessor(self):
        h = SerialExecutionHistory(program(), SCOPE)
        self.assertIsNone(h.snapshot(0).features['copy_route_switch'])
        h.commit(0, 11)
        self.assertFalse(h.snapshot(1).features['copy_route_switch'])
        self.assertTrue(h.snapshot(3).features['copy_route_switch'])
        h.commit(4, 2)
        self.assertEqual(h.snapshot(3).features['previous_service']['kind'], 'COPY')
        self.assertEqual(h.snapshot(3).features['idle_since_last_service_ns'], 2)

    def test_renaming_layers_strategies_buffers_and_operations_changes_no_feature(self):
        original, renamed = program(), program()
        mapping = {'a': 'layer-999:a', 'b': 'strategy-other:b', 'out': 'other:out', 'op': 'layer-new:op'}
        for t in renamed['graph']['workload']['tensors']:
            t['id'] = mapping[t['id']]
        for a in renamed['actions']:
            if 'target_id' in a:
                a['target_id'] = mapping[a['target_id']]
        for b in renamed['bindings']:
            b['operation_id'] = mapping[b['operation_id']]
            for side in ('operands', 'outputs'):
                b[side] = {role: mapping[tid] for role, tid in b[side].items()}
        renamed.update(layer_id='999', strategy='another', seed=1000,
                       original_simulation={'actual_target_ns': 1e99})
        a, b = SerialExecutionHistory(original, SCOPE), SerialExecutionHistory(renamed, SCOPE)
        for index in (0, 1, 2, 3, 4, 0):
            self.assertEqual(a.snapshot(index).to_dict(), b.snapshot(index).to_dict())
            a.commit(index, 5)
            b.commit(index, 5)

    def test_target_prediction_cannot_change_its_own_pre_state(self):
        a, b = SerialExecutionHistory(program(), SCOPE), SerialExecutionHistory(program(), SCOPE)
        for h in (a, b):
            h.commit(0, 10)
        before = deepcopy(a.snapshot(1).to_dict())
        self.assertEqual(before, b.snapshot(1).to_dict())
        a.commit(1, 20)
        b.commit(1, 200)
        self.assertEqual(before['features']['elapsed_predicted_ns'], '10')
        self.assertNotEqual(a.snapshot(2).to_dict(), b.snapshot(2).to_dict())

    def test_returned_snapshot_cannot_mutate_live_history(self):
        h = SerialExecutionHistory(program(), SCOPE)
        h.commit(0, 10)
        snapshot = h.snapshot(1)
        snapshot.features['predicted_busy_ns']['copy'] = 999
        snapshot.semantic['tensor']['storage_bytes'] = 999
        fresh = h.snapshot(1)
        self.assertEqual(fresh.features['predicted_busy_ns']['copy'], 10)
        self.assertEqual(fresh.semantic['tensor']['storage_bytes'], 16)

    def test_exact_fraction_and_scope_envelope(self):
        h = SerialExecutionHistory(program(), HistoryScope('a' * 64, max_elapsed_ns=3, max_completed_actions=2))
        h.commit(0, Fraction(1, 3))
        self.assertEqual(h.snapshot(1).features['elapsed_predicted_ns'], Fraction(1, 3))
        with self.assertRaises(UnknownParameter):
            h.commit(1, 3)
        self.assertEqual(h.time_ns, Fraction(1, 3))
        self.assertEqual(h.completed_actions, 1)
        h.commit(1, 1)
        with self.assertRaises(UnknownParameter):
            h.commit(2, 0)
        with self.assertRaises(UnknownParameter):
            h.idle(3)

    def test_unknown_response_is_not_silently_zero_or_hot_price(self):
        with self.assertRaises(UnknownParameter):
            evaluate_with_history(program(), SCOPE, lambda state: None)

    def test_provider_receives_only_prefix_and_inclusive_cost_is_added_once(self):
        seen = []
        def response(state):
            seen.append(state.to_dict())
            return Fraction(3, 2)
        result = evaluate_with_history(program(), SCOPE, response, repeats=2)
        self.assertEqual(result['duration_ns'], 15)
        self.assertEqual(len(seen), 10)
        self.assertEqual([row['features']['elapsed_predicted_ns'] for row in seen],
                         [str(Fraction(3 * i, 2)) for i in range(10)])
        self.assertFalse(result['hardware_accuracy_qualified'])

    def test_invalid_or_unsupported_descriptors_fail(self):
        for mutate in (
            lambda p: p['graph']['workload']['tensors'][0].update(storage_bytes=None),
            lambda p: p['graph']['workload']['tensors'][0].update(storage_bytes=True),
            lambda p: p['graph']['workload']['tensors'][0].update(alias_of='b'),
            lambda p: p['graph']['workload']['tensors'].append(deepcopy(p['graph']['workload']['tensors'][0])),
            lambda p: p['actions'][0].update(target_id='missing'),
            lambda p: p['actions'][0].update(src='ssd'),
            lambda p: p['actions'][2].update(device='npu'),
        ):
            p = program()
            mutate(p)
            with self.subTest(program=p), self.assertRaises(InvalidModel):
                SerialExecutionHistory(p, SCOPE)

    def test_unknown_scope_and_nonexact_time_rejected(self):
        for scope in ('', 'program-name', None, 'g' * 64):
            with self.subTest(scope=scope), self.assertRaises(InvalidModel):
                HistoryScope(scope)
        h = SerialExecutionHistory(program(), SCOPE)
        for value in (1.5, -1, None, True):
            with self.subTest(value=value), self.assertRaises(InvalidModel):
                h.commit(0, value)
        for index in (-1, 5, True):
            with self.subTest(index=index), self.assertRaises(InvalidModel):
                h.snapshot(index)


if __name__ == '__main__':
    unittest.main()
