"""A model-independent tensor DAG uses the generic schedule core directly."""
from __future__ import annotations

import dataclasses
import unittest

from llm_infer_model.tensor.engine import (
    ACTION_ADVANCE, ACTION_COMPUTE, ACTION_COPY, ACTION_EVICT, ACTION_READ_SSD,
    Action, MappingError, evaluate_mapping, initial_state, used_bytes,
)
from llm_infer_model.tensor.spec import COPY_RESERVED_COPY, InvalidInput, Unsupported, build_scenario


def generic_dag_documents(*, cpu_gpu=True, copy_compute=True):
    def tensor(name, role, locations=()):
        result = dict(id=name, name=name, role=role, dtype='GGML_TYPE_F32',
                      ne=[4, 1, 1, 1], nb=[4, 16, 16, 16], storage_bytes=16)
        if locations:
            result['initial_locations'] = list(locations)
        return result
    workload = dict(schema_version='0.2', id='opaque-fork-join',
        origin=dict(kind='synthetic_fixture', ggml_version=None, sample='generic scheduling protocol'),
        tensors=[tensor('a', 'input', ('ssd',)), tensor('b', 'input', ('ssd',)),
                 tensor('u', 'intermediate'), tensor('v', 'intermediate'), tensor('out', 'output')],
        operations=[dict(id='alpha', semantic_op='RELU', ggml_op='UNARY', unary_op='RELU',
                         op_params=[], inputs=['a'], output='u'),
                    dict(id='beta', semantic_op='RELU', ggml_op='UNARY', unary_op='RELU',
                         op_params=[], inputs=['b'], output='v'),
                    dict(id='omega', semantic_op='ADD', ggml_op='ADD', unary_op=None,
                         op_params=[], inputs=['u', 'v'], output='out')], outputs=['out'])
    allowed = {'ssd_to_dram': ['a', 'b'], 'dram_to_vram': ['b'],
               'vram_to_dram': ['v'], 'dram_to_ssd': []}
    scenario = dict(schema_version='0.2', id='opaque-fork-join',
        architecture=dict(dram_capacity_bytes=4096, vram_capacity_bytes=4096,
            dram_runtime_reserved_bytes=0, vram_runtime_reserved_bytes=0,
            dram_alignment_bytes=1, vram_alignment_bytes=1, cpu_compute_slots=1,
            gpu_compute_slots=1, pcie_copy_slots=1, ssd_io_slots=1),
        goal=dict(output_locations={'out': 'dram'}),
        costs=dict(source='synthetic_fixed', compute={
            'alpha': {'cpu': dict(duration_ns=5_000_000, workspace_bytes=16)},
            'beta': {'gpu': dict(duration_ns=5_000_000, workspace_bytes=16)},
            'omega': {'cpu': dict(duration_ns=1_000_000, workspace_bytes=16)}},
            transfer={route: {name: dict(duration_ns=3_000_000 if route == 'ssd_to_dram' else 100_000)
                              for name in names} for route, names in allowed.items()}),
        mapspace=dict(allowed_devices={'alpha': ['cpu'], 'beta': ['gpu'], 'omega': ['cpu']},
            allowed_transfers=allowed, allow_eviction=True,
            allow_copy_compute_overlap=copy_compute, allow_cpu_gpu_overlap=cpu_gpu,
            allow_recomputation=False),
        mapper=dict(algorithm='uniform_cost', max_expanded_states=1000, wall_time_limit_s=1))
    return workload, scenario


def generic_dag(**kwargs):
    return build_scenario(*generic_dag_documents(**kwargs))


def generic_facts(*, order=('alpha', 'beta', 'omega'), protected=frozenset()):
    from tensor_mapping.schedule import ScheduleFacts
    return ScheduleFacts(operation_order=order, managed_reusable_tensors=frozenset({'a', 'b'}),
                         authoritative_sources=(('a', 'ssd'), ('b', 'ssd')),
                         protected_tensors=protected, binding_id='opaque-dag-inputs')


def intersects(a, b):
    return max(a.start_ns, b.start_ns) < min(a.end_ns, b.end_ns)


class GenericScheduleTests(unittest.TestCase):
    def setUp(self):
        from tensor_mapping.schedule import (
            SchedulePolicy, legal_schedule_actions, make_schedule_context,
            plan_schedule, step_schedule,
        )
        self.Policy = SchedulePolicy
        self.context = make_schedule_context
        self.plan = plan_schedule
        self.legal = legal_schedule_actions
        self.step = step_schedule

    def policy(self, *, overlap=True, depth=1, dram=32, vram=16):
        return self.Policy(prefetch_depth=depth, overlap=overlap,
                           managed_capacity_bytes={'dram': dram, 'vram': vram})

    def test_opaque_non_llm_fork_join_has_actual_compute_and_transfer_overlap(self):
        scenario = generic_dag()
        context = self.context(scenario, generic_facts(), self.policy())
        plan = self.plan(context, max_actions=100)
        replay = evaluate_mapping(scenario, plan.actions)
        self.assertEqual(replay.status, 'valid', replay.reason)
        self.assertEqual(replay.makespan_ns, 12_200_000)
        self.assertEqual([event.target_id for event in replay.events if event.kind == ACTION_COMPUTE],
                         ['alpha', 'beta', 'omega'])
        computes = [event for event in replay.events if event.kind == ACTION_COMPUTE]
        transfers = [event for event in replay.events if event.bytes_transferred]
        self.assertTrue(any(intersects(c, t) for c in computes for t in transfers))
        self.assertTrue(any(intersects(a, b) for a in computes for b in computes
                            if a.device != b.device))
        self.assertTrue(plan.complete)
        self.assertEqual(replay.ssd_read_bytes, 32)

    def test_same_dag_serial_policy_uses_all_same_costs_without_timing_discount(self):
        scenario = generic_dag()
        context = self.context(scenario, generic_facts(), self.policy(overlap=False))
        replay = evaluate_mapping(scenario, self.plan(context, max_actions=100).actions)
        self.assertEqual(replay.status, 'valid', replay.reason)
        self.assertEqual(replay.makespan_ns, 17_200_000)
        timed = [event for event in replay.events if event.start_ns is not None]
        self.assertFalse(any(intersects(a, b) for i, a in enumerate(timed) for b in timed[i + 1:]))

    def test_explicit_operation_order_is_not_inferred_from_tensor_names(self):
        scenario = generic_dag()
        facts = generic_facts(order=('beta', 'alpha', 'omega'))
        plan = self.plan(self.context(scenario, facts, self.policy()), max_actions=100)
        replay = evaluate_mapping(scenario, plan.actions)
        self.assertEqual(replay.status, 'valid', replay.reason)
        self.assertEqual([event.target_id for event in replay.events if event.kind == ACTION_COMPUTE],
                         ['beta', 'alpha', 'omega'])
        self.assertEqual(replay.makespan_ns, 12_000_000)

    def test_generic_pool_eviction_preserves_sources_and_join_operands(self):
        scenario = generic_dag()
        plan = self.plan(self.context(scenario, generic_facts(), self.policy(dram=16)), max_actions=100)
        replay = evaluate_mapping(scenario, plan.actions)
        self.assertEqual(replay.status, 'valid', replay.reason)
        evictions = [action for action in plan.actions if action.kind == ACTION_EVICT]
        self.assertTrue(evictions)
        self.assertTrue(all(action.target_id in {'a', 'b'} and action.location != 'ssd'
                            for action in evictions))
        snapshot = replay.states[-1]
        self.assertEqual(dict(snapshot.ssd_status)['a'], 'READY')
        self.assertEqual(dict(snapshot.ssd_status)['b'], 'READY')
        self.assertEqual(replay.makespan_ns, 17_200_000)

    def test_non_source_reservation_is_charged_before_copy_finishes(self):
        workload, raw = generic_dag_documents()
        workload['tensors'][1]['initial_locations'] = ['vram', 'ssd']
        raw['mapspace']['allowed_transfers']['vram_to_dram'].append('b')
        raw['costs']['transfer']['vram_to_dram']['b'] = dict(duration_ns=100_000)
        scenario = build_scenario(workload, raw)
        context = self.context(scenario, generic_facts(), self.policy(dram=16))
        state = initial_state(scenario)
        state = self.step(context, state, Action(ACTION_READ_SSD, 'a'))
        self.assertEqual(state.copy_at('a', 'dram'), COPY_RESERVED_COPY)
        self.assertEqual(used_bytes(scenario, state, 'dram'), 16)
        copy = Action(ACTION_COPY, 'b', src='vram', dst='dram')
        self.assertNotIn(copy, self.legal(context, state))
        with self.assertRaises((MappingError, InvalidInput)):
            self.step(context, state, copy)

    def test_pending_reservation_is_not_a_legal_cache_victim(self):
        scenario = generic_dag()
        context = self.context(scenario, generic_facts(), self.policy(dram=16))
        state = self.step(context, initial_state(scenario), Action(ACTION_READ_SSD, 'a'))
        self.assertNotIn(Action(ACTION_EVICT, 'a', location='dram'), self.legal(context, state))

    def test_protected_tensor_is_not_selected_as_a_reusable_pool_victim(self):
        scenario = generic_dag()
        facts = dataclasses.replace(generic_facts(protected=frozenset({'a'})),
            managed_reusable_tensors=frozenset({'b'}), authoritative_sources=(('b', 'ssd'),))
        context = self.context(scenario, facts, self.policy(dram=16))
        state = self.step(context, initial_state(scenario), Action(ACTION_READ_SSD, 'a'))
        state = self.step(context, state, Action(ACTION_ADVANCE))
        state = self.step(context, state, Action(ACTION_COMPUTE, 'alpha', device='cpu'))
        state = self.step(context, state, Action(ACTION_ADVANCE))
        self.assertNotIn(Action(ACTION_EVICT, 'a', location='dram'), self.legal(context, state))

    def test_managed_protected_overlap_is_refused_as_a_contradictory_descriptor(self):
        with self.assertRaises(InvalidInput):
            self.context(generic_dag(), generic_facts(protected=frozenset({'a'})), self.policy())

    def test_complete_physical_capacity_still_counts_workspace_and_outputs(self):
        scenario = generic_dag()
        plan = self.plan(self.context(scenario, generic_facts(), self.policy()), max_actions=100)
        base = evaluate_mapping(scenario, plan.actions)
        for capacity, expected in ((base.peak_dram_bytes, 'valid'),
                                   (base.peak_dram_bytes - 1, 'invalid_mapping')):
            limited = dataclasses.replace(scenario, architecture=dataclasses.replace(
                scenario.architecture, dram_capacity_bytes=capacity))
            replay = evaluate_mapping(limited, plan.actions)
            self.assertEqual(replay.status, expected, replay.reason)

    def test_workspace_unknown_is_refused_without_zero_or_default_substitution(self):
        workload, raw = generic_dag_documents()
        raw['costs']['compute']['alpha']['cpu']['workspace_bytes'] = None
        with self.assertRaises((InvalidInput, Unsupported)):
            build_scenario(workload, raw)

    def test_window_does_not_sweep_unrequested_reusable_values(self):
        scenario = generic_dag()
        context = self.context(scenario, generic_facts(), self.policy(depth=0))
        legal = self.legal(context, initial_state(scenario))
        self.assertIn(Action(ACTION_READ_SSD, 'a'), legal)
        self.assertNotIn(Action(ACTION_READ_SSD, 'b'), legal)

    def test_generic_action_budget_keeps_partial_schedule_unfinished(self):
        scenario = generic_dag()
        plan = self.plan(self.context(scenario, generic_facts(), self.policy()), max_actions=1)
        self.assertFalse(plan.complete)
        self.assertEqual(plan.termination_reason, 'max_actions')
        self.assertNotEqual(evaluate_mapping(scenario, plan.actions).status, 'valid')


if __name__ == '__main__':
    unittest.main()
