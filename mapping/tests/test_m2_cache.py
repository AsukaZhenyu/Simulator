"""Synthetic MoE cache-policy semantics, independent of hardware timing data.

Five calls with expert accesses 0,1,2,0,1 and two 64-byte cache slots make
LRU reload one more expert than a trace-informed policy. All durations are
declared fixture costs, so this verifies strategy semantics, not speedups.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from llm_infer_model.m2.costs import parse_cost_table
from llm_infer_model.m2.expand import compile_invocation
from llm_infer_model.m2.runner import run_invocation
from llm_infer_model.m2.spec import (
    parse_invocation_document, parse_layer_document, state_in_from_state_out,
    to_state_document, to_state_in_document, weight_id,
)
from llm_infer_model.tensor.engine import (
    ACTION_ADVANCE, ACTION_COMPUTE, ACTION_COPY, ACTION_EVICT, ACTION_READ_SSD,
    Action, evaluate_mapping, load_mapping,
)
from llm_infer_model.tensor.spec import InvalidInput, Unsupported

from .test_m2_moe import moe_cost_document, moe_invocation_document, moe_layer_document


def cache_layer_document():
    document = moe_layer_document()
    document['id'] = 'moe-cache-top1'
    document['ffn_params']['top_k'] = 1
    return document


def cache_call(layer, index, experts, *, state=None, devices=('cpu',),
               hardware=None, phase=None, request_id='r0'):
    """Compile a declared call; source trace positions are never renumbered."""
    position = 0 if state is None else state.for_request(request_id).next_position
    phase = phase or ('prefill' if index == 0 else 'decode')
    request = dict(request_id=request_id, position_start=position, tokens=len(experts),
                   input_location='dram', output_location='dram')
    raw = moe_invocation_document(layer, devices=devices, phase=phase, requests=[request])
    raw['id'] = f'moe-cache-call-{index}'
    raw['layer_file'] = 'moe-cache-top1.layer.json'
    raw['costs_file'] = 'moe-cache-top1.costs.json'
    raw['route_source'] = 'synthetic_fixed_cache_access_sequence'
    raw['route_trace'] = [dict(request_id=request_id, layer_id=layer.id,
        absolute_position=position + offset, expert_ids=[expert])
        for offset, expert in enumerate(experts)]
    if state is not None:
        raw['state_in'] = to_state_in_document(state, layer=layer.id)
    if hardware is not None:
        raw['hardware'].update(hardware)
    invocation = parse_invocation_document(raw)
    table = parse_cost_table(moe_cost_document(layer, invocation))
    return compile_invocation(layer, invocation, table)


def cache_templates(*, accesses=((0,), (1,), (2,), (0,), (1,)), devices=('cpu',),
                    hardware=None, initial_state=None, layer_document=None):
    """Use a legal baseline to declare history shapes for later sequence replay.

    The sequence runner must replace subsequent weight snapshots with the
    selected policy's actual preceding state_out; these baseline copies are
    deliberately different from the constrained cache's outcome.
    """
    layer = parse_layer_document(cache_layer_document() if layer_document is None else layer_document)
    state = initial_state
    compiled = []
    for index, experts in enumerate(accesses):
        call = cache_call(layer, index, experts, state=state, devices=devices,
                          hardware=hardware)
        baseline = run_invocation(call)
        if baseline.status != 'valid':
            raise AssertionError(f'fixture baseline failed: {baseline.evaluation.reason}')
        compiled.append(call)
        state = state_in_from_state_out(baseline.state_out)
    return tuple(compiled)


def expert_reads(candidate):
    return [action.target_id for invocation in candidate.invocations
        for action in invocation.plan.actions
        if action.kind == ACTION_READ_SSD and action.target_id.startswith('w:')
        and action.target_id.rsplit(':', 1)[-1].startswith('E')]


def expert_evictions(candidate, *, location=None):
    return [action for invocation in candidate.invocations
        for action in invocation.plan.actions if action.kind == ACTION_EVICT
        and action.target_id.startswith('w:')
        and (location is None or action.location == location)]


def gpu_expert_placements(templates):
    return tuple({task.operation_id: 'gpu' if task.kind == 'expert_ffn' else 'cpu'
                  for task in compiled.tasks} for compiled in templates)


class CachePolicyTests(unittest.TestCase):
    def setUp(self):
        from tensor_mapping.m2_cache import CachePolicy
        from tensor_mapping.m2_cache_search import run_cache_sequence
        self.Policy = CachePolicy
        self.run_sequence = run_cache_sequence

    def policy(self, kind, *, dram=128, vram=None):
        return self.Policy(kind=kind,
                           expert_cache_capacity_bytes={'dram': dram, 'vram': vram})

    def test_lru_and_offline_farthest_have_explainable_miss_difference(self):
        templates = cache_templates()
        lru = self.run_sequence(templates, policy=self.policy('lru'))
        oracle = self.run_sequence(templates, policy=self.policy('offline_farthest'))
        self.assertEqual(lru.status, 'valid', lru.reason)
        self.assertEqual(oracle.status, 'valid', oracle.reason)
        self.assertEqual([tid.rsplit(':', 1)[-1] for tid in expert_reads(lru)],
                         ['E0', 'E1', 'E2', 'E0', 'E1'])
        self.assertEqual(len(expert_reads(oracle)), 4)
        self.assertEqual(lru.total_makespan_ns, 65_000_000)
        self.assertEqual(oracle.total_makespan_ns, 62_000_000)
        self.assertEqual(lru.total_makespan_ns - oracle.total_makespan_ns, 3_000_000)
        self.assertEqual((lru.cache_hits, lru.cache_misses), (0, 5))
        self.assertEqual((oracle.cache_hits, oracle.cache_misses), (1, 4))

    def test_replay_continues_kv_positions_versions_and_actual_weight_copies(self):
        candidate = self.run_sequence(cache_templates(), policy=self.policy('lru'))
        previous = None
        for index, invocation in enumerate(candidate.invocations):
            self.assertEqual(invocation.result.status, 'valid')
            state = invocation.result.state_out
            self.assertEqual(state.requests[0].next_position, index + 1)
            self.assertEqual(state.requests[0].version, index + 2)
            self.assertEqual([b.first_position for b in state.requests[0].blocks],
                             list(range(index + 1)))
            self.assertEqual(to_state_document(state)['format_version'], 'm2-state-0.2')
            if previous is not None:
                self.assertEqual(invocation.compiled.state_in,
                                 state_in_from_state_out(previous))
            for expert in (w for w in state.weight_copies if w.id.startswith('E')):
                self.assertIn('ssd', expert.locations)
                self.assertEqual(expert.version, 1)
            previous = state
        self.assertTrue(expert_evictions(candidate))
        self.assertEqual(candidate.invocations[-1].result.state_out.requests[0].next_position, 5)

    def test_each_group_is_computed_and_loaded_once_for_two_routed_tokens(self):
        candidate = self.run_sequence(cache_templates(accesses=((0, 0), (1,), (0,))),
                                      policy=self.policy('lru'))
        self.assertEqual(candidate.status, 'valid', candidate.reason)
        first = candidate.invocations[0]
        groups = [t for t in first.compiled.tasks if t.kind == 'expert_ffn']
        self.assertEqual([(t.expert_id, t.shape.tokens) for t in groups], [(0, 2)])
        self.assertEqual(sum(a.kind == ACTION_COMPUTE and a.target_id == groups[0].operation_id
                             for a in first.plan.actions), 1)
        self.assertEqual(expert_reads(candidate),
                         [weight_id(first.compiled.layer.id, 'E0'),
                          weight_id(first.compiled.layer.id, 'E1')])

    def test_causal_route_barrier_is_enforced_for_every_demand_read(self):
        candidate = self.run_sequence(cache_templates(), policy=self.policy('lru'))
        for invocation in candidate.invocations:
            actions = invocation.plan.actions
            routers = {task.operation_id for task in invocation.compiled.tasks
                       if task.kind == 'router'}
            router_indices = [i for i, action in enumerate(actions)
                              if action.kind == ACTION_COMPUTE and action.target_id in routers]
            for index, action in enumerate(actions):
                if action.kind == ACTION_READ_SSD and action.target_id.startswith('w:'):
                    self.assertGreater(index, max(router_indices) + 1)
        first = candidate.invocations[0]
        actions = list(first.plan.actions)
        read = next(i for i, action in enumerate(actions) if action.kind == ACTION_READ_SSD)
        early = [actions[read], Action(ACTION_ADVANCE), *actions[:read], *actions[read + 2:]]
        replay = run_invocation(first.compiled, actions=early)
        self.assertEqual(replay.status, 'invalid_mapping')
        self.assertEqual(replay.evaluation.error_code, 'TRANSFER_DEPENDENCY_NOT_READY')
        self.assertIsNone(replay.state_out)

    def test_no_cache_budget_uses_one_copy_per_touched_expert(self):
        candidate = self.run_sequence(cache_templates(), policy=self.policy('keep_all', dram=None))
        self.assertEqual(candidate.status, 'valid', candidate.reason)
        self.assertEqual(len(expert_reads(candidate)), 3)
        self.assertEqual(expert_evictions(candidate), [])
        self.assertEqual(candidate.total_makespan_ns, 59_000_000)

    def test_keep_all_cannot_ignore_the_same_declared_cache_quota(self):
        candidate = self.run_sequence(cache_templates(), policy=self.policy('keep_all'))
        self.assertNotEqual(candidate.status, 'valid')
        self.assertIsNone(candidate.total_makespan_ns)
        self.assertEqual(len(candidate.invocations), 3)
        self.assertIn('quota', candidate.reason)

    def test_cache_quota_uses_aligned_allocation_bytes(self):
        templates = cache_templates(accesses=((0,), (1,), (0,)),
                                    hardware={'dram_alignment_bytes': 128})
        one_slot = self.run_sequence(templates, policy=self.policy('lru', dram=128))
        two_slots = self.run_sequence(templates, policy=self.policy('lru', dram=256))
        self.assertEqual(one_slot.status, 'valid', one_slot.reason)
        self.assertEqual(two_slots.status, 'valid', two_slots.reason)
        self.assertEqual(len(expert_reads(one_slot)), 3)
        self.assertEqual(len(expert_reads(two_slots)), 2)
        self.assertEqual(one_slot.total_makespan_ns - two_slots.total_makespan_ns, 3_000_000)

    def test_zero_expert_cache_cannot_turn_rejected_call_into_zero_time(self):
        candidate = self.run_sequence(cache_templates(), policy=self.policy('lru', dram=0))
        self.assertNotEqual(candidate.status, 'valid')
        self.assertIsNone(candidate.total_makespan_ns)
        self.assertTrue(candidate.reason)

    def test_capacity_fields_and_policy_are_strict(self):
        for value in (True, -1, 1.0, '128'):
            with self.subTest(value=value), self.assertRaises(InvalidInput):
                self.run_sequence(cache_templates(), policy=self.policy('lru', dram=value))
        with self.assertRaises((InvalidInput, Unsupported)):
            self.run_sequence(cache_templates(), policy=self.policy('unknown-policy'))

    def test_vram_eviction_keeps_dram_staging_and_ssd_source_independent(self):
        templates = cache_templates(devices=('cpu', 'gpu'))
        placements = gpu_expert_placements(templates)
        constrained = self.run_sequence(templates, placements=placements,
            policy=self.policy('lru', dram=None, vram=128))
        retained = self.run_sequence(templates, placements=placements,
            policy=self.policy('lru', dram=None, vram=192))
        for candidate in (constrained, retained):
            self.assertEqual(candidate.status, 'valid', candidate.reason)
            self.assertEqual(len(expert_reads(candidate)), 3)
            for invocation in candidate.invocations:
                for expert in (w for w in invocation.result.state_out.weight_copies
                               if w.id.startswith('E')):
                    self.assertIn('ssd', expert.locations)
        def expert_h2d(candidate):
            return [a for invocation in candidate.invocations for a in invocation.plan.actions
                    if a.kind == ACTION_COPY and a.dst == 'vram'
                    and a.target_id.startswith('w:') and a.target_id.rsplit(':', 1)[-1].startswith('E')]
        self.assertEqual(len(expert_h2d(constrained)), 5)
        self.assertEqual(len(expert_h2d(retained)), 3)
        self.assertEqual(expert_evictions(constrained, location='dram'), [])
        self.assertGreater(len(expert_evictions(constrained, location='vram')), 0)
        self.assertEqual(constrained.total_makespan_ns - retained.total_makespan_ns, 200_000)

    def test_original_dram_only_expert_copy_is_pinned_and_version_preserved(self):
        layer_document = cache_layer_document()
        next(w for w in layer_document['weights'] if w['id'] == 'E0')['initial_location'] = 'dram'
        initial = cache_templates(layer_document=layer_document)[0]
        state = dataclasses.replace(initial.state_in, weight_copies=tuple(
            dataclasses.replace(weight, version=7)
            if weight.id == 'E0' else weight for weight in initial.state_in.weight_copies))
        templates = cache_templates(initial_state=state, layer_document=layer_document)
        candidate = self.run_sequence(templates, policy=self.policy('lru', dram=64))
        self.assertEqual(candidate.status, 'valid', candidate.reason)
        original = weight_id(initial.layer.id, 'E0')
        self.assertNotIn(original, expert_reads(candidate))
        self.assertFalse(any(action.target_id == original
                             for action in expert_evictions(candidate)))
        for invocation in candidate.invocations:
            weight = next(w for w in invocation.result.state_out.weight_copies if w.id == 'E0')
            self.assertEqual(weight.locations, ('dram',))
            self.assertEqual(weight.version, 7)

    def test_declared_snapshot_overrides_layer_initial_location_without_inventing_ssd(self):
        initial = cache_templates()[0]
        state = dataclasses.replace(initial.state_in, weight_copies=tuple(
            dataclasses.replace(weight, locations=('dram',), version=7)
            if weight.id == 'E0' else weight for weight in initial.state_in.weight_copies))
        templates = cache_templates(initial_state=state)
        candidate = self.run_sequence(templates, policy=self.policy('lru', dram=64))
        self.assertEqual(candidate.status, 'valid', candidate.reason)
        expert_id = weight_id(initial.layer.id, 'E0')
        self.assertNotIn(expert_id, expert_reads(candidate))
        self.assertFalse(any(action.target_id == expert_id
                             for action in expert_evictions(candidate)))
        for invocation in candidate.invocations:
            expert = next(w for w in invocation.result.state_out.weight_copies if w.id == 'E0')
            self.assertEqual(expert.locations, ('dram',))
            self.assertEqual(expert.version, 7)

    def test_lru_touches_dram_copy_used_as_h2d_source_but_fifo_does_not(self):
        templates = cache_templates(accesses=((0,), (1,), (0,), (2,), (0,)),
                                    devices=('cpu', 'gpu'))
        placements = gpu_expert_placements(templates)
        lru = self.run_sequence(templates, placements=placements,
                               policy=self.policy('lru', dram=128, vram=64))
        fifo = self.run_sequence(templates, placements=placements,
                                policy=self.policy('fifo', dram=128, vram=64))
        self.assertEqual(lru.status, 'valid', lru.reason)
        self.assertEqual(fifo.status, 'valid', fifo.reason)
        fourth = lru.invocations[3]
        dram_evictions = [a.target_id for a in fourth.plan.actions
                          if a.kind == ACTION_EVICT and a.location == 'dram']
        self.assertIn(weight_id(fourth.compiled.layer.id, 'E1'), dram_evictions)
        self.assertNotIn(weight_id(fourth.compiled.layer.id, 'E0'), dram_evictions)
        self.assertEqual(len(expert_reads(lru)), 3)
        self.assertEqual(len(expert_reads(fifo)), 4)
        self.assertLess(lru.total_makespan_ns, fifo.total_makespan_ns)

    def test_eviction_after_completed_goal_is_refused(self):
        candidate = self.run_sequence(cache_templates(), policy=self.policy('lru'))
        last = candidate.invocations[-1]
        final_weight = next(w for w in last.result.state_out.weight_copies
                            if w.id.startswith('E') and 'dram' in w.locations)
        actions = (*last.plan.actions, Action(ACTION_EVICT,
                   weight_id(last.compiled.layer.id, final_weight.id), location='dram'))
        replay = run_invocation(last.compiled, actions=actions)
        self.assertEqual(replay.status, 'invalid_mapping')
        self.assertIsNone(replay.state_out)
        self.assertIn('goal', replay.evaluation.reason.lower())

    def test_single_prefill_streams_three_experts_through_two_cache_slots(self):
        candidate = self.run_sequence(cache_templates(accesses=((0, 1, 2),)),
                                      policy=self.policy('lru'))
        self.assertEqual(candidate.status, 'valid', candidate.reason)
        invocation = candidate.invocations[0]
        groups = [task for task in invocation.compiled.tasks if task.kind == 'expert_ffn']
        self.assertEqual([(task.expert_id, task.shape.tokens) for task in groups],
                         [(0, 1), (1, 1), (2, 1)])
        self.assertEqual(len(expert_reads(candidate)), 3)
        self.assertEqual(len(expert_evictions(candidate)), 1)
        self.assertEqual({a.target_id for a in expert_evictions(candidate)},
                         {weight_id(invocation.compiled.layer.id, 'E0')})
        for task in groups:
            self.assertEqual(sum(a.kind == ACTION_COMPUTE and a.target_id == task.operation_id
                                 for a in invocation.plan.actions), 1)
        self.assertEqual(invocation.result.state_out.requests[0].next_position, 3)
        self.assertEqual(len(invocation.result.state_out.requests[0].blocks), 3)

    def test_dram_capacity_is_checked_against_all_live_ports_and_scratch(self):
        base = cache_templates(accesses=((0,),))
        candidate = self.run_sequence(base, policy=self.policy('lru'))
        peak = candidate.invocations[0].result.evaluation.peak_dram_bytes
        self.assertGreater(peak, 64)
        for capacity, expected in ((peak, 'valid'), (peak - 1, 'invalid_mapping')):
            call = base[0]
            invocation = dataclasses.replace(call.invocation,
                hardware=dataclasses.replace(call.invocation.hardware,
                                             dram_capacity_bytes=capacity))
            limited = compile_invocation(call.layer, invocation, call.cost_table)
            # Replay the same concrete plan at the boundary. A planner could
            # choose a different eviction at a lower physical capacity; that
            # would be a new strategy rather than validation of this plan.
            replay = run_invocation(limited, actions=candidate.invocations[0].plan.actions)
            self.assertEqual(replay.status, expected, replay.evaluation.reason)
            if expected == 'valid':
                self.assertEqual(replay.evaluation.peak_dram_bytes, peak)
            else:
                self.assertIsNone(replay.makespan_ns)

    def test_initial_capacity_overflow_is_not_repaired_by_fabricating_state(self):
        templates = cache_templates(accesses=((0,), (1,)))
        second = templates[1]
        # The first call's actual cache + W_proj/W_router + one KV + new input
        # occupy 144 bytes at the next initial state; capacity cannot be evaded
        # by deleting READY weight copies from the declared state snapshot.
        invocation = dataclasses.replace(second.invocation,
            hardware=dataclasses.replace(second.invocation.hardware,
                                         dram_capacity_bytes=143))
        second = dataclasses.replace(second, invocation=invocation)
        candidate = self.run_sequence((templates[0], second), policy=self.policy('lru'))
        self.assertNotEqual(candidate.status, 'valid')
        self.assertIsNone(candidate.total_makespan_ns)
        self.assertTrue(candidate.reason)


if __name__ == '__main__':
    unittest.main()
