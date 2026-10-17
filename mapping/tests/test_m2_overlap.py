"""Real engine intervals for synthetic MoE JIT, prefetch and overlap schedules."""
from __future__ import annotations

import dataclasses
import unittest

from llm_infer_model.m2.costs import parse_cost_table
from llm_infer_model.m2.expand import compile_invocation, plan_for
from llm_infer_model.m2.runner import run_invocation
from llm_infer_model.m2.spec import (
    parse_invocation_document, parse_layer_document, state_in_from_state_out,
    to_state_in_document, weight_id,
)
from llm_infer_model.tensor.engine import (
    ACTION_ADVANCE, ACTION_COMPUTE, ACTION_COPY, ACTION_EVICT, ACTION_READ_SSD,
    Action, MappingError, evaluate_mapping, initial_state, transition, used_bytes,
)
from llm_infer_model.tensor.spec import COPY_RESERVED_COPY, InvalidInput, build_scenario

from .test_m2_moe import moe_cost_document, moe_invocation_document, moe_layer_document


def overlap_layer_document():
    raw = moe_layer_document()
    raw['id'] = 'moe-overlap-small'
    raw['ffn_params']['top_k'] = 3
    return raw


def overlap_call(*, index=0, state=None, devices=('cpu',), tokens=1, routes=None):
    layer = parse_layer_document(overlap_layer_document())
    position = 0 if state is None else state.for_request('r0').next_position
    phase = 'prefill' if index == 0 else 'decode'
    requests = [dict(request_id='r0', position_start=position, tokens=tokens,
                     input_location='dram', output_location='dram')]
    raw = moe_invocation_document(layer, devices=devices, phase=phase, requests=requests)
    raw['id'] = f'moe-overlap-call-{index}'
    raw['layer_file'] = 'moe-overlap-small.layer.json'
    raw['costs_file'] = 'moe-overlap-small.costs.json'
    raw['route_source'] = 'synthetic_fixed_top3_overlap_fixture'
    selections = routes or [[0, 1, 2] for _ in range(tokens)]
    raw['route_trace'] = [dict(request_id='r0', layer_id=layer.id,
        absolute_position=position + offset, expert_ids=selected)
        for offset, selected in enumerate(selections)]
    if state is not None:
        raw['state_in'] = to_state_in_document(state, layer=layer.id)
    invocation = parse_invocation_document(raw)
    return compile_invocation(layer, invocation,
                              parse_cost_table(moe_cost_document(layer, invocation)))


def overlap_templates(*, devices=('cpu',), calls=2):
    result = []
    state = None
    for index in range(calls):
        compiled = overlap_call(index=index, state=state, devices=devices)
        baseline = run_invocation(compiled)
        if baseline.status != 'valid':
            raise AssertionError(baseline.evaluation.reason)
        result.append(compiled)
        state = state_in_from_state_out(baseline.state_out)
    return tuple(result)


def with_overlap_flags(compiled, *, copy_compute=None, cpu_gpu=None):
    document = {**compiled.scenario_document,
                'mapspace': dict(compiled.scenario_document['mapspace'])}
    if copy_compute is not None:
        document['mapspace']['allow_copy_compute_overlap'] = copy_compute
    if cpu_gpu is not None:
        document['mapspace']['allow_cpu_gpu_overlap'] = cpu_gpu
    return dataclasses.replace(compiled, scenario_document=document,
        scenario=build_scenario(compiled.workload_document, document))


def run_plan(compiled, plan):
    return run_invocation(compiled, actions=plan.actions)


def intersecting(first, second):
    return max(first.start_ns, second.start_ns) < min(first.end_ns, second.end_ns)


def compute_prefix(compiled):
    """Projection -> Attention -> Router completion, with no expert access."""
    actions = []
    for task in compiled.tasks:
        if task.kind not in {'projection', 'attention', 'router'}:
            break
        actions.extend((Action(ACTION_COMPUTE, task.operation_id, device='cpu'),
                        Action(ACTION_ADVANCE)))
    return tuple(actions)


def apply_prefix(compiled, actions, *, context=None):
    state = initial_state(compiled.scenario)
    if context is None:
        for action in actions:
            state = transition(compiled.scenario, state, action)
    else:
        from tensor_mapping.m2_overlap import step_schedule
        for action in actions:
            state = step_schedule(context, state, action)
    return state


class OverlapPlanTests(unittest.TestCase):
    def setUp(self):
        from tensor_mapping.m2_overlap import SchedulePolicy, plan_overlap
        self.Policy = SchedulePolicy
        self.plan = plan_overlap

    def policy(self, *, overlap=True, depth=2, dram=None, vram=None):
        return self.Policy(prefetch_depth=depth, overlap=overlap,
            expert_cache_capacity_bytes={'dram': dram, 'vram': vram})

    def test_prefetch_overlaps_actual_compute_interval_and_saves_six_ms(self):
        compiled = overlap_call()
        serial = self.plan(compiled, policy=self.policy(overlap=False, depth=0), max_actions=200)
        overlapped = self.plan(compiled, policy=self.policy(), max_actions=200)
        baseline = run_plan(compiled, serial)
        replay = run_plan(compiled, overlapped)
        self.assertEqual(baseline.status, 'valid', baseline.evaluation.reason)
        self.assertEqual(replay.status, 'valid', replay.evaluation.reason)
        self.assertEqual(baseline.makespan_ns, 29_000_000)
        self.assertEqual(replay.makespan_ns, 23_000_000)
        reads = [event for event in replay.evaluation.events if event.kind == ACTION_READ_SSD]
        computes = [event for event in replay.evaluation.events if event.kind == ACTION_COMPUTE
                    and ':expert.' in event.target_id]
        self.assertEqual(len(reads), 3)
        self.assertTrue(any(intersecting(read, compute) for read in reads for compute in computes))
        ordered = sorted(reads, key=lambda event: event.start_ns)
        self.assertTrue(all(a.end_ns <= b.start_ns for a, b in zip(ordered, ordered[1:])))
        self.assertEqual(replay.evaluation.ssd_read_bytes, 192)
        self.assertTrue(overlapped.complete)
        self.assertEqual(overlapped.termination_reason, 'goal_reached')

    def test_engine_flag_disables_copy_compute_overlap_with_same_prices(self):
        compiled = with_overlap_flags(overlap_call(), copy_compute=False)
        plan = self.plan(compiled, policy=self.policy(), max_actions=200)
        replay = run_plan(compiled, plan)
        self.assertEqual(replay.status, 'valid', replay.evaluation.reason)
        self.assertEqual(replay.makespan_ns, 29_000_000)
        reads = [event for event in replay.evaluation.events if event.kind == ACTION_READ_SSD]
        computes = [event for event in replay.evaluation.events if event.kind == ACTION_COMPUTE]
        self.assertFalse(any(intersecting(read, compute) for read in reads for compute in computes))

    def test_independent_cpu_gpu_experts_overlap_but_macro_start_order_is_fixed(self):
        compiled = overlap_call(devices=('cpu', 'gpu'))
        selected = {task.operation_id: ('gpu' if task.expert_id == 1 else 'cpu')
                    for task in compiled.tasks}
        plan = self.plan(compiled, placements=selected, policy=self.policy(), max_actions=500)
        replay = run_plan(compiled, plan)
        self.assertEqual(replay.status, 'valid', replay.evaluation.reason)
        cpu = [event for event in replay.evaluation.events if event.kind == ACTION_COMPUTE
               and event.device == 'cpu']
        gpu = [event for event in replay.evaluation.events if event.kind == ACTION_COMPUTE
               and event.device == 'gpu']
        self.assertTrue(any(intersecting(a, b) for a in cpu for b in gpu))
        starts = [event.target_id for event in replay.evaluation.events
                  if event.kind == ACTION_COMPUTE]
        self.assertEqual(starts, list(compiled.compute_ids))
        constrained = with_overlap_flags(compiled, cpu_gpu=False)
        plan = self.plan(constrained, placements=selected, policy=self.policy(), max_actions=500)
        replay = run_plan(constrained, plan)
        self.assertEqual(replay.status, 'valid', replay.evaluation.reason)
        cpu = [event for event in replay.evaluation.events if event.kind == ACTION_COMPUTE
               and event.device == 'cpu']
        gpu = [event for event in replay.evaluation.events if event.kind == ACTION_COMPUTE
               and event.device == 'gpu']
        self.assertFalse(any(intersecting(a, b) for a in cpu for b in gpu))

    def test_same_ssd_resource_rejects_two_concurrent_reads(self):
        compiled = overlap_call()
        state = apply_prefix(compiled, compute_prefix(compiled))
        state = transition(compiled.scenario, state,
            Action(ACTION_READ_SSD, weight_id(compiled.layer.id, 'E0')))
        with self.assertRaises(MappingError):
            transition(compiled.scenario, state,
                       Action(ACTION_READ_SSD, weight_id(compiled.layer.id, 'E1')))

    def test_ssd_and_pcie_overlap_and_gpu_weights_still_take_two_hops(self):
        compiled = overlap_call(devices=('cpu', 'gpu'))
        selected = {task.operation_id: ('gpu' if task.kind == 'expert_ffn' else 'cpu')
                    for task in compiled.tasks}
        plan = self.plan(compiled, placements=selected, policy=self.policy(), max_actions=500)
        replay = run_plan(compiled, plan)
        self.assertEqual(replay.status, 'valid', replay.evaluation.reason)
        reads = [event for event in replay.evaluation.events if event.kind == ACTION_READ_SSD]
        h2d = [event for event in replay.evaluation.events if event.kind == ACTION_COPY
               and event.src == 'dram' and event.dst == 'vram']
        self.assertTrue(any(intersecting(read, copy) for read in reads for copy in h2d))
        for expert in ('E0', 'E1', 'E2'):
            name = weight_id(compiled.layer.id, expert)
            read = next(event for event in reads if event.target_id == name)
            copy = next(event for event in h2d if event.target_id == name)
            self.assertLessEqual(read.end_ns, copy.start_ns)
        self.assertTrue(all(event.resource == 'ssd_io' for event in reads))

    def test_causal_router_done_is_required_before_expert_prefetch(self):
        compiled = overlap_call()
        actions = [Action(ACTION_READ_SSD, weight_id(compiled.layer.id, 'E0')),
                   Action(ACTION_ADVANCE), *plan_for(compiled)]
        replay = run_invocation(compiled, actions=actions)
        self.assertEqual(replay.status, 'invalid_mapping')
        self.assertEqual(replay.evaluation.error_code, 'TRANSFER_DEPENDENCY_NOT_READY')
        self.assertIsNone(replay.state_out)

    def test_prefetch_window_is_explicit_and_never_loads_inactive_expert(self):
        from tensor_mapping.m2_overlap import legal_schedule_actions, make_schedule_context
        compiled = overlap_call()
        state = apply_prefix(compiled, compute_prefix(compiled))
        demand = make_schedule_context(compiled, policy=self.policy(depth=0))
        ahead = make_schedule_context(compiled, policy=self.policy(depth=2))
        def read_targets(context):
            return {action.target_id for action in legal_schedule_actions(context, state)
                    if action.kind == ACTION_READ_SSD}
        self.assertEqual(read_targets(demand), {weight_id(compiled.layer.id, 'E0')})
        self.assertEqual(read_targets(ahead),
                         {weight_id(compiled.layer.id, expert) for expert in ('E0', 'E1', 'E2')})
        self.assertNotIn(weight_id(compiled.layer.id, 'E3'), read_targets(ahead))

    def test_pending_copy_reservation_counts_toward_cache_quota(self):
        from tensor_mapping.m2_overlap import (
            legal_schedule_actions, make_schedule_context, step_schedule,
        )
        original = overlap_call()
        state_in = dataclasses.replace(original.state_in, weight_copies=tuple(
            dataclasses.replace(weight, locations=('vram', 'ssd')) if weight.id == 'E1'
            else weight for weight in original.state_in.weight_copies))
        compiled = compile_invocation(original.layer, original.invocation,
                                      original.cost_table, state_in=state_in)
        context = make_schedule_context(compiled,
            policy=self.policy(dram=64, vram=128))
        state = apply_prefix(compiled, compute_prefix(compiled), context=context)
        name0 = weight_id(compiled.layer.id, 'E0')
        name1 = weight_id(compiled.layer.id, 'E1')
        before = used_bytes(compiled.scenario, state, 'dram')
        state = step_schedule(context, state, Action(ACTION_READ_SSD, name0))
        self.assertEqual(state.copy_at(name0, 'dram'), COPY_RESERVED_COPY)
        self.assertEqual(used_bytes(compiled.scenario, state, 'dram') - before, 64)
        blocked = Action(ACTION_COPY, name1, src='vram', dst='dram')
        self.assertNotIn(blocked, legal_schedule_actions(context, state))
        with self.assertRaises((MappingError, InvalidInput)):
            step_schedule(context, state, blocked)

    def test_pending_copy_and_read_locked_source_cannot_be_evicted(self):
        compiled = overlap_call(devices=('cpu', 'gpu'))
        prefix = compute_prefix(compiled)
        state = apply_prefix(compiled, prefix)
        name = weight_id(compiled.layer.id, 'E0')
        state = transition(compiled.scenario, state, Action(ACTION_READ_SSD, name))
        with self.assertRaises(MappingError):
            transition(compiled.scenario, state,
                       Action(ACTION_EVICT, name, location='dram'))
        state = transition(compiled.scenario, state, Action(ACTION_ADVANCE))
        state = transition(compiled.scenario, state,
                           Action(ACTION_COPY, name, src='dram', dst='vram'))
        for domain in ('dram', 'vram'):
            with self.subTest(domain=domain), self.assertRaises(MappingError):
                transition(compiled.scenario, state,
                           Action(ACTION_EVICT, name, location=domain))

    def test_cache_policy_does_not_evict_persistent_kv_even_during_attention(self):
        from tensor_mapping.m2_overlap import legal_schedule_actions, make_schedule_context
        compiled = overlap_templates()[1]
        context = make_schedule_context(compiled, policy=self.policy(dram=128))
        state = initial_state(compiled.scenario)
        first = compiled.tasks[0]
        state = transition(compiled.scenario, state,
                           Action(ACTION_COMPUTE, first.operation_id, device='cpu'))
        state = transition(compiled.scenario, state, Action(ACTION_ADVANCE))
        attention = next(task for task in compiled.tasks if task.kind == 'attention')
        state = transition(compiled.scenario, state,
                           Action(ACTION_COMPUTE, attention.operation_id, device='cpu'))
        kv = compiled.state_in.requests[0].blocks[0].id
        with self.assertRaises(MappingError):
            transition(compiled.scenario, state, Action(ACTION_EVICT, kv, location='dram'))
        self.assertFalse(any(action.kind == ACTION_EVICT and action.target_id == kv
                             for action in legal_schedule_actions(context, state)))

    def test_prefill_decode_binds_actual_cache_kv_and_versions(self):
        first, template = overlap_templates()
        state = dataclasses.replace(first.state_in, weight_copies=tuple(
            dataclasses.replace(weight, version=7) if weight.id == 'E0' else weight
            for weight in first.state_in.weight_copies))
        first = compile_invocation(first.layer, first.invocation, first.cost_table, state_in=state)
        policy = self.policy(dram=192)
        initial_plan = self.plan(first, policy=policy, max_actions=200)
        prefill = run_plan(first, initial_plan)
        self.assertEqual(prefill.status, 'valid', prefill.evaluation.reason)
        actual = state_in_from_state_out(prefill.state_out)
        second = compile_invocation(template.layer, template.invocation,
                                    template.cost_table, state_in=actual)
        decode_plan = self.plan(second, policy=policy, history=initial_plan.history, max_actions=200)
        decode = run_plan(second, decode_plan)
        self.assertEqual(decode.status, 'valid', decode.evaluation.reason)
        self.assertEqual(second.state_in, actual)
        self.assertEqual(decode.evaluation.ssd_read_bytes, 0)
        self.assertEqual((prefill.makespan_ns, decode.makespan_ns), (23_000_000, 20_000_000))
        self.assertEqual(decode.state_out.requests[0].next_position, 2)
        self.assertEqual(decode.state_out.requests[0].version, 3)
        self.assertEqual([block.first_position for block in decode.state_out.requests[0].blocks], [0, 1])
        for weight in decode.state_out.weight_copies:
            if weight.id.startswith('E'):
                self.assertIn('ssd', weight.locations)
        self.assertEqual(next(weight.version for weight in decode.state_out.weight_copies
                              if weight.id == 'E0'), 7)

    def test_same_concrete_overlap_plan_obeys_full_physical_peak_boundary(self):
        compiled = overlap_call()
        plan = self.plan(compiled, policy=self.policy(), max_actions=200)
        base = run_plan(compiled, plan)
        peak = base.evaluation.peak_dram_bytes
        for capacity, expected in ((peak, 'valid'), (peak - 1, 'invalid_mapping')):
            invocation = dataclasses.replace(compiled.invocation,
                hardware=dataclasses.replace(compiled.invocation.hardware,
                                             dram_capacity_bytes=capacity))
            limited = compile_invocation(compiled.layer, invocation, compiled.cost_table)
            replay = run_invocation(limited, actions=plan.actions)
            self.assertEqual(replay.status, expected, replay.evaluation.reason)

    def test_explicit_action_budget_does_not_make_partial_execution_valid(self):
        compiled = overlap_call()
        plan = self.plan(compiled, policy=self.policy(), max_actions=1)
        self.assertFalse(plan.complete)
        self.assertEqual(plan.termination_reason, 'max_actions')
        self.assertNotEqual(run_plan(compiled, plan).status, 'valid')

    def test_policy_fields_and_action_budget_are_strict(self):
        for depth in (True, -1, 1.0, '2'):
            with self.subTest(depth=depth), self.assertRaises(InvalidInput):
                self.plan(overlap_call(), policy=self.policy(depth=depth), max_actions=200)
        for enabled in (0, 1, None, 'true'):
            with self.subTest(enabled=enabled), self.assertRaises(InvalidInput):
                self.plan(overlap_call(), policy=self.policy(overlap=enabled), max_actions=200)
        for budget in (True, 0, -1, 1.0):
            with self.subTest(budget=budget), self.assertRaises(InvalidInput):
                self.plan(overlap_call(), policy=self.policy(), max_actions=budget)


if __name__ == '__main__':
    unittest.main()
