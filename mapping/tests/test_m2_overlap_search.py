"""Global episode UCS, concrete replay and honest stopping claims."""
from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from llm_infer_model.m2.expand import compile_invocation
from llm_infer_model.m2.runner import run_invocation
from llm_infer_model.m2.spec import state_in_from_state_out
from llm_infer_model.tensor.engine import ACTION_COMPUTE, evaluate_mapping, load_mapping
from llm_infer_model.tensor.spec import InvalidInput, Unsupported, build_scenario, scenario_fingerprint, workload_fingerprint
from tensor_mapping.artifacts import action_to_dict
from tensor_mapping.m2_cache import CachePolicy
from tensor_mapping.m2_cache_search import CacheInvocationResult, CacheSequenceCandidate
from tensor_mapping.m2_overlap import SchedulePolicy, plan_overlap
from tensor_mapping.m2_overlap_search import search_schedule_episode

from .test_m2_cache import cache_layer_document, cache_templates, expert_reads
from .test_m2_overlap import intersecting, overlap_call, overlap_templates, with_overlap_flags


def fixed_cpu(templates):
    return tuple({task.operation_id: 'cpu' for task in compiled.tasks} for compiled in templates)


def greedy_sequence(templates, policy, *, placements=None):
    placements = fixed_cpu(templates) if placements is None else placements
    actual = templates[0].state_in
    history = None
    invocations = []
    for template, selected in zip(templates, placements):
        compiled = compile_invocation(template.layer, template.invocation,
                                      template.cost_table, state_in=actual)
        plan = plan_overlap(compiled, placements=selected, policy=policy,
                            history=history, max_actions=1000)
        result = run_invocation(compiled, actions=plan.actions)
        if result.status != 'valid':
            raise AssertionError(result.evaluation.reason)
        invocations.append(CacheInvocationResult(compiled, tuple(selected.items()), plan, result))
        actual = state_in_from_state_out(result.state_out)
        history = plan.history
    return CacheSequenceCandidate(0,
        CachePolicy(kind='lru', expert_cache_capacity_bytes=policy.expert_cache_capacity_bytes),
        tuple(invocations), len(templates))


class EpisodeScheduleSearchTests(unittest.TestCase):
    def policy(self, *, depth=2, overlap=True, dram=192, vram=192):
        return SchedulePolicy(prefetch_depth=depth, overlap=overlap,
                              expert_cache_capacity_bytes={'dram': dram, 'vram': vram})

    def search(self, templates, *, policy=None, placements=None, expanded=1000, generated=2000,
               incumbent=None):
        return search_schedule_episode(templates, policy=self.policy() if policy is None else policy,
            placements=fixed_cpu(templates) if placements is None else placements,
            max_expanded_states=expanded, max_generated_states=generated, incumbent=incumbent)

    def test_global_prefill_decode_search_replays_actual_state_at_43_ms(self):
        templates = overlap_templates()
        result = self.search(templates)
        self.assertEqual(result.status, 'optimal_within_declared_space')
        self.assertTrue(result.optimality_proven)
        self.assertTrue(result.complete)
        self.assertTrue(result.search_complete)
        self.assertFalse(result.enumeration_complete)
        self.assertEqual(result.termination_reason, 'episode_goal_popped')
        self.assertEqual(result.best.total_makespan_ns, 43_000_000)
        first, second = result.best.invocations
        self.assertEqual((first.result.makespan_ns, second.result.makespan_ns),
                         (23_000_000, 20_000_000))
        self.assertEqual(second.compiled.state_in, state_in_from_state_out(first.result.state_out))
        self.assertEqual(second.result.evaluation.ssd_read_bytes, 0)
        self.assertEqual(second.result.state_out.requests[0].next_position, 2)
        self.assertEqual(second.result.state_out.requests[0].version, 3)

    def test_global_future_cache_effect_beats_lru_on_the_whole_episode(self):
        templates = cache_templates()
        policy = self.policy(depth=0, overlap=False, dram=128, vram=None)
        lru = greedy_sequence(templates, policy)
        result = self.search(templates, policy=policy)
        self.assertEqual(lru.total_makespan_ns, 65_000_000)
        self.assertEqual(result.status, 'optimal_within_declared_space')
        self.assertEqual(result.best.total_makespan_ns, 62_000_000)
        self.assertEqual(len(expert_reads(result.best)), 4)
        self.assertLess(result.best.total_makespan_ns, lru.total_makespan_ns)
        for previous, following in zip(result.best.invocations, result.best.invocations[1:]):
            self.assertEqual(following.compiled.state_in, state_in_from_state_out(previous.result.state_out))

    def test_result_latency_is_sum_of_replayed_calls_and_intersections_are_real(self):
        result = self.search(overlap_templates())
        total = 0
        intersects = 0
        for invocation in result.best.invocations:
            replay = run_invocation(invocation.compiled, actions=invocation.plan.actions)
            self.assertEqual(replay.status, 'valid', replay.evaluation.reason)
            total += replay.makespan_ns
            compute = [event for event in replay.evaluation.events if event.kind == ACTION_COMPUTE]
            transfer = [event for event in replay.evaluation.events
                        if event.bytes_transferred > 0]
            intersects += sum(intersecting(c, t) for c in compute for t in transfer)
            self.assertEqual([event.target_id for event in compute], list(invocation.compiled.compute_ids))
        self.assertEqual(total, result.best.total_makespan_ns)
        self.assertGreater(intersects, 0)

    def test_searches_declared_compute_device_choice_within_actions(self):
        document = cache_layer_document()
        next(w for w in document['weights'] if w['id'] == 'E0')['initial_location'] = 'dram'
        compiled = cache_templates(accesses=((0,),), devices=('cpu', 'gpu'),
                                   layer_document=document)[0]
        scenario_document = {**compiled.scenario_document,
            'mapspace': dict(compiled.scenario_document['mapspace'])}
        scenario_document['mapspace']['allowed_devices'] = {
            task.operation_id: ['cpu', 'gpu'] if task.kind == 'expert_ffn' else ['cpu']
            for task in compiled.tasks}
        scenario_document['costs'] = {**compiled.scenario_document['costs'],
            'compute': {task.operation_id: {device: cost for device, cost in
                compiled.scenario_document['costs']['compute'][task.operation_id].items()
                if device in scenario_document['mapspace']['allowed_devices'][task.operation_id]}
                for task in compiled.tasks}}
        compiled = dataclasses.replace(compiled, scenario_document=scenario_document,
            scenario=build_scenario(compiled.workload_document, scenario_document))
        result = search_schedule_episode((compiled,), policy=self.policy(),
            max_expanded_states=1000, max_generated_states=2000)
        self.assertEqual(result.status, 'optimal_within_declared_space')
        self.assertFalse(result.fixed_placements)
        placements = dict(result.best.invocations[0].placements)
        expert = next(task for task in compiled.tasks if task.kind == 'expert_ffn')
        self.assertEqual(placements[expert.operation_id], 'gpu')
        self.assertLess(result.best.total_makespan_ns, 10_000_000)

    def test_expanded_budget_without_goal_returns_unknown(self):
        result = self.search(overlap_templates(), expanded=1)
        self.assertEqual(result.status, 'unknown')
        self.assertIsNone(result.best)
        self.assertFalse(result.optimality_proven)
        self.assertFalse(result.enumeration_complete)
        self.assertEqual(result.termination_reason, 'max_expanded_states')
        self.assertEqual(result.expanded_states, 1)

    def test_generated_budget_without_goal_returns_unknown(self):
        result = self.search(overlap_templates(), generated=1)
        self.assertEqual(result.status, 'unknown')
        self.assertIsNone(result.best)
        self.assertFalse(result.enumeration_complete)
        self.assertEqual(result.termination_reason, 'max_generated_states')
        self.assertEqual(result.generated_states, 1)

    def test_validated_incumbent_survives_budget_without_gaining_optimality(self):
        templates = overlap_templates()
        policy = self.policy()
        seed = greedy_sequence(templates, policy)
        result = self.search(templates, policy=policy, expanded=1, generated=1, incumbent=seed)
        self.assertEqual(result.status, 'feasible')
        self.assertEqual(result.best.total_makespan_ns, 43_000_000)
        self.assertTrue(result.incumbent_seeded)
        self.assertFalse(result.optimality_proven)
        self.assertFalse(result.enumeration_complete)

    def test_generated_goal_is_feasible_until_its_priority_queue_pop(self):
        templates = overlap_templates(calls=1)
        full = self.search(templates)
        result = self.search(templates, expanded=full.expanded_states - 1)
        self.assertIsNotNone(result.best)
        self.assertTrue(result.incumbent_generated)
        self.assertEqual(result.status, 'feasible')
        self.assertFalse(result.optimality_proven)
        self.assertFalse(result.enumeration_complete)
        self.assertEqual(result.best.total_makespan_ns, 23_000_000)

    def test_zero_quota_exhaustion_proves_only_declared_space_infeasibility(self):
        result = self.search(overlap_templates(calls=1), policy=self.policy(dram=0))
        self.assertEqual(result.status, 'infeasible_within_declared_space')
        self.assertTrue(result.enumeration_complete)
        self.assertIsNone(result.best)
        self.assertFalse(result.optimality_proven)
        self.assertEqual(result.termination_reason, 'exhausted_declared_space')

    def test_later_initial_capacity_overflow_is_an_explicit_boundary_refusal(self):
        first, second = overlap_templates()
        second = dataclasses.replace(second, invocation=dataclasses.replace(second.invocation,
            hardware=dataclasses.replace(second.invocation.hardware, dram_capacity_bytes=271)))
        result = self.search((first, second))
        self.assertEqual(result.status, 'infeasible_within_declared_space')
        self.assertIsNone(result.best)
        self.assertGreater(result.rejected_boundaries, 0)
        self.assertTrue(result.boundary_refusals)
        self.assertTrue(all(item['code'] == 'initial_capacity_exceeded'
                            for item in result.boundary_refusals))

    def test_refused_fast_gpu_boundary_does_not_abort_slower_feasible_cpu_branch(self):
        document = cache_layer_document()
        next(w for w in document['weights'] if w['id'] == 'E0')['initial_location'] = 'dram'
        first, second = cache_templates(accesses=((0,), (0,)), devices=('cpu', 'gpu'),
                                        layer_document=document)
        declared = {**first.scenario_document, 'mapspace': dict(first.scenario_document['mapspace'])}
        declared['mapspace']['allowed_devices'] = {
            task.operation_id: ['cpu', 'gpu'] if task.kind == 'expert_ffn' else ['cpu']
            for task in first.tasks}
        declared['costs'] = {**first.scenario_document['costs'],
            'compute': {task.operation_id: {device: cost for device, cost in
                first.scenario_document['costs']['compute'][task.operation_id].items()
                if device in declared['mapspace']['allowed_devices'][task.operation_id]}
                for task in first.tasks}}
        first = dataclasses.replace(first, scenario_document=declared,
                                    scenario=build_scenario(first.workload_document, declared))
        invocation = dataclasses.replace(second.invocation, devices=('cpu',),
            hardware=dataclasses.replace(second.invocation.hardware, vram_capacity_bytes=32))
        second = compile_invocation(second.layer, invocation, second.cost_table)
        # GPU is faster for call zero but leaves a 64-byte resident weight;
        # the next 32-byte VRAM initial state rejects that branch. CPU remains
        # feasible and must still be explored as part of the same episode.
        result = search_schedule_episode((first, second), policy=self.policy(),
            max_expanded_states=1000, max_generated_states=2000)
        self.assertEqual(result.status, 'optimal_within_declared_space')
        self.assertEqual(result.best.total_makespan_ns, 20_000_000)
        self.assertGreater(result.rejected_boundaries, 0)
        self.assertEqual(set(dict(result.best.invocations[0].placements).values()), {'cpu'})

    def test_commuting_zero_time_actions_are_deduplicated_deterministically(self):
        templates = overlap_templates(calls=1)
        first = self.search(templates)
        second = self.search(templates)
        self.assertGreater(first.duplicate_states, 0)
        self.assertEqual(first.expanded_states, second.expanded_states)
        self.assertEqual(first.generated_states, second.generated_states)
        self.assertEqual(first.duplicate_states, second.duplicate_states)
        self.assertEqual(first.best.invocations[0].plan.actions,
                         second.best.invocations[0].plan.actions)

    def test_exported_best_plans_verify_fingerprints_and_replay_exactly(self):
        result = self.search(overlap_templates())
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for index, item in enumerate(result.best.invocations):
                scenario = item.compiled.scenario
                document = dict(schema_version=scenario.schema_version, scenario_id=scenario.id,
                    workload_fingerprint=workload_fingerprint(scenario.workload),
                    scenario_fingerprint=scenario_fingerprint(scenario),
                    actions=[action_to_dict(action, m1=True) for action in item.plan.actions])
                path = folder / f'call-{index}.json'
                path.write_text(json.dumps(document) + '\n', encoding='utf-8', newline='\n')
                loaded = load_mapping(path, scenario)
                self.assertTrue(loaded.fingerprint_verified)
                replay = evaluate_mapping(scenario, loaded.actions)
                self.assertEqual(replay.status, 'valid', replay.reason)
                self.assertEqual(replay.makespan_ns, item.result.makespan_ns)
                self.assertEqual(replay.ssd_read_bytes, item.result.evaluation.ssd_read_bytes)
                self.assertEqual(replay.peak_dram_bytes, item.result.evaluation.peak_dram_bytes)

    def test_incumbent_with_forged_claimed_latency_is_rejected(self):
        templates = overlap_templates()
        policy = self.policy()
        seed = greedy_sequence(templates, policy)
        invocation = seed.invocations[0]
        evaluation = dataclasses.replace(invocation.result.evaluation, makespan_ns=1)
        altered = dataclasses.replace(invocation, result=dataclasses.replace(
            invocation.result, evaluation=evaluation))
        seed = dataclasses.replace(seed, invocations=(altered, *seed.invocations[1:]))
        with self.assertRaises(InvalidInput):
            self.search(templates, policy=policy, incumbent=seed)

    def test_incumbent_scenario_flags_must_match_the_declared_space(self):
        templates = overlap_templates(calls=1)
        seed = greedy_sequence(templates, self.policy())
        # CPU-only actions remain behaviorally legal under this flag change.
        # The seed still belongs to a different scenario fingerprint.
        disabled = with_overlap_flags(templates[0], cpu_gpu=False)
        with self.assertRaises(InvalidInput):
            self.search((disabled,), incumbent=seed)

    def test_copy_compute_flag_survives_recompilation_at_call_boundary(self):
        first = overlap_call()
        state = state_in_from_state_out(run_invocation(first).state_out)
        second = overlap_call(index=1, state=state, routes=[[0, 1, 3]])
        templates = tuple(with_overlap_flags(call, copy_compute=False) for call in (first, second))
        result = self.search(templates)
        self.assertEqual(result.status, 'optimal_within_declared_space')
        self.assertEqual(result.best.total_makespan_ns, 52_000_000)
        self.assertEqual([item.result.makespan_ns for item in result.best.invocations],
                         [29_000_000, 23_000_000])
        self.assertFalse(result.best.invocations[1].compiled.scenario.mapspace.allow_copy_compute_overlap)

    def test_state_budgets_are_explicit_positive_integers(self):
        for name in ('max_expanded_states', 'max_generated_states'):
            for value in (0, -1, True, 1.0, '100'):
                with self.subTest(name=name, value=value), self.assertRaises(InvalidInput):
                    kwargs = dict(max_expanded_states=100, max_generated_states=100)
                    kwargs[name] = value
                    search_schedule_episode(overlap_templates(calls=1), policy=self.policy(), **kwargs)

    def test_physical_measured_contract_is_not_reinterpreted_as_synthetic_moe(self):
        compiled = overlap_templates(calls=1)[0]
        compiled = dataclasses.replace(compiled,
            layer=dataclasses.replace(compiled.layer, format_version='moe_measurements_v2'))
        with self.assertRaises(Unsupported):
            self.search((compiled,))


if __name__ == '__main__':
    unittest.main()
