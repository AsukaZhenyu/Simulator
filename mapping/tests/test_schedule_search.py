"""One shared search handles arbitrary tensor graphs, Dense and MoE adapters."""
from __future__ import annotations

import dataclasses
import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path

from llm_infer_model.m2.spec import state_in_from_state_out, weight_id
from llm_infer_model.tensor.engine import ACTION_COMPUTE, ACTION_COPY, ACTION_READ_SSD, evaluate_mapping
from llm_infer_model.tensor.spec import InvalidInput, Unsupported, build_scenario
from tensor_mapping.schedule import SchedulePolicy
from tensor_mapping.schedule_search import (
    ScheduleSequenceCandidate, TensorScenarioAdapter, run_schedule_sequence,
    search_schedule, search_schedule_episode,
)

from .test_m2_hybrid import hybrid_compiled, placements_for
from .test_m2_overlap import overlap_templates
from .test_schedule import generic_dag, generic_facts, intersects


def dense_templates():
    from llm_infer_model.m2.runner import run_invocation
    first = hybrid_compiled()
    baseline = run_invocation(first, placements=placements_for(first, ('cpu', 'cpu', 'cpu')))
    second = hybrid_compiled(state=state_in_from_state_out(baseline.state_out), phase='decode')
    return first, second


class GenericSearchTests(unittest.TestCase):
    def policy(self, *, dram=32, vram=16, depth=1, overlap=True):
        return SchedulePolicy(prefetch_depth=depth, overlap=overlap,
                              managed_capacity_bytes={'dram': dram, 'vram': vram})

    def solve_dag(self, *, policy=None, expanded=1000, generated=2000, incumbent=None):
        return search_schedule(generic_dag(), facts=generic_facts(),
            policy=self.policy() if policy is None else policy,
            max_expanded_states=expanded, max_generated_states=generated, incumbent=incumbent)

    def test_arbitrary_graph_needs_no_llm_adapter(self):
        result = self.solve_dag()
        self.assertEqual(result.status, 'optimal_within_declared_space')
        self.assertTrue(result.optimality_proven)
        self.assertTrue(result.complete)
        self.assertFalse(result.enumeration_complete)
        # Reading GPU operand b first lets its PCIe copy finish before CPU
        # operand a arrives. Both independent computations then start at 6 ms;
        # b's result returns at 11.1 ms and the join finishes at 12.1 ms.
        self.assertEqual(result.best.total_makespan_ns, 12_100_000)
        self.assertIsInstance(result.best, ScheduleSequenceCandidate)
        item = result.best.invocations[0]
        self.assertEqual(item.compiled.id, 'opaque-fork-join')
        replay = evaluate_mapping(item.compiled, item.plan.actions)
        self.assertEqual(replay, item.result.evaluation)

    def test_direct_episode_api_and_convenience_share_the_same_result_type(self):
        scenario = generic_dag()
        result = search_schedule_episode((scenario,), adapter=TensorScenarioAdapter(generic_facts()),
            policy=self.policy(), max_expanded_states=1000, max_generated_states=2000)
        convenience = self.solve_dag()
        self.assertEqual(type(result), type(convenience))
        self.assertEqual(result.best.total_makespan_ns, convenience.best.total_makespan_ns)
        self.assertEqual(result.best.invocations[0].plan.actions,
                         convenience.best.invocations[0].plan.actions)

    def test_generic_imports_do_not_load_model_family_adapters(self):
        mapping = Path(__file__).resolve().parents[1]
        modeling = mapping.parent / 'modeling'
        script = ('import sys; sys.path[:0]=sys.argv[1:]; '
                  'import tensor_mapping.schedule, tensor_mapping.schedule_search; '
                  'assert not any(n.startswith("llm_infer_model.m2") for n in sys.modules), '
                  '[n for n in sys.modules if n.startswith("llm_infer_model.m2")]')
        completed = subprocess.run([sys.executable, '-I', '-c', script, str(mapping), str(modeling)],
                                   cwd=mapping.parent, capture_output=True, text=True, timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_generated_budget_does_not_claim_infeasibility(self):
        result = self.solve_dag(generated=1)
        self.assertEqual(result.status, 'unknown')
        self.assertIsNone(result.best)
        self.assertFalse(result.complete)
        self.assertEqual(result.termination_reason, 'max_generated_states')

    def test_verified_generic_incumbent_is_only_feasible_after_truncation(self):
        scenario = generic_dag()
        adapter = TensorScenarioAdapter(generic_facts())
        seed = run_schedule_sequence((scenario,), adapter=adapter, policy=self.policy(), max_actions=100)
        self.assertTrue(seed.is_valid)
        result = self.solve_dag(expanded=1, generated=1, incumbent=seed)
        self.assertEqual(result.status, 'feasible')
        self.assertTrue(result.incumbent_seeded)
        self.assertFalse(result.optimality_proven)
        self.assertFalse(result.complete)
        self.assertEqual(result.best.total_makespan_ns, 12_200_000)

    def test_zero_copy_quota_exhausts_only_the_declared_space(self):
        result = self.solve_dag(policy=self.policy(dram=0))
        self.assertEqual(result.status, 'infeasible_within_declared_space')
        self.assertTrue(result.enumeration_complete)
        self.assertTrue(result.complete)
        self.assertIsNone(result.best)

    def test_repeated_runs_have_deterministic_state_dominance_and_actions(self):
        first = self.solve_dag()
        second = self.solve_dag()
        self.assertGreater(first.duplicate_states, 0)
        self.assertEqual(first.generated_states, second.generated_states)
        self.assertEqual(first.expanded_states, second.expanded_states)
        self.assertEqual(first.best.invocations[0].plan.actions,
                         second.best.invocations[0].plan.actions)

    def test_adapter_final_state_verification_checks_ledger_not_only_time(self):
        from llm_infer_model.tensor.engine import initial_state, transition
        from tensor_mapping.schedule_search import verify_replayed_state
        result = self.solve_dag()
        item = result.best.invocations[0]
        state = initial_state(item.compiled)
        for action in item.plan.actions:
            state = transition(item.compiled, state, action)
        verify_replayed_state(item.result.evaluation, state)
        forged = dataclasses.replace(state, dram_status=tuple(
            (tensor, 'ABSENT') if tensor == 'out' else (tensor, status)
            for tensor, status in state.dram_status))
        with self.assertRaises(InvalidInput):
            verify_replayed_state(item.result.evaluation, forged)


class SharedM2AdapterTests(unittest.TestCase):
    def setUp(self):
        from tensor_mapping.m2_schedule import M2ScheduleAdapter
        self.Adapter = M2ScheduleAdapter

    def policy(self, *, dram=None, vram=None, depth=1):
        return SchedulePolicy(prefetch_depth=depth, overlap=True,
                              managed_capacity_bytes={'dram': dram, 'vram': vram})

    def test_dense_and_moe_use_the_same_generic_search_and_candidate_classes(self):
        dense = dense_templates()
        dense_placements = tuple(placements_for(item) for item in dense)
        dense_result = search_schedule_episode(dense, adapter=self.Adapter(), policy=self.policy(),
            placements=dense_placements, max_expanded_states=1000, max_generated_states=2000)
        moe = overlap_templates()
        moe_placements = tuple({task.operation_id: 'cpu' for task in item.tasks} for item in moe)
        moe_result = search_schedule_episode(moe, adapter=self.Adapter(),
            policy=self.policy(dram=192, vram=192, depth=2), placements=moe_placements,
            max_expanded_states=1000, max_generated_states=2000)
        for result in (dense_result, moe_result):
            self.assertEqual(result.status, 'optimal_within_declared_space')
            self.assertIsInstance(result.best, ScheduleSequenceCandidate)
        self.assertEqual(type(dense_result), type(moe_result))
        self.assertEqual(moe_result.best.total_makespan_ns, 43_000_000)

    def test_dense_prefill_decode_carries_actual_kv_and_reuses_weight_copy(self):
        templates = dense_templates()
        placements = tuple(placements_for(item) for item in templates)
        result = search_schedule_episode(templates, adapter=self.Adapter(), policy=self.policy(vram=1024),
            placements=placements, max_expanded_states=1000, max_generated_states=2000)
        self.assertIsNotNone(result.best)
        first, second = result.best.invocations
        self.assertEqual(second.compiled.state_in, state_in_from_state_out(first.result.state_out))
        self.assertEqual(second.result.state_out.requests[0].next_position, 4)
        self.assertEqual(second.result.state_out.requests[0].version, 3)
        name = weight_id(first.compiled.layer.id, 'W_output')
        first_copies = [action for action in first.plan.actions if action.kind == ACTION_COPY
                        and action.target_id == name and action.dst == 'vram']
        second_copies = [action for action in second.plan.actions if action.kind == ACTION_COPY
                         and action.target_id == name and action.dst == 'vram']
        self.assertEqual(len(first_copies), 1)
        self.assertEqual(second_copies, [])
        self.assertIn('vram', second.compiled.scenario.workload.tensor_by_id[name].initial_locations)
        self.assertEqual(second.boundary_state_in, first.boundary_state_out)

    def test_dense_overlap_is_verified_in_events_and_legacy_costs_remain_synthetic(self):
        templates = dense_templates()
        result = search_schedule_episode(templates, adapter=self.Adapter(), policy=self.policy(),
            placements=tuple(placements_for(item) for item in templates),
            max_expanded_states=1000, max_generated_states=2000)
        self.assertIsNotNone(result.best)
        first = result.best.invocations[0]
        compute = [event for event in first.result.evaluation.events if event.kind == ACTION_COMPUTE]
        copies = [event for event in first.result.evaluation.events if event.kind == ACTION_COPY]
        self.assertTrue(any(intersects(c, t) for c in compute for t in copies))
        self.assertIn('synthetic', first.compiled.costs_source)
        self.assertLess(first.result.makespan_ns, 3100)

    def test_dense_all_gpu_baseline_can_evict_non_source_weights_and_still_finish(self):
        templates = dense_templates()
        selected = tuple({task.operation_id: 'gpu' for task in item.tasks} for item in templates)
        candidate = run_schedule_sequence(templates, adapter=self.Adapter(),
            policy=self.policy(vram=6208), placements=selected, max_actions=1000)
        self.assertTrue(candidate.is_valid, candidate.reason)
        self.assertGreater(candidate.eviction_count, 0)
        for item in candidate.invocations:
            for tensor in item.compiled.scenario.workload.tensors:
                if tensor.role == 'weight':
                    self.assertEqual(dict(item.result.evaluation.states[-1].dram_status)[tensor.id], 'READY')
            replay = evaluate_mapping(item.compiled.scenario, item.plan.actions)
            self.assertEqual(replay, item.result.evaluation)

    def test_dense_joint_device_search_selects_gpu_attention_without_a_moe_algorithm(self):
        compiled = dense_templates()[0]
        declared = copy.deepcopy(compiled.scenario_document)
        for task in compiled.tasks:
            if task.kind != 'attention':
                declared['mapspace']['allowed_devices'][task.operation_id] = ['cpu']
                del declared['costs']['compute'][task.operation_id]['gpu']
        compiled = dataclasses.replace(compiled, scenario_document=declared,
            scenario=build_scenario(compiled.workload_document, declared))
        result = search_schedule_episode((compiled,), adapter=self.Adapter(),
            policy=self.policy(vram=1024), max_expanded_states=1000, max_generated_states=2000)
        self.assertEqual(result.status, 'optimal_within_declared_space')
        self.assertFalse(result.fixed_placements)
        self.assertEqual([device for operation, device in result.best.invocations[0].placements],
                         ['cpu', 'gpu', 'cpu'])
        self.assertLess(result.best.total_makespan_ns, 3100)

    def test_dense_boundary_does_not_invent_native_weight_versions(self):
        templates = dense_templates()
        candidate = run_schedule_sequence(templates, adapter=self.Adapter(), policy=self.policy(),
            placements=tuple(placements_for(item) for item in templates), max_actions=1000)
        self.assertTrue(candidate.is_valid)
        for item in candidate.invocations:
            self.assertEqual(item.result.state_out.weight_copies, ())
            self.assertEqual(len(item.boundary_state_out['weights']), len(item.compiled.layer.weights))
            self.assertTrue(all(entry['version'] is None
                                and entry['version_semantics'] == 'immutable_layer_definition'
                                for entry in item.boundary_state_out['weights']))

    def assert_single_device_dense_chain(self, first, second, device, source):
        templates = (first, second)
        candidate = run_schedule_sequence(templates, adapter=self.Adapter(), policy=self.policy(),
            placements=tuple({task.operation_id: device for task in item.tasks} for item in templates),
            max_actions=1000)
        self.assertTrue(candidate.is_valid, candidate.reason)
        prefill, decode = candidate.invocations
        self.assertEqual(decode.compiled.state_in, state_in_from_state_out(prefill.result.state_out))
        self.assertEqual(decode.result.state_out.requests[0].next_position, 4)
        self.assertEqual(decode.boundary_state_in, prefill.boundary_state_out)
        for item in candidate.invocations:
            self.assertEqual(item.result.status, 'valid')
            self.assertEqual(evaluate_mapping(item.compiled.scenario, item.plan.actions),
                             item.result.evaluation)
            self.assertEqual({entry.provenance for entry in item.compiled.cost_table.entries.values()},
                             {'synthetic_layer'})
            final = item.result.evaluation.states[-1]
            ready = {domain: dict(final.statuses(domain))
                     for domain in ('dram', 'vram', 'ssd')}
            for entry in item.boundary_state_out['weights']:
                self.assertEqual(entry['authoritative_source'], source)
                self.assertIn(source, entry['locations'])
                self.assertEqual(set(entry['locations']),
                    {domain for domain, ledger in ready.items() if ledger[entry['tensor_id']] == 'READY'})

    def test_full_cpu_03_adapter_preserves_declared_source_and_actual_decode_state(self):
        from llm_infer_model.m2.expand import compile_invocation
        from llm_infer_model.m2.runner import run_invocation
        from llm_infer_model.m2.spec import parse_layer_document
        from .test_m2_full import full_layer_document, full_invocation, full_cost_table
        layer = parse_layer_document(full_layer_document())
        invocation = full_invocation(layer)
        first = compile_invocation(layer, invocation, full_cost_table(layer, invocation))
        history = state_in_from_state_out(run_invocation(first).state_out)
        decode = full_invocation(layer, phase='decode', state=history)
        decode = dataclasses.replace(decode, id=invocation.id + '-decode')
        second = compile_invocation(layer, decode, full_cost_table(layer, decode))
        self.assertEqual(first.layer.format_version, 'm2-layer-0.3')
        self.assert_single_device_dense_chain(first, second, 'cpu', 'dram')

    def test_full_gpu_04_adapter_preserves_declared_source_and_actual_decode_state(self):
        from llm_infer_model.m2.expand import compile_invocation
        from llm_infer_model.m2.runner import run_invocation
        from llm_infer_model.m2.spec import parse_layer_document
        from .test_m2_cuda import cuda_layer_document, cuda_invocation, cuda_cost_table
        layer = parse_layer_document(cuda_layer_document())
        invocation = cuda_invocation(layer)
        first = compile_invocation(layer, invocation, cuda_cost_table(layer, invocation))
        history = state_in_from_state_out(run_invocation(first).state_out)
        decode = cuda_invocation(layer, phase='decode', state=history)
        decode = dataclasses.replace(decode, id=invocation.id + '-decode')
        second = compile_invocation(layer, decode, cuda_cost_table(layer, decode))
        self.assertEqual(first.layer.format_version, 'm2-layer-0.4')
        self.assert_single_device_dense_chain(first, second, 'gpu', 'vram')

    def test_reusable_boundary_corrupted_inventory_or_source_cannot_bind_next_call(self):
        templates = dense_templates()
        adapter = self.Adapter()
        candidate = run_schedule_sequence(templates, adapter=adapter, policy=self.policy(),
            placements=tuple(placements_for(item) for item in templates), max_actions=1000)
        first = candidate.invocations[0]
        for corruption in ('layer_hash', 'missing_weight', 'source'):
            with self.subTest(corruption=corruption):
                claim = copy.deepcopy(first.boundary_state_out)
                if corruption == 'layer_hash':
                    claim['layer_sha256'] = '0' * 64
                elif corruption == 'missing_weight':
                    claim['weights'].pop()
                else:
                    claim['weights'][0]['authoritative_source'] = 'ssd'
                with self.assertRaises(InvalidInput):
                    adapter.bind(templates[1], dataclasses.replace(first, boundary_state_out=claim))

    def test_dense_forged_cached_location_is_rejected_against_final_ledger(self):
        templates = dense_templates()
        adapter = self.Adapter()
        candidate = run_schedule_sequence(templates, adapter=adapter, policy=self.policy(),
            placements=tuple(placements_for(item) for item in templates), max_actions=1000)
        first = candidate.invocations[0]
        claim = copy.deepcopy(first.boundary_state_out)
        entry = next(entry for entry in claim['weights'] if entry['id'] == 'W_q')
        self.assertEqual(entry['locations'], ['dram'])
        entry['locations'].append('vram')
        forged = dataclasses.replace(first, boundary_state_out=claim)
        with self.assertRaises(InvalidInput):
            adapter.bind(templates[1], forged)

    def test_future_declared_scenario_override_is_refused_instead_of_discarded(self):
        first, second = dense_templates()
        declared = copy.deepcopy(second.scenario_document)
        attention = next(task for task in second.tasks if task.kind == 'attention')
        declared['costs']['compute'][attention.operation_id]['cpu']['duration_ns'] = 9999
        second = dataclasses.replace(second, scenario_document=declared,
            scenario=build_scenario(second.workload_document, declared))
        with self.assertRaises(Unsupported):
            run_schedule_sequence((first, second), adapter=self.Adapter(), policy=self.policy(),
                placements=(placements_for(first), placements_for(second)), max_actions=1000)

    def test_moe_router_causality_still_comes_from_engine_under_generic_search(self):
        from tensor_mapping.schedule import legal_schedule_actions
        from llm_infer_model.tensor.engine import initial_state
        template = overlap_templates(calls=1)[0]
        adapter = self.Adapter()
        bound = adapter.initial(template)
        context = adapter.make_context(bound, self.policy(depth=20),
            {task.operation_id: 'cpu' for task in template.tasks})
        legal = legal_schedule_actions(context, initial_state(context.scenario))
        self.assertFalse(any(action.kind == ACTION_READ_SSD for action in legal))


if __name__ == '__main__':
    unittest.main()
