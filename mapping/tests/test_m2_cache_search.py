"""Bounded offline cache enumeration over an explicit synthetic episode."""
from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from llm_infer_model.m2.expand import compile_invocation
from llm_infer_model.tensor.engine import evaluate_mapping, load_mapping
from llm_infer_model.tensor.spec import InvalidInput, scenario_fingerprint, workload_fingerprint
from tensor_mapping.artifacts import action_to_dict
from tensor_mapping.m2_cache import CachePolicy
from tensor_mapping.m2_cache_search import run_cache_sequence, search_cache_sequence

from .test_m2_cache import cache_templates, expert_reads
from .test_m2_moe_search import with_budgets


class CacheSearchTests(unittest.TestCase):
    def policy(self, *, dram=128, vram=None):
        return CachePolicy(kind='branch',
                           expert_cache_capacity_bytes={'dram': dram, 'vram': vram})

    def test_complete_tiny_space_finds_four_load_plan(self):
        result = search_cache_sequence(cache_templates(), policy=self.policy(),
                                       max_candidates=100, max_expanded_states=100)
        self.assertTrue(result.enumeration_complete)
        self.assertEqual(result.status, 'optimal_within_declared_space')
        self.assertEqual(result.termination_reason, 'exhausted_declared_space')
        self.assertEqual(result.best.total_makespan_ns, 62_000_000)
        self.assertEqual(len(expert_reads(result.best)), 4)
        self.assertEqual(result.declared_placement_count, 1)
        self.assertTrue(result.ranking)
        self.assertEqual([c.total_makespan_ns for c in result.ranking],
                         sorted(c.total_makespan_ns for c in result.ranking))

    def test_candidate_budget_produces_feasible_without_optimality(self):
        result = search_cache_sequence(cache_templates(), policy=self.policy(),
                                       max_candidates=1, max_expanded_states=100)
        self.assertEqual(result.status, 'feasible')
        self.assertEqual(len(result.candidates), 1)
        self.assertFalse(result.enumeration_complete)
        self.assertEqual(result.termination_reason, 'max_candidates')
        self.assertIsNotNone(result.best)

    def test_prefix_budget_before_first_leaf_is_unknown(self):
        result = search_cache_sequence(cache_templates(), policy=self.policy(),
                                       max_candidates=100, max_expanded_states=1)
        self.assertEqual(result.status, 'unknown')
        self.assertEqual(result.candidates, ())
        self.assertEqual(result.ranking, ())
        self.assertEqual(result.expanded_states, 1)
        self.assertFalse(result.enumeration_complete)
        self.assertIsNone(result.best)
        self.assertEqual(result.termination_reason, 'max_expanded_states')

    def test_exact_leaf_and_state_budgets_distinguish_exhaustion(self):
        full = search_cache_sequence(cache_templates(), policy=self.policy(),
                                     max_candidates=100, max_expanded_states=100)
        exact = search_cache_sequence(cache_templates(), policy=self.policy(),
            max_candidates=len(full.candidates), max_expanded_states=full.expanded_states)
        self.assertTrue(exact.enumeration_complete)
        self.assertEqual(exact.status, 'optimal_within_declared_space')
        short = search_cache_sequence(cache_templates(), policy=self.policy(),
            max_candidates=len(full.candidates) - 1, max_expanded_states=100)
        self.assertFalse(short.enumeration_complete)
        self.assertEqual(short.status, 'feasible')
        self.assertEqual(short.termination_reason, 'max_candidates')

    def test_forced_single_slot_evictions_do_not_create_false_choice_branches(self):
        result = search_cache_sequence(cache_templates(accesses=((0,), (1,), (0,))),
            policy=self.policy(dram=64), max_candidates=1, max_expanded_states=1)
        self.assertTrue(result.enumeration_complete)
        self.assertEqual(result.status, 'optimal_within_declared_space')
        self.assertEqual(result.expanded_states, 1)
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.best.decisions, ())
        self.assertEqual(result.best.eviction_count, 2)
        self.assertEqual(result.best.total_makespan_ns, 39_000_000)

    def test_unvisited_infeasible_devices_cannot_prove_episode_infeasible(self):
        base = cache_templates(accesses=((0,),), devices=('cpu', 'gpu'))[0]
        limited = with_budgets(base, vram=1, cpu_scratch=1 << 30)
        partial = search_cache_sequence((limited,), policy=self.policy(),
            max_candidates=1, max_expanded_states=100)
        self.assertEqual(partial.status, 'unknown')
        self.assertIsNone(partial.best)
        self.assertFalse(partial.enumeration_complete)
        self.assertEqual(partial.candidates[0].total_makespan_ns, None)
        full = search_cache_sequence((limited,), policy=self.policy(),
            max_candidates=100, max_expanded_states=100)
        self.assertEqual(full.status, 'infeasible_within_declared_space')
        self.assertTrue(full.enumeration_complete)
        self.assertIsNone(full.best)

    def test_fixed_placement_controls_search_size_independently_of_devices(self):
        templates = cache_templates(devices=('cpu', 'gpu'))
        placements = tuple({task.operation_id: 'cpu' for task in template.tasks}
                           for template in templates)
        result = search_cache_sequence(templates, placements=placements,
            policy=self.policy(), max_candidates=100, max_expanded_states=100)
        self.assertTrue(result.fixed_placements)
        self.assertEqual(result.declared_placement_count, 1)
        self.assertEqual(result.best.total_makespan_ns, 62_000_000)
        for invocation in result.best.invocations:
            self.assertEqual(set(dict(invocation.placements).values()), {'cpu'})

    def test_enumerated_device_product_is_budgeted_without_materializing_it(self):
        templates = cache_templates(devices=('cpu', 'gpu'))
        result = search_cache_sequence(templates, policy=self.policy(),
            max_candidates=1, max_expanded_states=100)
        self.assertEqual(result.declared_placement_count,
                         2 ** sum(len(c.tasks) for c in templates))
        self.assertFalse(result.fixed_placements)
        self.assertFalse(result.enumeration_complete)
        self.assertEqual(result.placements_visited, 1)
        self.assertEqual(len(result.candidates), 1)

    def test_search_and_decision_order_are_reproducible(self):
        kwargs = dict(policy=self.policy(), max_candidates=100, max_expanded_states=100)
        first = search_cache_sequence(cache_templates(), **kwargs)
        second = search_cache_sequence(cache_templates(), **kwargs)
        self.assertEqual([(c.decisions, c.total_makespan_ns) for c in first.candidates],
                         [(c.decisions, c.total_makespan_ns) for c in second.candidates])
        self.assertEqual(first.best.index, second.best.index)
        self.assertEqual(first.expanded_states, second.expanded_states)

    def test_best_mapping_replays_with_bound_state_fingerprints(self):
        candidate = search_cache_sequence(cache_templates(), policy=self.policy(),
            max_candidates=100, max_expanded_states=100).best
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for index, invocation in enumerate(candidate.invocations):
                compiled = invocation.compiled
                mapping = dict(schema_version=compiled.scenario.schema_version,
                    scenario_id=compiled.scenario.id,
                    workload_fingerprint=workload_fingerprint(compiled.scenario.workload),
                    scenario_fingerprint=scenario_fingerprint(compiled.scenario),
                    actions=[action_to_dict(a, m1=True) for a in invocation.plan.actions])
                path = folder / f'mapping-{index}.json'
                path.write_text(json.dumps(mapping) + '\n', encoding='utf-8', newline='\n')
                loaded = load_mapping(path, compiled.scenario)
                self.assertTrue(loaded.fingerprint_verified)
                replay = evaluate_mapping(compiled.scenario, loaded.actions)
                self.assertEqual(replay.status, 'valid', replay.reason)
                self.assertEqual(replay.makespan_ns, invocation.result.makespan_ns)
                self.assertEqual(replay.ssd_read_bytes, invocation.result.evaluation.ssd_read_bytes)
                self.assertEqual(replay.peak_dram_bytes, invocation.result.evaluation.peak_dram_bytes)
                self.assertNotIn(b'\r\n', path.read_bytes())

    def test_positive_integer_search_budgets_are_strict(self):
        for field in ('max_candidates', 'max_expanded_states'):
            for value in (0, -1, True, 1.0, '1'):
                with self.subTest(field=field, value=value), self.assertRaises(InvalidInput):
                    search_cache_sequence(cache_templates(), policy=self.policy(), **{field: value})

    def test_baseline_policy_is_not_silently_reinterpreted_as_branch_search(self):
        policy = CachePolicy(kind='lru', expert_cache_capacity_bytes={'dram': 128, 'vram': None})
        with self.assertRaises(InvalidInput):
            search_cache_sequence(cache_templates(), policy=policy)

    def test_completed_path_cannot_carry_unused_choices(self):
        templates = cache_templates(accesses=((0,),))
        for decisions in ((0,), (True,), (-1,), (1.0,)):
            with self.subTest(decisions=decisions), self.assertRaises(InvalidInput):
                run_cache_sequence(templates, policy=self.policy(), decisions=decisions)

    def test_encountered_eviction_choice_rejects_out_of_range_index(self):
        with self.assertRaises(InvalidInput):
            run_cache_sequence(cache_templates(), policy=self.policy(), decisions=(99,))

    def test_later_request_positions_are_validated_against_actual_prior_state(self):
        templates = cache_templates()
        template = templates[1]
        request = dataclasses.replace(template.invocation.requests[0], position_start=100)
        template = dataclasses.replace(template, invocation=dataclasses.replace(
            template.invocation, requests=(request,)))
        with self.assertRaises(InvalidInput):
            run_cache_sequence((templates[0], template))

    def test_sequence_requires_exact_same_immutable_layer_definition(self):
        templates = cache_templates()
        different = dataclasses.replace(templates[1], layer=dataclasses.replace(
            templates[1].layer, id='different-layer'))
        with self.assertRaises(InvalidInput):
            run_cache_sequence((templates[0], different))
        with self.assertRaises(InvalidInput):
            search_cache_sequence(())


if __name__ == '__main__':
    unittest.main()
