"""Both M2 families export exact generic schedule/state-boundary artifacts."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from llm_infer_model.tensor.engine import evaluate_mapping, load_mapping
from llm_infer_model.tensor.spec import InvalidInput, build_scenario
from tensor_mapping.m2_schedule_cli import episode_raw, episode_snapshot, execute_episode, main

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples' / 'm2'


def copied_episode(folder, family):
    """Copy immutable input assets; no mutation reaches historical fixtures."""
    episode = EXAMPLES / f'{family}-schedule.episode.json'
    raw = json.loads(episode.read_text(encoding='utf-8'))
    for ref in raw['invocation_files']:
        source = episode.parent / ref
        invocation = json.loads(source.read_text(encoding='utf-8'))
        shutil.copy2(source, folder / source.name)
        for key in ('layer_file', 'costs_file'):
            artifact = source.parent / invocation[key]
            shutil.copy2(artifact, folder / artifact.name)
    result = folder / episode.name
    shutil.copy2(episode, result)
    return result


class GenericM2CliTests(unittest.TestCase):
    def execute(self, episode, output, *, expanded=1000, generated=2000, actions=1000):
        return execute_episode(episode, output, max_expanded_states=expanded,
                               max_generated_states=generated, max_baseline_actions=actions)

    def test_dense_and_moe_export_identical_generic_scheduler_contract(self):
        for family in ('dense', 'moe'):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                episode = copied_episode(folder, family)
                output = folder / 'result'
                summary = self.execute(episode, output, expanded=2 if family == 'dense' else 1000)
                self.assertEqual(summary['shared_scheduler'], 'tensor_mapping.schedule')
                self.assertEqual(summary['shared_searcher'], 'tensor_mapping.schedule_search')
                self.assertEqual(summary['workload_family'], family)
                self.assertEqual(summary['reusable_tensor_scope'], 'all_layer_weights')
                self.assertFalse(summary['hardware_measurement_performed'])
                self.assertFalse(summary['performance_calibrated'])
                self.assertIsNotNone(summary['best'])
                if family == 'moe':
                    self.assertEqual(summary['best_total_makespan_ns'], 43_000_000)
                    self.assertTrue(summary['search_complete'])
                    self.assertFalse(summary['enumeration_complete'])
                else:
                    self.assertEqual(summary['status'], 'feasible')
                    self.assertFalse(summary['optimality_within_declared_space'])

    def test_every_saved_best_call_has_independent_fingerprint_verified_replay(self):
        for family in ('dense', 'moe'):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                episode = copied_episode(folder, family)
                output = folder / 'result'
                summary = self.execute(episode, output, expanded=2 if family == 'dense' else 1000)
                total = 0
                prior_boundary = None
                for index, declared in enumerate(summary['best']['calls']):
                    call = output / 'best' / f'call-{index:03d}'
                    read = lambda name: json.loads((call / name).read_text(encoding='utf-8'))
                    scenario = build_scenario(read('workload.json'), read('scenario.json'))
                    mapping = load_mapping(call / 'mapping.json', scenario)
                    self.assertTrue(mapping.fingerprint_verified)
                    replay = evaluate_mapping(scenario, mapping.actions)
                    self.assertEqual(replay.status, 'valid', replay.reason)
                    self.assertEqual(replay.makespan_ns, declared['makespan_ns'])
                    self.assertEqual(replay.peak_dram_bytes, declared['peak_dram_bytes'])
                    self.assertEqual(replay.h2d_bytes, declared['h2d_bytes'])
                    total += replay.makespan_ns
                    incoming, outgoing = read('boundary_state_in.json'), read('boundary_state_out.json')
                    self.assertEqual(incoming['format_version'], 'm2-reusable-boundary-0.1')
                    if prior_boundary is not None:
                        self.assertEqual(incoming, prior_boundary)
                    prior_boundary = outgoing
                    ready = replay.states[-1]
                    for entry in outgoing['weights']:
                        expected = [domain for domain in ('dram', 'vram', 'ssd')
                                    if dict(ready.statuses(domain)).get(entry['tensor_id']) == 'READY']
                        self.assertEqual(entry['locations'], expected)
                self.assertEqual(total, summary['best_total_makespan_ns'])
                for path in output.rglob('*.json'):
                    self.assertNotIn(b'\r\n', path.read_bytes())

    def test_source_hashes_pin_each_raw_input_including_the_route_trace(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            episode = copied_episode(folder, 'moe')
            summary = self.execute(episode, folder / 'result')
            sources = summary['source_inputs']
            self.assertEqual(sources['episode']['sha256'], hashlib.sha256(episode.read_bytes()).hexdigest())
            for call in sources['calls']:
                for role in ('invocation', 'layer', 'costs'):
                    self.assertEqual(call[role]['sha256'],
                        hashlib.sha256(Path(call[role]['path']).read_bytes()).hexdigest())
                invocation = json.loads(Path(call['invocation']['path']).read_text(encoding='utf-8'))
                canonical = json.dumps(invocation['route_trace'], sort_keys=True, separators=(',', ':'),
                                       ensure_ascii=False, allow_nan=False).encode('utf-8')
                self.assertEqual(call['inline_route_trace']['sha256'], hashlib.sha256(canonical).hexdigest())

    def test_schema_never_guesses_omitted_policy_or_state_binding_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            episode = copied_episode(folder, 'moe')
            accepted = json.loads(episode.read_text(encoding='utf-8'))
            self.assertEqual(episode_raw(episode), accepted)
            for key in accepted:
                raw = copy.deepcopy(accepted)
                del raw[key]
                episode.write_text(json.dumps(raw) + '\n', encoding='utf-8', newline='\n')
                with self.subTest(key=key), self.assertRaises(InvalidInput):
                    episode_raw(episode)
            raw = dict(accepted, extra_unmodeled='ignored?')
            episode.write_text(json.dumps(raw) + '\n', encoding='utf-8', newline='\n')
            with self.assertRaises(InvalidInput):
                episode_raw(episode)

    def test_nonempty_output_is_refused_before_overwriting_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            episode = copied_episode(folder, 'moe')
            output = folder / 'result'
            output.mkdir()
            sentinel = output / 'summary.json'
            sentinel.write_bytes(b'existing independent evidence\n')
            with self.assertRaises(InvalidInput):
                self.execute(episode, output)
            self.assertEqual(sentinel.read_bytes(), b'existing independent evidence\n')

    def test_source_change_during_search_creates_no_result_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            episode = copied_episode(folder, 'moe')
            source = episode_snapshot(episode)
            changed = copy.deepcopy(source)
            changed['episode']['sha256'] = '0' * 64
            output = folder / 'result'
            with mock.patch('tensor_mapping.m2_schedule_cli.episode_snapshot',
                            side_effect=[source, source, changed]):
                with self.assertRaises(InvalidInput):
                    self.execute(episode, output)
            self.assertFalse(output.exists())

    def test_budgets_reject_boolean_float_zero_and_negative(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            episode = copied_episode(folder, 'moe')
            for key in ('max_expanded_states', 'max_generated_states', 'max_baseline_actions'):
                for value in (True, 0, -1, 1.0):
                    budgets = dict(max_expanded_states=1000, max_generated_states=2000, max_baseline_actions=1000)
                    budgets[key] = value
                    with self.subTest(key=key, value=value), self.assertRaises(InvalidInput):
                        execute_episode(episode, folder / 'result', **budgets)

    def test_unknown_search_has_no_best_mapping_or_completed_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            episode = copied_episode(folder, 'moe')
            output = folder / 'result'
            summary = self.execute(episode, output, expanded=1, generated=1, actions=1)
            self.assertEqual(summary['status'], 'unknown')
            self.assertIsNone(summary['best'])
            self.assertIsNone(summary['best_total_makespan_ns'])
            self.assertFalse((output / 'best').exists())

    def test_main_missing_input_returns_refusal_without_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            output = folder / 'result'
            with contextlib.redirect_stderr(io.StringIO()):
                code = main([str(folder / 'missing.json'), '--out', str(output),
                    '--max-expanded-states', '100', '--max-generated-states', '100',
                    '--max-baseline-actions', '100'])
            self.assertEqual(code, 1)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
