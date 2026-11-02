from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from profiling.strategy_calibration.calibrate import (
    compare, fit_models, main, plan_jobs, record_inventory, run_job, validate_suite, verify_inventory,
)

SUITE = Path(__file__).resolve().parents[1] / 'suite.json'


def suite(backend='cpu'):
    value = json.loads(SUITE.read_text(encoding='utf-8'))
    value['backends'] = [backend]
    value['cases'] = value['cases'][:1]
    return value


def rows(backend='cpu', split='profile'):
    result = []
    for repeat in range(2):
        common = dict(backend=backend, geometry_id='small_prefill_q3_h0', split=split,
                      repeat=repeat, status='measured')
        if backend == 'cpu':
            result.append({**common, 'native': {
                'stage_wall_ns': {'projection': [10]*4, 'attention': [20]*4, 'dense_ffn': [30]*4},
                'serial_wall_ns': [61 if split == 'profile' else 62]*4,
                'joint_wall_ns': [40]*4}})
        else:
            for stage, value in [('projection', 20), ('attention', 30), ('dense_ffn', 40), ('block', 75 if split == 'profile' else 80)]:
                result.append({**common, 'case': {'kind': stage}, 'native': {'wall_ns': [value]*4}})
    return result


class CalibrationContractTests(unittest.TestCase):
    def test_explicit_suite_and_predeclared_independent_process_seeds(self):
        value = validate_suite(suite())
        profile = plan_jobs(value, 'cpu', 'profile')
        heldout = plan_jobs(value, 'cpu', 'heldout')
        self.assertEqual({item['seed'] for item in profile}, {17})
        self.assertEqual({item['seed'] for item in heldout}, {29})
        self.assertEqual(len(profile), 2)
        self.assertEqual({item['repeat'] for item in profile}, {0, 1})

    def test_missing_protocol_is_not_guessed(self):
        value = suite()
        del value['cpu_protocol']['cpu_mask']
        with self.assertRaises(ValueError):
            validate_suite(value)

    def test_seed_leakage_is_rejected_even_when_run_labels_differ(self):
        value = suite()
        value['heldout_seeds'] = value['profile_seeds']
        with self.assertRaises(ValueError):
            validate_suite(value)

    def test_heldout_observations_cannot_fit_a_saved_model(self):
        with self.assertRaises(ValueError):
            fit_models(suite(), rows() + rows(split='heldout'))

    def test_pilot_is_excluded_from_coefficients(self):
        pilot = rows(split='pilot')
        pilot[0]['native']['serial_wall_ns'] = [999999]*4
        model = fit_models(suite(), rows() + pilot)
        self.assertEqual(model['models'][0]['same_variant_median_ns'], 61)
        self.assertEqual(model['models'][0]['serial_outer_clock_offset_ns'], 1)

    def test_same_partition_variant_and_joint_discrepancy_are_separate(self):
        value = suite()
        report = compare(value, rows() + rows(split='heldout'), fit_models(value, rows()))
        primary = next(item for item in report['comparisons'] if item['comparison'] == 'same_variant_partition_cost_sum')
        self.assertEqual(primary['predicted_ns'], 60)
        self.assertEqual(primary['observed_heldout_median_ns'], 62)
        self.assertAlmostEqual(primary['signed_relative_error'], -2/62)
        self.assertTrue(primary['calibrated'])
        discrepancy = next(item for item in report['comparisons'] if 'discrepancy' in item['comparison'])
        self.assertEqual(discrepancy['absolute_relative_error'], 0.5)
        self.assertFalse(discrepancy['same_execution_variant'])
        self.assertFalse(discrepancy['calibrated'])
        self.assertTrue(report['calibrated'])

    def test_gpu_joint_repeatability_never_proves_segmented_or_capacity_execution(self):
        value = suite('gpu')
        model = fit_models(value, rows('gpu'))
        report = compare(value, rows('gpu') + rows('gpu', 'heldout'), model)
        self.assertIsNone(model['models'][0]['workspace_bytes'])
        primary = report['comparisons'][0]
        self.assertEqual(primary['predicted_ns'], 75)
        self.assertEqual(primary['observed_heldout_median_ns'], 80)
        self.assertTrue(primary['calibrated'])
        self.assertFalse(report['capacity_calibrated'])
        self.assertFalse(report['mixed_or_offload_calibrated'])

    def test_unrelated_gpu_stage_noise_cannot_invalidate_joint_only_lookup(self):
        value = suite('gpu')
        observed = rows('gpu') + rows('gpu', 'heldout')
        for row in observed:
            if row['case']['kind'] == 'projection':
                row['native']['wall_ns'] = [20, 20, 20, 1000]
        report = compare(value, observed, fit_models(value, rows('gpu')))
        self.assertTrue(report['comparisons'][0]['calibrated'])
        self.assertFalse(report['comparisons'][1]['cv_within_predeclared_threshold'])

    def test_no_outlier_is_deleted_to_make_quality_gate_pass(self):
        value = suite()
        observed = rows() + rows(split='heldout')
        observed[-1]['native']['serial_wall_ns'] = [62, 62, 62, 999999]
        before = copy.deepcopy(observed)
        report = compare(value, observed, fit_models(value, rows()))
        self.assertEqual(observed, before)
        self.assertFalse(report['calibrated'])
        self.assertFalse(report['comparisons'][0]['cv_within_predeclared_threshold'])

    def test_shape_history_or_runtime_protocol_change_cannot_reuse_lookup(self):
        value = suite()
        model = fit_models(value, rows())
        value['cases'][0]['visible_history'] = 3
        with self.assertRaises(ValueError):
            compare(value, rows() + rows(split='heldout'), model)

    def test_bootstrap_is_reproducible_and_does_not_fit_heldout(self):
        value = suite()
        observed = rows() + rows(split='heldout')
        observed[-1]['native']['serial_wall_ns'] = [59, 62, 63, 65]
        model = fit_models(value, rows())
        before = copy.deepcopy(model)
        first = compare(value, observed, model)
        self.assertEqual(first, compare(value, observed, model))
        self.assertEqual(model, before)
        bounds = first['comparisons'][0]['heldout_median_bootstrap_interval_ns']
        self.assertLessEqual(bounds[0], bounds[1])

    def test_native_failure_preserves_returncode_stdout_and_stderr(self):
        job = plan_jobs(suite(), 'cpu', 'profile')[0]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'failed.json'
            with patch('profiling.strategy_calibration.calibrate.subprocess.run',
                       return_value=SimpleNamespace(returncode=7, stdout='partial payload\n', stderr='device failure\n')):
                row = run_job(job, Path(folder)/'probe.exe', [], path, 1)
            raw = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(row['status'], 'failed')
            self.assertEqual(raw['returncode'], 7)
            self.assertEqual(raw['stdout'], 'partial payload\n')
            self.assertEqual(raw['stderr'], 'device failure\n')
            self.assertNotIn(b'\r\n', path.read_bytes())

    def test_timeout_preserves_partial_bytes_without_hardware(self):
        job = plan_jobs(suite(), 'cpu', 'profile')[0]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'timeout.json'
            with patch('profiling.strategy_calibration.calibrate.subprocess.run',
                       side_effect=subprocess.TimeoutExpired(['probe'], 1, output=b'partial', stderr=b'busy')):
                row = run_job(job, Path(folder)/'probe.exe', [], path, 1)
            raw = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(row['status'], 'failed')
            self.assertTrue(raw['timed_out'])
            self.assertEqual(raw['stdout'], 'partial')
            self.assertEqual(raw['stderr'], 'busy')

    def test_nonempty_output_is_refused_before_hardware_or_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = root/'suite.json'
            config.write_text(json.dumps(suite()), encoding='utf-8')
            binary = root/'probe.exe'
            binary.write_bytes(b'unexecuted test placeholder')
            output = root/'output'
            output.mkdir()
            sentinel = output/'keep.txt'
            sentinel.write_bytes(b'keep')
            with patch('profiling.strategy_calibration.calibrate.subprocess.run') as native:
                with self.assertRaises(ValueError):
                    main(['--suite', str(config), '--cpu-binary', str(binary), '--runtime-dir', folder,
                          '--timeout-s', '1', '--output', str(output)])
                native.assert_not_called()
            self.assertEqual(sentinel.read_bytes(), b'keep')

    def test_changed_source_or_missing_runtime_file_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, library = root/'source.py', root/'provided.dll'
            source.write_bytes(b'original source')
            library.write_bytes(b'provided library')
            sources = record_inventory([source], 'source')
            runtimes = record_inventory([library], 'runtime')
            source.write_bytes(b'changed source')
            with self.assertRaisesRegex(ValueError, 'changed'):
                verify_inventory(sources, 'source')
            library.unlink()
            with self.assertRaisesRegex(ValueError, 'missing'):
                verify_inventory(runtimes, 'runtime')

    def test_child_flags_record_actual_inheritance_and_explicit_graph_disable(self):
        job = plan_jobs(suite('gpu'), 'gpu', 'profile')[0]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'raw.json'
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '3', 'OMP_NUM_THREADS': '7',
                                         'GGML_CUDA_DISABLE_GRAPHS': '0', 'GGML_CUDA_FORCE_MMQ': '1'}):
                with patch('profiling.strategy_calibration.calibrate.subprocess.run',
                           return_value=SimpleNamespace(returncode=7, stdout='', stderr='no hardware')) as native:
                    row = run_job(job, Path(folder)/'probe.exe', [], path, 1)
            raw = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(row['child_flags'], raw['child_flags'])
            self.assertEqual(raw['child_flags']['CUDA_VISIBLE_DEVICES'], '3')
            self.assertEqual(raw['child_flags']['OMP_NUM_THREADS'], '7')
            self.assertEqual(raw['child_flags']['GGML_CUDA_FORCE_MMQ'], '1')
            self.assertEqual(raw['child_flags']['GGML_CUDA_DISABLE_GRAPHS'], '1')
            self.assertEqual(native.call_args.kwargs['env']['GGML_CUDA_DISABLE_GRAPHS'], '1')


if __name__ == '__main__':
    unittest.main()
