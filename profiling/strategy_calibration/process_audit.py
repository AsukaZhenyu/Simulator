"""Posthoc spread/environment audit; does not modify gates or run hardware."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from calibrate import STAGES, dump, pooled_median, sha, summarize_samples

ENVIRONMENT_FLAGS = (
    'GGML_CUDA_FORCE_MMQ', 'GGML_CUDA_FORCE_CUBLAS',
    'GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F', 'GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F',
    'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'OMP_NUM_THREADS',
    'OMP_PROC_BIND', 'OMP_PLACES', 'GGML_CUDA_DISABLE_GRAPHS',
)


def audit(experiment):
    folder = Path(experiment)
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    rows = json.loads((folder/'measurements.json').read_text(encoding='utf-8'))
    report = json.loads((folder/'report.json').read_text(encoding='utf-8'))
    if len(rows) != manifest['planned_jobs'] or any(row['status'] != 'measured' for row in rows):
        raise ValueError('process audit requires the completed experiment')
    items = []
    for comparison in report['comparisons']:
        backend, geometry, label = (comparison[key] for key in ('backend', 'geometry_id', 'comparison'))
        metric = ('joint_wall_ns' if backend == 'cpu' and ('joint' in label)
                  else 'serial_wall_ns' if backend == 'cpu' else 'block')
        processes = []
        for row in rows:
            if (row['backend'], row['geometry_id']) != (backend, geometry) or row['split'] == 'pilot':
                continue
            if backend == 'gpu' and row['case']['kind'] != metric:
                continue
            samples = row['native'][metric] if backend == 'cpu' else row['native']['wall_ns']
            stats = summarize_samples(samples)
            observed = stats['median_ns']
            processes.append({'split': row['split'], 'seed': row['seed'], 'repeat': row['repeat'],
                'raw_output': row['raw_output'], 'median_ns': observed,
                'within_process_cv': stats['cv'],
                'signed_relative_error_of_saved_prediction': (comparison['predicted_ns'] - observed)/observed})
        spreads = {}
        for split in ('profile', 'heldout'):
            medians = [row['median_ns'] for row in processes if row['split'] == split]
            stats = summarize_samples(medians)
            spreads[split] = {'processes': len(medians), 'process_medians_ns': medians,
                'between_process_cv_of_medians': stats['cv'],
                'min_process_median_ns': min(medians), 'max_process_median_ns': max(medians),
                'relative_span_of_process_medians': (max(medians)-min(medians))/pooled_median([medians])}
        heldout_errors = [abs(row['signed_relative_error_of_saved_prediction']) for row in processes if row['split'] == 'heldout']
        items.append({'backend': backend, 'geometry_id': geometry, 'comparison': label,
            'same_execution_variant': comparison['same_execution_variant'],
            'original_pooled_point_estimate_gate_pass': comparison['calibrated'],
            'processes': processes, 'between_process_spread': spreads,
            'maximum_heldout_process_absolute_relative_error': max(heldout_errors),
            'new_acceptance_gate_applied': False})
    observed_graph_flags = sorted({row['native']['cuda_graph_disable_environment']
        for row in rows if row['backend'] == 'gpu'})
    return {'format_version': 'resident-calibration-process-audit-0.1',
        'hardware_execution_performed_by_auditor': False,
        'collected_at_utc': datetime.now(timezone.utc).isoformat(),
        'calibrated_field_interpretation': 'pooled-median point-estimate gates only; within-process CV plus pooled relative error',
        'between_process_drift_was_an_original_gate': False,
        'reports_or_gates_modified': False,
        'retained_offset_field_interpretation': 'profile-fitted-composition-offset = median(serial) - sum(median(stages)); not separately measured physical clock/dispatch overhead',
        'child_environment_at_run': {'status': 'unobserved_in_run_except_graph_disable',
            'unobserved_names': [name for name in ENVIRONMENT_FLAGS if name != 'GGML_CUDA_DISABLE_GRAPHS'],
            'graph_disable_observed_native_values': observed_graph_flags,
            'default_environment_asserted': False},
        'posthoc_current_environment': {'observation_scope': 'this audit process after measurement; not evidence of run-start environment',
                                       'values': {name: os.environ.get(name) for name in ENVIRONMENT_FLAGS}},
        'comparisons': items, 'analysis_source_sha256': sha(__file__),
        'manifest_sha256': sha(folder/'manifest.json'), 'report_sha256': sha(folder/'report.json')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError('audit output must be a new file')
    result = audit(args.experiment)
    dump(args.output, result)
    print('Saved', len(result['comparisons']), 'process comparisons; original gates unchanged')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as error:
        print('process_audit:', error)
        raise SystemExit(1)
