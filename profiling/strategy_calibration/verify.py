"""Read-only verification of saved resident calibration; never runs hardware."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace

from calibrate import (
    SIMULATOR, STAGES, canonical, compare, dump, fit_models, gpu_arguments,
    paired_command, plan_jobs, sha, summarize_samples, validate_measurement,
    validate_native_binding, validate_paired_binding, validate_suite,
)


def evidence_path(name):
    path = Path(name)
    return path if path.is_absolute() else SIMULATOR / path


def verify(experiment, suite_file, executed_source=None):
    folder = Path(experiment)
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    suite = validate_suite(manifest['suite'])
    if sha(suite_file) != manifest['suite_sha256'] or canonical(json.loads(Path(suite_file).read_text(encoding='utf-8'))) != canonical(suite):
        raise ValueError('declared suite file differs from the exact measured bytes')
    planned = json.loads((folder/'plan.json').read_text(encoding='utf-8'))
    rows = json.loads((folder/'measurements.json').read_text(encoding='utf-8'))
    expected = [job for split in ('pilot', 'profile', 'heldout') for backend in suite['backends']
                for job in plan_jobs(suite, backend, split)]
    if canonical(expected) != canonical(planned) or len(rows) != len(expected):
        raise ValueError('saved experiment does not contain every predeclared job')
    if manifest['completed_jobs'] != len(rows) or manifest['measured_jobs'] != len(rows):
        raise ValueError('a failed/incomplete experiment cannot verify native execution')
    sources_verified = 0
    snapshots_verified = 0
    for name, digest in manifest['source_sha256'].items():
        if name.endswith(('strategy_calibration/calibrate.py', 'strategy_calibration\\calibrate.py')) and executed_source is not None:
            if sha(executed_source) != digest:
                raise ValueError('executed source snapshot does not match the original measured source hash')
            old_tree = ast.parse(Path(executed_source).read_text(encoding='utf-8'))
            current_tree = ast.parse(evidence_path(name).read_text(encoding='utf-8'))
            functions = ('canonical', 'dump', 'positive_int', 'validate_suite', 'plan_jobs', 'sample_groups',
                         'pooled_median', 'boot_median', 'interval', 'physical_key', 'fit_models', 'compare')
            def definitions(tree):
                return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name in functions}
            if len(definitions(old_tree)) != len(functions) or definitions(old_tree) != definitions(current_tree):
                raise ValueError('current statistical reconstruction differs from the actual executed source')
            sources_verified += 1
            snapshots_verified += 1
            continue
        if sha(evidence_path(name)) != digest:
            raise ValueError('measured source changed: ' + name)
        sources_verified += 1
    for name, digest in manifest['runtime_sha256'].items():
        if sha(evidence_path(name)) != digest:
            raise ValueError('provided runtime changed: ' + name)
    before, after = manifest['historical_assets_before'], manifest['historical_assets_after']
    if canonical(before) != canonical(after) or manifest['historical_assets_unchanged'] is not True:
        raise ValueError('historical files changed during the experiment')
    for name, digest in before['cpu_frozen_files'].items():
        if sha(SIMULATOR / name) != digest:
            raise ValueError('historical frozen asset changed: ' + name)
    pinned_binaries = {str(evidence_path(name).resolve()): digest
                       for name, digest in before['binary_sha256'].items()}
    for name, digest in pinned_binaries.items():
        if sha(name) != digest:
            raise ValueError('existing native binary changed: ' + name)
    gpu_devices = set()
    quality_failures = 0
    for job, row in zip(expected, rows):
        if row['status'] != 'measured' or any(canonical(row[key]) != canonical(value) for key, value in job.items()):
            raise ValueError('measurement row lost its predeclared exact binding')
        raw_path = folder / row['raw_output']
        if sha(raw_path) != row['raw_sha256']:
            raise ValueError('raw output changed: ' + str(raw_path))
        raw = json.loads(raw_path.read_text(encoding='utf-8'))
        if raw['returncode'] != 0 or any(canonical(raw[key]) != canonical(value) for key, value in job.items()):
            raise ValueError('raw native execution did not match the expected job')
        if 'child_flags_before_by_backend' in manifest:
            if row.get('child_flags') != raw.get('child_flags') or raw.get('child_flags') != manifest['child_flags_before_by_backend'][job['backend']]:
                raise ValueError('native child environment changed or lost its recorded binding')
        binary = Path(raw['command'][0])
        if str(binary.resolve()) not in pinned_binaries:
            raise ValueError('native command uses an unpinned binary')
        protocol = SimpleNamespace(**job['protocol'], seed=job['seed'])
        command = paired_command(job['case'], binary, protocol) if job['backend'] == 'cpu' else [str(binary), *gpu_arguments(job['case'], protocol.warmup, protocol.samples, protocol.seed)]
        if command != raw['command']:
            raise ValueError('native command differs from the exact protocol')
        native = json.loads(raw['stdout'])
        if job['backend'] == 'cpu':
            validate_paired_binding(job['case'], native, protocol)
            groups = [*native['stage_wall_ns'].values(), native['serial_wall_ns'], native['joint_wall_ns']]
        else:
            validate_native_binding(job['case'], native, protocol)
            native['expected_samples'] = protocol.samples
            validate_measurement(native)
            groups = [native['wall_ns']]
            gpu_devices.add(native['device_info']['uuid'])
            if native['workspace_bytes'] is not None:
                raise ValueError('unknown GPU scratch was replaced by a number')
        if canonical(native) != canonical(row['native']):
            raise ValueError('measurement native fields disagree with raw stdout')
        cv = max(summarize_samples(group)['cv'] for group in groups)
        flags = ['cv_above_predeclared_threshold'] if cv > suite['quality']['cv_threshold'] else []
        if row['maximum_observed_cv'] != cv or row['quality_flags'] != flags:
            raise ValueError('CV or quality flags disagree with preserved samples')
        quality_failures += bool(flags)
    if len(gpu_devices) > 1:
        raise ValueError('CUDA runs switched physical device identity')
    model_path = folder/'prediction-model.json'
    if sha(model_path) != manifest['model_sha256_before_heldout']:
        raise ValueError('profile model changed after heldout started')
    model = json.loads(model_path.read_text(encoding='utf-8'))
    fitted = fit_models(suite, [row for row in rows if row['split'] == 'profile'])
    if canonical(fitted) != canonical(model):
        raise ValueError('saved model cannot be reconstructed from profile observations only')
    report = json.loads((folder/'report.json').read_text(encoding='utf-8'))
    if canonical(compare(suite, rows, model)) != canonical(report):
        raise ValueError('saved heldout comparison differs from independent recomputation')
    if manifest['calibration_claimed'] != report['calibrated']:
        raise ValueError('manifest calibration status disagrees with the quality/error gates')
    text_files = [path for path in folder.rglob('*') if path.is_file() and path.suffix in ('.json', '.md', '.py', '.csv', '.log')]
    if any(b'\r\n' in path.read_bytes() for path in text_files):
        raise ValueError('experiment text output must use LF')
    return {'format_version': 'resident-calibration-verification-0.1',
        'hardware_execution_performed_by_verifier': False,
        'raw_native_bindings_verified': len(rows), 'source_files_verified': sources_verified,
        'sources_verified_using_executed_snapshot': snapshots_verified,
        'statistical_functions_match_executed_snapshot': snapshots_verified > 0,
        'executed_source_snapshot_sha256': sha(executed_source) if executed_source is not None else None,
        'child_flags_run_observation': 'recorded_and_bound' if 'child_flags_before_by_backend' in manifest else 'legacy_run_unobserved_except_graph_disable',
        'provided_runtime_files_verified': len(manifest['runtime_sha256']),
        'cpu_frozen_files_unchanged': len(before['cpu_frozen_files']),
        'existing_native_binaries_unchanged': len(before['binary_sha256']),
        'gpu_uuids': sorted(gpu_devices), 'CV_flagged_jobs': quality_failures,
        'profile_model_reconstructed_without_heldout': True,
        'signed_absolute_errors_and_bootstrap_recomputed': True,
        'LF_text_files_verified': len(text_files), 'calibrated': report['calibrated'],
        'capacity_calibrated': False, 'mixed_or_offload_calibrated': False,
        'manifest_sha256': sha(folder/'manifest.json'), 'measurements_sha256': sha(folder/'measurements.json'),
        'prediction_model_sha256': sha(model_path), 'report_sha256': sha(folder/'report.json')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--executed-source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError('verification output must be a new file')
    result = verify(args.experiment, args.suite, args.executed_source)
    dump(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as error:
        print('verify_strategy_calibration:', error)
        raise SystemExit(1)
