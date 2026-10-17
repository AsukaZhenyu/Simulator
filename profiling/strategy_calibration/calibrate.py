"""Measure exact resident Dense execution variants and validate on new processes.

CPU partition timings predict the SAME partitioned serial execution. GPU joint
timings predict the SAME joint execution. Partition sums versus joint graphs are
reported separately as execution-variant discrepancy, never as mixed offload.
No experiment exports costs or converts unknown GPU workspace into capacity.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
from types import SimpleNamespace

PROFILING = Path(__file__).resolve().parents[1]
SIMULATOR = PROFILING.parent
sys.path[:0] = [str(PROFILING), str(SIMULATOR / 'modeling')]
from llm_infer_model.m2.dense_contract import build_plan as cpu_plan
from llm_infer_model.m2.dense_cuda_contract import build_plan as gpu_plan, native_arguments as gpu_arguments
from run_dense_paired_measurements import paired_command, validate_paired_binding
from run_dense_cuda_measurements import validate_native_binding
from run_measurements import environment
from profile_contract import summarize_samples, validate_measurement

VERSION = 'resident-strategy-calibration-0.1'
STAGES = ('projection', 'attention', 'dense_ffn')
ENVIRONMENT_FLAGS = (
    'GGML_CUDA_DISABLE_GRAPHS', 'GGML_CUDA_FORCE_MMQ', 'GGML_CUDA_FORCE_CUBLAS',
    'GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F', 'GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F',
    'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'OMP_NUM_THREADS', 'OMP_PROC_BIND',
    'OMP_PLACES', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def record_inventory(paths, label):
    result = {}
    for path in sorted({Path(path).resolve() for path in paths}):
        if not path.is_file():
            raise ValueError(label + ' file is missing: ' + str(path))
        result[str(path)] = sha(path)
    return result


def verify_inventory(expected, label):
    observed = record_inventory(expected, label)
    if observed != expected:
        changed = [path for path in expected if observed[path] != expected[path]]
        raise ValueError(label + ' changed during experiment: ' + repr(changed))
    return observed


def child_environment(runtime_dirs, backend):
    env = os.environ.copy()
    env['PATH'] = os.pathsep.join([str(path.resolve()) for path in runtime_dirs] + [env.get('PATH', '')])
    if os.name != 'nt':
        env['LD_LIBRARY_PATH'] = os.pathsep.join([str(path.resolve()) for path in runtime_dirs] + [env.get('LD_LIBRARY_PATH', '')])
    env.pop('LLAMA_VIZ_TIMING', None)
    if backend == 'gpu':
        env['GGML_CUDA_DISABLE_GRAPHS'] = '1'
    return env


def observed_child_flags(env):
    names = set(ENVIRONMENT_FLAGS) | {name for name in env if name.startswith(('GGML_', 'CUDA_', 'OMP_', 'CUBLAS_'))}
    # None means observed absent in this actual child environment; it never
    # means the effective library default or an unobserved legacy run value.
    return {name: env.get(name) for name in sorted(names)}


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                          encoding='utf-8', newline='\n')


def positive_int(value, label, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(label + ' must be an explicit integer >= ' + str(minimum))


def validate_suite(document):
    expected = {'format_version', 'cases', 'backends', 'cpu_protocol', 'gpu_protocol',
                'profile_seeds', 'heldout_seeds', 'repeats', 'pilot', 'quality'}
    if not isinstance(document, dict) or set(document) != expected or document['format_version'] != VERSION:
        raise ValueError('suite requires the complete explicit calibration schema')
    if document['backends'] not in (['cpu'], ['gpu'], ['cpu', 'gpu']):
        raise ValueError('backends must explicitly select cpu, gpu, or cpu then gpu')
    if not isinstance(document['cases'], list) or not document['cases']:
        raise ValueError('cases must be a nonempty exact Dense declaration')
    positive_int(document['repeats'], 'repeats')
    seen = set()
    for split in ('profile_seeds', 'heldout_seeds'):
        seeds = document[split]
        if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
            raise ValueError(split + ' must contain unique explicit seeds')
        for seed in seeds:
            positive_int(seed, split, 0)
            if seed > 0xFFFFFFFF or seed in seen:
                raise ValueError('profile and heldout seeds must be disjoint uint32 values')
            seen.add(seed)
    for backend in ('cpu', 'gpu'):
        protocol = document[backend + '_protocol']
        keys = {'warmup', 'samples'} | ({'threads', 'compute_mode', 'min_warmup_ms', 'cpu_mask'}
            if backend == 'cpu' else {'cuda_device', 'cuda_graphs'})
        if not isinstance(protocol, dict) or set(protocol) != keys:
            raise ValueError(backend + ' protocol has missing or unknown fields')
        positive_int(protocol['warmup'], 'warmup', 0)
        positive_int(protocol['samples'], 'samples', 4)
        if protocol['samples'] > 10000 or protocol['samples'] % 2:
            raise ValueError('samples require an even count in 4..10000')
        if backend == 'cpu':
            positive_int(protocol['threads'], 'threads')
            positive_int(protocol['min_warmup_ms'], 'min_warmup_ms', 0)
            mask = protocol['cpu_mask']
            if protocol['compute_mode'] not in ('graph', 'plan') or not isinstance(mask, list) or len(mask) != protocol['threads'] or len(set(mask)) != len(mask):
                raise ValueError('CPU mode and one distinct explicit processor per thread are required')
            for cpu in mask:
                positive_int(cpu, 'CPU processor', 0)
        else:
            positive_int(protocol['cuda_device'], 'cuda_device', 0)
            if protocol['cuda_graphs'] != 'disabled':
                raise ValueError('this executable requires explicit disabled CUDA graphs')
    pilot = document['pilot']
    if not isinstance(pilot, dict) or set(pilot) != {'case_ids', 'warmup', 'samples', 'cpu_min_warmup_ms'}:
        raise ValueError('pilot requires explicit cases, samples, warmup and minimum duration')
    positive_int(pilot['warmup'], 'pilot warmup', 0)
    positive_int(pilot['samples'], 'pilot samples', 4)
    positive_int(pilot['cpu_min_warmup_ms'], 'pilot minimum warmup', 0)
    if pilot['samples'] % 2 or pilot['samples'] > 10000:
        raise ValueError('pilot sample count requires an even count in 4..10000')
    ids = [case.get('id') for case in document['cases']]
    if len(ids) != len(set(ids)) or any(case.get('kind') != 'block' for case in document['cases']):
        raise ValueError('each case requires a unique id and complete block geometry')
    if not isinstance(pilot['case_ids'], list) or not pilot['case_ids'] or not set(pilot['case_ids']) <= set(ids):
        raise ValueError('pilot must select existing exact case IDs')
    quality = document['quality']
    if not isinstance(quality, dict) or set(quality) != {'cv_threshold', 'max_absolute_relative_error', 'bootstrap_iterations', 'bootstrap_confidence', 'bootstrap_seed'}:
        raise ValueError('all quality thresholds and bootstrap settings must be explicit')
    for key in ('cv_threshold', 'max_absolute_relative_error', 'bootstrap_confidence'):
        value = quality[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(key + ' must be finite in (0,1)')
    positive_int(quality['bootstrap_iterations'], 'bootstrap_iterations', 100)
    positive_int(quality['bootstrap_seed'], 'bootstrap_seed', 0)
    # Contract builders reject unknown arithmetic, dtype, phase, batch or shape.
    raw = {'schema_version': 'dense_layer_measurement_suite_v1', 'cases': document['cases']}
    cpu_plan(raw, document['cpu_protocol']['threads'])
    gpu_plan(raw, document['gpu_protocol']['cuda_device'], document['gpu_protocol']['cuda_graphs'])
    return document


def plan_jobs(suite, backend, split):
    protocol = copy.deepcopy(suite[backend + '_protocol'])
    cases = copy.deepcopy(suite['cases'])
    seeds = suite['profile_seeds'] if split in ('pilot', 'profile') else suite['heldout_seeds']
    repeats = suite['repeats']
    if split == 'pilot':
        cases = [case for case in cases if case['id'] in suite['pilot']['case_ids']]
        protocol.update(warmup=suite['pilot']['warmup'], samples=suite['pilot']['samples'])
        if backend == 'cpu':
            protocol['min_warmup_ms'] = suite['pilot']['cpu_min_warmup_ms']
        seeds, repeats = [seeds[0]], 1
    expanded = []
    for case in cases:
        if backend == 'cpu':
            expanded.append(case)
        else:
            for stage in (*STAGES, 'block'):
                item = copy.deepcopy(case)
                item['id'] += '__' + stage
                item['kind'] = stage
                item['calibration_geometry_id'] = case['id']
                expanded.append(item)
    # The shared contract refuses non-schema extension keys, so keep the
    # geometry ID outside the validated case and reconstruct it from base IDs.
    base_ids = {case['id'] for case in cases}
    for case in expanded:
        case.pop('calibration_geometry_id', None)
    raw = {'schema_version': 'dense_layer_measurement_suite_v1', 'cases': expanded}
    cases = cpu_plan(raw, protocol['threads']) if backend == 'cpu' else gpu_plan(raw, protocol['cuda_device'], protocol['cuda_graphs'])
    jobs = []
    for seed in seeds:
        for repeat in range(repeats):
            ordered = list(cases)
            random.Random(seed + repeat).shuffle(ordered)
            for case in ordered:
                geometry = case['id'] if backend == 'cpu' else case['id'].rsplit('__', 1)[0]
                if geometry not in base_ids:
                    raise ValueError('expanded job lost its geometry binding')
                jobs.append({'backend': backend, 'split': split, 'seed': seed,
                    'repeat': repeat, 'geometry_id': geometry, 'case': case, 'protocol': protocol})
    return jobs


def sample_groups(rows, backend, geometry, split, metric):
    result = []
    for row in rows:
        if (row['backend'], row['geometry_id'], row['split']) != (backend, geometry, split):
            continue
        if row['status'] != 'measured':
            raise ValueError('failed observation cannot enter a calibrated lookup')
        native = row['native']
        if backend == 'cpu':
            values = native['stage_wall_ns'][metric] if metric in STAGES else native[metric]
        elif row['case']['kind'] == metric:
            values = native['wall_ns']
        else:
            continue
        result.append(values)
    if not result:
        raise ValueError('missing exact measurements: ' + '/'.join((backend, geometry, split, metric)))
    return result


def pooled_median(groups):
    return statistics.median([value for group in groups for value in group])


def boot_median(groups, rng):
    # Processes are the first sampling unit; samples within each selected
    # process are the second. Desktop disturbance is not deleted as an outlier.
    selected = [rng.choice(groups) for _ in groups]
    return statistics.median([rng.choice(group) for group in selected for _ in group])


def interval(values, confidence):
    ordered = sorted(values)
    low = (1 - confidence) / 2
    return [ordered[math.floor(low * (len(ordered) - 1))],
            ordered[math.ceil((1 - low) * (len(ordered) - 1))]]


def physical_key(suite, backend, geometry):
    case = next(case for case in suite['cases'] if case['id'] == geometry)
    return canonical({'backend': backend, 'case': {key: case[key] for key in
        ('config', 'phase', 'batch', 'tokens', 'visible_history')},
        'protocol': suite[backend + '_protocol']})


def fit_models(suite, rows):
    if any(row['split'] == 'heldout' for row in rows):
        raise ValueError('heldout observations may not enter fitting')
    models = []
    for backend in suite['backends']:
        for case in suite['cases']:
            geometry = case['id']
            stages = {stage: pooled_median(sample_groups(rows, backend, geometry, 'profile', stage)) for stage in STAGES}
            total = sum(stages.values())
            target = 'serial_wall_ns' if backend == 'cpu' else 'block'
            observed = pooled_median(sample_groups(rows, backend, geometry, 'profile', target))
            joint = pooled_median(sample_groups(rows, backend, geometry, 'profile', 'joint_wall_ns' if backend == 'cpu' else 'block'))
            models.append({'backend': backend, 'geometry_id': geometry,
                'physical_key': physical_key(suite, backend, geometry),
                'training_split': 'profile', 'stage_median_ns': stages,
                'stage_sum_ns': total, 'same_variant_median_ns': observed,
                'joint_median_ns': joint,
                'serial_outer_clock_offset_ns': observed - total if backend == 'cpu' else None,
                'workspace_bytes': None,
                'workspace_status': 'unknown_cuda_private_scratch' if backend == 'gpu' else 'validated_components_retained_in_raw_not_exported_as_cost',
                'capacity_calibrated': False})
    return {'format_version': VERSION, 'fitting_inputs': 'profile only; pilot and heldout excluded',
            'models': models}


def compare(suite, rows, fitted):
    quality = suite['quality']
    rng = random.Random(quality['bootstrap_seed'])
    comparisons = []
    for model in fitted['models']:
        backend, geometry = model['backend'], model['geometry_id']
        if model['physical_key'] != physical_key(suite, backend, geometry):
            raise ValueError('prediction requires exactly matching geometry and execution protocol')
        metric = 'serial_wall_ns' if backend == 'cpu' else 'block'
        specifications = ([('same_variant_partition_cost_sum', metric, model['stage_sum_ns'], True),
                           ('same_variant_partition_cost_sum_with_profile_clock_offset', metric,
                            model['same_variant_median_ns'], True),
                           ('same_variant_joint_lookup', 'joint_wall_ns', model['joint_median_ns'], True),
                           ('partition_sum_to_joint_variant_discrepancy', 'joint_wall_ns', model['stage_sum_ns'], False)]
            if backend == 'cpu' else
            [('same_variant_joint_lookup', 'block', model['joint_median_ns'], True),
             ('isolated_stage_sum_to_joint_variant_discrepancy', 'block', model['stage_sum_ns'], False)])
        for label, target, predicted, same_variant in specifications:
            heldout = sample_groups(rows, backend, geometry, 'heldout', target)
            observed = pooled_median(heldout)
            signed = (predicted - observed) / observed
            used = [row for row in rows if row['backend'] == backend and row['geometry_id'] == geometry
                    and row['split'] in ('profile', 'heldout')]
            relevant = ({'joint_wall_ns'} if label == 'same_variant_joint_lookup' and backend == 'cpu'
                        else {'block'} if label == 'same_variant_joint_lookup'
                        else {*STAGES, target})
            cvs = []
            for row in used:
                if backend == 'cpu':
                    values = [row['native']['stage_wall_ns'][key] if key in STAGES else row['native'][key]
                              for key in relevant]
                else:
                    values = [row['native']['wall_ns']] if row['case']['kind'] in relevant else []
                cvs.extend(summarize_samples(group)['cv'] for group in values)
            acceptable = max(cvs) <= quality['cv_threshold'] and abs(signed) <= quality['max_absolute_relative_error']
            observed_boot = [boot_median(heldout, rng) for _ in range(quality['bootstrap_iterations'])]
            # Prediction is fixed BEFORE heldout. This interval quantifies
            # heldout uncertainty conditional on that saved model, not a CI
            # for unseen geometries or adaptive coefficient selection.
            error_boot = [(predicted - value) / value for value in observed_boot]
            comparisons.append({'backend': backend, 'geometry_id': geometry, 'comparison': label,
                'same_execution_variant': same_variant, 'predicted_ns': predicted, 'observed_heldout_median_ns': observed,
                'signed_error_ns': predicted - observed, 'absolute_error_ns': abs(predicted - observed),
                'signed_relative_error': signed, 'absolute_relative_error': abs(signed),
                'maximum_observed_cv': max(cvs), 'cv_within_predeclared_threshold': max(cvs) <= quality['cv_threshold'],
                'error_within_predeclared_tolerance': abs(signed) <= quality['max_absolute_relative_error'],
                'heldout_median_bootstrap_interval_ns': interval(observed_boot, quality['bootstrap_confidence']),
                'signed_error_bootstrap_interval': interval(error_boot, quality['bootstrap_confidence']),
                'bootstrap_scope': 'hierarchical process/sample resampling; fixed saved profile model',
                'calibrated': same_variant and acceptable,
                'capacity_calibrated': False, 'mixed_or_offload_calibrated': False})
    primary = [row for row in comparisons if row['same_execution_variant']]
    return {'format_version': VERSION, 'comparisons': comparisons,
        'calibrated': bool(primary) and all(row['calibrated'] for row in primary),
        'capacity_calibrated': False, 'mixed_or_offload_calibrated': False,
        'scope': 'exact declared resident CPU partition/joint and GPU joint latency only',
        'limitations': ['generated weights, not pretrained model execution',
            'heldout seeds/processes validate repeatability for exact measured geometries, not shape generalization',
            'GPU segmented chain, mixed device execution, cache, prefetch and overlap were not executed',
            'unknown GPU private workspace remains unknown; no capacity feasibility claim',
            'desktop apps remain open; samples and quality failures are retained']}


def preserved_snapshot(freeze_manifest, binaries):
    expected = json.loads(freeze_manifest.read_text(encoding='utf-8'))['files'] if freeze_manifest else {}
    observed = {name: sha(SIMULATOR / name) for name in expected}
    mismatches = [name for name in expected if expected[name] != observed[name]]
    if mismatches:
        raise ValueError('historical CPU frozen assets already differ: ' + repr(mismatches))
    return {'freeze_manifest': str(freeze_manifest) if freeze_manifest else None,
        'freeze_manifest_sha256': sha(freeze_manifest) if freeze_manifest else None,
        'cpu_frozen_files': observed, 'binary_sha256': {str(path): sha(path) for path in binaries}}


def run_job(job, binary, runtime_dirs, output, timeout):
    protocol = SimpleNamespace(**job['protocol'], seed=job['seed'])
    command = (paired_command(job['case'], binary, protocol) if job['backend'] == 'cpu'
               else [str(binary), *gpu_arguments(job['case'], protocol.warmup, protocol.samples, protocol.seed)])
    env = child_environment(runtime_dirs, job['backend'])
    flags = observed_child_flags(env)
    raw = {'command': command, **job, 'child_flags': flags,
           'started_at_utc': datetime.now(timezone.utc).isoformat()}
    row = {**job, 'child_flags': flags, 'raw_output': output.name, 'status': 'failed'}
    try:
        process = subprocess.run(command, env=env, capture_output=True, text=True, errors='replace', timeout=timeout)
        raw.update(returncode=process.returncode, stdout=process.stdout, stderr=process.stderr)
        if process.returncode:
            raise ValueError('native returned ' + str(process.returncode) + ': ' + process.stderr[-1200:])
        native = json.loads(process.stdout)
        if job['backend'] == 'cpu':
            validate_paired_binding(job['case'], native, protocol)
        else:
            validate_native_binding(job['case'], native, protocol)
            native['expected_samples'] = protocol.samples
            validate_measurement(native)
        row.update(status='measured', native=native)
        groups = ([*native['stage_wall_ns'].values(), native['serial_wall_ns'], native['joint_wall_ns']]
                  if job['backend'] == 'cpu' else [native['wall_ns']])
        row['maximum_observed_cv'] = max(summarize_samples(values)['cv'] for values in groups)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        if isinstance(error, subprocess.TimeoutExpired):
            raw.update(timed_out=True, stdout=error.stdout.decode(errors='replace') if isinstance(error.stdout, bytes) else error.stdout,
                       stderr=error.stderr.decode(errors='replace') if isinstance(error.stderr, bytes) else error.stderr)
        raw['error'] = row['error'] = str(error)
    raw['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    dump(output, raw)
    row['raw_sha256'] = sha(output)
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--cpu-binary', type=Path)
    parser.add_argument('--gpu-binary', type=Path)
    parser.add_argument('--runtime-dir', type=Path, action='append', required=True)
    parser.add_argument('--freeze-manifest', type=Path)
    parser.add_argument('--preserve-file', type=Path, action='append', default=[])
    parser.add_argument('--timeout-s', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    suite = validate_suite(json.loads(args.suite.read_text(encoding='utf-8')))
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        raise ValueError('timeout must be explicitly positive and finite')
    binaries = {backend: getattr(args, backend + '_binary') for backend in suite['backends']}
    if any(path is None or not path.is_file() for path in binaries.values()):
        raise ValueError('every selected backend needs an existing native executable')
    if any(not path.is_dir() for path in args.runtime_dir):
        raise ValueError('every supplied runtime directory must exist')
    jobs = [job for split in ('pilot', 'profile', 'heldout') for backend in suite['backends']
            for job in plan_jobs(suite, backend, split)]
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return 0
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('output must be a fresh empty directory')
    preserved = [*binaries.values(), *args.preserve_file]
    before = preserved_snapshot(args.freeze_manifest, preserved)
    lock_path = PROFILING / '.measurement.lock'
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as stream:
        stream.write(str(os.getpid()) + '\n')
    try:
        (output / 'raw').mkdir(parents=True)
        source_paths = [Path(__file__), PROFILING / 'run_dense_paired_measurements.py',
            PROFILING / 'run_dense_cuda_measurements.py', PROFILING / 'profile_contract.py',
            SIMULATOR / 'modeling/llm_infer_model/m2/dense_contract.py',
            SIMULATOR / 'modeling/llm_infer_model/m2/dense_cuda_contract.py']
        runtime_files = {path for folder in args.runtime_dir for path in folder.iterdir()
                         if path.name.lower().startswith(('ggml', 'libggml', 'cudart', 'libcudart', 'cublas', 'libcublas', 'nvrtc', 'libnvrtc'))
                         and (path.suffix in ('.dll', '.so', '.dylib') or '.so.' in path.name)}
        source_inventory = record_inventory(source_paths, 'source')
        runtime_inventory = record_inventory(runtime_files, 'supplied runtime')
        suite_inventory = record_inventory([args.suite], 'suite')
        (output / 'source').mkdir()
        executed_source = output / 'source' / 'calibrate.executed.py'
        executed_source.write_bytes(Path(__file__).read_bytes())
        manifest = {'format_version': VERSION, 'suite': suite, 'suite_sha256': sha(args.suite),
            'suite_path': str(args.suite.resolve()),
            'environment_before': environment(), 'historical_assets_before': before,
            'source_sha256': source_inventory, 'runtime_sha256': runtime_inventory,
            'executed_source_snapshot': 'source/calibrate.executed.py',
            'executed_source_snapshot_sha256': sha(executed_source),
            'child_flags_before_by_backend': {backend: observed_child_flags(child_environment(args.runtime_dir, backend))
                                              for backend in suite['backends']},
            'runtime_search_dirs': [str(path.resolve()) for path in args.runtime_dir],
            'runtime_hash_scope': 'supplied matching libraries; resolved loader paths not independently observed',
            'started_at_utc': datetime.now(timezone.utc).isoformat(),
            'planned_jobs': len(jobs), 'hardware_executed': True,
            'power_mode_changed': False, 'other_applications_closed': False,
            'calibration_claimed': False, 'capacity_calibrated': False,
            'native_source_linkage': 'existing executables retained unchanged; no new native compilation'}
        dump(output / 'manifest.json', manifest)
        dump(output / 'plan.json', jobs)
        rows = []
        fitted = None
        for index, job in enumerate(jobs, 1):
            if job['split'] == 'heldout' and fitted is None:
                fitted = fit_models(suite, [row for row in rows if row['split'] == 'profile'])
                dump(output / 'prediction-model.json', fitted)
                manifest['model_sha256_before_heldout'] = sha(output / 'prediction-model.json')
                dump(output / 'manifest.json', manifest)
            print(f"[{index}/{len(jobs)}] {job['backend']} {job['split']} {job['geometry_id']} {job['case']['kind']} seed={job['seed']} repeat={job['repeat']}", flush=True)
            row = run_job(job, binaries[job['backend']].resolve(), args.runtime_dir,
                          output / 'raw' / f'{index:03d}.json', args.timeout_s)
            if row['child_flags'] != manifest['child_flags_before_by_backend'][job['backend']]:
                row.update(status='failed', error='child environment changed from the recorded pre-experiment context')
            row['raw_output'] = 'raw/' + row['raw_output']
            row['quality_flags'] = (['cv_above_predeclared_threshold'] if row.get('maximum_observed_cv', 0) > suite['quality']['cv_threshold'] else [])
            rows.append(row)
            dump(output / 'measurements.json', rows)
            if row['status'] != 'measured':
                print(row['error'], file=sys.stderr)
                break
        provenance_error = None
        try:
            verify_inventory(source_inventory, 'source')
            verify_inventory(runtime_inventory, 'supplied runtime')
            verify_inventory(suite_inventory, 'suite')
        except ValueError as error:
            provenance_error = str(error)
        if provenance_error is not None:
            report = {'format_version': VERSION, 'calibrated': False, 'capacity_calibrated': False,
                      'mixed_or_offload_calibrated': False, 'reason': provenance_error, 'comparisons': []}
        elif len(rows) == len(jobs) and all(row['status'] == 'measured' for row in rows):
            if sha(output / 'prediction-model.json') != manifest['model_sha256_before_heldout']:
                raise ValueError('saved prediction model changed after heldout started')
            report = compare(suite, rows, fitted)
        else:
            report = {'format_version': VERSION, 'calibrated': False, 'capacity_calibrated': False,
                      'mixed_or_offload_calibrated': False, 'reason': 'native observation failed; raw failure preserved', 'comparisons': []}
        dump(output / 'report.json', report)
        after = preserved_snapshot(args.freeze_manifest, preserved)
        if before != after:
            raise ValueError('historical frozen files or native executable changed during experiment')
        manifest.update(completed_jobs=len(rows), measured_jobs=sum(row['status'] == 'measured' for row in rows),
            source_runtime_suite_verified_unchanged=provenance_error is None, provenance_error=provenance_error,
            historical_assets_after=after, historical_assets_unchanged=True,
            environment_after=environment(), calibration_claimed=report['calibrated'],
            finished_at_utc=datetime.now(timezone.utc).isoformat())
        dump(output / 'manifest.json', manifest)
        print('Measured', manifest['measured_jobs'], '/', len(jobs), 'calibrated=', report['calibrated'])
        return 0 if len(rows) == len(jobs) and manifest['measured_jobs'] == len(jobs) and provenance_error is None else 1
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print('strategy_calibration:', error, file=sys.stderr)
        raise SystemExit(1)
