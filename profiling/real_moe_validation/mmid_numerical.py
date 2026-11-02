"""Four fresh, request-held-out comparisons of a single private MMID fix.

There is no performance fitting or policy evaluation. Reference-generated
tokens may enter only predeclared numerical replay/capture controls. Requests
are selected from frozen metadata, never execution times or output equality.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import psutil

from .dataset import read, write
from .numerical_collect import verify_native
from .numerical_v6 import ENV_KEYS
from . import numerical_vectors as vectors
from .run import child_environment, file_sha, require
from .tap_v4 import receipt, sha
from .tap_campaign_v5 import available
from .postcollection import expected_forwards
from profiling.state_calibration import run as common

ROOT = Path(__file__).resolve().parents[2]
ORDER = ('original-generate', 'fixed-generate', 'fixed-replay', 'fixed-capture')
EXTRA_FLAGS = ('GGML_CUDA_GRAPH_OPT', 'GGML_CUDA_DISABLE_FUSION')


def select_requests(suite, excluded_ids):
    """First lexicographic held-out ID per domain/length; no target data read."""
    selected = []
    for domain in ('dialogue', 'code', 'math'):
        for length in ('short', 'long'):
            eligible = [r for r in suite['requests'] if r['domain'] == domain
                        and r['length_class'] == length and r['split'] == 'heldout'
                        and r['id'] not in excluded_ids]
            require(bool(eligible), 'missing independent metadata cell ' + domain + '/' + length)
            row = copy.deepcopy(sorted(eligible, key=lambda r: r['id'])[0])
            require('output_token_ids' not in row, 'selection must not contain target outputs')
            require(row['actual_prompt_tokens'] == len(row['prompt_token_ids']), 'actual tokenizer count differs')
            require(len(row['prompt_token_ids']) + 8 <= 2048, 'unchanged context envelope exceeded')
            selected.append(row)
    require(len({r['id'] for r in selected}) == 6, 'unique six-request selection required')
    return selected


def request_for(template, selected, job, out):
    require(job in ORDER, 'job outside declared four-process order')
    request = copy.deepcopy(template)
    request['mode'] = job.split('-', 1)[1]
    request['requests'] = []
    for index, source in enumerate(selected):
        row = copy.deepcopy(source)
        row.update(id=f'{job}-o{index:02d}', origin_case_id=source['id'],
            comparison_observation_key=source['id'], context_reset_index=index,
            max_output_tokens=8, logit_artifact_path=str(out / 'logits' / job / f'o{index:02d}.f32'))
        row.pop('output_token_ids', None)
        request['requests'].append(row)
    return request


def prepare(out, fix_build, suite_path):
    out, fix_build, suite_path = map(lambda p: Path(p).resolve(), (out, fix_build, suite_path))
    require(not out.exists(), 'new numerical acceptance directory required')
    fixed = read(fix_build / 'BUILD_COMPLETE.json')
    prepared = read(fix_build / 'PREPARED.json')
    require(fixed.get('compiled') is True and fixed.get('original_inputs_unchanged') is True,
            'completed source-bound private fence build required')
    old_build = Path(prepared['observer_build'])
    driver_build = Path(prepared['numerical_build'])
    old = read(ROOT / 'build/real-moe-validation-20261027/numerical-diagnostic-v6/manifest.json')
    template = read(ROOT / 'build/real-moe-validation-20261027/numerical-diagnostic-v6/requests/normal-SSLS.json')
    excluded = {r['origin_case_id'] for r in template['requests']}
    selected = select_requests(read(suite_path), excluded)
    environment = {key: os.environ.get(key) for key in ENV_KEYS}
    environment.update(NVIDIA_TF32_OVERRIDE='0', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1')
    require(environment['GGML_CUDA_DISABLE_GRAPHS'] is None, 'default graph mode must remain unset')
    require(all(os.environ.get(key) is None for key in EXTRA_FLAGS), 'unexported native graph/fusion override unsupported')
    template.update(expected_runtime_environment=environment)
    original_complete = read(old_build / 'BUILD_COMPLETE.json')
    driver_complete = read(driver_build / 'BUILD_COMPLETE.json')
    bindings = dict(original=dict(binary=driver_complete['binary'],
        runtime=[r for r in original_complete['binaries'] if Path(r['path']).suffix == '.dll'],
        runtime_dirs=[str(old_build / 'bin'), 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin']),
        fixed=dict(binary=next(r for r in fixed['binaries'] if Path(r['path']).suffix == '.exe'),
        runtime=[r for r in fixed['binaries'] if Path(r['path']).suffix == '.dll'],
        runtime_dirs=[str(fix_build / 'bin'), 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin']))
    require(bindings['original']['binary']['sha256'] == bindings['fixed']['binary']['sha256'], 'host driver changed across comparison')
    old_dll = {Path(r['path']).name: r['sha256'] for r in bindings['original']['runtime']}
    new_dll = {Path(r['path']).name: r['sha256'] for r in bindings['fixed']['runtime']}
    require(set(old_dll) == set(new_dll) and [name for name in old_dll if old_dll[name] != new_dll[name]] == ['ggml-cuda.dll'],
            'only CUDA DLL may differ')
    out.mkdir(parents=True)
    jobs = []
    for job in ORDER:
        request = request_for(template, selected, job, out)
        path = out / 'requests' / (job + '.json')
        (out / 'logits' / job).mkdir(parents=True)
        if request['mode'] == 'generate':
            write(path, request)
        else:
            # Template is frozen now; only actual reference output IDs may be
            # inserted after its predeclared first process, never chosen later.
            write(out / 'request-templates' / (job + '.json'), request)
        jobs.append(dict(job_id=job, binding=job.split('-', 1)[0],
            request=receipt(path) if request['mode'] == 'generate' else None,
            template=receipt(out / 'request-templates' / (job + '.json')) if request['mode'] != 'generate' else None,
            expected_request_schema='real-moe-numerical-request-6', expected_result_schema='real-moe-numerical-native-6',
            expected_source_sha256=template['source_sha256']))
    sources = []
    for path in (Path(__file__), Path(__file__).with_name('test_mmid_numerical.py'),
                 Path(__file__).with_name('numerical_collect.py'), Path(__file__).with_name('numerical_vectors.py'),
                 Path(__file__).with_name('tap_audit_v5.py'), Path(__file__).with_name('run.py'),
                 Path(__file__).with_name('postcollection.py'), ROOT / 'profiling/state_calibration/run.py'):
        target = out / 'executed_sources' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        sources.append(dict(source=receipt(path), archived=receipt(target)))
    manifest = dict(schema='real-moe-mmid-numerical-campaign-1', prepared_at_utc=datetime.now(timezone.utc).isoformat(),
        fixed_order=list(ORDER), jobs=jobs, bindings=bindings, sources=sources,
        fix_build=receipt(fix_build / 'BUILD_COMPLETE.json'), driver_build=receipt(driver_build / 'BUILD_COMPLETE.json'),
        source_suite=receipt(suite_path), source_smoke_manifest=old['source_smoke_manifest'],
        source_greedy_trajectory=old['source_greedy_trajectory'], old_failure_preserved=old['old_failure_preserved'],
        model=old['model'], capacity_policy=old['capacity_policy'], native_guard_bytes=old['native_guard_bytes'],
        parent_preflight_bytes=old['parent_preflight_bytes'], resident_peak_limit_bytes=old['resident_peak_limit_bytes'],
        selection=[dict(id=r['id'], domain=r['domain'], length_class=r['length_class'],
                        prompt_sha256=r['prompt_sha256'], actual_prompt_tokens=r['actual_prompt_tokens']) for r in selected],
        selection_rule='first lexicographic heldout ID per dialogue/code/math x short/long, excluding prior numerical calibration IDs',
        selection_uses_outputs_or_times=False, raw_expected_environment=environment,
        additional_parent_environment={key: None for key in EXTRA_FLAGS},
        workload=dict(processes=4, requests_per_process=6, max_outputs_per_request=8,
            capacity_limits_unchanged=True, retries=0, teacher_forcing='actual original-generate IDs only in fixed replay/capture controls'),
        numerical_gate='exact full F32 byte equality under verified equal actual-input prefixes; every mismatch retained',
        forced_token_equality_is_numeric_evidence=False, teacher_forcing_used_as_policy_input=False,
        whole_model_performance_accuracy_evaluated=False,
        native_runtime_DLL_scope_boilerplate_inapplicable_to_fixed_build=True,
        actual_DLL_provenance='per-job binding receipts override unchanged-driver diagnostic boilerplate')
    write(out / 'manifest.json', manifest)
    return dict(manifest=receipt(out / 'manifest.json'), requests=manifest['selection'], hardware_executed=False)


def freeze(out, tests):
    out, tests = Path(out).resolve(), Path(tests).resolve()
    evidence = read(tests)
    require(evidence['returncode'] == 0 and evidence['hardware_executed'] is False, 'passing software-only tests required')
    write(out / 'FREEZE_BARRIER.json', dict(schema='real-moe-mmid-numerical-freeze-1',
        manifest=receipt(out / 'manifest.json'), tests=receipt(tests), collection_authorized=False,
        root_authorization_required=True, thresholds_changed=False, prior_failure_preserved=True))
    return receipt(out / 'FREEZE_BARRIER.json')


def authorization(out):
    manifest = read(out / 'manifest.json')
    root = read(out / 'ROOT_AUTHORIZATION_RECEIPT.json')
    barrier = read(out / 'FREEZE_BARRIER.json')
    require(root.get('collection_authorized') is True and root.get('manifest_sha256') == sha(out / 'manifest.json')
        and root.get('freeze_barrier_sha256') == sha(out / 'FREEZE_BARRIER.json'), 'exact root collection authorization required')
    require(barrier['manifest']['sha256'] == sha(out / 'manifest.json'), 'manifest changed after freeze')
    return manifest


def derived_replay_request(template, reference):
    """Only actual prior reference token IDs may change a frozen template."""
    request = copy.deepcopy(template)
    actual = {row['origin_case_id']: row for row in reference['results']}
    require(len(actual) == len(reference['results']) == len(request['requests']), 'unique complete reference origin inventory required')
    require(set(actual) == {row['origin_case_id'] for row in request['requests']}, 'reference origin set differs')
    for row in request['requests']:
        source = actual[row['origin_case_id']]
        require(row['prompt_token_ids'] == source['prompt_token_ids'], 'reference prompt differs')
        outputs = source['output_token_ids']
        require(type(outputs) is list and 1 <= len(outputs) <= row['max_output_tokens']
            and all(type(value) is int and value >= 0 for value in outputs), 'bounded actual output-token stream required')
        row['output_token_ids'] = list(outputs)
    return request


def verified_pair_spec(out, spec):
    """Do not let a mutable replay-plan file redefine the frozen job binding."""
    plan = read(out / 'pair-plans' / (spec['job_id'] + '.json'))
    require(set(plan) == set(spec) | {'actual_reference', 'teacher_forcing_used_as_policy_input'}, 'pair-plan fields differ')
    require(all(plan[key] == value for key, value in spec.items() if key != 'request'), 'pair plan changed frozen job metadata')
    require(plan['teacher_forcing_used_as_policy_input'] is False, 'numerical replay cannot become a policy input')
    require(sha(spec['template']['path']) == spec['template']['sha256'], 'frozen replay template changed')
    execution = read(out / 'jobs/original-generate/execution.json')
    require(execution['status'] == 'completed' and plan['actual_reference'] == execution['native'], 'pair reference differs from acquired first process')
    require(sha(execution['native']['path']) == execution['native']['sha256'], 'actual reference changed')
    require(Path(plan['request']['path']).resolve() == (out / 'requests' / (spec['job_id'] + '.json')).resolve(), 'replay request path differs')
    require(sha(plan['request']['path']) == plan['request']['sha256'], 'derived request changed')
    expected = derived_replay_request(read(spec['template']['path']), read(execution['native']['path']))
    require(read(plan['request']['path']) == expected, 'derived request exceeds actual-reference-token-only change')
    return dict(spec, request=plan['request'])


def make_pairs(out):
    out = Path(out).resolve()
    manifest = authorization(out)
    execution = read(out / 'jobs/original-generate/execution.json')
    require(execution['status'] == 'completed', 'reference must complete before token replay derivation')
    require(sha(execution['native']['path']) == execution['native']['sha256'], 'acquired reference changed')
    reference = read(execution['native']['path'])
    for spec in manifest['jobs'][2:]:
        require(sha(spec['template']['path']) == spec['template']['sha256'], 'frozen replay template changed')
        request = derived_replay_request(read(spec['template']['path']), reference)
        path = out / 'requests' / (spec['job_id'] + '.json')
        write(path, request)
        write(out / 'pair-plans' / (spec['job_id'] + '.json'), dict(spec, request=receipt(path),
            actual_reference=receipt(out / 'jobs/original-generate/result.json'),
            teacher_forcing_used_as_policy_input=False))


def collect(out, job):
    out = Path(out).resolve()
    manifest = authorization(out)
    require(job in ORDER, 'job outside fixed order')
    for previous in ORDER[:ORDER.index(job)]:
        require(read(out / 'jobs' / previous / 'execution.json')['status'] == 'completed', 'earlier process failed/missing; no retry')
    spec = next(r for r in manifest['jobs'] if r['job_id'] == job)
    if spec['request'] is None:
        spec = verified_pair_spec(out, spec)
    request = read(spec['request']['path'])
    binding = manifest['bindings'][spec['binding']]
    folder = out / 'jobs' / job
    require(not folder.exists(), 'no process overwrite/retry')
    folder.mkdir(parents=True)
    write(folder / 'request.json', request)
    command = [binding['binary']['path'], str(folder / 'request.json'), str(folder / 'result.json')]
    record = dict(schema='real-moe-mmid-numerical-execution-1', job_id=job, status='failed', command=command,
        binding=spec['binding'], binary=binding['binary'], runtime=binding['runtime'], request=receipt(folder / 'request.json'),
        capacity_policy=manifest['capacity_policy'], timing_qualified_as_bare=False)
    process = None
    handles, samples = [], []
    low = 0
    with common.measurement_lock():
        try:
            resources = [spec['request'], manifest['fix_build'], manifest['driver_build'], manifest['source_suite'],
                manifest['source_smoke_manifest'], manifest['source_greedy_trajectory'], manifest['old_failure_preserved'],
                manifest['capacity_policy'], binding['binary'], *binding['runtime'],
                *[r[k] for r in manifest['sources'] for k in ('source', 'archived')]]
            for resource in resources:
                require(sha(resource['path']) == resource['sha256'], 'frozen resource changed: ' + resource['path'])
            require(file_sha(manifest['model']['path']) == manifest['model']['sha256'], 'model content changed')
            require(all(os.environ.get(key) is None for key in EXTRA_FLAGS), 'graph/fusion override changed')
            env = child_environment(binding['runtime_dirs'], request)
            env.update(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1')
            require({key: env.get(key) for key in ENV_KEYS} == manifest['raw_expected_environment'], 'raw runtime environment changed')
            record.update(raw_environment={key: env.get(key) for key in (*ENV_KEYS, *EXTRA_FLAGS)},
                available_before_launch_bytes=available())
            require(record['available_before_launch_bytes'] >= manifest['parent_preflight_bytes'], 'unchanged parent capacity guard not met')
            process, stdout, stderr, affinity = common.launch(command, env, folder / 'stdout.txt', folder / 'stderr.txt', request['expected_process_mask'])
            handles = [stdout, stderr]
            record.update(process_id=process.pid, process_affinity=affinity)
            observed = psutil.Process(process.pid)
            start = time.monotonic()
            write(folder / 'STARTED.json', dict(pid=process.pid, at_utc=datetime.now(timezone.utc).isoformat()))
            print(json.dumps(dict(started=job, pid=process.pid)), flush=True)
            with (folder / 'observer.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
                while process.poll() is None:
                    sample = dict(time_ns=time.perf_counter_ns(), available_bytes=available())
                    try:
                        sample['memory'] = observed.memory_info()._asdict()
                    except (psutil.NoSuchProcess, psutil.AccessDenied) as error:
                        sample['observer_error'] = type(error).__name__
                    samples.append(sample)
                    stream.write(json.dumps(sample) + '\n')
                    stream.flush()
                    low = low + 1 if sample['available_bytes'] < 512 * 1024**2 else 0
                    if low >= 2 or sample.get('memory', {}).get('peak_wset', 0) > manifest['resident_peak_limit_bytes'] or time.monotonic() - start > 300:
                        record['abort_reason'] = 'unchanged resident/available guard or fixed300s timeout'
                        process.terminate()
                        break
                    try:
                        process.wait(timeout=.2)
                    except subprocess.TimeoutExpired:
                        pass
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            record.update(returncode=process.returncode, process_exited_verified=True)
            require(process.returncode == 0, 'native failed; no retry')
            record['verification'] = verify_native(read(folder / 'result.json'), request, process.pid, spec)
            record.update(status='completed', native=receipt(folder / 'result.json'))
        except BaseException as error:
            record['error'] = repr(error)
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            if process is not None:
                record.update(returncode=process.returncode, process_exited_verified=process.poll() is not None)
        finally:
            for handle in handles:
                handle.close()
            record.update(minimum_observed_available_bytes=min((r['available_bytes'] for r in samples), default=None),
                maximum_observed_peak_wset_bytes=max((r.get('memory', {}).get('peak_wset', 0) for r in samples), default=None))
            write(folder / 'execution.json', record)
    print(json.dumps(dict(job=job, status=record['status'], error=record.get('error'))), flush=True)
    return record


def collect_all(out):
    out = Path(out).resolve()
    authorization(out)
    write(out / 'PIPELINE_STARTED.json', dict(order=list(ORDER), retries=0))
    for job in ORDER:
        if job == 'fixed-replay':
            make_pairs(out)
        record = collect(out, job)
        if record['status'] != 'completed':
            write(out / 'STOPPED.json', dict(failed=job, remaining=list(ORDER[ORDER.index(job) + 1:]), retry_performed=False))
            return dict(completed=False, failed=job)
    write(out / 'ACQUISITION_COMPLETE.json', dict(processes=4, request_observations=24,
        all_processes_exited=True, numerical_comparison_pending=True, performance_accuracy_evaluated=False))
    return dict(completed=True, processes=4)


def actual_prefix(row, output_index):
    require(type(output_index) is int and 0 <= output_index < row['output_count'], 'output correspondence outside row')
    forwards = expected_forwards(len(row['prompt_token_ids']), row['output_count'])
    boundaries = [i for i, f in enumerate(forwards) if f['output_rows']]
    return [r['actual_input_rows_sha256'] for r in row['actual_batch_inputs'][:boundaries[output_index] + 1]]


def compare_rows(left, right):
    require(left['origin_case_id'] == right['origin_case_id'], 'unrelated request origin')
    lm, lr, lv = vectors._load(left['full_logits_artifact'])
    rm, rr, rv = vectors._load(right['full_logits_artifact'])
    require(lm['columns'] == rm['columns'], 'vocabulary correspondence differs')
    columns = lm['columns']
    rows = []
    for index in range(min(lm['rows'], rm['rows'])):
        same = actual_prefix(left, index) == actual_prefix(right, index)
        result = dict(output_index=index, actual_input_prefix_equal=same)
        if same:
            start, end = index * columns, (index + 1) * columns
            result.update(vectors._metrics(lr[start * 4:end * 4], rr[start * 4:end * 4], lv[start:end], rv[start:end]))
            result['greedy_equal'] = left['sampled_greedy_token_ids'][index] == right['sampled_greedy_token_ids'][index]
            count = len(actual_prefix(left, index))
            result['graph_prefix_equal'] = ([g['graph_partition_fingerprint_sha256'] for g in left['graph_observations'][:count]]
                == [g['graph_partition_fingerprint_sha256'] for g in right['graph_observations'][:count]])
        else:
            result.update(comparison_status='unsupported_unequal_actual_input_prefix', bitwise_equal=None)
        rows.append(result)
    return dict(origin_case_id=left['origin_case_id'], left_outputs=lm['rows'], right_outputs=rm['rows'], rows=rows,
        missing_output_correspondences=abs(lm['rows'] - rm['rows']), vectors_filled_or_modified=False,
        first_bitwise_divergence=next((r['output_index'] for r in rows if r.get('bitwise_equal') is False), None))


def legacy_audit_view(native):
    """Explicit schema6 host-ledger projection for the unchanged strict v5 auditor.

    Raw records are never edited. Only observer artifact locations/IDs and the
    additive version label are normalized; execution inputs and outputs remain.
    """
    result = copy.deepcopy(native)
    require(result['schema'] == 'real-moe-numerical-native-6', 'v6 input required')
    result['schema'] = 'real-moe-tap-native-5'
    result['request']['schema'] = 'real-moe-tap-request-5'
    for row in result['request']['requests']:
        row['id'] = row['origin_case_id']
        row.pop('logit_artifact_path', None)
    for row in result['results']:
        row['id'] = row['origin_case_id']
    return result


def analyze(out):
    out = Path(out).resolve()
    manifest = authorization(out)
    native = {}
    for spec in manifest['jobs']:
        execution = read(out / 'jobs' / spec['job_id'] / 'execution.json')
        require(execution['status'] == 'completed', 'all four declared processes required; no filtering')
        require(sha(execution['native']['path']) == execution['native']['sha256'], 'native changed after acquisition')
        native[spec['job_id']] = read(execution['native']['path'])
        for row in native[spec['job_id']]['results']:
            vectors.inspect(row['full_logits_artifact'], row['logit_observations'])
    comparisons = []
    for left, right in (('original-generate', 'fixed-generate'), ('original-generate', 'fixed-replay'), ('fixed-replay', 'fixed-capture')):
        pairs = [compare_rows(a, b) for a, b in zip(native[left]['results'], native[right]['results'])]
        comparisons.append(dict(left_job=left, right_job=right, requests=pairs,
            qualified_output_rows=sum(r['actual_input_prefix_equal'] for p in pairs for r in p['rows']),
            bitwise_equal_output_rows=sum(r.get('bitwise_equal') is True for p in pairs for r in p['rows']),
            graph_prefix_equal_output_rows=sum(r.get('graph_prefix_equal') is True for p in pairs for r in p['rows']),
            unqualified_output_rows=sum(not r['actual_input_prefix_equal'] for p in pairs for r in p['rows']),
            missing_output_correspondences=sum(p['missing_output_correspondences'] for p in pairs)))
    from .tap_audit_v5 import pair_audit
    observer = pair_audit(legacy_audit_view(native['fixed-replay']), legacy_audit_view(native['fixed-capture']))
    result = dict(schema='real-moe-mmid-independent-numerical-evaluation-1',
        completed_processes=4, completed_request_observations=24, comparisons=comparisons,
        strict_fixed_replay_capture_audit=observer, prior_failure_preserved=manifest['old_failure_preserved'],
        exact_vector_gate_unchanged=True, original_race_cause_of_old_logits_proven=False,
        whole_model_performance_accuracy_evaluated=False, performance_parameters_fitted=False,
        comparison_sets_not_independent_samples=True, raw_records_modified=False,
        legacy_audit_projection='schema6 to5 additive host-ledger label; per-job IDs to unchanged origin IDs; remove only artifact path',
        gate_passed=observer['status'] == 'passed' and all(c['unqualified_output_rows'] == 0 and c['missing_output_correspondences'] == 0
            and c['qualified_output_rows'] == c['bitwise_equal_output_rows'] == c['graph_prefix_equal_output_rows'] for c in comparisons))
    write(out / 'NUMERICAL_EVALUATION.json', result)
    return dict(output=receipt(out / 'NUMERICAL_EVALUATION.json'), gate_passed=result['gate_passed'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'freeze', 'collect-all', 'collect', 'analyze'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--fix-build')
    parser.add_argument('--suite')
    parser.add_argument('--tests')
    parser.add_argument('--job', choices=ORDER)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.out, args.fix_build, args.suite)
    elif args.command == 'freeze':
        result = freeze(args.out, args.tests)
    elif args.command == 'analyze':
        result = analyze(args.out)
    elif args.command == 'collect':
        result = collect(args.out, args.job)
    else:
        result = collect_all(args.out)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
