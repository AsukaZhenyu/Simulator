"""Read-only independent evidence audit; no fitting/model/qualification imports.

This auditor rebuilds logical reuse coordinates from declarations, recomputes
action calibration means, composes frozen predictions, and checks native ledgers
and reported acceptance metrics. A failed scientific gate is a valid outcome.
Passing this audit establishes evidence consistency, not a physical mechanism or
accuracy outside the finite heldout cases.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


ROOT = Path(__file__).resolve().parents[2]
OLD_REPORT = ROOT.parent/'paper-write/第八教学周/experiment-reports/dac-accuracy-repair-20261026/REPORT_zh_v2.html'
OLD_REPORT_SHA = '061738066a4bec44dc22381ffeb93558168260b5903e35407fb235e12cbd725f'
PROVIDERS = ('old_hot', 'context_static', 'history_conditional')
QUALITY = dict(max_process_mean_ape=.10, max_trial_average_p90_ape=.20,
    max_process_trial_cv=.10, max_observer_relative_mean_difference=.05)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def close(actual, expected):
    if actual is None or expected is None:
        return actual is expected
    return (type(actual) in (int, float) and type(expected) in (int, float)
            and math.isfinite(actual) and math.isfinite(expected)
            and abs(actual-expected) <= max(1e-6, abs(expected)*1e-12))


def numeric_equal(actual, expected):
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            numeric_equal(a, b) for a, b in zip(actual, expected))
    return (type(actual) in (int, float) and type(expected) in (int, float)
        and math.isfinite(actual) and math.isfinite(expected)
        and abs(actual-expected) <= 5e-5*(1+abs(expected)))


def higher_p90(values):
    values = sorted(values)
    if not values:
        raise ValueError('empty independent order statistic')
    return values[math.ceil((len(values)-1)*.9)]


def sample_cv(values):
    return statistics.stdev(values)/statistics.mean(values) if len(values)>1 else None


def extract_semantic(program, index):
    """Independent extraction; no model helper or source IDs in the result."""
    a = program['actions'][index]
    tensors = {t['id']: t for t in program['graph']['workload']['tensors']}
    def geometry(tid):
        return {k: tensors[tid][k] for k in ('dtype', 'ne', 'nb', 'storage_bytes')}
    if a['kind'] == 'COPY':
        return dict(kind='COPY', src=a['src'], dst=a['dst'], tensor=geometry(a['target_id']))
    if a['kind'] == 'COMPUTE':
        b = next(b for b in program['bindings'] if b['operation_id'] == a['target_id'])
        return dict(kind='COMPUTE', device=a['device'], semantic_op=b['kind'],
            shape=b['shape'], op_params=b['op_params'],
            operands={role: geometry(tid) for role, tid in b['operands'].items()},
            outputs={role: geometry(tid) for role, tid in b['outputs'].items()},
            history=[dict(tokens=v['tokens'], first_position=v['first_position'], geometry=geometry(v['id'])) for v in b['history']],
            new_kv=[dict(tokens=v['tokens'], first_position=v['first_position'], geometry=geometry(v['id'])) for v in b['new_kv']],
            config=program['physical']['config'], threads=program['physical']['threads'],
            port_boundary=program['boundary_contract'])
    if a['kind'] in ('ADVANCE', 'EVICT'):
        return dict(kind=a['kind'], native_behavior='logical_control_no_allocation_or_wait')
    raise ValueError('unrecognized declared action')


def independent_coordinates(program, scope, *, repeats=8):
    """Logical distinct DRAM footprint; no measured/predicted time is needed.

    A full logical prime precedes each modeled block. Native reset writes and
    physical arena/cache aliasing are intentionally outside this hypothesis.
    """
    tensors = {t['id']: t for t in program['graph']['workload']['tensors']}
    bindings = {b['operation_id']: b for b in program['bindings']}
    last_touch = {}
    event = 0
    def advance(a):
        nonlocal event
        event += 1
        touched = set()
        if a['kind'] == 'COPY' and ('dram' in (a['src'], a['dst'])):
            touched.add(a['target_id'])
        elif a['kind'] == 'COMPUTE' and a['device'] == 'cpu':
            b = bindings[a['target_id']]
            touched.update(b['operands'].values())
            touched.update(b['outputs'].values())
            touched.update(v['id'] for v in b['history'])
            touched.update(v['id'] for v in b['new_kv'])
        for tid in touched:
            last_touch[tid] = event
    for a in program['actions']:
        advance(a)
    rows = []
    for invocation in range(repeats):
        for index, a in enumerate(program['actions']):
            semantic = extract_semantic(program, index)
            static = dict(semantic=semantic, condition={}, scope_sha256=scope)
            condition = {}
            if a['kind'] == 'COPY' and a['src'] == 'dram' and a['dst'] == 'vram':
                tid = a['target_id']
                previous = last_touch.get(tid)
                if previous is None:
                    bucket = 'unknown'
                else:
                    amount = sum(tensors[other]['storage_bytes'] for other, touched in last_touch.items()
                                 if other != tid and touched > previous)
                    bucket = 'zero' if amount == 0 else str(2**(amount.bit_length()-1))
                condition = {'h2d_source_reuse_bucket': bucket}
            rows.append(dict(invocation=invocation, action_index=index, static=static,
                             conditional=dict(semantic=semantic, condition=condition, scope_sha256=scope)))
            advance(a)
    return rows


class Auditor:
    def __init__(self):
        self.count = 0
        self.sections = defaultdict(int)

    def check(self, condition, section, message):
        self.count += 1
        self.sections[section] += 1
        if not condition:
            raise ValueError(section + ': ' + message)

    def receipt(self, value, section='hashes'):
        self.check(Path(value['path']).is_file(), section, 'missing evidence file: '+value['path'])
        self.check(sha(value['path']) == value['sha256'], section, 'changed evidence file: '+value['path'])


def audit_native(audit, native, request, program, oracle, pid):
    """Validate original QPC, action phases and independent numerical ports."""
    c = lambda condition, message: audit.check(condition, 'native', message)
    c(native['schema'] == 'h2d-history-native-1' and native['request'] == request, 'schema/request differs')
    c(native['source_sha256'] == request['source_sha256'] and native['caller_source_sha256_match'] is True,
      'native source SHA differs')
    c(native['process_id'] == pid and native['program_sha256'] == program['program_sha256'], 'PID/program differs')
    c(native['actions'] == program['actions'], 'mathematical action sequence differs')
    c(native['mode'] == request['mode'] and native['instrumentation'] == request['instrumentation']
      and native['memory_mode'] == request['memory_mode'], 'measurement mode differs')
    c(native['history_intervention'] == request.get('history_intervention', 'none'), 'intervention differs')
    thread, process = native['main_thread_affinity'], native['process_affinity']
    c(thread['requested_mask'] == thread['observed_mask'] == thread['final_mask'] == 4
      and thread['restored'] is True, 'owned main thread affinity differs')
    c(process['requested_mask'] == process['observed_mask'] == process['final_mask'] == 4294967295
      and process['enforced'] is True and process['applied_before_backend_initialization'] is True,
      'owned process affinity differs')
    c(native['observed_runtime_flags'] == dict(GGML_CUDA_DISABLE_GRAPHS='1', NVIDIA_TF32_OVERRIDE='0',
      matmul_precision='GGML_PREC_F32'), 'runtime flags differ')
    c(native['diagnostic_state_used_as_future_input'] is False
      and native['whole_program_time_used_as_parameter'] is False, 'future target input flag differs')
    prep = native['preparation_contract']
    repeats, trials = request['sampling']['repeats'], request['sampling']['trials']
    c(prep['reset_before_batch'] is True and prep['untimed_complete_prime_passes'] == 1
      and prep['fixed_input_replay'] is True and prep['blocks_per_trial'] == 1
      and prep['timed_replays_per_batch'] == repeats, 'warm preparation differs')
    registration = native['arena_registration']
    c(registration['mode'] == request['memory_mode'] and registration['restored'] is True,
      'owned host-memory registration was not restored')
    if registration['during'].get('status') == 'observed':
        category = 'registered_host' if request['memory_mode'] == 'registered' else 'unregistered_pageable'
        c(registration['before']['unregistered_pageable'] is True
          and registration['during'][category] is True and registration['after']['unregistered_pageable'] is True,
          'host memory category differs')
    else:
        c(request['memory_mode'] == 'pageable' and registration['during']['status'] == 'not_queried'
          and registration['during']['host_cpu_backend'] is True, 'unsupported unknown registration category')
    proof = native['verification_native']
    c(proof['copy_values_equal'] is True and native['native_repeat_consistency']['passed'] is True,
      'copy/replay numerical consistency failed')
    c(set(proof['ports']) == set(oracle), 'independent numerical port inventory differs')
    for key, expected in oracle.items():
        c(numeric_equal(proof['ports'][key], expected), 'independent numeric port mismatch: '+key)
    hz = native['clock']['frequency_hz']
    c(type(hz) is int and hz > 0 and native['physical_clock']['frequency_hz'] == hz, 'clock differs')
    c(len(native['trials']) == trials, 'trial count differs')
    previous_end = -1
    for trial, block in enumerate(native['trials']):
        c(block['trial'] == trial and block['block'] == 0 and block['repeats'] == repeats, 'trial/replay order differs')
        begin, end = block['begin_tick'], block['end_tick']
        c(end > begin >= previous_end, 'QPC block chronology differs')
        c(abs((end-begin)*1e9/hz-block['wall_ns']) <= 1.01, 'QPC block ledger differs')
        c(block['begin_ns'] == begin*1000000000//hz and block['end_ns'] == end*1000000000//hz,
          'absolute QPC floor-ns conversion differs')
        state = block['preblock_state']
        c(state['sample_tick'] <= begin and state['outside_timed_interval'] is True
          and state['derived_from_target_duration'] is False, 'precondition provenance differs')
        c(state['cpu_utilization'] is None and state['gpu_utilization'] is None
          and state['gpu_clock_mhz'] is None and state['gpu_temperature_c'] is None
          and state['memory_capacity_is_bandwidth'] is False, 'unknown sensors were inferred')
        stamped = request['instrumentation'] != 'bare'
        c(block['actual_stamped'] is stamped, 'observer indicator differs')
        indices = list(range(len(program['actions']))) if request['mode'] == 'full' else [request['target_action_index']]
        expected = [(r, i) for r in range(repeats) for i in indices] if stamped else []
        stamps = block['action_intervals']
        c(len(stamps) == len(expected), 'action stamp count differs')
        previous_action_end = begin
        for (invocation, index), item in zip(expected, stamps):
            c(item['trial'] == trial and item['invocation'] == invocation and item['action_index'] == index,
              'action identity/replay order differs')
            first, last = item['begin_tick'], item['end_tick']
            c(previous_action_end <= first <= last <= end, 'action chronology/batch containment differs')
            c(abs((last-first)*1e9/hz-item['wall_ns']) <= 1.01, 'action QPC duration differs')
            c(item['begin_ns'] == first*1000000000//hz and item['end_ns'] == last*1000000000//hz,
              'action absolute QPC conversion differs')
            if program['actions'][index]['kind'] == 'COPY':
                submitted, gpu = item['copy_return_tick'], item['gpu_return_tick']
                c(first <= submitted <= gpu <= last, 'copy phase order differs')
                c(item['submit_copy_api_ns'] == item['copy_call_ns'], 'inclusive copy API alias differs')
                c(abs((submitted-first)*1e9/hz-item['copy_call_ns']) <= 1.01
                  and abs((gpu-submitted)*1e9/hz-item['gpu_sync_ns']) <= 1.01
                  and abs((last-gpu)*1e9/hz-item['cpu_sync_ns']) <= 1.01, 'copy phase duration ledger differs')
                c(abs(item['copy_call_ns']+item['gpu_sync_ns']+item['cpu_sync_ns']-item['wall_ns']) <= 2,
                  'copy inclusive phase partition differs')
            previous_action_end = last
        previous_end = end
    if request['instrumentation'] == 'cupti':
        capture = native['cupti']
        retained = (capture['cupti_dropped_records'] == capture['local_record_overflow'] == 0
                    and not capture['callback_errors'] and capture['timestamp_callback_failed'] is False)
        c(capture['all_requested_records_retained'] is bool(retained), 'CUPTI retention flag differs')
        integrity = retained and all(r['kind'] == 'external_correlation' or r['timestamp_valid']
                                    for r in capture['records'])
        c(native['capture_integrity_pass'] is bool(integrity)
          and native['physical_activity_usable_for_diagnosis'] is bool(integrity),
          'CUPTI capture quality flag differs; an honestly failed capture is not fabricated evidence')


def recompute_case(prediction, records):
    """Independent direct equations; scientific gate failures are returned."""
    predictions = [float(Fraction(v)) for v in prediction['trial_mean_replay_ns']]
    predicted_mean = float(Fraction(prediction['mean_replay_ns']))
    if len(predictions) != 16 or not close(statistics.mean(predictions), predicted_mean):
        raise ValueError('frozen per-block/process prediction inconsistency')
    primary = [r for r in records if r['instrumentation'] == 'bare']
    observer = [r for r in records if r['instrumentation'] == 'stamped']
    if len(primary) != 4 or len(observer) != 4:
        raise ValueError('declared primary/observer denominator differs')
    means, cvs, observer_cvs, errors, effects = [], [], [], [], []
    process = {}
    for r in records:
        if r['status'] != 'completed':
            process[r['job_id']] = None
            continue
        values = r['values']
        average = statistics.mean(values)
        signed = [p-a for p, a in zip(predictions, values)]
        ape = [abs(e)/a for e, a in zip(signed, values)]
        cv = sample_cv(values)
        early, late = statistics.mean(values[:8]), statistics.mean(values[8:])
        process[r['job_id']] = dict(mean_ns=average, trial_cv=cv,
            mean_ape=abs(predicted_mean-average)/average,
            signed_mean_error_ns=predicted_mean-average,
            absolute_error_mean_ns=statistics.mean(abs(v) for v in signed),
            root_mean_square_error_ns=math.sqrt(statistics.mean(v*v for v in signed)),
            block_ape_p90=higher_p90(ape), early_mean_ns=early, late_mean_ns=late,
            late_over_early_minus_one=late/early-1)
        if r['instrumentation'] == 'bare':
            means.append(average)
            cvs.append(cv)
            errors.extend(ape)
        else:
            observer_cvs.append(cv)
    for round_id in range(4):
        a = next(r for r in primary if r['round'] == round_id)
        b = next(r for r in observer if r['round'] == round_id)
        effects.append(None if a['status'] != 'completed' or b['status'] != 'completed' else
            statistics.mean(b['values'])/statistics.mean(a['values'])-1)
    complete = all(r['status'] == 'completed' for r in records)
    max_ape = max((process[r['job_id']]['mean_ape'] for r in primary if r['status'] == 'completed'), default=None)
    pooled = higher_p90(errors) if errors else None
    gates = dict(all_declared_processes_completed=complete,
        numerical_oracles=all(r.get('numeric', False) for r in records),
        each_primary_process_mean_ape=bool(len(means) == 4 and max_ape <= QUALITY['max_process_mean_ape']),
        pooled_primary_block_p90_ape=bool(len(errors) == 64 and pooled <= QUALITY['max_trial_average_p90_ape']),
        each_primary_process_cv=bool(len(cvs) == 4 and all(v <= QUALITY['max_process_trial_cv'] for v in cvs)),
        each_observer_process_cv=bool(len(observer_cvs) == 4 and all(v <= QUALITY['max_process_trial_cv'] for v in observer_cvs)),
        each_paired_observer_relative_mean_difference=all(v is not None and abs(v) <= QUALITY['max_observer_relative_mean_difference'] for v in effects))
    original = {k: v for k, v in gates.items() if k != 'each_observer_process_cv'}
    return dict(gates=gates, fully_qualified=all(original.values()),
        qualified_original_primary_cv_gate=all(original.values()),
        qualified_conservative_all_process_cv_gate=all(gates.values()),
        failed_gates=[k for k, v in original.items() if not v],
        failed_conservative_gates=[k for k, v in gates.items() if not v],
        predicted_mean_ns=predicted_mean, observed_primary_mean_ns=statistics.mean(means) if means else None,
        maximum_primary_process_mean_ape=max_ape, pooled_primary_block_ape_p90=pooled,
        maximum_primary_process_cv=max(cvs, default=None), maximum_observer_process_cv=max(observer_cvs, default=None),
        maximum_paired_observer_difference=max((abs(v) for v in effects if v is not None), default=None),
        effects=effects, processes=process)


def audit(out):
    out = Path(out).resolve()
    a = Auditor()
    m, barrier, frozen, model, summary = [read(out/name) for name in
        ('manifest.json', 'frozen-barrier.json', 'predictions-frozen.json', 'model-frozen.json', 'summary.json')]
    a.check(sha(OLD_REPORT) == OLD_REPORT_SHA, 'preservation', 'old report v2 was changed')
    a.check(m['protocol']['quality'] == summary['quality'] == QUALITY, 'protocol', 'thresholds changed')
    for name, expected in barrier['files'].items():
        a.check(sha(out/name) == expected, 'frozen', 'frozen artifact changed: '+name)
    for value in m['sources']:
        a.receipt(value['source'])
        a.receipt(value['archived'])
    native_sources = [v['source'] for v in m['sources']
                      if Path(v['source']['path']).name == 'history_probe.cpp']
    a.check(len(native_sources) == 1, 'protocol', 'unique frozen native source missing')
    for value in [m['binary'], *m['runtime'], barrier['manifest'], barrier['profile_executions'], barrier['baseline_profile']]:
        a.receipt(value)
    a.check(frozen['target_time_used_for_fit'] is False and frozen['future_target_state_used'] is False
        and model['target_time_used_for_fit'] is False and model['future_target_state_used'] is False,
        'causality', 'future-target flags differ')
    a.check(model['condition_fields'] == ['h2d_source_reuse_bucket']
        and model['bucketing'] == 'power_of_two_floor_bytes_v1'
        and model['initialization'] == 'fresh_logical_history_then_one_full_prime'
        and model['unmodeled_reset_accesses'] is True
        and model['state_feature_depends_on_price_values'] is False,
        'causality', 'state/model hypothesis differs')
    a.check(model['scope_sha256'] == digest(model['scope']), 'scope', 'response scope digest differs')
    frozen_time = datetime.fromisoformat(barrier['frozen_at_utc'])
    started = read(out/'heldout-started.json')
    a.check(started['barrier'] == barrier and datetime.fromisoformat(started['at']) >= frozen_time,
            'ordering', 'heldout phase started before freeze')
    execution_cache, programs, oracles, records = {}, {}, {}, []
    calibration_sum, calibration_count = defaultdict(Fraction), defaultdict(int)
    calibration_coordinates = {}
    traces = {}
    for job in m['jobs']:
        a.check(job['request']['source_sha256'] == native_sources[0]['sha256'],
                'execution', 'requested native source differs from archived implementation')
        folder = out/'jobs'/job['id']
        execution = read(folder/'execution.json')
        execution_cache[job['id']] = execution
        a.check(execution['job'] == job, 'execution', 'declared/executed job differs')
        a.check(execution['process_exited_verified'] is True, 'execution', 'owned child did not exit')
        start = datetime.fromisoformat(execution['started_at_utc'])
        end = datetime.fromisoformat(execution['completed_at_utc'])
        a.check(end >= start, 'ordering', 'execution completion preceded start')
        if job['split'] == 'profile':
            a.check(execution['status'] == 'completed' and end <= frozen_time,
                    'ordering', 'calibration incomplete or after freeze')
        elif job['split'] == 'heldout':
            a.check(start >= datetime.fromisoformat(started['at']) >= frozen_time
                and execution['barrier'] == barrier, 'ordering', 'heldout execution not bound after freeze')
        if execution['status'] != 'completed':
            a.check(execution.get('error') is not None, 'execution', 'failed process lacks failure evidence')
            if job['split'] == 'heldout':
                records.append(dict(job_id=job['id'], program_key=job['program_key'],
                    instrumentation=job['request']['instrumentation'], status=execution['status'],
                    round=int(re.match(r'heldout-r(\d+)-', job['id']).group(1)), values=[], numeric=False))
            continue
        a.check(execution['returncode'] == 0 and execution['forced_termination'] is False,
                'execution', 'completed child exit differs')
        a.receipt(execution['native'])
        key = job['program_key']
        if key not in programs:
            binding = m['programs'][key]
            for name in ('program', 'reference'):
                a.receipt(binding[name])
            programs[key], oracles[key] = read(binding['program']['path']), read(binding['reference']['path'])
        native = read(execution['native']['path'])
        audit_native(a, native, job['request'], programs[key], oracles[key], execution['process_id'])
        if job['split'] == 'profile':
            a.check(native['instrumentation'] == 'stamped' and native['history_intervention'] == 'none'
                and native['memory_mode'] == 'pageable' and native['mode'] == 'full',
                'calibration', 'diagnostic or changed-boundary labels entered calibration')
            if key not in traces:
                traces[key] = independent_coordinates(programs[key], model['scope_sha256'])
            trace = traces[key]
            for block in native['trials']:
                for expected, item in zip(trace, block['action_intervals']):
                    for h, coordinate in {digest(expected[name]): expected[name] for name in ('static', 'conditional')}.items():
                        calibration_sum[h] += item['wall_ns']
                        calibration_count[h] += 1
                        calibration_coordinates[h] = coordinate
        elif job['split'] == 'heldout':
            records.append(dict(job_id=job['id'], program_key=key, instrumentation=native['instrumentation'],
                status='completed', round=int(re.match(r'heldout-r(\d+)-', job['id']).group(1)),
                values=[row['wall_ns']/row['repeats'] for row in native['trials']], numeric=True))
    profile_receipts = read(out/'profile-executions.json')
    a.check({e['job']['id'] for e in profile_receipts} == {j['id'] for j in m['jobs'] if j['split'] == 'profile'}
        and len(profile_receipts) == 18, 'calibration', 'profile receipt inventory differs')
    for e in profile_receipts:
        a.check(e == execution_cache[e['job']['id']], 'calibration', 'profile phase receipt differs from job receipt')
    prices = {}
    a.check(len(model['entries']) == len(calibration_sum) == len(calibration_coordinates),
            'calibration', 'frozen response-cell inventory differs')
    for entry in model['entries']:
        h = digest(entry['coordinate'])
        a.check(h == entry['key'] and entry['coordinate'] == calibration_coordinates[h],
                'calibration', 'response coordinate differs from independent logical replay')
        a.check(entry['sample_count'] == calibration_count[h]
            and Fraction(entry['mean_ns']) == calibration_sum[h]/calibration_count[h],
            'calibration', 'price differs from independent action-only mean')
        a.check(all(v['split'] == 'profile' and v['native'] in model['calibration_evidence'] for v in entry['evidence']),
                'calibration', 'unbound diagnostic/heldout response evidence')
        prices[h] = Fraction(entry['mean_ns'])
    prior = read(m['baseline_profile']['path'])
    old_prices = {}
    for row in prior['entries']:
        a.check(row['key'] == digest(row['coordinate']) and row['coordinate']['context'] == prior['context'],
                'prediction', 'older independent hot coordinate differs')
        old_prices[digest(row['coordinate']['semantic'])] = Fraction(row['mean_ns'])
    a.check(set(frozen['providers']) == set(PROVIDERS) == set(summary['providers']),
            'prediction', 'three-provider inventory differs')
    keys = sorted({j['program_key'] for j in m['jobs'] if j['split'] == 'heldout'})
    a.check(len(keys) == 6 and len(records) == 48, 'denominator', 'heldout case/process denominator differs')
    qualification = {}
    for provider in PROVIDERS:
        a.check(set(frozen['providers'][provider]) == set(keys), 'prediction', 'provider case inventory differs')
        provider_summary = summary['providers'][provider]
        reported_cases = {r['program_key']: r for r in provider_summary['cases']}
        a.check(set(reported_cases) == set(keys), 'metrics', 'reported case inventory differs')
        original_count = conservative_count = 0
        for key in keys:
            if key not in traces:
                traces[key] = independent_coordinates(programs[key], model['scope_sha256'])
            if provider == 'old_hot':
                total = sum((old_prices[digest(extract_semantic(programs[key], i))]
                             for i in range(len(programs[key]['actions']))), Fraction())
            else:
                field = 'static' if provider == 'context_static' else 'conditional'
                total = sum((prices[digest(row[field])] for row in traces[key]), Fraction())/8
            prediction = frozen['providers'][provider][key]
            a.check(Fraction(prediction['mean_replay_ns']) == total
                and len(prediction['trial_mean_replay_ns']) == 16
                and all(Fraction(v) == total for v in prediction['trial_mean_replay_ns']),
                'prediction', 'frozen prediction differs from independently composed prices')
            selected = [r for r in records if r['program_key'] == key]
            expected = recompute_case(prediction, selected)
            reported = reported_cases[key]
            for name in ('fully_qualified', 'qualified_original_primary_cv_gate', 'qualified_conservative_all_process_cv_gate',
                         'gates', 'failed_gates', 'failed_conservative_gates'):
                a.check(reported[name] == expected[name], 'metrics', 'scientific gate/result differs: '+name)
            for name in ('predicted_mean_ns', 'observed_primary_mean_ns', 'maximum_primary_process_mean_ape',
                         'pooled_primary_block_ape_p90', 'maximum_primary_process_cv',
                         'maximum_observer_process_cv', 'maximum_paired_observer_difference'):
                a.check(close(reported[name], expected[name]), 'metrics', 'case metric differs: '+name)
            reported_processes = {r['job_id']: r for r in reported['processes']}
            a.check(set(reported_processes) == {r['job_id'] for r in selected}, 'denominator', 'failed/native process omitted')
            for process in selected:
                r = reported_processes[process['job_id']]
                a.check(r['status'] == process['status'], 'metrics', 'reported native status differs')
                values = expected['processes'][process['job_id']]
                if values is None:
                    continue
                for name, value in values.items():
                    a.check(close(r[name], value), 'metrics', 'process metric differs: '+name)
            for i, effect in enumerate(expected['effects']):
                a.check(close(reported['observer_pairs'][i]['relative_mean_difference'], effect),
                        'metrics', 'paired observer effect differs')
            original_count += expected['qualified_original_primary_cv_gate']
            conservative_count += expected['qualified_conservative_all_process_cv_gate']
        a.check(provider_summary['fully_qualified_cases'] == provider_summary['qualified_original_primary_cv_cases'] == original_count,
                'metrics', 'original qualification count differs')
        a.check(provider_summary['qualified_conservative_all_process_cv_cases'] == conservative_count,
                'metrics', 'conservative qualification count differs')
        qualification[provider] = dict(original=original_count, conservative=conservative_count)
    completed = sum(e['status'] == 'completed' for e in execution_cache.values())
    heldout_completed = sum(r['status'] == 'completed' for r in records)
    a.check(summary['declared_native_processes'] == 48 and summary['completed_native_processes'] == heldout_completed,
            'denominator', 'summary native completion count differs')
    a.check({r['job_id'] for r in summary['failed_processes']} == {r['job_id'] for r in records if r['status'] != 'completed'},
            'denominator', 'failed processes omitted from summary')
    return dict(schema='h2d-history-independent-evidence-audit-1', passed=True,
        checks=a.count, sections=dict(a.sections), declared_native_processes=len(m['jobs']),
        completed_native_processes=completed, heldout_completed=heldout_completed,
        calibration_cells_checked=len(prices), static_cells=sum(not r['coordinate']['condition'] for r in model['entries']),
        conditional_cells=sum(bool(r['coordinate']['condition']) for r in model['entries']),
        qualification=qualification, old_report_v2_sha256=OLD_REPORT_SHA,
        summary_sha256=sha(out/'summary.json'), frozen_barrier_sha256=sha(out/'frozen-barrier.json'),
        audit_source_sha256=sha(Path(__file__)), measurements_performed=False,
        scientific_gate_failures_are_audit_failures=False,
        scope='evidence consistency on declared finite cases; no cache/staging mechanism or general dynamic accuracy proof')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    result = audit(args.out)
    destination = Path(args.output) if args.output else Path(args.out)/'independent-audit.json'
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, separators=(',', ':'), allow_nan=False)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
