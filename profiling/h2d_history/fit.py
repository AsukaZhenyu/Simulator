"""Freeze action-only calibration and predictions before fresh heldout runs.

Only stamped *action* response intervals from the declared profile split are
fitting labels. Diagnostic jobs, block/program durations and heldout stamps are
not read. State evolution and response selection are model-owned.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import shutil
import statistics

from llm_infer_model.performance.execution_history import HistoryScope, SerialExecutionHistory
from llm_infer_model.performance.history_conditioned_costs import (
    BUCKETING, INITIALIZATION, SCHEMA, HistoryConditionedResponseTable, response_coordinate,
)
from llm_infer_model.performance.reusable_action_costs import (
    ReusableResponseTable, coordinate, digest,
)
from profiling.dac_validation import e2a


ROOT = Path(__file__).resolve().parents[2]
read, write, sha, receipt, require = e2a.read, e2a.write, e2a.sha, e2a.receipt, e2a.require
FIELDS = ['h2d_source_reuse_bucket']


def calibration_scope(manifest):
    """Hardware/runtime/preparation scope without mathematical program IDs."""
    requests = [j['request'] for j in manifest['jobs'] if j['split'] == 'profile']
    require(requests, 'no declared action calibration jobs')
    first = requests[0]
    require(all(r['sampling'] == first['sampling'] and r['memory_mode'] == 'pageable'
        and r['mode'] == 'full' and r['instrumentation'] == 'stamped'
        and r['expected_process_mask'] == first['expected_process_mask']
        and r.get('history_intervention', 'none') == 'none' for r in requests),
        'profile timing/memory/preparation contracts differ')
    context = dict(hardware=manifest['baseline_context']['hardware'],
        runtime=[dict(path=r['path'], sha256=r['sha256']) for r in manifest['runtime']],
        binary_sha256=manifest['binary']['sha256'],
        native_source_sha256=first['source_sha256'],
        dtype='F32', precision='GGML_PREC_F32', cuda_graphs='disabled',
        threads=1, main_thread_mask=first['sampling']['main_thread_mask'],
        process_affinity_mask=first['expected_process_mask'], memory_mode='pageable',
        measurement_boundary='inclusive host-completed GGML action with per-invocation QPC stamping',
        preparation=dict(reset_before_batch=True, untimed_complete_prime_passes=1,
            fixed_input_replay=True, repeats=first['sampling']['repeats'],
            warmup_min_ms=first['sampling']['warmup_min_ms'],
            logical_state_initialization=INITIALIZATION),
        state_features=FIELDS, bucketing=BUCKETING,
        reset_writes_and_physical_cache_clock_state_modeled=False,
        target_outcome_or_program_identity_used=False)
    return context, digest(context)


def coordinate_trace(program, scope_sha256, prior_durations, *, repeats=8):
    """One full prime and replay coordinates using known independent prices.

    Logical reuse bins are independent of those price values.  They still use
    genuine predicted durations, rather than pretending that the prime is free.
    """
    require(len(prior_durations) == len(program['actions']), 'prior action inventory differs')
    history = SerialExecutionHistory(program, HistoryScope(scope_sha256))
    for index, value in enumerate(prior_durations):
        history.commit(index, value)
    rows = []
    for invocation in range(repeats):
        for index, value in enumerate(prior_durations):
            state = history.snapshot(index)
            static = response_coordinate(state, FIELDS, use_history=False)
            conditional = response_coordinate(state, FIELDS)
            rows.append(dict(invocation=invocation, action_index=index,
                static=static, conditional=conditional))
            history.commit(index, value)
    return rows


def action_samples(native, trace, *, trials, repeats):
    """Yield action labels only. Complete-block wall time is deliberately unused."""
    require(native['mode'] == 'full' and native['instrumentation'] == 'stamped'
        and native['memory_mode'] == 'pageable'
        and native.get('history_intervention', 'none') == 'none',
        'only unintervened stamped pageable profile action intervals may fit prices')
    prep = native['preparation_contract']
    require(prep['reset_before_batch'] is True and prep['untimed_complete_prime_passes'] == 1
        and prep['timed_replays_per_batch'] == repeats and prep['fixed_input_replay'] is True,
        'action calibration preparation differs')
    rows = native['trials']
    require(len(rows) == trials, 'calibration block count differs')
    for trial, block in enumerate(rows):
        require(block['trial'] == trial and block['repeats'] == repeats
            and block['actual_stamped'] is True, 'calibration trial/replay inventory differs')
        intervals = block['action_intervals']
        require(len(intervals) == len(trace), 'calibration action-stamp count differs')
        for wanted, interval in zip(trace, intervals):
            require(interval['trial'] == trial and interval['invocation'] == wanted['invocation']
                and interval['action_index'] == wanted['action_index'],
                'action interval order differs from declared causal replay')
            value = interval['wall_ns']
            require(type(value) is int and value >= 0, 'action response must be a known nonnegative integer ns')
            yield wanted, Fraction(value)


def aggregate_samples(groups, sources):
    """Mean response per exact semantic/declared state coordinate."""
    entries = []
    for key in sorted(groups):
        group = groups[key]
        values = group['values']
        require(values, 'empty calibration response cell')
        average = sum(values, Fraction()) / len(values)
        # Descriptive calibration spread only. No sample or failed process is
        # discarded, and these statistics never replace heldout gates.
        deviation = statistics.stdev(float(v) for v in values) if len(values) > 1 else 0.0
        entries.append(dict(key=key, coordinate=group['coordinate'], mean_ns=str(average),
            sample_count=len(values), evidence=sources[key],
            calibration_spread=dict(min_ns=str(min(values)), max_ns=str(max(values)),
                sample_cv=deviation / float(average) if average > 0 else None)))
    return entries


def _new_outputs(out):
    for name in ('model-frozen.json', 'predictions-frozen.json', 'frozen-barrier.json',
                 'prediction-ledgers', 'fit_sources'):
        require(not (out / name).exists(), 'frozen output already exists: ' + name)


def freeze(out):
    out = Path(out).resolve()
    manifest = read(out / 'manifest.json')
    require(manifest['schema'] == 'h2d-history-campaign-1', 'campaign schema differs')
    require(not (out / 'heldout-started.json').exists(), 'fit after heldout acquisition rejected')
    heldout_jobs = [j for j in manifest['jobs'] if j['split'] == 'heldout']
    require(heldout_jobs and not any((out / 'jobs' / j['id']).exists() for j in heldout_jobs),
            'heldout native directory already exists; target-blind freeze rejected')
    _new_outputs(out)
    from profiling.h2d_history.campaign import check
    check(manifest)
    expected = {j['id']: j for j in manifest['jobs'] if j['split'] == 'profile'}
    executions = read(out / 'profile-executions.json')
    require(len(expected) == manifest['protocol']['counts']['profile'] == 18,
            'declared action calibration process count differs')
    require(len(executions) == len(expected)
        and {e['job']['id'] for e in executions} == set(expected), 'profile receipt inventory differs')
    require(all(e['status'] == 'completed' and e['job'] == expected[e['job']['id']]
        and e['job']['split'] == 'profile' for e in executions),
        'all declared profile processes must complete; diagnostics/heldout cannot fit prices')
    context, scope = calibration_scope(manifest)
    prior_binding = manifest['baseline_profile']
    require(sha(prior_binding['path']) == prior_binding['sha256'], 'independent old hot profile changed')
    prior = read(prior_binding['path'])
    prior_table = ReusableResponseTable(tuple(prior['entries']))
    programs, traces = {}, {}
    groups, evidence = {}, defaultdict(list)
    calibration_receipts = []
    for execution in executions:
        job = execution['job']
        key = job['program_key']
        binding = manifest['programs'][key]['program']
        if key not in programs:
            require(sha(binding['path']) == binding['sha256'], 'declared calibration program changed')
            programs[key] = read(binding['path'])
            durations = [prior_table.lookup(coordinate(programs[key], i, prior['context']))
                         for i in range(len(programs[key]['actions']))]
            traces[key] = coordinate_trace(programs[key], scope, durations,
                repeats=job['request']['sampling']['repeats'])
        native_receipt = execution['native']
        require(sha(native_receipt['path']) == native_receipt['sha256'], 'profile native evidence changed')
        native = read(native_receipt['path'])
        require(native['request'] == job['request'] and native['process_id'] == execution['process_id']
            and native['program_sha256'] == programs[key]['program_sha256']
            and native['native_repeat_consistency']['passed'] is True,
            'profile native input/source/numerical receipt mismatch')
        per_file = defaultdict(lambda: dict(count=0, indices=set()))
        for wanted, value in action_samples(native, traces[key],
                trials=job['request']['sampling']['trials'], repeats=job['request']['sampling']['repeats']):
            coordinates = {digest(wanted[name]): wanted[name] for name in ('static', 'conditional')}
            for h, coord in coordinates.items():
                group = groups.setdefault(h, dict(coordinate=coord, values=[]))
                require(group['coordinate'] == coord, 'response coordinate hash collision')
                group['values'].append(value)
                per_file[h]['count'] += 1
                per_file[h]['indices'].add(wanted['action_index'])
        for h, counts in per_file.items():
            evidence[h].append(dict(native=native_receipt, split='profile',
                action_indices=sorted(counts['indices']), sampled_action_intervals=counts['count']))
        calibration_receipts.append(native_receipt)
    entries = aggregate_samples(groups, evidence)
    profile = dict(schema=SCHEMA, scope_sha256=scope, scope=context, condition_fields=FIELDS,
        bucketing=BUCKETING, initialization=INITIALIZATION, unmodeled_reset_accesses=True,
        entries=entries, frozen_at_utc=e2a.utc(),
        calibration_processes=len(executions), calibration_evidence=calibration_receipts,
        fitting_labels='per-invocation inclusive action wall_ns only; no whole-block labels',
        fitting_split='profile only; diagnostics and heldout excluded',
        static_baseline='same per-action calibration labels aggregated without H2D state bins',
        state_evolution='declared accesses; independent old static prices used only for prediction-clock advancement',
        state_feature_depends_on_price_values=False,
        target_time_used_for_fit=False, future_target_state_used=False,
        causal_mechanism_identified=False, hardware_accuracy_qualified=False,
        limitations=['piecewise-constant logical-footprint response is an empirical hypothesis',
            'reset writes, cache contents, driver staging, clocks and prior-block recovery are not identified',
            'calibration covers the declared semantic and logical-state bins only'])
    table = HistoryConditionedResponseTable.from_profile(profile)
    predictions = dict(schema='h2d-history-frozen-predictions-1', providers={name: {} for name in
        ('old_hot', 'context_static', 'history_conditional')},
        initialization=INITIALIZATION, target_time_used_for_fit=False, future_target_state_used=False,
        predicted_blocks='same one-prime logical history independently initialized for each of 16 blocks',
        old_hot_context='older independent isolated-hot profile; different calibration/execution context',
        state_feature_depends_on_price_values=False)
    ledgers = {}
    heldout_keys = sorted({j['program_key'] for j in heldout_jobs})
    require(len(heldout_keys) == 6, 'six complete Dense heldout candidates required')
    for key in heldout_keys:
        binding = manifest['programs'][key]['program']
        require(sha(binding['path']) == binding['sha256'], 'heldout mathematical input changed')
        program = read(binding['path'])
        sampling = next(j['request']['sampling'] for j in heldout_jobs if j['program_key'] == key)
        old = prior_table.evaluate(program, prior['context'])['duration_ns']
        predictions['providers']['old_hot'][key] = dict(mean_replay_ns=str(old),
            trial_mean_replay_ns=[str(old)] * sampling['trials'],
            context_comparability='different older isolated-hot calibration; not the causal state ablation')
        for name, enabled in (('context_static', False), ('history_conditional', True)):
            result = table.predict_blocks(program, blocks=1, repeats=sampling['repeats'], use_history=enabled)
            total = result['duration_ns']
            relative = 'prediction-ledgers/' + name + '-' + key + '.json'
            ledgers[relative] = result
            predictions['providers'][name][key] = dict(mean_replay_ns=str(total),
                trial_mean_replay_ns=[str(total)] * sampling['trials'],
                ledger=relative, context_comparability='same profile action labels; H2D conditioning ablation')
    # All target coordinates have been checked above, before any artifact is
    # published and before the first heldout child directory may be created.
    source_paths = [Path(__file__), ROOT/'modeling/llm_infer_model/performance/execution_history.py',
        ROOT/'modeling/llm_infer_model/performance/history_conditioned_costs.py',
        ROOT/'modeling/llm_infer_model/performance/reusable_action_costs.py',
        ROOT/'modeling/llm_infer_model/performance/parameters.py']
    sources = []
    for path in source_paths:
        target = out/'fit_sources'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        sources.append(dict(source=receipt(path), archived=receipt(target)))
    write(out/'model-frozen.json', profile)
    write(out/'predictions-frozen.json', predictions)
    for relative, ledger in ledgers.items():
        write(out/relative, e2a.number(ledger))
    bound = ['model-frozen.json', 'predictions-frozen.json', *ledgers]
    bound += [str(Path(s['archived']['path']).relative_to(out)) for s in sources]
    barrier = dict(schema='h2d-history-preheldout-freeze-1', frozen_at_utc=e2a.utc(),
        files={name: sha(out/name) for name in bound}, sources=sources,
        manifest=receipt(out/'manifest.json'), profile_executions=receipt(out/'profile-executions.json'),
        baseline_profile=prior_binding, target_time_used_for_fit=False,
        future_target_state_used=False, all_heldout_coordinates_known_before_acquisition=True,
        independent_heldout_seed_and_processes=True)
    write(out/'frozen-barrier.json', barrier)
    return dict(scope_sha256=scope, calibration_processes=len(executions), response_cells=len(entries),
        static_cells=sum(not row['coordinate']['condition'] for row in entries),
        conditional_cells=sum(bool(row['coordinate']['condition']) for row in entries),
        heldout_candidates=len(heldout_keys), frozen_barrier=receipt(out/'frozen-barrier.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    import json
    print(json.dumps(freeze(args.out), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
