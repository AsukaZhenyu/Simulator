"""Re-run the frozen 800-policy corpus after a public ready-selection change.

This script never re-solves T*. It re-executes exactly the frozen candidate
(scenario, regime, storage, choices) with the *current* compiler, re-verifies
every produced trace against the frozen independent `nodes.json`, and keeps the
frozen per-policy T* / lower-bound / upper-bound certificate. A new trace
strictly faster than the frozen upper bound becomes the new feasible witness,
so the certified interval can only tighten -- never be widened to hide a gap.

Usage:
    python experiments/coordination_metrics/rerun_ready_selection.py [--ids a b ...]
"""
from __future__ import annotations
import argparse
import csv
import json
import shutil
import sys
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'src', ROOT / 'mapping', ROOT / 'modeling'):
    sys.path.insert(0, str(p))

from experiments.ir_value import campaign as base
from experiments.dac_validation_e1_oracle import declared_dag
from experiments.coordination_metrics import exact_fairness as ef

FROZEN = ef.OUT
OUT = ROOT / 'build/coordination-metrics-rerun-20261030' / 'ready-selection'


def scenario_of(row):
    st = next(x for x in base.SCENARIOS if x[0] == row['scenario'])
    return base.scenario(*st, row['regime'])


def run_one(row, out):
    s = scenario_of(row)
    d = declared_dag(s)
    folder = out / 'runs' / row['id']
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    with base.build(s, d, row['choices'], row['storage'], folder) as compiled:
        result = compiled.run()
        result.save(folder / 'execution')
    events = ef.read(folder / 'execution' / 'events.json')
    if isinstance(events, dict):
        events = events.get('events', events)
    trace = ef.event_trace(events)
    nodes = ef.read(FROZEN / 'cases' / row['id'] / 'nodes.json')
    makespan = ef.verify(nodes, trace)
    return makespan, trace


def saved_trace(row, out):
    events = ef.read(Path(out) / 'runs' / row['id'] / 'execution' / 'events.json')
    if isinstance(events, dict):
        events = events.get('events', events)
    return ef.event_trace(events)


def run_job(job):
    """Worker entry: returns (id, makespan) or (id, error document)."""
    row, out = job
    try:
        makespan, _ = run_one(row, out)
        return row['id'], makespan
    except Exception as exc:
        return row['id'], dict(type=type(exc).__name__, message=str(exc),
                               traceback=traceback.format_exc())


def refresh_certificate(row, makespan, trace):
    c = row['certificate']
    lower, upper = c['lower_bound_ns'], c['upper_bound_ns']
    tightened = makespan < upper
    if tightened:
        upper = makespan
        witness = dict(kind='rerun_compiler_trace', duration_ns=makespan, events=trace)
    else:
        witness = c['feasible_optimal_or_upper_witness']
    assert lower <= upper, (row['id'], 'invalid bounds after rerun', lower, upper)
    certified = lower == upper
    c.update(lower_bound_ns=lower, upper_bound_ns=upper, T_star_ns=upper if certified else None,
             g=(makespan - upper) / upper if certified else None,
             g_interval=[(makespan - upper) / upper, (makespan - lower) / lower],
             feasible_optimal_or_upper_witness=witness,
             certified=certified,
             rerun_tightened_upper_bound=tightened)
    return c


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--ids', nargs='*', default=None)
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--jobs', type=int, default=1,
                    help='parallel case workers; each case is independent and verified')
    ap.add_argument('--summarize-only', action='store_true',
                    help='rewrite SUMMARY/PER_POLICY/WORST from an existing RESULTS.json')
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        rows = ef.read(out / 'RESULTS.json')['rows']
        summarize(out, rows)
        return 0
    frozen = ef.read(FROZEN / 'RESULTS.json')
    rows = frozen['rows']
    wanted = [r for r in rows if r['audit_status'] == 'pass']
    if args.ids:
        wanted = [r for r in wanted if r['id'] in set(args.ids)]
    ef.write(out / 'RERUN_PROTOCOL.json', dict(
        schema='per-policy-rerun-1',
        frozen_results=str(FROZEN / 'RESULTS.json'),
        frozen_results_sha256=ef.sha(FROZEN / 'RESULTS.json'),
        candidate_source='frozen scenario/regime/storage/choices; costs and T* not re-solved',
        current_compiler_sha256={str(p): ef.sha(p) for p in (
            ROOT / 'mapping/tensor_mapping/composition_policy.py',
            ROOT / 'mapping/tensor_mapping/request_scheduling.py',
            ROOT / 'mapping/tensor_mapping/python_frontend.py',
            ROOT / 'experiments/ir_value/campaign.py')},
        rule='Every new trace is re-verified against the frozen nodes.json. A trace shorter than the frozen upper bound replaces it as feasible witness; T* and lower bounds are never re-solved or weakened.',
        policy_count=len(wanted)))
    counts = Counter()
    # The independent per-case inventories are frozen evidence; copy them so
    # the new directory is self-contained and report() can read the same nodes
    # it verified against.
    shutil.copytree(FROZEN / 'cases', out / 'cases', dirs_exist_ok=True)
    for frozen_name in ('SOLVER_ENVIRONMENT.json', 'PROTOCOL.json'):
        if not (out / frozen_name).exists():
            shutil.copy2(FROZEN / frozen_name, out / frozen_name)
    jobs = [(row, out) for row in wanted]
    by_id = {row['id']: row for row in wanted}
    if args.jobs > 1:
        from multiprocessing import get_context
        pool = get_context('spawn').Pool(args.jobs)
        outcomes = pool.imap_unordered(run_job, jobs, chunksize=1)
    else:
        pool = None
        outcomes = map(run_job, jobs)
    try:
        for index, (identity, result) in enumerate(outcomes, 1):
            row = by_id[identity]
            if isinstance(result, dict):
                counts['failed'] += 1
                row['rerun_error'] = result
                print(f"[{index}/{len(wanted)}] {identity:46s} FAILED {result['type']}: {result['message']}", flush=True)
            else:
                makespan = result
                old = row['T_C_ns']
                row['frozen_T_C_ns'] = old
                row['T_C_ns'] = makespan
                trace = saved_trace(row, out) if makespan < row['certificate']['upper_bound_ns'] else None
                refresh_certificate(row, makespan, trace)
                counts['reproduced'] += 1
                if makespan != old:
                    counts['changed'] += 1
                if row['certificate'].get('rerun_tightened_upper_bound'):
                    counts['tightened_upper_bound'] += 1
                if row['certificate']['certified']:
                    counts['certified'] += 1
                print(f"[{index}/{len(wanted)}] {identity:46s} {old} -> {makespan} "
                      f"certified={row['certificate']['certified']} "
                      f"g=[{100*row['certificate']['g_interval'][0]:.2f}%,{100*row['certificate']['g_interval'][1]:.2f}%]",
                      flush=True)
            if index % 25 == 0:
                ef.write(out / 'RESULTS.json', dict(rows=rows))
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    ef.write(out / 'RESULTS.json', dict(rows=rows))
    print('collapse', dict(counts), flush=True)
    summarize(out, rows)
    ef.write(out / 'RERUN_SUMMARY.json', dict(counts=dict(counts), coverage=coverage(rows),
        provenance='Every row here was re-executed with the current compiler and re-verified against the frozen nodes; frozen T* and lower bounds are unchanged. Rows absent from coverage.rerun_policies keep their frozen trace.'))
    return 0


def summarize(out, rows):
    """Write the same per-policy quantities as the frozen report, with the
    frozen T* catalogue and a rerun-accurate narrative."""
    out = Path(out)
    valid = [r for r in rows if r['audit_status'] == 'pass']
    certified = [r for r in valid if r['certificate']['certified']]
    unresolved = [r for r in valid if not r['certificate']['certified']]
    worst = max(certified, key=lambda r: r['certificate']['g'])
    families = []
    for family in sorted({r['scenario'] for r in valid}):
        group = [r for r in valid if r['scenario'] == family]
        exact = [r for r in group if r['certificate']['certified']]
        families.append(dict(scenario=family, policies=len(group), certified=len(exact),
            unresolved=len(group) - len(exact),
            max_certified_g=max((r['certificate']['g'] for r in exact), default=None),
            max_g_lower_bound=max(r['certificate']['g_interval'][0] for r in group),
            max_g_upper_bound=max(r['certificate']['g_interval'][1] for r in group)))
    interval = [max(r['certificate']['g_interval'][0] for r in valid),
                max(r['certificate']['g_interval'][1] for r in valid)]
    summary = dict(
        policies=len(rows), independent_fidelity_passed=len(valid),
        certified_individual_optima=len(certified), unresolved_individual_optima=len(unresolved),
        negative_gap_count=sum(r['certificate']['g_interval'][0] < 0 for r in valid),
        violations_certified=sum(r['certificate']['g'] > 0 for r in certified),
        violations_with_positive_lower_bound=sum(r['certificate']['g_interval'][0] > 0 for r in valid),
        worst_certified_policy=worst['id'], max_certified_g=worst['certificate']['g'],
        maximum_over_entire_corpus_interval=interval,
        entire_corpus_max_gap_certified=interval[0] == interval[1],
        families=families, unresolved_policy_ids=[r['id'] for r in unresolved],
        frozen_results=str(FROZEN / 'RESULTS.json'),
        frozen_results_sha256=ef.sha(FROZEN / 'RESULTS.json'),
        limitations=[
            'Same frozen candidate, prices, DAG, placement and independent per-case nodes; only the compiler ready-selection changed and every trace was re-verified.',
            'T* is unchanged from the frozen certificate. A re-run trace below the frozen upper bound replaces that upper bound, so an interval can only tighten.',
            'T* is offline for the complete declared workload, permits deliberate idle time and knows request arrival times; it is not an online unknown-future optimum.',
            'Bounded analytical kernels and fixed single CPU/GPU/shared PCIe; no native accuracy claim.',
            'Unresolved rows keep g intervals and no T* scalar. The corpus maximum is certified only if the largest lower bound equals the largest upper bound.'])
    ef.write(out / 'SUMMARY.json', summary)
    columns = ['id', 'scenario', 'regime', 'storage', 'T_C_ns', 'frozen_T_C_ns', 'T_star_ns',
               'lower_bound_ns', 'upper_bound_ns', 'g', 'g_lower', 'g_upper', 'certified',
               'node_count', 'copy_nodes', 'OR_groups']
    with (out / 'PER_POLICY.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        for r in valid:
            c = r['certificate']
            writer.writerow({**{k: r.get(k) for k in ('id', 'scenario', 'regime', 'storage', 'T_C_ns',
                                                      'frozen_T_C_ns', 'node_count', 'copy_nodes', 'OR_groups')},
                             **{k: c[k] for k in ('T_star_ns', 'lower_bound_ns', 'upper_bound_ns', 'g', 'certified')},
                             'g_lower': c['g_interval'][0], 'g_upper': c['g_interval'][1]})
    trace_path = out / 'runs' / worst['id'] / 'execution' / 'events.json'
    current = saved_trace(dict(id=worst['id']), out) if trace_path.exists() else None
    ef.write(out / 'WORST_CERTIFIED_COUNTEREXAMPLE.json', dict(
        policy=worst, current_compiler_trace=current,
        optimal_trace=worst['certificate']['feasible_optimal_or_upper_witness']['events'],
        nodes=ef.read(out / 'cases' / worst['id'] / 'nodes.json'),
        conclusion='Same declared placement, preparation scope and independent inventory as the frozen corpus; only the public ready-selection order changed.'))
    print(json.dumps({k: v for k, v in summary.items() if k not in {'unresolved_policy_ids'}},
                     ensure_ascii=False, indent=2), flush=True)


def coverage(rows):
    valid = [r for r in rows if r['audit_status'] == 'pass' and 'rerun_error' not in r]
    def frozen(r):
        return r.get('frozen_T_C_ns', r['T_C_ns'])
    return dict(
        rerun_policies=sum(1 for r in valid if 'frozen_T_C_ns' in r),
        rerun_failures=sum(1 for r in rows if 'rerun_error' in r),
        improved=sum(1 for r in valid if r['T_C_ns'] < frozen(r)),
        unchanged=sum(1 for r in valid if r['T_C_ns'] == frozen(r)),
        regressed=sum(1 for r in valid if r['T_C_ns'] > frozen(r)),
        negative_gap=sum(1 for r in valid if r['certificate']['g_interval'][0] < 0))


if __name__ == '__main__':
    raise SystemExit(main())
