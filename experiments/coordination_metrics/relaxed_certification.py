"""Retry MILP certification with a 1 ns relaxation of proven start lower bounds.

HiGHS declares the frozen model infeasible for a handful of P/D rows where the
strengthened lower bounds coincide with feasible start times (zero slack plus
floating-point rounding). Relaxing every proven lower bound by 1 ns enlarges the
feasible set, so the relaxed optimum is still a valid lower bound; if its dual
bound reaches the verified witness makespan, the original optimum is proven.
Nothing outside this addendum is modified.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT, ROOT / 'src', ROOT / 'mapping', ROOT / 'modeling'):
    sys.path.insert(0, str(folder))

from experiments.coordination_metrics.exact_fairness import (  # noqa: E402
    read, solver_dependencies, strengthened_lower_bound, verify, write)

solver_dependencies()  # the venv lacks scipy; this activates the read-only fallback path

from scipy.optimize import Bounds, LinearConstraint, milp  # noqa: E402
from scipy.sparse import lil_matrix  # noqa: E402

RESULTS = ROOT / 'build/coordination-metrics-rerun-20261030/ready-selection/RESULTS.json'
CASES = ROOT / 'build/coordination-metrics-approved-20261030/exact/cases'
DEFAULT_OUT = ROOT / 'build/certification-extension-20261030/relaxed-retry'


def build_model(nodes, upper):
    identities = list(nodes)
    index = {k: i for i, k in enumerate(identities)}
    n = len(identities)
    groups = defaultdict(list)
    for k, v in nodes.items():
        groups[v['machine']].append(k)
    ancestors = {k: set(v['deps']) for k, v in nodes.items()}
    for _ in nodes:
        new = {k: set().union(v, *(ancestors[p] for p in v)) if v else set() for k, v in ancestors.items()}
        if new == ancestors:
            break
        ancestors = new
    pairs = [(a, b) for g in groups.values() for a, b in itertools.combinations(g, 2)
             if nodes[a]['duration'] and nodes[b]['duration']
             and a not in ancestors[b] and b not in ancestors[a]]
    or_groups = {k: v.get('dependency_alternatives', [[]]) for k, v in nodes.items() if 'dependency_alternatives' in v}
    alternatives = [(k, j) for k, group in or_groups.items() for j in range(len(group))]
    selected = {pair: n + 1 + len(pairs) + i for i, pair in enumerate(alternatives)}
    variables = n + 1 + len(pairs) + len(alternatives)
    horizon = upper['duration_ns'] + max(v['duration'] for v in nodes.values()) + 1
    constraints = []; low = []; high = []
    def add(v, lo=-np.inf, hi=np.inf):
        constraints.append(v); low.append(lo); high.append(hi)
    for k, v in nodes.items():
        add({n: 1, index[k]: -1}, lo=v['duration'])
        for parent in v['deps']:
            add({index[k]: 1, index[parent]: -1}, lo=nodes[parent]['duration'])
    for j, (a, b) in enumerate(pairs):
        binary = n + 1 + j
        add({index[a]: 1, index[b]: -1, binary: horizon}, hi=horizon - nodes[a]['duration'])
        add({index[b]: 1, index[a]: -1, binary: -horizon}, hi=-nodes[b]['duration'])
    for k, group in or_groups.items():
        add({selected[k, j]: 1 for j in range(len(group))}, lo=1, hi=1)
        for j, branch in enumerate(group):
            for parent in branch:
                add({index[k]: 1, index[parent]: -1, selected[k, j]: -horizon}, lo=nodes[parent]['duration'] - horizon)
    strong = strengthened_lower_bound(nodes)
    for target, parents in ancestors.items():
        for anchor in sorted(parents):
            work = defaultdict(int)
            for job in parents:
                if anchor in ancestors[job]:
                    work[nodes[job]['machine']] += nodes[job]['duration']
            if work:
                add({index[target]: 1, index[anchor]: -1}, lo=nodes[anchor]['duration'] + max(work.values()))
    matrix = lil_matrix((len(constraints), variables), dtype=float)
    for row, entries in enumerate(constraints):
        for col, value in entries.items():
            matrix[row, col] = value
    return (matrix.tocsr(), np.array(low), np.array(high), variables, identities, index, strong)


def solve_relaxed(nodes, upper, relax=1.0, limit=30.0):
    matrix, low, high, variables, identities, index, strong = build_model(nodes, upper)
    n = len(identities)
    lower = np.array([max(nodes[k]['release'],
                          strong['energetic_earliest_finish_ns'][k] - nodes[k]['duration'] - relax)
                      for k in identities] + [strong['lower_bound_ns'] - relax] + [0] * (variables - n - 1),
                     dtype=float)
    upper_bounds = np.array([upper['duration_ns']] * (n + 1) + [1] * (variables - n - 1), dtype=float)
    objective = np.array([0.0] * n + [1.0] + [0.0] * (variables - n - 1))
    integrality = np.array([0] * (n + 1) + [1] * (variables - n - 1))
    started = time.perf_counter()
    result = milp(objective, integrality=integrality, bounds=Bounds(lower, upper_bounds),
                  constraints=LinearConstraint(matrix, low, high),
                  options=dict(time_limit=limit, mip_rel_gap=0, threads=1))
    doc = dict(status=int(result.status), message=result.message,
               elapsed_seconds=time.perf_counter() - started, variables=variables,
               constraints=int(matrix.shape[0]), relax_ns=relax,
               objective_ns=None if result.fun is None else float(result.fun),
               dual_bound_ns=getattr(result, 'mip_dual_bound', None),
               mip_gap=getattr(result, 'mip_gap', None), integer_tick_witness_valid=False)
    if result.x is not None:
        trace = [dict(id=k, machine=nodes[k]['machine'], start_ns=int(round(result.x[index[k]])),
                      end_ns=int(round(result.x[index[k]])) + nodes[k]['duration']) for k in identities]
        try:
            makespan = verify(nodes, trace)
            doc.update(integer_tick_witness_valid=True, verified_makespan_ns=makespan, events=trace)
        except AssertionError as exc:
            doc['witness_failure'] = str(exc)
    return doc


def main(out=DEFAULT_OUT, ids=None, relax=1.0, limit=30.0):
    solver_dependencies()
    out = Path(out)
    if out.exists():
        raise SystemExit(f'refusing existing output directory: {out}')
    out.mkdir(parents=True)
    document = read(RESULTS)
    rows = [r for r in document['rows'] if r['audit_status'] == 'pass' and not r['certificate']['certified']]
    if ids:
        rows = [r for r in rows if r['id'] in set(ids)]
    records = []
    for row in rows:
        nodes = read(CASES / row['id'] / 'nodes.json')
        upper = row['certificate']['feasible_optimal_or_upper_witness']
        doc = solve_relaxed(nodes, upper, relax=relax, limit=limit)
        dual = doc.get('dual_bound_ns')
        lower = row['strengthened_lower_bound']['lower_bound_ns']
        if dual is not None and math.isfinite(dual):
            lower = max(lower, math.ceil(dual - 1e-4))
        certified = doc['status'] == 0 and lower == upper['duration_ns'] and doc['integer_tick_witness_valid']
        certificate = dict(method='relaxed_MILP_retry', lower_bound_ns=lower, upper_bound_ns=upper['duration_ns'],
                           witness_kind='MILP_integer_rechecked' if certified else upper['kind'],
                           certified=certified, T_star_ns=upper['duration_ns'] if certified else None,
                           g=(row['T_C_ns'] - upper['duration_ns']) / upper['duration_ns'] if certified else None,
                           g_interval=[(row['T_C_ns'] - upper['duration_ns']) / upper['duration_ns'],
                                       (row['T_C_ns'] - lower) / lower],
                           feasible_optimal_or_upper_witness=dict(kind='MILP_integer_rechecked', duration_ns=doc['verified_makespan_ns'],
                                                                  events=doc.get('events')) if certified else upper,
                           milp=doc, relax_ns=relax)
        record = dict(id=row['id'], scenario=row['scenario'], T_C_ns=row['T_C_ns'], certified=certified,
                      lower=lower, upper=upper['duration_ns'], g=certificate['g'],
                      g_interval=certificate['g_interval'], milp_status=doc['status'],
                      elapsed_seconds=doc['elapsed_seconds'])
        write(out / 'rows' / f"{row['id']}.json", record)
        records.append(record)
        print(f"{row['id']} certified={certified} status={doc['status']} dual={dual} upper={upper['duration_ns']}",
              flush=True)
    summary = dict(schema='relaxed-certification-retry-1', relax_ns=relax, attempted=len(records),
                   certified=sum(r['certified'] for r in records),
                   still_unresolved=sum(not r['certified'] for r in records),
                   certified_ids=[r['id'] for r in records if r['certified']],
                   records=records)
    write(out / 'SUMMARY.json', summary)
    print(summary['certified'], 'certified of', summary['attempted'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(DEFAULT_OUT))
    parser.add_argument('--ids', nargs='*')
    parser.add_argument('--relax', type=float, default=1.0)
    parser.add_argument('--limit', type=float, default=30.0)
    args = parser.parse_args()
    main(out=args.out, ids=args.ids, relax=args.relax, limit=args.limit)
