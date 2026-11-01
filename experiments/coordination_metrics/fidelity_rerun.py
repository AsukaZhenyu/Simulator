"""Re-run the independent fidelity contract on the post-fix 800-policy traces.

The frozen audit in ``fidelity_audit.py`` checked the *pre-fix* corpus.  The
frozen ``RESULTS.json`` rows were mutated in place by ``rerun_ready_selection``
(``T_C_ns`` replaced, certificate refreshed), so their copied ``fidelity``
booleans describe the old trace and cannot be cited as post-fix evidence.  This
driver closes that gap: it takes the same public declarations (scenario,
regime, storage, choices) and the same independent expected DAG, but reads the
graph and events produced by the current compiler from
``runs/<id>/execution``.

No expectation is read from compiler output.  The checker is imported, not
copied, so its falsification controls in the frozen audit remain the controls
for this run.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'src', ROOT / 'mapping', ROOT / 'modeling'):
    sys.path.insert(0, str(p))

from copy import deepcopy

from experiments.dac_validation_e1_oracle import declared_dag
from experiments.coordination_metrics.fidelity_audit import (audit_lru_sequence,
                                                             audit_public_records, audit_trace,
                                                             fresh_lru, read, save, sha)
from experiments.coordination_metrics.rerun_ready_selection import scenario_of

OUT = ROOT / 'build/coordination-metrics-rerun-20261030' / 'fidelity'
RUNS = ROOT / 'build/coordination-metrics-rerun-20261030' / 'ready-selection'
PHYSICAL = ROOT / 'build/coordination-metrics-rerun-20261030' / 'physical'
COMPILER = ('mapping/tensor_mapping/composition_policy.py',
            'mapping/tensor_mapping/request_scheduling.py',
            'mapping/tensor_mapping/python_frontend.py',
            'experiments/ir_value/campaign.py')


def mutation_controls(declaration, choices, dag, graph, events, bridge, records, public_events):
    """Deliberately wrong evidence; same mutations as the frozen audit, rerun
    with the declaration's public choices so the expected side is unchanged."""
    prepared = declaration['intent']['preparation']
    results = []

    def trace_mutation(name, mutate, category):
        ge, ev = deepcopy(graph), deepcopy(events)
        mutate(ge, ev)
        r = audit_trace(name, declaration, dag, ge, ev, choices=choices, preparation=prepared)
        results.append(dict(id=name, expected_detector=category,
                            caught=r['violation_counts'].get(category, 0) > 0, audit=r))

    def flip_placement(g, e):
        first = next(x for x in e if x['kind'] == 'COMPUTE')
        first['device'] = 'cpu' if first.get('device') != 'cpu' else 'gpu'

    trace_mutation('wrong-placement', flip_placement, 'placement')
    trace_mutation('duplicate-compute',
                   lambda g, e: e.append(deepcopy(next(x for x in e if x['kind'] == 'COMPUTE'))),
                   'execution_multiplicity')
    trace_mutation('missing-compute',
                   lambda g, e: e.remove(next(x for x in e if x['kind'] == 'COMPUTE')),
                   'execution_multiplicity')
    trace_mutation('wrong-copy-bytes',
                   lambda g, e: next(x for x in e if x['kind'] == 'COPY').update(bytes_transferred=1),
                   'copy_payload')
    trace_mutation('missing-required-input',
                   lambda g, e: next(x for x in g['workload']['operations'] if x['inputs'])['inputs'].pop(),
                   'required_dependencies')
    for name, field, value, category in [('wrong-lru-victim', 'expert', 0, 'lru_victim'),
                                         ('illegal-future-load', 'expert', 2, 'causal_route')]:
        acts = deepcopy(bridge['program']['actions'])
        next(x for x in acts if x['kind'] == ('RELEASE' if name == 'wrong-lru-victim' else 'LOAD'))[field] = value
        r = audit_lru_sequence(acts, [0, 1, 0, 2])
        results.append(dict(id=name, expected_detector=category,
                            caught=r['violation_counts'].get(category, 0) > 0, audit=r))
    forged = deepcopy(records)
    future = deepcopy(next(route for rec in forged for route in rec['observation']['route_history']
                           if route['absolute_position'] == 3))
    forged[0]['observation']['route_history'] = [future]
    r = audit_public_records(forged, [0, 1, 0, 2], public_events, 'illegal-future-revelation')
    results.append(dict(id='illegal-future-revelation', expected_detector='public_route_revelation',
                        caught=r['violation_counts'].get('public_route_revelation', 0) > 0, audit=r))
    return results


def main(out=OUT, physical=PHYSICAL):
    out = Path(out)
    if out.exists():
        raise ValueError('Fresh output directory required')
    results_path = RUNS / 'RESULTS.json'
    rows_in = [r for r in read(results_path)['rows'] if r['audit_status'] == 'pass']
    inputs = {str(results_path): sha(results_path)}
    rows = []
    mutation_source = None
    for index, row in enumerate(rows_in, 1):
        folder = RUNS / 'runs' / row['id'] / 'execution'
        graph_path, events_path = folder / 'compiled_graph.json', folder / 'events.json'
        inputs[str(graph_path)] = sha(graph_path)
        inputs[str(events_path)] = sha(events_path)
        workload = scenario_of(row)
        dag = declared_dag(workload)
        graph = read(graph_path)
        graph = graph.get('graph', graph)
        events = read(events_path)
        if isinstance(events, dict):
            events = events['events']
        declaration = dict(workload=workload, intent=dict(preparation=row['storage']))
        result = audit_trace(row['id'], declaration, dag, graph, events,
                             choices=row['choices'], preparation=row['storage'])
        result['cohort'] = 'rerun800'
        rows.append(result)
        save(out / 'cases' / (row['id'] + '.json'), result)
        if mutation_source is None and any(e['kind'] == 'COPY' for e in events):
            mutation_source = (declaration, row['choices'], dag, graph, events)
        if index % 100 == 0:
            print(f'{index}/{len(rows_in)} {row["id"]} violations={result["total_violations"]}', flush=True)
    protocol_path = Path(physical) / 'physical_PROTOCOL.json'
    inputs[str(protocol_path)] = sha(protocol_path)
    for condition in read(protocol_path)['conditions']:
        folder = Path(physical) / 'runs' / condition['id']
        graph_path, events_path = folder / 'execution-graph.json', folder / 'events.json'
        inputs[str(graph_path)] = sha(graph_path)
        inputs[str(events_path)] = sha(events_path)
        result = audit_trace(condition['id'], condition, declared_dag(condition['workload']),
                             read(graph_path), read(events_path),
                             preparation=condition['intent']['preparation'])
        result['cohort'] = 'physical48-postfix'
        rows.append(result)
        save(out / 'cases' / (condition['id'] + '.json'), result)
    fresh, bridge = fresh_lru(out)
    for row in fresh:
        row['cohort'] = 'fresh-postfix'
        rows.append(row)
    mutations = mutation_controls(
        *mutation_source, bridge,
        read(out / 'fresh/base/public-records.json'), read(out / 'fresh/base/events.json'))
    save(out / 'MUTATION_CONTROLS.json', mutations)
    totals, violations = Counter(), Counter()
    for r in rows:
        totals.update(r['checks'])
        violations.update(r['violation_counts'])
    summary = dict(
        schema='strategy-fidelity-rerun-1',
        utc=datetime.now(timezone.utc).isoformat(),
        cases=len(rows),
        cohort_counts=dict(Counter(r['cohort'] for r in rows)),
        checks=dict(totals),
        violations=dict(violations),
        total_violations=sum(violations.values()),
        cases_with_violations=[r['id'] for r in rows if r['total_violations']],
        mutation_controls_count=len(mutations),
        mutation_controls_caught=sum(r['caught'] for r in mutations),
        category_case_counts={k: dict(evaluated=sum(k in r['checks'] for r in rows),
                                      violated=sum(r['violation_counts'].get(k, 0) > 0 for r in rows))
                              for k in totals},
        results=str(results_path), results_sha256=sha(results_path),
        compiler_sha256={p: sha(ROOT / p) for p in COMPILER},
        input_hashes_unchanged=all(sha(p) == h for p, h in inputs.items()),
        scope=('Independent finite contract checks on traces produced by the current compiler for the '
               'frozen 800 (scenario, regime, storage, choices) candidates, the 48 paired physical '
               'variants and fresh public-API LRU and causal-prefix microcases. Expected operations, '
               'tensors, placement and preparation scope come from public declarations, never from the '
               'trace. No new hardware timings; no re-solve of T*.'),
        nonclaims=['no universal expression completeness',
                   'no performance prediction accuracy',
                   'the 800 candidates reuse frozen T* / bounds; only T_C was re-executed'])
    save(out / 'SUMMARY.json', summary)
    with (out / 'COUNTS.csv').open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['case', 'cohort', 'category', 'checks', 'violations'])
        writer.writeheader()
        for r in rows:
            for k, n in r['checks'].items():
                writer.writerow(dict(case=r['id'], cohort=r['cohort'], category=k,
                                     checks=n, violations=r['violation_counts'].get(k, 0)))
    save(out / 'INPUT_HASHES.json', inputs)
    save(out / 'OUTPUT_HASHES.json', {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps({k: v for k, v in summary.items() if k != 'category_case_counts'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--physical', type=Path, default=PHYSICAL)
    args = parser.parse_args()
    main(args.out, args.physical)
