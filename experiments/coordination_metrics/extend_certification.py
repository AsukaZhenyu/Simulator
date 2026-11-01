"""Re-certify unresolved policies with a longer MILP budget.

Frozen artefacts are read-only: the rerun RESULTS and the case node files are
never modified. Every attempt writes its own per-row record and a summary into
a fresh output directory, so a failed or interrupted run cannot corrupt the
frozen certificates. T_C is the post-repair makespan already stored in the
rerun rows; only the per-policy bounds are recomputed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT, ROOT / 'src', ROOT / 'mapping', ROOT / 'modeling'):
    sys.path.insert(0, str(folder))

from experiments.coordination_metrics.exact_fairness import certify, read, write  # noqa: E402

RESULTS = ROOT / 'build/coordination-metrics-rerun-20261030/ready-selection/RESULTS.json'
CASES = ROOT / 'build/coordination-metrics-approved-20261030/exact/cases'
DEFAULT_OUT = ROOT / 'build/certification-extension-20261030/pilot'


def row_detail(row, certificate, elapsed):
    old = row['certificate']
    return dict(
        id=row['id'],
        scenario=row['scenario'],
        regime=row['regime'],
        storage=row['storage'],
        T_C_ns=row['T_C_ns'],
        old=dict(certified=old['certified'], lower=old['lower_bound_ns'],
                 upper=old['upper_bound_ns'], g_interval=old['g_interval']),
        new=dict(certified=certificate['certified'], lower=certificate['lower_bound_ns'],
                 upper=certificate['upper_bound_ns'], g=certificate['g'],
                 g_interval=certificate['g_interval'],
                 method=certificate.get('method'),
                 witness_kind=certificate.get('witness_kind')),
        milp=certificate.get('milp'),
        elapsed_seconds=elapsed,
    )


def main(out=DEFAULT_OUT, limit=30.0, ids=None, families=None, max_rows=None, resume=False):
    out = Path(out)
    if out.exists() and not resume:
        raise SystemExit(f'refusing existing output directory: {out}')
    out.mkdir(parents=True, exist_ok=True)
    document = read(RESULTS)
    rows = [r for r in document['rows'] if r['audit_status'] == 'pass' and not r['certificate']['certified']]
    if ids:
        wanted = set(ids)
        rows = [r for r in rows if r['id'] in wanted]
    if families:
        rows = [r for r in rows if r['scenario'] in set(families)]
    if max_rows:
        rows = rows[:max_rows]
    print(f'attempting {len(rows)} unresolved rows at limit={limit}s', flush=True)

    records = []
    started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        row = deepcopy(row)
        row['lower_bound'] = row['strengthened_lower_bound']
        nodes = read(CASES / row['id'] / 'nodes.json')
        upper = row['certificate']['feasible_optimal_or_upper_witness']
        t0 = time.perf_counter()
        try:
            certificate = certify(row, nodes, upper, limit)
            record = row_detail(row, certificate, time.perf_counter() - t0)
            write(out / 'rows' / f"{row['id']}.json", record)
            records.append(record)
            print(f"[{index}/{len(rows)}] {row['id']} certified={certificate['certified']} "
                  f"bounds=[{certificate['lower_bound_ns']},{certificate['upper_bound_ns']}] "
                  f"g={certificate['g']} wall={record['elapsed_seconds']:.1f}s", flush=True)
        except Exception as exc:  # keep going; a failed row must not hide the others
            record = dict(id=row['id'], scenario=row['scenario'], error_type=type(exc).__name__,
                          error=str(exc), elapsed_seconds=time.perf_counter() - t0)
            write(out / 'rows' / f"{row['id']}.json", record)
            records.append(record)
            print(f"[{index}/{len(rows)}] {row['id']} ERROR {type(exc).__name__}: {exc}", flush=True)

    certified = [r for r in records if 'new' in r and r['new']['certified']]
    unresolved = [r for r in records if 'new' in r and not r['new']['certified']]
    errors = [r for r in records if 'error_type' in r]
    tightened = [r for r in records if 'new' in r and r['new']['g_interval']
                 and (r['new']['g_interval'][1] - r['new']['g_interval'][0])
                 < (r['old']['g_interval'][1] - r['old']['g_interval'][0]) - 1e-12]
    by_family = defaultdict(Counter)
    for r in records:
        if 'new' not in r:
            by_family[r['scenario']]['errors'] += 1
            continue
        key = 'certified' if r['new']['certified'] else 'unresolved'
        by_family[r['scenario']][key] += 1
    milp_seconds = [r['milp']['elapsed_seconds'] for r in records
                    if isinstance(r.get('milp'), dict) and isinstance(r['milp'].get('elapsed_seconds'), (int, float))]
    summary = dict(
        schema='certification-extension-1',
        generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        inputs=dict(results=str(RESULTS), cases=str(CASES), limit_seconds=limit,
                    attempted_rows=len(rows)),
        counts=dict(attempted=len(rows), newly_certified=len(certified),
                    still_unresolved=len(unresolved), errors=len(errors),
                    intervals_tightened=len(tightened)),
        by_family={family: dict(counter) for family, counter in sorted(by_family.items())},
        certified_ids=sorted(r['id'] for r in certified),
        total_wall_seconds=time.perf_counter() - started,
        milp_wall_seconds=dict(total=sum(milp_seconds), max=max(milp_seconds, default=None),
                               mean=sum(milp_seconds) / len(milp_seconds) if milp_seconds else None),
        scope=('Per-policy bounds only. T_C is the post-repair makespan from the rerun. '
               'Frozen certificates and the frozen corpus maximum are unchanged; '
               'a new certificate here is a separate, additional proof.'),
    )
    write(out / 'SUMMARY.json', summary)
    print(json.dumps(summary['counts'], ensure_ascii=False), flush=True)
    print(f"summaries written to {out}", flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(DEFAULT_OUT))
    parser.add_argument('--limit', type=float, default=30.0)
    parser.add_argument('--ids', nargs='*')
    parser.add_argument('--families', nargs='*')
    parser.add_argument('--max-rows', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    main(out=args.out, limit=args.limit, ids=args.ids, families=args.families,
         max_rows=args.max_rows, resume=args.resume)
