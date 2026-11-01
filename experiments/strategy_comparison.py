"""Paired finite strategy comparison on fixed, explicitly analytical hardware."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hardware_sensitivity import Point, make_inputs, stable_hash, write_csv, write_json
from experiments.strategy_comparison_metrics import timeline_metrics
from llm_simulator import AnalyticalCosts, Cache, Prefetch, Scheduling, Simulator, InvalidInput, Unsupported

WARNING = 'Uncalibrated analytical pilot, not hardware measurements or a search algorithm.'
CANDIDATES = tuple(f'e{e}-s{s}' for e in range(4) for s in range(2))
REFERENCE = 'e0-s0'
POINT = Point('fixed-512k', vram_bytes=512 * 1024)
CACHE_BYTES = 192 * 1024


def candidate_spec(kind, candidate):
    if kind not in ('dense', 'moe') or candidate not in CANDIDATES:
        raise ValueError('unknown declared model or candidate')
    execution, storage = (int(x[1:]) for x in candidate.split('-'))
    ffn = 'dense_ffn' if kind == 'dense' else 'expert_ffn'
    placements = ({}, {}, {ffn: 'cpu'}, {'attention': 'cpu'})[execution]
    return Scheduling(default_device='cpu' if execution == 1 else 'gpu',
        placements=placements, release='last_use',
        cache=Cache(rule='lru', budget_bytes=CACHE_BYTES),
        prefetch=Prefetch(next_layers=1, max_count=2, include_dense_weights=True)
                 if storage else Prefetch())


def frozen_evidence():
    """Record existing evidence if present, without making it a dependency."""
    index = ROOT / 'docs/data/single-cpu-gpu-acceptance-20261023.json'
    if not index.exists():
        return {'available': False, 'files': []}
    entries = json.loads(index.read_text(encoding='utf-8'))['accuracy_evidence']['files']
    rows = []
    for entry in entries:
        path = ROOT / entry['path']
        rows.append({'path': entry['path'], 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()
                     if path.exists() else None})
    return {'available': all(r['sha256'] is not None for r in rows), 'files': rows}


def traffic_metrics(result):
    tensors = {t.id: t for t in result.raw[1].observe().tensors}
    breakdown = defaultdict(Counter)
    copied_weights = set()
    reload_bytes = reload_count = weight_evictions = weight_released = 0
    for event in result.events:
        tensor = tensors.get(event.target_id)
        if event.kind == 'COPY':
            if tensor is None:
                raise ValueError('COPY target missing from final tensor catalogue')
            direction = f'{event.src}->{event.dst}'
            breakdown[direction][tensor.role] += event.bytes_transferred
            if tensor.role == 'weight':
                key = (event.target_id, event.src, event.dst)
                if key in copied_weights:
                    reload_count += 1
                    reload_bytes += event.bytes_transferred
                copied_weights.add(key)
        elif event.kind == 'EVICT' and tensor is not None and tensor.role == 'weight':
            weight_evictions += 1
            weight_released += event.bytes_released
    return dict(copy_bytes_by_direction_and_role={k: dict(v) for k, v in sorted(breakdown.items())},
        immutable_weight_reload_count=reload_count, immutable_weight_reload_bytes=reload_bytes,
        weight_eviction_count=weight_evictions, weight_released_allocation_bytes=weight_released)


def run_candidate(kind, candidate, output, retain_traces=False):
    folder = output / 'cases' / kind / candidate
    folder.mkdir(parents=True)
    model, hardware, workload = make_inputs(kind, POINT)
    scheduling = candidate_spec(kind, candidate)
    row = dict(model_kind=kind, candidate_id=candidate, calibrated=False,
               hardware_measurement=False, numerical_llm_execution=False,
               cost_model='analytical_roofline', completed_makespan_ns=None,
               unsupported_metrics={'ttft': 'no modeled output boundary extraction',
                   'tpot': 'no modeled output boundary extraction',
                   'cache_hit_rate': 'no first-demand residency observation',
                   'causal_wait_breakdown': 'not derived from elapsed idle or ADVANCE'})
    row['declared_inputs_sha256'] = stable_hash(dict(model=asdict(model), hardware=asdict(hardware),
        workload=asdict(workload), workspace_bytes=0))
    row['strategy_sha256'] = stable_hash(asdict(scheduling))
    started = time.perf_counter()
    try:
        simulator = Simulator(model=model, hardware=hardware, workload=workload,
                              cost_model=AnalyticalCosts(workspace_bytes=0))
        with simulator.compile(scheduling=scheduling, directory=folder / 'inputs') as compiled:
            ir = compiled.ir_document()
            write_json(folder / 'compiled-ir.json', ir)
            row['compiled_ir_audit_sha256'] = stable_hash(ir)
            result = compiled.run()
            row.update(result.summary())
            row.pop('input_directory', None)
            events = [asdict(e) for e in result.events]
            write_json(folder / 'events.json', events)
            row['event_sha256'] = stable_hash(events)
            row['event_counts'] = dict(Counter(e.kind for e in result.events))
            if result.success:
                row['completed_makespan_ns'] = result.makespan_ns
                row.update(timeline_metrics(result.events, result.makespan_ns))
                row.update(traffic_metrics(result))
            else:
                row['diagnostic_partial_makespan_ns'] = result.makespan_ns
            if retain_traces:
                result.save(folder / 'full-audit')
    except (Unsupported, InvalidInput) as error:
        row.update(status='unsupported_or_invalid_input', success=False,
                   error_type=type(error).__name__, error_message=str(error))
    row['harness_wall_seconds'] = time.perf_counter() - started
    write_json(folder / 'summary.json', row)
    return row


def relative_rows(records):
    index = {(r['model_kind'], r['candidate_id']): r for r in records}
    result = []
    for row in records:
        ref = index[(row['model_kind'], REFERENCE)]
        a, b = ref.get('completed_makespan_ns'), row.get('completed_makespan_ns')
        result.append(dict(model_kind=row['model_kind'], candidate_id=row['candidate_id'],
            success=row.get('success', False), completed_makespan_ns=b,
            reference_speedup=a / b if a and b else None,
            h2d_bytes=row.get('h2d_bytes'), d2h_bytes=row.get('d2h_bytes'),
            peak_dram_bytes=row.get('peak_dram_bytes'), peak_vram_bytes=row.get('peak_vram_bytes')))
    return result


def paired_interactions(records):
    index = {(r['model_kind'], r['candidate_id']): r for r in records}
    rows = []
    for kind in sorted({r['model_kind'] for r in records}):
        for e in range(1, 4):
            ids = [REFERENCE, f'e{e}-s0', 'e0-s1', f'e{e}-s1']
            times = [index[(kind, c)].get('completed_makespan_ns') for c in ids]
            row = dict(model_kind=kind, execution=f'e{e}', candidates=ids,
                       complete_four_cells=all(t is not None and t > 0 for t in times),
                       joint_gain_over_better_single=None, nonadditive_time_saving_ns=None,
                       strict_execution_order_reversal=None)
            if row['complete_four_cells']:
                t00, t10, t01, t11 = times
                single = min(t10, t01)
                row.update(joint_gain_over_better_single=(single-t11)/single,
                    nonadditive_time_saving_ns=t10+t01-t00-t11,
                    strict_execution_order_reversal=(t00-t10)*(t01-t11) < 0)
            rows.append(row)
    return rows


def report(output, records, comparisons, interactions):
    lines = ['# Fixed-hardware strategy comparison pilot', '', f'**{WARNING}**', '',
        'Eight predeclared execution/prefetch combinations per model. The cache rule and budget are identical; only placement and prefetch differ. This is a finite cold workload, not saturated serving.', '',
        '| Model | Candidate | Complete | Simulated ms | Reference ratio | H2D bytes | D2H bytes | Peak VRAM bytes |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for r in comparisons:
        t = '' if r['completed_makespan_ns'] is None else f"{r['completed_makespan_ns']/1e6:.6f}"
        speed = '' if r['reference_speedup'] is None else f"{r['reference_speedup']:.4f}"
        lines.append(f"| {r['model_kind']} | {r['candidate_id']} | {r['success']} | {t} | {speed} | {r['h2d_bytes']} | {r['d2h_bytes']} | {r['peak_vram_bytes']} |")
    lines += ['', '## Paired interaction cells', '',
        '| Model | Execution vs e0 | All four complete | Joint gain over better single | Nonadditive saving ns | Strict order reversal |',
        '|---|---|---|---:|---:|---|']
    for r in interactions:
        gain = '' if r['joint_gain_over_better_single'] is None else f"{100*r['joint_gain_over_better_single']:.3f}%"
        lines.append(f"| {r['model_kind']} | {r['execution']} | {r['complete_four_cells']} | {gain} | {r['nonadditive_time_saving_ns']} | {r['strict_execution_order_reversal']} |")
    lines += ['', '## Evidence boundaries', '',
        '- Synthetic F32 macro costs and idealized zero scratch, explicitly chosen; no new measurement or fit.',
        '- All candidates receive identical model, route trace, initial DRAM weights, empty KV and mandatory final DRAM goals.',
        '- Failures remain in runs.jsonl and never receive completed latency or a reference speedup.',
        '- CPU/GPU/COPY fields describe unions/intersections of event spans, not measured hardware utilization or attributed waits.',
        '- TTFT, TPOT, demand cache-hit rate and causal wait breakdown remain unavailable rather than zero.',
        '- The tested storage change is same-budget prefetch only. General cache-policy, partitions, expert adaptation and multi-request comparisons remain future experiments.',
        '- A finite analytical ranking is not a real-device speedup, universal optimum or proof of a new paper contribution.', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--models', nargs='+', choices=('dense', 'moe'), default=['dense', 'moe'])
    parser.add_argument('--retain-traces', action='store_true')
    args = parser.parse_args(argv)
    output = args.out.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('--out must be new or empty')
    output.mkdir(parents=True, exist_ok=True)
    kinds = tuple(dict.fromkeys(args.models))
    evidence_before = frozen_evidence()
    protocol = dict(format_version='strategy-comparison-pilot-0.1', warning=WARNING,
        calibrated=False, measured_profiles_used=False, frozen_evidence_before=evidence_before,
        candidates=list(CANDIDATES), reference=REFERENCE, models={},
        initial_state='cold weights in DRAM; empty KV',
        final_goals='mandatory hidden/KV in DRAM; optional retained weight copies may differ',
        candidate_budget=8, storage_ablation='same LRU rule and192KiB budget; only prefetch changes',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        metrics_sha256=hashlib.sha256((ROOT/'experiments/strategy_comparison_metrics.py').read_bytes()).hexdigest())
    for kind in kinds:
        model, hardware, workload = make_inputs(kind, POINT)
        protocol['models'][kind] = dict(model=asdict(model), hardware=asdict(hardware),
            workload=asdict(workload), strategies={c: asdict(candidate_spec(kind,c)) for c in CANDIDATES})
    write_json(output/'protocol.json', protocol)
    records=[]
    with (output/'runs.jsonl').open('w', encoding='utf-8', newline='\n') as handle:
        for kind in kinds:
            for candidate in CANDIDATES:
                row=run_candidate(kind,candidate,output,args.retain_traces)
                records.append(row)
                handle.write(json.dumps(row,allow_nan=False)+'\n'); handle.flush()
                print(f"{len(records)}/{len(kinds)*len(CANDIDATES)} {kind} {candidate}: {row.get('success')} {row.get('completed_makespan_ns')} ns",flush=True)
    comparisons, interactions=relative_rows(records),paired_interactions(records)
    write_csv(output/'runs.csv',records)
    write_csv(output/'comparisons.csv',comparisons)
    write_csv(output/'interactions.csv',interactions)
    report(output,records,comparisons,interactions)
    evidence_after=frozen_evidence()
    if evidence_before!=evidence_after:
        raise RuntimeError('existing accuracy evidence changed during the campaign')
    write_json(output/'completion.json',dict(complete=True,attempts=len(records),
        goal_reached=sum(bool(r.get('success')) for r in records),calibrated=False,
        hardware_measurement=False,protocol_sha256=stable_hash(protocol),
        frozen_evidence_unchanged=True,frozen_evidence_after=evidence_after))
    print(output/'REPORT.md',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
