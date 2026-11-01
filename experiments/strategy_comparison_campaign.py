"""Run every declared fixed-hardware Dense/MoE execution/storage combination.

The workload, candidate and cost/source catalogue is frozen before any case.
Analytical simulation only: no native probing, numerical LLM or model fitting.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hardware_sensitivity import stable_hash, write_csv, write_json
from experiments.strategy_comparison import frozen_evidence, traffic_metrics
from experiments.strategy_comparison_metrics import timeline_metrics
from experiments.strategy_comparison_scenarios import (
    ScenarioSpec, scenarios, candidate_ids, candidate_spec, make_inputs,
    catalogue_document,
)
from experiments.strategy_comparison_analysis import analyze
from llm_simulator import AnalyticalCosts, Simulator, InvalidInput, Unsupported

WARNING = 'Uncalibrated analytical simulation; not measured hardware speedups or a new search algorithm.'


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execution_source_manifest():
    files = set()
    for directory in ('src/llm_simulator', 'modeling/llm_infer_model', 'mapping/tensor_mapping'):
        files.update((ROOT / directory).rglob('*.py'))
    for name in ('hardware_sensitivity.py', 'strategy_comparison.py',
                 'strategy_comparison_metrics.py', 'strategy_comparison_scenarios.py',
                 'strategy_comparison_analysis.py', 'strategy_comparison_campaign.py'):
        files.add(ROOT / 'experiments' / name)
    files.update(p for p in (ROOT / 'src/llm_simulator/data').rglob('*') if p.is_file())
    return {str(p.relative_to(ROOT)).replace('\\', '/'): sha_file(p) for p in sorted(files)}


def unused_gpu_weight_traffic(result, observation):
    """Post-run evaluation only; no future information is fed to mapping.

    Count copies whose immutable weight ID has no actual GPU consumer anywhere
    in this finite run. This is a conservative subset of speculative waste, not
    an exact useful-prefetch/cache-hit or individual-copy-use classification.
    """
    ops = {op.id: op for op in observation.operations}
    weights = {t.id for t in observation.tensors if t.role == 'weight'}
    consumed = set()
    for event in result.events:
        if event.kind == 'COMPUTE' and event.device == 'gpu':
            consumed.update(t for t in ops[event.target_id].inputs if t in weights)
    copies = [e for e in result.events if e.kind == 'COPY' and e.dst == 'vram'
              and e.target_id in weights and e.target_id not in consumed]
    return dict(weight_gpu_copy_bytes_with_no_gpu_consumer=sum(e.bytes_transferred for e in copies),
                weight_gpu_copy_count_with_no_gpu_consumer=len(copies))


def verify_row_metrics(row):
    if row.get('success') is not True:
        if row.get('completed_makespan_ns') is not None:
            raise ValueError('failed run cannot have completed latency')
        return
    total = row['completed_makespan_ns']
    if total <= 0 or row['any_engine_event_span_ns'] + row['all_engine_event_uncovered_ns'] != total:
        raise ValueError('elapsed event coverage does not match completed latency')
    if row['cpu_gpu_event_overlap_ns'] > min(row['cpu_event_span_ns'], row['gpu_event_span_ns']):
        raise ValueError('CPU/GPU intersection exceeds its operands')
    if row['copy_compute_event_overlap_ns'] > min(row['copy_event_span_ns'], row['compute_event_span_ns']):
        raise ValueError('copy/compute intersection exceeds its operands')
    traffic = row['copy_bytes_by_direction_and_role']
    if sum(traffic.get('dram->vram', {}).values()) != row['h2d_bytes']:
        raise ValueError('H2D summary differs from event role accounting')
    if sum(traffic.get('vram->dram', {}).values()) != row['d2h_bytes']:
        raise ValueError('D2H summary differs from event role accounting')


def run_case(spec, candidate, output, retain_traces=False, *, hardware_override=None):
    spec = ScenarioSpec(**spec) if isinstance(spec, dict) else spec
    folder = Path(output) / 'cases' / spec.id / candidate
    folder.mkdir(parents=True, exist_ok=False)
    model, hardware, workload = make_inputs(spec)
    if hardware_override is not None:
        hardware = replace(hardware, **hardware_override)
    scheduling = candidate_spec(spec.model_kind, candidate)
    row = dict(scenario_id=spec.id, model_kind=spec.model_kind,
        prefill_tokens=spec.prefill_tokens, decode_steps=spec.decode_steps,
        route_pattern=spec.route_pattern, candidate_id=candidate, success=False,
        calibrated=False, hardware_measurement=False, numerical_llm_execution=False,
        cost_model='analytical_roofline', completed_makespan_ns=None,
        artifact_directory=str(folder.relative_to(output)).replace('\\','/'),
        unavailable_metrics=dict(ttft='no modeled output-boundary extraction',
            tpot='no modeled output-boundary extraction',
            demand_cache_hit_rate='no first-demand residency audit',
            causal_wait_breakdown='not inferred from event occupation or ADVANCE'))
    row['declared_inputs_sha256'] = stable_hash(dict(model=asdict(model), hardware=asdict(hardware),
        workload=asdict(workload), workspace_bytes=0))
    row['strategy_sha256'] = stable_hash(asdict(scheduling))
    started = time.perf_counter()
    try:
        simulator = Simulator(model=model, hardware=hardware, workload=workload,
                              cost_model=AnalyticalCosts(workspace_bytes=0))
        with simulator.compile(scheduling=scheduling, directory=folder/'inputs') as compiled:
            ir = compiled.ir_document()
            write_json(folder/'compiled-ir.json', ir)
            row['compiled_ir_audit_sha256'] = stable_hash(ir)
            result = compiled.run()
            row.update(result.summary())
            row.pop('input_directory', None)
            events = [asdict(e) for e in result.events]
            write_json(folder/'events.json', events)
            row['event_sha256'] = stable_hash(events)
            row['event_counts'] = dict(Counter(e.kind for e in result.events))
            observation = result.raw[1].observe()
            write_json(folder/'operation-catalogue.json', [asdict(o) for o in observation.operations])
            write_json(folder/'tensor-catalogue.json', [asdict(t) for t in observation.tensors])
            write_json(folder/'policy-audit.json', result.raw[2].audit_document())
            write_json(folder/'decision-log.json', [dict(index=d.index,
                t_ns=d.observation.t_ns, observation_sha256=d.observation_sha256,
                decision=asdict(d.decision) if d.decision is not None else None,
                accepted=d.accepted, error_code=d.error_code, error_message=d.error_message)
                for d in result.raw[0].decisions])
            if result.success:
                row['completed_makespan_ns'] = result.makespan_ns
                row.update(timeline_metrics(result.events, result.makespan_ns))
                row.update(traffic_metrics(result))
                row.update(unused_gpu_weight_traffic(result, observation))
            else:
                row['diagnostic_partial_makespan_ns'] = result.makespan_ns
            if retain_traces:
                result.save(folder/'full-audit')
    except (Unsupported, InvalidInput) as error:
        row.update(status='unsupported_or_invalid_input', success=False,
            error_type=type(error).__name__, error_message=str(error))
    row['harness_wall_seconds'] = time.perf_counter() - started
    verify_row_metrics(row)
    write_json(folder/'summary.json', row)
    files = {str(p.relative_to(folder)).replace('\\','/'): sha_file(p)
             for p in sorted(folder.rglob('*')) if p.is_file()}
    write_json(folder/'artifacts.json', files)
    return row


def verify_equivalence_controls(records, catalogue):
    indexed = {(r['scenario_id'], r['candidate_id']): r for r in records}
    pairs=[]
    for control, original in catalogue['equivalence_controls'].items():
        for candidate in candidate_ids('moe'):
            left, right = indexed.get((control,candidate)), indexed.get((original,candidate))
            if left is None or right is None:
                continue
            fields=('success','status','error_code','termination_reason','completed_makespan_ns',
                    'h2d_bytes','d2h_bytes','peak_dram_bytes','peak_vram_bytes','event_sha256')
            same=all(left.get(k)==right.get(k) for k in fields)
            pairs.append(dict(control=control,reference=original,candidate_id=candidate,
                              equivalent=same))
            if not same:
                raise ValueError(f'equivalent workload control disagrees: {control}/{candidate}')
    return pairs


def _audit_declarations(records, selected):
    for spec in selected:
        rows=[r for r in records if r['scenario_id']==spec.id]
        if len(rows)!=len(candidate_ids(spec.model_kind)):
            raise ValueError('candidate coverage differs from the declared matrix')
        if len({r['declared_inputs_sha256'] for r in rows})!=1:
            raise ValueError('candidate inputs differ inside a scenario')
        if len({r['strategy_sha256'] for r in rows})!=len(rows):
            raise ValueError('duplicate declared strategy contract')
        for row in rows:
            verify_row_metrics(row)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--scenarios',nargs='+',help='optional declared scenario IDs; default is full matrix')
    parser.add_argument('--retain-traces',action='store_true')
    args=parser.parse_args(argv)
    if not 1<=args.workers<=4:
        parser.error('--workers must be between1 and4')
    selected=tuple(s for s in scenarios() if args.scenarios is None or s.id in args.scenarios)
    if not selected or args.scenarios and set(args.scenarios)-{s.id for s in selected}:
        parser.error('unknown scenario selection')
    output=args.out.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('--out must be new or empty')
    output.mkdir(parents=True,exist_ok=True)
    source_before=execution_source_manifest()
    evidence_before=frozen_evidence()
    catalogue=catalogue_document()
    jobs=[(s,c) for s in selected for c in candidate_ids(s.model_kind)]
    protocol=dict(format_version='strategy-comparison-campaign-0.1',warning=WARNING,
        catalogue=catalogue,selected_scenarios=[s.id for s in selected],
        selected_attempts=len(jobs),full_matrix=len(selected)==len(scenarios()),
        execution_sources_sha256=stable_hash(source_before),
        frozen_accuracy_evidence_before=evidence_before,
        statistics='deterministic analytical cases, not independent repeated timing samples',
        worker_count=args.workers,worker_wall_time_is_not_simulated_latency=True)
    write_json(output/'source-manifest-before.json',source_before)
    write_json(output/'protocol.json',protocol)
    records=[]
    started=time.perf_counter()
    with (output/'runs.jsonl').open('w',encoding='utf-8',newline='\n') as handle:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending={pool.submit(run_case,asdict(s),c,str(output),args.retain_traces):(s,c)
                     for s,c in jobs}
            for future in as_completed(pending):
                row=future.result()
                records.append(row)
                handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                handle.flush()
                elapsed=time.perf_counter()-started
                print(f"{len(records)}/{len(jobs)} {row['scenario_id']} {row['candidate_id']} "
                      f"success={row.get('success')} T={row.get('completed_makespan_ns')}ns "
                      f"wall={elapsed:.1f}s",flush=True)
    order={(s.id,c):i for i,(s,c) in enumerate(jobs)}
    records.sort(key=lambda r:order[(r['scenario_id'],r['candidate_id'])])
    write_json(output/'records.json',records)
    write_csv(output/'runs.csv',records)
    _audit_declarations(records,selected)
    controls=verify_equivalence_controls(records,catalogue)
    write_json(output/'equivalence-controls.json',controls)
    analysis=analyze(records)
    write_json(output/'analysis.json',analysis)
    for name in ('comparisons','prefetch_pairs','cache_pairs','execution_rank_reversals',
                 'interactions','one_shot_selections'):
        write_csv(output/(name+'.csv'),analysis[name])
    source_after=execution_source_manifest()
    evidence_after=frozen_evidence()
    write_json(output/'source-manifest-after.json',source_after)
    if source_after!=source_before:
        raise RuntimeError('execution source changed during campaign; results are not source-frozen')
    if evidence_after!=evidence_before:
        raise RuntimeError('frozen accuracy evidence changed during campaign')
    artifacts={f'{s.id}/{c}':sha_file(output/'cases'/s.id/c/'artifacts.json') for s,c in jobs}
    write_json(output/'artifact-index.json',artifacts)
    write_json(output/'completion.json',dict(complete=True,full_matrix=protocol['full_matrix'],
        expected_attempts=len(jobs),attempted=len(records),
        goal_reached=sum(bool(r.get('success')) for r in records),
        calibrated=False,hardware_measurement=False,source_unchanged=True,
        execution_sources_sha256=stable_hash(source_before),
        frozen_accuracy_evidence_unchanged=True,accuracy_file_count=len(evidence_after['files']),
        protocol_sha256=stable_hash(protocol),analysis_sha256=sha_file(output/'analysis.json'),
        records_sha256=sha_file(output/'records.json'),
        artifact_index_sha256=sha_file(output/'artifact-index.json'),
        equivalence_control_pairs=len(controls),equivalence_controls_pass=all(r['equivalent'] for r in controls),
        harness_wall_seconds=time.perf_counter()-started))
    print('COMPLETED '+str(output),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
