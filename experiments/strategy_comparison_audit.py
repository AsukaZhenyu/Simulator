"""Independent stdlib-only, read-only audit of a completed strategy campaign.

No simulator/modeling/mapping/analysis helpers are imported. Output is a sibling
JSON receipt; retained campaign artifacts are never changed.
"""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import time


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def union(spans):
    merged=[]
    for a,b in sorted(spans):
        if a==b:
            continue
        if merged and a<=merged[-1][1]:
            merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
        else:
            merged.append((a,b))
    return merged


def length(spans):
    return sum(b-a for a,b in spans)


def intersect(a,b):
    # Deliberately independent Cartesian intersections of disjoint union spans.
    return sum(max(0,min(d,y)-max(c,x)) for c,d in a for x,y in b)


def main():
    if not __debug__:
        raise RuntimeError('This audit requires assertions enabled; do not use Python -O.')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1],
                        help='repository root used to verify execution and accuracy files')
    args=parser.parse_args()
    campaign=args.campaign.resolve()
    output=args.output.resolve()
    root=args.root.resolve()
    assert not output.is_relative_to(campaign), 'audit output must be outside the campaign'
    started=time.perf_counter()
    completion=read(campaign/'completion.json')
    protocol=read(campaign/'protocol.json')
    records=read(campaign/'records.json')
    analysis=read(campaign/'analysis.json')
    index=read(campaign/'artifact-index.json')
    assert completion['complete'] and completion['full_matrix']
    assert len(records)==completion['expected_attempts']==completion['attempted']==352
    assert completion['records_sha256']==sha(campaign/'records.json')
    assert completion['analysis_sha256']==sha(campaign/'analysis.json')
    assert completion['artifact_index_sha256']==sha(campaign/'artifact-index.json')
    assert completion['protocol_sha256']==canonical(protocol)
    before=read(campaign/'source-manifest-before.json')
    after=read(campaign/'source-manifest-after.json')
    assert before==after
    assert canonical(before)==completion['execution_sources_sha256']
    assert all(sha(root/path)==digest for path,digest in before.items())
    accuracy=protocol['frozen_accuracy_evidence_before']['files']
    assert all(sha(root/e['path'])==e['sha256'] for e in accuracy)
    def identity(r): return r['scenario_id'],r['candidate_id']
    indexed={identity(r):r for r in records}
    assert len(indexed)==352
    streamed=[json.loads(s) for s in (campaign/'runs.jsonl').read_text(encoding='utf-8').splitlines()]
    assert {identity(r):r for r in streamed}==indexed and len(streamed)==352
    assert analysis['totals']['complete_count']==completion['goal_reached']
    declared={s['spec']['id']:s for s in protocol['catalogue']['scenarios']}
    assert len(declared)==16
    counts=Counter(r['scenario_id'] for r in records)
    for sid,spec in declared.items():
        assert counts[sid]==(16 if spec['spec']['model_kind']=='dense' else 24)
        assert len({r['declared_inputs_sha256'] for r in records if r['scenario_id']==sid})==1

    checks=Counter()
    for row in records:
        assert row['success'] is True and row['status']=='valid' and row['termination_reason']=='goal_reached'
        assert row['calibrated'] is False and row['hardware_measurement'] is False
        assert row['numerical_llm_execution'] is False and row['cost_model']=='analytical_roofline'
        folder=campaign/row['artifact_directory']
        scenario=declared[row['scenario_id']]
        assert row['declared_inputs_sha256']==canonical(dict(model=scenario['model'],
            hardware=scenario['hardware'],workload=scenario['workload'],workspace_bytes=0))
        assert row['strategy_sha256']==canonical(
            protocol['catalogue']['strategies'][row['model_kind']][row['candidate_id']])
        assert read(folder/'summary.json')==row
        manifest=read(folder/'artifacts.json')
        assert sha(folder/'artifacts.json')==index['/'.join(identity(row))]
        assert set(manifest)=={str(p.relative_to(folder)).replace('\\','/') for p in folder.rglob('*')
                              if p.is_file() and p.name!='artifacts.json'}
        for rel,digest in manifest.items():
            assert sha(folder/rel)==digest,(identity(row),rel,'artifact digest')
            checks['case_artifact_files_hashed']+=1
        events=read(folder/'events.json')
        assert {e['kind'] for e in events}<={'COMPUTE','COPY','ADVANCE','EVICT'}
        assert row['ssd_read_bytes']==row['ssd_write_bytes']==0
        assert canonical(events)==row['event_sha256']
        assert canonical(read(folder/'compiled-ir.json'))==row['compiled_ir_audit_sha256']
        assert read(folder/'inputs/environment.json')['events']==[]
        metadata=read(folder/'inputs/analytical-metadata.json')
        assert metadata['calibrated'] is False and metadata['hardware_measurements_used'] is False
        assert metadata['workspace_bytes_per_macro']==0
        assert metadata['hardware']==scenario['hardware'] and metadata['model']==scenario['model']
        for key in ('prefill_tokens','decode_steps','batch_size','route_source'):
            assert metadata['workload'][key]==scenario['workload'][key]
        assert metadata['workload']['route_count']==len(scenario['workload']['routes'])
        generated_routes=[]
        for round_ in read(folder/'inputs/model.global.json')['rounds']:
            for invocation in round_['invocations']:
                generated_routes.extend(read(folder/'inputs'/invocation['invocation_file']).get('route_trace',[]))
        def route_signature(r):
            return r['request_id'],r['layer_id'],r['absolute_position'],tuple(r['expert_ids'])
        assert sorted(map(route_signature,generated_routes))==sorted(map(route_signature,scenario['workload']['routes']))
        checks['compiled_workload_and_route_inputs_checked']+=1
        ops={o['id']:o for o in read(folder/'operation-catalogue.json')}
        tensors={t['id']:t for t in read(folder/'tensor-catalogue.json')}
        total=row['completed_makespan_ns']
        spans=defaultdict(list)
        device_spans=defaultdict(list)
        roles=defaultdict(Counter)
        points=[]
        computed=Counter()
        weights=Counter()
        reload_bytes=0
        for e in events:
            if e['kind'] in ('COMPUTE','COPY'):
                assert 0<=e['start_ns']<=e['end_ns']<=total
                spans[e['resource']].append((e['start_ns'],e['end_ns']))
                points.extend([(e['start_ns'],e['action_index'],0,e),
                               (e['end_ns'],e['action_index'],1,e)])
            elif e['kind']=='EVICT':
                points.append((e['start_ns'],e['action_index'],0,e))
            if e['kind']=='COMPUTE':
                op=ops[e['target_id']]
                price=next(p for p in op['prices'] if p['device']==e['device'])
                assert e['end_ns']-e['start_ns']==price['duration_ns']
                assert e['workspace_bytes']==price['allocated_workspace_bytes']==0
                computed[e['target_id']]+=1
                device_spans[e['device']].append((e['start_ns'],e['end_ns']))
                checks['compute_events_price_checked']+=1
            elif e['kind']=='COPY':
                tensor=tensors[e['target_id']]
                price=next(p for p in tensor['transfers'] if (p['src'],p['dst'])==(e['src'],e['dst']))
                assert e['end_ns']-e['start_ns']==price['duration_ns']
                assert e['bytes_transferred']==tensor['storage_bytes']
                roles[e['src']+'->'+e['dst']][tensor['role']]+=e['bytes_transferred']
                device_spans['copy'].append((e['start_ns'],e['end_ns']))
                if tensor['role']=='weight':
                    key=(tensor['id'],e['src'],e['dst'])
                    if weights[key]: reload_bytes+=e['bytes_transferred']
                    weights[key]+=1
                checks['copy_events_price_checked']+=1
        assert set(computed)==set(ops) and all(v==1 for v in computed.values())
        assert all(o['status']=='DONE' for o in ops.values())
        for resource,ss in spans.items():
            ss.sort()
            assert all(a[1]<=b[0] for a,b in zip(ss,ss[1:])),(identity(row),resource,'resource overlap')
        assert {k:dict(v) for k,v in roles.items()}==row['copy_bytes_by_direction_and_role']
        assert sum(roles['dram->vram'].values())==row['h2d_bytes']
        assert sum(roles['vram->dram'].values())==row['d2h_bytes']
        assert sum(v-1 for v in weights.values())==row['immutable_weight_reload_count']
        assert reload_bytes==row['immutable_weight_reload_bytes']
        cpu,gpu,copy=(union(device_spans[k]) for k in ('cpu','gpu','copy'))
        comp=union(cpu+gpu)
        expected={'cpu_event_span_ns':length(cpu),'gpu_event_span_ns':length(gpu),
            'copy_event_span_ns':length(copy),'compute_event_span_ns':length(comp),
            'cpu_gpu_event_overlap_ns':intersect(cpu,gpu),
            'copy_compute_event_overlap_ns':intersect(comp,copy),
            'any_engine_event_span_ns':length(union(comp+copy)),
            'all_engine_event_uncovered_ns':total-length(union(comp+copy))}
        assert all(row[k]==v for k,v in expected.items())

        ready={(t['id'],loc) for t in tensors.values() for loc in t['initial_locations']}
        allocated=set(ready)
        locks=Counter()
        mem=Counter()
        def size(tid,loc): return dict(tensors[tid]['allocation_bytes'])[loc]
        for tid,loc in allocated: mem[loc]+=size(tid,loc)
        peak=dict(mem)
        for t,index_,end,e in sorted(points,key=lambda p:p[:3]):
            kind=e['kind'];tid=e['target_id']
            loc=e['device'] if kind=='COMPUTE' else e.get('dst')
            loc={'cpu':'dram','gpu':'vram'}.get(loc,loc)
            if kind=='COPY' and not end:
                assert (tid,e['src']) in ready and (tid,loc) not in allocated
                allocated.add((tid,loc));mem[loc]+=size(tid,loc)
                locks[(tid,e['src'])]+=1
            elif kind=='COPY':
                ready.add((tid,loc));locks[(tid,e['src'])]-=1
            elif kind=='COMPUTE':
                op=ops[tid]
                if not end:
                    for inp in op['inputs']:
                        assert (inp,loc) in ready,(identity(row),tid,inp,'compute input readiness')
                        locks[(inp,loc)]+=1
                    for out in op['outputs']:
                        if (out,loc) not in allocated:
                            allocated.add((out,loc));mem[loc]+=size(out,loc)
                else:
                    for inp in op['inputs']: locks[(inp,loc)]-=1
                    ready.update((out,loc) for out in op['outputs'])
            elif kind=='EVICT':
                loc=e['location']
                assert (tid,loc) in ready and locks[(tid,loc)]==0
                assert e['bytes_released']==size(tid,loc)
                ready.remove((tid,loc));allocated.remove((tid,loc));mem[loc]-=size(tid,loc)
            for domain,n in mem.items():
                assert n>=0
                peak[domain]=max(peak.get(domain,0),n)
        assert all(v==0 for v in locks.values())
        assert ready=={(t['id'],loc) for t in tensors.values() for loc,status in t['copies'] if status=='READY'}
        for tensor in tensors.values():
            if tensor['goal_location']:
                assert (tensor['id'],tensor['goal_location']) in ready
                checks['final_tensor_goals_readiness_checked']+=1
        hw=declared[row['scenario_id']]['hardware']
        assert peak.get('dram',0)==row['peak_dram_bytes']<=hw['dram_capacity_bytes']
        assert peak.get('vram',0)==row['peak_vram_bytes']<=hw['vram_capacity_bytes']
        checks['event_replays_with_exact_memory_peak_and_dependency_checks']+=1

    controls=read(campaign/'equivalence-controls.json')
    assert len(controls)==48 and all(p['equivalent'] for p in controls)
    for pair in controls:
        a=indexed[(pair['control'],pair['candidate_id'])]
        b=indexed[(pair['reference'],pair['candidate_id'])]
        for key in ('event_sha256','completed_makespan_ns','h2d_bytes','d2h_bytes','peak_dram_bytes','peak_vram_bytes'):
            assert a[key]==b[key]
    receipt=dict(format_version='independent-stdlib-campaign-audit-0.1',passed=True,
        campaign=str(campaign),attempts_checked=len(records),distinct_inputs=14,equivalence_controls=2,
        equivalence_control_pairs_checked=48,source_files_hashed=len(before),
        frozen_accuracy_files_hashed=len(accuracy),checks=dict(checks),
        completed_campaign_sha256=sha(campaign/'completion.json'),
        records_sha256=sha(campaign/'records.json'),analysis_sha256=sha(campaign/'analysis.json'),
        independent_audit_script_sha256=sha(Path(__file__)),
        limitations=['This checks artifact integrity and internal execution/metric consistency, not hardware accuracy.',
            'No simulator, modeling, mapping or campaign-analysis code is imported by this audit.',
            'Goal and copy granularity are the declared contracts; no physical realism is inferred.'],
        wall_seconds=time.perf_counter()-started)
    output.write_text(json.dumps(receipt,indent=2,ensure_ascii=False,allow_nan=False)+'\n',
        encoding='utf-8',newline='\n')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':
    main()
