"""Repair shared-copy demand OR readiness; retain every earlier reference.

Actual executions are unchanged. Independent W/H/K graph and policy choices
regenerate references; production event data is used onlyfor posthoc validation.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from .campaign import SCENARIOS,PRIORITIES,scenario,declared_dag,placement,reference_nodes,schedule,write

def run(source,out):
    source,out=Path(source),Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('newoutputrequired')
    out.mkdir(parents=True,exist_ok=True)
    original=json.loads((source/'REPORT.json').read_text(encoding='utf-8'))
    protocol=dict(method='Replace first-consumer shared-COPY demand gate by OR over all legal consumer producer-readiness sets; actualIRrunsunchanged.',
        input_report=str((source/'REPORT.json').resolve()),input_sha256=sha256((source/'REPORT.json').read_bytes()).hexdigest(),
        source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        campaign_source_sha256=sha256(Path(__file__).with_name('campaign.py').read_bytes()).hexdigest(),
        independent_inputs='handwritten workloaddag fromfixedgeometry, choices andstorage; noobservedexecution usedtoinventdependencies',
        frozen_before_recomputation=True)
    write(out/'PROTOCOL.json',protocol)
    rows=[];actual_errors=[]
    structures={r[0]:r for r in SCENARIOS}
    for row in original['rows']:
        row=deepcopy(row)
        if row['status']!='pass':rows.append(row);continue
        name,pattern,attention,decode,requests=structures[row['scenario']]
        s=scenario(name,pattern,attention,decode,requests,row['regime']);d=declared_dag(s)
        nodes=reference_nodes(s,d,placement(s,d,row['choices']),row['storage'])
        candidates=[schedule(nodes,p) for p in PRIORITIES];best=min(range(len(candidates)),key=lambda i:candidates[i]['duration_ns'])
        path=out/'references'/row['id']/'independent-reference.json'
        write(path,dict(nodes=nodes,best_priority=PRIORITIES[best],schedule_durations_ns=[r['duration_ns'] for r in candidates],best=candidates[best]))
        events=json.loads((Path(row['artifact_directory'])/'events.json').read_text(encoding='utf-8'))
        actual={e['target_id']:e for e in events if e['kind']=='COMPUTE'}
        for identity,n in nodes.items():
            if n['kind']=='copy':
                actual[identity]=next(e for e in events if e['kind']=='COPY' and e['target_id']==n['tensor'] and
                    e['src']==('dram' if n['src']=='cpu' else 'vram') and e['dst']==('dram' if n['dst']=='cpu' else 'vram'))
        for identity,n in nodes.items():
            t=actual[identity]['start_ns']
            if not all(actual[parent]['end_ns']<=t for parent in n['deps']):actual_errors.append((row['id'],identity,'mandatory'))
            if not any(all(actual[parent]['end_ns']<=t for parent in alt) for alt in n.get('dependency_alternatives',[[]])):
                actual_errors.append((row['id'],identity,'alternative'))
        row.update(original_reference_family_best_ns=row['reference_family_best_ns'],
            reference_family_best_ns=candidates[best]['duration_ns'],
            reference_gap_fraction=row['duration_ns']/candidates[best]['duration_ns']-1,
            reference_distinct_durations=len({r['duration_ns'] for r in candidates}),
            corrected_reference_path=str(path.resolve()))
        rows.append(row)
    summaries=[]
    for old in original['summary']:
        old=deepcopy(old);current=[r for r in rows if r['status']=='pass' and r['scenario']==old['scenario'] and r['regime']==old['regime']]
        ref=min(r['reference_family_best_ns'] for r in current)
        old.update(reference_family_best_ns=ref,reference_gap_fraction=old['best_ns']/ref-1)
        summaries.append(old)
    doc=deepcopy(original);doc.update(rows=rows,summary=summaries,reference_correction=dict(protocol_path=str((out/'PROTOCOL.json').resolve()),
        previous_report=str((source/'REPORT.json').resolve()),actual_executions_unchanged=True,actual_reference_readiness_errors=actual_errors))
    write(out/'REPORT.json',doc);print(json.dumps(dict(conditions=len(rows),actual_readiness_errors=len(actual_errors)),indent=2))
    assert not actual_errors
    return doc

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.source,a.out)
