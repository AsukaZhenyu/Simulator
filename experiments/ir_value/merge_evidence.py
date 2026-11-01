"""Combine retained base and corrected MoE fixture campaigns transparently."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

def digest(p):return sha256(Path(p).read_bytes()).hexdigest()

def merge(base,correction,out):
    base,correction,out=map(Path,(base,correction,out))
    if out.exists() and any(out.iterdir()):raise ValueError('mergedoutputmustbeempty')
    out.mkdir(parents=True,exist_ok=True)
    a=json.loads((base/'REPORT.json').read_text(encoding='utf-8'))
    b=json.loads((correction/'REPORT.json').read_text(encoding='utf-8'))
    assert len(b['rows'])==128 and all(r['scenario']=='moe' for r in b['rows'])
    initial_failures=[r for r in a['rows'] if r['scenario']=='moe']
    assert len(initial_failures)==128
    assert all(r.get('error')=='independent compute price mismatch g:request0:request_arrival' for r in initial_failures)
    rows=[]
    for source,data in ((base,[r for r in a['rows'] if r['scenario']!='moe']),(correction,b['rows'])):
        for row in data:
            prefix,index,storage=row['id'].rsplit('-',2)
            rows.append(dict(row,artifact_directory=str((source/prefix/f'{index}-{storage}').resolve())))
    doc=dict(schema='ir-value-combined-evidence-1',conditions=len(rows),passed=sum(r['status']=='pass' for r in rows),
        failed=sum(r['status']!='pass' for r in rows),rows=rows,summary=a['summary']+b['summary'],
        imported_production_sources_unchanged=a['imported_production_sources_unchanged'] and b['imported_production_sources_unchanged'],
        code_sha256_unchanged=a['code_sha256_unchanged'] and b['code_sha256_unchanged'],
        exact_dense_certificates_path=str((base/'exact-dense-certificates/REPORT.json').resolve()),
        retained_campaigns=[dict(path=str(p.resolve()),report_sha256=digest(p/'REPORT.json'),protocol_sha256=digest(p/'PROTOCOL.json')) for p in (base,correction)],
        corrective_rerun=dict(original_moe_assertion_failures=128,rerun_conditions=128,
            cause='Independentreference mistakenlypricedMoEinputpreparation32ns rather than declaredRequestFlow(hidden32+I32position4)36ns.',
            production_changes=0,original_failures_retained=True,base_campaign_source_snapshot=str((base/'campaign.source-v2.py').resolve())),
        scope='Analytical mechanismexperiment; boundedreference exceptexacttinyDensecertificate. Grid928includes128fixedDRAMhandoffcontractconflicts; allrawattempts1056retained.')
    (out/'REPORT.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(str(out/'REPORT.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--correction',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();merge(args.base,args.correction,args.out)
