"""Fresh profile freeze then heldout, and immutable raw evidence verification.

This effective model prices the declared traced evaluator's graph construction,
allocation, materialization and weight lifecycle plus actual decoder computation.
Scalar reference, validation, weight-source generation and output serialization
are excluded by construction, never subtracted from measured wall time.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from profiling.workload_accuracy.contract import (QUALITY, canonical, digest, integer, same, sha,
                                        stage_keys, validate_case, validate_native, workload_key)


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def now():
    return datetime.now(timezone.utc).isoformat()


def cv(values):
    return statistics.pstdev(values)/statistics.mean(values) if values and statistics.mean(values)>0 else math.inf


def build_binding(binary):
    binary=Path(binary).resolve()
    manifest=json.loads(binary.with_suffix('.build.json').read_text(encoding='utf-8'))
    same(manifest.get('schema'),'workload_accuracy_build/1','native build schema')
    same(manifest.get('binary'),str(binary),'native binary path')
    same(manifest.get('binary_sha256'),sha(binary),'native binary hash')
    for field in ('sources','ggml_headers','runtime','link_libraries'):
        if not manifest.get(field):
            raise ValueError('missing build binding '+field)
        for path,expected in manifest[field].items():
            same(sha(path),expected,field+' SHA256')
    python_sources={str(p.resolve()):sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    return dict(native=manifest,python_sources=python_sources)


def environment(binding):
    env=os.environ.copy()
    directories=sorted({str(Path(p).parent) for p in binding['native']['runtime']})
    # Match linked shared libraries and turn off CUDA graph capture explicitly.
    env['PATH']=os.pathsep.join(directories+[env.get('PATH','')])
    env['GGML_CUDA_DISABLE_GRAPHS']='1'
    env['NVIDIA_TF32_OVERRIDE']='0'
    return env


@contextmanager
def hardware_lock(path=None):
    path=Path(path) if path else Path(__file__).resolve().parents[1]/'.measurement.lock'
    with path.open('x',encoding='utf-8') as stream:
        stream.write(canonical(dict(owner='workload_accuracy',pid=os.getpid(),created_utc=now())))
    try:
        yield
    finally:
        path.unlink()


def derive_case(base,seed):
    case=copy.deepcopy(base);old=case['seed'];case['seed']=seed
    for rd in case['rounds']:
        for row in rd['hidden_input']:
            for j,v in enumerate(row):
                row[j]=v+math.sin((seed-old)*.071+j*.03)*.025
    return validate_case(case)


def collect(binary,case,case_path,*,samples,warmup,binding):
    write(case_path,case)
    command=[str(Path(binary).resolve()),'--case',str(Path(case_path).resolve()),'--samples',str(samples),'--warmup',str(warmup)]
    started=now()
    try:
        result=subprocess.run(command,text=True,capture_output=True,env=environment(binding),timeout=300)
    except subprocess.TimeoutExpired as error:
        return dict(schema='workload_accuracy_raw/1',case=case,command=command,started_utc=started,
                    completed_utc=now(),returncode=None,timed_out=True,
                    stdout=(error.stdout or b'').decode(errors='replace') if isinstance(error.stdout,bytes) else error.stdout or '',
                    stderr=(error.stderr or b'').decode(errors='replace') if isinstance(error.stderr,bytes) else error.stderr or '',
                    samples=samples,warmup=warmup),None
    except OSError as error:
        return dict(schema='workload_accuracy_raw/1',case=case,command=command,started_utc=started,
                    completed_utc=now(),returncode=None,launch_error=str(error),stdout='',stderr='',
                    samples=samples,warmup=warmup),None
    raw=dict(schema='workload_accuracy_raw/1',case=case,command=command,started_utc=started,
             completed_utc=now(),returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
             samples=samples,warmup=warmup)
    if result.returncode:
        return raw,None
    try:
        native=json.loads(result.stdout)
        validate_native(case,native,samples=samples,warmup=warmup)
    except (ValueError,KeyError,TypeError) as error:
        raw['validation_error']=str(error)
        return raw,None
    return raw,native


def statistics_for(record):
    case=record['case'];native=record['native'];keys=stage_keys(case)
    stage_values={}
    for index,key in enumerate(keys):
        values=[s['stages'][index]['compute_ns'] for s in native['sample_timings']]
        stage_values.setdefault(key['key'],[]).extend(values)
    full=[s['timing']['full_workload_wall_ns'] for s in native['sample_timings']]
    gap=[s['timing']['host_setup_materialization_gap_ns'] for s in native['sample_timings']]
    return dict(stages={key:dict(median_ns=statistics.median(values),cv=cv(values)) for key,values in stage_values.items()},
                full=dict(median_ns=statistics.median(full),cv=cv(full)),
                gap=dict(median_ns=statistics.median(gap),cv=cv(gap)))


def freeze(profile,binding,suite):
    stages={};gaps={}
    for record in profile:
        if record['split']!='profile':
            raise ValueError('heldout data cannot fit coefficients')
        stats=statistics_for(record)
        for key,item in stats['stages'].items():
            stages.setdefault(key,[]).append(item)
        gaps.setdefault(workload_key(record['case']),[]).append(stats['gap'])
    def summarize(groups):
        return {key:dict(median_ns=statistics.median([x['median_ns'] for x in rows]),
                         within_process_cv_max=max(x['cv'] for x in rows),
                         between_process_cv=cv([x['median_ns'] for x in rows]),processes=len(rows))
                for key,rows in groups.items()}
    return dict(schema='workload_accuracy_frozen_model/1',frozen_utc=now(),
                native_binding_sha256=digest(binding),quality=copy.deepcopy(suite['quality']),
                stages=summarize(stages),host_gaps=summarize(gaps),
                host_gap_is_effective=True,
                prediction_scope='inference_graph_setup_alloc_materialization_weight_lifecycle_release',
                suitable_for_uninstrumented_runtime_prediction=False,
                profile_raw_sha256=[x['raw_sha256'] for x in profile])


def evaluate(heldout,model,*,expected_processes):
    quality=model['quality'];rows=[];groups={}
    for record in heldout:
        if record['split']!='heldout':
            raise ValueError('evaluation expects heldout records')
        case=record['case'];stats=statistics_for(record);keys=stage_keys(case);gap=model['host_gaps'].get(workload_key(case))
        covered=gap is not None and all(k['key'] in model['stages'] for k in keys)
        prediction=(sum(model['stages'][k['key']]['median_ns'] for k in keys)+gap['median_ns']) if covered else None
        error=abs(prediction-stats['full']['median_ns'])/stats['full']['median_ns'] if covered else None
        sources=[model['stages'][k['key']] for k in keys]+[gap] if covered else []
        gates=dict(exact_stage_and_host_gap_coverage=covered,numerical_memory_checks=True,
                   within_process_cv=stats['full']['cv']<=quality['max_within_process_cv'] and covered and all(x['within_process_cv_max']<=quality['max_within_process_cv'] for x in sources),
                   profile_between_process_cv=covered and all(x['between_process_cv']<=quality['max_between_process_cv'] and x['processes']>=expected_processes for x in sources),
                   heldout_relative_error=error is not None and error<=quality['max_heldout_relative_error'])
        row=dict(case_id=case['id'],seed=case['seed'],predicted_ns=prediction,observed_ns=stats['full']['median_ns'],
                 absolute_relative_error=error,within_process_cv=stats['full']['cv'],gates=gates)
        rows.append(row);groups.setdefault(case['id'],[]).append(row)
    summaries=[]
    for ident,items in groups.items():
        between=cv([x['observed_ns'] for x in items]);complete=len(items)==expected_processes
        for row in items:
            row['gates']['heldout_process_coverage']=complete
            row['gates']['heldout_between_process_cv']=between<=quality['max_between_process_cv']
            row['accepted']=all(row['gates'].values())
        summaries.append(dict(case_id=ident,processes=len(items),between_process_cv=between,accepted=all(x['accepted'] for x in items)))
    return dict(schema='workload_accuracy_report/1',quality=quality,rows=rows,cases=summaries,
                accepted=bool(rows) and all(row['accepted'] for row in rows),
                physical_capacity_verified=False,host_gap_is_effective=True,
                suitable_for_uninstrumented_runtime_prediction=False)


def validate_suite(value):
    same(value.get('schema'),'workload_accuracy_suite/1','suite schema')
    same(value.get('quality'),QUALITY,'original quality thresholds')
    integer(value.get('samples'),'samples',3,100);integer(value.get('warmup'),'warmup',0,10)
    n=integer(value.get('processes'),'processes',3,20)
    profile=value.get('profile_seeds',[]);heldout=value.get('heldout_seeds',[])
    if len(profile)!=n or len(heldout)!=n or len(set(profile+heldout))!=2*n:
        raise ValueError('fresh disjoint profile/heldout seeds with exact process coverage required')
    for seed in profile+heldout:
        integer(seed,'seed',0,1000000)
    cases=[validate_case(case) for case in value.get('cases',[])]
    if not cases or len({c['id'] for c in cases})!=len(cases):
        raise ValueError('unique nonempty cases required')
    return value


def _run_locked(binary,suite,output):
    suite=validate_suite(suite);output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    binding=build_binding(binary);write(output/'binding.json',binding);write(output/'suite.json',suite)
    records=[];frozen_sha=None;profile=[]
    for split in ('profile','heldout'):
        for ci,base in enumerate(suite['cases']):
            for pi,seed in enumerate(suite[split+'_seeds']):
                # Sources and binaries must remain identical at each boundary.
                same(build_binding(binary),binding,'source/runtime stability before process')
                if split=='heldout':
                    same(sha(output/'frozen-model.json'),frozen_sha,'model frozen before every heldout process')
                case=derive_case(base,seed)
                from profiling.workload_accuracy.public_audit import export_audit
                audit_directory=output/f'{split}-{ci}-{pi}.public-model'
                audit=export_audit(case,audit_directory)
                stem=f'{split}-{ci}-{pi}';raw,native=collect(binary,case,output/(stem+'.case.json'),samples=suite['samples'],warmup=suite['warmup'],binding=binding)
                raw['public_model_audit']=str(audit_directory.relative_to(output)/'workload-audit.json')
                raw['public_model_audit_sha256']=sha(audit_directory/'workload-audit.json')
                raw['binding_sha256_at_start']=digest(binding)
                raw['split']=split;raw['frozen_model_sha256_at_start']=frozen_sha
                rawpath=output/(stem+'.raw.json');write(rawpath,raw)
                if native is None:
                    raise RuntimeError('native failed; preserved '+str(rawpath)+'\n'+raw['stderr'])
                same(len(audit['round_layer_checks']),len(native['rows']),'complete native/public layer coverage')
                for check,row in zip(audit['round_layer_checks'],native['rows']):
                    for field in ('round','layer','tokens','history_before','history_after','kv_used_bytes'):
                        same(row[field],check[field],'actual native/public compiler '+field)
                same(build_binding(binary),binding,'source/runtime stability after process')
                if split=='heldout':
                    same(sha(output/'frozen-model.json'),frozen_sha,'frozen model unchanged during heldout process')
                record=dict(split=split,case=case,native=native,raw_output=rawpath.name,raw_sha256=sha(rawpath))
                records.append(record)
                if split=='profile':
                    profile.append(record)
        if split=='profile':
            model=freeze(profile,binding,suite);write(output/'frozen-model.json',model);frozen_sha=sha(output/'frozen-model.json')
        same(sha(output/'frozen-model.json'),frozen_sha,'frozen model unchanged')
    report=evaluate([r for r in records if r['split']=='heldout'],model,expected_processes=suite['processes'])
    write(output/'report.json',report)
    write(output/'manifest.json',dict(schema='workload_accuracy_campaign/1',created_utc=now(),
          frozen_model_sha256=frozen_sha,binding_sha256=sha(output/'binding.json'),suite_sha256=sha(output/'suite.json'),
          report_sha256=sha(output/'report.json'),records=[{k:v for k,v in r.items() if k not in ('case','native')} for r in records]))
    return report


def run(binary,suite,output):
    with hardware_lock():
        return _run_locked(binary,suite,output)


def verify(directory):
    root=Path(directory).resolve();manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    same(manifest.get('schema'),'workload_accuracy_campaign/1','campaign schema')
    for filename,key in (('binding.json','binding_sha256'),('suite.json','suite_sha256'),('report.json','report_sha256'),('frozen-model.json','frozen_model_sha256')):
        same(sha(root/filename),manifest.get(key),filename+' hash')
    suite=validate_suite(json.loads((root/'suite.json').read_text(encoding='utf-8')))
    model=json.loads((root/'frozen-model.json').read_text(encoding='utf-8'));records=[];seen=set()
    binding=json.loads((root/'binding.json').read_text(encoding='utf-8'))
    same(model['native_binding_sha256'],digest(binding),'frozen build/source binding')
    base_by_id={c['id']:c for c in suite['cases']}
    for record in manifest['records']:
        path=(root/record['raw_output']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('raw path escapes campaign')
        same(sha(path),record['raw_sha256'],'raw hash');raw=json.loads(path.read_text(encoding='utf-8'))
        same(raw['returncode'],0,'native exit status');case=validate_case(raw['case'])
        split=raw['split'];same(split,record['split'],'split binding')
        if split not in ('profile','heldout') or case['seed'] not in suite[split+'_seeds']:
            raise ValueError('undeclared seed/split')
        ident=(split,case['id'],case['seed'])
        if ident in seen:
            raise ValueError('duplicate process evidence')
        seen.add(ident)
        if case['id'] not in base_by_id:
            raise ValueError('undeclared case')
        same(case,derive_case(base_by_id[case['id']],case['seed']),'native case from declared source trace')
        same(raw.get('binding_sha256_at_start'),digest(binding),'per-process source/runtime binding')
        audit_path=(root/raw['public_model_audit']).resolve()
        if not audit_path.is_relative_to(root):
            raise ValueError('public audit path escapes campaign')
        same(sha(audit_path),raw['public_model_audit_sha256'],'same-case public compiler audit hash')
        audit=json.loads(audit_path.read_text(encoding='utf-8'))
        same(audit['case_sha256'],digest(case),'public/native case identity')
        same(audit['real_public_ModelPlan_runtime_success'],True,'public ModelPlan runtime success')
        command=raw['command']
        same(command[0],binding['native']['binary'],'bound native executable command')
        same(command[1],'--case','case argument')
        casefile=root/Path(command[2]).name
        same(json.loads(casefile.read_text(encoding='utf-8')),case,'saved native case')
        same(command[3:],['--samples',str(suite['samples']),'--warmup',str(suite['warmup'])],'native runtime arguments')
        if split=='heldout':
            same(raw['frozen_model_sha256_at_start'],manifest['frozen_model_sha256'],'prior freeze')
            if raw['started_utc']<=model['frozen_utc']:
                raise ValueError('heldout process predates model freeze')
        native=json.loads(raw['stdout']);validate_native(case,native,samples=suite['samples'],warmup=suite['warmup'])
        same(len(audit['round_layer_checks']),len(native['rows']),'complete replayed native/public layer coverage')
        for check,row in zip(audit['round_layer_checks'],native['rows']):
            for field in ('round','layer','tokens','history_before','history_after','kv_used_bytes'):
                same(row[field],check[field],'replayed native/public '+field)
        records.append(dict(record,case=case,native=native))
    expected={(s,c['id'],seed) for s in ('profile','heldout') for c in suite['cases'] for seed in suite[s+'_seeds']}
    same(sorted(seen),sorted(expected),'full process coverage')
    regenerated_model=freeze([r for r in records if r['split']=='profile'],binding,suite)
    regenerated_model['frozen_utc']=model['frozen_utc']
    same(regenerated_model,model,'profile-only regenerated coefficients')
    regenerated=evaluate([r for r in records if r['split']=='heldout'],model,expected_processes=suite['processes'])
    same(regenerated,json.loads((root/'report.json').read_text(encoding='utf-8')),'regenerated report')
    return regenerated


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    collect_parser=sub.add_parser('run')
    collect_parser.add_argument('--binary',type=Path,required=True)
    collect_parser.add_argument('--suite',type=Path,required=True)
    collect_parser.add_argument('--output',type=Path,required=True)
    collect_parser.add_argument('--allow-hardware',action='store_true',required=True)
    verify_parser=sub.add_parser('verify');verify_parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    result=run(args.binary,json.loads(args.suite.read_text(encoding='utf-8')),args.output) if args.command=='run' else verify(args.directory)
    print(json.dumps(result,indent=2))
