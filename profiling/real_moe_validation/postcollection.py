"""Independent postcollection OLMoE route/shape audit and replay preparation.

This additive module never edits the frozen real-model executable, requests or
raw outputs. A valid route inventory establishes workload coverage, not matching
latency prices or equivalence of hidden routes in an uninstrumented execution.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

from .dataset import read,write
from .run import make_request,receipt,require

CONTRACT=dict(architecture='olmoe',layers=16,experts=64,top_k=8)


def effective_parameters(stderr):
    expected=dict(n_batch=256,n_ubatch=256,n_seq_max=1);observed={};missing=[];wrong=[]
    for key,value in expected.items():
        values=sorted({int(v) for v in re.findall(r'\b'+key+r'\s*=\s*(\d+)',stderr)})
        observed[key]=values
        if not values:missing.append(key)
        elif values!=[value]:wrong.append(key)
    status='failed' if wrong else 'unknown' if missing else 'passed'
    return dict(status=status,expected=expected,observed=observed,missing=missing,wrong=wrong,
        scope='effective values printed by the actual llama_context, not library defaults')


def expected_forwards(prompt_tokens,output_count):
    require(type(prompt_tokens) is int and prompt_tokens>0 and type(output_count) is int and output_count>0,'actual prompt/output inventory required')
    result=[]
    for position in range(0,prompt_tokens,256):
        tokens=min(256,prompt_tokens-position)
        result.append(dict(index=len(result),phase='prefill',position=position,tokens=tokens,
            output_rows=int(position+tokens==prompt_tokens)))
    for step in range(output_count-1):
        result.append(dict(index=len(result),phase='decode',position=prompt_tokens+step,tokens=1,output_rows=1))
    return result


def audit_native(trace,stderr,*,bare=None,prepared_profile_qs=(1,128,256)):
    """Pure structural audit; caller separately binds acquisition file receipts."""
    problems=[];cases=[];shapes=Counter();pruned=0
    def check(condition,message):
        if not condition:problems.append(message)
    check(trace.get('schema')=='real-moe-native-1' and trace.get('mode')=='trace','trace native schema/mode differs')
    req=trace.get('request',{})
    check(req.get('batch_size')==256,'request batch size is outside the reviewed 256/256 boundary')
    check(trace.get('observed_model_contract')==CONTRACT and req.get('model_contract')==CONTRACT,'OLMoE architecture/geometry contract differs')
    check(trace.get('threadpool_policy')=='explicit_persistent_owned','normal execution threadpool boundary differs')
    check(trace.get('model_weights_loaded') is True and trace.get('trace_callback_enabled') is True,'actual model trace evidence missing')
    check(trace.get('trace_callback_asks_all_nodes') is False,'trace callback did not preserve its routing-only ask filter')
    settings=effective_parameters(stderr)
    if settings['status']=='failed':problems.append('effective microbatch/sequence parameters differ from reviewed boundary')
    reference={r['id']:r for r in bare.get('results',[])} if bare is not None else {}
    if bare is not None:
        check(bare.get('mode') in ('generate','replay'),'normal reference mode differs')
        check(bare.get('model_sha256_verified_by_parent')==trace.get('model_sha256_verified_by_parent'),'paired model content differs')
        check(bare.get('source_sha256')==trace.get('source_sha256'),'paired executable source differs')
        check(bare.get('request',{}).get('placement')==req.get('placement'),'paired placement differs')
        check([r['id'] for r in bare.get('results',[])]==[r['id'] for r in trace.get('results',[])],'paired request ID order differs')
    source_rows={r['id']:r for r in req.get('requests',[])}
    check(len(source_rows)==len(req.get('requests',[])),'duplicate request IDs')
    check([r['id'] for r in trace.get('results',[])]==[r['id'] for r in req.get('requests',[])],'trace result/request inventory differs')
    for row in trace.get('results',[]):
        before=len(problems);identifier=row['id'];prefix=identifier+': '
        prompt=row['prompt_token_ids'];outputs=row['output_token_ids'];forwards=expected_forwards(len(prompt),len(outputs))
        source=source_rows[identifier]
        check(prompt==source.get('prompt_token_ids'),'prompt differs from frozen token request')
        check(outputs==source.get('output_token_ids'),'trace did not replay the frozen output token stream')
        check(row.get('teacher_forced') is True,prefix+'trace must be teacher-forced from an actual normal run')
        check(row.get('timing_usable_as_bare_reference') is False,prefix+'trace timing promoted to normal truth')
        check(row.get('all_greedy_tokens_match') is True,prefix+'teacher-forced stream differs from trace greedy trajectory')
        actual=row['forward_inventory']
        check(len(actual)==len(forwards),prefix+'forward count differs; possible internal split or extra final decode')
        for observed,expected in zip(actual,forwards):
            for key in ('index','phase','position','tokens','output_rows'):
                check(observed.get(key)==expected[key],prefix+f'forward {expected["index"]} {key} differs')
        check(row.get('forward_count')==len(forwards) and row.get('decode_forward_count')==len(outputs)-1,prefix+'forward/decode ledger differs')
        by_forward={}
        for route in row['routes']:
            index=route.get('forward_index');layer=route.get('layer')
            if type(index) is not int or not 0<=index<len(forwards) or type(layer) is not int or not 0<=layer<16:
                problems.append(prefix+'route has an unknown forward or layer');continue
            routes=by_forward.setdefault(index,{})
            if layer in routes:problems.append(prefix+f'duplicate actual route for forward {index}/layer {layer}; microbatches are not silently merged');continue
            routes[layer]=route
        summaries=[]
        for f in forwards:
            routes=by_forward.get(f['index'],{});counts=[]
            for layer in range(16):
                # OLMoE prunes final-layer FFN after Attention according to the
                # output mask. Earlier layers must retain every submitted row.
                n=f['tokens'] if layer<15 else f['output_rows']
                positions=list(range(f['position'],f['position']+f['tokens'])) if layer<15 else ([f['position']+f['tokens']-1] if n else [])
                route=routes.get(layer)
                if not n:
                    pruned+=1
                    check(route is None or (route.get('absolute_positions')==[] and route.get('expert_ids')==[]),prefix+f'forward {f["index"]}/layer15 fabricated pruned rows')
                    counts.append(0);continue
                shapes[n]+=1;counts.append(n)
                if route is None:problems.append(prefix+f'missing forward {f["index"]}/layer{layer} required {n} rows');continue
                check(route.get('phase')==f['phase'],prefix+'route phase differs')
                check(route.get('absolute_positions')==positions,prefix+f'forward {f["index"]}/layer{layer} absolute positions differ')
                ids=route.get('expert_ids',[])
                check(len(ids)==n,prefix+f'forward {f["index"]}/layer{layer} row count differs')
                check(all(isinstance(v,list) and len(v)==8 and len(set(v))==8 and all(type(x) is int and 0<=x<64 for x in v) for v in ids),prefix+'expert IDs/top-k invalid')
            summaries.append(dict(**f,required_route_rows_by_layer=counts))
        if bare is not None:
            normal=reference.get(identifier)
            check(normal is not None,prefix+'normal timing reference missing')
            if normal is not None:
                check(normal['prompt_token_ids']==prompt and normal['output_token_ids']==outputs,prefix+'normal/trace token streams differ')
                check(normal.get('timing_usable_as_bare_reference') is True and not normal.get('routes'),prefix+'reference is not a normal bare run')
                normal_forwards=[{k:x.get(k) for k in ('index','phase','position','tokens','output_rows')} for x in normal['forward_inventory']]
                check(normal_forwards==forwards,prefix+'normal/trace physical forward inventory differs')
        cases.append(dict(id=identifier,status='passed' if len(problems)==before else 'failed',prompt_tokens=len(prompt),output_tokens=len(outputs),forwards=summaries))
    route_status='failed' if problems else 'unknown' if settings['status']=='unknown' else 'passed'
    return dict(schema='real-moe-postcollection-shape-audit-1',routing_shape_status=route_status,
        effective_parameters=settings,problems=problems,cases=cases,
        required_nonempty_ffn_q_invocations={str(k):v for k,v in sorted(shapes.items())},pruned_final_layer_ffn_invocations=pruned,
        prepared_profile_qs=list(prepared_profile_qs),q_shapes_without_prepared_profile=sorted(set(shapes)-set(prepared_profile_qs)),
        q256_tail_q1_are_distinct=True,linear_extrapolation_used=False,
        exact_timing_coverage='unknown',timing_coverage_reason='prepared shape alone does not bind actual weights/routes/activations/cache/backend cuts and runtime transitions',
        normal_token_stream_pair_checked=bare is not None,hidden_bare_routes_proven_identical=False,
        trace_timing_is_performance_truth=False,whole_model_accuracy_certified=False,
        route_visibility='modeling workload evidence; this audit grants Mapping no access to future experts')


def audit_files(trace_path,out,bare_path=None):
    trace_path=Path(trace_path);trace=read(trace_path);stderr_path=trace_path.parent/'stderr.txt'
    result=audit_native(trace,stderr_path.read_text(encoding='utf-8',errors='replace') if stderr_path.exists() else '',bare=read(bare_path) if bare_path else None)
    evidence=[];integrity=[]
    for path in (trace_path,Path(bare_path) if bare_path else None):
        if path is None:continue
        execution=path.parent/'execution.json';item=dict(native=receipt(path),execution=None,status='unknown')
        if execution.exists():
            record=read(execution);item['execution']=receipt(execution)
            item['status']='passed' if record.get('status')=='completed' and record.get('native')==receipt(path) and record.get('process_id')==read(path).get('process_id') else 'failed'
        evidence.append(item);integrity.append(item['status'])
    result['acquisition_evidence']=evidence
    result['acquisition_integrity_status']='failed' if 'failed' in integrity else 'unknown' if 'unknown' in integrity else 'passed'
    statuses=(result['routing_shape_status'],result['acquisition_integrity_status'])
    result['status']='failed' if 'failed' in statuses else 'unknown' if 'unknown' in statuses else 'passed'
    if stderr_path.exists():result['stderr']=receipt(stderr_path)
    result['audit_tool']=receipt(__file__);write(out,result);return result


def prepare_pair(collection,bare_path,out):
    collection,out=Path(collection).resolve(),Path(out).resolve();bare=read(bare_path);manifest=read(collection/'manifest.json')
    acquisition_path=Path(bare_path).resolve().parent/'execution.json';acquisition=read(acquisition_path)
    require(acquisition.get('status')=='completed' and acquisition.get('native')==receipt(bare_path)
        and acquisition.get('process_id')==bare.get('process_id'),'completed normal collection receipt required')
    require(bare['mode']=='generate' and bare['request']['mode']=='generate','pair preparation requires actual normal generate output')
    require(bare['model_sha256_verified_by_parent']==manifest['model']['sha256'] and bare['source_sha256']==manifest['native_source_sha256'],'normal generation is not from this frozen collection')
    require(all(r['timing_usable_as_bare_reference'] and not r['routes'] for r in bare['results']),'routing trace cannot supply the normal reference')
    require(bare.get('model_weights_loaded') is True,'normal evidence did not execute actual model weights')
    require(not out.exists(),'new pair preparation directory required');out.mkdir(parents=True)
    commands=[];tag=bare['request']['placement'].replace('-','_')
    maximum=max(len(r['output_token_ids']) for r in bare['results'])
    for mode in ('replay','trace'):
        path=out/(mode+'-request.json');make_request(collection,bare_path,path,mode=mode,placement=bare['request']['placement'],
            max_output_tokens=maximum,threads=bare['request']['threads'],warmup_replays=bare['request']['warmup_replays'])
        job_id=tag+'-'+out.name+'-'+mode
        require(all(c.isalnum() or c in '-_' for c in job_id),'pair output directory must have a safe job-name basename')
        commands.append(dict(step=mode,job_id=job_id,request=receipt(path),
            argv=[sys.executable,'-m','profiling.real_moe_validation.run','collect','--out',str(collection),'--job-id',job_id,'--request',str(path),'--timeout','1800'],
            hardware_window_required=True))
    trace_result=collection/'jobs'/commands[1]['job_id']/'result.json'
    commands.append(dict(step='shape-routing-audit',argv=[sys.executable,'-m','profiling.real_moe_validation.postcollection','audit','--trace',str(trace_result),'--bare',str(Path(bare_path).resolve()),'--out',str(out/'shape-routing-audit.json')],hardware_window_required=False))
    plan=dict(schema='real-moe-replay-trace-plan-1',bare_reference=receipt(bare_path),source_collection_manifest=receipt(collection/'manifest.json'),commands=commands,
        bare_collection_execution=receipt(acquisition_path),
        input_tokens_from_actual_generation=True,outputs_from_actual_generation=True,no_actual_jobs_started=True,
        fixed_token_replay_is_reported=True,normal_reference_is_original_generate=True,
        independent_accuracy_certification=False,execute_only_in_parent_serial_window=True)
    write(out/'PAIR_PLAN.json',plan);return plan


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('audit','prepare-pair'));p.add_argument('--trace');p.add_argument('--bare');p.add_argument('--collection');p.add_argument('--out',required=True);a=p.parse_args()
    if a.command=='audit':
        try:r=audit_files(a.trace,a.out,a.bare)
        except (KeyError,ValueError,TypeError,FileNotFoundError) as error:
            r=dict(schema='real-moe-postcollection-shape-audit-1',status='unknown' if isinstance(error,FileNotFoundError) else 'failed',error=repr(error),raw_modified=False,whole_model_accuracy_certified=False);write(a.out,r)
        print(json.dumps(dict(out=a.out,status=r['status'],exact_timing_coverage=r.get('exact_timing_coverage','unknown'))))
        if r['status']!='passed':raise SystemExit(1 if r['status']=='failed' else 2)
    else:
        r=prepare_pair(a.collection,a.bare,a.out);print(json.dumps(dict(out=a.out,prepared_steps=len(r['commands']),no_actual_jobs_started=True)))


if __name__=='__main__':main()
