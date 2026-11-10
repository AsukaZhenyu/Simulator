"""Explicit preparation and serialized real-MoE process collection.

No profiling median is an end-to-end predictor. Collected data are actual model
workloads and references; model-side kernel coverage is separately required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from profiling import host
from profiling.dac_validation import e2a
from profiling.state_calibration import run as common_run

ROOT=Path(__file__).resolve().parents[2]
read,write,sha,receipt,require=e2a.read,e2a.write,e2a.sha,e2a.receipt,e2a.require


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def prepare(out, binary, model, runtime_dir, acquisition_proof):
    out,binary,model,runtime_dir=map(lambda p:Path(p).resolve(),(out,binary,model,runtime_dir))
    require(not out.exists(),'new collection root required')
    require(binary.is_file() and model.is_file() and runtime_dir.is_dir(),'explicit real-model assets missing')
    proof=read(acquisition_proof)
    require(proof['completed'] and proof['sha256_matches'] and Path(proof['model']).resolve()==model,'verified acquisition proof required')
    contract=proof['model_contract']
    require(set(contract)=={'architecture','layers','experts','top_k'} and isinstance(contract['architecture'],str)
        and 0<contract['top_k']<=contract['experts'] and contract['layers']>0,'explicit model contract invalid')
    out.mkdir(parents=True)
    with common_run.measurement_lock():
        binding=dict(path=str(model),sha256=file_sha(model),bytes=model.stat().st_size,mtime_ns=model.stat().st_mtime_ns)
    require(binding['sha256']==proof['sha256'] and binding['bytes']==proof['bytes'],'download proof and model differ')
    native=ROOT/'profiling/real_moe_validation/native/real_moe_probe.cpp'
    source_paths=[Path(__file__),native,native.parent/'CMakeLists.txt',ROOT/'profiling/real_moe_validation/dataset.py',ROOT/'profiling/real_moe_validation/acquire_model.py']
    proof_path=Path(acquisition_proof).resolve()
    source_paths.extend([proof_path,proof_path.parent/'acquisition-plan.json',proof_path.parent/'original-config.json',proof_path.parent/'original-tokenizer_config.json'])
    source_paths.extend(ROOT.parent/'llama.cpp-implement/release-b8705/llama.cpp-b8705'/p for p in
        ('include/llama.h','src/llama-model.h','src/llama-graph.h','ggml/include/ggml.h','ggml/include/ggml-backend.h'))
    sources=[]
    for i,path in enumerate(source_paths):
        target=out/'executed_sources'/f'{i:02d}-{path.name}';target.parent.mkdir(exist_ok=True)
        shutil.copy2(path,target);sources.append(dict(source=receipt(path),archived=receipt(target)))
    target=out/'executed_binary'/binary.name;target.parent.mkdir();shutil.copy2(binary,target)
    runtime=[receipt(p) for p in sorted(runtime_dir.glob('*.dll'))]
    require(any(Path(r['path']).name=='llama.dll' for r in runtime),'matching llama.dll required')
    m=dict(schema='real-moe-collection-manifest-3',execution_revision='olmoe-v3-persistent-pool',prepared_at_utc=e2a.utc(),model=binding,
           cpu_threadpool_policy='explicit_persistent_owned',
           model_contract=contract,acquisition_proof=receipt(acquisition_proof),
           official_tokenizer_config_sha256=sha(proof_path.parent/'original-tokenizer_config.json'),
           binary=receipt(target),sources=sources,native_source_sha256=sha(native),
           runtime=runtime,runtime_dirs=[str(runtime_dir)],jobs=[],
           scope='real-model corpus and normal runtime timing; no simulator accuracy certification',
           predictions_fitted=False,global_resource_settings_changed=False)
    write(out/'manifest.json',m);return dict(out=str(out),model=binding,binary=m['binary'])


def request(manifest, *, mode, placement, requests, threads=16, warmup_replays=1):
    require(mode in ('tokenize-only','generate','replay','trace'),'unknown native mode')
    require(placement in ('cpu','cpu-experts-gpu-other'),'unknown native placement')
    require(1<=threads<=32 and 0<=warmup_replays<=3,'explicit thread/warmup envelope invalid')
    return dict(schema='real-moe-request-1',source_sha256=manifest['native_source_sha256'],mode=mode,
        model_contract=manifest['model_contract'],
        tokenization_add_special=False,
        threadpool_policy='metadata_only_no_compute' if mode=='tokenize-only' else 'explicit_persistent_owned',
        model_path=manifest['model']['path'],model_sha256=manifest['model']['sha256'],model_bytes=manifest['model']['bytes'],
        placement=placement,threads=threads,context_size=2048,batch_size=256,
        expected_process_mask=4294967295,warmup_replays=warmup_replays,
        minimum_free_host_bytes=0 if mode=='tokenize-only' else manifest['model']['bytes']+2*1024**3,
        minimum_free_gpu_bytes=6*1024**3,cuda_graph_policy='runtime_default_enabled',tf32_override='0',requests=requests)


def child_environment(runtime_dirs,req):
    require(req['cuda_graph_policy']=='runtime_default_enabled' and req['tf32_override']=='0','unknown explicit native flags')
    env=dict(os.environ)
    directories=os.pathsep.join([*map(str,runtime_dirs),env.get('PATH','')])
    # The runtime directories go first on whichever variable this host's loader
    # reads. `PATH` alone starts a Linux child into a loader error naming a
    # library that is already in the list -- the same defect
    # `state_calibration.run.child_environment` names on its own line.
    env['PATH']=directories
    search=host.shared_library_variable()
    if search!='PATH':env[search]=directories
    env.pop('GGML_CUDA_DISABLE_GRAPHS',None)
    env['NVIDIA_TF32_OVERRIDE']='0'
    return env


def make_request(out, inputs, output, *, mode, placement, ids=None, max_output_tokens=64, threads=16, warmup_replays=1):
    manifest=read(Path(out)/'manifest.json');data=read(inputs)
    require(1<=max_output_tokens<=4096,'explicit output-token envelope invalid')
    if data.get('schema')=='real-moe-native-1' and data.get('mode')!='tokenize-only':
        require(data['model_sha256_verified_by_parent']==manifest['model']['sha256'],'input generated token stream belongs to a different model')
        original={r['id']:r for r in data['request']['requests']}
        rows=[]
        for measured in data['results']:
            source=original[measured['id']]
            rows.append(dict(source,prompt_token_ids=measured['prompt_token_ids'],
                output_token_ids=measured['output_token_ids'],max_output_tokens=max_output_tokens))
        require(mode in ('trace','replay'),'measured token stream is only replayed/traced')
    else:
        if 'model_sha256' in data:require(data['model_sha256']==manifest['model']['sha256'],'input suite belongs to a different model')
        elif 'model_sha256_verified_by_parent' in data:require(data['model_sha256_verified_by_parent']==manifest['model']['sha256'],'tokenizer output belongs to a different model')
        else:require(data.get('tokenizer_config_sha256')==manifest['official_tokenizer_config_sha256'],'candidate prompt template is not bound to this model')
        rows=data.get('requests',data.get('tokenized_requests'))
        require(isinstance(rows,list),'request inventory missing')
        rows=[dict(r,max_output_tokens=max_output_tokens) for r in rows]
    if ids is not None:
        wanted=ids.split(',');lookup={r['id']:r for r in rows}
        require(all(i in lookup for i in wanted) and len(wanted)==len(set(wanted)),'unknown/duplicate selected request ID')
        rows=[lookup[i] for i in wanted]
    require(rows,'empty selected request inventory')
    req=request(manifest,mode=mode,placement=placement,requests=rows,threads=threads,warmup_replays=warmup_replays)
    req['input_artifact']=receipt(inputs)
    req['selection_used_target_performance']=False
    write(output,req)
    return dict(request=receipt(output),mode=mode,placement=placement,requests=len(rows))


def verify(native, req, process_id):
    require(native['schema']=='real-moe-native-1' and native['request']==req,'native request differs')
    require(native['process_id']==process_id and native['source_sha256']==req['source_sha256'],'native identity differs')
    require(native['process_affinity']['observed_mask']==req['expected_process_mask']
        and native['process_affinity']['enforced_before_model_load'],'native process affinity differs')
    require(native['automatic_fit_used'] is False and native['all_prompt_tokens_marked_output'] is False
        and native['extra_final_decode'] is False,'normal execution-boundary guard failed')
    require(native['model_sha256_verified_by_parent']==req['model_sha256'],'model content binding differs')
    contract=req['model_contract']
    require(native['model_contract_verified'] is True and native['observed_model_contract']==contract,'model architecture/shape contract differs')
    require(native['tokenization_add_special'] is False,'rendered prompt gained implicit special tokens')
    require(native['threadpool_policy']==req['threadpool_policy'],'CPU threadpool lifetime differs')
    if req['mode']=='tokenize-only':
        require(native['model_weights_loaded'] is False,'vocab-only unexpectedly loaded weights')
        require(len(native['tokenized_requests'])==len(req['requests']),'tokenizer inventory differs')
        return dict(passed=True,tokenizer_only=True)
    require(len(native['results'])==len(req['requests']),'request result inventory differs')
    expected_ids=[r['id'] for r in req['requests']]
    require([r['id'] for r in native['results']]==expected_ids,'request ID order differs')
    hz=native['clock']['frequency_hz']
    for row,source in zip(native['results'],req['requests']):
        require(row['begin_tick']<row['prefill_end_tick']<=row['end_tick'],'timing order invalid')
        require(abs((row['end_tick']-row['begin_tick'])*1e9/hz-row['whole_request_wall_ns'])<=1,'request QPC ledger differs')
        require(row['output_count']==len(row['output_token_ids'])>=1,'actual output count differs')
        require(row['decode_forward_count']==len(row['core_decode_forward_ns'])==row['output_count']-1,'extra or missing final decode')
        require(row['timing_usable_as_bare_reference']==(req['mode']!='trace'),'trace promoted to bare truth')
        if 'prompt_token_ids' in source:require(row['prompt_token_ids']==source['prompt_token_ids'],'prompt tokens differ')
        if 'output_token_ids' in source:require(row['output_token_ids']==source['output_token_ids'],'frozen teacher-forced tokens differ')
        if req['mode']!='trace':require(not row['routes'],'bare mode collected routing')
        else:
            inventory={r['index']:r for r in row['forward_inventory']}
            by_forward={}
            for route in row['routes']:
                f=inventory[route['forward_index']]
                require(route['phase']==f['phase'],'route phase mismatch')
                require(len(route['absolute_positions'])==len(route['expert_ids']),'route position inventory differs')
                require(all(f['position']<=p<f['position']+f['tokens'] for p in route['absolute_positions']),'route position outside actual forward')
                require(all(len(ids)==len(set(ids)) and len(ids)==contract['top_k'] and all(0<=v<contract['experts'] for v in ids) for ids in route['expert_ids']),'expert ID/uniqueness mismatch')
                require(0<=route['layer']<contract['layers'],'route layer outside declared model')
                layers=by_forward.setdefault(route['forward_index'],{})
                require(route['layer'] not in layers,'duplicate layer routing in one forward')
                layers[route['layer']]=route
            require(native['model_layers']==contract['layers'],'real-MoE collection model-layer contract differs')
            for f in row['forward_inventory']:
                if f['phase']!='decode':continue
                routes=by_forward.get(f['index'],{})
                require(set(routes)==set(range(contract['layers'])),'decode route inventory missing model layers')
                require(all(r['absolute_positions']==[f['position']] for r in routes.values()),'decode route token mapping differs')
    return dict(passed=True,requests=len(req['requests']),normal_pruning=True,
        greedy_match_all=all(r['all_greedy_tokens_match'] for r in native['results']),
        routing_mode=req['mode']=='trace',
        matched_normal_generation=all(r['all_greedy_tokens_match'] for r in native['results']),
        routing_and_bare_timing_pair_qualified=False,simulator_accuracy_certified=False)


def collect_one(out, job_id, request_path, timeout=1800):
    out=Path(out).resolve();m=read(out/'manifest.json');req=read(request_path)
    require(req['model_sha256']==m['model']['sha256'] and req['source_sha256']==m['native_source_sha256'],'unbound request')
    require(job_id and all(c.isalnum() or c in '-_' for c in job_id),'safe unique job ID required')
    folder=out/'jobs'/job_id;require(not folder.exists(),'job already exists; no retry/overwrite')
    for r in [m['binary'],*m['runtime']]:require(sha(r['path'])==r['sha256'],'binary/runtime changed')
    for r in m['sources']:require(sha(r['source']['path'])==r['source']['sha256'],'bound source changed')
    folder.mkdir(parents=True);write(folder/'request.json',req)
    command=[m['binary']['path'],str(folder/'request.json'),str(folder/'result.json')]
    record=dict(schema='real-moe-native-execution-1',job_id=job_id,started_at_utc=e2a.utc(),status='failed',request=receipt(folder/'request.json'),command=command)
    proc=None;handles=[]
    with common_run.measurement_lock():
        try:
            require(file_sha(m['model']['path'])==m['model']['sha256'],'real model content changed')
            env=child_environment(m['runtime_dirs'],req);proc,stdout,stderr,affinity=common_run.launch(command,env,folder/'stdout.txt',folder/'stderr.txt',4294967295)
            handles=[stdout,stderr];record.update(process_id=proc.pid,process_affinity=affinity)
            code=proc.wait(timeout=timeout);record.update(returncode=code,process_exited_verified=True,forced_termination=False)
            require(code==0,'real model native failed; do not substitute or retry')
            result=read(folder/'result.json');record['verification']=verify(result,req,proc.pid)
            record.update(status='completed',native=receipt(folder/'result.json'))
        except BaseException as error:
            record['error']=repr(error)
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
                record['forced_termination']=True
            if proc is not None:record.update(returncode=proc.returncode,process_exited_verified=proc.poll() is not None)
        finally:
            for handle in handles:handle.close()
            record['completed_at_utc']=e2a.utc();write(folder/'execution.json',record)
    return record


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','make-request','collect'))
    p.add_argument('--out',required=True);p.add_argument('--binary');p.add_argument('--model');p.add_argument('--runtime-dir');p.add_argument('--job-id');p.add_argument('--request');p.add_argument('--timeout',type=int,default=1800)
    p.add_argument('--acquisition-proof')
    p.add_argument('--inputs');p.add_argument('--request-output');p.add_argument('--mode',choices=('tokenize-only','generate','replay','trace'))
    p.add_argument('--placement',choices=('cpu','cpu-experts-gpu-other'));p.add_argument('--ids');p.add_argument('--max-output-tokens',type=int,default=64);p.add_argument('--threads',type=int,default=16);p.add_argument('--warmup-replays',type=int,default=1)
    a=p.parse_args()
    if a.command=='prepare':result=prepare(a.out,a.binary,a.model,a.runtime_dir,a.acquisition_proof)
    elif a.command=='make-request':result=make_request(a.out,a.inputs,a.request_output,mode=a.mode,placement=a.placement,ids=a.ids,max_output_tokens=a.max_output_tokens,threads=a.threads,warmup_replays=a.warmup_replays)
    else:result=collect_one(a.out,a.job_id,a.request,a.timeout)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
