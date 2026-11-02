"""Packed-weight subgraph acquisition and strict export to model-owned prices.

No whole-model label is accepted as a subgraph price. This module measures and
estimates samples; modeling.performance.steady_action_costs owns composition.
Synthetic activations, homogeneous graph cuts and CUDA unknown workspace remain
explicit. None independently certifies a llama_decode end-to-end prediction.
"""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction
import json
import math
from pathlib import Path
import shutil
import subprocess

from llm_infer_model.performance.steady_action_costs import SteadyActionCost, SteadyActionKey, SteadyActionTable
from .dataset import digest,read,write
from .run import child_environment,file_sha,receipt,sha,require
from profiling.state_calibration import run as common_run

ROOT=Path(__file__).resolve().parents[2]
SCOPE='independent homogeneous completed packed-weight subgraph; excludes llama scheduler and inter-backend transitions'


def request(directory, *, operation, layer, tokens, backend, source_sha256, runtime_binding, input_file=None, cache_context='isolated_warm', tensor_name=None):
    require(operation in ('olmoe_cpu_ffn','regular_mul_mat'),'unknown exact subgraph')
    require(backend in ('CPU','CUDA0') and (operation!='olmoe_cpu_ffn' or backend=='CPU'),'heterogeneous FFN is unsupported')
    require(type(tokens) is int and 1<=tokens<=256 and type(layer) is int and 0<=layer<16,'subgraph geometry outside explicit envelope')
    require(cache_context in ('isolated_warm','foreign_cpu_touch_512MiB') and (cache_context=='isolated_warm' or backend=='CPU'),'unknown measured cache context')
    foreign=512*1024**2 if cache_context=='foreign_cpu_touch_512MiB' else 0
    tensors={t['name']:t for t in directory['tensors']};prefix=f'blk.{layer}.'
    names={'router':prefix+'ffn_gate_inp.weight','up':prefix+'ffn_up_exps.weight','gate':prefix+'ffn_gate_exps.weight','down':prefix+'ffn_down_exps.weight'} if operation=='olmoe_cpu_ffn' else {'weight':tensor_name or prefix+'attn_q.weight'}
    weights={role:tensors[name] for role,name in names.items()}
    if operation=='olmoe_cpu_ffn':
        require(weights['router']['dtype']=='F32' and all(weights[r]['dtype'] in ('Q4_K','Q6_K') for r in ('up','gate','down')),'actual mixed-quantized weights required')
    total=sum(t['bytes'] for t in weights.values());model=directory['model']
    r=dict(schema='real-moe-quantized-subgraph-request-1',source_sha256=source_sha256,operation=operation,backend=backend,
        model_path=model['path'],model_bytes=model['bytes'],model_sha256=model['sha256'],weights=weights,
        tokens=tokens,threads=16,warmup=5,warmup_min_ms=500,samples=15,top_k=8,experts=64,seed=8173,
        cuda_graph_policy='runtime_default_enabled',tf32_override='0',expected_process_mask=4294967295,
        minimum_free_host_bytes=total+512*1024**2+foreign,minimum_free_gpu_bytes=total+256*1024**2,
        cache_context=cache_context,foreign_working_set_bytes=foreign,
        cpu_threadpool_policy='explicit_persistent_owned' if backend=='CPU' else 'not_applicable',
        input_source='declared_synthetic_f32',input_f32_file=None,input_receipt=None,
        runtime_binding=runtime_binding,reference_scope=SCOPE,calibration_role='calibration',
        target_outcome_used=False,whole_model_label_used=False)
    if input_file is not None:
        path=Path(input_file).resolve();hidden=weights['router' if operation=='olmoe_cpu_ffn' else 'weight']['shape'][0]
        require(path.stat().st_size==hidden*tokens*4,'captured F32 input exact byte count differs')
        r.update(input_source='external_f32_binary',input_f32_file=str(path),input_receipt=receipt(path))
    return r


def prepare(out,collection,directory_path,binary):
    out=Path(out).resolve();require(not out.exists(),'new quantized profile root required')
    m=read(Path(collection)/'manifest.json');directory=read(directory_path)
    require(directory['model']==m['model'] and directory['tensor_payloads_materialized'] is False,'GGUF directory does not bind to selected model')
    binary=Path(binary).resolve();source=Path(__file__).parent/'quantized_native/quantized_probe.cpp'
    out.mkdir(parents=True);target=out/'executed_binary'/binary.name;target.parent.mkdir();shutil.copy2(binary,target)
    sources=[]
    for index,path in enumerate((Path(__file__),Path(__file__).parent/'inventory.py',source,source.parent/'CMakeLists.txt')):
        copy=out/'executed_sources'/f'{index:02d}-{path.name}';copy.parent.mkdir(exist_ok=True);shutil.copy2(path,copy);sources.append(dict(source=receipt(path),archived=receipt(copy)))
    profile=dict(schema='real-moe-quantized-profile-manifest-1',model=m['model'],model_contract=m['model_contract'],
        collection_manifest=receipt(Path(collection)/'manifest.json'),directory=receipt(directory_path),binary=receipt(target),
        native_source_sha256=sha(source),sources=sources,runtime=m['runtime'],runtime_dirs=m['runtime_dirs'],
        whole_model_accuracy_certified=False,measurement_completed=False)
    write(out/'manifest.json',profile)
    runtime=digest(dict(binary=profile['binary']['sha256'],dlls=[dict(name=Path(x['path']).name,sha256=x['sha256']) for x in profile['runtime']]))
    jobs=[]
    # Two actual mixed-dtype layouts, no invented F32 expert alternative.
    for layer in (0,2):
        for tokens in (1,128,256):
            req=request(directory,operation='olmoe_cpu_ffn',layer=layer,tokens=tokens,backend='CPU',source_sha256=profile['native_source_sha256'],runtime_binding=runtime)
            name=f'cpu-ffn-layer{layer}-q{tokens}';path=out/'requests'/(name+'.json');write(path,req);jobs.append(dict(id=name,request=receipt(path)))
        req=request(directory,operation='olmoe_cpu_ffn',layer=layer,tokens=1,backend='CPU',source_sha256=profile['native_source_sha256'],runtime_binding=runtime,cache_context='foreign_cpu_touch_512MiB')
        name=f'cpu-ffn-layer{layer}-q1-foreign512';path=out/'requests'/(name+'.json');write(path,req);jobs.append(dict(id=name,request=receipt(path)))
    for backend in ('CPU','CUDA0'):
        for tensor,tag,grid in (('blk.0.attn_q.weight','q4',(1,128,256)),('blk.0.attn_v.weight','q6',(1,128,256)),('output.weight','head-q6',(1,))):
            for tokens in grid:
                req=request(directory,operation='regular_mul_mat',layer=0,tokens=tokens,backend=backend,source_sha256=profile['native_source_sha256'],runtime_binding=runtime,tensor_name=tensor)
                name=f'{backend.lower()}-regular-{tag}-q{tokens}';path=out/'requests'/(name+'.json');write(path,req);jobs.append(dict(id=name,request=receipt(path)))
    write(out/'prepared-jobs.json',dict(jobs=jobs,source='real GGUF packed directory',measurement_started=False))
    return dict(out=str(out),prepared_profiles=len(jobs),no_hardware_measurements=True)


def validate_native(native,req,pid):
    require(native['schema']=='real-moe-quantized-subgraph-native-1' and native['request']==req,'subgraph request differs')
    require(native['source_sha256']==req['source_sha256'] and native['process_id']==pid,'subgraph process/source identity differs')
    require(native['observed_process_mask']==req['expected_process_mask'],'profile affinity differs')
    require(native['packed_weights_read_from_model'] and not native['weights_dequantized_or_substituted'],'packed model weights replaced')
    require(native['output_finite'] and not native['whole_model_accuracy_certified'],'unqualified whole-model accuracy claim')
    require(native['cache_context']==req['cache_context'] and native['foreign_working_set_bytes']==req['foreign_working_set_bytes'] and native['cache_conditioning_in_target_interval'] is False,'profile cache boundary differs')
    require(native['cpu_threadpool_policy']==req['cpu_threadpool_policy'],'CPU threadpool lifetime differs')
    require(len(native['samples_ns'])==req['samples'] and all(type(x) is int and x>0 for x in native['samples_ns']),'invalid completed sample inventory')
    require({k:v.upper() for k,v in native['actual_weight_types'].items()}=={k:v['dtype'] for k,v in req['weights'].items()},'actual packed dtypes differ')
    ops=Counter(n['op'] for n in native['graph_nodes'])
    if req['operation']=='olmoe_cpu_ffn':
        require(req['backend']=='CPU' and ops['MUL_MAT_ID']==3 and ops['MUL_MAT']==1 and ops['ADD']==7 and ops['GLU']==1,'vectorized OLMoE source topology differs')
        require(ops['SOFT_MAX']==1 and ops['ARGSORT']==1 and ops['GET_ROWS']==1,'original router topology differs')
        ids=native['observed_routes_flat'];require(isinstance(ids,list) and len(ids)==req['tokens']*req['top_k'],'observed routing geometry differs')
        for first in range(0,len(ids),req['top_k']):
            row=ids[first:first+req['top_k']];require(len(set(row))==req['top_k'] and all(type(x) is int and 0<=x<req['experts'] for x in row),'invalid observed expert selection')
    else:require(ops['MUL_MAT']==1 and len(native['graph_nodes'])==1,'single MUL_MAT topology differs')
    return dict(passed=True,static_source_topology_checked=True,numerical_equivalence_to_original_model=False,
        actual_runtime_partition_equivalence=False,whole_model_accuracy_certified=False)


def collect(out,job_id,request_path,timeout=600):
    out=Path(out).resolve();m=read(out/'manifest.json');req=read(request_path)
    require(req['source_sha256']==m['native_source_sha256'] and req['model_sha256']==m['model']['sha256'],'unbound packed-weight request')
    require(job_id and all(c.isalnum() or c in '-_' for c in job_id),'safe unique profile ID required')
    folder=out/'jobs'/job_id;require(not folder.exists(),'no profile retry or overwrite')
    for r in [m['binary'],*m['runtime']]:require(sha(r['path'])==r['sha256'],'profile binary/runtime changed')
    for r in m['sources']:require(sha(r['source']['path'])==r['source']['sha256'],'profile source changed')
    if req['input_receipt'] is not None:require(sha(req['input_receipt']['path'])==req['input_receipt']['sha256'],'captured activation changed')
    folder.mkdir(parents=True);write(folder/'request.json',req);command=[m['binary']['path'],str(folder/'request.json'),str(folder/'result.json')]
    record=dict(schema='real-moe-quantized-profile-execution-1',job_id=job_id,status='failed',command=command);proc=None;handles=[]
    with common_run.measurement_lock():
        try:
            require(file_sha(m['model']['path'])==m['model']['sha256'],'packed model content changed')
            proc,stdout,stderr,affinity=common_run.launch(command,child_environment(m['runtime_dirs'],req),folder/'stdout.txt',folder/'stderr.txt',4294967295)
            handles=[stdout,stderr];record.update(process_id=proc.pid,affinity=affinity)
            code=proc.wait(timeout=timeout);record.update(returncode=code,process_exited_verified=True,forced_termination=False)
            require(code==0,'packed-weight native profile failed; no fallback or replacement')
            result=read(folder/'result.json');record['verification']=validate_native(result,req,proc.pid);record.update(status='completed',native=receipt(folder/'result.json'))
        except BaseException as error:
            record['error']=repr(error)
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
                record['forced_termination']=True
            if proc is not None:record.update(returncode=proc.returncode,process_exited_verified=proc.poll() is not None)
        finally:
            for f in handles:f.close()
            write(folder/'execution.json',record)
    return record


def export_price(result_path,execution_path):
    native=read(result_path);execution=read(execution_path);req=native['request']
    require(execution['status']=='completed' and execution['native']==receipt(result_path),'completed independent acquisition evidence required')
    validate_native(native,req,execution['process_id'])
    require(req['calibration_role']=='calibration' and not req['target_outcome_used'] and not req['whole_model_label_used'],'target label cannot become a component price')
    means=Fraction(sum(native['samples_ns']),len(native['samples_ns']))
    variance=sum((Fraction(x)-means)**2 for x in native['samples_ns'])/len(native['samples_ns'])
    cv=math.sqrt(float(variance))/float(means)
    require(cv<=.10,'unstable isolated samples cannot become a qualified scoped price')
    leaf_roles={v['name']:k for k,v in req['weights'].items()}
    graph=[dict(op=n['op'],geometry=n['geometry'],op_params=n['op_params'],inputs=[dict(node=x['node'],geometry=x['geometry'],leaf_role=leaf_roles.get(x['name'])) for x in n['inputs']]) for n in native['graph_nodes']]
    semantic=dict(operation=req['operation'],backend=req['backend'],tokens=req['tokens'],
        weights={k:{x:w[x] for x in ('dtype','shape','bytes')} for k,w in req['weights'].items()},
        threads=req['threads'],runtime=req['runtime_binding'],graph_policy=req['cuda_graph_policy'],
        graph_fingerprint=digest(graph),input_source=req['input_source'],
        cache_context=req['cache_context'],foreign_working_set_bytes=req['foreign_working_set_bytes'],
        cpu_threadpool_policy=req['cpu_threadpool_policy'],
        observed_route_pattern_sha256=None if native['observed_routes_flat'] is None else digest(native['observed_routes_flat']))
    key=SteadyActionKey(digest(semantic),req['backend'],digest(dict(tokens=req['tokens'],weights=semantic['weights'])),SCOPE)
    cost=SteadyActionCost(key,means,
        (str(Path(result_path).resolve())+'#sha256='+sha(result_path),),len(native['samples_ns']),(execution['process_id'],))
    return dict(schema='real-moe-scoped-subgraph-price-1',semantic=semantic,table=SteadyActionTable((cost,)).to_dict(),
        modeling_composer='llm_infer_model.performance.steady_action_costs.compile_steady_actions',
        calibrated_scope=SCOPE,original_model_activation_measured=False,
        external_input_supplied=req['input_source']=='external_f32_binary',activation_capture_provenance_import_supported=False,
        isolated_sample_cv=cv,source_process_count=1,cross_process_stability_verified=False,
        full_runtime_mapping_supported=False,whole_model_prediction_supported=False,
        unsupported_reasons=['llama scheduler and transitions not priced','hybrid backend partition not matched',
            'Attention/KV/Embedding/LM head not jointly covered','captured activation or numerical graph-port equivalence may be missing'])


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','collect','export'));p.add_argument('--out',required=True);p.add_argument('--collection');p.add_argument('--directory');p.add_argument('--binary');p.add_argument('--job-id');p.add_argument('--request');p.add_argument('--result');p.add_argument('--execution');p.add_argument('--timeout',type=int,default=600);a=p.parse_args()
    if a.command=='prepare':r=prepare(a.out,a.collection,a.directory,a.binary)
    elif a.command=='collect':r=collect(a.out,a.job_id,a.request,a.timeout)
    else:r=export_price(a.result,a.execution);write(a.out,r);r=dict(out=a.out,whole_model_prediction_supported=False)
    print(json.dumps(r))


if __name__=='__main__':main()
