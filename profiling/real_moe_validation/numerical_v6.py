"""Versioned numerical diagnostics: unchanged v5b kernels, host snapshots only.

Preparing/building/freezing is not permission to run a model. All six processes
are predeclared and contain four controlled requests; no failed sample is retried.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
from .tap_v4 import receipt,sha,write,run_record
from .dataset import read

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ENV_KEYS=('CUBLAS_WORKSPACE_CONFIG','CUDA_DEVICE_MAX_CONNECTIONS','CUDA_LAUNCH_BLOCKING','CUDA_MODULE_LOADING','CUDA_VISIBLE_DEVICES','NVIDIA_TF32_OVERRIDE','GGML_CUDA_DISABLE_GRAPHS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS')
ORDER=(('normal-SSLS','generate','SSLS'),('replay-LSSS','replay','LSSS'),('capture-SSLS','capture','SSLS'),('normal-LSSS','generate','LSSS'),('replay-SSLS','replay','SSLS'),('capture-LSSS','capture','LSSS'))

def native_v6(text):
    replacements=[
      ('// Revision v5: additive same-stream GPU consumer observation, not bare timing.','// Revision v6: preserve v5b kernels, persist existing host logits outside timing.'),
      ('#include "gpu_tap_pod.hpp"','#include "gpu_tap_pod.hpp"\n#include "host_snapshots.hpp"'),
      ('int main(int argc,char **argv){try{', 'int main(int argc,char **argv){try{\n if(argc==3&&std::string(argv[1])=="--host-export-self-test"){const float values[]={0.f,1.f,-2.f,3.5f,.25f,-0.f,100.f,1e-7f};auto artifact=save_host_logits(std::filesystem::absolute(std::filesystem::u8path(argv[2])),values,2,4);std::cout<<json{{"passed",true},{"model_loaded",false},{"GPU_exercised",false},{"artifact",artifact}}.dump()<<\'\\n\';return 0;}'),
      ('json graph_observations=json::array();int64_t logit_copy_ns=0,graph_observer_ns=0;','json graph_observations=json::array();int64_t logit_copy_ns=0,graph_observer_ns=0;\n ActualBatchLedger actual_batches(tokens.size()+outputs,(tokens.size()+chunk-1)/chunk+outputs);'),
      ('const auto a=tick();require(llama_decode(ctx,batch.value)==0,"prefill llama_decode failed");','actual_batches.record(batch.value,forward);\n  const auto a=tick();require(llama_decode(ctx,batch.value)==0,"prefill llama_decode failed");'),
      ('const auto a=tick();require(llama_decode(ctx,batch.value)==0,"decode llama_decode failed");','actual_batches.record(batch.value,forward);\n  const auto a=tick();require(llama_decode(ctx,batch.value)==0,"decode llama_decode failed");'),
      ('json forwards=json::array();for(const auto &s:forward_stamps)', 'require(observe_logits&&request.contains("logit_artifact_path"),"v6 requires explicit full host logit artifact");\n const auto logit_artifact=save_host_logits(std::filesystem::u8path(request.at("logit_artifact_path").get<std::string>()),logit_snapshots.data(),generated.size(),vocab_size);\n json forwards=json::array();for(const auto &s:forward_stamps)'),
      ('return json{{"id",request.at("id")}', 'return json{{"full_logits_artifact",logit_artifact},{"actual_batch_inputs",actual_batches.output()},{"origin_case_id",request.at("origin_case_id")},{"context_reset_index",request.at("context_reset_index")},{"memory_clear_data_requested",true},{"memory_clear_call_count_for_this_request",1},{"reset_completion_not_independently_instrumented",true},{"first_observable_numeric_boundary","output logits after normal prefill; inner-layer first divergence remains unknown"},{"id",request.at("id")}'),
      ('real-moe-tap-request-5','real-moe-numerical-request-6'),('real-moe-tap-native-5','real-moe-numerical-native-6'),
      ('require(request.at("tokenization_add_special")==false,','require(request.at("diagnostic_protocol")=="full-host-logits-cold-reset-1","diagnostic protocol differs");\n require(diagnostic_environment()==request.at("expected_runtime_environment"),"explicit runtime environment differs");\n require(request.at("tokenization_add_special")==false,'),
      ('{"observed_runtime_flags",json{{"GGML_CUDA_DISABLE_GRAPHS",nullptr}', '{"raw_runtime_environment",diagnostic_environment()},{"runtime_DLL_scope","CPU/CUDA/llama DLLs unchanged from frozen v5b; only native host export/input ledger differs"},{"observed_runtime_flags",json{{"GGML_CUDA_DISABLE_GRAPHS",nullptr}'),
      ('for(auto row:normalized){\n  if(mode=="replay")', 'for(auto row:normalized){\n  require(row.at("context_reset_index").get<size_t>()==results.size(),"explicit context reset index differs");\n  if(mode=="replay")'),
    ]
    for before,after in replacements:
        if text.count(before)!=1:raise ValueError('pinned v5b host diagnostic anchor differs: '+before[:110])
        text=text.replace(before,after,1)
    return text

def prepare_build(out,v5_build):
    out=Path(out).resolve();old=Path(v5_build).resolve()
    if out.exists():raise ValueError('new numerical driver build root required')
    (out/'source').mkdir(parents=True);(out/'objects').mkdir();(out/'bin').mkdir()
    for p in (old/'source').iterdir():
        if p.suffix in ('.h','.hpp'):shutil.copy2(p,out/'source'/p.name)
    shutil.copy2(HERE/'numerical_native_v6/host_snapshots.hpp',out/'source/host_snapshots.hpp')
    (out/'source/real_moe_tap_probe.cpp').write_text(native_v6((old/'source/real_moe_tap_probe.cpp').read_text(encoding='utf-8')),encoding='utf-8',newline='\n')
    original=read(old/'BUILD_COMPLETE.json');m=dict(schema='real-moe-numerical-driver-prepared-6',prepared_at_utc=datetime.now(timezone.utc).isoformat(),v5_build=receipt(old/'BUILD_COMPLETE.json'),original_native_source=receipt(old/'source/real_moe_tap_probe.cpp'),runtime=[r for r in original['binaries'] if Path(r['path']).suffix=='.dll'],runtime_dirs=[str(old/'bin'),'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin'],sources=[receipt(p) for p in sorted((out/'source').iterdir())],native_source_sha256=sha(out/'source/real_moe_tap_probe.cpp'),CPU_rebuilt=False,CUDA_rebuilt=False,kernels_or_runtime_flags_modified=False,existing_logit_host_copy_reused=True,additional_GPU_reads=0,extra_synchronizations=0,compiled=False,hardware_executed=False)
    write(out/'PREPARED.json',m);return m

def build(out,cl,link,env):
    out=Path(out).resolve();m=read(out/'PREPARED.json');old=Path(m['v5_build']['path']).parent
    for r in m['sources']+m['runtime']:
        if sha(r['path'])!=r['sha256']:raise ValueError('numerical driver source/runtime binding changed')
    source=m['native_source_sha256'];old_source=read(old/'PREPARED.json')['native_source_sha256']
    args=read(old/'NATIVE_COMPILE.json')['command'];args=[cl]+[x.replace(str(old/'source'),str(out/'source')).replace(str(old/'objects'),str(out/'objects')).replace(old_source,source) for x in args[1:]]
    run_record(out,'NATIVE_COMPILE',args,env)
    args=read(old/'NATIVE_LINK.json')['command'];args=[link]+[x.replace(str(old/'objects'),str(out/'objects')).replace(str(old/'bin'),str(out/'bin')).replace('real-moe-tap-probe.exe','real-moe-numerical-probe.exe') for x in args[1:]]
    run_record(out,'NATIVE_LINK',args,env)
    result=dict(schema='real-moe-numerical-driver-build-complete-6',prepared=receipt(out/'PREPARED.json'),binary=receipt(out/'bin/real-moe-numerical-probe.exe'),runtime=m['runtime'],runtime_dirs=m['runtime_dirs'],executions=[receipt(out/(x+'.json')) for x in ('NATIVE_COMPILE','NATIVE_LINK')],CPU_rebuilt=False,CUDA_rebuilt=False,compiled=True,hardware_executed=False)
    write(out/'BUILD_COMPLETE.json',result);return result

def prepare_cohort(out,build_dir,v5_smoke):
    out=Path(out).resolve();build_dir=Path(build_dir).resolve();v5_smoke=Path(v5_smoke).resolve()
    if out.exists():raise ValueError('new numerical diagnostic root required')
    b=read(build_dir/'BUILD_COMPLETE.json');prepared=read(build_dir/'PREPARED.json');old=read(v5_smoke/'manifest.json');normal=read(v5_smoke/'jobs/hybrid-private-normal/result.json')
    request=normal['request'];rows=normal['results'];short,long=sorted(rows,key=lambda r:len(r['prompt_token_ids']))
    sources={r['id']:r for r in request['requests']};shapes={'S':short,'L':long}
    environment={k:os.environ.get(k) for k in ENV_KEYS};environment.update(NVIDIA_TF32_OVERRIDE='0',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
    if environment['GGML_CUDA_DISABLE_GRAPHS'] is not None:raise ValueError('default-enabled graph environment must remain unset')
    out.mkdir(parents=True);jobs=[]
    for job,mode,sequence in ORDER:
        observations=[]
        for index,kind in enumerate(sequence):
            measured=shapes[kind];source=sources[measured['id']];observation_id=f'{job}-o{index:02d}-{kind}'
            row=dict(source,id=observation_id,comparison_observation_key=f'{sequence}-{index:02d}-{kind}',origin_case_id=source['id'],context_reset_index=index,max_output_tokens=8,prompt_token_ids=measured['prompt_token_ids'],logit_artifact_path=str(out/'logits'/job/(observation_id+'.f32')))
            if mode!='generate':row['output_token_ids']=measured['output_token_ids']
            else:row.pop('output_token_ids',None)
            observations.append(row)
        req=dict(request,schema='real-moe-numerical-request-6',source_sha256=prepared['native_source_sha256'],mode=mode,diagnostic_protocol='full-host-logits-cold-reset-1',expected_runtime_environment=environment,requests=observations)
        path=out/'requests'/(job+'.json');path.parent.mkdir(exist_ok=True);write(path,req);(out/'logits'/job).mkdir(parents=True)
        jobs.append(dict(job_id=job,mode=mode,sequence=sequence,request=receipt(path),expected_request_schema=req['schema'],expected_result_schema='real-moe-numerical-native-6',expected_source_sha256=prepared['native_source_sha256']))
    m=dict(schema='real-moe-numerical-campaign-6',prepared_at_utc=datetime.now(timezone.utc).isoformat(),driver_build=receipt(build_dir/'BUILD_COMPLETE.json'),binary=b['binary'],runtime=b['runtime'],runtime_dirs=b['runtime_dirs'],source_smoke_manifest=receipt(v5_smoke/'manifest.json'),source_greedy_trajectory=receipt(v5_smoke/'jobs/hybrid-private-normal/result.json'),old_failure_preserved=receipt(v5_smoke/'SMOKE_ANALYSIS.json'),model=old['model'],capacity_policy=old['capacity_policy'],native_guard_bytes=old['native_guard_bytes'],parent_preflight_bytes=old['parent_preflight_bytes'],resident_peak_limit_bytes=old['resident_peak_limit_bytes'],fixed_order=[x[0] for x in ORDER],jobs=jobs,raw_expected_environment=environment,workload=dict(processes=6,requests_per_process=4,observations=24,max_output_tokens=8,sequences=['SSLS','LSSS'],normal_forced=False,replay_capture_tokens='frozen prior actual same greedy streams, not future observations',observation_ids_globally_unique=True,comparison_by='comparison_observation_key and origin_case_id with exact actual inputs; never only renamed observation ID',origin_case_ids_preserved=True),scope='numerical diagnostic, not performance calibration; complete host F32 vectors exported after end tick; first inner-layer divergence remains unobserved',hypotheses=['first context initialization/input/quantization state','same-context reset and graph-cache/allocator history','shape/order sensitivity','GPU algorithm/workspace state: observed environment only, not silently changed'],whole_model_accuracy_evaluated=False,thresholds_relaxed=False,CPU_DLL_changed=False,CUDA_DLL_changed=False,sources=[])
    for path in (Path(__file__),HERE/'numerical_collect.py',HERE/'numerical_vectors.py',HERE/'test_numerical_v6.py',HERE/'test_numerical_collect.py',HERE/'test_numerical_vectors.py',HERE/'tap_v4.py',HERE/'tap_campaign_v5.py',HERE/'postcollection.py',HERE/'run.py',HERE/'dataset.py',ROOT/'profiling/state_calibration/run.py'):
        target=out/'executed_sources'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);m['sources'].append(dict(source=receipt(path),archived=receipt(target)))
    write(out/'manifest.json',m);return receipt(out/'manifest.json')

def freeze(out,tests):
    out=Path(out).resolve();t=read(tests)
    if t['returncode'] or t.get('hardware_executed') is not False:raise ValueError('passing software-only tests required')
    write(out/'FREEZE_BARRIER.json',dict(schema='real-moe-numerical-freeze-6',frozen_at_utc=datetime.now(timezone.utc).isoformat(),manifest=receipt(out/'manifest.json'),software_tests=receipt(tests),new_native_and_unchanged_v5b_DLLs=True,collection_authorized=False,root_review_required=True,prior_failure_not_replaced=True))
    return receipt(out/'FREEZE_BARRIER.json')

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare-build','build','prepare-cohort','freeze'));p.add_argument('--out',required=True);p.add_argument('--v5-build');p.add_argument('--build-dir');p.add_argument('--v5-smoke');p.add_argument('--cl');p.add_argument('--link');p.add_argument('--tests');a=p.parse_args()
    if a.command=='prepare-build':r=prepare_build(a.out,a.v5_build)
    elif a.command=='build':r=build(a.out,a.cl,a.link,dict(os.environ))
    elif a.command=='prepare-cohort':r=prepare_cohort(a.out,a.build_dir,a.v5_smoke)
    else:r=freeze(a.out,a.tests)
    print(json.dumps(r if a.command in ('prepare-cohort','freeze') else {'schema':r['schema'],'hardware_executed':False},ensure_ascii=False))
if __name__=='__main__':main()
