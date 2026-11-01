"""Frozen extraction/fixture preparation for MMID<8> shared-memory ordering.

No prepare/test action initializes CUDA. Builds and GPU/racecheck runs require
separate root windows. Original source/runtime/results are never modified.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET
from .dataset import read,write
from .tap_v4 import receipt,sha,run_record
from .mmid_reference import reference

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
TOKENS=(1,3,31,32,33,125,256)
PATTERNS=('cyclic','sparse','skewed')
LAYOUTS=(('up_strided',64,1,1),('down_compact',8,8,8))
ANCHOR='    nex_prev = warp_reduce_sum<warp_size>(nex_prev);\n\n'
FENCE='    __syncwarp(0xffffffff);\n'

def function_text(text,marker):
    if text.count(marker)!=1:raise ValueError('pinned dependency/kernel marker differs: '+marker)
    start=text.index(marker);brace=text.index('{',start);depth=0
    # Reviewed fragments contain no braces in comments/string literals.
    for i in range(brace,len(text)):
        if text[i]=='{':depth+=1
        elif text[i]=='}':
            depth-=1
            if depth==0:return text[start:i+1]+'\n'
    raise ValueError('pinned function closing brace missing')

def extract(mmid,common):
    marker='template <int n_expert_used_template>\n__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)\nstatic __global__ void mm_ids_helper('
    kernel=function_text(mmid,marker)
    if kernel.count(ANCHOR)!=1 or kernel.count('static __global__ void mm_ids_helper(')!=1:raise ValueError('pinned synchronization site differs')
    struct_start='// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.'
    if mmid.count(struct_start)!=1:raise ValueError('pinned packed-store declaration differs')
    packed=mmid[mmid.index(struct_start):mmid.index(marker)]
    definitions=[]
    for macro,value in (('WARP_SIZE','32'),('GGML_CUDA_CC_AMPERE','800')):
        matches=re.findall(r'^#define '+macro+r'\s+'+value+r'\s*$',common,re.M)
        if len(matches)!=1:raise ValueError('pinned CUDA dependency macro differs: '+macro)
        definitions.append(matches[0].rstrip())
    helpers=[function_text(common,m) for m in ('static constexpr __device__ int ggml_cuda_get_physical_warp_size()', 'template<int width = WARP_SIZE>\nstatic __device__ __forceinline__ int warp_reduce_sum(int x)', 'template<int width = WARP_SIZE>\nstatic __device__ __forceinline__ int warp_reduce_any(int x)')]
    original=kernel.replace('void mm_ids_helper(','void mm_ids_helper_original(',1)
    fixed=kernel.replace(ANCHOR,ANCHOR.rstrip('\n')+'\n'+FENCE+'\n',1).replace('void mm_ids_helper(','void mm_ids_helper_syncwarp(',1)
    header='#pragma once\n#include <cuda_runtime.h>\n#include <cstdint>\n#include <climits>\n'+ '\n'.join(definitions)+'\n\n'+'\n'.join(helpers)+'\n'+packed+original+'\n'+fixed
    manifest=dict(schema='real-moe-mmid-extraction-1',normalized_original_kernel_sha256=hashlib.sha256(kernel.encode()).hexdigest(),normalized_packed_store_sha256=hashlib.sha256(packed.encode()).hexdigest(),normalized_helper_sha256=[hashlib.sha256(h.encode()).hexdigest() for h in helpers],copied_macros=definitions,normalization='UTF-8 newline-normalized source fragments; whole original files separately SHA-bound',original_variant_changes=['kernel symbol renamed only'],fixed_variant_changes=['kernel symbol renamed','one fullmask __syncwarp immediately after register reduction and before shared reads'],algorithms_or_integer_inputs_changed=False,GGML_host_runtime_linked=False,architecture_scope='NVIDIA CUDA arch89, warp32, TOP_K8 only; no HIP/generic routing certification')
    return header,manifest,original,fixed

def make_case(n,pattern,layout,index):
    name,si1,sis1,channels=layout;storage=[-424242]*(n*si1);sparse=(0,3,11,17,29,47,53,63)
    for t in range(n):
        if pattern=='cyclic':ids=[(t*3+k*5)%64 for k in range(8)]
        elif pattern=='sparse':ids=[sparse[(t+k)%8] for k in range(8)]
        elif pattern=='skewed':
            base=[0]+[1+(t*13+k*7)%63 for k in range(7)];ids=[base[(k*5+t)%8] for k in range(8)]
        else:raise ValueError('unknown deterministic route pattern')
        if len(set(ids))!=8:raise ValueError('fixture top8 not unique')
        storage[t*si1:t*si1+8]=ids
    oracle=reference(storage,n_experts=64,n_tokens=n,si1=si1,sis1=sis1,nchannels_y=channels)
    return dict(id=f'q{n:03d}-{pattern}-{name}',case_index=index,n_tokens=n,n_experts=64,top_k=8,si1=si1,sis1=sis1,nchannels_y=channels,pattern=pattern,ids_storage=storage,reference={k:oracle[k] for k in ('ids_src1','ids_dst','expert_bounds')})

def make_suite():
    cases=[]
    for n in TOKENS:
        for pattern in PATTERNS:
            for layout in LAYOUTS:cases.append(make_case(n,pattern,layout,len(cases)))
    return dict(schema='real-moe-mmid-primitive-suite-1',top_k=8,launch_repeats=3,variants=['original','syncwarp'],cases=cases,case_count=42,planned_launches=252,CUDA_graph_capture_replay=False,model_loaded=False,performance_timing_evaluated=False)

def prepare(out,llama_root,cuda_root,cl):
    out=Path(out).resolve();llama_root=Path(llama_root).resolve();cuda_root=Path(cuda_root).resolve();cl=Path(cl).resolve()
    if out.exists():raise ValueError('new MMID primitive directory required')
    source=llama_root/'ggml/src/ggml-cuda/mmid.cu';common=llama_root/'ggml/src/ggml-cuda/common.cuh';header,extraction,original,fixed=extract(source.read_text(encoding='utf-8'),common.read_text(encoding='utf-8'))
    (out/'source').mkdir(parents=True);(out/'bin').mkdir()
    (out/'source/mmid_extracted.cuh').write_text(header,encoding='utf-8',newline='\n');shutil.copy2(HERE/'mmid_native/mmid_probe.cu',out/'source/mmid_probe.cu')
    write(out/'EXTRACTION.json',dict(extraction,original_mmid_source=receipt(source),original_common_source=receipt(common)))
    (out/'fence-only-kernel.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),fixed.splitlines(True),fromfile='original symbol',tofile='syncwarp symbol')),encoding='utf-8',newline='\n')
    suite=make_suite();write(out/'suite.json',suite)
    case=next(c for c in suite['cases'] if c['id']=='q125-sparse-up_strided')
    for variant in ('original','syncwarp'):write(out/f'racecheck-{variant}.json',dict(suite,launch_repeats=1,variants=[variant],cases=[case],case_count=1,planned_launches=1))
    nvcc=cuda_root/'bin/nvcc.exe';sanitizer=cuda_root/'compute-sanitizer/compute-sanitizer.exe'
    if not nvcc.is_file() or not cl.is_file():raise ValueError('explicit compiler path unavailable')
    project=llama_root/'build/ggml/src/ggml-cuda/ggml-cuda.vcxproj';ns={'m':'http://schemas.microsoft.com/developer/msbuild/2003'}
    releases=[g for g in ET.parse(project).getroot().findall('m:ItemDefinitionGroup',ns) if "'Release|x64'" in g.get('Condition','')]
    if len(releases)!=1:raise ValueError('one pinned CUDA Release compilation contract required')
    cu=releases[0].find('m:CudaCompile',ns);options=cu.findtext('m:AdditionalOptions',namespaces=ns)
    if '--generate-code=arch=compute_89,code=[compute_89,sm_89]' not in options or cu.findtext('m:Optimization',namespaces=ns)!='O2' or cu.findtext('m:Runtime',namespaces=ns)!='MD' or cu.findtext('m:FastMath',namespaces=ns)!='true' or cu.findtext('m:CudaRuntime',namespaces=ns)!='Static':raise ValueError('pinned arch89/O2/MD/fastmath/static cudart contract differs')
    definitions=[d for d in cu.findtext('m:Defines',namespaces=ns).split(';') if d and '%(' not in d]
    command=[str(nvcc),str(out/'source/mmid_probe.cu'),'-o',str(out/'bin/mmid-probe.exe'),'-ccbin',str(cl.parent),'-m64','-O2','--use_fast_math','-std=c++17','-extended-lambda','--generate-code=arch=compute_89,code=[compute_89,sm_89]','--cudart=static','-lineinfo','-Xcompiler=/EHsc,/Ob2,/MD,/Zc:preprocessor','-I'+str(out/'source'),'-I'+str(llama_root/'vendor')]+['-D'+d for d in definitions]
    write(out/'COMPILE_CONTRACT.json',dict(schema='real-moe-mmid-compile-contract-1',original_project=receipt(project),original_CUDA_additional_options=options,original_definitions=definitions,architecture='89',optimization='O2',host_runtime='MD',CUDA_runtime='Static',fast_math=True,standalone_extra_diagnostic_flag='-lineinfo',GPU_algorithm_or_helper_transform_other_than_fence=False,GGML_runtime_linked=False))
    exe=str(out/'bin/mmid-probe.exe');plans=[dict(job_id='contract-both',kind='native',command=[exe,'--run',str(out/'suite.json'),str(out/'contract-result.json')],expected_launches=252)]
    for variant in ('original','syncwarp'):plans.append(dict(job_id=f'racecheck-{variant}',kind='racecheck',command=[str(sanitizer),'--tool','racecheck','--error-exitcode','97','--log-file',str(out/f'racecheck-{variant}.log'),exe,'--run',str(out/f'racecheck-{variant}.json'),str(out/f'racecheck-{variant}-result.json')],expected_launches=1,tool_exit_code_alone_proves_no_hazard=False))
    m=dict(schema='real-moe-mmid-primitive-prepared-1',prepared_at_utc=datetime.now(timezone.utc).isoformat(),original_sources=[receipt(source),receipt(common),receipt(llama_root/'vendor/nlohmann/json.hpp')],extraction=receipt(out/'EXTRACTION.json'),sources=[receipt(p) for p in sorted((out/'source').iterdir())],suite=receipt(out/'suite.json'),racecheck_suites=[receipt(out/f'racecheck-{v}.json') for v in ('original','syncwarp')],nvcc=receipt(nvcc),cl=receipt(cl),compute_sanitizer=dict(available=sanitizer.is_file(),file=receipt(sanitizer) if sanitizer.is_file() else None,version_process_executed=False,help_process_executed=False),planned_build_command=command,planned_GPU_jobs=plans,planned_host_validation=[exe,'--validate-only',str(out/'suite.json'),str(out/'suite-validated.json')],tooling_python_sources=[receipt(p) for p in (Path(__file__),HERE/'mmid_reference.py',HERE/'test_mmid_reference.py',HERE/'test_mmid_primitive.py')],owned_GPU_buffer_budget_bytes=1024*1024,additional_CUDA_driver_context_memory='not measured by software preparation; separate from owned buffers',host_free_reserve_bytes=512*1024*1024,device_free_reserve_bytes=64*1024*1024,maximum_dynamic_shared_bytes=1024,shared_memory_attribute_scope='logical <=1KiB dynamic shared allocation; no opt-in shared-memory limit API needed for these cases',compiled=False,GPU_executed=False,racecheck_executed=False,model_loaded=False,old_MoE_logit_root_cause_proven=False)
    m['compile_contract']=receipt(out/'COMPILE_CONTRACT.json')
    write(out/'PREPARED.json',m);return receipt(out/'PREPARED.json')

def build(out,env):
    out=Path(out).resolve();m=read(out/'PREPARED.json')
    for r in m['sources']+m['original_sources']+[m['nvcc'],m['cl']]:
        if sha(r['path'])!=r['sha256']:raise ValueError('prepared source/tool changed')
    run_record(out,'BUILD',m['planned_build_command'],env)
    result=dict(schema='real-moe-mmid-primitive-build-1',prepared=receipt(out/'PREPARED.json'),binary=receipt(out/'bin/mmid-probe.exe'),execution=receipt(out/'BUILD.json'),compiled=True,GPU_executed=False,racecheck_executed=False)
    write(out/'BUILD_COMPLETE.json',result);return result

def main():
    import os
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','build'));p.add_argument('--out',required=True);p.add_argument('--llama-root');p.add_argument('--cuda-root');p.add_argument('--cl');a=p.parse_args()
    r=prepare(a.out,a.llama_root,a.cuda_root,a.cl) if a.command=='prepare' else build(a.out,dict(os.environ));print(json.dumps(r if a.command=='prepare' else {'compiled':r['compiled'],'GPU_executed':False}))
if __name__=='__main__':main()
