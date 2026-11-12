"""Private v5 GPU observer. No frozen source/runtime mutation; no inference CLI."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import difflib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from .tap_v4 import sha, receipt, write, run_record, private_context_header

HERE=Path(__file__).resolve().parent
DECL='static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {'
DISPATCH='''    const ggml_tensor * ids  = dst->src[2];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);'''
GRAPH='''    if (use_cuda_graph && cuda_graph_update_required) {
        // Start CUDA graph capture'''
HOOK=r'''
// Private observation v5: no math/dispatch/graph-configuration changes.
#include "gpu_tap_api.h"
static sim_gpu_reserve sim_gpu_reserve_v5 = nullptr;
static sim_gpu_copy_status sim_gpu_status_v5 = nullptr;
static sim_gpu_graph_event sim_gpu_graph_v5 = nullptr;
static void * sim_gpu_user_v5 = nullptr;
extern "C" void simulator_gpu_tap_set_v5(sim_gpu_reserve reserve, sim_gpu_copy_status status, sim_gpu_graph_event graph, void * user) {
    sim_gpu_user_v5 = user; sim_gpu_status_v5 = status; sim_gpu_graph_v5 = graph; sim_gpu_reserve_v5 = reserve;
}
extern "C" int simulator_gpu_tap_host_alloc_v5(size_t bytes, void ** ptr) { return int(cudaHostAlloc(ptr, bytes, cudaHostAllocDefault)); }
extern "C" int simulator_gpu_tap_host_free_v5(void * ptr) { return int(cudaFreeHost(ptr)); }
'''
DISPATCH_NEW='''    const ggml_tensor * ids  = dst->src[2];

    if (sim_gpu_reserve_v5) {
        const size_t row_bytes = size_t(ids->ne[0])*sizeof(int32_t);
        void * host = sim_gpu_reserve_v5(dst, ids, (void *) ctx.stream(), row_bytes, sim_gpu_user_v5);
        if (host) {
            // Same stream as the real consumer; distinct pinned slot per invocation.
            const cudaError_t status = cudaMemcpy2DAsync(host, row_bytes, ids->data, ids->nb[1], row_bytes, size_t(ids->ne[1]), cudaMemcpyDeviceToHost, ctx.stream());
            if (sim_gpu_status_v5) sim_gpu_status_v5(int(status), sim_gpu_user_v5);
        }
    }

    GGML_ASSERT(src1->type == GGML_TYPE_F32);'''
GRAPH_NEW='''    if (sim_gpu_graph_v5) {
#ifdef USE_CUDA_GRAPH
        const int simulator_graph_compiled = 1;
#else
        const int simulator_graph_compiled = 0;
#endif
        sim_gpu_graph_v5(cgraph, simulator_graph_compiled, int(use_cuda_graph), int(cuda_graph_update_required), sim_gpu_user_v5);
    }

    if (use_cuda_graph && cuda_graph_update_required) {
        // Start CUDA graph capture'''

def patched_cuda(text):
    for anchor in (DECL,DISPATCH,GRAPH):
        if text.count(anchor)!=1:raise ValueError('pinned CUDA observer anchor differs')
    return text.replace(DECL,HOOK+'\n'+DECL,1).replace(DISPATCH,DISPATCH_NEW,1).replace(GRAPH,GRAPH_NEW,1)

def native_v5(text):
    replacements=[
      # The dense channel is a property of the CPU candidate ledger this stage
      # already imports, so both model shapes are exercised where the driver
      # runs its self-test: a mixture model sizes a real ledger, and a dense
      # model must record nothing rather than fail on a zero-sized one.
      ('require(hash_format(std::string(64,\'a\'))&&!hash_format("unknown"),"SHA parser failed");',
       'require(hash_format(std::string(64,\'a\'))&&!hash_format("unknown"),"SHA parser failed");\n'
       ' {Capture dense;dense.prepare(2,3,1,0,0);require(dense.output().is_array()&&dense.output().empty(),"a dense capture must record no expert windows");\n'
       '  Capture moe;moe.prepare(2,3,1,4,2);require(moe.top_k==2&&moe.experts==4,"mixture capture sizing failed");}'),
      ('// Revision v4: observe CPU-consumed expert IDs; no llama scheduler callback.','// Revision v5: additive same-stream GPU consumer observation, not bare timing.'),
      ('#include "route_context_inspect.hpp"','#include "route_context_inspect.hpp"\n#include "gpu_tap_pod.hpp"'),
      ('{"passed",true},{"model_loaded",false}', '{"passed",true},{"gpu_POD_contract_checks",gpu_tap_pod_contract()},{"GPU_DMA_tested",false},{"model_loaded",false}'),
      ('const json &request,Capture &capture,bool record)','const json &request,Capture &capture,GPUCapture &gpu,bool record)'),
      ('capture.enabled=record;const auto begin=tick();','gpu.prepare(static_cast<int>((tokens.size()+chunk-1)/chunk)+outputs,static_cast<int>(tokens.size())+outputs,capture.layers,capture.experts,capture.top_k,record);\n capture.enabled=record;const auto begin=tick();'),
      ('if(record){capture.forward=forward;capture.decode=false;capture.position=static_cast<int>(first);capture.submitted=n;}','if(record){capture.forward=forward;capture.decode=false;capture.position=static_cast<int>(first);capture.submitted=n;}\n  gpu.forward=forward;gpu.decode=false;gpu.position=static_cast<int>(first);gpu.submitted=n;'),
      ('if(record){capture.forward=forward;capture.decode=true;capture.position=static_cast<int>(tokens.size())+i;capture.submitted=1;}','if(record){capture.forward=forward;capture.decode=true;capture.position=static_cast<int>(tokens.size())+i;capture.submitted=1;}\n  gpu.forward=forward;gpu.decode=true;gpu.position=static_cast<int>(tokens.size())+i;gpu.submitted=1;'),
      ('const auto end=tick();capture.enabled=false;const auto perf=llama_perf_context(ctx);','const auto end=tick();capture.enabled=false;gpu.enabled=false;const auto perf=llama_perf_context(ctx);\n auto gpu_routes=record?gpu.output():json::array();'),
      ('{"sampled_greedy_token_ids",sampled_tokens}', '{"gpu_routes",gpu_routes},{"gpu_capture_error_code",gpu.error},{"gpu_capture_status",gpu.error?"unsupported":"observed"},{"cuda_graph_decisions",gpu.graph_output()},{"gpu_copy_inventory",gpu.copy_output()},{"gpu_observer_copy_count",gpu.copy_count},{"gpu_pinned_capacity_bytes",gpu.pinned_bytes},{"gpu_host_reservation_ns",elapsed(0,gpu.enqueue_ticks)},{"observer_extra_flush_ns",0},{"observer_completion_boundary","existing llama_synchronize only"},{"sampled_greedy_token_ids",sampled_tokens}'),
      ('{"timing_usable_as_bare_reference",!record&&!observe_logits&&!observe_graph}','{"timing_usable_as_bare_reference",false}'),
      ('real-moe-tap-request-4','real-moe-tap-request-5'),('real-moe-tap-native-4','real-moe-tap-native-5'),
      ('TapBinding tap_binding(capture,mode=="capture");','TapBinding tap_binding(capture,mode=="capture");GPUCapture gpu;GPUBinding gpu_binding(gpu,mode=="capture");'),
      ('execute(context.value,vocab,row,capture,false)','execute(context.value,vocab,row,capture,gpu,false)'),
      ('execute(context.value,vocab,row,capture,mode=="capture")','execute(context.value,vocab,row,capture,gpu,mode=="capture")'),
      ('measured["routes"]=mode=="capture"?capture.output():json::array();','measured["routes"]=mode=="capture"?capture.output():json::array();\n  for(auto &route:measured["gpu_routes"])measured["routes"].push_back(route);'),
      ('result["tap_hook_available"]=tap_binding.setter!=nullptr;','result["tap_hook_available"]=tap_binding.setter!=nullptr;result["gpu_tap_hook_available"]=gpu_binding.setter!=nullptr;result["gpu_graph_replay_route_policy"]="unsupported if any GPU expert partition uses CUDA graph; no cached route reuse";'),
      ('{"actual_graph_capture_use_status","not exported by backend"}','{"actual_graph_capture_use_status","per-partition actual decisions exported in results"}'),
      ('normal output mask; callback-free CPU-consumer tap; capture or logit/graph observer timings are diagnostic only','normal output mask; CPU tap plus same-stream GPU D2H; all v5 timings observer-inclusive diagnostic only'),
    ]
    for before,after in replacements:
        if text.count(before)!=1:raise ValueError('pinned native replacement differs: '+before[:100])
        text=text.replace(before,after,1)
    return text

def prepare(out,llama_root,v4_build):
    out=Path(out).resolve();lr=Path(llama_root).resolve();old=Path(v4_build).resolve()
    if out.exists():raise ValueError('new private v5 build root required')
    for d in ('source','objects','bin','lib'):(out/d).mkdir(parents=True)
    original=lr/'ggml/src/ggml-cuda/ggml-cuda.cu';text=original.read_text(encoding='utf-8');patched=patched_cuda(text)
    (out/'source/ggml-cuda.cu').write_text(patched,encoding='utf-8',newline='\n')
    (out/'gpu-consumer-tap.patch').write_text(''.join(difflib.unified_diff(text.splitlines(True),patched.splitlines(True),fromfile='original/ggml-cuda.cu',tofile='private/ggml-cuda.cu')),encoding='utf-8',newline='\n')
    for name in ('route_tap_api.h','route_tap_pod.hpp','route_context_inspect.hpp'):
        shutil.copy2(HERE/'tap_native'/name,out/'source'/name)
    (out/'source/private_llama_context.h').write_text(
        private_context_header((lr/'src/llama-context.h').read_text(encoding='utf-8')),encoding='utf-8',newline='\n')
    for name in ('gpu_tap_api.h','gpu_tap_pod.hpp'):shutil.copy2(HERE/'tap_native_v5'/name,out/'source'/name)
    (out/'source/real_moe_tap_probe.cpp').write_text(native_v5((HERE/'tap_native/real_moe_tap_probe.cpp').read_text(encoding='utf-8')),encoding='utf-8',newline='\n')
    objs=sorted((lr/'build/ggml/src/ggml-cuda/ggml-cuda.dir/Release').glob('*.obj'))
    original_obj=lr/'build/ggml/src/ggml-cuda/ggml-cuda.dir/Release/ggml-cuda.obj'
    if original_obj not in objs:raise ValueError('matching original CUDA objects unavailable')
    m=dict(schema='real-moe-gpu-consumer-build-5',prepared_at_utc=datetime.now(timezone.utc).isoformat(),llama_root=str(lr),v4_build=receipt(old/'BUILD_COMPLETE.json'),original_cuda_source=receipt(original),original_cuda_object=receipt(original_obj),reused_cuda_objects=[receipt(p) for p in objs if p!=original_obj],source_files=[receipt(p) for p in sorted((out/'source').iterdir())],native_source_sha256=sha(out/'source/real_moe_tap_probe.cpp'),cuda_graph_policy='runtime_default_enabled unchanged; actual GPU expert CUDA graph capture/replay unsupported',pinned_policy='bounded unique slots before consumer; existing synchronization; no layer sync',fusion_policy='no disable or fallback; unobserved fused consumers fail inventory',timing_usable_as_bare_reference=False,common_source_modified=False,CPU_rebuilt=False,llama_rebuilt=False,compiled=False,hardware_executed=False)
    write(out/'PREPARED.json',m);return m

def build(out,nvcc,cl,link,env):
    out=Path(out).resolve();m=json.loads((out/'PREPARED.json').read_text());lr=Path(m['llama_root']);old=Path(m['v4_build']['path']).parent;src=out/'source';cuda=Path(nvcc).parent.parent
    for r in m['source_files']+m['reused_cuda_objects']:
        if sha(r['path'])!=r['sha256']:raise ValueError('prepared source/object changed: '+r['path'])
    project=lr/'build/ggml/src/ggml-cuda/ggml-cuda.vcxproj';ns={'m':'http://schemas.microsoft.com/developer/msbuild/2003'}
    releases=[g for g in ET.parse(project).getroot().findall('m:ItemDefinitionGroup',ns) if "'Release|x64'" in g.get('Condition','')]
    if len(releases)!=1:raise ValueError('one original x64 Release CUDA contract required')
    cu=releases[0].find('m:CudaCompile',ns);options=cu.findtext('m:AdditionalOptions',namespaces=ns)
    arch=re.search(r'--generate-code=arch=compute_(\d+),code=\[compute_\1,sm_\1\]',options)
    if not arch or cu.findtext('m:Runtime',namespaces=ns)!='MD' or cu.findtext('m:Optimization',namespaces=ns)!='O2' or cu.findtext('m:FastMath',namespaces=ns)!='true':raise ValueError('original math/build switches differ')
    original_defs=[d for d in cu.findtext('m:Defines',namespaces=ns).split(';') if d and '%(' not in d]
    if 'GGML_CUDA_USE_GRAPHS' not in original_defs:raise ValueError('original CUDA graph compile flag missing')
    linked=(project.parent/'ggml-cuda.dir/Release/ggml-cuda.tlog/link.command.1.tlog').read_text(encoding='utf-16').splitlines()[1:]
    names=re.findall(r'"[^"\r\n]*[\\/]([^"\\/]+\.OBJ)"','\n'.join(linked),re.I)
    expected=[Path(r['path']).name.lower() for r in m['reused_cuda_objects']]+['ggml-cuda.obj']
    if sorted(n.lower() for n in names)!=sorted(expected):raise ValueError('original CUDA linked object inventory differs')
    includes=[src,lr/'ggml/src/ggml-cuda',lr/'ggml/src',lr/'ggml/include',cuda/'include']
    defs=original_defs+['SIMULATOR_GPU_TAP_BUILD']
    args=[nvcc,'-ccbin',str(Path(cl).parent),'-x','cu','-m64','-c','-cudart','static','-std=c++17','-extended-lambda','--use_fast_math',f'--generate-code=arch=compute_{arch[1]},code=[compute_{arch[1]},sm_{arch[1]}]','-Xcompiler=/EHsc,/Ob2,/MD,/Zc:preprocessor','-O2']+['-I'+str(p) for p in includes]+['-D'+d for d in defs]+['-o',str(out/'objects/ggml-cuda.obj'),str(src/'ggml-cuda.cu')]
    write(out/'COMPILE_EQUIVALENCE.json',dict(schema='real-moe-gpu-compile-equivalence-5',project=receipt(project),original_cuda_defines=original_defs,added_observer_define='SIMULATOR_GPU_TAP_BUILD',architecture=arch[1],runtime='MD',optimization='O2',fast_math=True,original_additional_options=options,graph_compile_flag_retained=True,original_link_object_inventory_matches=True,CUDA_graph_mode_changed=False,mathematical_flags_changed=False))
    start=time.monotonic();run_record(out,'CUDA_COMPILE',args,env)
    lib=lr/'build/ggml/src/Release';cudalib=cuda/'lib/x64'
    objs=[r['path'] for r in m['reused_cuda_objects']]+[str(out/'objects/ggml-cuda.obj')]
    deps=[str(lib/'ggml-base.lib')]+[str(cudalib/(x+'.lib')) for x in ('cudart','cublas','cuda','cublasLt','cudadevrt','cudart_static')]+[x+'.lib' for x in ('kernel32','user32','gdi32','winspool','shell32','ole32','oleaut32','uuid','comdlg32','advapi32')]
    rsp=out/'cuda-link.rsp';rsp.write_text('\n'.join('"'+x+'"' for x in objs+deps)+'\n/DLL /NOLOGO /INCREMENTAL:NO /MACHINE:X64 /OUT:"'+str(out/'bin/ggml-cuda.dll')+'" /IMPLIB:"'+str(out/'lib/ggml-cuda.lib')+'"\n',encoding='utf-8',newline='\n')
    run_record(out,'CUDA_LINK',[link,'@'+str(rsp)],env)
    oldcompile=json.loads((old/'NATIVE_COMPILE.json').read_text())['command'];args=[cl]+oldcompile[1:]
    args=[x.replace(str(old/'source'),str(src)).replace(str(old/'objects'),str(out/'objects')).replace(json.loads((old/'PREPARED.json').read_text())['native_source_sha256'],m['native_source_sha256']) for x in args]
    run_record(out,'NATIVE_COMPILE',args,env)
    oldlink=json.loads((old/'NATIVE_LINK.json').read_text())['command'];args=[link]+[x.replace(str(old/'bin'),str(out/'bin')).replace(str(old/'objects'),str(out/'objects')) for x in oldlink[1:]]
    run_record(out,'NATIVE_LINK',args,env)
    for name in ('ggml-base.dll','ggml-cpu.dll','ggml.dll','llama.dll'):shutil.copy2(old/'bin'/name,out/'bin'/name)
    result=dict(schema='real-moe-gpu-consumer-build-complete-5',prepared=receipt(out/'PREPARED.json'),binaries=[receipt(p) for p in sorted((out/'bin').iterdir())],execution_receipts=[receipt(out/(n+'.json')) for n in ('CUDA_COMPILE','CUDA_LINK','NATIVE_COMPILE','NATIVE_LINK')],elapsed_seconds=time.monotonic()-start,compiled=True,hardware_executed=False)
    write(out/'BUILD_COMPLETE.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','build'));p.add_argument('--out',required=True);p.add_argument('--llama-root');p.add_argument('--v4-build');p.add_argument('--nvcc');p.add_argument('--cl');p.add_argument('--link');a=p.parse_args()
    m=prepare(a.out,a.llama_root,a.v4_build) if a.command=='prepare' else build(a.out,a.nvcc,a.cl,a.link,dict(os.environ));print(json.dumps(m,ensure_ascii=False))
if __name__=='__main__':main()
