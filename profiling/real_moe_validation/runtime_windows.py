"""Private exact-runtime observation build for real MoE performance profiles.

No original source, mathematical graph, CUDA object or original artifact changes.
CPU node-start stamps use the existing worker-zero dispatch and ordinary barriers.
GPU submissions are NOT marked as completion; natural host windows are retained.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from .dataset import read
from .tap_v4 import receipt,sha,write,run_record

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

BASE_HOOK='''
// Private read-only, no-sync runtime observer ABI. Registered outside execution.
typedef void (*simulator_runtime_window_callback)(int,ggml_backend_t,const ggml_cgraph*,const ggml_tensor*,uint64_t,void*);
static simulator_runtime_window_callback simulator_runtime_observer=nullptr;
static void *simulator_runtime_user=nullptr;
extern "C" __declspec(dllexport) void simulator_runtime_window_set(simulator_runtime_window_callback cb,void*user){simulator_runtime_user=user;simulator_runtime_observer=cb;}
'''
CPU_HOOK='''
typedef void (*simulator_runtime_cpu_window_callback)(int,void*,const void*,const struct ggml_tensor*,uint64_t,void*);
static simulator_runtime_cpu_window_callback simulator_runtime_cpu_observer=NULL;
static void *simulator_runtime_cpu_user=NULL;
__declspec(dllexport) void simulator_runtime_cpu_window_set(simulator_runtime_cpu_window_callback cb,void*user){simulator_runtime_cpu_user=user;simulator_runtime_cpu_observer=cb;}
'''

def replace_once(text,before,after):
    if text.count(before)!=1:raise ValueError('pinned observer anchor differs: '+before[:100])
    return text.replace(before,after,1)

def patched_base(text):
    text=replace_once(text,'// backend buffer type',BASE_HOOK+'\n// backend buffer type')
    text=replace_once(text,'    return backend->iface.graph_compute(backend, cgraph);',
        '    if(simulator_runtime_observer)simulator_runtime_observer(1,backend,cgraph,nullptr,0,simulator_runtime_user);\n'
        '    const auto simulator_result=backend->iface.graph_compute(backend, cgraph);\n'
        '    if(simulator_runtime_observer)simulator_runtime_observer(2,backend,cgraph,nullptr,0,simulator_runtime_user);\n'
        '    return simulator_result;')
    text=replace_once(text,'    backend->iface.synchronize(backend);',
        '    if(simulator_runtime_observer)simulator_runtime_observer(3,backend,nullptr,nullptr,0,simulator_runtime_user);\n'
        '    backend->iface.synchronize(backend);\n'
        '    if(simulator_runtime_observer)simulator_runtime_observer(4,backend,nullptr,nullptr,0,simulator_runtime_user);')
    for signature,kind,backend,tensor in (
        ('void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {',5,'backend','tensor'),
        ('void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {',6,'backend','tensor'),
        ('void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {',7,'nullptr','tensor'),
        ('void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {',8,'nullptr','tensor')):
        text=replace_once(text,signature,signature+'\n    if(simulator_runtime_observer)simulator_runtime_observer('+str(kind)+','+backend+',nullptr,'+tensor+',uint64_t(size),simulator_runtime_user);')
    return text

def patched_cpu(text):
    text=replace_once(text,'// ggml_compute_forward_mul_mat_id\n',CPU_HOOK+'\n// ggml_compute_forward_mul_mat_id\n')
    anchor='    // Observe the exact CPU-consumed IDs before ordinary/repack dispatch.'
    return replace_once(text,anchor,
        '    if(params->ith==0&&simulator_runtime_cpu_observer)simulator_runtime_cpu_observer(10,NULL,NULL,tensor,0,simulator_runtime_cpu_user);\n'+anchor)

def patched_native(text):
    text=replace_once(text,'#include "gpu_tap_pod.hpp"','#include "gpu_tap_pod.hpp"\n#include "runtime_windows.hpp"')
    text=replace_once(text,'capture.enabled=record;const auto begin=tick();','capture.enabled=record;runtime_windows->begin(request.value("runtime_window_observer",false));const auto begin=tick();runtime_windows->mark(20,begin);')
    text=replace_once(text,'  const auto a=tick();require(llama_decode(ctx,batch.value)==0,"prefill llama_decode failed");const auto b=tick();',
        '  runtime_windows->context(forward,n,int(first),false,int(first/chunk));\n'
        '  const auto a=tick();runtime_windows->mark(21,a);require(llama_decode(ctx,batch.value)==0,"prefill llama_decode failed");const auto b=tick();runtime_windows->mark(22,b);')
    text=replace_once(text,' llama_synchronize(ctx);prefill_end=tick();',' llama_synchronize(ctx);prefill_end=tick();runtime_windows->mark(23,prefill_end);')
    text=replace_once(text,' for(int i=0;i<outputs;++i){const auto sampled=llama_sampler_sample(sampler.value,ctx,-1);sampled_tokens.push_back(sampled);',
        ' for(int i=0;i<outputs;++i){runtime_windows->context(forward,1,int(tokens.size())+i,true,i);runtime_windows->mark(24);const auto sampled=llama_sampler_sample(sampler.value,ctx,-1);runtime_windows->mark(25);sampled_tokens.push_back(sampled);')
    text=replace_once(text,'  const auto a=tick();require(llama_decode(ctx,batch.value)==0,"decode llama_decode failed");llama_synchronize(ctx);const auto b=tick();',
        '  runtime_windows->context(forward,1,int(tokens.size())+i,true,i);\n'
        '  const auto a=tick();runtime_windows->mark(21,a);require(llama_decode(ctx,batch.value)==0,"decode llama_decode failed");llama_synchronize(ctx);const auto b=tick();runtime_windows->mark(22,b);')
    text=replace_once(text,' const auto end=tick();capture.enabled=false;gpu.enabled=false;',
        ' const auto end=tick();runtime_windows->mark(26,end);runtime_windows->finish();capture.enabled=false;gpu.enabled=false;')
    text=replace_once(text,' return json{{"id",request.at("id")}', ' return json{{"final_full_logits_sha256",tap_sha256(llama_get_logits_ith(ctx,-1),size_t(vocab_size)*sizeof(float))},{"runtime_windows",runtime_windows->output()},{"id",request.at("id")}')
    text=replace_once(text,' ThreadpoolOwner pool;', ' ThreadpoolOwner pool;') if text.count(' ThreadpoolOwner pool;')==1 else text
    text=replace_once(text,' TapBinding tap_binding(capture,mode=="capture");',' RuntimeWindows window_owner;runtime_windows=&window_owner;\n TapBinding tap_binding(capture,mode=="capture");')
    text=replace_once(text,'  row["observe_logits"]=request.value("observe_logits",false);row["observe_graph"]=request.value("observe_graph",false);',
        '  row["observe_logits"]=request.value("observe_logits",false);row["observe_graph"]=request.value("observe_graph",false);row["runtime_window_observer"]=request.at("runtime_window_observer");')
    text=text.replace('real-moe-tap-request-5','real-moe-runtime-window-request-1').replace('real-moe-tap-native-5','real-moe-runtime-window-native-1')
    # warmup uses the same declared request lifecycle but its samples are never prices.
    text=replace_once(text,'  for(int w=0;w<warmups;++w)execute(context.value,vocab,row,capture,gpu,false);',
        '  for(int w=0;w<warmups;++w){auto warm=row;warm["runtime_window_observer"]=false;execute(context.value,vocab,warm,capture,gpu,false);}')
    text=replace_once(text,'{"timing_usable_as_bare_reference",false}',
        '{"timing_usable_as_bare_reference",!record&&!observe_logits&&!observe_graph&&!request.value("runtime_window_observer",false)}')
    text=replace_once(text,'GPUBinding gpu_binding(gpu,mode=="capture");',
        'GPUBinding gpu_binding(gpu,mode=="capture");if(mode!="capture")gpu_binding.setter(nullptr,nullptr,nullptr,nullptr);')
    text=replace_once(text,'  std::cout<<json{{"passed",true},{"gpu_POD_contract_checks",gpu_tap_pod_contract()}',
        '  RuntimeWindows contract;contract.begin(true);contract.context(0,1,0,false,0);contract.mark(20);ggml_tensor fake{};std::strcpy(fake.name,"ffn_moe_up-0");RuntimeWindows::callback(10,nullptr,nullptr,&fake,0,&contract);contract.mark(26);contract.finish();require(contract.output().at("events").size()==3,"runtime-window POD observer contract failed");\n'
        '  std::cout<<json{{"runtime_window_POD_contract",true},{"observer_POD_capacity_bytes",contract.output().at("POD_capacity_bytes")},{"passed",true},{"gpu_POD_contract_checks",gpu_tap_pod_contract()}')
    return text

def prepare(out,llama_root,observer_build,fixed_build):
    out,llama_root,observer_build,fixed_build=map(lambda p:Path(p).resolve(),(out,llama_root,observer_build,fixed_build))
    if out.exists():raise ValueError('new runtime-window build directory required')
    cpu_build=Path(read(observer_build/'PREPARED.json')['v4_build']['path']).parent
    sources={
        'ggml-backend.cpp':patched_base((llama_root/'ggml/src/ggml-backend.cpp').read_text(encoding='utf-8')),
        'ggml-cpu.c':patched_cpu((cpu_build/'source/ggml-cpu.c').read_text(encoding='utf-8')),
        'real_moe_tap_probe.cpp':patched_native((observer_build/'source/real_moe_tap_probe.cpp').read_text(encoding='utf-8'))}
    for name in ('source','objects','bin','lib'):(out/name).mkdir(parents=True,exist_ok=True)
    for p in (observer_build/'source').iterdir():
        if p.suffix in ('.h','.hpp'):shutil.copy2(p,out/'source'/p.name)
    shutil.copy2(HERE/'runtime_windows_native/runtime_windows.hpp',out/'source/runtime_windows.hpp')
    for name,text in sources.items():(out/'source'/name).write_text(text,encoding='utf-8',newline='\n')
    m=dict(schema='real-moe-runtime-window-build-prepared-1',llama_root=str(llama_root),observer_build=str(observer_build),cpu_build=str(cpu_build),fixed_build=receipt(fixed_build/'BUILD_COMPLETE.json'),
        sources=[receipt(p) for p in sorted((out/'source').iterdir())],native_source_sha256=sha(out/'source/real_moe_tap_probe.cpp'),
        original_base_source=receipt(llama_root/'ggml/src/ggml-backend.cpp'),original_cpu_source=receipt(cpu_build/'source/ggml-cpu.c'),original_native_source=receipt(observer_build/'source/real_moe_tap_probe.cpp'),
        mathematical_graph_changed=False,extra_GPU_syncs=0,CPU_barriers_added=0,CUDA_objects_rebuilt=False,whole_model_price_supported=False,hardware_executed=False)
    write(out/'PREPARED.json',m);return m

def build(out,cl,link,env):
    out=Path(out).resolve();m=read(out/'PREPARED.json');lr=Path(m['llama_root']);old=Path(m['observer_build']);oldcpu=Path(m['cpu_build']);src=out/'source'
    for r in m['sources']:
        if sha(r['path'])!=r['sha256']:raise ValueError('prepared runtime-window source changed')
    # Use the exact original project command-line switches for its base object.
    command_file=lr/'build/ggml/src/ggml-base.dir/Release/ggml-base.tlog/CL.command.1.tlog'
    lines=command_file.read_text(encoding='utf-16').splitlines()
    original=None
    import shlex
    for i,line in enumerate(lines):
        if line.startswith('^') and 'GGML-BACKEND.CPP' in line.upper():original=lines[i+1];break
    if original is None:raise ValueError('original backend compile switches unavailable')
    # Windows command-line parser is supplied by the Python ctypes shell API.
    import ctypes
    argc=ctypes.c_int();parse=ctypes.windll.shell32.CommandLineToArgvW;parse.argtypes=[ctypes.c_wchar_p,ctypes.POINTER(ctypes.c_int)];parse.restype=ctypes.POINTER(ctypes.c_wchar_p)
    argv=parse('cl '+original,ctypes.byref(argc));args=[argv[i] for i in range(1,argc.value)];free=ctypes.windll.kernel32.LocalFree;free.argtypes=[ctypes.c_void_p];free.restype=ctypes.c_void_p;free(argv)
    # Historical tlogs used the checkout's former path; map only that prefix.
    historical_root=None
    for x in args:
        if x.lower().endswith('ggml-backend.cpp'):historical_root=str(Path(x).parents[2]);break
    if historical_root is None:raise ValueError('original backend source path missing')
    args=[x for x in args if not x.upper().startswith(('/FO','/FD')) and not x.lower().endswith('ggml-backend.cpp')]
    args=[x.replace(historical_root,str(lr)) for x in args]
    args+=[str(src/'ggml-backend.cpp'),'/Fo'+str(out/'objects/ggml-backend.cpp.obj')]
    run_record(out,'BASE_COMPILE',[cl,*args],env)
    old_link=(lr/'build/ggml/src/ggml-base.dir/Release/ggml-base.tlog/link.command.1.tlog').read_text(encoding='utf-16').splitlines()[1:]
    import re
    linked=' '.join(old_link);objs=list(dict.fromkeys(re.findall(r'"([^"\r\n]+\.obj)"',linked,re.I)))
    if not objs:raise ValueError('original base object inventory unavailable')
    resolved=[];replaced_backend=0
    for p in objs:
        p=p.replace(historical_root,str(lr));pp=Path(p);pp=pp if pp.is_absolute() else lr/'build/ggml/src'/pp
        replace_backend=pp.name.lower() in ('ggml-backend.obj','ggml-backend.cpp.obj')
        replaced_backend+=int(replace_backend)
        resolved.append(str(out/'objects/ggml-backend.cpp.obj') if replace_backend else str(pp.resolve()))
    if replaced_backend!=1:raise ValueError('must link exactly one new runtime observer backend object')
    rsp=out/'base-link.rsp';rsp.write_text('\n'.join('"'+p+'"' for p in resolved)+'\nkernel32.lib user32.lib advapi32.lib\n/DLL /NOLOGO /INCREMENTAL:NO /MACHINE:X64 /OUT:"'+str(out/'bin/ggml-base.dll')+'" /IMPLIB:"'+str(out/'lib/ggml-base.lib')+'"\n',encoding='utf-8',newline='\n')
    run_record(out,'BASE_LINK',[link,'@'+str(rsp)],env)
    # Retain the observer CPU object's original switches; only add no-sync stamps.
    args=read(oldcpu/'CPU_COMPILE.json')['command'];args=[cl]+[x.replace(str(oldcpu/'source'),str(src)).replace(str(oldcpu/'objects'),str(out/'objects')) for x in args[1:]]
    for x in args:
        if str(oldcpu) in x:raise ValueError('recorded CPU build path survived the rewrite: '+x)
    run_record(out,'CPU_COMPILE',args,env)
    text=(oldcpu/'cpu-link.rsp').read_text(encoding='utf-8').replace(str(oldcpu/'objects'),str(out/'objects')).replace(str(oldcpu/'bin'),str(out/'bin')).replace(str(oldcpu/'lib'),str(out/'lib'))
    if str(oldcpu) in text:raise ValueError('recorded CPU build path survived the link rewrite')
    (out/'cpu-link.rsp').write_text(text,encoding='utf-8',newline='\n');run_record(out,'CPU_LINK',[link,'@'+str(out/'cpu-link.rsp')],env)
    oldsha=read(old/'PREPARED.json')['native_source_sha256']
    # The recorded commands name the directory they were actually built in, which
    # is not necessarily PREPARED.observer_build: reusing a patched source tree
    # keeps its original request on disk. Rewriting against observer_build alone
    # would silently leave the old /Fo and /OUT paths in place and write the new
    # object and executable into the historical build directory.
    compile_recorded=read(old/'NATIVE_COMPILE.json')['command'][1:]
    link_recorded=read(old/'NATIVE_LINK.json')['command'][1:]
    named=next((x for x in compile_recorded if x.lower().endswith('.cpp')),None)
    if named is None:raise ValueError('recorded native compile names no source file')
    recorded=Path(named).parent.parent
    compile_swaps=((str(recorded/'source'),str(src)),(str(recorded/'objects'),str(out/'objects')),(oldsha,m['native_source_sha256']))
    link_swaps=((str(recorded/'objects'),str(out/'objects')),(str(recorded/'bin'),str(out/'bin')),('real-moe-tap-probe.exe','real-moe-runtime-window-probe.exe'))
    def rewrite(argv,swaps):
        result=[]
        for x in argv:
            for before,after in swaps:x=x.replace(before,after)
            result.append(x)
        return result
    args=[cl,*rewrite(compile_recorded,compile_swaps)]
    for x in args:
        if str(recorded) in x:raise ValueError('recorded observer path survived the compile rewrite: '+x)
    run_record(out,'NATIVE_COMPILE',args,env)
    args=[link,*rewrite(link_recorded,link_swaps)]
    for x in args:
        if str(recorded) in x:raise ValueError('recorded observer path survived the link rewrite: '+x)
    run_record(out,'NATIVE_LINK',args,env)
    fixed=Path(m['fixed_build']['path']).parent
    for name in ('ggml-cuda.dll','ggml.dll','llama.dll'):shutil.copy2(fixed/'bin'/name,out/'bin'/name)
    return finalize(out,link)

def finalize(out,link):
    out=Path(out).resolve()
    for step in ('BASE_COMPILE','BASE_LINK','CPU_COMPILE','CPU_LINK','NATIVE_COMPILE','NATIVE_LINK'):
        if read(out/(step+'.json'))['returncode']!=0:raise ValueError('native build step failed: '+step)
    dumper=str(Path(link).with_name('dumpbin.exe'))
    for filename,symbol in (('ggml-base.dll','simulator_runtime_window_set'),('ggml-cpu.dll','simulator_runtime_cpu_window_set')):
        result=subprocess.run([dumper,'/exports',str(out/'bin'/filename)],env=dict(os.environ),capture_output=True,text=True,encoding='utf-8',errors='replace')
        write(out/(filename+'.exports.json'),dict(command=[dumper,'/exports',str(out/'bin'/filename)],returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,hardware_executed=False))
        if result.returncode or symbol not in result.stdout:raise ValueError('required runtime-window export missing: '+symbol)
    result=dict(schema='real-moe-runtime-window-build-complete-1',prepared=receipt(out/'PREPARED.json'),binaries=[receipt(p) for p in sorted((out/'bin').iterdir())],compiled=True,mathematical_graph_changed=False,extra_GPU_syncs=0,CPU_barriers_added=0,CUDA_objects_rebuilt=False,hardware_executed=False)
    write(out/'BUILD_COMPLETE.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','build','finalize'));p.add_argument('--out',required=True);p.add_argument('--llama-root');p.add_argument('--observer-build');p.add_argument('--fixed-build');p.add_argument('--cl');p.add_argument('--link');a=p.parse_args()
    r=prepare(a.out,a.llama_root,a.observer_build,a.fixed_build) if a.command=='prepare' else finalize(a.out,a.link) if a.command=='finalize' else build(a.out,a.cl,a.link,dict(os.environ));print(json.dumps(r,ensure_ascii=False))
if __name__=='__main__':main()
