"""Private CPU-consumer tap build; never mutates the public/frozen runtime.

Only the pinned ggml-cpu.c translation unit is recompiled. Other CPU objects,
CUDA/llama libraries, graph construction and allocator are retained unchanged.
Building is software work, not authorization to execute inference.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
HOOK='''
// Private observer ABI; registration is outside all owned graph computation.
#include "route_tap_api.h"
static simulator_moe_tap_callback simulator_moe_observer = NULL;
static void * simulator_moe_observer_user = NULL;
void simulator_moe_tap_set(simulator_moe_tap_callback callback, void * user) {
    simulator_moe_observer_user = user;
    simulator_moe_observer = callback;
}
'''
ANCHOR='''    // extra_buffer op?
    if (ggml_cpu_extra_compute_forward(params, tensor)) {'''
REPLACEMENT='''    // Observe the exact CPU-consumed IDs before ordinary/repack dispatch.
    // Read-only callback: no extra graph node, barrier, flag, copy or sync.
    if (params->ith == 0 && tensor->op == GGML_OP_MUL_MAT_ID && simulator_moe_observer) {
        simulator_moe_observer(tensor, tensor->src[2], simulator_moe_observer_user);
    }

    // extra_buffer op?
    if (ggml_cpu_extra_compute_forward(params, tensor)) {'''
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def receipt(path):return dict(path=str(Path(path).resolve()),sha256=sha(path),bytes=Path(path).stat().st_size)
def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n')
# The two inline accessors the observer reads from the internal context. They are
# named here, not inline in `prepare`, so a second build path (the Linux chain)
# inserts the same bytes instead of a second transcription of them.
CONTEXT_NEEDLE='    uint32_t graph_max_nodes(uint32_t n_tokens) const;'
CONTEXT_METHODS='''    // Private read-only inline accessors; no field/layout/runtime changes.
    llm_graph_result * simulator_tap_graph_result() const { return gf_res_prev.get(); }
    ggml_backend_sched_t simulator_tap_scheduler() const { return sched.get(); }
'''
def private_context_header(text):
    """`llama-context.h` with the two accessors added, or a refusal."""
    if text.count(CONTEXT_NEEDLE)!=1:raise ValueError('pinned internal context accessor anchor differs')
    return text.replace(CONTEXT_NEEDLE,CONTEXT_METHODS+CONTEXT_NEEDLE)
def patched_cpu(source):
    if source.count(ANCHOR)!=1:raise ValueError('pinned CPU dispatch anchor differs')
    marker='// ggml_compute_forward_mul_mat_id\n'
    if source.count(marker)!=1:raise ValueError('pinned CPU observer declaration anchor differs')
    return source.replace(ANCHOR,REPLACEMENT,1).replace(marker,HOOK+'\n'+marker,1)
def prepare(out,llama_root,native_dir=None):
    native=Path(native_dir).resolve() if native_dir else HERE/'tap_native'
    out=Path(out).resolve();llama_root=Path(llama_root).resolve()
    if out.exists():raise ValueError('new private build root required')
    (out/'source').mkdir(parents=True);(out/'objects').mkdir();(out/'bin').mkdir();(out/'lib').mkdir()
    original=llama_root/'ggml/src/ggml-cpu/ggml-cpu.c';text=original.read_text(encoding='utf-8');patched=patched_cpu(text)
    if patched.count('void simulator_moe_tap_set(')!=1:raise ValueError('observer export insertion differs')
    (out/'source/ggml-cpu.c').write_text(patched,encoding='utf-8',newline='\n')
    (out/'cpu-consumer-tap.patch').write_text(''.join(difflib.unified_diff(text.splitlines(True),patched.splitlines(True),fromfile='original/ggml-cpu.c',tofile='private/ggml-cpu.c')),encoding='utf-8',newline='\n')
    for name in ('route_tap_api.h','route_tap_pod.hpp','route_context_inspect.hpp','real_moe_tap_probe.cpp'):
        shutil.copy2(native/name,out/'source'/name)
    h=llama_root/'src/llama-context.h';ht=h.read_text(encoding='utf-8')
    (out/'source/private_llama_context.h').write_text(private_context_header(ht),encoding='utf-8',newline='\n')
    objects=sorted((llama_root/'build/ggml/src/ggml-cpu.dir/Release').rglob('*.obj'))
    original_object=llama_root/'build/ggml/src/ggml-cpu.dir/Release/ggml-cpu/ggml-cpu.c.obj'
    if original_object not in objects:raise ValueError('matching original CPU object unavailable')
    manifest=dict(schema='real-moe-consumer-tap-build-4',prepared_at_utc=datetime.now(timezone.utc).isoformat(),llama_root=str(llama_root),
        original_cpu_source=receipt(original),original_context_header=receipt(h),original_cpu_object=receipt(original_object),
        reused_cpu_objects=[receipt(p) for p in objects if p!=original_object],
        source_files=[receipt(p) for p in sorted((out/'source').iterdir())],
        native_source_sha256=sha(out/'source/real_moe_tap_probe.cpp'),
        common_source_modified=False,CUDA_rebuilt=False,llama_rebuilt=False,graph_or_allocator_modified=False,
        hook_scope='CPU MUL_MAT_ID consumption, ith=0, before extra/repack; configure only while no graph runs',
        hardware_executed=False,compiled=False)
    write(out/'PREPARED.json',manifest);return manifest
def run_record(out,name,argv,env):
    start=datetime.now(timezone.utc).isoformat();result=subprocess.run(argv,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
    record=dict(command=argv,started_at_utc=start,ended_at_utc=datetime.now(timezone.utc).isoformat(),returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,hardware_executed=False)
    write(out/(name+'.json'),record)
    if result.returncode:raise RuntimeError(name+' failed; retained execution receipt')
def build(out,cl,link,env):
    out=Path(out).resolve();m=json.loads((out/'PREPARED.json').read_text());lr=Path(m['llama_root']);src=out/'source';lib=lr/'build/ggml/src/Release';llama=lr/'build/src/Release/llama.lib'
    for r in m['source_files']+m['reused_cpu_objects']:
        if sha(r['path'])!=r['sha256']:raise ValueError('prepared source/object changed: '+r['path'])
    inc=[src,lr/'ggml/src',lr/'ggml/src/ggml-cpu',lr/'ggml/include',lr/'include',lr/'src',lr/'vendor']
    includes=[a for p in inc for a in ('/I',str(p))]
    definitions=['_WINDLL','_MBCS','WIN32','_WINDOWS','NDEBUG','GGML_BACKEND_BUILD','GGML_BACKEND_SHARED','GGML_USE_OPENMP','GGML_USE_LLAMAFILE','GGML_USE_CPU_REPACK','GGML_AVX512','_CRT_SECURE_NO_WARNINGS','GGML_SCHED_MAX_COPIES=4','_XOPEN_SOURCE=600','GGML_SHARED','ggml_cpu_EXPORTS','SIMULATOR_MOE_TAP_BUILD']
    args=[cl,'/nologo','/c','/O2','/Ob2','/MD','/GS','/arch:AVX512','/fp:precise','/openmp','/std:c11','/TC','/utf-8','/bigobj']+includes+['/D'+d for d in definitions]+['/Fo'+str(out/'objects/ggml-cpu.c.obj'),str(src/'ggml-cpu.c')]
    run_record(out,'CPU_COMPILE',args,env)
    objs=[r['path'] for r in m['reused_cpu_objects']]+[str(out/'objects/ggml-cpu.c.obj')]
    rsp=out/'cpu-link.rsp';rsp.write_text('\n'.join('"'+x+'"' for x in objs+[str(lib/'ggml-base.lib'),'kernel32.lib','user32.lib','advapi32.lib'])+'\n/DLL /NOLOGO /INCREMENTAL:NO /MACHINE:X64 /OUT:"'+str(out/'bin/ggml-cpu.dll')+'" /IMPLIB:"'+str(out/'lib/ggml-cpu.lib')+'"\n',encoding='utf-8',newline='\n')
    run_record(out,'CPU_LINK',[link,'@'+str(rsp)],env)
    native_defs=['GGML_SHARED','GGML_BACKEND_SHARED','REAL_MOE_SOURCE_SHA256="'+m['native_source_sha256']+'"']
    args=[cl,'/nologo','/c','/O2','/Ob2','/MD','/EHsc','/std:c++17','/utf-8','/bigobj']+includes+['/D'+d for d in native_defs]+['/Fo'+str(out/'objects/real_moe_tap_probe.obj'),str(src/'real_moe_tap_probe.cpp')]
    run_record(out,'NATIVE_COMPILE',args,env)
    args=[link,'/NOLOGO','/MACHINE:X64','/OUT:'+str(out/'bin/real-moe-tap-probe.exe'),str(out/'objects/real_moe_tap_probe.obj'),str(llama),str(lib/'ggml.lib'),str(lib/'ggml-base.lib'),str(lib/'ggml-cpu.lib'),'bcrypt.lib','kernel32.lib']
    run_record(out,'NATIVE_LINK',args,env)
    for name in ('ggml-base.dll','ggml-cuda.dll','ggml.dll','llama.dll'):
        shutil.copy2(lr/'build/bin/Release'/name,out/'bin'/name)
    result=dict(schema='real-moe-consumer-tap-build-complete-4',prepared=receipt(out/'PREPARED.json'),binaries=[receipt(p) for p in sorted((out/'bin').iterdir())],execution_receipts=[receipt(out/(n+'.json')) for n in ('CPU_COMPILE','CPU_LINK','NATIVE_COMPILE','NATIVE_LINK')],compiled=True,hardware_executed=False)
    write(out/'BUILD_COMPLETE.json',result);return result
def main():
    import os
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','build'));p.add_argument('--out',required=True);p.add_argument('--llama-root');p.add_argument('--cl');p.add_argument('--link');p.add_argument('--native-dir');a=p.parse_args()
    r=prepare(a.out,a.llama_root,a.native_dir) if a.command=='prepare' else build(a.out,a.cl,a.link,dict(os.environ));print(json.dumps(r,ensure_ascii=False))
if __name__=='__main__':main()
