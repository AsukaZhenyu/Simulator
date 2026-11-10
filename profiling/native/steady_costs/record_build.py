"""Record all explicit native build inputs without running hardware."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def backend_fields(backend,architecture):
    """The receipt's claims about the backend, as the binary actually behaves.

    The parent receipt was written for a Win32 CUDA build and says two things a
    Metal build makes false: that the event instrumentation exists, and that the
    CUDA stream schedule was observed. A probe built here refuses `events` at
    request construction and has no CUDA flags to query, so its manifest says so
    rather than inheriting the claim.

    `device_memory` is the same three-way rule `profiling.host.device_memory()` uses, and
    it is spelled here rather than imported because this script runs from a CMake
    post-build hook with no package on `sys.path`. A Mac's accelerator is the SoC's own
    GPU, so `metal` decides `shared`. CUDA on `aarch64` does not decide anything: a
    Grace-Blackwell host can carry a discrete card over PCIe, and a receipt that claimed
    `shared` there would price every copy at zero. The owner states it with
    `IR_DEVICE_MEMORY`, and until they do the receipt says `unknown`.
    """
    arch=(architecture or '').lower()
    memory='shared' if backend=='metal' else 'separate'
    if backend!='metal' and arch in ('aarch64','arm64'):
        declared=os.environ.get('IR_DEVICE_MEMORY','').strip().lower()
        memory=declared if declared in ('shared','separate') else 'unknown'
    fields=dict(backend=backend,architecture=architecture,device_memory=memory)
    if backend=='metal':
        fields.update(supported_instrumentation=['bare'],
            cuda_schedule_observation='not applicable: this build is Metal, which has no CUDA device flags to query',
            cuda_schedule_override_environment=None,
            cuda_schedule_override_values=[],
            cuda_schedule_unset_behavior='not applicable',
            cuda_schedule_requested_mode_must_be_observed=False,
            timing_events_scope='host-side steady clock between records; this build has no CUDA events')
    return fields


def main():
    parser=argparse.ArgumentParser()
    for name in ('binary','source-manifest','compiler','cmake-cache','runtime-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--compiler-version',required=True)
    # The backend this binary was linked for. The parent receipt describes the
    # Win32 CUDA build it was written for, and two of its claims are false of a
    # Metal build: that the event instrumentation exists, and that a CUDA stream
    # schedule was observed. A manifest that kept them would let a campaign cite
    # a capability this binary refuses at request construction.
    parser.add_argument('--backend',choices=('cuda','metal'),default=None)
    parser.add_argument('--architecture',default=None)
    args=parser.parse_args()
    parent=Path(__file__).resolve().parents[1]/'state_response/record_build.py'
    spec=importlib.util.spec_from_file_location('steady_parent_record',parent)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    value=module.build_record(args.binary,args.source_manifest,args.compiler,args.compiler_version,args.cmake_cache,args.runtime_dir)
    # The parent receipt describes the Win32 build it was written for. This probe
    # also builds on Linux, where the thread id is `SYS_gettid` behind the same
    # spelling, so the receipt says which one it was.
    value['main_thread_id_provider']=('Windows GetCurrentThreadId' if sys.platform=='win32'
        else 'POSIX gettid behind GetCurrentThreadId from portable/platform.hpp'
             + (' (pthread_self on macOS)' if sys.platform=='darwin' else ''))
    value.update(format_version='steady-costs-native-build-1',
        primary_execution_boundary='unchanged Executor.action; batched QPC wall, optional full-context stamps; no CPU cycle queries',
        component_execution_boundary='explicit new GGML component graph, procedural inputs, independent NumPy oracle required',
        supported_programs=['highlevel-accuracy-program-v1','steady-component-program-1'],
        supported_modes=['full','full-stamped','action','all-actions','async-graph','sync-idle'],
        supported_instrumentation=['bare','events'],
        timing_events_scope='pinned GGML actual stream event span, including host submission gaps; not GPU active kernel time',
        collected_cpu_cycles=False,performance_parameters_fitted=False,hardware_accuracy_qualified=False)
    if args.backend is not None:
        value.update(backend_fields(args.backend,args.architecture))
    for path in (parent,Path(__file__)):
        value['bound_files'][str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    with (args.binary.parent/'build-manifest.json').open('w',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


if __name__=='__main__':main()
