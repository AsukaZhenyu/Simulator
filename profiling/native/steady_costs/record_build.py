"""Record all explicit native build inputs without running hardware."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    for name in ('binary','source-manifest','compiler','cmake-cache','runtime-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--compiler-version',required=True)
    args=parser.parse_args()
    parent=Path(__file__).resolve().parents[1]/'state_response/record_build.py'
    spec=importlib.util.spec_from_file_location('steady_parent_record',parent)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    value=module.build_record(args.binary,args.source_manifest,args.compiler,args.compiler_version,args.cmake_cache,args.runtime_dir)
    value.update(format_version='steady-costs-native-build-1',
        primary_execution_boundary='unchanged Executor.action; batched QPC wall, optional full-context stamps; no CPU cycle queries',
        component_execution_boundary='explicit new GGML component graph, procedural inputs, independent NumPy oracle required',
        supported_programs=['highlevel-accuracy-program-v1','steady-component-program-1'],
        supported_modes=['full','full-stamped','action','all-actions','async-graph','sync-idle'],
        supported_instrumentation=['bare','events'],
        timing_events_scope='pinned GGML actual stream event span, including host submission gaps; not GPU active kernel time',
        collected_cpu_cycles=False,performance_parameters_fitted=False,hardware_accuracy_qualified=False)
    for path in (parent,Path(__file__)):
        value['bound_files'][str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    with (args.binary.parent/'build-manifest.json').open('w',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


if __name__=='__main__':main()
