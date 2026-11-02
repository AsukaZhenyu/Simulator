"""Collect auditable CUDA memory checkpoints, without producing timing costs."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

from cuda_memory_contract import validate_memory_probe, sha
from run_dense_cuda_measurements import measurement_signature
from llm_infer_model.m2.dense_cuda_contract import build_plan, native_arguments
from run_measurements import dump, environment
from llm_infer_model.tensor.spec import SpecError

ROOT = Path(__file__).resolve().parent


def native_command(case, binary_dir, args):
    suffix = '.exe' if os.name == 'nt' else ''
    return [str(binary_dir/('ggml-dense-layer-cuda-memory-probe'+suffix)),
            *native_arguments(case,args.warmup,args.samples,args.seed)]


def _text(value):
    if isinstance(value,bytes):
        return value.decode('utf-8',errors='replace')
    return value or ''


def write_report(output, profile_id, plan, rows):
    successful = [row for row in rows if row['status']=='observed']
    device_info = successful[0]['native']['device_info'] if successful else None
    report = {'schema':'dense_cuda_memory_profile/1','profile_id':profile_id,
        'purpose':'memory_diagnostics_only','complete':len(rows)==len(plan) and len(successful)==len(plan),
        'device_info':device_info,'workspace_bytes':None,'workspace_bound_known':False,
        'physical_capacity_verified':False,'model_capacity_under_declared_budget':None,
        'timing_usable_as_cost':False,'records':rows}
    dump(output/'memory_report.json',report)
    with (output/'summary.csv').open('w',encoding='utf-8',newline='') as stream:
        fields=['case_id','kind','phase','tokens','visible_history','status','resident_allocated_bytes',
                'gallocr_allocated_bytes','max_observed_device_used_bytes','workspace_bytes','raw_output']
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
        writer.writeheader()
        for row in rows:
            observation=row.get('observation',{})
            known=observation.get('known_allocations',{})
            writer.writerow({'case_id':row['case_id'],
                **{key:row['case'][key] for key in ('kind','phase','tokens','visible_history')},
                'status':row['status'],'resident_allocated_bytes':known.get('resident_allocated_bytes'),
                'gallocr_allocated_bytes':known.get('internal_allocated_buffer_bytes'),
                'max_observed_device_used_bytes':observation.get('max_observed_used_bytes'),
                'workspace_bytes':None,'raw_output':row['raw_output']})
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',required=True,type=Path)
    parser.add_argument('--native-bin',required=True,type=Path)
    parser.add_argument('--runtime-dir',action='append',default=[],type=Path)
    parser.add_argument('--cuda-device',required=True,type=int)
    parser.add_argument('--cuda-graphs',required=True,choices=['disabled'])
    parser.add_argument('--warmup',type=int,default=1)
    parser.add_argument('--samples',type=int,default=3,help='validation repetitions only; timings are not costs')
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--timeout-s',type=float,default=180)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--keep-going',action='store_true')
    args=parser.parse_args(argv)
    if args.cuda_device<0 or args.warmup<0 or not 3<=args.samples<=10000 or args.seed<0 or not math.isfinite(args.timeout_s) or args.timeout_s<=0:
        parser.error('device/warmup/seed must be nonnegative, samples 3..10000, timeout positive finite')
    suite_path=args.suite.resolve()
    suite=json.loads(suite_path.read_text(encoding='utf-8'))
    plan=build_plan(suite,args.cuda_device,args.cuda_graphs)
    random.Random(args.seed).shuffle(plan)
    binary_dir=args.native_bin.resolve()
    commands=[native_command(case,binary_dir,args)for case in plan]
    if args.dry_run:
        print(json.dumps({'purpose':'memory_diagnostics_only','plan':plan,'commands':commands},indent=2))
        return 0
    output=args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('output directory must be empty: '+str(output))
    binary=Path(commands[0][0])
    if not binary.is_file():
        raise ValueError('memory native binary does not exist: '+str(binary))
    runtime_dirs=[path.resolve()for path in args.runtime_dir]
    if any(not path.is_dir() for path in runtime_dirs):
        raise ValueError('all runtime directories must exist')
    child_env=os.environ.copy()
    child_env['PATH']=os.pathsep.join([str(path)for path in runtime_dirs]+[child_env.get('PATH','')])
    if os.name!='nt':
        child_env['LD_LIBRARY_PATH']=os.pathsep.join([str(path)for path in runtime_dirs]+[child_env.get('LD_LIBRARY_PATH','')])
    child_env['GGML_CUDA_DISABLE_GRAPHS']='1'
    child_env.pop('LLAMA_VIZ_TIMING',None)
    lock_path=ROOT/'.measurement.lock'
    try:
        lock=lock_path.open('x',encoding='utf-8',newline='\n')
    except FileExistsError:
        raise ValueError('another profiling measurement holds '+str(lock_path))
    with lock:
        lock.write(str(os.getpid())+'\n')
    try:
        output.mkdir(parents=True,exist_ok=True)
        (output/'raw').mkdir()
        profile_id='dense_cuda_memory_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        runtime_files={binary}
        for directory in runtime_dirs:
            for pattern in ('ggml*.dll','libggml*.so*','libggml*.dylib'):
                runtime_files.update(directory.glob(pattern))
        source_files=[ROOT/'run_dense_cuda_memory.py',ROOT/'cuda_memory_contract.py',ROOT/'run_dense_cuda_measurements.py',
                      ROOT/'native/dense_layer_cuda_probe.cpp',ROOT/'native/dense_layer_cuda_memory_probe.cpp',ROOT/'native/CMakeLists.txt']
        manifest={'schema':'dense_cuda_memory_manifest/1','profile_id':profile_id,'suite':suite,
            'suite_path':str(suite_path),'suite_sha256':sha(suite_path),'environment':environment(),
            'protocol':{'cuda_device':args.cuda_device,'cuda_graphs':'disabled','warmup':args.warmup,
                'samples':args.samples,'seed':args.seed,'sequential_execution':True,
                'purpose':'memory_diagnostics_only','checkpoints':'synchronized_device_wide_availability_not_process_peak'},
            'software':{'benchmark_source_sha256':{str(path.relative_to(ROOT)):sha(path)for path in source_files},
                'runtime_sha256':{str(path):sha(path)for path in sorted(runtime_files)},
                'runtime_hash_scope':'memory executable and supplied GGML libraries; final resolved loader paths not observed',
                'runtime_search_dirs':[str(path)for path in runtime_dirs],
                'child_flags':{key:child_env.get(key)for key in ('GGML_CUDA_DISABLE_GRAPHS','CUDA_VISIBLE_DEVICES')}}}
        dump(output/'manifest.json',manifest)
        dump(output/'plan.json',plan)
        rows=[]
        observed_device=None
        for index,(case,command)in enumerate(zip(plan,commands),1):
            print(f"[{index}/{len(plan)}] {case['id']}",flush=True)
            raw_name=f'raw/{index:03d}.json'
            raw={'case':case,'command':command}
            row={'case_id':case['id'],'case':case,'raw_output':raw_name,
                 'signature':measurement_signature(case,sha(binary))}
            try:
                result=subprocess.run(command,capture_output=True,text=True,errors='replace',env=child_env,timeout=args.timeout_s)
                raw.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
                if result.returncode!=0:
                    raise ValueError('native memory probe failed: '+result.stderr.strip())
                native=json.loads(result.stdout)
                observation=validate_memory_probe(case,native,args)
                if observed_device is None:
                    observed_device=native['device_info']
                elif observed_device!=native['device_info']:
                    raise ValueError('memory suite changed actual CUDA device identity')
                row.update(status='observed',native=native,observation=observation)
            except subprocess.TimeoutExpired as error:
                raw.update(returncode=None,stdout=_text(error.stdout),stderr=_text(error.stderr),timed_out=True,error='native timeout')
                row.update(status='timed_out',error='native timeout')
            except (ValueError,SpecError,OSError) as error:
                raw['error']=str(error)
                row.update(status='failed',error=str(error))
            dump(output/raw_name,raw)
            row['raw_sha256']=sha(output/raw_name)
            rows.append(row)
            write_report(output,profile_id,plan,rows)
            if row['status']!='observed' and not args.keep_going:
                break
        report=write_report(output,profile_id,plan,rows)
        print(f"Observed {sum(row['status']=='observed'for row in rows)}/{len(plan)} CUDA memory cases: {output}; workspace remains unknown")
        return 0 if report['complete'] else 1
    finally:
        lock_path.unlink(missing_ok=True)


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (ValueError,OSError,SpecError)as error:
        print('run_dense_cuda_memory: '+str(error),file=sys.stderr)
        raise SystemExit(1)
