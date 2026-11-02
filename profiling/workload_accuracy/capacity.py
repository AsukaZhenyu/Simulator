"""Direct safe below/exact/above tracked-budget checks with real allocations.

This tests the declared ledger capacity gate, not physical device capacity. The
baseline is bounded by the native 64 MiB tensor-payload guard; a one-byte-low
budget rejects before the allocation that would cross the measured arena peak.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from profiling.workload_accuracy.contract import digest, same, validate_case
from profiling.workload_accuracy.run import build_binding, collect, hardware_lock, write


def check_capacity(binary,case,output):
    case=validate_case(case);output=Path(output).resolve();output.mkdir(exist_ok=False,parents=True)
    binding=build_binding(binary);write(output/'binding.json',binding)
    rows=[]
    with hardware_lock():
        baseline,native=collect(binary,case,output/'baseline.case.json',samples=3,warmup=0,binding=binding)
        write(output/'baseline.raw.json',baseline)
        if native is None:
            raise RuntimeError('capacity baseline failed: '+baseline.get('validation_error',baseline['stderr']))
        peak=native['memory']['peak_tracked_reserved_bytes']
        for label,budget,accepted in (('one_byte_below',peak-1,False),('exact_peak',peak,True),('one_byte_above',peak+1,True)):
            same(build_binding(binary),binding,'capacity source/runtime stability before execution')
            trial=copy.deepcopy(case);trial['tracked_budget_bytes']=budget
            raw,result=collect(binary,trial,output/(label+'.case.json'),samples=3,warmup=0,binding=binding)
            write(output/(label+'.raw.json'),raw)
            same(build_binding(binary),binding,'capacity source/runtime stability after execution')
            if accepted:
                if result is None:
                    raise RuntimeError('tracked capacity acceptance failed: '+label+' '+raw.get('validation_error',raw['stderr']))
                same(result['memory']['peak_tracked_reserved_bytes'],peak,'same real allocation peak')
                same(result['memory']['final_tracked_reserved_bytes'],0,'lifecycle releases all tracked bytes')
            elif raw['returncode']==0 or 'tracked_budget_exceeded before allocation' not in raw['stderr']:
                raise ValueError('near-budget rejection must come from explicit pre-allocation capacity check')
            rows.append(dict(case=label,budget_bytes=budget,expected_tracked_acceptance=accepted,
                             actual_tracked_acceptance=raw['returncode']==0,
                             physical_capacity_verified=False))
    loads=[(layer['layer'],event['expert']) for layer in native['residency'] for event in layer['events'] if event['event']=='load']
    report=dict(schema='workload_accuracy_capacity/1',case_sha256=digest(case),
                tracked_peak_bytes=peak,rows=rows,passed=True,
                allocation_release_checked=True,expert_reload_observed=len(loads)>len(set(loads)),
                physical_capacity_verified=False,
                process_peak_bytes=None,cuda_private_workspace_bytes=None,
                scope='explicit_allocator_payload_budget_only')
    write(output/'capacity-report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-hardware',action='store_true',required=True)
    args=parser.parse_args()
    print(json.dumps(check_capacity(args.binary,json.loads(args.case.read_text(encoding='utf-8')),args.output),indent=2))
