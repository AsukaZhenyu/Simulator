"""Strict evidence for CUDA memory checkpoints; never infer a workspace bound."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from run_dense_cuda_measurements import validate_native_binding, measurement_signature
from llm_infer_model.m2.dense_cuda_contract import build_plan, native_arguments, validate_case

LABELS = ('after_context_initialization', 'after_upstream_preparation', 'after_stage_allocation',
          'after_setup', 'after_warmup', 'after_samples', 'after_stage_release')
SOURCE = 'cudaMemGetInfo_after_cudaDeviceSynchronize'
SCOPE = 'device_wide_cuda_memory_availability'


def same(actual, expected, label):
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError(label + ' differs from its declared evidence')


def integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{label} must be an integer >= {minimum}')
    return value


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(label + ' must be a lowercase SHA256')
    return value


def validate_memory_probe(case, payload, args):
    if not isinstance(payload, dict):
        raise ValueError('memory native output must be an object')
    fixed = {'schema':'ggml_dense_layer_cuda_memory_probe/1','purpose':'memory_diagnostics_only',
             'timing_usable_as_cost':False,'workspace_bytes':None,'workspace_bound':'unknown',
             'snapshot_scope':SCOPE,'source':SOURCE,'process_peak_bytes':None,
             'hardware_compute_performed':True,'snapshot_count':len(LABELS),
             'capacity_verified':False,'peak_measurement_claimed':False,'process_attribution_verified':False,
             'max_observed_used_bytes_scope':'maximum_of_seven_device_wide_snapshots_not_process_peak',
             'dense_validation_exit_code':0,
             'backend':f"CUDA{case['cuda_device']}",'cuda_device':case['cuda_device'],
             'cuda_graphs':'disabled'}
    for key, value in fixed.items():
        if key not in payload:
            raise ValueError('memory evidence is missing ' + key)
        same(payload[key], value, 'memory native ' + key)
    native = payload.get('dense_validation')
    validate_native_binding(case, native, args)
    same(payload.get('device_info'), native['device_info'], 'memory observed device identity')
    same(payload.get('upstream_setup_compute_repetitions'),native['upstream_setup_compute_repetitions'],'memory upstream preparation count')
    snapshots = payload.get('snapshots')
    if not isinstance(snapshots, list) or len(snapshots) != len(LABELS):
        raise ValueError('memory evidence must include all seven ordered checkpoints')
    checkpoints = []
    previous = -1
    baseline = None
    for expected_label, snapshot in zip(LABELS, snapshots):
        if not isinstance(snapshot, dict):
            raise ValueError('memory snapshot must be an object')
        same(snapshot.get('label'), expected_label, 'memory checkpoint order')
        same(integer(snapshot.get('device_index'), 'snapshot device'), case['cuda_device'], 'snapshot device')
        elapsed = integer(snapshot.get('elapsed_ns'), 'snapshot elapsed_ns')
        if elapsed <= previous:
            raise ValueError('memory checkpoint elapsed times must strictly increase')
        previous = elapsed
        total = integer(snapshot.get('total_bytes'), 'snapshot total bytes', 1)
        free = integer(snapshot.get('free_bytes'), 'snapshot free bytes')
        used = integer(snapshot.get('used_bytes'), 'snapshot used bytes')
        if free > total:
            raise ValueError('snapshot free bytes exceed total bytes')
        same(used, total-free, 'snapshot used=total-free')
        if baseline is None:
            baseline = used
        checkpoints.append({**copy.deepcopy(snapshot), 'delta_used_from_context_bytes':used-baseline})
    maximum = max(snapshot['used_bytes'] for snapshot in snapshots)
    same(integer(payload.get('max_observed_used_bytes'), 'maximum observed used bytes'), maximum, 'maximum sampled used bytes')
    accounting = native['memory_accounting']
    return {'device_info':copy.deepcopy(native['device_info']),'snapshot_scope':SCOPE,'source':SOURCE,
            'checkpoints':checkpoints,'max_observed_used_bytes':maximum,
            'max_observed_increase_from_context_bytes':maximum-baseline,
            'known_allocations':copy.deepcopy(accounting),
            'resident_padding_bytes':accounting['resident_allocated_bytes']-accounting['resident_logical_bytes'],
            'upstream_preparation_compute_repetitions':native['upstream_setup_compute_repetitions'],
            'runtime_history_includes_upstream_preparation':native['upstream_setup_compute_repetitions'] > 0,
            'workspace_bytes':None,'process_peak_bytes':None,'workspace_bound_known':False,
            'physical_capacity_verified':False,'timing_usable_as_cost':False,
            'model_capacity_under_declared_budget':None}


def load_memory_profile(directory):
    """Verify saved native evidence before a declared budget may reference it.

    Runtime SHA values bind source identity; the runtime need not remain installed
    on a machine reading a portable profile. This is not an attestation service.
    """
    directory = Path(directory).resolve()
    manifest_path = directory/'manifest.json'
    report_path = directory/'memory_report.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    report = json.loads(report_path.read_text(encoding='utf-8'))
    same(manifest.get('schema'), 'dense_cuda_memory_manifest/1', 'memory manifest schema')
    for key, expected in {'schema':'dense_cuda_memory_profile/1','purpose':'memory_diagnostics_only',
        'complete':True,'workspace_bytes':None,'workspace_bound_known':False,
        'physical_capacity_verified':False,'model_capacity_under_declared_budget':None,
        'timing_usable_as_cost':False}.items():
        if key not in report:
            raise ValueError('memory profile is missing ' + key)
        same(report[key], expected, 'memory profile ' + key)
    profile_id = manifest.get('profile_id')
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError('memory profile_id must be explicit')
    same(report.get('profile_id'), profile_id, 'memory profile identity')
    protocol = manifest.get('protocol')
    if not isinstance(protocol, dict):
        raise ValueError('memory protocol is missing')
    device = integer(protocol.get('cuda_device'), 'memory CUDA device')
    same(protocol.get('cuda_graphs'), 'disabled', 'memory CUDA graphs')
    same(protocol.get('sequential_execution'), True, 'memory sequential execution')
    args = SimpleNamespace(warmup=integer(protocol.get('warmup'), 'memory warmup'),
                           samples=integer(protocol.get('samples'), 'memory samples',3),
                           seed=integer(protocol.get('seed'), 'memory seed'))
    if args.samples > 10000:
        raise ValueError('memory samples exceed native limit')
    plan = build_plan(manifest.get('suite'), device, 'disabled')
    plan_by_id = {case['id']:case for case in plan}
    saved_plan = json.loads((directory/'plan.json').read_text(encoding='utf-8'))
    if not isinstance(saved_plan,list) or len(saved_plan)!=len(plan):
        raise ValueError('saved memory plan is incomplete')
    same({case['id']:case for case in saved_plan},plan_by_id,'saved memory plan')
    records = report.get('records')
    if not isinstance(records,list) or len(records)!=len(plan):
        raise ValueError('memory records must cover the full declared plan')
    software = manifest.get('software',{})
    same(software.get('child_flags',{}).get('GGML_CUDA_DISABLE_GRAPHS'),'1','memory graph disable environment')
    runtime_hashes = software.get('runtime_sha256')
    if not isinstance(runtime_hashes,dict) or not runtime_hashes:
        raise ValueError('memory runtime hash evidence is missing')
    for path, value in runtime_hashes.items():
        if not isinstance(path,str) or not path:
            raise ValueError('memory runtime identity must be nonempty')
        digest(value,'runtime SHA256')
    seen = set()
    observed_device = None
    for row in records:
        if not isinstance(row,dict) or row.get('status')!='observed':
            raise ValueError('failed memory records cannot bind a budget')
        case = validate_case(row.get('case'))
        same(row.get('case_id'),case['id'],'memory case identity')
        if case['id'] in seen or case['id'] not in plan_by_id:
            raise ValueError('duplicate or undeclared memory case')
        seen.add(case['id'])
        same(case,plan_by_id[case['id']],'memory declared case')
        name = row.get('raw_output')
        if not isinstance(name,str) or not name or Path(name).is_absolute():
            raise ValueError('memory raw path must be relative')
        raw_path = (directory/name).resolve()
        if not raw_path.is_relative_to(directory):
            raise ValueError('memory raw path escapes profile')
        same(sha(raw_path),digest(row.get('raw_sha256'),'memory raw SHA256'),'memory raw file hash')
        raw = json.loads(raw_path.read_text(encoding='utf-8'))
        same(raw.get('case'),case,'memory raw case')
        same(raw.get('returncode'),0,'memory native returncode')
        if raw.get('timed_out') or 'error' in raw:
            raise ValueError('memory raw execution failed')
        command = raw.get('command')
        if not isinstance(command,list) or not command or not isinstance(command[0],str):
            raise ValueError('memory raw command is missing')
        basename = command[0].replace(chr(92),'/').split('/')[-1]
        if basename not in ('ggml-dense-layer-cuda-memory-probe','ggml-dense-layer-cuda-memory-probe.exe'):
            raise ValueError('memory raw executable identity is incorrect')
        same(command[1:],native_arguments(case,args.warmup,args.samples,args.seed),'memory raw arguments')
        binary_hash = digest(runtime_hashes.get(command[0]),'memory native binary hash')
        same(row.get('signature'),measurement_signature(case,binary_hash),'memory math/device signature')
        if not isinstance(raw.get('stdout'),str) or not isinstance(raw.get('stderr'),str):
            raise ValueError('memory stdout/stderr must be retained')
        native = json.loads(raw['stdout'])
        same(row.get('native'),native,'memory raw native output')
        observation = validate_memory_probe(case,native,args)
        same(row.get('observation'),observation,'regenerated memory observation')
        if observed_device is None:
            observed_device = native['device_info']
        same(native['device_info'],observed_device,'single memory profile device')
    same(report.get('device_info'),observed_device,'memory report device identity')
    return {'device_info':copy.deepcopy(observed_device),'records':copy.deepcopy(records),
            'source_hashes':{'manifest_sha256':sha(manifest_path),'memory_report_sha256':sha(report_path)},
            'workspace_bound_known':False,'physical_capacity_verified':False}
