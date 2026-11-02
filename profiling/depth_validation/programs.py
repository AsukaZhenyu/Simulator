"""Small depth-test configurations plus explicit binary verification inputs.

Only profiling's real executor needs numeric weights. The simulator and strategy
coordinates use the unchanged shapes, dependencies, placements and byte counts.
No timing is performed here; array generation and references are software work.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np

from profiling.highlevel_accuracy.bridge import (
    BOUNDARY, FORMAT, _values, export_plan, reference_ports, sha, validate_program,
)
from profiling.highlevel_accuracy.multilayer import build_multilayer_case

SCHEMA = 'dense-initial-assets-1'
RECIPES = (('cpu','cpu','cpu'), ('gpu','gpu','gpu'), ('gpu','cpu','gpu'))
DTYPES = {'GGML_TYPE_F32': np.dtype('<f4'), 'GGML_TYPE_I32': np.dtype('<i4')}


def require(ok, message):
    if not ok: raise ValueError(message)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''): h.update(block)
    return h.hexdigest()


def receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha(path), bytes=path.stat().st_size)


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, separators=(',', ':'), allow_nan=False); stream.write('\n')


def matrix():
    return [dict(id=f'D-{phase}-L{layers}-S{strategy}', phase=phase, layers=layers,
                 strategy=strategy, tokens=tokens, history=history)
            for phase, depths, tokens, history in (('prefill',(4,8,16,32),64,0),
                                                    ('decode',(16,32),1,128))
            for layers in depths for strategy in range(3)]


def compact_program(program, asset_pool):
    """Replace initializer arrays with one content-addressed, shared byte file.

    ``program`` remains unchanged. Values are not converted to Python lists or
    decimal JSON. Exact dtype, shape, bytes, file offsets and hashes remain bound.
    """
    validate_program(program)
    tensors = {t['id']: t for t in program['graph']['workload']['tensors']}
    require(set(program['initial_values']) == {tid for tid,t in tensors.items() if t.get('initial_locations')},
            'initial tensor inventory differs')
    asset_pool = Path(asset_pool).resolve(); asset_pool.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix='depth-initial-', suffix='.bin.tmp', dir=asset_pool)
    import os
    os.close(handle)
    temporary = Path(temporary)
    entries, offset, digest = {}, 0, hashlib.sha256()
    try:
        with temporary.open('wb') as output:
            for tid in sorted(program['initial_values']):
                tensor = tensors[tid]; dtype = tensor['dtype']
                require(dtype in DTYPES, 'unsupported initial asset dtype')
                value = np.ascontiguousarray(program['initial_values'][tid], dtype=DTYPES[dtype]).reshape(-1)
                require(value.size == math.prod(tensor['ne']) and value.nbytes == tensor['storage_bytes'],
                        'numeric initializer geometry differs')
                require(np.isfinite(value).all(), 'non-finite numeric initializer')
                payload = memoryview(value).cast('B')
                entries[tid] = dict(offset_bytes=offset, bytes=value.nbytes, elements=value.size,
                                    dtype=dtype, sha256=hashlib.sha256(payload).hexdigest())
                output.write(payload); digest.update(payload); offset += value.nbytes
        target = asset_pool/(digest.hexdigest()+'.bin')
        if target.exists():
            require(target.stat().st_size == offset and file_sha(target) == digest.hexdigest(),
                    'existing content-addressed initializer changed')
            temporary.unlink()
        else:
            temporary.rename(target)
    finally:
        if temporary.exists(): temporary.unlink()
    body = {key:deepcopy(value) for key,value in program.items()
            if key not in ('initial_values','identity','program_sha256')}
    body['initial_values'] = {}
    body['initial_assets'] = dict(schema=SCHEMA,
        file=dict(path=str(target),sha256=digest.hexdigest(),bytes=offset), tensors=entries)
    body['materialized_identity'] = deepcopy(program['identity'])
    body['materialized_program_sha256'] = program['program_sha256']
    body['identity'] = {key:sha(body[key]) for key in program['identity']}
    for key in ('initial_assets','materialized_identity','materialized_program_sha256'):
        body['identity'][key] = sha(body[key])
    body['program_sha256'] = sha(body)
    validate_compact(body)
    return body


def validate_compact(program):
    require(program.get('format_version')==FORMAT and program.get('boundary_contract')==BOUNDARY,
            'compact numeric assets require the exact F32 Dense boundary')
    require(program.get('initial_values')=={}, 'compact metadata cannot contain numerical tensor values')
    body = {key:value for key,value in program.items() if key!='program_sha256'}
    require(sha(body)==program.get('program_sha256'), 'compact program identity changed')
    for key,wanted in program['identity'].items():
        require(sha(program[key])==wanted, 'compact field identity changed: '+key)
    asset = program.get('initial_assets', {})
    require(asset.get('schema')==SCHEMA, 'unsupported initializer asset schema')
    path = Path(asset['file']['path'])
    require(path.is_file() and path.stat().st_size==asset['file']['bytes']
            and file_sha(path)==asset['file']['sha256'], 'initializer file bytes/hash changed')
    tensors = {t['id']:t for t in program['graph']['workload']['tensors']}
    expected = {tid for tid,t in tensors.items() if t.get('initial_locations')}
    require(set(asset['tensors'])==expected, 'initializer asset tensor inventory differs')
    offset = 0
    with path.open('rb') as stream:
        for tid in sorted(expected):
            entry,tensor = asset['tensors'][tid],tensors[tid]
            require(entry['offset_bytes']==offset and offset%4==0, 'initializer offset/overlap differs')
            require(entry['dtype']==tensor['dtype'] and entry['dtype'] in DTYPES
                    and entry['elements']==math.prod(tensor['ne'])
                    and entry['bytes']==tensor['storage_bytes']==4*entry['elements'],
                    'initializer dtype/geometry differs')
            payload = stream.read(entry['bytes'])
            require(len(payload)==entry['bytes'] and hashlib.sha256(payload).hexdigest()==entry['sha256'],
                    'initializer tensor bytes/hash changed')
            offset += entry['bytes']
    require(offset==asset['file']['bytes'], 'unbound trailing initializer bytes')
    return program


def materialize(program):
    """Load read-only NumPy views for independent numerical reference only."""
    validate_compact(program)
    omitted = {'initial_values','initial_assets','materialized_identity',
               'materialized_program_sha256','identity','program_sha256'}
    restored = {key:deepcopy(value) for key,value in program.items() if key not in omitted}
    asset = program['initial_assets']
    raw = np.memmap(asset['file']['path'], mode='r', dtype=np.uint8)
    restored['initial_values'] = {tid:np.frombuffer(raw, dtype=DTYPES[entry['dtype']],
            count=entry['elements'], offset=entry['offset_bytes'])
        for tid,entry in asset['tensors'].items()}
    restored['identity'] = deepcopy(program['materialized_identity'])
    restored['program_sha256'] = program['materialized_program_sha256']
    validate_program(restored)
    return restored


def reseed_case(template, directory, *, seed, asset_pool=None, with_reference=True):
    """Reuse an exact compiled graph with independent numerical inputs.

    This does not call the compiler, simulator or target timing reader. Geometry,
    bindings, accepted actions, scope identities and source-bound declarations
    remain unchanged. All initial numerical values are generated from the new
    explicit seed using the original bridge's per-tensor generator.
    """
    require(type(seed) is int and 0 <= seed < 2**32, 'seed must be uint32')
    directory = Path(directory).resolve()
    require(not directory.exists(), 'new reseeded case directory required')
    if isinstance(template, dict):
        compact = template
        provenance = dict(program_sha256=compact.get('program_sha256'), path=None)
    else:
        source = Path(template).resolve()
        compact = json.loads(source.read_text(encoding='utf-8'))
        provenance = dict(**receipt(source),program_sha256=compact.get('program_sha256'))
    validate_compact(compact)
    boundary = compact['graph'].get('boundary') or {}
    require(not (boundary.get('ffn_execution') or {}).get('weight_partitions'),
            'reseeded depth cases do not guess partition-parent numerical values')
    omitted = {'initial_values','initial_assets','materialized_identity',
               'materialized_program_sha256','identity','program_sha256'}
    program = {key:deepcopy(value) for key,value in compact.items() if key not in omitted}
    tensors = {t['id']:t for t in program['graph']['workload']['tensors']}
    norm_roles = {'input_norm','post_attention_norm','q_norm','k_norm'}
    norms = {tid for weights in program.get('layer_weight_bindings', {}).values()
             for role,tid in weights.items() if role in norm_roles}
    positions = {}
    for binding in program['bindings']:
        norms.update(tid for role,tid in binding['operands'].items() if role in norm_roles)
        if binding['kind'] != 'projection': continue
        tid = binding['operands']['positions']
        start,n = binding['position_start'],binding['shape']['tokens']
        require(type(start) is int and start >= 0 and type(n) is int and n > 0,
                'projection position range is not explicit')
        values = np.arange(start,start+n,dtype=np.int32)
        require(tid not in positions or np.array_equal(positions[tid],values),
                'one position tensor has incompatible projection ranges')
        positions[tid] = values
    initial = {}
    for tid,tensor in tensors.items():
        if not tensor.get('initial_locations'): continue
        if tensor['dtype'] == 'GGML_TYPE_I32':
            require(tid in positions and positions[tid].size==math.prod(tensor['ne']),
                    'unrecognized initial position tensor')
            initial[tid] = positions[tid]
        else:
            require(tensor['dtype']=='GGML_TYPE_F32','depth reseed requires F32 data')
            initial[tid] = _values(tensor,seed,gamma=tid in norms)
    program['seed'] = seed
    program['initial_values'] = initial
    program['identity'] = {key:sha(program[key]) for key in compact['materialized_identity']}
    program['program_sha256'] = sha(program)
    validate_program(program)
    projection = [b for b in program['bindings'] if b['kind']=='projection']
    tokens = {b['shape']['tokens'] for b in projection}
    histories = {b['shape']['visible_history'] for b in projection}
    require(len(tokens)==len(histories)==1, 'depth reseed requires homogeneous token/history shape')
    placements = []
    operations = {b['operation_id']:b['kind'] for b in program['bindings']}
    for kind in ('projection','attention','dense_ffn'):
        devices = {a['device'] for a in program['actions'] if a['kind']=='COMPUTE'
                   and operations[a['target_id']]==kind}
        require(len(devices)==1,'depth reseed requires homogeneous complete macro placement')
        placements.append(next(iter(devices)))
    require(tuple(placements) in RECIPES,'unknown depth placement recipe')
    strategy = RECIPES.index(tuple(placements)); tokens=tokens.pop(); history=histories.pop()
    layers = len(program.get('invocation_scopes', [])) or 1
    directory.mkdir(parents=True)
    output = compact_program(program,asset_pool or directory/'assets')
    write_new(directory/'program.json',output)
    if with_reference: write_new(directory/'reference.json',reference_ports(program))
    result = dict(schema='connected-dense-depth-case-1',layers=layers,strategy=strategy,
        phase='decode' if tokens==1 and history>0 else 'prefill',seed=seed,tokens=tokens,history=history,
        program=receipt(directory/'program.json'),initial_assets=output['initial_assets']['file'],
        source_generator=receipt(Path(__file__)),reseeded_from=provenance,
        actual_connected_hidden_seams=layers-1,native_timing_executed=False,
        numeric_tensor_values_in_program_json=False,compiler_or_simulator_rerun=False,
        reference=receipt(directory/'reference.json') if with_reference else None)
    write_new(directory/'case.json',result)
    return result


def build_one(directory, *, layers, strategy, phase='prefill', seed=2026102701,
              asset_pool=None, with_reference=True, hidden=512, intermediate=1024,
              heads=8, kv_heads=2, head_dim=64, tokens=None, history=None):
    require(type(layers) is int and 1 <= layers <= 32, 'depth must be within 1..32')
    require(type(strategy) is int and 0 <= strategy < 3, 'strategy must be S0, S1 or S2')
    require(phase in ('prefill','decode'), 'unknown inference phase')
    directory = Path(directory).resolve(); require(not directory.exists(), 'new case directory required')
    directory.mkdir(parents=True)
    tokens = (64 if phase=='prefill' else 1) if tokens is None else tokens
    history = (0 if phase=='prefill' else 128) if history is None else history
    plan = build_multilayer_case(directory/'inputs', layers=layers, hidden=hidden,
        intermediate=intermediate, heads=heads, kv_heads=kv_heads, head_dim=head_dim,
        tokens=tokens, history=history, threads=1, devices=RECIPES[strategy])
    program = export_plan(plan.compile(), seed=seed, numeric_arrays=True,
                          element_limit=128_000_000, lightweight_actions=True)
    compact = compact_program(program, asset_pool or directory/'assets')
    write_new(directory/'program.json',compact)
    reference = None
    if with_reference:
        reference = reference_ports(program)
        write_new(directory/'reference.json',reference)
    result = dict(schema='connected-dense-depth-case-1', layers=layers, strategy=strategy,
        phase=phase, seed=seed, tokens=tokens, history=history, program=receipt(directory/'program.json'),
        initial_assets=compact['initial_assets']['file'], native_timing_executed=False,
        source_generator=receipt(Path(__file__)),
        numeric_tensor_values_in_program_json=False,
        actual_connected_hidden_seams=layers-1, reference=receipt(directory/'reference.json') if reference is not None else None)
    write_new(directory/'case.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--layers',type=int,required=True)
    p.add_argument('--strategy',type=int,default=2);p.add_argument('--phase',choices=('prefill','decode'),default='prefill')
    p.add_argument('--seed',type=int,default=2026102701);p.add_argument('--asset-pool')
    p.add_argument('--no-reference',action='store_true');a=p.parse_args()
    print(json.dumps(build_one(a.out,layers=a.layers,strategy=a.strategy,phase=a.phase,seed=a.seed,
        asset_pool=a.asset_pool,with_reference=not a.no_reference),ensure_ascii=False),flush=True)
