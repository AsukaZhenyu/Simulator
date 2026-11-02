"""Compact representative F32 probes and independent deterministic oracles."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


def require(condition,message):
    if not condition:raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def semantic_program(program):
    """Numeric seed is not a performance table coordinate."""
    if program.get('format_version')=='steady-component-program-1':
        return {key:value for key,value in program.items() if key not in {'seed','program_sha256'}}
    return {key:program[key] for key in ('boundary_contract','physical','graph','bindings','actions')}


def representative_components():
    budget=1<<30
    rows=[('cpu-decode4096','matmul','cpu',dict(k=4096,m=4096,n=1)),
          ('gpu-decode4096','matmul','gpu',dict(k=4096,m=4096,n=1)),
          ('gpu-prefill4096-q64','matmul','gpu',dict(k=4096,m=4096,n=64)),
          ('gpu-pv4096-q1-kv4096','attention_pv','gpu',dict(heads=32,kv_heads=8,head_dim=128,query_tokens=1,kv_tokens=4096)),
          ('gpu-pv4096-q16-kv2048','attention_pv','gpu',dict(heads=32,kv_heads=8,head_dim=128,query_tokens=16,kv_tokens=2048)),
          ('copy-dram-vram-16m','transfer','gpu',dict(bytes=16<<20,src='dram',dst='vram')),
          ('copy-vram-dram-16m','transfer','gpu',dict(bytes=16<<20,src='vram',dst='dram')),
          ('copy-dram-dram-16m','transfer','cpu',dict(bytes=16<<20,src='dram',dst='dram'))]
    return [dict(id=name,program=dict(format_version='steady-component-program-1',kind=kind,device=device,
                threads=1,**({'cuda_device':0} if device=='gpu' else {}),
                shape=shape,allocation_budget_bytes=budget)) for name,kind,device,shape in rows]


def validate_component(program):
    required={'format_version','kind','seed','device','shape','allocation_budget_bytes','program_sha256','threads'}
    if program.get('device')=='gpu':required.add('cuda_device')
    require(set(program)==required,'component fields differ')
    require(program['program_sha256']==digest({k:v for k,v in program.items() if k!='program_sha256'}),'component semantic hash differs')
    require(program['format_version']=='steady-component-program-1','component schema differs')
    require(type(program['seed']) is int and 0<=program['seed']<=0xffffffff,'uint32 seed required')
    require(program['device'] in {'cpu','gpu'},'explicit CPU/GPU backend required')
    require(program['threads']==1 and type(program['threads']) is int,'explicit one-thread component boundary required')
    if program['device']=='gpu':require(type(program['cuda_device']) is int and program['cuda_device']==0,'explicit CUDA0 required')
    require(type(program['allocation_budget_bytes']) is int and 0<program['allocation_budget_bytes']<=1<<30,'explicit allocation budget required')
    kind,shape=program['kind'],program['shape']
    fields={'matmul':{'k','m','n'},'attention_pv':{'heads','kv_heads','head_dim','query_tokens','kv_tokens'},
            'transfer':{'bytes','src','dst'}}
    require(kind in fields and set(shape)==fields[kind],'component geometry fields differ')
    for key,value in shape.items():
        if key not in {'src','dst'}:require(type(value) is int and value>0,'positive integer geometry required')
    if kind=='matmul':sizes=[shape['k']*shape['m'],shape['k']*shape['n'],shape['m']*shape['n']]
    elif kind=='attention_pv':
        require(shape['heads']%shape['kv_heads']==0,'PV broadcast ratio differs')
        sizes=[shape['head_dim']*shape['kv_tokens']*shape['kv_heads'],shape['kv_tokens']*shape['query_tokens']*shape['heads'],
               shape['head_dim']*shape['query_tokens']*shape['heads']]
    else:
        require(shape['src'] in {'dram','vram'} and shape['dst'] in {'dram','vram'},'copy route differs')
        require(shape['src']!='vram' or shape['dst']!='vram','VRAM-to-VRAM not in this campaign')
        require(program['device']==('gpu' if 'vram' in (shape['src'],shape['dst']) else 'cpu'),'copy backend differs')
        sizes=[]
    nominal=4*sum(sizes) if sizes else 2*shape['bytes']
    require(nominal<=program['allocation_budget_bytes'],'declared component allocations exceed budget')
    return program


def mix32(seed,count,ordinal):
    """Independent vectorized implementation of native mix32-index-v1."""
    require(type(count) is int and count>=0,'explicit element count required')
    z=(np.arange(count,dtype=np.uint64)+1)*np.uint64(0x9e3779b9)
    z=(z+np.uint64(seed)+np.uint64((ordinal+1)*0x85ebca6b)) & np.uint64(0xffffffff)
    z^=z>>np.uint64(16);z=(z*np.uint64(0x7feb352d))&np.uint64(0xffffffff)
    z^=z>>np.uint64(15);z=(z*np.uint64(0x846ca68b))&np.uint64(0xffffffff)
    z^=z>>np.uint64(16)
    return z.astype(np.uint32)


def uniform(seed,count,ordinal,scale=1.0):
    return (((mix32(seed,count,ordinal)>>np.uint32(8)).astype(np.float64)/(1<<24)-.5)*scale).astype(np.float32)


def component_reference(program):
    validate_component(program)
    seed,s=program['seed'],program['shape']
    if program['kind']=='matmul':
        w=uniform(seed,s['k']*s['m'],0,1/math.sqrt(s['k'])).reshape(s['m'],s['k'])
        x=uniform(seed,s['k']*s['n'],1).reshape(s['n'],s['k'])
        # Bounded oracle blocks; independent double accumulation, not GGML.
        out=np.empty((s['n'],s['m']),dtype=np.float64)
        for begin in range(0,s['m'],256):out[:,begin:begin+256]=x.astype(np.float64)@w[begin:begin+256].astype(np.float64).T
        return dict(kind='float_output',values=out.ravel().tolist(),oracle='independent NumPy float64 GEMM blocks')
    if program['kind']=='attention_pv':
        h,kh,d,q,k=(s[n] for n in ('heads','kv_heads','head_dim','query_tokens','kv_tokens'))
        v=uniform(seed,d*k*kh,0).reshape(kh,k,d).astype(np.float64)
        p=((mix32(seed,k*q*h,1)>>np.uint32(8)).astype(np.float64)+1)/(1<<24)
        p=p.reshape(h,q,k);p=(p/p.sum(axis=-1,keepdims=True)).astype(np.float32).astype(np.float64)
        out=np.stack([p[i]@v[i//(h//kh)] for i in range(h)])
        return dict(kind='float_output',values=out.transpose(1,0,2).ravel().tolist(),oracle='independent NumPy grouped PV merged q/head/dim')
    content=(mix32(seed,s['bytes'],0)&np.uint32(255)).astype(np.uint8).tobytes()
    return dict(kind='byte_sha256',sha256=hashlib.sha256(content).hexdigest(),bytes=len(content),
                oracle='independent procedural bytes and hashlib.sha256')


def verify_float_output(expected,actual,*,atol=5e-5,rtol=5e-5):
    left,right=np.asarray(expected,dtype=np.float64),np.asarray(actual,dtype=np.float64)
    require(left.shape==right.shape and np.all(np.isfinite(right)),'native floating output shape/finiteness differs')
    require(np.allclose(left,right,atol=atol,rtol=rtol),'independent component numerical check failed')
    return dict(passed=True,maximum_absolute_error=float(np.max(np.abs(left-right))) if left.size else 0.,atol=atol,rtol=rtol)
