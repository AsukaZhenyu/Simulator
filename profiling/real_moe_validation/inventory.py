"""Read-only GGUF directory and conservative OLMoE profile coverage audit.

Tensor payloads are mapped by the pinned reader but never materialized here.
The directory establishes necessary weight geometry, not runtime graph/fusion
equivalence, actual compute placement or a prediction accuracy result.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib
from pathlib import Path
import sys

from .dataset import read, write


def inventory(manifest_path, gguf_python_dir):
    manifest=read(manifest_path);model=manifest['model'];path=Path(model['path'])
    stat=path.stat()
    if stat.st_size!=model['bytes'] or stat.st_mtime_ns!=model['mtime_ns']:
        raise ValueError('model file changed since full-hash collection binding')
    root=Path(gguf_python_dir).resolve();sys.path.insert(0,str(root))
    gguf=importlib.import_module('gguf')
    if not Path(gguf.__file__).resolve().is_relative_to(root):raise ValueError('GGUF reader is not from the explicitly selected source')
    reader=gguf.GGUFReader(path,'r');rows=[]
    for tensor in reader.tensors:
        row=dict(name=tensor.name,dtype=tensor.tensor_type.name,shape=[int(v) for v in tensor.shape],
            bytes=int(tensor.n_bytes),data_offset=int(tensor.data_offset))
        if row['data_offset']<reader.data_offset or row['data_offset']+row['bytes']>stat.st_size:
            raise ValueError('tensor payload is outside the pinned file')
        rows.append(row)
    return dict(schema='real-moe-gguf-directory-1',model=model,model_contract=manifest['model_contract'],
        reader_source=dict(path=str(root/'gguf/gguf_reader.py'),sha256=hashlib.sha256((root/'gguf/gguf_reader.py').read_bytes()).hexdigest()),
        tensor_count=len(rows),type_counts=dict(Counter(r['dtype'] for r in rows)),tensors=rows,
        tensor_payloads_materialized=False,model_inference_executed=False,
        full_hash_source='collection manifest; file size and mtime rechecked; parent rehashes before acquisition')


def audit(directory):
    contract=directory['model_contract']
    if contract!={'architecture':'olmoe','layers':16,'experts':64,'top_k':8}:
        raise ValueError('this source audit only covers the explicitly checked OLMoE graph')
    tensors={r['name']:r for r in directory['tensors']};layers=[]
    for layer in range(contract['layers']):
        prefix=f'blk.{layer}.'
        weights={name:tensors[prefix+name+'.weight'] for name in
            ('ffn_gate_exps','ffn_up_exps','ffn_down_exps','ffn_gate_inp','attn_q','attn_k','attn_v','attn_output')}
        expected={'ffn_gate_exps':[2048,1024,64],'ffn_up_exps':[2048,1024,64],
            'ffn_down_exps':[1024,2048,64],'ffn_gate_inp':[2048,64]}
        for name,shape in expected.items():
            allowed={'F32'} if name=='ffn_gate_inp' else {'Q4_K','Q6_K'}
            if weights[name]['dtype'] not in allowed or weights[name]['shape']!=shape:raise ValueError('unsupported actual quantized expert layout')
        layers.append(dict(layer=layer,weights=weights,expert_weight_bytes=sum(weights[n]['bytes'] for n in expected if n!='ffn_gate_inp')))
    gaps=[
        dict(family='expert_ffn',actual='layer-specific Q4_K/Q6_K MUL_MAT_ID, fused SWIGLU, weighted views and seven ordered ADDs',
             existing='legacy CPU F32 per-expert grouped MUL_MAT',qualified=False,reason='different dtypes, dispatch topology and materialization'),
        dict(family='regular_projection_and_lm_head',actual='mixed Q4_K/Q6_K MUL_MAT with real output-row pruning',
             existing='isolated Q4_K/Q6_K MUL_MAT probe',qualified=False,reason='old probe forces CUDA graphs off; no matching whole-runtime boundary'),
        dict(family='attention_and_kv',actual='Q/K RMSNorm, model RoPE, padded F16 KV and backend-selected graph',
             existing='Dense partial/complete probes with declared F32 KV boundaries',qualified=False,reason='KV dtype/layout, rope and scheduler cuts must match'),
        dict(family='router_and_combine',actual='real F32 router, SOFTMAX/ARGSORT/GET_ROWS and ordered aggregation',
             existing='prescribed-route F32 probe',qualified=False,reason='probe explicitly does not reproduce the original model router/kernel'),
        dict(family='embedding_and_normalization',actual='quantized GET_ROWS and model norm tensors',
             existing='no complete real-model bound stage profile',qualified=False,reason='missing exact measured boundary'),
        dict(family='runtime_and_backend_transitions',actual='llama decode scheduling, sampling, GPU/CPU transfers and joins',
             existing='steady-action inclusive response composition',qualified=False,reason='complete action boundaries differ; omitted overlap/transition terms need heldout testing'),
    ]
    return dict(schema='real-moe-profile-coverage-audit-1',directory_model_sha256=directory['model']['sha256'],layers=layers,
        checked_model_layers=16,families=gaps,qualified_existing_families=0,total_required_families=len(gaps),
        graph_source='llama.cpp-b8705/src/models/olmoe.cpp and llama-graph.cpp build_moe_ffn',
        current_end_to_end_prediction_supported=False,
        minimal_new_probe='real packed weights, CPU homogeneous post-FFN-norm -> MoE output, exact vectorized MUL_MAT_ID topology',
        hybrid_ffn_supported=False,hybrid_reason='actual GPU-router/CPU-expert/backend copy split must be observed before assigning stage prices',
        conclusions_are_static_source_audit=True,hardware_accuracy_certified=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--gguf-python-dir',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();directory=inventory(args.manifest,args.gguf_python_dir)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    write(out/'directory.json',directory);coverage=audit(directory);write(out/'coverage.json',coverage)
    print(__import__('json').dumps(dict(out=str(out),tensor_count=directory['tensor_count'],type_counts=directory['type_counts'],qualified_existing_families=0)))


if __name__=='__main__':main()
