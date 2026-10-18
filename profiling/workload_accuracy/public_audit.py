"""Bind the same trace to public Model/Workload/ModelPlan shape/state semantics.

Analytical costs here are compiler inputs only. They never fit or score native
times. MoE's existing opaque synthetic router boundary is explicitly different
from a trace-selected physical SwiGLU evaluator; an exact profile bind is refused.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import copy
import shutil
import sys

from .contract import dense_config, digest, same, sha, validate_case


def add_repository_paths():
    root=Path(__file__).resolve().parents[2]
    for sub in ('src','modeling','mapping'):
        value=str(root/sub)
        if value not in sys.path:
            sys.path.insert(0,value)
    return root


def export_audit(case,directory,*,native=None):
    case=validate_case(case)
    add_repository_paths()
    from llm_simulator import (AnalyticalCosts, Device, Hardware, Model, Route,
                               Scheduling, Simulator, Workload)
    from llm_infer_model.m2.costs import parse_cost_table
    from llm_infer_model.m2.expand import compile_invocation
    from llm_infer_model.m2.spec import load_invocation, load_layer
    kinds={x['kind'] for x in case['layers']}
    if len(kinds)!=1:
        return _export_mixed_audit(case,directory,native=native)
    if any(len(rd['hidden_input'])!=1 for rd in case['rounds'][1:]):
        raise ValueError('public Workload facade supports one token per post-prefill round')
    if case['attention_gate']!='none' or case['qk_norm']!='none' or case['rope_dims']!=case['head_dim'] or case['rope_base']!=10000 or case['norm_eps']!=1e-6 or case['threads']!=1:
        raise ValueError('case differs from explicit public Model attention contract; no silent fallback')
    first=case['layers'][0]
    if any(layer!=first for layer in case['layers']):
        raise ValueError('public Model facade requires identical per-layer expert declarations')
    params=dict(layers=len(case['layers']),hidden_size=case['hidden'],intermediate_size=case['intermediate'],
                heads=case['heads'],kv_heads=case['kv_heads'])
    model=Model.dense(**params) if first['kind']=='dense' else Model.moe(experts=first['experts'],top_k=first['top_k'],**params)
    routes=[];position=0
    for rd in case['rounds']:
        for li,layer in enumerate(case['layers']):
            if layer['kind']=='moe':
                for t,(ids,scores) in enumerate(zip(rd['routes'][li],rd['route_scores'][li])):
                    routes.append(Route('request0',f'layer{li}',position+t,tuple(ids),tuple(scores)))
        position+=len(rd['hidden_input'])
    workload=Workload(prefill_tokens=len(case['rounds'][0]['hidden_input']),decode_steps=len(case['rounds'])-1,
                      routes=tuple(routes),route_source='same workload_accuracy_case/1 '+digest(case) if routes else None)
    # Rates are deliberately synthetic; only compiler structure/state is audited.
    device=Device(1e9,1e9)
    hardware=Hardware(cpu=device if case['backend']=='CPU' else None,
                      gpu=device if case['backend']=='CUDA' else None,
                      dram_capacity_bytes=128*1024**2,vram_capacity_bytes=128*1024**2,
                      transfer_bandwidth_bytes_per_second=1e9,ssd_read_bandwidth_bytes_per_second=1e9,
                      ssd_write_bandwidth_bytes_per_second=1e9)
    simulator=Simulator(model=model,hardware=hardware,workload=workload,cost_model=AnalyticalCosts())
    directory=Path(directory).resolve()
    scheduling=Scheduling(default_device='cpu' if case['backend']=='CPU' else 'gpu')
    with simulator.compile(scheduling=scheduling,directory=directory) as compiled:
        ir=compiled.ir_document()
        checks=[];history=0
        for ri,rd in enumerate(case['rounds']):
            round_id='prefill' if ri==0 else f'decode{ri-1}'
            n=len(rd['hidden_input'])
            for li,layer in enumerate(case['layers']):
                scope=f'layer{li}-{round_id}'
                desc=load_layer(directory/f'layer{li}.layer.json')
                invocation=load_invocation(directory/(scope+'.json'))
                costs=parse_cost_table(json.loads((directory/(scope+'.costs.json')).read_text(encoding='utf-8')))
                expanded=compile_invocation(desc,invocation,costs)
                request=invocation.requests[0]
                same(request.position_start,history,'public absolute position')
                same(request.tokens,n,'public current tokens')
                old=invocation.state_in.requests[0].blocks
                new=expanded.new_blocks['request0']
                same(len(old),history,'public persistent KV block count')
                same(sum(b.storage_bytes for b in old),history*model.kv_bytes_per_token,'public KV history bytes')
                same(sum(b.storage_bytes for b in new),n*model.kv_bytes_per_token,'public new KV bytes')
                same([b.first_position for b in new],list(range(history,history+n)),'public appended positions')
                tensors={t['id']:t for t in expanded.workload_document['tensors']}
                same(tensors[expanded.hidden_out_ids['request0']]['storage_bytes'],n*model.hidden_bytes_per_token,'public hidden output boundary bytes')
                task_kinds=[task.kind for task in expanded.tasks]
                if layer['kind']=='dense':
                    same(task_kinds,['projection','attention','dense_ffn'],'dense stage topology')
                    same(desc.physical.config,dense_config(case),'complete Dense numerical configuration')
                    weight_bytes=sum(w.storage_bytes for w in desc.weights)
                    expected=4*(2*case['hidden']+2*case['hidden']**2+2*case['hidden']*case['kv_heads']*case['head_dim']+3*case['hidden']*case['intermediate'])
                    same(weight_bytes,expected,'persistent Dense F32 logical weight bytes')
                else:
                    if not all(kind in task_kinds for kind in ('projection','attention','router','expert_ffn','combine')):
                        raise ValueError('public opaque MoE stage topology incomplete')
                    same([tuple(row.expert_ids) for row in invocation.route_trace],
                         [tuple(ids) for ids in rd['routes'][li]],'public current trace route ids')
                    weight_bytes=sum(w.storage_bytes for w in desc.weights)
                check=dict(round=ri,layer=li,tokens=n,history_before=history,history_after=history+n,
                           kv_used_bytes=(history+n)*model.kv_bytes_per_token,
                           native_kv_capacity_bytes=case['max_context']*model.kv_bytes_per_token,
                           modeled_kv_allocation='logical_used_blocks',native_kv_allocation='fixed_capacity_buffer',
                           hidden_output_bytes=n*model.hidden_bytes_per_token,task_kinds=task_kinds,
                           declared_logical_weight_bytes=weight_bytes)
                if native is not None:
                    row=native['rows'][ri*len(case['layers'])+li]
                    for name in ('tokens','history_before','history_after','kv_used_bytes'):
                        same(row[name],check[name],'native/public '+name)
                checks.append(check)
            history+=n
        result=compiled.run()
        same(result.success,True,'public ModelPlan runtime completion')
    files={p.name:sha(p) for p in sorted(directory.glob('*.json'))}
    report=dict(schema='workload_public_model_audit/1',case_sha256=digest(case),
                model=asdict(model),workload={**asdict(workload),'routes':[asdict(r) for r in routes]},
                real_public_ModelPlan_compiled=True,real_public_ModelPlan_runtime_status=result.status,
                real_public_ModelPlan_runtime_success=result.success,
                modelplan_ir=ir,round_layer_checks=checks,source_file_sha256=files,
                shape_position_route_output_boundary_match=True,
                dense_numerical_operation_semantics_match=first['kind']=='dense',
                exact_native_physical_profile_bind_allowed=False,
                memory_capacity_equivalent=False,
                exclusions=['Analytical timing is synthetic and is not compared with native timing.',
                            'Native fixed KV capacity, graph buffers, metadata and real expert allocations differ from logical Modeling blocks.',
                            'Native per-head GGML graphs and host materialization differ from frozen dense probe copies; old macro costs cannot be reused.',
                            'Current public MoE topology has an opaque packed projection and router; native replays selected trace scores with an explicit physical decoder FFN.',
                            'Facade scheduling does not claim to reproduce native expert cache lifetimes.'])
    (directory/'workload-audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


def _export_mixed_audit(case,directory,*,native=None):
    """Compose public-generated layer inputs through real heterogeneous ModelPlan.

    The homogeneous convenience Model is not silently extended. Its ordinary
    per-layer M2 inputs are combined through the existing lower-level API.
    """
    from llm_simulator import ModelPlan, Scheduling
    if any(len(rd['hidden_input'])!=1 for rd in case['rounds']):
        raise ValueError('existing mixed Dense/opaque-MoE ModelPlan hidden layouts match only for single-token rounds')
    root=Path(directory).resolve()
    root.mkdir(parents=True,exist_ok=False)
    templates={};reports={}
    moe_specs=[layer for layer in case['layers'] if layer['kind']=='moe']
    first=moe_specs[0]
    if any((layer['experts'],layer['top_k'])!=(first['experts'],first['top_k']) for layer in moe_specs):
        raise ValueError('bounded mixed public export requires common expert geometry')
    for kind in ('dense','moe'):
        source=copy.deepcopy(case)
        source['layers']=[{'kind':'dense'} if kind=='dense' else copy.deepcopy(first) for _ in case['layers']]
        for rd in source['rounds']:
            for li in range(len(source['layers'])):
                if kind=='dense':
                    rd['routes'][li]=None;rd['route_scores'][li]=None
                elif rd['routes'][li] is None:
                    k=first['top_k'];n=len(rd['hidden_input'])
                    rd['routes'][li]=[list(range(k)) for _ in range(n)]
                    rd['route_scores'][li]=[[1/k]*k for _ in range(n)]
        templates[kind]=root/(kind+'-public-inputs')
        reports[kind]=export_audit(source,templates[kind])
    composed=root/'composed-model';composed.mkdir()
    for filename in ('model.global.json','environment.json'):
        shutil.copyfile(templates['dense']/filename,composed/filename)
    checks=[]
    for ri in range(len(case['rounds'])):
        round_id='prefill' if ri==0 else f'decode{ri-1}'
        for li,layer in enumerate(case['layers']):
            kind=layer['kind'];source=templates[kind]
            if ri==0:
                shutil.copyfile(source/f'layer{li}.layer.json',composed/f'layer{li}.layer.json')
            scope=f'layer{li}-{round_id}'
            for suffix in ('.json','.costs.json'):
                shutil.copyfile(source/(scope+suffix),composed/(scope+suffix))
            check=reports[kind]['round_layer_checks'][ri*len(case['layers'])+li]
            checks.append(check)
            if native is not None:
                row=native['rows'][ri*len(case['layers'])+li]
                for key in ('tokens','history_before','history_after','kv_used_bytes'):
                    same(row[key],check[key],'mixed native/public '+key)
    authored=ModelPlan(id='mixed-workload-audit',model_kind='global_m2',
                       model_file=composed/'model.global.json',environment_file=composed/'environment.json',
                       base_dir=composed,scheduling=Scheduling(default_device='cpu' if case['backend']=='CPU' else 'gpu'))
    compiled=authored.compile();result=compiled.run()[0]
    same(result.evaluation.status,'valid','heterogeneous public ModelPlan status')
    same(result.termination_reason,'goal_reached','heterogeneous public ModelPlan completed work')
    report=dict(schema='workload_public_model_audit/1',case_sha256=digest(case),
                model={'kind':'mixed','layers':case['layers']},
                convenience_models=[reports[kind]['model'] for kind in ('dense','moe')],
                composition='public Model generated per-layer inputs combined by existing global_m2 ModelPlan',
                real_public_ModelPlan_compiled=True,real_public_ModelPlan_runtime_status='valid',
                real_public_ModelPlan_runtime_success=True,modelplan_ir=compiled.ir_document(),
                round_layer_checks=checks,source_file_sha256={str(p.relative_to(root)):sha(p) for p in composed.glob('*.json')},
                shape_position_route_output_boundary_match=True,dense_numerical_operation_semantics_match=True,
                exact_native_physical_profile_bind_allowed=False,memory_capacity_equivalent=False,
                exclusions=reports['moe']['exclusions'])
    (root/'workload-audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report
