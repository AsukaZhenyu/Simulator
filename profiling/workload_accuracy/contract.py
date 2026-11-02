"""Bounded traced execution and evidence contracts, independent of native timing."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
from pathlib import Path

QUALITY = {'max_within_process_cv': .15, 'max_between_process_cv': .20,
           'max_heldout_relative_error': .20}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def integer(value, label, low=0, high=2**63-1):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{label}: expected integer {low}..{high}')
    return value


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(label + ': expected finite number')
    return value


def same(actual, expected, label):
    if canonical(actual) != canonical(expected):
        raise ValueError(label + ' differs from declared evidence')


def validate_case(raw):
    case = copy.deepcopy(raw)
    required = {'schema','id','backend','cuda_device','threads','seed','hidden','intermediate',
                'heads','kv_heads','head_dim','rope_dims','rope_base','norm_eps','attention_gate',
                'qk_norm','max_context','layers','rounds','tracked_budget_bytes'}
    if not isinstance(case, dict) or set(case) != required:
        raise ValueError('case must declare exactly ' + str(sorted(required)))
    same(case['schema'], 'workload_accuracy_case/1', 'case schema')
    if not isinstance(case['id'], str) or not case['id']:
        raise ValueError('case id required')
    if case['backend'] not in ('CPU','CUDA'):
        raise ValueError('CPU or CUDA required; no fallback')
    for key, low, high in (('cuda_device',0,15),('threads',1,64),('seed',0,1000000),
                           ('hidden',2,256),('intermediate',1,512),('heads',1,32),('kv_heads',1,32),
                           ('head_dim',2,256),('max_context',1,4096),('tracked_budget_bytes',1,512*1024**2)):
        integer(case[key],key,low,high)
    h,d,heads,kv=case['hidden'],case['head_dim'],case['heads'],case['kv_heads']
    if h != d*heads or heads % kv:
        raise ValueError('invalid head geometry')
    integer(case['rope_dims'],'rope_dims',2,d)
    if case['rope_dims'] % 2:
        raise ValueError('rope_dims must be even')
    if finite(case['rope_base'],'rope_base') <= 1 or finite(case['norm_eps'],'norm_eps') <= 0:
        raise ValueError('invalid RoPE/norm parameters')
    if case['attention_gate'] not in ('none','sigmoid') or case['qk_norm'] not in ('none','rms'):
        raise ValueError('unsupported attention variant')
    if not isinstance(case['layers'],list) or not 2 <= len(case['layers']) <= 8:
        raise ValueError('requires 2..8 actual layers')
    for layer in case['layers']:
        if layer == {'kind':'dense'}:
            continue
        if not isinstance(layer,dict) or set(layer) != {'kind','experts','top_k','residency'} or layer['kind'] != 'moe':
            raise ValueError('invalid Dense/MoE layer')
        integer(layer['experts'],'experts',1,32)
        integer(layer['top_k'],'top_k',1,layer['experts'])
        if layer['residency'] not in ('all','active'):
            raise ValueError('expert residency must be all or active')
    if not isinstance(case['rounds'],list) or not 2 <= len(case['rounds']) <= 32:
        raise ValueError('requires 2..32 actual rounds')
    history=0
    for rd in case['rounds']:
        if not isinstance(rd,dict) or set(rd) != {'hidden_input','routes','route_scores'}:
            raise ValueError('invalid round fields')
        x=rd['hidden_input']
        if not isinstance(x,list) or not 1 <= len(x) <= 32:
            raise ValueError('requires 1..32 tokens per round')
        history += len(x)
        if history > case['max_context']:
            raise ValueError('trace exceeds KV capacity')
        for row in x:
            if not isinstance(row,list) or len(row)!=h:
                raise ValueError('hidden input shape')
            for v in row:
                finite(v,'hidden input')
        for key in ('routes','route_scores'):
            if not isinstance(rd[key],list) or len(rd[key]) != len(case['layers']):
                raise ValueError('route arrays must cover every layer')
        for li,layer in enumerate(case['layers']):
            ids,scores=rd['routes'][li],rd['route_scores'][li]
            if layer['kind']=='dense':
                if ids is not None or scores is not None:
                    raise ValueError('dense routes must be null')
                continue
            if not isinstance(ids,list) or not isinstance(scores,list) or len(ids)!=len(x) or len(scores)!=len(x):
                raise ValueError('routes must exactly cover current tokens')
            for route,coeff in zip(ids,scores):
                if len(route)!=layer['top_k'] or len(coeff)!=len(route):
                    raise ValueError('top-k route shape')
                for e in route:
                    integer(e,'expert',0,layer['experts']-1)
                if len(set(route))!=len(route):
                    raise ValueError('duplicate expert')
                if any(finite(v,'score')<0 for v in coeff) or not math.isclose(sum(coeff),1,abs_tol=1e-5):
                    raise ValueError('nonnegative selected scores must sum to one')
    payload=h*h*len(case['layers'])*4+h*case['intermediate']*32*len(case['layers'])*12+case['max_context']*kv*d*8*len(case['layers'])
    if payload >= 64*1024**2:
        raise ValueError('bounded probe payload limit 64 MiB')
    return case


def dense_config(case):
    """Exact immutable dense_contract geometry for the native stage boundary."""
    from llm_infer_model.m2.dense_contract import validate_config, weight_shapes
    config={key:case[key] for key in ('hidden','intermediate','heads','kv_heads','head_dim',
                                     'attention_gate','qk_norm','rope_dims','rope_base')}
    config.update(norm_epsilon=case['norm_eps'],rope_mode='normal')
    config['weight_dtypes']={name:'f32' for name in weight_shapes(config)}
    return validate_config(config)


def stage_keys(case):
    """Exact semantic/shape/history/device keys. Never include observed time."""
    case=validate_case(case)
    keys=[];history=0
    common={k:case[k] for k in ('backend','cuda_device','threads','hidden','intermediate','heads',
                               'kv_heads','head_dim','rope_dims','rope_base','norm_eps','attention_gate','qk_norm')}
    for ri,rd in enumerate(case['rounds']):
        n=len(rd['hidden_input'])
        for li,layer in enumerate(case['layers']):
            kinds=['projection','attention','dense_ffn' if layer['kind']=='dense' else 'moe_ffn_trace']
            counts={}
            if layer['kind']=='moe':
                for route in rd['routes'][li]:
                    for e in route:
                        counts[e]=counts.get(e,0)+1
            for kind in kinds:
                semantics=dict(common,kind=kind,tokens=n,history=history,
                               active_expert_row_counts=sorted(counts.values()) if kind=='moe_ffn_trace' else [])
                keys.append({'round':ri,'layer':li,'kind':kind,'key':digest(semantics),'semantics':semantics})
        history+=n
    return keys


def workload_key(case):
    """Host-gap scope includes topology, capacity, routes and residency churn."""
    item=copy.deepcopy(validate_case(case))
    for name in ('id','seed','tracked_budget_bytes'):
        item.pop(name)
    # Equal-shape independent hidden input/weights may transfer. Routes remain in
    # this effective-overhead key because allocation churn is route dependent.
    for rd in item['rounds']:
        rd['hidden_input']={'tokens':len(rd['hidden_input'])}
        rd.pop('route_scores')
    return digest(item)


def replay_memory(memory):
    same(memory.get('physical_capacity_verified'),False,'physical capacity claim')
    same(memory.get('process_peak_bytes'),None,'process physical peak')
    same(memory.get('cuda_private_workspace_bytes'),None,'unknown CUDA workspace')
    same(memory.get('cuda_private_workspace_bound_known'),False,'unknown CUDA workspace bound')
    current=peak=0;live={};budget=integer(memory.get('budget_bytes'),'budget',1)
    components={}
    for event in memory.get('events',[]):
        ident=event.get('id');action=event.get('event')
        requested=integer(event.get('requested_bytes'),'requested')
        reserved=integer(event.get('reserved_bytes'),'reserved',requested)
        if action=='allocate':
            if ident in live:
                raise ValueError('duplicate live allocation')
            live[ident]=(requested,reserved,event['component']);current+=reserved
            components[event['component']]=max(components.get(event['component'],0),reserved)
        elif action=='release':
            same(live.pop(ident,None),(requested,reserved,event['component']),'released allocation')
            current-=reserved
        else:
            raise ValueError('unknown memory lifetime event')
        same(event.get('live_reserved_bytes'),current,'allocation lifetime total')
        if current > budget:
            raise ValueError('tracked budget exceeded')
        peak=max(peak,current)
    if live or current:
        raise ValueError('unreleased allocations')
    same(memory.get('peak_tracked_reserved_bytes'),peak,'actual tracked peak')
    same(memory.get('final_tracked_reserved_bytes'),0,'final memory')
    same(memory.get('tracked_budget_pass'),True,'tracked budget acceptance')
    return {'tracked_peak_bytes':peak,'components_max_single_allocation_bytes':components,
            'physical_capacity_verified':False,'unknown_private_workspace_bytes':None}


def validate_route_event_order(case,layer_index,events):
    """Assert trace/router barrier ordering independently of residency snapshots."""
    previous=-1;boundaries={}
    allowed={'projection_complete','attention_complete','route_known','load','unload',
             'route_visible','expert_ffn_begin','expert_ffn_complete'}
    for event in events:
        sequence=integer(event.get('sequence'),'causal event sequence')
        if sequence<=previous:
            raise ValueError('causal event sequence must strictly increase')
        previous=sequence
        if event.get('event') not in allowed:
            raise ValueError('unknown causal route event')
    for ri,rd in enumerate(case['rounds']):
        current=[event for event in events if event.get('round')==ri]
        kinds=[event['event'] for event in current]
        fixed=['projection_complete','attention_complete','route_known','route_visible','expert_ffn_begin','expert_ffn_complete']
        if [kind for kind in kinds if kind not in ('load','unload')]!=fixed:
            raise ValueError('route_known must follow attention and precede residency/expert work')
        by_kind={event['event']:event for event in current if event['event'] not in ('load','unload')}
        known=by_kind['route_known'];visible=by_kind['route_visible'];begin=by_kind['expert_ffn_begin']
        for event in current:
            if event['event'] in ('load','unload') and not known['sequence']<event['sequence']<visible['sequence']:
                raise ValueError('route-based load/evict occurred before route_known or after residency snapshot')
        same(known.get('routes'),rd['routes'][layer_index],'current IDs revealed at trace barrier')
        same(known.get('selected_scores'),rd['route_scores'][layer_index],'current scores revealed at trace barrier')
        same(known.get('learned_router_executed'),False,'trace barrier is not a learned router')
        boundaries[ri]=(known['sequence'],begin['sequence'])
    if len(events)!=sum(sum(event.get('round')==ri for event in events) for ri in range(len(case['rounds']))):
        raise ValueError('causal events outside declared rounds')
    return boundaries


def validate_residency(case,payload):
    allocations={e['id']:e for e in payload['memory']['events'] if e['event']=='allocate'}
    releases={e['id']:e for e in payload['memory']['events'] if e['event']=='release'}
    same(len(payload.get('residency',[])),len(case['layers']),'layer lifetime coverage')
    final_history=sum(len(rd['hidden_input']) for rd in case['rounds'])
    for li,layer in enumerate(case['layers']):
        events=payload['residency'][li]
        same(events.get('layer'),li,'residency layer')
        same(events.get('final_history'),final_history,'final persistent history')
        for component in ('resident_weights','kv_capacity')+(() if layer['kind']=='moe' else ('resident_dense_ffn',)):
            matches=[e for e in allocations.values() if e['component']==component and e['layer']==li]
            same(len(matches),1,'one persistent layer allocation '+component)
            allocation=matches[0]
            same(allocation['round'],-1,'weight/KV allocation before rounds')
            same(releases[allocation['id']]['round'],len(case['rounds']),'persistent lifetime through all rounds')
        if layer['kind']=='dense':
            same(events.get('events'),[],'dense has no expert routing')
            continue
        boundaries=validate_route_event_order(case,li,events['events'])
        snapshots=[e for e in events['events'] if e['event']=='route_visible']
        same(len(snapshots),len(case['rounds']),'one route observation per current round')
        previous={}
        for ri,(snapshot,rd) in enumerate(zip(snapshots,case['rounds'])):
            active=sorted({e for route in rd['routes'][li] for e in route})
            expected=active if layer['residency']=='active' else list(range(layer['experts']))
            same(snapshot.get('round'),ri,'current route round')
            same(snapshot.get('information_scope'),'current_layer_round_only','route information boundary')
            same(snapshot.get('active_experts'),active,'routes visible now')
            same(snapshot.get('resident_experts'),expected,'actual expert residency')
            ids=snapshot.get('allocation_ids',{})
            same(sorted(ids),sorted(str(e) for e in expected),'resident allocation identities')
            for expert,ident in ids.items():
                alloc=allocations.get(ident)
                if alloc is None:
                    raise ValueError('resident expert lacks actual allocation')
                same(alloc['layer'],li,'resident allocation layer')
                same(alloc['component'],'resident_expert.'+expert,'resident expert allocation')
                same(alloc['requested_bytes'],12*case['hidden']*case['intermediate'],'expert logical weight bytes')
                if alloc['round']>ri or releases[ident]['round']<ri:
                    raise ValueError('resident expert outside allocation lifetime')
                if layer['residency']=='active' and alloc['round']==ri:
                    known,begin=boundaries[ri]
                    if not known<integer(alloc.get('sequence'),'expert allocation causal sequence')<begin:
                        raise ValueError('actual expert allocation occurred before route_known or after expert FFN began')
                if expert in previous:
                    same(ident,previous[expert],'reuse continuing resident weight allocation')
            previous=ids
        if layer['residency']=='active':
            for alloc in allocations.values():
                if alloc['layer']==li and alloc['component'].startswith('resident_expert.'):
                    released=releases[alloc['id']]
                    if released['round']<len(case['rounds']):
                        known,begin=boundaries[released['round']]
                        if not known<integer(released.get('sequence'),'expert release sequence')<begin:
                            raise ValueError('actual expert eviction before route_known or after expert FFN began')


def validate_native(case, payload, *, samples, warmup):
    case=validate_case(case)
    for key,value in {'schema':'ggml_workload_accuracy/2','case_id':case['id'],'backend':case['backend'],
                      'cuda_device':case['cuda_device'],'hardware_compute_performed':True,'numeric_pass':True,
                      'dtype':'f32','samples':samples,'warmup':warmup,'each_sample_resets_workload_state':True,
                      'execution':'actual_input_trace_multilayer_persistent_kv',
                      'causal_route_boundary':'trace_known_after_attention_before_route_based_residency_and_ffn',
                      'learned_router_executed':False}.items():
        same(payload.get(key),value,'native '+key)
    same(payload.get('precision'),{'matmul':'GGML_PREC_F32',
                                  'NVIDIA_TF32_OVERRIDE':'0' if case['backend']=='CUDA' else None,
                                  'GGML_CUDA_DISABLE_GRAPHS':'1' if case['backend']=='CUDA' else None},'native precision policy')
    rows=payload.get('rows',[])
    if len(rows)!=len(case['rounds'])*len(case['layers']):
        raise ValueError('complete layer/round coverage required')
    history=0;index=0
    for ri,rd in enumerate(case['rounds']):
        n=len(rd['hidden_input']);previous=[struct.unpack('f',struct.pack('f',v))[0] for row in rd['hidden_input'] for v in row]
        for li,layer in enumerate(case['layers']):
            row=rows[index];index+=1
            for key,value in {'round':ri,'layer':li,'tokens':n,'kind':layer['kind'],'history_before':history,
                              'history_after':history+n,'kv_used_bytes':(history+n)*case['kv_heads']*case['head_dim']*8,
                              'kv_capacity_bytes':case['max_context']*case['kv_heads']*case['head_dim']*8}.items():
                same(row.get(key),value,'row '+key)
            same(row.get('hidden_input'),previous,'real layer output feeding next layer')
            expected_ports={'q','k_new','v_new','persistent_k','persistent_v','attention_output','hidden_output'}
            if case['attention_gate']=='sigmoid':
                expected_ports.add('gate_proj')
            checks=row.get('checks',[])
            if len(checks)!=len(expected_ports) or {x.get('port') for x in checks}!=expected_ports:
                raise ValueError('complete numerical checks required')
            port_sizes={'q':case['hidden']*n,'k_new':case['kv_heads']*case['head_dim']*n,
                        'v_new':case['kv_heads']*case['head_dim']*n,
                        'persistent_k':case['kv_heads']*case['head_dim']*(history+n),
                        'persistent_v':case['kv_heads']*case['head_dim']*(history+n),
                        'attention_output':case['hidden']*n,'hidden_output':case['hidden']*n,
                        'gate_proj':case['hidden']*n}
            for check in checks:
                same(check.get('passed'),True,'numeric stage acceptance')
                same(check.get('elements'),port_sizes[check['port']],'numeric port shape')
                finite(check.get('max_abs_error'),'numeric error')
            previous=row.get('hidden_output')
            if not isinstance(previous,list) or len(previous)!=case['hidden']*n:
                raise ValueError('output boundary shape')
            reference=row.get('reference_output')
            if not isinstance(reference,list) or len(reference)!=len(previous):
                raise ValueError('independent reference shape')
            for actual,expected in zip(previous,reference):
                if abs(finite(actual,'output')-finite(expected,'reference'))>2e-4+2e-3*abs(expected):
                    raise ValueError('independent numerical mismatch')
        history+=n
    same(len(payload.get('sample_timings',[])),samples,'sample coverage')
    keys=stage_keys(case)
    for sample in payload['sample_timings']:
        same(len(sample.get('stages',[])),len(keys),'stage timing coverage')
        for stage,expected in zip(sample['stages'],keys):
            for key in ('round','layer','kind'):
                same(stage.get(key),expected[key],'timed stage boundary')
            integer(stage.get('compute_ns'),'stage time',1)
        timing=sample['timing'];wall=integer(timing['full_workload_wall_ns'],'workload wall',1)
        compute=sum(x['compute_ns'] for x in sample['stages'])
        same(timing['stage_compute_sum_ns'],compute,'stage timing sum')
        same(timing['host_setup_materialization_gap_ns'],wall-compute,'host gap')
        same(timing.get('scope'),'inference_graph_setup_alloc_materialization_weight_lifecycle_release','inference scope')
        for flag in ('scalar_reference_in_timed_region','port_serialization_in_timed_region',
                     'memory_event_logging_in_timed_region','source_weight_generation_in_timed_region'):
            same(timing.get(flag),False,'timing excludes '+flag)
        output_checks=sample.get('output_checks_after_timing',[])
        same(len(output_checks),len(case['rounds']),'timed complete round output coverage')
        for check in output_checks:
            same(check.get('passed'),True,'timed output checked after measurement')
        same(sample.get('peak_tracked_reserved_bytes'),payload['memory']['peak_tracked_reserved_bytes'],'timed/validation allocation topology')
        if wall<compute:
            raise ValueError('wall excludes stage interval')
    memory=replay_memory(payload['memory'])
    validate_residency(case,payload)
    return memory
