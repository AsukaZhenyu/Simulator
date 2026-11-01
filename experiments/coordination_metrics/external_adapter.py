"""Independent, generic static-obligation adapter for retained external interfaces.

Consumes only common mathematical workload ports and high-level intent. Does
not consume our compiler IR/actions, predictions, or physical replay ledger.
It deliberately does not implement a serving scheduler or a causal MoE router.
Consequently its automatic static contracts are not whole-tool policy results.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
import math
from external_interfaces import InterfaceBundle

CATEGORIES = ('allocation_alias', 'copy_payload', 'reader_lifetime', 'preparation_dependency')

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def tensor_bytes(tensor):
    if tensor['dtype'] not in ('GGML_TYPE_F32','GGML_TYPE_I32'):
        raise ValueError('Unimplemented dtype contract')
    return 4*math.prod(tensor['ne'])

def public_condition(condition):
    """Drop model-private future routing before the policy-side adapter call."""
    w = {k:v for k,v in condition['workload'].items() if k not in ('future','seed','causal_mode')}
    p = {k:v for k,v in condition['intent'].items() if k != 'future_expert'}
    return dict(id=condition['id'],pair=condition['pair'],family=condition['family'],
                split=condition['split'],variant=condition['variant'],change=condition['change'],
                workload=w,intent=p,geometry=condition['geometry'],hardware=condition['hardware'],
                cost_contract=condition['cost_contract'])

def placement(c, op, context):
    p=c['intent'];kind=op['semantic_op'];identity=op['id']
    if kind == 'state_update': return 'cpu'
    if c['family'] == 'PDhandoff':
        return p['prefill_device'] if context['step']==0 else p['decode_device']
    if kind.startswith('ffn_'):
        if '.host.' in identity:return 'cpu'
        return 'gpu'
    if kind in ('attention_qk','attention_softmax'):return p['attention_front_device']
    if kind.startswith('attention_'):return 'gpu'
    if kind=='projection':return p['projection_device']
    if kind=='attention':return p['attention_device']
    if kind in ('expert_ffn','shared_expert_ffn'):return p['expert_device']
    return p['default_device']

def derive_allocation_contract(c, dag, devices):
    tensors=dag['tensors'];views=[];potential={}
    for tid,t in tensors.items():
        n=tensor_bytes(t)
        if n != t['storage_bytes']:raise ValueError('Common mathematical tensor size mismatch')
        for domain in ('dram','vram'):
            aid=f'buffer:{tid}@{domain}'
            potential[aid]=dict(id=aid,tensor_id=tid,domain=domain,bytes=n,
                               initially_present=domain in t.get('initial_locations',[]))
    for oid,op in dag['operations'].items():
        device=devices[oid];domain='vram' if device=='gpu' else 'dram'
        for tid in op['inputs']:
            t=tensors[tid]
            record=dict(id=f'view:{oid}:{tid}',consumer=oid,tensor_id=tid,
                        allocation_id=f'buffer:{tid}@{domain}',domain=domain,
                        physical_bytes=tensor_bytes(t),logical_bytes=tensor_bytes(t),
                        relation='whole_value')
            if (t['role']=='weight' and op['semantic_op'] in ('ffn_gate','ffn_up','ffn_down')
                    and len(op.get('op_params',[]))==2):
                start,end=op['op_params'];h=c['geometry']['hidden']
                if not 0<=start<end<=c['geometry']['intermediate']:raise ValueError('Invalid FFN bounds')
                record.update(slice_bounds=[start,end],logical_bytes=4*h*(end-start))
                if c['intent']['ffn_storage']=='shared_parent_slice_views':
                    record['relation']='shared_parent_slice'
                    if tensor_bytes(t)!=4*h*c['geometry']['intermediate']:
                        raise ValueError('View contract has no full parent')
                else:
                    record['relation']='materialized_shard'
                    if tensor_bytes(t)!=record['logical_bytes']:raise ValueError('Materialized shard size mismatch')
            views.append(record)
    return dict(potential_allocations=list(potential.values()),operand_views=views,
                scope='Potential physical copies and operand views, not a live allocation timeline or OOM result')

def derive_copy_contract(c, dag, devices, allocations):
    producers={tid:oid for oid,op in dag['operations'].items()
               for tid in [op['output'],*op.get('extra_outputs',[])]}
    copies={};operand_guards=[]
    for view in allocations['operand_views']:
        tid=view['tensor_id'];consumer=view['consumer'];target=view['domain'];t=dag['tensors'][tid]
        producer=producers.get(tid)
        source=('dram' if devices[producer]=='cpu' else 'vram') if producer else (
            t.get('initial_locations') or [None])[0]
        if source is None:raise ValueError('Value has neither an initial source nor a producer')
        if source==target:
            dependency=dict(kind='producer_complete' if producer else 'initial_copy_ready',
                            tensor_id=tid,domain=target,producer=producer,consumer=consumer)
        else:
            key=(tid,source,target)
            copy_id='copy:'+tid+':'+source+'>'+target
            row=copies.setdefault(key,dict(id=copy_id,tensor_id=tid,src=source,dst=target,
                physical_bytes=view['physical_bytes'],source_producer=producer,consumers=[],
                trigger=c['intent']['preparation'] if t['role']=='weight' else 'demand',
                additional_reload_policy='Possible later reloads require runtime cache/retention state'))
            if consumer not in row['consumers']:row['consumers'].append(consumer)
            dependency=dict(kind='copy_complete',tensor_id=tid,domain=target,
                            copy_id=copy_id,consumer=consumer)
        operand_guards.append(dependency)
    return dict(unique_potential_transfers=list(copies.values()),operand_readiness=operand_guards,
                producers=producers,scope='Required first-residency/copy templates; not actual executed byte totals')

def derive_reader_contract(c, dag, devices, allocations):
    readers=defaultdict(list)
    for view in allocations['operand_views']:
        readers[view['allocation_id']].append(view['consumer'])
    return [dict(allocation_id=aid,readers=sorted(set(ids)),
                 release_guard='active_readers == 0 and valid_backing_or_no_future_use',
                 last_use_guard='all_declared_consumers_complete',
                 retention_rule=c['intent']['cache_rule'] if dag['tensors'][aid[7:].rsplit('@',1)[0]]['role']=='weight'
                    else c['intent']['release']) for aid,ids in sorted(readers.items())]

def derive_preparation_contract(c, dag, copy_contract):
    return [dict(copy_id=x['id'],source_guard='initial_ready' if x['source_producer'] is None else 'producer_complete',
                 source_producer=x['source_producer'],capacity_guard='reserve destination physical bytes before copy',
                 bytes_to_reserve=x['physical_bytes'],trigger=x['trigger'],
                 scheduling_guard='spare_resource_and_capacity' if x['trigger']=='spare' else 'known_consumer_demand',
                 consumes_no_future_route=True,compute_waits_for=x['consumers'])
            for x in copy_contract['unique_potential_transfers']]

def compass_calls(bundle,c,dag):
    calls=[];h=c['geometry']['hidden'];i=c['geometry']['intermediate']
    for oid,op in dag['operations'].items():
        ctx=dag['contexts'][oid];q=ctx['n'];kind=op['semantic_op']
        weights=[dag['tensors'][tid] for tid in op['inputs'] if dag['tensors'][tid]['role']=='weight']
        result=None
        if kind in ('ffn_gate','ffn_up'):
            a,b=op['op_params'];result=bundle.matmul([q,h],[h,b-a])
        elif kind=='ffn_down':
            a,b=op['op_params'];result=bundle.matmul([q,b-a],[b-a,h])
        elif kind=='attention_qk':
            result=bundle.matmul([q,c['workload']['head_dim']],[c['workload']['head_dim'],ctx['past']+q])
        elif kind=='attention_pv':
            result=bundle.matmul([q,ctx['past']+q],[ctx['past']+q,c['workload']['head_dim']])
        elif kind=='attention_softmax':result=bundle.softmax([c['geometry']['heads'],q,ctx['past']+q])
        elif kind=='attention_output':result=bundle.matmul([q,h],[h,h])
        elif kind=='projection':
            result=dict(kind='projection_shape_components',components=[bundle.matmul([q,h],[h,out])
                for out in (h,c['workload']['head_dim'],c['workload']['head_dim'])])
        elif kind=='dense_ffn':
            result=dict(kind='ffn_shape_components',components=[bundle.matmul([q,h],[h,i]),
                bundle.matmul([q,h],[h,i]),bundle.matmul([q,i],[i,h])])
        elif kind=='attention':
            result=dict(kind='attention_shape_components',components=[
                bundle.matmul([q,c['workload']['head_dim']],[c['workload']['head_dim'],ctx['past']+q]),
                bundle.softmax([c['geometry']['heads'],q,ctx['past']+q]),
                bundle.matmul([q,ctx['past']+q],[ctx['past']+q,c['workload']['head_dim']])])
        if result is not None:calls.append(dict(operation_id=oid,exact_interface_result=result))
    return dict(executed_operator_calls=calls,
                excluded_semantics=['Full macro fusion/kernel timing','CPU/GPU runtime scheduling','Physical allocation lifetime'],
                note='Calls validate constituent shapes, not a claim that an opaque macro equals a sum of these timings')

def serving_calls(bundle,c,dag,devices,allocations,copies):
    rows=[];row_evidence=[];view_lookup={(x['consumer'],x['tensor_id']):x for x in allocations['operand_views']}
    for oid,op in dag['operations'].items():
        weights=[tid for tid in op['inputs'] if dag['tensors'][tid]['role']=='weight']
        activation=[tid for tid in op['inputs'] if dag['tensors'][tid]['role']!='weight']
        outputs=[op['output'],*op.get('extra_outputs',[])]
        # Source trace rows accept physical bytes independently of computation.
        # Per-operation device identity is in our companion record, not invented
        # as an extra field in the upstream eleven-field schema.
        row=[oid,'1','LOCAL',str(sum(tensor_bytes(dag['tensors'][t]) for t in activation)),
             'REMOTE:0' if devices[oid]=='gpu' and weights else 'LOCAL',
             str(sum(view_lookup[(oid,t)]['physical_bytes'] for t in weights)),
             'LOCAL',str(sum(tensor_bytes(dag['tensors'][t]) for t in outputs)),'NONE','0','NONE']
        rows.append(row)
        row_evidence.append(dict(operation_id=oid,execution_device=devices[oid],
                                 physical_weight_byte_rule='sum of physical parent/shard operands',
                                 original_weight_ids=weights))
    parsed=bundle.parse_rows(rows)
    for row,item in zip(rows,parsed):
        if item['weight_bytes']!=int(row[5]) or item['compute_ns']!=1:raise ValueError('Upstream row roundtrip differs')
    config=dict(hidden_size=c['geometry']['hidden'],intermediate_size=c['geometry']['intermediate'],
        num_attention_heads=c['geometry']['heads'],num_key_value_heads=c['geometry']['kv_heads'],
        head_dim=c['workload']['head_dim'],vocab_size=32)
    native_sizes=[bundle.serving_sizes(config,stage,c['geometry']['prefill_tokens'])
                  for stage in ('gate_up_proj','act_fn','down_proj')]
    resolved=bundle.serving_placement('REMOTE:0')
    return dict(exact_parser_rows=rows,parsed=parsed,companion_operation_devices=row_evidence,
                native_size_helper_results=native_sizes,native_location_resolution=resolved,
                computed_time_is_placeholder_not_performance=True,
                note='Parser acceptance and size/configuration helpers; no System Simulator, execution planner or ASTRA run')

def audit_case(bundle,c,dag):
    if c['workload']['kind']=='moe':
        return dict(id=c['id'],pair=c['pair'],family=c['family'],split=c['split'],variant=c['variant'],
            status='not_evaluated_causal_router_adapter_not_implemented',whole_task_score=None,
            extra_coordination_score=None,categories={k:dict(status='not_evaluated',extra_declaration=None) for k in CATEGORIES},
            private_route_exposed_to_adapter=False,private_declared_dag_loaded_by_adapter=False,
            reason='No policy-side callback for router revelations is implemented in this external audit. Native causal MoE support is not denied.')
    devices={oid:placement(c,op,dag['contexts'][oid]) for oid,op in dag['operations'].items()}
    alloc=derive_allocation_contract(c,dag,devices)
    copies=derive_copy_contract(c,dag,devices,alloc)
    readers=derive_reader_contract(c,dag,devices,alloc)
    prep=derive_preparation_contract(c,dag,copies)
    expected=sum(len(op['inputs']) for op in dag['operations'].values())
    if len(alloc['operand_views'])!=expected or len(copies['operand_readiness'])!=expected:
        raise ValueError('Every mathematical operand must have one view and readiness guard')
    if any(x['bytes_to_reserve']<=0 for x in prep):raise ValueError('Nonpositive transfer reservation')
    categories={k:dict(status='automatically_derived_static_contract',extra_declaration=0,
        origin='frozen_parameterized_adapter',whole_runtime_validated=False) for k in CATEGORIES}
    result=dict(id=c['id'],pair=c['pair'],family=c['family'],split=c['split'],variant=c['variant'],
        status='static_interface_and_adapter_obligations_only',whole_task_score=None,
        extra_coordination_score=None,static_contract_extra_categories=0,categories=categories,
        common_input_hashes=dict(public_intent_sha256=digest(c),mathematical_workload_sha256=digest(dag)),
        extra_user_coordination_input={},operation_devices=devices,allocation_contract=alloc,
        copy_contract=copies,reader_contract=readers,preparation_contract=prep,
        limitations=['No complete external scheduler or causal policy execution',
            'Static templates do not prove actual copy counts, live-memory peaks or OOM',
            'Compiler production traces and physical summaries were not adapter inputs'],
        LLMCompass=compass_calls(bundle,c,dag),LLMServingSim=serving_calls(bundle,c,dag,devices,alloc,copies))
    return result
