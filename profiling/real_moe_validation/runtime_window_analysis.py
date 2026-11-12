"""Estimate real runtime-window prices; compose only through modeling.

The native observations partition one request's host wall clock. CPU windows
are single actual GGML nodes closed by the ordinary next-node barrier/return.
Other windows are explicitly inclusive runtime bridges or host/sampler work.
No CUDA async-submit interval is interpreted as device completion.
"""
from __future__ import annotations
from collections import defaultdict
from fractions import Fraction
import hashlib
import itertools
import json
import re

import numpy as np
from llm_infer_model.performance.moe_execution import RuntimeWindowAction,RuntimeWindowPrice,compile_runtime_windows,MECHANISM_POLICY,MECHANISM_POLICY_V2,MECHANISM_POLICIES,state_applies
from llm_infer_model.performance.window_work import WindowWorkSpec,FEATURES,INTERCEPT,CPU_ATTENTION_STAGE

COEFFICIENT_GRID=1_000_000_000


def require(ok,message):
    if not ok:raise ValueError(message)

def key_text(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True)

def node_layer(name,previous=None):
    match=re.search(r'-(\d+)$',name)
    if match:return int(match[1])
    if name.startswith(('inp_','result_','output')):return None
    return previous

def forward_kv(events,nodes):
    candidates=defaultdict(set)
    for e in events:
        if e['kind']!=1:continue
        for n in nodes[e['first_node']:e['first_node']+e['node_count']]:
            if n['name'].startswith(('kq-','kq_soft_max-','kq_scaled-')):
                candidates[e['forward']].add(int(n['ne'][0]))
    out={}
    for f,values in candidates.items():
        require(len(values)==1,'inconsistent actual Attention KV geometry within forward '+str(f)+': '+repr(values))
        out[f]=next(iter(values))
    return out

def decode_history(forward, depth_rank, first_decode, closes_request=False):
    """Name a decode forward by its position among the forwards at its padded KV depth.

    The first two forwards at a depth carry a setup cost the steady state does
    not, and a generation that crosses a depth pays it again, so the position is
    read inside the depth rather than inside the request. The window that closes
    the request is the exception: its identity exists once per request and its
    cost is the host's response to the sampler, not the device work that enters
    a depth (see PREREG-decode-depth-windows-v2.md).
    """
    if not closes_request and forward in depth_rank:
        return 'decode:'+{0:'first',1:'settling'}.get(depth_rank[forward],'continuation')
    return 'decode:'+('first' if forward<=first_decode else 'continuation')


def cpu_consumed_ns(left,right,cycles_per_second=None):
    """CPU time the calling thread spent between two observer boundaries.

    Two stamp formats exist. A nanosecond CPU stamp is used directly. A cycle
    stamp is per thread, so a window whose ends ran on different threads has no
    CPU division at all and returns None rather than a number that silently
    subtracts one thread's counter from another's. A window with no CPU stamp
    on either end - an older probe - also returns None.
    """
    a,b=left.get('cpu_ns'),right.get('cpu_ns')
    if a is not None and b is not None:return int(b-a)
    a,b=left.get('thread_cycles'),right.get('thread_cycles')
    if a is None or b is None or not cycles_per_second:return None
    if left.get('thread_id') is None or left.get('thread_id')!=right.get('thread_id'):return None
    return Fraction(int(b-a))*1_000_000_000/Fraction(cycles_per_second)

def extract_windows(row, *, placement,frequency_hz,feature_policy=MECHANISM_POLICY,capture_ticks=False):
    """Return measured windows plus a timing-free replayable workload inventory.

    feature_policy is the declaration rule set, never a fit choice: v1 reads
    work from each window's own nodes, v2 additionally reads one ordering
    dependency when a bridge's transfer carries the output of a host-resident
    attention stage in the same forward and layer.
    """
    require(isinstance(feature_policy,str) and feature_policy in MECHANISM_POLICIES,'unknown mechanism feature policy')
    raw=row['runtime_windows'];events=raw['events'];nodes=raw['nodes']
    require(raw['schema']=='real-moe-runtime-windows-1' and raw['added_GPU_synchronizations']==0,'unexpected observation boundary')
    require(events and events[0]['kind']==20 and events[-1]['kind']==26,'complete request clock boundaries required')
    require(all(a['tick']<=b['tick'] for a,b in zip(events,events[1:])),'nonmonotonic observation timestamps')
    require(events[0]['tick']==row['begin_tick'] and events[-1]['tick']==row['end_tick'],'window request clock differs')
    kv=forward_kv(events,nodes)
    require(kv,'actual padded Attention KV dimension missing; no guessed KV geometry')
    require(row['prefill_end_tick']>=row['begin_tick'],'missing Prefill completed boundary')
    pending_gpu=[];payload=0;copy_calls=0;sync_calls=0;first_copy_tick=None;left=events[0];left_node=None;last_cpu_layer={};windows=[];graph_counter=defaultdict(int);cpu_occurrences=defaultdict(int);graph_depth=0;current_phase='prefill'
    # (forward, layer) -> the actual host attention node that completed there.
    host_attention_stage={}
    actual_forwards=row.get('forward_inventory',[])
    decode_forwards=[v['index'] for v in actual_forwards if v['phase']=='decode']
    if not decode_forwards:decode_forwards=[v['forward'] for v in events if v['kind']==21 and v.get('phase')=='decode']
    first_decode=min(decode_forwards) if decode_forwards else min(kv)
    # A decode forward's measured cost depends on how many forwards of the same
    # padded KV depth preceded it: the first two cost 40-70% more than the steady
    # state, and the excess reappears whenever the depth changes inside a
    # generation (see PREREG-decode-depth-windows.md). One price keyed by the
    # request's first decode forward alone fits a line across both, so the
    # position within the depth is declared instead.
    depth_rank={}
    depth_seen=defaultdict(int)
    for forward in sorted(set(decode_forwards)):
        # A forward with no actual Attention geometry has no padded depth to be
        # positioned in; the per-request fallback in decode_history keeps it out
        # of a class it was never measured in.
        if forward not in kv:continue
        depth_rank[forward]=depth_seen[kv[forward]]
        depth_seen[kv[forward]]+=1
    prior_cpu_name='request_begin'
    for e in events[1:]:
        kind=e['kind']
        if kind==1:graph_depth+=1
        elif kind==2:
            graph_depth-=1;require(graph_depth>=0,'graph API end without matching beginning')
        if kind==1 and not e['cpu']:
            count=graph_counter[e['forward']];graph_counter[e['forward']]+=1
            for node_index,n in enumerate(nodes[e['first_node']:e['first_node']+e['node_count']]):
                pending_gpu.append(dict(identity='forward'+str(e['forward'])+':gpu_graph'+str(count)+':node'+str(node_index)+':'+n['name']+':op'+str(n['op']),node=n,source=e))
        if kind in (5,7) and not e['cpu']:
            payload+=e['requested_bytes'];copy_calls+=1
            if first_copy_tick is None:first_copy_tick=e['tick']
        if kind==4:sync_calls+=1
        boundary=(kind in (10,23,24,25,26) or (kind==2 and e['cpu'])
            or (kind==4 and not e['cpu'] and graph_depth==0))
        if not boundary:continue
        right_node=nodes[e['first_node']] if kind==10 else None
        require(not(left_node is not None and pending_gpu),'CPU-node window unexpectedly contains new GPU submissions')
        gpu_source_forwards={n['source']['forward'] for n in pending_gpu}
        require(len(gpu_source_forwards)<=1,'inclusive GPU bridge contains multiple source forwards; this scoped response model does not assign them a guessed single q')
        context=left if left_node is not None else pending_gpu[0]['source'] if pending_gpu else e
        f=context['forward']
        # The terminal host/sampler interval can point to the next ordinal but
        # its actual preceding graph retains the same attention history.
        if f not in kv:
            prior=[k for k in kv if k<=f];f=max(prior) if prior else min(kv)
        phase=current_phase
        q=max(1,int(context['query_tokens']));ordinal=int(context['ordinal'])
        if phase=='prefill':
            history=phase+':'+('first' if f==min(kv) else 'continuation')
        else:
            history=decode_history(f,depth_rank,first_decode,closes_request=(kind==26))
        covered=();layer=None;op=None;global_operation=False
        if left_node is not None:
            scope='CPU_NODE';name=left_node['name'];op='ggml_op_'+str(left_node['op'])
            layer=node_layer(name,last_cpu_layer.get(f));last_cpu_layer[f]=layer;global_operation=layer is None
            occurrence_key=(f,name,left_node['op']);occurrence=cpu_occurrences[occurrence_key];cpu_occurrences[occurrence_key]+=1
            identity=dict(name=name,op=left_node['op'],type=left_node['type'],weight_type=left_node['weight_type'],occurrence=occurrence)
            covered=('cpu:'+name+':op'+str(left_node['op'])+':instance'+str(occurrence),)
            # Pinned GGML b8705 enum: MUL_MAT=29, MUL_MAT_ID=30.
            # Actual normal output pruning can make the final FFN/head q=1
            # while the Attention forward still processes all prompt tokens.
            if left_node['op']==29:q=int(left_node['ne'][1])
            elif left_node['op']==30:q=int(left_node['ne'][2])
            require(q>0,'actual dispatched matrix node has no query rows')
        elif left['kind']==24:
            scope='SAMPLER';identity=dict(stage='greedy_sample')
        elif pending_gpu:
            scope='RUNTIME_BRIDGE';identity=dict(left=prior_cpu_name,right=right_node['name'] if right_node else 'boundary_'+str(kind),GPU_node_names=[n['node']['name'] for n in pending_gpu],GPU_node_ops=[n['node']['op'] for n in pending_gpu])
            covered=tuple(n['identity'] for n in pending_gpu)
        else:
            scope='HOST';identity=dict(left_kind=left['kind'],right_kind=kind,left_cpu=prior_cpu_name,right_cpu=right_node['name'] if right_node else None)
        dependency=None;dependency_evidence=()
        if scope=='RUNTIME_BRIDGE' and feature_policy==MECHANISM_POLICY_V2 and copy_calls:
            prior_layer=last_cpu_layer.get(f);producer=host_attention_stage.get((f,prior_layer))
            if producer is not None:
                # The bridge stages an activation that a host attention stage
                # just produced in this very forward and layer; both the start
                # time and the transferred buffer depend on that stage.
                dependency=CPU_ATTENTION_STAGE
                dependency_evidence=('actual_producer:'+prior_cpu_name,
                    'host_attention_stage:'+producer+':layer'+str(prior_layer),
                    'carried_transfer:'+str(copy_calls)+' copy call(s)')
        work=WindowWorkSpec.from_metadata(scope,copy_calls=copy_calls,
            cpu_node=left_node if scope=='CPU_NODE' else None,
            gpu_nodes=[n['node'] for n in pending_gpu] if scope=='RUNTIME_BRIDGE' else (),
            dependency=dependency,dependency_evidence=dependency_evidence)
        if scope=='CPU_NODE' and work.operators[0].role=='attention_core' and layer is not None:
            host_attention_stage[(f,layer)]=name
        if scope=='CPU_NODE' and work.operators[0].query_axis is not None:
            q=int(work.operators[0].ne[work.operators[0].query_axis])
        elif scope=='RUNTIME_BRIDGE' and work.query_basis in ('actual_matrix_rows','actual_pointwise_rows','actual_attention_rows'):
            role={'actual_matrix_rows':'matrix','actual_pointwise_rows':'pointwise','actual_attention_rows':'attention_core'}[work.query_basis]
            rows={node.ne[node.query_axis] for node in work.operators if node.role==role and node.query_axis is not None}
            require(len(rows)==1,'declared runtime query basis requires unique actual local rows')
            q=next(iter(rows))
        work.validate_coordinates(q,kv[f],payload)
        key=key_text(dict(placement=placement,scope=scope,phase=phase,history=history,identity=identity,
            feature_policy=feature_policy,work_signature=work.structural_signature()))
        duration=Fraction(e['tick']-left['tick'])*1_000_000_000/frequency_hz
        inventory=dict(id='window:'+str(len(windows)),key=key,scope=scope,query_tokens=q,kv_padded=kv[f],history=history,phase=phase,
            layer=layer,op=op,covered_nodes=list(covered),global_operation=global_operation,forward_index=f,payload_bytes=payload,
            work_spec=work.to_dict(),
            metadata=dict(CPU_node=left_node,GPU_nodes=[n['node'] for n in pending_gpu],copy_calls=copy_calls,natural_sync_calls=sync_calls,
                actual_forward_ordinal=ordinal,actual_forward_query_tokens=int(context['query_tokens']),
                query_coordinate='actual work-spec local row axis when structurally identified; forward q is only an unknown-work domain guard otherwise',
                mechanism_feature_policy=feature_policy+'; no KV proxy for cache, expert history, CPU availability or ordinary matrix work'
                    +('; bridge KV only through an actual host attention stage in the same forward/layer' if feature_policy==MECHANISM_POLICY_V2 else ''),
                mechanism_dependency=work.dependency,mechanism_dependency_evidence=list(work.dependency_evidence),
                history_coordinate='prefill first/continuation, or decode first/settling/continuation by position among the decode forwards of the same padded KV depth, except the request-closing window which keeps the request position; not a claim of identical CUDA graph mode',
                sampler_scope='SAMPLER is the residual sampler tail after natural-sync partitioning; HOST/bridge can contain preceding sampler work',
                GPU_source_forwards=[dict(forward_index=source,query_tokens=pending_gpu[0]['source']['query_tokens'],kv_padded=kv[source]) for source in sorted(gpu_source_forwards)],
                GPU_source_coordinate='source submission forward, never the later natural-completion boundary; mixed-source bridges are explicitly unsupported',
                layer_identity='actual_named_node_or_preceding_named_layer' if layer is not None else 'global_or_unassigned_actual_node',
                payload_scope='actual H2D set/set_async requested bytes, including submitted padding; not inferred bus traffic'))
        # Validate a single CPU node was not expanded into an opaque CPU model.
        if scope=='CPU_NODE':require(len(covered)==1,'one actual CPU node required')
        if not capture_ticks:
            windows.append(dict(inventory=inventory,observed_ns=duration))
        else:
            ticks=dict(left=int(left['tick']),end=int(e['tick']),
                first_copy=None if first_copy_tick is None else int(first_copy_tick))
            # Thread CPU time is optional: only probes that stamp it carry a
            # CPU-side observable. When present, wall minus the CPU-time delta is
            # the part of the window the calling thread spent blocked, which wall
            # time alone cannot distinguish from work.
            consumed=cpu_consumed_ns(left,e,raw.get('thread_cycles_per_second'))
            if consumed is not None:
                ticks['cpu_ns']=consumed
                ticks['blocked_ns']=duration-Fraction(consumed)
            windows.append(dict(inventory=inventory,observed_ns=duration,ticks=ticks))
        if left_node is not None:prior_cpu_name=left_node['name']
        pending_gpu=[];payload=copy_calls=sync_calls=0;first_copy_tick=None;left=e;left_node=right_node
        if kind==23:current_phase='decode'
    require(not pending_gpu and graph_depth==0,'unclosed submitted GPU work inventory')
    total=sum((w['observed_ns'] for w in windows),Fraction(0))
    exact=Fraction(row['end_tick']-row['begin_tick'])*1_000_000_000/frequency_hz
    require(total==exact,'mutually exclusive windows do not conserve the complete request clock')
    cpu=[w for w in windows if w['inventory']['scope']=='CPU_NODE']
    expert_layers={w['inventory']['layer'] for w in cpu if w['inventory']['metadata']['CPU_node']['name'].startswith(('ffn_moe_up-','ffn_moe_gate-','ffn_moe_down-')) and w['inventory']['layer'] is not None}
    return dict(schema='real-moe-runtime-window-ledger-1',windows=windows,total_ns=total,source_request_id=row['id'],
        workload_inventory=[w['inventory'] for w in windows],CPU_node_windows=len(cpu),expert_layers=sorted(expert_layers),
        actual_forward_KV=kv,clock_conservation_passed=True,additional_device_costs=False,whole_model_multiplier_fitted=False)

def action_from_inventory(w):
    fields=('id','key','scope','query_tokens','kv_padded','history','phase','layer','op','covered_nodes','global_operation','forward_index','payload_bytes')
    d={k:w[k] for k in fields};d['covered_nodes']=tuple(d['covered_nodes'])
    if 'work_spec' in w:d['work_spec']=WindowWorkSpec.from_dict(w['work_spec'])
    # The machine state is declared on the window, never inferred from a duration.
    if w.get('device_state') is not None:d['device_state']=w['device_state']
    return RuntimeWindowAction(**d)

def design_row(inventory,state_multiplier=1):
    """One calibration row: the intercept and the work, scaled by its declared state."""
    return [1.,float(state_multiplier)*inventory['query_tokens'],float(state_multiplier)*inventory['query_tokens']*inventory['kv_padded'],
            float(state_multiplier)*inventory['payload_bytes']]

def fit_nonnegative(samples,allowed_features=None,state_multipliers=None,reference='nominal'):
    """Tiny explicit active-set least squares, no network/DNN or heldout labels.

    Candidate columns are 1, q, q*KV and requested H2D bytes. The actual work
    declaration restricts this set before any calibration times are fitted.
    Missing declarations retain the explicit legacy test/helper contract only.
    A sample measured in a declared machine state enters as its work terms times
    that state's multiplier; the intercept is never scaled, which is what keeps
    the clock-independent part of the response identifiable.
    """
    require(samples,'nonempty calibration windows required')
    if allowed_features is None:
        declared=samples[0]['inventory'].get('work_spec')
        allowed_features=WindowWorkSpec.from_dict(declared).allowed_features if declared else FEATURES
    require(allowed_features and set(allowed_features)<=set(FEATURES) and 'intercept' in allowed_features,'invalid structural feature set')
    allowed_indices={i for i,name in enumerate(FEATURES) if name in allowed_features}
    x=np.asarray([design_row(s['inventory'],sample_state_multiplier(s,state_multipliers,reference)) for s in samples])
    y=np.asarray([float(s['observed_ns']) for s in samples])
    scales=np.maximum(np.max(np.abs(x),axis=0),1.);design=x/scales
    best=None
    for mask in range(1,16):
        selected=[k for k in range(4) if mask&(1<<k)]
        if not set(selected)<=allowed_indices:continue
        coefficients=np.linalg.lstsq(design[:,selected],y,rcond=None)[0]
        if np.any(coefficients < -1e-6):continue
        beta=np.zeros(4);beta[selected]=np.maximum(coefficients,0)
        error=float(np.sum((design@beta-y)**2))
        # Prefer fewer active terms when residuals are numerically equivalent.
        rank=(round(error,6),len(selected),mask)
        if best is None or rank<best[0]:best=(rank,beta/scales)
    require(best is not None,'nonnegative affine estimation failed')
    # Common grid bounds the sum denominator. Independent limit_denominator
    # calls produce mutually prime denominators and enormous whole-ledger LCMs.
    # Exact Fraction rounding is within half a grid step of the binary LS value.
    values=[Fraction(round(Fraction.from_float(float(v))*COEFFICIENT_GRID),COEFFICIENT_GRID) for v in best[1]]
    return values

def sample_state_multiplier(sample,state_multipliers,reference='nominal'):
    """The declared multiplier of one calibration window.

    `state_multipliers` is None for a table that does not declare the machine
    state at all, which is every cohort before this one; those windows are all
    priced at one and every earlier table reproduces exactly. The reference
    state carries one because it is the state the table is priced in, not
    because it is nominal.
    """
    if state_multipliers is None:return 1
    if not state_applies(sample['inventory']['scope']):return 1
    state=sample['inventory'].get('device_state',reference)
    table={name:Fraction(value) for name,value in dict(state_multipliers).items()}
    table.setdefault(reference,Fraction(1))
    require(state in table,'calibration window in an undeclared machine state: '+str(state))
    require(table[state]>0,'a declared state multiplier must be positive')
    return table[state]

INTERCEPT_POLICIES=('per_key','shared_within_placement')

def _slope_design(inventories):
    x=np.asarray([[i['query_tokens'],i['query_tokens']*i['kv_padded'],i['payload_bytes']] for i in inventories],dtype=float)
    return x,np.maximum(np.max(np.abs(x),axis=0),1.)

def fit_slopes_nonnegative(design,y,scales,allowed_features):
    """Nonnegative active-set least squares over the non-intercept columns.

    The intercept is supplied as a fixed offset so that one placement-wide value
    can be fitted while every key keeps only its own slopes. Column scaling and
    the parsimony tie-break match fit_nonnegative exactly, so the two estimators
    differ in nothing but where the intercept comes from.
    """
    allowed={name for name in allowed_features if name in FEATURES and name!=INTERCEPT}
    indices=[k for k,name in enumerate(FEATURES[1:]) if name in allowed]
    scaled=design/scales
    best=None
    for size in range(len(indices)+1):
        for selected in itertools.combinations(indices,size):
            selected=list(selected)
            beta=np.zeros(3)
            if selected:
                coefficients=np.linalg.lstsq(scaled[:,selected],y,rcond=None)[0]
                if np.any(coefficients < -1e-6):continue
                beta[selected]=np.maximum(coefficients,0)
            error=float(np.sum((scaled@beta-y)**2))
            rank=(round(error,6),len(selected),tuple(selected))
            if best is None or rank<best[0]:best=(rank,beta/scales)
    require(best is not None,'nonnegative slope estimation failed')
    return best[1]

def fit_shared_intercept(grouped,allowed_by_key,sweeps=60):
    """One intercept for the whole placement; the slopes stay key-specific.

    A price key holds about fifteen windows, so the per-key split between the
    intercept and the query column is set by a handful of nearly collinear
    points. Those arbitrary splits cancel inside a key and do not cancel when a
    request sums thousands of windows from thousands of keys, which is how the
    fitted table can miss its own calibration totals. Sharing one intercept
    removes the free parameter that the data cannot identify. The objective is
    convex and separable, so alternating the closed-form intercept with the
    per-key nonnegative slopes converges to its minimum.

    REJECTED as a default. Leave-one-request-out over the six calibration
    requests favours it strongly (attn total APE 12.29% -> 7.19%), but scoring
    the frozen v15 heldout with everything else held fixed reverses that: total
    APE 4.81% -> 6.61%, decode-with-sampling 8.14% -> 34.17%, passing rows
    32/48 -> 15/48. A constant charged to every window is negligible in prefill,
    which has many windows, and dominant in decode, which has few and short
    ones. The calibration fold never decomposes by phase, so it could not see
    this. Kept only so the negative stays reproducible; 'per_key' is the
    default and reproduces the frozen price table bit for bit.
    """
    require(grouped,'nonempty calibration groups required')
    prepared={}
    for key,samples in grouped.items():
        design,scales=_slope_design([s['inventory'] for s in samples])
        prepared[key]=(design,scales,np.asarray([float(s['observed_ns']) for s in samples]))
    total=sum(float(s['observed_ns']) for samples in grouped.values() for s in samples)
    a=total/sum(len(samples) for samples in grouped.values())
    for _ in range(sweeps):
        slopes={};numer,denom=0.0,0
        for key,(design,scales,y) in prepared.items():
            beta=fit_slopes_nonnegative(design,y-a,scales,allowed_by_key[key])
            slopes[key]=beta
            numer+=float(np.sum(y-design@beta));denom+=y.size
        updated=max(0.0,numer/denom)
        converged=abs(updated-a)<=1e-9*max(1.0,abs(a))
        a=updated
        if converged:break
    return a,slopes

def declared_states(state_multipliers,reference='nominal'):
    """The state family every key of one table is calibrated for.

    The reference state is the identification point: it carries multiplier one
    because a key measured entirely in one state cannot separate that state from
    its own coefficients, so the coefficients are in the units of the state the
    calibration actually ran in. Every other state is stated relative to it.
    """
    table={reference:Fraction(1)}
    if state_multipliers is not None:
        for name,value in dict(state_multipliers).items():
            require(name!=reference,'the reference multiplier is the identification point, never supplied')
            table[name]=Fraction(value)
            require(table[name]>0,'a declared state multiplier must be positive')
    return tuple(sorted(table.items()))

def fit_prices(ledgers,evidence,feature_policy=MECHANISM_POLICY,intercept_policy='per_key',state_multipliers=None,reference_state='nominal'):
    require(isinstance(feature_policy,str) and feature_policy in MECHANISM_POLICIES,'unknown mechanism feature policy')
    require(intercept_policy in INTERCEPT_POLICIES,'unknown intercept identification policy')
    require(state_multipliers is None or intercept_policy=='per_key','a shared intercept has no state-scaled slope identification')
    grouped=defaultdict(list)
    for ledger in ledgers:
        for window in ledger['windows']:grouped[window['inventory']['key']].append(window)
    works,allowed_by_key={},{}
    for key,samples in grouped.items():
        first=samples[0]['inventory'];declared=first.get('work_spec')
        work=WindowWorkSpec.from_dict(declared) if declared else None
        if work is not None:
            require(all(WindowWorkSpec.from_dict(s['inventory']['work_spec']).structural_signature()==work.structural_signature() for s in samples),'one price cannot mix structural work declarations')
        works[key]=work
        allowed_by_key[key]=work.allowed_features if work is not None else FEATURES
    shared=None
    if intercept_policy=='shared_within_placement':
        shared,shared_slopes=fit_shared_intercept(grouped,allowed_by_key)
        shared=Fraction(round(Fraction.from_float(float(shared))*COEFFICIENT_GRID),COEFFICIENT_GRID)
    prices=[]
    for key,samples in sorted(grouped.items()):
        first=samples[0]['inventory'];work=works[key]
        if shared is None:
            a,b,c,d=fit_nonnegative(samples,allowed_by_key[key],state_multipliers,reference_state)
        else:
            if any(float(s['observed_ns'])!=0.0 for s in samples):
                a=shared
                b,c,d=(Fraction(round(Fraction.from_float(float(v))*COEFFICIENT_GRID),COEFFICIENT_GRID) for v in shared_slopes[key])
            else:
                # A key whose every window resolved to zero keeps the explicit
                # resolution-limited record; a positive intercept would price a
                # response the clock never saw.
                a=b=c=d=Fraction(0)
        price=RuntimeWindowPrice(key=key,scope=first['scope'],intercept_ns=a,per_query_token_ns=b,per_attention_pair_ns=c,
            observed_query_tokens=tuple(sorted({s['inventory']['query_tokens'] for s in samples})),
            observed_kv_padded=tuple(sorted({s['inventory']['kv_padded'] for s in samples})),
            history_domain=tuple(sorted({s['inventory']['history'] for s in samples})),evidence=tuple(evidence),
            per_payload_byte_ns=d,observed_payload_bytes=tuple(sorted({s['inventory']['payload_bytes'] for s in samples})),
            feature_policy=feature_policy if work is not None else 'legacy_affine_v1',work_spec=work,
            zero_duration_approximation=all(v==0 for v in (a,b,c,d)),
            state_multipliers=declared_states(state_multipliers,reference_state),reference_state=reference_state)
        prices.append(price)
    return tuple(prices)

def price_fit_diagnostics(ledgers,prices):
    """Expose sparse/collinear calibration support without claiming validation."""
    grouped=defaultdict(list)
    for ledger in ledgers:
        for window in ledger['windows']:
            grouped[window['inventory']['key']].append((ledger,window))
    result={}
    for price in prices:
        samples=grouped[price.key]
        coordinates=sorted({(w['inventory']['query_tokens'],w['inventory']['kv_padded'],w['inventory']['payload_bytes']) for _,w in samples})
        x=np.asarray([[1.,q,q*kv,payload] for q,kv,payload in coordinates])
        allowed=price.work_spec.allowed_features if price.work_spec is not None else FEATURES
        indices=[i for i,term in enumerate(FEATURES) if term in allowed]
        x=x[:,indices];scaled=x/np.maximum(np.max(np.abs(x),axis=0),1.)
        terms=FEATURES
        coefficients=(price.intercept_ns,price.per_query_token_ns,price.per_attention_pair_ns,price.per_payload_byte_ns)
        processes={l.get('source_process_id') for l,_ in samples}-{None}
        result[price.key]=dict(sample_count=len(samples),distinct_request_count=len({l['source_request_id'] for l,_ in samples}),
            calibration_process_count=len(processes),unique_workload_coordinates=[list(v) for v in coordinates],
            scaled_design_rank=int(np.linalg.matrix_rank(scaled)),design_columns=len(indices),allowed_features=list(allowed),
            feature_policy=price.feature_policy,work_kind=price.work_spec.kind if price.work_spec is not None else 'legacy_unrestricted',
            unknown_mechanisms=list(price.work_spec.unknown_mechanisms) if price.work_spec is not None else [],
            prohibited_nonzero_coefficients=[term for term,value in zip(terms,coefficients) if value and term not in allowed],
            individual_terms_identifiable=int(np.linalg.matrix_rank(scaled))==len(indices),
            active_terms=[term for term,value in zip(terms,coefficients) if value],
            coefficient_grid_denominator=COEFFICIENT_GRID,coefficient_grid_step_ns=Fraction(1,COEFFICIENT_GRID),
            maximum_mean_response_representation_error_ns=max(Fraction(1+q+q*kv+payload,2*COEFFICIENT_GRID) for q,kv,payload in coordinates),
            representation_error_bound_scope='relative to binary floating-point least-squares coefficients, not a modeling-error claim; completed windows also round to integer ns',
            calibration_fit_is_independent_accuracy=False)
    return result

def state_design(ledgers,reference='nominal'):
    """Unscaled per-key work, its states, and its response, prepared once."""
    grouped=defaultdict(list)
    for ledger in ledgers:
        for window in ledger['windows']:
            grouped[window['inventory']['key']].append(window)
    prepared={}
    for key,samples in grouped.items():
        declared=samples[0]['inventory'].get('work_spec')
        allowed=WindowWorkSpec.from_dict(declared).allowed_features if declared else FEATURES
        work=np.asarray([[s['inventory']['query_tokens'],s['inventory']['query_tokens']*s['inventory']['kv_padded'],
                          s['inventory']['payload_bytes']] for s in samples],dtype=float)
        prepared[key]=dict(work=work,state=[s['inventory'].get('device_state',reference) if state_applies(s['inventory']['scope']) else reference for s in samples],
            y=np.asarray([float(s['observed_ns']) for s in samples]),
            indices=[i for i,name in enumerate(FEATURES) if name in allowed])
    mixed=[entry for entry in prepared.values() if len(set(entry['state']))>1]
    fixed=sum(_key_error(entry,np.ones(len(entry['state']))) for entry in prepared.values()
              if len(set(entry['state']))==1)
    return dict(mixed=mixed,fixed=fixed,keys=len(prepared),mixed_keys=len(mixed))

def _key_error(entry,scale):
    """Best nonnegative active-set error of one key at one per-sample scaling."""
    x=np.column_stack((np.ones(len(scale)),entry['work']*scale[:,None]))
    design=x/np.maximum(np.max(np.abs(x),axis=0),1.)
    y=entry['y'];best=None
    for mask in range(1,16):
        selected=[k for k in range(4) if mask&(1<<k)]
        if not set(selected)<=set(entry['indices']):continue
        coefficients=np.linalg.lstsq(design[:,selected],y,rcond=None)[0]
        if np.any(coefficients<-1e-6):continue
        beta=np.zeros(4);beta[selected]=np.maximum(coefficients,0)
        error=float(np.sum((design@beta-y)**2))
        rank=(round(error,6),len(selected),mask)
        if best is None or rank<best[0]:best=(rank,)
    require(best is not None,'nonnegative affine estimation failed')
    return best[0][0]

def _key_sse(prepared,multipliers,reference='nominal'):
    """Least-squares error of every key at one declared state family.

    A key whose windows all ran in one state cannot identify the state at all:
    a uniform scale is absorbed exactly by its own coefficients, so its error is
    fixed and is evaluated once, outside the search.
    """
    total=prepared['fixed']
    for entry in prepared['mixed']:
        scale=np.asarray([1.0 if state==reference else multipliers[state] for state in entry['state']])
        total+=_key_error(entry,scale)
    return total

def fit_state_multipliers(ledgers,states=('p4','p5'),limits=None,step=0.05,iterations=3,reference='nominal'):
    """One declared multiplier per machine state, on the calibration windows only.

    The reference state is the identification point: it is the state the
    calibration actually ran in, it carries multiplier one, and every other
    state is stated relative to it. The multipliers are estimated by coordinate
    descent on the summed squared error of the same active-set fit each key uses,
    so the state is part of the price model rather than a correction applied
    afterwards. Nothing here reads a heldout time.
    """
    require(bool(states) and len(set(states))==len(states) and reference not in states,'distinct states away from the reference required')
    observed={(s['inventory'].get('device_state',reference)) for l in ledgers for s in l['windows']}
    require(set(states)<=observed,'no calibration window in a declared state: '+repr(sorted(observed)))
    limits=dict(limits or {})
    grid={s:[round(limits.get(s,(0.5,6.0))[0]+index*step,4)
             for index in range(int(round((limits.get(s,(0.5,6.0))[1]-limits.get(s,(0.5,6.0))[0])/step))+1)] for s in states}
    prepared=state_design(ledgers,reference)
    multipliers={s:1.0 for s in states}
    errors={}
    for _ in range(iterations):
        for state in states:
            plateau=[(c,_key_sse(prepared,dict(multipliers,**{state:c}),reference)) for c in grid[state]]
            errors[state]=plateau
            # The smallest improving multiplicity wins: equal errors at 1.05 and
            # 1.60 must not be resolved towards the larger claim.
            best=min(plateau,key=lambda item:(round(item[1],6),item[0]))
            multipliers[state]=best[0]
    return ({state:Fraction(round(multipliers[state]*1000),1000) for state in states},errors)

def declare_states(windows,declared):
    """Attach the declared machine state to every window of one prediction.

    `declared` is a state name for a uniform declaration, or a mapping from
    (phase, forward index) to the state observed while that forward ran.
    """
    if declared is None:return tuple(windows)
    out=[]
    for window in windows:
        if isinstance(declared,str):
            state=declared
        else:
            state=declared.get((window['phase'],int(window['forward_index'])))
            require(state is not None,'no declared machine state for window '+str(window['id'])+' in '+window['phase']+'/'+str(window['forward_index']))
        item=dict(window);item['device_state']=state;out.append(item)
    return tuple(out)

def predict_inventory(inventory,prices, *, allow_extrapolation=False,declared_states=None):
    actions=tuple(action_from_inventory(w) for w in declare_states(inventory,declared_states))
    compiled=compile_runtime_windows(actions,tuple(prices),allow_extrapolation=allow_extrapolation)
    total=compiled.evaluate().duration_ns
    phase={}
    for name in ('prefill','decode'):
        subset=tuple(a for a in actions if a.phase==name)
        phase[name]=compile_runtime_windows(subset,tuple(prices),allow_extrapolation=allow_extrapolation).evaluate().duration_ns if subset else 0
    return dict(total_ns=total,phase_ns=phase,compiled=compiled.to_dict(),source_observed_durations_used=False)
