"""Independent serving invariants over declared requests and observed execution.

The caller persists the declared request/scope map before compilation. These
checks never infer expected counts, slices, bytes or dependencies from lowered
compiler output. Runtime documents are privileged observed evidence only.
"""
from __future__ import annotations

from dataclasses import asdict

from experiments import strategy_fidelity as ref


def observed(result):
    observation=result.raw[1].observe()
    return ({o.id:o for o in observation.operations},{t.id:t for t in observation.tensors},
            {e.target_id:e for e in result.events if e.kind=="COMPUTE"})


def complete_and_capacity(result, *, cache_budget,weight_inventory=None):
    ref.require(result.success,"full serving goal not reached")
    ops,tensors,computes=observed(result)
    ref.require(ops and all(o.status=="DONE" for o in ops.values()),"final graph contains incomplete operations")
    actual_weights={t.id:t.storage_bytes for t in tensors.values() if t.role=="weight"}
    if weight_inventory is not None:
        ref.equal(actual_weights,weight_inventory,"exact shared physical weight ID/byte inventory; no per-request clones")
    peak=0
    for observation in [r.observation for r in result.raw[0].decisions]+[result.raw[1].observe()]:
        reserved=sum(t.storage_bytes for t in observation.tensors
                     if t.role=="weight" and t.status_at("vram")!="ABSENT")
        peak=max(peak,reserved)
        ref.require(reserved<=cache_budget,"shared cache budget exceeded (including in-flight reservations)")
        for memory in observation.memory:
            if memory.capacity_bytes is not None:
                ref.require(memory.used_bytes<=memory.capacity_bytes,"physical storage capacity exceeded")
    for event in result.events:
        if event.kind=="COPY":ref.equal(event.bytes_transferred,tensors[event.target_id].storage_bytes,"full physical COPY payload charged")
    edges=ref.check_dependency_lifetimes(result)
    return dict(completed_operations=len(ops),dependency_edges=edges,peak_weight_bytes=peak,
                physical_weight_tensors=len(actual_weights),physical_weight_bytes=sum(actual_weights.values()),
                one_runtime=True,summary=result.summary())


def lifecycle(result, *, requests, scope_map, kinds):
    """requests: declared {id:{arrival_ns,prefill_tokens,decode_steps,layers}}."""
    ops,tensors,events=observed(result)
    ref.equal(set(scope_map),{o.invocation_id for o in ops.values() if o.kind!="state_update"},"exact declared request/layer/round scopes")
    for row in result.raw[0].decisions:
        ref.equal(set(row.observation.arrived_requests),
            {request for request,d in requests.items() if d["arrival_ns"]<=row.observation.t_ns},
            "public arrived requests reveal only current/past arrivals")
        ref.require(set(row.observation.retired_requests).issubset(row.observation.arrived_requests),"unarrived request retired")
        for request,declared in requests.items():
            if row.observation.t_ns>=declared["arrival_ns"]:continue
            forbidden=[o for o in row.observation.operations if o.request_id==request
                       or any(work[0]==request for work in o.work_items)]
            ref.require(not forbidden,"request-specific operation leaked before arrival")
            ref.require(not any((":"+request+":") in t.id and t.role!="weight" for t in row.observation.tensors),
                        "request-specific activation/KV leaked before arrival")
    counts={}
    for scope,contract in scope_map.items():
        selected=[o for o in ops.values() if o.invocation_id==scope and o.kind!="state_update"]
        expected_kinds=kinds[contract["kind"]]
        # Dynamic MoE expert counts are checked separately against revealed
        # routes; their fixed non-expert chain still occurs once per leaf.
        for kind in expected_kinds:
            ref.equal(len([o for o in selected if o.kind==kind]),1,"declared macro executes once per request/layer/round")
        for op in selected:
            ref.require(events[op.id].start_ns>=requests[contract["request_id"]]["arrival_ns"],"compute starts before arrival")
            ref.equal(sum(e.kind=="COMPUTE" and e.target_id==op.id for e in result.events),1,"intended operation charged exactly once")
        counts[scope]=len(selected)
    # Retirement is finite trace completion, not a claim to implement EOS
    # sampling or reclaim every now-dead KV allocation immediately.
    last_compute={}
    for request,declared in requests.items():
        expected_scopes=[s for s,c in scope_map.items() if c["request_id"]==request]
        ref.equal(len(expected_scopes),declared["layers"]*(1+declared["decode_steps"]),"unequal per-request round counts")
        own=[e for oid,e in events.items() if ops[oid].invocation_id in expected_scopes]
        last_compute[request]=max(e.end_ns for e in own)
    final=result.raw[1].observe()
    ref.equal(set(final.arrived_requests),set(requests),"all finite requests arrived")
    ref.equal(set(final.retired_requests),set(requests),"all finite requests retired")
    observations=[row.observation for row in result.raw[0].decisions]+[final]
    retirement={r:min(o.t_ns for o in observations if r in o.retired_requests) for r in requests}
    for request in requests:
        ref.require(retirement[request]>=last_compute[request],"request retired while its compute was unfinished")
    releases=[];records=result.raw[0].decisions
    for index,row in enumerate(records):
        if row.decision.kind!="EVICT" or not row.decision.target_id.startswith("kv:"):continue
        request=next((r for r in requests if (":"+r+":") in row.decision.target_id),None)
        if request not in row.observation.retired_requests:continue
        following=records[index+1].observation if index+1<len(records) else final
        before={m.domain:m.used_bytes for m in row.observation.memory}
        after={m.domain:m.used_bytes for m in following.memory}
        domain=row.decision.location
        ref.require(after[domain]<=before[domain]-32,"retired KV EVICT did not release its physical 32-byte allocation")
        remaining=[e.target_id for e in events.values() if e.start_ns>row.observation.t_ns
                   and scope_map.get(ops[e.target_id].invocation_id,{}).get("request_id") not in (request,None)]
        if remaining:
            releases.append(dict(request_id=request,tensor_id=row.decision.target_id,location=domain,
                t_ns=row.observation.t_ns,released_bytes=before[domain]-after[domain],later_other_request_operations=remaining))
    ref.require(releases,"no actual retired-request KV release followed by another request's progress")
    return dict(scope_operation_counts=counts,request_retired_ns=retirement,
                request_last_compute_end_ns=last_compute,
                request_arrival_ns={r:d["arrival_ns"] for r,d in requests.items()},retired_kv_release_witnesses=releases)


def asynchronous_progress(result,scope_map):
    ops,_,events=observed(result);witnesses=[]
    for op in ops.values():
        own=scope_map.get(op.invocation_id)
        if own is None or own["step_index"]<1 or op.kind!="projection":continue
        a=events[op.id]
        for other in ops.values():
            foreign=scope_map.get(other.invocation_id)
            if foreign is None or foreign["request_id"]==own["request_id"] or foreign["step_index"]>=own["step_index"]:continue
            b=events[other.id]
            overlap=min(a.end_ns,b.end_ns)-max(a.start_ns,b.start_ns)
            if overlap>0:witnesses.append(dict(next_round_operation=op.id,earlier_round_other_request=other.id,overlap_ns=overlap))
    ref.require(witnesses,"no request makes next-round progress while another request is still in an earlier round")
    return witnesses


def execution_overlap(result,scope_map):
    ops,_,events=observed(result);witnesses=[]
    for cpu in ops.values():
        if cpu.kind not in {"attention","attention_core","attention_qk","attention_softmax","attention_pv"} or events[cpu.id].device!="cpu":continue
        a=events[cpu.id];own=scope_map[cpu.invocation_id]["request_id"]
        for gpu in ops.values():
            if gpu.kind not in {"dense_ffn","expert_ffn","shared_expert_ffn"} or events[gpu.id].device!="gpu":continue
            if scope_map[gpu.invocation_id]["request_id"]==own:continue
            b=events[gpu.id];overlap=min(a.end_ns,b.end_ns)-max(a.start_ns,b.start_ns)
            if overlap>0:witnesses.append(dict(cpu=cpu.id,gpu=gpu.id,overlap_ns=overlap))
    ref.require(witnesses,"no positive-duration cross-request CPU Attention/GPU FFN overlap")
    return witnesses


def pd_handoff(result, *, requests,scope_map,destination="dram",kv_bytes_per_token=32):
    ops,tensors,events=observed(result);checked=[]
    raw_ops=result.raw[1].execution_document()["workload"]["operations"]
    for request,declared in requests.items():
        copies=[];kv_ids=[]
        for layer in range(declared["layers"]):
            for position in range(declared["prefill_tokens"]):
                tid=f"kv:layer{layer}:{request}:{position}";kv_ids.append(tid)
                ref.equal(tensors[tid].storage_bytes,kv_bytes_per_token,"handwritten KV geometry")
                found=[e for e in result.events if e.kind=="COPY" and e.target_id==tid and e.src=="vram" and e.dst==destination]
                ref.equal(len(found),1,"each prefill KV handoff copied exactly once")
                ref.equal(found[0].bytes_transferred,kv_bytes_per_token,"handoff payload bytes")
                copies.append(found[0])
        if not declared["decode_steps"]:continue
        kind=scope_map[f"{request}-prefill-layer{declared['layers']-1}"]["kind"]
        input_port="in.hidden" if kind=="dense" else "in"
        first_decode_input=f"g:{request}-decode0-layer0:t:layer0:{request}:{input_port}"
        gates=[o for o in raw_ops if o["output"]==first_decode_input]
        ref.equal(len(gates),1,"one first-Decode input gate")
        last="ffn" if kind=="dense" else "combine"
        prefill_hidden=f"g:{request}-prefill-layer{declared['layers']-1}:t:layer{declared['layers']-1}:{request}:{last}"
        ref.equal(gates[0]["inputs"],[prefill_hidden,*kv_ids],"first Decode gate consumes final Prefill hidden plus every layer/token KV")
        ready=max(e.end_ns for e in copies)
        prefill=[events[o.id] for o in ops.values() if scope_map.get(o.invocation_id,{}).get("request_id")==request and o.phase=="prefill"]
        prefill_done=max(e.end_ns for e in prefill)
        decode=[events[o.id] for o in ops.values() if scope_map.get(o.invocation_id,{}).get("request_id")==request and o.phase=="decode" and o.kind!="state_update"]
        ref.require(decode,"handoff declared with no Decode work")
        ref.require(min(e.start_ns for e in decode)>=max(ready,prefill_done),"Decode begins before full-layer KV handoff/Prefill completion")
        checked.append(dict(request_id=request,kv_ids=kv_ids,kv_bytes=len(kv_ids)*kv_bytes_per_token,
            gate_inputs=[prefill_hidden,*kv_ids],prefill_done_ns=prefill_done,
            all_layers_kv_ready_ns=ready,first_decode_start_ns=min(e.start_ns for e in decode)))
        expected_history={f"kv:layer{layer}:{request}:{p}" for layer in range(declared["layers"])
                          for p in range(declared["prefill_tokens"]+declared["decode_steps"])}
        actual_history={tid for tid in tensors if tid.startswith("kv:") and (":"+request+":") in tid}
        ref.equal(actual_history,expected_history,"persistent KV history grows to the exact finite length")
        for tid in expected_history:
            ref.equal(tensors[tid].storage_bytes,kv_bytes_per_token,"preserved growing KV block bytes")
            ref.equal(len([o for o in ops.values() if tid in o.outputs]),1,"stable KV identity has one producer")
    return checked


def expert_visibility(result):
    checked=0
    for row in result.raw[0].decisions:
        for op in row.observation.operations:
            if op.kind not in {"expert_ffn","expert_gate","expert_up","expert_swiglu","expert_down","expert_sum"} or not isinstance(op.expert_id,int):continue
            ref.require(op.work_items,"dynamic expert region lacks row ownership")
            for request,position in op.work_items:
                ref.require(any(route.request_id==request and route.layer_id==op.layer_id
                    and route.absolute_position==position and op.expert_id in route.expert_ids
                    for route in row.observation.route_history),"expert publicly visible before its row route is revealed")
        if row.decision.kind=="COMPUTE":
            op=next(o for o in row.observation.operations if o.id==row.decision.target_id)
            if op.kind=="expert_ffn":checked+=1
    ref.require(checked>0,"no revealed expert compute checked")
    return checked


def prefetch_witnesses(result,scope_map):
    ops,tensors,events=observed(result)
    audits=[r for r in result.raw[2].audit_document()["decisions"] if r["reason"]=="cross_layer_public_history_prefetch"]
    witnesses=[]
    for row in audits:
        target=row.get("evidence",{}).get("tensor")
        for event in result.events:
            if event.kind!="COPY" or event.target_id!=target or event.start_ns!=row["t_ns"]:continue
            ref.require(target.startswith("w:layer1:"),"two-layer prefetch target must belong to next layer")
            active=[o.id for o in ops.values() if o.layer_id=="layer0" and events[o.id].start_ns<=event.start_ns<events[o.id].end_ns]
            if active:
                ref.equal(event.bytes_transferred,tensors[target].storage_bytes,"speculative COPY payload")
                witnesses.append(dict(tensor_id=target,bytes=event.bytes_transferred,start_ns=event.start_ns,active_operations=active))
    ref.require(witnesses,"no audited charged cross-layer prefetch overlaps active computation")
    return witnesses


def canonical_prefix(result, *, before_ns=None, before_route_position=None):
    prefix=[]
    for row in result.raw[0].decisions:
        if before_ns is not None and row.observation.t_ns>=before_ns:break
        if before_route_position is not None and any(r.absolute_position>=before_route_position for r in row.observation.route_history):break
        prefix.append(dict(observation=asdict(row.observation),decision=asdict(row.decision)))
    ref.require(prefix,"empty causal comparison prefix")
    return prefix
