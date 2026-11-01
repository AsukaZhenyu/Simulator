"""Hand-authored finite DAG and physical action replay; no simulator imports.

The expected graph is constructed from a frozen scenario, never compiler output.
This is a correct low-level baseline, including gates, copies, resources and
capacity. Durations supplied in the observed scenario are independently replayed,
not asserted to be measured hardware prices.
"""
from __future__ import annotations
from collections import Counter
import math


def need(ok,message):
    if not ok:raise AssertionError(message)


def scope(r,step,l):
    return f"request{r}-{'prefill' if step==0 else 'decode'+str(step-1)}-layer{l}"


def tid(r,step,l,suffix):
    return f"g:{scope(r,step,l)}:t:layer{l}:request{r}:{suffix}"


def oid(r,step,l,suffix,request=None):
    return f"g:{scope(r,step,l)}:op:layer{l}:{'request'+str(r) if request is None else request}:{suffix}"


def kv(r,l,p):return f"kv:layer{l}:request{r}:{p}"


def weight(l,name):return f"w:layer{l}:{name}"


def declared_dag(s):
    """Specify exact identities/shapes/operands independently of compiler helpers."""
    tensors={};ops={};contexts={}
    h,i,d=s["hidden"],s["intermediate"],s["head_dim"]
    def tensor(identity,shape,dtype="GGML_TYPE_F32",role="intermediate",initial=False):
        value=dict(id=identity,ne=[*shape,*([1]*(4-len(shape)))],dtype=dtype,
            storage_bytes=4*math.prod(shape),role=role,initial_locations=["dram"] if initial else [])
        need(identity not in tensors or tensors[identity]==value,"handwritten tensor collision")
        tensors[identity]=value
        return identity
    def operation(identity,kind,inputs,outputs,params=(),context=None):
        need(identity not in ops,"handwritten op collision")
        ops[identity]=dict(id=identity,semantic_op=kind,inputs=list(inputs),output=outputs[0],
            extra_outputs=list(outputs[1:]),op_params=list(params))
        if context is not None:contexts[identity]=context
        return identity
    for l in range(s["layers"]):
        if s["kind"]=="dense":
            shapes={"W_input_norm":(h,),"W_q":(h,h),"W_k":(h,d),"W_v":(h,d),
                "W_output":(h,h),"W_post_attention_norm":(h,),"W_gate":(h,i),"W_up":(h,i),"W_down":(i,h)}
            for name,shape in shapes.items():
                if s["ffn_refined"] and s["pattern"]!="A12" and name in {"W_gate","W_up","W_down"}:
                    for shard,start,end in (("host",0,i//4),("accelerator",i//4,i)):
                        leafshape=(end-start,h) if name=="W_down" else (h,end-start)
                        tensor(weight(l,name)+":ffn:"+shard,leafshape,role="weight",initial=True)
                else:tensor(weight(l,name),shape,role="weight",initial=True)
        else:
            tensor(weight(l,"W_proj"),(3*h*h,),role="weight",initial=True)
            tensor(weight(l,"W_router"),(3*h,),role="weight",initial=True)
            for e in range(3):tensor(weight(l,"E"+str(e)),(3*h*i,),role="weight",initial=True)
    terminal=[]
    for r in range(s["requests"]):
        previous=None
        for step in range(s["decode_steps"]+1):
            n=s["q"] if step==0 else 1;past=0 if step==0 else s["q"]+step-1
            input_suffix="in.hidden" if s["kind"]=="dense" else "in"
            first_input=tid(r,step,0,input_suffix)
            tensor(first_input,(h,n) if s["kind"]=="dense" else (h*n,))
            positions=[]
            if s["kind"]=="dense":
                for l in range(s["layers"]):positions.append(tensor(tid(r,step,l,"in.positions"),(n,),"GGML_TYPE_I32"))
            gate=f"g:request{r}:request_arrival" if step==0 else f"g:request{r}-request{r}-decode{step-1}:trace_next:request{r}"
            gateinputs=[] if step==0 else [previous]
            if step==1 and s["handoff"]:
                gateinputs+=sorted(kv(r,l,p) for l in range(s["layers"]) for p in range(s["q"]))
            operation(gate,"state_update",gateinputs,[first_input,*positions],context=dict(r=r,step=step,l=0,n=n,past=past))
            current=first_input
            for l in range(s["layers"]):
                context=dict(r=r,step=step,l=l,n=n,past=past)
                history=[kv(r,l,p) for p in range(past)]
                new=[tensor(kv(r,l,p),(2*d,),role="output") for p in range(past,past+n)]
                attn=tensor(tid(r,step,l,"attn"),(h,n) if s["kind"]=="dense" else (h*n,))
                if s["kind"]=="dense":
                    ports=[tensor(tid(r,step,l,"proj."+p),(d,n,2 if p=="q" else 1)) for p in ("q","k","v")]
                    operation(oid(r,step,l,"proj"),"projection",[weight(l,name) for name in ("W_input_norm","W_q","W_k","W_v")]+[current,positions[l]],ports,context=context)
                    if s["attention_refined"]:
                        scores=tensor(tid(r,step,l,"attn.scores"),(past+n,n,2))
                        prob=tensor(tid(r,step,l,"attn.probabilities"),(past+n,n,2))
                        ctx=tensor(tid(r,step,l,"attn.context"),(h,n))
                        operation(oid(r,step,l,"attn.qk"),"attention_qk",[*ports[:2],*history],[scores],context=context)
                        operation(oid(r,step,l,"attn.softmax"),"attention_softmax",[scores],[prob],context=context)
                        operation(oid(r,step,l,"attn.pv"),"attention_pv",[prob,*ports[1:],*history],[ctx,*new],context=context)
                        operation(oid(r,step,l,"attn.output"),"attention_output",[ctx,weight(l,"W_output"),current],[attn],context=context)
                    else:
                        operation(oid(r,step,l,"attn"),"attention",[weight(l,"W_output"),*ports,current,*history],[attn,*new],context=context)
                    final=tensor(tid(r,step,l,"ffn"),(h,n),role="output")
                    if s["ffn_refined"]:
                        cut=i*3//4 if r else i//4
                        norm=tensor(tid(r,step,l,"ffn.ffn.norm"),(h,n))
                        operation(oid(r,step,l,"ffn.ffn.norm"),"ffn_norm",[weight(l,"W_post_attention_norm"),attn],[norm],context=context)
                        partials=[]
                        for shard,start,end in (("host",0,cut),("accelerator",cut,i)):
                            base="ffn.ffn."+shard
                            vals={p:tensor(tid(r,step,l,base+"."+p),(end-start,n)) for p in ("gate","up","swiglu")}
                            for stage in ("gate","up"):
                                w=weight(l,"W_"+stage)+(":ffn:"+shard if s["pattern"]!="A12" else "")
                                operation(oid(r,step,l,base+"."+stage),"ffn_"+stage,[w,norm],[vals[stage]],(start,end),context)
                            operation(oid(r,step,l,base+".swiglu"),"ffn_swiglu",[vals["gate"],vals["up"]],[vals["swiglu"]],(start,end),context)
                            part=tensor(tid(r,step,l,base+".partial"),(h,n));partials.append(part)
                            w=weight(l,"W_down")+(":ffn:"+shard if s["pattern"]!="A12" else "")
                            operation(oid(r,step,l,base+".down"),"ffn_down",[w,vals["swiglu"]],[part],(start,end),context)
                        total=tensor(tid(r,step,l,"ffn.ffn.sum"),(h,n))
                        operation(oid(r,step,l,"ffn.ffn.sum"),"ffn_sum",partials,[total],(2,),context)
                        operation(oid(r,step,l,"ffn.ffn.residual"),"ffn_residual",[total,attn],[final],context=context)
                    else:operation(oid(r,step,l,"ffn"),"dense_ffn",[weight(l,name) for name in ("W_post_attention_norm","W_gate","W_up","W_down")]+[attn],[final],context=context)
                else:
                    proj=tensor(tid(r,step,l,"proj"),(h*n,))
                    operation(oid(r,step,l,"proj"),"projection",[weight(l,"W_proj"),current],[proj],context=context)
                    operation(oid(r,step,l,"attn"),"attention",[proj,current,*history],[attn,*new],context=context)
                    route=tensor(tid(r,step,l,"router"),(1,n),"GGML_TYPE_I32")
                    operation(oid(r,step,l,"router"),"router",[weight(l,"W_router"),attn],[route],context=context)
                    grouped={}
                    for p in range(past,past+n):
                        e=p%2 if p<s["q"] else s["future"]
                        grouped.setdefault(e,[]).append(p)
                    expertouts=[]
                    for e,positions_e in sorted(grouped.items()):
                        out=tensor(f"g:{scope(r,step,l)}:t:layer{l}:__moe_batch__:expert.{e}",(h*len(positions_e),))
                        expertouts.append(out)
                        operation(oid(r,step,l,"expert."+str(e),request="__moe_batch__"),"expert_ffn",[weight(l,"E"+str(e)),attn,route],[out],context=dict(**context,expert=e,work_positions=positions_e))
                    final=tensor(tid(r,step,l,"combine"),(h*n,),role="output")
                    operation(oid(r,step,l,"combine"),"combine",[attn,route,*expertouts],[final],context=context)
                current=final
                if step==s["decode_steps"] and l==s["layers"]-1:terminal.append(final)
            previous=current
    return dict(schema="e1-handwritten-dag-v1",tensors=tensors,operations=ops,contexts=contexts,
        terminal_hidden=terminal,expected_all_kv=[k for k in tensors if k.startswith("kv:")],
        dependency_contract="exact tensor identities; one immutable producer; no inferred compiler expectations")


def check_graph(s,expected,graph,events):
    tensors={t["id"]:t for t in graph["workload"]["tensors"]}
    operations={o["id"]:o for o in graph["workload"]["operations"]}
    need(set(tensors)==set(expected["tensors"]),"tensor inventory differs: "+str(set(tensors)^set(expected["tensors"])))
    need(set(operations)==set(expected["operations"]),"operation inventory differs: "+str(set(operations)^set(expected["operations"])))
    for identity,wanted in expected["tensors"].items():
        actual=tensors[identity]
        for field in ("ne","dtype","storage_bytes"):
            need(actual[field]==wanted[field],f"tensor {identity} {field} differs")
        need(actual.get("initial_locations",[])==wanted["initial_locations"],f"tensor {identity} initial residency differs")
    for identity,wanted in expected["operations"].items():
        actual=operations[identity]
        for field in ("semantic_op","inputs","output","op_params"):
            need(actual[field]==wanted[field],f"op {identity} {field} differs: {actual[field]!r} != {wanted[field]!r}")
        need(actual.get("extra_outputs",[])==wanted["extra_outputs"],f"op {identity} extra outputs differ")
    computes={e["target_id"]:e for e in events if e["kind"]=="COMPUTE"}
    need(len(computes)==len(operations),"each intended compute must execute once")
    producer={tid:oid for oid,op in expected["operations"].items() for tid in [op["output"],*op["extra_outputs"]]}
    edges=0
    for oid_,op in expected["operations"].items():
        for ti in op["inputs"]:
            if ti in producer:
                need(computes[oid_]["start_ns"]>=computes[producer[ti]]["end_ns"],"dependency violated")
                edges+=1
    return dict(exact_tensors=len(tensors),exact_operations=len(operations),exact_dependency_edges=edges,
        independently_authored_expected_dag=True,partially_ordered=True)


def replay_physical(s,expected,graph,events,makespan):
    """Full payload and allocation replay with independent discrete/shared ledger."""
    tensors=expected["tensors"];ops=expected["operations"]
    scenario=graph["scenario"];prices=scenario["costs"];goals=scenario["goal"]["output_locations"]
    shared=s["memory_mode"]=="shared";cap=s["physical_capacity_bytes"]
    status={};refs={};allocation=0;running=[];done=set();now=0;peak=Counter();events_count=Counter()
    for tid_,t in tensors.items():
        for loc in t["initial_locations"]:
            status[tid_,loc]="READY";refs[tid_,loc]=allocation;allocation+=1
    def usage():
        if shared:
            by={a:t for (t,l),a in refs.items()};return {"shared":sum(tensors[t]["storage_bytes"] for t in by.values())+sum(r["workspace"] for r in running)}
        return {loc:sum(tensors[t]["storage_bytes"] for (t,l) in refs if l==loc)+sum(r["workspace"] for r in running if r["loc"]==loc) for loc in ("dram","vram")}
    def check():
        for loc,size in usage().items():
            capacity=s.get("vram_capacity_bytes",cap) if loc=="vram" else cap
            need(size<=capacity,"independent physical capacity overflow");peak[loc]=max(peak[loc],size)
    def finish(t):
        nonlocal running,now
        need(t>=now,"action time reversed")
        kept=[]
        for task in running:
            if task["end"]>t:kept.append(task);continue
            for dest in task["outputs"]:status[dest]="READY"
            if task.get("op"):done.add(task["op"])
        running=kept;now=t
    check()
    for event in events:
        finish(event["t_ns"]);kind=event["kind"];identity=event.get("target_id");events_count[kind]+=1
        if kind=="ADVANCE":pass
        elif kind=="EVICT":
            key=(identity,event["location"]);need(status.get(key)=="READY","evict absent/unready")
            a=refs[key];need(not any(a in r["read_allocations"] for r in running),"evict live physical reader")
            del refs[key];del status[key]
            released=0 if a in refs.values() else tensors[identity]["storage_bytes"]
            need(event.get("bytes_released",0)==released,"incorrect physical release")
        elif kind=="SHARE":
            source=(identity,event["src"]);dest=(identity,event["dst"])
            need(shared and status.get(source)=="READY" and dest not in status,"invalid share")
            need(event["start_ns"]==event["end_ns"]==now and event["bytes_transferred"]==0,"share charged payload")
            status[dest]="READY";refs[dest]=refs[source]
        elif kind in {"COMPUTE","COPY"}:
            if kind=="COMPUTE":
                op=ops[identity];loc="dram" if event["device"]=="cpu" else "vram"
                inputs=[(x,loc) for x in op["inputs"]];outputs=[(x,loc) for x in [op["output"],*op["extra_outputs"]]]
                need(identity not in done and not any(r.get("op")==identity for r in running),"duplicate compute")
                price=prices["compute"][identity][event["device"]];workspace=price.get("workspace_bytes",0)
                resource=event["device"]+"_compute"
            else:
                inputs=[(identity,event["src"])];outputs=[(identity,event["dst"])];loc=event["dst"]
                route=event["src"]+"_to_"+event["dst"]
                price=prices["transfer"][route][identity];workspace=0;resource=event["resource"]
                need(event["bytes_transferred"]==tensors[identity]["storage_bytes"],"copy physical payload differs")
            need(all(status.get(x)=="READY" for x in inputs),"action reads unready/wrong version")
            need(all(x not in status for x in outputs),"action overwrites live tensor")
            need(not any(r["resource"]==resource for r in running),"exclusive resource overlaps")
            need(event["start_ns"]==now and event["end_ns"]-now==price["duration_ns"],"price/timestamp differs from independent replay")
            reads={refs[x] for x in inputs}
            for output in outputs:
                refs[output]=allocation;allocation+=1;status[output]="RESERVED"
            running.append(dict(end=event["end_ns"],outputs=outputs,read_allocations=reads,workspace=workspace,
                loc=loc,resource=resource,op=identity if kind=="COMPUTE" else None))
        else:raise AssertionError("unsupported action in independent replay: "+kind)
        check()
    finish(makespan);check()
    need(not running and len(done)==len(ops),"unfinished independent replay")
    need(all(status.get((t,loc))=="READY" for t,loc in goals.items()),"terminal goals differ")
    return dict(events=dict(events_count),completed_operations=len(done),terminal_goals=len(goals),
        peak_physical_bytes=dict(peak),makespan_ns=now,exact_tick_agreement=True,
        note="Replay uses independently authored identities/allocations and frozen declared prices; not a hardware accuracy test.")
