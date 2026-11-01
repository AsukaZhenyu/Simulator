"""Enumerate a frozen finite family of legal default-action-DAG lowerings.

No kernel/copy coalescing, cache replacement, placement or prefetch-window change
is allowed. We enumerate all 24 permutations of four ready-action classes plus the production
default, retain full payloads and release dependencies, and validate every legal
trace against an independent physical ledger AND the production runtime. This is
a finite, deliberately conservative residual subspace, not all possible schedules.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import traceback

from experiments.dac_validation_e1_oracle import need,declared_dag,check_graph,replay_physical

PRIORITIES=tuple("_".join(p) for p in itertools.permutations(("cpu","gpu","copy","evict")))


def action_dag(s,expected,graph,events):
    """Infer only observed-version lifetime edges; expected operands are independent.

    Preserve the default's memory-command order, start epochs, cache choices and
    copy inventory. End edges protect readers/producers; start edges preserve the
    same declared visible prefetch epoch. Each edge points backward in the default
    trace, providing an inspectable certificate that the default belongs to L(p).
    """
    nodes=[];writers={};readers={};absent_after={};previous_memory=None;last_compute=None
    for ti,t in expected["tensors"].items():
        for loc in t["initial_locations"]:writers[ti,loc]=None;readers[ti,loc]=[]
    ops=expected["operations"]
    for e in events:
        if e["kind"]=="ADVANCE":continue
        j=len(nodes);kind=e["kind"];target=e.get("target_id");end=set();start=set();inputs=[];outputs=[]
        if kind=="COMPUTE":
            op=ops[target];loc="dram" if e["device"]=="cpu" else "vram"
            inputs=[(ti,loc) for ti in op["inputs"]];outputs=[(ti,loc) for ti in [op["output"],*op["extra_outputs"]]]
        elif kind=="COPY":
            inputs=[(target,e["src"])];outputs=[(target,e["dst"])]
        elif kind=="EVICT":
            key=(target,e["location"])
            if writers.get(key) is not None:end.add(writers[key])
            end.update(readers.get(key,[]));writers.pop(key,None);readers.pop(key,None);absent_after[key]=j
        else:raise AssertionError("finite residual space excludes "+kind)
        for key in inputs:
            need(key in writers,"default reads absent version during DAG extraction")
            if writers[key] is not None:end.add(writers[key])
            readers.setdefault(key,[]).append(j)
        for key in outputs:
            if key in absent_after:end.add(absent_after[key])
            writers[key]=j;readers[key]=[]
        if kind in {"COPY","EVICT"}:
            if previous_memory is not None:start.add(previous_memory)
            previous_memory=j
            if last_compute is not None:start.add(last_compute)
        if kind=="COMPUTE":last_compute=j
        # An opaque expert's *weight* is public but this family freezes demand
        # loading, so its future-route-dependent copy cannot precede revelation.
        if kind=="COPY" and target and ":E" in target:
            uses=[(index,n) for index,n in enumerate(nodes) if n["event"]["kind"]=="COMPUTE" and
                expected["operations"][n["event"]["target_id"]]["semantic_op"]=="router"]
            if uses:end.add(uses[-1][0])
        context=expected["contexts"].get(target,{})
        release=2000*context.get("r",0) if kind=="COMPUTE" and target.endswith(":request_arrival") else 0
        duration=0 if kind=="EVICT" else e["end_ns"]-e["start_ns"]
        nodes.append(dict(id=j,event=deepcopy(e),end_dependencies=sorted(end),
            start_dependencies=sorted(start),inputs=[list(x) for x in inputs],outputs=[list(x) for x in outputs],
            release_ns=release,duration_ns=duration))
    for node in nodes:
        need(all(i<node["id"] for i in node["end_dependencies"]+node["start_dependencies"]),"non-causal default dependency extraction")
    return nodes


def reschedule(s,expected,nodes,priority):
    """Independent finite event scheduler with full physical allocation checks."""
    remaining=set(range(len(nodes)));started={};done=set();running=[];now=0;events=[]
    tensors=expected["tensors"];present={(t,loc) for t,v in tensors.items() for loc in v["initial_locations"]}
    ready=set(present)
    def current_usage():
        return {loc:sum(tensors[t]["storage_bytes"] for t,l in present if l==loc)+
            sum(r["workspace"] for r in running if r["loc"]==loc) for loc in ("dram","vram")}
    def rank(index):
        e=nodes[index]["event"]
        tag=e.get("device") if e["kind"]=="COMPUTE" else "evict" if e["kind"]=="EVICT" else "copy"
        order=priority.split("_");return (order.index(tag) if tag in order else 3,e.get("target_id") or "",index)
    def finish():
        nonlocal running
        keep=[]
        for r in running:
            if r["end"]>now:keep.append(r);continue
            ready.update(r["outputs"]);done.add(r["node"])
        running=keep
    for _ in range(20*len(nodes)+100):
        finish()
        if not remaining and not running:break
        available=[]
        used=current_usage()
        for index in remaining:
            node=nodes[index];e=node["event"];kind=e["kind"]
            if node["release_ns"]>now or not set(node["end_dependencies"])<=done or not set(node["start_dependencies"])<=set(started):continue
            if any(tuple(key) not in ready for key in node["inputs"]):continue
            if kind=="EVICT":
                if (e["target_id"],e["location"]) not in ready:continue
                available.append(index);continue
            resource=e["resource"]
            if any(r["resource"]==resource for r in running):continue
            outputs=[tuple(x) for x in node["outputs"]]
            if any(x in present for x in outputs):continue
            loc=("dram" if e["device"]=="cpu" else "vram") if kind=="COMPUTE" else e["dst"]
            extra=sum(tensors[t]["storage_bytes"] for t,_ in outputs)+e.get("workspace_bytes",0)
            capacity=s.get("vram_capacity_bytes",s["physical_capacity_bytes"]) if loc=="vram" else s["physical_capacity_bytes"]
            if used[loc]+extra>capacity:continue
            available.append(index)
        if available:
            index=min(available,key=rank);node=nodes[index];e=deepcopy(node["event"]);kind=e["kind"]
            remaining.remove(index);started[index]=now;e.update(action_index=len(events),t_ns=now,start_ns=now,end_ns=now+node["duration_ns"])
            if kind=="EVICT":
                key=(e["target_id"],e["location"]);present.remove(key);ready.remove(key);done.add(index)
            else:
                outputs=[tuple(x) for x in node["outputs"]];present.update(outputs)
                loc=("dram" if e["device"]=="cpu" else "vram") if kind=="COMPUTE" else e["dst"]
                running.append(dict(node=index,end=e["end_ns"],outputs=outputs,resource=e["resource"],
                    loc=loc,workspace=e.get("workspace_bytes",0)))
            events.append(e);continue
        future=[r["end"] for r in running if r["end"]>now]
        future.extend(nodes[j]["release_ns"] for j in remaining if nodes[j]["release_ns"]>now)
        if not future:return dict(status="infeasible",reason="No legal finite ready action or future completion",events=events)
        now=min(future)
        events.append(dict(action_index=len(events),kind="ADVANCE",target_id=None,resource=None,t_ns=now,
            start_ns=None,end_ns=now,bytes_transferred=0,workspace_bytes=0,bytes_released=0,device=None,src=None,dst=None,location=None))
    else:return dict(status="timeout",reason="Finite scheduler step bound exceeded",events=events)
    return dict(status="pass",duration_ns=now,events=events)


def verify_runtime(compiled,events,*,future_position=None):
    """Replay exactly the rescheduled commands through unmodified production runtime."""
    from llm_simulator import Decision
    from tensor_mapping.causal import compile_command
    from tensor_mapping.unified_strategy import create_session
    from llm_infer_model.tensor.runtime import dispatch_command
    session=create_session(compiled.plan._compiled);prefix=[]
    for e in events:
        if e["kind"]=="ADVANCE":continue
        for _ in range(10000):
            obs=session.observe()
            if obs.t_ns>=e["t_ns"]:break
            step=dispatch_command(session,compile_command(Decision("ADVANCE"),obs))
            need(step.ok,"production timer rejected finite lowering")
        obs=session.observe();need(obs.t_ns==e["t_ns"],"runtime time differs from finite lowering")
        decision=Decision(e["kind"],e["target_id"],device=e.get("device"),src=e.get("src"),dst=e.get("dst"),location=e.get("location"))
        if future_position is not None and not any(r.absolute_position>=future_position for r in obs.route_history):
            prefix.append(dict(observation=asdict(obs),decision=asdict(decision)))
        step=dispatch_command(session,compile_command(decision,obs))
        need(step.ok,"production runtime rejected finite lowering: "+str(step.error_code)+" "+str(step.error_message))
    final=max(e["end_ns"] for e in events)
    for _ in range(10000):
        obs=session.observe()
        if obs.t_ns>=final:break
        step=dispatch_command(session,compile_command(Decision("ADVANCE"),obs))
        need(step.ok,"production finish timer rejected")
    result=session.result()
    need(result.evaluation.status=="valid","production result is not valid")
    need(session.observe().complete,"finite lowering did not reach production goal")
    return dict(status="pass",commands=len([e for e in events if e["kind"]!="ADVANCE"]),
        duration_ns=session.observe().t_ns,causal_prefix=prefix)


def scenarios():
    from experiments.dac_validation_e1 import spec
    for graph_type,pattern in (("partition_merge","A03"),("cross_layer_prefetch","A04"),
            ("two_requests","A09"),("causal_expert_cache","A06")):
        for regime in ("compute_constrained","transfer_constrained","capacity_constrained"):
            candidates=[]
            for c in range(4):
                s=spec(pattern,8,1)
                s.update(regime=regime,projection_device="cpu" if c&1 else "gpu",
                    attention_device="gpu" if c&2 else "cpu")
                if graph_type=="causal_expert_cache":s.update(layers=1,decode_steps=1)
                if graph_type=="cross_layer_prefetch":s.update(decode_steps=0)
                if regime=="capacity_constrained":
                    s.update(vram_capacity_bytes=3072,cache_budget_bytes=3072)
                candidates.append((f"P{c}",s))
            yield graph_type,regime,candidates


def run_lowering(output):
    from experiments.dac_validation_e1 import build,sha
    from experiments.strategy_fidelity import write_json
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    declarations=[dict(graph_type=g,regime=r,candidates=[dict(id=i,scenario=s) for i,s in cs]) for g,r,cs in scenarios()]
    manifest=dict(schema="e1-finite-residual-manifest-v1",groups=declarations,priorities=PRIORITIES,
        lowerings=["production_default",*PRIORITIES],candidate_tie_break="lexicographic candidate ID",
        expert_objective="equally weighted mean of two model routes with identical public prefix; one ready-order policy shared by both routes",
        constraints="same default action multiset, physical payloads, memory action order, producer/reader lifetime edges and conservative public compute-start epochs; no kernel/coalescing/cache/prefetch/placement changes",
        limitation="Finite declared subset only: all 24 priority permutations of CPU/GPU/COPY/EVICT plus production default per candidate. Does not enumerate all continuous schedules, copy batching/coalescing, CUDA Graphs, layout or kernels.")
    write_json(output/"manifest.json",manifest);groups=[]
    for g,r,candidates in scenarios():
        group=dict(id=g+"-"+r,graph_type=g,regime=r,candidates=[])
        for identity,s in candidates:
            folder=output/group["id"]/identity;folder.mkdir(parents=True)
            row=dict(id=identity,scenario=s,lowerings=[])
            compiled=[];graphs=[];expects=[];defaults=[];dags=[]
            try:
                futures=(1,2) if g=="causal_expert_cache" else (1,)
                for future in futures:
                    local=dict(s,future=future);sub=folder/f"future-{future}"
                    expected=declared_dag(local);write_json(sub/"expected.json",expected)
                    c=build(local,sub);compiled.append(c)
                    result=c.run();need(result.success,"production default infeasible")
                    graph=result.raw[1].execution_document();events=[asdict(e) for e in result.events]
                    check_graph(local,expected,graph,events);replay_physical(local,expected,graph,events,result.makespan_ns)
                    result.save(sub/"default-execution");write_json(sub/"graph.json",graph)
                    nodes=action_dag(local,expected,graph,events);write_json(sub/"action-dag.json",nodes)
                    graphs.append(graph);expects.append(expected);defaults.append((events,result.makespan_ns));dags.append(nodes)
                for lower in ("production_default",*PRIORITIES):
                    alt=dict(id=lower,routes=[]);prefixes=[];trace_hashes=[]
                    for index,future in enumerate(futures):
                        local=dict(s,future=future)
                        schedule=dict(status="pass",events=defaults[index][0],duration_ns=defaults[index][1]) if lower=="production_default" else reschedule(local,expects[index],dags[index],lower)
                        route=dict(future=future,status=schedule["status"])
                        if schedule["status"]=="pass":
                            try:
                                route["physical"]=replay_physical(local,expects[index],graphs[index],schedule["events"],schedule["duration_ns"])
                                receipt=verify_runtime(compiled[index],schedule["events"],future_position=1 if g=="causal_expert_cache" else None)
                                prefixes.append(receipt.pop("causal_prefix"));route.update(duration_ns=schedule["duration_ns"],runtime=receipt)
                            except Exception as error:route.update(status="rejected",error=repr(error),traceback=traceback.format_exc())
                        write_json(folder/f"{lower}-route-{future}.json",dict(**route,events=schedule["events"]))
                        trace_hashes.append(hashlib.sha256(json.dumps(schedule["events"],sort_keys=True,separators=(",",":")).encode()).hexdigest())
                        alt["routes"].append(route)
                    valid=all(x["status"]=="pass" for x in alt["routes"])
                    if g=="causal_expert_cache" and valid:
                        valid=bool(prefixes[0]) and prefixes[0]==prefixes[1]
                        alt["joint_causal_prefix_equal"]=valid;alt["causal_prefix_decisions"]=len(prefixes[0])
                        write_json(folder/f"{lower}-joint-prefixes.json",[[dict(
                            observation_sha256=hashlib.sha256(json.dumps(item["observation"],sort_keys=True,separators=(",",":")).encode()).hexdigest(),
                            decision=item["decision"],t_ns=item["observation"]["t_ns"],
                            route_history=item["observation"]["route_history"]) for item in prefix] for prefix in prefixes])
                        alt["causal_prefix_sha256"]=[hashlib.sha256(json.dumps(p,sort_keys=True,separators=(",",":")).encode()).hexdigest() for p in prefixes]
                    alt["status"]="pass" if valid else "infeasible_or_rejected"
                    alt["route_trace_sha256"]=trace_hashes
                    if valid:alt["duration_ns"]=sum(x["duration_ns"] for x in alt["routes"])/len(futures)
                    row["lowerings"].append(alt)
                default=row["lowerings"][0]
                need(default["status"]=="pass","default not certified in L(p)")
                valid=[x for x in row["lowerings"] if x["status"]=="pass"]
                row.update(status="pass",default_duration_ns=default["duration_ns"],
                    best_duration_ns=min(x["duration_ns"] for x in valid),enumeration_complete=True,
                    declared_lowerings=25,enumerated_priority_policies=24,valid_lowerings=len(valid),default_in_L=True,
                    distinct_legal_trace_tuples=len({tuple(x["route_trace_sha256"]) for x in valid}),
                    distinct_legal_durations=len({x["duration_ns"] for x in valid}),
                    duration_range_ns=[min(x["duration_ns"] for x in valid),max(x["duration_ns"] for x in valid)],
                    observed_nontrivial_timing_freedom=len({x["duration_ns"] for x in valid})>1)
                row["compiler_loss"]=row["default_duration_ns"]/row["best_duration_ns"]-1
            except Exception as error:row.update(status="fail",error=repr(error),traceback=traceback.format_exc())
            finally:
                for c in compiled:c.close()
            write_json(folder/"result.json",row);group["candidates"].append(row)
        valid=[row for row in group["candidates"] if row["status"]=="pass"]
        group["enumeration_complete"]=len(valid)==4
        if valid:
            selected=min(valid,key=lambda row:(row["default_duration_ns"],row["id"]))
            best=min(valid,key=lambda row:(row["best_duration_ns"],row["id"]))
            group.update(default_selected=selected["id"],joint_selected=best["id"],
                selection_loss=selected["best_duration_ns"]/best["best_duration_ns"]-1,
                max_compiler_loss=max(row["compiler_loss"] for row in valid))
        write_json(output/group["id"]/"result.json",group);groups.append(group)
        print("lowering",group["id"],"complete" if group["enumeration_complete"] else "incomplete",group.get("selection_loss"),flush=True)
    summary=dict(schema="e1-finite-residual-summary-v1",manifest_sha256=sha(output/"manifest.json"),groups=groups,
        groups_completed=sum(x["enumeration_complete"] for x in groups),groups_total=12,
        candidates_completed=sum(c["status"]=="pass" for g in groups for c in g["candidates"]),candidates_total=48,
        max_selection_loss=max((g.get("selection_loss",0) for g in groups),default=None),
        max_compiler_loss=max((g.get("max_compiler_loss",0) for g in groups),default=None),
        ranking_changed_groups=sum(g.get("default_selected")!=g.get("joint_selected") for g in groups if g["enumeration_complete"]),
        candidates_with_nontrivial_timing_freedom=sum(c.get("observed_nontrivial_timing_freedom",False) for g in groups for c in g["candidates"]),
        scope_limitation="Only 24 fixed-inventory priority permutations plus production default. Zero selection loss cannot establish absence of backend optimization, transfer coalescing benefits, or all-schedule optimality.")
    summary["enumeration_complete"]=summary["groups_completed"]==12 and summary["candidates_completed"]==48
    summary["target_pass"]=summary["enumeration_complete"] and summary["max_selection_loss"]<=0.05
    summary["complete"]=summary["enumeration_complete"]
    write_json(output/"summary.json",summary)
    return summary
