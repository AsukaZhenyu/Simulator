"""Reevaluate bounded coordination loss with independently scheduled DAGs.

The primary compiler executions are retained, frozen analytical traces. This
module actually reruns the independent sixteen-priority reference for every
admitted candidate, and recomputes the tiny Dense optimum certificates. Ten
fresh compiler runs check the selected candidate of each group; they are a
separate reproducibility check, never substituted into the frozen global min.
No native CPU/GPU execution or profile fitting takes place.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from experiments.ir_value import campaign as ref
from experiments.ir_value import exact_oracle


DEFAULT_SOURCE = ref.ROOT / "build/ir-value-20261029-final-evidence"
DEFAULT_OUT = ref.ROOT / "build/coordination-metrics-20261030/lowering"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vhash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write("\n")


def source_manifest():
    paths=[Path(__file__),Path(ref.__file__),Path(exact_oracle.__file__)]
    paths += [ref.ROOT/"experiments"/p for p in
              ("dac_validation_e1.py","dac_validation_e1_oracle.py","dac_validation_e1_numeric.py")]
    return {str(p.resolve()):sha(p) for p in paths}


def production_manifest():
    return {str(p.resolve()):sha(p) for folder in
            (ref.ROOT/"src/llm_simulator",ref.ROOT/"mapping/tensor_mapping",ref.ROOT/"modeling/llm_infer_model")
            for p in folder.rglob("*.py")}


def stats(values):
    vals=np.asarray(values,dtype=float)
    if not len(vals):raise ValueError("empty candidate distribution")
    return dict(count=len(vals),minimum_percent=float(vals.min()),median_percent=float(np.median(vals)),
                p90_percent=float(np.quantile(vals,.9)),maximum_percent=float(vals.max()),
                negative_count=int((vals < -1e-12).sum()),zero_count=int((abs(vals)<=1e-12).sum()),
                positive_count=int((vals > 1e-12).sum()))


def verify_schedule(nodes, events):
    """Independent obligations, release times and nonpreemptive resources."""
    byid={e["id"]:e for e in events}
    if len(byid)!=len(events) or set(byid)!=set(nodes):
        raise AssertionError("schedule inventory is not the declared independent DAG")
    for identity,node in nodes.items():
        event=byid[identity];start=event["start_ns"]
        if event["machine"]!=node["machine"] or event["end_ns"]-start!=node["duration"]:
            raise AssertionError("resource or component cost changed")
        if start<node["release"]:
            raise AssertionError("release obligation violated")
        if any(byid[p]["end_ns"]>start for p in node["deps"]):
            raise AssertionError("mandatory producer/copy guard violated")
        alternatives=node.get("dependency_alternatives",[[]])
        if not any(all(byid[p]["end_ns"]<=start for p in alt) for alt in alternatives):
            raise AssertionError("shared-copy demand OR guard violated")
    machines=defaultdict(list)
    for event in events:
        if event["end_ns"]>event["start_ns"]:
            machines[event["machine"]].append(event)
    for group in machines.values():
        order=sorted(group,key=lambda x:(x["start_ns"],x["end_ns"],x["id"]))
        for a,b in zip(order,order[1:]):
            if a["end_ns"]>b["start_ns"]:
                raise AssertionError("simultaneous work exceeds declared resource")
    return max(e["end_ns"] for e in events)


def map_retained_trace(nodes, events):
    compute={e["target_id"]:e for e in events if e["kind"]=="COMPUTE"}
    copies=defaultdict(list)
    for event in events:
        if event["kind"]=="COPY":
            key=(event["target_id"],event["src"],event["dst"])
            copies[key].append(event)
    output=[]
    for identity,node in nodes.items():
        if node["kind"]=="compute":
            event=compute.pop(identity)
            if event["device"]!=node["machine"]:
                raise AssertionError("fixed strategy placement changed")
        else:
            key=(node["tensor"],"dram" if node["src"]=="cpu" else "vram",
                 "dram" if node["dst"]=="cpu" else "vram")
            matches=copies.pop(key)
            if len(matches)!=1 or matches[0]["bytes_transferred"]!=node["bytes"]:
                raise AssertionError("physical COPY inventory/payload differs")
            event=matches[0]
        output.append(dict(id=identity,machine=node["machine"],start_ns=event["start_ns"],end_ns=event["end_ns"]))
    if compute or copies:
        raise AssertionError("retained trace contains undeclared compute/copies")
    return output


def metrics(rows, selected_id):
    """Both minima use exactly the same admitted P; only coordination varies."""
    selected=next(r for r in rows if r["id"]==selected_id)
    compiler_best=min(r["compiler_duration_ns"] for r in rows)
    if selected["compiler_duration_ns"]!=compiler_best:
        raise ValueError("frozen selected strategy is not a compiler-time minimum")
    reference_best=min(r["reference_family_best_ns"] for r in rows)
    selected_reference=selected["reference_family_best_ns"]
    lower=100*(compiler_best/reference_best-1)
    choice=100*(selected_reference/reference_best-1)
    selected_gap=100*(compiler_best/selected_reference-1)
    if abs((1+lower/100)-(1+choice/100)*(1+selected_gap/100))>1e-12:
        raise AssertionError("gap decomposition failed")
    return dict(compiler_best_ns=compiler_best, reference_family_global_best_ns=reference_best,
                selected_reference_best_ns=selected_reference,
                lower_signed_relative_gap_percent=lower, choice_regret_percent=choice,
                selected_candidate_lowering_signed_gap_percent=selected_gap,
                selected_id=selected_id,
                candidate_gap_distribution=stats([r["candidate_signed_gap_percent"] for r in rows]))


def baseline_sets(rows):
    """Original Table II restrictions; not the later MoE strong union."""
    execution=[r for r in rows if r["storage"]=="demand"]
    storage=[r for r in rows if set(r["choices"].values()) in ({"cpu"},{"gpu"})]
    union={r["id"] for r in execution+storage}
    e=min(execution,key=lambda r:(r["compiler_duration_ns"],r["id"]))
    s=min(storage,key=lambda r:(r["compiler_duration_ns"],r["id"]))
    return dict(execution_only_candidates=len(execution),storage_only_candidates=len(storage),
                isolated_union_candidates=len(union),
                execution_only_best_ns=e["compiler_duration_ns"],execution_only_best_id=e["id"],
                storage_only_best_ns=s["compiler_duration_ns"],storage_only_best_id=s["id"],
                better_isolated_best_ns=min(e["compiler_duration_ns"],s["compiler_duration_ns"]))


def fresh_check(row, s, dag, nodes, out):
    folder=out/"fresh-selected"/row["id"]
    before=time.perf_counter_ns()
    compiled=ref.build(s,dag,row["choices"],row["storage"],folder)
    compile_ns=time.perf_counter_ns()-before
    with compiled:
        save(folder/"compiled-ir.json",compiled.ir_document())
        before=time.perf_counter_ns();result=compiled.run();run_ns=time.perf_counter_ns()-before
        if not result.success:raise AssertionError("fresh selected run failed: "+row["id"])
        result.save(folder/"execution")
        graph=result.raw[1].execution_document();events=[asdict(x) for x in result.events]
        save(folder/"graph.json",graph);save(folder/"events.json",events)
        before=time.perf_counter_ns()
        graph_check=ref.check_graph(s,dag,graph,events)
        physical=ref.replay_physical(s,dag,graph,events,result.makespan_ns)
        numerical=ref.check_numerical(s,dag,graph,events);numerical.pop("outputs",None)
        observed=verify_schedule(nodes,map_retained_trace(nodes,events))
        if observed!=result.makespan_ns:raise AssertionError("fresh completed boundary mismatch")
        verification_ns=time.perf_counter_ns()-before
    record=dict(id=row["id"],retained_compiler_duration_ns=row["duration_ns"],
                fresh_duration_ns=result.makespan_ns,
                matches_retained_duration=result.makespan_ns==row["duration_ns"],
                compile_wall_ns=compile_ns,run_wall_ns=run_ns,verification_wall_ns=verification_ns,
                graph=graph_check,physical=physical,numerical=numerical,
                primary_frozen_results_replaced=False,
                timing_scope="One analytical compile including its input materialization; simulator run excludes report serialization; verification separate; no native timing")
    save(folder/"result.json",record)
    return record


def run(source=DEFAULT_SOURCE,out=DEFAULT_OUT,*,fresh=True):
    source,out=Path(source).resolve(),Path(out).resolve()
    if out.exists():raise ValueError("new lowering output directory required")
    source_report=source/"REPORT.json";original=read(source_report)
    valid=[r for r in original["rows"] if r["status"]=="pass"]
    rejected=[r for r in original["rows"] if r["status"]!="pass"]
    expected_error="first Decode Attention placement conflicts with the phase handoff destination"
    if any(r.get("error")!=expected_error for r in rejected):
        raise ValueError("unexpected old failure must be reviewed before constructing admitted P")
    groups={(r["scenario"],r["regime"]):r for r in original["summary"]}
    if len(groups)!=10 or len(valid)!=800 or len(rejected)!=128:
        raise ValueError("retained source is not the declared five-structure/two-cost matrix")
    source_before=source_manifest();production_before=production_manifest()
    raw_paths={source_report}
    for row in valid:
        folder=Path(row["artifact_directory"])
        raw_paths.update(folder/name for name in ("events.json","graph.json","result.json","compiled-ir.json"))
        raw_paths.add(Path(row["corrected_reference_path"]))
    raw_hashes={str(p.resolve()):sha(p) for p in sorted(raw_paths)}
    out.mkdir(parents=True)
    protocol=dict(schema="coordination-lowering-metric-protocol-1",created=datetime.now(timezone.utc).isoformat(),
        primary_evidence="Frozen retained compiler traces; fresh independent analytical reference reevaluation",
        source_directory=str(source),source_report_sha256=sha(source_report),
        source_hashes=source_before,raw_input_hashes="RAW_INPUT_HASHES.json",
        target_grid=928,admitted_candidates=800,fixed_contract_rejections=128,
        scenarios=[s[0] for s in ref.SCENARIOS],regimes=ref.BYTE_NS,
        geometry=dict(hidden=8,intermediate=16,layers=1,F32=True),
        fixed="Within each candidate, placement, preparation, work DAG, physical allocation/copy inventory, component prices and CPU/GPU/sharedPCIe resources are identical; vary resource-local coordination order only",
        reference_family=dict(size=16,priority_pairs=ref.PRIORITIES,
                              method="independently generated non-delay schedules; same shared-copy OR-readiness guards"),
        definitions=dict(L_lower="min_p T(C(p)) / min_{p,a in A(p)} T(a) - 1",
                         R_choice="min_{a in A(p_hat)} T(a) / min_{p,a in A(p)} T(a) - 1; p_hat frozen from compiler timing",
                         candidate_gap="T(C(p))/min_{a in A_16(p)}T(a)-1",
                         non_dense_scope="signed relative gap to sixteen-priority family; not exhaustive optimality loss",
                         dense_scope="global denominator independently certified only for the32 Dense candidates and fixed inventory"),
        frozen_choices={s+"/"+r:g["best"] for (s,r),g in groups.items()},
        fresh_check="Exactly one frozen selected candidate per group; separate source version and timing; never substituted in global minima" if fresh else "disabled",
        native_measurements=False,parameter_fitting=False,production_edits=False,
        old_source_provenance=dict(retained_campaigns=original.get("retained_campaigns"),
                                   corrective_rerun=original.get("corrective_rerun"),
                                   reference_correction=original.get("reference_correction")))
    save(out/"PROTOCOL.json",protocol);save(out/"RAW_INPUT_HASHES.json",raw_hashes)
    save(out/"CURRENT_PRODUCTION_SOURCES_BEFORE.json",production_before)
    descriptors={s[0]:s for s in ref.SCENARIOS};rows=[];fresh_results=[];reference_wall=0
    start=time.perf_counter()
    for (name,regime),old_group in groups.items():
        desc=descriptors[name];s=ref.scenario(*desc,regime);dag=ref.declared_dag(s)
        save(out/"declared-dags"/(s["id"]+".json"),dag)
        subset=[r for r in valid if (r["scenario"],r["regime"])==(name,regime)]
        for row in subset:
            before=time.perf_counter_ns()
            devices=ref.placement(s,dag,row["choices"])
            nodes=ref.reference_nodes(s,dag,devices,row["storage"])
            schedules=[]
            for priority in ref.PRIORITIES:
                candidate=ref.schedule(nodes,priority)
                if verify_schedule(nodes,candidate["events"])!=candidate["duration_ns"]:
                    raise AssertionError("reference completion boundary differs")
                schedules.append(dict(priority=list(priority),**candidate))
            reference_wall+=time.perf_counter_ns()-before
            best=min(range(len(schedules)),key=lambda i:schedules[i]["duration_ns"])
            best_ns=schedules[best]["duration_ns"]
            old_ref=read(row["corrected_reference_path"])
            nodes_match=vhash(nodes)==vhash(old_ref["nodes"])
            duration_match=best_ns==row["reference_family_best_ns"]
            if not nodes_match or not duration_match:
                raise AssertionError("independent reference no longer matches frozen inputs: "+row["id"])
            original_events=read(Path(row["artifact_directory"])/"events.json")
            trace=map_retained_trace(nodes,original_events)
            if verify_schedule(nodes,trace)!=row["duration_ns"]:
                raise AssertionError("retained compiler trace violates common reference boundary")
            record=dict(nodes=nodes,best_priority=list(ref.PRIORITIES[best]),
                        schedule_durations_ns=[x["duration_ns"] for x in schedules],
                        best=schedules[best],all_schedules=schedules,
                        retained_compiler_trace=trace)
            rp=out/"references"/row["id"]/"independent-reference.json";save(rp,record)
            if name=="dense":
                case_name=row["id"].removeprefix(name+"-"+regime+"-")
                save(out/"dense-certificate-inputs"/s["id"]/case_name/"independent-reference.json",record)
            entry=dict(id=row["id"],scenario=name,regime=regime,
                choices=row["choices"],storage=row["storage"],compiler_duration_ns=row["duration_ns"],
                reference_family_best_ns=best_ns,
                candidate_signed_gap_percent=100*(row["duration_ns"]/best_ns-1),
                reference_distinct_durations=len(set(x["duration_ns"] for x in schedules)),
                reference_path=str(rp),reference_sha256=sha(rp),
                retained_artifact_directory=row["artifact_directory"],
                nodes_match_retained_corrected_reference=nodes_match,
                reference_duration_matches_retained=duration_match,
                independent_trace_obligations_checked=True)
            rows.append(entry)
            if fresh and row["id"]==old_group["best"]:
                fresh_results.append(fresh_check(row,s,dag,nodes,out))
        print("lowering",name,regime,len(subset),"candidates /",len(subset)*16,"independent schedules",flush=True)
    # Recompute exact global Dense certificates; do not just quote old results.
    dense_certificates=exact_oracle.certify(out/"dense-certificate-inputs")
    certificates={r["scenario_id"]:r for r in dense_certificates["summary"]}
    if len(certificates)!=2 or not all(r["global_optimum_certified"] for r in certificates.values()):
        raise AssertionError("tiny Dense global certificate unresolved")
    summary=[]
    for (name,regime),old_group in groups.items():
        subset=[r for r in rows if (r["scenario"],r["regime"])==(name,regime)]
        item=dict(scenario=name,regime=regime,transfer_ns_per_byte=ref.BYTE_NS[regime],
                  admitted_candidates=len(subset),**metrics(subset,old_group["best"]),
                  table_ii_baselines=baseline_sets(subset))
        if name=="dense":
            certificate=certificates[name+"-"+regime]
            if certificate["upper_bound_ns"]!=item["reference_family_global_best_ns"]:
                raise AssertionError("Dense reference denominator differs from certificate")
            item.update(global_denominator_certified=True,dense_exact_certificate=certificate,
                        metric_scope="L_lower is certified for this bounded Dense global comparison; individual candidate gaps still use A16 unless separately certified")
        else:
            item.update(global_denominator_certified=False,
                        metric_scope="Signed relative gap to A16 priority family, not global-optimal lowering loss")
        summary.append(item)
    source_after=source_manifest();production_after=production_manifest()
    raw_after={path:sha(path) for path in raw_hashes}
    save(out/"INPUT_SOURCE_INTEGRITY.json",dict(raw_inputs_unchanged=raw_hashes==raw_after,
        experiment_sources_unchanged=source_before==source_after,
        current_production_sources_unchanged=production_before==production_after,
        experiment_sources_after=source_after,current_production_sources_after=production_after))
    if raw_hashes!=raw_after or source_before!=source_after or production_before!=production_after:
        raise AssertionError("inputs or source changed during analytical reevaluation")
    sorted_gaps=sorted(rows,key=lambda r:(r["candidate_signed_gap_percent"],r["id"]))
    result=dict(schema="coordination-lowering-metrics-1",protocol_sha256=sha(out/"PROTOCOL.json"),
        native_measurements=False,analytical_reference_reexecuted=True,
        primary_compiler_traces="Frozen ir-value-20261029-final-evidence (retained original/corrected campaigns)",
        admitted_candidates=len(rows),reference_schedule_executions=len(rows)*len(ref.PRIORITIES),
        fixed_contract_rejections=len(rejected),summary=summary,candidates=rows,
        all_candidate_signed_gap_distribution=stats([r["candidate_signed_gap_percent"] for r in rows]),
        largest_positive_examples=list(reversed(sorted_gaps[-5:])),largest_negative_examples=sorted_gaps[:5],
        fresh_recompile=dict(count=len(fresh_results),unchanged_duration_count=sum(r["matches_retained_duration"] for r in fresh_results),
                             rows=fresh_results,primary_results_replaced=False),
        dense_certificate_file=str(out/"dense-certificate-inputs/exact-dense-certificates/REPORT.json"),
        timing=dict(independent_reference_compute_seconds=reference_wall/1e9,
                    whole_derivation_wall_seconds=time.perf_counter()-start,
                    fresh_compile_seconds=sum(r["compile_wall_ns"] for r in fresh_results)/1e9,
                    fresh_run_seconds=sum(r["run_wall_ns"] for r in fresh_results)/1e9,
                    peak_host_ram_bytes=None),
        observations=["Best-of-space closeness does not bound the loss of every individual strategy",
                      "A negative per-candidate signed gap means the16priority family misses a better compiler ordering",
                      "A16 candidate selection regret is nonnegative but not regret to unknown exhaustive schedules",
                      "No native performance or universal expression-completeness claim follows"],
        baseline_definition=dict(E_only="All admitted placements with preparation fixed to demand; reoptimize placement",
            S_only="Demand/spare preparation within fixed uniform allCPU or allGPU placements; choose best admitted one",
            isolated_union="union(E_only,S_only), not a new third independent baseline",
            joint="all admitted placement x preparation candidates",
            distinction="The later MoE E5xS3 B_STRONG additionally includes shape-aware and resident-path hybrid families and is a different workload/implementation domain"))
    save(out/"SUMMARY.json",result)
    paper=[]
    for row in summary:
        d=row["candidate_gap_distribution"];b=row["table_ii_baselines"]
        paper.append(dict(structure=row["scenario"],copy_ns_per_byte=row["transfer_ns_per_byte"],
                          admitted=row["admitted_candidates"],compiler_best_us=row["compiler_best_ns"]/1000,
                          reference_best_us=row["reference_family_global_best_ns"]/1000,
                          L_lower_signed_percent=row["lower_signed_relative_gap_percent"],
                          R_choice_percent=row["choice_regret_percent"],
                          selected_lowering_percent=row["selected_candidate_lowering_signed_gap_percent"],
                          candidate_min_percent=d["minimum_percent"],candidate_median_percent=d["median_percent"],
                          candidate_p90_percent=d["p90_percent"],candidate_max_percent=d["maximum_percent"],
                          E_only_us=b["execution_only_best_ns"]/1000,S_only_us=b["storage_only_best_ns"]/1000,
                          dense_global_certified=row["global_denominator_certified"]))
    save(out/"PAPER_TABLE.json",paper)
    import csv
    with (out/"PAPER_TABLE.csv").open("x",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(paper[0]));writer.writeheader();writer.writerows(paper)
    save(out/"OUTPUT_HASHES.json",{str(p.relative_to(out)):sha(p) for p in out.rglob("*") if p.is_file()})
    print(json.dumps(dict(groups=len(summary),candidates=len(rows),schedules=len(rows)*16,
                          fresh_checks=len(fresh_results),distribution=result["all_candidate_signed_gap_distribution"],
                          output=str(out)),indent=2),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=DEFAULT_SOURCE)
    parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--no-fresh",action="store_true")
    args=parser.parse_args();run(args.source,args.out,fresh=not args.no_fresh)
