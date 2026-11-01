"""Controlled, uncalibrated design-space experiments; no hardware ranking.

Cold and GPU-resident states are materialized before compilation. The resident
case puts immutable weights in VRAM only, leaves input in DRAM and KV empty,
and does not subtract measured/simulated initialization from any latency.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.hardware_sensitivity import stable_hash, write_json, write_csv
from experiments.hardware_system_campaign import execution_source_manifest as base_manifest
from experiments.strategy_comparison import frozen_evidence, traffic_metrics
from experiments.strategy_comparison_metrics import timeline_metrics
from experiments.strategy_comparison_campaign import verify_row_metrics
from llm_simulator import (AnalyticalCosts, Cache, Device, Hardware, InvalidInput,
                          Model, Prefetch, Route, Scheduling, Simulator, Unsupported, Workload)

VERSION = "paper-design-study-0.1"
GIB = 2**30
GEOMETRY = dict(layers=2, hidden_size=6144, intermediate_size=16384, heads=24, kv_heads=6)
HARDWARE_POINTS = tuple((capacity, link) for capacity in (2, 4, 12) for link in (2, 8, 24))
WARNING = "Hypothetical analytical sensitivity, not measured hardware or proven optimal search."


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    result = base_manifest()
    result[Path(__file__).relative_to(ROOT).as_posix()] = sha_file(__file__)
    return result


@dataclass(frozen=True)
class Case:
    workload: str
    decode_steps: int
    initial_state: str
    vram_gib: int
    link_gbs: int
    candidate: str

    @property
    def group(self):
        return f"{self.workload}-d{self.decode_steps}-{self.initial_state}-v{self.vram_gib}-b{self.link_gbs}"

    @property
    def key(self):
        return self.group, self.candidate


def candidate_ids(workload):
    return tuple(f"e{e}-s{s}" for e in range(4 if workload == "dense" else 5) for s in range(2))


def jobs(mode):
    if mode == "pilot":
        return tuple(Case(*values) for values in (
            ("dense", 1, "cold", 2, 8, "e0-s1"),
            ("dense", 8, "gpu_resident", 4, 8, "e0-s0"),
            ("dense", 1, "gpu_resident", 2, 8, "e0-s0"),
            ("moe_hot", 8, "cold", 4, 8, "e4-s1"),
            ("moe_rotating", 8, "gpu_resident", 12, 8, "e4-s0"),
            ("moe_rotating", 32, "cold", 4, 8, "e0-s0")))
    if mode not in ("main", "long"):
        raise ValueError("mode must be pilot/main/long")
    return tuple(Case(w, d, state, capacity, link, candidate)
        for w in ("dense", "moe_hot", "moe_rotating")
        for d in ((1,) if mode == "main" else (8, 32))
        for state in ("cold", "gpu_resident")
        for capacity, link in (HARDWARE_POINTS if mode == "main" else ((12, 8),))
        for candidate in candidate_ids(w))


def inputs(case):
    model = (Model.dense(**GEOMETRY) if case.workload == "dense" else
             Model.moe(**GEOMETRY, experts=4, top_k=2))
    routes = tuple(Route("request0", f"layer{layer}", p,
        (0, 1) if case.workload == "moe_hot" else ((p+layer)%4, (p+layer+1)%4))
        for layer in range(2) for p in range(8+case.decode_steps)) if model.kind == "moe" else ()
    workload = Workload(prefill_tokens=8, decode_steps=case.decode_steps, routes=routes,
        route_source=f"declared_synthetic_{case.workload}" if routes else None)
    hardware = Hardware(cpu=Device(0.6e12, 50e9, 500), gpu=Device(8e12, 300e9, 5000),
        dram_capacity_bytes=32*GIB, vram_capacity_bytes=case.vram_gib*GIB,
        transfer_bandwidth_bytes_per_second=case.link_gbs*1e9,
        ssd_read_bandwidth_bytes_per_second=3e9, ssd_write_bandwidth_bytes_per_second=2e9,
        transfer_latency_ns=1000)
    e, s = (int(part[1:]) for part in case.candidate.split("-"))
    ffn = "dense_ffn" if model.kind == "dense" else "expert_ffn"
    policy = Scheduling(default_device="cpu" if e == 1 else "gpu",
        placements={2: {ffn: "cpu"}, 3: {"attention": "cpu"}}.get(e, {}),
        expert_rule="current_cost" if e == 4 else "fixed", release="last_use",
        cache=Cache(rule="lru", budget_bytes=hardware.vram_capacity_bytes),
        prefetch=Prefetch(next_layers=1, max_count=2, include_dense_weights=True) if s else Prefetch())
    return model, hardware, workload, policy


def weight_bytes(model):
    # Use emitted graph weights, independently checked against tensor catalogue.
    from llm_simulator.analytical import _layer_document
    version = "m2-layer-0.5" if model.kind == "dense" else "m2-layer-0.6"
    return sum(w["storage_bytes"] for w in _layer_document(model, "layer0", version)["weights"]) * model.layers


@dataclass(frozen=True)
class InitialStateCosts(AnalyticalCosts):
    initial_state: str = "cold"

    def build_inputs(self, directory, model, hardware, workload):
        result = super().build_inputs(directory, model, hardware, workload)
        directory = Path(directory)
        if self.initial_state == "gpu_resident":
            for path in directory.glob("*.layer.json"):
                document = read_json(path)
                for weight in document["weights"]:
                    weight["initial_location"] = "vram"
                write_json(path, document)
            if model.kind == "moe":
                for path in directory.glob("*.json"):
                    document = read_json(path)
                    if "state_in" in document:
                        for copy in document["state_in"].get("weight_copies", []):
                            copy["locations"] = ["vram"]
                        write_json(path, document)
        result["metadata"]["storage"] = (
            "F32 immutable weights initially VRAM only; inputs DRAM; empty KV at first round; later KV continuous"
            if self.initial_state == "gpu_resident" else "F32 immutable weights and inputs initially DRAM; empty KV")
        result["metadata"]["initial_state_experiment"] = self.initial_state
        write_json(directory / "analytical-metadata.json", result["metadata"])
        return result


def run_case(raw_case, output):
    case = Case(**raw_case) if isinstance(raw_case, dict) else raw_case
    folder = Path(output) / "cases" / case.group / case.candidate
    folder.mkdir(parents=True, exist_ok=False)
    model, hardware, workload, policy = inputs(case)
    row = dict(**asdict(case), artifact_directory=folder.relative_to(output).as_posix(),
               success=False, completed_makespan_ns=None, calibrated=False, hardware_measurement=False,
               initial_weight_bytes=weight_bytes(model), total_latency_boundary="finite P8 plus D rounds and mandatory DRAM hidden/KV goals",
               resident_boundary="all weights VRAM only; prepositioning excluded by declared initial condition, not subtracted")
    declaration = dict(case=asdict(case), model=asdict(model), hardware=asdict(hardware),
                       workload=asdict(workload), policy=asdict(policy))
    row["shared_input_sha256"] = stable_hash({k:v for k,v in declaration.items() if k not in ("case", "policy")})
    write_json(folder / "declarations.json", declaration)
    start = time.perf_counter()
    if case.initial_state == "gpu_resident" and weight_bytes(model) > hardware.vram_capacity_bytes:
        row.update(status="initial_capacity_infeasible", reason="all immutable weights exceed VRAM before any compute")
    else:
        try:
            simulator = Simulator(model=model, hardware=hardware, workload=workload,
                cost_model=InitialStateCosts(workspace_bytes=0, initial_state=case.initial_state))
            with simulator.compile(scheduling=policy, directory=folder/"inputs") as compiled:
                write_json(folder/"compiled-ir.json", compiled.ir_document())
                result = compiled.run()
                row.update(result.summary())
                row.pop("input_directory", None)
                events = [asdict(event) for event in result.events]
                observation = result.raw[1].observe()
                write_json(folder/"events.json", events)
                write_json(folder/"tensor-catalogue.json", [asdict(t) for t in observation.tensors])
                write_json(folder/"operation-catalogue.json", [asdict(o) for o in observation.operations])
                write_json(folder/"policy-audit.json", result.raw[2].audit_document())
                row["event_sha256"] = stable_hash(events)
                row["event_counts"] = dict(Counter(e.kind for e in result.events))
                if result.success:
                    row["completed_makespan_ns"] = result.makespan_ns
                    row.update(timeline_metrics(result.events, result.makespan_ns))
                    row.update(traffic_metrics(result))
                else:
                    row["diagnostic_partial_makespan_ns"] = result.makespan_ns
        except (Unsupported, InvalidInput) as error:
            row.update(status="unsupported_or_invalid", reason=str(error), error_type=type(error).__name__)
    row["harness_wall_seconds"] = time.perf_counter()-start
    verify_row_metrics(row)
    write_json(folder/"summary.json", row)
    write_json(folder/"artifacts.json", {p.relative_to(folder).as_posix():sha_file(p) for p in sorted(folder.rglob("*")) if p.is_file()})
    return row


def audit(records, expected, output):
    """Replay saved raw metrics and initialization without calling runtime/modeling."""
    actual = [(Case(**{k:r[k] for k in Case.__dataclass_fields__}).key) for r in records]
    assert len(actual) == len(set(actual)) and set(actual) == {c.key for c in expected}
    count, initial_checks = 0, 0
    inputs_per_group = defaultdict(set)
    for row in records:
        case = Case(**{k:row[k] for k in Case.__dataclass_fields__})
        folder = Path(output)/row["artifact_directory"]
        inputs_per_group[case.group].add(row["shared_input_sha256"])
        for path, digest in read_json(folder/"artifacts.json").items():
            assert sha_file(folder/path) == digest, path
        if not row["success"]:
            assert row["completed_makespan_ns"] is None
            continue
        events = read_json(folder/"events.json")
        tensors = read_json(folder/"tensor-catalogue.json")
        assert stable_hash(events) == row["event_sha256"]
        assert sum(t["storage_bytes"] for t in tensors if t["role"] == "weight") == row["initial_weight_bytes"]
        location = "vram" if case.initial_state == "gpu_resident" else "dram"
        for tensor in tensors:
            if tensor["role"] == "weight":
                assert tensor["initial_locations"] == [location], tensor["id"]
                initial_checks += 1
        assert max(e["end_ns"] for e in events) == row["completed_makespan_ns"]
        for src, dst, metric in (("dram","vram","h2d_bytes"),("vram","dram","d2h_bytes")):
            assert sum(e["bytes_transferred"] for e in events if e["kind"] == "COPY" and e["src"] == src and e["dst"] == dst) == row[metric]
        assert row["peak_vram_bytes"] <= case.vram_gib*GIB
        count += len(events)
    assert all(len(values) == 1 for values in inputs_per_group.values())
    return dict(pass_=True, cases=len(records), replayed_events=count, weight_initial_location_checks=initial_checks,
        checks=["complete matrix", "file hashes", "same within-group workload/hardware", "event makespan", "raw transfer payloads", "initial weight locations and bytes", "VRAM physical peak"])


def analyze(records):
    groups=defaultdict(list)
    for row in records:
        case=Case(**{k:row[k] for k in Case.__dataclass_fields__})
        groups[case.group].append(row)
    outcomes=[]
    def best(rows):
        valid=[r for r in rows if r["success"]]
        return min(valid,key=lambda r:(r["completed_makespan_ns"],r["candidate"])) if valid else None
    for group, rows in sorted(groups.items()):
        joint=best(rows)
        if joint is None:
            outcomes.append(dict(group=group, feasible=False, statuses=dict(Counter(r["status"] for r in rows))))
            continue
        execution=best([r for r in rows if r["candidate"].endswith("s0")])
        storage=best([r for r in rows if r["candidate"].startswith("e0-")])
        chosen_e=execution["candidate"].split("-")[0] if execution else None
        stagewise=best([r for r in rows if r["candidate"].startswith(chosen_e+"-")]) if chosen_e else None
        choices={}
        for name,result in (("joint",joint),("execution_only_s0",execution),("storage_only_e0",storage),("stagewise_execution_then_storage",stagewise)):
            choices[name]=None if result is None else dict(candidate=result["candidate"],latency_ns=result["completed_makespan_ns"],
                regret_vs_finite_best=(result["completed_makespan_ns"]/joint["completed_makespan_ns"]-1),
                evaluations={"joint":len(rows),"execution_only_s0":len(rows)//2,"storage_only_e0":2,
                    "stagewise_execution_then_storage":len(rows)//2+1}[name])
        center=next((rs for rs in groups.values() if rs[0]["workload"]==joint["workload"] and
            rs[0]["decode_steps"]==joint["decode_steps"] and rs[0]["initial_state"]==joint["initial_state"] and
            rs[0]["vram_gib"]==12 and rs[0]["link_gbs"]==8),None)
        center_best=best(center) if center else None
        frozen=next((r for r in rows if center_best and r["candidate"]==center_best["candidate"]),None)
        outcomes.append(dict(group=group, feasible=True, workload=joint["workload"],decode_steps=joint["decode_steps"],
            initial_state=joint["initial_state"],vram_gib=joint["vram_gib"],link_gbs=joint["link_gbs"], choices=choices,
            tied_best=[r["candidate"] for r in rows if r["success"] and r["completed_makespan_ns"]==joint["completed_makespan_ns"]],
            center_frozen_candidate=center_best["candidate"] if center_best else None,
            center_frozen_regret=None if not frozen or not frozen["success"] else frozen["completed_makespan_ns"]/joint["completed_makespan_ns"]-1,
            center_frozen_status="center_infeasible_or_absent" if center_best is None else ("success" if frozen and frozen["success"] else "candidate_infeasible")))
    return dict(warning=WARNING,groups=outcomes,
        scope="Finite candidate optimum; joint contains restricted choices by construction. Gains measure magnitude, not search algorithm superiority.",
        stagewise="choose execution at s0, then optimize storage for that execution; duplicate s0 evaluation reused",
        center="12 GiB, 8 GB/s; no center policy invented where resident initial state is infeasible")


def protocol(cases, mode):
    return dict(version=VERSION, mode=mode, warning=WARNING, declared_cases=len(cases),jobs=[asdict(c) for c in cases],
        geometry=GEOMETRY,hardware_points=HARDWARE_POINTS,precision="F32",workspace_bytes=0,
        workspace_provenance="explicit simplified assumption, not a measured zero", prefilling_tokens=8,
        initial_states=dict(cold="all immutable weights DRAM; input DRAM; empty KV",
            gpu_resident="all immutable weights VRAM only, charged in capacity; input DRAM; empty KV; infeasible initial states retained"),
        final_goals="same DRAM hidden and KV views; optional cached weight copies may differ",
        policy_descriptions=dict(e0="all GPU",e1="all CPU",e2="CPU FFN/experts, GPU others",e3="CPU attention, GPU others",
            e4="MoE causal current_cost expert placement, GPU others",s0="LRU without prefetch",s1="same LRU plus next-layer prefetch max 2"),
        limitations=["No task partition experiment: independently priced split stage contracts absent from analytical facade",
            "All performance rates synthetic; no physical hardware ranking or calibrated prediction claim",
            "GPU-resident condition is an initial placement comparison, not a steady-state decoding throughput metric",
            "No simulated token sampling or numerical inference", "No statistical CI for deterministic scenario calculations"],
        comparison_methods=["joint", "fixed storage s0/execution only", "fixed execution e0/storage only", "execution at s0 then storage", "center fixed policy vs reselect"])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--mode",choices=("pilot","main","long"),default="pilot")
    parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args(argv)
    if not 1<=args.workers<=2: parser.error("workers must be 1 or 2")
    output=args.out.resolve()
    if output.exists() and any(output.iterdir()): parser.error("output must be new or empty")
    output.mkdir(parents=True,exist_ok=True)
    cases=jobs(args.mode); before=sources(); evidence=frozen_evidence(); declaration=protocol(cases,args.mode)
    write_json(output/"protocol.json",declaration);write_json(output/"source-manifest-before.json",before)
    start=time.perf_counter();records=[]
    with (output/"runs.jsonl").open("w",encoding="utf-8",newline="\n") as stream:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(run_case,asdict(case),str(output)):case for case in cases}
            for future in as_completed(futures):
                row=future.result();records.append(row)
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n");stream.flush()
                print(f"{len(records)}/{len(cases)} {futures[future].group}/{row['candidate']} {row['success']} wall={time.perf_counter()-start:.1f}s",flush=True)
    records.sort(key=lambda r:(Case(**{k:r[k] for k in Case.__dataclass_fields__}).key))
    write_json(output/"records.json",records);write_csv(output/"runs.csv",records)
    verification=audit(records,cases,output);write_json(output/"independent-artifact-audit.json",verification)
    write_json(output/"analysis.json",analyze(records));after=sources();write_json(output/"source-manifest-after.json",after)
    assert before==after and evidence==frozen_evidence(), "sources or frozen accuracy evidence changed"
    completion=dict(complete=True,cases=len(cases),success=sum(r["success"] for r in records),
        statuses=dict(Counter(r.get("status") for r in records)),elapsed_wall_seconds=time.perf_counter()-start,
        sources_unchanged=True,accuracy_evidence_unchanged=True,protocol_sha256=stable_hash(declaration),
        records_sha256=stable_hash(records),audit=verification)
    write_json(output/"completion.json",completion);print(json.dumps(completion,ensure_ascii=False),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
