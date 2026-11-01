"""Independent, read-only replay of the controlled paper design study.

Imports only the standard library and an independently authored stdlib event
replayer. It does not import the study runner, compiler, runtime, or cost model.
Passing establishes consistency of the retained analytical evidence, not real
hardware accuracy or numerical correctness of inference.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.hardware_system_audit import (
    AuditError, audit_trace, canonical, need, read, sha, within,
)

GIB = 2**30
FIELDS = ("workload", "decode_steps", "initial_state", "vram_gib", "link_gbs", "candidate")


def identity(row):
    return tuple(row[key] for key in FIELDS)


def expected_jobs(mode):
    """Independent finite matrix; do not obtain expected cases from the runner."""
    if mode == "pilot":
        return {
            ("dense", 1, "cold", 2, 8, "e0-s1"),
            ("dense", 8, "gpu_resident", 4, 8, "e0-s0"),
            ("dense", 1, "gpu_resident", 2, 8, "e0-s0"),
            ("moe_hot", 8, "cold", 4, 8, "e4-s1"),
            ("moe_rotating", 8, "gpu_resident", 12, 8, "e4-s0"),
            ("moe_rotating", 32, "cold", 4, 8, "e0-s0"),
        }
    need(mode in ("main", "long"), "undeclared study mode")
    points = [(v, b) for v in (2, 4, 12) for b in (2, 8, 24)] if mode == "main" else [(12, 8)]
    return {(workload, decode, state, vram, link, f"e{e}-s{s}")
            for workload in ("dense", "moe_hot", "moe_rotating")
            for decode in ((1,) if mode == "main" else (8, 32))
            for state in ("cold", "gpu_resident")
            for vram, link in points
            for e in range(4 if workload == "dense" else 5) for s in range(2)}


def check_matrix(records, protocol):
    keys = [identity(row) for row in records]
    expected = expected_jobs(protocol["mode"])
    need(len(keys) == len(set(keys)), "duplicate candidate case")
    need(set(keys) == expected, "missing or undeclared candidate case")
    need(protocol["declared_cases"] == len(expected), "declared denominator differs from matrix")
    jobs = [identity(row) for row in protocol["jobs"]]
    need(len(jobs) == len(set(jobs)) and set(jobs) == expected, "predeclared jobs differ from matrix")
    return len(expected)


def expected_weights(workload):
    """F32 asset byte oracle from H=6144, I=16384, KV hidden=1536."""
    h, intermediate, kv = 6144, 16384, 1536
    if workload == "dense":
        layer = dict(W_input_norm=4*h, W_q=4*h*h, W_k=4*h*kv,
                     W_v=4*h*kv, W_output=4*h*h, W_post_attention_norm=4*h,
                     W_gate=4*h*intermediate, W_up=4*h*intermediate,
                     W_down=4*intermediate*h)
    else:
        layer = dict(W_proj=4*(2*h*h+2*h*kv), W_router=4*h*4,
                     **{f"E{i}": 4*3*h*intermediate for i in range(4)})
    return {f"w:layer{layer_index}:{name}": amount
            for layer_index in range(2) for name, amount in layer.items()}


def check_geometry_and_initial_state(row, tensors):
    expected = expected_weights(row["workload"])
    weights = {t["id"]: t for t in tensors if t["role"] == "weight"}
    need(set(weights) == set(expected), "immutable weight asset catalogue differs from geometry")
    location = "vram" if row["initial_state"] == "gpu_resident" else "dram"
    for tid, size in expected.items():
        t = weights[tid]
        need(t["storage_bytes"] == size and dict(t["allocation_bytes"]) == {"dram": size, "vram": size},
             "immutable weight bytes differ from independent F32 geometry")
        need(t["initial_locations"] == [location], "initial weight has missing owner or free second copy")
        expected_goal = None if row["workload"] == "dense" else location
        need(t["goal_location"] == expected_goal, "immutable weight final boundary differs from declared model family")
    need(sum(expected.values()) == row["initial_weight_bytes"], "initial weight byte total differs from geometry")
    need(not any(t["initial_locations"] for t in tensors if t["id"].startswith("kv:")),
         "first round contains pre-existing KV")
    return len(expected)


def check_final_goals(row, tensors):
    catalog = {t["id"]: t for t in tensors}
    count = row["decode_steps"] + 8
    kv = {f"kv:layer{layer}:request0:{position}" for layer in range(2) for position in range(count)}
    need({tid for tid in catalog if tid.startswith("kv:")} == kv, "KV block coverage differs from P8 plus Decode")
    suffix = "ffn" if row["workload"] == "dense" else "combine"
    hidden = f"g:layer1-decode{row['decode_steps']-1}:t:layer1:request0:{suffix}"
    for tid in kv | {hidden}:
        need(tid in catalog and catalog[tid]["goal_location"] == "dram", "required final KV/hidden DRAM goal missing")
        need(dict(catalog[tid]["copies"])["dram"] == "READY", "final KV/hidden not ready in DRAM")
        need(catalog[tid]["storage_bytes"] == (12288 if tid in kv else 24576), "final KV/hidden byte count differs")
    nonweight_goals = {t["id"] for t in tensors if t.get("goal_location") and t["role"] != "weight"}
    need(nonweight_goals == kv | {hidden}, "unexpected nonweight final goal boundary")
    return len(kv) + 1


def check_graph_dependencies(operations, tensors, events):
    """Check single producers and producer completion for every declared edge."""
    producer = {}
    for op in operations:
        for tid in op["outputs"]:
            need(tid not in producer, "tensor has multiple compute producers")
            producer[tid] = op["id"]
    catalog = {t["id"]: t for t in tensors}
    computes = {e["target_id"]: e for e in events if e["kind"] == "COMPUTE"}
    edges = 0
    for op in operations:
        need(op["id"] in computes, "declared operation has no compute event")
        for tid in op["inputs"]:
            need(tid in catalog, "compute graph input absent from tensor catalogue")
            if tid in producer:
                source = producer[tid]
                need(source in computes and computes[source]["end_ns"] <= computes[op["id"]]["start_ns"],
                     "dependent computation starts before producer completion")
                edges += 1
            else:
                need(bool(catalog[tid]["initial_locations"]), "unproduced graph input is not initial data")
    return edges


def check_macro_catalogue(row, operations):
    """Verify expected stage and active-expert coverage from declared workload."""
    expected = {}
    for round_index in range(row["decode_steps"]+1):
        round_id = "prefill" if round_index == 0 else f"decode{round_index-1}"
        tokens = 8 if round_index == 0 else 1
        if round_index:
            expected[f"g:{round_id}:trace_next:request0"] = ("state_update", 1)
        for layer in range(2):
            prefix = f"g:layer{layer}-{round_id}:op:layer{layer}:"
            stages = {"proj": "projection", "attn": "attention"}
            if row["workload"] == "dense":
                stages["ffn"] = "dense_ffn"
            else:
                stages.update(router="router", combine="combine")
                if row["workload"] == "moe_hot":
                    active = {0: tokens, 1: tokens}
                elif round_index == 0:
                    active = {expert: 4 for expert in range(4)}
                else:
                    position = 8+round_index-1
                    active = {(position+layer)%4: 1, (position+layer+1)%4: 1}
                for expert, count in active.items():
                    expected[f"{prefix}__moe_batch__:expert.{expert}"] = ("expert_ffn", count)
            for stage, kind in stages.items():
                expected[f"{prefix}request0:{stage}"] = (kind, tokens)
    actual = {op["id"]: (op["kind"], op["token_count"]) for op in operations}
    need(actual == expected and len(actual) == len(operations), "macro stage/active expert/token coverage differs from independent workload")
    return len(expected)


def check_execution_devices(row, operations, events):
    execution = int(row["candidate"].split("-")[0][1:])
    by_id = {op["id"]: op for op in operations}
    checked = 0
    for event in events:
        if event["kind"] != "COMPUTE":
            continue
        kind = by_id[event["target_id"]]["kind"]
        if execution == 4 and kind == "expert_ffn":
            need(event["device"] in ("cpu", "gpu"), "dynamic expert uses undeclared device")
            continue  # Current-cost decision optimality is not reimplemented.
        cpu = (kind == "state_update" or execution == 1
               or (execution == 2 and kind in ("dense_ffn", "expert_ffn"))
               or (execution == 3 and kind == "attention"))
        need(event["device"] == ("cpu" if cpu else "gpu"), "executed device differs from declared candidate placement")
        checked += 1
    return checked


def check_declarations(row, declaration):
    need(declaration["case"] == {key: row[key] for key in FIELDS}, "declaration case identity differs")
    model, hw, wl, policy = (declaration[k] for k in ("model", "hardware", "workload", "policy"))
    expected_model = dict(kind="dense" if row["workload"] == "dense" else "moe", layers=2,
                          hidden_size=6144, intermediate_size=16384, heads=24, kv_heads=6,
                          experts=0 if row["workload"] == "dense" else 4,
                          top_k=0 if row["workload"] == "dense" else 2)
    # Dense dataclass retains its unused default expert dimensions. Only active
    # dimensions enter the geometry oracle; no MoE capability is inferred.
    need(all(model[k] == v for k, v in expected_model.items() if k not in ("experts", "top_k")),
         "model geometry differs from fixed study")
    if model["kind"] == "moe":
        need(model["experts"] == 4 and model["top_k"] == 2, "MoE active dimensions differ")
    need(hw["memory_mode"] == "separate" and hw["vram_capacity_bytes"] == row["vram_gib"]*GIB
         and hw["dram_capacity_bytes"] == 32*GIB and hw["transfer_bandwidth_bytes_per_second"] == row["link_gbs"]*1e9,
         "hardware point differs from independent matrix")
    need(hw["cpu"] == dict(compute_flops_per_second=0.6e12, memory_bandwidth_bytes_per_second=50e9, launch_overhead_ns=500)
         and hw["gpu"] == dict(compute_flops_per_second=8e12, memory_bandwidth_bytes_per_second=300e9, launch_overhead_ns=5000),
         "uncontrolled compute or memory bandwidth change")
    need(wl["prefill_tokens"] == 8 and wl["decode_steps"] == row["decode_steps"] and wl["batch_size"] == 1,
         "workload horizon differs from declared P8/D count")
    expected_routes = []
    if row["workload"] != "dense":
        for layer in range(2):
            for position in range(8+row["decode_steps"]):
                pair = [0, 1] if row["workload"] == "moe_hot" else [(position+layer)%4,(position+layer+1)%4]
                expected_routes.append(dict(request_id="request0",layer_id=f"layer{layer}",absolute_position=position,expert_ids=pair,scores=None))
    need(wl["routes"] == expected_routes, "expert route trace differs from fixed workload")
    e, s = (int(part[1:]) for part in row["candidate"].split("-"))
    ffn = "dense_ffn" if row["workload"] == "dense" else "expert_ffn"
    expected_policy = dict(placements={2:{ffn:"cpu"},3:{"attention":"cpu"}}.get(e,{}),request_placements={},
        expert_rule="current_cost" if e == 4 else "fixed",release="last_use",
        cache=dict(rule="lru",unit="tensor",budget_bytes=hw["vram_capacity_bytes"],alpha=None,top_p=None),
        prefetch=dict(next_layers=s,max_count=2*s,ranking="history_frequency",include_dense_weights=bool(s)),
        transport=dict(engine=None,stage_rules=[],slot_reuse="release_unlocked_backed_copy",boundary_release="declared_dead_intermediates"),
        reconfiguration=None,default_device="cpu" if e == 1 else "gpu",expert_order="graph_order",request_pipeline=None,phase_handoffs=[])
    need(policy == expected_policy, "candidate policy differs from declared execution/storage coordinates")
    need(policy["cache"]["budget_bytes"] == hw["vram_capacity_bytes"], "cache budget differs from physical VRAM")
    need(row["shared_input_sha256"] == canonical({key: value for key, value in declaration.items() if key not in ("case", "policy")}),
         "within-group input hash differs")


def check_selection_analysis(records, analysis):
    """Re-evaluate finite selection and evaluation budgets from all raw rows."""
    groups = defaultdict(list)
    for row in records:
        key = (f"{row['workload']}-d{row['decode_steps']}-{row['initial_state']}"
               f"-v{row['vram_gib']}-b{row['link_gbs']}")
        groups[key].append(row)
    emitted = {row["group"]: row for row in analysis["groups"]}
    need(set(emitted) == set(groups) and len(emitted) == len(analysis["groups"]), "analysis omitted or duplicated candidate group")
    def choose(rows):
        feasible = [r for r in rows if r["success"]]
        return min(feasible,key=lambda r:(r["completed_makespan_ns"],r["candidate"])) if feasible else None
    checks = 0
    for key, rows in groups.items():
        actual = emitted[key]
        joint = choose(rows)
        need(actual["feasible"] == (joint is not None), "analysis changed group feasibility")
        if joint is None:
            need(actual["statuses"] == dict(Counter(r["status"] for r in rows)), "infeasible group denominator changed")
            continue
        execution = choose([r for r in rows if r["candidate"].endswith("-s0")])
        storage = choose([r for r in rows if r["candidate"].startswith("e0-")])
        stagewise = choose([r for r in rows if execution and r["candidate"].split("-")[0] == execution["candidate"].split("-")[0]])
        expected = (("joint",joint,len(rows)),("execution_only_s0",execution,len(rows)//2),
                    ("storage_only_e0",storage,2),("stagewise_execution_then_storage",stagewise,len(rows)//2+1))
        for method, winner, evaluations in expected:
            item = actual["choices"][method]
            need((item is None) == (winner is None), "analysis discards failed restricted selection")
            if winner is not None:
                need(item == dict(candidate=winner["candidate"],latency_ns=winner["completed_makespan_ns"],
                                  regret_vs_finite_best=winner["completed_makespan_ns"]/joint["completed_makespan_ns"]-1,
                                  evaluations=evaluations), "finite winner, regret or evaluation budget differs")
                checks += 1
        tied = {r["candidate"] for r in rows if r["success"] and r["completed_makespan_ns"] == joint["completed_makespan_ns"]}
        need(set(actual["tied_best"]) == tied, "exact tied winner set differs")
    return checks


def audit_campaign(campaign, *, check_current_sources=True):
    folder = Path(campaign).resolve()
    started = time.perf_counter()
    counts, results, issues = Counter(), [], []
    receipt = dict(format_version="paper-design-independent-audit-0.1", passed=False,
                   campaign=str(folder), counts=counts, cases=results, issues=issues,
                   limitations=["Checks retained analytical graph/event evidence, not numerical inference or real hardware accuracy.",
                       "Checks expected macro/active-expert coverage and replays declared ports and exact prices; does not derive mathematical operators or cost formulas.",
                       "MoE private future noninterference is covered by expression studies, not proven by this single-trace replay.",
                       "Resident weights initially occupy VRAM only; no free DRAM replica. MoE retains original immutable backing as a final goal; Dense has no explicit weight goal.",
                       "Only terminal hidden is a global hidden goal; CPU trace-token gates additionally require each preceding round's final hidden in DRAM. All KV blocks require final DRAM views; eager KV copies are policy behavior, not a per-layer completion barrier.",
                       "Static separate-memory resources only; dynamic trajectories, shared-memory coherence and allocator fragmentation are outside this study."])
    try:
        completion, protocol, records = (read(folder/name) for name in ("completion.json", "protocol.json", "records.json"))
        need(completion["complete"] is True, "campaign not complete")
        counts["declared_cases"] = check_matrix(records, protocol)
        need(len(records) == completion["cases"] and canonical(records) == completion["records_sha256"], "completion records digest differs")
        need(canonical(protocol) == completion["protocol_sha256"], "completion protocol digest differs")
        streamed = [json.loads(line) for line in (folder/"runs.jsonl").read_text(encoding="utf-8").splitlines()]
        need(len(streamed) == len(records) and {identity(r): r for r in streamed} == {identity(r): r for r in records}, "streamed records differ")
        before, after = (read(folder/name) for name in ("source-manifest-before.json", "source-manifest-after.json"))
        need(before == after, "study execution source manifest changed")
        if check_current_sources:
            for name, digest in before.items():
                need(sha(within(ROOT, name)) == digest, f"current executed source changed: {name}")
                counts["execution_sources_hashed"] += 1
        receipt["study_protocol_sha256"] = canonical(protocol)
        receipt["study_records_sha256"] = canonical(records)
        receipt["study_source_manifest_sha256"] = canonical(before)
        groups = defaultdict(set)
        for row in records:
            key = identity(row)
            item = dict(case=list(key), passed=False)
            results.append(item)
            try:
                case_dir = within(folder, row["artifact_directory"])
                need(read(case_dir/"summary.json") == row, "case summary differs from aggregate")
                manifest = read(case_dir/"artifacts.json")
                actual_files = {p.relative_to(case_dir).as_posix() for p in case_dir.rglob("*") if p.is_file() and p.name != "artifacts.json"}
                need(actual_files == set(manifest), "case artifact manifest coverage differs")
                for name, digest in manifest.items():
                    need(sha(within(case_dir, name)) == digest, f"case artifact digest differs: {name}")
                    counts["case_artifact_files_hashed"] += 1
                declaration = read(case_dir/"declarations.json")
                check_declarations(row, declaration)
                groups[key[:-1]].add(row["shared_input_sha256"])
                need(row["calibrated"] is False and row["hardware_measurement"] is False, "analytical result mislabeled measured")
                expected_bytes = sum(expected_weights(row["workload"]).values())
                infeasible = row["initial_state"] == "gpu_resident" and expected_bytes > row["vram_gib"]*GIB
                if infeasible:
                    need(not row["success"] and row["status"] == "initial_capacity_infeasible"
                         and row["completed_makespan_ns"] is None, "impossible initial placement mislabeled feasible")
                    need(row["initial_weight_bytes"] == expected_bytes and not (case_dir/"events.json").exists(), "initial rejection byte oracle or boundary differs")
                    counts["initial_capacity_infeasible"] += 1
                else:
                    need(row["success"] is True and row["status"] == "valid", "feasible campaign case failed; do not discard its denominator")
                    events, operations, tensors = (read(case_dir/name) for name in ("events.json", "operation-catalogue.json", "tensor-catalogue.json"))
                    need(canonical(events) == row["event_sha256"], "event digest differs")
                    metadata = read(case_dir/"inputs/analytical-metadata.json")
                    need(metadata["initial_state_experiment"] == row["initial_state"] and metadata["calibrated"] is False, "metadata boundary/provenance differs")
                    need(read(case_dir/"inputs/environment.json")["events"] == [], "undeclared resource trajectory")
                    counts["weight_assets_checked"] += check_geometry_and_initial_state(row, tensors)
                    counts["final_hidden_kv_goals_checked"] += check_final_goals(row, tensors)
                    counts["macro_operations_checked"] += check_macro_catalogue(row, operations)
                    counts["fixed_execution_placements_checked"] += check_execution_devices(row, operations, events)
                    counts["dependency_edges_checked"] += check_graph_dependencies(operations, tensors, events)
                    replay = audit_trace(events, operations, tensors, row, declaration["hardware"])
                    item["replay"] = replay
                    counts["successful_traces_replayed"] += 1
                    for name in ("events", "compute_events", "share_events", "copy_events", "zero_release_events"):
                        counts[name] += replay[name]
                item["passed"] = True
            except (AuditError, KeyError, TypeError, ValueError, OSError) as error:
                item["error"] = f"{type(error).__name__}: {error}"
                issues.append(dict(case=list(key), error=item["error"]))
        need(all(len(values) == 1 for values in groups.values()), "same-group candidates use different workloads/hardware")
        counts["within_group_input_equivalence_checks"] = len(groups)
        need(counts["successful_traces_replayed"] == completion["success"], "successful replay count differs from completed campaign")
        counts["finite_selection_and_budget_checks"] = check_selection_analysis(records, read(folder/"analysis.json"))
        receipt["passed"] = not issues
    except (AuditError, KeyError, TypeError, ValueError, OSError) as error:
        issues.append(dict(error=f"{type(error).__name__}: {error}"))
    receipt["audit_sources_sha256"] = {p.relative_to(ROOT).as_posix(): sha(p) for p in
        (Path(__file__), ROOT/"experiments/hardware_system_audit.py", ROOT/"tests/test_paper_design_independent_audit.py")}
    receipt["elapsed_wall_seconds"] = time.perf_counter()-started
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-current-source-check", action="store_true")
    args = parser.parse_args(argv)
    result = audit_campaign(args.campaign, check_current_sources=not args.skip_current_source_check)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, ensure_ascii=False), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
