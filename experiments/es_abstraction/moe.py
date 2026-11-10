"""Finite E/S policy study using the existing serial expert-policy boundary.

This module generates declarations, compiles them through the common policy and
command compiler, and predicts with frozen *physical* profiles. It never starts
native measurements. Shared workload payloads avoid repeating large input arrays
for every strategy; ``load_program`` materializes the exact native JSON program.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "src", ROOT / "mapping", ROOT / "modeling", ROOT / "profiling"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np

from moe_policy_validation.bridge import BOUNDARY, compile_expert_program
from moe_policy_validation.reference import copy_traffic, reference_outputs
from llm_infer_model.performance.expert_policy_response import evaluate_expert_policy
from tensor_mapping.python_frontend import Cache, Scheduling


GEOMETRIES = {"small": (256, 512), "real": (2048, 1024)}
E_RULES = {
    "E_CPU": "fixed", "E_GPU": "fixed", "E_SHAPE": "current_cost",
    "E_RESIDENT": "fiddler_paths", "E_PATH": "current_cost",
}
S_RULES = ("S_RELEASE", "S_LRU", "S_FREQ")
PATTERNS = ("fixed_heavy_light_pairs", "halfway_heavy_light_swap", "rotating_disjoint_pairs")
GUARDS = {"host_reserve_bytes": 512 * 1024**2, "gpu_reserve_bytes": 256 * 1024**2}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def value_hash(value):
    return sha256(_canonical(value)).hexdigest()


def _save(path, value, *, compact=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False,
                  indent=None if compact else 2, separators=(",", ":") if compact else None)
        stream.write("\n")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def native_config(geometry, k):
    if geometry not in GEOMETRIES or k not in (2, 4, 8):
        raise ValueError("geometry must be small/real and slots must be 2/4/8")
    h, intermediate = GEOMETRIES[geometry]
    return dict(hidden=h, intermediate=intermediate, experts=8, gpu_slots=k,
                threads=1, cuda_device=0)


def workload_specs(q_b, include_transfer=False):
    """Return the preregistered 24 main/control/geometry cases, optionally +12.

    q_b is selected once using the small-geometry calibration and is reused for
    real matrix geometry and the long-sequence transfer set.
    """
    if q_b not in (4, 8, 16):
        raise ValueError("q_b must be selected from exact rows 4,8,16")
    result = []

    def add(group, geometry, calls, seeds, patterns, light_rows):
        for k in (2, 4):
            for pattern in patterns:
                for seed in seeds:
                    cid = f"{group}-{geometry}-k{k}-n{calls}-{pattern}-s{seed}"
                    result.append(dict(id=cid, group=group, geometry=geometry, k=k,
                        calls=calls, seed=seed, pattern=pattern, heavy_rows=32,
                        light_rows=light_rows, q_total=32 + light_rows))

    add("main", "small", 8, (41030, 41031), PATTERNS, q_b)
    add("control", "small", 8, (41034,), PATTERNS[:2], 1)
    add("geometry", "real", 8, (41035, 41036), (PATTERNS[0], PATTERNS[2]), q_b)
    if include_transfer:
        add("transfer", "small", 32, (41032, 41033), PATTERNS, q_b)
    return result


def make_steps(spec):
    """Make auditable token rows; every expert receives exactly heavy/light rows.

    Expert-ID permutation and F32 inputs have independent deterministic streams.
    No policy receives this complete future list: the scoped bridge reveals one
    step only after the previous COMBINE.
    """
    h, _ = GEOMETRIES[spec["geometry"]]
    seed, count = int(spec["seed"]), int(spec["calls"])
    heavy, light = int(spec["heavy_rows"]), int(spec["light_rows"])
    if count <= 0 or heavy <= 0 or light <= 0 or spec["pattern"] not in PATTERNS:
        raise ValueError("invalid workload row counts/calls/pattern")
    ids = np.random.default_rng(seed + 7919).permutation(8).tolist()
    pairs = [ids[2*j:2*j+2] for j in range(4)]
    rng = np.random.default_rng(seed + 104729)
    steps = []
    for index in range(count):
        if spec["pattern"] == PATTERNS[0]:
            high_pair, low_pair = pairs[0], pairs[1]
        elif spec["pattern"] == PATTERNS[1]:
            high_pair, low_pair = ((pairs[0], pairs[1]) if index < count // 2
                                   else (pairs[1], pairs[0]))
        else:
            high_pair, low_pair = pairs[index % 4], pairs[(index + 1) % 4]
        routes = [list(high_pair) for _ in range(heavy)] + [list(low_pair) for _ in range(light)]
        steps.append(dict(input=rng.uniform(-0.9, 0.9, (heavy + light, h)).astype(np.float32).tolist(),
                          routes=routes, scores=[[0.6, 0.4] for _ in routes]))
    return steps


def policy_candidates(include_queue_diagnostic=False):
    rows = [dict(id=f"{e}__{s}", execution=e, storage=s, main=True)
            for e in E_RULES for s in S_RULES]
    if include_queue_diagnostic:
        rows.append(dict(id="E_QUEUE__S_LRU", execution="E_QUEUE", storage="S_LRU", main=False))
    return rows


def scheduling_for(candidate, config):
    e, s = candidate["execution"], candidate["storage"]
    if e not in E_RULES and e != "E_QUEUE" or s not in S_RULES:
        raise ValueError("unknown declared E/S candidate")
    weight_bytes = 12 * config["hidden"] * config["intermediate"]
    scheduling = Scheduling(default_device="gpu",
        placements={"expert_ffn": "cpu"} if e == "E_CPU" else {},
        expert_rule="hybrimoe_queues" if e == "E_QUEUE" else E_RULES[e],
        expert_order="graph_order", release="last_use",
        cache=Cache(rule="history_frequency" if s == "S_FREQ" else "lru",
                    unit="expert_group", budget_bytes=config["gpu_slots"] * weight_bytes))
    return scheduling, ("release_after_use" if s == "S_RELEASE" else "policy"), (
        "compute_only" if e == "E_SHAPE" else "full_path")


def baseline_membership(candidate):
    e, s = candidate["execution"], candidate["storage"]
    if not candidate["main"]:
        return ["diagnostic"]
    groups = ["B_JOINT"]
    if s == "S_RELEASE":
        groups.append("B_E")
    if e in ("E_CPU", "E_GPU"):
        groups.append("B_S")
    if e == "E_SHAPE":
        groups.append("B_FIXED_HYBRID")
    if e == "E_RESIDENT":
        groups.append("B_RESIDENT")
    if len(groups) > 1:
        groups.append("B_STRONG")
    return groups


def select_boundary_rows(calibrations):
    """Calibration-only selection shared by K=2/4; never examines validation.

    calibrations maps (geometry, K) to an exact physical profile. Missing prices
    are errors; no interpolation or manufactured crossover is allowed.
    """
    scores = []
    for q in (4, 8, 16):
        details, eligible, distance = [], True, 0.0
        for k in (2, 4):
            p = calibrations[("small", k)]["prices_ns"]
            c, g, load = (p[f"EXPERT:cpu:q={q}"], p[f"EXPERT:gpu:q={q}"], p["LOAD"])
            if min(c, g) < 0 or load <= 0:
                raise ValueError("exact nonnegative EXPERT and positive LOAD profiles required")
            cell = g < c < g + load
            eligible &= cell
            distance += abs((c - g) / load - 0.5)
            details.append(dict(k=k, cpu_ns=c, gpu_ns=g, load_ns=load, crossover=cell))
        scores.append(dict(q=q, eligible=eligible, normalized_midpoint_distance=distance, cells=details))
    valid = [row for row in scores if row["eligible"]]
    chosen = min(valid, key=lambda row: (row["normalized_midpoint_distance"], row["q"]))["q"] if valid else 16
    return dict(q_b=chosen, found_shared_crossover=bool(valid), candidates=scores,
                selected_using="small geometry calibration only; same q_b reused for all K and geometry transfer",
                fallback="q16 if no shared crossover; absence retained")


def load_calibrations(manifest_path):
    """Read {small:{2:profile-or-path,4:...},real:{2:...,4:...}}."""
    path = Path(manifest_path).resolve()
    data = _read(path)
    data = data.get("calibrations", data)
    result = {}
    for geometry, by_slots in data.items():
        if geometry not in GEOMETRIES:
            continue
        for slots, source in by_slots.items():
            if isinstance(source, str):
                target = Path(source)
                if not target.is_absolute():
                    target = path.parent / target
                profile = _read(target)
            else:
                profile = source
            result[(geometry, int(slots))] = profile
    return result


def _compile(config, steps, seed, candidate, physical_profile):
    scheduling, retention, view = scheduling_for(candidate, config)
    return compile_expert_program(config=config, steps=steps, seed=seed,
        scheduling=scheduling, retention=retention, decision_cost_view=view,
        prices=physical_profile, resource_guards=GUARDS)


def check_causal_prefix(config, steps, seed, candidate, physical_profile, original=None):
    split = max(1, len(steps) // 2)
    altered = deepcopy(steps)
    for step in altered[split:]:
        step["routes"] = [[(expert + 1) % config["experts"] for expert in row] for row in step["routes"]]
    original = original or _compile(config, steps, seed, candidate, physical_profile)
    changed = _compile(config, altered, seed, candidate, physical_profile)
    before = [row for row in original["compilation_evidence"]["records"] if row["step"] < split]
    after = [row for row in changed["compilation_evidence"]["records"] if row["step"] < split]
    if before != after:
        raise AssertionError("unrevealed route suffix changed a public prefix")
    return dict(prefix_steps=split, future_suffix_changed=True, records_equal=True,
                prefix_sha256=value_hash(before))


def _source_receipts():
    files = [Path(__file__), ROOT / "profiling/moe_policy_validation/bridge.py",
             ROOT / "profiling/moe_policy_validation/reference.py",
             ROOT / "modeling/llm_infer_model/performance/expert_policy_response.py"]
    files += [ROOT / "mapping/tensor_mapping" / name for name in
              ("composition_policy.py", "composition_observation.py", "causal.py", "policy_executor.py",
               "expert_scheduling.py", "python_frontend.py", "policy_ir.py")]
    return {str(path.resolve()): sha256(path.read_bytes()).hexdigest() for path in files}


def compile_all(calibrations, out, *, q_b, include_transfer=False, workloads=None,
                compute_references=False, verify_prefix=True, include_queue_diagnostic=False):
    """Freeze finite candidates and physical predictions, without native execution.

    A partial workload list is permitted for preparation/pilot, but it is recorded
    explicitly and cannot stand in for the full preregistered matrix.
    """
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    specs = workload_specs(q_b, include_transfer) if workloads is None else list(workloads)
    if not specs or len({spec["id"] for spec in specs}) != len(specs):
        raise ValueError("workloads must have unique nonempty identities")
    candidates = policy_candidates(include_queue_diagnostic)
    sources = _source_receipts()
    physical_profiles = {}
    for geometry, k in sorted({(spec["geometry"], spec["k"]) for spec in specs}):
        profile = deepcopy(calibrations[(geometry, k)])
        if profile.get("config") != native_config(geometry, k) or profile.get("boundary") != BOUNDARY:
            raise ValueError("physical profile configuration/boundary mismatch")
        key = f"{geometry}-k{k}"
        physical_profiles[key] = dict(path=f"profiles/{key}.json", sha256=value_hash(profile),
                                     geometry=geometry, k=k)
        _save(out / physical_profiles[key]["path"], profile)
    _save(out / "PROTOCOL.json", dict(schema="es-abstraction-moe-campaign-1", q_b=q_b,
        native_measurements_started=False, source_hashes=sources, workloads=specs,
        candidates=candidates, physical_profiles=physical_profiles, causal_prefix_checks=verify_prefix,
        input_boundary=BOUNDARY, all_core_mechanisms_existing=True,
        custom_changes="ordinary Scheduling bridge inlet and compute-only decision price ablation",
        full_preregistered_matrix=(workloads is None),
        claim="serial routed expert FFN; no async/full-layer/total-VRAM certification"))

    programs, labels, workload_rows = {}, [], []
    for spec in specs:
        config, steps = native_config(spec["geometry"], spec["k"]), make_steps(spec)
        pkey = f"{spec['geometry']}-k{spec['k']}"
        physical = calibrations[(spec["geometry"], spec["k"])]
        wpath = f"workloads/{spec['id']}/workload.json"
        payload = dict(schema="moe-expert-policy-native-1", config=config,
                       seed=spec["seed"], resource_guards=GUARDS, steps=steps)
        _save(out / wpath, payload, compact=True)
        _save(out / f"workloads/{spec['id']}/spec.json", spec)
        reference_path = None
        if compute_references:
            reference_path = f"workloads/{spec['id']}/reference.json"
            _save(out / reference_path, reference_outputs(payload), compact=True)
        workload_rows.append(dict(**spec, payload_path=wpath, reference_path=reference_path,
                                  payload_sha256=value_hash(payload)))
        for candidate in candidates:
            started = perf_counter_ns()
            compiled = _compile(config, steps, spec["seed"], candidate, physical)
            compile_ns = perf_counter_ns() - started
            program = compiled["program"]
            started = perf_counter_ns()
            prediction = evaluate_expert_policy(program, physical)
            evaluate_ns = perf_counter_ns() - started
            for key, count in copy_traffic(program).items():
                if prediction[key] != count:
                    raise AssertionError(f"independent physical byte ledger differs: {key}")
            causal = check_causal_prefix(config, steps, spec["seed"], candidate, physical,
                                         original=compiled) if verify_prefix else None
            phash = value_hash(program)
            pid = "program-" + phash[:24]
            label = dict(id=f"{spec['id']}--{candidate['id']}", workload_id=spec["id"],
                geometry=spec["geometry"], k=spec["k"], group=spec["group"],
                policy_id=candidate["id"], execution=candidate["execution"], storage=candidate["storage"],
                main=candidate["main"], baseline_families=baseline_membership(candidate),
                program_id=pid, program_sha256=phash, predicted_ns=prediction["duration_ns"],
                compile_ns=compile_ns, evaluate_ns=evaluate_ns)
            labels.append(label)
            _save(out / f"labels/{label['id']}.json", dict(**label,
                compilation=compiled["compilation_evidence"], causal_prefix=causal))
            if pid in programs:
                if programs[pid]["program_sha256"] != phash or programs[pid]["predicted_ns"] != prediction["duration_ns"]:
                    raise AssertionError("alias hash or physical prediction differs")
                programs[pid]["labels"].append(label["id"])
                programs[pid]["policy_ids"].append(candidate["id"])
            else:
                actions_path = f"programs/{pid}/actions.json"
                _save(out / actions_path, program["actions"])
                _save(out / f"programs/{pid}/prediction.json", prediction)
                programs[pid] = dict(id=pid, program_id=pid, program_sha256=phash,
                    workload_id=spec["id"], geometry=spec["geometry"], k=spec["k"],
                    group=spec["group"], labels=[label["id"]], policy_ids=[candidate["id"]],
                    workload_path=wpath, actions_path=actions_path, physical_profile_key=pkey,
                    physical_profile_sha256=physical_profiles[pkey]["sha256"],
                    predicted_ns=prediction["duration_ns"],
                    action_counts=dict(Counter(a["kind"] for a in program["actions"])),
                    weight_h2d_bytes=prediction["weight_h2d_bytes"],
                    total_h2d_bytes=prediction["total_h2d_bytes"], total_d2h_bytes=prediction["total_d2h_bytes"],
                    peak_resident_experts=prediction["peak_resident_experts"])
        print(f"compiled {spec['id']}: {len(candidates)} labels", flush=True)
    for pid, metadata in programs.items():
        _save(out / f"programs/{pid}/metadata.json", metadata)
    after = _source_receipts()
    if after != sources:
        raise RuntimeError("source changed during compilation; do not use partial campaign")
    manifest = dict(schema="es-abstraction-frozen-predictions-1", q_b=q_b,
        workload_count=len(specs), nominal_conditions=len(labels), unique_programs=len(programs),
        source_hashes=sources, physical_profiles=physical_profiles,
        workloads=workload_rows, labels=labels, programs=list(programs.values()),
        native_measurements_started=False, no_validation_timing_used=True,
        decision_ablation="E_SHAPE hides LOAD price only in policy observations; model always uses physical profile",
        equivalence="Complete native program: geometry, weights seed, steps, action order, guards and boundary",
        storage="One common workload payload plus exact actions; load_program reconstructs and verifies canonical hash")
    _save(out / "FROZEN_PREDICTIONS.json", manifest)
    _save(out / "SOURCE_UNCHANGED.json", dict(unchanged=True, source_hashes=after))
    return manifest


def load_program(campaign_dir, program_id):
    folder = Path(campaign_dir).resolve()
    if not isinstance(program_id, str) or not program_id.startswith("program-") or not all(
            c in "0123456789abcdef" for c in program_id[len("program-"):]):
        raise ValueError("invalid canonical program identity")
    metadata = _read(folder / "programs" / program_id / "metadata.json")
    payload = _read(folder / metadata["workload_path"])
    payload["actions"] = _read(folder / metadata["actions_path"])
    if value_hash(payload) != metadata["program_sha256"]:
        raise ValueError("retained workload/actions differ from frozen native program")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select-boundary")
    select.add_argument("--profiles", type=Path, required=True)
    select.add_argument("--out", type=Path, required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--profiles", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--q-b", type=int, choices=(4, 8, 16), required=True)
    prepare.add_argument("--include-transfer", action="store_true")
    prepare.add_argument("--include-queue-diagnostic", action="store_true")
    prepare.add_argument("--compute-references", action="store_true")
    prepare.add_argument("--geometry", choices=("small", "real"))
    prepare.add_argument("--limit-workloads", type=int,
                         help="Explicit partial pilot preparation; not full-confirmation evidence")
    args = parser.parse_args(argv)
    profiles = load_calibrations(args.profiles)
    if args.command == "select-boundary":
        result = select_boundary_rows(profiles)
        _save(args.out, result)
        print(json.dumps(result, indent=2))
    else:
        specs = None
        if args.geometry or args.limit_workloads is not None:
            specs = workload_specs(args.q_b, args.include_transfer)
            if args.geometry:
                specs = [spec for spec in specs if spec["geometry"] == args.geometry]
            if args.limit_workloads is not None:
                if args.limit_workloads <= 0:
                    raise ValueError("limit-workloads must be positive")
                specs = specs[:args.limit_workloads]
        result = compile_all(profiles, args.out, q_b=args.q_b, include_transfer=args.include_transfer,
            workloads=specs, compute_references=args.compute_references,
            include_queue_diagnostic=args.include_queue_diagnostic)
        print(json.dumps({key: result[key] for key in ("workload_count", "nominal_conditions", "unique_programs")}, indent=2))


if __name__ == "__main__":
    main()
