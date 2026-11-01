"""Independent, context-matched calibration diagnostic for native expert FFNs.

No old validation outputs are read. Each eight-call program is measured bare
and stamped in balanced randomized blocks. Prices remain separated by workload
context; modeling, not this module, evaluates the independent validation seed.
This is a small diagnostic, not a replacement for the 24-workload study.

From Simulator::

    python -m profiling.es_abstraction.context_repair prepare --out build/NEW --baseline-root build/es-abstraction-20261030
    python -m profiling.es_abstraction.context_repair acquire --out build/NEW --phase calibration
    python -m profiling.es_abstraction.context_repair freeze --out build/NEW
    python -m profiling.es_abstraction.context_repair acquire --out build/NEW --phase validation
    python -m profiling.es_abstraction.context_repair summarize --out build/NEW

``run`` performs that sequence. Preparation and unit tests do not run native
code. Only ``acquire``/``run`` invoke the existing executor under its lock and
unchanged memory guards. Partial or failed data remain visible, never replaced.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import statistics as st
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

from profiling.moe_policy_validation import campaign as base
from profiling.moe_policy_validation import reference
from profiling.es_abstraction import calibration as environment
from llm_infer_model.performance import expert_policy_response as model

CALIBRATION_SEED = 32030
VALIDATION_SEED = 42030
GEOMETRIES = {"small": (256, 512), "real": (2048, 1024)}
MODES = ("bare", "stamped")
DEFAULT_BUILD = base.ROOT / "build/moe-policy-native-20261029/BUILD_COMPLETE.json"


def value_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def specifications():
    result = []
    for q in (1, 8):
        for lifecycle in ("lru", "release"):
            result.append(dict(id=f"small-k4-q{q}-{lifecycle}", geometry="small", k=4,
                               light_q=q, lifecycle=lifecycle, calls=8, pattern="fixed"))
    for k in (2, 4):
        result.append(dict(id=f"real-k{k}-q8-cpu", geometry="real", k=k,
                           light_q=8, lifecycle="cpu", calls=8, pattern="fixed"))
    return result


def make_program(spec, seed, *, dimensions=None):
    """Direct known native diagnostic actions, not a new policy compiler."""
    if spec["calls"] != 8 or spec["pattern"] != "fixed":
        raise ValueError("this diagnostic fixes eight calls and fixed route identities")
    if spec["light_q"] not in (1, 8) or spec["lifecycle"] not in ("cpu", "lru", "release"):
        raise ValueError("context is outside the declared diagnostic matrix")
    h, intermediate = dimensions or GEOMETRIES[spec["geometry"]]
    config = dict(hidden=h, intermediate=intermediate, experts=8, gpu_slots=spec["k"],
                  threads=1, cuda_device=0)
    # K and lifecycle do not alter inputs, weights or routes. Thus the real
    # pure-CPU K=2/4 comparison changes only an unused GPU configuration field.
    rng = np.random.default_rng(seed + spec["light_q"])
    steps, actions, owners = [], [], {}
    for step in range(8):
        routes = [[0, 1] for _ in range(32)] + [[2, 3] for _ in range(spec["light_q"])]
        steps.append(dict(input=rng.uniform(-.9, .9, (len(routes), h)).astype(np.float32).tolist(),
                          routes=routes, scores=[[.6, .4] for _ in routes]))
        for expert in range(4):
            if spec["lifecycle"] == "cpu":
                actions.append(dict(kind="EXPERT", step=step, expert=expert, device="cpu"))
            else:
                if expert not in owners:
                    if len(owners) >= spec["k"]:
                        raise ValueError("small diagnostic requires four simultaneous weight slots")
                    slot = min(set(range(spec["k"])) - set(owners.values()))
                    owners[expert] = slot
                    actions.append(dict(kind="LOAD", expert=expert, slot=slot))
                actions.append(dict(kind="EXPERT", step=step, expert=expert, device="gpu", slot=owners[expert]))
                if spec["lifecycle"] == "release":
                    actions.append(dict(kind="RELEASE", expert=expert, slot=owners.pop(expert)))
        actions.append(dict(kind="COMBINE", step=step))
    program = dict(schema="moe-expert-policy-native-1", config=config, seed=seed,
                   resource_guards=dict(base.GUARDS), steps=steps, actions=actions)
    model.action_price_keys(program)
    return program


def blocked_order(specs, blocks, phase):
    if blocks < 2 or blocks % 2:
        raise ValueError("an even number of blocks >=2 is required for exact mode-order balance")
    if phase not in ("calibration", "validation"):
        raise ValueError("unknown measurement phase")
    seed = CALIBRATION_SEED if phase == "calibration" else VALIDATION_SEED
    identifiers = sorted(s["id"] for s in specs)
    # Pair order is randomized each block. Within a pair, the first mode
    # alternates by block, with randomized per-program initial orientation.
    orientation = {cid: random.Random(seed + i*101).randrange(2) for i, cid in enumerate(identifiers)}
    result = []
    for block in range(blocks):
        order = list(identifiers)
        random.Random(seed + 1009 + block).shuffle(order)
        for cid in order:
            modes = list(MODES) if phase == "calibration" else ["bare"]
            if phase == "calibration" and (orientation[cid] + block) % 2:
                modes.reverse()
            for mode in modes:
                result.append(dict(index=len(result), block=block, case_id=cid, mode=mode))
    return result


def source_manifest():
    # These are actually used measurement/oracle/model sources. Unused mapping
    # modules imported transitively by legacy helpers are not frozen here;
    # another task may repair lowering while this independent probe runs.
    return {str(Path(p).resolve()): base.digest(p) for p in
            (__file__, base.__file__, reference.__file__, environment.__file__)}


def baseline_tables(root, specs):
    """Read only the allowlisted old calibration files, never old validation."""
    root=Path(root).resolve(); tables={}
    for geometry,k in sorted({(s["geometry"],s["k"]) for s in specs}):
        path=root/"calibration"/geometry/f"k{k}"/"calibration.json"
        profile=base.read(path)
        if profile.get("schema")!="moe-expert-policy-calibration-1" or profile.get("boundary")!=model.TIMING_BOUNDARY:
            raise ValueError("old baseline has an incompatible calibration schema/boundary")
        tables[f"{geometry}-k{k}"]=dict(origin_path=str(path), origin_sha256=base.digest(path), profile=profile)
    return tables


def checked_build(path):
    build = base.read(path)
    for record in [build["binary"]] + build["runtimes"]:
        if base.digest(record["path"]) != record["sha256"]:
            raise ValueError("native runtime differs from its build receipt: " + record["path"])
    return build


def check_sources(protocol):
    for path, expected in protocol["sources"].items():
        if base.digest(path) != expected:
            raise ValueError("diagnostic source changed: " + path)
    return checked_build(protocol["build_receipt"])


def prepare(out, *, blocks=4, build_path=DEFAULT_BUILD, baseline_root=None):
    out = Path(out).resolve()
    if out.exists():
        raise ValueError("new diagnostic directory required")
    specs = specifications()
    orders = {phase: blocked_order(specs, blocks, phase) for phase in ("calibration", "validation")}
    build = checked_build(build_path)
    sources = source_manifest()
    baselines=baseline_tables(baseline_root,specs) if baseline_root is not None else {}
    out.mkdir(parents=True)
    baseline_receipts={}
    for key,row in baselines.items():
        relative=f"baseline_profiles/{key}.json"
        base.save(out/relative,row["profile"])
        baseline_receipts[key]=dict(path=relative, file_sha256=base.digest(out/relative),
                                    origin_path=row["origin_path"],origin_sha256=row["origin_sha256"])
    programs = []
    for phase, seed in (("calibration", CALIBRATION_SEED), ("validation", VALIDATION_SEED)):
        for spec in specs:
            program = make_program(spec, seed)
            relative = Path("programs") / phase / (spec["id"] + ".json")
            base.save(out/relative, program)
            programs.append(dict(phase=phase, case_id=spec["id"], path=relative.as_posix(),
                                 file_sha256=base.digest(out/relative), program_sha256=value_hash(program),
                                 seed=seed, action_counts=dict(Counter(a["kind"] for a in program["actions"])),
                                 priced_action_counts=dict(Counter(model.action_price_keys(program)))))
        a, b = (make_program(next(s for s in specs if s["id"]==f"real-k{k}-q8-cpu"), seed) for k in (2,4))
        a["config"]["gpu_slots"] = b["config"]["gpu_slots"]
        if a != b:
            raise ValueError("pure CPU K comparison changes more than gpu_slots")
    if source_manifest() != sources:
        raise ValueError("source changed during diagnostic preparation")
    protocol = dict(schema="es-context-repair-protocol-1", created=base.utc(), blocks=blocks,
        calibration_seed=CALIBRATION_SEED, validation_seed=VALIDATION_SEED, specs=specs,
        order=orders, programs=programs, sources=sources, build_receipt=str(Path(build_path).resolve()),
        build=build, request=dict(warmup=3, samples=10, repeats=1),
        old_baseline_profiles=baseline_receipts,
        modeling_source_at_prepare=dict(path=str(Path(model.__file__).resolve()),sha256=base.digest(model.__file__),
                                        scope="Recorded separately; unrelated mapping/core files are not acquisition drift locks"),
        expected_calibration_processes=len(orders["calibration"]),
        expected_validation_processes=len(orders["validation"]),
        timing="Eight complete serial calls; cold logical cache per sequence, retained native graph/slot allocations, same program bytes for bare and stamped",
        independent_unit="fresh-process mean; modes are separate adjacent processes within a matched randomized block",
        paired_mode_order="Each condition has equal bare-first and stamped-first blocks; program order is deterministically shuffled each block",
        price_partition=["geometry", "gpu_slots", "light_q", "lifecycle", "calls", "pattern"],
        price_rule="Equal-weight mean of per-process stamped category means, independently for each exact context; no bare residual correction or cross-context pooling",
        pure_cpu_check="Same inputs/weights/action coordinates; only unused gpu_slots differs. Report K2/K4 contrast; do not automatically average or fit them",
        stamp_check="Report per-block stamped-minus-bare and percent difference on identical bytes; attribution remains unknown",
        validation_rule="Seed42030 inputs/weights, frozen exact-context prices; model evaluates before validation. No old665/865data imported",
        unknown_states=["background CPU/GPU occupancy", "clock/thermal history", "OS ready/block timing", "cache microstate"],
        nonclaims=["No causal attribution to OS or H2D", "No complete24-workload validation", "No compiler novelty claim", "No new modeling formula", "No removal of unfavorable samples"],
        guards=dict(base.GUARDS))
    base.save(out/"PROTOCOL.json", protocol)
    print(json.dumps({"prepared": str(out), "calibration_processes": len(orders["calibration"]),
                      "validation_processes": len(orders["validation"])}, indent=2), flush=True)
    return protocol


def load_program(out, protocol, phase, case_id):
    record = next(p for p in protocol["programs"] if (p["phase"], p["case_id"]) == (phase, case_id))
    path = Path(out) / record["path"]
    if base.digest(path) != record["file_sha256"]:
        raise ValueError("prepared program changed: " + str(path))
    return path, base.read(path)


def read_raw(path):
    path = Path(path)
    if path.exists():
        return base.read(path)
    with gzip.open(path.with_suffix(".json.gz"), "rt", encoding="utf-8") as stream:
        return json.load(stream)


def compress_raw(path):
    path = Path(path)
    dest = path.with_suffix(".json.gz")
    expected = base.digest(path)
    with path.open("rb") as source, gzip.open(dest, "xb", compresslevel=6) as target:
        while data := source.read(1024**2):
            target.write(data)
    check = hashlib.sha256()
    with gzip.open(dest, "rb") as source:
        while data := source.read(1024**2):
            check.update(data)
    if check.hexdigest() != expected:
        raise RuntimeError("compressed evidence checksum differs")
    base.save(dest.with_suffix(".receipt.json"), dict(original_sha256=expected, compressed_sha256=base.digest(dest)))
    path.unlink()


def category_means(program, native):
    keys = model.action_price_keys(program)
    samples = native.get("action_samples_ns")
    if not samples:
        raise ValueError("stamped result requires per-action durations")
    cells = defaultdict(list)
    for sample in samples:
        if len(sample) != len(keys):
            raise ValueError("stamped action count differs from the actual program")
        for key, value in zip(keys, sample):
            if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValueError("invalid measured action duration")
            cells[key].append(value)
    return {key: st.mean(values) for key, values in cells.items()}


def acquire(out, phase):
    out = Path(out).resolve(); protocol = base.read(out/"PROTOCOL.json")
    build = check_sources(protocol)
    if phase not in ("calibration", "validation"):
        raise ValueError("unknown phase")
    if phase == "validation":
        frozen = base.read(out/"FROZEN_PREDICTIONS.json")
        for rel, expected in frozen["profile_sha256"].items():
            if base.digest(out/rel) != expected:
                raise ValueError("a frozen context calibration changed")
    env = environment.native_env(build)
    with base.measurement_lock():
        # Reference construction is outside timed native intervals, and under
        # the same measurement lock so it cannot compete with another campaign.
        for spec in protocol["specs"]:
            rp = out/"references"/phase/(spec["id"]+".json")
            if not rp.exists():
                _, program = load_program(out, protocol, phase, spec["id"])
                base.save(rp, reference.reference_outputs(program))
        for item in protocol["order"][phase]:
            cid, block, mode = item["case_id"], item["block"], item["mode"]
            folder = out/phase/cid/mode
            raw = folder/f"process-{block}.json"
            verification_path = raw.with_suffix(".verification.json")
            if verification_path.exists():
                if not (raw.exists() or raw.with_suffix(".json.gz").exists()):
                    raise ValueError("verification exists without retained raw evidence")
                continue
            program_path, program = load_program(out, protocol, phase, cid)
            start = time.perf_counter()
            if raw.exists() or raw.with_suffix(".json.gz").exists():
                native = read_raw(raw)
                resumed = True
            else:
                # Existing failed request/process receipts are not overwritten.
                # They require explicit review, rather than an automatic retry.
                native = base.invoke(Path(build["binary"]["path"]), env, program_path,
                                     dict(mode=mode, **protocol["request"]), raw)
                resumed = False
            verification = reference.verify_result(program, native, expected=base.read(out/"references"/phase/(cid+".json")))
            vals = native["samples_ns"]
            if len(vals) != protocol["request"]["samples"] or any(v <= 0 or not np.isfinite(v) for v in vals):
                raise ValueError("native complete-sequence sample count/value mismatch")
            verification.update(phase=phase, case_id=cid, block=block, mode=mode,
                program_sha256=value_hash(program), process_mean_ns=st.mean(vals),
                process_sample_cv=st.stdev(vals)/st.mean(vals),
                component_means_ns=category_means(program, native) if mode=="stamped" else None,
                source_or_verify_wall_seconds=time.perf_counter()-start,
                resumed_raw_not_a_new_process=resumed,
                allocated_gpu_weight_bytes=native["gpu_weight_allocated_bytes"],
                workspace_bytes=native["workspace_bytes"],
                graph_port_allocated_bytes=native["graph_port_allocated_bytes"])
            if mode == "stamped":
                verification["mean_sequence_minus_action_stamps_ns"] = st.mean(vals) - st.mean(sum(x) for x in native["action_samples_ns"])
            base.save(verification_path, verification)
            if raw.exists():
                compress_raw(raw)
            print("context", phase, item["index"]+1, "/", len(protocol["order"][phase]),
                  cid, mode, block, round(verification["process_mean_ns"]/1e6, 4), "ms", flush=True)
            if not verification["numerically_equal"]:
                raise RuntimeError("numerical failure retained; no calibration claim: " + cid)
    check_sources(protocol)
    receipt=out/(phase.upper()+"_SOURCES_UNCHANGED.json")
    if not receipt.exists():
        base.save(receipt, dict(sources=protocol["sources"], unchanged=True))


def records(out, protocol, phase, cid, mode):
    return [base.read(Path(out)/phase/cid/mode/f"process-{block}.verification.json")
            for block in range(protocol["blocks"])]


def paired_contrast(first, second, *, seed=52030):
    """Second minus first, matched by independent acquisition block."""
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("paired contrasts require equal independent block counts")
    a, b = np.asarray(first, float), np.asarray(second, float)
    if (a <= 0).any() or (b <= 0).any():
        raise ValueError("timings must be positive")
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(a), (10000, len(a)))
    ratios = 100 * (b[index].mean(axis=1)/a[index].mean(axis=1)-1)
    return dict(first_mean_ns=float(a.mean()), second_mean_ns=float(b.mean()),
                second_vs_first_percent=float(100*(b.mean()/a.mean()-1)),
                paired_block_differences_ns=(b-a).tolist(),
                nominal_95ci_percent=np.quantile(ratios,[.025,.975]).tolist(), independent_blocks=len(a),
                inference="Matched-block bootstrap; nominal small-sample interval, not causal attribution or equivalence certification")


def diagnostic(out, protocol):
    contexts, mode_checks = {}, []
    for spec in protocol["specs"]:
        cid = spec["id"]
        bare = records(out, protocol, "calibration", cid, "bare")
        stamped = records(out, protocol, "calibration", cid, "stamped")
        all_rows = bare + stamped
        if not all(x["numerically_equal"] and x["compiled_action_sequence_equal"] and x["bounded_weight_slots"] for x in all_rows):
            raise ValueError("failed numerical/action/physical calibration cannot produce prices")
        keys = set(stamped[0]["component_means_ns"])
        if any(set(row["component_means_ns"]) != keys for row in stamped):
            raise ValueError("same context produced different action coordinates")
        cells = {key:[row["component_means_ns"][key] for row in stamped] for key in sorted(keys)}
        contexts[cid] = dict(spec=spec, prices_ns={key: st.mean(values) for key,values in cells.items()},
                             per_process_category_means_ns=cells,
                             bare_process_means_ns=[x["process_mean_ns"] for x in bare],
                             stamped_process_means_ns=[x["process_mean_ns"] for x in stamped],
                             stamp_unattributed_gaps_ns=[x["mean_sequence_minus_action_stamps_ns"] for x in stamped])
        mode_checks.append(dict(case_id=cid, comparison="stamped minus bare, identical program bytes",
            **paired_contrast(contexts[cid]["bare_process_means_ns"], contexts[cid]["stamped_process_means_ns"])))
    cpu_checks = []
    for mode in MODES:
        field = mode+"_process_means_ns"
        cpu_checks.append(dict(mode=mode, comparison="pure CPU K4 minus K2; GPU not used",
            **paired_contrast(contexts["real-k2-q8-cpu"][field], contexts["real-k4-q8-cpu"][field])))
    # Same expert q32, call count and lifecycle, with light-route context varied.
    q32_checks=[]
    for lifecycle in ("lru", "release"):
        key="EXPERT:gpu:q=32"
        a=contexts[f"small-k4-q1-{lifecycle}"]["per_process_category_means_ns"][key]
        b=contexts[f"small-k4-q8-{lifecycle}"]["per_process_category_means_ns"][key]
        q32_checks.append(dict(lifecycle=lifecycle, coordinate=key, comparison="lightq8 minus lightq1 context",
                              **paired_contrast(a,b)))
    return dict(schema="es-context-calibration-diagnostic-1", contexts=contexts,
                mode_comparisons=mode_checks, pure_cpu_k_comparisons=cpu_checks,
                same_q32_context_comparisons=q32_checks, cross_context_pooling=False,
                missing_state_measurements=protocol["unknown_states"],
                interpretation="Differences establish context sensitivity if resolved; no unique OS, H2D, cache or instrumentation-cause inference")


def freeze(out):
    out=Path(out).resolve(); protocol=base.read(out/"PROTOCOL.json")
    check_sources(protocol)
    if (out/"FROZEN_PREDICTIONS.json").exists():
        raise ValueError("predictions already frozen; no refitting permitted")
    if (out/"validation").exists():
        raise ValueError("validation collection must not precede calibration freeze")
    report=diagnostic(out,protocol)
    base.save(out/"CALIBRATION_DIAGNOSTIC.json",report)
    predictions=[]; profile_hashes={}
    modeling_before=base.digest(model.__file__)
    for spec in protocol["specs"]:
        cid=spec["id"]
        _, calprogram=load_program(out,protocol,"calibration",cid)
        _, valprogram=load_program(out,protocol,"validation",cid)
        context=report["contexts"][cid]
        profile=dict(schema="moe-expert-policy-calibration-1", config=calprogram["config"],
            boundary=model.TIMING_BOUNDARY, seed=CALIBRATION_SEED,
            prices_ns=context["prices_ns"], measured_process_means_ns=context["per_process_category_means_ns"],
            context=spec, calls=8, measurement_mode="stamped", created=base.utc(),
            provenance="Independent seed32030 matched eight-call context; no total-time residual fitting and no old validation data",
            limitations="Exact context only; bare/stamped discrepancy is separately retained, not subtracted")
        rel=f"profiles/{cid}.json"; base.save(out/rel,profile)
        profile_hashes[rel]=base.digest(out/rel)
        # Prediction is delegated to shared modeling; no formula or correction
        # is implemented in this measurement utility.
        prediction=model.evaluate_expert_policy(valprogram,profile)
        record=dict(case_id=cid, profile=rel, prediction=prediction,
                    program_sha256=value_hash(valprogram), seed=VALIDATION_SEED)
        baseline=protocol.get("old_baseline_profiles",{}).get(f"{spec['geometry']}-k{spec['k']}")
        if baseline is not None:
            if base.digest(out/baseline["path"])!=baseline["file_sha256"]:
                raise ValueError("frozen old profile copy changed")
            old_profile=base.read(out/baseline["path"])
            record["old_baseline_prediction"]=model.evaluate_expert_policy(valprogram,old_profile)
            record["old_baseline_profile"]=baseline
        predictions.append(record)
    if base.digest(model.__file__)!=modeling_before:
        raise ValueError("shared evaluator changed while freezing predictions")
    frozen=dict(schema="es-context-frozen-predictions-1", created=base.utc(),
                profile_sha256=profile_hashes, cases=predictions,
                calibration_processes=len(protocol["order"]["calibration"]), sources=protocol["sources"],
                price_partition=protocol["price_partition"], calibration_seed=CALIBRATION_SEED,
                validation_seed=VALIDATION_SEED, old_validation_data_used=False,
                old_baseline_profiles=protocol.get("old_baseline_profiles",{}),
                modeling_source_at_freeze=dict(path=str(Path(model.__file__).resolve()),sha256=modeling_before))
    base.save(out/"FROZEN_PREDICTIONS.json",frozen)
    return frozen


def summarize(out):
    out=Path(out).resolve(); protocol=base.read(out/"PROTOCOL.json")
    check_sources(protocol)
    frozen=base.read(out/"FROZEN_PREDICTIONS.json")
    for path,sha in frozen["profile_sha256"].items():
        if base.digest(out/path)!=sha:
            raise ValueError("frozen profile changed during independent validation")
    rows=[]
    for case in frozen["cases"]:
        vs=records(out,protocol,"validation",case["case_id"],"bare")
        vals=[v["process_mean_ns"] for v in vs]
        mean=st.mean(vals); prediction=case["prediction"]["duration_ns"]
        passed=all(v["numerically_equal"] and v["compiled_action_sequence_equal"] and v["bounded_weight_slots"] for v in vs)
        row=dict(case_id=case["case_id"], predicted_ns=prediction,
            measured_mean_ns=mean, process_means_ns=vals,
            absolute_percentage_error=100*abs(prediction-mean)/mean,
            signed_percentage_error=100*(prediction/mean-1),
            numerical_action_physical_checks=passed, max_abs_error=max(v["max_abs_error"] for v in vs))
        if "old_baseline_prediction" in case:
            old=case["old_baseline_prediction"]["duration_ns"]
            row.update(old_baseline_predicted_ns=old,
                       old_baseline_absolute_percentage_error=100*abs(old-mean)/mean,
                       old_baseline_signed_percentage_error=100*(old/mean-1),
                       ape_change_percentage_points=row["absolute_percentage_error"]-100*abs(old-mean)/mean)
        rows.append(row)
    result=dict(schema="es-context-independent-results-1", created=base.utc(),
        calibration_seed=CALIBRATION_SEED, validation_seed=VALIDATION_SEED,
        calibration_processes=len(protocol["order"]["calibration"]),
        validation_processes=len(protocol["order"]["validation"]), independent_blocks=protocol["blocks"],
        all_checks_passed=all(r["numerical_action_physical_checks"] for r in rows),
        mape_percent=st.mean(r["absolute_percentage_error"] for r in rows),
        max_ape_percent=max(r["absolute_percentage_error"] for r in rows), rows=rows,
        calibration_diagnostic="CALIBRATION_DIAGNOSTIC.json", old_validation_data_used=False,
        causal_attribution="unresolved", cross_context_pooling=False,
        scope="Six fixed-route eight-call expert programs; independent input/weight seeds, unchanged modeling and native executor. Not the earlier24-workload acceptance set.")
    if all("old_baseline_predicted_ns" in row for row in rows):
        result["old_baseline_mape_percent"]=st.mean(r["old_baseline_absolute_percentage_error"] for r in rows)
        result["old_baseline_max_ape_percent"]=max(r["old_baseline_absolute_percentage_error"] for r in rows)
        result["baseline_comparison_boundary"]="Old and new predictions were frozen before the same seed42030 bare measurements; old prices only, no old validation values"
    base.save(out/"RESULTS.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},ensure_ascii=False,indent=2),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","acquire","freeze","summarize","run"))
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--blocks",type=int,default=4)
    parser.add_argument("--build",type=Path,default=DEFAULT_BUILD)
    parser.add_argument("--baseline-root",type=Path,
                        help="Old run root; only calibration/{geometry}/k*/calibration.json is read")
    parser.add_argument("--phase",choices=("calibration","validation"))
    args=parser.parse_args()
    if args.command in ("prepare","run"):
        prepare(args.out,blocks=args.blocks,build_path=args.build,baseline_root=args.baseline_root)
    if args.command=="acquire":
        if args.phase is None:
            parser.error("acquire requires --phase")
        acquire(args.out,args.phase)
    if args.command=="run":
        acquire(args.out,"calibration")
    if args.command in ("freeze","run"):
        freeze(args.out)
    if args.command=="run":
        acquire(args.out,"validation")
    if args.command in ("summarize","run"):
        summarize(args.out)


if __name__=="__main__":
    main()
