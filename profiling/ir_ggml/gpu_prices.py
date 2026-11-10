#!/usr/bin/env python3
"""Measure the GPU half of a model's price table and hand it to `run.run`.

The frozen export under `prices/<model>/costs.json` is CPU-only, so every corpus topology
that places an operator on the accelerator is unpriceable (`no_price_for_device`).  This
module closes that gap without touching the frozen file:

    python profiling/ir_ggml/gpu_prices.py suite   --model qwen3-0.6b
    python profiling/run_dense_cuda_measurements.py --suite <suite> --kv-dtype f16 ...
    python profiling/ir_ggml/gpu_prices.py export  --model qwen3-0.6b --run <results dir>
    python -m profiling.ir_ggml.run --model qwen3-0.6b --prices prices/qwen3-0.6b/gpu_costs.json

`suite` takes its geometry from the CPU export, so a GPU price and a CPU price at the same
(kind, phase, tokens, visible_history) describe the same mathematics on two devices.  The
cache element type is `f16`, which is what the CPU table declares; `m2.dense_cuda_contract`
appends `--kv-dtype f16` for the same reason.

`export` copies nothing it did not measure: each entry carries the native record's own
median, its sample count and coefficient of variation, and the receipt path it came from.
A case that failed, or whose samples were unstable, is refused rather than exported with a
flag -- a price is either measured or absent.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIMULATOR = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import instantiate as inst  # noqa: E402

# The CUDA probe's stages, per envelope.  A sparse model's routed feed-forward is measurable
# under `moe_cuda_contract`; under the dense envelope it stays unpriceable on the accelerator
# and is reported as such rather than estimated.
CUDA_STAGES = ("projection", "attention", "dense_ffn")
SPARSE_STAGES = ("projection", "attention", "moe_ffn")
KV_DTYPE = "f16"
# A price whose repeats disagree with themselves cannot be defended as one number.
MAX_CV = 0.10

# The dense CUDA contract describes a block, not a sparse one.  A sparse model's CPU export
# therefore carries fields it cannot read: router/expert geometry.  The projection and
# attention stages never touch the router or the experts, so those fields can be dropped and
# recorded.  A query/key/value bias is different: it is part of the attention graph, so the
# dense contract declares it (``attention_bias``) and the CUDA probe builds the bias adds.
DENSE_CONFIG_KEYS = ("hidden", "intermediate", "heads", "kv_heads", "head_dim", "rope_dims",
                     "attention_gate", "qk_norm", "rope_mode", "rope_base", "norm_epsilon",
                     "attention_bias", "weight_dtypes")
SPARSE_ONLY_KEYS = ("experts", "top_k", "normalize_topk", "shared_intermediate")
SPARSE_ONLY_ROLES = ("router", "gate_exps", "up_exps", "down_exps", "gate_inp_shexp",
                     "gate_shexp", "up_shexp", "down_shexp")


def dense_config(model: str, kind: str, signature: dict) -> tuple[dict, dict]:
    """The dense subset of a CPU entry's config, plus a record of what was left out."""
    config = dict(signature["config"])
    extra = sorted(set(config) - set(DENSE_CONFIG_KEYS))
    unknown = [key for key in extra if key not in SPARSE_ONLY_KEYS]
    if unknown:
        raise SystemExit(f"{model}: {kind} config carries {unknown}, which the dense "
                         f"contract cannot express")
    weights = config.get("weight_dtypes") or {}
    dropped_roles = sorted(set(weights) & set(SPARSE_ONLY_ROLES))
    config = {key: value for key, value in config.items() if key in DENSE_CONFIG_KEYS}
    config["weight_dtypes"] = {role: dtype for role, dtype in weights.items()
                               if role not in SPARSE_ONLY_ROLES}
    # A sparse export has no dense `gate`/`up`/`down`: its feed-forward is the expert block.
    # That is fine for the projection and attention stages, which never touch those weights;
    # the contract only requires the roles of the stage the case names.
    record = {"dropped_keys": extra, "dropped_weight_roles": dropped_roles}
    return config, record


def _entries(model: str) -> list[dict]:
    document = json.loads((inst.EXPORTS[model] / "costs.json").read_text(encoding="utf-8"))
    return [entry for entry in document["entries"] if entry.get("device") == "cpu"]


def build_suite(model: str) -> tuple[dict, dict]:
    """One case per CPU price key the CUDA probe can measure, at the same geometry."""
    cases = []
    projections: dict[str, dict] = {}
    for entry in _entries(model):
        if entry["kind"] not in CUDA_STAGES:
            continue
        config, projection = dense_config(model, entry["kind"], entry["physical_signature"])
        case = {
            "id": f"gpu_{model}_{entry['kind']}_{entry['phase']}"
                  f"_q{entry['tokens']}_h{entry['visible_history']}",
            "kind": entry["kind"], "phase": entry["phase"], "batch": 1,
            "tokens": entry["tokens"], "visible_history": entry["visible_history"],
            "kv_dtype": KV_DTYPE,
            "config": config,
            "source": {"cpu_price": str(inst.EXPORTS[model] / "costs.json"),
                       "cpu_duration_ns": entry["duration_ns"]},
        }
        # The case itself must declare exactly the dense contract's fields, so what was
        # left out is recorded on the suite, keyed by case id.
        if projection["dropped_keys"] or projection["dropped_weight_roles"]:
            projections[case["id"]] = projection
        cases.append(case)
    if not cases:
        raise SystemExit(f"{model}: no CPU price at a CUDA-measurable stage")
    document = {"schema_version": "dense_layer_measurement_suite_v1",
                "comment": f"GPU price suite for {model}, geometry cloned from its CPU export; "
                           f"cache {KV_DTYPE} as the CPU table declares",
                "cases": cases}
    return document, projections


def build_sparse_suite(model: str) -> dict:
    """The same cells for a sparse model, under `moe_cuda_contract`.

    Here nothing is projected away: a sparse config is exactly what the sparse CUDA
    envelope declares, including the routed feed-forward that the dense envelope has no
    way to express.  The case therefore carries the CPU entry's config verbatim, and a
    GPU price and a CPU price at one key describe the same block on two devices.
    """
    cases = []
    for entry in _entries(model):
        if entry["kind"] not in SPARSE_STAGES:
            continue
        cases.append({
            "id": f"gpu_{model}_{entry['kind']}_{entry['phase']}"
                  f"_q{entry['tokens']}_h{entry['visible_history']}",
            "kind": entry["kind"], "phase": entry["phase"], "batch": 1,
            "tokens": entry["tokens"], "visible_history": entry["visible_history"],
            "kv_dtype": KV_DTYPE,
            "config": dict(entry["physical_signature"]["config"]),
            "source": {"cpu_price": str(inst.EXPORTS[model] / "costs.json"),
                       "cpu_duration_ns": entry["duration_ns"]},
        })
    if not cases:
        raise SystemExit(f"{model}: no CPU price at a CUDA-measurable sparse stage")
    return {"schema_version": "moe_layer_measurement_suite_v1",
            "comment": f"GPU price suite for the sparse block of {model}, geometry cloned "
                       f"from its CPU export; cache {KV_DTYPE} as the CPU table declares",
            "cases": cases}


def export(model: str, run_dirs: list[Path]) -> dict:
    """Turn one or more measurement runs into `{"prices": [...]}` for `run.table_for`.

    Several runs are merged by price key, an earlier run winning a key it already answered:
    a stage added to the suite later (`moe_ffn`) can therefore be measured separately
    without re-measuring the stages that were already priced.
    """
    document = json.loads((inst.EXPORTS[model] / "costs.json").read_text(encoding="utf-8"))
    cpu_keys = {(entry["kind"], entry["phase"], entry["tokens"], entry["visible_history"]):
                entry["duration_ns"] for entry in document["entries"] if entry.get("device") == "cpu"}
    prices, refused, seen = [], [], set()
    for run_dir in run_dirs:
        rows = json.loads((run_dir / "measurements.json").read_text(encoding="utf-8"))
        for row in rows:
            case = row["case"]
            key = (case["kind"], case["phase"], case["tokens"], case["visible_history"])
            if row.get("status") != "measured":
                refused.append({"key": key, "reason": row.get("error", "not measured")})
                continue
            summary = row["summary"]
            if summary["cv"] > MAX_CV:
                refused.append({"key": key, "reason": f"cv {summary['cv']:.4f} > {MAX_CV}"})
                continue
            if key in seen:
                refused.append({"key": key, "reason": f"already priced by {run_dir.name}"})
                continue
            seen.add(key)
            prices.append({
                "kind": case["kind"], "device": "gpu", "phase": case["phase"], "batch": 1,
                "tokens": case["tokens"], "visible_history": case["visible_history"],
                "duration_ns": summary["median_ns"],
                "workspace_bytes": None,
                "provenance": "measured_macro",
                "accounting": "per_request",
                "physical_signature": row["signature"]["physical_signature"],
                "contract_hash": row["signature"]["physical_contract_sha256"],
                "workspace_scope": "known_cuda_gallocr_capacity_excludes_private_backend_scratch",
                "workspace_source": "unavailable_cuda_backend_private_scratch_and_runtime_overheads",
                "measurement": {"profiles": [str(run_dir / row["raw_output"])],
                                "samples": row["native"]["samples"] if "samples" in row["native"] else None,
                                "cv": summary["cv"],
                                "cpu_price_ns": cpu_keys.get(key)},
            })
    if refused:
        for item in refused:
            print(f"refused {item['key']}: {item['reason']}", file=sys.stderr)
    return {"schema": "ir-ggml-gpu-prices/v1", "model": model, "kv_dtype": KV_DTYPE,
            "runs": [str(run_dir) for run_dir in run_dirs], "prices": prices}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("suite", "export"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--contract", choices=("dense", "sparse"), default="dense",
                        help="`sparse` is moe_cuda_contract: it declares the same block for a "
                             "model whose feed-forward is a routed expert set")
    parser.add_argument("--run", type=Path, action="append",
                        help="measurement directory (export only; repeat to merge runs)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    sparse = args.contract == "sparse"
    if args.action == "suite":
        out = args.out or SIMULATOR / "profiling" / "cases" / (
            f"moe_layer_cuda_{args.model}.json" if sparse else f"dense_layer_cuda_{args.model}.json")
        if sparse:
            document, projections = build_sparse_suite(args.model), {}
        else:
            document, projections = build_suite(args.model)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
        print(f"{len(document['cases'])} cases -> {out}")
        if projections:
            record = out.with_suffix(".projection.json")
            record.write_text(json.dumps(projections, indent=1) + "\n", encoding="utf-8")
            print(f"{len(projections)} projected configs -> {record}")
        return 0
    if not args.run:
        parser.error("export needs --run")
    document = export(args.model, args.run)
    out = args.out or inst.EXPORTS[args.model] / "gpu_costs.json"
    out.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    print(f"{len(document['prices'])} GPU prices -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
