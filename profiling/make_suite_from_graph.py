"""Extract resident MUL_MAT cases from the existing GGML graph collector format."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from profile_contract import validate_case


def extract_suite(document: dict, *, graph_id: str | None, layer_ids: set[int] | None,
                  tokens: list[int], dtype_mode: str = "native") -> dict:
    if not isinstance(document, dict) or not isinstance(document.get("graphs"), list):
        raise ValueError("expected collector JSON with a graphs array")
    if not tokens or any(isinstance(t, bool) or not isinstance(t, int) or t < 1 for t in tokens):
        raise ValueError("tokens must be a nonempty list of positive integers")
    if dtype_mode not in {"native", "f32"}:
        raise ValueError("dtype_mode must be native or f32")
    selected = [g for g in document["graphs"] if graph_id is None or g.get("id") == graph_id]
    if not selected:
        raise ValueError(f"graph id not found: {graph_id}")
    unique = {}
    skipped = []
    for graph in selected:
        tensors = {t["id"]: t for t in graph.get("tensors", [])}
        for node in graph.get("nodes", []):
            if node.get("op") != "MUL_MAT" or layer_ids is not None and node.get("layer_id") not in layer_ids:
                continue
            inputs = node.get("input_tensor_ids", [])
            if not inputs or inputs[0] not in tensors:
                raise ValueError(f"missing weight input for node {node.get('id')}")
            weight = tensors[inputs[0]]
            if weight.get("role") not in {"weight", "parameter"}:
                skipped.append({"node": node["id"], "reason": "MUL_MAT without a persistent weight; this probe covers resident projections"})
                continue
            shape = weight.get("shape", [])
            if weight.get("is_view") or len(shape) != 2:
                skipped.append({"node": node["id"], "reason": "weight must be a non-view 2D tensor"})
                continue
            original_dtype = str(weight.get("ggml_type", "")).lower()
            dtype = "f32" if dtype_mode == "f32" else original_dtype
            if dtype not in {"f32", "f16", "q4_k", "q6_k"}:
                skipped.append({"node": node["id"], "reason": f"unsupported weight dtype {original_dtype}"})
                continue
            k, m = shape
            for n in sorted(set(tokens)):
                key = (m, k, n, dtype)
                if key not in unique:
                    case = validate_case({"id": f"mulmat_m{m}_k{k}_t{n}_{dtype}", "kind": "compute",
                                          "m": m, "k": k, "n": n, "dtype": dtype})
                    case["source"] = {"tensor_names": [], "node_ids": [], "original_dtypes": [],
                                      "shape_source": "exported GGML weight dimensions; requested token count",
                                      "dtype_mode": dtype_mode, "weight_values": "generated in probe; not original model weights",
                                      "fusion_note": "isolated MUL_MAT; actual backend fusion must be checked before binding a macro task"}
                    unique[key] = case
                source = unique[key]["source"]
                for field, value in (("tensor_names", weight.get("name")), ("node_ids", node["id"]), ("original_dtypes", original_dtype)):
                    if value not in source[field]:
                        source[field].append(value)
    if not unique:
        raise ValueError("no supported resident MUL_MAT cases matched the graph/layer selection")
    return {"schema_version": "measurement_suite_v1", "description": "Measured projections with real exported shapes and generated tensor contents",
            "model_source": document.get("model"), "source_label": document.get("source_label"),
            "compute_cases": sorted(unique.values(), key=lambda c: c["id"]), "transfer_cases": [], "skipped": skipped}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--graph-id", help="Select one exported graph, e.g. g1; default deduplicates all graphs")
    parser.add_argument("--layer-id", type=int, action="append", help="Repeat to select layers; default all")
    parser.add_argument("--tokens", default="1,32,128", help="Total tokens for each isolated matrix multiplication")
    parser.add_argument("--dtype-mode", choices=["native", "f32"], default="native")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        raw = args.graph.read_bytes()
        suite = extract_suite(json.loads(raw), graph_id=args.graph_id,
                              layer_ids=set(args.layer_id) if args.layer_id else None,
                              tokens=[int(n) for n in args.tokens.split(",")], dtype_mode=args.dtype_mode)
        suite["source_graph"] = {"file": args.graph.name, "sha256": hashlib.sha256(raw).hexdigest()}
        if args.output.exists():
            raise ValueError(f"output already exists: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(suite, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"Created {len(suite['compute_cases'])} cases; recorded {len(suite['skipped'])} skipped nodes: {args.output}")
        return 0
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
