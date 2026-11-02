"""Independent numerical oracle for selected-row F32 MoE execution.

This is a portable correctness check, not a performance probe. It produces no
timings or M2 cost entries. Whole-token scalar evaluation is compared with an
independently materialized gather / gate / up / SwiGLU / down / weighted
scatter-add graph. Routes and scores are explicit test inputs, not predicted
or claimed to come from model weights.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import sys


SCHEMA = "moe-rows-reference-0.1"
REPORT_SCHEMA = "moe-rows-verification-0.1"
FIELDS = {"format_version", "id", "dtype", "hidden_size", "intermediate_size",
          "expert_count", "top_k", "shared_expert", "seed", "atol", "rtol", "work_items"}


def _int(value, name, *, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} exceeds explicit verification budget {maximum}")
    return value


def _number(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be a finite number >= {minimum}")
    return value


def validate_case(document):
    if not isinstance(document, dict) or set(document) != FIELDS:
        raise ValueError("reference case must contain exactly the documented fields")
    case = deepcopy(document)
    if case["format_version"] != SCHEMA or case["dtype"] != "f32":
        raise ValueError("only explicit moe-rows-reference-0.1 F32 cases are supported")
    if not isinstance(case["id"], str) or not case["id"].strip():
        raise ValueError("id must be a nonempty string")
    for key, limit in (("hidden_size", 256), ("intermediate_size", 512),
                       ("expert_count", 64), ("top_k", 64)):
        _int(case[key], key, minimum=1, maximum=limit)
    if case["top_k"] > case["expert_count"]:
        raise ValueError("top_k cannot exceed expert_count")
    if not isinstance(case["shared_expert"], bool):
        raise ValueError("shared_expert must be an explicit boolean")
    _int(case["seed"], "seed", maximum=2**32 - 1)
    _number(case["atol"], "atol")
    _number(case["rtol"], "rtol")
    items = case["work_items"]
    if not isinstance(items, list) or not 1 <= len(items) <= 256:
        raise ValueError("work_items must contain 1..256 explicitly routed token rows")
    # Prevent accidental huge scalar workloads, even within individual limits.
    operations = len(items) * (case["top_k"] + int(case["shared_expert"]))
    if operations * case["hidden_size"] * case["intermediate_size"] > 4_000_000:
        raise ValueError("case exceeds the portable scalar verification budget")
    seen = set()
    for item in items:
        if not isinstance(item, dict) or set(item) != {
                "request_id", "absolute_position", "expert_ids", "scores"}:
            raise ValueError("each row needs request_id, absolute_position, expert_ids and scores")
        if not isinstance(item["request_id"], str) or not item["request_id"]:
            raise ValueError("request_id must be nonempty")
        _int(item["absolute_position"], "absolute_position")
        key = (item["request_id"], item["absolute_position"])
        if key in seen:
            raise ValueError("duplicate token work item")
        seen.add(key)
        experts, scores = item["expert_ids"], item["scores"]
        if not isinstance(experts, list) or len(experts) != case["top_k"]:
            raise ValueError("expert_ids must have exactly top_k entries")
        for expert in experts:
            _int(expert, "expert_id", maximum=case["expert_count"] - 1)
        if len(set(experts)) != len(experts):
            raise ValueError("a token cannot repeat the same routed expert")
        if not isinstance(scores, list) or len(scores) != case["top_k"]:
            raise ValueError("scores must align with expert_ids")
        for score in scores:
            _number(score, "score")
            if score > 1e6:
                raise ValueError("score exceeds the portable numerical verification budget")
    return case


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _matrix(rng, rows, cols, scale):
    # Non-power-of-two scaling avoids the old exact-binary-input blind spot.
    return [[f32(rng.uniform(-0.43, 0.43) * scale) for _ in range(cols)]
            for _ in range(rows)]


def materialize(case):
    case = validate_case(case)
    h, intermediate = case["hidden_size"], case["intermediate_size"]
    rng = random.Random(case["seed"])
    weights = {}
    for expert in [*(str(e) for e in range(case["expert_count"])),
                   *(["shared"] if case["shared_expert"] else [])]:
        weights[expert] = {
            "gate": _matrix(rng, intermediate, h, 1 / math.sqrt(h)),
            "up": _matrix(rng, intermediate, h, 1 / math.sqrt(h)),
            "down": _matrix(rng, h, intermediate, 1 / math.sqrt(intermediate)),
        }
    return {"hidden": _matrix(rng, len(case["work_items"]), h, 1), "weights": weights}


def _scalar_expert(row, weights):
    # Scalar per-token path: high precision reductions, F32 materialized ports.
    gate = [f32(math.fsum(w * x for w, x in zip(vector, row)))
            for vector in weights["gate"]]
    up = [f32(math.fsum(w * x for w, x in zip(vector, row)))
          for vector in weights["up"]]
    activation = [f32(f32(g / (1 + math.exp(-g))) * u) for g, u in zip(gate, up)]
    return [f32(math.fsum(w * x for w, x in zip(vector, activation)))
            for vector in weights["down"]]


def whole_token_reference(case, data):
    """No grouping, gathers or scatter indices; each token evaluated directly."""
    case = validate_case(case)
    outputs = []
    for item, row in zip(case["work_items"], data["hidden"]):
        contributions = []
        for expert, score in zip(item["expert_ids"], item["scores"]):
            result = _scalar_expert(row, data["weights"][str(expert)])
            contributions.append([f32(f32(score) * value) for value in result])
        if case["shared_expert"]:
            contributions.append(_scalar_expert(row, data["weights"]["shared"]))
        outputs.append([f32(math.fsum([row[col], *(values[col] for values in contributions)]))
                        for col in range(case["hidden_size"])])
    return outputs


def _staged_matmul(rows, weights):
    # Deliberately independent implementation and F32 reduction order.
    result = [[0.0] * len(weights) for _ in rows]
    for token in range(len(rows)):
        for output in range(len(weights)):
            for channel in range(len(rows[token])):
                product = f32(rows[token][channel] * weights[output][channel])
                result[token][output] = f32(result[token][output] + product)
    return result


def selected_row_graph(case, data, *, group_order=None):
    """Materialize all explicit stages and weighted scatter-add; keep raw ports."""
    case = validate_case(case)
    groups = defaultdict(list)
    for token, item in enumerate(case["work_items"]):
        for slot, expert in enumerate(item["expert_ids"]):
            groups[str(expert)].append((token, slot))
    if case["shared_expert"]:
        groups["shared"] = [(token, None) for token in range(len(case["work_items"]))]
    order = sorted(groups, key=lambda key: (key == "shared", int(key) if key != "shared" else 0))
    if group_order is not None:
        if len(group_order) != len(order) or set(group_order) != set(order):
            raise ValueError("group_order must be a permutation of every active group")
        order = list(group_order)
    output = deepcopy(data["hidden"])
    ports = {}
    for expert in order:
        rows = groups[expert]
        gathered = [list(data["hidden"][token]) for token, _ in rows]
        weights = data["weights"][expert]
        gate = _staged_matmul(gathered, weights["gate"])
        up = _staged_matmul(gathered, weights["up"])
        swiglu = [[f32(f32(g / (1 + math.exp(-g))) * u) for g, u in zip(grow, urow)]
                  for grow, urow in zip(gate, up)]
        down = _staged_matmul(swiglu, weights["down"])
        weighted = []
        for row_index, (token, slot) in enumerate(rows):
            score = 1.0 if slot is None else f32(case["work_items"][token]["scores"][slot])
            weighted.append([f32(score * value) for value in down[row_index]])
            for col, value in enumerate(weighted[-1]):
                output[token][col] = f32(output[token][col] + value)
        ports[expert] = {"token_indices": [token for token, _ in rows],
                         "route_slots": [slot for _, slot in rows],
                         "work_items": [{"request_id": case["work_items"][token]["request_id"],
                                         "absolute_position": case["work_items"][token]["absolute_position"]}
                                        for token, _ in rows],
                         "gather": gathered, "gate": gate, "up": up,
                         "swiglu": swiglu, "down": down, "weighted": weighted}
    return output, ports


def compare_outputs(reference, actual, *, atol, rtol):
    _number(atol, "atol")
    _number(rtol, "rtol")
    if len(reference) != len(actual) or any(len(a) != len(b) for a, b in zip(reference, actual)):
        raise ValueError("output shape mismatch")
    pairs = [(a, b) for rrow, arow in zip(reference, actual) for a, b in zip(rrow, arow)]
    if not pairs:
        raise ValueError("empty outputs cannot establish equivalence")
    finite = all(math.isfinite(a) and math.isfinite(b) for a, b in pairs)
    if not finite:
        return {"passed": False, "all_outputs_finite": False,
                "max_abs_error": None, "max_rel_error": None, "atol": atol, "rtol": rtol}
    return {"passed": all(abs(a - b) <= atol + rtol * abs(a) for a, b in pairs),
            "all_outputs_finite": True,
            "max_abs_error": max(abs(a - b) for a, b in pairs),
            "max_rel_error": max(abs(a - b) / max(abs(a), 1e-30) for a, b in pairs),
            "atol": atol, "rtol": rtol}


def _float_digest(rows):
    return hashlib.sha256(b"".join(struct.pack("<f", value) for row in rows for value in row)).hexdigest()


def verify_case(document):
    case = validate_case(document)
    data = materialize(case)
    reference = whole_token_reference(case, data)
    selected, groups = selected_row_graph(case, data)
    check = compare_outputs(reference, selected, atol=case["atol"], rtol=case["rtol"])
    h_bytes, i_bytes = case["hidden_size"] * 4, case["intermediate_size"] * 4
    request_tokens = defaultdict(int)
    for item in case["work_items"]:
        request_tokens[item["request_id"]] += 1
    accounting = {}
    for expert, ports in groups.items():
        selected_count = len(ports["token_indices"])
        involved = set(item["request_id"] for item in ports["work_items"])
        original_count = sum(request_tokens[request] for request in involved)
        accounting[expert] = {
            "row_count": selected_count, "whole_request_row_count": original_count,
            "selected_input_bytes": selected_count * h_bytes,
            "whole_request_input_bytes": original_count * h_bytes,
            "grouped_output_bytes": selected_count * h_bytes,
            "gate_bytes": selected_count * i_bytes, "up_bytes": selected_count * i_bytes,
            "swiglu_bytes": selected_count * i_bytes,
            "scatter_payload_bytes": selected_count * h_bytes,
            "weight_logical_bytes": 3 * case["hidden_size"] * case["intermediate_size"] * 4,
            "workspace_bytes": None,
            "workspace_status": "not_measured_by_numerical_oracle",
        }
    raw = {"case": case, "generated_inputs": data, "whole_token_output": reference,
           "selected_rows_output": selected, "group_ports": groups}
    report = {"schema_version": REPORT_SCHEMA, "case_id": case["id"], "numeric_check": check,
              "port_sha256": {"whole_token_output": _float_digest(reference),
                              "selected_rows_output": _float_digest(selected)},
              "row_accounting": accounting, "routing_mode": "explicit_prescribed_scored_trace",
              "scores_normalized_by_verifier": False, "random_weights_not_model_weights": True,
              "scalar_reference_verified": check["passed"],
              "GGML_execution_verified": False, "GPU_execution_verified": False,
              "performance_measured": False, "cost_table_exported": False,
              "workspace_measured": False,
              "numeric_scope": "MoE_FFN_after_attention_with_shared_and_residual; no_attention_or_router"}
    return report, raw


def _rows(value, name):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must contain explicit nonempty token row scopes")
    seen = set()
    for row in value:
        if not isinstance(row, dict) or set(row) != {"request_id", "absolute_position"}:
            raise ValueError(f"{name} has invalid work item")
        if not isinstance(row["request_id"], str) or not row["request_id"]:
            raise ValueError(f"{name} has invalid request_id")
        _int(row["absolute_position"], "absolute_position")
        key = (row["request_id"], row["absolute_position"])
        if key in seen:
            raise ValueError(f"{name} repeats a token row")
        seen.add(key)
    return tuple((row["request_id"], row["absolute_position"]) for row in value)


def validate_row_execution_document(workload_document, boundary, *, expected_groups=None):
    """Check expanded Modeling row scopes and logical bytes without importing it.

    This proves row accounting only. It does not make the synthetic task costs,
    whole expert weight aliases or allocator scratch physically calibrated.
    """
    metadata = boundary.get("row_execution") if isinstance(boundary, dict) else None
    if not isinstance(metadata, dict):
        raise ValueError("physical boundary must explicitly contain row_execution metadata")
    if metadata.get("format_version") != "moe-execution-0.1":
        raise ValueError("unsupported row execution metadata version")
    h_bytes = _int(metadata.get("hidden_bytes_per_token"), "hidden_bytes_per_token", minimum=1)
    variant = metadata.get("expert_variant")
    i_bytes = metadata.get("intermediate_bytes_per_token")
    if variant not in ("opaque", "swiglu_staged"):
        raise ValueError("row execution must explicitly name opaque or swiglu_staged")
    if variant == "swiglu_staged":
        _int(i_bytes, "intermediate_bytes_per_token", minimum=1)
    if h_bytes % 4 or (variant == "swiglu_staged" and i_bytes % 4):
        raise ValueError("F32 row widths must be multiples of four bytes")
    if not isinstance(workload_document, dict) or not isinstance(workload_document.get("tensors"), list) \
            or not isinstance(workload_document.get("operations"), list):
        raise ValueError("workload document must explicitly declare tensors and operations")
    for tensor in workload_document["tensors"]:
        if not isinstance(tensor, dict) or not isinstance(tensor.get("id"), str):
            raise ValueError("invalid workload tensor descriptor")
        _int(tensor.get("storage_bytes"), "tensor storage_bytes", minimum=1)
    for operation in workload_document["operations"]:
        if not isinstance(operation, dict) or not isinstance(operation.get("id"), str) \
                or not isinstance(operation.get("output"), str) or not isinstance(operation.get("inputs"), list):
            raise ValueError("invalid workload operation descriptor")
    tensors = {item["id"]: item for item in workload_document["tensors"]}
    operations = {item["id"]: item for item in workload_document["operations"]}
    if len(tensors) != len(workload_document["tensors"]) or len(operations) != len(workload_document["operations"]):
        raise ValueError("duplicate tensor or operation id")
    scopes = metadata.get("tensor_row_scopes")
    if not isinstance(scopes, dict):
        raise ValueError("tensor_row_scopes must be an explicit object")
    scope_rows = {name: _rows(rows, "tensor_row_scopes") for name, rows in scopes.items()}
    for name in scope_rows:
        if name not in tensors:
            raise ValueError("row scope references an unknown tensor")
    regions = metadata.get("regions")
    if not isinstance(regions, list) or not regions:
        raise ValueError("row_execution regions must be explicit and nonempty")
    actual_groups = {}
    checked = []
    request_scatter = defaultdict(list)
    for region in regions:
        if not isinstance(region, dict):
            raise ValueError("row execution region must be an object")
        rows = _rows(region.get("work_items"), "region work_items")
        expert = region.get("expert_id")
        key = "shared" if expert == "shared" else str(_int(expert, "expert_id"))
        if key in actual_groups:
            raise ValueError("duplicate expert row region")
        actual_groups[key] = rows
        for role in ("grouped_input_tensor", "grouped_output_tensor"):
            name = region.get(role)
            if name not in tensors or scope_rows.get(name) != rows:
                raise ValueError(f"{role} must match the region row order")
            if tensors[name]["storage_bytes"] != len(rows) * h_bytes:
                raise ValueError(f"{role} bytes must equal actual selected row count * hidden bytes")
        gather = region.get("gather_operation_id")
        if gather not in operations or operations[gather]["output"] != region["grouped_input_tensor"] \
                or operations[gather].get("semantic_op") != "gather_rows":
            raise ValueError("gather operation must produce the grouped input")
        expert_ids = region.get("expert_operation_ids")
        if not isinstance(expert_ids, list) or not expert_ids:
            raise ValueError("region must list its expert stages")
        kinds = [operations.get(oid, {}).get("semantic_op") for oid in expert_ids]
        expected_kinds = ["expert_ffn"] if variant == "opaque" else [
            "expert_gate", "expert_up", "expert_swiglu", "expert_down"]
        if kinds != expected_kinds:
            raise ValueError("expert stages must match the explicit variant without dropped or duplicated stages")
        for operation in expert_ids:
            if operation not in operations:
                raise ValueError("unknown expert stage")
            output = operations[operation]["output"]
            if output not in tensors or scope_rows.get(output) != rows:
                raise ValueError("expert stage row order differs from its group")
            kind = operations[operation].get("semantic_op")
            expected = i_bytes if kind in ("expert_gate", "expert_up", "expert_swiglu") else h_bytes
            if tensors[output]["storage_bytes"] != len(rows) * expected:
                raise ValueError("expert stage logical bytes mismatch")
        if operations[expert_ids[-1]]["output"] != region["grouped_output_tensor"]:
            raise ValueError("final expert stage must produce the declared grouped output")
        compact = region["grouped_input_tensor"]
        if variant == "opaque":
            if compact not in operations[expert_ids[0]]["inputs"]:
                raise ValueError("expert must consume its actual gathered rows")
        else:
            gate, up, swiglu, down = (operations[oid] for oid in expert_ids)
            if compact not in gate["inputs"] or compact not in up["inputs"] \
                    or swiglu["inputs"] != [gate["output"], up["output"]] \
                    or swiglu["output"] not in down["inputs"]:
                raise ValueError("staged SwiGLU task dependencies do not match gate/up/activation/down")
        scatter_ids = region.get("scatter_operation_ids")
        if not isinstance(scatter_ids, list) or not scatter_ids:
            raise ValueError("region must list its request scatter stages")
        scattered = []
        request_seen = set()
        for operation in scatter_ids:
            if operation not in operations:
                raise ValueError("unknown scatter stage")
            output = operations[operation]["output"]
            scoped = scope_rows.get(output)
            if scoped is None or len({req for req, _ in scoped}) != 1:
                raise ValueError("scatter output must retain one request's selected row scope")
            request = scoped[0][0]
            if request in request_seen or scoped != tuple(row for row in rows if row[0] == request):
                raise ValueError("scatter stages must cover each selected row in its original request order")
            request_seen.add(request)
            if operations[operation].get("semantic_op") != "scatter_rows" \
                    or region["grouped_output_tensor"] not in operations[operation]["inputs"]:
                raise ValueError("scatter must consume the declared expert output")
            if tensors[output]["storage_bytes"] != len(scoped) * h_bytes:
                raise ValueError("scatter output logical bytes mismatch")
            scattered.extend(scoped)
            request_scatter[request].append(output)
        if len(scattered) != len(rows) or set(scattered) != set(rows):
            raise ValueError("scatter stages must cover each selected row exactly once")
        checked.append({"expert_id": expert, "row_count": len(rows),
                        "selected_input_bytes": len(rows) * h_bytes})
    # Real expansions retain one complete output for each request. Minimal
    # isolated-region documents may omit combine; their region checks remain
    # useful but do not claim full-layer graph coverage.
    combines = [op for op in operations.values() if op.get("semantic_op") == "combine_rows"]
    combined_requests = set()
    for combine in combines:
        rows = scope_rows.get(combine["output"])
        if rows is None or len({req for req, _ in rows}) != 1:
            raise ValueError("combine must produce one complete request's explicit row scope")
        request = rows[0][0]
        if request in combined_requests:
            raise ValueError("a request cannot have duplicate combine outputs")
        combined_requests.add(request)
        if tensors[combine["output"]]["storage_bytes"] != len(rows) * h_bytes:
            raise ValueError("combine output logical bytes mismatch")
        actual_inputs = [tid for tid in combine["inputs"] if tid in scope_rows]
        if len(actual_inputs) != len(request_scatter[request]) or set(actual_inputs) != set(request_scatter[request]):
            raise ValueError("combine must consume each own-request contribution exactly once")
        expected_rows = {row for group in actual_groups.values() for row in group if row[0] == request}
        if set(rows) != expected_rows:
            raise ValueError("combine output rows must cover the routed request exactly")
    if combines and combined_requests != set(request_scatter):
        raise ValueError("combine stages must cover every routed request")
    if expected_groups is not None:
        expected = {str(key): tuple(tuple(row) for row in rows) for key, rows in expected_groups.items()}
        if actual_groups != expected:
            raise ValueError("expanded expert groups differ from the true explicit routing rows")
    return {"row_accounting_verified": True, "region_count": len(checked), "regions": checked,
            "full_request_combine_checked": bool(combines),
            "numeric_execution_verified": False, "performance_calibrated": False}


def _dump(path, document):
    Path(path).write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def run_verification(case_path, output, *, workload_path=None, boundary_path=None):
    case_path, output = Path(case_path), Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output directory must be absent or empty; evidence is never overwritten")
    if (workload_path is None) != (boundary_path is None):
        raise ValueError("--workload and --boundary must be supplied together")
    sources = {str(case_path.resolve()): case_path.read_bytes(),
               str(Path(__file__).resolve()): Path(__file__).read_bytes()}
    report, raw = verify_case(json.loads(sources[str(case_path.resolve())]))
    if workload_path is not None:
        workload_path, boundary_path = Path(workload_path), Path(boundary_path)
        for path in (workload_path, boundary_path):
            sources[str(path.resolve())] = path.read_bytes()
        report["modeling_row_accounting"] = validate_row_execution_document(
            json.loads(sources[str(workload_path.resolve())]),
            json.loads(sources[str(boundary_path.resolve())]))
    if any(Path(path).read_bytes() != content for path, content in sources.items()):
        raise ValueError("a verification source changed while it was being checked")
    output.mkdir(parents=True, exist_ok=True)
    _dump(output / "raw.json", raw)
    report["raw_sha256"] = hashlib.sha256((output / "raw.json").read_bytes()).hexdigest()
    _dump(output / "report.json", report)
    _dump(output / "manifest.json", {"schema_version": REPORT_SCHEMA,
        "sources_sha256": {path: hashlib.sha256(content).hexdigest() for path, content in sources.items()},
        "artifacts_sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                             for name in ("raw.json", "report.json")},
        "python_version": sys.version, "implementation": "stdlib_scalar_F32_numerical_oracle",
        "backend_timing_measured": False, "physical_cost_export": False})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workload", type=Path)
    parser.add_argument("--boundary", type=Path)
    args = parser.parse_args(argv)
    try:
        report = run_verification(args.case, args.out, workload_path=args.workload, boundary_path=args.boundary)
        print(json.dumps({"case_id": report["case_id"], "numeric_check": report["numeric_check"],
                          "performance_measured": False, "output": str(args.out)}, allow_nan=False))
        return 0 if report["numeric_check"]["passed"] else 3
    except (ValueError, OSError, OverflowError) as exc:
        print(f"invalid_input: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
