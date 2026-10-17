"""Finite generation/layer/microbatch IR with explicit versioned tensor swaps.

Expresses FlexGen's block traversal and same-layer weight reuse. This synthetic
template does not implement its LP, compression or CPU-attention delegation.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

from llm_infer_model.tensor.spec import InvalidInput, Unsupported, SpecError, build_scenario, ROUTE_ENDS
from .artifacts import write_json
from .strategy_ir import parse_strategy_ir
from .strategy_compiler import compile_strategy, _read_object
from .strategy_compiler import run_strategy
from .strategy import write_run

FIELDS = {"format_version", "id", "provenance", "generations", "initial_kv_positions", "layers",
          "microbatches", "activation_home", "kv_home", "order", "weight_release", "hardware", "transfer_entries"}


def _int(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise InvalidInput(f"{name} must be an integer >= {minimum}")
    return value


def _tensor(tid, role, size, locations=()):
    if size % 4:
        raise Unsupported("block template materializes contiguous F32 ports; bytes must be multiples of four")
    return {"id": tid, "name": tid, "role": role, "dtype": "GGML_TYPE_F32",
            "ne": [size // 4, 1, 1, 1], "nb": [4, size, size, size], "storage_bytes": size,
            "initial_locations": list(locations)}


@dataclass(frozen=True)
class BlockCompilation:
    document: dict
    workload: dict
    scenario: dict
    strategy: dict
    compiled: object


def compile_blocks(document, *, sources=()):
    if not isinstance(document, dict) or set(document) != FIELDS or document["format_version"] != "block-execution-0.1":
        raise InvalidInput("block template requires exactly the block-execution-0.1 fields")
    if document["provenance"] != "synthetic_block_template":
        raise Unsupported("block template currently requires synthetic_block_template provenance")
    if not isinstance(document["id"], str) or not document["id"]:
        raise InvalidInput("block template id must be nonempty")
    generations = _int(document["generations"], "generations", 1)
    initial = _int(document["initial_kv_positions"], "initial_kv_positions")
    if document["order"] not in ("layer_major", "batch_major") or document["weight_release"] not in ("after_group", "retain"):
        raise Unsupported("block order/weight_release is unsupported")
    for name in ("activation_home", "kv_home"):
        if document[name] not in ("dram", "ssd"):
            raise Unsupported(f"{name} currently requires dram or ssd for explicit GPU swaps")
    layers, batches = document["layers"], document["microbatches"]
    for rows, fields, name in ((layers, {"id", "weight_storage_bytes", "weight_home", "duration_ns", "workspace_bytes"}, "layers"),
                              (batches, {"id", "hidden_storage_bytes", "kv_storage_bytes"}, "microbatches")):
        if not isinstance(rows, list) or not rows:
            raise InvalidInput(f"{name} must be a nonempty array")
        ids = set()
        for row in rows:
            if not isinstance(row, dict) or set(row) != fields or not isinstance(row["id"], str) or not row["id"] or row["id"] in ids:
                raise InvalidInput(f"{name} entries have missing/unknown fields or duplicate IDs")
            ids.add(row["id"])
            for key in fields - {"id", "weight_home"}:
                _int(row[key], key, 0 if key == "workspace_bytes" else 1)
            if name == "layers" and row["weight_home"] not in ("dram", "ssd", "vram"):
                raise Unsupported("weight_home must be dram/ssd/vram")
    prices = {}
    if not isinstance(document["transfer_entries"], list):
        raise InvalidInput("transfer_entries must be an explicit array")
    for row in document["transfer_entries"]:
        if not isinstance(row, dict) or set(row) != {"route", "storage_bytes", "duration_ns"} or row["route"] not in ROUTE_ENDS:
            raise InvalidInput("invalid block transfer price")
        key = row["route"], _int(row["storage_bytes"], "transfer storage_bytes", 1)
        if key in prices:
            raise InvalidInput("duplicate exact block transfer cost")
        prices[key] = _int(row["duration_ns"], "transfer duration_ns", 1)
    tensors, operations, tasks, goals = [], [], [], {}
    for layer in layers:
        tensors.append(_tensor("w:" + layer["id"], "weight", layer["weight_storage_bytes"], (layer["weight_home"],)))
    hidden, kvs = {}, {}
    for batch in batches:
        tid = "x:" + batch["id"]; hidden[batch["id"]] = tid
        tensors.append(_tensor(tid, "input", batch["hidden_storage_bytes"], (document["activation_home"],)))
        for layer in layers:
            key = layer["id"], batch["id"]; kvs[key] = []
            for position in range(initial):
                tid = f"kv:{layer['id']}:{batch['id']}:{position}"
                tensors.append(_tensor(tid, "input", batch["kv_storage_bytes"], (document["kv_home"],)))
                kvs[key].append(tid)
    for generation in range(generations):
        order = ((layer, batch) for layer in layers for batch in batches) if document["order"] == "layer_major" else (
            (layer, batch) for batch in batches for layer in layers)
        for layer, batch in order:
            layer_id, batch_id = layer["id"], batch["id"]; key = layer_id, batch_id
            oid = f"op:g{generation}:{layer_id}:{batch_id}"
            out = f"h:g{generation}:{layer_id}:{batch_id}"
            kv = f"kv:{layer_id}:{batch_id}:{initial + generation}"
            inputs = [hidden[batch_id], "w:" + layer_id, *kvs[key]]
            tensors.append(_tensor(out, "output" if generation == generations - 1 and layer is layers[-1] else "intermediate",
                                   batch["hidden_storage_bytes"]))
            tensors.append(_tensor(kv, "output", batch["kv_storage_bytes"]))
            operations.append({"id": oid, "semantic_op": "dense_ffn", "ggml_op": "SYNTHETIC_MACRO",
                               "unary_op": None, "op_params": [], "inputs": inputs, "output": out, "extra_outputs": [kv]})
            tasks.append((generation, layer, batch, oid, tuple(inputs), out, kv))
            hidden[batch_id] = out; kvs[key].append(kv); goals[kv] = document["kv_home"]
    for tid in hidden.values(): goals[tid] = document["activation_home"]
    workload = {"schema_version": "0.2", "id": document["id"], "origin": {"kind": "synthetic_layer"},
                "tensors": tensors, "operations": operations, "outputs": list(goals)}
    routes = {route: [t["id"] for t in tensors] for route in ROUTE_ENDS}
    transfers = {}
    for route, ids in routes.items():
        transfers[route] = {}
        for tensor in tensors:
            key = route, tensor["storage_bytes"]
            if key not in prices:
                raise Unsupported(f"missing exact block transfer cost {key}")
            transfers[route][tensor["id"]] = {"duration_ns": prices[key]}
    scenario = {"schema_version": "0.2", "id": document["id"], "architecture": document["hardware"],
                "costs": {"source": "synthetic_block_template", "compute": {
                    oid: {"gpu": {"duration_ns": layer["duration_ns"], "workspace_bytes": layer["workspace_bytes"]}}
                    for _, layer, _, oid, _, _, _ in tasks}, "transfer": transfers},
                "mapspace": {"allowed_devices": {oid: ["gpu"] for _, _, _, oid, _, _, _ in tasks},
                             "allowed_transfers": routes, "allow_eviction": True,
                             "allow_copy_compute_overlap": True, "allow_cpu_gpu_overlap": True},
                "goal": {"output_locations": goals}, "mapper": {"algorithm": "uniform_cost"}}
    built = build_scenario(workload, scenario)
    resident = {t["id"]: set(t["initial_locations"]) for t in tensors}; program = []
    def transfer(tid, destination):
        if destination in resident[tid]: return
        source = "dram" if "dram" in resident[tid] else "vram" if "vram" in resident[tid] else "ssd"
        if {source, destination} == {"ssd", "vram"}:
            transfer(tid, "dram"); source = "dram"
        program.append({"kind": "transfer", "tensor_id": tid, "src": source, "dst": destination})
        program.append({"kind": "wait", "operations": [], "copies": [{"tensor_id": tid, "location": destination}]})
        resident[tid].add(destination)
    def release(tid, location):
        if location in resident[tid]:
            program.append({"kind": "release", "tensor_id": tid, "location": location}); resident[tid].remove(location)
    for index, (generation, layer, batch, oid, inputs, out, kv) in enumerate(tasks):
        for tid in inputs: transfer(tid, "vram")
        program.append({"kind": "compute", "operation_id": oid})
        program.append({"kind": "wait", "operations": [oid], "copies": []})
        resident[out].add("vram"); resident[kv].add("vram")
        for tid in inputs:
            if tid.startswith("w:"): continue
            if "vram" not in next(t["initial_locations"] for t in tensors if t["id"] == tid): release(tid, "vram")
        end_group = document["order"] == "batch_major" or index == len(tasks) - 1 or tasks[index + 1][1]["id"] != layer["id"] or tasks[index + 1][0] != generation
        if end_group and document["weight_release"] == "after_group" and layer["weight_home"] != "vram":
            release("w:" + layer["id"], "vram")
        # Deliver before dropping a GPU version; last goal completion is the
        # final wait and no cleanup follows it.
        transfer(out, document["activation_home"])
        release(out, "vram")
        if index != len(tasks) - 1:
            transfer(kv, document["kv_home"]); release(kv, "vram")
        else:
            transfer(kv, document["kv_home"])
    strategy = {"format_version": "strategy-ir-0.1", "id": document["id"], "scenario_file": "compiled.scenario.json",
                "information_mode": "offline_fixed", "regions": [], "placements": {o["id"]: "gpu" for o in operations},
                "program": program, "device_states": {}}
    compiled = compile_strategy(parse_strategy_ir(strategy), built, sources=sources)
    return BlockCompilation(document, workload, scenario, strategy, compiled)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("blocks", type=Path); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
            raise InvalidInput("output must be new or empty")
        data = args.blocks.read_bytes(); source = ({"path": str(args.blocks.resolve()), "sha256": hashlib.sha256(data).hexdigest()},)
        compiled = compile_blocks(_read_object(args.blocks, data), sources=source)
        result = run_strategy(compiled.compiled)
        write_run(args.out, compiled.compiled, result)
        write_json(args.out / "blocks.json", compiled.document)
        write_json(args.out / "workload.json", compiled.workload)
        write_json(args.out / "scenario.json", compiled.scenario)
        summary = json.loads((args.out / "summary.json").read_bytes())
        summary.update(paper_mechanism="flexgen_block_traversal", paper_strategy_reproduced=False,
                       compression_supported=False, cpu_attention_delegation_supported=False,
                       ssd_capacity_modeled=False, finite_unrolled_graph=True)
        write_json(args.out / "summary.json", summary)
    except (SpecError, OSError, ValueError) as error:
        print(f"blocks: {error}", file=sys.stderr); return 2
    print(json.dumps({"status": result.evaluation.status, "makespan_ns": result.evaluation.makespan_ns}))
    return 0 if result.evaluation.status == "valid" else 3


if __name__ == "__main__":
    raise SystemExit(main())
