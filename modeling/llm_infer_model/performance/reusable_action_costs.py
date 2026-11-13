"""Serial composition of independently measured semantic action responses.

The reusable coordinate deliberately contains no program, action ordinal, tensor
identifier, strategy name, seed or end-to-end measurement. It preserves the exact
kernel semantics, operand geometry/layout and runtime/hardware boundary instead.
Inclusive response prices already contain native dispatch and synchronization;
this model adds no guessed switching correction and rejects unknown coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import os
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def semantic_action(program, index):
    action = program["actions"][index]
    tensors = {t["id"]: t for t in program["graph"]["workload"]["tensors"]}
    def geometry(tid):
        tensor = tensors[tid]
        return {key: tensor[key] for key in ("dtype", "ne", "nb", "storage_bytes")}
    kind = action["kind"]
    if kind == "COMPUTE":
        binding = next(b for b in program["bindings"] if b["operation_id"] == action["target_id"])
        return dict(kind=kind, device=action["device"], semantic_op=binding["kind"],
            shape=binding["shape"], op_params=binding["op_params"],
            operands={role: geometry(tid) for role, tid in binding["operands"].items()},
            outputs={role: geometry(tid) for role, tid in binding["outputs"].items()},
            history=[dict(tokens=b["tokens"], first_position=b["first_position"], geometry=geometry(b["id"])) for b in binding["history"]],
            new_kv=[dict(tokens=b["tokens"], first_position=b["first_position"], geometry=geometry(b["id"])) for b in binding["new_kv"]],
            config=program["physical"]["config"], threads=program["physical"]["threads"],
            port_boundary=program["boundary_contract"])
    if kind == "COPY":
        row = dict(kind=kind, src=action["src"], dst=action["dst"], tensor=geometry(action["target_id"]))
        # A copy's measured cost depends on what else the device is doing, and no
        # other field of the key carries that.  DEPTH_COPY_KEY=placement declares
        # the placement (which operator kind runs where) part of the copy's
        # identity, so one key no longer pools an all-CPU program with a hybrid
        # one.  Off by default: every frozen table keeps the keys it was audited
        # with.
        if os.environ.get("DEPTH_COPY_KEY") == "placement":
            where = {}
            for other in program["actions"]:
                if other.get("kind") != "COMPUTE" or not other.get("target_id"):
                    continue
                binding = next((b for b in program["bindings"]
                                if b["operation_id"] == other["target_id"]), None)
                if binding is not None:
                    where[binding["kind"]] = other.get("device")
            row["placement"] = dict(sorted(where.items()))
        return row
    if kind in {"ADVANCE", "EVICT"}:
        return dict(kind=kind, native_behavior="logical_control_no_allocation_or_wait")
    raise ValueError("unsupported native action: " + kind)


def coordinate(program, index, context):
    return dict(schema="reusable-native-action-coordinate-1", semantic=semantic_action(program, index), context=context)


@dataclass(frozen=True)
class ReusableResponseTable:
    entries: tuple[dict, ...]

    def __post_init__(self):
        keys = set()
        for row in self.entries:
            key = digest(row["coordinate"])
            if key != row["key"] or key in keys:
                raise ValueError("changed/duplicate reusable coordinate")
            keys.add(key)
            if not row.get("evidence") or row.get("sample_count", 0) < 1:
                raise ValueError("independent measurement evidence required")
            value = row.get("mean_ns")
            if value is None or Fraction(value) < 0:
                raise ValueError("unknown or negative response")

    def lookup(self, value):
        key = digest(value)
        for row in self.entries:
            if row["key"] == key and row["coordinate"] == value:
                return Fraction(row["mean_ns"])
        raise KeyError("unknown exact semantic response coordinate: " + key)

    def evaluate(self, program, context):
        durations = tuple(self.lookup(coordinate(program, i, context)) for i in range(len(program["actions"])))
        # Serial native executor: every compute/copy completes before its successor.
        return dict(duration_ns=sum(durations, Fraction()), action_ns=durations,
            composition="serial inclusive completed action sum", extra_transition_ns=0,
            extra_transition_assumption="not separately identified; heldout tests this omission",
            overlapping_execution=False, target_outcome_used=False)


def predict_program(program, frozen):
    """Public ledger for architecture studies; unknown workspace stays unknown."""
    result = ReusableResponseTable(tuple(frozen["entries"])).evaluate(program, frozen["context"])
    rows = []
    for index, (action, duration) in enumerate(zip(program["actions"], result["action_ns"])):
        semantic = semantic_action(program, index)
        rows.append(dict(index=index, kind=action["kind"], duration_ns=duration,
            device=action.get("device"), semantic=semantic, workspace_bytes=None,
            workspace_scope="not exported by this steady executor; CUDA private workspace unknown",
            inclusive_dispatch_and_synchronization=True,
            key=digest(coordinate(program, index, frozen["context"]))))
    return dict(total_ns=result["duration_ns"], action_rows=rows, extra_overhead_rows=[],
        missing_extra_mechanisms=result["extra_transition_assumption"],
        full_memory_qualification=False, primary_predictor="independent isolated semantic response sum")
