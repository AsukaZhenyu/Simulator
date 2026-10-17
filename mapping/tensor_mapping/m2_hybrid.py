"""Enumerate Dense macro placements using the shared M2/M1 replay semantics.

This is a deliberately small search space: each macro chooses an allowed CPU or
GPU device, while ``plan_for`` supplies a serial materialized-port schedule. It
does not search transfer order, prefetch, overlap, eviction or recomputation.
An optimum here is therefore only an optimum among those serial plans, under
the declared costs and capacity budget. ``compiled.scenario`` remains available
to the general uniform-cost mapper for a larger action-space experiment.

Run from the mapping package environment::

    python -m tensor_mapping.m2_hybrid INVOCATION.json --out DIRECTORY

No hardware measurements are performed by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from llm_infer_model.m2.expand import CompiledInvocation, load_and_compile, plan_for
from llm_infer_model.m2.runner import LayerInvocationResult, run_invocation
from llm_infer_model.m2.spec import to_state_document
from llm_infer_model.tensor.engine import Action, MappingError, STATUS_VALID
from llm_infer_model.tensor.spec import (
    InvalidInput, SpecError, Unsupported, scenario_fingerprint, workload_fingerprint,
)

from .artifacts import action_to_dict, events_document, states_document, write_json

SEARCH_SPACE = "serial_materialized_macro_device_assignments"
HYBRID_FORMAT_VERSION = "m2-layer-0.5"
_DEVICE_ORDER = ("cpu", "gpu")

__all__ = [
    "SEARCH_SPACE", "HybridCandidate", "HybridSearchResult",
    "enumerate_device_assignments", "search_hybrid", "candidate_document",
    "summary_document", "write_search_artifacts", "main",
]


@dataclass(frozen=True)
class HybridCandidate:
    """One placement and the outcome of its shared-kernel replay."""

    index: int
    placements: tuple[tuple[str, str], ...]
    actions: tuple[Action, ...]
    result: LayerInvocationResult | None
    plan_error: MappingError | None = None

    @property
    def id(self) -> str:
        return f"candidate-{self.index:03d}"

    @property
    def status(self) -> str:
        return self.result.status if self.result is not None else "invalid_mapping"

    @property
    def reason(self) -> str:
        if self.result is not None:
            return self.result.evaluation.reason
        assert self.plan_error is not None
        return self.plan_error.message

    @property
    def makespan_ns(self) -> int | None:
        return self.result.makespan_ns if self.result is not None else None

    @property
    def is_valid(self) -> bool:
        return self.status == STATUS_VALID


@dataclass(frozen=True)
class HybridSearchResult:
    """An exhaustive comparison of the explicitly restricted serial space."""

    compiled: CompiledInvocation
    candidates: tuple[HybridCandidate, ...]
    best_index: int | None
    source_inputs: Mapping[str, Any] | None = None

    @property
    def best(self) -> HybridCandidate | None:
        return None if self.best_index is None else self.candidates[self.best_index]

    @property
    def ranking(self) -> tuple[HybridCandidate, ...]:
        # A valid replay always has a makespan; no numeric default for a refusal.
        feasible = [candidate for candidate in self.candidates if candidate.is_valid]
        assert all(candidate.makespan_ns is not None for candidate in feasible)
        return tuple(sorted(feasible, key=lambda candidate: (candidate.makespan_ns, candidate.index)))


def _check_scope(compiled: CompiledInvocation) -> None:
    if compiled.layer.format_version != HYBRID_FORMAT_VERSION:
        raise Unsupported(f"hybrid search requires {HYBRID_FORMAT_VERSION}")
    if compiled.invocation.batch != 1:
        raise Unsupported("hybrid serial macro search currently requires batch=1")
    if compiled.layer.ffn != "dense" or compiled.layer.attention != "full_causal":
        raise Unsupported("hybrid serial macro search currently covers Dense full_causal only")
    kinds = tuple(task.kind for task in compiled.tasks)
    if len(kinds) != 3 or set(kinds) != {"projection", "attention", "dense_ffn"}:
        raise Unsupported("hybrid serial macro search requires one projection, attention and dense_ffn")


def enumerate_device_assignments(compiled: CompiledInvocation) -> tuple[dict[str, str], ...]:
    """Return the Cartesian product of each macro's *declared* device choices.

    CPU precedes GPU to make ties reproducible. No device is added to the
    mapspace, so restricting a macro to one device also restricts this search.
    """
    _check_scope(compiled)
    choices: list[tuple[str, ...]] = []
    for task in compiled.tasks:
        allowed = compiled.scenario.mapspace.devices_for(task.operation_id)
        if not allowed or any(device not in _DEVICE_ORDER for device in allowed):
            raise InvalidInput(f"macro {task.operation_id!r} needs declared CPU/GPU choices")
        choices.append(tuple(device for device in _DEVICE_ORDER if device in allowed))
    operation_ids = tuple(task.operation_id for task in compiled.tasks)
    return tuple(dict(zip(operation_ids, devices)) for devices in itertools.product(*choices))


def search_hybrid(
    compiled: CompiledInvocation, *, source_inputs: Mapping[str, Any] | None = None,
) -> HybridSearchResult:
    """Evaluate every declared macro-device assignment, without reimplementing M1.

    Capacity/dependency rejections remain candidate outcomes. Schema errors,
    missing transfer costs, and unknown macro workspace remain input refusals;
    they must not be hidden by dropping a candidate from the comparison.
    """
    candidates: list[HybridCandidate] = []
    for index, placements in enumerate(enumerate_device_assignments(compiled)):
        actions: tuple[Action, ...] = ()
        try:
            actions = plan_for(compiled, placements=placements)
            result = run_invocation(compiled, actions=actions)
            candidate = HybridCandidate(index, tuple(placements.items()), actions, result)
        except MappingError as error:
            candidate = HybridCandidate(index, tuple(placements.items()), actions, None, error)
        candidates.append(candidate)
    feasible = [candidate for candidate in candidates if candidate.is_valid]
    assert all(candidate.makespan_ns is not None for candidate in feasible)
    best = min(feasible, key=lambda candidate: (candidate.makespan_ns, candidate.index), default=None)
    return HybridSearchResult(compiled, tuple(candidates), None if best is None else best.index, source_inputs)


def _identity(compiled: CompiledInvocation) -> dict[str, Any]:
    return {
        "invocation_id": compiled.invocation.id,
        "layer_id": compiled.layer.id,
        "scenario_id": compiled.scenario.id,
        "scenario_fingerprint": scenario_fingerprint(compiled.scenario),
        "workload_fingerprint": workload_fingerprint(compiled.scenario.workload),
        "cost_source": compiled.costs_source,
        "cost_provenance": sorted(compiled.cost_table.provenances),
    }


def candidate_document(candidate: HybridCandidate) -> dict[str, Any]:
    evaluation = None if candidate.result is None else candidate.result.evaluation
    error = candidate.plan_error
    return {
        "id": candidate.id,
        "index": candidate.index,
        "placements": dict(candidate.placements),
        "status": candidate.status,
        "reason": candidate.reason,
        "makespan_ns": candidate.makespan_ns,
        "peak_dram_bytes": None if evaluation is None else evaluation.peak_dram_bytes,
        "peak_vram_bytes": None if evaluation is None else evaluation.peak_vram_bytes,
        "h2d_bytes": None if evaluation is None else evaluation.h2d_bytes,
        "d2h_bytes": None if evaluation is None else evaluation.d2h_bytes,
        "ssd_read_bytes": None if evaluation is None else evaluation.ssd_read_bytes,
        "ssd_write_bytes": None if evaluation is None else evaluation.ssd_write_bytes,
        "action_count": None if evaluation is None else evaluation.action_count,
        "declared_action_count": len(candidate.actions),
        "error_code": error.code if error is not None else (None if evaluation is None else evaluation.error_code),
        "error_action_index": error.action_index if error is not None else (None if evaluation is None else evaluation.error_action_index),
        "error_t_ns": error.t_ns if error is not None else (None if evaluation is None else evaluation.error_t_ns),
        "actions": [action_to_dict(action, m1=True) for action in candidate.actions],
        "state_out": None if candidate.result is None or candidate.result.state_out is None
        else to_state_document(candidate.result.state_out),
    }


def summary_document(result: HybridSearchResult) -> dict[str, Any]:
    compiled = result.compiled
    return {
        "format_version": "m2_hybrid_search/1",
        **_identity(compiled),
        "source_inputs": result.source_inputs,
        "hardware_measurement_performed": False,
        "status": "optimal_within_declared_space" if result.best is not None else "infeasible_within_declared_space",
        "search_space": SEARCH_SPACE,
        "optimality_within_declared_space": result.best is not None,
        "global_schedule_optimality_proven": False,
        "search_scope": {
            "macro_device_assignment": True,
            "copy_policy": "serial_just_in_time_inputs_then_serial_goal_delivery",
            "compute_policy": "serial_materialized_macros_in_compiled_task_order",
            "transfer_order_search": False,
            "prefetch_search": False,
            "overlap_search": False,
            "eviction_search": False,
            "recomputation_search": False,
        },
        "candidate_count": len(result.candidates),
        "valid_candidate_count": len(result.ranking),
        "rejected_candidate_count": len(result.candidates) - len(result.ranking),
        "best_candidate_id": None if result.best is None else result.best.id,
        "ranking": [candidate.id for candidate in result.ranking],
        "physical_boundary": compiled.physical_boundary,
        "assumptions": [
            "Times and capacity verdicts use the declared cost table and hardware budget.",
            "Optimality applies only to the enumerated serial macro-device plans.",
            "No hardware measurements or end-to-end performance validation are performed here.",
            "The expanded scenario is also available to the general action-space mapper.",
        ],
        "candidates": [candidate_document(candidate) for candidate in result.candidates],
    }


def _mapping_document(compiled: CompiledInvocation, candidate: HybridCandidate) -> dict[str, Any]:
    return {
        "schema_version": compiled.scenario.schema_version,
        "scenario_id": compiled.scenario.id,
        "comment": f"{candidate.id}: {candidate.status}; serial macro-device placement only; no global schedule optimum claim.",
        "workload_fingerprint": workload_fingerprint(compiled.scenario.workload),
        "scenario_fingerprint": scenario_fingerprint(compiled.scenario),
        "actions": [action_to_dict(action, m1=True) for action in candidate.actions],
    }


def write_search_artifacts(result: HybridSearchResult, output: str | Path) -> None:
    """Write replayable candidates, logs, and the selected persistent state as LF JSON."""
    folder = Path(output)
    if folder.exists() and any(folder.iterdir()):
        raise InvalidInput("hybrid search output directory must be absent or empty")
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "summary.json", summary_document(result))
    write_json(folder / "workload.json", result.compiled.workload_document)
    write_json(folder / "scenario.json", result.compiled.scenario_document)
    for candidate in result.candidates:
        candidate_folder = folder / "candidates" / candidate.id
        candidate_folder.mkdir(parents=True, exist_ok=True)
        write_json(candidate_folder / "candidate.json", candidate_document(candidate))
        write_json(candidate_folder / "mapping.json", _mapping_document(result.compiled, candidate))
        evaluation = None if candidate.result is None else candidate.result.evaluation
        write_json(candidate_folder / "events.json", events_document(
            mode="hybrid_serial_search", scenario=result.compiled.scenario,
            evaluation=evaluation, note=SEARCH_SPACE,
        ))
        write_json(candidate_folder / "states.json", states_document(
            mode="hybrid_serial_search", scenario=result.compiled.scenario,
            evaluation=evaluation, note=SEARCH_SPACE,
        ))
    if result.best is not None:
        assert result.best.result is not None and result.best.result.state_out is not None
        write_json(folder / "best_mapping.json", _mapping_document(result.compiled, result.best))
        write_json(folder / "state_out.json", to_state_document(result.best.result.state_out))


def _source_snapshot(invocation_path: Path) -> dict[str, Any]:
    """Record file hashes, with inline costs covered by the invocation hash."""
    path = invocation_path.resolve()
    payload = path.read_bytes()
    raw = json.loads(payload.decode("utf-8"))
    if not isinstance(raw, dict):
        raise InvalidInput("invocation must be a JSON object")
    documents: dict[str, Any] = {
        "invocation": {
            "path": str(path), "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        },
    }
    for role, key in (("layer", "layer_file"), ("costs", "costs_file")):
        reference = raw.get(key)
        if isinstance(reference, str):
            source_path = (path.parent / reference).resolve()
            source_bytes = source_path.read_bytes()
            documents[role] = {
                "path": str(source_path), "sha256": hashlib.sha256(source_bytes).hexdigest(),
                "size_bytes": len(source_bytes),
            }
    if "costs" in raw:
        documents["inline_costs"] = {"covered_by": "invocation.sha256", "json_pointer": "/costs"}
    return documents


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("invocation", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        before = _source_snapshot(args.invocation)
        compiled = load_and_compile(args.invocation)
        after = _source_snapshot(args.invocation)
        if before != after:
            raise InvalidInput("source files changed during compilation; rerun from stable inputs")
        result = search_hybrid(compiled, source_inputs=after)
        write_search_artifacts(result, args.out)
    except (OSError, UnicodeError, json.JSONDecodeError, SpecError) as error:
        print(f"hybrid search refused input: {error}", file=sys.stderr)
        return 1
    best = result.best
    if best is None:
        print(f"no feasible serial macro-device plan among {len(result.candidates)} candidates")
        return 2
    print(f"{best.id}: {best.makespan_ns} ns; best among {len(result.candidates)} serial macro-device plans")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
