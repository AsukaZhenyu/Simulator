"""Budgeted MoE device-placement search over an explicitly offline route trace.

Each expanded macro chooses one declared CPU/GPU device. The shared M2 planner
supplies serial just-in-time copies, and the shared tensor engine verifies every
candidate. This module does not search overlap, prefetch, eviction, route choice,
or compute/copy ordering. A finite budget may stop enumeration before completion.

Even causal_replay gates expert access at router completion, placement search sees
the complete expanded routing trace. Its information mode is therefore always
offline_oracle; it is not a claim about an online adaptive scheduling policy.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from llm_infer_model.m2.expand import CompiledInvocation, load_and_compile, plan_for
from llm_infer_model.m2.runner import run_invocation
from llm_infer_model.m2.spec import to_state_document
from llm_infer_model.tensor.engine import MappingError
from llm_infer_model.tensor.spec import InvalidInput, SpecError, Unsupported
from .artifacts import events_document, states_document, write_json
from .m2_hybrid import HybridCandidate, candidate_document, _identity, _mapping_document, _source_snapshot

MOE_FORMAT_VERSION = "m2-layer-0.6"
SEARCH_SPACE = "serial_trace_expanded_moe_macro_device_assignments"
DEFAULT_MAX_CANDIDATES = 256
_DEVICE_ORDER = ("cpu", "gpu")

__all__ = ["SEARCH_SPACE", "MoeSearchResult", "enumerate_device_assignments",
           "search_moe", "summary_document", "write_search_artifacts", "main"]


@dataclass(frozen=True)
class MoeSearchResult:
    compiled: CompiledInvocation
    candidates: tuple[HybridCandidate, ...]
    best_index: int | None
    declared_candidate_count: int
    max_candidates: int
    source_inputs: Mapping[str, Any] | None = None

    @property
    def best(self) -> HybridCandidate | None:
        return None if self.best_index is None else self.candidates[self.best_index]

    @property
    def ranking(self) -> tuple[HybridCandidate, ...]:
        valid = [candidate for candidate in self.candidates if candidate.is_valid]
        assert all(candidate.makespan_ns is not None for candidate in valid)
        return tuple(sorted(valid, key=lambda candidate: (candidate.makespan_ns, candidate.index)))

    @property
    def enumeration_complete(self) -> bool:
        return len(self.candidates) == self.declared_candidate_count


def _budget(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InvalidInput("max_candidates must be a positive integer")
    return value


def _choices(compiled: CompiledInvocation) -> tuple[tuple[str, ...], ...]:
    if compiled.layer.format_version != MOE_FORMAT_VERSION:
        raise Unsupported(f"MoE search requires {MOE_FORMAT_VERSION}")
    if compiled.layer.ffn != "moe" or compiled.layer.attention != "full_causal":
        raise Unsupported("MoE serial search requires MoE full_causal")
    if not compiled.tasks:
        raise InvalidInput("MoE search has no expanded tasks")
    kinds = {task.kind for task in compiled.tasks}
    if not {"projection", "attention", "router", "expert_ffn", "combine"}.issubset(kinds):
        raise Unsupported("MoE search requires the expanded projection/attention/router/expert/combine graph")
    if getattr(compiled.invocation, "information_mode", None) not in {"causal_replay", "offline_oracle"}:
        raise InvalidInput("MoE invocation must explicitly declare information_mode")
    operation_ids = tuple(task.operation_id for task in compiled.tasks)
    if len(set(operation_ids)) != len(operation_ids):
        raise InvalidInput("expanded MoE operation ids must be unique")
    choices: list[tuple[str, ...]] = []
    for operation_id in operation_ids:
        allowed = compiled.scenario.mapspace.devices_for(operation_id)
        if not allowed or any(device not in _DEVICE_ORDER for device in allowed):
            raise InvalidInput(f"macro {operation_id!r} needs declared CPU/GPU choices")
        choices.append(tuple(device for device in _DEVICE_ORDER if device in allowed))
    return tuple(choices)


def enumerate_device_assignments(
    compiled: CompiledInvocation, *, max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> tuple[dict[str, str], ...]:
    """Return a reproducible budgeted prefix; never materialize the full product."""
    limit = _budget(max_candidates)
    choices = _choices(compiled)
    operation_ids = tuple(task.operation_id for task in compiled.tasks)
    return tuple(dict(zip(operation_ids, devices)) for devices in
                 itertools.islice(itertools.product(*choices), limit))


def search_moe(
    compiled: CompiledInvocation, *, max_candidates: int = DEFAULT_MAX_CANDIDATES,
    source_inputs: Mapping[str, Any] | None = None,
) -> MoeSearchResult:
    """Replay a bounded prefix of offline trace-informed serial placements.

    Mapping failures remain outcomes. Missing cost/unknown workspace and schema
    refusals propagate instead of shrinking the declared comparison silently.
    """
    limit = _budget(max_candidates)
    choices = _choices(compiled)
    total = math.prod(len(choice) for choice in choices)
    operation_ids = tuple(task.operation_id for task in compiled.tasks)
    candidates: list[HybridCandidate] = []
    for index, devices in enumerate(itertools.islice(itertools.product(*choices), limit)):
        placements = dict(zip(operation_ids, devices))
        actions = ()
        try:
            actions = plan_for(compiled, placements=placements)
            candidate = HybridCandidate(index, tuple(placements.items()), actions,
                                        run_invocation(compiled, actions=actions))
        except MappingError as error:
            candidate = HybridCandidate(index, tuple(placements.items()), actions, None, error)
        candidates.append(candidate)
    valid = [candidate for candidate in candidates if candidate.is_valid]
    assert all(candidate.makespan_ns is not None for candidate in valid)
    best = min(valid, key=lambda candidate: (candidate.makespan_ns, candidate.index), default=None)
    return MoeSearchResult(compiled, tuple(candidates), None if best is None else best.index,
                           total, limit, source_inputs)


def summary_document(result: MoeSearchResult) -> dict[str, Any]:
    if result.enumeration_complete:
        status = "optimal_within_declared_space" if result.best else "infeasible_within_declared_space"
    else:
        status = "feasible" if result.best else "unknown"
    return {
        "format_version": "m2_moe_search/1",
        **_identity(result.compiled),
        "source_inputs": result.source_inputs,
        "status": status,
        "hardware_measurement_performed": False,
        "performance_calibrated": False,
        "search_space": SEARCH_SPACE,
        "search_information_mode": "offline_oracle",
        "replay_information_mode": result.compiled.invocation.information_mode,
        "online_policy_claimed": False,
        "routing_trace_preexpanded": True,
        "route_source": result.compiled.invocation.route_source,
        "normalized_route_trace_sha256": hashlib.sha256(json.dumps(
            [asdict(entry) for entry in result.compiled.invocation.route_trace],
            sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")).hexdigest(),
        "optimality_within_declared_space": result.enumeration_complete and result.best is not None,
        "global_schedule_optimality_proven": False,
        "enumeration_complete": result.enumeration_complete,
        "termination_reason": "exhausted_declared_space" if result.enumeration_complete else "max_candidates",
        "max_candidates": result.max_candidates,
        "declared_candidate_count": result.declared_candidate_count,
        "candidate_count": len(result.candidates),
        "valid_candidate_count": len(result.ranking),
        "rejected_candidate_count": len(result.candidates) - len(result.ranking),
        "best_candidate_id": None if result.best is None else result.best.id,
        "ranking": [candidate.id for candidate in result.ranking],
        "physical_boundary": result.compiled.physical_boundary,
        "search_scope": {
            "expanded_macro_device_assignment": True,
            "route_choice_search": False,
            "copy_policy": "serial_just_in_time_inputs_then_serial_goal_delivery",
            "compute_policy": "serial_macros_in_compiled_task_order",
            "transfer_order_search": False,
            "prefetch_search": False,
            "overlap_search": False,
            "eviction_search": False,
            "recomputation_search": False,
        },
        "assumptions": [
            "All costs are synthetic; this is semantic and declared-space search validation.",
            "Every placement is chosen with knowledge of the entire expanded routing trace.",
            "Causal replay gates expert access at route revelation but is not an online policy claim.",
            "A truncated enumeration makes no optimum or infeasibility claim for unvisited candidates.",
            "Any optimum applies only to the declared serial placements and declared hardware budget.",
        ],
        "candidates": [candidate_document(candidate) for candidate in result.candidates],
    }


def write_search_artifacts(result: MoeSearchResult, output: str | Path) -> None:
    """Write exact replay plans, state and metrics; refuse stale nonempty outputs."""
    folder = Path(output)
    if folder.exists() and any(folder.iterdir()):
        raise InvalidInput("MoE search output directory must be absent or empty")
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
            mode="moe_serial_search", scenario=result.compiled.scenario,
            evaluation=evaluation, note=SEARCH_SPACE))
        write_json(candidate_folder / "states.json", states_document(
            mode="moe_serial_search", scenario=result.compiled.scenario,
            evaluation=evaluation, note=SEARCH_SPACE))
    if result.best is not None:
        assert result.best.result is not None and result.best.result.state_out is not None
        write_json(folder / "best_mapping.json", _mapping_document(result.compiled, result.best))
        write_json(folder / "state_out.json", to_state_document(result.best.result.state_out))


def _source_snapshot_moe(path: Path) -> dict[str, Any]:
    snapshot = _source_snapshot(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "route_trace" in raw:
        try:
            canonical = json.dumps(raw["route_trace"], sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise InvalidInput("route_trace must be finite canonical JSON") from error
        snapshot["inline_route_trace"] = {
            "covered_by": "invocation.sha256", "json_pointer": "/route_trace",
            "canonical_sha256": hashlib.sha256(canonical).hexdigest(),
            "route_source": raw.get("route_source"),
        }
    return snapshot


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("invocation", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    args = parser.parse_args(argv)
    try:
        before = _source_snapshot_moe(args.invocation)
        compiled = load_and_compile(args.invocation)
        after = _source_snapshot_moe(args.invocation)
        if before != after:
            raise InvalidInput("source files changed during compilation; rerun from stable inputs")
        result = search_moe(compiled, max_candidates=args.max_candidates, source_inputs=after)
        write_search_artifacts(result, args.out)
    except (OSError, UnicodeError, json.JSONDecodeError, SpecError) as error:
        print(f"MoE search refused input: {error}", file=sys.stderr)
        return 1
    status = summary_document(result)["status"]
    if result.best is None:
        print(f"{status}: no feasible plan in {len(result.candidates)}/{result.declared_candidate_count} compared placements")
        return 2
    print(f"{status}: {result.best.id}, {result.best.makespan_ns} ns; compared "
          f"{len(result.candidates)}/{result.declared_candidate_count} offline serial placements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
