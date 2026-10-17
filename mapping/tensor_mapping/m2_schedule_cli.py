"""Dense and MoE adapters for the shared heterogeneous tensor schedule search.

All budgets, state binding and reusable-copy quotas are explicit. This command
uses declared costs and performs no hardware measurement or calibration.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from llm_infer_model.m2.expand import load_and_compile
from llm_infer_model.m2.spec import to_state_document, to_state_in_document
from llm_infer_model.tensor.engine import evaluate_mapping, load_mapping
from llm_infer_model.tensor.spec import (
    InvalidInput, SpecError, build_scenario, scenario_fingerprint, workload_fingerprint,
)
from .artifacts import action_to_dict, events_document, states_document, write_json
from .m2_hybrid import _source_snapshot
from .m2_overlap_cli import overlap_metrics

FORMAT = "m2-schedule-episode-0.1"
STATE_BINDING = "declared_first_then_previous_state_out"
BASELINES = ("serial_lru", "demand_overlap_lru", "prefetch_lru", "prefetch_fifo",
             "prefetch_farthest", "prefetch_gpu")


@dataclass(frozen=True)
class ScheduleEpisode:
    id: str
    templates: tuple[Any, ...]
    reusable_cache_capacity_bytes: dict[str, int | None]
    placement_mode: str
    prefetch_depth: int
    overlap: bool
    baseline_policies: tuple[str, ...]
    source_inputs: dict[str, Any]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def episode_raw(path):
    from .schedule import SchedulePolicy
    raw = read(path)
    required = {"format_version", "id", "invocation_files", "state_binding",
                "reusable_cache_capacity_bytes", "placement_mode", "prefetch_depth",
                "overlap", "baseline_policies"}
    if not isinstance(raw, dict) or set(raw) != required:
        raise InvalidInput(f"schedule episode must explicitly contain {sorted(required)}")
    if raw["format_version"] != FORMAT or raw["state_binding"] != STATE_BINDING:
        raise InvalidInput("unsupported schedule episode format or state binding")
    if not isinstance(raw["id"], str) or not raw["id"].strip():
        raise InvalidInput("episode id must be nonempty")
    refs = raw["invocation_files"]
    if not isinstance(refs, list) or not refs or any(not isinstance(ref, str) or not ref.strip() for ref in refs):
        raise InvalidInput("invocation_files must explicitly list nonempty file references")
    SchedulePolicy(prefetch_depth=raw["prefetch_depth"], overlap=raw["overlap"],
                   managed_capacity_bytes=raw["reusable_cache_capacity_bytes"]).validate()
    if raw["placement_mode"] not in ("fixed_first_declared", "enumerate"):
        raise InvalidInput("placement_mode must be fixed_first_declared or enumerate")
    policies = raw["baseline_policies"]
    if (not isinstance(policies, list) or not policies
            or any(policy not in BASELINES for policy in policies)
            or len(set(policies)) != len(policies)):
        raise InvalidInput("baseline_policies must explicitly select unique supported names")
    if "prefetch_gpu" in policies and raw["placement_mode"] != "enumerate":
        raise InvalidInput("prefetch_gpu baseline requires enumerate placement_mode")
    return raw


def episode_snapshot(path):
    path = Path(path).resolve()
    raw = episode_raw(path)
    payload = path.read_bytes()
    calls = []
    for ref in raw["invocation_files"]:
        invocation_path = (path.parent / ref).resolve()
        call = _source_snapshot(invocation_path)
        invocation = read(invocation_path)
        if "route_trace" in invocation:
            try:
                canonical = json.dumps(invocation["route_trace"], sort_keys=True, separators=(",", ":"),
                                       ensure_ascii=False, allow_nan=False).encode("utf-8")
            except (TypeError, ValueError) as error:
                raise InvalidInput("route_trace must be finite canonical JSON") from error
            call["inline_route_trace"] = dict(covered_by="invocation.sha256",
                json_pointer="/route_trace", sha256=hashlib.sha256(canonical).hexdigest())
        calls.append(call)
    return dict(episode=dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest(),
                            size_bytes=len(payload)), calls=calls)


def load_episode(path):
    from .m2_schedule import M2ScheduleAdapter
    path = Path(path).resolve()
    before = episode_snapshot(path)
    raw = episode_raw(path)
    templates = tuple(load_and_compile((path.parent / ref).resolve()) for ref in raw["invocation_files"])
    adapter = M2ScheduleAdapter()
    adapter.validate_templates(templates)
    layer_hashes = {call.get("layer", {}).get("sha256") for call in before["calls"]}
    if None in layer_hashes or len(layer_hashes) != 1:
        raise InvalidInput("all calls must preserve one immutable layer document SHA256")
    if episode_snapshot(path) != before:
        raise InvalidInput("episode sources changed during loading")
    return ScheduleEpisode(raw["id"], templates, dict(raw["reusable_cache_capacity_bytes"]),
        raw["placement_mode"], raw["prefetch_depth"], raw["overlap"],
        tuple(raw["baseline_policies"]), before), adapter


def candidate_document(candidate, label):
    calls = []
    offset = 0
    for index, item in enumerate(candidate.invocations):
        evaluation = None if item.result is None else item.result.evaluation
        calls.append(dict(call_index=index, invocation_id=item.compiled.invocation.id,
            status=item.status, reason=item.reason, placements=dict(item.placements),
            start_ns=offset, makespan_ns=None if evaluation is None else evaluation.makespan_ns,
            peak_dram_bytes=None if evaluation is None else evaluation.peak_dram_bytes,
            peak_vram_bytes=None if evaluation is None else evaluation.peak_vram_bytes,
            h2d_bytes=None if evaluation is None else evaluation.h2d_bytes,
            d2h_bytes=None if evaluation is None else evaluation.d2h_bytes,
            ssd_read_bytes=None if evaluation is None else evaluation.ssd_read_bytes,
            ssd_write_bytes=None if evaluation is None else evaluation.ssd_write_bytes,
            eviction_count=None if item.plan is None else item.plan.eviction_count,
            schedule_complete=False if item.plan is None else item.plan.complete,
            schedule_termination_reason=None if item.plan is None else item.plan.termination_reason,
            overlap_metrics=None if evaluation is None else overlap_metrics(evaluation)))
        if evaluation is not None and evaluation.makespan_ns is not None:
            offset += evaluation.makespan_ns
    return dict(candidate_id=candidate.id, strategy_label=label, status=candidate.status,
        reason=candidate.reason, total_makespan_ns=candidate.total_makespan_ns,
        completed_prefix_time_ns=offset, declared_invocation_count=candidate.declared_invocation_count,
        cache_hits=candidate.cache_hits, cache_misses=candidate.cache_misses,
        eviction_count=candidate.eviction_count,
        cache_statistics="managed_input_residency_on_first_preparation_window_entry_for_selected_device",
        calls=calls)


def write_candidate(candidate, folder, label):
    folder.mkdir(parents=True)
    write_json(folder / "candidate.json", candidate_document(candidate, label))
    timeline = []
    offset = 0
    for index, item in enumerate(candidate.invocations):
        call_folder = folder / f"call-{index:03d}"
        call_folder.mkdir()
        compiled = item.compiled
        write_json(call_folder / "workload.json", compiled.workload_document)
        write_json(call_folder / "scenario.json", compiled.scenario_document)
        write_json(call_folder / "state_in.json", to_state_in_document(compiled.state_in, layer=compiled.layer.id))
        write_json(call_folder / "physical_boundary.json", compiled.physical_boundary)
        write_json(call_folder / "boundary_state_in.json", item.boundary_state_in)
        write_json(call_folder / "boundary_state_out.json", item.boundary_state_out)
        if item.plan is None:
            write_json(call_folder / "refusal.json", dict(status=item.status, reason=item.reason,
                error_code=getattr(item.error, "code", None), prefix_is_complete_mapping=False))
            continue
        write_json(call_folder / "schedule_plan.json", asdict(item.plan))
        mapping = dict(schema_version=compiled.scenario.schema_version, scenario_id=compiled.scenario.id,
            workload_fingerprint=workload_fingerprint(compiled.scenario.workload),
            scenario_fingerprint=scenario_fingerprint(compiled.scenario),
            actions=[action_to_dict(action, m1=True) for action in item.plan.actions])
        write_json(call_folder / "mapping.json", mapping)
        rebuilt = build_scenario(read(call_folder / "workload.json"), read(call_folder / "scenario.json"))
        loaded = load_mapping(call_folder / "mapping.json", rebuilt)
        replay = evaluate_mapping(rebuilt, loaded.actions)
        if not loaded.fingerprint_verified or replay != item.result.evaluation:
            raise InvalidInput("schedule export did not reproduce its verified mapping")
        write_json(call_folder / "events.json", events_document(mode="generic_schedule_episode",
            scenario=compiled.scenario, evaluation=replay, note="declared tensor schedule space"))
        write_json(call_folder / "states.json", states_document(mode="generic_schedule_episode",
            scenario=compiled.scenario, evaluation=replay, note="declared tensor schedule space"))
        if item.result.state_out is not None:
            write_json(call_folder / "state_out.json", to_state_document(item.result.state_out))
        for event in replay.events:
            raw = asdict(event)
            for field in ("t_ns", "start_ns", "end_ns"):
                if raw.get(field) is not None:
                    raw[field] += offset
            timeline.append(dict(call_index=index, **raw))
        if replay.makespan_ns is not None:
            offset += replay.makespan_ns
    write_json(folder / "timeline.json", dict(schema_version="generic_schedule_timeline_v1",
        events=timeline, total_makespan_ns=candidate.total_makespan_ns, overlap_between_invocations=False))


def execute_episode(path, output, *, max_expanded_states, max_generated_states, max_baseline_actions):
    from .schedule import SchedulePolicy
    from .schedule_search import run_schedule_sequence, search_schedule_episode
    folder = Path(output)
    if folder.exists() and any(folder.iterdir()):
        raise InvalidInput("schedule episode output directory must be absent or empty")
    if any(type(value) is not int or value < 1 for value in
           (max_expanded_states, max_generated_states, max_baseline_actions)):
        raise InvalidInput("all search/baseline budgets must be explicit positive integers")
    episode, adapter = load_episode(path)
    fixed = tuple({task.operation_id: task.device for task in compiled.tasks} for compiled in episode.templates)
    baselines = {}
    for index, label in enumerate(episode.baseline_policies):
        policy = SchedulePolicy(
            prefetch_depth=0 if label in ("serial_lru", "demand_overlap_lru") else episode.prefetch_depth,
            overlap=False if label == "serial_lru" else episode.overlap,
            eviction_kind="offline_farthest" if label == "prefetch_farthest" else "fifo" if label == "prefetch_fifo" else "lru",
            managed_capacity_bytes=episode.reusable_cache_capacity_bytes)
        selected = fixed
        if label == "prefetch_gpu":
            if any("gpu" not in compiled.scenario.mapspace.devices_for(task.operation_id)
                   for compiled in episode.templates for task in compiled.tasks):
                raise InvalidInput("prefetch_gpu requires every macro to explicitly allow GPU")
            selected = tuple({task.operation_id: "gpu" for task in compiled.tasks} for compiled in episode.templates)
        baselines[label] = run_schedule_sequence(episode.templates, adapter=adapter, policy=policy,
            placements=selected, max_actions=max_baseline_actions, index=index)
    policy = SchedulePolicy(prefetch_depth=episode.prefetch_depth, overlap=episode.overlap,
                            managed_capacity_bytes=episode.reusable_cache_capacity_bytes)
    seed = min((candidate for label, candidate in baselines.items()
                if label.startswith("prefetch_") and candidate.is_valid),
               key=lambda candidate: candidate.total_makespan_ns, default=None)
    result = search_schedule_episode(episode.templates, adapter=adapter, policy=policy,
        placements=fixed if episode.placement_mode == "fixed_first_declared" else None,
        max_expanded_states=max_expanded_states, max_generated_states=max_generated_states, incumbent=seed)
    if episode_snapshot(path) != episode.source_inputs:
        raise InvalidInput("episode sources changed during search")
    summary = dict(format_version="m2_generic_schedule_search/1", episode_id=episode.id,
        source_inputs=episode.source_inputs, state_binding=STATE_BINDING,
        state_boundary_contract="actual_KV_state_plus_explicit_READY_weight_residency",
        shared_scheduler="tensor_mapping.schedule", shared_searcher="tensor_mapping.schedule_search",
        workload_family=episode.templates[0].layer.ffn,
        cost_provenance=[compiled.costs_source for compiled in episode.templates],
        reusable_tensor_scope="all_layer_weights", reusable_cache_capacity_bytes=episode.reusable_cache_capacity_bytes,
        placement_mode=episode.placement_mode, prefetch_depth=episode.prefetch_depth, overlap_enabled=episode.overlap,
        cache_pool="aligned_READY_and_RESERVED_COPY_non_source_managed_reusable_copies",
        status=result.status, termination_reason=result.termination_reason,
        search_complete=result.complete, enumeration_complete=result.enumeration_complete,
        state_space_exhausted=result.termination_reason == "exhausted_declared_space",
        optimality_within_declared_space=result.optimality_proven,
        max_expanded_states=max_expanded_states, max_generated_states=max_generated_states,
        max_baseline_actions=max_baseline_actions, expanded_states=result.expanded_states,
        generated_states=result.generated_states, duplicate_states=result.duplicate_states,
        popped_states=result.popped_states, stale_pops=result.stale_pops, pruned_states=result.pruned_states,
        rejected_actions=result.rejected_actions, goal_states_generated=result.goal_states_generated,
        rejected_boundaries=result.rejected_boundaries, boundary_refusals=list(result.boundary_refusals),
        initial_refusal=result.initial_refusal, incumbent_seeded=result.incumbent_seeded,
        incumbent_generated=result.incumbent_generated,
        best_candidate_id=None if result.best is None else result.best.id,
        best_total_makespan_ns=None if result.best is None else result.best.total_makespan_ns,
        search_information_mode="offline_oracle", online_policy_claimed=False,
        hardware_measurement_performed=False, performance_calibrated=False,
        global_schedule_optimality_proven=False,
        search_scope=dict(compute_start_order="fixed_declared_operation_order",
            managed_copy_eviction_search=True, eviction_trigger="demand_or_preparation_window_capacity_pressure",
            prefetch_search=True, overlap_search=True, overlap_between_invocations=False,
            activation_eviction_search=False, recomputation_search=False),
        baselines=[candidate_document(candidate, label) for label, candidate in baselines.items()],
        best=None if result.best is None else candidate_document(result.best, "uniform_cost_search"),
        assumptions=["Resource legality, dependency gates, capacity and time all come from the shared tensor engine.",
            "Dense and MoE use the same scheduling algorithm; workload adapters supply their tensor graphs and persistent state.",
            "Source copies and protected tensors are not eviction targets; transfer reservations also consume the reusable quota.",
            "Dense legacy state_out records KV only; boundary_state_out additionally records actual READY weight copies.",
            "Declared static prices and independent resources are not a hardware performance calibration.",
            "Offline preexpanded route traces do not establish an online route prediction policy."])
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "summary.json", summary)
    for label, candidate in baselines.items():
        write_candidate(candidate, folder / "baselines" / label, label)
    if result.best is not None:
        write_candidate(result.best, folder / "best", "uniform_cost_search")
    if episode_snapshot(path) != episode.source_inputs:
        raise InvalidInput("episode sources changed during export")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-expanded-states", type=int, required=True)
    parser.add_argument("--max-generated-states", type=int, required=True)
    parser.add_argument("--max-baseline-actions", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        summary = execute_episode(args.episode, args.out, max_expanded_states=args.max_expanded_states,
            max_generated_states=args.max_generated_states, max_baseline_actions=args.max_baseline_actions)
    except (OSError, UnicodeError, json.JSONDecodeError, SpecError) as error:
        print(f"schedule episode refused: {error}", file=sys.stderr)
        return 1
    print(f"{summary['status']}: {summary['best_total_makespan_ns']} ns; "
          f"{summary['expanded_states']} expanded, {summary['generated_states']} generated")
    return 0 if summary["best"] is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
