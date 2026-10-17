"""Explicit MoE episode: cache eviction, input prefetch and engine overlap search.

Every plan is checked by the same tensor engine. Later calls receive the
candidate's actual preceding state_out. This entry point makes no measurements.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import shutil

from llm_infer_model.m2.expand import load_and_compile
from llm_infer_model.m2.runner import run_invocation
from llm_infer_model.m2.spec import state_in_from_state_out
from llm_infer_model.tensor.engine import MappingError
from llm_infer_model.tensor.spec import InitialCapacityExceeded, InvalidInput, SpecError
from .artifacts import write_json
from .m2_cache import CacheHistory, CachePolicy
from .m2_cache_cli import CacheEpisode, STATE_BINDING, _write_candidate, candidate_document, fixed_placements
from .m2_cache_search import CacheInvocationResult, CacheSequenceCandidate
from .m2_moe import _source_snapshot_moe

FORMAT = "m2-moe-schedule-episode-0.1"
BASELINES = ("serial_lru", "demand_overlap_lru", "prefetch_lru", "prefetch_fifo", "prefetch_farthest", "prefetch_gpu")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def episode_raw(path):
    raw = read(path)
    keys = {"format_version", "id", "invocation_files", "state_binding", "expert_cache_capacity_bytes",
            "placement_mode", "prefetch_depth", "overlap", "baseline_policies"}
    if not isinstance(raw, dict) or set(raw) != keys:
        raise InvalidInput(f"schedule episode must explicitly contain {sorted(keys)}")
    if raw["format_version"] != FORMAT or raw["state_binding"] != STATE_BINDING:
        raise InvalidInput("unsupported schedule episode format or state binding")
    if not isinstance(raw["id"], str) or not raw["id"].strip():
        raise InvalidInput("episode id must be nonempty")
    refs = raw["invocation_files"]
    if not isinstance(refs, list) or not refs or any(not isinstance(ref, str) or not ref.strip() for ref in refs):
        raise InvalidInput("invocation_files must explicitly list nonempty file references")
    CachePolicy(expert_cache_capacity_bytes=raw["expert_cache_capacity_bytes"]).validate()
    if raw["placement_mode"] not in ("fixed_first_declared", "enumerate"):
        raise InvalidInput("placement_mode must be fixed_first_declared or enumerate")
    if type(raw["prefetch_depth"]) is not int or raw["prefetch_depth"] < 0:
        raise InvalidInput("prefetch_depth must be an explicit nonnegative integer")
    if type(raw["overlap"]) is not bool:
        raise InvalidInput("overlap must be an explicit boolean")
    policies = raw["baseline_policies"]
    if (not isinstance(policies, list) or not policies or any(policy not in BASELINES for policy in policies)
            or len(set(policies)) != len(policies)):
        raise InvalidInput("baseline_policies must explicitly select unique supported names")
    if "prefetch_gpu" in policies and raw["placement_mode"] != "enumerate":
        raise InvalidInput("prefetch_gpu baseline requires enumerate placement_mode")
    return raw


def episode_snapshot(path):
    path = Path(path).resolve()
    raw = episode_raw(path)
    payload = path.read_bytes()
    return dict(episode=dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload)),
                calls=[_source_snapshot_moe((path.parent / ref).resolve()) for ref in raw["invocation_files"]])


def load_episode(path):
    path = Path(path).resolve()
    before = episode_snapshot(path)
    raw = episode_raw(path)
    templates = tuple(load_and_compile((path.parent / ref).resolve()) for ref in raw["invocation_files"])
    if any(item.layer.format_version != "m2-layer-0.6" or item.layer.ffn != "moe" for item in templates):
        raise InvalidInput("schedule episode requires single-layer synthetic m2-layer-0.6 MoE calls")
    layer_hashes = {call.get("layer", {}).get("sha256") for call in before["calls"]}
    if None in layer_hashes or len(layer_hashes) != 1 or any(item.layer != templates[0].layer for item in templates):
        raise InvalidInput("all calls must preserve one immutable layer document SHA256 and definition")
    if len({item.invocation.id for item in templates}) != len(templates):
        raise InvalidInput("invocation ids must be unique")
    if episode_snapshot(path) != before:
        raise InvalidInput("episode sources changed during loading")
    return raw, CacheEpisode(raw["id"], templates, raw["expert_cache_capacity_bytes"], raw["placement_mode"],
                            tuple(raw["baseline_policies"]), before)


def union_ns(intervals):
    end = None
    total = 0
    for start, stop in sorted((a, b) for a, b in intervals if a < b):
        if end is None or start >= end:
            total += stop - start
        elif stop > end:
            total += stop - end
        end = stop if end is None else max(end, stop)
    return total


def overlap_metrics(evaluation):
    events = [event for event in evaluation.events if event.resource is not None
              and event.start_ns is not None and event.end_ns is not None]
    by_resource = {}
    for event in events:
        by_resource.setdefault(event.resource, []).append((event.start_ns, event.end_ns))
    for resource, intervals in by_resource.items():
        ordered = sorted(intervals)
        if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
            raise InvalidInput(f"replay has simultaneous tasks on exclusive resource {resource}")
    def intersection(left, right):
        return union_ns((max(a.start_ns, b.start_ns), min(a.end_ns, b.end_ns))
                        for a in events if a.resource in left for b in events if b.resource in right)
    computes = {"cpu_compute", "gpu_compute"}
    transfers = {"pcie_copy", "ssd_io"}
    return dict(compute_transfer_overlap_ns=intersection(computes, transfers),
                cpu_gpu_overlap_ns=intersection({"cpu_compute"}, {"gpu_compute"}),
                ssd_pcie_overlap_ns=intersection({"ssd_io"}, {"pcie_copy"}),
                resource_busy_ns={resource: union_ns(intervals) for resource, intervals in sorted(by_resource.items())},
                interval_source="original_engine_events", exclusive_resources_verified=True)


def schedule_candidate_document(candidate, label):
    document = candidate_document(candidate)
    document["strategy_label"] = label
    document["cache_statistics"] = "expert_input_residency_on_first_entry_into_preparation_window_for_selected_device"
    for item, saved in zip(candidate.invocations, document["calls"]):
        saved["overlap_metrics"] = None if item.result is None else overlap_metrics(item.result.evaluation)
        saved["schedule_termination_reason"] = getattr(item.plan, "termination_reason", None)
    return document


def write_candidate(candidate, folder, label, state_artifact_format):
    """Preserve the legacy event format; optionally compress large state logs."""
    _write_candidate(candidate, folder)
    write_json(folder / "candidate.json", schedule_candidate_document(candidate, label))
    if state_artifact_format == "json-gzip":
        index = []
        for source in sorted(folder.glob("call-*/states.json")):
            destination = source.with_suffix(".json.gz")
            original = source.read_bytes()
            digest = hashlib.sha256(original).hexdigest()
            size = len(original)
            del original
            with source.open("rb") as stream, destination.open("wb") as target:
                with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as compressed:
                    shutil.copyfileobj(stream, compressed)
            with gzip.open(destination, "rb") as compressed:
                restored = compressed.read()
            if len(restored) != size or hashlib.sha256(restored).hexdigest() != digest:
                raise InvalidInput("compressed state artifact failed lossless verification")
            del restored
            index.append(dict(path=destination.relative_to(folder).as_posix(), compression="gzip",
                              uncompressed_sha256=digest, uncompressed_size_bytes=size))
            source.unlink()  # Only this freshly exported, verified redundant JSON file.
        write_json(folder / "state-artifacts.json", dict(format="json-gzip", artifacts=index))


def run_overlap_sequence(templates, *, placements, policy, max_actions, index=0):
    from .m2_overlap import compile_schedule_invocation, plan_overlap
    from .m2_cache import expert_accesses
    state_in = templates[0].state_in
    history = CacheHistory(templates[0].layer.id)
    invocations = []
    accesses = [expert_accesses(template, placement) for template, placement in zip(templates, placements)]
    for i, (template, placement) in enumerate(zip(templates, placements)):
        compiled = compile_schedule_invocation(template, state_in)
        try:
            plan = plan_overlap(compiled, placements=placement, policy=policy, history=history,
                                max_actions=max_actions, future_accesses=tuple(x for seq in accesses[i + 1:] for x in seq))
        except (MappingError, InitialCapacityExceeded) as error:
            invocations.append(CacheInvocationResult(compiled, tuple(placement.items()), None, None, error))
            break
        result = run_invocation(compiled, actions=plan.actions)
        invocations.append(CacheInvocationResult(compiled, tuple(placement.items()), plan, result))
        if result.status != "valid":
            break
        history = plan.history
        state_in = state_in_from_state_out(result.state_out)
    cache_policy = CachePolicy(kind=policy.eviction_kind, expert_cache_capacity_bytes=policy.expert_cache_capacity_bytes)
    return CacheSequenceCandidate(index, cache_policy, tuple(invocations), len(templates))


def execute_episode(path, output, *, max_expanded_states, max_generated_states, max_baseline_actions,
                    state_artifact_format="json"):
    from .m2_overlap import SchedulePolicy
    from .m2_overlap_search import search_schedule_episode
    folder = Path(output)
    if folder.exists() and any(folder.iterdir()):
        raise InvalidInput("schedule episode output directory must be absent or empty")
    if any(type(value) is not int or value < 1 for value in
           (max_expanded_states, max_generated_states, max_baseline_actions)):
        raise InvalidInput("all search/baseline budgets must be explicit positive integers")
    if state_artifact_format not in ("json", "json-gzip"):
        raise InvalidInput("state_artifact_format must be json or json-gzip")
    raw, episode = load_episode(path)
    fixed = fixed_placements(episode)
    baselines = {}
    for i, label in enumerate(episode.baseline_policies):
        kind = "offline_farthest" if label == "prefetch_farthest" else "fifo" if label == "prefetch_fifo" else "lru"
        policy = SchedulePolicy(prefetch_depth=0 if label in ("serial_lru", "demand_overlap_lru") else raw["prefetch_depth"],
                                overlap=False if label == "serial_lru" else raw["overlap"], eviction_kind=kind,
                                expert_cache_capacity_bytes=episode.expert_cache_capacity_bytes)
        baseline_placements = fixed
        if label == "prefetch_gpu":
            if any("gpu" not in template.scenario.mapspace.devices_for(task.operation_id)
                   for template in episode.templates for task in template.tasks):
                raise InvalidInput("prefetch_gpu requires every macro to explicitly allow GPU")
            baseline_placements = tuple({task.operation_id: "gpu" for task in template.tasks}
                                        for template in episode.templates)
        candidate = run_overlap_sequence(episode.templates, placements=baseline_placements, policy=policy,
                                         max_actions=max_baseline_actions, index=i)
        baselines[label] = candidate
    policy = SchedulePolicy(prefetch_depth=raw["prefetch_depth"], overlap=raw["overlap"], eviction_kind="lru",
                            expert_cache_capacity_bytes=episode.expert_cache_capacity_bytes)
    seed = min((candidate for label, candidate in baselines.items()
                if label.startswith("prefetch_") and candidate.is_valid),
               key=lambda candidate: candidate.total_makespan_ns, default=None)
    search = search_schedule_episode(episode.templates, policy=policy,
        placements=fixed if episode.placement_mode == "fixed_first_declared" else None,
        max_expanded_states=max_expanded_states, max_generated_states=max_generated_states, incumbent=seed)
    if episode_snapshot(path) != episode.source_inputs:
        raise InvalidInput("episode sources changed during search")
    folder.mkdir(parents=True, exist_ok=True)
    summary = dict(format_version="m2_moe_prefetch_overlap_search/1", episode_id=episode.id,
        source_inputs=episode.source_inputs, state_binding=STATE_BINDING, placement_mode=episode.placement_mode,
        prefetch_depth=raw["prefetch_depth"], overlap_enabled=raw["overlap"],
        expert_cache_capacity_bytes=episode.expert_cache_capacity_bytes,
        cache_pool="aligned_READY_and_RESERVED_COPY_non_source_expert_weight_copies",
        cache_statistics="per_strategy_documented_input_residency; not a hardware cache or SSD-read statistic",
        status=search.status, termination_reason=search.termination_reason,
        enumeration_complete=search.enumeration_complete, search_complete=search.complete,
        state_space_exhausted=search.termination_reason == "exhausted_declared_space",
        optimality_within_declared_space=search.optimality_proven,
        max_expanded_states=max_expanded_states, max_generated_states=max_generated_states,
        max_baseline_actions=max_baseline_actions, expanded_states=search.expanded_states,
        state_artifact_format=state_artifact_format,
        generated_states=search.generated_states, duplicate_states=search.duplicate_states,
        rejected_boundaries=search.rejected_boundaries,
        boundary_refusals=list(search.boundary_refusals), initial_refusal=search.initial_refusal,
        popped_states=search.popped_states, stale_pops=search.stale_pops, pruned_states=search.pruned_states,
        rejected_actions=search.rejected_actions, incumbent_seeded=search.incumbent_seeded,
        goal_states_generated=search.goal_states_generated,
        incumbent_seed_strategy="best_valid_prefetch_baseline_revalidated_against_search_space" if seed is not None else None,
        best_candidate_id=None if search.best is None else search.best.id,
        best_total_makespan_ns=None if search.best is None else search.best.total_makespan_ns,
        search_information_mode="offline_oracle", online_policy_claimed=False,
        cost_provenance="synthetic_layer_and_synthetic_transfer", hardware_measurement_performed=False,
        performance_calibrated=False, global_schedule_optimality_proven=False,
        search_scope=dict(macro_device_assignment=episode.placement_mode == "enumerate",
            compute_start_order="fixed_compiled_task_order", expert_copy_eviction_search=True,
            expert_eviction_trigger="demand_or_prefetch_window_capacity_pressure",
            prefetch_search=True, prefetch_window="current_unstarted_macro_plus_prefetch_depth_following_macros",
            overlap_search=True, overlap_resources=["cpu_compute", "gpu_compute", "pcie_copy", "ssd_io"],
            overlap_between_invocations=False, inactive_expert_loading=False, recomputation_search=False,
            activation_eviction_search=False, event_time_semantics="ADVANCE_to_next_engine_completion"),
        baselines=[schedule_candidate_document(candidate, label) for label, candidate in baselines.items()],
        best=None if search.best is None else schedule_candidate_document(search.best, "uniform_cost_search"),
        assumptions=["All costs are static declared synthetic values; measured context-specific v2 CPU costs are not reused.",
            "Reservations consume expert quota before transfer completion; whole-domain capacity includes all tensors and scratch.",
            "Only active-call graph inputs and required output delivery are moved; causal router barriers remain engine dependencies.",
            "Complete episode search crosses actual state_out boundaries; individual-call optima are not combined as a proof.",
            "Search is offline and bounded; a cutoff without a full episode result is unknown, not infeasible.",
            "Overlap validity and benefit are conditional on the declared independent resources and constant costs."])
    write_json(folder / "summary.json", summary)
    for label, candidate in baselines.items():
        write_candidate(candidate, folder / "baselines" / label, label, state_artifact_format)
    if search.best is not None:
        write_candidate(search.best, folder / "best", "uniform_cost_search", state_artifact_format)
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
    parser.add_argument("--state-artifact-format", choices=("json", "json-gzip"), default="json")
    args = parser.parse_args(argv)
    try:
        summary = execute_episode(args.episode, args.out, max_expanded_states=args.max_expanded_states,
                                  max_generated_states=args.max_generated_states,
                                  max_baseline_actions=args.max_baseline_actions,
                                  state_artifact_format=args.state_artifact_format)
    except (OSError, UnicodeError, json.JSONDecodeError, SpecError) as error:
        print(f"schedule episode refused: {error}", file=sys.stderr)
        return 1
    print(f"{summary['status']}: {summary['best_total_makespan_ns']} ns; "
          f"{summary['expanded_states']} expanded, {summary['generated_states']} generated")
    return 0 if summary["best_candidate_id"] is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
