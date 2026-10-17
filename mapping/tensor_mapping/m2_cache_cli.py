"""Load an explicit single-layer MoE cache episode and export replayable plans.

Later calls use the candidate's actual preceding state_out, as explicitly
declared by the episode. Invocation files provide call/route/cost templates.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from llm_infer_model.m2.expand import CompiledInvocation, load_and_compile
from llm_infer_model.m2.spec import to_state_document, to_state_in_document
from llm_infer_model.tensor.engine import evaluate_mapping, load_mapping
from llm_infer_model.tensor.spec import InvalidInput, SpecError
from llm_infer_model.tensor.spec import build_scenario, scenario_fingerprint, workload_fingerprint
from .artifacts import action_to_dict, events_document, states_document, write_json
from .m2_moe import _source_snapshot_moe

EPISODE_FORMAT = "m2-moe-cache-episode-0.1"
STATE_BINDING = "declared_first_then_previous_state_out"
POLICIES = ("keep_all", "drop_all", "lru", "fifo", "offline_farthest")


@dataclass(frozen=True)
class CacheEpisode:
    id: str
    templates: tuple[CompiledInvocation, ...]
    expert_cache_capacity_bytes: dict[str, int | None]
    placement_mode: str
    baseline_policies: tuple[str, ...]
    source_inputs: dict[str, Any]


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _episode_raw(path):
    raw = _read(path)
    keys = {"format_version", "id", "invocation_files", "state_binding",
            "expert_cache_capacity_bytes", "placement_mode", "baseline_policies"}
    if not isinstance(raw, dict) or set(raw) != keys:
        raise InvalidInput(f"cache episode must explicitly contain {sorted(keys)}")
    if raw["format_version"] != EPISODE_FORMAT or raw["state_binding"] != STATE_BINDING:
        raise InvalidInput("unsupported cache episode format or state binding")
    if not isinstance(raw["id"], str) or not raw["id"].strip():
        raise InvalidInput("cache episode id must be nonempty")
    refs = raw["invocation_files"]
    if not isinstance(refs, list) or not refs or any(not isinstance(ref, str) or not ref.strip() for ref in refs):
        raise InvalidInput("invocation_files must explicitly list nonempty file references")
    capacity = raw["expert_cache_capacity_bytes"]
    if not isinstance(capacity, dict) or set(capacity) != {"dram", "vram"}:
        raise InvalidInput("expert cache capacity must explicitly contain dram and vram")
    for domain, value in capacity.items():
        if value is not None and (type(value) is not int or value < 0):
            raise InvalidInput(f"cache capacity {domain} must be a nonnegative integer or explicit null")
    if raw["placement_mode"] not in ("fixed_first_declared", "enumerate"):
        raise InvalidInput("placement_mode must be fixed_first_declared or enumerate")
    policies = raw["baseline_policies"]
    if not isinstance(policies, list) or not policies or any(policy not in POLICIES for policy in policies):
        raise InvalidInput("baseline_policies must explicitly select supported baseline names")
    if len(set(policies)) != len(policies):
        raise InvalidInput("duplicate baseline policy")
    return raw


def episode_snapshot(path):
    path = Path(path).resolve()
    raw = _episode_raw(path)
    payload = path.read_bytes()
    return dict(episode=dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload)),
        calls=[_source_snapshot_moe((path.parent / ref).resolve()) for ref in raw["invocation_files"]])


def load_cache_episode(path) -> CacheEpisode:
    path = Path(path).resolve()
    before = episode_snapshot(path)
    raw = _episode_raw(path)
    templates = tuple(load_and_compile((path.parent / ref).resolve()) for ref in raw["invocation_files"])
    if any(template.layer.format_version != "m2-layer-0.6" or template.layer.ffn != "moe" for template in templates):
        raise InvalidInput("cache episode requires single-layer m2-layer-0.6 MoE calls")
    source_layer_hashes = [call.get("layer", {}).get("sha256") for call in before["calls"]]
    if None in source_layer_hashes or len(set(source_layer_hashes)) != 1:
        raise InvalidInput("all calls must use the same immutable layer document SHA256")
    if any(template.layer != templates[0].layer for template in templates[1:]):
        raise InvalidInput("cache episode cannot mix layer definitions")
    if len({template.invocation.id for template in templates}) != len(templates):
        raise InvalidInput("cache episode invocation ids must be unique")
    after = episode_snapshot(path)
    if before != after:
        raise InvalidInput("cache episode sources changed during loading")
    return CacheEpisode(raw["id"], templates, dict(raw["expert_cache_capacity_bytes"]),
                        raw["placement_mode"], tuple(raw["baseline_policies"]), after)


def fixed_placements(episode):
    return tuple({task.operation_id: task.device for task in compiled.tasks}
                 for compiled in episode.templates)


def candidate_document(candidate):
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
            cache_hit_by_domain=None if item.plan is None else dict(item.plan.cache_hit_by_domain),
            cache_miss_by_domain=None if item.plan is None else dict(item.plan.cache_miss_by_domain),
            error_code=getattr(item.error, "code", None) if evaluation is None else evaluation.error_code))
        if evaluation is not None and evaluation.makespan_ns is not None:
            offset += evaluation.makespan_ns
    return dict(candidate_id=candidate.id, policy=candidate.policy.kind,
        status=candidate.status, reason=candidate.reason, total_makespan_ns=candidate.total_makespan_ns,
        completed_prefix_time_ns=offset, declared_invocation_count=candidate.declared_invocation_count,
        decision_path=list(candidate.decisions), cache_hits=candidate.cache_hits,
        cache_misses=candidate.cache_misses, eviction_count=candidate.eviction_count, calls=calls)


def _write_candidate(candidate, folder):
    folder.mkdir(parents=True)
    write_json(folder / "candidate.json", candidate_document(candidate))
    offset = 0
    timeline = []
    for index, item in enumerate(candidate.invocations):
        call_folder = folder / f"call-{index:03d}"
        call_folder.mkdir()
        compiled = item.compiled
        write_json(call_folder / "workload.json", compiled.workload_document)
        write_json(call_folder / "scenario.json", compiled.scenario_document)
        write_json(call_folder / "state_in.json", to_state_in_document(compiled.state_in, layer=compiled.layer.id))
        write_json(call_folder / "boundary.json", compiled.physical_boundary)
        if item.plan is None:
            write_json(call_folder / "refusal.json", dict(status=item.status, reason=item.reason,
                error_code=getattr(item.error, "code", None),
                error_action_index=getattr(item.error, "action_index", None),
                committed_prefix_actions=[action_to_dict(action, m1=True) for action in getattr(item.error, "actions", ())],
                prefix_is_complete_mapping=False))
            continue
        write_json(call_folder / "cache_plan.json", asdict(item.plan))
        mapping = dict(schema_version=compiled.scenario.schema_version, scenario_id=compiled.scenario.id,
            workload_fingerprint=workload_fingerprint(compiled.scenario.workload),
            scenario_fingerprint=scenario_fingerprint(compiled.scenario),
            actions=[action_to_dict(action, m1=True) for action in item.plan.actions])
        write_json(call_folder / "mapping.json", mapping)
        rebuilt = build_scenario(_read(call_folder / "workload.json"), _read(call_folder / "scenario.json"))
        loaded = load_mapping(call_folder / "mapping.json", rebuilt)
        replay = evaluate_mapping(rebuilt, loaded.actions)
        if not loaded.fingerprint_verified or replay != item.result.evaluation:
            raise InvalidInput("cache plan export did not reproduce its verified mapping")
        write_json(call_folder / "events.json", events_document(mode="moe_cache_sequence",
            scenario=compiled.scenario, evaluation=replay, note="serial demand cache plan"))
        write_json(call_folder / "states.json", states_document(mode="moe_cache_sequence",
            scenario=compiled.scenario, evaluation=replay, note="serial demand cache plan"))
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
    write_json(folder / "timeline.json", dict(schema_version="moe_cache_serial_timeline_v1", events=timeline,
        total_makespan_ns=candidate.total_makespan_ns, overlap_between_invocations=False))


def run_episode(episode, *, max_candidates, max_expanded_states):
    from .m2_cache import CachePolicy
    from .m2_cache_search import run_cache_sequence, search_cache_sequence
    fixed = fixed_placements(episode)
    baselines = tuple(run_cache_sequence(episode.templates, placements=fixed,
        policy=CachePolicy(kind=kind, expert_cache_capacity_bytes=episode.expert_cache_capacity_bytes), index=i)
        for i, kind in enumerate(episode.baseline_policies))
    search = search_cache_sequence(episode.templates,
        placements=fixed if episode.placement_mode == "fixed_first_declared" else None,
        policy=CachePolicy(kind="branch", expert_cache_capacity_bytes=episode.expert_cache_capacity_bytes),
        max_candidates=max_candidates, max_expanded_states=max_expanded_states)
    return baselines, search


def write_episode_artifacts(episode, baselines, search, output):
    folder = Path(output)
    if folder.exists() and any(folder.iterdir()):
        raise InvalidInput("cache episode output directory must be absent or empty")
    folder.mkdir(parents=True, exist_ok=True)
    summary = dict(format_version="m2_moe_cache_search/1", episode_id=episode.id,
        source_inputs=episode.source_inputs, state_binding=STATE_BINDING,
        placement_mode=episode.placement_mode, baseline_placement_mode="fixed_first_declared",
        expert_cache_capacity_bytes=episode.expert_cache_capacity_bytes,
        cache_pool="aligned_non_source_routed_and_shared_expert_weight_copies_in_each_domain",
        cache_statistics="expert_compute_input_copy_residency; includes pinned source hits; not SSD read count",
        status=search.status, enumeration_complete=search.enumeration_complete,
        termination_reason=search.termination_reason, max_candidates=search.max_candidates,
        max_expanded_states=search.max_expanded_states, expanded_states=search.expanded_states,
        declared_placement_count=search.declared_placement_count, placements_visited=search.placements_visited,
        candidate_count=len(search.candidates), valid_candidate_count=len(search.ranking),
        best_candidate_id=None if search.best is None else search.best.id,
        best_total_makespan_ns=None if search.best is None else search.best.total_makespan_ns,
        search_information_mode="offline_oracle", online_policy_claimed=False,
        cost_provenance="synthetic_layer_and_synthetic_transfer", hardware_measurement_performed=False,
        performance_calibrated=False, global_schedule_optimality_proven=False,
        optimality_within_declared_space=search.enumeration_complete and search.best is not None,
        search_scope=dict(macro_device_assignment=episode.placement_mode == "enumerate",
            expert_copy_eviction_search=True, eviction_trigger="cache_quota_or_engine_demand_capacity_pressure",
            compute_order="fixed_compiled_task_order", copy_order="serial_just_in_time",
            anticipatory_boundary_eviction_search=False, prefetch_search=False, overlap_search=False,
            inactive_expert_loading=False, recomputation_search=False),
        baselines=[candidate_document(candidate) for candidate in baselines],
        candidates=[candidate_document(candidate) for candidate in search.candidates],
        assumptions=["All plans use the original tensor engine's dependencies and full capacity ledger.",
            "Every later call receives the preceding candidate's actual READY KV and weight copies.",
            "Cache quota is additional to whole-domain capacity; immutable sources, KV and activations are not cache victims.",
            "An initial-state overflow is a refusal within this policy space; state_in is never edited to bypass it.",
            "A truncated search does not prove unvisited candidates infeasible or globally optimal.",
            "Offline farthest-next-use is a baseline heuristic for varying sizes/costs, not an optimality proof."])
    write_json(folder / "summary.json", summary)
    for candidate in baselines:
        _write_candidate(candidate, folder / "baselines" / candidate.policy.kind)
    (folder / "candidates").mkdir()
    for candidate in search.candidates:
        write_json(folder / "candidates" / (candidate.id + ".json"), candidate_document(candidate))
    if search.best is not None:
        _write_candidate(search.best, folder / "best")
    return summary


def execute_episode(path, output, *, max_candidates, max_expanded_states):
    episode = load_cache_episode(path)
    baselines, search = run_episode(episode, max_candidates=max_candidates, max_expanded_states=max_expanded_states)
    if episode_snapshot(path) != episode.source_inputs:
        raise InvalidInput("cache episode inputs changed during search")
    summary = write_episode_artifacts(episode, baselines, search, output)
    if episode_snapshot(path) != episode.source_inputs:
        raise InvalidInput("cache episode inputs changed during export")
    return summary


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, required=True)
    parser.add_argument("--max-expanded-states", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        summary = execute_episode(args.episode, args.out, max_candidates=args.max_candidates,
                                  max_expanded_states=args.max_expanded_states)
    except (OSError, UnicodeError, json.JSONDecodeError, SpecError) as error:
        print(f"cache episode refused: {error}", file=sys.stderr)
        return 1
    print(f"{summary['status']}: {summary['best_total_makespan_ns']} ns; "
          f"{summary['candidate_count']} episode candidates, {summary['expanded_states']} expanded decision prefixes")
    return 0 if summary["best_candidate_id"] is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
