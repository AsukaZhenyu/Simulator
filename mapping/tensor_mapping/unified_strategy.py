"""Source-bound causal execution, including continuous M2 and replay checkpoints."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys

from llm_infer_model.m2.routing_scores import ScoredCausalSession, parse_routing_scores
from llm_infer_model.m2.row_execution import parse_execution_spec
from llm_infer_model.tensor.runtime import dispatch_command
from llm_infer_model.tensor.environment import CODE_NO_PROGRESS
from llm_infer_model.tensor.services import parse_services
from llm_infer_model.tensor.spec import InvalidInput, SpecError, Unsupported

from .artifacts import write_json
from .causal import CausalRun, DecisionRecord, compile_command, observation_hash
from .causal_cli import parse_environment, verify_sources
from .policy_ir import parse_history, parse_policy
from .strategy_compiler import _read_object, _resolve_scenario
from .unified_policy import UnifiedPolicy, parse_transport


FIELDS = {"format_version", "id", "model_file", "execution_file", "policy_file",
          "services_file", "environment_file", "history_file", "routing_scores_file",
          "transport", "max_decisions"}


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def runtime_fingerprint():
    import llm_infer_model
    import tensor_mapping
    rows = []
    for label, package in (("modeling", llm_infer_model), ("mapping", tensor_mapping)):
        root = Path(package.__file__).resolve().parent
        rows.extend((f"{label}/{file.relative_to(root).as_posix()}", hashlib.sha256(file.read_bytes()).hexdigest())
                    for file in sorted(root.rglob("*.py")))
    return _hash(rows)


@dataclass(frozen=True)
class CompiledUnifiedStrategy:
    document: dict
    model: object
    global_model: bool
    execution: object
    policy: object
    services: object
    environment: tuple
    history: tuple
    routing_scores: tuple
    sources: tuple
    composition: object | None = None
    reclaim_requests: tuple = ()
    control_costs: object | None = None
    visibility_mode: str = "scope_barriers"
    frontend_sha256: str | None = None


def strategy_fingerprint(compiled):
    """Bind replay to both source bytes and the selected execution semantics."""
    return _hash({"sources": compiled.sources, "document": compiled.document,
                  "visibility_mode": compiled.visibility_mode, "frontend_sha256": compiled.frontend_sha256})


def compile_unified_strategy_file(path):
    path = Path(path).resolve()
    snapshots, documents = {}, {}
    def read(source):
        source = Path(source).resolve()
        if source not in snapshots:
            data = source.read_bytes()
            snapshots[source] = data
            documents[source] = _read_object(source, data)
        return documents[source]
    document = read(path)
    if document.get("format_version") in {"unified-strategy-0.2", "unified-strategy-0.3"}:
        from .composition_ir import compile_composition_file
        return compile_composition_file(path)
    if set(document) != FIELDS or document.get("format_version") != "unified-strategy-0.1":
        raise InvalidInput("unified strategy requires exactly the unified-strategy-0.1 fields")
    for name in ("id", "model_file", "policy_file", "environment_file"):
        if not isinstance(document[name], str) or not document[name]:
            raise InvalidInput(f"{name} must be explicit and nonempty")
    for name in ("execution_file", "services_file", "history_file", "routing_scores_file"):
        if document[name] is not None and (not isinstance(document[name], str) or not document[name]):
            raise InvalidInput(f"{name} must be explicit null or a nonempty path")
    if type(document["max_decisions"]) is not int or document["max_decisions"] < 1:
        raise InvalidInput("max_decisions must be positive")
    parse_transport(document["transport"])
    model_path = path.parent / document["model_file"]
    global_model = read(model_path).get("format_version") == "global-execution-0.1"
    if global_model:
        from llm_infer_model.m2.global_execution import compile_global_execution_file
        if document["execution_file"] is not None:
            raise InvalidInput("global model binds execution per invocation; top-level execution_file must be null")
        model = compile_global_execution_file(model_path)
        execution = None
        for source in model.sources:
            read(source["path"])
        initial_positions = {}
        for invocation in model.document["rounds"][0]["invocations"]:
            raw = read(model_path.parent / invocation["invocation_file"])
            initial_positions.update({(raw["layer"], row["request_id"]): row["position_start"] for row in raw["requests"]})
    else:
        _, model = _resolve_scenario(model_path, read)
        if model is None:
            raise Unsupported("unified causal strategy requires an M2 invocation or global model")
        execution = parse_execution_spec(read(path.parent / document["execution_file"])) if document["execution_file"] else None
        initial_positions = {(model.layer.id, row.request_id): row.position_start for row in model.invocation.requests}
    policy = parse_policy(read(path.parent / document["policy_file"]))
    services = parse_services(read(path.parent / document["services_file"])) if document["services_file"] else None
    environment = parse_environment(read(path.parent / document["environment_file"]))
    history = parse_history(read(path.parent / document["history_file"])) if document["history_file"] else ()
    scores = parse_routing_scores(read(path.parent / document["routing_scores_file"])) if document["routing_scores_file"] else ()
    sources = tuple({"path": str(file), "sha256": hashlib.sha256(data).hexdigest()}
                    for file, data in sorted(snapshots.items(), key=lambda row: str(row[0])))
    verify_sources(sources)
    compiled = CompiledUnifiedStrategy(document, model, global_model, execution, policy,
                                       services, environment, history, scores, sources)
    session = create_session(compiled)
    catalogue = {}
    for expert in session.observe().experts:
        catalogue.setdefault(expert.layer_id, set()).add(expert.expert_id)
    for row in history:
        key = (row.layer_id, row.request_id)
        if key not in initial_positions or row.absolute_position >= initial_positions[key]:
            raise InvalidInput("imported routing history must precede the first invocation of its layer/request")
        if not set(row.expert_ids) <= catalogue.get(row.layer_id, set()):
            raise InvalidInput("imported routing history references an unknown layer or expert")
        if row.router_scores_scope == "all_experts" and {expert for expert, _ in row.router_scores} != {
                expert for expert in catalogue.get(row.layer_id, set()) if isinstance(expert, int)}:
            raise InvalidInput("historical full router vector must cover the exact public routed-expert catalogue")
    if policy.cache_rule == "mrs" and not scores:
        raise Unsupported("MRS requires full current router-score evidence; top-k alone is insufficient")
    # Validate every private task's explicit kind/device binding without giving
    # any private task or future routing data to the policy callback.
    graph = session.execution_document()
    placements = dict(policy.placements)
    overrides = {request: dict(bindings) for request, bindings in policy.request_placements}
    for task in graph["tasks"]:
        kind, request, oid = task["kind"], task["request_id"], task["operation_id"]
        allowed = graph["scenario"]["mapspace"]["allowed_devices"][oid]
        if kind == "state_update":
            if len(allowed) != 1:
                raise InvalidInput("trace input gate must bind one explicit device")
            continue
        device = overrides.get(request, {}).get(kind, placements.get(kind))
        if device is None or device not in allowed:
            raise Unsupported(f"explicit placement for {request}/{kind} is absent or has no allowed exact cost")
        if policy.expert_device_rule in ("fiddler_paths", "hybrimoe_queues") and kind in {
                "expert_gate", "expert_up", "expert_swiglu", "expert_down"}:
            raise Unsupported("opaque paper expert selectors cannot reinterpret staged subweights")
    requests_and_kinds = {}
    for task in graph["tasks"]:
        requests_and_kinds.setdefault(task["request_id"], set()).add(task["kind"])
    if any(request not in requests_and_kinds or set(bindings) - requests_and_kinds[request]
           for request, bindings in overrides.items()):
        raise InvalidInput("request placement references an absent request or task kind")
    UnifiedPolicy(policy, transport=document["transport"], history=history)._estimate_observation(session.observe())
    return compiled


def create_session(compiled):
    if compiled.composition is not None:
        from .composition_observation import CompositionSession
        kind = compiled.document["model"]["kind"]
        if kind == "block":
            from .composition_block import create_session as create_block_session
            session = create_block_session(compiled.model, services=compiled.services,
                                           environment_events=compiled.environment,
                                           reclaim_requests=compiled.reclaim_requests, control_costs=compiled.control_costs)
        elif kind == "global_m2":
            from llm_infer_model.m2.global_execution import GlobalM2Session
            session = GlobalM2Session(compiled.model, routing_scores=compiled.routing_scores,
                                      environment_events=compiled.environment, services=compiled.services,
                                      reclaim_requests=compiled.reclaim_requests, control_costs=compiled.control_costs,
                                      visibility_mode=compiled.visibility_mode)
        else:
            session = ScoredCausalSession(compiled.model, execution_spec=compiled.execution,
                routing_scores=compiled.routing_scores, environment_events=compiled.environment, services=compiled.services,
                reclaim_requests=compiled.reclaim_requests, control_costs=compiled.control_costs)
        return CompositionSession(session, compiled.model, kind)
    if compiled.global_model:
        from llm_infer_model.m2.global_execution import GlobalM2Session
        return GlobalM2Session(compiled.model, routing_scores=compiled.routing_scores,
                              environment_events=compiled.environment, services=compiled.services,
                              visibility_mode=compiled.visibility_mode)
    return ScoredCausalSession(compiled.model, execution_spec=compiled.execution, routing_scores=compiled.routing_scores,
                               environment_events=compiled.environment, services=compiled.services)


def _runtime_signature(session):
    result = session.result()
    document = {"state": asdict(result.final_state), "environment_events": [asdict(row) for row in result.environment_events],
                "service_intervals": [asdict(row) for row in result.service_intervals],
                "nominal_ends": result.nominal_ends, "action_sources": result.action_sources}
    document["reclaim_records"] = [asdict(row) for row in getattr(result, "reclaim_records", ())]
    document["control_events"] = [asdict(row) for row in getattr(result, "control_events", ())]
    document["compute_progress"] = [asdict(row) for row in result.compute_progress]
    observation = session.observe()
    document["pending_reclaim"] = asdict(observation.pending_reclaim) if observation.pending_reclaim else None
    document["control_running"] = asdict(observation.control_running) if observation.control_running else None
    if session.observe().services is not None:
        document["services"] = session.service_document()
    return _hash(document)


def run_unified_strategy(compiled, *, checkpoint=None, max_decisions=None, policy_factory=None):
    """Run one strategy on the common runtime with a fresh policy instance.

    A factory may inspect the compiled strategy while constructing a trusted
    policy. The execution callback itself still receives only immutable public
    observations. Factories must return fresh deterministic policy state for
    each invocation so command replay has the same initial conditions.
    """
    verify_sources(compiled.sources)
    session = create_session(compiled)
    if policy_factory is not None:
        if not callable(policy_factory):
            raise InvalidInput("policy_factory must be callable")
        policy = policy_factory(compiled)
        if not callable(policy) or not callable(getattr(policy, "observe_final", None)):
            raise InvalidInput("policy_factory must return a callable policy with observe_final")
    elif compiled.composition is not None:
        from .composition_policy import CompositionPolicy
        policy = CompositionPolicy(compiled.composition, history=compiled.history)
    else:
        policy = UnifiedPolicy(compiled.policy, transport=compiled.document["transport"], history=compiled.history)
    records, replayed = [], 0
    if checkpoint is not None:
        fields = {"format_version", "source_sha256", "runtime_sha256", "commands", "runtime_signature",
                  "resume_semantics", "simulated_replay_overhead_ns"}
        if not isinstance(checkpoint, dict) or set(checkpoint) != fields or checkpoint["format_version"] != "unified-checkpoint-0.1":
            raise InvalidInput("checkpoint fields/version differ")
        if checkpoint["source_sha256"] != strategy_fingerprint(compiled) or checkpoint["runtime_sha256"] != runtime_fingerprint():
            raise InvalidInput("checkpoint sources or runtime changed; refusing stale replay")
        if (checkpoint["resume_semantics"] != "deterministic_command_replay_same_absolute_clock" or
                type(checkpoint["simulated_replay_overhead_ns"]) is not int or checkpoint["simulated_replay_overhead_ns"] != 0):
            raise InvalidInput("checkpoint must declare reconstruction with no additional simulated service")
        if not isinstance(checkpoint["commands"], list):
            raise InvalidInput("checkpoint commands must be a list")
        for index, expected in enumerate(checkpoint["commands"]):
            observation = session.observe()
            decision = policy(observation)
            actual = {"observation_sha256": observation_hash(observation), "decision": asdict(decision) if decision else None}
            if actual != expected:
                raise InvalidInput(f"checkpoint causal decision/observation differs at command {index}")
            command = compile_command(decision, observation)
            step = dispatch_command(session, command, source_index=index)
            if not step.ok:
                raise InvalidInput(f"checkpoint reconstructs a rejected command at {index}")
            records.append(DecisionRecord(index, actual["observation_sha256"], observation, decision, command.action, True))
        if _runtime_signature(session) != checkpoint["runtime_signature"]:
            raise InvalidInput("checkpoint inventory, in-flight services or environment differs after reconstruction")
        replayed = len(records)
    budget = compiled.document["max_decisions"] if max_decisions is None else max_decisions
    if type(budget) is not int or budget < 1:
        raise InvalidInput("additional decision budget must be positive")
    reason, compilation_error = "decision_budget_exhausted", None
    for index in range(len(records), len(records) + budget):
        observation = session.observe()
        if observation.complete or observation.error_code:
            reason = ("goal_reached" if observation.complete else
                      "policy_dead_end" if observation.error_code == CODE_NO_PROGRESS else "runtime_failure")
            break
        try:
            decision = policy(observation)
            if decision is None:
                reason = "policy_stopped"
                break
            command = compile_command(decision, observation)
        except (InvalidInput, Unsupported) as error:
            compilation_error, reason = str(error), "decision_rejected"
            break
        step = dispatch_command(session, command, source_index=index)
        records.append(DecisionRecord(index, observation_hash(observation), observation, decision, command.action,
                                      step.ok, step.error_code, step.error_message))
        if not step.ok:
            reason = "policy_dead_end" if step.error_code == CODE_NO_PROGRESS else "runtime_failure"
            break
    if session.observe().complete:
        reason = "goal_reached"
    policy.observe_final(session.observe())
    return CausalRun(session.result(), tuple(records), reason, compilation_error=compilation_error), session, policy, replayed


def checkpoint_document(compiled, result):
    run, session, _, _ = result
    if any(not row.accepted for row in run.decisions) or session.observe().error_code or run.compilation_error:
        raise Unsupported("failed execution cannot produce a resumable checkpoint")
    return {"format_version": "unified-checkpoint-0.1", "source_sha256": strategy_fingerprint(compiled),
            "runtime_sha256": runtime_fingerprint(), "runtime_signature": _runtime_signature(session),
            "commands": [{"observation_sha256": row.observation_sha256, "decision": asdict(row.decision)} for row in run.decisions],
            "resume_semantics": "deterministic_command_replay_same_absolute_clock", "simulated_replay_overhead_ns": 0}


def write_unified_result(output, compiled, result):
    verify_sources(compiled.sources)
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise InvalidInput("output must be new or empty")
    run, session, policy, replayed = result
    runtime, evaluation = run.runtime_result, run.evaluation
    output.mkdir(parents=True, exist_ok=True)
    reclaim_records = tuple(getattr(runtime, "reclaim_records", ()))
    all_requests_satisfied = all(row.status == "accepted" for row in reclaim_records)
    control_events = tuple(getattr(runtime, "control_events", ()))
    write_json(output / "summary.json", {"format_version": "unified-result-0.1", "id": compiled.document["id"],
        "status": evaluation.status, "reason": evaluation.reason, "termination_reason": run.termination_reason,
        "compilation_error": run.compilation_error, "makespan_ns": evaluation.makespan_ns,
        "decision_count": len(run.decisions), "checkpoint_replayed_commands": replayed,
        "peak_dram_bytes": evaluation.peak_dram_bytes, "peak_vram_bytes": evaluation.peak_vram_bytes,
        "h2d_bytes": evaluation.h2d_bytes, "d2h_bytes": evaluation.d2h_bytes,
        "ssd_read_bytes": evaluation.ssd_read_bytes, "ssd_write_bytes": evaluation.ssd_write_bytes,
        "error_code": evaluation.error_code, "continuous_multilayer_runtime": compiled.global_model or (
            compiled.composition is not None and compiled.document["model"]["kind"] == "block"),
        "composed_mechanisms": compiled.composition is not None,
        "visibility_mode": compiled.visibility_mode, "strategy_sha256": strategy_fingerprint(compiled),
        "one_clock_and_inventory": True, "information_mode": "causal_observation",
        "future_routes_in_policy_api": False, "future_environment_events_in_policy_api": False,
        "hardware_measurement": False, "physical_calibration_claimed": False, "policy_host_overhead_measured": False,
        "search_performed": False, "optimality_proven": False, "paper_strategy_reproduced": False,
        "pcie_timing_scope": "payload_only_no_launch_latency" if compiled.services else "exact_table_contract",
        "sources": list(compiled.sources), "runtime_sha256": runtime_fingerprint()})
    if compiled.document["format_version"] == "unified-strategy-0.3":
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        summary.update(reconfiguration_enabled=True, published_reclaim_request_count=len(reclaim_records),
            accepted_reclaim_request_count=sum(row.status == "accepted" for row in reclaim_records),
            unmet_reclaim_request_count=sum(row.status != "accepted" for row in reclaim_records),
            pending_reclaim_request_count=sum(row.status == "pending" for row in reclaim_records),
            all_published_reclaim_requests_satisfied=all_requests_satisfied,
            inference_goal_reached=run.termination_reason == "goal_reached",
            resource_compliant_success=evaluation.status == "valid" and run.termination_reason == "goal_reached" and all_requests_satisfied,
            control_cost_ns=sum(row.cost_ns if row.end_ns is not None else min(row.cost_ns, session.observe().t_ns - row.start_ns)
                                for row in control_events if row.kind != "WAIT_UNTIL"),
            control_committed_cost_ns=sum(row.cost_ns for row in control_events if row.kind != "WAIT_UNTIL"),
            control_cost_source=compiled.control_costs.source,
            control_timing_scope=compiled.control_costs.timing_scope)
        write_json(output / "summary.json", summary)
        write_json(output / "reconfiguration.json", {
            "visibility_scope": "published_requests_and_committed_control_only",
            "requests": [asdict(row) for row in reclaim_records],
            "controls": [asdict(row) for row in control_events],
            "all_published_requests_satisfied": all_requests_satisfied,
            "pending_request": asdict(session.observe().pending_reclaim) if session.observe().pending_reclaim else None,
            "control_running": asdict(session.observe().control_running) if session.observe().control_running else None})
    write_json(output / "strategy.json", compiled.document)
    write_json(output / "decisions.json", {"visibility_scope": "policy_observations_only", "decisions": [asdict(row) for row in run.decisions]})
    write_json(output / "execution.json", {"actions": [dict(**asdict(action), source_index=runtime.action_sources[index])
                                                          for index, action in enumerate(evaluation.actions)]})
    write_json(output / "events.json", {"events": [dict(**asdict(event), nominal_end_ns=runtime.nominal_ends[index])
                                                    for index, event in enumerate(evaluation.events)]})
    write_json(output / "states.json", {"visibility_scope": "modeling_private_debug_not_policy_input", "states": [asdict(row) for row in evaluation.states]})
    write_json(output / "environments.json", {"events": [asdict(row) for row in runtime.environment_events]})
    write_json(output / "service_intervals.json", {
        "intervals": [asdict(row) for row in runtime.service_intervals],
        "compute_progress": [asdict(row) for row in runtime.compute_progress],
        "compute_rate_semantics": "reference_cost_ns_per_wall_ns; exact_work; completion_ceil_to_integer_ns"})
    write_json(output / "persistent_state.json", session.persistent_state_document())
    write_json(output / "policy_audit.json", policy.audit_document())
    write_json(output / "history_out.json", policy.history_document())
    write_json(output / "compiled_graph.json", {"visibility_scope": "modeling_private_debug_not_policy_input", "graph": session.execution_document()})
    if compiled.services:
        write_json(output / "services.json", session.service_document())
    if not session.observe().error_code and not run.compilation_error:
        write_json(output / "checkpoint.json", checkpoint_document(compiled, result))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("strategy", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-decisions", type=int, help="additional decisions, excluding reconstruction")
    args = parser.parse_args(argv)
    try:
        if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
            raise InvalidInput("output must be new or empty")
        compiled = compile_unified_strategy_file(args.strategy)
        checkpoint = _read_object(args.resume, args.resume.read_bytes()) if args.resume else None
        result = run_unified_strategy(compiled, checkpoint=checkpoint, max_decisions=args.max_decisions)
        write_unified_result(args.out, compiled, result)
    except (SpecError, OSError, ValueError) as error:
        print(f"unified strategy: {error}", file=sys.stderr)
        return 2
    printed = {"status": result[0].evaluation.status, "makespan_ns": result[0].evaluation.makespan_ns,
               "reconstructed_commands": result[3], "output": str(args.out.resolve())}
    if compiled.document["format_version"] == "unified-strategy-0.3":
        printed["resource_compliant_success"] = result[0].evaluation.status == "valid" and all(
            row.status == "accepted" for row in result[0].runtime_result.reclaim_records)
    print(json.dumps(printed))
    if compiled.document["format_version"] == "unified-strategy-0.3" and result[0].evaluation.status == "valid" and any(
            row.status != "accepted" for row in result[0].runtime_result.reclaim_records):
        return 4
    return 0 if result[0].evaluation.status == "valid" else 3


if __name__ == "__main__":
    raise SystemExit(main())
