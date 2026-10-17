"""Source-bound causal mechanisms from offloading papers; no search or calibration."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import sys

from llm_infer_model.m2.routing_scores import parse_routing_scores, ScoredCausalSession
from llm_infer_model.tensor.spec import InvalidInput, SpecError, Unsupported
from .artifacts import write_json
from .causal import run_causal
from .causal_cli import parse_environment, verify_sources, write_causal_run
from .causal_strategy import _validate_history
from .policy_ir import parse_policy, policy_document, parse_history
from .policy_executor import DeclarativePolicy
from .strategy_compiler import _read_object, _resolve_scenario

FIELDS = {"format_version", "id", "paper", "invocation_files", "environment_file", "execution_file",
          "policy_file", "history_file", "routing_scores_file", "max_decisions"}


@dataclass(frozen=True)
class CompiledPaperStrategy:
    document: dict
    binding: object
    execution_spec: object
    policy: object
    history: tuple
    routing_scores: tuple
    environment: tuple
    sources: tuple


def compile_paper_strategy_file(path):
    path = Path(path).resolve(); snapshots, documents = {}, {}
    def read(source):
        source = Path(source).resolve()
        if source not in snapshots:
            data = source.read_bytes(); snapshots[source] = data
            documents[source] = _read_object(source, data)
        return documents[source]
    doc = read(path)
    if set(doc) != FIELDS or doc.get("format_version") != "paper-strategy-0.1":
        raise InvalidInput("paper strategy requires exactly the paper-strategy-0.1 fields")
    if doc["paper"] not in ("fiddler", "hybrimoe", "neo"):
        raise Unsupported("causal paper strategy supports fiddler, hybrimoe, neo")
    if not isinstance(doc["id"], str) or not doc["id"] or type(doc["max_decisions"]) is not int or doc["max_decisions"] < 1:
        raise InvalidInput("paper strategy requires nonempty id and positive explicit decision budget")
    files = doc["invocation_files"]
    if not isinstance(files, list) or not files or any(not isinstance(p, str) or not p for p in files) or len(set(files)) != len(files):
        raise InvalidInput("invocation_files must be a nonempty array of unique paths")
    for field in ("environment_file", "execution_file", "policy_file"):
        if not isinstance(doc[field], str) or not doc[field]:
            raise InvalidInput(f"{field} must be a nonempty path")
    for field in ("history_file", "routing_scores_file"):
        if doc[field] is not None and (not isinstance(doc[field], str) or not doc[field]):
            raise InvalidInput(f"{field} must be explicit null or a nonempty path")
    bindings = tuple(_resolve_scenario(path.parent / item, read)[1] for item in files)
    if any(binding is None for binding in bindings):
        raise Unsupported("paper strategy inputs must be M2 invocations")
    execution_doc = read(path.parent / doc["execution_file"])
    if doc["paper"] == "neo":
        from llm_infer_model.m2.attention_execution import parse_attention_execution, compile_attention_subbatches
        execution = parse_attention_execution(execution_doc)
        execution_parent = (path.parent / doc["execution_file"]).resolve().parent
        if tuple((execution_parent / item).resolve() for item in execution.invocation_files) != tuple(
                (path.parent / item).resolve() for item in files):
            raise InvalidInput("execution invocation_files differ from the strategy's exact source paths")
        binding = compile_attention_subbatches(bindings, execution)
        session_execution = None  # already lowered, do not transform a second time
    else:
        from llm_infer_model.m2.row_execution import parse_execution_spec
        if len(bindings) != 1:
            raise Unsupported("routed paper mechanisms currently require exactly one invocation")
        binding = bindings[0]; execution = parse_execution_spec(execution_doc); session_execution = execution
    policy = parse_policy(read(path.parent / doc["policy_file"]))
    expected_rule = {"fiddler": "fiddler_paths", "hybrimoe": "hybrimoe_queues", "neo": "fixed"}[doc["paper"]]
    if policy.expert_device_rule != expected_rule:
        raise InvalidInput("paper mechanism label does not match the declared device rule")
    if doc["paper"] != "neo" and (execution.variant != "selected_rows" or execution.expert_variant != "opaque"):
        raise Unsupported("Fiddler/HybriMoE mechanism rules require opaque selected-row experts")
    history = parse_history(read(path.parent / doc["history_file"])) if doc["history_file"] else ()
    if binding.layer.ffn == "moe":
        _validate_history(history, binding)
        count = binding.layer.ffn_params["experts"]
        for row in history:
            if row.router_scores_scope == "all_experts" and tuple(i for i, _ in row.router_scores) != tuple(range(count)):
                raise InvalidInput("historical router vector must cover every catalogue expert")
    elif history or doc["routing_scores_file"]:
        raise Unsupported("Dense attention mechanism cannot receive MoE history or route scores")
    scores = parse_routing_scores(read(path.parent / doc["routing_scores_file"])) if doc["routing_scores_file"] else ()
    if policy.cache_rule == "mrs" and (not scores or any(r.router_scores_scope != "all_experts" for r in history)):
        raise Unsupported("MRS requires explicit full expert probabilities for current and imported historical routes")
    environment = parse_environment(read(path.parent / doc["environment_file"]))
    session = ScoredCausalSession(binding, execution_spec=session_execution, routing_scores=scores,
                                 environment_events=environment)
    graph = session.execution_document()
    declared = policy_document(policy)
    request_kinds = {}
    for task in graph["tasks"]:
        request_kinds.setdefault(task["request_id"], set()).add(task["kind"])
    for request, overrides in declared.get("request_placements", {}).items():
        if request not in request_kinds or set(overrides) - request_kinds[request]:
            raise InvalidInput("request_placements references an unknown request or absent task kind")
    for task in graph["tasks"]:
        kind, request, oid = task["kind"], task["request_id"], task["operation_id"]
        preferred = declared.get("request_placements", {}).get(request, {}).get(kind, declared["placements"].get(kind))
        if preferred is None:
            raise InvalidInput(f"missing policy placement for {request}/{kind}")
        if preferred not in graph["scenario"]["mapspace"]["allowed_devices"][oid]:
            raise Unsupported(f"placement {request}/{kind}={preferred} lacks allowed exact cost")
    if doc["paper"] == "neo":
        for subbatch in execution.subbatches:
            for request in subbatch.request_ids:
                if declared.get("request_placements", {}).get(request, {}).get("attention_core",
                        declared["placements"].get("attention_core")) != subbatch.attention_device:
                    raise InvalidInput("policy attention_core placement contradicts the explicit subbatch")
    sources = tuple({"path": str(p), "sha256": hashlib.sha256(data).hexdigest()}
                    for p, data in sorted(snapshots.items(), key=lambda row: str(row[0])))
    verify_sources(sources)
    return CompiledPaperStrategy(doc, binding, session_execution, policy, history, scores, environment, sources)


def run_paper_strategy(compiled):
    verify_sources(compiled.sources)
    session = ScoredCausalSession(compiled.binding, execution_spec=compiled.execution_spec,
                                 routing_scores=compiled.routing_scores, environment_events=compiled.environment)
    policy = DeclarativePolicy(compiled.policy, history=compiled.history)
    run = run_causal(policy, session, max_decisions=compiled.document["max_decisions"])
    policy.observe_final(session.observe())
    return run, session, policy


def write_paper_result(output, compiled, result):
    run, session, policy = result; verify_sources(compiled.sources)
    write_causal_run(output, run, session, compiled.sources,
                     policy_name=compiled.document["id"], max_decisions=compiled.document["max_decisions"])
    output = Path(output)
    write_json(output / "strategy.json", compiled.document)
    write_json(output / "policy.json", policy_document(compiled.policy))
    write_json(output / "policy_audit.json", policy.audit_document())
    write_json(output / "history_out.json", policy.history_document())
    write_json(output / "compiled_graph.json", dict(visibility_scope="private_debug_not_policy_input", graph=session.execution_document()))
    summary = json.loads((output / "summary.json").read_bytes())
    summary.update(paper_mechanism=compiled.document["paper"], paper_strategy_reproduced=False,
                   hardware_measurement=False, physical_calibration_claimed=False,
                   policy_host_overhead_measured=False, continuous_multilayer_runtime=False)
    write_json(output / "summary.json", summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("strategy", type=Path); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
            raise InvalidInput("output must be new or empty")
        compiled = compile_paper_strategy_file(args.strategy)
        result = run_paper_strategy(compiled); write_paper_result(args.out, compiled, result)
    except (SpecError, OSError, ValueError) as error:
        print(f"paper strategy: {error}", file=sys.stderr); return 2
    print(json.dumps({"status": result[0].evaluation.status, "makespan_ns": result[0].evaluation.makespan_ns,
                      "paper": compiled.document["paper"], "output": str(args.out.resolve())}))
    return 0 if result[0].evaluation.status == "valid" else 3


if __name__ == "__main__":
    raise SystemExit(main())
