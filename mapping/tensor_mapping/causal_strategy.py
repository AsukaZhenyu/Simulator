"""Compile an explicit execution variant and causal policy, then evaluate it.

The compiler/driver owns input files and the private evaluation graph. The
policy executor receives only public observations; it never receives this
CompiledCausalStrategy object or an input filename.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from llm_infer_model.m2.causal import CausalM2Session
from llm_infer_model.tensor.spec import InvalidInput, SpecError, Unsupported

from .artifacts import write_json
from .causal import run_causal
from .causal_cli import parse_environment, verify_sources, write_causal_run
from .strategy_compiler import _read_object, _resolve_scenario


FORMAT_VERSION = "causal-strategy-0.1"
_FIELDS = {"format_version", "id", "invocation_file", "environment_file", "execution_file",
           "policy_file", "history_file", "max_decisions"}


@dataclass(frozen=True)
class CausalStrategyIR:
    format_version: str
    id: str
    invocation_file: str
    environment_file: str
    execution_file: str | None
    policy_file: str
    history_file: str | None
    max_decisions: int


def parse_causal_strategy(document: Any) -> CausalStrategyIR:
    if not isinstance(document, dict) or set(document) != _FIELDS:
        raise InvalidInput(f"causal strategy requires exactly {sorted(_FIELDS)}")
    if document["format_version"] != FORMAT_VERSION:
        raise Unsupported(f"causal strategy requires {FORMAT_VERSION}")
    for name in ("id", "invocation_file", "environment_file", "policy_file"):
        if not isinstance(document[name], str) or not document[name].strip():
            raise InvalidInput(f"causal strategy {name} must be a nonempty string")
    for name in ("execution_file", "history_file"):
        if document[name] is not None and (not isinstance(document[name], str) or not document[name].strip()):
            raise InvalidInput(f"causal strategy {name} must be a nonempty string or explicit null")
    budget = document["max_decisions"]
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise InvalidInput("causal strategy max_decisions must be an explicit positive integer")
    return CausalStrategyIR(**document)


@dataclass(frozen=True)
class CompiledCausalStrategy:
    ir: CausalStrategyIR
    binding: Any
    environment_events: tuple[Any, ...]
    execution_spec: Any
    execution_input: dict | None
    policy: Any
    history: tuple[Any, ...]
    sources: tuple[dict, ...]


@dataclass(frozen=True)
class CausalStrategyResult:
    run: Any
    session: CausalM2Session
    policy_executor: Any

    @property
    def evaluation(self):
        return self.run.evaluation


def _validate_history(history, binding):
    # A declared history file cannot inject the current invocation's route as
    # past evidence. Other requests remain explicit trusted historical input.
    starts = {request.request_id: request.position_start for request in binding.invocation.requests}
    expert_count = binding.layer.ffn_params["experts"]
    top_k = binding.layer.ffn_params["top_k"]
    for row in history:
        if row.layer_id != binding.layer.id:
            raise InvalidInput("history layer_id does not match the current layer")
        if len(row.expert_ids) != top_k or any(expert >= expert_count for expert in row.expert_ids):
            raise InvalidInput("historical experts do not match the static catalogue/top_k")
        if row.request_id in starts and row.absolute_position >= starts[row.request_id]:
            raise InvalidInput("history contains an unrevealed current/future position for this request")


def compile_causal_strategy_file(path: str | Path) -> CompiledCausalStrategy:
    from llm_infer_model.m2.row_execution import parse_execution_spec
    from .policy_ir import parse_history, parse_policy
    path = Path(path).resolve()
    snapshots, documents = {}, {}

    def read(source):
        source = Path(source).resolve()
        if source not in snapshots:
            data = source.read_bytes()
            documents[source] = _read_object(source, data)
            snapshots[source] = data
        return documents[source]

    ir = parse_causal_strategy(read(path))
    _, binding = _resolve_scenario((path.parent / ir.invocation_file).resolve(), read)
    if binding is None:
        raise Unsupported("causal strategy requires an M2 MoE invocation")
    if binding.layer.ffn != "moe" or binding.layer.format_version != "m2-layer-0.6":
        raise Unsupported("causal strategy currently requires synthetic MoE m2-layer-0.6")
    events = parse_environment(read(path.parent / ir.environment_file))
    execution_input = read(path.parent / ir.execution_file) if ir.execution_file is not None else None
    execution = parse_execution_spec(execution_input) if execution_input is not None else None
    policy = parse_policy(read(path.parent / ir.policy_file))
    history = parse_history(read(path.parent / ir.history_file)) if ir.history_file is not None else ()
    _validate_history(history, binding)
    # Compilation checks exact stage prices, graph semantics, and field choices
    # in a trusted driver. None of the private graph is passed to the policy.
    session = CausalM2Session(binding, execution_spec=execution, environment_events=events)
    from .policy_ir import policy_document
    declared_kinds = set(policy_document(policy)["placements"])
    required_kinds = {task["kind"] for task in session.execution_document()["tasks"]}
    if required_kinds - declared_kinds:
        raise InvalidInput(f"policy placements missing execution kinds {sorted(required_kinds - declared_kinds)}")
    graph = session.execution_document()
    preferred_devices = policy_document(policy)["placements"]
    for task in graph["tasks"]:
        operation_id, kind = task["operation_id"], task["kind"]
        device = preferred_devices[kind]
        devices = graph["scenario"]["mapspace"]["allowed_devices"][operation_id]
        prices = graph["scenario"]["costs"]["compute"][operation_id]
        if device not in devices or device not in prices:
            raise Unsupported(f"policy placement {kind}={device} has no allowed exact execution cost")
    sources = tuple({"path": str(source), "sha256": hashlib.sha256(data).hexdigest()}
                    for source, data in sorted(snapshots.items(), key=lambda item: str(item[0])))
    verify_sources(sources)
    return CompiledCausalStrategy(ir, binding, events, execution, execution_input, policy, history, sources)


def run_causal_strategy(compiled: CompiledCausalStrategy) -> CausalStrategyResult:
    from .policy_executor import DeclarativePolicy
    if not isinstance(compiled, CompiledCausalStrategy):
        raise InvalidInput("run_causal_strategy requires CompiledCausalStrategy")
    verify_sources(compiled.sources)
    session = CausalM2Session(compiled.binding, execution_spec=compiled.execution_spec,
                             environment_events=compiled.environment_events)
    executor = DeclarativePolicy(compiled.policy, history=compiled.history)
    run = run_causal(executor, session, max_decisions=compiled.ir.max_decisions)
    executor.observe_final(session.observe())
    return CausalStrategyResult(run, session, executor)


def write_strategy_result(output, compiled: CompiledCausalStrategy, result: CausalStrategyResult):
    from .policy_ir import policy_document
    output = Path(output)
    # Build complete documents and check provenance before creating artifacts.
    verify_sources(compiled.sources)
    audit = result.policy_executor.audit_document()
    history = result.policy_executor.history_document()
    graph = result.session.execution_document()
    policy = policy_document(compiled.policy)
    write_causal_run(output, result.run, result.session, compiled.sources,
                     policy_name=compiled.ir.id, max_decisions=compiled.ir.max_decisions)
    write_json(output / "strategy.json", asdict(compiled.ir))
    write_json(output / "policy.json", policy)
    write_json(output / "policy_audit.json", audit)
    write_json(output / "history_out.json", history)
    write_json(output / "compiled_graph.json", dict(
        format_version="compiled-causal-graph-0.1",
        visibility_scope="modeling_debug_complete_graph_not_policy_input",
        compiler_choices="explicit_execution_variant_and_policy_only",
        graph=graph,
    ))
    if compiled.execution_input is not None:
        write_json(output / "execution_spec.json", compiled.execution_input)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    summary.update(strategy_format_version=FORMAT_VERSION,
                   execution_variant="whole_request" if compiled.execution_spec is None else "selected_rows",
                   policy_format_version=policy["format_version"],
                   historical_input_count=len(compiled.history),
                   multi_invocation_continuous_runtime=False,
                   policy_prediction_runtime_overhead="not_profiled_host_control_logic")
    write_json(output / "summary.json", summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("strategy", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
            raise InvalidInput("output must be a new or empty directory")
        compiled = compile_causal_strategy_file(args.strategy)
        result = run_causal_strategy(compiled)
        write_strategy_result(args.out, compiled, result)
    except (SpecError, OSError, ValueError) as error:
        print(f"causal strategy: {error}", file=sys.stderr)
        return 2
    print(json.dumps(dict(status=result.evaluation.status, makespan_ns=result.evaluation.makespan_ns,
                          termination_reason=result.run.termination_reason, output=str(args.out.resolve()))))
    return 0 if result.evaluation.status == "valid" else 3


if __name__ == "__main__":
    raise SystemExit(main())
