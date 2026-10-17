"""Run a causal MoE baseline against modeling's private environment trace.

This entry point does not search or reproduce a paper's complete algorithm.
The policy callback receives observations only; inputs and debug artifacts are
owned by the trusted modeling driver, not forwarded to that callback.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

from llm_infer_model.m2.causal import CausalM2Session
from llm_infer_model.m2.spec import to_state_document
from llm_infer_model.tensor.engine import STATUS_VALID
from llm_infer_model.tensor.environment import EnvironmentEvent
from llm_infer_model.tensor.spec import InvalidInput, SpecError, Unsupported

from .artifacts import action_to_dict, write_json
from .causal import SerialDemandPolicy, run_causal
from .strategy_compiler import _read_object, _resolve_scenario


def parse_environment(document):
    if not isinstance(document, dict) or set(document) != {"format_version", "events"}:
        raise InvalidInput("environment must contain exactly format_version and events")
    if document["format_version"] != "environment-trace-0.1":
        raise Unsupported("environment requires environment-trace-0.1")
    if not isinstance(document["events"], list):
        raise InvalidInput("environment.events must be an explicit array")
    events = []
    allowed = {"t_ns", "dram_external_bytes", "vram_external_bytes", "cpu_available"}
    for index, row in enumerate(document["events"]):
        if not isinstance(row, dict) or "t_ns" not in row or set(row) - allowed:
            raise InvalidInput(f"environment.events[{index}] has missing/unknown fields")
        if any(value is None for value in row.values()):
            raise InvalidInput(f"environment.events[{index}]: null is not an occupancy/availability declaration")
        events.append(EnvironmentEvent(**row))
    return tuple(events)


def load_session(invocation_path, environment_path):
    """Compile exactly the snapshotted bytes in the trusted driver."""
    invocation_path, environment_path = Path(invocation_path).resolve(), Path(environment_path).resolve()
    snapshots = {}
    documents = {}

    def read(path):
        path = Path(path).resolve()
        if path not in snapshots:
            data = path.read_bytes()
            documents[path] = _read_object(path, data)
            snapshots[path] = data
        return documents[path]

    _, binding = _resolve_scenario(invocation_path, read)
    if binding is None:
        raise Unsupported("causal CLI currently requires an M2 MoE invocation")
    events = parse_environment(read(environment_path))
    session = CausalM2Session(binding, environment_events=events)
    sources = tuple({"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}
                    for path, data in sorted(snapshots.items(), key=lambda item: str(item[0])))
    verify_sources(sources)
    return session, sources


def verify_sources(sources):
    for source in sources:
        if hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise InvalidInput(f"input changed during compilation/replay: {source['path']}")


def write_causal_run(output, run, session, sources, *, policy_name, max_decisions):
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise InvalidInput("output must be a new or empty directory")
    verify_sources(sources)
    evaluation, runtime = run.evaluation, run.runtime_result
    # Validate state export before creating any artifacts.
    state_document = to_state_document(session.state_out()) if evaluation.status == STATUS_VALID else None
    history = asdict(session.observe())["route_history"]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", dict(
        format_version="causal-result-0.1", information_mode="causal_observation",
        policy=policy_name, max_decisions=max_decisions, decision_count=len(run.decisions),
        termination_reason=run.termination_reason, compilation_error=run.compilation_error,
        status=evaluation.status, reason=evaluation.reason, makespan_ns=evaluation.makespan_ns,
        error_code=evaluation.error_code, error_action_index=evaluation.error_action_index,
        error_t_ns=evaluation.error_t_ns, peak_dram_bytes=evaluation.peak_dram_bytes,
        peak_vram_bytes=evaluation.peak_vram_bytes, h2d_bytes=evaluation.h2d_bytes,
        d2h_bytes=evaluation.d2h_bytes, ssd_read_bytes=evaluation.ssd_read_bytes,
        ssd_write_bytes=evaluation.ssd_write_bytes, environment_event_count=len(runtime.environment_events),
        search_performed=False, optimality_proven=False, global_infeasibility_proven=False,
        physical_calibration_claimed=False, paper_strategy_reproduced=False,
        cpu_contention_model="explicit_service_blackout_not_utilization_percentage",
        future_routes_in_policy_api=False, future_environment_events_in_policy_api=False,
        sources=list(sources),
    ))
    write_json(output / "decisions.json", dict(
        format_version="causal-decisions-0.1", visibility_scope="policy_observations_only",
        decisions=[asdict(record) for record in run.decisions],
    ))
    write_json(output / "execution.json", dict(
        format_version="causal-execution-0.1",
        actions=[dict(index=index, action=action_to_dict(action, m1=True),
                      decision_index=runtime.action_sources[index])
                 for index, action in enumerate(evaluation.actions)],
    ))
    write_json(output / "events.json", dict(
        format_version="causal-events-0.1", completion_semantics="actual_wall_span_end_null_while_unfinished",
        busy_time_source="service_intervals.json_excludes_cpu_blackouts",
        events=[dict(**asdict(event), nominal_end_ns=runtime.nominal_ends[index])
                for index, event in enumerate(evaluation.events)],
    ))
    write_json(output / "service_intervals.json", dict(
        format_version="causal-service-0.1", intervals=[asdict(interval) for interval in runtime.service_intervals],
    ))
    write_json(output / "environments.json", dict(
        format_version="causal-environments-0.1", visibility_scope="already_processed_environment_events",
        events=[asdict(event) for event in runtime.environment_events],
        rejected_allocation_semantics="recorded_attempt_not_committed_or_accepted_by_os",
    ))
    write_json(output / "states.json", dict(
        format_version="causal-states-0.1", visibility_scope="modeling_debug_complete_graph_not_policy_input",
        states=[asdict(state) for state in evaluation.states],
    ))
    write_json(output / "revealed_history.json", dict(
        format_version="revealed-route-history-0.1", entries=history,
        note="Only routes already revealed during this invocation; a stateful policy may retain this history across calls.",
    ))
    if state_document is not None:
        write_json(output / "state_out.json", state_document)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("invocation", type=Path)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--policy", choices=("serial_cpu", "serial_gpu"), required=True)
    parser.add_argument("--max-decisions", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
            raise InvalidInput("output must be a new or empty directory")
        session, sources = load_session(args.invocation, args.environment)
        policy = SerialDemandPolicy(args.policy.removeprefix("serial_"))
        run = run_causal(policy, session, max_decisions=args.max_decisions)
        write_causal_run(args.out, run, session, sources,
                         policy_name=args.policy, max_decisions=args.max_decisions)
    except (SpecError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"causal: {error}", file=sys.stderr)
        return 2
    print(json.dumps(dict(status=run.evaluation.status, makespan_ns=run.evaluation.makespan_ns,
                          termination_reason=run.termination_reason, output=str(args.out.resolve()))))
    return 0 if run.evaluation.status == STATUS_VALID else 3


if __name__ == "__main__":
    raise SystemExit(main())
