"""Explicit slot ownership and CE/SM program IR over the shared tensor ledger."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import sys

from llm_infer_model.tensor.engine import Action, ACTION_COMPUTE, ACTION_COPY, ACTION_EVICT, ACTION_READ_SSD, ACTION_WRITE_SSD
from llm_infer_model.tensor.program import WaitStep, _check_wait, _wait_done
from llm_infer_model.tensor.services import parse_services, ServiceSession
from llm_infer_model.tensor.runtime import ExecutionCommand, dispatch_command
from llm_infer_model.tensor.spec import InvalidInput, SpecError, Unsupported
from .artifacts import write_json
from .causal_cli import parse_environment, verify_sources
from .strategy_compiler import _read_object, _resolve_scenario

FIELDS = {"format_version", "id", "scenario_file", "services_file", "environment_file",
          "information_mode", "pcie_cost_source", "program"}


@dataclass(frozen=True)
class ServiceProgram:
    document: dict
    scenario: object
    services: object
    environment: tuple
    steps: tuple
    sources: tuple


def compile_service_program_file(path):
    path = Path(path).resolve(); snapshots, docs = {}, {}
    def read(source):
        source = Path(source).resolve()
        if source not in snapshots:
            data = source.read_bytes(); snapshots[source] = data; docs[source] = _read_object(source, data)
        return docs[source]
    doc = read(path)
    if set(doc) != FIELDS or doc.get("format_version") != "service-program-0.1":
        raise InvalidInput("service program requires exactly the service-program-0.1 fields")
    if doc["information_mode"] != "offline_fixed" or doc["pcie_cost_source"] != "services_payload_rates":
        raise Unsupported("service program requires explicit offline_fixed/services_payload_rates contracts")
    for name in ("id", "scenario_file", "services_file", "environment_file"):
        if not isinstance(doc[name], str) or not doc[name]: raise InvalidInput(f"{name} must be nonempty")
    scenario, binding = _resolve_scenario(path.parent / doc["scenario_file"], read)
    if binding is not None and binding.layer.ffn == "moe":
        raise Unsupported("offline service program is not an information-isolated MoE policy")
    services = parse_services(read(path.parent / doc["services_file"]))
    environment = parse_environment(read(path.parent / doc["environment_file"]))
    if not isinstance(doc["program"], list) or not doc["program"]:
        raise InvalidInput("service program must contain explicit steps")
    steps, computes = [], set()
    for index, raw in enumerate(doc["program"]):
        if not isinstance(raw, dict): raise InvalidInput(f"instruction {index} must be an object")
        kind = raw.get("kind")
        if kind == "compute":
            if set(raw) != {"kind", "operation_id", "device"}: raise InvalidInput("compute fields differ")
            oid = raw["operation_id"]
            if oid not in scenario.workload.operation_by_id or oid in computes:
                raise InvalidInput("compute references unknown or duplicate operation")
            if raw["device"] not in scenario.mapspace.devices_for(oid): raise Unsupported("compute device lacks exact allowed cost")
            computes.add(oid); steps.append((Action(ACTION_COMPUTE, oid, device=raw["device"]), None))
        elif kind == "transfer":
            if set(raw) != {"kind", "tensor_id", "src", "dst", "engine"}: raise InvalidInput("transfer requires explicit engine")
            if raw["engine"] not in ("ce", "sm") or {raw["src"], raw["dst"]} != {"dram", "vram"}:
                raise Unsupported("CE/SM transfer must name a dram/vram route")
            action = Action(ACTION_COPY, raw["tensor_id"], src=raw["src"], dst=raw["dst"])
            if raw["tensor_id"] not in scenario.workload.tensor_by_id or not scenario.mapspace.allows(action.route, raw["tensor_id"]):
                raise InvalidInput("transfer references an unknown tensor or forbidden route")
            steps.append((action, raw["engine"]))
        elif kind in ("read_ssd", "write_ssd"):
            if set(raw) != {"kind", "tensor_id"}: raise InvalidInput("SSD operation fields differ")
            action = Action(ACTION_READ_SSD if kind == "read_ssd" else ACTION_WRITE_SSD, raw["tensor_id"])
            if raw["tensor_id"] not in scenario.workload.tensor_by_id or not scenario.mapspace.allows(action.route, raw["tensor_id"]):
                raise InvalidInput("SSD instruction references an unknown tensor or forbidden route")
            steps.append((action, None))
        elif kind == "release":
            if set(raw) != {"kind", "tensor_id", "location"} or raw["location"] not in ("dram", "vram"):
                raise InvalidInput("release fields/location differ")
            if raw["tensor_id"] not in scenario.workload.tensor_by_id: raise InvalidInput("release names unknown tensor")
            steps.append((Action(ACTION_EVICT, raw["tensor_id"], location=raw["location"]), None))
        elif kind == "wait":
            if set(raw) != {"kind", "operations", "copies"} or not isinstance(raw["operations"], list) or not isinstance(raw["copies"], list):
                raise InvalidInput("wait must declare operations and copies")
            if not raw["operations"] and not raw["copies"]: raise InvalidInput("wait must have a condition")
            if len(set(raw["operations"])) != len(raw["operations"]): raise InvalidInput("duplicate wait operation")
            copies = []
            for item in raw["copies"]:
                if not isinstance(item, dict) or set(item) != {"tensor_id", "location"} or item["location"] not in ("dram", "vram", "ssd"):
                    raise InvalidInput("invalid wait copy")
                copies.append((item["tensor_id"], item["location"]))
            if len(set(copies)) != len(copies): raise InvalidInput("duplicate wait copy")
            if any(oid not in scenario.workload.operation_by_id for oid in raw["operations"]) or any(tid not in scenario.workload.tensor_by_id for tid, _ in copies):
                raise InvalidInput("wait names unknown object")
            steps.append((WaitStep(operations=tuple(raw["operations"]), copies=tuple(copies), source_index=index), None))
        else: raise Unsupported(f"unsupported service instruction {kind}")
    if computes != set(scenario.workload.operation_by_id): raise InvalidInput("program must compute every operation exactly once")
    sources = tuple({"path": str(p), "sha256": hashlib.sha256(data).hexdigest()} for p, data in sorted(snapshots.items(), key=lambda row: str(row[0])))
    verify_sources(sources)
    # Validate static allocation/rate constraints before running any action.
    ServiceSession(scenario, services, environment_events=environment)
    return ServiceProgram(doc, scenario, services, environment, tuple(steps), sources)


def run_service_program(compiled):
    verify_sources(compiled.sources)
    session = ServiceSession(compiled.scenario, compiled.services, environment_events=compiled.environment)
    for index, (step, engine) in enumerate(compiled.steps):
        if isinstance(step, WaitStep):
            if session.finished:
                session.submit(Action("ADVANCE"), source_index=index); break
            try:
                _check_wait(session.scenario, session.state, step)
            except Exception as error:
                # Only modeled wait failures are folded into runtime status.
                from llm_infer_model.tensor.engine import MappingError
                if not isinstance(error, MappingError): raise
                session._fail(error.code, str(error), action_index=len(session.result().evaluation.actions)); break
            while not _wait_done(session.scenario, session.state, step):
                result = session.advance(source_index=index)
                if not result.ok: break
        else:
            result = dispatch_command(session, ExecutionCommand(step, engine), source_index=index)
            if not result.ok: break
        if session.result().evaluation.status == "invalid_mapping": break
    return session


def write_service_result(output, compiled, session):
    verify_sources(compiled.sources); output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())): raise InvalidInput("output must be new or empty")
    result = session.result(); e = result.evaluation; output.mkdir(parents=True, exist_ok=True)
    write_json(output / "program.json", compiled.document)
    write_json(output / "services.json", session.service_document())
    write_json(output / "execution.json", {"actions": [dict(**asdict(a), source_index=result.action_sources[i]) for i, a in enumerate(e.actions)]})
    write_json(output / "events.json", {"events": [dict(**asdict(event), nominal_end_ns=result.nominal_ends[i]) for i, event in enumerate(e.events)]})
    write_json(output / "states.json", {"states": [asdict(s) for s in e.states]})
    write_json(output / "environments.json", {"events": [asdict(s) for s in result.environment_events]})
    write_json(output / "summary.json", {"status": e.status, "reason": e.reason, "makespan_ns": e.makespan_ns,
        "peak_dram_bytes": e.peak_dram_bytes, "peak_vram_bytes": e.peak_vram_bytes,
        "h2d_bytes": e.h2d_bytes, "d2h_bytes": e.d2h_bytes, "ssd_read_bytes": e.ssd_read_bytes, "ssd_write_bytes": e.ssd_write_bytes,
        "error_code": e.error_code, "error_action_index": e.error_action_index, "sources": list(compiled.sources),
        "paper_mechanism": "atsinfer_slots_and_engines", "paper_strategy_reproduced": False,
        "physical_calibration_claimed": False, "search_performed": False, "load_aware_dp_implemented": False,
        "pcie_timing_scope": "payload_only_no_launch_latency", "sm_gpu_rule": "exclusive_stream_no_partial_sm_occupancy"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", type=Path); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())): raise InvalidInput("output must be new or empty")
        compiled = compile_service_program_file(args.program); session = run_service_program(compiled)
        write_service_result(args.out, compiled, session)
    except (SpecError, OSError, ValueError) as error:
        print(f"service program: {error}", file=sys.stderr); return 2
    e = session.result().evaluation
    print(json.dumps({"status": e.status, "makespan_ns": e.makespan_ns}))
    return 0 if e.status == "valid" else 3


if __name__ == "__main__":
    raise SystemExit(main())
