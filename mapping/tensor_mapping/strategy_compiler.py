"""Lower an explicit strategy IR to the modeling execution program.

There is no search or device choice here. Regions are labels over declared
nodes, not atomic execution barriers. Transfers through SSD are lowered through
DRAM with a visible wait; costs always come from the selected complete scenario.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from llm_infer_model.tensor.engine import (
    ACTION_COMPUTE, ACTION_COPY, ACTION_EVICT, ACTION_READ_SSD, ACTION_WRITE_SSD,
    Action,
)
from llm_infer_model.tensor.program import (
    ActionStep, DeviceStateStep, ProgramResult, WaitStep, execute_program,
    validate_device_state,
)
from llm_infer_model.tensor.spec import (
    M1Costs, InvalidInput, ROUTE_ENDS, Scenario, Unsupported, build_scenario,
    scenario_fingerprint, workload_fingerprint,
)

from .artifacts import action_to_dict
from .strategy_ir import StrategyIR, parse_strategy_ir, strategy_to_document

_ROUTE_FOR = {ends: name for name, ends in ROUTE_ENDS.items()}


@dataclass(frozen=True)
class CompiledStrategy:
    ir: StrategyIR
    scenario: Scenario
    steps: tuple[ActionStep | WaitStep | DeviceStateStep, ...]
    device_states: Mapping[str, Scenario]
    sources: tuple[dict[str, Any], ...] = ()
    m2_binding: Any = field(default=None, repr=False, compare=False)


def _check_scenario(scenario: Scenario) -> None:
    if not scenario.is_m1 or not isinstance(scenario.costs, M1Costs):
        raise Unsupported("strategy-ir-0.1 requires the heterogeneous tensor scenario 0.2")


def compile_strategy(
    ir: StrategyIR, scenario: Scenario, *, device_states: Mapping[str, Scenario] | None = None,
    sources: tuple[dict[str, Any], ...] = (), m2_binding: Any = None,
) -> CompiledStrategy:
    """Compile a fixed plan. Legality that depends on state belongs to modeling."""
    ir = parse_strategy_ir(strategy_to_document(ir))
    _check_scenario(scenario)
    validate_device_state(scenario, scenario)
    profiles = dict(device_states or {})
    if set(profiles) != set(ir.device_states):
        raise InvalidInput("device_states must resolve every and only the IR's named state")
    for candidate in profiles.values():
        _check_scenario(candidate)
        validate_device_state(scenario, candidate)

    operations = scenario.workload.operation_by_id
    tensors = scenario.workload.tensor_by_id
    if set(ir.placements) != set(operations):
        missing = sorted(set(operations) - set(ir.placements))
        extra = sorted(set(ir.placements) - set(operations))
        raise InvalidInput(f"placements must cover the graph exactly; missing={missing}, extra={extra}")
    for operation_id, device in ir.placements.items():
        if device not in scenario.mapspace.devices_for(operation_id):
            raise Unsupported(f"operation {operation_id!r} does not permit {device!r}")
        for profile in (scenario, *profiles.values()):
            if profile.costs.compute_cost(operation_id, device) is None:
                raise Unsupported(f"missing exact cost for {operation_id!r} on {device!r}")
    for region in ir.regions:
        unknown = set(region.members) - set(operations)
        if unknown:
            raise InvalidInput(f"region {region.id!r} references unknown operations {sorted(unknown)}")

    steps: list[ActionStep | WaitStep | DeviceStateStep] = []
    seen_compute: set[str] = set()

    def require_tensor(tensor_id: str) -> None:
        if tensor_id not in tensors:
            raise InvalidInput(f"unknown tensor {tensor_id!r}")

    def transfer(tensor_id: str, src: str, dst: str, source_index: int) -> None:
        require_tensor(tensor_id)
        route = _ROUTE_FOR[(src, dst)]
        if not scenario.mapspace.allows(route, tensor_id):
            raise Unsupported(f"transfer {tensor_id!r} on {route!r} is not permitted")
        for profile in (scenario, *profiles.values()):
            if profile.costs.transfer_cost(route, tensor_id) is None:
                raise Unsupported(f"missing exact transfer cost for {tensor_id!r} on {route!r}")
        if src == "ssd":
            action = Action(ACTION_READ_SSD, tensor_id)
        elif dst == "ssd":
            action = Action(ACTION_WRITE_SSD, tensor_id)
        else:
            action = Action(ACTION_COPY, tensor_id, src=src, dst=dst)
        steps.append(ActionStep(action, source_index))

    for index, instruction in enumerate(ir.program):
        kind = instruction.kind
        if kind == "compute":
            op_id = instruction.operation_id
            if op_id not in operations:
                raise InvalidInput(f"instruction {index}: unknown operation {op_id!r}")
            if op_id in seen_compute:
                raise InvalidInput(f"instruction {index}: duplicate compute for {op_id!r}")
            seen_compute.add(op_id)
            steps.append(ActionStep(Action(ACTION_COMPUTE, op_id, device=ir.placements[op_id]), index))
        elif kind == "transfer":
            src, dst = instruction.src, instruction.dst
            if {src, dst} == {"ssd", "vram"}:
                transfer(instruction.tensor_id, src, "dram", index)
                steps.append(WaitStep(copies=((instruction.tensor_id, "dram"),), source_index=index))
                transfer(instruction.tensor_id, "dram", dst, index)
            else:
                transfer(instruction.tensor_id, src, dst, index)
        elif kind == "release":
            require_tensor(instruction.tensor_id)
            if not scenario.mapspace.allow_eviction:
                raise Unsupported("the scenario does not permit release/eviction")
            steps.append(ActionStep(Action(ACTION_EVICT, instruction.tensor_id,
                                          location=instruction.location), index))
        elif kind == "wait":
            for op_id in instruction.operations:
                if op_id not in operations:
                    raise InvalidInput(f"instruction {index}: wait references unknown operation {op_id!r}")
            for copy in instruction.copies:
                require_tensor(copy.tensor_id)
            steps.append(WaitStep(operations=instruction.operations,
                                  copies=tuple((copy.tensor_id, copy.location) for copy in instruction.copies),
                                  source_index=index))
        elif kind == "drain":
            steps.append(WaitStep(drain=True, source_index=index))
        elif kind == "device_state":
            steps.append(DeviceStateStep(instruction.name, profiles[instruction.name], index))
        else:  # The parser rejects this; keep direct API mistakes loud too.
            raise Unsupported(f"unsupported instruction {kind!r}")
    missing_compute = set(operations) - seen_compute
    if missing_compute:
        raise InvalidInput(f"program omits compute instructions for {sorted(missing_compute)}")
    return CompiledStrategy(ir, scenario, tuple(steps), profiles, sources, m2_binding)


def _read_object(path: Path, data: bytes | None = None) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise InvalidInput(f"{path}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    def constant(value):
        raise InvalidInput(f"{path}: non-finite JSON constant {value}")

    value = json.loads((path.read_bytes() if data is None else data).decode("utf-8"),
                       object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(value, dict):
        raise InvalidInput(f"{path}: expected a JSON object")
    return value


def _resolve_scenario(path: Path, read: Callable[[Path], dict[str, Any]]):
    document = read(path)
    if "schema_version" in document:
        reference = document.get("workload_file")
        if not isinstance(reference, str) or not reference:
            raise InvalidInput(f"{path}: workload_file must be a nonempty string")
        scenario_document = dict(document)
        del scenario_document["workload_file"]
        workload = read((path.parent / reference).resolve())
        return build_scenario(workload, scenario_document, where=str(path)), None
    if "format_version" in document and "layer_file" in document:
        from llm_infer_model.m2.costs import parse_cost_table
        from llm_infer_model.m2.expand import compile_invocation
        from llm_infer_model.m2.spec import parse_invocation_document, parse_layer_document
        invocation = parse_invocation_document(document, where=str(path))
        layer_path = (path.parent / invocation.layer_file).resolve()
        layer = parse_layer_document(read(layer_path), where=str(layer_path))
        table_document = (invocation.raw_cost_table if invocation.raw_cost_table is not None
                          else read((path.parent / invocation.costs_file).resolve()))
        table = parse_cost_table(table_document, where=f"{path}:costs")
        binding = compile_invocation(layer, invocation, table, where=str(path))
        return binding.scenario, binding
    raise Unsupported(f"{path}: expected tensor scenario or M2 invocation")


def compile_strategy_file(path: str | Path) -> CompiledStrategy:
    path = Path(path).resolve()
    snapshots: dict[Path, bytes] = {}
    documents: dict[Path, dict[str, Any]] = {}

    def read(source: Path) -> dict[str, Any]:
        source = source.resolve()
        if source not in snapshots:
            # Hash the exact bytes used by every parser, not a later re-read.
            data = source.read_bytes()
            documents[source] = _read_object(source, data)
            snapshots[source] = data
        return documents[source]

    ir = parse_strategy_ir(read(path), where=str(path))
    base_path = (path.parent / ir.scenario_file).resolve()
    scenario, binding = _resolve_scenario(base_path, read)
    profiles = {}
    for name, reference in ir.device_states.items():
        profile_path = (path.parent / reference).resolve()
        profiles[name], _ = _resolve_scenario(profile_path, read)
    sources = tuple({"path": str(source), "sha256": hashlib.sha256(snapshots[source]).hexdigest()}
                    for source in sorted(snapshots, key=str))
    for source, snapshot in snapshots.items():
        if source.read_bytes() != snapshot:
            raise InvalidInput(f"input changed during compilation: {source}")
    return compile_strategy(ir, scenario, device_states=profiles, sources=sources, m2_binding=binding)


def run_strategy(compiled: CompiledStrategy) -> ProgramResult:
    return execute_program(compiled.scenario, compiled.steps)


def compiled_document(compiled: CompiledStrategy) -> dict[str, Any]:
    lowered = []
    for index, step in enumerate(compiled.steps):
        row: dict[str, Any] = {"index": index, "ir_instruction_index": step.source_index}
        if isinstance(step, ActionStep):
            row.update(kind="action", action=action_to_dict(step.action, m1=True))
        elif isinstance(step, WaitStep):
            row.update(kind="wait", operations=list(step.operations),
                       copies=[{"tensor_id": t, "location": loc} for t, loc in step.copies], drain=step.drain)
        else:
            row.update(kind="device_state", name=step.name, scenario_fingerprint=scenario_fingerprint(step.scenario))
        lowered.append(row)
    return {"format_version": "compiled-strategy-0.1", "strategy_id": compiled.ir.id,
            "information_mode": "offline_fixed", "online_policy_claimed": False,
            "scenario_id": compiled.scenario.id,
            "scenario_fingerprint": scenario_fingerprint(compiled.scenario),
            "workload_fingerprint": workload_fingerprint(compiled.scenario.workload),
            "regions": [{"id": r.id, "kind": r.kind, "members": list(r.members),
                         "execution_barrier": False} for r in compiled.ir.regions],
            "placements": dict(compiled.ir.placements), "program": lowered,
            "sources": list(compiled.sources),
            "device_states": {name: {"scenario_fingerprint": scenario_fingerprint(profile),
                                     "cost_source": profile.costs.source}
                              for name, profile in compiled.device_states.items()}}
