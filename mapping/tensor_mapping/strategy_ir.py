"""Strict, declarative input for compiling a fixed tensor execution plan.

This module describes intent; it does not load scenarios, choose placements,
insert transfers, or execute actions. Regions are flat descriptive groups and
introduce no execution barrier. References are preserved verbatim for the
compiler to resolve relative to the IR file.

The only information contract in version 0.1 is ``offline_fixed``. It neither
implements an online policy nor hides future information behind that label.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from llm_infer_model.tensor.spec import InvalidInput, Unsupported


FORMAT_VERSION = "strategy-ir-0.1"
INFORMATION_MODE = "offline_fixed"
DEVICES = frozenset({"cpu", "gpu"})
LOCATIONS = frozenset({"dram", "vram", "ssd"})
INSTRUCTION_KINDS = frozenset({"compute", "transfer", "release", "wait", "drain", "device_state"})
_TOP_FIELDS = frozenset({"format_version", "id", "scenario_file", "information_mode",
                         "regions", "placements", "program", "device_states"})
_INSTRUCTION_FIELDS = {
    "compute": frozenset({"kind", "operation_id"}),
    "transfer": frozenset({"kind", "tensor_id", "src", "dst"}),
    "release": frozenset({"kind", "tensor_id", "location"}),
    "wait": frozenset({"kind", "operations", "copies"}),
    "drain": frozenset({"kind"}),
    "device_state": frozenset({"kind", "name"}),
}


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidInput(f"{where} must be a nonempty string")
    return value


def _object(value: Any, fields: frozenset[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InvalidInput(f"{where} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise InvalidInput(f"{where} must have string keys")
    missing, extra = fields - value.keys(), value.keys() - fields
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing fields {sorted(missing)}")
        if extra:
            details.append(f"unknown fields {sorted(extra)}")
        raise InvalidInput(f"{where}: {'; '.join(details)}")
    return value


def _array(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise InvalidInput(f"{where} must be an array")
    return value


def _location(value: Any, where: str, *, releasable: bool = False) -> str:
    value = _text(value, where)
    allowed = frozenset({"dram", "vram"}) if releasable else LOCATIONS
    if value not in allowed:
        raise InvalidInput(f"{where} must be one of {sorted(allowed)}")
    return value


def _unique_texts(values: tuple[Any, ...], where: str, *, nonempty: bool = False) -> None:
    if not isinstance(values, tuple):
        raise InvalidInput(f"{where} must be a tuple in the typed IR")
    if nonempty and not values:
        raise InvalidInput(f"{where} must contain at least one member")
    for index, value in enumerate(values):
        _text(value, f"{where}[{index}]")
    if len(set(values)) != len(values):
        raise InvalidInput(f"{where} contains duplicate identifiers")


@dataclass(frozen=True)
class Region:
    id: str
    kind: str
    members: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.id, "region.id")
        _text(self.kind, "region.kind")
        _unique_texts(self.members, "region.members", nonempty=True)


@dataclass(frozen=True)
class CopyRef:
    tensor_id: str
    location: str

    def __post_init__(self) -> None:
        _text(self.tensor_id, "copy.tensor_id")
        _location(self.location, "copy.location")


@dataclass(frozen=True)
class Instruction:
    kind: str
    operation_id: str | None = None
    tensor_id: str | None = None
    src: str | None = None
    dst: str | None = None
    location: str | None = None
    operations: tuple[str, ...] = ()
    copies: tuple[CopyRef, ...] = ()
    name: str | None = None

    def __post_init__(self) -> None:
        _validate_instruction(self, "instruction")


def _validate_instruction(instruction: Instruction, where: str) -> None:
    kind = _text(instruction.kind, f"{where}.kind")
    if kind not in INSTRUCTION_KINDS:
        raise Unsupported(f"{where}.kind {kind!r} is not supported by {FORMAT_VERSION}")
    permitted = _INSTRUCTION_FIELDS[kind] - {"kind"}
    for field in ("operation_id", "tensor_id", "src", "dst", "location", "name"):
        value = getattr(instruction, field)
        if field not in permitted and value is not None:
            raise InvalidInput(f"{where}.{field} is not used by {kind}")
    if kind != "wait" and (instruction.operations != () or instruction.copies != ()):
        raise InvalidInput(f"{where}.operations/copies are only used by wait")
    if kind == "compute":
        _text(instruction.operation_id, f"{where}.operation_id")
    elif kind == "transfer":
        _text(instruction.tensor_id, f"{where}.tensor_id")
        _location(instruction.src, f"{where}.src")
        _location(instruction.dst, f"{where}.dst")
        if instruction.src == instruction.dst:
            raise InvalidInput(f"{where} transfer src and dst must differ")
    elif kind == "release":
        _text(instruction.tensor_id, f"{where}.tensor_id")
        _location(instruction.location, f"{where}.location", releasable=True)
    elif kind == "wait":
        _unique_texts(instruction.operations, f"{where}.operations")
        if not isinstance(instruction.copies, tuple):
            raise InvalidInput(f"{where}.copies must be a tuple in the typed IR")
        if any(not isinstance(copy, CopyRef) for copy in instruction.copies):
            raise InvalidInput(f"{where}.copies must contain CopyRef values")
        for copy in instruction.copies:
            _text(copy.tensor_id, f"{where}.copies.tensor_id")
            _location(copy.location, f"{where}.copies.location")
        if len(set(instruction.copies)) != len(instruction.copies):
            raise InvalidInput(f"{where}.copies contains duplicate copy references")
        if not instruction.operations and not instruction.copies:
            raise InvalidInput(f"{where} wait must declare at least one operation or copy")
    elif kind == "device_state":
        _text(instruction.name, f"{where}.name")


def _string_mapping(value: Any, where: str, *, devices: bool = False) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise InvalidInput(f"{where} must be an object")
    result = {}
    for key, item in value.items():
        _text(key, f"{where} key")
        _text(item, f"{where}.{key}")
        if devices and item not in DEVICES:
            raise InvalidInput(f"{where}.{key} must be cpu or gpu")
        result[key] = item
    return result


@dataclass(frozen=True)
class StrategyIR:
    format_version: str
    id: str
    scenario_file: str
    information_mode: str
    regions: tuple[Region, ...]
    placements: Mapping[str, str]
    program: tuple[Instruction, ...]
    device_states: Mapping[str, str]

    def __post_init__(self) -> None:
        _validate_strategy(self, "strategy")
        # Remove aliases to callers' mutable dictionaries. Serialization and
        # compiler entry validation also recheck the resulting typed value.
        object.__setattr__(self, "placements", dict(self.placements))
        object.__setattr__(self, "device_states", dict(self.device_states))


def _validate_strategy(ir: StrategyIR, where: str) -> None:
    if not isinstance(ir, StrategyIR):
        raise InvalidInput(f"{where} must be a StrategyIR")
    version = _text(ir.format_version, f"{where}.format_version")
    if version != FORMAT_VERSION:
        raise Unsupported(f"{where}.format_version {version!r} is not supported")
    _text(ir.id, f"{where}.id")
    _text(ir.scenario_file, f"{where}.scenario_file")
    mode = _text(ir.information_mode, f"{where}.information_mode")
    if mode != INFORMATION_MODE:
        raise Unsupported(f"{where}.information_mode {mode!r} is not supported; use offline_fixed")
    if not isinstance(ir.regions, tuple) or any(not isinstance(region, Region) for region in ir.regions):
        raise InvalidInput(f"{where}.regions must be a tuple of Region values")
    seen_ids, seen_members = set(), set()
    for index, region in enumerate(ir.regions):
        at = f"{where}.regions[{index}]"
        _text(region.id, f"{at}.id")
        _text(region.kind, f"{at}.kind")
        _unique_texts(region.members, f"{at}.members", nonempty=True)
        if region.id in seen_ids:
            raise InvalidInput(f"{at}.id duplicates region {region.id!r}")
        seen_ids.add(region.id)
        overlap = seen_members.intersection(region.members)
        if overlap:
            raise InvalidInput(f"{at}.members overlaps other regions: {sorted(overlap)}")
        seen_members.update(region.members)
    _string_mapping(ir.placements, f"{where}.placements", devices=True)
    states = _string_mapping(ir.device_states, f"{where}.device_states")
    if not isinstance(ir.program, tuple) or any(not isinstance(item, Instruction) for item in ir.program):
        raise InvalidInput(f"{where}.program must be a tuple of Instruction values")
    for index, instruction in enumerate(ir.program):
        at = f"{where}.program[{index}]"
        _validate_instruction(instruction, at)
        if instruction.kind == "device_state" and instruction.name not in states:
            raise InvalidInput(f"{at}.name references undeclared device state {instruction.name!r}")


def parse_strategy_ir(document: Any, where: str = "strategy") -> StrategyIR:
    """Validate explicit JSON structure; graph identifiers are compiler-checked."""
    raw = _object(document, _TOP_FIELDS, where)
    regions = []
    for index, item in enumerate(_array(raw["regions"], f"{where}.regions")):
        at = f"{where}.regions[{index}]"
        item = _object(item, frozenset({"id", "kind", "members"}), at)
        members = tuple(_array(item["members"], f"{at}.members"))
        regions.append(Region(item["id"], item["kind"], members))
    program = []
    for index, item in enumerate(_array(raw["program"], f"{where}.program")):
        at = f"{where}.program[{index}]"
        if not isinstance(item, dict):
            raise InvalidInput(f"{at} must be an object")
        kind = _text(item.get("kind"), f"{at}.kind")
        if kind not in INSTRUCTION_KINDS:
            raise Unsupported(f"{at}.kind {kind!r} is not supported by {FORMAT_VERSION}")
        item = _object(item, _INSTRUCTION_FIELDS[kind], at)
        values = {key: value for key, value in item.items() if key != "kind"}
        if kind == "wait":
            values["operations"] = tuple(_array(item["operations"], f"{at}.operations"))
            copies = []
            for copy_index, copy in enumerate(_array(item["copies"], f"{at}.copies")):
                copy_at = f"{at}.copies[{copy_index}]"
                copy = _object(copy, frozenset({"tensor_id", "location"}), copy_at)
                copies.append(CopyRef(copy["tensor_id"], copy["location"]))
            values["copies"] = tuple(copies)
        program.append(Instruction(kind=kind, **values))
    # Preserve the caller-provided location prefix for cross-field errors.
    ir = StrategyIR(raw["format_version"], raw["id"], raw["scenario_file"],
                    raw["information_mode"], tuple(regions),
                    _string_mapping(raw["placements"], f"{where}.placements", devices=True),
                    tuple(program), _string_mapping(raw["device_states"], f"{where}.device_states"))
    _validate_strategy(ir, where)
    return ir


def strategy_to_document(ir: StrategyIR) -> dict[str, Any]:
    """Return canonical explicit JSON fields, rejecting corrupted typed values."""
    _validate_strategy(ir, "strategy")
    program = []
    for instruction in ir.program:
        item = {"kind": instruction.kind}
        for field in sorted(_INSTRUCTION_FIELDS[instruction.kind] - {"kind"}):
            value = getattr(instruction, field)
            if field == "operations":
                value = list(value)
            elif field == "copies":
                value = [{"tensor_id": copy.tensor_id, "location": copy.location} for copy in value]
            item[field] = value
        program.append(item)
    return {"format_version": ir.format_version, "id": ir.id,
            "scenario_file": ir.scenario_file, "information_mode": ir.information_mode,
            "regions": [{"id": region.id, "kind": region.kind, "members": list(region.members)}
                        for region in ir.regions],
            "placements": dict(ir.placements), "program": program,
            "device_states": dict(ir.device_states)}


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidInput(f"strategy JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise InvalidInput(f"strategy JSON contains non-finite number {value}")


def load_strategy_ir(path: str | Path) -> StrategyIR:
    """Read strict UTF-8 JSON without loading any referenced scenario file."""
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"),
                              object_pairs_hook=_unique_json_object,
                              parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InvalidInput(f"{source}: cannot read strategy JSON: {error}") from error
    return parse_strategy_ir(document, where=str(source))
