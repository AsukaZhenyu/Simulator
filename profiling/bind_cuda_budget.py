"""Bind an explicit VRAM budget to an existing CUDA M2 invocation.

The declarations are policy inputs, not inferred from device-wide free memory.
This utility neither changes costs nor executes a model or hardware. In
particular, an external reservation does not make unknown CUDA workspace known.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class BudgetError(ValueError):
    """A declared budget or its source binding is invalid."""


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BudgetError(f"{label} must be an integer >= {minimum}")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BudgetError(f"{label} must be an explicit nonempty string")
    return value


def _object(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise BudgetError(f"{label} must be a JSON object")
    return value


def _hashes(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise BudgetError("source_hashes must be a nonempty path-to-SHA256 object")
    result = {}
    for key, digest in value.items():
        _text(key, "source_hashes path")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise BudgetError("source_hashes values must be lowercase SHA256")
        result[key] = digest
    return result


def _canonical_hash(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BudgetError("invocation must contain finite JSON values") from exc
    return hashlib.sha256(encoded).hexdigest()


def bind_budget(invocation: dict, *, observed_device: dict,
                capacity_bytes: int, runtime_reserved_bytes: int,
                alignment_bytes: int, reserve_source: str, device_uuid: str,
                source_hashes: dict[str, str]) -> tuple[dict, dict]:
    """Pure binding of caller declarations to a validated single-device profile.

    The file wrapper resolves relative references before calling this function.
    observed_device must come from a memory profile already checked to contain
    one exact CUDA device. Only the three VRAM hardware values are changed.
    No snapshot free/used value participates in the declaration.
    """
    _object(invocation, "invocation")
    if invocation.get("format_version") != "m2-layer-0.4":
        raise BudgetError("budget binding currently requires m2-layer-0.4")
    if invocation.get("devices") != ["gpu"]:
        raise BudgetError("budget binding requires exactly devices=['gpu']")
    hardware = _object(invocation.get("hardware"), "invocation.hardware")
    for key in ("vram_capacity_bytes", "vram_runtime_reserved_bytes", "vram_alignment_bytes"):
        if key not in hardware:
            raise BudgetError(f"existing invocation.hardware.{key} is required")
        _integer(hardware[key], f"existing invocation.hardware.{key}",
                 minimum=0 if key == "vram_runtime_reserved_bytes" else 1)
    device = _object(observed_device, "observed_device")
    _integer(device.get("index"), "observed_device.index")
    _text(device.get("name"), "observed_device.name")
    observed_uuid = _text(device.get("uuid"), "observed_device.uuid")
    total = _integer(device.get("total_memory_bytes"), "observed_device.total_memory_bytes", 1)
    declared_uuid = _text(device_uuid, "device_uuid")
    if declared_uuid != observed_uuid:
        raise BudgetError("declared device_uuid does not match the validated memory profile")
    capacity = _integer(capacity_bytes, "capacity_bytes", 1)
    reserved = _integer(runtime_reserved_bytes, "runtime_reserved_bytes")
    alignment = _integer(alignment_bytes, "alignment_bytes", 1)
    source = _text(reserve_source, "reserve_source")
    hashes = _hashes(source_hashes)
    if capacity > total:
        raise BudgetError("declared capacity exceeds the observed device total memory")
    if reserved >= capacity:
        raise BudgetError("runtime_reserved_bytes must be strictly smaller than capacity_bytes")
    if alignment > capacity:
        raise BudgetError("alignment_bytes must not exceed capacity_bytes")
    original = copy.deepcopy(invocation)
    bound = copy.deepcopy(invocation)
    bound["hardware"].update(vram_capacity_bytes=capacity,
        vram_runtime_reserved_bytes=reserved, vram_alignment_bytes=alignment)
    report = {
        "schema_version": "cuda_budget_binding_v1",
        "declared_budget": {
            "capacity_bytes": capacity,
            "runtime_reserved_bytes": reserved,
            "alignment_bytes": alignment,
            "available_for_modeled_allocations_bytes": capacity-reserved,
            "reserve_source": source,
            "source_kind": "explicit_caller_declaration",
            "reserve_scope": "static_external_runtime_reservation_excludes_macro_workspace",
            "device_uuid": declared_uuid,
        },
        "observed_device": copy.deepcopy(device),
        "source_hashes": dict(sorted(hashes.items())),
        "input_invocation_document_sha256": _canonical_hash(original),
        "bound_invocation_document_sha256": _canonical_hash(bound),
        "modified_hardware_fields": ["vram_capacity_bytes",
            "vram_runtime_reserved_bytes", "vram_alignment_bytes"],
        "reserve_inferred_from_device_free_memory": False,
        "model_capacity_under_declared_budget": None,
        "workspace_bound_known": False,
        "physical_capacity_verified": False,
        "execution_performed": False,
        "costs_modified": False,
        "scope": "explicit_budget_binding_only_unknown_workspace_still_rejected",
    }
    return bound, report


def _read(path: Path) -> tuple[dict, str]:
    try:
        data = path.read_bytes()
        document = json.loads(data)
    except (OSError, ValueError) as exc:
        raise BudgetError(f"cannot read JSON source {path}: {exc}") from exc
    return _object(document, str(path)), hashlib.sha256(data).hexdigest()


def prepare_invocation(path: str | Path, *, observed_device: dict,
                       capacity_bytes: int, runtime_reserved_bytes: int,
                       alignment_bytes: int, reserve_source: str,
                       device_uuid: str,
                       source_hashes: dict[str, str]) -> tuple[dict, dict]:
    """Read, bind, and preserve referenced files across an output-directory move."""
    source = Path(path).resolve()
    invocation, source_hash = _read(source)
    invocation = copy.deepcopy(invocation)
    hashes = dict(_hashes(source_hashes))
    if str(source) in hashes and hashes[str(source)] != source_hash:
        raise BudgetError("source invocation hash conflicts with supplied evidence")
    hashes[str(source)] = source_hash
    references = {}
    for field in ("layer_file", "costs_file"):
        if field not in invocation:
            continue
        relative = _text(invocation[field], f"invocation.{field}")
        resolved = (source.parent / relative).resolve()
        referenced, digest = _read(resolved)
        if str(resolved) in hashes and hashes[str(resolved)] != digest:
            raise BudgetError(f"{field} hash conflicts with supplied evidence")
        hashes[str(resolved)] = digest
        invocation[field] = str(resolved)
        references[field] = {
            "original_reference": relative,
            "resolved_path": str(resolved),
            "sha256": digest,
        }
        if field == "layer_file":
            layer = referenced.get("layer", referenced)
            physical = _object(_object(layer, "layer").get("physical"), "layer.physical")
            if _integer(physical.get("cuda_device"), "layer.physical.cuda_device") != observed_device.get("index"):
                raise BudgetError("layer.physical.cuda_device does not match the memory profile")
    bound, report = bind_budget(invocation, observed_device=observed_device,
        capacity_bytes=capacity_bytes, runtime_reserved_bytes=runtime_reserved_bytes,
        alignment_bytes=alignment_bytes, reserve_source=reserve_source,
        device_uuid=device_uuid, source_hashes=hashes)
    report["source_invocation_path"] = str(source)
    report["reference_bindings"] = references
    report["reference_resolution"] = "absolute_paths_from_original_invocation_directory"
    return bound, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--invocation", type=Path, required=True)
    parser.add_argument("--memory-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity-bytes", type=int, required=True)
    parser.add_argument("--runtime-reserved-bytes", type=int, required=True)
    parser.add_argument("--alignment-bytes", type=int, required=True)
    parser.add_argument("--reserve-source", required=True)
    parser.add_argument("--device-uuid", required=True)
    args = parser.parse_args(argv)
    # The reader is installed with the memory-evidence producer. It verifies the
    # saved profile rather than invoking CUDA or interpreting free bytes as policy.
    try:
        from cuda_memory_contract import load_memory_profile
        evidence = load_memory_profile(args.memory_profile)
        bound, report = prepare_invocation(args.invocation,
            observed_device=evidence["device_info"],
            capacity_bytes=args.capacity_bytes,
            runtime_reserved_bytes=args.runtime_reserved_bytes,
            alignment_bytes=args.alignment_bytes,
            reserve_source=args.reserve_source, device_uuid=args.device_uuid,
            source_hashes=evidence["source_hashes"])
        output = args.output.resolve()
        if output == args.invocation.resolve():
            raise BudgetError("budget output must not overwrite its source invocation")
        source_paths = {Path(p).resolve() for p in report["source_hashes"]}
        report_path = output.with_name(output.stem+".budget.json")
        profile_directory = args.memory_profile.resolve()
        if (output in source_paths or report_path in source_paths
                or output.is_relative_to(profile_directory)
                or report_path.is_relative_to(profile_directory)):
            raise BudgetError("budget outputs must not overwrite any source evidence or memory profile")
        if report_path == output:
            raise BudgetError("invocation and budget report paths must differ")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(bound, ensure_ascii=False, indent=2)+"\n",
                          encoding="utf-8", newline="\n")
        report["bound_invocation_path"] = str(output)
        report["bound_invocation_file_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n",
                               encoding="utf-8", newline="\n")
    except (BudgetError, ImportError, ValueError, OSError) as exc:
        parser.exit(2, f"budget binding failed: {exc}\n")
    print(f"bound invocation: {output}\nbudget evidence: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
