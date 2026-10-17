"""Export exact CPU, batch-one Attention/FFN evidence into physical M2 costs.

No interpolation, history reuse, fixture replacement or hardware execution occurs.
A successful export certifies the declared partial macro boundary, not calibration
of a complete decoder layer. --allow-unstable exports are for validation only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

MODELING = Path(__file__).resolve().parents[1] / "modeling"
if str(MODELING) not in sys.path:
    sys.path.insert(0, str(MODELING))

from llm_infer_model.m2.costs import MacroShape, canonical_contract_hash, parse_cost_table
from llm_infer_model.m2.physical import select_measurement_signature, signature_for, validate_physical_layer
from llm_infer_model.m2.spec import load_layer
from llm_infer_model.tensor.spec import InvalidInput, Unsupported
from macro_contract import CONTRACT, build_macro_plan, macro_signature, validate_macro_case
from profile_contract import summarize_samples, validate_measurement
from run_macro_measurements import native_command, validate_native_binding


class ExportError(ValueError):
    """Evidence cannot support an exact physical M2 macro entry."""


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _same(actual: Any, expected: Any, label: str) -> None:
    if _json(actual) != _json(expected):
        raise ExportError(f"{label} does not match its evidence or exact contract")


def _int(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ExportError(f"{label} must be an integer >= {minimum}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ExportError(f"{label} must be a lowercase SHA256")
    return value


def _resident_accounting(layer: Any, case: dict, native: dict) -> dict:
    physical = layer.physical
    q = case["tokens"]
    history = case["visible_history"]
    kind = case["kind"]
    weights = sum(layer.weight_by_id(name).storage_bytes for name in physical.weight_groups[kind].values())
    output = layer.hidden_size * q * 4
    if kind == "attention":
        ports = {name: math.prod(shape) * 4 for name, shape in physical.input_ports(q).items()}
        kv_per_token = layer.kv_bytes_per_token
        kv_history, kv_new = history * kv_per_token, q * kv_per_token
        kv_cache = kv_history + kv_new
    else:
        ports = {"hidden": output}
        kv_history = kv_new = kv_cache = 0
    expected = {"weight_bytes": weights, "input_bytes": sum(ports.values()), "output_bytes": output,
                "kv_history_bytes": kv_history, "kv_new_bytes": kv_new, "kv_cache_bytes": kv_cache}
    for key, value in expected.items():
        _same(_int(native.get(key), key), value, key)
    accounting = native["memory_accounting"]
    resident = weights + sum(ports.values()) + output + kv_cache
    _same(_int(accounting.get("resident_logical_bytes"), "resident_logical_bytes"), resident, "resident logical accounting")
    allocated = _int(accounting.get("resident_allocated_bytes"), "resident_allocated_bytes")
    if allocated < resident:
        raise ExportError("resident allocated bytes cannot be smaller than logical resident bytes")
    _same(_int(accounting.get("internal_logical_tensor_bytes_sum"), "internal_logical_tensor_bytes_sum"),
          _int(native.get("internal_tensor_bytes_upper_bound"), "internal_tensor_bytes_upper_bound"),
          "internal tensor accounting")
    if accounting.get("kv_new_bytes_semantics") != "appended_slots_in_cache_distinct_from_new_kv_input_tensors":
        raise ExportError("KV append slots and supplied new-KV tensors must have distinct accounting")
    return {"input_ports_bytes": ports, **expected, "resident_logical_bytes": resident,
            "resident_allocated_bytes": allocated,
            "resident_padding_bytes_not_charged_by_model": allocated-resident,
            "workspace_bytes": native["workspace_bytes"], "workspace_source": native["workspace_source"],
            "workspace_native_scope": accounting["workspace_scope"],
            "workspace_bound": accounting["workspace_bound"],
            "model_memory_scope": "logical_independent_resident_allocations_plus_running_macro_internal_scratch"}


def export_cost_table(layer: Any, records: list[dict], *, raw_directory: str | Path,
                      manifest: dict, allow_unstable: bool = False,
                      source_evidence: dict | None = None) -> tuple[dict, dict]:
    """Return (cost table, evidence report); never write files or run hardware.

    raw_directory is the measurement run directory containing raw_output paths.
    Every supplied record must pass; selection is an explicit caller operation.
    The full stable physical signature binds history even for a history-independent
    FFN, so one observed shape never silently supplies an unmeasured shape.
    """
    if getattr(layer, "physical", None) is None:
        raise ExportError("export requires an explicit m2-layer-0.2 physical partial-block descriptor")
    validate_physical_layer(layer)
    if not isinstance(records, list) or not records:
        raise ExportError("records must be a nonempty list")
    if not isinstance(allow_unstable, bool):
        raise ExportError("allow_unstable must be boolean")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "macro_measurement_profile_v1" or manifest.get("contract") != CONTRACT:
        raise ExportError("manifest must declare the current macro measurement profile contract")
    protocol = manifest.get("protocol", {})
    for key, expected in {"backend": "CPU", "batch": 1, "sequential_execution": True,
                          "graphs_per_sample": 1, "setup_compute_repetitions": 1}.items():
        _same(protocol.get(key), expected, f"manifest.protocol.{key}")
    threads = _int(protocol.get("threads"), "manifest.protocol.threads", 1)
    args = SimpleNamespace(warmup=_int(protocol.get("warmup"), "warmup"),
                           samples=_int(protocol.get("samples"), "samples", 1),
                           seed=_int(protocol.get("seed"), "seed"))
    if protocol.get("total_untimed_compute_repetitions") != args.warmup + 1:
        raise ExportError("manifest setup/warmup count is inconsistent")
    plan = {case["id"]: case for case in build_macro_plan(manifest.get("suite"), threads)}
    runtime_hashes = manifest.get("software", {}).get("runtime_sha256", {})
    if not isinstance(runtime_hashes, dict):
        raise ExportError("manifest runtime hash evidence must be an object")
    directory = Path(raw_directory).resolve()
    entries, evidence = [], []
    seen = set()
    for row in records:
        if not isinstance(row, dict) or row.get("status") != "measured":
            raise ExportError("failed or incomplete records cannot be exported")
        case = validate_macro_case(row.get("case"))
        if row.get("case_id") != case["id"] or case["id"] not in plan:
            raise ExportError("case identity is absent from its manifest suite")
        _same(case, plan[case["id"]], "measurement case and manifest plan")
        shape = MacroShape(case["phase"], case["batch"], case["tokens"], case["visible_history"])
        key = (case["kind"], "cpu", shape)
        if key in seen:
            raise ExportError("duplicate exact kind/device/phase/batch/tokens/history entry")
        seen.add(key)
        signature = row.get("signature")
        if not isinstance(signature, dict):
            raise ExportError("record lacks its physical signature")
        binary_hash = _digest(signature.get("native_binary_sha256"), "native_binary_sha256")
        _same(signature, macro_signature(case, native_binary_sha256=binary_hash), "regenerated measurement signature")
        selected = select_measurement_signature(signature)
        expected = signature_for(layer, case["kind"], shape)
        _same(selected, expected, "model physical signature")
        raw_name = row.get("raw_output")
        if not isinstance(raw_name, str) or not raw_name or Path(raw_name).is_absolute():
            raise ExportError("raw_output must be a relative run-directory path")
        raw_path = (directory / raw_name).resolve()
        if not raw_path.is_relative_to(directory):
            raise ExportError("raw_output escapes the measurement run directory")
        raw = _load(raw_path)
        if not isinstance(raw, dict) or isinstance(raw.get("returncode"), bool) or raw.get("returncode") != 0 or "error" in raw or raw.get("timed_out"):
            raise ExportError("raw native execution did not succeed")
        _same(raw.get("case"), case, "raw case")
        command = raw.get("command")
        if not isinstance(command, list) or not command or not isinstance(command[0], str):
            raise ExportError("raw command is missing")
        _same(command, native_command(case, Path(command[0]).parent, args), "raw native command")
        if runtime_hashes.get(command[0]) != binary_hash:
            raise ExportError("native binary signature hash is not bound to its manifest runtime evidence")
        if not isinstance(raw.get("stdout"), str) or not isinstance(raw.get("stderr"), str):
            raise ExportError("raw stdout/stderr must be captured strings")
        native_raw = json.loads(raw["stdout"])
        native = row.get("native")
        if not isinstance(native, dict):
            raise ExportError("record native payload is missing")
        # These two fields are the only additions made by the Python runner.
        _same({k: v for k, v in native.items() if k not in {"expected_samples", "summary"}},
              native_raw, "raw stdout and recorded native payload")
        validate_native_binding(case, native_raw, args)
        if native.get("workspace_bytes") is None:
            raise ExportError("workspace is unknown; it cannot become zero in an M2 cost")
        _same(native.get("expected_samples"), args.samples, "runner expected samples")
        checked = validate_measurement(native)
        summary = summarize_samples(native["wall_ns"])
        _same(native.get("summary"), summary, "native summary and raw samples")
        _same(row.get("summary"), summary, "record summary and raw samples")
        _same(checked["summary"], summary, "validated sample summary")
        flags = row.get("quality_flags")
        if not isinstance(flags, list) or any(not isinstance(flag, str) for flag in flags) or len(set(flags)) != len(flags):
            raise ExportError("quality_flags must be an explicit unique string list")
        unstable = summary["cv"] > 0.1
        if set(flags) - {"high_observed_variability_cv_gt_0.1"}:
            raise ExportError("unsupported quality flags cannot be waived with --allow-unstable")
        if flags or unstable:
            if not allow_unstable:
                raise ExportError(f"{case['id']}: unstable or flagged measurement (cv={summary['cv']}); only explicit --allow-unstable validation export is permitted")
        memory = _resident_accounting(layer, case, native)
        duration = summary["median_ns"]
        # Native wall samples are integral nanoseconds. An even-count median can
        # be half a nanosecond; round upward once and record that conversion.
        duration_ns = math.ceil(duration)
        if duration_ns < 1:
            raise ExportError("median rounds below one nanosecond")
        entries.append({"kind": case["kind"], "device": "cpu", "phase": shape.phase,
            "batch": shape.batch, "tokens": shape.tokens, "visible_history": shape.visible_history,
            "duration_ns": duration_ns, "workspace_bytes": native["workspace_bytes"],
            "provenance": "measured_macro", "accounting": "per_request",
            "physical_signature": selected, "contract_hash": canonical_contract_hash(selected),
            "workspace_scope": "macro_internal_scratch", "workspace_source": native["workspace_source"]})
        evidence.append({"case_id": case["id"], "shape": shape.key(), "kind": case["kind"],
            "raw_output": raw_name, "raw_sha256": _sha(raw_path), "native_binary_sha256": binary_hash,
            "measurement_signature": copy.deepcopy(signature), "sample_summary": summary,
            "quality_flags": list(flags), "cv_gt_0_1": unstable,
            "duration_conversion": "ceil_observed_median_ns", "memory_boundary_check": memory,
            "legacy_eligibility": copy.deepcopy(row.get("eligibility")),
            "checks": {"signature_reconstructed": True, "model_signature_exact": True,
                "raw_native_equal": True, "native_binding_valid": True,
                "samples_summary_valid": True, "logical_resident_accounting_valid": True}})
    entries.sort(key=lambda entry: (entry["kind"], entry["phase"], entry["batch"], entry["tokens"], entry["visible_history"]))
    validation_only = allow_unstable
    source = manifest.get("profile_id")
    if not isinstance(source, str) or not source:
        raise ExportError("manifest profile_id must be a nonempty string")
    records_hash = hashlib.sha256(_json(records).encode("utf-8")).hexdigest()
    manifest_hash = hashlib.sha256(_json(manifest).encode("utf-8")).hexdigest()
    costs = {"format_version": "m2-layer-0.2",
             "source": source + "-manifest-" + manifest_hash[:16] + "-records-" + records_hash[:16] + "-exact-macro-export",
             "validation_only": validation_only, "entries": entries, "comment": "Exact CPU batch-one partial macro evidence; complete-layer calibration is not established. "
             + ("validation_only=true; unstable measurements explicitly allowed." if validation_only else "No instability waiver used.")}
    parsed = parse_cost_table(costs)
    for entry in entries:
        shape = MacroShape(entry["phase"], entry["batch"], entry["tokens"], entry["visible_history"])
        parsed.lookup(entry["kind"], "cpu", shape, physical_signature=entry["physical_signature"])
    report = {"schema_version": "m2_macro_export_evidence_v1", "source_profile_id": source,
              "validation_only": validation_only, "allow_unstable": allow_unstable,
              "calibration_claimed": False, "complete_layer": False, "hardware_measurement_performed": False,
              "boundary": "CPU batch-one post-positional materialized QKV -> attention+O -> SwiGLU FFN partial block",
              "excluded_operations": list(layer.physical.excluded_operations),
              "limitations": ["deterministic generated inputs and weights, not pretrained contents",
                  "isolated resident CPU timings, not concurrent or cold-start timings",
                  "logical resident storage excludes allocator padding and host/runtime overheads",
                  "no complete decoder layer or original Qwen implementation calibration"],
              "source_evidence": copy.deepcopy(source_evidence or {}),
              "source_protocol": copy.deepcopy(protocol),
              "source_software": copy.deepcopy(manifest.get("software")),
              "source_environment": copy.deepcopy(manifest.get("environment")),
              "records_canonical_sha256": records_hash,
              "manifest_canonical_sha256": manifest_hash,
              "entries": evidence}
    return costs, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", required=True, type=Path)
    parser.add_argument("--layer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-unstable", action="store_true", help="explicit validation-only export; never establishes calibration")
    parser.add_argument("--phase", choices=("prefill", "decode"), help="explicitly select one phase; omitted records are listed in the evidence report")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ExportError(f"output directory must be empty: {output}")
    measurements = args.measurements.resolve()
    layer_path = args.layer.resolve()
    manifest_path = measurements.parent / "manifest.json"
    layer = load_layer(layer_path)
    all_records = _load(measurements)
    if not isinstance(all_records, list):
        raise ExportError("measurements must contain a JSON array")
    if any(not isinstance(row, dict) or not isinstance(row.get("case"), dict)
           or row["case"].get("phase") not in ("prefill", "decode") for row in all_records):
        raise ExportError("each input record must declare its phase before explicit phase selection")
    records = all_records if args.phase is None else [row for row in all_records
        if isinstance(row, dict) and isinstance(row.get("case"), dict) and row["case"].get("phase") == args.phase]
    costs, report = export_cost_table(layer, records, raw_directory=measurements.parent,
        manifest=_load(manifest_path), allow_unstable=args.allow_unstable,
        source_evidence={"measurements": str(measurements), "measurements_sha256": _sha(measurements),
            "layer": str(layer_path), "layer_sha256": _sha(layer_path),
            "manifest": str(manifest_path), "manifest_sha256": _sha(manifest_path)})
    report["record_selection"] = {"requested_phase": args.phase, "input_count": len(all_records),
        "selected_case_ids": [row.get("case_id") for row in records],
        "omitted_case_ids": [row.get("case_id") if isinstance(row, dict) else None
            for row in all_records if row not in records], "silent_quality_filtering": False,
        "omitted_records": [{"case_id": row["case_id"], "phase": row["case"]["phase"],
            "status": row.get("status"), "quality_flags": row.get("quality_flags")}
            for row in all_records if row not in records]}
    output.mkdir(parents=True, exist_ok=True)
    for filename, document in (("costs.json", costs), ("export_report.json", report)):
        with (output / filename).open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    print(f"Exported {len(costs['entries'])} exact partial macro costs to {output}; validation_only={report['validation_only']}; calibration_claimed=false")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, InvalidInput, Unsupported) as error:
        print(f"export_m2_macros: {error}", file=sys.stderr)
        raise SystemExit(1)
