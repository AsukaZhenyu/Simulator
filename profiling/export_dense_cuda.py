"""Export exact CUDA B1 Dense decoder evidence as non-executable observations.

Only exact, complete four-stage shape groups are exported. No interpolation,
history reuse, fixture replacement or hardware execution occurs.
The total CUDA workspace remains unknown. Every exported table is validation-only
and cannot support capacity-validated M2 execution. This export verifies the
declared mathematical and device boundary, not calibration of a model runtime.
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
from llm_infer_model.m2.dense_cuda_contract import (CONTRACT, build_plan, validate_case, signature, resident_accounting)
from llm_infer_model.m2.spec import load_layer
from llm_infer_model.m2.cuda_physical import validate_cuda_physical_layer as validate_model_layer
from llm_infer_model.tensor.spec import InvalidInput, Unsupported
from profile_contract import summarize_samples, validate_measurement
from run_dense_cuda_measurements import native_command, validate_native_binding, measurement_signature


class ExportError(ValueError):
    """Evidence cannot support an exact CUDA Dense observation entry."""


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


def model_signature(layer, kind, shape):
    return signature(layer.physical.config, kind, shape.phase, shape.tokens,
                     shape.visible_history, layer.physical.cuda_device, batch=shape.batch,
                     cuda_graphs=layer.physical.cuda_graphs)


def _resident_accounting(layer: Any, case: dict, native: dict) -> dict:
    expected = resident_accounting(layer.physical.config, case["kind"], case["tokens"], case["visible_history"])
    for key, value in expected.items():
        _same(native.get(key), value, "resident " + key)
    accounting = native["memory_accounting"]
    resident = sum(expected[key] for key in ("weight_bytes", "input_bytes", "output_bytes", "kv_cache_bytes"))
    _same(_int(accounting.get("resident_logical_bytes"), "resident_logical_bytes"), resident, "resident logical accounting")
    allocated = _int(accounting.get("resident_allocated_bytes"), "resident_allocated_bytes")
    if allocated < resident:
        raise ExportError("resident allocated bytes cannot be smaller than logical resident bytes")
    return {**expected, "resident_logical_bytes": resident, "resident_allocated_bytes": allocated,
            "resident_padding_bytes_not_charged_by_model": allocated-resident,
            "workspace_bytes": native["workspace_bytes"], "workspace_source": native["workspace_source"],
            "workspace_native_scope": accounting["workspace_scope"], "workspace_bound": accounting["workspace_bound"],
            "known_internal_allocated_buffer_bytes": accounting["internal_allocated_buffer_bytes"],
            "known_internal_logical_tensor_bytes_sum": accounting["internal_logical_tensor_bytes_sum"],
            "backend_scratch_bytes": None, "runtime_overhead_bytes": None,
            "capacity_verified": False,
            "model_memory_scope": "logical_resident_inventory_only_cuda_total_workspace_unknown"}


def composition_report(groups: dict, *, max_relative_error: float | None = None, allow_composition_error: bool = False) -> list[dict]:
    """Observe additive stage error against one joint graph with matching ports.

    A tolerance is a caller-supplied acceptance rule, never inferred calibration.
    """
    if max_relative_error is not None and (isinstance(max_relative_error, bool) or not isinstance(max_relative_error, (int, float)) or not math.isfinite(max_relative_error) or max_relative_error < 0):
        raise ExportError("max_relative_error must be an explicit nonnegative finite tolerance")
    if not isinstance(allow_composition_error, bool):
        raise ExportError("allow_composition_error must be boolean")
    result = []
    for shape, stages in sorted(groups.items()):
        if set(stages) != {"projection", "attention", "dense_ffn", "block"}:
            raise ExportError(f"{shape}: complete exact shape group requires projection, attention, dense_ffn and block; missing or extra stages")
        joint = stages["block"]
        for kind in ("projection", "attention", "dense_ffn"):
            stage = stages[kind]
            _same(stage["native_binary_sha256"], joint["native_binary_sha256"], "composition native executable")
            _same(stage["config"], joint["config"], "composition configuration")
            _same(stage["device_info"], joint["device_info"], "composition actual CUDA device")
            for port, digest in stage["stage_port_digests"].items():
                _same(joint["stage_port_digests"].get(port), digest, f"composition materialized port {port}")
        projection = stages["projection"]
        attention = stages["attention"]
        ffn = stages["dense_ffn"]
        for port, digest in projection["input_port_digests"].items():
            _same(joint["input_port_digests"].get(port), digest, f"composition projection input {port}")
        for port, digest in attention["input_port_digests"].items():
            upstream = joint["input_port_digests"] if port == "hidden" else projection["stage_port_digests"]
            _same(upstream.get(port), digest, f"composition attention input {port}")
        for port, digest in ffn["input_port_digests"].items():
            _same(attention["stage_port_digests"].get(port), digest, f"composition FFN input {port}")
        predicted = sum(math.ceil(stages[kind]["summary"]["median_ns"]) for kind in ("projection", "attention", "dense_ffn"))
        observed = joint["summary"]["median_ns"]
        signed = (predicted-observed) / observed
        within = abs(signed) <= max_relative_error if max_relative_error is not None else None
        if within is False and not allow_composition_error:
            raise ExportError(f"{shape}: composition relative error {abs(signed)} exceeds explicit tolerance {max_relative_error}")
        result.append({"shape": shape, "case_ids": {kind: stage["case_id"] for kind, stage in stages.items()},
            "predicted_serial_stage_sum_ns": predicted, "observed_joint_block_median_ns": observed,
            "signed_relative_error": signed, "absolute_relative_error": abs(signed),
            "explicit_max_relative_error": max_relative_error, "within_explicit_tolerance": within,
            "stage_cvs": {kind: stage["summary"]["cv"] for kind, stage in stages.items()},
            "materialized_ports_equal": True, "upstream_input_lineage_equal": True, "calibration_claimed": False,
            "comparison_scope": "same_declared_cuda_graph_variant_device_and_seed_with_timed_materialized_stage_copies"})
    return result


def export_cost_table(layer: Any, records: list[dict], *, raw_directory: str | Path,
                      manifest: dict, allow_unstable: bool = False,
                      source_evidence: dict | None = None,
                      max_relative_error: float | None = None,
                      allow_composition_error: bool = False) -> tuple[dict, dict]:
    """Return (cost table, evidence report); never write files or run hardware.

    raw_directory is the measurement run directory containing raw_output paths.
    Every supplied record must pass; selection is an explicit caller operation.
    The full stable physical signature binds history even for a history-independent
    FFN, so one observed shape never silently supplies an unmeasured shape.
    """
    if getattr(layer, "physical", None) is None:
        raise ExportError("export requires an explicit m2-layer-0.4 CUDA complete declared decoder descriptor")
    validate_model_layer(layer)
    if not isinstance(records, list) or not records:
        raise ExportError("records must be a nonempty list")
    if not isinstance(allow_unstable, bool):
        raise ExportError("allow_unstable must be boolean")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "dense_cuda_measurement_profile_v1" or manifest.get("contract") != CONTRACT:
        raise ExportError("manifest must declare the current macro measurement profile contract")
    protocol = manifest.get("protocol", {})
    for key, expected in {"backend": f"CUDA{layer.physical.cuda_device}", "batch": 1,
                          "cuda_device": layer.physical.cuda_device, "cuda_graphs": layer.physical.cuda_graphs,
                          "sequential_execution": True,
                          "graphs_per_sample": 1, "setup_compute_repetitions": 1}.items():
        _same(protocol.get(key), expected, f"manifest.protocol.{key}")
    if "threads" in protocol:
        raise ExportError("CUDA manifest cannot carry a CPU thread contract")
    cuda_device = _int(protocol.get("cuda_device"), "manifest.protocol.cuda_device")
    cuda_graphs = protocol.get("cuda_graphs")
    _same(manifest.get("software", {}).get("child_flags", {}).get("GGML_CUDA_DISABLE_GRAPHS"),
          "1", "manifest explicit disabled CUDA graph environment")
    args = SimpleNamespace(warmup=_int(protocol.get("warmup"), "warmup"),
                           samples=_int(protocol.get("samples"), "samples", 3),
                           seed=_int(protocol.get("seed"), "seed"),
                           cuda_device=cuda_device, cuda_graphs=cuda_graphs)
    if protocol.get("measured_graph_untimed_compute_repetitions") != args.warmup + 1:
        raise ExportError("manifest setup/warmup count is inconsistent")
    _same(protocol.get("upstream_setup_compute_repetitions_by_kind"), {kind: 0 if kind == "block" else 1 for kind in ("projection", "attention", "dense_ffn", "block")}, "manifest upstream preparation count")
    _same(protocol.get("total_untimed_compute_repetitions_by_kind"), {kind: args.warmup + (1 if kind == "block" else 2) for kind in ("projection", "attention", "dense_ffn", "block")}, "manifest total untimed computation count")
    plan = {case["id"]: case for case in build_plan(manifest.get("suite"), cuda_device, cuda_graphs)}
    runtime_hashes = manifest.get("software", {}).get("runtime_sha256", {})
    if not isinstance(runtime_hashes, dict):
        raise ExportError("manifest runtime hash evidence must be an object")
    directory = Path(raw_directory).resolve()
    entries, evidence, groups = [], [], {}
    seen = set()
    observed_device = None
    for row in records:
        if not isinstance(row, dict) or row.get("status") != "measured":
            raise ExportError("failed or incomplete records cannot be exported")
        case = validate_case(row.get("case"))
        if row.get("case_id") != case["id"] or case["id"] not in plan:
            raise ExportError("case identity is absent from its manifest suite")
        _same(case, plan[case["id"]], "measurement case and manifest plan")
        shape = MacroShape(case["phase"], case["batch"], case["tokens"], case["visible_history"])
        key = (case["kind"], "gpu", shape)
        if key in seen:
            raise ExportError("duplicate exact kind/device/phase/batch/tokens/history entry")
        seen.add(key)
        signature = row.get("signature")
        if not isinstance(signature, dict):
            raise ExportError("record lacks its physical signature")
        binary_hash = _digest(signature.get("native_binary_sha256"), "native_binary_sha256")
        _same(signature, measurement_signature(case, binary_hash), "regenerated measurement signature")
        selected = signature["physical_signature"]
        expected = model_signature(layer, case["kind"], shape)
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
        if observed_device is None:
            observed_device = copy.deepcopy(native_raw["device_info"])
        else:
            _same(native_raw["device_info"], observed_device, "observed CUDA device across records")
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
        if case["kind"] != "block":
            entries.append({"kind": case["kind"], "device": "gpu", "phase": shape.phase,
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
        groups.setdefault(shape.key(), {})[case["kind"]] = {"case_id": case["id"],
            "summary": summary, "stage_port_digests": native["stage_port_digests"],
            "input_port_digests": native["input_port_digests"],
            "native_binary_sha256": binary_hash, "config": case["config"],
            "device_info": copy.deepcopy(native["device_info"])}
    composition = composition_report(groups, max_relative_error=max_relative_error, allow_composition_error=allow_composition_error)
    entries.sort(key=lambda entry: (entry["kind"], entry["phase"], entry["batch"], entry["tokens"], entry["visible_history"]))
    # The backend scratch/runtime bytes are unobserved, even for stable samples.
    validation_only = True
    source = manifest.get("profile_id")
    if not isinstance(source, str) or not source:
        raise ExportError("manifest profile_id must be a nonempty string")
    records_hash = hashlib.sha256(_json(records).encode("utf-8")).hexdigest()
    manifest_hash = hashlib.sha256(_json(manifest).encode("utf-8")).hexdigest()
    costs = {"format_version": "m2-layer-0.4",
             "source": source + "-manifest-" + manifest_hash[:16] + "-records-" + records_hash[:16] + "-exact-cuda-dense-observations",
             "validation_only": True, "entries": entries,
             "comment": "Exact CUDA B1 declared Dense stage observations. validation_only=true; capacity_verified=false; execution_ready=false; calibration_claimed=false. Total workspace is unknown and cannot become zero; complete graph composition is reported without calibration."}
    # Observation tables can be parsed for inspection. Do not perform an
    # executable lookup: unknown workspace must reject capacity validation.
    parse_cost_table(costs)
    report = {"schema_version": "m2_dense_cuda_decoder_export_evidence_v1", "source_profile_id": source,
              "validation_only": validation_only, "allow_unstable": allow_unstable,
              "allow_composition_error": allow_composition_error, "max_relative_error": max_relative_error,
              "calibration_claimed": False, "capacity_verified": False, "execution_ready": False,
              "cost_table_parse_verified": True, "executable_lookup_performed": False,
              "complete_declared_block_variant": True, "hardware_measurement_performed": False,
              "boundary": "CUDA B1 complete declared pre-norm Dense block with timed materialized stage ports",
              "excluded_operations": ["allocation", "quantization", "io", "external_concurrency"],
              "limitations": ["deterministic generated inputs and weights, not pretrained contents",
                  "isolated resident CUDA timings with CUDA graphs disabled, not concurrent or cold-start timings",
                  "logical resident storage excludes allocator padding; CUDA private scratch/runtime overheads remain unknown",
                  "separate projections, explicit F32 KV and normal RoPE; original Qwen implementation is not reproduced"],
              "observed_cuda_device": observed_device,
              "source_evidence": copy.deepcopy(source_evidence or {}),
              "source_protocol": copy.deepcopy(protocol),
              "source_software": copy.deepcopy(manifest.get("software")),
              "source_environment": copy.deepcopy(manifest.get("environment")),
              "records_canonical_sha256": records_hash,
              "manifest_canonical_sha256": manifest_hash,
              "entries": evidence, "composition": composition}
    return costs, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", required=True, type=Path)
    parser.add_argument("--layer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-unstable", action="store_true", help="explicit validation-only export; never establishes calibration")
    parser.add_argument("--max-relative-error", type=float, help="explicit acceptance tolerance for stage-sum versus joint-block median; omitted means report only")
    parser.add_argument("--allow-composition-error", action="store_true", help="validation-only export despite an explicitly requested composition tolerance failure")
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
        max_relative_error=args.max_relative_error, allow_composition_error=args.allow_composition_error,
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
    print(f"Exported {len(costs['entries'])} exact CUDA Dense stage observations to {output}; validation_only={report['validation_only']}; calibration_claimed=false")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, InvalidInput, Unsupported) as error:
        print(f"export_dense_cuda: {error}", file=sys.stderr)
        raise SystemExit(1)
