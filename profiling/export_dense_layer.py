"""Export exact CPU B1 decoder stage evidence into physical M2 costs.

Only exact, complete four-stage shape groups are exported. No interpolation, history reuse, fixture replacement or hardware execution occurs.
A successful export certifies the declared decoder-block boundary -- dense or
sparse, as the layer document itself declares -- not calibration of an original
model implementation. --allow-unstable exports are for validation only.
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

from llm_infer_model.m2 import dense_contract
from llm_infer_model.m2.costs import MacroShape, canonical_contract_hash, parse_cost_table
from llm_infer_model.m2.full_physical import validate_full_physical_layer as validate_model_layer
from llm_infer_model.tensor.spec import InvalidInput, Unsupported
import contracts
from history_invariant_prices import ReplicateError, pool_history_invariant, require_replicates
from profile_contract import summarize_samples, validate_measurement
from run_dense_layer_measurements import native_command, validate_native_binding, measurement_signature


def _contract(layer: Any):
    """The contract module the layer document declares (dense or sparse)."""
    return contracts.for_layer(layer)


def _validate_layer(layer: Any, contract) -> None:
    """The layer's own complete-format check, in the format it declares.

    A sparse layer was already verified byte for byte by
    ``moe_contract.load_layer_document``; it has no mapping format to check.
    """
    if contract is dense_contract:
        validate_model_layer(layer)


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


def _cpu_mask(value: Any, threads: int, placement: Any) -> list[int] | None:
    """Recover the logical-CPU mask the recorded protocol passed to the probe.

    The raw command is rebuilt from the manifest, so the mask has to come from the
    same record. It stays optional: an unpinned campaign is still exportable, but
    its entries keep the ``os_scheduler_choice_of_logical_cpus`` placement and the
    protocol rule that only controlled placement reaches a main result.
    """
    if value is None:
        if placement is not None and placement != "os_scheduler_choice_of_logical_cpus":
            raise ExportError("manifest.protocol.thread_placement disagrees with an absent cpu_mask")
        return None
    if not isinstance(value, list) or any(isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 for cpu in value):
        raise ExportError("manifest.protocol.cpu_mask must be a list of nonnegative logical CPU ids")
    if len(value) != threads or len(set(value)) != len(value):
        raise ExportError("manifest.protocol.cpu_mask must name exactly one distinct logical CPU per thread")
    if value != sorted(value):
        raise ExportError("manifest.protocol.cpu_mask must be sorted")
    if any(cpu >= 64 for cpu in value):
        raise ExportError("manifest.protocol.cpu_mask exceeds the single processor group the probe supports")
    if placement is not None and placement != "explicit_strict_cpu_mask":
        raise ExportError("manifest.protocol.thread_placement disagrees with a recorded cpu_mask")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ExportError(f"{label} must be a lowercase SHA256")
    return value


def model_signature(layer, kind, shape, config=None, *, contract=None, price_policy=None):
    """The signature a layer document claims for this stage.

    Boundary stages multiply vocabulary-wide matrices the decoder-block document
    does not describe, so their signature comes from the measured case's own
    declared configuration; the prediction binds it to a model by looking the
    exact (kind, phase, tokens, history, vocab, word_dtype) key up.
    """
    contract = contract or _contract(layer)
    declared_cpu_ids = getattr(layer.physical, "cpu_ids", None)
    return contract.signature(config if config is not None else layer.physical.config, kind,
                              shape.phase, shape.tokens, shape.visible_history,
                              layer.physical.threads, batch=shape.batch,
                              residency=contract.signature_residency(
                                  getattr(layer.physical, "residency", None)),
                              process_scope=contract.signature_process_scope(
                                  getattr(layer.physical, "process_scope", None)),
                              cpu_ids=(list(declared_cpu_ids) if declared_cpu_ids else None),
                              price_policy=price_policy)


def collapse_history_invariant_prices(groups: dict, durations: dict, evidence: dict,
                                      policy: str) -> list[str]:
    """Replace each history-invariant decode price with one pooled median.

    ``groups`` maps a stage kind to its ``(entry, case id)`` members across
    visible history. The stage does the same work at every depth, so those
    entries are replicates of one physical task and the spread between them is
    measurement and machine state, not a depth effect. The entries keep their
    own keys and signatures; only the price becomes the pooled estimate, and
    the signature says so. Raw readings stay in the entry and in the evidence.
    """
    label = dense_contract.PRICE_POLICIES[policy]
    pooled_kinds = []
    for kind in sorted(groups):
        members = groups[kind]
        # One depth is not a set of replicates; there is nothing to pool and the
        # entry keeps the price it was measured at.
        if len({entry["visible_history"] for entry, _ in members}) < 2:
            continue
        try:
            require_replicates([entry for entry, _ in members], kind)
            pooled = pool_history_invariant(
                [(entry["visible_history"], case_id, entry["duration_ns"])
                 for entry, case_id in members])
        except ReplicateError as error:
            raise ExportError(f"{kind}: {error}") from None
        # The table itself gains no field: an entry is still a price under a
        # signature, and the signature now says which estimator produced it.
        # Each case's own reading stays in the evidence through sample_summary.
        for entry, case_id in members:
            entry["duration_ns"] = pooled["pooled_ns"]
            durations[case_id] = pooled["pooled_ns"]
            record = evidence[case_id]
            record["history_invariant_price_policy"] = label
            record["history_replicates"] = copy.deepcopy(pooled["replicates"])
            record["history_replicate_relative_spread"] = pooled["relative_spread"]
        pooled_kinds.append(kind)
    return pooled_kinds


def _stage_config(layer: Any, case: dict, contract) -> dict:
    return case["config"] if case["kind"] in contract.BOUNDARY_STAGES else layer.physical.config


def _resident_accounting(layer: Any, case: dict, native: dict, contract) -> dict:
    expected = contract.resident_accounting(_stage_config(layer, case, contract), case["kind"],
                                            case["tokens"], case["visible_history"])
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
            "model_memory_scope": "logical_independent_resident_allocations_plus_running_macro_internal_scratch"}


def composition_report(groups: dict, *, stages: tuple[str, ...] = dense_contract.STAGES,
                       max_relative_error: float | None = None, allow_composition_error: bool = False,
                       durations: dict[str, int] | None = None) -> list[dict]:
    """Observe additive stage error against one joint graph with matching ports.

    ``stages`` is the contract's own stage list, so a sparse block compares its
    three declared stages (projection, attention, moe_ffn) against its joint
    block exactly as a dense block does.  A tolerance is a caller-supplied
    acceptance rule, never inferred calibration.

    ``durations`` maps a case id to the exact nanosecond price the table will
    actually export for it.  Without it the joint-block comparison would be made
    against the raw per-case medians, which is the wrong question whenever the
    exported price was estimated from more than one case.
    """
    if max_relative_error is not None and (isinstance(max_relative_error, bool) or not isinstance(max_relative_error, (int, float)) or not math.isfinite(max_relative_error) or max_relative_error < 0):
        raise ExportError("max_relative_error must be an explicit nonnegative finite tolerance")
    if not isinstance(allow_composition_error, bool):
        raise ExportError("allow_composition_error must be boolean")
    result = []
    # The joint block names its third stage after the feed-forward it contains.
    ffn_stage = stages[2]
    decoder = set(stages)
    for shape, all_stages in sorted(groups.items()):
        # Boundary stages share a shape key with the block when both are measured
        # at the same token count; they are not part of the joint-block claim.
        measured = {kind: stage for kind, stage in all_stages.items() if kind in decoder}
        # A shape group that holds only boundary stages has no joint-block
        # composition claim to make; those groups are left out entirely.
        if not measured:
            continue
        if set(measured) != decoder:
            raise ExportError(f"{shape}: complete exact shape group requires "
                              f"{', '.join(sorted(decoder))}; missing or extra stages")
        joint = measured["block"]
        for kind in stages[:3]:
            stage = measured[kind]
            _same(stage["native_binary_sha256"], joint["native_binary_sha256"], "composition native executable")
            _same(stage["config"], joint["config"], "composition configuration")
            for port, digest in stage["stage_port_digests"].items():
                _same(joint["stage_port_digests"].get(port), digest, f"composition materialized port {port}")
        projection = measured["projection"]
        attention = measured["attention"]
        ffn = measured[ffn_stage]
        for port, digest in projection["input_port_digests"].items():
            _same(joint["input_port_digests"].get(port), digest, f"composition projection input {port}")
        for port, digest in attention["input_port_digests"].items():
            upstream = joint["input_port_digests"] if port == "hidden" else projection["stage_port_digests"]
            _same(upstream.get(port), digest, f"composition attention input {port}")
        for port, digest in ffn["input_port_digests"].items():
            _same(attention["stage_port_digests"].get(port), digest, f"composition FFN input {port}")
        def exported(stage):
            price = math.ceil(stage["summary"]["median_ns"])
            return price if durations is None else durations.get(stage["case_id"], price)

        predicted = sum(exported(measured[kind]) for kind in stages[:3])
        observed = joint["summary"]["median_ns"]
        signed = (predicted-observed) / observed
        within = abs(signed) <= max_relative_error if max_relative_error is not None else None
        if within is False and not allow_composition_error:
            raise ExportError(f"{shape}: composition relative error {abs(signed)} exceeds explicit tolerance {max_relative_error}")
        result.append({"shape": shape, "case_ids": {kind: stage["case_id"] for kind, stage in measured.items()},
            "predicted_serial_stage_sum_ns": predicted, "observed_joint_block_median_ns": observed,
            "signed_relative_error": signed, "absolute_relative_error": abs(signed),
            "explicit_max_relative_error": max_relative_error, "within_explicit_tolerance": within,
            "stage_cvs": {kind: stage["summary"]["cv"] for kind, stage in measured.items()},
            "materialized_ports_equal": True, "upstream_input_lineage_equal": True, "calibration_claimed": False,
            "stage_prices_from": ("exported_table_durations" if durations is not None
                                  else "raw_per_case_medians"),
            "comparison_scope": "same_declared_graph_variant_and_seed_with_timed_materialized_stage_copies"})
    return result


def export_cost_table(layer: Any, records: list[dict], *, raw_directory: str | Path,
                      manifest: dict, allow_unstable: bool = False,
                      source_evidence: dict | None = None,
                      max_relative_error: float | None = None,
                      allow_composition_error: bool = False,
                      price_policy: str | None = None) -> tuple[dict, dict]:
    """Return (cost table, evidence report); never write files or run hardware.

    raw_directory is the measurement run directory containing raw_output paths.
    Every supplied record must pass; selection is an explicit caller operation.
    The full stable physical signature binds history even for a history-independent
    FFN, so one observed shape never silently supplies an unmeasured shape.
    """
    if getattr(layer, "physical", None) is None:
        raise ExportError("export requires an explicit complete declared decoder descriptor")
    contract = _contract(layer)
    _validate_layer(layer, contract)
    if not isinstance(records, list) or not records:
        raise ExportError("records must be a nonempty list")
    if not isinstance(allow_unstable, bool):
        raise ExportError("allow_unstable must be boolean")
    # None is the historical own-shape estimator; anything else is a declared
    # estimator difference and enters the signature.
    policy = contract.signature_price_policy(price_policy)
    invariant_kinds = set(contract.history_invariant_kinds())
    if (not isinstance(manifest, dict)
            or manifest.get("schema_version") != contracts.PROFILE_SCHEMAS[contract.CONTRACT]
            or manifest.get("contract") != contract.CONTRACT):
        raise ExportError("manifest must declare the current macro measurement profile contract")
    protocol = manifest.get("protocol", {})
    for key, expected in {"backend": "CPU", "batch": 1, "sequential_execution": True,
                          "graphs_per_sample": 1, "setup_compute_repetitions": 1}.items():
        _same(protocol.get(key), expected, f"manifest.protocol.{key}")
    threads = _int(protocol.get("threads"), "manifest.protocol.threads", 1)
    args = SimpleNamespace(warmup=_int(protocol.get("warmup"), "warmup"),
                           samples=_int(protocol.get("samples"), "samples", 1),
                           seed=_int(protocol.get("seed"), "seed"))
    # Absent in manifests written before the regime was declarable, which is the
    # resident protocol: those tables keep exporting unchanged.
    args.evict_bytes = _int(protocol.get("evict_bytes", 0), "manifest.protocol.evict_bytes")
    declared_regime = ("streaming" if args.evict_bytes else "resident")
    if protocol.get("residency_regime") is not None:
        _same(protocol.get("residency_regime"), contract.RESIDENCY_REGIMES[declared_regime],
              "manifest.protocol.residency_regime")
    # Absent in manifests written before the scope was declarable, which is the
    # per-case protocol: those tables keep exporting unchanged.
    declared_scope = contract.declared_process_scope(protocol.get("process_scope"))
    args.cpu_ids = _cpu_mask(protocol.get("cpu_mask"), threads, protocol.get("thread_placement"))
    # A recorded mask says how the run was launched; it is an identity only when
    # the run itself declared it one. Manifests written before the dimension
    # existed record the mask (or nothing) without that declaration and keep
    # exporting against the historical signature, which is exactly what lets an
    # already published table be reproduced from its own evidence. A record
    # that disagrees either way is refused rather than reconciled.
    declares_mask = protocol.get("cpu_mask_in_physical_signature")
    if declares_mask is not None and declares_mask is not True:
        raise ExportError("manifest.protocol.cpu_mask_in_physical_signature must be true when present")
    if declares_mask is True and args.cpu_ids is None:
        raise ExportError("manifest declares the cpu mask as physical identity but records none")
    placement = args.cpu_ids if declares_mask is True else None
    if protocol.get("measured_graph_untimed_compute_repetitions") != args.warmup + 1:
        raise ExportError("manifest setup/warmup count is inconsistent")
    plan = {case["id"]: case for case in contract.build_plan(manifest.get("suite"), threads)}
    kinds = sorted({case["kind"] for case in plan.values()})
    prepared = set(contract.STAGES[:3])
    _same(protocol.get("upstream_setup_compute_repetitions_by_kind"), {kind: (1 if kind in prepared else 0) for kind in kinds}, "manifest upstream preparation count")
    _same(protocol.get("total_untimed_compute_repetitions_by_kind"), {kind: args.warmup + (2 if kind in prepared else 1) for kind in kinds}, "manifest total untimed computation count")
    runtime_hashes = manifest.get("software", {}).get("runtime_sha256", {})
    if not isinstance(runtime_hashes, dict):
        raise ExportError("manifest runtime hash evidence must be an object")
    directory = Path(raw_directory).resolve()
    entries, evidence, groups = [], [], {}
    exported_durations, invariant_groups = {}, {}
    seen = set()
    for row in records:
        if not isinstance(row, dict) or row.get("status") != "measured":
            raise ExportError("failed or incomplete records cannot be exported")
        case = contract.validate_case(row.get("case"))
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
        _same(signature, measurement_signature(case, binary_hash, contract,
                                               "streaming" if args.evict_bytes else None,
                                               declared_scope, placement),
              "regenerated measurement signature")
        selected = signature["physical_signature"]
        # The estimator is an export-side declaration, not a property of the raw
        # measurement, so the record's own signature is checked against the
        # unchanged contract and the entry then adds the policy it was priced by.
        base = model_signature(layer, case["kind"], shape,
                               _stage_config(layer, case, contract), contract=contract)
        _same(selected, base, "model physical signature")
        case_policy = (policy if policy is not None and shape.phase == "decode"
                       and case["kind"] in invariant_kinds else None)
        expected = (base if case_policy is None else
                    model_signature(layer, case["kind"], shape,
                                    _stage_config(layer, case, contract), contract=contract,
                                    price_policy=case_policy))
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
        _same(command, native_command(case, Path(command[0]).parent, args, contract), "raw native command")
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
        validate_native_binding(case, native_raw, args, contract)
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
        memory = _resident_accounting(layer, case, native, contract)
        duration = summary["median_ns"]
        # Native wall samples are integral nanoseconds. An even-count median can
        # be half a nanosecond; round upward once and record that conversion.
        duration_ns = math.ceil(duration)
        if duration_ns < 1:
            raise ExportError("median rounds below one nanosecond")
        if case["kind"] != "block":
            entry = {"kind": case["kind"], "device": "cpu", "phase": shape.phase,
                "batch": shape.batch, "tokens": shape.tokens, "visible_history": shape.visible_history,
                "duration_ns": duration_ns, "workspace_bytes": native["workspace_bytes"],
                "provenance": "measured_macro", "accounting": "per_request",
                "physical_signature": expected, "contract_hash": canonical_contract_hash(expected),
                "workspace_scope": "macro_internal_scratch", "workspace_source": native["workspace_source"]}
            entries.append(entry)
            exported_durations[case["id"]] = duration_ns
            if case_policy is not None:
                invariant_groups.setdefault(case["kind"], []).append((entry, case["id"]))
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
            "native_binary_sha256": binary_hash, "config": case["config"]}
    pooled_kinds = []
    if policy is not None:
        pooled_kinds = collapse_history_invariant_prices(
            invariant_groups, exported_durations,
            {record["case_id"]: record for record in evidence}, policy)
    composition = composition_report(groups, stages=contract.STAGES,
                                     max_relative_error=max_relative_error,
                                     allow_composition_error=allow_composition_error,
                                     durations=(exported_durations if policy is not None else None))
    entries.sort(key=lambda entry: (entry["kind"], entry["phase"], entry["batch"], entry["tokens"], entry["visible_history"]))
    validation_only = allow_unstable or allow_composition_error
    source = manifest.get("profile_id")
    if not isinstance(source, str) or not source:
        raise ExportError("manifest profile_id must be a nonempty string")
    ffn_stage = contract.STAGES[2]
    label = "MoE" if ffn_stage == "moe_ffn" else "Dense"
    records_hash = hashlib.sha256(_json(records).encode("utf-8")).hexdigest()
    manifest_hash = hashlib.sha256(_json(manifest).encode("utf-8")).hexdigest()
    costs = {"format_version": "m2-layer-0.3",
             "source": source + "-manifest-" + manifest_hash[:16] + "-records-" + records_hash[:16] + "-exact-" + label.lower() + "-export",
             "validation_only": validation_only, "entries": entries, f"comment": "Exact CPU B1 declared {label} stage evidence; joint composition error is reported and calibration is not claimed. "
             + ("validation_only=true; explicit measurement or composition acceptance waiver used." if validation_only else "No measurement or composition acceptance waiver used.")}
    parsed = parse_cost_table(costs)
    for entry in entries:
        shape = MacroShape(entry["phase"], entry["batch"], entry["tokens"], entry["visible_history"])
        parsed.lookup(entry["kind"], "cpu", shape, physical_signature=entry["physical_signature"])
    report = {"schema_version": "m2_dense_decoder_export_evidence_v1", "source_profile_id": source,
              "validation_only": validation_only, "allow_unstable": allow_unstable,
              "allow_composition_error": allow_composition_error, "max_relative_error": max_relative_error,
              "calibration_claimed": False, "complete_declared_block_variant": True, "hardware_measurement_performed": False,
              "history_invariant_price_policy": (None if policy is None else dense_contract.PRICE_POLICIES[policy]),
              "history_invariant_stages": sorted(pooled_kinds),
              # The exporter's contract, not the layer's format version, decides which
              # third stage the joint block should contain.
              "contract": contract.CONTRACT, "decoder_stages": list(contract.STAGES),
              "boundary": f"CPU B1 complete declared pre-norm {label} block with timed materialized stage ports",
              "excluded_operations": ["allocation", "quantization", "io", "external_concurrency"],
              "limitations": ["deterministic generated inputs and weights, not pretrained contents",
                  "isolated resident CPU timings, not concurrent or cold-start timings",
                  "logical resident storage excludes allocator padding and host/runtime overheads",
                  "separate projections, explicit F32 KV and normal RoPE; original Qwen implementation is not reproduced"],
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
    parser.add_argument("--history-invariant-price", choices=("own_shape", "median_of_history_replicates"),
        default="own_shape", help="how a decode price is estimated for a stage that reads no history: "
        "each shape keeps its own median (default, the historical tables) or one pooled median over the "
        "depth replicates, which declares itself in the signature")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ExportError(f"output directory must be empty: {output}")
    measurements = args.measurements.resolve()
    layer_path = args.layer.resolve()
    manifest_path = measurements.parent / "manifest.json"
    layer = contracts.load_layer(layer_path)
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
        price_policy=args.history_invariant_price,
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
    print(f"Exported {len(costs['entries'])} exact {report['contract']} stage costs to {output}; "
          f"validation_only={report['validation_only']}; calibration_claimed=false")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, InvalidInput, Unsupported) as error:
        print(f"export_dense_layer: {error}", file=sys.stderr)
        raise SystemExit(1)
