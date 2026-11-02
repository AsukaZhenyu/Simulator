"""Training-only adapter for explicit prepared transfer response profiles.

The mathematical family and calibration-only LOO selector are unchanged.
Rows are independently measured inclusive host-completed means. Diagnostic
copy-call aliases, explicit phase timings and GPU event intervals are not fit
targets and are never added to these means.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from llm_infer_model.performance.prepared_transfer import (
    PARAMETER_FIELDS, PreparedTransferModel, validate_contract,
)
from profiling.dac_followup.transfer_fit import fit_from_training


def _receipt(receipt):
    if not isinstance(receipt, dict) or set(receipt) != {"path", "sha256"}:
        raise ValueError("explicit path/sha256 evidence receipt required")
    path = Path(receipt["path"])
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"]:
        raise ValueError("measurement/provenance source missing or changed: " + str(path))
    return dict(path=str(path.resolve()), sha256=receipt["sha256"])


def _provenance(value):
    if not isinstance(value, dict) or not {"hardware", "binary", "runtime", "sources"} <= set(value):
        raise ValueError("hardware, binary, runtime and source provenance required")
    if not isinstance(value["hardware"], dict) or not value["hardware"]:
        raise ValueError("explicit hardware observations, including unknowns, required")
    if not isinstance(value["runtime"], (list, tuple)) or not value["runtime"]:
        raise ValueError("runtime receipts required")
    if not isinstance(value["sources"], (list, tuple)) or not value["sources"]:
        raise ValueError("collector/source receipts required")
    copied = json.loads(json.dumps(value, allow_nan=False))
    copied["binary"] = _receipt(value["binary"])
    copied["runtime"] = [_receipt(row) for row in value["runtime"]]
    copied["sources"] = [_receipt(row) for row in value["sources"]]
    return copied


def fit_training(rows, *, contract, provenance, breakpoints=(16384, 262144, 1048576)):
    """Freeze one memory/API/preparation contract using calibration rows only.

    Each row needs split='calibration', bytes, mean_ns, raw={path,sha256},
    host_pointer, and either an explicit complete contract or the native top
    contract fields. Unknown pointer categories fail instead of being guessed.
    """
    contract = validate_contract(contract)
    provenance = _provenance(provenance)
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("nonempty independent calibration rows required")
    if not isinstance(breakpoints, (list, tuple)) or len(set(breakpoints)) != len(breakpoints):
        raise ValueError("explicit unique candidate breakpoints required")
    if any(type(b) is not int or b <= 0 for b in breakpoints):
        raise ValueError("positive integer candidate breakpoints required")
    training = []
    evidence = []
    retained = []
    for row in rows:
        if row.get("split") != "calibration":
            raise ValueError("only calibration rows may fit or select a transfer model")
        size, mean = row.get("bytes"), row.get("mean_ns")
        if type(size) is not int or size <= 0 or type(mean) not in (int, float) or not math.isfinite(mean) or mean <= 0:
            raise ValueError("positive integer bytes and finite positive inclusive mean required")
        if "contract" in row:
            observed_contract = validate_contract(row["contract"])
        else:
            observed_contract = validate_contract(dict(contract,
                memory_mode=row.get("memory_mode"), measurement_api=row.get("measurement_api"),
                completion_semantic=row.get("completion_semantic"),
                preparation_contract=row.get("preparation_contract")))
        if observed_contract != contract:
            raise ValueError("calibration rows mix memory/API/preparation/runtime contracts")
        for name in ("route", "memory_mode", "measurement_api", "completion_semantic", "preparation_contract", "scope_sha256"):
            if name in row and row[name] != contract[name]:
                raise ValueError("row metadata contradicts its explicit contract: " + name)
        pointer = row.get("host_pointer")
        expected = "registered_host" if contract["memory_mode"] == "registered" else "unregistered_pageable"
        if not isinstance(pointer, dict) or pointer.get(expected) is not True:
            raise ValueError("actual host-pointer registration does not establish the memory mode")
        if pointer.get("registered_host") is True and pointer.get("unregistered_pageable") is True:
            raise ValueError("contradictory host-pointer registration categories")
        raw = _receipt(row.get("raw"))
        training.append(dict(split="calibration", bytes=size, mean_ns=mean))
        evidence.append(raw)
        retained.append(json.loads(json.dumps(row, allow_nan=False)))
    sizes = sorted({row["bytes"] for row in training})
    # Every LOO fold must retain observations on both sides. This also avoids
    # zero design columns when an endpoint is omitted from a very small grid.
    admitted = tuple(b for b in breakpoints
        if sum(size < b for size in sizes) >= 2 and sum(size > b for size in sizes) >= 2)
    fit = fit_from_training(training, breakpoints=admitted)
    selected = fit["selected"]
    result = dict(schema="prepared-transfer-response-profile-1", contract=contract,
        parameters={key: selected[key] for key in PARAMETER_FIELDS},
        minimum_bytes=min(row["bytes"] for row in training),
        maximum_bytes=max(row["bytes"] for row in training),
        qualified=False, qualification_receipt=None,
        evidence=evidence, provenance=provenance, calibration_rows=retained,
        candidate_selection=fit,
        requested_breakpoints=list(breakpoints), admitted_breakpoints=list(admitted),
        family="existing nonnegative affine or one continuous hinge",
        parameter_target="inclusive bare host-completed response mean_ns only",
        complete_strategy_times_used=False, heldout_rows_used=False,
        diagnostic_phase_or_event_cost_added=False,
        physical_payload_bandwidth_identified=False)
    PreparedTransferModel.from_profile(result)
    return result


def freeze_training_profile(path, rows, *, contract, provenance,
                            breakpoints=(16384, 262144, 1048576)):
    """Write once after training. Existing freezes are never overwritten."""
    path = Path(path)
    result = fit_training(rows, contract=contract, provenance=provenance, breakpoints=breakpoints)
    result["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    content = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
    return result

