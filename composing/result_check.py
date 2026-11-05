"""Accept a result that came back from another machine, or say what is missing.

Measurements are taken on more than one machine: the repository is pulled,
measured on another device, and the result comes back on its own branch.  A
result is only comparable if it carries the environment it was taken in, and a
number whose binary, CPU set or prediction cannot be identified is not evidence
for or against a model.

This module does not re-run anything and does not judge accuracy.  It answers
one question -- *is this file the result it claims to be?* -- and writes the
answer as a manifest that can be archived next to the raw result.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import schema
from .spec import SHAPES, dump_json, sha256_file

# Every one of these is part of what a wall clock means.  A file that omits one
# cannot be compared with a run from another machine, so it is refused here
# rather than discovered later in a disagreement.
REQUIRED_HEADER = ("llama_bench_sha256", "threads", "cpu_mask", "bench_protocol",
                   "decode_grouping", "warmup", "prediction", "models")


def _cpu_mask_hex(cpu_mask) -> str:
    mask = 0
    for cpu in cpu_mask:
        mask |= 1 << int(cpu)
    return hex(mask)


def _prediction_sha(directory: Path, prediction: Path | None) -> tuple[str | None, str]:
    """``(sha, where it came from)``; ``(None, reason)`` when it cannot be found."""
    local = directory / "predicted.json"
    if local.is_file():
        return sha256_file(local), str(local)
    if prediction is not None:
        return sha256_file(prediction), str(prediction)
    return None, "no predicted.json in the result directory and none supplied"


def inspect_result(directory: Path, *, prediction: Path | None = None,
                   expect_llama_bench_sha256: str | None = None) -> dict:
    """Check one result directory and describe it.  Problems are listed, not raised.

    The caller decides what to do with a bad result; a batch of results returning
    from several branches should not stop at the first one that is missing a
    field.
    """
    directory = Path(directory)
    path = directory / "measured.json"
    if not path.is_file():
        raise ValueError(f"{directory}: no measured.json; this is not a result directory")
    document = json.loads(path.read_text(encoding="utf-8"))
    problems: list[str] = []

    try:
        schema.validate_measurement(document, where=str(path))
    except ValueError as error:
        problems.append(str(error))
    for key in REQUIRED_HEADER:
        # Present *and* a value: "warmup: null" names the field without saying
        # which sample was dropped, and a null grouping does not say whether the
        # decode points shared a process.
        if document.get(key) is None:
            problems.append(f"the header does not declare {key!r}")

    digest, source = _prediction_sha(directory, prediction)
    declared = (document.get("prediction") or {})
    if digest is None:
        problems.append(source)
    elif declared.get("sha256") != digest:
        problems.append(f"measured.json was taken against prediction {declared.get('sha256')}, "
                        f"but the prediction on hand ({source}) is {digest}; these rows "
                        "belong to another pre-registration")
    actual_bench = document.get("llama_bench_sha256")
    if expect_llama_bench_sha256 and actual_bench != expect_llama_bench_sha256:
        problems.append(f"llama-bench is {actual_bench}, expected {expect_llama_bench_sha256}; "
                        "another binary is another implementation of the same flags")

    header_threads = document.get("threads")
    header_mask = document.get("cpu_mask")
    expected_mask = (_cpu_mask_hex(header_mask)
                     if isinstance(header_mask, list) and header_mask else None)
    models = {}
    for model_id, rows in (document.get("models") or {}).items():
        per_config = {}
        for config, row in rows.items():
            samples = row.get("samples_ns") or []
            measured_by = row.get("llama_bench") or {}
            if measured_by.get("n_threads") != header_threads:
                problems.append(f"{model_id}/{config}: row ran {measured_by.get('n_threads')} "
                                f"threads, header says {header_threads}")
            if expected_mask and measured_by.get("cpu_mask") != expected_mask:
                problems.append(f"{model_id}/{config}: row reports cpu_mask="
                                f"{measured_by.get('cpu_mask')}, header says {expected_mask}")
            if config in SHAPES:
                _, tokens, history = SHAPES[config]
                expected_depth = 0 if SHAPES[config][0] == "prefill" else history
                if measured_by.get("n_depth") not in (None, expected_depth):
                    problems.append(f"{model_id}/{config}: row reports n_depth="
                                    f"{measured_by.get('n_depth')}, shape is {expected_depth}")
            per_config[config] = {"avg_ns": row.get("avg_ns"), "samples": len(samples),
                                  "observed_cv": _cv(samples)}
        models[model_id] = per_config

    return {"schema_version": schema.RESULT_MANIFEST, "ok": not problems,
            "result_directory": str(directory.resolve()),
            "measured_json": str(path.resolve()), "measured_sha256": sha256_file(path),
            "prediction": {"sha256": digest, "source": source,
                           "declared_sha256": declared.get("sha256")},
            "environment": {"llama_bench": document.get("llama_bench"),
                            "llama_bench_sha256": actual_bench,
                            "threads": header_threads, "cpu_mask": header_mask,
                            "thread_placement": document.get("thread_placement"),
                            "decode_grouping": document.get("decode_grouping"),
                            "warmup": document.get("warmup"),
                            # The prediction names the file; this says which
                            # directory the measuring machine read it from.
                            "models_directory": document.get("models_directory"),
                            "bench_protocol": document.get("bench_protocol"),
                            "measured_at_utc": document.get("measured_at_utc")},
            "models": models, "problems": problems}


def _cv(samples) -> float | None:
    if not samples:
        return None
    mean = sum(samples) / len(samples)
    if not mean:
        return None
    variance = sum((value - mean) ** 2 for value in samples) / len(samples)
    return (variance ** 0.5) / mean


def require_result(directory: Path, **kwargs) -> dict:
    """``inspect_result`` for callers that want a bad result to stop the run."""
    manifest = inspect_result(directory, **kwargs)
    if not manifest["ok"]:
        raise ValueError(f"{directory}: the result does not pass its own check:\n  "
                         + "\n  ".join(manifest["problems"]))
    return manifest


def write_manifest(path: Path, manifest: dict) -> Path:
    dump_json(path, manifest)
    return path
