"""The document versions this package writes, and the checks it will not skip.

Three documents make one claim and they are written in a fixed order:

``predicted.json``   what the tables say a token step costs (written first)
``measured.json``    what the wall clock said, and under which protocol
``comparison.json``  the two, side by side

Each carries its own ``schema_version``.  A reader that assumed a shape instead
of checking it would be reading the wrong document after a rename, and the
numbers would still look plausible -- which is exactly the failure these checks
exist to prevent.
"""

from __future__ import annotations

PREDICTION = "dense_endtoend_prediction_v2"
MEASUREMENT = "dense_endtoend_measurement_v1"
# v3 adds, beside each row's point estimate, the spread of the repetitions that
# produced it and a session record of which rows were wider than the state band.
# A v2 comparison (no ``observation`` block) says nothing about its band, which
# is not the same claim as "the band was zero", so it is a different version
# rather than an older spelling of this one.  Archived v2 files stay readable by
# ``decompose``, which reads the rows and does not validate a version.
COMPARISON = "dense_endtoend_comparison_v3"
DECOMPOSITION = "dense_endtoend_decomposition_v1"
RESULT_MANIFEST = "composing_result_manifest_v1"

# A v2 prediction written before declared terms existed carries no
# ``composition_terms`` key.  That is zero declared terms, not a different
# schema: absent and empty mean the same thing and are read the same way.
_OPTIONAL_PREDICTION_KEYS = {"composition_terms", "composition_rule"}


def require_schema(document, expected: str, *, where: str) -> dict:
    if not isinstance(document, dict):
        raise ValueError(f"{where} must be a JSON object, got {type(document).__name__}")
    declared = document.get("schema_version")
    if declared != expected:
        raise ValueError(f"{where}: schema_version is {declared!r}, expected {expected!r}; "
                         "this reader does not guess at an older or newer layout")
    return document


def validate_prediction(document, *, where: str = "predicted.json") -> dict:
    require_schema(document, PREDICTION, where=where)
    models = document.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError(f"{where}.models must be a non-empty object")
    for model_id, entry in models.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{where}.models[{model_id!r}] must be an object")
        shapes = entry.get("shapes")
        if not isinstance(shapes, dict) or not shapes:
            raise ValueError(f"{where}.models[{model_id!r}].shapes must be a non-empty object")
        for name, shape in shapes.items():
            if not isinstance(shape.get("total_ns"), int):
                raise ValueError(f"{where}: {model_id}/{name} has no integer total_ns")
            if not isinstance(shape.get("block_sum_ns"), int):
                raise ValueError(f"{where}: {model_id}/{name} has no integer block_sum_ns")
        if not isinstance(entry.get("layers"), int) or entry["layers"] < 1:
            raise ValueError(f"{where}: {model_id} has no positive layer count")
    return document


def validate_measurement(document, *, where: str = "measured.json") -> dict:
    require_schema(document, MEASUREMENT, where=where)
    # The header is the environment the clock was taken in.  A file that cannot
    # say which CPUs, threads and binary produced its rows cannot be compared
    # with anything, least of all with a result arriving from another machine.
    for key in ("llama_bench_sha256", "threads", "cpu_mask", "bench_protocol",
                "decode_grouping", "prediction"):
        if key not in document:
            raise ValueError(f"{where}: the header does not declare {key!r}; a wall clock "
                             "without its protocol is not a comparable number")
    prediction = document["prediction"]
    if not isinstance(prediction, dict) or not prediction.get("sha256"):
        raise ValueError(f"{where}.prediction must record the sha256 of predicted.json")
    models = document.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError(f"{where}.models must be a non-empty object")
    for model_id, rows in models.items():
        if not isinstance(rows, dict) or not rows:
            raise ValueError(f"{where}.models[{model_id!r}] has no measured rows")
        for config, row in rows.items():
            samples = row.get("samples_ns")
            if not isinstance(samples, list) or not samples:
                raise ValueError(f"{where}: {model_id}/{config} has no samples")
            if not isinstance(row.get("avg_ns"), int):
                raise ValueError(f"{where}: {model_id}/{config} has no integer avg_ns")
    return document


def validate_comparison(document, *, where: str = "comparison.json") -> dict:
    require_schema(document, COMPARISON, where=where)
    models = document.get("models")
    if not isinstance(models, dict):
        raise ValueError(f"{where}.models must be an object")
    for model_id, rows in models.items():
        for config, row in rows.items():
            for key in ("predicted_ns", "observed_ns", "signed_relative_error",
                        "observation"):
                if key not in row:
                    raise ValueError(f"{where}: {model_id}/{config} lacks {key!r}")
    if "session" not in document:
        raise ValueError(f"{where}: no session record; the band the rows were taken "
                         "inside is part of what a comparison claims")
    # A comparison whose decode rows came from one process per depth and one
    # that grouped the depths are not the same measurement: the per-shape
    # protocol charges every depth its own first-decode cost and reports a
    # lower slope (2026-11-05 B2).  Like the band, a missing grouping is not
    # the same claim as "the grouping did not matter".
    if not document.get("decode_grouping"):
        raise ValueError(f"{where}: no decode grouping; the slope of a decode run "
                         "depends on which depths shared a process")
    return document
