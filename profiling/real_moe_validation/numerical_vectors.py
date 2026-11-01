"""Bounded, exact-file F32 logit diagnostics using only the Python stdlib.

The current diagnosis exports at most 8 x 50304 values. No acceptance tolerance
is inferred: bitwise equality is reported separately from numerical distances.
ULP distance uses numeric float ordering with +0 and -0 at the same location;
their distinct bits still count as a bitwise difference. Nonfinite values are
rejected. Relative L2 is undefined (None) for a zero reference and nonzero
difference; two numerically zero vectors use the explicit zero-distance rule.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
import struct

FORMAT = "little-endian IEEE754 F32 row-major"
MAX_ROWS = 8
MAX_COLUMNS = 50304
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_F32 = struct.Struct("<f")
_U32 = struct.Struct("<I")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _dimension(value, limit, name):
    _require(type(value) is int and 1 <= value <= limit, "invalid or unsupported " + name)
    return value


def _load(meta):
    _require(isinstance(meta, dict), "vector metadata object required")
    _require(meta.get("format") == FORMAT, "vector format must be explicit little-endian IEEE754 F32 row-major")
    rows = _dimension(meta.get("rows"), MAX_ROWS, "rows")
    columns = _dimension(meta.get("columns"), MAX_COLUMNS, "columns")
    size = rows * columns * 4
    _require(type(meta.get("bytes")) is int and meta["bytes"] == size, "metadata bytes do not equal declared rows*columns*4")
    digest = meta.get("sha256")
    _require(isinstance(digest, str) and _SHA.fullmatch(digest), "explicit vector sha256 required")
    _require(isinstance(meta.get("path"), str) and bool(meta["path"]), "explicit vector path required")
    path = Path(meta["path"])
    _require(path.is_absolute() and path.is_file(), "absolute existing vector file path required")
    _require(path.stat().st_size == size, "actual vector bytes differ from metadata/shape")
    raw = path.read_bytes()
    _require(len(raw) == size, "actual vector file length changed while reading")
    actual = hashlib.sha256(raw).hexdigest()
    _require(actual == digest, "vector file sha256 mismatch")
    values = tuple(value[0] for value in struct.iter_unpack("<f", raw))
    _require(len(values) == rows * columns and all(math.isfinite(value) for value in values), "nonfinite or incomplete IEEE754 F32 vector")
    return dict(path=str(path.resolve()),sha256=actual,bytes=size,format=FORMAT,rows=rows,columns=columns), raw, values


def inspect(meta, logit_observations):
    """Validate whole file, finite F32 values and every output's recorded hash."""
    checked, raw, values = _load(meta)
    rows, columns = checked["rows"], checked["columns"]
    _require(isinstance(logit_observations, list) and len(logit_observations) == rows, "complete per-output logit observation inventory required")
    indexed = {}
    for observation in logit_observations:
        _require(isinstance(observation, dict), "logit observation object required")
        index = observation.get("output_index")
        _require(type(index) is int and 0 <= index < rows and index not in indexed, "logit observation output index duplicate or out of bounds")
        indexed[index] = observation
    per_output = []
    for index in range(rows):
        observation = indexed[index]
        digest = observation.get("full_f32_logits_sha256")
        _require(isinstance(digest, str) and _SHA.fullmatch(digest), "full_f32_logits_sha256 missing/invalid")
        row = raw[index * columns * 4:(index + 1) * columns * 4]
        actual = hashlib.sha256(row).hexdigest()
        _require(actual == digest, "vector row hash differs from recorded output " + str(index))
        if "vocabulary_size" in observation:
            _require(type(observation["vocabulary_size"]) is int and observation["vocabulary_size"] == columns,
                     "recorded vocabulary size differs from vector columns")
        part = values[index * columns:(index + 1) * columns]
        per_output.append(dict(output_index=index,full_f32_logits_sha256=actual,entries=len(part),
                               finite=True,minimum=min(part),maximum=max(part)))
    return dict(schema="real-moe-logit-vector-inspection-1",status="passed",metadata=checked,
                entries_total=rows * columns,finite_entries=rows * columns,
                output_rows_bound_to_observations=rows,per_output=per_output,
                file_modified=False,vectors_generated_or_filled=False,hardware_accuracy_certified=False)


def _ordered(bits):
    # Collapse signed zeros while preserving one unit between adjacent numeric
    # F32 values on either side. NaN/Inf have already been rejected by _load.
    return 0x80000000 - (bits & 0x7fffffff) if bits & 0x80000000 else 0x80000000 + bits


def _metrics(left_raw, right_raw, left_values, right_values):
    equal_bits = 0
    max_ulp = 0
    first = None
    differences = []
    squares = []
    reference_squares = []
    for index, (left, right) in enumerate(zip(left_values, right_values)):
        left_bits = _U32.unpack_from(left_raw, index * 4)[0]
        right_bits = _U32.unpack_from(right_raw, index * 4)[0]
        same = left_bits == right_bits
        equal_bits += int(same)
        if not same and first is None:
            first = dict(index=index,left_value=left,right_value=right,
                         left_bits=f"0x{left_bits:08x}",right_bits=f"0x{right_bits:08x}")
        difference = right - left
        differences.append(abs(difference))
        squares.append(difference * difference)
        reference_squares.append(left * left)
        max_ulp = max(max_ulp, abs(_ordered(left_bits) - _ordered(right_bits)))
    count = len(left_values)
    square_sum = math.fsum(squares)
    reference_sum = math.fsum(reference_squares)
    if reference_sum:
        relative = math.sqrt(square_sum / reference_sum)
        relative_status = "defined"
    elif not square_sum:
        relative, relative_status = 0.0, "both_numerically_zero_distance_convention"
    else:
        relative, relative_status = None, "undefined_zero_reference_norm"
    return dict(bitwise_equal=equal_bits == count,entries_total=count,entries_equal_bits=equal_bits,
                entries_different_bits=count - equal_bits,max_abs=max(differences),
                mean_abs=math.fsum(differences) / count,relative_l2=relative,relative_l2_status=relative_status,
                max_ulp=max_ulp,first_difference=first)


def compare(left_meta, right_meta):
    """Compare every value; caller separately invokes inspect to bind row hashes.

    No threshold is chosen. No differences are filtered, sampled, rounded away
    or fitted. Reported ULP is a numeric distance, not an acceptance decision.
    """
    left, left_raw, left_values = _load(left_meta)
    right, right_raw, right_values = _load(right_meta)
    _require((left["rows"], left["columns"]) == (right["rows"], right["columns"]), "vector row/column correspondence differs")
    columns = left["columns"]
    metrics = _metrics(left_raw, right_raw, left_values, right_values)
    first = metrics["first_difference"]
    if first is not None:
        first = dict(first,output_index=first["index"] // columns,vocab_index=first["index"] % columns)
    metrics["first_difference"] = first
    per_output = []
    for index in range(left["rows"]):
        start, end = index * columns, (index + 1) * columns
        row = _metrics(left_raw[start * 4:end * 4], right_raw[start * 4:end * 4], left_values[start:end], right_values[start:end])
        if row["first_difference"] is not None:
            row["first_difference"] = dict(row["first_difference"],output_index=index,vocab_index=row["first_difference"]["index"])
        per_output.append(dict(output_index=index,**row))
    return dict(schema="real-moe-logit-vector-comparison-1",status="compared",left=left,right=right,
                **metrics,per_output=per_output,
                ULP_semantics="numeric IEEE754 F32 ordering; +0 and -0 have ULP distance 0 but unequal bits",
                relative_l2_reference="left vector",acceptance_threshold_introduced=False,
                per_output_hash_binding_checked_here=False,
                row_binding_requirement="inspect(metadata, logit_observations) must be called separately for both inputs",
                vectors_generated_or_filled=False,files_modified=False,hardware_accuracy_certified=False)
