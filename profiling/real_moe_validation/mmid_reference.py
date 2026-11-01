"""Exact integer oracle for the bounded CUDA mm_ids_helper<8> diagnosis.

All strides are in int32 elements, not bytes.  Padding is preserved in the
input storage but is not interpreted as expert IDs.  This oracle has no CUDA,
NumPy, model, timing, or fitting dependency.  Its intentionally bounded scope
is one sequence/microbatch, 1..256 tokens and 8..1024 experts; this is narrower
than the kernel's 22-bit token representation, and is reported explicitly.
"""
from __future__ import annotations

from collections.abc import Sequence


TOP_K = 8
MAX_TOKENS = 256
MAX_EXPERTS = 1024
INT32_MAX = (1 << 31) - 1
INT32_MIN = -(1 << 31)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _positive_int(value, name):
    _require(type(value) is int and 1 <= value <= INT32_MAX,
             name + " must be a positive int32 integer")
    return value


def reference(ids_storage, *, n_experts, n_tokens, si1, sis1, nchannels_y):
    """Return all three arrays consumed by the real CUDA MMQ implementation.

    ``ids_storage`` has exactly ``n_tokens * si1`` int32 elements, including
    any final row padding.  The eight IDs in token ``it`` begin at ``it*si1``.
    With unique top-8 IDs the helper groups by expert, preserving token order.
    Each grouped (token, slot) produces ``token*sis1 + slot%nchannels_y`` in
    ids_src1 and ``token*8 + slot`` in ids_dst.  Empty experts have repeated
    adjacent expert_bounds.  No unsupported input is clipped or repaired.
    """
    n_experts = _positive_int(n_experts, "n_experts")
    n_tokens = _positive_int(n_tokens, "n_tokens")
    si1 = _positive_int(si1, "si1")
    sis1 = _positive_int(sis1, "sis1")
    nchannels_y = _positive_int(nchannels_y, "nchannels_y")
    _require(TOP_K <= n_experts <= MAX_EXPERTS,
             "n_experts outside bounded oracle scope 8..1024")
    _require(n_tokens <= MAX_TOKENS,
             "n_tokens outside bounded oracle scope 1..256")
    _require(si1 >= TOP_K, "si1 overlaps the eight expert-ID slots")
    _require(sis1 >= nchannels_y,
             "sis1 must cover the declared source-channel row")
    _require((n_tokens - 1) * sis1 + min(TOP_K, nchannels_y) - 1 <= INT32_MAX,
             "ids_src1 arithmetic would exceed int32")
    _require(isinstance(ids_storage, Sequence)
             and not isinstance(ids_storage, (str, bytes, bytearray)),
             "ids_storage must be an int32 element sequence")
    _require(len(ids_storage) == n_tokens * si1,
             "ids_storage must contain the complete declared strided rows")
    _require(all(type(value) is int and INT32_MIN <= value <= INT32_MAX
                 for value in ids_storage),
             "ids_storage contains a non-int32 element")

    counts = [0] * n_experts
    rows = []
    for token in range(n_tokens):
        row = tuple(ids_storage[token * si1:token * si1 + TOP_K])
        _require(all(0 <= expert < n_experts for expert in row),
                 "expert ID out of bounds in token " + str(token))
        _require(len(set(row)) == TOP_K,
                 "top-8 expert IDs must be unique within each token")
        rows.append(row)
        for expert in row:
            counts[expert] += 1

    bounds = [0]
    for count in counts:
        bounds.append(bounds[-1] + count)
    _require(bounds[-1] == n_tokens * TOP_K,
             "internal grouped element count mismatch")
    ids_src1 = [None] * bounds[-1]
    ids_dst = [None] * bounds[-1]
    cursors = bounds[:-1].copy()
    for token, row in enumerate(rows):
        for slot, expert in enumerate(row):
            grouped_index = cursors[expert]
            ids_src1[grouped_index] = token * sis1 + slot % nchannels_y
            ids_dst[grouped_index] = token * TOP_K + slot
            cursors[expert] += 1
    _require(cursors == bounds[1:], "internal expert segment extent mismatch")

    return dict(
        schema="real-moe-mmid-top8-reference-1",
        status="computed",
        n_experts=n_experts,
        n_tokens=n_tokens,
        n_expert_used=TOP_K,
        si1=si1,
        sis1=sis1,
        nchannels_y=nchannels_y,
        stride_unit="int32 elements",
        source_storage_elements=len(ids_storage),
        source_required_read_elements=(n_tokens - 1) * si1 + TOP_K,
        grouped_elements=n_tokens * TOP_K,
        ids_src1=ids_src1,
        ids_dst=ids_dst,
        expert_bounds=bounds,
        grouping_order="expert ascending, token ascending, unique selected slot",
        oracle_scope=dict(max_tokens=MAX_TOKENS, max_experts=MAX_EXPERTS,
                          top_k=TOP_K, kernel_token_encoding_bits=22),
        floating_point_math=False,
        hardware_executed=False,
        hardware_accuracy_certified=False,
    )
