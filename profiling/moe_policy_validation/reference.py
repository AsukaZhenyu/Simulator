"""Independent NumPy oracle for the bounded F32 expert execution boundary.

No measured latency, compiler state, or native output participates in this oracle.
The native GGML graph and this NumPy expression are independent implementations.
"""
from __future__ import annotations

import math

import numpy as np


def weights(seed: int, expert: int, hidden: int, intermediate: int):
    result = []
    for role, (width, outputs) in enumerate(
        ((hidden, intermediate), (hidden, intermediate), (intermediate, hidden))
    ):
        index = np.arange(width * outputs, dtype=np.uint64)
        z = (seed + (index + 1) * 0x9E3779B9 + (3 * expert + role + 1) * 0x85EBCA6B) & 0xFFFFFFFF
        z ^= z >> 16
        z = (z * 0x7FEB352D) & 0xFFFFFFFF
        z ^= z >> 15
        z = (z * 0x846CA68B) & 0xFFFFFFFF
        z ^= z >> 16
        data = (((z >> 8).astype(np.float64) / 16777216.0 - 0.5) / math.sqrt(width)).astype(np.float32)
        result.append(data.reshape(outputs, width))
    return tuple(result)


def reference_outputs(program: dict) -> list:
    c = program["config"]
    bank = [tuple(w.astype(np.float64) for w in weights(program["seed"], expert, c["hidden"], c["intermediate"]))
            for expert in range(c["experts"])]
    outputs = []
    for step in program["steps"]:
        # Quantize the declared native input/score boundary once, then evaluate
        # the mathematical function in double precision, independently of GGML.
        x = np.asarray(step["input"], dtype=np.float32).astype(np.float64)
        scores = np.asarray(step["scores"], dtype=np.float32).astype(np.float64)
        y = x.copy()
        for expert in sorted({expert for selected in step["routes"] for expert in selected}):
            rows = [row for row, selected in enumerate(step["routes"]) if expert in selected]
            selected_scores = np.asarray([scores[row, step["routes"][row].index(expert)] for row in rows])
            gate, up, down = bank[expert]
            g, u = x[rows] @ gate.T, x[rows] @ up.T
            value = ((g / (1.0 + np.exp(-g))) * u) @ down.T
            y[rows] += selected_scores[:, None] * value
        outputs.append(y.tolist())
    return outputs


def copy_traffic(program: dict) -> dict:
    """Derive physical payload bytes from actions and declared token rows."""
    c = program["config"]
    weights = activations = 0
    for action in program["actions"]:
        if action["kind"] == "LOAD":
            weights += 3 * c["hidden"] * c["intermediate"] * 4
        elif action["kind"] == "EXPERT" and action["device"] == "gpu":
            rows = sum(action["expert"] in ids for ids in program["steps"][action["step"]]["routes"])
            activations += rows * c["hidden"] * 4
    return dict(weight_h2d_bytes=weights, total_h2d_bytes=weights + activations, total_d2h_bytes=activations)


def verify_result(program: dict, native: dict, *, expected=None, atol=2e-5, rtol=2e-4) -> dict:
    if native["actions"] != program["actions"]:
        raise ValueError("native action sequence differs from the exported compiled program")
    if native["config"] != program["config"] or native["seed"] != program["seed"]:
        raise ValueError("native geometry/seed differs")
    if expected is None:
        expected = reference_outputs(program)
    if len(expected) != len(native["outputs"]):
        raise ValueError("native output step count differs")
    differences = []
    accepted = True
    for index, (ref, result) in enumerate(zip(expected, native["outputs"])):
        ref, result = np.asarray(ref), np.asarray(result)
        if ref.shape != result.shape or not np.isfinite(result).all():
            raise ValueError("native output shape/nonfinite mismatch")
        err = np.abs(ref - result)
        valid = bool(np.all(err <= atol + rtol * np.abs(ref)))
        accepted &= valid
        relative_l2 = float(np.linalg.norm(ref - result) / max(np.linalg.norm(ref), np.finfo(float).tiny))
        differences.append(dict(step=index, max_abs_error=float(err.max()), rms_error=float(np.sqrt(np.mean(err**2))), relative_l2_error=relative_l2, numerically_equal=valid))
    c = program["config"]
    weight_bytes = 3 * c["hidden"] * c["intermediate"] * 4
    for field, expected_bytes in copy_traffic(program).items():
        if native[field] != expected_bytes:
            raise ValueError(f"physical {field} differs from compiled actions")
    if native["gpu_weight_logical_bytes"] not in (0, c["gpu_slots"] * weight_bytes):
        raise ValueError("GPU weights exceed the declared slot pool")
    if native["peak_resident_experts"] > c["gpu_slots"]:
        raise ValueError("physical live experts exceed capacity")
    return dict(numerically_equal=accepted, atol=atol, rtol=rtol, max_abs_error=max(d["max_abs_error"] for d in differences), steps=differences,
                compiled_action_sequence_equal=True, bounded_weight_slots=True,
                oracle="Independent NumPy float64 SwiGLU/weighted-residual expression on declared F32 inputs and weights",
                scope="Routed expert FFN subgraph; no router, attention, tokenizer or language-model quality claim")
