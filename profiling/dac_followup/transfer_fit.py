"""Fit transparent nonnegative transfer terms on calibration observations only."""
from __future__ import annotations

from itertools import combinations
import math
import statistics

import numpy as np
from llm_infer_model.performance.transfer_response import inclusive_response_ns


def design(sizes, breakpoint):
    x = np.asarray(sizes, dtype=float)
    if breakpoint is None:
        return np.column_stack((np.ones_like(x), x))
    return np.column_stack((np.ones_like(x), np.minimum(x, breakpoint), np.maximum(0, x-breakpoint)))


def nonnegative_relative_fit(sizes, means, breakpoint):
    """Exact small active-set NNLS with relative-error weights, no black box."""
    y = np.asarray(means, dtype=float)
    if len(y) < 3 or not np.all(np.isfinite(y)) or np.any(y <= 0):
        raise ValueError("positive finite independent size means required")
    matrix = design(sizes, breakpoint) / y[:, None]
    target = np.ones_like(y)
    # Rescaling columns improves conditioning without changing constraints.
    norms = np.linalg.norm(matrix, axis=0)
    normalized = matrix / norms
    best = None
    for count in range(1, matrix.shape[1] + 1):
        for active in combinations(range(matrix.shape[1]), count):
            values = np.linalg.lstsq(normalized[:, active], target, rcond=None)[0]
            if np.any(values < -1e-12):
                continue
            coeff = np.zeros(matrix.shape[1])
            coeff[list(active)] = np.maximum(values, 0) / norms[list(active)]
            loss = float(np.mean((matrix @ coeff - target) ** 2))
            if best is None or loss < best[0]:
                best = (loss, coeff)
    if best is None:
        raise ValueError("nonnegative relative fit unavailable")
    values = best[1].tolist()
    return dict(fixed_ns=values[0], small_ns_per_byte=values[1],
                large_ns_per_byte=values[1] if breakpoint is None else values[2],
                breakpoint_bytes=breakpoint, calibration_relative_mse=best[0])


def predict(model, size):
    return inclusive_response_ns(model, size)


def fit_from_training(rows, *, breakpoints=(16384, 262144, 1048576)):
    """LOO model selection uses training sizes only; heldout never read here."""
    grouped = {}
    for row in rows:
        if row.get("split") != "calibration":
            raise ValueError("heldout observation cannot fit or choose transfer model")
        grouped.setdefault(row["bytes"], []).append(row["mean_ns"])
    sizes = sorted(grouped)
    means = [statistics.mean(grouped[size]) for size in sizes]
    if len(sizes) < 5:
        raise ValueError("at least five independent calibration sizes required")
    candidates = []
    for breakpoint in (None, *breakpoints):
        if breakpoint is not None and not sizes[1] <= breakpoint <= sizes[-2]:
            continue
        errors = []
        for index, observed in enumerate(means):
            fit = nonnegative_relative_fit(sizes[:index]+sizes[index+1:], means[:index]+means[index+1:], breakpoint)
            errors.append(abs(predict(fit, sizes[index])-observed)/observed)
        fitted = nonnegative_relative_fit(sizes, means, breakpoint)
        candidates.append(dict(fitted, loo_relative_mse=statistics.mean(v*v for v in errors),
                               loo_maximum_ape=max(errors), calibration_sizes=sizes))
    chosen = min(candidates, key=lambda model: (model["loo_relative_mse"], model["breakpoint_bytes"] is not None))
    return dict(selected=chosen, candidates=candidates, selection="minimum leave-one-calibration-size-out relative MSE; no heldout input")
