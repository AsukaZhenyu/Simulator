"""Legacy measurement summaries; no performance prediction equations."""
from __future__ import annotations

import statistics


def statistics_ns(values):
    """Preserve the historical positive-integer timing summary contract.

    Every supplied sample is retained. The median is evidence for a measured
    timing boundary, not a derived compute throughput or bandwidth parameter.
    This helper deliberately does not change the historical numerical schema.
    New physical parameter workflows use parameter_calibration instead.
    """
    if not isinstance(values, list) or len(values) < 3:
        raise ValueError("timing requires at least three samples")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("sample_ns must be an integer in the supported range")
    mean = statistics.mean(values)
    return {"median_ns": statistics.median(values), "mean_ns": mean,
            "min_ns": min(values), "max_ns": max(values),
            "cv": statistics.pstdev(values) / mean, "count": len(values)}
