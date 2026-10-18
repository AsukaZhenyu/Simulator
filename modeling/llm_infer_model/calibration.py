"""Compatibility names for the calibration workflow now owned by profiling.

Importing this module does not load estimators or measurement runners. Legacy
function access loads the standard-library-only llm_infer_profile namespace;
performance prediction remains in llm_infer_model.
"""
from __future__ import annotations

from importlib import import_module

__all__ = [
    "H2D_PATTERN", "parse_llama_bench_payload", "run_llama_bench",
    "run_h2d_bandwidth_test", "evaluate_context_validation",
    "fit_scheduler_overhead", "fit_pipeline_service", "fit_host_staging",
    "fit_state_transfer", "fit_kv_storage_tiers", "calibrate_configs",
]


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    module = import_module("llm_infer_profile.legacy_calibration")
    try:
        return getattr(module, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__():
    return sorted(set(globals()) | set(__all__))
