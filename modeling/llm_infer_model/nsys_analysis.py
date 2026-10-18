"""Compatibility names for Nsight measurement analysis now owned by profiling.

The backend is imported only when a legacy helper is accessed. Normal modeling
and performance evaluation do not import SQLite parsing or sample statistics.
"""
from __future__ import annotations

from importlib import import_module

__all__ = ["expected_graph_fragments", "analyze_nsys_sqlite"]


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    module = import_module("llm_infer_profile.legacy_nsys_analysis")
    try:
        return getattr(module, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__():
    return sorted(set(globals()) | set(__all__))
