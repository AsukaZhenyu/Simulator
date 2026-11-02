"""Source-bound profiling of the model-owned stateful service response.

No hardware is launched by these functions. Calibration reads only profile
actions/owned-worker history; whole invocation timings are validation targets.
"""
from .evidence import extract_job

__all__ = ["extract_job", "freeze_profile", "make_barrier", "evaluate_frozen"]


def freeze_profile(*args, **kwargs):
    from .calibration import freeze_profile as function
    return function(*args, **kwargs)


def make_barrier(*args, **kwargs):
    from .validation import make_barrier as function
    return function(*args, **kwargs)


def evaluate_frozen(*args, **kwargs):
    from .validation import evaluate_frozen as function
    return function(*args, **kwargs)
