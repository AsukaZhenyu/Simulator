"""Necessary preparation compatibility checks for timing/observer comparisons.

Matching preparation permits an observer-effect check; it never proves that
instrumentation is unbiased. Unknown fields cannot establish compatibility.
"""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PreparationContract:
    reset_before_batch: bool | None = None
    untimed_complete_prime_passes: int | None = None
    timed_replays_per_batch: int | None = None
    fixed_input_replay: bool | None = None
    warmup_iterations_min: int | None = None
    warmup_ms_min: int | None = None

    def __post_init__(self):
        for name in ("reset_before_batch", "fixed_input_replay"):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise ValueError(name + " must be explicit bool or unknown")
        for name in ("untimed_complete_prime_passes", "timed_replays_per_batch",
                     "warmup_iterations_min", "warmup_ms_min"):
            value = getattr(self, name)
            minimum = 1 if name == "timed_replays_per_batch" else 0
            if value is not None and (type(value) is not int or value < minimum):
                raise ValueError(name + " has an invalid explicit count")


def compare_preparation(target, observer):
    wanted, actual = asdict(target), asdict(observer)
    unknown = [name for name in wanted if wanted[name] is None or actual[name] is None]
    differences = {name: {"target": wanted[name], "observer": actual[name]}
                   for name in wanted if wanted[name] is not None
                   and actual[name] is not None and wanted[name] != actual[name]}
    status = "mismatch" if differences else "unknown" if unknown else "match"
    return dict(status=status, allows_unbiased_observer_check=status == "match",
                differences=differences, unknown_fields=unknown,
                scope="necessary preparation check only; not an observer-bias certificate")


def require_matched_preparation(target, observer):
    result = compare_preparation(target, observer)
    if result["status"] != "match":
        raise ValueError("observer preparation " + result["status"] + ": " + str(result))
    return result
