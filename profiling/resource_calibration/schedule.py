"""Explicit offered-resource timelines, independent of any timing predictor.

Every phase specifies finite owned work and owned allocation sizes.  A requested
replay worker is not a utilization percentage, and an allocated byte is not a
claim that physical memory remains resident.  Those require separate telemetry.
"""
from __future__ import annotations

from copy import deepcopy
import itertools
import random
from typing import Any

SCHEMA = "resource-trajectory-schedule-0.1"
RESOURCE_KEYS = ("cpu_replay", "gpu_replay", "host_memory_bytes", "device_memory_bytes")
MAX_PHASES = 256
MAX_PHASE_HOLD_MS = 60_000
MAX_MEMORY_BYTES = 1 << 30


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def validate_resources(resources: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(resources, dict) or set(resources) != set(RESOURCE_KEYS):
        raise ValueError("resources require exactly cpu_replay, gpu_replay, host_memory_bytes, device_memory_bytes")
    for key in RESOURCE_KEYS[:2]:
        if type(resources[key]) is not bool:
            raise ValueError(key + " must be a boolean offered-work request")
    for key in RESOURCE_KEYS[2:]:
        _integer(resources[key], key, 0, MAX_MEMORY_BYTES)
    return deepcopy(resources)


def validate_schedule(schedule: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(schedule, dict) or schedule.get("schema") != SCHEMA:
        raise ValueError("unsupported resource schedule schema")
    if set(schedule) != {"schema", "id", "split", "repeat", "seed", "phases"}:
        raise ValueError("unknown or missing schedule field")
    if not isinstance(schedule["id"], str) or not schedule["id"] or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in schedule["id"]):
        raise ValueError("schedule id must be a nonempty safe identifier")
    if schedule["split"] not in ("train", "heldout", "diagnostic"):
        raise ValueError("schedule split must be train, heldout, or diagnostic")
    _integer(schedule["repeat"], "repeat", 0, 100_000)
    _integer(schedule["seed"], "seed", 0, (1 << 63) - 1)
    phases = schedule["phases"]
    if not isinstance(phases, list) or not 1 <= len(phases) <= MAX_PHASES:
        raise ValueError("schedule requires 1 through 256 explicit phases")
    for index, phase in enumerate(phases):
        if not isinstance(phase, dict) or set(phase) != {"index", "hold_ms", "resources"} or phase["index"] != index or isinstance(phase["index"], bool):
            raise ValueError("phase identity or fields invalid")
        _integer(phase["hold_ms"], "phase hold_ms", 200, MAX_PHASE_HOLD_MS)
        validate_resources(phase["resources"])
    return deepcopy(schedule)


def make_schedules(*, phase_duration_ms: int = 1000, host_memory_bytes: int = 64 << 20,
                   device_memory_bytes: int = 64 << 20, seed: int = 80217,
                   repeats: int = 3, smoke: bool = False) -> list[dict[str, Any]]:
    """Balanced binary offered controls with independently shuffled trajectories.

    Heldout uses a disjoint shuffle seed and order.  The numeric control vectors
    are deliberately shared, so validation tests a new temporal ordering rather
    than silently claiming extrapolation to unseen memory amounts or duty rates.
    Smoke retains all sixteen numeric combinations, using one independent
    process per split.  It cannot satisfy a three-process acceptance criterion.
    """
    _integer(phase_duration_ms, "phase_duration_ms", 200, MAX_PHASE_HOLD_MS)
    _integer(host_memory_bytes, "host_memory_bytes", 0, MAX_MEMORY_BYTES)
    _integer(device_memory_bytes, "device_memory_bytes", 0, MAX_MEMORY_BYTES)
    _integer(seed, "seed", 0, (1 << 62) - 1)
    _integer(repeats, "repeats", 1, 100)
    if type(smoke) is not bool:
        raise ValueError("smoke must be boolean")
    vectors = [{"cpu_replay": cpu, "gpu_replay": gpu,
                "host_memory_bytes": host_memory_bytes if host else 0,
                "device_memory_bytes": device_memory_bytes if device else 0}
               for cpu, gpu, host, device in itertools.product((False, True), repeat=4)]
    schedules = []
    splits = ("train", "heldout")
    for split in splits:
        for repeat in range(1 if smoke else repeats):
            trajectory_seed = seed + repeat + (1_000_003 if split == "heldout" else 0)
            ordered = deepcopy(vectors)
            random.Random(trajectory_seed).shuffle(ordered)
            item = {"schema": SCHEMA, "id": f"{split}-{repeat}", "split": split,
                    "repeat": repeat, "seed": trajectory_seed,
                    "phases": [{"index": i, "hold_ms": phase_duration_ms, "resources": vector}
                               for i, vector in enumerate(ordered)]}
            schedules.append(validate_schedule(item))
    return schedules


def phase_for_interval(events: list[dict[str, Any]], begin_tick: int, end_tick: int) -> dict[str, Any]:
    """Conservative exact containment: never label a crossing as steady state.

    Phase transitions are retained.  An interval outside all steady intervals
    gets an explicit transition/outside annotation, not an invented load vector.
    """
    _integer(begin_tick, "begin_tick", 0, (1 << 127) - 1)
    _integer(end_tick, "end_tick", begin_tick + 1, (1 << 127) - 1)
    previous_end = -1
    for event in events:
        left, right = event["steady_begin_tick"], event["steady_end_tick"]
        _integer(left, "steady_begin_tick", 0, (1 << 127) - 1)
        _integer(right, "steady_end_tick", left, (1 << 127) - 1)
        if left < previous_end:
            raise ValueError("steady phase intervals overlap or are out of order")
        previous_end = right
        validate_resources(event["requested_resources"])
    for event in events:
        left, right = event["steady_begin_tick"], event["steady_end_tick"]
        if left <= begin_tick and end_tick <= right:
            return {"kind": "steady", "phase_index": event["phase_index"],
                    "offered_resources": validate_resources(event["requested_resources"])}
    return {"kind": "transition_or_outside", "phase_index": None, "offered_resources": None}
