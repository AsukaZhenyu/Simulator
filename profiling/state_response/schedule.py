"""Predeclared, exogenous offered-work trajectories; no timing predictor.

CPU/GPU values request owned replay work, not utilization percentages.  Memory
is allocated before each trajectory and held constant through its final sample.
The temporal holdout changes order and dwell time, not numerical state support.
"""
from __future__ import annotations

from copy import deepcopy
import itertools
import random

STATE_KEYS = ("cpu_replay", "gpu_replay", "host_memory_bytes", "device_memory_bytes")
SCHEMA = "state-response-schedule-1"


def _int(value, name, minimum=0, maximum=1 << 63):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def state(cpu, gpu, host_bytes, device_bytes):
    for key, value in zip(STATE_KEYS[:2], (cpu, gpu)):
        _int(value, key, maximum=1)
    for key, value in zip(STATE_KEYS[2:], (host_bytes, device_bytes)):
        _int(value, key, maximum=1 << 30)
    return dict(zip(STATE_KEYS, (cpu, gpu, host_bytes, device_bytes)))


def make_jobs(*, repeats=3, seed=20261021, memory_bytes=64 << 20,
              profile_hold_ms=1200, settle_ms=150, heldout_cycles=4, profile_state_holds_ms=None):
    """Return all jobs before observing hardware; equal process counts per cell."""
    _int(repeats, "repeats", 1, 20)
    _int(seed, "seed", maximum=1 << 40)
    _int(memory_bytes, "memory_bytes", 1, 1 << 30)
    _int(profile_hold_ms, "profile_hold_ms", 200, 10000)
    _int(settle_ms, "settle_ms", 0, profile_hold_ms - 1)
    _int(heldout_cycles, "heldout_cycles", 1, 16)
    if profile_state_holds_ms is not None:
        if not isinstance(profile_state_holds_ms, (tuple, list)) or len(profile_state_holds_ms) != 4:
            raise ValueError("exactly four profile state holds required")
        for hold in profile_state_holds_ms:
            _int(hold, "profile state hold", max(200, settle_ms + 1), 10000)
        profile_holds = dict(zip(itertools.product((0, 1), repeat=2), profile_state_holds_ms))
    else:
        profile_holds = dict.fromkeys(itertools.product((0, 1), repeat=2), profile_hold_ms)
    result = []
    for split in ("profile", "heldout"):
        for repeat in range(repeats):
            for memory_index, (host, device) in enumerate(itertools.product((0, memory_bytes), repeat=2)):
                local_seed = seed + repeat * 101 + memory_index * 7 + (1000003 if split == "heldout" else 0)
                rng = random.Random(local_seed)
                vectors = list(itertools.product((0, 1), repeat=2))
                order = []
                for _ in range(1 if split == "profile" else heldout_cycles):
                    part = list(vectors)
                    rng.shuffle(part)
                    if order and part[0] == order[-1]:
                        part[0], part[1] = part[1], part[0]
                    order.extend(part)
                phases = []
                elapsed = 0
                for index, (cpu, gpu) in enumerate(order):
                    # Independent, shorter dwell periods deliberately test
                    # recovery and operations crossing a resource change.
                    hold_ms = profile_holds[(cpu, gpu)] if split == "profile" else rng.choice((37, 61, 103, 181))
                    duration = hold_ms * 1000000
                    phases.append({"index": index, "begin_ns": elapsed,
                                   "end_ns": elapsed + duration,
                                   "calibration_begin_ns": elapsed + (settle_ms * 1000000 if split == "profile" else 0),
                                   "state": state(cpu, gpu, host, device)})
                    elapsed += duration
                job = {"id": f"{split}-r{repeat}-m{memory_index}", "split": split,
                       "repeat": repeat, "numeric_seed": local_seed + (200000000 if split == "heldout" else 0),
                       "schedule": {"schema": SCHEMA, "seed": local_seed,
                                    "state_semantics": "offered_load_controlled",
                                    "memory_fixed_for_trajectory": True,
                                    "duration_ns": elapsed, "phases": phases}}
                validate_schedule(job["schedule"])
                result.append(job)
    return result


def validate_schedule(value):
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("unsupported state-response schedule")
    if value.get("state_semantics") != "offered_load_controlled" or value.get("memory_fixed_for_trajectory") is not True:
        raise ValueError("offered controls and fixed memory must be explicit")
    phases = value.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("a nonempty phase sequence is required")
    elapsed, memory = 0, None
    for index, phase in enumerate(phases):
        if set(phase) != {"index", "begin_ns", "end_ns", "calibration_begin_ns", "state"}:
            raise ValueError("unknown/missing phase fields")
        if type(phase["index"]) is not int or phase["index"] != index:
            raise ValueError("invalid phase identity")
        if _int(phase["begin_ns"], "begin_ns") != elapsed:
            raise ValueError("phase windows must be contiguous from zero")
        end = _int(phase["end_ns"], "end_ns", elapsed + 1)
        _int(phase["calibration_begin_ns"], "calibration_begin_ns", elapsed, end - 1)
        numeric = phase["state"]
        if not isinstance(numeric, dict) or set(numeric) != set(STATE_KEYS):
            raise ValueError("state requires exactly four numeric coordinates")
        state(*(numeric[key] for key in STATE_KEYS))
        current_memory = tuple(numeric[key] for key in STATE_KEYS[2:])
        if memory is not None and current_memory != memory:
            raise ValueError("mid-trajectory allocation changes require a different protocol")
        memory, elapsed = current_memory, end
    if type(value.get("duration_ns")) is not int or value["duration_ns"] != elapsed:
        raise ValueError("schedule duration differs from its phases")
    return deepcopy(value)


def timed_segments(schedule, *, start_tick, frequency_hz):
    """Bind planned requests to QPC; no foreground time enters these boundaries."""
    validate_schedule(schedule)
    _int(start_tick, "start_tick")
    _int(frequency_hz, "frequency_hz", 1)
    segments = []
    for phase in schedule["phases"]:
        convert = lambda ns: start_tick + (ns * frequency_hz + 999999999) // 1000000000
        segments.append({"phase_index": phase["index"], "begin_tick": convert(phase["begin_ns"]),
                         "end_tick": convert(phase["end_ns"]),
                         "calibration_begin_tick": convert(phase["calibration_begin_ns"]),
                         "state": deepcopy(phase["state"])})
    return segments
