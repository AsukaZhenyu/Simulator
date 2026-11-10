"""Read the machine state beside a run and declare it per forward.

The device performance state is an input of a runtime-window price, not a
property of the request: the same graph takes 4.1 ms at 8001 MHz of memory clock
and 31.5 ms at 810 MHz on the same host.  This module turns an external clock
sample series into one declared state per forward, using the interval each
forward actually occupied.

The report lags the state.  A clock read at time t describes the device at
t - lag, so the state of a forward is the clock read `LAG_NS` after its midpoint.
The lag was measured at 0.166 s on this device; the constant is fixed here and is
not adjusted to any timing result.
"""
from __future__ import annotations

from bisect import bisect_left

# Declared states, ordered from the fastest to the slowest memory clock.  The
# bounds are open above the last one and are applied to the reported memory
# clock in MHz; a reading the tiers do not cover is an error, never a guess.
STATES = ("nominal", "p4", "p5")
BOUNDS = {"nominal": (7800.0, None), "p4": (5500.0, 7800.0), "p5": (0.0, 1500.0)}
LAG_NS = 150_000_000


def tier(memory_mhz):
    """The declared state of one memory-clock reading."""
    if not isinstance(memory_mhz, int) or memory_mhz <= 0:
        raise ValueError("positive observed memory clock required")
    for state in STATES:
        low, high = BOUNDS[state]
        if memory_mhz > low and (high is None or memory_mhz < high):
            return state
    raise ValueError("memory clock outside every declared state: %d MHz" % memory_mhz)


class ClockSeries:
    """External `(qpc, memory MHz, sm MHz)` samples of one job, in time order."""

    def __init__(self, samples, lag_ns=LAG_NS):
        if not samples:
            raise ValueError("no usable clock sample")
        self.ticks = [int(qpc) for qpc, _, _ in samples]
        self.memory = [int(memory) for _, memory, _ in samples]
        self.sm = [int(sm) for _, _, sm in samples]
        if self.ticks != sorted(self.ticks):
            raise ValueError("clock samples are not in time order")
        self.lag_s = lag_ns / 1_000_000_000

    def at(self, qpc):
        """The reading in force at `qpc`: the last sample not after it."""
        index = bisect_left(self.ticks, int(qpc))
        if index == 0:
            return self.memory[0], self.sm[0]
        return self.memory[index - 1], self.sm[index - 1]

    def state(self, begin_tick, end_tick, frequency_hz):
        """The declared state while the interval was executing."""
        if end_tick < begin_tick:
            raise ValueError("interval ends before it begins")
        midpoint_s = (int(begin_tick) + int(end_tick)) / 2 / frequency_hz
        memory, _ = self.at(int(round((midpoint_s + self.lag_s) * frequency_hz)))
        return tier(memory)


def load_clocks(path):
    """Read a `qpc,mem_mhz,sm_mhz` sample file."""
    samples = []
    with open(path, encoding="utf-8") as stream:
        header = stream.readline().strip()
        if header != "qpc,mem_mhz,sm_mhz":
            raise ValueError("unexpected clock sample header: " + header)
        for line in stream:
            if not line.strip():
                continue
            qpc, memory, sm = (int(value) for value in line.split(","))
            if memory > 0:
                samples.append((qpc, memory, sm))
    return ClockSeries(samples)


def forward_states(native, clocks):
    """Declare the state of every forward of every request in one native result.

    Keyed by (request id, phase, forward index), which is what a runtime window
    carries.  A bare result carries the same forward intervals as an observed
    one, so a state can be declared beside an unobserved run.
    """
    frequency = native["clock"]["frequency_hz"]
    declared = {}
    for row in native["results"]:
        for forward in row["forward_inventory"]:
            key = (row["id"], forward["phase"], int(forward["index"]))
            declared[key] = clocks.state(forward["begin_tick"], forward["end_tick"], frequency)
    return declared


def state_census(states):
    """Count forwards per state, for a campaign's instrument check."""
    counts = {state: 0 for state in STATES}
    for state in states.values():
        counts[state] += 1
    return counts
