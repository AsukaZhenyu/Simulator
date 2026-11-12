"""Keep the device in one state for the whole of a measurement campaign.

A runtime-window price is fitted in one device state and is only valid in that
state: the same graph takes 4.1 ms at 8001 MHz of memory clock and 31.5 ms at
810 MHz on this host, and the band a machine sits in can change between two days
with no code change at all. A campaign that starts on a cold machine, drifts ten
degrees, or changes clocks half way through is measuring several machines and
reporting one number.

This module is the three parts a driver needs to keep that state fixed:

- a reading, taken in-process through the NVML provider that the state
  calibration protocol already uses (`profiling/state_calibration/telemetry.py`),
  so a sample costs microseconds and spawns no process;
- a plateau predicate, so "warm" is a measured condition rather than a sleep;
- a declared band with an explicit verdict, so a cohort taken outside the band
  is refused by name instead of being averaged into the others.

Nothing here changes a device setting unless `lock` is asked for, and a lock is
never reported as held without a read back that shows it.
"""
from __future__ import annotations

import json
import math
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

# The fields a band may name.  Everything else a provider reports is recorded
# but not gated: a band that silently forgot a field would read as satisfied.
FIELDS = ("temperature_c", "sm_clock_mhz", "memory_clock_mhz", "power_w",
          "gpu_util_percent", "memory_used_bytes")


class StateUnavailable(RuntimeError):
    """This host cannot be asked for its device state."""


class Provider:
    """The narrow interface this module needs: one read, one metadata line.

    The NVML provider is opened lazily so that importing this module works on a
    machine with no driver, which is how the tests run and how a dry run on a
    new host stays possible.
    """

    def __init__(self, read: Callable[[], dict[str, Any]], metadata: dict[str, Any] | None = None,
                 close: Callable[[], None] | None = None) -> None:
        self.read = read
        self.metadata = dict(metadata or {})
        self.close = close or (lambda: None)

    @classmethod
    def nvml(cls, device_index: int = 0) -> "Provider":
        from profiling.state_calibration.telemetry import NVMLProvider
        try:
            provider = NVMLProvider(device_index=device_index)
        except (OSError, RuntimeError, ValueError) as error:
            raise StateUnavailable("NVML is not readable on this host: %s" % error) from error
        return cls(provider.read, provider.metadata(), provider.close)


def sample(provider: Provider) -> dict[str, Any]:
    """One reading, with the fields the band gates and the ones it records.

    A field the driver does not report is absent, never zero: a zero clock would
    satisfy every lower bound while meaning "not asked".
    """
    reading = provider.read()
    # The NVML provider reports `values` beside what it could not read, so a
    # field NVML refused is absent rather than zero; a flat provider is accepted
    # so the tests need no provider of their own.
    values = reading.get("values", reading) if isinstance(reading, dict) else reading
    out = {name: values[name] for name in FIELDS if values.get(name) is not None}
    unknown = reading.get("unknown") if isinstance(reading, dict) else None
    if unknown:
        out["unreadable"] = sorted(name for name in unknown if name in FIELDS)
    out["taken_monotonic_ns"] = time.monotonic_ns()
    return out


@dataclass(frozen=True)
class Band:
    """The state a cohort is allowed to be measured in, as closed intervals."""

    intervals: dict[str, tuple[float, float]]

    @classmethod
    def parse(cls, declared: dict[str, Any]) -> "Band":
        intervals = {}
        for name, bounds in declared.items():
            if name not in FIELDS:
                raise ValueError("band names a field no reading provides: " + name)
            low, high = (float(value) for value in bounds)
            if not low < high:
                raise ValueError("band %s is not an interval: %r" % (name, bounds))
            intervals[name] = (low, high)
        if not intervals:
            raise ValueError("an empty band admits every state")
        return cls(intervals)

    def check(self, reading: dict[str, Any]) -> tuple[bool, str]:
        """Whether one reading is inside the band, and the first field that is not."""
        for name, (low, high) in self.intervals.items():
            value = reading.get(name)
            if value is None:
                return False, "%s was not reported" % name
            if not (low <= value <= high):
                return False, "%s is %s, outside [%g, %g]" % (name, _fmt(value), low, high)
        return True, ""


def _fmt(value: Any) -> str:
    return ("%.3g" % value) if isinstance(value, float) else str(value)


def block_means(rows: list[dict[str, Any]], field: str, blocks: int = 3) -> list[float]:
    """The mean of one field over equal spans of time across the rows.

    A mean is the quantity a price is sensitive to; the extremes of a bursty
    workload are not.  The spans are equal in time rather than equal in count,
    so a gap in the trace cannot give one block more weight than another.
    """
    if blocks < 1:
        raise ValueError("blocks must be at least one")
    values = [(row["taken_monotonic_ns"], row[field]) for row in rows if field in row]
    if not values:
        return []
    low, high = values[0][0], values[-1][0]
    if high <= low:
        return [sum(value for _, value in values) / len(values)]
    width = (high - low) / blocks
    out = []
    for index in range(blocks):
        start = low + index * width
        end = low + (index + 1) * width
        inside = [value for stamp, value in values
                  if start <= stamp < end or (index == blocks - 1 and stamp == high)]
        if inside:
            out.append(sum(inside) / len(inside))
    return out


@dataclass(frozen=True)
class Plateau:
    """What "the machine has settled" means, in numbers fixed before the run.

    Revision 3 gates drift, not jitter.  The campaign a machine is measured on
    is bursty by construction: the accelerator works for a few seconds and
    idles for longer while the host does its own part, so it heats and cools
    by several degrees between bursts.  On the desktop, over the last minute
    of a settled campaign the temperature still spans 4 C, and over the whole
    campaign the means of three equal blocks span 1.1 C: the extremes describe
    the workload, the means describe the machine.  A price is a mean, so the
    predicate is a bound on the spread of the block means over the window.

    The clock bound is separate and optional.  With clocks unlocked the SM
    clock follows the kernel mix the campaign happens to schedule (180 to
    2685 MHz on the same machine within one campaign), so a bound on it
    measures the workload rather than the machine's state.  A profile that
    does not lock clocks declares `clock_fraction: null`: the clock is then
    recorded and reported, not gated.  Where the profile does lock, the lock's
    read back fixes the clock and the bound still applies.
    """

    window_s: float | None = 60.0
    temperature_c: float = 1.5
    clock_fraction: float | None = 0.02
    blocks: int = 3

    def verdict(self, readings: Iterable[dict[str, Any]]) -> tuple[bool, str]:
        rows = list(readings)
        if not rows:
            return False, "no readings"
        span_s = (rows[-1]["taken_monotonic_ns"] - rows[0]["taken_monotonic_ns"]) / 1e9
        if self.window_s is None:
            # The whole discarded campaign is the analysed span: the workload is
            # the campaign's own, and its length differs by column, so a fixed
            # trailing window would judge different fractions of it per machine.
            window, where = rows, "the whole campaign"
        else:
            if span_s < self.window_s:
                return False, "only %.1f s of readings, window is %.1f s" % (span_s, self.window_s)
            cutoff = rows[-1]["taken_monotonic_ns"] - int(self.window_s * 1e9)
            window, where = [row for row in rows if row["taken_monotonic_ns"] >= cutoff], \
                "the last %.0f s" % self.window_s
        means = block_means(window, "temperature_c", self.blocks)
        if not means:
            return False, "the window lacks temperature readings"
        if len(means) < 2:
            return False, "the window covers a single block, so no drift can be judged"
        drift = max(means) - min(means)
        if drift > self.temperature_c:
            return False, "mean temperature drifted %.2f C across %d blocks of %s" % (
                drift, len(means), where)
        clocks = [row["sm_clock_mhz"] for row in window if "sm_clock_mhz" in row]
        if self.clock_fraction is None:
            return True, ("mean temperature drifted %.2f C across %d blocks of %s; "
                          "SM clock recorded, not gated (no clock lock declared)" % (
                              drift, len(means), where))
        if not clocks:
            return False, "the window lacks clock readings"
        low, high = min(clocks), max(clocks)
        spread = (high - low) / high if high else math.inf
        if spread > self.clock_fraction:
            return False, "SM clock moved %.1f%% in %s (%d-%d MHz)" % (
                spread * 100, where, low, high)
        return True, "mean temperature drifted %.2f C across %d blocks of %s; SM clock %.1f%%" % (
            drift, len(means), where, spread * 100)


class Sampler:
    """Read the device on a cadence while something else runs, on a thread.

    The reader is in-process NVML, so the sampler's own cost is microseconds of
    driver call per interval and it does not add a process to the machine it is
    describing.  Samples are taken with a monotonic stamp; the cadence is a
    lower bound, never an interpolation.
    """

    def __init__(self, provider: Provider, interval_s: float = 2.0) -> None:
        if interval_s <= 0:
            raise ValueError("sampling interval must be positive")
        self.provider = provider
        self.interval_s = float(interval_s)
        self.samples: list[dict[str, Any]] = []
        self.problems: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(sample(self.provider))
            except (OSError, RuntimeError, ValueError) as error:
                # One unreadable sample does not end the campaign, but it is
                # recorded: a gap in the trace is not the same as a flat one.
                self.problems.append(str(error))
            self._stop.wait(self.interval_s)

    def start(self) -> "Sampler":
        self._thread = threading.Thread(target=self._loop, name="device-state-sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(10.0, self.interval_s * 2))
        self._thread = None
        return self.samples

    def __enter__(self) -> "Sampler":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()


@dataclass
class Record:
    """One cohort's state evidence: the samples and what they imply."""

    samples: list[dict[str, Any]] = field(default_factory=list)
    out_of_band: list[str] = field(default_factory=list)

    def review(self, band: Band | None, stability: dict[str, float] | None = None,
               blocks: int = 3) -> tuple[bool, str]:
        """Whether the cohort is admissible, and the first reading or drift that is not.

        Two different questions, both declared in advance.  The band is where the
        device was: every reading has to be inside it.  Stability is whether it
        stayed there: a cohort whose mean moved has measured two machines, and
        the average over it describes neither.  The drift is judged on block
        means, not on the extremes, because the campaign's own bursts move the
        extremes on a machine whose mean never moves.
        """
        if not self.samples:
            return False, "no device-state readings"
        self.out_of_band = []
        if band is not None:
            for reading in self.samples:
                ok, reason = band.check(reading)
                if not ok:
                    self.out_of_band.append(reason)
            if self.out_of_band:
                first = self.out_of_band[0]
                return False, "%d of %d readings outside the band; first: %s" % (
                    len(self.out_of_band), len(self.samples), first)
        if stability:
            for name, limit in stability.items():
                # A field read as an interval is judged in its own units; a field
                # read as a fraction is judged against its median, because clocks
                # legitimately differ between machines and a fixed MHz swing says
                # nothing about whether *this* machine moved.
                if name.endswith("_fraction"):
                    field = name[:-len("_fraction")]
                    values = [row[field] for row in self.samples if field in row]
                    if not values:
                        return False, "no %s readings to judge stability" % field
                    span = max(values) - min(values)
                    scale = abs(statistics.median(values)) or 1.0
                    if span / scale > limit:
                        return False, "%s moved %.1f%% within the cohort (limit %.1f%%)" % (
                            field, 100 * span / scale, 100 * limit)
                    continue
                # A field read in its own units is judged the same way as the
                # precondition is (revision 3): the block means of the cohort,
                # because a bursty workload moves the extremes without moving
                # the state the cohort was measured in.
                means = block_means(self.samples, name, blocks)
                if not means:
                    return False, "no %s readings to judge stability" % name
                if len(means) < 2:
                    return False, "the cohort covers a single block, so no drift can be judged"
                drift = max(means) - min(means)
                if drift > limit:
                    return False, "%s drifted %s across %d blocks within the cohort (limit %s)" % (
                        name, _fmt(drift), len(means), _fmt(limit))
        inside = ("inside the band" if band is not None else "no band declared")
        return True, "%d readings %s and stable" % (len(self.samples), inside)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"samples": len(self.samples), "out_of_band": len(self.out_of_band)}
        for name in ("temperature_c", "sm_clock_mhz", "memory_clock_mhz", "power_w"):
            values = [row[name] for row in self.samples if name in row]
            if values:
                out[name] = {"min": min(values), "max": max(values),
                             "median": statistics.median(values)}
        return out


def lock_clocks(sm_mhz: float, memory_mhz: float | None, hours: float = 2.0,
                tool: str | None = None) -> dict[str, Any]:
    """Pin the clocks, then read them back; report the read back, not the command.

    NVIDIA's tool wants administrator rights on Windows and refuses on most
    laptop parts; both are recorded as refusals with the tool's own words.  This
    never raises for a refused lock: a machine that cannot lock clocks can still
    be measured, it just has to be gated by its band instead.
    """
    from pathlib import Path

    from profiling import host
    executable = tool or host.nvidia_smi_path()
    if executable is None:
        return {"status": "unavailable", "reason": "nvidia-smi is not on this host"}
    if not Path(executable).is_file():
        return {"status": "refused",
                "reason": "nvidia-smi is not a file: %s" % executable}
    commands = [["--lock-gpu-clocks=%d,%d" % (sm_mhz, sm_mhz)]]
    if memory_mhz is not None:
        commands.append(["--lock-memory-clocks=%d,%d" % (memory_mhz, memory_mhz)])
    out = {"requested": {"sm_clock_mhz": sm_mhz, "memory_clock_mhz": memory_mhz},
           "commands": [], "status": "attempted"}
    for arguments in commands:
        try:
            done = subprocess.run([executable, *arguments], stdin=subprocess.DEVNULL,
                                  capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as error:
            out.update(status="refused", reason=str(error))
            return out
        out["commands"].append({"arguments": arguments, "returncode": done.returncode,
                                "stdout": done.stdout.strip()[:400],
                                "stderr": done.stderr.strip()[:400]})
        if done.returncode != 0:
            out.update(status="refused",
                       reason=(done.stderr or done.stdout).strip()[:200] or "rc=%d" % done.returncode)
            return out
    out["status"] = "locked"
    return out


def verify_lock(provider: Provider, requested: dict[str, Any], tolerance_fraction: float = 0.01,
                attempts: int = 5, wait_s: float = 0.5) -> dict[str, Any]:
    """Read back the clocks the lock asked for, after the driver has applied it.

    The NVML clock reading lags a settings change, so a single immediate read
    can show the old clock and turn a working lock into a false refusal.
    """
    wanted = {name: value for name, value in requested.items()
              if name in ("sm_clock_mhz", "memory_clock_mhz") and value is not None}
    observed: dict[str, Any] = {}
    for _ in range(max(1, attempts)):
        reading = sample(provider)
        observed = {name: reading.get(name) for name in wanted}
        if wanted and all(
                observed[name] is not None and abs(observed[name] - value) <= tolerance_fraction * value
                for name, value in wanted.items()):
            return {"status": "verified", "requested": wanted, "observed": observed}
        time.sleep(wait_s)
    # Half a lock is not a lock: one clock pinned and the other free is a state
    # that no band describes, so the verdict is all-or-nothing and the fields
    # that did match are reported for the operator, not counted.
    matched = sorted(name for name, value in wanted.items()
                     if observed.get(name) is not None
                     and abs(observed[name] - value) <= tolerance_fraction * value)
    return {"status": "not_verified", "requested": wanted, "observed": observed,
            "matched": matched, "tolerance_fraction": tolerance_fraction}


def host_matches(declared: dict[str, Any], platform_name: str,
                 device_name: str | None) -> tuple[bool, str]:
    """Whether this machine is the one a profile was written for.

    A profile carries paths for one machine, and a `~` expands to whatever home
    directory is running it: the wrong machine would otherwise resolve every
    path successfully and measure a device the profile does not describe.
    """
    wanted = declared.get("platform")
    if wanted and wanted != platform_name:
        return False, "the profile is for %s and this host is %s" % (wanted, platform_name)
    wanted_name = declared.get("device_name_contains")
    if wanted_name:
        if not device_name:
            return False, "the profile names device %r and this host has no device name" % wanted_name
        if wanted_name.lower() not in device_name.lower():
            return False, "the profile names device %r and this host has %r" % (
                wanted_name, device_name)
    return True, ""


def write_jsonl(path: Any, rows: Iterable[dict[str, Any]]) -> None:
    from pathlib import Path
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
