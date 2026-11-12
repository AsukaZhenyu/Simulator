#!/usr/bin/env python
"""Run the frozen campaign N times on this machine, in one declared device state.

A campaign is a timing measurement, and a timing measurement belongs to the
device state that produced it. This driver makes that state an input it controls
rather than a property it discovers afterwards:

  1. precondition -- run the campaign once, discarded, while reading the device,
     until the temperature and the SM clock have stopped moving;
  2. lock (optional) -- pin the clocks, then read them back; a lock that is not
     verified is reported as refused, never as held;
  3. repeat -- run the frozen campaign N times into fresh roots, reading the
     device throughout, and refuse any cohort whose readings left the band;
  4. aggregate -- the ensemble number the device protocol defines: the mean of
     the cohort means, its between-run standard deviation, the per-run pass
     count, and the state occupancy of the per-request first forward.

Each machine supplies a profile (adapter, not method): interpreter, campaign
arguments, band, plateau, repetitions. The procedure below is the same on every
machine, and `--dry-run` resolves and prints it without measuring anything, so a
new host can check its profile before it spends an hour.

    python tools/run_device_ensemble.py --profile profiling/real_moe_validation/devices/laptop-4070.json --dry-run
    python tools/run_device_ensemble.py --profile .../laptop-4070.json

The driver never edits the campaign and never refits anything; it decides which
cohorts may be counted and how they are combined.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profiling import host
from profiling.real_moe_validation import device_state_control as state
from profiling.real_moe_validation import ensemble

PROFILE_SCHEMA = "device-ensemble-profile-1"
ENSEMBLE_SCHEMA = "device-ensemble-1"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class Refused(SystemExit):
    """A refusal, with the sentence the operator needs to read."""


def log(message: str) -> None:
    print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} {message}", flush=True)


def resolve(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else ROOT / candidate


def unresolved_paths(profile: dict) -> list[str]:
    """Profile values a machine has not filled in yet.

    A template that ships with a plausible-looking default measures the wrong
    file silently; `REPLACE_ME` is refused by name instead.
    """
    found = []

    def walk(value: Any, where: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{where}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{where}[{index}]")
        elif isinstance(value, str) and "REPLACE_ME" in value:
            found.append(where)

    walk(profile, "profile")
    return found


def load_profile(path: Path) -> dict:
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile.get("schema") != PROFILE_SCHEMA:
        raise Refused(f"{path} is not a {PROFILE_SCHEMA} profile")
    for key in ("device", "campaign", "ensemble"):
        if key not in profile:
            raise Refused(f"{path} has no {key!r} block")
    campaign = profile["campaign"]
    for key in ("python", "runner", "root_pattern", "arguments"):
        if key not in campaign:
            raise Refused(f"{path} campaign block has no {key!r}")
    unfilled = unresolved_paths(profile)
    if unfilled:
        raise Refused(f"{path} still carries placeholders: " + ", ".join(unfilled))
    return profile


def campaign_command(profile: dict, root: Path) -> list[str]:
    """The frozen campaign, with this machine's profile substituted in.

    The arguments are expanded here rather than by a shell, because the list is
    handed to `subprocess` whole: a `~` that reached the campaign would be a
    directory named "~", and a relative path would resolve against whichever
    directory the driver happened to be started from.
    """
    campaign = profile["campaign"]
    return [str(resolve(campaign["python"])), str(resolve(campaign["runner"])),
            "--root", str(root),
            *[os.path.expanduser(str(item)) for item in campaign["arguments"]]]


# ------------------------------------------------------------------ phases

def precondition(profile: dict, provider: state.Provider | None, root: Path) -> dict:
    """Run one discarded campaign and wait for the device to stop moving.

    The workload is the campaign itself, so the machine is warmed by exactly the
    work it will be measured on; a synthetic burn would settle a different
    thermal path.
    """
    declared = profile.get("state", {}).get("precondition", {})
    if not declared.get("enabled", True):
        return {"status": "skipped", "reason": "the profile disabled preconditioning"}
    plateau = state.Plateau(**{key: declared["plateau"][key] for key in
                               ("window_s", "temperature_c", "clock_fraction", "blocks")
                               if key in declared.get("plateau", {})})
    interval = float(declared.get("sample_interval_s", 2.0))
    target = root / "precondition"
    command = campaign_command(profile, target)
    clock = ("gated at %.0f%%" % (100 * plateau.clock_fraction)
             if plateau.clock_fraction is not None else "recorded, not gated")
    window = ("the whole campaign" if plateau.window_s is None
              else "the last %g s" % plateau.window_s)
    log(f"precondition: {target.name}, sampling every {interval:g} s, "
        f"plateau over {window} in {plateau.blocks} blocks, SM clock {clock}")
    record = state.Record()
    started = time.monotonic()
    if provider is not None:
        sampler = state.Sampler(provider, interval).start()
        try:
            done = run_campaign(command, target)
        finally:
            record.samples = sampler.stop()
        state.write_jsonl(target / "device-state.jsonl", record.samples)
    else:
        done = run_campaign(command, target)
    elapsed = time.monotonic() - started
    if done.returncode != 0:
        return {"status": "refused", "reason": f"the campaign exited {done.returncode}",
                "root": str(target), "seconds": round(elapsed, 1)}
    if provider is None:
        return {"status": "unverified", "reason": "no device state to read",
                "root": str(target), "seconds": round(elapsed, 1)}
    settled, why = plateau.verdict(record.samples)
    return {"status": "settled" if settled else "not_settled", "reason": why,
            "root": str(target), "seconds": round(elapsed, 1),
            "readings": len(record.samples)}


def run_campaign(command: list[str], root: Path) -> subprocess.CompletedProcess:
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "campaign.log"
    with log_path.open("w", encoding="utf-8") as stream:
        return subprocess.run(command, stdin=subprocess.DEVNULL, stdout=stream,
                              stderr=subprocess.STDOUT, cwd=str(ROOT))


def one_repetition(profile: dict, provider: state.Provider | None, root: Path,
                   band: state.Band | None, interval: float,
                   stability: dict[str, float] | None = None) -> dict:
    command = campaign_command(profile, root)
    record = state.Record()
    started = time.monotonic()
    if provider is not None:
        sampler = state.Sampler(provider, interval).start()
        try:
            done = run_campaign(command, root)
        finally:
            record.samples = sampler.stop()
        state.write_jsonl(root / "device-state.jsonl", record.samples)
    else:
        done = run_campaign(command, root)
    entry: dict = {"root": str(root), "seconds": round(time.monotonic() - started, 1),
                   "returncode": done.returncode, "state": record.summary()}
    if done.returncode != 0:
        entry.update(status="refused", reason=f"the campaign exited {done.returncode}")
        return entry
    blocks = int(profile.get("state", {}).get("precondition", {})
                 .get("plateau", {}).get("blocks", 3))
    in_band, why = record.review(band, stability, blocks)
    entry["state"].update(in_band=in_band, reason=why)
    if not in_band:
        entry.update(status="excluded", reason="device state left the band: " + why)
        return entry
    try:
        evaluation = ensemble.evaluation(root)
    except ensemble.Unreadable as error:
        entry.update(status="excluded", reason=str(error))
        return entry
    if evaluation["complete_endpoint_conditions"] < 12:
        entry.update(status="excluded",
                     reason="only %d of 12 conditions completed; the protocol excludes it"
                            % evaluation["complete_endpoint_conditions"])
        return entry
    entry.update(status="counted", mean_APE=evaluation["mean_APE"],
                 worst_APE=evaluation["worst_APE"],
                 passed=evaluation["passed_conditions"],
                 conditions=evaluation["complete_endpoint_conditions"],
                 phase_APEs=[{"placement": condition.get("placement"),
                              "request_id": condition.get("request_id"),
                              "APE": {name: value for name, value in condition["APE"].items()
                                      if name in ("total", "prefill", "decode_with_sampling")},
                              "passed": condition.get("passed")}
                             for condition in evaluation["conditions"]],
                 first_forward_ms=ensemble.first_forward_ms(root))
    return entry


def markdown(profile: dict, result: dict) -> str:
    if "ensemble" not in result:
        lines = [f"# Device ensemble: {profile['device']}", "",
                 f"Created {result['created_at_utc']}. **Not an ensemble number:** "
                 f"{result.get('incomplete', 'the run stopped early')}.", "",
                 "| cohort | status | mean total APE | worst | conditions passed |",
                 "|---|---|---|---|---|"]
        for entry in result["cohorts"]:
            mean = (f"{ensemble.total_ape(entry['mean_APE']) * 100:.2f}%"
                    if "mean_APE" in entry else "-")
            lines.append(f"| {Path(entry['root']).name} | {entry['status']} | {mean} | - | - |")
        return "\n".join(lines) + "\n"
    combined = result["ensemble"]
    lines = [
        f"# Device ensemble: {profile['device']}",
        "",
        f"Created {result['created_at_utc']}. Cohorts counted: {combined['cohorts']} "
        f"of {combined['attempts']} attempted.",
        "",
        "| cohort | status | mean total APE | worst | conditions passed |",
        "|---|---|---|---|---|",
    ]
    for entry in result["cohorts"]:
        name = Path(entry["root"]).name
        mean = (f"{ensemble.total_ape(entry['mean_APE']) * 100:.2f}%"
                if "mean_APE" in entry else "-")
        worst = (f"{ensemble.total_ape(entry['worst_APE']) * 100:.2f}%"
                 if "worst_APE" in entry else "-")
        passed = f"{entry['passed']}/{entry['conditions']}" if "passed" in entry else "-"
        lines.append(f"| {name} | {entry['status']} | {mean} | {worst} | {passed} |")
    lines += [
        "",
        f"**Ensemble mean {combined['mean_of_means'] * 100:.2f}% "
        f"+- {combined['sd_of_means'] * 100:.2f}% (between-run sd), "
        f"range {combined['min_of_means'] * 100:.2f}-{combined['max_of_means'] * 100:.2f}%.**",
    ]
    if "phase_APE" in combined:
        phase = combined["phase_APE"]
        lines.append(f"Per-condition phase APEs pooled over the cohorts: n={phase['n']}, "
                     f"mean {phase['mean'] * 100:.2f}%, median {phase['median'] * 100:.2f}%, "
                     f"p90 {phase['p90'] * 100:.2f}%, max {phase['max'] * 100:.2f}%.")
    if "state" in combined:
        where = combined["state"]
        lines.append(f"First-forward state: {where['slow']}/{where['requests']} bare requests "
                     f"above {where['boundary_ms']:.1f} ms "
                     f"({(where['fraction'] or 0) * 100:.0f}%); boundary from {where['source']}.")
    if result.get("lock"):
        lines.append(f"Clock lock: {result['lock'].get('status')} {result['lock'].get('reason', '')}".rstrip())
    if result.get("precondition"):
        lines.append(f"Preconditioning: {result['precondition'].get('status')} "
                     f"({result['precondition'].get('reason', '')}).")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None,
                        help="ensemble directory; default build/ensemble-<device>-<date>")
    parser.add_argument("--repetitions", type=int, default=None,
                        help="override the profile's repetitions")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve and print the plan; measure nothing")
    parser.add_argument("--skip-precondition", action="store_true")
    parser.add_argument("--precondition-only", action="store_true",
                        help="measure one discarded campaign and report the state trace; "
                             "this is how a new machine sets its band and tolerances")
    parser.add_argument("--attempt-limit", type=int, default=None,
                        help="how many cohorts may be run to obtain --repetitions counted ones")
    arguments = parser.parse_args(argv)

    profile = load_profile(arguments.profile)
    device = profile["device"]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = resolve(arguments.root or Path("build") / f"ensemble-{device}-{stamp}")
    repetitions = arguments.repetitions or int(profile["ensemble"].get("repetitions", 10))
    attempts_allowed = arguments.attempt_limit or int(
        profile["ensemble"].get("attempt_limit", repetitions + 3))
    state_block = profile.get("state", {})
    band = state.Band.parse(state_block["band"]) if state_block.get("band") else None
    stability = state_block.get("stability") or None
    interval = float(state_block.get("sample_interval_s", 5.0))

    log(f"device {device} -> {root}")
    log(f"{repetitions} repetitions, at most {attempts_allowed} cohorts, "
        f"band {'declared' if band else 'not declared'}")

    platform_name = "windows" if sys.platform.startswith("win") else (
        "linux" if sys.platform.startswith("linux") else sys.platform)
    provider = None
    if state_block.get("required", True):
        try:
            provider = state.Provider.nvml(int(state_block.get("device_index", 0)))
        except state.StateUnavailable as error:
            raise Refused(f"this profile requires device state and the host has none: {error}")
        metadata = provider.metadata
        log("device state: " + json.dumps({k: metadata.get(k) for k in
                                           ("device_name", "device_uuid") if k in metadata}))

    matched, why = state.host_matches(profile.get("host", {}), platform_name,
                                      (provider.metadata.get("device_name") if provider else None))
    if not matched:
        raise Refused("this is not the machine the profile describes: " + why)

    if arguments.dry_run:
        command = campaign_command(profile, root / "rep01")
        log("dry run: the campaign command is " + " ".join(command))
        log(f"dry run: the ensemble directory would be {root}")
        return 0

    root.mkdir(parents=True, exist_ok=True)
    result: dict = {"schema": ENSEMBLE_SCHEMA, "device": device, "created_at_utc": utc(),
                    "profile": str(arguments.profile),
                    "profile_sha256": hashlib.sha256(arguments.profile.read_bytes()).hexdigest(),
                    "repetitions": repetitions, "cohorts": []}

    if arguments.skip_precondition:
        result["precondition"] = {"status": "skipped", "reason": "asked on the command line"}
    else:
        result["precondition"] = precondition(profile, provider, root)
        log(f"precondition: {result['precondition']['status']} -- "
            f"{result['precondition'].get('reason', '')}")
        if arguments.precondition_only:
            write_result(root, profile, result, incomplete="preconditioning only")
            log("precondition-only: the trace is in " + str(Path(result["precondition"]["root"]) / "device-state.jsonl"))
            return 0
        if result["precondition"]["status"] not in ("settled", "unverified"):
            write_result(root, profile, result, incomplete="not preconditioned")
            raise Refused("the device did not settle before the measured repetitions: "
                          + result["precondition"].get("reason", ""))

    declared_lock = state_block.get("lock", {})
    if declared_lock.get("enabled"):
        lock = state.lock_clocks(float(declared_lock["sm_clock_mhz"]),
                                 declared_lock.get("memory_clock_mhz"))
        if lock["status"] == "locked" and provider is not None:
            lock["verification"] = state.verify_lock(provider, lock["requested"])
            if lock["verification"]["status"] != "verified":
                lock["status"] = "unverified"
        result["lock"] = lock
        log(f"clock lock: {lock['status']} {lock.get('reason', '')}".rstrip())
        if lock["status"] != "locked" and declared_lock.get("required"):
            write_result(root, profile, result, incomplete="clock lock refused")
            raise Refused("the profile requires a clock lock and this host refused it")

    counted, attempts = 0, 0
    while counted < repetitions and attempts < attempts_allowed:
        attempts += 1
        cohort_root = resolve(str(profile["campaign"]["root_pattern"]).format(
            device=device, date=datetime.now().strftime("%Y%m%d"), nn=f"{attempts:02d}"))
        log(f"cohort {attempts}: {cohort_root.name}")
        entry = one_repetition(profile, provider, cohort_root, band, interval, stability)
        result["cohorts"].append(entry)
        if entry["status"] == "counted":
            counted += 1
            log(f"    counted ({counted}/{repetitions}): mean total APE "
                f"{ensemble.total_ape(entry['mean_APE']) * 100:.2f}%, "
                f"worst {ensemble.total_ape(entry['worst_APE']) * 100:.2f}%, "
                f"{entry['passed']}/{entry['conditions']} conditions pass")
        else:
            log(f"    {entry['status']}: {entry.get('reason', '')}")

    if counted < repetitions:
        write_result(root, profile, result,
                     incomplete=f"{counted} of {repetitions} cohorts counted")
        raise Refused(f"only {counted} of {repetitions} cohorts could be counted")

    result["ensemble"] = ensemble.aggregate(
        result["cohorts"], attempts,
        profile.get("ensemble", {}).get("state_boundary_ms"))
    write_result(root, profile, result)
    combined = result["ensemble"]
    log(f"ensemble: {combined['mean_of_means'] * 100:.2f}% +- "
        f"{combined['sd_of_means'] * 100:.2f}% over {combined['cohorts']} cohorts")
    return 0


def write_result(root: Path, profile: dict, result: dict, incomplete: str | None = None) -> None:
    if incomplete:
        result = dict(result, incomplete=incomplete)
    (root / "ENSEMBLE.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    (root / "ENSEMBLE.md").write_text(markdown(profile, result), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
