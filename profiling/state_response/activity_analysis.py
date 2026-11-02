"""Read-only profile diagnostics of owned-process CPU activity and runtimes.

Request bits are commands, not actual CPU intensity. Positive post-call process
CPU increments can show that call completion did not end all runtime activity.
Zero increments never prove quiescence. No timing predictor or fit lives here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import ntpath
from pathlib import Path
import statistics

from profiling.resource_calibration import counters

ACTORS = ("cpu-worker", "gpu-worker", "main")


def _require(value, message):
    if not value:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _encoded(value):
    value = Fraction(value)
    return dict(numerator=value.numerator, denominator=value.denominator)


def _quantile(values, p):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * p
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def _distribution(values):
    values = list(values)
    return dict(count=len(values), min=min(values, default=None), p10=_quantile(values, .1),
        p50=_quantile(values, .5), p90=_quantile(values, .9), p99=_quantile(values, .99),
        max=max(values, default=None), mean=None if not values else statistics.mean(values))


class SourceAudit:
    """One analysis transaction's current-byte receipts; no persistent cache."""
    def __init__(self):
        self.receipts = {}

    def read(self, path, expected=None):
        path = Path(path).resolve()
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if expected is not None and actual != expected:
            raise ValueError("consumed profile raw bytes changed: " + str(path))
        self.receipts[str(path)] = dict(sha256=actual, bytes=len(raw))
        return raw

    def json(self, path, expected=None):
        return json.loads(self.read(path, expected))


def cpu_windows(activity, *, actor_id, expected_pid, clock):
    """Recompute cached deltas; keep true query envelopes and unresolved reads."""
    _require(activity.get("used_as_predictor") is False, "activity file claims predictor use")
    _require(activity.get("actors", {}).get(actor_id, {}).get("process_id") == expected_pid,
             "owned actor PID receipt differs")
    metadata = activity["metadata"]
    for key in ("clock_id", "frequency_hz"):
        _require(metadata["clock"][key] == clock[key], "activity clock differs")
    rows, unknown, previous, baselines = [], [], None, 0
    for sample in activity["records"]:
        raw = sample.get("snapshots", {}).get(actor_id)
        index = sample["sample_index"]
        if not isinstance(raw, dict) or raw.get("schema") != counters.SCHEMA:
            unknown.append(dict(sample_index=index, reason="counter snapshot missing or provider error"))
            previous = None
            continue
        if previous is None:
            previous = raw
            baselines += 1
            continue
        delta = counters.delta(previous, raw)
        cached = sample.get("activities", {}).get(actor_id)
        if cached is not None:
            _require(_canonical(delta) == _canonical(cached), "cached CPU delta does not reconstruct")
        lp, rp = previous["process"], raw["process"]
        points = (lp.get("cpu_counter_begin_tick"), lp.get("cpu_counter_end_tick"),
                  rp.get("cpu_counter_begin_tick"), rp.get("cpu_counter_end_tick"))
        value = delta.get("values", {}).get("cpu_time_100ns")
        if not all(type(point) is int for point in points) or not points[0] <= points[1] <= points[2] <= points[3] or value is None:
            unknown.append(dict(sample_index=index, reason="CPU query span/counter delta unresolved", delta=delta))
        else:
            rational = delta["exact_core_equivalents"]
            rows.append(dict(sample_index=index, outer_begin_tick=points[0], outer_end_tick=points[3],
                left_query_begin_tick=points[0], left_query_end_tick=points[1],
                right_query_begin_tick=points[2], right_query_end_tick=points[3],
                cpu_delta_ns=value * 100, core_equivalents=float(Fraction(rational["numerator"], rational["denominator"])),
                exact_core_equivalents=rational,
                nominal_endpoint_elapsed_ns=float(Fraction((points[3] - points[1]) * 1_000_000_000, clock["frequency_hz"])),
                min_possible_elapsed_ticks=points[2] - points[1], max_possible_elapsed_ticks=points[3] - points[0],
                missed_periods=sample.get("missed_periods"), lateness_ticks=sample.get("lateness_ticks")))
        previous = raw
    return rows, dict(counter_observations=len(activity["records"]), valid_windows=len(rows),
        baseline_only_snapshots=baselines,
        unresolved_windows=len(unknown), unresolved_details=unknown,
        nominal_interval_ms=metadata.get("interval_ms"), actual_cadence_guaranteed=False,
        sampler_thread_error=metadata.get("thread_error"),
        missed_periods_sum=sum(sample.get("missed_periods", 0) for sample in activity["records"]),
        clock_tick_ns=_encoded(Fraction(1_000_000_000, clock["frequency_hz"])),
        endpoint_elapsed_ns=_distribution(row["nominal_endpoint_elapsed_ns"] for row in rows),
        CPU_query_duration_ticks=_distribution(value for row in rows for value in (
            row["left_query_end_tick"] - row["left_query_begin_tick"],
            row["right_query_end_tick"] - row["right_query_begin_tick"])),
        smallest_observed_positive_CPU_counter_increment_ns=min((row["cpu_delta_ns"] for row in rows if row["cpu_delta_ns"] > 0), default=None),
        CPU_accounting_resolution_claimed=False)


def contained(windows, begin, end):
    return [row for row in windows if begin <= row["outer_begin_tick"] and row["outer_end_tick"] <= end]


def summarize(windows):
    windows = list(windows)
    positive = [row for row in windows if row["cpu_delta_ns"] > 0]
    sequences, sequence = [], []
    for row in windows:
        if row["cpu_delta_ns"] > 0:
            if sequence and row["sample_index"] != sequence[-1]["sample_index"] + 1:
                sequences.append(sequence)
                sequence = []
            sequence.append(row)
        elif sequence:
            sequences.append(sequence)
            sequence = []
    if sequence:
        sequences.append(sequence)
    return dict(window_count=len(windows), positive_increment_windows=len(positive),
        zero_increment_windows=len(windows) - len(positive),
        core_equivalents=_distribution(row["core_equivalents"] for row in windows),
        CPU_increment_ns=_distribution(row["cpu_delta_ns"] for row in windows),
        longest_consecutive_positive_counter_windows=max((len(part) for part in sequences), default=0),
        positive_sequences=[dict(windows=len(part), outer_begin_tick=part[0]["outer_begin_tick"],
            outer_end_tick=part[-1]["outer_end_tick"], observed_CPU_increment_ns=sum(row["cpu_delta_ns"] for row in part)) for part in sequences],
        last_positive_outer_end_tick=max((row["outer_end_tick"] for row in positive), default=None),
        zero_increment_proves_quiescence=False, consecutive_positive_increments_prove_continuous_CPU_busy=False)


def cpu_off_tails(go, worker, windows_by_actor, *, support_end_tick):
    """Only after both the off request and last old-request call completion."""
    intervals = [row for block in worker["blocks"] for row in block["iterations"]]
    changes, previous = [], 0
    for phase in go["phases"]:
        current = phase["cpu_replay"]
        if current != previous:
            changes.append((phase["begin_tick"], current))
        previous = current
    if previous:
        changes.append((go["end_tick"], 0))
    result, last_on = [], None
    for index, (edge, state) in enumerate(changes):
        if state:
            last_on = edge
            continue
        next_on = next((tick for tick, value in changes[index + 1:] if value), None)
        end = support_end_tick if next_on is None else min(next_on, support_end_tick)
        old = [row for row in intervals if last_on is not None and last_on <= row["enabled_check_tick"] < edge]
        terminal = max(old, key=lambda row: row["enabled_check_tick"]) if old else None
        anchor = None if terminal is None else max(edge, terminal["end_tick"])
        row = dict(off_request_tick=edge, next_on_request_tick=next_on, support_end_tick=end,
            last_old_request_call=terminal, after_request_and_call_begin_tick=anchor,
            reason=None if anchor is not None and anchor < end else "no complete old-call proof or no observed off-tail interval",
            process_quiescence_inferred=False)
        row["actors"] = {actor: summarize(contained(windows, anchor, end)) if anchor is not None and anchor < end else summarize([])
                         for actor, windows in windows_by_actor.items()}
        result.append(row)
    return result


def runtime_observations(paths, disk_files, expected_pids, environment):
    disk = {str(row["path"]).casefold(): row for row in disk_files.get("modules", [])}
    actors = {}
    for actor in ACTORS:
        observed = paths.get(actor, {})
        identity = observed.get("process_identity", {})
        binding = identity.get("status") == "verified" and identity.get("observed_pid") == expected_pids[actor]
        modules = []
        for row in observed.get("modules", []):
            path = row.get("path")
            if path and ntpath.basename(path).lower().startswith("vcomp"):
                receipt = disk.get(str(path).casefold())
                modules.append(dict(path=path, native_loaded_path_observed=bool(binding and observed.get("observed_loaded_path_proof")),
                    disk_receipt=receipt, mapped_image_bytes_verified=False))
        actors[actor] = dict(handle_PID_binding_verified=binding, module_snapshot_status=observed.get("status", "unknown"),
            before_after_inventory_equal=observed.get("same_module_inventory_before_after"),
            VCOMP_modules=modules, VCOMP_status="observed" if modules else "not observed in this snapshot; absence is not lifetime proof")
    return dict(actors=actors, launch_environment=environment,
        launch_mapping_recorded=isinstance(environment, dict) and environment.get("status") == "recorded",
        remote_child_environment_read=False, runtime_default_values_inferred=False,
        child_runtime_environment_interpretation_verified=False)


def vcomp_consistency(runs):
    """Unknown receipts can never establish full native/runtime consistency."""
    signatures, unknown, missing, observed = [], [], 0, 0
    def version_known(value):
        return isinstance(value, list) and len(value) == 4 and all(type(part) is int and 0 <= part <= 65535 for part in value)
    for run in runs:
        for actor_id in ACTORS:
            actor = run.get("runtime", {}).get("actors", {}).get(actor_id, {})
            modules = actor.get("VCOMP_modules", [])
            if not modules:
                missing += 1
                unknown.append(dict(run_id=run.get("run_id"), actor_id=actor_id,
                                    reasons=["VCOMP module path not observed for expected actor"]))
                continue
            for index, module in enumerate(modules):
                observed += 1
                issues = []
                if actor.get("handle_PID_binding_verified") is not True or module.get("native_loaded_path_observed") is not True:
                    issues.append("native handle/PID/loaded-path observation not verified")
                path = module.get("path")
                if not isinstance(path, str) or not path or not (ntpath.isabs(path) or Path(path).is_absolute()):
                    issues.append("absolute observed module path unknown")
                receipt = module.get("disk_receipt")
                if not isinstance(receipt, dict):
                    receipt = {}
                    issues.append("bound disk receipt missing")
                if not isinstance(path, str) or str(receipt.get("path", "")).casefold() != path.casefold():
                    issues.append("disk receipt path does not match observed native path")
                value = receipt.get("on_disk_sha256")
                if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                    issues.append("disk SHA-256 unknown or invalid")
                version = receipt.get("file_version")
                details = version.get("value") if isinstance(version, dict) else None
                if not isinstance(version, dict) or version.get("status") != "observed" or not isinstance(details, dict) or not (
                    version_known(details.get("file_version")) and version_known(details.get("product_version"))):
                    issues.append("observed numeric file/product version unknown or invalid")
                if issues:
                    unknown.append(dict(run_id=run.get("run_id"), actor_id=actor_id, module_index=index,
                                        path=path, reasons=issues))
                else:
                    signatures.append(_canonical(dict(path=path.casefold(), sha256=value,
                        file_version=details["file_version"], product_version=details["product_version"])).decode())
    subset_equal = bool(signatures) and len(set(signatures)) == 1
    return dict(rule_version="complete-native-and-known-disk-receipts-2",
        expected_actor_entries=len(runs) * len(ACTORS), observed_actor_entries=observed,
        actors_without_VCOMP_observation=missing, known_complete_entries=len(signatures),
        unknown_entries=len(unknown), unknown_details=unknown,
        distinct_path_hash_version_receipts=[json.loads(value) for value in sorted(set(signatures))],
        known_subset_receipts_identical=subset_equal,
        known_receipts_identical=bool(subset_equal and not unknown),
        known_consistency_requires="every expected actor has verified native PID/path plus matching known disk SHA and observed numeric file/product versions",
        None_equality_is_known_consistency=False, lifetime_or_mapped_image_identity_proven=False)


def analyze_run(activity, events, workers, paths, disk_files, *, context, run_id):
    go, clock = events["go"], events["go"]["clock"]
    expected = {actor: workers[actor]["process_id"] for actor in ACTORS}
    windows_by_actor, actors = {}, {}
    for actor in ACTORS:
        _require(workers[actor].get("go_schedule") == go and workers[actor].get("role") == actor, "worker/main GO or role differs")
        windows, coverage = cpu_windows(activity, actor_id=actor, expected_pid=expected[actor], clock=clock)
        windows_by_actor[actor] = windows
        states, phases, assigned = defaultdict(list), [], set()
        for phase in go["phases"]:
            selected = contained(windows, phase["begin_tick"], phase["end_tick"])
            state = dict(cpu_replay=phase["cpu_replay"], gpu_replay=phase["gpu_replay"],
                host_memory_bytes=go["memory"]["dram_hold_bytes"], device_memory_bytes=go["memory"]["vram_hold_bytes"])
            states[_canonical(state).decode()].extend(selected)
            assigned.update(row["sample_index"] for row in selected)
            phases.append(dict(phase_index=phase["phase_index"], state=state, begin_tick=phase["begin_tick"],
                               end_tick=phase["end_tick"], activity=summarize(selected)))
        actors[actor] = dict(process_id=expected[actor], coverage=coverage,
            native_handle_PID_binding_verified=bool(paths.get(actor, {}).get("process_identity", {}).get("status") == "verified" and
                paths.get(actor, {}).get("process_identity", {}).get("observed_pid") == expected[actor]),
            pre_GO_idle=summarize([row for row in windows if row["outer_end_tick"] <= go["start_tick"]]),
            by_requested_state=[dict(state=json.loads(state), activity=summarize(rows)) for state, rows in sorted(states.items())],
            phases=phases, crossing_or_outside_phase_windows=sum(row["sample_index"] not in assigned for row in windows),
            all_observed_windows=summarize(windows))
    support_end = max(row["end_tick"] for row in events["known_segments"])
    return dict(run_id=run_id, split="profile", actors=actors,
        CPU_off_after_last_call=cpu_off_tails(go, workers["cpu-worker"], windows_by_actor, support_end_tick=support_end),
        runtime=runtime_observations(paths, disk_files, expected, context.get("openmp_launch_environment")),
        foreground_durations_used_for_fit=False, heldout_used=False,
        observation_scope="owned-process CPU counter increments, including runtime threads/spin; not useful work or instantaneous external contention",
        containment_scope="both GetProcessTimes query spans and their full outer envelope must fit a declared interval",
        CPU_action_correlation_computed=False, process_quiescence_inferred=False)


def analyze_profile_campaign(campaign):
    root, audit = Path(campaign).resolve(), SourceAudit()
    manifest = audit.json(root / "manifest.json")
    executions = audit.json(root / "profile-processes.json")
    successful = [row for row in executions if row.get("status") == "measured"]
    expected_ids = [job["id"] for job in manifest["jobs"] if job["split"] == "profile"]
    _require([row["job"]["id"] for row in successful] == expected_ids, "profile process inventory differs or has survivors omitted")
    runs = []
    for execution in successful:
        normalized = Path(execution["normalized_path"]).resolve()
        _require(normalized.is_relative_to(root), "normalized process record escapes campaign")
        record = audit.json(normalized, execution["normalized_sha256"])
        _require(record["split"] == "profile" and record["process_id"] == execution["process_id"], "profile process identity differs")
        unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
        _require(hashlib.sha256(_canonical(unsigned)).hexdigest() == record["record_sha256"], "normalized profile digest differs")
        def raw(name):
            path = record["raw_bindings"][name]
            return audit.json(path, record["source_files"][path])
        events, activity = raw("events"), raw("worker_activity")
        workers = {"cpu-worker": raw("worker_cpu"), "gpu-worker": raw("worker_gpu"), "main": raw("native")}
        directory = normalized.parent
        runtime_paths, runtime_disk = directory / "loaded-runtime-paths.json", directory / "loaded-runtime-disk-files.json"
        paths = audit.json(runtime_paths, record["source_files"][str(runtime_paths)])
        disk = audit.json(runtime_disk, record["source_files"][str(runtime_disk)])
        # Only the actually observed VCOMP files are consumed for this narrow
        # diagnostic. Other runtime receipts remain metadata, not current-byte
        # validation of every SDK DLL or mapped image.
        for module in disk.get("modules", []):
            if ntpath.basename(module["path"]).lower().startswith("vcomp") and module.get("on_disk_sha256"):
                _require(record["source_files"].get(module["path"]) == module["on_disk_sha256"], "VCOMP disk receipt lacks matching source binding")
                audit.read(module["path"], module["on_disk_sha256"])
        runs.append(analyze_run(activity, events, workers, paths, disk, context=record["context"], run_id=record["run_id"]))
    return dict(schema="profile-owned-worker-activity-analysis-2", campaign=str(root), runs=runs,
        expected_profile_processes=len(expected_ids), retained_failed_attempts=[dict(attempt_id=row.get("attempt_id"), error=row.get("error")) for row in executions if row.get("status") != "measured"],
        VCOMP_consistency=vcomp_consistency(runs),
        launch_environment_declarations_identical=len({_canonical(run["runtime"]["launch_environment"]) for run in runs}) == 1,
        consumed_source_files=audit.receipts,
        current_consumed_raw_bytes_verified=True, non_VCOMP_SDK_disk_bytes_rehashed=False,
        analyzer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        predictions_computed=False, parameters_fitted=False, heldout_counters_used=False,
        zero_increment_proves_quiescence=False,
        limitations=["GetProcessTimes is finite-resolution accounting with non-atomic QPC queries.",
            "Positive counter windows do not identify individual spin, useful-work or CPU event times.",
            "Commands are not actual load strength; compare all process/state distributions before attributing timing drift.",
            "Launch mappings and on-disk versions do not prove runtime defaults or mapped-image bytes.",
            "No action correlation, recovery threshold or stop-response parameter is fitted."])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("analysis output must be a new file")
    result = analyze_profile_campaign(args.campaign)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(profile_processes=result["expected_profile_processes"], output=str(args.out),
                         fitted=False, heldout_used=False)))


if __name__ == "__main__":
    main()
