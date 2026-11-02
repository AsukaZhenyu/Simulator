"""Read-only, source-bound thread/state and action-boundary diagnosis.

All time distributions are observations. Thread states are sampled counts, not
durations; raw cycles are never converted to CPU execution time. This module
does not estimate a latency model, choose a policy, or predict heldout data.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics

SCHEMA = "owned-thread-action-diagnosis-1"
FOCUS_ACTIONS = (0, 14, 39, 85)
AGE_BOUNDARIES_NS = (0, 50_000_000, 150_000_000, 500_000_000, 1_000_000_000, 2_000_000_000)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * p
    left = int(position)
    right = min(left + 1, len(values) - 1)
    return values[left] + (values[right] - values[left]) * (position - left)


def distribution(values):
    values = list(values)
    mean = statistics.mean(values) if values else None
    return dict(count=len(values), min=min(values, default=None), p10=percentile(values, .1),
        p50=percentile(values, .5), p90=percentile(values, .9), p99=percentile(values, .99),
        max=max(values, default=None), mean=mean,
        population_cv=None if mean in (None, 0) or len(values) < 2 else statistics.pstdev(values) / abs(mean))


class SourceAudit:
    def __init__(self):
        self.receipts = {}

    def read(self, path, expected=None):
        path = Path(path).resolve()
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        require(expected is None or sha == expected, "consumed raw/source bytes changed: " + str(path))
        self.receipts[str(path)] = dict(sha256=sha, bytes=len(raw))
        return raw

    def json(self, path, expected=None):
        return json.loads(self.read(path, expected))

    def bound_json(self, path, sources):
        path = str(Path(path).resolve())
        require(path in sources, "consumed input lacks a recorded source SHA: " + path)
        return self.json(path, sources[path])


def age_bin(value_ns):
    require(value_ns >= 0, "negative phase age")
    index = len(AGE_BOUNDARIES_NS) - 1
    for i, edge in enumerate(AGE_BOUNDARIES_NS[1:]):
        if value_ns < edge:
            index = i
            break
    return index


def interval_stratum(go, begin, end):
    """Classify entire intervals; never squeeze crossings into a steady bin."""
    require(type(begin) is int and type(end) is int and end >= begin, "invalid interval")
    phases, frequency = go["phases"], go["clock"]["frequency_hz"]
    matches = [(i, p) for i, p in enumerate(phases)
               if p["begin_tick"] <= begin < p["end_tick"] and end <= p["end_tick"]]
    if not matches:
        return dict(category="crossing_or_outside", phase_index=None, age_bin=None)
    index, phase = matches[0]
    state = [phase["cpu_replay"], phase["gpu_replay"]]
    first = Fraction((begin - phase["begin_tick"]) * 1_000_000_000, frequency)
    last = Fraction((end - phase["begin_tick"]) * 1_000_000_000, frequency)
    first_bin, last_bin = age_bin(first), age_bin(last)
    # A right endpoint at an age boundary belongs to the preceding interval.
    if end > begin and last in AGE_BOUNDARIES_NS[1:]:
        last_bin -= 1
    same_bin = first_bin == last_bin
    return dict(category="contained_phase", phase_index=phase["phase_index"], request_state=state,
        predecessor_request_state=None if index == 0 else [phases[index - 1]["cpu_replay"], phases[index - 1]["gpu_replay"]],
        phase_kind="initial_00" if state == [0, 0] and index == 0 else "post_load_00" if state == [0, 0] else "requested_load",
        age_origin="declared_phase_entry_not_physical_quiescence",
        age_bin=first_bin if same_bin else "crosses_age_boundary",
        age_bin_ns=None if not same_bin else [AGE_BOUNDARIES_NS[first_bin],
            AGE_BOUNDARIES_NS[first_bin + 1] if first_bin + 1 < len(AGE_BOUNDARIES_NS) else None])


def group_key(value):
    return canonical(value).decode("utf-8")


class ThreadCounts:
    """Counts only: no interpolation, interval weights or implied percentages."""
    def __init__(self):
        self.snapshots = 0
        self.seen_snapshots, self.seen_actor_snapshots = set(), set()
        self.actors = defaultdict(lambda: dict(snapshot_count=0, missing_main_thread_snapshots=0,
            unknown_actor_snapshots=0, unknown_thread_entries=0,
            roles={role: dict(state=Counter(), wait=Counter(), state_codes=Counter(), wait_codes=Counter(), unknown=0, entries=0)
                   for role in ("main", "auxiliary", "unidentified")}))

    def add(self, snapshot, identities):
        for actor in snapshot["actors"]:
            for thread in actor["threads"]:
                self.add_state(dict(snapshot=snapshot, actor=actor, thread=thread), identities)

    def add_state(self, observation, identities):
        snapshot, actor, thread = (observation[key] for key in ("snapshot", "actor", "thread"))
        identity = identities[actor["actor_id"]]
        require(actor["process_id"] == identity["process_id"], "thread actor PID differs")
        index = snapshot["sample_index"]
        if index not in self.seen_snapshots:
            self.seen_snapshots.add(index)
            self.snapshots += 1
        row = self.actors[actor["actor_id"]]
        if (actor["actor_id"], index) not in self.seen_actor_snapshots:
            self.seen_actor_snapshots.add((actor["actor_id"], index))
            row["snapshot_count"] += 1
            row["unknown_actor_snapshots"] += bool(actor.get("unknown"))
            tid = identity.get("main_thread_id")
            if tid is not None and not any(t["thread_id"] == tid for t in actor["threads"]):
                row["missing_main_thread_snapshots"] += 1
        tid = identity.get("main_thread_id")
        role = "unidentified" if tid is None else "main" if thread["thread_id"] == tid else "auxiliary"
        counts = row["roles"][role]
        counts["entries"] += 1
        counts["state"][thread.get("state_name") or "unknown"] += 1
        provider = thread.get("thread_state_provider", "unspecified_recorded_provider")
        counts["state_codes"][group_key(dict(namespace=provider, code=thread.get("state_code"), name=thread.get("state_name")))] += 1
        if thread.get("state_name") == "Wait":
            counts["wait"][thread.get("wait_reason_name") or "unknown"] += 1
            native = thread.get("native_wait_reason_code")
            reason = (dict(namespace=provider + ":native_wait_reason", code=native, name=thread.get("native_wait_reason_name"))
                      if native is not None else dict(namespace=provider + ":managed_or_unspecified_wait_reason",
                        code=thread.get("wait_reason_code"), name=thread.get("wait_reason_name")))
            counts["wait_codes"][group_key(reason)] += 1
        if thread.get("unknown") or thread.get("state_name") is None:
            counts["unknown"] += 1
            row["unknown_thread_entries"] += 1

    def report(self):
        return dict(outer_snapshots=self.snapshots, actors={key: dict(
            snapshot_count=value["snapshot_count"], unknown_actor_snapshots=value["unknown_actor_snapshots"],
            missing_main_thread_snapshots=value["missing_main_thread_snapshots"],
            unknown_thread_entries=value["unknown_thread_entries"],
            roles={role: dict(thread_entries=counts["entries"], state_counts=dict(counts["state"]),
                wait_reason_counts=dict(counts["wait"]), unknown_entries=counts["unknown"])
                for role, counts in value["roles"].items()}) for key, value in self.actors.items()},
            state_code_observations={actor: {role: [dict(json.loads(key), count=count) for key, count in counts["state_codes"].items()]
                for role, counts in values["roles"].items()} for actor, values in self.actors.items()},
            wait_code_observations={actor: {role: [dict(json.loads(key), count=count) for key, count in counts["wait_codes"].items()]
                for role, counts in values["roles"].items()} for actor, values in self.actors.items()},
            counts_are_durations=False, counts_are_time_fractions=False,
            runtime_spin_or_useful_work_identified=False,
            wait_reason_identifies_specific_runtime_function=False)


class SnapshotIndex:
    def __init__(self, document, identities, clock):
        self.identities = identities
        if document is None:
            self.rows, self.starts, self.states, self.state_starts = [], [], [], []
            return
        require(document.get("schema") == "owned-thread-snapshots-1", "unsupported thread snapshot schema")
        require(document.get("clock") == clock, "native/thread clock differs")
        require(document.get("exact_thread_state_durations_claimed") is False and document.get("used_as_predictor") is False,
                "thread observation scope changed")
        self.rows = document["records"]
        self.starts = []
        states = []
        previous = None
        for index, row in enumerate(self.rows):
            require(row["sample_index"] == index and row["query_begin_tick"] <= row["query_end_tick"], "invalid snapshot order/span")
            require(previous is None or previous <= row["query_begin_tick"], "overlapping/backwards outer snapshots")
            previous = row["query_end_tick"]
            self.starts.append(row["query_begin_tick"])
            actors = {a["actor_id"]: a for a in row["actors"]}
            require(len(actors) == len(row["actors"]) and set(actors) == set(identities), "snapshot actor inventory differs")
            for actor_id, actor in actors.items():
                expected = identities[actor_id]
                require(actor["process_id"] == expected["process_id"] and
                    actor["creation_time_100ns"] == expected["creation_time_100ns"], "snapshot process lifetime differs")
                require(row["query_begin_tick"] <= actor["query_begin_tick"] <= actor["query_end_tick"] <= row["query_end_tick"],
                        "actor query escaped full snapshot envelope")
                for thread in actor["threads"]:
                    require(thread.get("handle_owner_pid") in (None, expected["process_id"]), "thread handle owner differs")
                    require(actor["query_begin_tick"] <= thread["state_snapshot_begin_tick"] <=
                        thread["state_snapshot_end_tick"] <= actor["query_end_tick"], "thread state query escaped actor envelope")
                    require(actor["query_begin_tick"] <= thread["query_begin_tick"] <=
                        thread["query_end_tick"] <= actor["query_end_tick"], "thread counter query escaped actor envelope")
                    states.append(dict(snapshot=row, actor=actor, thread=thread,
                        begin_tick=thread["state_snapshot_begin_tick"], end_tick=thread["state_snapshot_end_tick"]))
        self.states = sorted(states, key=lambda item: item["begin_tick"])
        self.state_starts = [item["begin_tick"] for item in self.states]

    def contained(self, begin, end):
        start = bisect_left(self.starts, begin)
        for row in self.rows[start:]:
            if row["query_begin_tick"] > end:
                break
            if row["query_end_tick"] <= end:
                yield row

    def contained_states(self, begin, end):
        start = bisect_left(self.state_starts, begin)
        for row in self.states[start:]:
            if row["begin_tick"] > end:
                break
            if row["end_tick"] <= end:
                yield row


class ObservationGroup:
    def __init__(self, stratum):
        self.stratum = stratum
        self.values = defaultdict(list)
        self.missing = Counter()
        self.thread_counts = ThreadCounts()

    def value(self, key, value):
        if value is None:
            self.missing[key] += 1
        else:
            self.values[key].append(value)

    def report(self):
        return dict(stratum=self.stratum, observations={key: distribution(values) for key, values in self.values.items()},
            missing_observations=dict(self.missing), thread_snapshots=self.thread_counts.report())


def add_cpu_bracket(group, bracket, *, wall_ns):
    for name in ("process_cpu_ns", "main_thread_cpu_ns", "main_thread_cycles"):
        group.value(name, bracket.get(name))
    process, main, cycles = (bracket.get(name) for name in ("process_cpu_ns", "main_thread_cpu_ns", "main_thread_cycles"))
    group.value("process_minus_main_counter_ns", None if process is None or main is None else process - main)
    group.value("raw_main_cycles_per_wall_ns", None if cycles is None or not wall_ns else cycles / wall_ns)


def summarize_job(job, program, native, go, thread_document, identities):
    clock = {key: native["clock"][key] for key in ("clock_id", "frequency_hz")}
    require(clock == {key: go["clock"][key] for key in clock}, "native/GO clock differs")
    require(native["go_schedule"] == go and native["actions"] == program["actions"], "native schedule/action contract differs")
    require(identities["main"]["process_id"] == native["process_id"], "native main PID differs from identity")
    require(identities["main"]["main_thread_id"] == native["main_thread_id"], "native main TID differs from identity")
    snapshots = SnapshotIndex(thread_document, identities, clock)
    actions = program["actions"]
    groups = {}
    action_sample_counts, action_call_counts, sampled_action_calls = Counter(), Counter(), Counter()
    legacy_outer_counts, state_entry_counts = Counter(), Counter()
    def obtain(kind, stratum, action=None):
        metadata = dict(stratum, scope=kind)
        if action is not None:
            metadata["action_scope"] = action
        key = group_key(metadata)
        if key not in groups:
            groups[key] = ObservationGroup(metadata)
        return groups[key]
    intervals = [row for block in native["blocks"] for row in block["action_intervals"]]
    cpu_rows = [row for block in native["blocks"] for row in block.get("thread_action_diagnostics", [])]
    cpu_index = {(row["iteration"], row["action_index"]): row for row in cpu_rows}
    require(len(cpu_index) == len(cpu_rows), "duplicate per-action CPU diagnostic")
    invocations = [row for block in native["blocks"] for row in block["iterations"]]
    invocation_queries, invocation_action_sum = defaultdict(int), defaultdict(int)
    for row in intervals:
        index = row["action_index"]
        action = actions[index]
        stratum = interval_stratum(go, row["begin_tick"], row["end_tick"])
        names = (["index:" + str(index)] if index in FOCUS_ACTIONS else [])
        if action["kind"] in ("COMPUTE", "COPY"):
            names.append("all_" + action["kind"].lower())
        item = cpu_index.get((row["iteration"], index))
        if item is not None:
            require(item["action_begin_tick"] == row["begin_tick"] and item["action_end_tick"] == row["end_tick"] and
                item["action_elapsed_ns"] == row["elapsed_ns"], "CPU bracket/action stamp differs")
            invocation_queries[row["iteration"]] += item["before_query_ns"] + item["after_query_ns"]
        invocation_action_sum[row["iteration"]] += row["elapsed_ns"]
        selected = [point for point in snapshots.contained_states(row["begin_tick"], row["end_tick"])
                    if interval_stratum(go, point["begin_tick"], point["end_tick"])["category"] == "contained_phase"]
        action_sample_counts[index] += len({point["snapshot"]["sample_index"] for point in selected})
        state_entry_counts[index] += len(selected)
        legacy_outer_counts[index] += sum(1 for _ in snapshots.contained(row["begin_tick"], row["end_tick"]))
        action_call_counts[index] += 1
        sampled_action_calls[index] += bool(selected)
        for name in names:
            group = obtain("action", stratum, name)
            group.value("action_wall_ns", row["elapsed_ns"])
            group.value("contained_state_entries_per_action", len(selected))
            if item is not None:
                for name2 in ("before_query_ns", "after_query_ns", "outer_elapsed_ns"):
                    group.value(name2, item[name2])
                add_cpu_bracket(group, item["cpu_counter_bracket"], wall_ns=row["elapsed_ns"])
            for point in selected:
                point_stratum = interval_stratum(go, point["begin_tick"], point["end_tick"])
                state_group = obtain("action_thread_state", point_stratum, name)
                state_group.thread_counts.add_state(point, identities)
    for row in invocations:
        group = obtain("invocation", interval_stratum(go, row["begin_tick"], row["end_tick"]))
        group.value("invocation_wall_ns", row["elapsed_ns"])
        if cpu_rows:
            group.value("sum_counter_query_wall_ns", invocation_queries[row["iteration"]])
        if intervals:
            group.value("sum_action_wall_ns", invocation_action_sum[row["iteration"]])
            group.value("invocation_minus_sum_action_wall_ns", row["elapsed_ns"] - invocation_action_sum[row["iteration"]])
    for point in snapshots.states:
        stratum = interval_stratum(go, point["begin_tick"], point["end_tick"])
        group = obtain("thread_state_snapshot", stratum)
        if stratum["category"] == "contained_phase":
            group.thread_counts.add_state(point, identities)
        else:
            group.value("unassigned_crossing_or_outside_state_entries", 1)
        group.value("state_query_bracket_ns", float(Fraction((point["end_tick"] - point["begin_tick"]) *
            1_000_000_000, clock["frequency_hz"])))
    previous_cycles, cycle_problems = {}, Counter()
    for snapshot in snapshots.rows:
        current_keys = set()
        for actor in snapshot["actors"]:
            identity = identities[actor["actor_id"]]
            for thread in actor["threads"]:
                tid, created, cycles = thread["thread_id"], thread.get("thread_creation_time_100ns"), thread.get("cycle_total")
                if created is None or cycles is None:
                    cycle_problems["unknown_lifetime_or_cycles"] += 1
                    continue
                key = (actor["actor_id"], tid, created)
                current_keys.add(key)
                previous = previous_cycles.get(key)
                if previous is not None and previous[0]["sample_index"] == snapshot["sample_index"] - 1:
                    old_snapshot, old_thread = previous
                    if cycles < old_thread["cycle_total"]:
                        cycle_problems["counter_moved_backwards"] += 1
                    else:
                        begin, end = old_snapshot["query_begin_tick"], snapshot["query_end_tick"]
                        stratum = interval_stratum(go, begin, end)
                        role = "unidentified" if identity.get("main_thread_id") is None else "main" if tid == identity["main_thread_id"] else "auxiliary"
                        cycle_group = obtain("thread_cycle_window", stratum,
                            dict(actor_id=actor["actor_id"], thread_role=role, thread_id=tid, creation_time_100ns=created))
                        delta = cycles - old_thread["cycle_total"]
                        wall_ns = Fraction((end - begin) * 1_000_000_000, clock["frequency_hz"])
                        cycle_group.value("raw_thread_cycle_increment", delta)
                        cycle_group.value("outer_cycle_window_wall_ns", float(wall_ns))
                        cycle_group.value("raw_cycles_per_wall_ns", float(delta / wall_ns) if wall_ns else None)
                        old_cpu = (old_thread.get("kernel_time_100ns"), old_thread.get("user_time_100ns"))
                        new_cpu = (thread.get("kernel_time_100ns"), thread.get("user_time_100ns"))
                        cpu_delta = None
                        if all(value is not None for value in old_cpu + new_cpu):
                            if all(new >= old for old, new in zip(old_cpu, new_cpu)):
                                cpu_delta = (sum(new_cpu) - sum(old_cpu)) * 100
                            else:
                                cycle_problems["CPU_counter_moved_backwards"] += 1
                        cycle_group.value("quantized_thread_CPU_counter_increment_ns", cpu_delta)
                elif previous is not None:
                    cycle_problems["nonconsecutive_thread_observation"] += 1
                previous_cycles[key] = (snapshot, thread)
        previous_cycles = {key: value for key, value in previous_cycles.items() if key in current_keys}
    gpu_count = 0
    for block in native["blocks"]:
        for row in block.get("target_diagnostics", []):
            gpu_count += 1
            group = obtain("gpu_target", interval_stratum(go, row["begin_tick"], row["end_tick"]), "index:" + str(row["action_index"]))
            for key in ("host_elapsed_ns", "host_enqueue_ns", "host_sync1_ns", "host_sync2_ns", "stream_span_ns"):
                group.value(key, row.get(key))
            add_cpu_bracket(group, row["cpu_total"], wall_ns=row["host_elapsed_ns"])
            for name, cpu_key in (("enqueue", "cpu_enqueue_bracket"), ("sync1", "cpu_sync1_bracket"), ("sync2", "cpu_sync2_bracket")):
                begin, end = row["host_" + name + "_begin_tick"], row["host_" + name + "_end_tick"]
                part = obtain("gpu_" + name, interval_stratum(go, begin, end), "index:" + str(row["action_index"]))
                wall = row["host_" + name + "_ns"]
                part.value("host_phase_wall_ns", wall)
                add_cpu_bracket(part, row[cpu_key], wall_ns=wall)
                for point in snapshots.contained_states(begin, end):
                    point_stratum = interval_stratum(go, point["begin_tick"], point["end_tick"])
                    if point_stratum["category"] == "contained_phase":
                        state_group = obtain("gpu_" + name + "_thread_state", point_stratum, "index:" + str(row["action_index"]))
                        state_group.thread_counts.add_state(point, identities)
    cadence = [float(Fraction((right["query_begin_tick"] - left["query_begin_tick"]) * 1_000_000_000, clock["frequency_hz"]))
               for left, right in zip(snapshots.rows, snapshots.rows[1:])]
    return dict(job_id=job["id"], condition=job["condition"], repeat=job["repeat"], process_id=native["process_id"],
        cpu_threads=program["physical"]["threads"], instrumentation=native["instrumentation"],
        owned_roles=sorted(identities), main_thread_id=native["main_thread_id"],
        invocation_count=len(invocations), action_interval_count=len(intervals), cpu_bracket_count=len(cpu_rows), gpu_diagnostic_count=gpu_count,
        overall_invocation_wall_ns=distribution(row["elapsed_ns"] for row in invocations),
        overall_sum_counter_query_wall_ns=distribution(invocation_queries[row["iteration"]] for row in invocations) if cpu_rows else None,
        outer_snapshot_query_wall_ns=distribution(float(Fraction((row["query_end_tick"] - row["query_begin_tick"]) *
            1_000_000_000, clock["frequency_hz"])) for row in snapshots.rows),
        actual_snapshot_start_cadence_ns=distribution(cadence), snapshot_count=len(snapshots.rows),
        thread_observer_status="declared_absent" if thread_document is None else "observed",
        sampler_missed_periods=None if thread_document is None else thread_document["missed_periods"],
        state_query_bracket_ns=distribution(float(Fraction((point["end_tick"] - point["begin_tick"]) *
            1_000_000_000, clock["frequency_hz"])) for point in snapshots.states),
        thread_cycle_window_problems=dict(cycle_problems),
        focus_action_metadata={str(i): actions[i] for i in FOCUS_ACTIONS if i < len(actions)},
        focus_contained_snapshot_counts={str(i): action_sample_counts[i] for i in FOCUS_ACTIONS},
        focus_contained_state_entries={str(i): state_entry_counts[i] for i in FOCUS_ACTIONS},
        focus_legacy_whole_outer_snapshot_counts={str(i): legacy_outer_counts[i] for i in FOCUS_ACTIONS},
        focus_action_calls={str(i): dict(observed=action_call_counts[i], with_contained_snapshot=sampled_action_calls[i])
            for i in FOCUS_ACTIONS},
        groups=[groups[key].report() for key in sorted(groups)])


def analyze(campaign):
    root, audit = Path(campaign).resolve(), SourceAudit()
    manifest = audit.json(root / "manifest.json")
    require(manifest["protocol"].get("diagnostic_only") is True, "diagnostic campaign required")
    rows = audit.json(root / "diagnostic-rows.json")
    results, failures = [], []
    for row in rows:
        job = row["job"]
        attempt_id = row.get("attempt_id", job["id"])
        directory = root / "jobs" / attempt_id
        require(directory.resolve().is_relative_to(root), "job path escapes campaign")
        execution = audit.json(directory / "execution.json")
        require(canonical(execution) == canonical(row), "execution differs from campaign inventory")
        if row["status"] != "measured":
            failures.append(dict(job_id=job["id"], attempt_id=attempt_id, execution=execution))
            continue
        normalized = audit.json(row["normalized_path"], row["normalized_sha256"])
        sources = normalized["source_files"]
        def bound(path):
            return audit.bound_json(path, sources)
        native, go = (bound(directory / name) for name in ("main/result.json", "go.json"))
        if job.get("thread_observer", True):
            thread = bound(directory / "thread-activity.json")
            config = bound(directory / "thread-sampler/config.json")
            identities = {a["actor_id"]: dict(a) for a in config["actors"]}
        else:
            require(not (directory / "thread-activity.json").exists(), "declared absent thread observer produced a trace")
            thread = None
            identities = {role: dict(actor_id=role, creation_time_100ns=None) for role in job["owned_roles"]}
        binding = manifest["programs"][job["id"]]
        program = bound(binding["path"])
        require(audit.receipts[str(Path(binding["path"]).resolve())]["sha256"] == binding["sha256"], "program binding differs")
        for actor, identity in identities.items():
            ready = bound(directory / actor / "ready.json")
            if "process_id" in identity:
                require(ready["process_id"] == identity["process_id"], "actor READY identity differs")
            identity["process_id"] = ready["process_id"]
            identity["main_thread_id"] = ready.get("main_thread_id")
        numerical = bound(directory / "numerical.json")
        require(numerical.get("passed") is True, "independent numerical check failed")
        results.append(summarize_job(job, program, native, go, thread, identities))
    return dict(schema=SCHEMA, campaign=str(root), runs=results, failed_attempts=failures,
        consumed_sources=audit.receipts,
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        focus_actions=list(FOCUS_ACTIONS), phase_age_boundaries_ns=list(AGE_BOUNDARIES_NS),
        interval_join_scope="individual thread state-snapshot query bracket contained by action and phase; separate later cycle/counter queries; no midpoint assignment",
        snapshots_are_exact_state_durations=False, cycles_converted_to_time=False,
        CPU_accounting_is_exact_per_action_time=False, GPU_stream_span_is_kernel_time=False,
        host_CPU_GPU_intervals_are_additive=False,
        actual_CPU_service_inferred_from_request_bits=False, latency_fitted_or_predicted=False,
        timing_scope="diagnostic modes include observer/counter overhead; not a formal cost table",
        source_verification_scope="every consumed raw input checked against the normalized recorded SHA; inventory files receipted as consumed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze(args.campaign)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
