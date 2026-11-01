"""Source-bound statistics for model-owned state-conditioned action responses.

The measured quantity is completed host-action wall time, including the backend
work/waits inside that boundary. It is never converted to pure GPU throughput.
All predictions are delegated to ``modeling.conditioned_response``.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics

STATE_KEYS = ("cpu_replay", "gpu_replay", "host_memory_bytes", "device_memory_bytes")
RUN_SCHEMA = "state-response-run-1"
FROZEN_SCHEMA = "state-response-frozen-1"
EVALUATION_SCHEMA = "state-response-evaluation-1"
DEFAULT_GATES = {"minimum_independent_runs": 3, "within_process_cv_max": .15,
                 "between_process_cv_max": .20, "relative_error_max": .20}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, label, minimum=0):
    _require(type(value) is int and value >= minimum, f"{label} requires integer >= {minimum}")
    return value


def _hash(value):
    _require(isinstance(value, str) and len(value) == 64 and
             all(char in "0123456789abcdef" for char in value), "SHA256 required")
    return value


def _utc(value):
    _require(isinstance(value, str), "UTC timestamp required")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require(parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0,
             "UTC timestamp requires explicit zero offset")
    return parsed.astimezone(timezone.utc)


def exact(value):
    if isinstance(value, Fraction):
        return value
    if type(value) is int:
        return Fraction(value)
    _require(isinstance(value, dict) and set(value) == {"numerator", "denominator"}, "exact rational required")
    return Fraction(_integer(value["numerator"], "numerator"),
                    _integer(value["denominator"], "denominator", 1))


def encoded(value):
    if value is None:
        return None
    value = exact(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


def median(values):
    values = sorted(exact(value) for value in values)
    if not values:
        return None
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def cv(values):
    values = [float(exact(value)) for value in values]
    if not values:
        return None
    mean = statistics.mean(values)
    return 0.0 if mean == 0 else statistics.pstdev(values) / mean


def validate_state(state):
    _require(isinstance(state, dict) and set(state) == set(STATE_KEYS),
             "state requires exactly four offered-load/allocation coordinates")
    result = dict(state)
    for key in STATE_KEYS[:2]:
        # JSON booleans from the historical collector are normalized to numeric
        # commands. They remain requests, never utilization percentages.
        if type(result[key]) is bool:
            result[key] = int(result[key])
        _require(type(result[key]) is int and result[key] in (0, 1), key + " requires numeric 0/1")
    for key in STATE_KEYS[2:]:
        _integer(result[key], key)
    return result


def state_key(state):
    return canonical(validate_state(state)).decode()


def contract_identity(program):
    """Exclude numeric initialization/seed while retaining execution geometry."""
    data = deepcopy(program)
    for key in ("program_sha256", "seed", "initial_values"):
        data.pop(key, None)
    if isinstance(data.get("identity"), dict):
        data["identity"].pop("initial_values", None)
    return digest(data)


def token_inventory(program):
    actions = program.get("actions")
    _require(isinstance(actions, list) and actions, "program requires nonempty action sequence")
    result = []
    for index, action in enumerate(actions):
        _require(isinstance(action, dict) and isinstance(action.get("kind"), str), "action kind required")
        result.extend([dict(token=f"action:{index}:gap_before", part="gap", action_index=index,
                            kind="HOST_GAP", original_cv_criterion=False),
                       dict(token=f"action:{index}:duration", part="duration", action_index=index,
                            kind=action["kind"], original_cv_criterion=action["kind"] in {"COMPUTE", "COPY"})])
    result.append(dict(token="tail", part="tail", action_index=None, kind="HOST_TAIL", original_cv_criterion=False))
    return result


def _native_invocations(native):
    if isinstance(native.get("blocks"), list):
        result = []
        for bi, block in enumerate(native["blocks"]):
            iterations = block.get("iterations")
            actions = block.get("action_intervals")
            _require(isinstance(iterations, list) and isinstance(actions, list), "native block intervals missing")
            by_iteration = defaultdict(list)
            for action in actions:
                by_iteration[action["iteration"]].append(action)
            _require(set(by_iteration) <= {row["iteration"] for row in iterations}, "orphan native action interval")
            for item in iterations:
                row = deepcopy(item)
                row["observation_id"] = f"block:{bi}:iteration:{item['iteration']}"
                row["action_intervals"] = deepcopy(by_iteration[item["iteration"]])
                result.append(row)
        return result
    _require(isinstance(native.get("iterations"), list), "native iterations missing")
    return [dict(deepcopy(row), observation_id=f"iteration:{index}") for index, row in enumerate(native["iterations"])]


def _known_segments(events):
    result = []
    for event in events:
        if "steady_begin_tick" in event:
            result.append(dict(begin_tick=event["steady_begin_tick"], end_tick=event["steady_end_tick"],
                               state=validate_state(event["requested_resources"])))
        else:
            _require({"begin_tick", "end_tick", "state"} <= set(event), "known segment fields differ")
            result.append(dict(begin_tick=event["begin_tick"], end_tick=event["end_tick"],
                               state=None if event["state"] is None else validate_state(event["state"])))
    previous = None
    for row in result:
        _integer(row["begin_tick"], "segment begin")
        _integer(row["end_tick"], "segment end", row["begin_tick"] + 1)
        _require(previous is None or previous <= row["begin_tick"], "known segments overlap or are reordered")
        previous = row["end_tick"]
    return result


def _event_sets(events):
    if isinstance(events, dict):
        _require(isinstance(events.get("known_segments"), list), "events dictionary requires known_segments")
        dynamic = events["known_segments"]
        calibration = events.get("events", dynamic)
    else:
        _require(isinstance(events, list), "events require an array or known_segments dictionary")
        dynamic = calibration = events
    return _known_segments(dynamic), _known_segments(calibration)


def _state_for_interval(segments, begin, end):
    for row in segments:
        if row["begin_ns"] <= begin and ((begin == end and begin < row["end_ns"]) or
                                         (begin < end and end <= row["end_ns"])):
            return row["state"]
    return None


def _verify_sources(record):
    _require(isinstance(record.get("source_files"), dict) and record["source_files"], "source file bindings required")
    for name, expected in record["source_files"].items():
        _require(Path(name).is_absolute(), "bound source paths must be absolute")
        if file_sha(name) != _hash(expected):
            raise ValueError("bound raw source bytes changed: " + name)


def make_run_record(program, native, events, *, context, run_id, split, seed,
                    process_id, started_at_utc, ended_at_utc, source_files,
                    numerical_verification, raw_bindings=None):
    """Normalize immutable collector evidence; no scalar is fitted or predicted.

    ``events`` are known external state segments, or historical steady events.
    Initialization/teardown holes remain unknown. All times share native QPC.
    Numerical verification is an explicit collector receipt, bound with raw
    source files; this normalizer does not pretend to recompute NumPy outputs.
    """
    _require(isinstance(context, dict) and context, "exact context required")
    _require(context.get("state_semantics") == "offered_load_controlled", "external offered-load state semantics required")
    _require(split in {"profile", "heldout", "diagnostic"}, "invalid run split")
    _require(isinstance(run_id, str) and run_id, "run_id required")
    _integer(seed, "seed")
    _integer(process_id, "process ID", 1)
    _require(_utc(started_at_utc) <= _utc(ended_at_utc), "run UTC chronology reversed")
    _require(native.get("process_id") == process_id and native.get("program_sha256") == program.get("program_sha256"),
             "native process/program identity differs")
    _require(native.get("actions") == program["actions"], "native complete action inventory differs")
    _require(native.get("role", "main") == "main", "worker timings are not complete foreground replay evidence")
    _require(native.get("diagnostic_only") is not True or split == "diagnostic", "instrumented diagnosis cannot fit or accept accuracy")
    _require(program.get("seed", seed) == seed, "numeric seed differs")
    _require(isinstance(numerical_verification, dict) and numerical_verification.get("passed") is True,
             "explicit numerical verification receipt required")
    clock = native.get("clock", native.get("native_clock"))
    _require(isinstance(clock, dict) and isinstance(clock.get("clock_id"), str), "native clock required")
    frequency = _integer(clock["frequency_hz"], "clock frequency", 1)
    rows = _native_invocations(native)
    _require(rows, "native has no retained invocations")
    origin = _integer(native["go_schedule"]["start_tick"] if context.get("time_origin") == "planned_go_start_v1"
                      else rows[0]["begin_tick"], "native origin")
    convert = lambda tick: round(Fraction((tick - origin) * 1_000_000_000, frequency))
    dynamic_events, calibration_events = _event_sets(events)
    def normalized_segments(rows):
        return [dict(begin_ns=max(0, convert(row["begin_tick"])), request_begin_ns=convert(row["begin_tick"]),
                     end_ns=convert(row["end_tick"]), state=row["state"])
                for row in rows if row["end_tick"] > origin]
    segments, calibration_segments = normalized_segments(dynamic_events), normalized_segments(calibration_events)
    invocations, previous, seen = [], origin, set()
    inventory = token_inventory(program)
    bare = native.get("instrumentation") in {"bare", "per_iteration_host_completed_work_only"}
    for row in rows:
        begin, end = row["begin_tick"], row["end_tick"]
        _integer(begin, "invocation begin", previous)
        _integer(end, "invocation end", begin + 1)
        _require(abs(row["elapsed_ns"] - round(Fraction((end - begin) * 1_000_000_000, frequency))) <= 1,
                 "native invocation duration differs from its clock interval")
        _require(row["observation_id"] not in seen, "duplicate invocation ID")
        seen.add(row["observation_id"])
        intervals = row["action_intervals"]
        _require(len(intervals) == len(program["actions"]) or (bare and not intervals and split != "profile"),
                 "profile/trace requires all actions; only heldout bare replay may omit action timers")
        tokens, cursor = [], begin
        for index, item in enumerate(intervals):
            left, right = item["begin_tick"], item["end_tick"]
            _require(item.get("action_index") == index and cursor <= left <= right <= end,
                     "native action order/inventory/chronology differs")
            _require(item.get("kind", program["actions"][index]["kind"]) == program["actions"][index]["kind"],
                     "native action kind differs")
            _require(abs(item["elapsed_ns"] - round(Fraction((right - left) * 1_000_000_000, frequency))) <= 1,
                     "native action duration differs from its clock interval")
            for token, a, b in ((f"action:{index}:gap_before", cursor, left),
                                (f"action:{index}:duration", left, right)):
                a_ns, b_ns = convert(a), convert(b)
                tokens.append(dict(token=token, begin_ns=a_ns, end_ns=b_ns,
                                   elapsed_ns=b_ns - a_ns,
                                   steady_state=_state_for_interval(segments, a_ns, b_ns),
                                   calibration_state=_state_for_interval(calibration_segments, a_ns, b_ns)))
            cursor = right
        if intervals:
            left_ns, right_ns = convert(cursor), convert(end)
            tokens.append(dict(token="tail", begin_ns=left_ns, end_ns=right_ns,
                               elapsed_ns=right_ns - left_ns,
                               steady_state=_state_for_interval(segments, left_ns, right_ns),
                               calibration_state=_state_for_interval(calibration_segments, left_ns, right_ns)))
        invocations.append(dict(observation_id=row["observation_id"], begin_ns=convert(begin), end_ns=convert(end),
            elapsed_ns=convert(end) - convert(begin), tokens=tokens,
            steady_state=_state_for_interval(segments, convert(begin), convert(end))))
        previous = end
    result = dict(schema=RUN_SCHEMA, context=deepcopy(context), run_id=run_id, split=split, seed=seed,
        process_id=process_id, started_at_utc=started_at_utc, ended_at_utc=ended_at_utc,
        program_sha256=_hash(program["program_sha256"]), contract_identity=contract_identity(program),
        action_inventory=deepcopy(program["actions"]), token_inventory=inventory,
        clock=deepcopy(clock), origin_tick=origin, instrumentation="bare" if bare else "stamped",
        segments=segments, calibration_segments=calibration_segments, invocations=invocations,
        source_files=deepcopy(source_files), raw_bindings=deepcopy(raw_bindings),
        raw_extraction_reconstructable=raw_bindings is not None,
        numerical_verification=deepcopy(numerical_verification),
        numerical_verification_scope="collector receipt; raw bytes rehashed; scalar extraction/NumPy not rerun here",
        state_provenance="external controller request schedule/owned allocation lifetimes; request is not instantaneous actual occupancy; no foreground timing/utilization predictor",
        source_verified=True, all_samples_retained=True, outliers_removed=0)
    _verify_sources(result)
    if raw_bindings is not None:
        inputs = _read_inputs(result)
        _require(inputs == (program, native, events, numerical_verification),
                 "bound raw inputs disagree with normalized extraction inputs")
    result["record_sha256"] = digest(result)
    return result


def _read_inputs(record):
    bindings = record.get("raw_bindings")
    required = {"program", "native", "events", "numerical"}
    _require(isinstance(bindings, dict) and required <= set(bindings) <= required | {"worker_cpu", "worker_gpu", "worker_activity"},
             "raw input bindings require program/native/events/numerical; optional worker_cpu/worker_gpu/worker_activity")
    values = {}
    for key in required:
        name = bindings[key]
        _require(name in record["source_files"], "raw input lacks source byte hash: " + key)
        values[key] = json.loads(Path(name).read_text(encoding="utf-8"))
    events = values["events"]
    _require(isinstance(events, (dict, list)), "events raw file requires a segment/event array or dictionary")
    if isinstance(events, dict) and "known_segments" not in events:
        events = events.get("events")  # Legacy timeline wrapper.
    return values["program"], values["native"], events, values["numerical"]


def _validate_record(record, program, context, split, verify_sources):
    _require(record.get("schema") == RUN_SCHEMA and record.get("split") == split, "wrong run schema/split")
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    _require(digest(unsigned) == record.get("record_sha256"), "run record digest mismatch")
    _require(record["context"] == context and record["contract_identity"] == contract_identity(program),
             "exact context/program contract differs")
    _require(record["action_inventory"] == program["actions"] and record["token_inventory"] == token_inventory(program),
             "run action/token inventory differs")
    _require(record["numerical_verification"].get("passed") is True, "numerical receipt not passed")
    _integer(record["process_id"], "PID", 1)
    _require(_utc(record["started_at_utc"]) <= _utc(record["ended_at_utc"]), "run chronology reversed")
    if verify_sources:
        _verify_sources(record)
        if record.get("raw_bindings") is not None:
            raw_program, native, events, numerical = _read_inputs(record)
            rebuilt = make_run_record(raw_program, native, events, context=record["context"],
                run_id=record["run_id"], split=record["split"], seed=record["seed"],
                process_id=record["process_id"], started_at_utc=record["started_at_utc"],
                ended_at_utc=record["ended_at_utc"], source_files=record["source_files"],
                numerical_verification=numerical, raw_bindings=record["raw_bindings"])
            _require(canonical(rebuilt) == canonical(record), "run normalization does not reconstruct from bound raw data")


def _validate_runs(records, program, context, split, verify_sources=True):
    records = tuple(records)
    _require(records, "nonempty run cohort required")
    _require(len({row["run_id"] for row in records}) == len(records), "duplicate/renamed run ID")
    for row in records:
        _validate_record(row, program, context, split, verify_sources)
    _require(all(row["clock"] == records[0]["clock"] for row in records), "cohort clock identities differ")
    for i, a in enumerate(records):
        for b in records[i + 1:]:
            _require(a["source_files"] != b["source_files"], "raw source inventory reused across independent runs")
            if a["process_id"] == b["process_id"]:
                _require(max(_utc(a["started_at_utc"]), _utc(b["started_at_utc"])) >
                         min(_utc(a["ended_at_utc"]), _utc(b["ended_at_utc"])), "same PID has overlapping run lifetimes")
    return records


def _gates(gates=None):
    gates = deepcopy(DEFAULT_GATES if gates is None else gates)
    _require(set(gates) == set(DEFAULT_GATES), "gate inventory differs")
    _integer(gates["minimum_independent_runs"], "minimum runs", 3)
    for name, value in gates.items():
        if name != "minimum_independent_runs":
            _require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "invalid gate " + name)
    return gates


def _sample_index(records):
    """Index all retained calibration samples once without changing their order."""
    indexed = []
    for record in records:
        # Eligibility is a declared calibration window, not merely a requested
        # state during an uncalibrated transition/recovery grace interval.
        states = {tuple(segment["state"][key] for key in STATE_KEYS)
                  for segment in record["calibration_segments"] if segment["state"] is not None}
        samples = defaultdict(list)
        for invocation in record["invocations"]:
            for item in invocation["tokens"]:
                state = item["calibration_state"]
                if state is not None:
                    samples[(item["token"], tuple(state[key] for key in STATE_KEYS))].append(item["elapsed_ns"])
        indexed.append((record, states, samples))
    return indexed


def _cell_statistics(records, token, state, gates, *, sample_index=None, minimum_samples=None):
    processes, medians = [], []
    state_tuple = tuple(state[key] for key in STATE_KEYS)
    indexed = _sample_index(records) if sample_index is None else sample_index
    for record, states, samples in indexed:
        if state_tuple not in states:
            continue  # Other independently declared memory-hold cells are not missing replicates of this cell.
        values = samples.get((token, state_tuple), [])
        central, variability = median(values), cv(values)
        if central is not None:
            medians.append(central)
        processes.append(dict(run_id=record["run_id"], process_id=record["process_id"], seed=record["seed"],
                              sample_count=len(values), samples_ns=values, median_ns=encoded(central), cv=variability,
                              within_process_cv_passed=variability is not None and variability <= gates["within_process_cv_max"]))
        if minimum_samples is not None:
            processes[-1].update(minimum_sample_count=minimum_samples,
                                 minimum_sample_count_passed=len(values) >= minimum_samples)
    between = cv(medians)
    issues = []
    if len(medians) < gates["minimum_independent_runs"]:
        issues.append("insufficient_independent_processes")
    if any(not row["within_process_cv_passed"] for row in processes):
        issues.append("within_process_cv_gate_failed_or_missing")
    if minimum_samples is not None and any(not row["minimum_sample_count_passed"] for row in processes):
        issues.append("minimum_samples_per_process_gate_failed")
    if between is None or between > gates["between_process_cv_max"]:
        issues.append("between_process_cv_gate_failed_or_missing")
    return dict(median_ns=encoded(median(medians)), processes=processes, between_process_cv=between,
                evidence_issues=issues, qualified=not issues,
                estimator="exact median of per-process exact medians; equal process weight; all observations retained")


def freeze_profile(program, records, *, context, required_states, frozen_at_utc,
                   protocol, gates=None, verify_sources=True):
    """Estimate scoped completed-action durations without a total-time fit.

    Unstable numeric cells remain available for explicitly diagnostic model
    evaluation and retain ``qualified=False``. Missing cells remain unknown.
    """
    gates = _gates(gates)
    records = _validate_runs(records, program, context, "profile", verify_sources)
    cutoff = _utc(frozen_at_utc)
    _require(all(_utc(row["ended_at_utc"]) <= cutoff for row in records), "profile source ended after freeze")
    _require(isinstance(protocol, dict) and protocol, "predeclared protocol required")
    states = [validate_state(state) for state in required_states]
    _require(states and len({state_key(state) for state in states}) == len(states), "unique required state grid needed")
    inventory = token_inventory(program)
    sample_index = _sample_index(records)
    minimum_samples = protocol.get("minimum_cell_samples_per_process")
    if minimum_samples is not None:
        _integer(minimum_samples, "predeclared minimum cell samples", 1)
    cells = []
    for definition in inventory:
        for state in states:
            cell = dict(token=definition["token"], state=state,
                        **_cell_statistics(records, definition["token"], state, gates, sample_index=sample_index,
                                           minimum_samples=minimum_samples))
            cell["original_cv_criterion"] = definition["original_cv_criterion"]
            cell["criterion_scope"] = ("original_compute_copy_source_criterion" if definition["original_cv_criterion"]
                                       else "diagnostic_microgap_or_control_not_original_source_criterion")
            cells.append(cell)
    result = dict(schema=FROZEN_SCHEMA, context=deepcopy(context), contract_identity=contract_identity(program),
        token_inventory=inventory, required_states=states, cells=cells, gates=gates, protocol=deepcopy(protocol),
        frozen_at_utc=frozen_at_utc, profile_runs=list(records),
        profile_run_ids=[row["run_id"] for row in records], profile_seeds=sorted({row["seed"] for row in records}),
        timing_scope="completed host action and disjoint before gaps/tail; no API/device additive decomposition",
        response_semantics="effective invocation completion under exact action/backend/context; not pure hardware throughput",
        heldout_used_for_estimation=False, total_duration_used_as_parameter=False,
        predictions_computed_in_profiling=False,
        source_qualified=all(cell["qualified"] for cell in cells if cell["original_cv_criterion"]),
        raw_extraction_reconstructable=all(row["raw_extraction_reconstructable"] for row in records),
        diagnostic_numeric_cells_retained=True,
        unknown_cells=sum(cell["median_ns"] is None for cell in cells))
    if protocol.get("request_response_lags", {}).get("enabled") is True:
        _require(context.get("time_origin") == "planned_go_start_v1", "lag response requires the external GO time origin")
        from .lags import estimate_request_delays
        lag_parameters, lag_evidence = estimate_request_delays(records, config=protocol["request_response_lags"])
        result.update(request_response_delays=lag_parameters.to_dict(), profile_lag_evidence=lag_evidence)
    result["artifact_sha256"] = digest(result)
    return result


def load_frozen(program, frozen, *, verify_sources=True):
    if isinstance(frozen, (str, Path)):
        frozen = json.loads(Path(frozen).read_text(encoding="utf-8"))
    frozen = deepcopy(frozen)
    _require(frozen.get("schema") == FROZEN_SCHEMA, "unsupported frozen response schema")
    _require(frozen.get("artifact_sha256") == digest({key: value for key, value in frozen.items() if key != "artifact_sha256"}),
             "frozen artifact digest mismatch")
    rebuilt = freeze_profile(program, frozen["profile_runs"], context=frozen["context"],
        required_states=frozen["required_states"], frozen_at_utc=frozen["frozen_at_utc"],
        protocol=frozen["protocol"], gates=frozen["gates"], verify_sources=verify_sources)
    _require(canonical(rebuilt) == canonical(frozen), "frozen statistics do not reconstruct from profile evidence")
    return frozen


def response_table(program, frozen):
    """Create the model-owned table; no performance equation is defined here."""
    from llm_infer_model.performance.conditioned_response import ResponseCell, ResponseState, ResponseTable
    context_sha = digest(dict(context=frozen["context"], contract_identity=contract_identity(program),
                             token_inventory=frozen["token_inventory"]))
    cells = []
    for cell in frozen["cells"]:
        evidence = tuple(f"record:{row['record_sha256']}:{cell['token']}" for row in frozen["profile_runs"]
                         if any(segment["state"] == cell["state"] for segment in row["segments"]))
        cells.append(ResponseCell(cell["token"], ResponseState(**cell["state"]),
            None if cell["median_ns"] is None else exact(cell["median_ns"]),
            evidence=evidence or ("missing profile state/action evidence",), qualified=cell["qualified"]))
    return ResponseTable(context_sha, tuple(item["token"] for item in frozen["token_inventory"]), tuple(cells))


def response_trajectory(record, fallback_state, *, start_ns=None):
    """Preserve unknown holes; unavailable points carry no priced state claim."""
    from llm_infer_model.performance.conditioned_response import ResponsePoint, ResponseState, ResponseTrajectory
    initial, available = validate_state(fallback_state), False
    transitions = {}
    last = initial
    for segment in record["segments"]:
        state = last if segment["state"] is None else segment["state"]
        known = segment["state"] is not None
        if segment["begin_ns"] == 0:
            initial, available = state, known
        else:
            transitions[segment["begin_ns"]] = (state, known)
        transitions[segment["end_ns"]] = (state, False)
        last = state
    ordered = sorted(transitions.items())
    if start_ns is not None:
        active_available = available
        for at, (_, known) in ordered:
            if at <= start_ns:
                active_available = known
        if not active_available:
            return ResponseTrajectory(ResponseState(**initial), (), initial_available=False)
        # Unavailable is represented as a model pause only to terminate support:
        # never pass a later recovery after an unknown hole. The common engine
        # can finish before this boundary or report blocked/unknown; it cannot
        # invent progress or a zero-service price through the missing interval.
        for at, (_, known) in ordered:
            if at > start_ns and not known:
                ordered = [(point, pair) for point, pair in ordered if point <= at]
                break
    points = tuple(ResponsePoint(at_ns=at, state=ResponseState(**state), available=known)
                   for at, (state, known) in ordered if at > 0)
    return ResponseTrajectory(ResponseState(**initial), points, initial_available=available)


def interval_coverage(record, begin, end):
    """Observation-only containment/coverage; never a performance prediction."""
    pieces = []
    for segment in record["segments"]:
        left, right = max(begin, segment["begin_ns"]), min(end, segment["end_ns"])
        if left < right and segment["state"] is not None:
            pieces.append((left, right, segment["state"]))
    known_ns = sum(right - left for left, right, _ in pieces)
    states = {state_key(state) for _, _, state in pieces}
    fully_known = known_ns == end - begin
    if not fully_known:
        category = "unknown_transition_or_outside"
    elif len(states) > 1:
        category = "known_state_crossing"
    else:
        category = "steady"
    return dict(category=category, known_ns=known_ns, unknown_ns=end - begin - known_ns,
                distinct_states=len(states), steady_state=pieces[0][2] if category == "steady" else None,
                inside_profile_calibration_window=_state_for_interval(record["calibration_segments"], begin, end) is not None)


def _known_request_history(record, entry_ns):
    """Actuator state needs every request from GO0, not only entry containment."""
    cursor = 0
    for segment in record["segments"]:
        if segment["begin_ns"] > cursor or segment["state"] is None:
            return False
        cursor = max(cursor, segment["end_ns"])
        if cursor >= entry_ns:
            return True
    return False


def _quantile(values, fraction):
    values = sorted(float(value) for value in values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    left = math.floor(position)
    return values[left] + (values[math.ceil(position)] - values[left]) * (position - left)


def _bucket_summary(rows, records):
    processes = []
    for record in records:
        selected = [row for row in rows if row["run_id"] == record["run_id"]]
        errors = [row["relative_error"] for row in selected if row["relative_error"] is not None]
        central = None if not errors else statistics.median(errors)
        processes.append(dict(run_id=record["run_id"], process_id=record["process_id"], sample_count=len(selected),
            predictable_count=len(errors), median_absolute_relative_error=central,
            p90_absolute_relative_error=_quantile(errors, .90), worst_absolute_relative_error=max(errors, default=None),
            observed_median_ns=encoded(median(row["observed_ns"] for row in selected)),
            observed_raw_cv=cv(row["observed_ns"] for row in selected)))
    medians = [row["median_absolute_relative_error"] for row in processes if row["median_absolute_relative_error"] is not None]
    errors = [row["relative_error"] for row in rows if row["relative_error"] is not None]
    return dict(sample_count=len(rows), predictable_count=len(errors),
                process_balanced_median_absolute_relative_error=None if not medians else statistics.median(medians),
                worst_process_median_absolute_relative_error=max(medians, default=None),
                pooled_p90_absolute_relative_error=_quantile(errors, .90),
                worst_absolute_relative_error=max(errors, default=None), processes=processes,
                raw_cv_scope="diagnostic only across different states; no pooled-CV accuracy claim")


def _verify_barrier(barrier, frozen, records):
    if barrier is None:
        return False
    _require(isinstance(barrier, dict) and barrier.get("schema") == "state-response-heldout-barrier-1", "invalid heldout barrier")
    _require(barrier.get("frozen_artifact_sha256") == frozen["artifact_sha256"], "heldout barrier frozen-model mismatch")
    expected = sorted(row["record_sha256"] for row in frozen["profile_runs"])
    _require(barrier.get("profile_record_sha256s") == expected, "barrier profile inventory differs")
    moment = _utc(barrier["created_at_utc"])
    _require(moment >= _utc(frozen["frozen_at_utc"]), "barrier precedes parameter freeze")
    _require(all(_utc(row["started_at_utc"]) >= moment for row in records), "heldout process launched before barrier")
    return True


def _primary_validation(frozen, records, predictions, *, barrier_verified, raw_verified, state_stability):
    """Validate the frozen choice using existing predictions, never predict again.

    Observation-only stability is shared with the legacy report. Error summaries
    and prediction coverage must instead come from the selected model, including
    its failures. Descriptive medians omit unknown errors; acceptance never does.
    """
    model = frozen["protocol"].get("primary_prediction_model", "dynamic_state_integration")
    _require(model in {"dynamic_state_integration", "lagged_dynamic_state_integration",
                       "no_owned_cpu_gpu_load", "entry_state_constant"}, "unknown primary validation model")
    selected = []
    for row in predictions:
        if model == "dynamic_state_integration":
            price = row
        else:
            _require(model in row["comparisons"], "selected model has no corresponding prediction")
            price = row["comparisons"][model]
        _require((price["predicted_ns"] is None) == (price["relative_error"] is None),
                 "selected prediction and error availability differ")
        selected.append(dict(run_id=row["run_id"], process_id=row["process_id"],
            observation_id=row["observation_id"], observed_ns=row["observed_ns"], coverage=row["coverage"],
            predicted_ns=price["predicted_ns"], relative_error=price["relative_error"]))
    gates = frozen["gates"]
    overall = _bucket_summary(selected, records)
    buckets = {name: _bucket_summary([row for row in selected if row["coverage"]["category"] == name], records)
               for name in ("steady", "known_state_crossing", "unknown_transition_or_outside")}
    measured = {state_key(row["state"]): row for row in state_stability}
    _require(set(measured) == {state_key(state) for state in frozen["required_states"]},
             "primary and legacy observation-stability state inventories differ")
    states = []
    for state in frozen["required_states"]:
        rows = [row for row in selected if row["coverage"]["steady_state"] == state]
        summary = _bucket_summary(rows, records)
        complete = summary["sample_count"] > 0 and summary["predictable_count"] == summary["sample_count"]
        error = summary["process_balanced_median_absolute_relative_error"]
        measurement = measured[state_key(state)]
        summary.update(state=state, complete_prediction_coverage=complete,
            measurement_stability_qualified=measurement["measurement_stability_qualified"],
            between_process_observed_median_cv=measurement["between_process_observed_median_cv"],
            accuracy_passed=bool(complete and error is not None and error <= gates["relative_error_max"]))
        states.append(summary)
    lag_required = model == "lagged_dynamic_state_integration"
    if lag_required:
        _require("request_response_delays" in frozen, "primary actuator model requires frozen worker delay parameters")
        lag_qualified = frozen["request_response_delays"]["qualified"]
        _require(type(lag_qualified) is bool, "lag source qualification must be explicit")
    else:
        lag_qualified = None
    source_qualified = frozen["source_qualified"] and (not lag_required or lag_qualified)
    evidence_ready = bool(barrier_verified and raw_verified and source_qualified)
    steady_accepted = bool(evidence_ready and len(records) >= gates["minimum_independent_runs"] and
        all(row["measurement_stability_qualified"] and row["accuracy_passed"] for row in states))
    crossing = buckets["known_state_crossing"]
    crossing_rows = [row for row in selected if row["coverage"]["category"] == "known_state_crossing"]
    crossing_runs = {row["run_id"] for row in crossing_rows}
    crossing_pids = {row["process_id"] for row in crossing_rows}
    crossing_complete = crossing["sample_count"] > 0 and crossing["predictable_count"] == crossing["sample_count"]
    crossing_error = crossing["process_balanced_median_absolute_relative_error"]
    crossing_accepted = bool(evidence_ready and len(crossing_runs) >= gates["minimum_independent_runs"] and
        len(crossing_pids) >= gates["minimum_independent_runs"] and crossing_complete and
        crossing_error is not None and crossing_error <= gates["relative_error_max"])
    total, predictable = len(selected), sum(row["predicted_ns"] is not None for row in selected)
    known = sum(row["coverage"]["category"] != "unknown_transition_or_outside" for row in selected)
    error = overall["process_balanced_median_absolute_relative_error"]
    return dict(schema="state-response-primary-validation-1", chosen_model=model,
        selection_source="frozen protocol" if "primary_prediction_model" in frozen["protocol"] else "legacy dynamic default",
        prediction_provider="existing modeling predictions; no prediction, parameter fit or model selection performed here",
        scope="request-conditioned completed-action response; not an identified actual CPU/GPU available-service transfer function",
        gates=deepcopy(gates), barrier_verified=barrier_verified, raw_sources_and_extraction_verified=raw_verified,
        original_compute_copy_profile_source_qualified=frozen["source_qualified"],
        microgap_control_qualification_scope="diagnostic, unchanged from the original source criterion",
        lag_source_required=lag_required, lag_source_qualified=lag_qualified,
        required_profile_source_gates_passed=bool(source_qualified), overall=overall, buckets=buckets,
        by_steady_state=states,
        coverage=dict(invocations=total, predictable_invocations=predictable,
            known_external_state_invocations=known, unknown_external_state_invocations=total - known,
            complete_prediction_coverage=bool(total and predictable == total),
            crossing_independent_runs=len(crossing_runs), crossing_distinct_process_ids=len(crossing_pids),
            crossing_complete_prediction_coverage=crossing_complete),
        error_statistic_scope="descriptive errors use predictable samples; all state/crossing acceptance additionally requires complete prediction coverage",
        steady_accepted=steady_accepted, known_crossing_accepted=crossing_accepted,
        statistical_protocol_accepted=bool(total and steady_accepted and crossing_accepted and known == total and predictable == total),
        median_error_within_threshold=error is not None and error <= gates["relative_error_max"],
        instrumentation_control_status="not independently established for this native/action/state scope",
        actual_CPU_GPU_available_service_transfer_identified=False,
        full_accuracy_qualified=False, accepted=False,
        full_accuracy_qualification_issues=["matching_stamped_vs_bare_instrumentation_control_not_independently_established",
            "actual_CPU_GPU_available_service_transfer_not_identified"])


def evaluate_heldout(program, frozen, records, *, barrier=None, verify_sources=True, retain_model_details=False):
    """Compare fresh bare invocations with modeling under exogenous state traces.

    The measured invocation *start* is supplied as a conditional query origin;
    measured internal action boundaries/durations are never prediction inputs.
    This is not a free-running end-to-end rollout of invocation launch times.
    """
    from llm_infer_model.performance import InvalidModel
    from llm_infer_model.performance.conditioned_response import (
        RequestResponseDelays, ResponseState, ResponseTrajectory, evaluate_response,
    )
    frozen = load_frozen(program, frozen, verify_sources=verify_sources)
    records = _validate_runs(records, program, frozen["context"], "heldout", verify_sources)
    _require(not set(row["run_id"] for row in records).intersection(frozen["profile_run_ids"]), "heldout run ID leaked into profile")
    _require(not set(row["seed"] for row in records).intersection(frozen["profile_seeds"]), "heldout numeric seed leaked into profile")
    for record in records:
        _require(_utc(record["started_at_utc"]) >= _utc(frozen["frozen_at_utc"]), "heldout began before parameter freeze")
        for profile in frozen["profile_runs"]:
            if record["process_id"] == profile["process_id"]:
                _require(_utc(record["started_at_utc"]) > _utc(profile["ended_at_utc"]), "heldout reused an overlapping profile PID")
            _require(record["source_files"] != profile["source_files"], "heldout raw evidence reused from profile")
    barrier_verified = _verify_barrier(barrier, frozen, records)
    table = response_table(program, frozen)
    predictions, telemetry_ignored = [], True
    lag_parameters = (None if "request_response_delays" not in frozen else
                      RequestResponseDelays.from_dict(frozen["request_response_delays"]))
    comparison_names = ["no_owned_cpu_gpu_load", "entry_state_constant"]
    if lag_parameters is not None:
        comparison_names.append("lagged_dynamic_state_integration")
    comparison_rows = {name: [] for name in comparison_names}
    primary_model = frozen["protocol"].get("primary_prediction_model", "dynamic_state_integration")
    _require(primary_model in {"dynamic_state_integration", *comparison_names}, "unknown predeclared primary prediction model")
    settle_ms = frozen["protocol"].get("profile_settle_ms")
    if settle_ms is not None:
        _integer(settle_ms, "predeclared profile settle ms")
    for record in records:
        for observation in record["invocations"]:
            trajectory = response_trajectory(record, frozen["required_states"][0], start_ns=observation["begin_ns"])
            coverage = interval_coverage(record, observation["begin_ns"], observation["end_ns"])
            entry_segment = next((segment for segment in record["segments"]
                if segment["begin_ns"] <= observation["begin_ns"] < segment["end_ns"]), None)
            state_age = None if entry_segment is None else observation["begin_ns"] - entry_segment["request_begin_ns"]
            coverage.update(entry_request_state_age_ns=state_age,
                before_declared_profile_settle_age=None if settle_ms is None or state_age is None else state_age < settle_ms * 1_000_000)
            model, model_digest, model_end, problem, predicted = None, None, None, None, None
            try:
                if _state_for_interval(record["segments"], observation["begin_ns"], observation["begin_ns"]) is None:
                    raise InvalidModel("unknown external state at invocation entry; no zero-service substitution")
                result = evaluate_response(table, trajectory, start_ns=observation["begin_ns"])
                predicted, model_end = result.duration_ns, result.end_ns
                details = result.to_dict()
                model_digest = digest(details)
                if retain_model_details:
                    model = details
            except InvalidModel as error:
                problem = dict(type=type(error).__name__, message=str(error))
            observed = observation["elapsed_ns"]
            entry_state = _state_for_interval(record["segments"], observation["begin_ns"], observation["begin_ns"])
            comparisons = {}
            for name in comparison_rows:
                comparator, failure = None, None
                if entry_state is None:
                    failure = dict(type="UnknownExternalState", message="entry state is unknown; comparison not priced")
                else:
                    state = dict(entry_state)
                    if name == "no_owned_cpu_gpu_load":
                        state.update(cpu_replay=0, gpu_replay=0)
                    try:
                        if name == "lagged_dynamic_state_integration":
                            if not _known_request_history(record, observation["begin_ns"]):
                                raise InvalidModel("unknown request history before entry; frozen actuator lag cannot reconstruct effective state")
                            effective = ResponseState(0, 0, trajectory.initial_state.host_memory_bytes,
                                                      trajectory.initial_state.device_memory_bytes)
                            comparator = evaluate_response(table, trajectory, start_ns=observation["begin_ns"],
                                request_delays=lag_parameters, initial_effective_state=effective)
                        else:
                            comparator = evaluate_response(table, ResponseTrajectory(ResponseState(**state)),
                                                           start_ns=observation["begin_ns"])
                    except InvalidModel as error:
                        failure = dict(type=type(error).__name__, message=str(error))
                comparison = dict(predicted_ns=None if comparator is None else comparator.duration_ns,
                    predicted_end_ns=None if comparator is None else comparator.end_ns,
                    relative_error=None if comparator is None else abs(comparator.duration_ns - observed) / observed,
                    model_problem=failure)
                comparisons[name] = comparison
                comparison_rows[name].append(dict(comparison, run_id=record["run_id"], process_id=record["process_id"],
                    observation_id=observation["observation_id"], observed_ns=observed, coverage=coverage))
            predictions.append(dict(run_id=record["run_id"], process_id=record["process_id"],
                observation_id=observation["observation_id"], begin_ns=observation["begin_ns"], end_ns=observation["end_ns"],
                coverage=coverage, observed_ns=observed, predicted_ns=predicted, predicted_end_ns=model_end,
                model_result=model, model_result_sha256=model_digest,
                model_problem=problem, relative_error=None if predicted is None else abs(predicted - observed) / observed,
                residual_observed_minus_predicted_ns=None if predicted is None else observed - predicted,
                comparisons=comparisons))
    buckets = {name: _bucket_summary([row for row in predictions if row["coverage"]["category"] == name], records)
               for name in ("steady", "known_state_crossing", "unknown_transition_or_outside")}
    buckets["known_state_recovery_outside_profile_settle_window"] = _bucket_summary(
        [row for row in predictions if row["coverage"]["category"] == "steady" and not row["coverage"]["inside_profile_calibration_window"]], records)
    buckets["settled_within_profile_calibration_window"] = _bucket_summary(
        [row for row in predictions if row["coverage"]["category"] == "steady" and row["coverage"]["inside_profile_calibration_window"]], records)
    buckets["recovery_before_declared_profile_settle_age"] = _bucket_summary(
        [row for row in predictions if row["coverage"]["category"] == "steady" and row["coverage"]["before_declared_profile_settle_age"] is True], records)
    buckets["settled_after_declared_profile_settle_age"] = _bucket_summary(
        [row for row in predictions if row["coverage"]["category"] == "steady" and row["coverage"]["before_declared_profile_settle_age"] is False], records)
    overall = _bucket_summary(predictions, records)
    comparison_summaries = {}
    for name, rows in comparison_rows.items():
        comparison_summaries[name] = dict(overall=_bucket_summary(rows, records),
            buckets={bucket: _bucket_summary([row for row in rows if row["coverage"]["category"] == bucket], records)
                     for bucket in ("steady", "known_state_crossing", "unknown_transition_or_outside")},
            table_artifact_sha256=frozen["artifact_sha256"], refitted=False,
            prediction_provider="llm_infer_model.performance.conditioned_response.evaluate_response",
            assumption=("CPU/GPU offered replay fixed at 0/0; actual held-memory coordinates preserved" if name == "no_owned_cpu_gpu_load"
                        else "entry offered state held constant; future changes ignored" if name == "entry_state_constant"
                        else "GO requests transformed by four frozen profile-only worker response delays; initial effective CPU/GPU idle; held-memory coordinates preserved"))
        if name == "lagged_dynamic_state_integration":
            comparison_summaries[name].update(request_response_delays=frozen["request_response_delays"],
                delay_source_qualified=lag_parameters.qualified, heldout_worker_intervals_used=False,
                source_qualified=bool(frozen["source_qualified"] and lag_parameters.qualified))
    by_state = []
    gates = frozen["gates"]
    for state in frozen["required_states"]:
        rows = [row for row in predictions if row["coverage"]["steady_state"] == state]
        summary = _bucket_summary(rows, records)
        processes = [process for process in summary["processes"] if process["sample_count"]]
        medians = [exact(process["observed_median_ns"]) for process in processes]
        between = cv(medians)
        stable = len(processes) >= gates["minimum_independent_runs"] and all(
            process["observed_raw_cv"] <= gates["within_process_cv_max"] for process in processes) and (
                between is not None and between <= gates["between_process_cv_max"])
        summary.update(state=state, between_process_observed_median_cv=between,
                       measurement_stability_qualified=stable,
                       accuracy_passed=summary["process_balanced_median_absolute_relative_error"] is not None and
                       summary["process_balanced_median_absolute_relative_error"] <= gates["relative_error_max"])
        by_state.append(summary)
    source_qualified = frozen["source_qualified"]
    raw_verified = verify_sources and frozen["raw_extraction_reconstructable"] and all(row["raw_extraction_reconstructable"] for row in records)
    sample_count = len(predictions)
    known_count = sum(row["coverage"]["category"] != "unknown_transition_or_outside" for row in predictions)
    predicted_count = sum(row["predicted_ns"] is not None for row in predictions)
    crossing_runs = {row["run_id"] for row in predictions if row["coverage"]["category"] == "known_state_crossing"}
    accuracy = overall["process_balanced_median_absolute_relative_error"]
    steady_accepted = bool(barrier_verified and raw_verified and source_qualified and
        len(records) >= gates["minimum_independent_runs"] and all(row["measurement_stability_qualified"] and row["accuracy_passed"] for row in by_state))
    crossing_error = buckets["known_state_crossing"]["process_balanced_median_absolute_relative_error"]
    crossing_accepted = bool(barrier_verified and raw_verified and source_qualified and
        len(crossing_runs) >= gates["minimum_independent_runs"] and crossing_error is not None and
        crossing_error <= gates["relative_error_max"] and
        buckets["known_state_crossing"]["predictable_count"] == buckets["known_state_crossing"]["sample_count"])
    result = dict(schema=EVALUATION_SCHEMA, frozen_artifact_sha256=frozen["artifact_sha256"],
        heldout_record_sha256s=[row["record_sha256"] for row in records],
        prediction_provider="llm_infer_model.performance.conditioned_response.evaluate_response",
        prediction_scope="whole replay duration conditional on an observed invocation start; no internal heldout action times",
        state_input_scope="exogenous controller request trace/owned allocation bytes; not target-inclusive CPU/GPU utilization",
        full_per_invocation_model_details_retained=bool(retain_model_details),
        heldout_used_for_estimation=False, total_duration_used_as_parameter=False,
        numerical_verification_scope="bound collector receipts; offline raw extraction reconstructed, independent NumPy not rerun",
        barrier_verified=barrier_verified, barrier=deepcopy(barrier),
        chronology_scope="declared UTC/barrier metadata checked; actual Popen/software-inventory proof belongs to collector",
        raw_sources_and_extraction_verified=raw_verified, profile_source_qualified=source_qualified,
        profile_microgap_control_cv_scope="reported separately; not silently added to original compute/copy source criterion",
        original_source_failed_cells=sum(not cell["qualified"] for cell in frozen["cells"] if cell["original_cv_criterion"]),
        microgap_control_failed_diagnostic_cells=sum(not cell["qualified"] for cell in frozen["cells"] if not cell["original_cv_criterion"]),
        gates=gates, overall=overall, buckets=buckets, by_steady_state=by_state,
        comparison_models=comparison_summaries,
        comparison_protocol="predeclared controls and optional profile-only lag candidate; identical frozen action cells; no heldout selection/refit",
        predeclared_primary_model=primary_model,
        predeclared_primary_summary=overall if primary_model == "dynamic_state_integration" else comparison_summaries[primary_model]["overall"],
        predeclared_primary_buckets=buckets if primary_model == "dynamic_state_integration" else comparison_summaries[primary_model]["buckets"],
        coverage=dict(invocations=sample_count, known_external_state_invocations=known_count,
            predictable_invocations=predicted_count, unknown_external_state_invocations=sample_count - known_count,
            known_state_fraction=known_count / sample_count, prediction_fraction=predicted_count / sample_count,
            observed_unknown_ns=sum(row["coverage"]["unknown_ns"] for row in predictions),
            observed_total_ns=sum(row["observed_ns"] for row in predictions),
            crossing_independent_runs=len(crossing_runs)),
        steady_accepted=steady_accepted, known_crossing_accepted=crossing_accepted,
        statistical_protocol_accepted=bool(steady_accepted and crossing_accepted and known_count == sample_count and predicted_count == sample_count),
        statistical_acceptance_scope="new conditional-state protocol only; does not replace prior single-layer acceptance or prove instrumentation neutrality",
        instrumentation_control_status="not independently established for this native/action/state scope",
        full_accuracy_qualified=False, accepted=False,
        full_accuracy_qualification_issues=["matching_stamped_vs_bare_instrumentation_control_not_independently_established"],
        profile_settle_ms=settle_ms,
        median_error_within_threshold=accuracy is not None and accuracy <= gates["relative_error_max"],
        residual_cause="unresolved; qualified estimates and controlled stage/CPU-active diagnostics needed for attribution",
        model_assumption="memoryless effective invocation progress in exact numeric state; transition/recovery accuracy separately tested",
        total_telemetry_used_as_predictor=not telemetry_ignored, predictions=predictions)
    result["legacy_validation_scope"] = dict(model="dynamic_state_integration",
        fields=["predictions", "overall", "buckets", "by_steady_state", "coverage", "steady_accepted",
                "known_crossing_accepted", "statistical_protocol_accepted", "median_error_within_threshold"],
        purpose="unchanged historical unlagged fields; do not use as validation of another declared primary model",
        primary_model_selection_applied_to_these_fields=False)
    result["primary_validation"] = _primary_validation(frozen, records, predictions,
        barrier_verified=barrier_verified, raw_verified=raw_verified, state_stability=by_state)
    result["evaluation_sha256"] = digest(result)
    return result


def import_legacy_campaign(campaign, *, split="profile"):
    """Recover full action traces from the old collector, never its Ridge fit.

    This imports measured completed-action responses, not pure hardware rates.
    Historical process counts, unknown transitions and instability are retained.
    """
    _require(split in {"profile", "heldout"}, "legacy split requires profile or heldout")
    root = Path(campaign).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema") == "resource-trajectory-campaign-0.1", "unsupported legacy campaign")
    old_split = "train" if split == "profile" else split
    cohort_path = root / (old_split + "-rows.json")
    executions = json.loads(cohort_path.read_text(encoding="utf-8"))
    jobs = [job for job in manifest["jobs"] if job["split"] == old_split]
    _require([row["job"] for row in executions] == jobs and jobs, "legacy cohort inventory differs")
    records = []
    for execution in executions:
        job = execution["job"]
        directory = (root / "jobs" / job["id"]).resolve()
        _require(directory.is_relative_to(root), "legacy job path escapes campaign")
        _require(execution == json.loads((directory / "execution.json").read_text(encoding="utf-8")), "legacy execution receipt differs")
        _require(execution.get("status") == "measured" and execution.get("cleanup_verified") is True, "legacy process did not complete cleanly")
        sources = {str(manifest_path): file_sha(manifest_path), str(cohort_path): file_sha(cohort_path),
                   str(directory / "execution.json"): file_sha(directory / "execution.json")}
        for name, expected in execution["artifacts_sha256"].items():
            path = (directory / name).resolve()
            _require(path.is_relative_to(directory), "legacy bound path escapes run")
            _require(file_sha(path) == expected, "legacy raw artifact changed")
            sources[str(path)] = expected
        binding = manifest["programs"][job["program_key"]]
        program_path = Path(binding["path"]).resolve()
        _require(file_sha(program_path) == binding["sha256"], "legacy program changed")
        sources[str(program_path)] = binding["sha256"]
        paths = dict(program=str(program_path), native=str(directory / "main/result.json"),
                     events=str(directory / "timeline.json"), numerical=str(directory / "numerical.json"))
        program, native, timeline, numerical = (json.loads(Path(paths[name]).read_text(encoding="utf-8"))
                                               for name in ("program", "native", "events", "numerical"))
        context = dict(state_semantics="offered_load_controlled", protocol_id=manifest["protocol_id"],
            historical_source_inventory_sha256=manifest["source_inventory_sha256"],
            execution_boundary=native["execution_boundary"], mode=job["mode"],
            source_scope="historical recorded bytes; no current-source equivalence claim")
        _require(job["mode"] == "mixed", "isolated-target legacy replay cannot price complete action sequence")
        records.append(make_run_record(program, native, timeline["events"], context=context,
            run_id=job["id"], split=split, seed=job["seed"], process_id=execution["process_id"],
            started_at_utc=execution["started_utc"], ended_at_utc=execution["completed_utc"],
            source_files=sources, numerical_verification=numerical, raw_bindings=paths))
    return records
