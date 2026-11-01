"""Read immutable per-job native evidence without loading old frozen models.

The native producer uses two-space pretty JSON. The streaming reader checks
that layout and preserves every invocation/action, including zero durations.
Owned-worker completed intervals are replay activity, never hardware busy time.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

SCHEMA = "service-response-run-1"
DEFAULT_COMPETITORS = (
    dict(name="cpu", request_key="cpu_replay", actor_role="cpu-worker"),
    dict(name="gpu", request_key="gpu_replay", actor_role="gpu-worker"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, name + " requires an integer")
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def utc(value):
    require(isinstance(value, str), "explicit UTC timestamp required")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(result.utcoffset() is not None and result.utcoffset().total_seconds() == 0, "timestamp must be UTC")
    return result


class SourceAudit:
    def __init__(self, verify=True):
        self.verify = verify
        self.files = {}

    def hash(self, path, expected=None):
        path = Path(path).resolve()
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                hasher.update(chunk)
        actual = hasher.hexdigest()
        require(not self.verify or expected is None or actual == expected, "consumed source SHA mismatch: " + str(path))
        self.files[str(path)] = dict(sha256=actual, bytes=path.stat().st_size)
        return actual

    def json(self, path, expected=None):
        self.hash(path, expected)
        return json.loads(Path(path).read_text(encoding="utf-8"))


def tail_field(path, name):
    """Extract one top-level producer field, with bounded incremental I/O."""
    path = Path(path)
    marker = ("\n  " + json.dumps(name) + ": ").encode()
    size = path.stat().st_size
    with path.open("rb") as stream:
        count = min(size, 65536)
        while True:
            stream.seek(size - count)
            data = stream.read(count)
            position = data.rfind(marker)
            if position >= 0:
                try:
                    return json.JSONDecoder().raw_decode(data[position + len(marker):].decode("utf-8"))[0]
                except (UnicodeDecodeError, json.JSONDecodeError):
                    pass
            require(count < size, "missing/malformed two-space producer field: " + name)
            count = min(size, count * 2)


def read_native(path):
    """Return small metadata plus compact tick tuples from native producer JSON.

    Iterations: (block, id, begin, end, elapsed, enabled_check).
    Actions: (block, iteration, index, begin, end, elapsed, kind).
    Large ports and other unused arrays are never deserialized.
    """
    wanted = {"format_version", "role", "process_id", "main_thread_id", "program_sha256", "actions",
        "clock", "native_clock", "instrumentation", "execution_boundary", "diagnostic_only", "go_schedule",
        "sampling_begin_tick", "sampling_end_tick", "ready_tick", "target_action_index", "worker_work_scope",
        "warmup_repeat_consistency", "post_replay_consistency", "cuda_schedule", "observed_runtime_flags",
        "untimed_reset_count", "sampling_release_scope", "initial_work_history"}
    meta, iterations, actions = {}, [], []
    key, field, block_index, row_kind, row = None, [], -1, None, None

    def finish():
        if field:
            value = json.loads("{" + "".join(field).rstrip().rstrip(",") + "}")
            require(list(value) == [key], "malformed native top-level field")
            meta.update(value)

    with Path(path).open(encoding="utf-8", newline=None) as stream:
        first = next(stream, None)
        require(first == "{\n", "native reader requires two-space pretty producer JSON")
        for line in stream:
            match = re.match(r'^  "([^"\\]+)":', line)
            if match:
                finish()
                key = match.group(1)
                field = [line] if key in wanted else []
            elif line == "}\n" or line == "}":
                finish(); field = []
            elif field:
                field.append(line)
            if key != "blocks":
                continue
            if line == "    {\n":
                block_index += 1
            elif line in ('      "iterations": [\n', '      "action_intervals": [\n'):
                require(row_kind is None and row is None, "nested native sample array")
                row_kind = "iterations" if '"iterations"' in line else "actions"
            elif row_kind is not None and line.startswith("      ]"):
                require(row is None, "unterminated native sample")
                row_kind = None
            elif row_kind is not None and line == "        {\n":
                require(row is None, "nested native sample")
                row = [line]
            elif row_kind is not None and row is not None:
                row.append(line)
                if line.startswith("        }"):
                    item = json.loads("".join(row).rstrip().rstrip(","))
                    if row_kind == "iterations":
                        iterations.append((block_index, item["iteration"], item["begin_tick"], item["end_tick"],
                                           item["elapsed_ns"], item.get("enabled_check_tick")))
                    else:
                        actions.append((block_index, item["iteration"], item["action_index"], item["begin_tick"],
                                        item["end_tick"], item["elapsed_ns"], item.get("kind")))
                    row = None
    require(row is None and row_kind is None and iterations, "missing/unterminated native invocations")
    require(meta.get("format_version") == "state-response-native-v1", "unsupported native schema")
    require(len({(r[0], r[1]) for r in iterations}) == len(iterations), "duplicate native invocation identity")
    return meta, iterations, actions


def validate_competitors(specs):
    require(isinstance(specs, (tuple, list)) and specs, "explicit named competitors required")
    result = [dict(spec) for spec in specs]
    for spec in result:
        require(set(spec) == {"name", "request_key", "actor_role"} and
                all(isinstance(value, str) and value for value in spec.values()), "invalid competitor binding")
    for key in ("name", "request_key", "actor_role"):
        require(len({spec[key] for spec in result}) == len(result), "duplicate competitor " + key)
    return result


def _normalize_segments(events, origin, frequency, competitors, *, calibration=False):
    convert = lambda tick: round(Fraction((tick - origin) * 1_000_000_000, frequency))
    raw = events["events"] if calibration else events["known_segments"]
    result, previous = [], None
    for item in raw:
        left = item["steady_begin_tick"] if calibration else item["begin_tick"]
        right = item["steady_end_tick"] if calibration else item["end_tick"]
        state = item["requested_resources"] if calibration else item["state"]
        integer(left, "segment begin tick"); integer(right, "segment end tick", left)
        require(previous is None or previous <= left, "request segments reordered/overlapping")
        previous = right
        if left == right:
            continue
        requests = None
        if state is not None:
            requests = {}
            for spec in competitors:
                value = state[spec["request_key"]]
                if type(value) is bool: value = int(value)
                require(type(value) is int and value in (0, 1), "binary offered request required")
                requests[spec["name"]] = value
        result.append(dict(begin_ns=convert(left), end_ns=convert(right), requests=requests,
            host_memory_bytes=None if state is None else integer(state["host_memory_bytes"], "host memory bytes"),
            device_memory_bytes=None if state is None else integer(state["device_memory_bytes"], "device memory bytes")))
    return result


def state_for_interval(segments, begin, end):
    for segment in segments:
        if segment["begin_ns"] <= begin and ((begin == end and begin < segment["end_ns"]) or
                                             (begin < end <= segment["end_ns"])):
            return segment["requests"]
    return None


def iter_tokens(record):
    """Yield baseline/control and compute/copy occurrences without copying them."""
    for invocation in record["invocations"]:
        cursor = invocation["begin_ns"]
        for index, left, right in invocation["actions"]:
            yield invocation, "action:" + str(index) + ":gap_before", cursor, left
            yield invocation, "action:" + str(index) + ":duration", left, right
            cursor = right
        if invocation["actions"]:
            yield invocation, "tail", cursor, invocation["end_ns"]


def token_inventory(actions):
    result = []
    for index, action in enumerate(actions):
        result.extend((dict(token=f"action:{index}:gap_before", action_index=index, kind="HOST_GAP", transformed=False),
                       dict(token=f"action:{index}:duration", action_index=index, kind=action["kind"],
                            transformed=action["kind"] in {"COMPUTE", "COPY"})))
    result.append(dict(token="tail", action_index=None, kind="HOST_TAIL", transformed=False))
    return result


def extract_job(manifest_path, execution_row, *, competitors=None, verify_sources=True):
    """Extract a source-bound record from state_response.collect.execute_job output.

    The old normalized JSON is only stream-hashed and its final source-binding
    map read. Raw native files are consumed one job/actor at a time. Hardware,
    NumPy or old frozen models are not invoked. No default idle history is made.
    """
    audit = SourceAudit(verify_sources)
    manifest_path = Path(manifest_path).resolve()
    manifest = audit.json(manifest_path)
    manifest_sha = audit.files[str(manifest_path)]["sha256"]
    row = execution_row
    require(row.get("status") == "measured" and row.get("cleanup_verified") is True and
            row.get("all_created_processes_exited") is True, "run must be measured with verified cleanup")
    require(row.get("source_inventory_sha256") == manifest["source_inventory_sha256"], "execution source inventory differs")
    job = row["job"]
    require(job["split"] in {"profile", "heldout", "diagnostic"}, "invalid run split")
    require(utc(row["started_at_utc"]) <= utc(row["ended_at_utc"]), "run chronology reversed")
    specs = validate_competitors(DEFAULT_COMPETITORS if competitors is None else competitors)
    normalized = Path(row["normalized_path"]).resolve()
    audit.hash(normalized, row["normalized_sha256"])
    sources = tail_field(normalized, "source_files")
    raw_bindings = tail_field(normalized, "raw_bindings")
    folder = normalized.parent
    execution_path = folder / "execution.json"
    recorded_execution = audit.json(execution_path)
    require(recorded_execution == row, "execution row differs from immutable per-job receipt")

    def bound(name):
        path = (folder / name).resolve()
        require(str(path) in sources, "missing raw source binding: " + name)
        return audit.json(path, sources[str(path)])

    def native(name):
        path = (folder / name).resolve()
        require(str(path) in sources, "missing native source binding: " + name)
        audit.hash(path, sources[str(path)])
        return read_native(path)

    binding = manifest["programs"][job["id"]]
    program = audit.json(binding["path"], binding["sha256"])
    require(str(Path(binding["path"]).resolve()) == raw_bindings["program"], "program raw binding differs")
    meta, calls, actions = native("main/result.json")
    events, numerical, ready = bound("events.json"), bound("numerical.json"), bound("main/ready.json")
    require(numerical.get("passed") is True, "independent numerical oracle receipt not passed")
    require(numerical.get("main", {}).get("passed") is True, "foreground independent oracle receipt missing")
    require(meta["process_id"] == row["process_id"] == ready["process_id"], "owned foreground PID differs")
    require(meta["role"] == "main" and meta["actions"] == program["actions"], "foreground action contract differs")
    require(meta["program_sha256"] == program["program_sha256"] == ready["program_sha256"], "program/numeric initialization differs")
    require(program.get("seed") == job["numeric_seed"], "numeric seed receipt differs")
    require(meta.get("diagnostic_only") is not True or job["split"] == "diagnostic", "diagnostic instrumentation cannot qualify a formal profile")
    require(events["go"] == meta["go_schedule"], "foreground external GO differs")
    clock = meta.get("clock", meta.get("native_clock"))
    frequency = integer(clock["frequency_hz"], "clock frequency", 1)
    clock_receipts = [clock,ready["clock"],events["go"]["clock"]]
    if meta.get("native_clock") is not None: clock_receipts.append(meta["native_clock"])
    for receipt in clock_receipts:
        require(receipt.get("clock_id") == clock["clock_id"] and receipt.get("frequency_hz") == frequency and
                (receipt.get("cross_process_monotonic_clock") is True or receipt.get("cross_process") is True),
                "foreground/READY/GO clock identity, scale or cross-process semantics differ")
    origin = integer(events["go"]["start_tick"], "planned GO origin")
    require(meta["ready_tick"] == ready["ready_tick"] <= origin and
            meta["sampling_begin_tick"] <= meta["sampling_end_tick"], "foreground READY/sampling chronology differs")
    pin_verified = None
    expected_pin = manifest["context"].get("main_thread_affinity_mask")
    if expected_pin is not None:
        before, after = bound("main-thread-affinity-before-go.json"), bound("main-thread-affinity-after-sampling.json")
        require(before["set_succeeded"] and before["requested_affinity_mask"] == expected_pin and
                before["main_thread_id"] == ready["main_thread_id"] == meta["main_thread_id"] and
                before["process_id"] == ready["process_id"] == before["owned_process_identity"]["process_id"] and
                before["owned_thread_identity"]["process_id"] == ready["process_id"] and
                before["owned_thread_identity"]["thread_id"] == ready["main_thread_id"], "main pin identity/request differs")
        require(ready["ready_tick"] <= before["pin_begin_tick"] <= before["pin_end_tick"] < origin and
                after["query_begin_tick"] >= meta["sampling_end_tick"], "main pin chronology differs")
        for observation in (before["observations"][-1], after):
            require(observation["status"] == "verified" and observation["observed_affinity_mask"] == expected_pin and
                    observation["observed_group"] == before["requested_group"] and
                    observation["thread_identity"] == before["owned_thread_identity"] and
                    observation["process_identity"] == before["owned_process_identity"], "main pin actual mask/lifetime differs")
        pin_verified = dict(endpoint_masks_verified=True,mask=expected_pin,continuous_residency_proven=False)
    convert = lambda tick: round(Fraction((tick - origin) * 1_000_000_000, frequency))
    require(manifest["context"].get("time_origin") == "planned_go_start_v1", "explicit planned-GO origin required")
    segments = _normalize_segments(events, origin, frequency, specs)
    calibration = _normalize_segments(events, origin, frequency, specs, calibration=True)
    by_call = defaultdict(list)
    for bi, iteration, index, left, right, elapsed, kind in actions:
        integer(index, "action occurrence index")
        require(index < len(program["actions"]) and kind in (None, program["actions"][index]["kind"]), "action kind/index differs")
        require(abs(elapsed - round(Fraction((right-left)*1_000_000_000,frequency))) <= 1, "action elapsed differs from QPC")
        by_call[(bi, iteration)].append((index, left, right))
    require(set(by_call) <= {(c[0], c[1]) for c in calls}, "orphan action intervals")
    invocations, previous = [], origin
    bare = meta["instrumentation"] in {"bare", "per_iteration_host_completed_work_only"}
    for bi, iteration, left, right, elapsed, _ in calls:
        require(meta["sampling_begin_tick"] <= left < right <= meta["sampling_end_tick"] and previous <= left and
                abs(elapsed-round(Fraction((right-left)*1_000_000_000,frequency))) <= 1,
                "foreground invocation order/duration differs")
        stamps = by_call[(bi, iteration)]
        require(len(stamps) == len(program["actions"]) or (bare and not stamps and job["split"] != "profile"),
                "profile needs complete actions; bare heldout alone may omit them")
        cursor = left
        for expected, (index, a, z) in enumerate(stamps):
            require(index == expected and cursor <= a <= z <= right, "action ordering/bounds differ")
            cursor = z
        require([s[0] for s in stamps] == list(range(len(stamps))), "action occurrence inventory differs")
        invocations.append(dict(observation_id=f"block:{bi}:iteration:{iteration}", begin_ns=convert(left), end_ns=convert(right),
            actions=[[index,convert(a),convert(z)] for index,a,z in stamps]))
        previous = right
    del by_call, actions, calls
    workers = {}
    children = {child["name"]:child for child in row["children"]}
    for spec in specs:
        role, name = spec["actor_role"], spec["name"]
        require(role in children and numerical.get(role, {}).get("passed") is True, "missing owned worker/numeric receipt")
        worker, intervals, worker_actions = native(role + "/result.json")
        require(worker["role"] == role and worker["process_id"] == children[role]["pid"], "worker owned identity differs")
        require(worker["program_sha256"] == meta["program_sha256"] and worker["actions"] == meta["actions"], "worker action contract differs")
        require(worker["clock"] == clock and worker["go_schedule"] == events["go"], "worker clock/GO differs")
        require(worker["execution_boundary"] == meta["execution_boundary"], "worker execution boundary differs")
        require(worker["ready_tick"] <= origin and not worker_actions, "worker readiness or bare instrumentation differs")
        normalized_calls, last = [], worker["sampling_begin_tick"]
        for bi, iteration, left, right, elapsed, decision in intervals:
            require(type(decision) is int and last <= left < right <= worker["sampling_end_tick"] and
                    worker["sampling_begin_tick"] <= decision <= left, "worker authorized-call chronology differs")
            requested = state_for_interval(segments, convert(decision), convert(decision))
            require(requested is not None and requested[name] == 1, "worker call was not authorized by request")
            require(abs(elapsed-round(Fraction((right-left)*1_000_000_000,frequency))) <= 1, "worker QPC duration differs")
            normalized_calls.append([convert(decision),convert(left),convert(right)])
            last = right
        workers[name] = dict(actor_role=role, process_id=worker["process_id"], target_action_index=worker["target_action_index"],
            work_scope=worker.get("worker_work_scope"), ready_ns=convert(worker["ready_tick"]),
            sampling_begin_ns=convert(worker["sampling_begin_tick"]), sampling_end_ns=convert(worker["sampling_end_tick"]),
            authorized_calls=normalized_calls, intervals_are_hardware_busy_time=False)
    inventory = token_inventory(program["actions"])
    contract = dict(program)
    for name in ("program_sha256", "seed", "initial_values"): contract.pop(name, None)
    if isinstance(contract.get("identity"),dict):
        contract["identity"] = {k:v for k,v in contract["identity"].items() if k != "initial_values"}
    result = dict(schema=SCHEMA, run_id=job["id"], split=job["split"], seed=job["numeric_seed"], process_id=row["process_id"],
        started_at_utc=row["started_at_utc"], ended_at_utc=row["ended_at_utc"], context=manifest["context"],
        context_sha256=digest(manifest["context"]), contract_sha256=digest(contract), manifest_sha256=manifest_sha,
        source_inventory_sha256=manifest["source_inventory_sha256"], original_normalized_sha256=row["normalized_sha256"],
        service_protocol_sha256=None if "service_response" not in manifest["protocol"] else digest(manifest["protocol"]["service_response"]),
        competitors=specs, token_inventory=inventory, action_inventory=program["actions"], clock=clock, origin_tick=origin,
        instrumentation="bare" if bare else "stamped", segments=segments, calibration_segments=calibration,
        invocations=invocations, workers=workers, numerical_verified=True,
        numerical_verification_scope="SHA-bound independent collector receipt; numeric oracle not rerun during extraction",
        source_files=audit.files, source_verified=bool(verify_sources), all_samples_retained=True, outliers_removed=0,
        extraction_inputs=dict(manifest_path=str(manifest_path),execution_path=str(execution_path)),
        initial_work_history_receipt=meta.get("initial_work_history"),
        main_thread_affinity=pin_verified,
        worker_history_scope="Authorized target-compute call intervals, not instantaneous utilization, CPU service or literal device queues")
    result["record_sha256"] = digest(result)
    return result


def validate_record(record, *, split=None, verify_sources=True):
    require(record.get("schema") == SCHEMA and (split is None or record["split"] == split), "run schema/split differs")
    require(record.get("record_sha256") == digest({k:v for k,v in record.items() if k != "record_sha256"}), "run record digest differs")
    require(record["numerical_verified"] and record["all_samples_retained"] and record["outliers_removed"] == 0, "run evidence/numerics differ")
    require(record["source_verified"] or not verify_sources, "unverified sources cannot become verified during validation")
    if verify_sources:
        audit = SourceAudit()
        for name, receipt in record["source_files"].items(): audit.hash(name, receipt["sha256"])
        require(record.get("extraction_inputs") is not None, "raw extraction recipe required for verified derivation")
        inputs = record["extraction_inputs"]
        rebuilt = extract_job(inputs["manifest_path"], audit.json(inputs["execution_path"]),
                              competitors=record["competitors"],verify_sources=True)
        require(canonical(rebuilt) == canonical(record), "source-bound extraction does not reconstruct")
    return record
