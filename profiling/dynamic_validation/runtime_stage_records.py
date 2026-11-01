"""Ordered target API responses and intercall gaps for the common g framework.

These are inclusive host responses, not active CPU, accepted GPU work, or DMA.
Opaque handles identify an observed stream inside one process only. Callers
must explicitly bind their stable stream roles before comparing processes.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json


def require(value, message):
    if not value: raise ValueError(message)


def coordinate(metadata, stream_roles):
    require(isinstance(metadata, dict) and isinstance(metadata.get("name"), str), "API semantic metadata missing")
    require(metadata.get("parameters_available") is True, "API parameters unavailable")
    handle = metadata.get("stream_handle_opaque_process_local")
    require(type(handle) is int and handle in stream_roles and isinstance(stream_roles[handle], str) and stream_roles[handle],
            "observed stream has no explicit cross-process role; do not infer ordinary stream equals PTDS")
    result = {key: metadata.get(key) for key in ("name", "symbol_name", "grid", "block", "shared_memory_bytes", "bytes", "copy_kind_cudaMemcpyKind")}
    if result["name"].startswith("cudaLaunchKernel"):
        require(isinstance(result["symbol_name"], str) and result["symbol_name"] and metadata.get("symbol_truncated") is False,
                "complete actual kernel symbol required for semantic coordinate")
        require(all(isinstance(result[k], list) and len(result[k]) == 3 and all(type(x) is int and x>0 for x in result[k]) for k in ("grid", "block")), "actual launch geometry missing")
        require(type(result["shared_memory_bytes"]) is int and result["shared_memory_bytes"]>=0, "actual shared memory bytes missing")
    elif result["name"].startswith("cudaMemcpyAsync"):
        require(type(result["bytes"]) is int and result["bytes"]>0 and type(result["copy_kind_cudaMemcpyKind"]) is int,
                "actual copy extent/direction missing")
    elif not result["name"].startswith("cudaStreamSynchronize"):
        raise ValueError("API family outside retained callback protocol")
    result["stream_role"] = stream_roles[handle]
    return result


def semantic_key(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def coordinate_sha256(value):
    return hashlib.sha256(semantic_key(value).encode("utf-8")).hexdigest()


def index_callbacks(callbacks):
    """Build once for all phases; avoid repeatedly scanning the full trace."""
    require(callbacks.get("all_selected_calls_retained") is True and callbacks.get("activity_driver_kernel_collection_enabled") is False,
            "callback integrity/scope not qualified")
    grouped = defaultdict(list)
    for ordinal, row in enumerate(callbacks["callback_records"]):
        require(row.get("ordinal") == ordinal and type(row.get("phase_id")) is int, "callback inventory order/phase malformed")
        grouped[row["phase_id"]].append(row)
    require(callbacks.get("record_count", len(callbacks["callback_records"])) == len(callbacks["callback_records"]), "callback record count differs")
    return dict(grouped)


def ordered_stages(callbacks, phase, frequency_hz, *, stream_roles, phase_records=None):
    """One completed action. Overlapping callbacks cannot become a serial graph."""
    require(type(frequency_hz) is int and frequency_hz>0, "actual QPC frequency required")
    require(callbacks.get("all_selected_calls_retained") is True and callbacks.get("activity_driver_kernel_collection_enabled") is False,
            "callback integrity/scope not qualified")
    require(type(phase.get("phase_id")) is int and phase["begin_tick"]<phase["end_tick"], "completed target action required")
    metadata = callbacks["metadata"]
    rows = list(phase_records) if phase_records is not None else [x for x in callbacks["callback_records"] if x["phase_id"] == phase["phase_id"]]
    require(bool(rows), "target action has no selected API records")
    require(all(x["phase_id"] == phase["phase_id"] for x in rows), "foreign phase callback supplied")
    rows.sort(key=lambda x:(x["enter_tick"], x["ordinal"]))
    result, previous_tick, previous_coordinate = [], phase["begin_tick"], None
    for index, row in enumerate(rows):
        require(row.get("exit_received") is True and row.get("return_status") == 0, "successful paired API return required")
        require(type(row.get("metadata_index")) is int and 0<=row["metadata_index"]<len(metadata), "API coordinate index invalid")
        begin, end = row["enter_tick"], row["exit_tick"]
        require(previous_tick<=begin<=end<=phase["end_tick"], "nested/overlapping/out-of-phase API responses cannot be serialized")
        current = coordinate(metadata[row["metadata_index"]], stream_roles)
        gap = dict(kind="host_gap", position="prefix" if index==0 else "between_APIs", previous=previous_coordinate, next=current)
        result.append(dict(coordinate=gap, begin_tick=previous_tick, end_tick=begin, response_ns=(begin-previous_tick)*1e9/frequency_hz,
                           work=1, unit="inclusive_completed_response_ns", source="observed_QPC_gap"))
        result.append(dict(coordinate=dict(kind="host_API_response", API=current), begin_tick=begin, end_tick=end,
                           response_ns=(end-begin)*1e9/frequency_hz, work=1, unit="inclusive_completed_response_ns",
                           source="selected_runtime_callback_enter_exit", correlation_id_diagnostic_only=row["correlation_id"]))
        previous_tick, previous_coordinate = end, current
    result.append(dict(coordinate=dict(kind="host_gap", position="suffix", previous=previous_coordinate, next=None), begin_tick=previous_tick,
                       end_tick=phase["end_tick"], response_ns=(phase["end_tick"]-previous_tick)*1e9/frequency_hz,
                       work=1, unit="inclusive_completed_response_ns", source="observed_QPC_gap"))
    ticks = sum(x["end_tick"]-x["begin_tick"] for x in result)
    require(ticks==phase["end_tick"]-phase["begin_tick"], "serial response+gap partition does not close")
    for ordinal, stage in enumerate(result):
        role = "api" if stage["coordinate"]["kind"] == "host_API_response" else "gap"
        stage.update(id=f"{role}_{ordinal:04d}", role=role, coordinate_sha256=coordinate_sha256(stage["coordinate"]))
    return dict(schema="target-ordered-runtime-response-stages-1", phase_id=phase["phase_id"], stages=result,
                selected_API_calls=len(rows), serial_stages=len(result), host_QPC_partition_closes=True,
                complete_attention_physical_GPU_work_identified=False, API_response_is_active_CPU=False,
                internal_GPU_work_added=False, parameters_qualified=False,
                source_phase_times_for_profile_labels_only=True, prediction_reanchors_to_observed_times=False)


def core_scope(native):
    """Never pool same-core and other-core conditions by a global CPU label."""
    target = native["target_thread_affinity"]["observed_mask"]
    actor = native["actors"]["cpu"]["thread_affinity"]["observed_mask"]
    require(type(target) is int and type(actor) is int and target>0 and actor>0, "observed target/actor affinities missing")
    return dict(target_thread_mask=target, CPU_actor_thread_mask=actor, logical_affinity_overlap=bool(target&actor),
                physical_core_overlap="caller must bind verified topology; logical-mask disjointness alone is insufficient",
                global_CPU_utilization_is_scope=False)


def snapshot_evidence(index, begin, end, *, actor_id, target_thread_id):
    """Read-only snapshot counts inside an API; never duration reconstruction."""
    states, waits, providers = Counter(), Counter(), Counter();unknown = 0;count = 0
    for observation in index.contained_states(begin, end):
        actor, thread = observation["actor"], observation["thread"]
        if actor["actor_id"] != actor_id or thread["thread_id"] != target_thread_id: continue
        count += 1;provider = thread.get("thread_state_provider", "unknown")
        providers[provider] += 1
        # Keep provider namespace and raw codes. Managed/native enums are not interchangeable.
        key = semantic_key(dict(provider=provider, raw_code=thread.get("state_code"), name=thread.get("state_name")))
        states[key] += 1
        if thread.get("state_name") == "Wait":
            wait = semantic_key(dict(provider=provider, native_code=thread.get("native_wait_reason_code"), native_name=thread.get("native_wait_reason_name"),
                                     managed_code=thread.get("wait_reason_code"), managed_name=thread.get("wait_reason_name")))
            waits[wait] += 1
        unknown += bool(thread.get("unknown")) or thread.get("state_name") is None
    return dict(snapshot_count=count, raw_state_counts=[dict(json.loads(k), count=v) for k,v in states.items()],
                raw_wait_reason_counts=[dict(json.loads(k), count=v) for k,v in waits.items()], provider_counts=dict(providers),
                unknown_snapshots=unknown, exact_state_durations=False, snapshot_counts_are_time_fractions=False,
                driver_blocking_cause_identified=False, useful_work_vs_runtime_spin_identified=False,
                target_inclusive_diagnostic_only=True, forecast_input=False)
