# Cooperative VRAM reclaim execution contract

Implemented on 2026-10-12 in the existing tensor execution sessions. This is an
opt-in simulator protocol; no operating-system or physical calibration claim is
made. Existing forced `EnvironmentEvent` allocation semantics remain unchanged.

## Public interface

```python
from llm_infer_model.tensor.environment import (
    EnvironmentReclaimRequest, ControlCosts, SimulationSession,
)
from llm_infer_model.tensor.runtime import ControlCommand, ExecutionCommand, dispatch_command

session = SimulationSession(
    scenario,
    reclaim_requests=(EnvironmentReclaimRequest("app", 100, 256, 160),),
    control_costs=ControlCosts(
        ack_ns=5, decline_ns=2, decision_ns=10, source="synthetic_example",
        timing_scope="dedicated_serial_control_resource",
    ),
)
```

`EnvironmentReclaimRequest(id, t_notice_ns, target_external_bytes, deadline_ns,
domain="vram")` contains an **absolute external allocation target**. Nonnegative
integer fields exclude booleans. IDs must be unique, notice windows ordered and
strictly non-overlapping. This first protocol supports one pending VRAM request;
disjoint later requests are supported without arbitration. The target must
exceed actual external allocation when the notice appears. An invalid target is
recorded as `invalid_target`, never charged to memory.

When requests or control commands are used, `ControlCosts` must be explicitly
provided. Zero is allowed as a declared experiment assumption; absence is not
silently converted to zero. `source` identifies that assumption or measurement.

The same `reclaim_requests` and `control_costs` keywords are accepted by
`ServiceSession` and `create_runtime`.

`ExecutionCommand.action` accepts either an existing tensor `Action` or a
`ControlCommand`. `dispatch_command` routes the latter to `submit_control`.
Transfer engines remain applicable only to tensor `COPY`.

| Control command | Required fields | Meaning |
| --- | --- | --- |
| `ACK_RECLAIM` | `request_id` | Pay `ack_ns`, then check deadline and complete physical capacity; atomically accept target only if both pass |
| `DECLINE_RECLAIM` | `request_id`, nonempty `reason` | Pay `decline_ns`, explicitly record an unmet request |
| `DECIDE` | optional public `request_id`/`reason` | Pay `decision_ns` on the dedicated serial controller |
| `WAIT_UNTIL` | integer `until_ns > current time`, optional public `request_id`/`reason` | Wait up to a caller-chosen public time; earlier runtime boundaries interrupt the wait and return control to the policy |

`ControlCommand(kind, request_id=None, until_ns=None, reason=None)` is not a
tensor action. Its events are written to a separate control ledger and are
never counted as tensor copies or transfer bytes. An unknown/future request ID,
duplicate response, or invalid timer is rejected and audited without changing
actual external allocation. A rejected command returns `SessionStep(ok=False)`.
A syntactically valid ACK launch returns `ok=True`; its eventual resource result
is recorded when it completes, including capacity/deadline rejection.

## Time and resource ownership

The controller is a **dedicated serial synthetic service**. It is independent of
CPU compute availability and does not model CPU utilization, host-thread
scheduling, allocator internals or graph rebuild implementation. Existing tensor
work proceeds while a control operation runs. During non-wait control service,
the policy must advance rather than launch further tensor/control operations.
This explicitly charges decision/response delay without pretending it is an
accurate operating-system model.

`DECIDE` is charged only when explicitly submitted. The runtime does not infer
how often a policy decides or silently amortize this cost over tensor actions.
`EVICT` remains a zero-time logical release under the existing tensor model;
allocator/free/synchronization cost is not claimed to have been measured. The
control cost can represent a stated aggregate assumption, but its source and
scope must remain visible in experimental results.

At a boundary: existing tensor completions run first, a due control completion
runs next, then same-time actual environment changes and new request notices.
The deadline at equality leaves one final response window. The next `ADVANCE`
seals an unresponded request before progressing beyond it. Thus zero-cost ACK at
the deadline can succeed, while ACK with positive cost launched at the deadline
misses it. A control completing exactly at the deadline can succeed. No future
recovery time is exposed by this protocol.

`WAIT_UNTIL` is an upper-bound timer, not a forced blind sleep. Tensor completion,
environment changes, request notice or deadline can wake it earlier. Its event
records actual elapsed time and `interrupted` when appropriate. The policy must
choose a new timer if it still wants to wait; no private event timestamp is
returned before that event becomes visible.

## Memory safety and admission

Notice does not change physical allocation. While pending, the old external
allocation remains charged. New GPU tensor/copy/output reservations and newly
reserved GPU workspace are prohibited, even when a fixed slot was already
reserved. In-flight work continues and CPU/DRAM work can proceed legally.

The model publicly supplies `safe_weight_release_candidates`: immutable leaf
weights with READY VRAM and DRAM copies and a currently legal existing EVICT
transition. Tensor identity is version identity for these immutable weights.
The check reuses the existing read-lock, reservation and reachability rules.
During pending reclaim, explicit VRAM eviction is limited to these clean backed
weights. Mutable KV, outputs, activations, in-flight destinations, GPU-read-locked
weights and weights with no DRAM backing cannot be released under this protocol.
It does not implement dirty KV writeback, task cancellation or partial migration.

A CPU task reading the DRAM backing does not lock an untouched VRAM copy. It is
therefore safe to release that GPU copy; the CPU task keeps its DRAM input,
output reservation and workspace while CPU service is paused.

Fixed-slot owners may be safe logical candidates, but evicting them returns
**zero physical bytes**. The slot's capacity remains in runtime reserve. Policies
must use physical allocation size, not payload size, when selecting a release
set. ACK always checks the complete physical ledger:

```text
runtime reserve + tensor/copy/output allocations + workspace + target external
    <= VRAM capacity
```

Only completed successful ACK commits the absolute target. Forced overcommit
without notice still gives terminal `external_memory_conflict`; no automatic
eviction or conversion into a request is performed.

## Observation and result additions

New immutable fields appended with backward-compatible defaults:

```text
Observation.pending_reclaim: ReclaimObservation | None
Observation.safe_weight_release_candidates: tuple[str, ...]
Observation.reclaim_records: tuple[ReclaimRecord, ...]
Observation.control_running: ControlRunningObservation | None
Observation.control_costs: ControlCosts | None
SessionResult.reclaim_records: tuple[ReclaimRecord, ...]
SessionResult.control_events: tuple[ControlEvent, ...]
```

`ReclaimObservation` has `id, domain, target_external_bytes, deadline_ns, status,
t_notice_ns`. Public request records contain only notices already exposed:
`id, domain, target_external_bytes, t_notice_ns, deadline_ns, status,
responded_ns, reason`. Status is `pending`, `accepted`, `declined`,
`capacity_rejected`, `deadline_rejected` or `invalid_target`. No future notice
queue, actual future recovery, or private routing trace is disclosed.

Control records contain `index, kind, request_id, start_ns, end_ns, cost_ns,
status, reason, source_index`. In-flight control exposes its own declared end
time, which follows from a command and public cost, not from the environment.

An unmet request does not terminate inference. Evaluation must report inference
completion and request fulfillment separately: `declined` or rejected requests
cannot be called successful resource-sharing policies merely because latency is
shorter. Published pending requests also prevent `finished=True`; notices that
would occur strictly after inference's goal are not drained, matching existing
future environment behavior. Experiments must specify their lifecycle horizon
if they require resource changes beyond an inference run.

The runtime supports deterministic replay from explicit commands and input
contracts. Native serialized checkpoint handling belongs to the mapping/M2
adapters, which must include request/control input and command histories.

## Acceptance evidence

New hand-calculated suite: `modeling/tests/test_reclaim_runtime.py`.

```powershell
$env:PYTHONPATH = 'modeling'
& modeling/.venv/Scripts/python.exe -m unittest discover -s modeling/tests -p test_reclaim_runtime.py -v
```

26 tests passed on 2026-10-12. They cover delayed ACK accounting, safe eviction
and paid reload, GPU read locks, unbacked and nonweight protection, fixed-slot
negative case, deadline equality/late response, blocked output/workspace/CE
admission, in-flight transfer progress, identical future-prefix observations,
timer interruption, explicit control overhead under CPU blackout, duplicate
responses, decline accounting, strict inputs, unchanged forced-overcommit,
factory forwarding, published/future request lifecycle, multiple absolute
targets, exact replay and paused CPU memory ownership.

Existing targeted environment and services suites passed: 22 + 27 tests. These
counts are targeted evidence, not a whole-project accuracy claim. All costs in
these tests are synthetic. Real-hardware validation remains separate.
