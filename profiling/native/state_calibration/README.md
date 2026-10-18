# State-aligned native diagnostic

This wrapper reuses the exact compiled ModelPlan GGML executor without editing
historical sources. It compares repeated isolated execution of one COMPUTE action
against complete mixed serial replay using identical program data and prebuilt
graphs. A sampler thread in the parent Python runner observes the same monotonic
clock; the measured workload runs in a separate native child process.

```text
state-calibration-probe PROGRAM.json isolated|mixed TARGET_ACTION_INDEX WARMUP BLOCKS MIN_BLOCK_MS stamped|bare OUTPUT.json
```

Optional `STATE_PROBE_READY_FILE` names a caller-owned new file. The probe creates
it exclusively after complete validation and warmup, immediately before the
first timed block. It contains clock metadata, PID, target, mode, and
`ready_tick`. Readiness does not itself prove concurrent execution; use actual
iteration intervals in the completed output to audit overlap. Optional
`STATE_PROBE_STOP_FILE` names a distinct caller-owned path that must not exist at
startup. When the caller creates it, the probe stops gracefully after its next
complete invocation. Existence is checked outside work timers. A stopped final
block may be shorter than the requested minimum and is explicitly marked by
`graceful_stop_requested`; completed raw intervals and final numerical checks are
still emitted. The probe never creates, overwrites, or deletes the stop file.

Optional `STATE_PROBE_FIXED_ITERATIONS` requests an exact positive decimal count
of invocations per block (at most 200000). In this mode `MIN_BLOCK_MS` is checked
and reported, never used to silently extend sampling. Every block reports
`minimum_elapsed_satisfied` and `fixed_iteration_count_satisfied`; the top-level
`iteration_policy`, `requested_iterations_per_block`, and `full_fixed_count`
record the protocol. A short block remains visible for downstream alignment or
envelope rejection. Graceful stop can truncate a fixed-count block and is
explicitly reported. When the variable is absent, blocks use the original
minimum-wall-time policy. Every individual invocation is retained in both modes;
block averages must not replace invocation-level variability gates.

`STATE_PROBE_EXECUTION_BOUNDARY` optionally selects one of three boundaries. Its
absence defaults to `legacy_double_sync`; unknown values fail explicitly.

| Boundary | Selected GPU action | Diagnostics |
| --- | --- | --- |
| `legacy_double_sync` | The original executor action is unchanged: synchronous public `graph_compute`, then another explicit backend synchronization. | Original outer action intervals; no CUDA events. Internal enqueue/synchronization split remains unknown. |
| `async_one_sync` | Public `graph_compute_async`, then one backend synchronization. | Host enqueue and synchronization intervals. |
| `async_events` | Timing events around `graph_compute_async`, then one backend synchronization. | Matched host intervals and the CUDA event interval. |

Only the selected GPU COMPUTE action changes. Every other action and independent
validation replay uses the historical executor. Alternate boundaries are
diagnostic experiments and must not enter legacy calibration cells. Compare
`async_events` with `async_one_sync` to assess event instrumentation overhead.
Each block's `target_diagnostics` stores all alternate-boundary invocations.
For legacy, that array is empty and the original `action_intervals` provide the
outer host duration. Timestamps and elapsed-time query costs remain measured.
Alternate modes also add QPC subtimestamps and a diagnostic JSON record. The
record is constructed after the outer target stamp but within the invocation,
so it can affect following work and later iterations. The legacy/async
comparison therefore does not isolate the cost of one extra synchronization;
that requires a legacy control with matched diagnostic instrumentation.

Timing events are created once outside sampling using `cudaEventDefault`.
The pinned `ggml_backend_event` wraps their raw CUDA handles; the actual GGML
CUDA callback records them on its actual execution stream. The wrapper is never
freed through GGML; its own CUDA handles are destroyed after sampling. No CUDA
context layout or default-stream assumption is used. Events are read only after
the single backend synchronization, without an extra event synchronization.
The event interval includes stream work and host launch gaps between its events,
and excludes queueing before the first event executes. It is **not summed kernel
activity** and cannot establish which kernel ran.

Without a fixed invocation count, each block lasts at least `MIN_BLOCK_MS`
(minimum 200 ms). Fixed-count blocks retain their actual duration even when
shorter than this minimum. Warmup is separate.
`stamped` records every action's host completed-work interval; `bare` records only
whole invocation intervals. Both synchronize each actual GGML compute/copy, as
required by the historical executor. Compare both modes to measure the cost of
per-action instrumentation. Mixed replay restores initial inputs before each
invocation; restoration lies outside invocation timers but inside block wall
time. JSON interval recording is included in invocation timers. This is an
instrumented synchronous execution boundary, not a production CUDA graph.

Clock metadata uses Windows QPC ticks/frequency, or POSIX CLOCK_MONOTONIC
nanoseconds. A telemetry sample's query interval must overlap an execution
interval before making an association. GPU utilization or clock readings are
observations of an interval, not proof that a particular kernel caused a delay.

The output contains every final port and the target's physical GGML nodes,
shapes, backend, requested precision, known workspace, and exact input-byte
immutability checks. It checks timed outputs against an untimed native replay.
An external independent NumPy/scalar oracle remains necessary. CUDA private
workspace, physical capacity, loaded DLL resolution, and kernel choice remain
unproven.

`build.py` requires explicit MSVC, SDK, CUDA toolkit, llama.cpp source/build, and runtime paths.
It writes only a fresh build directory, records exact sources/header/library/DLL
hashes, and rejects modified historical helper bytes. Use the explicitly bound
runtime directory first in `PATH`; the builder records the requested runtime,
not the dynamic loader's resolved paths.
