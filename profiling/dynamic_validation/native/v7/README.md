# H2D primitive stages and host progress diagnosis

This additive diagnostic keeps the frozen v1–v6 results unchanged. It does not
fit a target-level multiplier or declare the remaining variability random.

## What can be identified

The retained H2D path is a host-buffer `ggml_backend_tensor_copy` followed by
CUDA buffer `set_tensor`. The bound original sources select
`cudaMemcpyAsync(..., cudaStreamPerThread)` and then synchronize that same
per-thread default stream. The new primitive explicitly executes those two
APIs on the original backing buffers and retains the extra GGML GPU/CPU
synchronization calls. It is a separately named execution boundary, not an
instrumented original GGML action. The original action remains a control.

Each primitive copy records ordered QPC intervals for the copy API, inner
synchronization and extra GPU/CPU synchronization. Their union plus the gaps
closes the host wall interval. These are inclusive responses; none is called
active CPU work or an exact enqueue-acceptance instant.

An optional pair of CUDA events brackets the copy in that explicit stream.
The elapsed span is a second, overlapping observation. It is not a CUPTI DMA
activity interval, has no asserted QPC placement, and cannot be added to the
host synchronization time. Event creation and elapsed-time queries are outside
target intervals. The event observer must pass its own neutrality comparison
before its values can be considered for parameter calibration.

The probe records actual source memory category, device, buffer names, source
dispatch receipts and the explicit primitive stream handle. There is no public
GGML get-stream API here; the original bare action's internal stream is not
claimed to have been dynamically observed. Unknown source/dispatch evidence
rejects preparation rather than substituting a compute stream.

## Small, independent protocol

Use `profiling.dynamic_validation.h2d_v7` to prepare a new campaign only after
an explicit private build exists. It does not build or launch on import.

| Input | Calibration | Independent acceptance |
| --- | --- | --- |
| Copy bytes | 128 KiB, 512 KiB, 2 MiB | 256 KiB, 1 MiB |
| Requested idle | 0, 20 ms | 2, 10 ms |
| Host memory | Registered and pageable, separately | Same categories, fresh contexts |
| Mathematical input | Retained profile program | Distinct frozen program seed |
| Contexts | Two per memory category | Two per category |
| Observation modes | GGML bare, primitive QPC, primitive QPC + events | Same modes |

Twelve trials balance size and condition positions; the second fresh context
reverses the order. Each burst contains sixteen completed copies. Report first
copy and suffix copies 1–15 separately, with early six and late six trials.
Every burst follows the complete untimed original prime. PCIe queries occur
before that prime and after the burst, never between prime and first copy.
Their age is retained. Agreement of these boundary snapshots cannot establish
constant link state inside a copy, and link generation is not available
bandwidth. No theoretical generation ratio is used.

`protocol()` predeclares numeric/source/clock checks, a 5% observer discrepancy
limit for every first/suffix/burst cell, 10% early/late and cross-context base
support limits, and independent 20% mean-error plus 10% measured-CV limits.
An unsupported observer or base stops parameter export; it is not corrected
with the heldout mean. Root must separately authorize each split and freeze
the model and all heldout predictions before independent acquisition. Missing
state support remains unknown. Existing failures are preserved.

## External competition: reused instrumentation

`runtime_stage_records.py` partitions actual v6 Attention callbacks into
ordered API responses and prefix/intercall/suffix gaps. API coordinates contain
actual call metadata and an explicitly supplied stream role; process-local
handles, strategy IDs and target-total durations are not cost coordinates.
Nested or overlapping callbacks are rejected for serial-stage modeling.
`runtime_stage_profile.py` forms profile-only local stage observations and exact
four offered-actor state cells. Crossing observations are retained separately.
It refuses mixing affinity scopes and never certifies its own observer gate.
Predictions must obtain state from their own recursively predicted clock.

The existing owned .NET/NtQuerySystemInformation helper can supply bounded 2 ms
snapshots on a separate verified physical core. Join its raw state and wait
reason codes to long API intervals with `snapshot_evidence`. These are sample
counts, not exact running/ready/waiting durations or available FLOPS. Raw thread
cycles are not converted to active time. Precise scheduler intervals require
the separately authorized existing ETW CSwitch/ReadyThread path; even a Wait
state does not identify a specific driver function as the cause.

The new phase labels belong to profiling. Execution-state evolution and
duration calculation remain in the existing modeling response/submission
framework; no parallel performance model is introduced here.

## Build boundary

The native CMake target is `dynamic-h2d-primitive-v7`, using explicit
`LLAMA_ROOT`, `GGML_BUILD_DIR`, `STEADY_GENERATED_DIR`, `CUDA_ROOT` and
`CUPTI_ROOT` paths. The last is a header dependency inherited from the retained
common include; this target neither links nor initializes CUPTI. Use the
existing x64 Release toolchain and runtime. Preparation and compilation are
not measurements. Software tests do not validate actual device events or DMA.
