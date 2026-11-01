# Device-state response calibration — 21 October 2026

## Ownership and scope

Performance equations belong to `modeling/llm_infer_model/performance`.
`conditioned_response.py` converts an exact, scoped completed-action response
table and an exogenous state trajectory into invocation work for the existing
`performance.evaluate` event engine. There is no second timing integrator,
total-layer regression, DNN, or automatic conversion of utilization to speed.

`profiling/state_response` owns raw native records, source and timing-boundary
checks, independent numerical receipts, sample statistics, parameter freezing,
and comparison with subsequently collected observations. Its predeclared
prediction candidates all call the same modeling implementation.

The response quantity is inclusive host-completed action wall time, including
backend submission, device work, and waits within that action. It is **not** a
measurement of pure GPU FLOP/s. Disjoint action-before gaps, controls, and the
invocation tail are explicit response stages. Internal host/API/GPU intervals
must not be added to those inclusive prices again.

For a stage, the declared approximation is

`d(reference invocation progress)/dt = 1 / completed_stage_duration(state(t))`.

Changing state retains already-completed work. An explicitly supplied zero
service rate pauses work; an unknown state is rejected. A measured statistical
zero-duration cell uses a separately tagged zero-cost approximation; it does not
prove physically instantaneous work or replace an unknown with zero.

## State inputs and missing mechanisms

The declared state space has four numeric coordinates:

| Coordinate | Meaning |
|---|---|
| `cpu_replay` | Request 0/1 for an owned persistent CPU target-replay process |
| `gpu_replay` | Request 0/1 for an owned persistent GPU target-replay process |
| `host_memory_bytes` | Owned page-touched allocation held through the trajectory |
| `device_memory_bytes` | Owned explicit CUDA allocation held through the trajectory |

Requests are known before foreground durations are observed. They are not
instantaneous resource occupancy percentages. CPU/GPU total counters and clocks
are diagnostic observations only. Allocated bytes do not automatically imply a
bandwidth multiplier. Background contexts, residual submitted work, host
scheduling, and recovery can make the memoryless response assumption fail.
Independent transition tests must establish its useful scope.

The model now also accepts four named request-response delays: CPU start, CPU
stop, GPU start, and GPU stop. Profiling estimates these from **profile worker**
QPC intervals relative to predeclared request edges, independently of foreground
latency. Modeling applies them to the full GO-relative request history, using an
explicit initial idle state; opposite requests cancel pending edges. No heldout
worker activation/completion timestamp enters a prediction. The proxy describes
worker invocation busy brackets, not pure device activity or an OS scheduler.

The native diagnostic variants separately retain main-thread/process CPU time,
unconverted CPU cycles, submission/synchronization wall intervals, and actual
GGML-stream CUDA event spans. CPU time is quantized; stream spans include launch
gaps and queueing. These overlapping quantities are not additive physical
components. Diagnostic modes never supply formal response prices.

## Independent experiment protocol

The first geometry is the public-compiled H128/I256/q16 F32 Dense layer with
supported Attention stages and FFN partitioning. It uses the unchanged strict
F32 runtime, two CPU threads, materialized tensor ports, explicit serial copies,
prebuilt graphs, and disabled CUDA graphs. It is not a production-size serving
benchmark. Native outputs are compared with independently computed NumPy ports.

The planned allocation grid has four combinations using 0 or 64 MiB per domain.
Only **m0, with zero owned DRAM/VRAM allocation, has been measured** in the
completed independent experiment below. The other cohorts are not hardware
validation results. Within a measured cohort there are three independent
profile foreground processes and three fresh heldout foreground processes.
Every foreground has two separately owned,
preinitialized target-replay actors. The same foreground remains alive while
CPU/GPU requests change; contexts are not recreated at each phase boundary.

Profile trajectories contain each CPU/GPU request combination, with a
predeclared 150 ms settling window excluded from parameter estimation. All
original observations, including those settling windows, remain retained.
Heldout trajectories have independent numerical seeds, different orders, and
shorter dwell times. They retain all complete invocations, including operations
crossing state changes. The heldout launch barrier follows parameter freezing.

Four predeclared candidates share the exact same frozen per-action table:

1. Dynamic state integration.
2. Constant no-owned-CPU/GPU-work state, with the actual fixed allocation bytes.
3. The state at invocation entry, held constant for the entire invocation.
4. Dynamic integration with four profile-only worker start/stop response delays.

The fourth candidate is the declared primary in development-v3-m0 and
development-v4-m0; it was not
chosen after inspecting heldout errors. The original instantaneous predictions
remain available for comparison.

Predictions are conditional on each externally timed invocation start; heldout
internal action durations are not supplied to the predictor. This does not
establish a free-running rollout of all future invocation launch times.

All actors announce that sampling is complete before any actor performs final
tensor reads, validation, serialization, or context teardown. Memory remains
held through a predeclared one-second end-of-schedule support window. Only
caller-owned finite processes are stopped; user applications, global timers,
device clocks, and power settings are not changed.

## Acceptance and evidence

Results must separately report point prediction error, source stability,
heldout stability, known-state coverage, state-crossing error, recovery error,
and source/instrumentation compatibility. Original failed gates and historical
campaigns remain unchanged. A favorable median alone is not complete hardware
accuracy acceptance; a new stamped-versus-bare compatibility control is needed
for the new boundary before a complete qualified claim.

Native startup failures in `development-v1` are retained: the first pilot
rejected the 117-action program because its initial fixed record allocation
exceeded the declared two-million-record envelope. No GO/timed foreground work
was produced. The revised bound derives the maximum iteration count from the
actual action count, preserves equal bare/stamped buffers, and still fails
explicitly on exhaustion. Superseded native source bytes remain archived.

Hardware results below come from immutable, independently collected artifacts.
Successful software tests alone are not accuracy evidence. Physical
workspace/capacity, arbitrary application interference,
unseen-state extrapolation, concurrent streams, and mid-trajectory allocation
transitions remain outside this first measured scope.

## Completed development-v2 measurement

This pilot deliberately covered only the no-owned-allocation cohort, four
CPU/GPU request states, and one complete small Dense strategy. Profile dwells
were 400 ms, with 50 ms excluded from estimation for settling. Three successful
profile processes supplied 389 invocations. A fourth profile attempt failed
before READY/GO and is retained; its separately declared engineering retry
succeeded. Three fresh bare heldout processes supplied 117 invocations. Every
successful foreground and background actor passed independent NumPy port checks.

The frozen artifact and heldout barrier precede every heldout launch. The metric
below is the median of each process's median **per-invocation absolute relative
error**, giving each process equal weight. It differs from the earlier 14-case
single-layer study's error of pooled median durations.

| Frozen-table prediction | Process-balanced median ARE | Pooled P90 ARE | Worst invocation ARE |
|---|---:|---:|---:|
| Dynamic request-state integration | 9.29% | 27.28% | 303.76% |
| No owned CPU/GPU work; same held bytes | 24.62% | 86.07% | 94.77% |
| Entry request state held constant | 9.29% | 33.28% | 930.79% |

All 117 calls have known state support. The 106 within-state calls have 8.90%
median ARE; the 11 crossing calls have 21.59%. Complete accuracy acceptance is
false. Source instability and missing stamped/bare compatibility evidence are
not repaired by a favorable median.

An independent profile-only CPU audit identified a primary parameter problem:
in request state `CPU=1,GPU=1`, projection had only `[2,3,1]` settled samples per
process, and QK had `[3,2,1]`. Some FFN stages lacked a third process entirely.
FFN-up/down process-median CV was 0.938/0.836. In state `0/0`, QK/up/down had
dozens of samples per process and process-median CV about 0.005/0.012/0.002.
This is insufficient evidence for a stable busy-state price.

For the worst call, prediction was 20.08 ms versus 4.97 ms observed. Subsequent
CPU FFN prices contributed 14.89 ms; the modeled GPU PV contributed only 2.31 ms.
Another large error occurred entirely within one state. Consequently the tail
cannot be attributed solely to missing GPU phases or boundary integration.

Read-only Windows topology evidence also established that the old affinity
mask `3` names two SMT logical processors on **one physical core**. Foreground
and CPU background each used two GGML threads on that same core. This is a
strong contention setup and does not by itself identify each long-tail cause.
Profile worker start lags were 0.23–2.54 ms; in-flight stop tails were
0.05–9.26 ms. Those lags motivate an explicit model, but do not demonstrate that
lag is the dominant source of error.

All prior raw results and a verified snapshot of their original source bytes
are retained under `profiling/state_response/results/development-v2`. The CPU
audit is in `diagnosis/cpu-profile-v1`. A newer source version must not rewrite
these old manifests to make them appear current.

## Development-v3 independent validation scope

The next source-bound campaign uses 2000 ms profile dwells, a 150 ms settling
window, and a minimum of 10 samples per eligible process/action/state cell.
Unstable or undersampled numerical estimates remain visible and unqualified.
Foreground and CPU worker affinity mask `5` uses two distinct physical cores;
GPU worker mask `48` uses a disjoint third physical core. No global machine
configuration is changed. This context is different from development-v2, so a
cross-campaign difference cannot isolate the contribution of the new mechanism.

The new primary candidate is declared **before collection** as dynamic response
with the four profile-only worker delays. Instantaneous dynamic response, no
owned work, and entry-state-constant remain controls using exactly the same
frozen cells. The report retains whichever candidate actually performs worse.

Memory cohorts may be prepared and frozen separately to bound offline JSON
analysis memory. Within each cohort there are three profile and three fresh
heldout foreground PIDs. This does not establish allocation changes within a
single trajectory. Startup retries are finite and permitted only before GO,
with verified owned-process exit; failed timed observations are never retried
as startup or removed from the evidence.

## Completed development-v3-m0 results

Three successful profile foreground PIDs supplied **3,398** complete invocations.
After parameter freezing and the independent heldout barrier, three fresh bare
heldout foreground PIDs supplied **487** invocations. Every successful foreground
and owned background actor passed the independent numerical port checks. Profile
and heldout seeds are disjoint. All 487 heldout calls have known external request
support; no heldout worker timing was used to reconstruct a predictor input.

The metric remains the median of the three process medians of per-invocation
absolute relative error (ARE). P90 and worst are over the retained heldout calls.
Every candidate uses the same frozen action-response cells and the same modeling
event engine.

| Predeclared candidate | Process-balanced median ARE | Pooled P90 ARE | Worst invocation ARE |
|---|---:|---:|---:|
| **Primary: dynamic integration with profile worker delays** | **13.243%** | **68.125%** | **122.575%** |
| Instantaneous dynamic integration | 14.390% | 68.269% | 122.575% |
| No owned CPU/GPU work; same held bytes | 23.066% | 75.126% | 98.338% |
| Entry request state held constant | 14.390% | 68.395% | 122.575% |

For the primary, the 444 within-state calls have median ARE **12.679%**, P90
**67.070%**; the 43 crossing calls have median **26.885%**, P90 **88.618%**.
Crossings occur in all three independent heldout PIDs. The primary reduces the
typical error slightly relative to the instantaneous baseline, but leaves a
large tail. It does not solve dynamic-state accuracy.

Process variation remains material: the primary's per-process median ARE is
13.243%, 68.385% and 10.004%. The corresponding observed median call times are
3.915 ms, 10.389 ms and 3.653 ms. Thus the second heldout process is much slower
overall; the aggregate median cannot conceal it. These durations are validation
targets, not fitted PID-specific offsets.

`profile_source_qualified`, `accepted` and `full_accuracy_qualified` are false.
The report retains **180 failed original compute/copy source cells** and 760
failed gap/control diagnostic cells. The latter are not falsely advertised as
the original source criterion. A matching stamped-versus-bare instrument control
is still not independently established for this boundary. A median below 20%
does not override failed source stability, the large point-error tail or the
crossing result.

The authoritative results are in
[`development-v3-m0/evaluation.json`](../profiling/state_response/results/development-v3-m0/evaluation.json),
with the frozen parameter artifact, barrier, all process records and raw native
outputs beside it. Original source snapshots under `recorded-sources/` preserve
the exact collector/modeling implementation used by that campaign; later code
does not replace its manifests or rewrite old results.

## Development-v4 profile observations: requests do not fix load intensity

Development-v4-m0 adds owned-process CPU counter sampling, actual loaded-module
path observations, on-disk runtime identities and the exact OpenMP launch
environment. This section uses **only its three profile foreground PIDs**. It
describes profile observations separately from the completed independent
heldout validation below and fits no new parameter.

Counter deltas are reconstructed from GetProcessTimes snapshots. A window enters
a state group only if the complete outer envelope of both actual CPU query spans
lies inside that request interval. Core equivalents are owned-process CPU time
divided by elapsed time, include all its threads, and are not normalized to
machine/affinity capacity or interpreted as useful instructions.

For the same `CPU=1,GPU=1` request, the observed unweighted means of the contained
counter-window core equivalents differ sharply:

| Profile process | CPU worker | Main foreground process | GPU worker's host process |
|---|---:|---:|---:|
| r0 | 0.962 | 0.035 | 1.006 |
| r1 | 0.016 | 0.987 | 0.988 |
| r2 | 0.981 | 0.027 | 1.002 |

The CPU worker obtains almost no observed CPU time in r1, whereas the main
process obtains approximately one core equivalent. In r0/r2 the relationship is
reversed. Even `CPU=1,GPU=0` gives CPU-worker means 0.855, 0.272 and 0.280 across
the three PIDs. These are observations of differing realized process CPU
activity under identical request bits; they do not yet identify the responsible
runtime scheduling, blocking, spinning or other mechanism. A request-state table
therefore cannot presume a reproducible background resource intensity merely
because the same worker was requested. Correcting edge delay alone is insufficient
to explain this evidence.

Five identifiable CPU-off intervals also have counter windows wholly after both
the off request and completion of the last old-request worker call, before the
next on request or the end of observed support. Their CPU-worker positive-window
counts are **5/96, 7/101, 5/159, 7/95 and 1/101**. Up to six consecutive windows
show positive cumulative increments. Thus a completed target call is not proof
that all process CPU activity has stopped. These are accounting observations,
not a precise timestamp of post-call instructions, proof of continuous spin, or
a fitted recovery/quiescence threshold. Zero increments do not establish silence.

The nominal 10 ms cadence was not achieved exactly. Median actual endpoint
intervals are approximately **15.5–15.7 ms**; sampler missed-period totals are
345/349/355. Each actor has 617/612/606 valid delta windows and no unresolved
counter reads in r0/r1/r2. QPC has a 100 ns tick, but the smallest observed positive
CPU accumulation is **15.625 ms**. QPC resolution must not be mistaken for CPU
accounting resolution or used to interpolate these counters into small actions.

All nine observed actors load `C:\Windows\System32\vcomp140.dll`, with on-disk
file/product version **14.51.36247.0** and SHA-256
`95d4ce4a6802d1e18b5e0e1722cc30ea72ca7e033f83828f05c0b7b993fe7cbf`.
The observed path/version/hash receipts are consistent across these actors;
this is not a verification of the mapped image's memory bytes or lifetime module
history. The exact launch mappings contain no `OMP_`, `KMP_` or `VCOMP_` values.
This describes the environment supplied at launch, not every later value inside
the child: the inspected GGML CPU initialization sets `KMP_BLOCKTIME=200` when
it is absent. Whether this VCOMP runtime honors that variable is unknown. Neither
the launch receipt nor that source assignment proves an actual waiting policy.
Runtime identity consistency excludes a simple differing-DLL-version explanation
within these observations, not every runtime-state explanation.

The reusable read-only analyzer and source-bound output are
[`activity_analysis.py`](../profiling/state_response/activity_analysis.py) and
[`activity-profile-v2.json`](../profiling/state_response/results/development-v4-m0/diagnosis/activity-profile-v2.json).
The original v1 analyzer bytes and output remain preserved. V2 requires complete
native PID/path, disk hash and version evidence before calling runtime receipts
known and identical; all nine actual receipts meet that stronger check, and the
per-run CPU statistics are unchanged from v1.
Consumed raw files and observed VCOMP disk bytes are SHA-checked. No host-total
counter, heldout activity trace, foreground-total fit or action correlation enters
a predictor through this diagnostic.

## Completed development-v4-m0 heldout validation

The three profile foreground processes supplied **2,226** complete invocations.
After freezing, three fresh bare heldout foreground processes supplied **383**
invocations, all with known external request support and computable predictions.
The heldout activity observations discussed below were inspected only after
evaluation; they did not supply response prices, delays or predictor inputs.
The primary remains the lagged dynamic candidate declared before collection.

| Predeclared candidate | Process-balanced median ARE | Pooled P90 ARE | Worst invocation ARE |
|---|---:|---:|---:|
| **Primary: dynamic integration with profile worker delays** | **64.027%** | **69.412%** | **309.376%** |
| Instantaneous dynamic integration | 64.027% | 69.877% | 516.357% |
| No owned CPU/GPU work; same held bytes | 68.189% | 82.727% | 98.291% |
| Entry request state held constant | 64.027% | 71.696% | 671.425% |

The primary has median ARE **64.054%**, P90 **68.594%** on 343 calls wholly
inside one requested state. On the 40 calls crossing a known request boundary,
the corresponding values are **63.101%** and **97.620%**. Here "within-state"
does not mean a settled or history-independent hardware state. The result is a
failed accuracy validation, including ordinary within-state calls, rather than
an error confined to transitions. Being predictable is coverage, not accuracy.

| Heldout foreground | Calls | Observed median call time | Primary median ARE |
|---|---:|---:|---:|
| r0, PID 37392 | 58 | 11.083 ms | 68.004% |
| r1, PID 14540 | 77 | 9.476 ms | 64.027% |
| r2, PID 8092 | 248 | 3.753 ms | 8.593% |

The first two processes are consistently slower in the common request states.
The following table compares actual invocation medians with the median of the
already-frozen primary predictions, without changing any parameter. A missing
row means that this process produced no complete invocation wholly inside that
short requested state; crossing calls remain in the preceding aggregate.

| Foreground | Requested CPU/GPU | Complete calls | Observed median | Primary predicted median |
|---|---|---:|---:|---:|
| r0 | 0/0 | 29 | 9.720 ms | 3.014 ms |
| r0 | 0/1 | 1 | 135.768 ms | 28.460 ms |
| r0 | 1/0 | 15 | 11.286 ms | 3.807 ms |
| r1 | 0/0 | 36 | 9.033 ms | 3.014 ms |
| r1 | 1/0 | 30 | 9.782 ms | 3.807 ms |
| r2 | 0/0 | 125 | 3.199 ms | 3.014 ms |
| r2 | 0/1 | 13 | 19.229 ms | 28.460 ms |
| r2 | 1/0 | 91 | 4.171 ms | 3.807 ms |
| r2 | 1/1 | 3 | 31.824 ms | 44.920 ms |

Profile itself already contains different regimes: the complete within-request
0/0 invocation medians are 8.888/3.227/3.200 ms across its three PIDs; for 1/0
they are 10.267/4.037/4.028 ms. These are observed invocation statistics, including
retained settling observations, not an alternative timing formula or a sum of
action medians. Thus profile contains one slow and two fast processes, while
heldout contains two slow and one fast process. An equal-process median does not
make these regimes reproducible. Development-v3 also had a slow heldout process,
but only one of its three process medians was large. That difference helps explain
why the reported median changes from 13.243% to 64.027%; it does not establish
which implementation or hardware mechanism caused the regimes. Different
schedules and history coverage also prevent treating this cross-campaign
comparison as a controlled attribution experiment.

### Post-evaluation CPU activity and history checks

Consumed heldout activity, schedule and native CPU-diagnostic bytes were checked
against their record SHA bindings. Counter deltas were reconstructed, with no
unresolved reads in the 194/180/187 valid windows per actor in r0/r1/r2. Median
actual endpoint spacing is approximately 15.4/15.5/15.5 ms. The tables below use
only windows whose complete CPU-query envelopes lie inside the **GO phases**;
the later observation/support window is excluded. Ratios are the sum of CPU-time
increments divided by the sum of query-endpoint elapsed times for those contained
windows. They are diagnostic core equivalents, not available foreground rates.

| Heldout foreground | CPU worker, request 1/0 | Main, request 1/0 | CPU worker, request 1/1 | Main, request 1/1 |
|---|---:|---:|---:|---:|
| r0 | 0.924 | 0.419 | 0.973 | 0.000 |
| r1 | 0.059 | 1.607 | 0.000 | 1.002 |
| r2 | 0.254 | 0.972 | 0.331 | 0.552 |

In r1 the background CPU worker receives little observed CPU time, yet foreground
0/0 and 1/0 invocations remain slow. Background request intensity alone therefore
does not account for all observed error. The GPU worker's **host-process** ratios
in 1/1 are approximately 1.012/1.002/0.995; these are not measurements of GPU
device activity, queue occupancy or clocks.

Native whole-sampling CPU counters provide another independent symptom:

| Heldout foreground | Process CPU / wall time | Mean process CPU / completed call | Mean main-thread CPU / completed call | Mean other-thread CPU / completed call |
|---|---:|---:|---:|---:|
| r0 | 0.546 | 13.739 ms | 4.310 ms | 9.429 ms |
| r1 | 1.239 | 20.901 ms | 4.464 ms | 16.437 ms |
| r2 | 0.939 | 5.292 ms | 4.347 ms | 0.945 ms |

These are whole-trajectory CPU-accounting totals divided by invocation count,
not per-state medians or per-operator costs. Their scope includes the replay
loop/reset and runtime threads. The slow processes have substantially more
non-main-thread CPU time per completed invocation while main-thread CPU time per
invocation is similar. This supports an unresolved execution/runtime-state
mechanism; it does not identify useful work, spinning, blocking or VCOMP as its
cause. CPU and GPU intervals must not be added to these inclusive wall costs.

History remains a material confound. A subsequent raw-GO audit corrects the
earlier claim that all profile 0/0 intervals occur before the first CPU-worker
on request. Actual profile orders, where each pair denotes CPU/GPU requests,
are:

| Profile foreground | Exact request order and planned duration |
|---|---|
| r0 | 10 (3 s), 01 (1.5 s), 11 (3 s), 00 (0.6 s) |
| r1 | 00 (0.6 s), 11 (3 s), 10 (3 s), 01 (1.5 s) |
| r2 | 11 (3 s), 01 (1.5 s), 10 (3 s), 00 (0.6 s) |

Thus r1 has 177 complete initial 0/0 calls before any scheduled CPU-on request,
whereas r0/r2 reach 0/0 after a load. These are different foreground processes,
not a matched recovery control. Profile r0 and r2 both reach 0/0 last, yet one
is slow and the other fast; simple phase position alone is insufficient.
The original numerical results and raw records are unchanged by this correction.

Heldout r0 has two complete initial 0/0 calls before its first scheduled
CPU-worker on request, already slow at 14.6752 and 13.9439 ms. Heldout r1 starts
with CPU requested on, and heldout r2 starts with GPU requested on, so neither
supplies a comparable initial 0/0 control. This small initial heldout sample
does not rule out recovery/history effects. All actors have already performed
initial validation and warmup before GO: even an initial scheduled 0/0 interval
is not a claim of a pristine runtime or a CPU worker that has never executed.

After an earlier CPU on request, 0/0 observed medians by age since the latest CPU
off request are:

| Foreground | Age < 50 ms | Age 50–150 ms | Age >= 150 ms |
|---|---:|---:|---:|
| r0 | 10.064 ms (14 calls) | 9.437 ms (11) | 8.624 ms (2) |
| r1 | 9.293 ms (17) | 8.932 ms (16) | 8.787 ms (3) |
| r2 | 3.248 ms (42) | 3.200 ms (70) | 2.897 ms (13) |

Age here is the requested off age, not an inferred physical recovery time. The
slow difference persists beyond the first few milliseconds, but long off-age
coverage is small and differently ordered. No history-independent limiting
duration or recovery law can be inferred from these groups.

A focused read-only audit of profile action timestamps further rejects a single
whole-program slowdown multiplier. For each cell below, the complete action
interval lies inside the declared 0/0 calibration window after the 150 ms
exclusion. Entries are action-wall medians in milliseconds; the parenthesized
number is the retained action count.

| Compiled action occurrence | Profile r0 | Profile r1 | Profile r2 |
|---|---:|---:|---:|
| 0: CPU projection | 0.1130 (46) | 0.0994 (133) | 0.1020 (128) |
| 14: GPU Attention PV | 4.80265 (46) | 1.67175 (132) | 1.58780 (128) |
| 28: GPU Attention output | 0.3039 (47) | 0.0959 (133) | 0.0977 (127) |
| 39: CPU FFN gate, left partition | 0.0347 (47) | 0.1644 (133) | 0.1633 (127) |
| 52: GPU FFN gate, right partition | 0.1675 (47) | 0.0424 (133) | 0.0431 (127) |
| 85: D2H KV copy | 0.07345 (46) | 0.01710 (133) | 0.01790 (128) |

The process with the slow complete layer has a faster CPU FFN-gate action but
slower GPU actions and D2H copies. These inclusive wall intervals do not identify
whether the difference is host execution, ready waiting, runtime blocking,
device work or a combination. They motivate separate measurements at precisely
these boundaries; they do not establish an additive overhead. Action counts
differ because containment is tested per action, so summing these medians is
not a decomposition of the whole-invocation median.

Positive CPU-worker counter increments also occur in windows wholly after both
the off request and the last old-request call's return. The latest such outer
windows, measured from that joint boundary in each run, extend to approximately
548–564 ms, 876–892 ms and 674–689 ms for r0/r1/r2. Counter granularity, deferred
accounting and non-atomic query spans prevent translating them into exact
instruction or silence timestamps. Zero increments are likewise not proof of
quiescence. None of these heldout observations was used to refit a stop delay.

The evaluation retains **179 failed original compute/copy source cells** and
760 failed gap/control diagnostic cells. Source qualification, complete accuracy
qualification and acceptance remain false; the matching stamped-versus-bare
control is still missing. Primary results above come explicitly from the
predeclared lagged-model fields; legacy top-level instantaneous-summary fields
must not be relabeled as the primary. Historical v3 and earlier acceptance
results remain unchanged.

The authoritative result and evidence are
[`development-v4-m0/evaluation.json`](../profiling/state_response/results/development-v4-m0/evaluation.json),
its frozen parameters, pre-heldout barrier, process records and raw native/counter
outputs. The useful conclusion is that numeric worker requests plus fixed delays
are **not yet a sufficient reproducible service-state description** in this
configuration. The new observations narrow the problem; they do not solve it.

## Software checks and diagnostic feasibility

The subsequent software regression checks reported by the coordinating run are
separate from the hardware accuracy results above:

| Check scope | Reported total | Outcome |
|---|---:|---|
| Modeling | 467 tests, 7 skipped | No failures |
| Public API / distribution model | 63 tests | All passed |
| Legacy profiling | 491 tests, 1 skipped | No failures |
| State-response profiling | 96 tests, including 12 primary-result guard tests | All passed |
| New native software checks | 12 tests | All passed |
| Full mapping regression | 1,119 tests | All passed |

The first two mapping attempts had runner/discovery configuration errors
(relative imports and a relative `PYTHONPATH` inherited by child working
directories). The final run uses corrected discovery and absolute import paths
and completes all 1,119 tests in 232.316 seconds. These checks cannot
validate physical response prices, remove failed source-CV gates, or turn the
v4 result into qualified accuracy. Primary-summary guard changes likewise
require their own source receipts; they do not silently replace the archived
implementation or the original v4 evidence.

At that stage, read-only Windows tracing feasibility checks found WPR, Xperf and WPAExporter,
but the current token lacks `SeSystemProfilePrivilege`. No scheduler recording
was started, and a process filter has not been proven sufficient to exclude all
unrelated events. A scheduler-level attribution study would therefore need a
privileged session and a bounded diagnostic protocol. Tool availability is not
evidence that a particular runtime caused the measured regimes, nor a promise
that one change will reduce error below 20%.

The subsequent explicitly authorized kernel capture and sixteen controlled runs
are reported in [the 2026-10-22 thread/runtime/recovery investigation](THREAD_RUNTIME_RECOVERY_DIAGNOSIS_20261022.md).
They supersede the earlier statement that no scheduler capture had been run;
they do not supersede this report's frozen accuracy failures.

The additive primary-report fix was checked separately against saved v3/v4
predictions. Its derived receipts are
[`v3 primary validation`](../profiling/state_response/results/development-v3-m0/diagnosis/primary-validation-derived-v2.json)
and
[`v4 primary validation`](../profiling/state_response/results/development-v4-m0/diagnosis/primary-validation-derived-v2.json).
They verify original evaluation digests and archived source bindings, then
recompute only acceptance summaries. Both retain failed primary qualification.
They perform no prediction or fit and are not new hardware validation of the
current source. Original results are unchanged; the v4 source snapshot preserves
445 bound repository source files before subsequent reporting changes.

## Remaining scope and next evidence

Only m0 has completed independent hardware validation. Prepared m1/m2/m3 folders
are unmeasured plans, not completion of the 16 numerical states. After a source
change, those cohorts require a fresh prepare/source binding before collection.
Separate cohort freezes bound offline RAM; they do not change this coverage limit.

The next task is to identify a reproducible actual service state and its
independently measured response. The current evidence does **not** provide a
CPU-counter-to-available-throughput formula. A useful experiment must distinguish
worker requests, realized owned-process activity and the service received by
foreground operations, with explicit clock windows and causal availability of
state observations. Longer controls or scheduler/thread-level diagnostics may
be needed to separate active work, blocking and runtime recovery. Any new
response law must be frozen from profile-only evidence and tested on fresh
heldout data. PID-specific corrections and total-latency offsets would conceal
the unresolved mechanism rather than establish it.
