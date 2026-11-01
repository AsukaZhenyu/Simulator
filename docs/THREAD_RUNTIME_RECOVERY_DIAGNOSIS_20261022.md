# Thread scheduling, runtime and recovery diagnosis

**Accuracy remains unqualified.** This investigation identifies a substantial
missing CPU interruption component and excludes several simple explanations.
It does not replace failed calibration with another fitted layer multiplier.
There is no new claim that complete dynamic-state predictions pass 20% error.

The new frozen-profile/fresh-process check covers **686 invocations**: its
predeclared primary model has a **10.50% process-balanced median absolute
relative error**, **39.80% pooled P90** and **97.15% worst error**. GPU-load and
state-crossing subsets remain inaccurate, and source qualification still fails.
These results are reported together; the aggregate median is not an accuracy
acceptance.

## What was measured

The workload is the compiled H128/q16/history0 F32 mixed CPU/GPU layer,
including Attention, FFN partitioning, KV operations and transfers: 117 serial
action occurrences per invocation. The original executor, strict F32 runtime,
disabled CUDA graphs, numeric inputs, CPU affinity, thread count, actual CUDA
flags, GO trajectory and sources are recorded. Independent numerical checks
cover every completed run. Instrumented runs remain diagnostic observations.

New evidence consists of:

* Two thread-observer pilots, each from one fresh process. Both missed the
  requested 2 ms cadence: legacy .NET snapshots in the first pilot and NT
  snapshots in the second each took about 9 ms in these workload pilots.
  Snapshot counts are never converted into exact state durations.
* One explicitly authorized, bounded Windows kernel recording of a new process,
  with stamped ordinary actions and no thread-query observer. The recording
  contains 354 complete invocations and 41,418 action intervals.
* Sixteen further runs: eight conditions in two independently seeded processes
  each, with the thread-query observer disabled. Controls include four explicit
  CUDA wait policies, the unset policy, one CPU thread, bare execution and
  matched load/recovery trajectories.
* A subsequent thread-placement intervention with the process mask unchanged,
  followed by three profile processes and three fresh bare heldout processes
  in an explicitly fixed main-thread/core configuration.

Raw results and source archives remain under
`profiling/state_response/results/`. The raw system ETL remains local and is
excluded from Git. Portable summaries retain owned observations, not unrelated
process names or command lines.

## Scheduling assignment is not application CPU execution

The kernel trace uses QPC at 10 MHz, matching native receipts. The owned
process/thread lifetime starts occur inside the recorded launch-to-READY
brackets. Raw begin/end marker events occur inside their command brackets.
Both initial and post-processing loss counters report zero lost events/buffers.
These checks establish alignment; an existing ETL file alone would not.

The native sampling envelope lasts 5.0133705 s. CSwitch/ReadyThread reconstruction
assigns the main thread approximately:

| Scheduler state | Duration | Meaning |
|---|---:|---|
| Assigned Running | 4.9423488 s | A CPU is assigned to this thread; interrupts can preempt its instructions |
| Ready | 71.0217 ms | Runnable but awaiting a CPU |
| Observed Waiting | 0 | No blocked interval reconstructed inside this sampling window |

The begin/end marker span is 9.164 s, and the last actual sampled-profile
event occurs 13.023 s after the start command begins. The stop/merge command
takes 36.615 s. These are distinct measurements: command wall time is not
sample duration, and last-event timestamps do not identify the exact recorder
stop instant. The portable evidence retains the raw timestamps.

Thus ordinary ready-queue delay is about 1.42% of this window. It cannot by
itself explain the multiple-fold PV slowdown observed here. This is a statement
about this one traced slow process, not every process or loaded state.

Crucially, **2.4395134 s of the assigned-Running interval overlaps
same-CPU interrupt/DPC execution**, using a union that counts nested overlaps
once. This is 49.36% of assigned occupancy. Assigned Running must therefore
not be relabeled as useful application computation, CUDA arithmetic, or even
uninterrupted execution of this thread.

The native main-thread CPU-accounting total is approximately 2.281 s. Subtracting
the diagnostic interrupt overlap from assigned occupancy leaves 2.5028354 s;
the remaining discrepancy is retained. Counter accounting, interval precision
and interruption reconstruction are different observations, not interchangeable
clocks or grounds for fitting a residual. Exact decoder/precision qualifications
are recorded in the derived scheduler evidence.

For all 354 PV action-14 intervals, the **disjoint interval sums** are:

| Observed PV wall component | Aggregate |
|---|---:|
| Assigned Running outside same-CPU interrupts | 1.3025629 s |
| Same-CPU interrupt/DPC overlap | 1.4994567 s |
| Ready queue | 0.0214628 s |
| Total PV wall intervals | 2.8234824 s |

These sums partition the actual observed intervals; they are not a sum of
separately measured medians or extra costs to add to a GPU response. In this
trace interruption overlap occupies about 53.1% of PV wall time. It may include
workload-generated GPU-driver activity and tracing effects, not merely external
background load. No fast/slow interruption comparison or observer-neutrality
claim follows from one traced slow process.

The raw interruption decoder is cross-checked against all 166,510 exported
interrupt/DPC events: keys, descriptors and routine addresses match; durations
agree within xperf's rounded-microsecond precision. Final unions use raw QPC at
100 ns ticks, not the rounded durations. Sampled PCs are split inside/outside
those same-CPU intervals before interpreting module locations. Most sampled
`nvlddmkm` PCs in the main-thread context fall inside interrupt intervals;
they cannot be called application CPU work.

An auxiliary thread whose creation entry maps to VCOMP also has substantial
assigned occupancy. Its creation module alone does not identify what it executes
later, and auxiliary CPU time cannot be added to main-thread wall time. Sampled
instruction-module attribution must distinguish PCs inside interrupts from PCs
outside them. No spin-loop or function-level diagnosis follows from a module
name alone.

The assignment is asymmetric: the main thread stays on logical CPU0 throughout
this window, while that auxiliary thread stays on CPU2. Global interrupt unions
within the window are 2.4428067 s on CPU0 and 0.0096942 s on CPU2. The auxiliary
thread's own assigned interval overlaps just 0.0095566 s of interruptions. This
motivates the direct placement intervention below; it does not by itself prove
the source of every fast/slow process.

## CUDA wait policy does not remove the baseline regimes

All values below are whole-invocation medians, in milliseconds. Each entry is
one new process; two processes per condition are below the existing minimum of
three for source qualification.

| Condition | Repeat 0 | Repeat 1 |
|---|---:|---:|
| Bare replay, thread observer off | 8.286 | 8.411 |
| Explicit CUDA auto | 8.424 | 2.785 |
| Explicit CUDA spin | 8.314 | 2.778 |
| Explicit CUDA blocking | 3.587 | 9.217 |
| Explicit CUDA yield | 8.389 | 8.516 |
| Unset CUDA policy, GPU-phase diagnostic | 8.423 | 8.593 |
| One CPU compute thread | 8.948 | 8.890 |
| Matched recovery trajectory | 8.467 | 2.950 |

The explicit modes use the same setter-before-executor initialization procedure.
After warmup and after STOP, observed flags preserve the requested masks:
auto=8, spin=9, yield=10, blocking=12 including the non-schedule flag bits.
Unset invokes no setter. Auto is not interpreted as a resolved runtime behavior.

Auto, spin and blocking still have strongly different repeated-process timings.
Changing one wait flag therefore does not establish reproducible prices. The
one-thread controls remain slow, so OpenMP auxiliary work is not established as
a necessary cause either. These observations do not prove that either feature
is irrelevant on other workloads.

The selected GPU PV action is occurrence 14. In explicit auto repeat 0, its
median host enqueue is 4.77965 ms and terminal synchronization is 0.0032 ms;
repeat 1 gives 1.4233 ms and 0.0013 ms. Inclusive PV wall is 4.94185/1.4913 ms.
The large difference is predominantly inside the enqueue boundary, rather than
the separately timed final synchronize call.

CUDA stream spans are 4.84096/1.4592 ms in those same runs. They include gaps
between submissions and are not active-kernel time. Enqueue, sync and GPU spans
overlap and must not be added or subtracted to invent independent costs.

Earlier [operation-scoped Nsight evidence](SINGLE_LAYER_CUDA_WAIT_DIAGNOSIS_20261018.md)
also localized large wall intervals inside non-sync CUDA APIs while correlated
GPU activity was much shorter. That older instrumented experiment has profiler
compatibility limitations. Its kernel times are not transplanted into this ETW
trace, and its clocks are not merged with QPC.

## Recovery is a separate effect from the persistent baseline

Each recovery process begins with 00 for two seconds, exercises all three
requested load states in a randomized order for one second each, and returns
to 00 for two seconds after each. Here 10 means CPU replay, 01 GPU replay and
11 both. These are offered workloads, not utilization percentages.

| Complete invocations inside 00 phase | Slow process | Fast process |
|---|---:|---:|
| Initial 00 | 8.3828 ms | 2.9226 ms |
| After CPU load 10 | 8.5127 ms | 2.9834 ms |
| After GPU load 01 | 7.9364 ms | 2.8857 ms |
| After combined load 11 | 8.5749 ms | 2.8893 ms |

Even the 1000–2000 ms off-age bins retain roughly 8.01–8.66 versus 2.93–2.98 ms.
A short stop-response tail alone therefore cannot explain the persistent
threefold baseline difference. Off-age is measured from the request boundary;
it is not a certified timestamp of physical quiescence.

Load-on interference is still substantial. Whole-invocation medians under
requested GPU load 01/11 are approximately 87.87/115.00 ms in the slow process
and 36.21/47.03 ms in the fast process. PV terminal synchronization under GPU
load also rises to roughly 0.94–1.21 ms. Baseline regime, active contention and
post-load recovery must not be collapsed into one coefficient.

## Direct intervention: main-thread CPU placement

The process remains allowed on CPUs0/2 (mask5), and the CPU backend retains two
threads. After warmup but before GO, the controller changes **only the owned main
thread's** affinity. It holds a verified thread handle, checks PID/TID and birth
identities, verifies mask/process group, and rechecks the actual mask after
sampling. It changes no unrelated thread or system interrupt configuration.

| Main-thread binding | Whole p50 / p90 | PV14 p50 | CPU FFN gate39 p50 | D2H85 p50 |
|---|---:|---:|---:|---:|
| CPU2, mask4 | 2.8109 / 3.50532 ms | 1.46575 ms | 0.0853 ms | 0.0159 ms |
| CPU0, mask1 | 8.0953 / 8.91102 ms | 4.6432 ms | 0.0290 ms | 0.0577 ms |

This first matched geometry/input pair strongly supports main-thread placement
as an important hidden context dimension. GPU/copy paths get faster on CPU2,
while this CPU FFN action gets slower. A uniform whole-layer multiplier would
miss that opposite change. Each placement currently has one process in this
pilot, so it is not a complete causal isolation of interrupt routing from all
other per-core effects, or a qualified price table.

The observed pin is applied before GO and preserved after sampling in both
processes; the process mask remains5. Enqueue/stream components were not
instrumented in this stamped pilot and remain unknown, rather than being copied
from another run. Full provenance is retained in the portable affinity evidence.

## Changes to modeling and profiling

Profiling owns measurement and evidence:

* Actual native main TIDs, per-action CPU-counter query brackets and raw cycles.
* Optional owned-lifetime NT snapshots with measured cadence and unknown ranges.
* Explicit CUDA wait-mode endpoint receipts and controlled actor/thread cases.
* Same-process predecessor/off-age groups, with crossings retained separately.
* Optional bounded kernel tracing, raw-clock/loss/marker checks and same-core
  scheduler/interruption decomposition. Neither sampled states nor CPU counters
  are silently upgraded to exact useful-work measurements.

The common modeling engine owns execution-time composition:

* A `Task.waits_for` completion gate waits for declared device tasks without
  charging their duration a second time. A blocked caller can release a host
  resource; an explicitly occupied polling resource can remain reserved.
* `thread_cpu_ns` is active service demand for one thread, separate from elapsed
  fixed latency. Its explicit supply has unit `thread_cpu_ns/s`, bounded by
  0..1e9 for one thread. Zero supply pauses remaining work; recovery resumes it.
* Interruptions consume CPU availability separately from thread service. A
  same-core union must avoid double counting nested ISR/DPC intervals and must
  preserve unrelated GPU work and memory-state changes.

`compose_cpu_interruptions` implements that last operation with an explicit
shared timebase, logical-CPU inventory, source evidence and static thread/core
binding. It restores the latest base supply after an interruption, including
rate changes that occurred inside it. The base must exclude the same supplied
interruptions. It does not predict workload-induced future interrupts or thread
migration. The [runnable declared example](../examples/library/cpu_interruptions.py)
needs no saved profile or GPU: overlapping [3,7) and [5,9) ms interruptions
delay 10 ms of CPU work to 16 ms while an independent GPU task still ends at
12 ms. It explicitly reports `accuracy_qualified=false`.

These are representational mechanisms, not an implementation of the Windows
scheduler. Named resources remain exclusive; there is no inferred time slicing
or automatic sharing. No retrospective heldout CSwitch or DPC timeline may be
used as an ordinary prediction input. An explicitly labeled diagnostic replay
can study such a timeline but cannot qualify forecast accuracy.

The general mapping compiler has not automatically lowered every existing LLM
action into these finer host/device/interrupt stages. The ordinary LLM cost
workflow is not recalibrated by adding the mechanisms. Existing inclusive
response tables keep their old scope; fine-stage prices remain unknown until
independent measurement identifies them.

No DNN, PID identifier, layer-total offset, fitted policy multiplier, or
`sync wall - stream span` residual was introduced. Existing inclusive response
cells are not mixed with new host/GPU components, which would double count.

## Independent accuracy acceptance: improved baseline, remaining GPU/state gap

The previous frozen request/response model still fails independent accuracy:
the v4 primary process-balanced median error is about 64.03%, with source
qualification false. Its parameters and historical results are unchanged. See
the [earlier calibration report](STATE_RESPONSE_CALIBRATION_20261021.md).

The sixteen control runs are independent diagnostic evidence, **not** the
profile/freeze/heldout accuracy acceptance reported below. The new mechanisms
still lack a qualified predictive law for runtime/driver service and workload-induced
interruptions. A diagnostic trace of realized interruptions cannot be used to
pretend those future interruptions were predicted.

Following the placement intervention, cohort `development-v5-core2-m0` performs
a separate actual profile/freeze/heldout experiment. Main-thread mask 4 and all
three process masks are part of the frozen context. Three stamped profile
processes cover offered states 00/01/10/11 with 2000/2500/3000/3000 ms holds and
a predeclared 1000 ms settling exclusion before price extraction. Three fresh
bare processes use different seeds and dynamic state orders/durations, with
short request phases of 37–181 ms. No ETW interruption timeline,
heldout internal action times or heldout utilization enters prediction.

The response equation and equal-process-median estimator are unchanged. Source
stability and timing-error thresholds are unchanged. This checks reproducibility
under a known placement; it does not establish an endogenous-interruption law.
The profile-settle/hold protocol also differs from v4, so an error difference
between the cohorts cannot all be causally assigned to the pinning intervention.

The frozen file was fully written before a temporary orchestration script's
summary counter raised `KeyError: observations` (the schema uses `invocations`).
The original script and error log are retained in `freeze-wrapper-recovery.json`.
No parameter file or measurement was replaced. The heldout collector subsequently
reconstructed the frozen statistics from original evidence before creating its
barrier and launching new processes. Its frozen-file SHA is
`93d75dbc68d98c5136daec25cb05e359bf5d2aa076b31f5d71fe2dea3b359369`.

All six jobs' actual main-thread masks are 4 at both endpoints, while the
foreground and CPU worker process masks remain 5 and the GPU worker's host
process mask remains 48.
Owned process/thread identities and independent numerical checks pass. The
three profile processes finish before the frozen barrier, and all three fresh
heldout processes start after it. No timed run was discarded or retried.

The predeclared primary predictor is `lagged_dynamic_state_integration`. Its
results below come from `primary_validation`, not another retrospectively
chosen predictor or the legacy top-level summary.

| Primary validation subset | Invocations | Process-balanced median ARE | Pooled P90 ARE |
|---|---:|---:|---:|
| All heldout invocations | 686 | 10.50% | 39.80% |
| Constant known request throughout invocation | 639 | 9.70% | 33.32% |
| Invocation crossing a known request boundary | 47 | 37.69% | 90.09% |

ARE is `abs(predicted - observed) / observed`. The median statistic is the
median of the three within-process median errors, giving each process equal
weight. The three overall process medians are 10.08%, 21.00% and 10.50%; an
aggregate below 20% does not mean every process is below 20%. The 686 invocations
are repeated observations from three independent processes, not 686 independent
process-level repetitions. All 686 have predictions; no unpredictable sample
was removed from these error statistics.

| Requested CPU/GPU replay state, constant-request calls only | Invocations | Process-balanced median ARE | P90 ARE | Measurement stability qualified |
|---|---:|---:|---:|---|
| 00: neither replay | 336 | 10.94% | 27.87% | Yes |
| 10: CPU replay only | 260 | 8.57% | 20.74% | Yes |
| 01: GPU replay only | 23 | 56.98% | 70.66% | No |
| 11: both replays | 20 | 91.39% | 92.86% | No |

The stored bucket name `steady` means only that the request remains constant
throughout an invocation, including the known terminal-state grace interval.
One final 00 call extends past the original scheduled end into that grace
interval; it remains included. This bucket does **not** establish settled
physical device state. In particular, the heldout phases are much shorter than
the profile settling exclusion. State 11 has just 15/2/3 constant-request calls
across the three processes: its 91.39% statistic indicates a serious failure
in this small sample, not a precise population-level estimate. State 00's worst
process median is 21.71% despite its lower process-balanced summary.

Profiling supports the baseline diagnosis: the three 00 whole-invocation
medians are 2.904/2.843/2.912 ms, with between-process CV about 1.31% and max/min
about 1.024. Under the identical GPU replay-on request 01, profile medians are
instead 5.171/19.300/31.370 ms. Explicit main-thread placement stabilizes the
quiet baseline in this cohort; a binary GPU replay request still fails to
identify a reproducible available-service state.

**The independent acceptance remains false.** The original compute/copy source
has 178 failed cells, the lag source is unqualified, and the required profile,
steady-state, crossing and statistical-protocol gates do not pass. Matching
stamped-versus-bare instrumentation controls have not been independently
established for this exact scope. Actual CPU/GPU available-service transfer
also remains unidentified. The equation has not been refitted after seeing
heldout results, and no diagnostic interruption trace was substituted as a
prediction input.

This distinguishes three remaining mechanisms rather than absorbing them
into one mean: host placement/interruption affects the baseline; offered GPU
work does not uniquely specify actual queue/service interference; request
crossings and recovery retain history that the current response description
does not accurately resolve. GPU queue/backlog and host/device service are the
next identification targets, not already established predictive explanations.
The new explicit thread-service/interruption graph is not the source of a
claim that these formal predictions use newly calibrated physical stages.

The next physical calibration must therefore:

1. Measure backend API/materialization counts and independently observable
   runtime/interrupt service parameters, keeping thread CPU work, interrupt
   CPU work, GPU activity and runnable queue delays distinct.
2. Establish a source-stable state description across at least three profile
   processes; do not name the state by PID or by the target layer's measured
   latency. Preserve unknown parameters when identification fails.
3. Freeze the stage graph, inputs, parameters and instrumentation controls.
4. Launch new bare heldout processes with unseen load orders/durations. Supply
   only causally available external inputs, and evaluate whole-layer latency,
   tails, recovery/crossing subsets and unpredictable cases separately.

This task has measured a major missing component, added explicit mechanisms to
the common engine and completed a new independent validation. It has not
completed physical calibration of the remaining GPU contention/recovery gap.
Reporting only the improved median would conceal the open issue.
This Dense single-layer diagnosis also does not qualify MoE, multilayer, SSD or
additional memory-hold configurations.

## Evidence and verification

* [Portable control observations](data/thread-runtime-controls-20261022.json).
* [Portable owned scheduler/interruption evidence](data/thread-runtime-etw-20261022.json).
* [Portable owned placement and formal-cohort observations](data/thread-runtime-affinity-20261022.json).
* [Portable frozen-profile/fresh-process accuracy validation](data/thread-runtime-validation-20261022.json).
* [Reusable measurement protocol](../profiling/state_response/THREAD_DIAGNOSIS.md).
* Local kernel recording: `profiling/state_response/etw/results/LLMSimThread_50137046b44849e991531d6026a666ce/`.
* Local control observations: `build/temp-thread-analysis/thread-controls-v3-observations.json`.
* Exact original source archives: pilot v1 453 files, pilot v2 457, ETW v2 461,
  control v3 458, placement pilot 464 and formal v5 463. SHA comparisons precede
  subsequent source edits.

Software semantics and hardware accuracy are separate. Modeling regression
currently covers the completion gates, 16 hand-derived CPU-service cases and
23 interruption cases; the full run reports 526 tests, 7 explicit skips and no
failures. State-response
profiling reports 180 passing tests, including the optional actual C# parser
fixtures, and its production analyzer successfully processes all sixteen real
observer-off jobs. Further interruption/parser checks are recorded with their
own source receipts rather than silently attributed to an earlier build.
The public library/distribution suite reports 63 passing tests; the updated
public performance facade has also passed its 7 boundary checks. Native source
and provenance fixtures report 18 passing tests. The declared interruption
example returns the hand-computed 16/12 ms CPU/GPU completion times and false
accuracy qualification without loading any saved measurements.
