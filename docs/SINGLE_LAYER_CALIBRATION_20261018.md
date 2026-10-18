# Complete single-layer calibration — 18 October 2026

## Question and boundaries

This study tests whether prices measured for separate operations predict the
latency of the actual compiled single-layer strategy. It distinguishes parameter
scope, omitted intervals, statistical composition and execution-state variation.
The default library remains an uncalibrated analytical estimator.

The formal campaign contains 14 **complete F32 Dense layer** programs: H64/I128
with q1/history128, H128/I256 with q16/history0, and H256/I512 with q32/history0.
The programs include CPU, GPU and hybrid macro placements, split Attention, and
three FFN split points at each geometry. They are bounded small test workloads,
not production-size model validation. The native executor reuses the actual
public-compiled action sequence, independent contiguous tensor ports, prebuilt
GGML graphs, explicit serial copies, CPU threads=2, default threadpool polling,
strict F32 and disabled CUDA graphs. Each native process is compared with the
independent NumPy reference for all final ports and copy/action identities.

The history128 factory represents history as 128 one-token logical blocks with
materialized independent backing. The H64 hybrid case therefore has 136 explicit
timed COPY actions; the CPU case has none. This is the declared graph/layout
boundary, not a production contiguous or paged-KV implementation. Its many small
copies and internal layout work help explain why these small GPU timings cannot
be read as ordinary model-serving throughput. Compare policies within this
boundary; do not transfer their absolute times to a real LLM runtime.

This does not validate MoE, quantization, batch>1, concurrent streams,
multi-round token generation, physical memory capacity or unseen geometry.
Native backing is preallocated; logical eviction is not evidence of physical
allocator reclamation. GPU private scratch remains outside the capacity claim.

## Corrections and their ownership

| Finding | Evidence | Change | Remaining limit |
|---|---|---|---|
| An isolated action price can have the wrong context | The old probe repeats one action; the actual layer alternates CPU, GPU and copies | Profiling now retains each completed action in real execution order, alongside the isolated baseline | A static context label does not capture every time-varying runtime state |
| Control and dispatch intervals were not explicitly priced | A timestamp partition covers every action, preceding gap and final tail | Modeling constructs an explicit serial dependency graph with separate nonoverlapping parameters | Inclusive action prices already contain device work and waits; they are not pure throughput parameters |
| Action medians do not add to a whole-program median | The two-case development study differs by 4.27–8.62% even within the instrumented trace | Modeling additionally propagates supplied marginal distributions and reports exact expected duration | Secondary p50/p90 assume conditional independence; a shared state change can violate it |
| Runtime-state drift survives the same PID | Identically labelled blocks change by over three times with unchanged program and thread count | Preserve all raw samples, same-process controls, CPU counters and synchronized device observations | The physical source of the waits is not identified by these controls alone |

Profiling owns source verification, observed samples, process-weighted medians,
histograms and frozen parameter evidence. Modeling owns all execution-time
calculations. The new workflow never fits whole-layer totals into an offset or
uses heldout totals as a latency parameter. No DNN or regression predictor is
introduced. No negative instrumentation correction is subtracted.

The deterministic alternatives are:

1. **Isolated components:** separately measured compute/copy prices; explicit
   zero declarations for the original ignored control/gap intervals.
2. **Context components:** compute/copy prices from the real action order;
   the same explicit zero declarations for controls/gaps.
3. **Context actions and gaps:** every completed action plus disjoint host
   gaps and tail. This remains the primary model under the original gates.
4. **Secondary distribution candidate:** choose one profiled process with
   equal weight, then draw each serial interval independently from that
   process's supplied marginal distribution. It does not reuse an entire
   measured iteration. Exact mean follows linearity; median/tail need the stated
   conditional-independence assumption. Fixed Monte Carlo seed 20261018 and
   20,000 trials were declared before formal measurements.

The distribution model's numerical integration bounds are conditional on its supplied distributions;
they are not hardware confidence intervals. Mean error and interval coverage do
not replace the primary median/CV acceptance gates.

## Development measurements and same-process controls

The development campaign retained 36 profile processes and 12 fresh bare
heldout processes at the two larger FFN-split examples, under both default and
zero polling. All numerical/action checks passed. Default-pool primary heldout errors
were 9.56% and 16.59%; zero polling still produced a 42.18% H128 error. All four
groups failed the complete quality gates. These are development evidence, not
the formal post-correction acceptance result.

Measured dispatch gaps occupied only about 0.15–0.25% of the context total;
adding them cannot explain the large millisecond errors. The median composition
gap is a different issue: a sum of marginal medians is not a median of sums.
It must not be renamed as an unobserved physical overhead.

The subsequent diagnostic uses 24 independent native processes, 96 ABBA/BAAB
blocks, 30 warmup iterations per block and 101 retained samples. It compares
instrumented versus bare execution, and isolated versus in-context Attention
PV, inside the same PID. All declared processes and blocks are retained.
See the [paired diagnosis](SINGLE_LAYER_PAIRED_DIAGNOSIS_20261018.md).

The four group median instrumentation ratios are all within 5%, but only 9/12
individual process ratios meet that interval. A fixed instrumentation penalty
cannot explain the earlier 26–28% difference between independent process
cohorts. No such cohort difference is fitted or subtracted.

Identically labelled isolated PV blocks change from 4.8392 to 1.4932 ms under
default polling, and from 4.7863 to 1.4394 ms under zero polling. Another pair of
context PV blocks changes from 2.9878 to 7.5231 ms. Neither changing PID nor
nonzero polling is necessary for large latency drift. The same controls do not
establish that context, polling, cache, clocks or driver scheduling is its cause.

Synchronized query containment provides useful limits on simple explanations.
For the 2.9878→7.5231 ms pair, all contained SM-clock queries report 1605 MHz
and all temperature queries report 68°C. H128's slower blocks do not have lower
median memory clocks. These observations cannot justify a single clock-ratio
correction. They also do not exclude unsampled transients. The extra H128 wall
time largely lacks corresponding main-thread CPU-time growth; GPU/driver waits
and host descheduling still need to be separated. CPU-counter envelopes include
reset where documented; they are not identical to the invocation timer.

The [reset audit](SINGLE_LAYER_RESET_BOUNDARY_AUDIT.md) confirms that all 14
programs initially place tensors in DRAM only. Reset performs CPU writes; it
does not hide initial-weight or historical-KV H2D transfers outside the timer.
All explicit strategy copies remain inside the measured invocation. Reset can
still affect CPU cache/entry state, but no causal latency claim is made for it.

NVML utilization and power retain their rolling-window semantics. Query
containment does not make those hidden averaging windows contained. No telemetry
sample is interpolated into a short action or treated as exogenous contention.

## Formal independent protocol

The formal source snapshot contains 145 byte-verified files. All 14 geometry/strategy
cases, process counts, input seeds, candidate models, fixed gates and
Monte Carlo settings were declared before collection. A numerical seed offset
of 200,000,000 separates the formal cohort from development. Within the formal
cohort, profile and heldout inputs use disjoint seeds; sibling split policies
use the same underlying numerical weights for fair comparison.

Three fresh native processes per case and profile mode give 126 profile
processes. Only after **every** model is frozen does the heldout phase launch
42 new bare processes. Each process retains 101 samples after 30 warmups.
Heldout data cannot update any price or model. This validates new numerical
inputs and process/time cohorts at profiled shapes and policies; it is not a
heldout-shape or unprofiled-strategy generalization test.

The unchanged complete acceptance conjunction requires numerical/source checks,
at least three independent processes, compute/copy source within-process CV
≤15%, source between-process median CV ≤20%, heldout within-process CV ≤15%,
heldout between-process median CV ≤20%, instrumented/bare profile-cohort
relative difference ≤5%, and pooled-median absolute prediction error ≤20%.
The cohort instrumentation check is conservative and not a causal clock-cost
estimate. Microscopic gap CVs remain separately reported diagnostics, rather
than being relabelled as passing original source criteria.

Failures, all original samples, process inventories and provenance remain
visible. A small pooled error alone is not complete calibration acceptance.

## Formal results

The formal campaign completed **126/126 profile processes and 42/42 fresh heldout processes**, with all independent numerical/action checks passing. The primary context-action model has **8.54% median absolute relative error, 18.39% worst error, and 14/14 pooled-median errors within 20%**. The same heldout data give the isolated-component baseline a worst error of **82.61%**. These are matched-cohort comparisons, unlike comparisons against the historical 152.95% failure from another campaign.

**Complete quality acceptance is still 0/14.** Every case has at least one failed compute/copy source-stability criterion; only 2/14 satisfy all heldout within-process CV checks. All 14 satisfy the between-process heldout-median CV criterion, and 4/14 satisfy the conservative profile-cohort instrumentation comparison. No threshold was relaxed.

| Frozen alternative | Median error | Worst error | Errors ≤20% | All quality gates |
|---|---:|---:|---:|---:|
| Isolated components | 10.88% | 82.61% | 10/14 | 0/14 |
| Context compute/copy | 8.09% | 19.65% | 14/14 | 0/14 |
| Context actions + gaps (primary) | 8.54% | 18.39% | 14/14 | 0/14 |
| Distribution p50 (secondary) | 8.28% | 24.51% | 12/14 | 0/14 |
| Distribution expected mean (separate mean target) | 8.03% | 27.28% | 12/14 | 0/14 |

The expected-mean row compares against observed mean latency, while the other rows compare against the observed pooled median. Its last column preserves the original p50-based candidate gates; mean error is not an alternative acceptance criterion. The secondary distribution model does **not** consistently improve the primary model: p50 fails the 20% error threshold in two cases, and empirical p10–p90 coverage ranges from 23.10% to 95.38% (median 70.13%). The intervals are not calibrated probability guarantees. Keep this model as a diagnostic, not a promoted default.

| Complete layer / policy | Observed median (ms) | Isolated error | Primary prediction (ms) | Primary error | Secondary p50 error |
|---|---:|---:|---:|---:|---:|
| H64 q1/history128 / hybrid macros | 14.1898 | 10.43% | 12.9810 | 8.52% | 1.22% |
| H64 q1/history128 / cpu macros | 1.7278 | 6.41% | 1.5043 | 12.94% | 10.52% |
| H64 q1/history128 / gpu macros | 14.8042 | 13.65% | 14.4327 | 2.51% | 3.13% |
| H64 q1/history128 / Attention + FFN cut 60 | 10.4957 | 26.90% | 10.5402 | 0.42% | 13.13% |
| H64 q1/history128 / Attention + FFN cut 64 | 10.2967 | 25.12% | 10.8969 | 5.83% | 24.51% |
| H64 q1/history128 / Attention + FFN cut 68 | 10.3724 | 23.28% | 8.4646 | 18.39% | 3.80% |
| H128 q16/history0 / hybrid macros | 2.7663 | 10.86% | 2.6195 | 5.31% | 4.32% |
| H128 q16/history0 / Attention + FFN cut 124 | 3.0836 | 10.89% | 2.9622 | 3.94% | 3.40% |
| H128 q16/history0 / Attention + FFN cut 128 | 3.2950 | 82.61% | 3.7002 | 12.30% | 20.09% |
| H128 q16/history0 / Attention + FFN cut 132 | 3.3177 | 0.64% | 3.6228 | 9.20% | 18.84% |
| H256 q32/history0 / hybrid macros | 5.8657 | 17.13% | 5.3632 | 8.57% | 2.39% |
| H256 q32/history0 / Attention + FFN cut 252 | 6.7771 | 6.50% | 5.7584 | 15.03% | 6.04% |
| H256 q32/history0 / Attention + FFN cut 256 | 6.1612 | 6.18% | 6.7483 | 9.53% | 13.20% |
| H256 q32/history0 / Attention + FFN cut 260 | 6.3278 | 6.19% | 6.5151 | 2.96% | 16.91% |

The largest isolated-baseline error is H128/q16/FFN cut128: observed 3.2950 ms; isolated prediction 6.0171 ms; context compute/copy prediction 3.6447 ms; primary prediction 3.7002 ms. Its GPU PV price changes from 4.7333 ms in isolated profiling to 2.0870 ms in context profiling. Replacing isolated prices with context-scoped observations supplies the dominant numerical correction for that case. Mode, time and process state still differ between profile cohorts; this is not a causal estimate of an intrinsic context penalty. Adding control/gap intervals increases the prediction by only 0.0555 ms; it is not the main explanation. Context PV process medians still range from 1.5364 to 3.2309 ms, so the new price is not established as a stable universal constant.

The primary model is not better in every case. For H128/q16/cut132 the isolated baseline error is 0.64%, compared with the primary model’s 9.20%. We retain that result, use the predeclared primary for all cases, and do not select the best alternative after looking at each heldout result.

The supported conclusion is narrower than “single-layer inference is fully calibrated”: **prices collected in the real execution order eliminate the four >20% pooled-median failures of the isolated baseline in this fresh 14-case cohort**. Remaining within-run tails, source-regime changes and cohort incompatibilities prevent a full accepted accuracy claim.

## Where the remaining time is observed

The separate [operation-scoped CUDA diagnosis](SINGLE_LAYER_CUDA_WAIT_DIAGNOSIS_20261018.md)
validates all 808 selected PV ranges across two instrumented H128/q16 processes.
Each range has 52 correlated kernels, 33 device-to-device copies and two stream
synchronizations. Host PV p50/p90 is 1.898/4.916 ms, while the correlated GPU
activity union stays between 148.128 and 149.378 microseconds. Non-synchronization
CUDA runtime API **host wall intervals** occupy a median 90.90% of the host PV
range; the two explicit synchronization APIs occupy a median 0.666%.

For this representation and disabled-CUDA-graph configuration, the dominant
observable interval is therefore inside many backend runtime API calls rather
than the final synchronization or arithmetic on the GPU. API wall time can
include internal blocking and thread descheduling; it is not measured CPU
arithmetic. API and GPU intervals overlap and must not be summed. This supports
modeling backend materialization/call granularity and host/device overlap when
finer physical prices are required, instead of deriving a pure GPU rate from
completed PV latency or adding one fixed synchronization penalty.

The capture is diagnostic only: NVTX/CUPTI perturbs execution, some APIs are
explicitly skipped, and Nsight warns about its CUDA-driver compatibility. Exact
target/correlation counts pass, but those facts do not make traced timings valid
uninstrumented prices. None of these samples updates the formal frozen models.
The distinction between internal driver blocking and host descheduling remains
unresolved.

## What remains unresolved

A fixed switching penalty, inverse clock multiplier or independent marginal distribution is insufficient to explain all observed latency regimes. The additional controls isolate missing state/queue/wait behavior as an investigation target without attributing it to a particular operating-system or GPU mechanism. A separate operation-scoped NVTX/CUDA diagnostic has been collected; its profiler-instrumented timings are never fed into these frozen models or this acceptance result.

Current prices apply to their exact serial graph, shape, execution boundary and backend. They do not silently replace the library’s analytical costs, generalize to unseen mappings, or predict per-invocation tails reliably. The next physical decomposition must distinguish host submission, device activity, queueing and completion/resumption while respecting overlap; those measured intervals cannot simply be summed as mutually exclusive costs.

The reusable workflow, source guards, independent numerical oracle and recorded failures are deliverables alongside the improved point predictions. They do not turn an unresolved residual into a known calibrated physical parameter.

Before using these results to validate design-space exploration, separately test
policies and layouts not included in profiling. Measuring every action within a
known complete policy validates replay at that policy; it does not establish
composable predictions for arbitrary new splits, KV layouts or overlaps.

## Source version and validation

The checked-in source for the formal measurements is commit
`e447e4e1171523bfa46a10cd9e49c12acafcdc04`. The 145 recorded source
files also have a byte-verified local archive. Later evidence-only record-set
guards and whitespace cleanup do not alter performance equations or frozen
results; they must not be presented as a new hardware campaign. Use the recorded
commit and declared native/runtime build inputs to reproduce this campaign.

The original offline verification checks every numerical port and source byte;
the portable summarizer checks signatures, cached predictions and inventories
without repeating NumPy or native execution. The two verification scopes are
explicitly different. Actual profile/heldout file trees also match the declared
126/42 record sets exactly.

Local validation totals **2,734 distinct tests: 2,726 passed, eight existing
optional-dependency/artifact skips, zero failures**. This includes independent
wheel/sdist installation without profiles, the original full modeling/mapping
regressions, calibration evidence guards, native source generation and interval
attribution. Repeated driver checks are not counted twice. The
[verification receipt](data/single-layer-calibration-verification-20261018.json)
binds component counts, local logs and experiment chronology.

## Reuse and evidence

- [Source-checkout calibration workflow](../profiling/single_layer_calibration/README.md)
- [Same-process diagnostic workflow](../profiling/single_layer_context_pair/README.md)
- [Modeling/profiling responsibility boundary](PERFORMANCE_BOUNDARIES.md)
- [Portable formal results](data/single-layer-calibration-20261018.json)
- [Reset and COPY audit](SINGLE_LAYER_RESET_BOUNDARY_AUDIT.md)
- [Independent model integration review](SINGLE_LAYER_MODELING_AUDIT_20261018.md)
- [CUDA runtime/API decomposition](SINGLE_LAYER_CUDA_WAIT_DIAGNOSIS_20261018.md)
- [Verification counts and chronology](data/single-layer-calibration-verification-20261018.json)
- Read-only portable summarizer: `tools/summarize_single_layer_calibration.py`
- Local development campaign: `profiling/single_layer_calibration/results/diagnostic-v2/`
- Local paired campaign: `profiling/single_layer_context_pair/results/diagnostic-v1/`
- Local formal campaign: `profiling/single_layer_calibration/results/formal-v1/`

Native executables, runtime assets, large raw observations and build directories
remain local ignored artifacts. Portable aggregates bind their source files by
SHA256 but do not replace the raw observations for an independent audit.
