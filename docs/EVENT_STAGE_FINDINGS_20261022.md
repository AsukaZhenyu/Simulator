# What this correction establishes — 2026-10-22

**The implementation and measurement boundaries improved; heterogeneous
performance accuracy is not yet qualified.** The next priority is to identify
compute, transfer and recurrent runtime costs separately at representative
shapes. A periodic forecast of OS-related interference is not an adequate
substitute for a supplied available-resource trace.

The full tables are in [the experiment report](EVENT_STAGE_CALIBRATION_20261022.md).
The startup/steady-use distinction is in [timing boundaries](TIMING_METRICS.md).

## Results on 1,152 fresh, bare-target calls

| Predeclared hypothesis | Coverage | Median absolute error | P90 absolute error | Qualification |
|---|---:|---:|---:|---|
| PROFILE scalar baseline | 1,152/1,152 | 36.75% | 74.45% | fail |
| Cycle response, continuous BEFORE estimate | 1,152/1,152 | 14.84% | 40.90% | fail |
| BEFORE-only periodic host pause | 541/1,152 | 104.20% | 846.85% | fail; conditional errors only |
| Periodic pause with completed-past-call updates | 576/1,152 | 11.77% | 40.36% | fail; conditional errors only |

Percentiles use every applicable call; unknown phase is not replaced by a
default. The two overlap assumptions produce identical predictions in this
dataset because their fitted completion terms are all zero. They do not
establish that host/device overlap has no effect.

On the **same 576-call subset** where the history observer produces a forecast,
the continuous estimate has median/P90 errors of **14.77%/84.76%**, versus
**11.77%/40.36%** with history. This retrospective paired diagnostic suggests
useful information in completed history; it is not an additional selected
candidate or a full-coverage pass. The history candidate's worst error is
**1,272.24%**, from a badly placed predicted pause. It is not a safe default.

The continuous candidate's P90 errors are 26.80% idle, 40.97% CPU competition,
42.07% GPU competition and 85.79% joint competition. All four conditions fail.
All 12 heldout processes also have whole-call CV above 10%; variability is
retained, not filtered out to improve accuracy.

These are comparisons **within this round**. Changed parking and counter
protocols prevent treating differences from earlier experiment rounds as a
controlled improvement.

## Why startup is not the main explanation here

Executor/backend construction and 20 warmups finish before READY. The target
timer excludes process initialization, those warmups and the fixture reset.
Recurring submissions, launches, copies and synchronization remain inside.

The traced diagnostics measure GPU kernel envelopes of 0.098–0.126 ms and
memcpy envelopes of 0.091–0.108 ms per call, versus whole-call means of
6.707–12.226 ms. API unions are 5.806–11.408 ms. These overlapping envelopes
must not be summed and are not a causal decomposition. The tiny graph is
dominated by runtime/host response rather than representative large-matrix
GPU execution. Calling all of that response "cold startup" would be wrong;
calling all of it "useful computation" would also be wrong.

## Remaining identification failures

- **Counter interference:** event+cycle mean overhead is +6.96%, +1.81% and
  +5.55% across the three controls. The predeclared 5% gate fails. No overhead
  is subtracted from measured or predicted heldout latency.
- **Component separation:** all **1,404/1,404** local action/process
  unprojected completion residuals are negative. The constrained local fits
  consequently have zero completion terms. This is a failure to identify
  independent GPU completion demand from these scopes, not evidence that
  GPU work is free. Raw CPU cycles contain runtime/counter work and potentially
  spinning; the reference rate is a workload-specific empirical response.
- **Availability forecasting:** a recurring PROFILE pause has a candidate
  period of 64.178 ms and width of 32.121 ms. Recurrence does not establish a
  transferable future phase. The BEFORE-only calendar contradicts observed
  invocation entry in **329/541** known predictions; history reduces this to
  **45/576**, but false pauses remain severe. No claim about a specific OS
  scheduling mechanism follows from these timings alone.

The common modeling engine can integrate declared work against a supplied
time-varying service trace. **Predicting that future trace from stale probes is
a different accuracy problem.** Our simulator need not implement an OS
scheduler to represent its effect as external availability.

## Recommended work order

1. Keep small graphs for dependency, memory and overlap correctness. Use
   representative attention/FFN/matmul shapes for hardware-efficiency
   calibration; passing a noisy tiny-graph tail test is not a prerequisite to
   collecting those component profiles.
2. Independently profile device execution and transfer intervals, plus
   recurrent submission/synchronization microprograms. Establish which
   components overlap before assigning additive prices. Reduce or separately
   measure probe overhead; do not fit it into useful compute demand.
3. First validate trace replay with a declared or independently measured
   resource history. Report online forecasting with completed history as a
   separate optional capability, with coverage and uncertainty. Keep the
   periodic calendar explicitly diagnostic.
4. Compare execution/storage strategies using warm end-to-end strategy
   latency. Report useful activity and cold-start latency as complementary
   metrics. Include recurring offload and launch costs because changing the
   partition or granularity changes those costs.

## Verification and retained evidence

- 37/37 effective jobs pass the integrity audit. One controller file-read
  failure is retained and replaced as a whole same-configuration unit under
  an append-only index, independently of target latency. There were 209 normal
  process exits in accepted jobs and four normal exits in that failed attempt;
  no forced termination was needed.
- Software: **869 tests run, 862 passed, seven skipped**, with all suites
  successful. This validates software behavior, not hardware accuracy.
- Parameters and evaluation sources were frozen before heldout collection;
  frozen file SHA256 is
  `df3c620188cd058c5f854cea343c210587c8711392244654276098908045b08f`.
  Twenty-eight Python source files have byte-identical archived copies.
- Current target outcomes never generate their own predictions. Past target
  outcomes are used only by the explicitly named causal-history candidate.
  Heldout raw cycle counters, including reference counters, do not feed any
  predicted capacity or fitted parameter.
