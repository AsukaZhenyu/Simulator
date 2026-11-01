# Event-stage correction and fresh validation — 2026-10-22

**This round fixes measurement and modeling mechanisms. It does not establish an accurate, continuous available-resource model.**
The full qualification remains false: point accuracy, observer noninterference, source integrity and coverage are separate requirements. The cycle observer's 5% gate is reported independently and cannot be repaired by adjusting predicted means.

## What changed

1. Profiling actors block on parent-owned Windows GO/STOP events after READY. The old periodic file polling is retained only as a declared matched control. The executed layer actions and synchronization boundaries are unchanged.
2. Modeling conserves raw CPU cycle work using `cpu_cycle/s`. It does not convert cycles with an assumed GHz. PROFILE and independent BEFORE capacity estimates use the same complete reference-iteration wall denominator.
3. A submission graph separates enqueue acceptance, device-stream execution and completion waits. An already-enqueued device command can continue while the host has no CPU service. Inclusive host response cannot also be charged as device work.
4. A coarse cycle-response graph tests explicit `after_host` and `overlap_host` assumptions in the common evaluator. Its fitted completion response is not measured GPU-active service or a reconstructed CUDA submission DAG.
5. A fixed-parameter phase observer distinguishes BEFORE-only forecasting from updates using completed PAST target calls. Each prediction is sealed before that call's outcome is extracted; current/future target latency never supplies its own prediction.

## Protocol and evidence

- PROFILE: 12 new processes × 96 calls; fresh bare-target heldout: 12 processes × 96 calls (1152 calls). Three CPU/GPU contention diagnostics use CUPTI; nine idle controls compare polling, events and cycle instrumentation.
- One H=128, q=16, history=0 F32 dense layer, fixed single-CPU-thread placement and the same RTX 4070 Laptop GPU. This is a small controlled mixed CPU/GPU workload, not production-scale LLM validation.
- PROFILE target cycle counters are enabled; heldout target counters are disabled. Heldout reference counters are collected but never read as prediction inputs. Legal capacity inputs are frozen PROFILE work and independent BEFORE reference wall/QPC observations.
- Every heldout call, including long pauses, contributes to its applicable metric. Unknown calendar phase remains unknown. Conditional known-subset errors cannot establish full coverage.
- Invocation entry timestamps are observed legal inputs. These are conditional per-call predictions, not an autonomous forecast of an entire token stream's future start times.
- Frozen parameter file SHA256: `df3c620188cd058c5f854cea343c210587c8711392244654276098908045b08f`. No heldout fitting, DNN, state-label lookup or whole-layer offset is used.
- Audit: 37/37 declared jobs included; 0 excluded; 0 pending; 0 source errors. Integrity audit pass: **True**.

- Retained infrastructure failures: **1**. The Windows completion-file read failure is excluded as a whole unit and replaced under an append-only same-configuration index. Original raw files and hashes remain intact; target latency never selects a retry.

Heldout process stability (whole-call CV ≤10%): **0/12**. This describes measured variation, not whether a time-varying model correctly predicts individual slow calls.

## Observer controls

| Repeat | Poll mean (ms) | Event mean (ms) | Event + cycles mean (ms) | Event versus poll | Cycles versus event |
|---|---:|---:|---:|---:|---:|
| 0 | 4.2969 | 2.7988 | 2.9936 | -34.87% | 6.96% |
| 1 | 2.9865 | 2.8120 | 2.8628 | -5.84% | 1.81% |
| 2 | 2.8542 | 2.7522 | 2.9049 | -3.57% | 5.55% |

Cycle-observer ≤5% gate: **fail**. These are separate process controls in fixed declared order. Their differences are observed protocol effects, not a randomized causal attribution. No observer overhead is subtracted from heldout measurements or predictions.

## Fresh heldout results

| Declared candidate | Known / all calls | Median APE | P90 APE | Worst APE | Full point gate |
|---|---:|---:|---:|---:|---|
| profile_scalar_median | 1152/1152 | 36.75% | 74.45% | 88.40% | fail |
| cycle_after_host_no_calendar | 1152/1152 | 14.84% | 40.90% | 90.97% | fail |
| cycle_after_host_before_only | 541/1152 | 104.20% | 846.85% | 1275.17% | fail |
| cycle_after_host_causal_history | 576/1152 | 11.77% | 40.36% | 1272.24% | fail |
| cycle_overlap_host_no_calendar | 1152/1152 | 14.84% | 40.90% | 90.97% | fail |
| cycle_overlap_host_before_only | 541/1152 | 104.20% | 846.85% | 1275.17% | fail |
| cycle_overlap_host_causal_history | 576/1152 | 11.77% | 40.36% | 1272.24% | fail |

APE = |predicted − observed| / observed. Percentiles pool all applicable calls, rather than averaging process percentiles. The point gate requires complete coverage, median APE ≤20% and P90 APE ≤20%. Hardware qualification additionally requires the observer and validity gates.

### Offered-load condition: both

| Declared candidate | Known / all calls | Median APE | P90 APE | Worst APE | Full point gate |
|---|---:|---:|---:|---:|---|
| profile_scalar_median | 288/288 | 15.19% | 87.46% | 88.40% | fail |
| cycle_after_host_no_calendar | 288/288 | 12.46% | 85.79% | 87.62% | fail |
| cycle_after_host_before_only | 253/288 | 86.09% | 278.94% | 508.86% | fail |
| cycle_after_host_causal_history | 288/288 | 7.64% | 33.95% | 757.37% | fail |
| cycle_overlap_host_no_calendar | 288/288 | 12.46% | 85.79% | 87.62% | fail |
| cycle_overlap_host_before_only | 253/288 | 86.09% | 278.94% | 508.86% | fail |
| cycle_overlap_host_causal_history | 288/288 | 7.64% | 33.95% | 757.37% | fail |

### Offered-load condition: cpu

| Declared candidate | Known / all calls | Median APE | P90 APE | Worst APE | Full point gate |
|---|---:|---:|---:|---:|---|
| profile_scalar_median | 288/288 | 49.71% | 80.92% | 87.20% | fail |
| cycle_after_host_no_calendar | 288/288 | 15.63% | 40.97% | 90.97% | fail |
| cycle_after_host_before_only | 288/288 | 390.02% | 1002.08% | 1275.17% | fail |
| cycle_after_host_causal_history | 288/288 | 15.25% | 146.63% | 1272.24% | fail |
| cycle_overlap_host_no_calendar | 288/288 | 15.63% | 40.97% | 90.97% | fail |
| cycle_overlap_host_before_only | 288/288 | 390.02% | 1002.08% | 1275.17% | fail |
| cycle_overlap_host_causal_history | 288/288 | 15.25% | 146.63% | 1272.24% | fail |

### Offered-load condition: gpu

| Declared candidate | Known / all calls | Median APE | P90 APE | Worst APE | Full point gate |
|---|---:|---:|---:|---:|---|
| profile_scalar_median | 288/288 | 13.85% | 31.47% | 46.06% | fail |
| cycle_after_host_no_calendar | 288/288 | 16.52% | 42.07% | 78.11% | fail |
| cycle_after_host_before_only | 0/288 | unknown | unknown | unknown | fail |
| cycle_after_host_causal_history | 0/288 | unknown | unknown | unknown | fail |
| cycle_overlap_host_no_calendar | 288/288 | 16.52% | 42.07% | 78.11% | fail |
| cycle_overlap_host_before_only | 0/288 | unknown | unknown | unknown | fail |
| cycle_overlap_host_causal_history | 0/288 | unknown | unknown | unknown | fail |

### Offered-load condition: idle

| Declared candidate | Known / all calls | Median APE | P90 APE | Worst APE | Full point gate |
|---|---:|---:|---:|---:|---|
| profile_scalar_median | 288/288 | 52.82% | 74.24% | 87.28% | fail |
| cycle_after_host_no_calendar | 288/288 | 13.44% | 26.80% | 42.42% | fail |
| cycle_after_host_before_only | 0/288 | unknown | unknown | unknown | fail |
| cycle_after_host_causal_history | 0/288 | unknown | unknown | unknown | fail |
| cycle_overlap_host_no_calendar | 288/288 | 13.44% | 26.80% | 42.42% | fail |
| cycle_overlap_host_before_only | 0/288 | unknown | unknown | unknown | fail |
| cycle_overlap_host_causal_history | 0/288 | unknown | unknown | unknown | fail |

## What the fitted quantities can and cannot establish

The PROFILE-only host-pause hypothesis has period `64177900.0` ns and width `32121100.0` ns. Phase is supplied separately, never fitted from a heldout call's own outcome. A periodic host-response pattern does not prove an OS scheduling mechanism or independently measure neutral CPU availability.

The GPU reference feature includes host launch/wait response. CPU work and completion residuals are therefore scoped empirical response estimates. Nonnegative least squares being solvable does not establish causal identification, a neutral GPU capacity or independent physical components. Negative unprojected residuals remain in the frozen evidence; the constrained fit is explicitly underidentified.

The completion residual is calibrated under an additive `after_host` assumption. Reusing it under `overlap_host` tests structural sensitivity; it is not a separately calibrated physical execution model. Fixed head/tail bounds and PROFILE counter-query work can retain instrumentation bias.

The BEFORE references are stale workload-response probes. Causal-history updates use past target outcomes and must be reported separately from open-loop prediction. Calendar center predictions and assumed uncertainty growth are diagnostic forecasts, not physically proven future resource trajectories.

## Startup, useful work and strategy latency

Process/Executor construction and 20 warmups occur before READY and outside the per-invocation timer. The isolated benchmark reset is also outside that timer. Recurrent submissions, launches, copies and synchronizations remain inside. We optimize warm strategy latency while retaining activity diagnostics; a long API wall is not automatically a kernel-launch cost. See [Timing boundaries](TIMING_METRICS.md).

| Traced diagnostic | Whole-call mean (ms) | GPU kernel envelope (ms) | GPU memcpy envelope (ms) | API union (ms) |
|---|---:|---:|---:|---:|
| cpu | 8.3313 | 0.1264 | 0.1080 | 5.8129 |
| gpu | 6.7067 | 0.1135 | 0.1010 | 5.8062 |
| both | 12.2259 | 0.0975 | 0.0910 | 11.4080 |

These columns overlap and must not be added. GPU intervals are activity envelopes, not guaranteed useful arithmetic service. Pure CPU compute time remains unknown. CUPTI-instrumented diagnostics are separate from the bare-target accuracy cohort.

## Remaining accuracy work

- Lower or explicitly isolate per-action counter overhead before treating PROFILE parameters as noninterfering measurements.
- Identify enqueue/host/wait components with independently timed backend microprograms; do not infer GPU active demand by subtracting host wall time.
- Validate availability traces and recovery behavior independently of the target outcome. Separate replay with a supplied resource trace from online prediction of an unknown future trace.
- After the small layer passes both median and tail gates, broaden to different shapes, token phases, partitions and hardware. This round cannot justify generalizing to production Dense/MoE strategies.

## Reproduce

```powershell
python -m profiling.event_stage_evaluate evaluate --campaign profiling/available_service/results/event-v1-h128-t1 `
  --frozen profiling/available_service/results/event-v1-h128-t1/profile-frozen.json `
  --barrier profiling/available_service/results/event-v1-h128-t1/event-frozen-barrier.json --output build/event-reevaluation.json
python -m profiling.event_stage_audit profiling/available_service/results/event-v1-h128-t1 --output build/event-reaudit.json
```

Existing outputs are write-once. Use new output paths for verification; do not overwrite frozen evidence.
