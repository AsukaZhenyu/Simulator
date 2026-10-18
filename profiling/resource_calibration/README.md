# Unified resource-trajectory calibration

This optional source-checkout workflow fits **one execution-time functional**
from numeric resource inputs. It does not select a model by a named background
condition. See the [modeling contract](../../docs/UNIFIED_RESOURCE_FUNCTIONAL.md).
The installed library remains usable without profiles or CUDA.

The [first implementation and hardware check](../../docs/UNIFIED_RESOURCE_RESULTS_20261018.md)
records 241 passing relevant software tests and a completed 16-state smoke;
the first fitted response still fails hardware accuracy requirements.

## Inputs and prediction

An exact operation contract fixes shape, dtype, hardware/backend, execution
context and completed-work timing boundary. Its varying inputs are CPU pressure,
GPU pressure, DRAM allocations, and VRAM allocations. A frozen response surface
provides the operation's effective service rate at a state; the same prediction
path integrates remaining work over a supplied time trace.

`functional.py` is standard-library-only. It implements training-only fitting,
explicit source/protocol/clock identities, exact rational progress, bounded
support, independent heldout checks, and capacity checks separate from timing.
CPU, GPU and copy operation responses use the same numerical interface. A small
GPU-PV hardware campaign does not calibrate the CPU and copy responses as well.

The first feature basis is `[1, cpu, gpu, dram/capacity, vram/capacity, cpu*gpu]`.
Ridge regularization is declared **before collection** (default `1/1000`, with an
unpenalized intercept). Memory features can be much smaller than compute
features, so this choice shrinks memory coefficients strongly. It is not evidence
that memory allocations have no effect. The model makes no causality or accuracy
claim. It rejects unknown operation contracts, mismatched input meanings,
out-of-range features, nonpositive predictions, and incomplete input traces.

## Controlled input contract

Hardware collection currently uses `offered_load_controlled`, not measured
external occupancy percentages:

| Numeric input | Actual intervention |
|---|---|
| CPU `0/1` | No owned CPU replay / one owned native CPU replay |
| GPU `0/1` | No owned GPU replay / one owned native GPU replay |
| DRAM bytes | Owned, page-touched Python allocation |
| VRAM bytes | Owned explicit `cudaMalloc` allocation, initialized before readiness |

Total GPU/system utilization, clocks, power, memory counters and owned-process
CPU activity are collected separately as diagnostic observations. They are not
used as target-inclusive explanatory variables. Replay workers also initialize
backend contexts and buffers: their side effects are not magically isolated to
one physical resource. Owned allocation bytes are not total external memory,
working-set residency, bandwidth pressure, or a physical capacity certificate.

Each trajectory contains all 16 on/off combinations in an independently shuffled
order. Defaults are 64 MiB for each allocation and 1000 ms steady holds. The
foreground continues in the **same native process** across all phases. Requested
commands, initialization/teardown intervals, actual readiness and steady holds
are distinct recorded intervals. Background invocation-bracket overlap is
recorded, with its wait/descheduling limitation retained.

The physical example is the existing small F32 H128/q16 single-layer Dense graph,
with CPU/GPU Attention stages and FFN partitions. The modeled hardware target is
its GPU `attention_pv` action; the complete graph and all action intervals are
also retained. `mixed` replays actions serially; `isolated` repeats only PV.
Prebuilt graphs, completed-work synchronization and disabled CUDA graphs are
explicit constants. This is not pretrained generation or a production decode
benchmark.

## Run

Install NumPy for independent numerical references with
`python -m pip install ".[profiling]"`. Supply an explicitly built native probe
and its compatible runtime DLLs; see the
[native build guide](../native/state_calibration/README.md).

Run from the repository root. Always use a new output directory:

```powershell
python -m profiling.resource_calibration.collect prepare `
  --out profiling/resource_calibration/results/run-new `
  --binary profiling/native/state_calibration/build-direct-v4/state-calibration-probe.exe `
  --runtime-dir profiling/native/highlevel_accuracy/build-runtime-f32-v2/bin `
  --runtime-dir "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin" `
  --cuda-runtime "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/cudart64_12.dll"

python -m profiling.resource_calibration.collect collect `
  --out profiling/resource_calibration/results/run-new --split train

python -m profiling.resource_calibration.evaluate fit `
  --campaign profiling/resource_calibration/results/run-new `
  --model profiling/resource_calibration/results/run-new/frozen-model.json

python -m profiling.resource_calibration.collect collect `
  --out profiling/resource_calibration/results/run-new --split heldout `
  --frozen-model profiling/resource_calibration/results/run-new/frozen-model.json

python -m profiling.resource_calibration.evaluate evaluate `
  --campaign profiling/resource_calibration/results/run-new `
  --model profiling/resource_calibration/results/run-new/frozen-model.json `
  --out profiling/resource_calibration/results/run-new/evaluation.json
```

`prepare --smoke --phase-ms 500` keeps all 16 states but uses one foreground
process per split. It is an integration check with insufficient independent
processes for hardware acceptance. The normal protocol uses three train and
three heldout processes, independent numerical seeds and independently shuffled
phase orders. Hardware capacity totals must actually be observed; missing totals
are not replaced with defaults.

The collector owns only the children it starts, bounds their runtime and memory,
and records cleanup. It does not close user applications, alter clocks or power
settings, or change global timer resolution. Collection shares the existing
exclusive measurement lock. Do not run builds or regression suites concurrently.

## Evaluation boundaries

Fit weights are equal per run/steady-phase, using exact medians of **all** its
retained complete target invocations. More repetitions in a faster phase do not
give that phase more regression weight. Raw CV is computed from individual
invocations; it is not replaced by CV of smoothed averages.

One model is frozen and verified against its actual training rows before any
heldout process launches. Heldout timings cannot refit coefficients or tune
regularization. The evaluator reconstructs numerical checks, phase assignments,
source and raw-file hashes, and reports unknown/transition coverage explicitly.
The original 15% within-process CV, 20% between-process CV, 20% timing-error and
three-independent-process requirements remain relevant; a smoke run cannot waive
them. Diagnostic artifacts conservatively make no hardware acceptance claim.

Transitions remain unpriced where the actual offered input is not known during
worker initialization or teardown. Therefore phase-wise hardware errors validate
only settled-segment response. Exact cross-transition integration is tested in
software, but cross-transition **hardware accuracy remains unverified**.
The first response is memoryless in the current input vector. Thermal history,
driver state, memory clocks, polling and host scheduling can still be missing
state. Unexplained error remains error.

```powershell
python -m unittest discover -s profiling/resource_calibration/tests -v
python -m profiling.resource_calibration.evaluate verify `
  --campaign profiling/resource_calibration/results/run-new `
  --model profiling/resource_calibration/results/run-new/frozen-model.json
```

Raw results stay under ignored `results/`. Old campaigns and rejected calibration
evidence are not rewritten to conform to this newer model.
