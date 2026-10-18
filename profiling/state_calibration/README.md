# State-aligned performance calibration

This optional source-checkout workflow measures identical numerical programs
under declared execution regimes, records device observations on the same
monotonic clock as native execution, and validates profile-only conditional
costs in fresh heldout processes. It does not require any previously generated
profile. The installed simulator continues to work without this workflow.

Install the optional numerical-reference dependency with
`python -m pip install ".[profiling]"`. Native builds require an explicitly
supplied GGML/CUDA development environment; see
[the native build guide](../native/state_calibration/README.md).

## What is measured

The prepared suite generates genuine public `ModelPlan` compilations and exact
native programs, not hand-written substitutes. Two formerly problematic F32
Attention PV geometries are included: H128/q16 and H256/q32, with supported
materialized Attention/FFN transforms. Every primary process checks all output
ports against an independent NumPy reference outside the timed work.

Each primary cell has three fresh profile and three fresh heldout processes.
Identical profile matrices are reused across conditions and execution contexts;
heldout numerical seeds are disjoint. Each process retains all 256 invocations
in each of three blocks. Individual invocation samples, not block means, feed
the unchanged CV gates. A block shorter than its declared minimum remains a
failed coverage/alignment observation and is never silently extended.

The declared regimes are baseline, an owned CPU-target replay and an owned
GPU-target replay. Backgrounds use bounded native processes and a cooperative
stop file. Their actual recorded invocation envelopes establish overlap;
readiness only establishes completed initialization/warmup. Both background
types own a full native executor, including resident CUDA buffers. The CPU
condition therefore is not a pure intervention on CPU utilization alone.

No user applications are closed, and no clocks, power policies or global timer
resolution are changed. [Telemetry semantics](TELEMETRY.md) distinguish bounded
observations, unsupported counters, rolling utilization, memory occupancy and
coarse temporal resolution. Short actions are never assigned a nearby sample
by interpolation. Sustained blocks permit descriptive state association.

The campaign default is a declared 50 ms interval, with a private Windows
high-resolution waitable timer when available. A prior 10 ms pilot exposed
missed deadlines and expensive queries; its records are retained. The longer
declared interval reduces observer pressure and does not relax any quality
gate or imply per-action state resolution. Actual query bounds and missed
deadlines are retained at either interval.

## Execution boundaries and observation controls

The primary workflow retains the historical completed-work boundary, named
`legacy_double_sync`. The underlying `ggml_backend_graph_compute` already waits
for completion; the historical executor then synchronizes again. Two additional
diagnostic boundaries operate on the selected GPU target only:

- `async_one_sync`: submit the graph asynchronously and synchronize once;
- `async_events`: the same asynchronous boundary with timing-enabled CUDA
  events recorded on the actual GGML stream.

These alternate boundaries are diagnostic jobs, excluded from every model fit.
They expose host submission/wait intervals and CUDA stream spans. A stream span
includes host submission gaps between device commands; it is not kernel-active
time and omits waiting before the start event. Event instrumentation changes the
boundary and must be compared with the matched event-free diagnostic.

Separate controls toggle per-action timestamping and device sampling. Their
matched-cohort ratios include fresh-process variation. No overhead is subtracted
from any observation, and a lower error does not override failed quality gates.

## Build, collect and verify

Use new output paths. Historical sources, binaries, raw samples and rejected
campaigns are never overwritten. Run software tests and builds before hardware
collection; all native collections share `profiling/.measurement.lock`.

```powershell
python profiling/state_calibration/run.py prepare `
  --out profiling/state_calibration/results/campaign-new `
  --binary profiling/native/state_calibration/build-new/state-calibration-probe.exe `
  --runtime-dir profiling/native/highlevel_accuracy/build-runtime-f32-v2/bin `
  --runtime-dir "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin"

python profiling/state_calibration/run.py collect `
  --out profiling/state_calibration/results/campaign-new --split diagnostic
python profiling/state_calibration/run.py collect `
  --out profiling/state_calibration/results/campaign-new --split profile
python profiling/state_calibration/run.py collect `
  --out profiling/state_calibration/results/campaign-new --split heldout

python -m profiling.state_calibration.verify `
  profiling/state_calibration/results/campaign-new --output new-verification.json
python profiling/state_calibration/analysis.py `
  profiling/state_calibration/results/campaign-new --output new-description
python profiling/state_calibration/diagnose_background.py `
  --campaign profiling/state_calibration/results/campaign-new --output new-background.json
```

`prepare --smoke` generates one geometry and one process per primary cell. It
tests integration but cannot meet the three-process statistical gate. The native
binary and supplied shared libraries are source-bound; this workflow does not
observe every resolved operating-system loader path or CUDA kernel selection.

The read-only background diagnostic verifies retained identities, hashes and
clock intervals before clipping invocation brackets to main blocks and target
actions. Bracket coverage includes waiting and is not CPU/GPU active occupancy.
It never fits heldout features or modifies the frozen model. The optional
`plot_results.py` script uses Matplotlib to visualize a portable evidence summary;
Matplotlib is not a simulator runtime dependency.

`thread_context.py --campaign CAMPAIGN --out NEW` runs a separate, bounded
12-process CPU-thread-configuration diagnostic using the campaign's exact H128
graph and numerical reference. It compares one/two requested CPU threads in
isolated/mixed execution, using three minimum-wall-time blocks and no new model
fit. It retains variable invocation counts and must not be merged into the
fixed-count formal acceptance table. Run it only after other local tests and
hardware measurements have finished.

## What the calibrated model means

`model.py` freezes two models using only declared profile rows. Both have the same
exact geometry, execution context, timestamping and sampling boundaries. The
constant model pools the observed named conditions; the conditional model gives
each verified condition its own price. A missing condition, shape or context is
unknown, including for the constant comparison. No interpolation, hidden
fallback, utilization-to-slowdown rule or heldout fitting is performed.

Source and target within-process CV must be <=15%, between-process median CV
<=20%, heldout pooled-median relative error <=20%, with at least three independent
process records per split, equal invocation counts and full numerical/protocol
coverage. Conditional prices can remain descriptive when these gates fail;
failed or unknown scopes stay in the denominator.

The [runtime adapter](runtime_adapter.py) binds an exact measured GPU target to
a sealed one-operation timing carrier using rational service rates. It refuses
to apply a PV price to an arbitrary LLM graph, CPU work or a whole-iteration
price. Unaccepted source prices require explicit diagnostic opt-in. The carrier's
small synthetic memory fixture is not a physical memory model for the target.
See [runtime semantics](../../docs/DEVICE_STATE_RUNTIME.md) for mid-operation
rate changes, exact remaining work, causal observations and checkpoint evidence.

The experiments do not establish one universal GPU slowdown factor, a causal
relationship from telemetry to latency, pretrained generation, arbitrary
external applications, quantized arithmetic, SSD/UMA behavior or physical VRAM
capacity. Private CUDA workspace remains unknown.

## Startup precision is part of the contract

`NVIDIA_TF32_OVERRIDE=0` and graph-control flags must be supplied in the child
environment before process creation. Setting them only in native `main` can be
too late for dependency initialization. An exact-input paired diagnostic
observed a numerical mismatch with late setup and a pass with prelaunch setup,
at the unchanged `5e-5` absolute/relative tolerance. The rejected launch protocol
and raw outputs are retained. Runtime flags reported after startup alone cannot
prove the arithmetic path used during initialization.

```powershell
python -B -m unittest discover -s profiling/state_calibration/tests -v
python -B -m unittest discover -s profiling/native/state_calibration -p 'test_*.py' -v
```
