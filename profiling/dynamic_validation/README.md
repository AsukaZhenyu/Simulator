# Dynamic-resource validation

This directory adds new experiments. Historical H2D history sources, binaries,
prices and acceptance outcomes are unchanged. Hardware collection is serialized
through the repository's `profiling/.measurement.lock`.

## First experiment: idle and recovery

The standalone `native/idle_probe.cpp` executes the actual 2 MiB `W_up` H2D
action from the retained two-layer Dense Executor arena. It uses the unchanged
inclusive `Executor.action` boundary, with no CUPTI or tensor logging in primary
timers. It does not identify physical PCIe active time or a unique driver cause.

Each process has one explicit host-memory mode. The entire owned DRAM arena is
either observed pageable or explicitly registered and restored. Each condition
has the same reset and complete untimed prime, then a requested idle followed by
sixteen target copies. Sixteen trials balance the four condition positions
within the process. Zero idle skips the sleep call; it does not execute
`Sleep(0)`. Requested and actually observed idle intervals are separate fields.

The profile grid is 0, 1, 5 and 20 ms; heldout is 0, 2, 10 and 20 ms. Each split
has three fresh processes per memory mode, six total. Heldout uses different
initial values and buffers. Its acquisition is blocked until a source-bound
parameter file and all exact requested-idle/ordinal predictions are frozen.

Build the CMake target with explicit existing GGML, generated Executor, CUDA and
import-library paths. On the current Windows toolchain, x64-host MSVC and
`CMAKE_TRY_COMPILE_CONFIGURATION=Release` avoid a local Debug PDB mismatch. No
historical source tree or executable needs rebuilding.

```text
python -m profiling.dynamic_validation.idle_campaign prepare --out NEW --binary PROBE
python -m profiling.dynamic_validation.idle_campaign profile --out NEW
python -m profiling.dynamic_validation.idle_campaign summarize-profile --out NEW
```

`prepare --smoke` declares only one pageable profile process for protocol
checking; it is not the six-process calibration cohort. Preparation and the
native parser self-test do not execute the target. Collection launches hardware
work and must not overlap other validation campaigns.

After profile inspection, select and fit an interpretable model, freeze its
parameters and all heldout predictions, then use:

```text
python -m profiling.dynamic_validation.idle_campaign freeze --out NEW --model MODEL --predictions PREDICTIONS
python -m profiling.dynamic_validation.idle_campaign heldout --out NEW
python -m profiling.dynamic_validation.idle_campaign summarize-heldout --out NEW
```

Predictions use schema `dynamic-idle-frozen-predictions-1` with rows containing
`memory_mode`, `requested_idle_ms`, `ordinal` and `predicted_ns`. All 128 unique
memory/idle/ordinal coordinates are required. Profile files only estimate
parameters; no complete Dense latency or heldout outcome enters a fit.

The modeling candidate in `performance/idle_recovery.py` has no default
coefficients. Its scalar proxy fills during idle and decays during its own
predicted completed invocations. It is an optional empirical hypothesis whose
scope is this declared boundary, payload, memory category and preparation. It
does not assert a real OS queue or power state, and it is not automatically the
general CPU/GPU response model. The law must be selected from profile evidence
and independently tested; merely implementing it does not establish accuracy.

Actual completed idle can be used only for separately declared conditional
diagnosis: it was known at target start, but it is not the pre-run requested
timeline. Primary forecasts must use the declared planned input and report
controller adherence instead of silently replacing it with future observations.

## Next experiment: external plans

The next target is two continuous Dense streams (GPU-oriented and CPU-Attention
plus GPU-FFN), each with four predeclared external plans: no actor, CPU on/off,
GPU on/off, and staggered CPU/GPU. Three fresh processes per cell give 24 heldout
processes. External work must start and stop from a fixed common clock, not from
target completion. Actor duty cycle is offered work, not directly available
compute percentage. Memory reservation and bandwidth competition are distinct.

The external controller and native actor campaign are implemented separately
from the idle probe. The retained `external-v1` campaign completed its 24
profile processes, but its primitive-state support gate refused a fit: the
joint-actor GPU Attention response was not observed entirely inside one state.
No independent external-plan accuracy result follows from those profiles.
The idle cohort must not be presented as completed external-resource validation.

No raw CV threshold is a prediction-accuracy gate. Report original CV and
observer effects separately from per-process mean error, per-burst error and
transient response error. Keep failed processes, timeouts, unsupported response
coordinates and all raw timing observations in their declared denominators.

## Repair diagnostics (27 October 2026)

`baseline_audit.py` tests the single-baseline assumption before `idle_fit.py`
can estimate parameters. It retains all zero-idle suffix samples and checks
chronological early/late drift and independent-process spread. A refusal is a
model-support result, not evidence that the fluctuation is inherently random.
Historical frozen fits and their results are retained unchanged.

The versioned diagnostics keep collection and performance calculation separate:

- `h2d_v3.py` and `h2d_v3_analysis.py` correlate API and device-copy intervals
  and compare pinned versus wide initialization. Inclusive host response is
  partitioned; device activity is never added a second time to host waits.
- `h2d_v4.py` uses same-process ABA/BAB observer contrasts and records actual
  PCIe link states. Missing direct correlation is unsupported, never filled by
  a nearest-timestamp match. Pre/post link agreement does not prove that every
  copy inside the burst experienced a constant link state.
- `external_v5.py` locates long Attention response inside runtime launch/copy
  calls, with separately correlated short GPU activities. Its in-run activity
  finalization changes later actor exposure; subtracting that gap cannot
  reconstruct the unobserved control execution.
- `external_v6.py` records only target-thread Attention runtime callbacks into
  preallocated records, and serializes after the target and actor join. It
  compares a CPU actor on the target core with a verified different physical
  core. API enter-to-exit spans are inclusive responses, not active CPU work,
  GPU service time, or identified enqueue-acceptance points.

Each diagnostic has its own manifest, binary/source hashes, original control
and all failed/unsupported observations. Observer and state-coverage gates
must pass before phase prices can be transferred to a predictor. Diagnostic
completion alone does not establish hardware prediction accuracy.

Multi-size, state-conditional H2D collection lives in `../link_calibration/`.
The modeling-side `performance/pcie_response.py` only selects measured
`TransferResponse` parameters and checks provenance, scope and causal timing.
It never converts theoretical PCIe generation rates into measured costs. Its
existence does not qualify any current calibration dataset.

## Continuous lifecycle calibration (28 October 2026)

`h2d_bare_v9.py` measures the original completed GGML COPY response without
adding CUPTI, NVML queries or CUDA events in the target loop. Named affine,
idle-recovery and prepared-state parameters are estimated by profiling;
`modeling.performance.transfer_recovery` owns the equations. Failed baseline
support remains failed. `h2d_v9_diagnostic_holdout.py` can separately score
already frozen, unqualified hypotheses on new sizes/idles/processes; that
diagnostic neither exports qualified costs nor changes the original gate.

The current complete-Attention path is `continuous_macro_campaign.py` with
`native/v10/continuous_probe.cpp` and `continuous_macro_fit.py`. It records one
calibration-only warm invocation followed by all consecutive invocations until
the predeclared duration or safety cap. Prices are averaged within each fresh
context, then equally across contexts. This matches the foreground lifecycle
more closely than transferring a single-call price to a continuous stream.
Warm responses, every priced call, intercall gaps and stop reasons are retained.
An observation lasting a second does not itself prove physical stationarity.

The same `modeling.performance.PlannedCompletedResponse` integrates the four
offered CPU/GPU regimes on its own predicted completion clock. Profiling does
not add a whole-target multiplier or infer active CPU/DMA work from inclusive
response time. Parameters and all forecasts are frozen before independent
processes and schedules. Heldout entry and accuracy thresholds remain explicit;
recorded target boundaries never reanchor a primary forecast.

`continuous_price_ablation.py` freezes a separate same-context comparison of the
first invocation versus later continuous prices before validation. Both score
the same raw targets; the former remains an explicitly mismatched diagnostic
baseline. These experiments concern one fixed-shape Attention component and
declared GEMM actors, not general whole-LLM dynamic accuracy or an OS simulator.
