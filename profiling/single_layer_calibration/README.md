# Single-layer latency calibration

This optional source-checkout workflow separates measurement and parameter
estimation in **profiling** from execution-time calculation in **modeling**. The
installed library still works without measurement files, NumPy, CUDA or a native
probe. No previously generated profile is required to prepare a new campaign.

The bounded executor runs the actual compiled single-layer F32 Dense action
sequence, including explicit Attention/FFN splits and copies. It uses prebuilt
GGML graphs, materialized independent tensor ports, serial execution and disabled
CUDA graphs. Numerical validation compares every final port with an independent
NumPy reference. This is not an original-model kernel or physical-capacity claim.

Three measurements retain different questions:

* `isolated-components`: does repeating the same action produce a price that
  transfers to a complete mixed CPU/GPU sequence?
* `context-trace`: how much completed host time belongs to each action in the
  actual sequence, including control actions and the gaps between actions?
* `bare-replay`: how long does the entire sequence take with invocation clocks
  alone? This is the separate full-layer validation boundary.

Timestamp records use preallocated POD storage; JSON assembly and input reset
stay outside invocation timers. Context clocks can still perturb execution.
Default polling and explicitly requested `poll=0` are separate execution
variants at the same configured thread count. The workflow does not silently
change affinity, clocks, power, precision or CUDA graph settings.

Build the [native wrapper](../native/single_layer_calibration/README.md) in a new
directory using explicitly supplied GGML sources, libraries and runtime assets.
Then run the phases in order, from the repository root:

```powershell
python profiling/single_layer_calibration/run.py prepare `
  --suite diagnostic --processes 3 --warmup 30 --samples 101 `
  --poll default --seed-offset 200000000 `
  --distribution-trials 20000 --distribution-seed 20261018 `
  --executable profiling/native/single_layer_calibration/build-release-v2/Release/single-layer-probe.exe `
  --runtime-dir profiling/native/highlevel_accuracy/build-runtime-f32-v2/bin `
  --out profiling/single_layer_calibration/results/new-campaign
python profiling/single_layer_calibration/run.py profile --out profiling/single_layer_calibration/results/new-campaign
python profiling/single_layer_calibration/run.py freeze --out profiling/single_layer_calibration/results/new-campaign
python profiling/single_layer_calibration/run.py heldout --out profiling/single_layer_calibration/results/new-campaign
python profiling/single_layer_calibration/run.py evaluate --out profiling/single_layer_calibration/results/new-campaign
```

The paths above are examples of explicitly built local artifacts. Supply your
own successful build/runtime paths. `prepare` does not execute hardware. Output
directories must be new; sources, binary, runtime assets and prepared inputs are
bound before collection. Hardware phases share `profiling/.measurement.lock`.
Do not modify bound source files or run competing tests/builds during collection.

All profile processes finish before a global model-freeze barrier. Heldout uses
new native PIDs and disjoint numerical seeds after that barrier. Profiling
preserves raw observations, per-process medians, variation and provenance.
Model-owned `performance.serial_actions` computes predictions from the frozen
action/gap parameters. Whole-layer observations are not fitted into a correction.

An explicit numerical seed offset separates development and fresh validation
campaigns. Use a new offset after changing the model; heldout seeds also remain
disjoint from the same campaign's profile seeds. Keep the source version and
runtime assets unchanged through all phases.

The combined frozen artifact preserves three deterministic alternatives:
isolated compute/copy prices; context compute/copy prices; and every context
action plus disjoint dispatch gaps. The third remains the primary model with
the original median-error and CV gates. A fourth, predeclared secondary
alternative passes profile-only interval histograms to model-owned
`performance.serial_distributions`. It chooses a profile process with equal
weight, then independently draws each interval within that process. It never
resamples whole measured traces or uses invocation totals as prices.

Expected duration follows linearity of expectation; no interval-independence
assumption is needed for that mean. The secondary median and tail predictions
do assume conditional independence, which may fail under shared device/OS
state changes. Mean error and interval coverage are reported against fresh
bare measurements without replacing the primary acceptance result. Monte Carlo
rank bounds describe numerical integration of the supplied model, not hardware
confidence or measurement uncertainty.

Acceptance checks source stability, independent heldout stability, numerical
correctness, instrumented-versus-bare compatibility and prediction error. Every
failure remains visible; a failed quality gate is not an accurate profile merely
because one pooled error is small. No sample filtering or hidden fallback is
used. These prices remain bound to their exact sequence, execution context,
backend and pool configuration; transfer to new shapes or strategies requires
additional evidence.

Use the [same-process paired controls](../single_layer_context_pair/README.md)
to distinguish context differences from between-process cohorts and to inspect
clock instrumentation. They are diagnostic only and never supply fitted or
heldout-acceptance observations. CI runs the contract tests and native source
guards without compiling GGML or executing hardware.
