# Completed-work chain calibration

This is a new protocol and source-bound native executable. Historical native
sources, builds, raw measurements and models remain intact. The native probe
reuses the pinned `execution_transforms_dual/compute_context.cpp` and
`matched_mechanisms/compute.cpp` graph/data/reference helpers. A generated include
renames only the context file's terminal `main`; it does not edit either source.

The supported boundary is F32 compact selected-row gate/up/SwiGLU/down FFN,
including materialized CPU/GPU intermediate partitions and CPU output addition;
and post-RoPE QKV causal GQA QK/softmax/PV, sigmoid gate, O projection and residual.
FFN norm/residual and MoE routing/gather/scatter/combine are outside this probe.
All immutable inputs start resident in both memories. Results finish in DRAM.
The preload path reloads three GPU-shard weights every invocation through pinned
staging. It makes no cold-cache or copy/compute-overlap claim. CUDA graphs are
disabled, matmuls explicitly request GGML_PREC_F32, and the runner sets
NVIDIA_TF32_OVERRIDE=0. Fixtures are generated F32 values, not pretrained weights.

## Timing and diagnostics

`wall_ns` retains positive integer nanoseconds for isolated components and
separate uninstrumented completed-work path invocations. Strategies include full
CPU, full GPU, actual monolithic GGML joint graphs, hybrid FFN serial-stage
branches, hybrid FFN joint-graph branches, and CPU QK/softmax to GPU PV/output
attention. CPU and GPU independent shards run on the same explicit main/worker
affinities used by the parallel chain. Native affinity readbacks are required.

`chain_traces` is a separate set of diagnostic invocations, following each path's
uninstrumented samples. Every real graph/copy/preload/merge has steady_clock start
and end stamps. Hybrid FFN also records dispatch and join, CPU/GPU completed-work
brackets, signed branch-start skew, join wait and bracket overlap. Trace
validation requires exact stage/thread inventories, ordering and dependencies.
These are host completed-work intervals, not CUDA kernel timelines. They cannot
identify physical context-switch costs or prove simultaneous GPU kernels.

`timestamp_overhead_ns` separately measures a clock pair, an empty stage pair,
and no-op traces with the same number of stage and branch timestamp calls. The
no-op traces execute on the main thread; they measure timestamp/storage cost,
not the two-thread execution mechanism. Empty worker handshake samples are also
retained independently. No instrumentation/handshake cost is subtracted from any
observation. Instrumented/uninstrumented ratios are diagnostic, not corrections.

## Frozen models and acceptance

`suite.json` declares 23 FFN shape/split and attention token/history/GQA cases,
including 1/8/32 tokens, 128/256/512 hidden width, CPU fractions from 1/8 to 7/8,
and history through 1024 tokens. Two larger q1 points cover FFN H2048/I5632 with
CPU cut1408 and attention H2048/heads16/KVheads4/head_dim128/history128. The chain
protocol explicitly supports H<=2048/I<=8192; its wrapper validates actual wider
geometry before using bounded historical parser placeholders for shared option/
affinity handling, restoring actual geometry before any allocation or execution.
The original parser and all historical graph sources remain unchanged.
`make_suite.py` constructs this grid without
reading measurements. Seeds 1901 and 2027 are disjoint; three repeats yield 138
fresh native processes. The complete plan, order, recipes, sources, executable
and runtime DLL inventory freeze before collection. Grouped component/path
measurement order within each native process is explicit, not randomized.

The independent sum/max component model and a separate profile in-chain stage/gap
fork/join model both freeze before the first heldout process. The latter uses
`accuracy_suite/trace_model.py`, is explicitly calibrated on profile joint trace
intervals, and predicts only the exact profiled strategy/geometry. Complete chain
totals are not its lookup prices or offsets. Join completion depends on both
branches, so wait is not added twice. Heldout traces never enter either fit.
Shared coefficient models and every whole-case leave-one-out coefficient fold
also freeze before heldout in `shape-models.json`. They use the immutable
`composition_calibration/model.py` through source-bound `accuracy_suite/shape_models.py`,
and declare new single-invocation observations with batch_repetitions=1, without
altering raw integers or reusing historical batched coefficients. CPU/joint paths
are excluded from that model's narrower primary GPU/hybrid strategy scope.
Each fold excludes the complete-chain training targets of one declared shape-
and-partition case but retains that case's independent component prices. Other
partitions of the same underlying shape may remain in training. This tests
coefficient transfer across declared cases, not leave-all-partitions-of-one-base-
shape-out validation or prediction without profiling. `shape-report.json` adds these
diagnostics; baseline CV/error gates remain unchanged.
Every heldout raw record binds all three frozen model hashes.

The original gates remain within-process CV <= 0.15, between-process CV of process
medians <= 0.20, and pooled-median heldout relative error <= 0.20, with at least
three independent processes per split and all numerical/coverage checks passing.
`analysis/report.json` uses independent component source CVs.
`trace-report.json` separately reports and gates every profile trace stage/gap/
launch cost CV, including tiny variable gaps, and separate uninstrumented heldout
target CVs. Trace acceptance never silently replaces baseline source gates.
Near-crossing pairs are declared by a frozen profile prediction ratio <=1.25;
heldout outcomes do not select which pairs count. Ranking comparisons use common DRAM output and resident-input
contracts; preload paths are grouped separately. Interval overlap is inconclusive.

All samples, stderr/stdout, nonzero exits, timeouts and failed validation remain
in the new experiment directory. No filtering, outlier removal or automatic
retries occur. The shared exclusive `profiling/.measurement.lock` covers the
entire campaign. A preexisting lock is never deleted or bypassed.

## Build and run

The native directory includes CMake and an equivalent direct MSVC release build
script. The direct route retains exact compiler command, output, linked-library
hashes, source/generated-include hashes and executable hash. It avoids debug PDB
compiler-probe failures in the local Visual Studio installation. Both routes
require explicit existing GGML/CUDA/toolchain paths and reject changed historical
source hashes. Use a new build/output directory for each attempt.

From Simulator, the validated local direct build command is:

```powershell
python profiling/native/chain_calibration/build.py `
  --compiler 'C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe' `
  --ggml-source '../llama.cpp-implement/release-b8705/llama.cpp-b8705/ggml' `
  --ggml-build '../llama.cpp-implement/release-b8705/llama.cpp-b8705/build' `
  --cuda-root 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6' `
  --msvc-root 'C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207' `
  --sdk-root 'C:/Program Files (x86)/Windows Kits/10' --sdk-version 10.0.26100.0 `
  --output profiling/native/chain_calibration/build-new
```

Select explicit allowed logical CPUs. The declared suite uses main CPU 0; use an
observed distinct allowed dispatch CPU, then run the campaign exclusively:

```powershell
$env:GGML_CALIBRATION_GPU_DISPATCH_CPU_ID = '1'
python profiling/chain_calibration/run.py `
  --suite profiling/chain_calibration/suite.json `
  --binary profiling/native/chain_calibration/build-new/chain-calibration-probe.exe `
  --runtime-dir profiling/native/highlevel_accuracy/build-runtime-f32-v2/bin `
  --runtime-dir 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin' `
  --timeout-s 180 --output profiling/chain_calibration/results/campaign-new
```

`--dry-run` prints the complete process plan without a lock or hardware execution.
Do not run any other timings, tests or builds during a real campaign. The default
warmup alone covers dozens of paths per process, so 138 processes take sustained
exclusive runtime. Unknown CUDA private scratch prevents a capacity acceptance
claim.

Offline verification and synthetic protocol tests:

```powershell
python profiling/chain_calibration/verify.py profiling/chain_calibration/results/campaign-new
python -m unittest discover -s profiling/chain_calibration/tests -v
```

`verify.py` replays plan/process coverage, raw SHA256 bindings, numerical/trace
validation, profile-only frozen costs, derived trace predictions/gates, and
diagnostic summaries without running hardware.
