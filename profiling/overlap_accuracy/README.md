# CE prefetch and current-compute overlap accuracy

This is a new source-bound F32 experiment. It does not alter any historical probe,
model, raw record, source or build. The native executable reuses immutable GGML
FFN/reference helpers through a generated include that renames only the helper's
terminal main. The current computation is compact gate/up/SwiGLU/down FFN on
resident weights; it excludes normalization, residual and MoE routing.

The copy source is initialized page-locked host storage. Each asynchronous H2D
flow writes a distinct independently allocated next-payload buffer. Neither
destination aliases any current FFN input/output port, the other destination,
or a weight used by current compute. The copied payload is next-step data; this
experiment completes its prefetch but does not claim a complete multi-step
decoder implementation or a production prefetch speedup.

Four completed-work paths are fixed before collection:

| Path | Native execution | Frozen isolated-cost hypothesis |
|---|---|---|
| serial/one_copy_compute | A copy completes, then current FFN completes | A + compute |
| overlap/one_copy_compute | A async copy starts, FFN completes, then copy completion is joined | max(A, compute) |
| serial/two_copy_compute | A completes, B completes, FFN completes | A + B + compute |
| overlap/two_copy_compute | Distinct streams enqueue A/B, FFN completes, both copy completions are joined | max(A + B, compute), one shared link |

Each flow is also measured in isolation, as is the completed current FFN graph.
The two-flow report retains an alternative ideal independent-link prediction
max(A, B, compute) to expose disagreement with a shared-link hypothesis. This is
a diagnostic comparison, not a second physical PCIe resource or an accepted
general contention model.

The native calls use real GGML CUDA computation and cudaMemcpyAsync on explicit
nonblocking streams. Uninstrumented totals use only an outer steady_clock timer
and include submission/completion synchronization. Output and full payload
byte checks, current-input checks, allocations and initialization occur outside
those timers. Diagnostic invocations are separate: they retain host submission,
compute and join intervals plus CUDA copy-event elapsed observations. Event reads
are outside each diagnostic total. Clock pairs and empty completed CUDA event
pairs are measured separately and never subtracted. These observations do not
prove concurrent compute kernels, isolate physical context switches, or replace
a CUDA profiler. CUDA private workspace and physical capacity remain unknown.

## Common compiler/runtime binding

Every exact-case path is exported as F32 tensors, an opaque measured compact FFN
operation with its actual four materialized inputs, explicit next-payload COPY
actions and completed VRAM goals. It runs through the repository's common tensor
scenario, strategy IR parser, compiler and runtime. This is timing-only tensor IR
coverage; it is not high-level ModelPlan or transform-lowering coverage.

The runtime has one GPU compute resource and one shared PCIe service resource.
For two flows, it queues the second link service after A while current compute
can remain active. The native submits both streams before computing; this
difference in host submission/queue mechanics is explicit in each exported
binding. Host overhead is tested as residual error, not silently assigned a
fitted offset. Native preallocated destinations versus absent initial logical
VRAM copies are a declared lifecycle difference, not a capacity validation.

Only isolated profile components enter prices. Their exact-case pooled medians
are rounded upward to integer nanoseconds for the runtime. Full tensor workload,
scenario, strategy IR, events and predictions freeze with costs before any
heldout process starts. No heldout target or trace enters fitting.

## Collection and acceptance

`suite.json` contains 18 cases: H256/512/1024, intermediate=2H, q1/q32, and
1/16/64 MiB per copy flow. Profile seed2903 and heldout seed3911 are disjoint.
Three repeats give 108 new native processes and 72 case/path validations.
Default warmup is ten invocations and at least100ms per component/path, with
31 retained samples. Both independent streams and the FFN are bounded far below
the local GPU's VRAM. No shape or process is skipped for runtime convenience.

The shared exclusive `profiling/.measurement.lock` covers the whole campaign.
Every process writes its stdout/stderr, command, flags, seed, failure reason and
timings to a new raw file. Timeouts and launch failures are retained. There is no
outlier removal or automatic retry. Source snapshots and source/binary/runtime/
header/link-library hashes are checked before and after every process. The
predeclared plan and frozen profile model are also bound.

Original gates are unchanged: within-process CV<=0.15 for all used profile
components and heldout totals; between-process CV of process medians<=0.20;
pooled-median heldout relative error<=0.20; at least three independent processes
per split; complete numerical/process coverage; successful common compiler
replay. A low error with failed stability gates is not accepted accuracy.

## Build, untimed validation and run

The direct MSVC release build uses explicit existing GGML/CUDA/toolchain paths.
It records its command, source/generated include hashes, exact linked libraries,
headers, compiler output and executable hash. Each new attempt needs a new
directory. CMake is supplied as an alternative build description; the collector
requires the direct builder's build-record.json for its checked provenance.

From Simulator:

```powershell
python profiling/native/overlap_accuracy/build.py `
  --compiler 'C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe' `
  --ggml-source '../llama.cpp-implement/release-b8705/llama.cpp-b8705/ggml' `
  --ggml-build '../llama.cpp-implement/release-b8705/llama.cpp-b8705/build' `
  --cuda-root 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6' `
  --msvc-root 'C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207' `
  --sdk-root 'C:/Program Files (x86)/Windows Kits/10' --sdk-version 10.0.26100.0 `
  --output profiling/native/overlap_accuracy/build-new
```

The native interface requires all documented legacy FFN shape/context/sampling
options, plus `--payload-bytes` and `--mode measure|validate`. Validate mode runs
each path once, checks data and storage, and emits no timing samples. It is not
heldout accuracy evidence. The pre-formal H256/q1/1MiB validation used seed29;
its raw result and an earlier missing-DLL launch failure remain in separate debug
directories. Correct GGML runtime DLLs are under build/bin/Release, not the
import-library directories under build/ggml/src.

Once granted an exclusive hardware slot, run the complete suite:

```powershell
python profiling/overlap_accuracy/run.py `
  --suite profiling/overlap_accuracy/suite.json `
  --binary profiling/native/overlap_accuracy/build-direct-v2/overlap-accuracy-probe.exe `
  --runtime-dir profiling/native/highlevel_accuracy/build-runtime-f32-v2/bin `
  --runtime-dir 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin' `
  --timeout-s 180 --output profiling/overlap_accuracy/results/formal-v1
```

`--dry-run` prints the full plan without
hardware execution. The runner forces TF32 override0 and CUDA graphs disabled;
the native graph explicitly requests GGML_PREC_F32. Do not run other timings,
builds or heavy tests during a real campaign. Allow sustained exclusive runtime
for all108 processes (conservative estimate12–20minutes on this machine).

When a supplied runtime directory contains strict-f32-runtime.json, the collector
automatically validates and inventories every original/input/output/evidence
SHA256 plus the manifest itself. Every raw process binds that manifest hash;
offline verification revalidates its bytes. The private backend compiles one
regular-MUL_MAT router translation unit and reuses the other byte-bound objects.
It does not claim a clean GGML rebuild, changed MUL_MAT_ID arithmetic, or observed
loader resolution. Full backend provenance hashing adds wall time outside all
native timing intervals; the conservative native campaign estimate excludes it.

Offline replay and synthetic tests:

```powershell
python profiling/overlap_accuracy/verify.py profiling/overlap_accuracy/results/formal-v1
python -m unittest discover -s profiling/overlap_accuracy/tests -v
```

New JSON exporters use explicit LF newlines. Synthetic tests certify protocol,
source separation and common runtime scheduling only; their durations are never
presented as measured hardware evidence.
