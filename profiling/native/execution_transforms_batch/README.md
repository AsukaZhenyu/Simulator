# Batched execution-transform probe

This is a new timing variant. It does not edit or overwrite the historical
single-invocation probes, binaries, data or acceptance decisions. The source
generator retains their mathematical graphs, input fixtures, numerical oracle,
tolerances, CPU/GPU dispatch contexts, explicit F32 precision and TF32 metadata.
No 27 KiB C++ source duplicate is checked in.

## Build

The generator reads `../execution_transforms_dual/compute_context.cpp` and
`../matched_mechanisms/compute.cpp`. It requires their audited SHA256 identities
(normalized LF) and the exact expected anchors/counts. Source drift fails
explicitly and requires a new audit. Generated C++ uses an absolute include for
the unchanged helper so it can live entirely in a build directory.

```powershell
cmake -S profiling/native/execution_transforms_batch -B <new-build> `
  -DGGML_SOURCE_DIR=<existing-ggml-source-root> `
  -DGGML_BUILD_DIR=<existing-ggml-build-root>
cmake --build <new-build> --config Release --target execution-transform-batch-probe
```

The dependencies are explicit existing GGML base/CPU/CUDA libraries, CUDA
Toolkit, C++17 threads and Python 3.10 or newer for source generation. The build
does not download or rebuild GGML. `source-manifest.json` records generator,
base-source, helper and generated-source hashes. After a successful target
build, a CMake `POST_BUILD` hook runs `record_build.py`. It copies the generated
`compute_batch.cpp` and `source-manifest.json` byte-identically beside the actual
target executable and writes `compute-batch-build.json` there. Same-file copies
are skipped. This also works with Visual Studio's `Release/` or `Debug/` output
subdirectories, so the measurement runner receives the same adjacent artifact
layout for single-config and multi-config generators.

The record includes the binary hash, full source manifest, CMake reproduction
command/configuration/generator, toolchain identity, CMake source and metadata
helper hashes. The reproduction command is not a captured compiler/linker
invocation. The helper never compiles or runs a probe: its default manual mode
records identities only and explicitly sets `compile_verification_performed`
to false. The CMake hook declares its invocation context after a successful
target build; that declaration is not compiler tracing or cryptographic proof.
Building does not run measurements.

The separately compiled workstation binary under `build-local/` retains its
original `compute-batch-build.json` with the actual adapted historical `cl.exe`
command. The metadata fix does not replace that record or modify that binary.
An initial campaign verified before the CMake metadata fix retains the earlier
CMake source snapshot; it must not be relabeled as using this later build hook.

To generate source or run generator tests without a compiler/GPU:

```powershell
python profiling/native/execution_transforms_batch/generate_probe.py `
  --output profiling/native/execution_transforms_batch/build-local/compute_batch.cpp `
  --manifest profiling/native/execution_transforms_batch/build-local/source-manifest.json
python -m unittest discover -s profiling/native/execution_transforms_batch -p test_generate_probe.py -v
python -m unittest discover -s profiling/native/execution_transforms_batch -p test_record_build.py -v
```

The metadata tests use a fake executable and exercise byte-preserving copies,
same-file handling, source-drift rejection and single-/multi-config layouts.
They are CPU-only tests, not evidence that CMake compiled a native executable.

## Explicit runtime contract

Every historical CLI dimension remains required and unchanged, including
`--cpu-intermediate`. The runner must additionally supply:

```text
GGML_CALIBRATION_BATCH_REPETITIONS=<integer from 1 through 1024>
GGML_CALIBRATION_GPU_DISPATCH_CPU_ID=<explicit logical CPU different from --cpu-id>
NVIDIA_TF32_OVERRIDE=0
```

There is no implicit repetition count. Missing, malformed or out-of-range
batch counts fail before model allocation. The original Windows two-logical-CPU
affinity checks remain mandatory. The existing explicit `GGML_PREC_F32` and
externally supplied TF32 setting retain their original semantics.

`measure_batched` calls the unchanged historical `measure` helper with a
function that completes N invocations sequentially. Each invocation still
contains its original completion synchronizations, dispatch/wait, pinned
staging and delivery. Warmup count and minimum warmup duration apply to these
whole batches; allocation and numerical checks remain outside timing. Repeated
Attention invocations reuse the same immutable cache/input fixture; they are
not advancing N real decode tokens. Repeated preload invocations copy all
three original weight payloads each time and do not claim a cold cache.

The raw output has a distinct schema:

```json
{
  "schema": "execution-transform-native-batch-0.1",
  "batch_repetitions": 16,
  "timing_unit": "completed_work_batch_total",
  "batch_wall_ns": {"metric_name": [123456, 123987, 123700]}
}
```

The array values are **total nanoseconds for N completed invocations**. C++
never divides them and does not emit a normalized `wall_ns`. A downstream
analysis may explicitly derive an average per invocation and must retain these
raw totals, N and the changed sample unit. A steadier batch mean cannot prove
that isolated-invocation latency jitter disappeared or that a previous failed
quality gate should now pass.

All historical metrics remain. FFN additionally measures each weight H2D
independently with the same pinned-staging/completion boundary:

- `hybrid/preload_gate_weight_h2d`
- `hybrid/preload_up_weight_h2d`
- `hybrid/preload_down_weight_h2d`

These are independently timed batches, not fractions of the existing aggregate
`hybrid/preload_weights` metric. Their sum need not equal an independently
measured aggregate. The untimed host-overlap diagnostic remains a single
execution and is explicitly labeled as such; it does not establish CUDA kernel
overlap. GPU private scratch remains unknown.

This variant can diagnose steady repeated execution. It cannot by itself
validate full-model inference timing, numerical equivalence of arbitrary
compiler transforms, physical memory capacity, operating-system scheduling,
or actual language-generation behavior. The original calibration failures and
their declared gates remain part of the evidence.
