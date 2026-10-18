# High-level strategy accuracy bridge

This adapter runs an actual public `ModelPlan.compile()` / `CompiledPlan.run()`
strategy through the original simulator and a new GGML executor. It exports the
complete compiled frontend IR, workload graph, operation operands, materialized
ports, exact accepted action sequence, physical descriptor and real tensor data.
No compact FFN or hand-authored tensor IR is relabelled as high-level coverage.

Supported execution is bounded explicitly: one complete F32 Dense invocation,
CPU/CUDA macro placements, optional public `AttentionStages` and
`FFNPartition` transforms, normal adjacent-pair RoPE, optional Q/K RMSNorm and
sigmoid attention gating, causal GQA, both hidden RMSNorms and both residuals.
The partitioned FFN performs one norm, exact immutable matrix slices, complete
SwiGLU partials, one ordered sum and one residual. KV blocks are physical
position-major K/V tensors with independent copy allocations. Every output port,
including scores, probabilities, shard partials and appended KV, is compared
against a separate NumPy numerical implementation.

`SerialExecutionPolicy` is an explicit high-level policy. The bridge rejects
overlapping launches, multiple invocations, other transforms, services,
disturbances, SSD and unsupported dtypes. Export does not change or silently
serialize a strategy. The native executor checks graph operands and exact
independent port storage, rejects unavailable operators, checks copy contents,
and enforces copy/eviction liveness and final output delivery.

The native executor reserves independent backing allocations before timing.
Eviction changes logical validity. It does **not** establish native allocation
or memory-capacity accuracy; the artifacts state this limitation explicitly.
The executor has a persistent CPU threadpool, real CUDA backend execution with
graphs disabled, `NVIDIA_TF32_OVERRIDE=0`, explicit `GGML_PREC_F32` matmul requests,
and no CPU fallback for requested CUDA operations. Every native result reports
the environment values and requested GGML precision parameter. Those values
alone do not establish the internal CUDA arithmetic precision. Final timed output ports are read outside the timer and
independently checked. CUDA private backend scratch is unknown; reported
workspace covers gallocr capacity and CPU cplan scratch only.

## Build

From `Simulator`, using the existing GGML source and shared library build:

```powershell
cmake -S profiling/native/highlevel_accuracy -B profiling/native/highlevel_accuracy/build `
  -DLLAMA_ROOT="C:/path/to/llama.cpp-b8705" `
  -DGGML_BUILD_DIR="C:/path/to/llama.cpp-b8705/build" `
  -DCMAKE_TRY_COMPILE_CONFIGURATION=Release `
  -DCMAKE_EXE_LINKER_FLAGS_RELEASE="/INCREMENTAL:NO" `
  -DCMAKE_CXX_FLAGS_RELEASE="/O2 /MD /DNDEBUG"
cmake --build profiling/native/highlevel_accuracy/build --config Release --parallel 2
```

The Release try-compile and flags avoid the installed Windows toolchain's Debug
PDB DLL mismatch. Historical probes, sources, artifacts and builds are untouched.

### Private strict-F32 CUDA runtime

The pinned GGML b8705 CUDA router selects custom MMF for suitable F32 matrices
with up to 16 output columns. Its explicit TF32 PTX ignores the ordinary
`GGML_PREC_F32` request and `NVIDIA_TF32_OVERRIDE`. The original representative-v2
collection is retained as a failed diagnostic: all four H128/q16 plans failed
the independent numerical checks. The other ten ledgers failed source stability
gates, and no heldout or global profile barrier was produced. Gates and numerical
tolerances were kept unchanged.

Build a NEW private runtime before collecting a replacement campaign:

```powershell
python profiling/native/highlevel_accuracy/build_f32_runtime.py `
  --ggml-source "C:/path/to/llama.cpp-b8705/ggml" `
  --ggml-build "C:/path/to/llama.cpp-b8705/build" `
  --output profiling/native/highlevel_accuracy/build-runtime-f32-v2 `
  --cuda-root "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6" `
  --vs-dev-cmd "C:/Program Files/Microsoft Visual Studio/2022/Community/Common7/Tools/VsDevCmd.bat"
```

This bounded Windows MSVC/CUDA Release builder pins the original router bytes,
copies the GGML source and supplied objects into a new directory, applies one
guard so regular `MUL_MAT` requests with `GGML_PREC_F32` bypass MMF, compiles that
one translation unit and relinks the copied objects. It preserves the originals
and refuses an existing output. `MUL_MAT_ID` is outside the patch scope. This is
an explicitly recorded partial rebuild, not a clean build of the supplied
objects. `bin/strict-f32-runtime.json` records the source, compiler, commands,
patch, all reused objects, import libraries, output DLLs and hashes. The campaign
runner verifies those current bytes before every process and at phase end.
Resolved DLL loader paths are not observed; the supplied runtime search
directories and all their shared-library bytes are recorded.

## Profile and heldout protocol

The runner defaults to three independent native processes and 31 samples per
process. `prepare` is software only. It creates new complete physical model
sources, exact compilation placeholders, genuine high-level plans and frozen
program JSON. Original placeholder simulation durations remain explicitly
labelled as unmeasured and never become accuracy predictions.

```powershell
python profiling/highlevel_accuracy/run.py prepare `
  --out profiling/highlevel_accuracy/runs/representative-v3 --suite representative
python profiling/highlevel_accuracy/run.py profile-all `
  --out profiling/highlevel_accuracy/runs/representative-v3 `
  --executable profiling/native/highlevel_accuracy/build/Release/highlevel-probe.exe `
  --runtime-dir profiling/native/highlevel_accuracy/build-runtime-f32-v2/bin `
  --runtime-dir "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin"
python profiling/highlevel_accuracy/run.py heldout-all `
  --out profiling/highlevel_accuracy/runs/representative-v3 `
  --executable profiling/native/highlevel_accuracy/build/Release/highlevel-probe.exe `
  --runtime-dir profiling/native/highlevel_accuracy/build-runtime-f32-v2/bin `
  --runtime-dir "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin"
```

Both hardware phases use the shared `profiling/.measurement.lock`. The runner
never overwrites prior evidence. A failed run leaves its raw output for review;
start a new campaign directory after fixing its cause.

The representative suite contains 14 plans: H64/I128 decode q1/history128 on
CPU, CUDA and hybrid macros plus intermediate cuts 60/64/68; H128/I256 q16 and
H256/I512 q32 prefill on hybrid macros plus three close cuts per shape.
Staged cases use CPU QK/softmax and CUDA PV/output and mixed CPU/CUDA FFN shards.
`--suite smoke` prepares three smaller full CPU, full CUDA and mixed staged plans.

`profile-all` runs matching native graphs and exact tensor-copy components,
aggregates fresh-process samples, records raw hashes, source/binary/library
hashes and per-component CVs, and freezes every ledger before writing the global
profile barrier. Profile inputs come from a complete untimed execution of the
same graph. It then rebinds the same compiled high-level model's leaf costs to
the measured ledger and reruns its actual policy. Every selected price must be
measured; other device/transfer choices are removed. The complete action trace
must stay identical, and a separate engine replay must agree with the high-level
model's makespan. No old timing is divided or interpolated.

`heldout-all` validates every ledger and the global barrier before launching any
heldout. Fresh processes use numerical seeds absent from profiling. Everything
affecting the graph, shapes, operands, physical ports and actions stays identical.
The heldout wall time includes whole action dispatch, completed graphs and
copies. Its samples never enter the frozen component ledger. Reports give the
signed and absolute relative error per process, independent numerical and copy
checks, and exact identity evidence. Required gates are fixed: each source
component and heldout process must have CV <= 0.15; source component process
medians and heldout process medians must have CV <= 0.20; pooled heldout median
prediction error must be <= 0.20. Every source process retains its separate
samples and statistics. An optional `--max-relative-error` supplies an additional
per-process check. Source, binary and supplied runtime-library inventories are
checked before each process and after each phase. A successful-link build record
binds native sources and linked libraries to the executable. Failures and timeouts
preserve stdout/stderr and aggregate failed/unknown coverage while remaining
cases continue. Source or binary changes after freeze reject the comparison.

## Reusable API and tests

```python
from highlevel_accuracy.bridge import export_plan, SerialExecutionPolicy
from highlevel_accuracy.profiles import freeze_profile, run_plan_with_profile, compare_heldout

# Author transforms, placements and explicit serial policy through ModelPlan.
plan.runtime_policy(SerialExecutionPolicy)
compiled = plan.compile()
program = export_plan(compiled, seed=1234)
```

For an exported program the native interface is:

```text
highlevel-probe PROGRAM.json validate|components|replay WARMUP SAMPLES OUTPUT.json
```

`validate` executes once and emits numerical/copy evidence without sampling any
durations. Python tests check changed operands/shapes/boundaries/actions,
unsupported transforms, complete numerical ports, exact macro/partition
equivalence, source-bound profile reuse, measured high-level regeneration and
seed/chronology separation. Synthetic duration fixtures in these tests certify
bookkeeping only.

```powershell
$env:PYTHONPATH="modeling;mapping;profiling"
python -m unittest discover -s profiling/tests -p test_highlevel_accuracy.py -v
python -m unittest profiling.tests.test_highlevel_f32_runtime -v
```

Set `HIGHLEVEL_PROBE` to the built executable and `HIGHLEVEL_RUNTIME_DIR` to
the GGML DLL directory to add untimed real-CPU full/staged numerical tests.
