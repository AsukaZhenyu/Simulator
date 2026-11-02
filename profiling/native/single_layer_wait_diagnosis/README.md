# Operation-scoped CUDA wait diagnosis

This separate native executable adds header-only NVTX annotations to byte-pinned
copies of the existing executor and paired diagnostic. It never edits historical
sources or supplied GGML runtime DLLs. The actual GGML compute, copy, reset,
validation and pool behavior remain unchanged. Its profiler overhead is retained;
all results are diagnostic only, never parameter estimates or accuracy acceptance.

```
wait-diagnosis-probe PROGRAM.json context_vs_isolated|instrumentation TARGET WARMUP SAMPLES default|0 ABBA|BAAB OUTPUT.json
```

The JSON schema and full-port numerical evidence are the existing paired v2
contract plus `nvtx_diagnostic`. Check every block's
`raw.verification_native` with the independent NumPy oracle. The operation label
borrows the original action's `target_id` string; it does not build dynamic
labels or substitute operators. Categories are block=1, CPU COMPUTE=10, GPU
COMPUTE=11, COPY=20, ADVANCE=30 and EVICT=31. COPY messages are tensor IDs.

Each block is enclosed by one main-thread NVTX parent named `pair_block_0` through
`pair_block_3`. The raw `block_order` and A/B labels identify its mode. For the
selected GPU PV operation, every block must contain exactly
`1 + WARMUP + SAMPLES` matching GPU COMPUTE ranges: one validation invocation,
then warmup, then samples. **Only the final SAMPLES ranges are timed.** Reject
missing, extra or dropped events; do not silently infer a selection from a
process-wide aggregate. Other operations in isolated A have validation ranges
only, and port readback can issue CUDA copies outside action ranges.

Capture CUDA and NVTX together with Nsight Systems; disable CPU statistical
sampling when it is not needed. For example, with an explicitly chosen local
`nsys` executable and a new output prefix:

```
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --stats=false --force-overwrite=false --output=NEW_PREFIX wait-diagnosis-probe PROGRAM.json context_vs_isolated 14 30 101 default ABBA NEW_OUTPUT.json
```

Set the same explicit GGML/CUDA runtime search paths as the source campaign.
Keep the command, actual executable/runtime hashes, profiler version and outputs.
Use the NVTX host range and thread to associate CUDA APIs, then API correlation
IDs to associate GPU kernels/memcpy. Inclusive API synchronization waits and GPU
work overlap; their durations must not be summed as disjoint costs. A kernel
interval or a host wait alone does not prove which operating-system or device
mechanism caused a delay. Profiling itself can change the observed regimes.

Configure a fresh CMake build with explicit `LLAMA_ROOT`, `GGML_BUILD_DIR`,
`RUNTIME_DIR`, `NVTX_INCLUDE_DIR` (the installed include root containing
`nvtx3/nvToolsExt.h`) and `Python3_EXECUTABLE`. No downloads or NVTX import library
are required. The build record binds all three input executors, the exact allowed
generated changes, provided GGML/NVTX headers, libraries, runtime assets, compiler
and CMake cache; the original files must still match after linking. Actual loader
resolution and transitive SDK provenance remain unproven.

```
python -m unittest discover -s profiling/native/single_layer_wait_diagnosis -p test_build.py -v
```

These tests verify source transformations and metadata without running hardware.
