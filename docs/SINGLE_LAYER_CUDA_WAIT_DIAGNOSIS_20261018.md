# Operation-scoped CUDA wall-time diagnosis

In this **profiler-instrumented H128/q16 F32 configuration**, the dominant
observable interval is **host wall time inside non-synchronization CUDA runtime
APIs**, not the final two stream-synchronization calls. Correlated GPU kernel
and device-copy activity occupies a much smaller interval. This is a bounded
diagnostic result, not a universal GPU-overhead constant or accepted calibration.

The independent [compact JSON evidence](data/single-layer-wait-diagnosis-20261018.json)
summarizes all 808 timed PV invocations from two fresh native processes, using
ABBA/BAAB order. A repeats only the selected PV graph; B executes that graph
inside the complete real action sequence. Every block passed the saved
independent full-port numerical/interval checks. The stdlib analyzer rehashes
those records but does not rerun NumPy or hardware.

## Exact attribution

Four main-thread NVTX parents per process identify `pair_block_0` through
`pair_block_3`. Each parent contains exactly **132** GPU COMPUTE ranges whose
message matches the native target operation ID: one validation invocation,
30 warmup invocations, and 101 timed invocations. Only the last 101 are selected.
All counts, nesting, labels, thread identities and source target bindings pass.
No missing range or unmatched selected launch/copy correlation was ignored.

CUDA APIs must be on that same thread and wholly inside the selected NVTX range.
Their correlation IDs identify kernels and memory copies in the same CUDA
process/context. All activity completes inside the host range. Each of the
808 invocations has **52 correlated kernels, 33 correlated device copies and
two stream-synchronization APIs**. Verification readbacks outside operation
ranges are excluded. Nsight timestamps are compared only with other Nsight
timestamps; no mapping from native QPC is assumed.

The kernel multiset is identical across all invocations:

- One `gemmSN_TN_kernel` and one `k_compute_batched_ptrs`.
- One `cpy_scalar_transpose`, 33 `cpy_scalar` and 16 `concat_f32_dim0` kernels.
- The 33 memcpy activities are device-to-device: one 8,192 B copy and 32 512 B
  copies, identified by the export's memcpy-kind enum.

The timed action therefore contains substantial layout, output materialization
and KV packing work around the matrix multiplication. It is not a bare GEMM.
These copies are internal device-to-device activities, not excluded H2D input
loads. The separate formal single-layer campaign starts with CPU inputs in
DRAM and explicitly executes its declared input transfers.

## Host and GPU intervals

All values below are microseconds; the host column also shows its p90. GPU active
is the interval union of correlated kernels and memcpy, counting overlap once.
API columns are host wall intervals and may include blocking or descheduling.
Columns overlap and must **not** be added as independent costs.

| Order / block | Host PV p50 / p90 | GPU activity union p50 | Non-sync API wall union p50 | First sync p50 | Second sync p50 |
|---|---:|---:|---:|---:|---:|
| ABBA / A0 | 2117.6 / 2403.9 | 148.54 | 1952.7 | 12.30 | 1.25 |
| ABBA / B1 | 2229.6 / 2518.1 | 148.64 | 2005.9 | 14.93 | 1.24 |
| ABBA / B2 | 1874.0 / 2261.2 | 148.61 | 1704.8 | 11.12 | 1.20 |
| ABBA / A3 | 1877.7 / 2137.8 | 148.57 | 1692.4 | 11.08 | 1.18 |
| BAAB / B0 | 1844.0 / 5229.3 | 148.61 | 1675.9 | 11.00 | 1.20 |
| BAAB / A1 | 1778.2 / 5162.6 | 148.54 | 1621.5 | 9.74 | 1.19 |
| BAAB / A2 | 1768.0 / 5442.6 | 148.58 | 1606.9 | 9.67 | 1.18 |
| BAAB / B3 | 1894.6 / 5367.3 | 148.61 | 1713.4 | 11.89 | 1.22 |

Across all 808 invocations:

- Host PV spans range from **1.583 to 6.018 ms**; p50 is 1.898 ms and p90
  4.916 ms.
- Correlated GPU activity union ranges from **148.128 to 149.378 us**; p50 is
  148.580 us and p90 148.807 us. The observed union changes by less than 1% while
  host spans change by almost four times.
- The per-invocation non-sync API wall-time share has p50 **90.90%**, p10 88.71%
  and p90 92.06%. This is an observable wall-time location, not measured CPU
  arithmetic or proof of one driver mechanism.
- The combined two synchronization APIs have p50 share **0.666%**. The first
  synchronization has p50 11.481 us and maximum 729.209 us; the second has p50
  1.203 us and maximum 5.941 us. Rare synchronization delays remain visible,
  but the second explicit wait is not the dominant interval in these traces.

GPU activity span is much larger than GPU activity union because the correlated
operations have gaps between them. For BAAB/A2, for example, their p50 span is
1.727 ms while union is 0.149 ms. Such gaps may reflect command submission,
dependencies, scheduling or work outside the traced target. They are not proof
that the entire device was idle.

The JSON retains signed GPU-start minus host-API-return differences, including
negative values when GPU work starts before its launch/copy API returns. These
are not pure queue delays. Their tails and maxima are preserved; host API time
and GPU time cannot be subtracted to recover an independent submission cost.

## What this changes in the diagnosis

A completed PV time cannot be divided by its GEMM FLOPs and labelled a pure GPU
throughput parameter. Here its observed cost includes many backend calls,
layout kernels, device copies, synchronization and host execution effects.
Similarly, adding one fixed final-sync penalty cannot account for the observed
host wall-time distribution.

The decomposition supports representing backend call/materialization granularity
and its host/GPU overlap explicitly when finer physical parameters are needed.
Profiling must measure the corresponding boundaries, while modeling computes
their composition. This experiment does not fit those parameters or establish
that one API-cost equation transfers to other kernels, shapes or hardware.

The present capture cannot distinguish internal driver blocking from main-thread
descheduling inside runtime APIs, nor assign untraced competing GPU activity.
The first and second synchronization calls sometimes vary, but their ordinary
duration does not identify the underlying cause of the longer API wall spans.
No claim about a universal performance law follows.

Representation also matters. This probe uses independently materialized ports
and block-shaped KV tensors. The broader H64/history128 factory uses **128
single-token history blocks**, unlike a contiguous or production paged KV cache.
No H64 Nsight capture is included here. Call counts and overheads from this
H128/q16 representation must not be extrapolated to those other layouts or
treated as production LLM kernel costs.

## Profiler limitations and preserved failures

The export is from Nsight Systems **2024.4.2**, SQLite schema **3.11.3**. Both
traces explicitly warn that installed CUDA driver **13.0 is unsupported by this
Nsight build**, so CUDA tracing uses libraries for driver **12.6**. The warning
is retained verbatim in the JSON. No dropped-event diagnostic is reported; exact
target counts and selected kernel/copy correlations pass, but this does not
remove the profiler compatibility limitation.

Capture metadata also says `CUDA_SKIP_SOME_API_CALLS=true`. Reported API unions
therefore describe the recorded runtime API set, not proof that every host
CUDA-related function was traced. `host_outside_recorded_runtime_api_ns` retains
the rest of each NVTX interval without labelling it pure CPU work.

NVTX/CUPTI collection and launch injection change execution overhead and may
change performance regimes. These two traced PIDs cannot validate the timing
of uninstrumented runs. The earlier two sandbox-launch timeouts remain in
`results/nsight-v1`; the successful captures and derived SQLite exports are
under `results/nsight-v2`. No timeout was turned into a successful sample.
These diagnostics are excluded from parameter fitting and accuracy gates.

## Reusable analysis and evidence

```powershell
python profiling/single_layer_wait_diagnosis/analyze.py `
  --campaign profiling/single_layer_context_pair/results/nsight-v2 `
  --out NEW-FULL.json --compact-out NEW-COMPACT.json
python -m unittest discover -s profiling/single_layer_wait_diagnosis/tests -v
```

The analyzer uses stdlib/read-only SQLite, rejects an existing output, checks
build/program/profiler/retained-file bytes, and binds exported SQLite/report
hashes. It does not independently repeat the Nsight export. Synthetic interval,
target-count, thread, process/context, correlation, boundary, second-sync,
dropped-event, signed-delay and compact-export tests passed **13/13** without hardware.

The full local output retains every invocation, all queue differences and
kernel/copy multisets. The 88 KB portable JSON retains all eight blocks,
min/p10/p50/p90/p99/max statistics, per-API aggregates, validation guards, all
profiler warnings and source SHA-256s. It binds the complete local decomposition
`profiling/single_layer_context_pair/results/nsight-v2/wait-decomposition.json`
by SHA-256; detailed invocation rows are omitted only from the portable copy.
Raw traces remain local. Resolved loader paths, uninstrumented kernel identity and complete
physical memory accounting are not established.

Final analyzer SHA-256:
`e58d7f2e911398f82eb1ef0f06c947ba71c78e30e5a0693a6864f846f1a0da97`.
Full local JSON SHA-256:
`bcd797b690c1afe6518d5f91d0f02954cd413a5bb08af5675732a17a89cd74f1`.
Portable JSON SHA-256:
`9f6ff82920d75b2888b37c47dc44fa999f233f8628c0f2a93c979b55872e49a0`.
