# GPU consumer observation v5

This private runtime adds observation to the original CUDA `MUL_MAT_ID`
consumer. It retains the v4b CPU observer DLL, the original CUDA mathematics,
backend selection, fusion conditions, graph configuration and other objects.
The public GGML source and all earlier artifacts remain unchanged.

The CPU and GPU hooks read the expert IDs supplied to actual up/gate/down
consumers. Requested placement and the buffer holding expert weights do not
prove which backend executed a consumer. Qualification uses the observed
consumer backend together with the actual graph/partition inventory; weight
residency is reported separately. A CPU-resident weight is not, by itself,
evidence of CPU computation.

The CUDA consumer entry reserves a unique, bounded pinned-host slot, then
enqueues `cudaMemcpy2DAsync` on its actual consumer stream before the original
dispatch. Supported IDs are two-dimensional I32 tensors with contiguous slots
(`nb[0] == 4`) and the actual source row pitch (`nb[1] >= top_k * 4`).
`cudaMemcpy2DAsync` copies those rows into a packed destination; a noncontiguous
slot dimension is explicitly rejected, not treated as packed. Existing
`llama_synchronize` calls complete these copies before parsing; there is no new
layer synchronization, event, callback split or graph-output flag. Captured
stream identifiers, strides, destination offsets and byte counts are exported.
They are observation commands and consume resources. **Every v5 timing is
observer-inclusive diagnostic data, never bare latency or a cost-table value.**

## CUDA graph boundary

The environment remains `runtime_default_enabled`. The private backend records
the actual per-partition `use_cuda_graph` and `cuda_graph_update_required`
decisions immediately before the original execution. A graph decision event is
host submission metadata, not device completion. A default-enabled flag alone
does not prove that a particular partition used capture/replay.

A normal CUDA graph replay bypasses host tensor dispatch. Therefore a GPU expert
partition that actually uses CUDA graph capture/replay is explicitly unsupported
for this observer (error 10). Previous IDs are never reused as current routes.
Graph-enabled partitions without GPU experts retain their normal behavior.
The observer never disables graph capture or forces fallback/cuBLAS. Unobserved
fusion paths remain missing in the strict stage inventory; they are not filled
from a neighboring stage or previous forward.

## Acceptance boundaries

Software tests and the native self-test exercise source transformation, POD
reservation/grouping and rejection contracts. They do not execute GPU DMA,
validate the real model, establish numerical equivalence, or certify 16-layer
coverage. The native self-test labels this explicitly.

Real-model qualification requires independent original/private normal,
teacher-forced replay and capture jobs at the same placement and configuration.
Compare every greedy token and complete-logit hash, actual graph/partition
fingerprints, and all nonempty forward/layer up/gate/down route records. Forced
tokens alone are not evidence of equality. Report prefill and decode separately,
all failures and denominators, actual graph decisions, observer capacity and
unchanged safety guards. Whole-model performance accuracy remains unassessed.

A zero capture error does not establish route coverage. The strict inventory
must contain every expected nonempty forward/layer and all three expert stages,
with consistent IDs and shapes. Missing/fused paths, unsupported graph replay,
capacity failures and asynchronous copy errors remain explicit failures or
unsupported outcomes. They must not become empty successful traces, inferred
routes, or reused IDs from an earlier forward.

Each job binds its own request/result schema, native executable, DLL inventory
and prepared-source receipts. In particular, the v5 executable requires the
private GPU observer ABI even outside capture mode: it cannot be paired with
the original CUDA DLL as an original-normal control. The original-normal job
uses its matching v4b executable/schema and original runtime; private v5 jobs
use their matching v5 executable/schema and private runtime. Comparisons must
retain these differences in their receipts rather than silently mixing files.

The v5 first build's native const-qualification error is retained. The corrected
v5b build uses GGML's non-const public graph getter only through a documented
read-only cast. Neither build has permission to run inference merely by existing.

## Build interface

From the repository root, using actual paths in place of the placeholders:

```text
python -m profiling.real_moe_validation.tap_v5 prepare --out NEW_DIRECTORY --llama-root PINNED_LLAMA_ROOT --v4-build FROZEN_V4B_BUILD
python -m profiling.real_moe_validation.tap_v5 build --out NEW_DIRECTORY --nvcc NVCC_PATH --cl CL_PATH --link LINK_PATH
```

Run the build in an initialized x64 Visual Studio environment. It compiles one
CUDA translation unit, reuses the exact original linked object inventory, and
compiles the new native driver. CUDA Release flags are read and checked from the
original project. The pinned original llama/GGML source, matching built Release
objects, project/link records, CUDA toolchain and completed v4b private build
are required. This is a source-bound diagnostic build, not a general portable
llama.cpp build command; incompatible anchors, objects or flags are rejected.

All output directories are new and output files are created exclusively. The
native executable accepts schema `real-moe-tap-request-5` and produces
`real-moe-tap-native-5`; old request/result schemas and artifacts are preserved.
Preparing or building does not run inference or authorize hardware collection.
These optional profiling tools are separate from the simulator core: using the
core library does not require CUDA, these private binaries, or generated profile
artifacts.
