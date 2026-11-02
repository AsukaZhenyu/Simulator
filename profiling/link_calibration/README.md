# Registered H2D size and observed-link calibration

This additive tool separates a fixed completed-copy cost from a byte-dependent
component using measured sizes. It does not infer a rate from a theoretical
PCIe-generation ratio, fit model parameters, or certify whole-LLM accuracy.
Existing source, runtime and measurements are unchanged.

The retained F32 Dense Executor owns the original 2 MiB DRAM/VRAM tensors and
backends. Each size is an exact, contiguous F32 prefix view with the same source
and destination addresses. Aliases use the unchanged `Executor.action` COPY
path (`ggml_backend_tensor_copy`, GPU sync, CPU sync). Views change the byte
extent, without replacing the arena, context, registration, kernels or buffers.

Only registered host memory and zero requested idle are studied here. Every
burst follows an original reset plus one complete untimed Dense prime. The
common warmup remains at least 20 full calls and 500 ms. Before that warmup, a
non-vacuous qualification poisons the entire GPU target, checks it differs from
the selected source, runs the same alias COPY, and checks all 2 MiB: selected
bytes equal the source and bytes outside the slice retain poison. Original
reset/prime then restore the arena before formal measurements. Poison and
readback are not inserted between a formal prime and burst.

Each fresh process has twelve position-balanced size trials. Each size receives
two matched bursts: sixteen bare completed copies and sixteen completed copies
with PCIe queries immediately before/after the burst. Their order is balanced
overall and within each size. The first copy and subsequent fifteen are retained
separately. Every raw duration, including slow samples, remains in the output.
Bare controls have NVML initialized in the process but no per-burst queries;
they are not claimed to be fresh processes without the NVML library.

Calibration uses 128 KiB, 512 KiB and 2 MiB in three fresh processes. Independent
validation uses unseen 256 KiB and 1 MiB in three fresh processes and the retained
independent mathematical-input seed. `heldout` refuses to run without a root-owned
`frozen-barrier.json` of schema `link-copy-cost-freeze-1`, the current manifest
receipt, the `profile_summary` receipt, `predictor_parameters_frozen=true`, a
nonempty `model_scope` string, and hashes in a nonempty relative-path `files`
dictionary. The summary must report 3/3 contexts and passed exact-state support.
Profiling never fits those costs.

The native optional NVML reader binds CUDA0 by its actual PCI bus ID. It loads
only an explicit absolute module or System32 NVML, records the actual loaded
module identity, and reads current/max PCIe generation and width. Each query
retains its QPC bracket, return status and value. Missing fields stay unknown.
The parent binds the actual module SHA. There are no per-copy GPU observations,
CUDA events, CUPTI, link-speed controls, power controls or theoretical fallbacks.

A predictor input may use only the completed pre-burst state queries. Post-burst
state can reject a presumed single-state calibration scope, but cannot define
the input key. Equal boundary readings do not prove within-burst constancy or
measure effective bandwidth. Unknown and transition rows remain visible. An
exact state includes all four current/max generation/width values; states are
never pooled just because generation agrees. Each required state × calibration
size cell needs three distinct process contexts. Both observed Gen3 and Gen4
must have such supported cells before claiming those two states are covered.
Missing cells stop the fixed campaign; there is no infinite retry or extra sample
selection. Root owns modeling, parameter fitting/freezing and final reporting.

Build the single `native/cmake/CMakeLists.txt` target with explicit matching
`LLAMA_ROOT`, `GGML_BUILD_DIR`, `STEADY_GENERATED_DIR` and `CUDA_ROOT` paths. Then:

```text
python -m profiling.link_calibration.campaign prepare --out NEW_DIR --binary LINK_COPY_PROBE --nvml-dll ABSOLUTE_NVML_DLL
python -m profiling.link_calibration.campaign profile --out NEW_DIR
python -m profiling.link_calibration.campaign summarize-profile --out NEW_DIR
# root reviews exact-state support and freezes its model before validation
python -m profiling.link_calibration.campaign heldout --out NEW_DIR
```

No command is an authorization to overlap another hardware experiment. Real
collection requires the global serialized measurement window and lock. The
software tests use only synthetic metadata and never initialize a backend/NVML.

The initial `native/CMakeLists.txt` is retained as an input receipt of the first
manual build; its CUDA import-library search path is incomplete. The active
`native/cmake` entry includes the CPU/CUDA library subdirectories. No compiled
CPP/header/runtime bytes were changed by this packaging correction.
