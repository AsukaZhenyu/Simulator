# H2D context and execution-history validation

This study separates measurement from prediction. Native collection, optional
CUPTI/NVML observations, parameter estimation, and acceptance accounting live in
`profiling/h2d_history`. Logical history evolution and response calculation live
in `llm_infer_model.performance.execution_history` and
`llm_infer_model.performance.history_conditioned_costs`.

The optional response provider is an empirical, interpretable hypothesis. It
does not simulate a hardware cache, an operating-system scheduler, a CUDA driver
queue, or a power governor. It changes no existing default provider.

## Measurement boundary

The native probe executes the existing F32 GGML Dense action stream with one CPU
thread. CPU/GPU COPY completes before the next action. The inclusive GGML H2D
call already includes an asynchronous CUDA copy followed by synchronization on
the per-thread CUDA stream. The subsequent backend fence is recorded separately.
Runtime API spans and device memcpy activity overlap and must not be added.

Every block resets inputs, performs one complete untimed prime, and times eight
fixed-input replays. Bare, stamped and CUPTI processes use sixteen blocks and the
same preparation, avoiding the previous difference in observer replay history.
They are separate fresh processes: their mean difference includes between-process
variation and is not a pure measurement of instrumentation overhead.

The registration control registers and restores only the owned DRAM arena.
Registration is outside steady timers and is not free initialization. It never
silently substitutes for the pageable target. Source-read and idle interventions
are diagnostic; intervened block totals are not strategy-accuracy labels.

## Causal response provider

For each declared action, modeling computes a semantic coordinate and a logical
history snapshot before predicting its duration. It then advances history using
that predicted duration and the declared accesses. H2D conditioning uses only the
power-of-two bin of distinct other logical host-buffer bytes accessed since the
source's previous access. Buffer handles identify reuse internally but do not
appear in fitted coordinates. Geometry, layout and kernel semantics remain exact.

This logical footprint is not physical cache occupancy. Block initialization
starts a new logical history and explicitly replays the prime; it does not claim
that physical caches, clocks or driver state were reset. Reset writes and
cross-block device recovery are not identified by this provider. Unknown
coordinates reject instead of using guessed prices or interpolation.

Calibration uses per-action measurements from eighteen profile processes only.
Diagnostics and complete-block times do not fit response prices. A static table
using the same calibration labels is the state ablation. The older isolated-hot
table is a historical reference with a different measurement context.

## Reproduction

Build `native/CMakeLists.txt` with explicit GGML source/import-library/generated
Executor paths and CUDA/CUPTI toolkit paths. Preserve the runtime DLL receipts.
Then run from the repository root, with `modeling`, `mapping`, `src`, and the root
on `PYTHONPATH`:

```text
python -m profiling.h2d_history.campaign prepare --out NEW --binary PROBE
python -m profiling.h2d_history.campaign diagnostic --out NEW
python -m profiling.h2d_history.analysis --out NEW
python -m profiling.h2d_history.campaign profile --out NEW
python -m profiling.h2d_history.fit --out NEW
python -m profiling.h2d_history.campaign heldout --out NEW
python -m profiling.h2d_history.qualification --out NEW
python -m profiling.h2d_history.independent_audit --out NEW
```

The freeze rejects any prior heldout acquisition. All three providers and six
Dense candidates are fixed before the forty-eight heldout processes start.
Collection is serialized through the existing measurement lock, never retries a
failed process, and preserves failed jobs in acceptance denominators. Artifacts
are exclusive-create UTF-8/LF files with source/binary/runtime SHA256 receipts.

The unchanged primary gates are: every bare-process mean APE at most 10%, pooled
block-average APE P90 at most 20%, every bare-process trial CV at most 10%, and
every matched observer mean discrepancy at most 5%. Observer CV is an additional
reported check. A failed scientific gate is a valid result, not an audit error.

Optional `clock_diagnostic` and `idle_diagnostic` commands create separate
diagnostic cohorts. NVML observations include the target's own activity and are
never future inputs to its prediction. Actual query cadence and unsupported
fields remain explicit. An observed clock/link change or an idle intervention
does not, by itself, identify a unique power-management or scheduling cause.

These tests qualify only their stated warm, finite-shape, serial Dense scope.
They do not establish arbitrary dynamic-load accuracy, cold-copy jitter,
unmeasured memory modes, UMA, or full production LLM inference.
