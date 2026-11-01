# Scoped steady costs

This campaign measures a finite, declared baseline before calibrating any
resource-contention response. It does not predict future device utilization.
Existing user applications are neither stopped nor assumed absent. Each native
process warms its own inputs, then times fixed-input batches; process startup,
allocation, warmup, verification and output serialization stay outside prices.

Two boundaries are deliberately separate:

- Complete hybrid Dense layers reuse the existing Executor without changing its
  action implementation. PROFILE records inclusive action times in their full
  sequence context. An independent bare-sequence control measures the effect of
  those clocks. New-seed heldout sequences test the sum through modeling's common
  Graph evaluator. A fixed-input replay does not advance token-generation state.
- Standalone F32 GGML components measure 4096-wide GEMMs, grouped attention PV,
  and three explicit copy routes. Procedural inputs have independent NumPy or
  byte/hash references. These are hot-working-set batch averages, not cold-call
  latency or the cache history of a multi-layer streamed workload.

The predeclared set contains 93 processes: 11 cases, three PROFILE processes and
three heldout processes per case, nine full-sequence observer controls and 18
separate diagnostics. GPU stream spans include possible host gaps; enqueue,
synchronization and device spans never get added to an already inclusive price.
Known allocations are checked against an explicit budget, while private CUDA
workspace stays unknown. This is not a capacity qualification.

Run from the repository with its root, `src`, `modeling`, `mapping` and `profiling`
on `PYTHONPATH` and the explicitly selected runtime directories:

```text
python -m profiling.steady_costs.campaign prepare --out NEW_DIRECTORY --binary EXE --runtime-dir DLL_DIRECTORY --seed 202610220
python -m profiling.steady_costs.campaign collect --out NEW_DIRECTORY --split profile
python -m profiling.steady_costs.campaign collect --out NEW_DIRECTORY --split observer
python -m profiling.steady_costs.campaign collect --out NEW_DIRECTORY --split diagnostic
python -m profiling.steady_costs.campaign freeze --out NEW_DIRECTORY
python -m profiling.steady_costs.campaign collect --out NEW_DIRECTORY --split heldout --barrier NEW_DIRECTORY/frozen-barrier.json
python -m profiling.steady_costs.offline_evaluator_v2 --out NEW_DIRECTORY --bind-sources
python -m profiling.steady_costs.offline_evaluator_v2 --out NEW_DIRECTORY
```

`prepare` writes inputs and independent references without launching native
timing. Campaign-owned receipts, programs and reference JSON explicitly use LF
endings; reused `build_case()` input configuration files follow the existing
Windows helper's newline convention. Collection uses one shared
measurement lock, finite owned processes and verified cleanup. Sources, binaries,
programs and references are hashed. A source change invalidates the campaign;
heldout cannot launch without frozen PROFILE prices. Costs are exact arithmetic
means over every retained trial, with no residual fit, offset or interpolation.
Unknown prices remain unknown. Keys include exact shape, action context,
hardware/software identity and repetition boundary.

The canonical offline evaluator is v2. It binds its own new source and focused
tests before evaluating, and verifies the original source inventory and frozen
prices. Windows can recycle numeric PIDs after a process exits: v2 distinguishes
processes by PID plus the recorded launch epoch, allowing a recycled number only
after a verified normal exit with no interval overlap. This is a receipt launch
epoch, not an OS creation-time sample. The source correction and initial rejected
attempt are preserved; prices, thresholds and raw trials stay unchanged. The
historical `campaign evaluate` command retains its original frozen evaluator.

Predeclared gates are mean error at most 10% for each heldout process, P90 error
at most 20% over trial averages, process CV at most 10%, and at most 5% change in
matched full-stamped versus bare sequence means. Failures stay in the report.
Trial averages do not establish individual-invocation tail accuracy. The
standalone components have no per-action observer, and the 5% sequence observer
gate applies to the layer cases only.

Tiny data-only tests, including a fake freeze/heldout pipeline:

```text
python -m unittest discover -s profiling/steady_costs/tests -v
```
