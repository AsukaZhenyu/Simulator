# Same-process context pairing diagnostic

```
context-pair-probe PROGRAM.json context_vs_isolated|instrumentation TARGET_ACTION_INDEX WARMUP SAMPLES default|0 ABBA|BAAB OUTPUT.json
```

This diagnostic reuses one Executor, program, persistent CPU pool and process for
four predeclared blocks. In `context_vs_isolated`, A repeats only the selected GPU
`attention_pv` action in isolation. In `instrumentation`, A executes the full real
action sequence with only invocation clocks (`bare-replay`). In both pairs, B
executes the full sequence with the existing low-overhead POD context trace.
Explicit `ABBA` and `BAAB` orders support counterbalancing across
independent PIDs; the caller chooses the order before seeing data. A useful fixed
protocol is three independent PIDs, alternating orders, 30 warmup invocations and
101 retained samples per block. The program/target, pool configuration, sample
count and numerical data remain the same across all four blocks within each PID.

Each block first validates the full sequence, then repeats its own warmup. Input
restore is outside invocation clocks in B, exactly as in the original context
probe. Every final port is captured outside timing for independent NumPy checking.
`Executor.validate` convenience records are cleared between blocks so each raw
port/copy report describes exactly one validation replay, without changing calls
inside any measured execution. Windows CPU diagnostics and monotonic sampling
envelopes are retained in each block. JSON conversion and port readback occur
between blocks and can influence subsequent device state; warmup repeats after
each boundary but does not prove exact GPU/OS state restoration.

The root schema `single-layer-context-pair-native-v2` has `process_id`, `seed`,
`program_sha256`, `target_action_index`, `block_order` and four `blocks`. Each block
has `block_index`, `label`, begin/end ticks, wall time and `raw`. B's `raw` is the
unchanged single-layer context schema. Isolated A's `raw` uses that schema with
`component_subset=true` and exactly one component; bare A has empty per-action
intervals and retains all invocation durations. Root `pair_kind`, `A_label` and
`B_label` declare the comparison explicitly. Check
`raw['verification_native']` using the existing independent full-port verifier;
derive B's target samples from its per-invocation `action_intervals`.

These results are **diagnostic only**: `usable_for_parameter_fit` and
`usable_for_heldout_acceptance` are always false. An isolated subset must never be
treated as a complete cost table. Same-process counterbalancing tests whether a
context effect persists without changing PID; it does not by itself identify
clocks, scheduling, cache, polling, kernel identity or another mechanism. The
comparison still retains context instrumentation and uncontrolled device drift.

Build with a new CMake directory and explicit `LLAMA_ROOT`, `GGML_BUILD_DIR`,
`RUNTIME_DIR`, `Python3_EXECUTABLE`. This directory copies two byte-pinned existing
sources with only their `main` symbols renamed. It never edits the original
single-layer or historical executors. Precompile and post-link manifests bind the
supplied source/header/library/runtime bytes. Actual loader resolution, transitive
system SDK inputs and historical runtime build flags remain unproven.
