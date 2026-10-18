# Same-process paired diagnostics

This source-only runner counterbalances four blocks in each fresh native PID.
It reuses only the three **profile** programs per case from a source-bound
single-layer campaign. Existing heldout programs or measurements are not read.

The two controls are `context_vs_isolated` (A: repeated GPU attention PV;
B: the same target in the complete action sequence) and `instrumentation`
(A: whole replay with invocation clocks; B: whole replay with action clocks).
ABBA/BAAB order is fixed before measurements. Two cases, three repetitions, two
poll settings and two controls declare 24 independent processes.

```powershell
python profiling/single_layer_context_pair/runner.py prepare `
  --campaign profiling/single_layer_calibration/results/diagnostic-v2 `
  --executable profiling/native/single_layer_context_pair/build-release-v2/Release/context-pair-probe.exe `
  --out profiling/single_layer_context_pair/results/diagnostic-v1
python profiling/single_layer_context_pair/runner.py run `
  --out profiling/single_layer_context_pair/results/diagnostic-v1
python profiling/single_layer_context_pair/runner.py summary `
  --out profiling/single_layer_context_pair/results/diagnostic-v1
```

Prepare never starts hardware. Run acquires the common measurement lock, retains
every failure, records the actual Popen PID and inherited affinity, and samples
device observations every 50 ms. Every block is checked through the independent
NumPy full-port oracle. Raw monotonic intervals bind sample/target/block order,
the same program/PID/pool, and the unchanged completed-action boundary.

The original campaign, source/binary/runtime inventory, paired build record,
runner source and exact profile program bytes are bound before collection.
Full byte hashes are checked at phase boundaries; source/executable hashes and
runtime metadata are checked before every job. Summary rechecks raw hashes and
all declared successes/failures. Supplied libraries are bound; actual loader
resolution and kernel identity remain unknown.

All outputs are **diagnostic only** and cannot be used as parameter profiles or
independent accuracy acceptance. Ratios describe paired blocks. They do not
establish a universal correction, a pure instrumentation cost, or a cause such
as clocks, scheduling, caches or CPU polling. Block validation, warmup, port
readback and JSON conversion can influence subsequent state. No observation is
removed, no overhead is subtracted, and no hardware setting is changed.
