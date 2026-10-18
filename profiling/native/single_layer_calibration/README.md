# Single-layer calibration probe

This independent wrapper calls the unchanged historical GGML `Executor.action`.
Its generated helper differs only by renaming `main`. It does not edit historical
sources, builds, DLLs or measurements. The execution boundary is synchronized,
serial, F32, with CUDA graphs disabled; it is not a production serving runtime.

```
single-layer-probe PROGRAM.json MODE WARMUP SAMPLES default|0 OUTPUT.json
```

The three explicit modes run in separate processes:

* `isolated-components`: prepare exact upstream values with one full replay,
  then warm and repeatedly time each COMPUTE/COPY action, matching the historical
  isolated-component context.
* `context-trace`: execute the complete action sequence on every invocation,
  recording every action, including EVICT/ADVANCE, in preallocated POD storage.
* `bare-replay`: execute the same complete sequence with only invocation clocks.
  Use this independent boundary for whole-layer accuracy acceptance.

Input reset is outside each invocation timer. All instrumentation storage is
allocated before sampling. Timestamp conversion, JSON construction, output reads
and file writing occur after sampling. The actual GGML calls can allocate their
own internal storage, as in the historical executor. Context clocks and POD writes
still have overhead: compare context-trace with bare-replay; do not call the
instrumented duration an uninstrumented hardware truth.

`default` retains the original default persistent CPU pool. `0` explicitly creates
an otherwise identical pool with poll=0, switches it only after synchronization,
and frees the old pool. Both requested and effective public parameter structures
and configured thread count are recorded. The supplied GGML header declares a
thread-count getter that this pinned source/DLL does not implement or export, so
`observed_n_threads` is honestly null. This is a controlled execution
variant, not a silent optimization or a change to thread count/affinity.

Output schema `single-layer-calibration-native-v1` contains:

* `iterations`: invocation ticks/duration plus `action_intervals` with index,
  kind, ticks, and begin/end offsets in ns; bare action intervals are empty.
* `components`: isolated samples, exact actions, timestamps and known workspace.
* `native_clock`: QPC or POSIX monotonic clock identity and frequency.
* `verification_native`: compatible with the independent NumPy full-port
  `highlevel_accuracy.bridge.verify_native(program, raw['verification_native'])`.
* `threadpool`: explicit default/0 request and effective public parameters.
* `cpu_diagnostics`: Windows process/main/non-main CPU counters and wall time over
  sampling envelopes; non-Windows CPU counters are null. The non-main difference
  includes all native auxiliary workers and does not prove GGML polling. Separate
  counter queries and OS accounting granularity can produce a small signed
  difference. These diagnostics are not prediction parameters.

Use offsets for exact accounting: gap before an action is its begin offset minus
the previous action's end offset (zero for the previous end before action zero).
Tail is invocation elapsed minus the final end offset. Action duration equals end
offset minus begin offset. These telescope exactly to the invocation duration,
without independently rounding each raw QPC difference. Retain zeros honestly.
No CUDA stream-active time is measured and host/device intervals must not be
added as disjoint components.

Configure CMake in a new build directory with explicit `LLAMA_ROOT`,
`GGML_BUILD_DIR`, `RUNTIME_DIR`, and `Python3_EXECUTABLE`; build Release. The supplied
GGML libraries are linked as-is. The build manifest binds native sources,
generated helper, supplied GGML headers, JSON header, libraries, runtime assets,
compiler, source manifest and CMake cache. It verifies precompile input hashes
again after linking. Runtime search is explicit in the Python runner; actual
loader resolution is not proven. The compiler/SDK's transitive system headers
and the original historical DLL compiler invocations are outside this manifest.

Source/provenance tests do not execute hardware:

```
python -m unittest discover -s profiling/native/single_layer_calibration -p 'test_*.py' -v
```
