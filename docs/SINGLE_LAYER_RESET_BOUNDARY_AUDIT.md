# Reset boundary audit for the formal single-layer campaign

All 14 audited prepared cases place their initial tensors in **DRAM only**.
`Executor.reset()` therefore performs CPU buffer writes in this campaign; it
does **not** upload initial weights or history to VRAM outside the invocation
timer. The hypothesis that these resets hide H2D transfers is not supported by
the prepared programs. Actual explicit H2D/D2H COPY actions remain inside the
measured sequence.

This is a read-only source and metadata audit, not a latency experiment, a new
calibration, or an attribution of the remaining performance error. It reads one
prepared `profile-0` program for every case in
`profiling/single_layer_calibration/results/formal-v1`; the graph/action/placement
contract is fixed for the other profile and heldout seeds of that case. The
large numerical `initial_values` arrays are deliberately not parsed. Source
inspection and metadata extraction do not independently prove all runtime input
bytes remain unchanged during repeated execution.

## Per-invocation reset and in-timer copies

| Prepared case | Initial tensors | DRAM reset B | VRAM reset B | Timed COPY count (H2D/D2H) |
|---|---:|---:|---:|---:|
| h64-q1-history128-hybrid-macros | 142 | 197,508 | 0 | 136 (134/2) |
| h64-q1-history128-cpu-macros | 142 | 197,508 | 0 | 0 (0/0) |
| h64-q1-history128-gpu-macros | 142 | 197,508 | 0 | 144 (142/2) |
| h64-q1-history128-attention-ffn-cut60 | 145 | 197,508 | 0 | 142 (139/3) |
| h64-q1-history128-attention-ffn-cut64 | 145 | 197,508 | 0 | 142 (139/3) |
| h64-q1-history128-attention-ffn-cut68 | 145 | 197,508 | 0 | 142 (139/3) |
| h128-q16-history0-hybrid-macros | 14 | 664,896 | 0 | 23 (6/17) |
| h128-q16-history0-attention-ffn-cut124 | 17 | 664,896 | 0 | 29 (11/18) |
| h128-q16-history0-attention-ffn-cut128 | 17 | 664,896 | 0 | 29 (11/18) |
| h128-q16-history0-attention-ffn-cut132 | 17 | 664,896 | 0 | 29 (11/18) |
| h256-q32-history0-hybrid-macros | 14 | 2,525,568 | 0 | 39 (6/33) |
| h256-q32-history0-attention-ffn-cut252 | 17 | 2,525,568 | 0 | 45 (11/34) |
| h256-q32-history0-attention-ffn-cut256 | 17 | 2,525,568 | 0 | 45 (11/34) |
| h256-q32-history0-attention-ffn-cut260 | 17 | 2,525,568 | 0 | 45 (11/34) |

The reset byte counts are sums over **initial tensor locations**, not allocated
DRAM/VRAM capacity, scratch space, bus traffic, or elapsed time. H2D/D2H counts
refer to explicit COPY actions; internal graph materializations and backend
copies can exist inside COMPUTE actions. Full per-route payload sizes, weight
versus other reset bytes, prepared program bindings, and metadata-prefix hashes
are in [the machine-readable audit](data/single-layer-reset-boundary-audit.json).

## What the implementation does

* [Historical executor](../profiling/native/highlevel_accuracy/highlevel_probe.cpp)
  `Executor.reset` loops over every tensor's `initial_locations` and calls
  `ggml_backend_tensor_set` with its complete initial byte array. This includes
  weights, positions, hidden input and initial history blocks when present.
* The supplied CPU buffer implementation dispatches this to `memcpy`. The CUDA
  buffer implementation would dispatch a VRAM set to H2D followed by stream
  synchronization; no audited reset has a VRAM destination.
* [The current single-layer wrapper](../profiling/native/single_layer_calibration/probe.cpp)
  calls reset before each warmup and timed invocation in `context-trace` and
  `bare-replay`. Reset is outside each invocation timer but inside the sampling
  envelope. `isolated-components` performs a full validation replay first, then
  repeats individual actions without resetting before every sample.
* COPY remains the unchanged `Executor.action`: it physically copies the named
  tensor along its declared route and completes synchronization. No COPY action
  was removed, moved outside the invocation timer or replaced by an assumed hit.

## Numerical and inference interpretation

For all 14 audited graphs, initial tensor identities are disjoint from computed
output identities. The executor uses independent allocated tensor ports and
non-in-place compute construction; new KV outputs are separate from historical
input blocks. Repeating the same static invocation should therefore not require
rewriting every read-only weight for mathematical correctness. A future runtime
claim still needs before/after input-byte checks and independent full-port
numerical verification; this audit does not establish that by measurement.

The current result is a **reset-conditioned repeated static invocation**.
Rewriting DRAM inputs and weights can alter CPU cache contents, memory traffic,
worker activity or the entry state of later operations even when the reset's
own wall time is excluded. These are possible boundary effects, not established
causes of the observed error. Their direction and magnitude are unknown here.
They must not be turned into an estimated fixed overhead or a GPU multiplier.

Removing resets alone would still repeat the same hidden input, positions and
history. It would not become realistic continuous decoding with new tokens,
growing KV cache or changing requests. Both the current experiment and any
future persistent-input variant must keep that scope explicit. No reset
ablation is implemented in this audit; CUDA API/wait tracing is a separate
diagnostic and no existing source, binary, raw result or manifest is changed.
