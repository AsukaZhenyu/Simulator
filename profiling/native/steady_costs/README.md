# Steady-state native costs

This new probe leaves all historical executors and frozen experiments intact.
It accepts:

```text
steady-costs-probe PROGRAM.json REQUEST.json OUTPUT.json
```

`PROGRAM` is either the existing `highlevel-accuracy-program-v1` full Dense
program, or a compact `steady-component-program-1` specification. The latter
builds real F32 GGML GEMM, grouped Attention PV or exact-byte memory-copy work
without materializing large input weights in JSON. CPU and GPU device choices,
thread count, CUDA device and the allocation budget are explicit.

The request schema is `steady-cost-request-1`, with explicit `mode`,
`target_action_index`, `warmup`, `warmup_min_ms`, `trials`, `repeats`, `poll`,
`instrumentation` and `main_thread_mask`. Zero is a valid original action
index. All price fields are totals over `repeats`.

- `full` performs the unchanged legacy action sequence with two QPC reads per
  trial. `full-stamped` adds preallocated per-action QPC intervals in the same
  sequence. Both reset **once outside** each entire trial and then repeat the
  same fixed-input inference. They do not advance generation history.
- `action` repeats one original action, or one compact component. Initial
  upstream inputs are prepared and checked outside the timer.
- `all-actions` is an isolated hot-action diagnostic with grouped trial
  arrays; it is not a mixed-context cost table.
- `async-graph` is an explicitly different compute boundary: repeated GGML
  async graph submissions and one final backend synchronization. Optional
  timing events are recorded through the pinned **actual GGML backend stream**
  API. The stream envelope may include host submission gaps and is never
  called active kernel time.
- `sync-idle` measures synchronization on a completed, empty selected backend.
  It does not characterize a busy device wait.

Warmup executes at least the requested count and minimum duration, with a
fixed 120-second safety deadline. It precedes every trial timer. Construction,
input initialization, resets, numerical reads and JSON serialization are not
in the timed trial. No CPU cycle counters, DNN or residual-based inference of
GPU active work are used. Host and device time columns must not be added.

The native process pins only its own main thread before backend construction,
records initial/final affinity, and restores the original mask before output.
Unknown/rejected affinity is an error, never an unpinned fallback.

Compact input generation is `mix32-index-v1`, with uint32 wrapping throughout:

```text
z = seed + (index+1)*0x9e3779b9 + (ordinal+1)*0x85ebca6b
z ^= z >> 16; z *= 0x7feb352d
z ^= z >> 15; z *= 0x846ca68b
z ^= z >> 16
u = (z >> 8) / 2^24
```

GEMM weight ordinal 0 is `(u-.5)/sqrt(k)`; input ordinal 1 is `u-.5`, with one
final F32 conversion. PV V ordinal 0 is `u-.5`; P ordinal 1 is positive
`((z>>8)+1)/2^24`, normalized over KV in double precision then converted to
F32. PV executes V transpose/packing, mul_mat, head merge and output-port
materialization. Its flattened result is **query, head, head-dimension** order.
Transfer bytes are ordinal-0 `z & 255`, checked over the full copied buffer and
exported as SHA256. An independent NumPy/hashlib oracle must accept outputs
before any timing can enter calibration.

Build manifests bind pinned source helpers, GGML event ABI files, CUDA headers,
runtime DLLs, compiler, cache and new source bytes. `--self-test` exercises data
and parser invariants only; it constructs no Executor or backend and performs
no hardware benchmark.

## Building on the host you are measuring

The target is `steady-costs-probe`. It needs a llama.cpp tree (for `ggml`), a
CUDA toolkit and, at run time, the directory holding the ggml/CUDA shared
libraries it loads. The build writes `build-manifest.json` beside the binary,
which is what `prepare_portable` binds a campaign to.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_ROOT=PATH/llama.cpp -DCUDA_ROOT=PATH/CUDA -DRUNTIME_DIR=PATH/CUDA/bin
cmake --build build --config Release -j
```

`profiling.dac_validation.build_probe` runs exactly those two steps, refuses an
existing binary and a runtime directory that holds no library the loader can
use, then requires the probe's `build-manifest.json`. Visual Studio is a
multi-config generator (the binary lands in `Release/`, and `--config Release`
belongs to the build command); Ninja and Unix Makefiles are single-config (the
binary lands in the build root, and `-DCMAKE_BUILD_TYPE=Release` is required or
the measurement is compiled unoptimised).

macOS builds with `-DPROBE_BACKEND=metal` and needs no toolkit: the CUDA event
vocabulary the pinned fragments use is answered by
`profiling/native/backend/metal/include`, which the project puts first on the
include path and uses as `CUDA_ROOT`'s default. The e2a column itself still
needs its strict-F32 runtime, and that runtime's patch is a CUDA bypass router
with no Metal counterpart yet -- the probe builds, the column is not ready, and
`tools.device_check` says so. Linux and Windows build with
`-DPROBE_BACKEND=cuda`; on Linux the probe links `pthread` and `dl`.
