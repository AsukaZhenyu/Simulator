# Native physical stage capture

This optional Windows probe executes the unchanged byte-pinned GGML
`Executor.action` and records native host boundaries plus direct CUPTI
concurrent-kernel, memcpy, runtime API, driver API and external-correlation activity.
It does not estimate an available FLOP rate from utilization or foreground
slowdown. GPU activity and host API waits overlap; do not add them twice.

The existing strict F32 GGML runtime is reused. Building does not launch the
probe or any workload. Supply explicit local paths; no downloads are performed.

```powershell
cmake -S profiling/native/physical_stages -B profiling/native/physical_stages/build-release-v1 -G "Visual Studio 17 2022" -A x64 -DCMAKE_TRY_COMPILE_CONFIGURATION=Release -DLLAMA_ROOT="EXISTING_LLAMA_ROOT" -DGGML_BUILD_DIR="EXISTING_GGML_BUILD" -DRUNTIME_DIR="MATCHING_F32_RUNTIME" -DCUDA_ROOT="EXISTING_CUDA_ROOT" -DCUPTI_ROOT="EXISTING_CUPTI_ROOT" -DPython3_EXECUTABLE="EXPLICIT_PYTHON"
cmake --build profiling/native/physical_stages/build-release-v1 --config Release
```

The executable and `build-manifest.json` are under the build's `Release/`
directory. The receipt binds provided libraries, headers, SDK assets, compiler,
generated sources and unchanged historical executor sources. It does not prove
the dynamic loader's transitive resolution.

The stable command has eleven arguments after the executable:

```text
physical-stages-probe PROGRAM.json full|target cupti|bare TARGET_INDEX|none WARMUP SAMPLES default|0 READY.json GO.json STOP.json OUTPUT.json
```

All four handshake/output paths must be distinct and absent initially. The
parent supplies the matching GGML DLL, CUDA `bin/`, and CUPTI `lib64/` runtime
search paths. Even `bare` loads the linked CUPTI DLL but never starts capture.
`full` requires target `none`; `target` repeats one original COMPUTE action with
its exact prepared upstream bytes. The explicit pool-poll option is the same
historical `default` or `0` contract. It is not an OS scheduling policy.

After initial validation and warmup, the probe writes READY with raw QPC clock
metadata under `clock` and the physical nanosecond clock under `physical_clock`.
The parent then exclusively creates GO:

```json
{
  "schema": "physical-stages-go-1",
  "clock_id": "windows_qpc",
  "frequency_hz": 10000000,
  "start_tick": 123456789,
  "end_tick": 124456789
}
```

Numbers above are schema illustrations only. The actual frequency comes from
READY; the parent chooses a future start and an explicit end. Windows QPC ticks
are converted to ns using quotient/remainder and floor rounding. Before enabling
any activity, CUPTI receives the same conversion as its timestamp callback.
Native action and CUPTI activity timestamps therefore share
`windows_qpc_floor_ns` / `absolute_qpc_zero`. No Nsight clock-offset guess is used.

After exactly SAMPLES complete invocations, `sampling-complete.json` is written
beside READY. The probe waits for owned STOP before flushing CUPTI, final port
readback, verification, destruction or JSON serialization. Root orchestration
must retain partial failures and stop all owned actors on failure; the probe
does not retry or discard a slow sample. A single invocation begun within GO
may finish after its end; the recorded timestamps remain authoritative.

`cupti` and `bare` use the same native action clocks and batched CPU counters.
Here `bare` means CUPTI disabled, not original clock-free execution. A paired
observer check is still necessary before claiming uninstrumented accuracy.
Capture overhead is retained, not subtracted from a whole-layer total.

Output schema `physical-stage-native-1` retains every original action index and
per-invocation external ID. CUPTI runtime API correlations connect enqueue and
synchronization APIs to kernel/memcpy records; `external_correlation` links them
to actions. Kernel/memcpy intervals are device activity, while runtime APIs are
inclusive host intervals. Unknown/zero timestamps, dropped CUPTI records, local
buffer overflow and callback errors remain explicit. Any such failure must
prevent physical parameter qualification; a numerical success alone is not
complete capture qualification.

Batch process/main-thread kernel+user CPU counters include runtime, reset and
profiling work. They do not provide exact tiny-action active service or separate
Running/Ready/Blocked states. Numerical port evidence reuses the existing
untimed executor reference and must also be checked with the independent oracle
outside the native timer. The executable does not claim to apply that oracle.

```powershell
python -m unittest discover -s profiling/native/physical_stages -p test_build.py -v
```

These tests are source/receipt checks, not hardware calibration.
