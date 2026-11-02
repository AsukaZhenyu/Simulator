# Bounded, synchronized device observations

`telemetry.py` uses only Python's standard library. `NVMLProvider` loads NVML
directly through `ctypes`; it does not start `nvidia-smi` for each sample or change
clocks, power settings, affinity, user applications, or device state. Unsupported
queries retain an explicit reason in `unknown`; a missing value is not zero.
The default nominal sampling period is 10 ms and is configurable.

The native probe and Python sampler use the same process-independent monotonic
clock: Windows `QueryPerformanceCounter` ticks and its queried frequency, or
POSIX `CLOCK_MONOTONIC` nanoseconds. Both clock ID and frequency must agree before
alignment. UTC is provenance only; it is never used to align intervals.

```python
from profiling.state_calibration.telemetry import DeviceTelemetrySampler, align_intervals

with DeviceTelemetrySampler(device_index=0, interval_ms=10) as sampler:
    native = run_native_probe()  # caller's subprocess; not provided by telemetry
raw_binding = sampler.write_jsonl("new-observations.jsonl")  # refuses overwrite
metadata = sampler.metadata()
aligned = align_intervals(
    native["blocks"], sampler.records, metadata["clock"],
    interval_ms=metadata["interval_ms"], action_clock=native["clock"],
)
```

Each sample includes absolute deadline, begin/end ticks around **all** provider
queries, individual provider query bounds, lateness, missed cadence periods,
successful observations, and field-level failures. Sampling follows absolute
deadlines: a slow query skips deadlines explicitly instead of pretending a
10 ms cadence was maintained. A sampler thread failure is retained in metadata.
Raw JSONL, metadata source SHA-256, clock, version and provider information allow
the caller to bind an experiment. No observer latency is subtracted.

On Windows, cadence waits use a process-owned high-resolution one-shot timer
created with `CreateWaitableTimerExW(CREATE_WAITABLE_TIMER_HIGH_RESOLUTION)` and a
manual stop event. `WaitForMultipleObjects` wakes for either the deadline timer
or explicit `SetEvent` cancellation. Relative negative 100 ns due times are
derived from remaining monotonic deadlines. This does not call `timeBeginPeriod`,
change the system timer resolution, alter power settings, or spin while waiting.
If the API or high-resolution flag is unavailable, metadata explicitly records
the reason and the `threading.Event.wait` fallback. `wait_policy` records the
actual API and whether high-resolution creation succeeded. Neither successful
creation nor the requested period guarantees OS scheduling latency; all late,
missed and long-query observations remain subject to the same quality gates.

The default GPU observations are UUID/index, P-state, SM/memory/graphics clocks,
temperature, power, device-wide memory occupancy, and GPU/memory utilization.
Machine CPU busy fraction is computed from consecutive OS counter reads:
`GetSystemTimes` on Windows, `/proc/stat` on Linux. Windows kernel time includes
idle and must not have idle added twice. On systems above 64 logical CPUs,
`GetSystemTimes` covers the calling thread's primary processor group; the
metadata records this scope rather than claiming a machine total.

## What the observations establish

An observation belongs to an action summary only if the **entire query interval**
is contained in that action. A boundary-overlapping sample is retained as
ambiguous and excluded from its numerical summary. CPU busy averages additionally
require their complete OS counter interval to be contained in that action.
There is no nearest-sample filling or interpolation.

`quality.model_ready` means only sufficient cadence for a **descriptive sustained
block association**. It requires at least three nominal periods and three fully
contained observations, no edge or internal gap above 2.5 periods, a query
duration no longer than one period, and at least one observed field. It does not
certify any particular model feature, independent observations, a calibrated
performance model, or causal attribution. Short actions retain the explicit
insufficient-resolution reason. Callers must additionally require that each
feature used by their model is present, supported and in its calibrated range.

NVML GPU utilization is time with one or more kernels active over a
product-dependent rolling window (approximately 1 second to 1/6 second). Memory
utilization is the fraction of that window with memory reads/writes; it is **not**
bandwidth saturation or allocated-memory fraction. Repeated 10 ms reads can
repeat the same underlying observation. On Ampere except GA100 and newer
architectures, the queried power is averaged over 1 second. Query bounds do not
reveal these rolling windows or turn them into instantaneous action measurements.
The measured workload itself contributes to utilization, so a high value alone
does not prove external contention or explain a slowdown. Device-wide memory
occupancy does not measure private workspace or process allocation lifetimes.

Before drawing a performance conclusion, the caller should compare sampler on/off
and stamped/bare timing controls with a fixed protocol. The experiment runner,
not this observation layer, controls those comparisons. No automatic clock to
throughput scaling, outlier removal, hardware setting changes, or causal claim is
implemented here.

References: [NVIDIA utilization definition](https://docs.nvidia.com/deploy/nvml-api/latest/api/structnvmlUtilization__t.html),
[NVIDIA power query semantics](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html),
[Microsoft GetSystemTimes](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getsystemtimes).
[Microsoft CreateWaitableTimerExW](https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-createwaitabletimerexw),
[Microsoft SetWaitableTimer](https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-setwaitabletimer),
[Microsoft WaitForMultipleObjects](https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-waitformultipleobjects).
