# Thread, runtime-wait and recovery measurements

This is an optional Windows measurement workflow. It supplies observations and
parameter evidence; `modeling/llm_infer_model/performance` owns predictions.
Diagnostic output is not a qualified cost table.

## Questions and controls

The same process covers initial idle, CPU load followed by idle, GPU load
followed by idle, and combined load followed by idle. Every load lasts one
second and every idle recovery lasts two seconds. All three predecessor states
occur in each recovery process, in a predeclared randomized order. Three
independent processes are the default. Separately compare:

* `solo`: foreground only, two CPU threads, per-action CPU brackets.
* `idle-actors`: both initialized background actors remain present but receive
  no replay requests. Their initialization creates CPU and GPU contexts.
* `recovery`: matched predecessor/recovery sequences in one foreground process.
* `solo-one-thread`: otherwise identical geometry with one CPU thread.
* `solo-bare`: ordinary action path without per-action counter instrumentation.
* `solo-bare-observer-off`: the same bare path without thread snapshots; other
  telemetry remains enabled. This controls observer effects separately.
* `solo-gpu-phases`, `recovery-gpu-phases`: targeted GPU enqueue, synchronization
  and stream-span diagnostics; these intervals overlap and cannot be added.

Optional `solo-gpu-auto`, `solo-gpu-spin`, `solo-gpu-blocking` and `solo-gpu-yield` explicitly
request CUDA scheduling flags in the foreground process. The native actor
records the requested and actual flags. Unset means no override, not a guessed
CUDA policy. `auto` does not identify the driver's instantaneous wait behavior.
These controls never change another process's CUDA context.
Compare the four explicit policies using the same preinitialization procedure:
calling the setter can itself initialize the CUDA context before the executor.
An unset-versus-explicit comparison also changes that initialization order and
cannot by itself isolate the waiting policy's effect.

## Build and collect

Build `profiling/native/state_response` into a fresh directory with the explicit
GGML source/build, strict F32 runtime and CUDA toolkit paths described in its
CMake configuration. Never overwrite binaries bound to an earlier campaign.

The thread observer requires .NET Framework and, for the NT snapshot provider,
an explicit MSVC compiler and Windows SDK include directory. The compiler checks
the structure sizes and offsets against that SDK; unsupported layouts fail.

```powershell
python -m profiling.state_response.thread_activity --build-out <NEW_HELPER_BUILD> --native-compiler <CL_EXE> --sdk-include <SDK_INCLUDE_VERSION>
python -m profiling.state_response.mechanism_collect prepare --out <NEW_CAMPAIGN> --binary <NATIVE_EXE> --runtime-dir <GGML_RUNTIME> --runtime-dir <CUDA_BIN> --cuda-runtime <CUDART_DLL> --thread-helper <HELPER_EXE> --seed 20261022 --repeats 3
python -m profiling.state_response.mechanism_collect collect --out <NEW_CAMPAIGN>
python -m profiling.state_response.thread_analysis --campaign <NEW_CAMPAIGN> --out <NEW_ANALYSIS_JSON>
```

`--conditions` can select an explicit subset before collection. Requests,
conditions, actor presence, thread counts, seeds, source hashes and timing
boundaries are stored in the manifest. A source change requires fresh prepare.
All measurements use the common measurement lock and clean up only owned
actors. At most one verified pre-GO startup retry is allowed; timed failures
remain failures. Run no compiler, tests or second hardware experiment during
collection. Outputs are create-only and use UTF-8/LF.

Use `--disable-thread-observer` to omit the query helper in every selected
condition while preserving native instrumentation and ordinary telemetry. The
real NT observer pilots had approximately 9 ms query spans despite a nominal
2 ms period. The helper is therefore not an exact short-action scheduler trace,
and changing its implementation did not establish instrumentation neutrality.

For the separately authorized bounded kernel diagnosis, use
[`etw/PROTOCOL.md`](etw/PROTOCOL.md) and the `etw_collect` controller. A unique
named session coexists with existing Windows loggers; no foreign logger is
stopped or cancelled. The controller binds sources before UAC and launches
actors only after the recorder is ready. Preserve failure receipts as well as
successful recordings. Raw system ETLs remain local; exports must filter owned
lifetimes and retain only numerical interruption events needed for accounting.

## What is measured

The native program records actual main thread IDs. `thread-diagnostic` preserves
the normal `Executor.action` path, but surrounds every action with CPU-accounting
and cycle queries. Action timers exclude those queries; whole invocation timers
include them. Every query bracket and its overhead remains recorded. Raw cycles
are not converted into time. CPU accounting has limited resolution and is not
an exact measurement of each small action.

The optional observer makes one `NtQuerySystemInformation` call per sampling
round and retains only specified PID lifetimes. Each state has the actual NT
query bracket, distinct from subsequent cycle/counter queries. State samples
join an action only when their complete query bracket is contained in that
action and phase. Actual cadence, missed periods and unavailable observations
remain visible. Native wait-reason codes retain their namespace; they are not
cast to managed wait-reason enums. See the [Microsoft API contract](https://learn.microsoft.com/en-us/windows/win32/api/winternl/nf-winternl-ntquerysysteminformation).

Snapshots distinguish observed Running, Ready and Wait states; they do not
establish exact time spent in those states or distinguish useful instructions
from runtime spinning. That requires finer scheduler/stack evidence. The tool
does not suspend target threads, start ETW or change global timer settings.

## Modeling boundary and independent validation

The common event engine can represent a completion gate with `Task.waits_for`.
A gate launches after its `depends_on` tasks and completes when the awaited
tasks finish, without a second GPU duration. Holding a declared host resource
represents a polling wait; holding a separate wait resource lets other host work
run during a blocking wait. A later resume task must acquire its host resource
normally. Service rates, wait mode and resume cost remain explicit inputs;
the engine does not infer Windows scheduling from a utilization percentage.

Host CPU service can use explicit `thread_cpu_ns` demand and
`thread_cpu_ns/s` supply. This separates time needing CPU service from a fixed
elapsed delay: zero supply pauses remaining work. Supply for one thread cannot
exceed 1e9. Resources remain exclusive and there is no automatic OS time slicing
or allocation of shared CPU service.

CSwitch Running means scheduler assignment, not uninterrupted thread execution.
Interrupts/DPC may execute on the same CPU without a thread switch. Intersect
their same-CPU interval union before interpreting assigned occupancy as thread
service; never sum nested ISR/DPC time twice. Likewise a sampled kernel PC in
such an interval is not automatically application driver work. An observed
heldout interruption timeline can support explicitly labeled diagnostic replay,
but must not become a forecast input or fitted remainder.

Diagnostic observations first test which mechanism is missing. Formal prices
must then come from separately scoped profile measurements, be frozen before
fresh heldout launches, and be evaluated by the same modeling implementation.
No heldout action duration, retrospective thread state or PID-specific offset
may become a prediction parameter. Low median error alone does not qualify
source stability, observer neutrality, recovery behavior or tail accuracy.

The [2026-10-22 investigation](../../docs/THREAD_RUNTIME_RECOVERY_DIAGNOSIS_20261022.md)
records the bounded kernel trace, sixteen controls and a main-thread placement
intervention. A separate frozen-profile/fresh-process validation reports 686
calls, primary median error 10.50% and P90 39.80%; GPU contention and crossings
still fail. It retains failed accuracy qualification and does not promote
diagnostic interruption timelines into predictive inputs.
