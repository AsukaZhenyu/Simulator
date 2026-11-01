# Staged, bounded scheduler diagnosis

This is a bounded Windows scheduler diagnosis, separate from timing calibration.
The profile is accepted by the local 10.0.26100 WPR; it declares 32 MiB kernel
buffers and 1 MiB marker buffers. Recording requires an explicitly authorized
elevated token containing SeSystemProfilePrivilege.

## Exact local support

`wpr -help advanced` confirms `-instancename` applies to recording commands and
**must be the last argument**. Every start, status, marker and stop in the
broker uses one fresh `LLMSimThread_<32 lowercase hex>` instance. It issues no
cancel, implicit kernel stop, global timer/power command or profile-interval
change. An unrecognized `xperf -loggers` inventory or a collision with the new
instance name causes refusal. Existing Windows loggers may coexist; their actual
elevated-token inventory is retained. A normal token may hide these sessions.
No existing recorder is stopped to make room. Windows supports multiple named
SystemTraceProvider sessions; if their limit is reached, this start must fail
without stopping any unconfirmed or foreign session. See Microsoft's
[SystemTraceProvider documentation](https://learn.microsoft.com/en-us/windows/win32/etw/configuring-and-starting-a-systemtraceprovider-session).

`OwnedThreads.wprp` requests CSwitch, ReadyThread, ProcessThread, Loader,
SampledProfile, CpuConfig, DPC and Interrupt, with scheduling/sample stacks.
Unlike the broad built-in CPU profile it omits unrelated RPC, .NET, JScript and
Win32k providers. Kernel events cannot be restricted to owned PIDs at recording
time with this profile. The raw ETL therefore stays local; derived outputs must
retain only owned PID/TID lifetimes, referring to others only as `other`.

## Prepared launch; requires a privileged session

Do not run this until the coordinating collector is ready and the required
Windows permission has been obtained. Replace `NEW_RUN` with an absent child
directory and generate a fresh GUID for every attempt; never reuse an instance.
The script does not self-elevate or launch UAC.

```powershell
$root = (Resolve-Path .).Path   # run from the repository root
$broker = Join-Path $root 'profiling\state_response\etw\capture-owned-threads.ps1'
$out = Join-Path $root 'profiling\state_response\etw\results\NEW_RUN'
$instance = 'LLMSimThread_' + [Guid]::NewGuid().ToString('N')
& $broker -OutDir $out -InstanceName $instance -MaximumTraceSeconds 30 -Capture
```

This call belongs in a privileged PowerShell process. Normal collection waits
for `capture-ready.json` **before launching its native actors**, avoiding any
UAC delay inside their 30-second native GO deadline. Keep one diagnostic GO job
to about five seconds, followed by the declared idle recovery support. Log the
native binary SHA, exact launch environment, PID creation times, native main TID,
and every actor's QPC clock/READY/GO/completion receipts. Do not launch a second
hardware job, compiler or test suite during this capture.

After sampling and recovery support, while actors still await STOP, the normal
collector exclusively creates `$out\stop.request.json` with UTF-8/LF JSON:

```json
{"schema":"llm-sim-etw-stop-request-1","instance_name":"LLMSimThread_REPLACE_WITH_SAME_GUID","owned_job_id":"diagnostic-job-0"}
```

The broker stops only the instance that it successfully started. It also stops
that instance on its 30-second deadline or broker exception. The start-command
bracket is included in the deadline. A failed start never triggers a stop of an
unconfirmed recorder. Stop failure is retained as failure, not hidden cleanup.
Wait for `capture-stopped.json`/`broker-final.json` before releasing native actors
to verification/teardown. Do not forcibly kill a successful broker: process death
can prevent any in-process cleanup. No claim of a complete trace follows merely
from an ETL file existing or the stop command returning success.

## Concrete export commands; no export executed yet

The following local help was verified: `xperf -help dumper`, `tracestats`,
`format`, `query`, `wpr -help tracing`, and `tracerpt -?`. Export to fresh files:

```powershell
xperf -i "$out\owned-thread-trace.etl" -o "$out\trace-stats.txt" -a tracestats -timespan actual -detail stack
xperf -i "$out\owned-thread-trace.etl" -o "$out\events-dump.txt" -a dumper -add_rawdata -add_fieldnames -stacktimeshifting -exc_dpcisr
tracerpt "$out\owned-thread-trace.etl" -o "$out\raw-events.xml" -of XML -rts
tracerpt "$out\owned-thread-trace.etl" -o "$out\decoded-events.xml" -of XML -summary "$out\trace-summary.txt"
```

`tracerpt -rts` cannot be combined with its summary/report option. Xperf dump
timestamps are microseconds relative to the trace start, not absolute QPC ticks.
Do not align them by guessing an offset. Inspect TRACE_LOGFILE_HEADER clock type,
PerfFreq, EventsLost and BuffersLost (SDK `shared/evntrace.h` documents these
fields), and verify the raw marker timestamps against the broker's QPC query
brackets and frequency. A marker is inserted through the recorded WPR control
provider inside each `InvokeTool` QPC bracket. If raw timestamps are not QPC,
frequency differs, marker checks fail or event losses occur, exact timeline
reconstruction is unqualified. Keep raw ETL, exports and explicit unknown ranges.
Do not silently treat a sampling gap as blocked time.

## Reconstruction contract; parser not yet implemented

Use only kernel process/thread lifetime evidence matching the collector's owned
PID, creation time, executable identity and native main TID. PID/TID reuse must
create separate identities. For each CPU, CSwitch `NewThreadId` begins a
scheduled-running interval and the matching switch-out ends it. This is on-core
scheduling, not useful compute; DPC/ISR overlap must remain separate. A switch-out
with Ready state starts a ready-but-not-scheduled interval. A switch-out with
Waiting state starts blocked time; the matching ReadyThread event ends blocking
and starts ready time, until the next switch-in. Missing boundary/lifetime or
ready event means unknown duration, never an invented split. Verify payload
fields/provider versions from the actual exports before implementing a parser.

Align these intervals with native action and counter-query brackets. Scheduling
events identify running/ready/blocked, **not** GGML computation versus VCOMP spin.
Classify sampled/CSwitch stacks only where recorded module mappings and symbols
support it: GGML, VCOMP, CUDA host API, kernel wait or unknown. Preserve raw
addresses and unknown symbols; no automatic symbol-network download is requested.
WaitReason alone does not identify the application-level cause. ETW is a separate
perturbing diagnostic, not an action-price source or qualified timing control.

Start with one five-second diagnostic. Only after clock, identity, loss and
instrumentation support are checked should three independent repeats be planned.
Freeze any resulting model structure and profile-only parameters before fresh
heldout collection. Never supply heldout per-action CPU or scheduler durations
as prediction features or subtract overlapping CPU/GPU spans to fit a residual.
