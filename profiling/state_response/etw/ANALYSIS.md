# Offline owned-thread ETW analysis

This workflow reads an existing ETL. It never starts, stops, cancels, or changes a recorder. Recording remains a separate, explicit Windows-permission operation documented in `PROTOCOL.md`. Ordinary simulator use does not require these Windows-only diagnostic dependencies.

## Dependencies and optional readers

Install Windows Performance Toolkit (`xperf.exe`), MSVC, the Windows SDK, CMake, and provide an explicit include directory containing `nlohmann/json.hpp`. The pinned llama.cpp vendor directory used by the native probes is one supported source of that header. No dependency is downloaded and no input path or unknown measurement is guessed.

From an MSVC developer command prompt, use a fresh build directory:

```powershell
cmake -S profiling/native/state_response/etw-tools -B NEW_BUILD -DNLOHMANN_JSON_INCLUDE_DIR=EXPLICIT_VENDOR_DIR
cmake --build NEW_BUILD --config Release
```

`etw_export` is the exact promoted source used to obtain SDK `TRACE_LOGFILE_HEADER` clock and loss counters, including post-`ProcessTrace` receipts. Classic kernel TDH schemas may be unavailable; its output stays unqualified and is not used to guess missing scheduler fields. `etw_marker_export` scans every provider's raw payload for the exact instance's begin/end markers. Process command-line matches are retained as raw candidates and are **not** accepted as emitted markers. `etw_interrupt_export` exports raw initial/end QPC candidates and payloads, which Python must qualify against xperf's independent schema.

Identity JSON must supply an instance name, owned actor PID/main TID, native launch/READY/sampling ticks and clock frequency, and SHA-256-bound process/result evidence. The interrupt reader additionally requires `window_begin_qpc` and `window_end_qpc`. See the retained launch identity and receipts referenced by the experiment report; every output directory must be new.

```powershell
etw_export.exe TRACE.etl IDENTITY.json NEW_HEADER_EXPORT
etw_marker_export.exe TRACE.etl IDENTITY.json NEW_MARKER_SCAN.json
etw_interrupt_export.exe TRACE.etl INTERRUPT_IDENTITY.json NEW_INTERRUPT_EXPORT
```

## Independent schema export

Use the installed xperf, read-only, with the actual ETL. Separate provider exports bound output size and keep the scheduler, module, and interrupt schemas explicit:

```powershell
xperf -i TRACE.etl -o NEW_SCHEDULE.csv -a dumper -provider "{3d6fa8d1-fe05-11d0-9dda-00c04fd7ba7c}" "{3d6fa8d0-fe05-11d0-9dda-00c04fd7ba7c}" -add_rawdata
xperf -i TRACE.etl -o NEW_PERF.csv -a dumper -provider "{ce1dbfb4-137e-4da6-87b0-3f59aa102cbc}" -add_rawdata
xperf -i TRACE.etl -o NEW_IMAGES.csv -a dumper -provider "{2cb15d1d-5fc1-11d2-abe1-00a0c911f518}" -add_rawdata
```

These intermediate dumps may contain unrelated OS process names and stay in ignored local experiment storage. The Python analysis only emits the owned process's schedule and image data, relevant PID-zero kernel image ranges, exact instance markers, and global interrupt intervals needed to interpret owned CPU assignment. It removes unrelated scheduler process/thread identities.

```powershell
python -m profiling.state_response.etw_analysis --dump NEW_SCHEDULE.csv --raw NEW_HEADER_EXPORT/raw-export.json --identity IDENTITY.json --result NATIVE_RESULT.json --out NEW_SCHEDULE_ANALYSIS
python -m profiling.state_response.etw_samples --analysis NEW_SCHEDULE_ANALYSIS/analysis.json --identity IDENTITY.json --images NEW_IMAGES.csv --perf NEW_PERF.csv --actions NEW_SCHEDULE_ANALYSIS/action-scheduler-states.jsonl --marker-scan NEW_MARKER_SCAN.json --timelines NEW_SCHEDULE_ANALYSIS/timelines.json --raw-interrupts NEW_INTERRUPT_EXPORT/raw-interrupts.jsonl --raw-interrupt-receipt NEW_INTERRUPT_EXPORT/raw-interrupt-export.json --out NEW_SAMPLE_ANALYSIS
```

## Qualification and meaning

The pipeline checks zero loss counters, the raw QPC clock/frequency, unique owned process start within native launch/READY, thread lifetimes, exact marker payload/provider/opcode and command QPC brackets, unique address/time image ranges, and all raw interrupt descriptors/routines/durations against xperf. Unsupported layouts are rejected. Missing ready transitions or unbounded timeline tails remain unknown. Extra xperf columns are retained explicitly, never silently discarded.

Scheduler `Running` means CPU assignment, including overlapping ISR/DPC. Raw `InitialTimeQPC` and event-end QPC form interruption intervals; the analyzer unions them per CPU before intersecting owned assignment. Nested ISR/DPC are never counted twice. Threaded DPC opcode 66 is not silently treated as interrupt preemption. The currently supported raw descriptors are the actually verified 64-bit PerfInfo v2 ISR50/24 bytes, ISR67/20 bytes, DPC68/16 bytes, timer-DPC69/16 bytes. New layouts require independent qualification.

Sampled PCs only identify module location; they do not establish a function, useful computation, spin loop, or a duration. The output separates samples inside and outside same-CPU interruption intervals. A thread's creation entry module is not evidence of its later execution location. Auxiliary CPU time overlaps main wall time and must not be added to it. Interrupts may be generated by the workload itself; do not assume they are unrelated background pressure or charge an overhead twice. Counter-accounting residuals remain unknown. No latency predictor, parameter fitting, or CUDA kernel timing is produced here.

The raw initial-QPC layout and opcode reference is Microsoft's [PerfView KernelTraceEventParser](https://github.com/microsoft/perfview/blob/main/src/TraceEvent/Parsers/KernelTraceEventParser.cs), `ISRTraceData`/`DPCTraceData` and PerfInfo event registrations. Actual descriptors, addresses, and durations still require the independent xperf checks above.

Synthetic regression tests run without ETW, administrative privileges, native programs, or hardware workloads:

```powershell
python -m unittest profiling.state_response.tests.test_etw_analysis
```
