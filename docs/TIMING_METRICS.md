# Timing boundaries for heterogeneous inference

Our primary strategy metric is **warm strategy latency**: elapsed time from a
declared ready input/state to the declared ready output/state. We also report
device activity and runtime diagnostics. A useful-compute-only metric cannot
replace strategy latency when comparing partitions, offload and overlap.

| Quantity | Treatment | Reason |
|---|---|---|
| Process creation, imports, backend/context construction | Separate cold-start metric | Usually paid once per session |
| Initial model placement | Declare separately; include if evaluating first-response latency | May be a startup choice, unlike recurring offload |
| Repeated kernel launch, host submission, synchronization | Include in warm strategy latency | Changes with task granularity and CPU/GPU cooperation |
| Repeated weight/KV movement required by a strategy | Include | This is storage scheduling, even when an API calls it a load |
| Device work already submitted while the host is unavailable | Continue according to the device's own supplied service | A host pause does not imply a device pause |
| External CPU/GPU contention and unavailable memory | Input resource history, with explicit source and uncertainty | We do not simulate the OS scheduler |
| Repeated wake/recovery after idle or contention | Model as state/history dependent when observed | Repetition makes it part of steady usage; it is not automatically a removable cold start |
| Input restoration performed only to repeat an isolated benchmark | Outside the declared target interval, with its own timestamps | It is benchmark preparation, not unreported inference work |

For example, a GPU partition that performs 1 ms of arithmetic but needs 0.6 ms
of recurring submission/transfer is not automatically faster than a CPU
partition that finishes the same output in 1.3 ms. Both arithmetic efficiency
and the complete critical path matter. Concurrent intervals must be composed
through dependencies, not added twice.

## Current event-stage experiment

`profiling/native/event_stages/probe.cpp` constructs the Executor before entering
the sampling routine. It validates and performs **20 warmup executions** before
READY. The per-invocation timer begins after the isolated benchmark's reset and
ends after its declared action sequence. Process creation, Executor/context
construction and those warmups are outside that timer. This does **not** prove
that clocks, temperature or post-idle runtime state are stationary.

The current complete-action boundary still includes recurring GGML execution,
launches, copies and synchronizations, including the executor's existing
synchronization choices. These measurements characterize this executor, not
an optimized production backend.

Three complementary views are kept:

1. **Warm whole-call latency:** the objective when comparing execution and
   storage schedules. Background interference can lengthen it through the
   supplied/estimated resource history; it is not relabeled useful compute.
2. **GPU kernel and memcpy envelopes:** CUPTI intervals attributed by exact
   correlation. They are diagnostic device envelopes, not necessarily pure
   useful arithmetic or preemption-free service. Kernel, copy and API columns
   can overlap and cannot simply be summed.
3. **CPU/runtime diagnostics:** raw thread cycles, API spans, gaps and wait
   dependencies. Raw cycles include runtime work and possible spin waiting;
   CPU useful-compute time remains **unknown** where not independently
   measured. A long CUDA API span is not automatically a launch cost.

Cold-start latency can be a future separate benchmark with explicit endpoints.
It must not be silently folded into the warm profile, or silently subtracted
from a recurrent cost. Reports must say which metric is being predicted.

## Choosing calibration workloads

Small graphs test dependency and accounting correctness. Hardware-efficiency
calibration should also use representative matrix/attention/FFN sizes, where
device execution is identifiable separately from measurement and runtime
overheads. We do not need to fit every unpredictable tiny-call pause before
collecting those component profiles. Stable-resource calibration, replay with
a supplied availability trace, and forecasting an unknown future trace are
three separate evaluations.
