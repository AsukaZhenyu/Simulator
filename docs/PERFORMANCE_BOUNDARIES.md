# Performance modeling and profiling boundaries

The simulator computes performance in **modeling**. **Profiling** measures and
summarizes evidence, estimates named parameters, and checks those parameters and
the model against independent measurements. Mapping chooses the execution and
storage strategy; it does not supply an alternative timing predictor.

This distinction concerns ownership, not whether a parameter came from a formula
or an experiment. A bandwidth estimated by a median is still a parameter consumed
by modeling. The median calculation and its measurement evidence belong to
profiling.

## One path from a strategy to a prediction

This diagram states the ownership boundary. In this version, the new physical
graph is supplied explicitly; general lowering from the existing compiler is
still a separate integration step, as detailed below.

```text
Strategy/compiler contracts + explicit tasks, dependencies and work demands
                                                    |
Declared hardware parameters ---------------------->|
Profiled, frozen parameter bundle ----------------->| modeling
Resource state timeline --------------------------->| evaluate
                                                    |
                                      execution times and capacity checks
                                                    |
Independent native measurements ------------------>| profiling comparison
```

`llm_infer_model.performance` is the installed, standard-library-only performance
module. The optional source-checkout profiling tools call this module to produce
predictions; the installed module does not import those tools, launch a native
probe, read a saved campaign, fit coefficients, or compute sample medians.

The same types and evaluator are available as `llm_simulator.performance`; this
is a public facade, rather than a separate implementation. The optional workflow
is [parameter calibration](../profiling/parameter_calibration/README.md).
The [18 October implementation report](PERFORMANCE_BOUNDARY_REFACTOR_20261018.md)
records the migration, regression checks and retained-data evidence limits.

| Responsibility | Owner | Examples |
|---|---|---|
| Work and dependencies | Modeling and compiler contracts | FLOPs, byte traffic, task prerequisites |
| Strategy choices | Mapping | Placement, partitioning, prefetch, retention |
| Parameter values and provenance | Declared inputs or profiling artifacts | Effective FLOP/s, byte/s, fixed latency in ns |
| Sample statistics and estimation | Profiling | Medians, variability, calibration splits |
| Progress, scheduling and completion | Modeling | Remaining work, busy resources, state transitions |
| Capacity accounting | Modeling | Declared live reservations plus external allocations |
| Prediction versus independent observation | Profiling | Error, coverage and failed quality gates |

## Explainable inputs, rather than a total-time fit

The new model represents a task as work on named resources. A work demand names
the parameter that converts its work amount into service time. Its units remain
explicit: FLOPs with FLOP/s, bytes with byte/s, and fixed overheads in ns. A task
can express compute and local-memory demands with the declared overlap rule;
serial host work, transfers and dependencies can be separate tasks.

The distinction matters for calibration. Measuring a completed PV action does
not identify how much of that action was host work, device arithmetic, memory
traffic, or waiting. A tool must not silently reinterpret a completed-action
median as pure compute throughput. Existing exact-cost tables may retain
explicitly scoped composite durations. The new physical-parameter workflow
keeps total PV observations as references, rather than pure-rate or
exclusive-host parameters.

Memory occupancy has a separate role. Allocated bytes consume capacity; they do
not automatically lower bandwidth or compute speed. A resource timeline can
explicitly change available service. Inferring that service from utilization,
clock frequency, allocation size, or a background-work command requires its own
measurement and modeling evidence.

For example, three explicitly declared tasks can represent host submission,
GPU work, and host completion processing. Independent CPU work can overlap the
GPU task. Modeling computes the dependency critical path; it does not sum the
durations of all tasks. A CUDA stream interval and the overlapping host interval
must not both be inserted as sequential costs for the same work.

## What a calibration artifact must establish

Parameter estimation and evaluation are separate operations:

1. Retain the original observations and their units, timing boundaries, numerical
   checks, hardware/backend identities, source hashes and collection protocol.
2. Use only the calibration split to estimate the declared parameters. Record
   the estimator and its source observations, including unknowns and variability.
3. Freeze the parameter bundle before evaluating independent data. A validation
   duration cannot be copied into a parameter used to predict that same duration.
4. Ask modeling to evaluate the declared graph, parameter bundle and state
   timeline. Report the resulting prediction alongside the actual observation.
5. Keep statistical quality gates, prediction error and software correctness as
   separate results. Successful execution of a model is not hardware validation.

Calibration can identify a wrong parameter value or a missing mechanism. Changing
a parameter and changing a model are different actions and must be reported as
such. Heldout failures must not silently tune either.

## Existing APIs and historical evidence

`AnalyticalCosts` remains the profile-independent convenience API. Callers supply
effective compute and memory rates and fixed overheads. Its output remains an
uncalibrated estimate unless an independent experiment establishes otherwise.
The analytical builder also declares macro work and traffic approximations;
moving the numerical formula into modeling does not validate these counts.

Legacy sample statistics, MoE measurement conversion, calibration runners and
Nsight parsing now live in the installed `llm_infer_profile` package, whose
source directory is `profiling/llm_infer_profile`. The old modeling import
paths retain lazy compatibility forwards. The performance core never imports
that package; existing scripts can still request the legacy tools explicitly.

The earlier `profiling/resource_calibration/functional.py` is a historical Ridge
response diagnostic. Its fitted total-time response and single-operation runtime
carrier do not define the new modeling API. They are retained to reproduce the
previous report, including its failed accuracy checks. Their old numbers must
not be relabeled as measurements of individual physical mechanisms.

Historical campaigns bind the exact source bytes that produced them. Source
changes require a separate versioned verification path or a verified source
snapshot; rewriting an old manifest or raw result to match new code would destroy
that evidence. A verification of frozen historical sources is distinct from a
new validation of the current implementation.

## Complete single-layer measurements

The optional `profiling/single_layer_calibration` workflow measures the actual
compiled serial Dense action inventory in three ways: isolated actions, actions
in their real execution order, and independently timed whole invocations.
Profiling owns the observations, medians, histograms, source checks and frozen
artifacts. Model-owned `performance.serial_actions` constructs the dependency
chain from inclusive completed-action durations and explicitly disjoint gaps.
An inclusive action price already includes submission, device work and waits;
internal kernel or synchronization prices must not be added to it again.

Adding action medians is a deterministic approximation to whole-program median
latency, not a statistical identity. `performance.serial_distributions` supplies
a bounded secondary model: choose one profiled process with equal weight, then
draw serial intervals independently from that process's supplied marginals.
Its expected duration follows linearity; its median and tail estimates require
conditional independence. Profiling estimates the marginals; modeling computes
the resulting latency. Complete measured totals are not input prices or fitted
corrections. Fresh uninstrumented measurements validate both alternatives while
the original primary acceptance gates remain unchanged.

These scoped host-completed prices do not establish pure FLOP/s, memory bandwidth,
CUDA kernel durations, physical memory capacity or transfer to arbitrary unseen
strategies. The new adapter does not automatically recalibrate `AnalyticalCosts`.

## State-conditioned completed actions

`performance.conditioned_response` adds an explicitly scoped response form to
the **same** event evaluator. Its input is an exact action/state table, not a
whole-layer fitted correction. The reference work of an inclusive completed
action is one invocation; the declared approximation integrates its remaining
progress at `1 / T(state(t))`. Already completed work survives a state change.
The tagged zero-duration approximation is distinct from an unknown price and
from an explicit zero service rate that pauses work.

`RequestResponseDelays` describes the gap between a known owned-worker request
and its effective response. CPU and GPU start/stop delays are independent named
parameters. Modeling transforms the external request history, cancels a pending
edge when an opposite request supersedes it, and then calls the existing event
engine. It does not estimate delays, read measurements, or predict an operating
system scheduler. Memory-hold and availability changes retain their original
external timestamps. An unknown required delay is rejected.

The optional `profiling/state_response` collector measures action intervals,
disjoint gaps and owned-worker intervals on a shared clock. It estimates exact
per-process medians, then gives independent processes equal weight, freezes the
parameters, and validates on fresh processes with independent request orders
and seeds. Worker response parameters use **profile background-worker**
intervals only; heldout worker activity and internal action durations do not
become prediction inputs. See the [calibration report](STATE_RESPONSE_CALIBRATION_20261021.md).

This response form requires its exact program, shape, backend, hardware,
threadpool, affinity and synchronization context. Binary replay controls are
offered work, not measured utilization. Held allocation bytes do not establish
physical residency or bandwidth loss. These inclusive action prices cannot be
overlapped or supplemented with the same internal GPU work a second time.
Neither this adapter nor a favorable median calibrates the library's analytical
defaults or establishes arbitrary unseen-policy performance.

## Scope of the physical graph implementation

The first resource graph model is a mechanism-oriented building block, with
explicit work, parameter units, dependencies, resource timelines and capacity
reservations. It is not a complete operating-system, cache, driver, kernel or
distributed-collective simulator. Those effects require explicit tasks,
parameters, contention rules or documented residuals.

The public `AnalyticalCosts` compute and transfer methods now delegate their
numerical calculation to modeling. Existing mapping compilations continue to
consume their generated exact costs in the tensor runtime. The new physical
`Graph` evaluator is also directly usable, but a general compiler lowering from
every mapping transform to this finer physical graph is not automatic. Supplying
a profiled parameter bundle to a standalone graph does not silently recalibrate
the existing full LLM convenience workflow.

Within one `Task`, multiple work demands explicitly use a roofline maximum:
demands progress together and the last demand determines completion. Sequential
stages require dependency-linked tasks. A resource is exclusive; other shared
engines must be declared through additional resource locks. Input tuple order is
the deterministic priority among simultaneously ready tasks, not a search for
the fastest schedule.

In particular, moving performance calculation to modeling fixes the architecture
boundary. It does **not** by itself explain the previously observed fast/slow PV
latency groups, calibrate arbitrary external applications, establish full-model
hardware accuracy, or implement a new optimal strategy searcher.

## Thread service, completion waits and interruptions

`Task.waits_for` is a completion gate, with launch dependencies separate from
the tasks it awaits. It holds only declared resources and memory reservations.
A virtual blocked-wait resource can release CPU availability; explicitly
reserving a CPU resource represents exclusive polling occupancy. Gates add no
second GPU duration. A subsequent host resume stage must receive CPU service.
This is not an implementation of a preemptive OS scheduler.

`WorkDemand(unit="thread_cpu_ns")` describes active service for one thread,
including runtime work where explicitly measured. Its parameter unit is
`thread_cpu_ns/s`, bounded to 0..1e9. Zero supply pauses remaining work; a fixed
elapsed `ns` parameter continues with wall time. Neither supply nor useful
computation is inferred from CPU utilization, raw cycles, or a replay request.

Interrupt/DPC execution can overlap a CSwitch-assigned Running interval without
an application thread switch. CPU availability must therefore distinguish
same-core interruption time from thread service and avoid counting nested
interruptions twice. The [diagnostic investigation](THREAD_RUNTIME_RECOVERY_DIAGNOSIS_20261022.md)
measures this distinction. A replay of observed heldout interruption times is a
diagnostic, not a forecast or accepted calibration. Predicting workload-induced
interruptions and driver service still requires independently qualified source
parameters and a causal model; the default LLM cost path has not acquired them.

## External available resources are the physical input boundary

New physical experiments should use `AvailableResources`, `ServiceBinding` and
`evaluate_available_resources`. The external trajectory supplies absolute CPU/GPU
service, bandwidth and already-available memory budgets. OS scheduling and other
applications are outside the simulator. Modeling computes LLM work, dependencies,
own resource locks and waits against that trajectory; profiling supplies scoped
physical parameters. No background-pressure multiplier is applied to the same
available service again. See the [interface and executable example](AVAILABLE_RESOURCE_MODEL_20261022.md).

The older request-response and interruption adapters below remain explicit
diagnostic forms. Their inputs must not be relabeled as independently measured
available service or accepted as new physical calibration evidence.

## Shared stateful service response (diagnostic hypothesis)

`ServiceResponseModel` and `build_service_response` put the state-to-service law
in modeling. Named competitors have explicit requested work and finite residual
interference states. A full target-by-competitor coefficient matrix describes
same-resource and cross-resource effects with the same equation for CPU, GPU
and transfer targets. After a request stops, residual interference drains instead
of resetting at each invocation. The common evaluator integrates the resulting
rates and resolves actual graph boundaries. A normalized reservoir is an explicit
response hypothesis; it is not evidence of an actual GPU FIFO or lossless queue.

The optional `profiling/service_response` package owns synchronized measurement,
source reconstruction, per-process statistics, frozen parameters and independent
validation. It imports modeling to predict durations; it does not contain a
second performance calculator. The first hardware adapter prices inclusive
completed actions in `invocation/s`. Those prices include internal submission,
work and waiting; adding those internal stages again would double count. Signed
coefficients are permitted only in this inclusive response scope, with a proven
positive denominator. Physical service coefficients remain nonnegative.

Baseline-fixed control overhead, declared proxy dynamics, background feedback,
instrumentation neutrality and arbitrary graph transfer remain qualification
limits. The [experiment report](UNIFIED_SERVICE_RESPONSE_20261022.md) separates
numerical integration checks, source stability and hardware prediction error.
The analytical LLM convenience workflow is not silently recalibrated by this
optional exact-context experiment.
