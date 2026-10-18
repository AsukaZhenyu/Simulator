# Performance ownership and calibration refactor — 18 October 2026

Performance prediction belongs to modeling. Profiling supplies observations,
statistics and estimated parameters, then compares independent measurements
with predictions from modeling. Mapping supplies execution and storage choices.

## Implemented data flow

```text
profiling: raw observations -> process medians -> parameter estimates + evidence
                                                     |
modeling: work + dependencies + parameters + resource trajectory
                                                     |
                            execution times, waits and capacity checks
                                                     |
profiling: independent observations -> parameter/model calibration report
```

The installed `llm_infer_model.performance` package implements this computation;
`llm_simulator.performance` exposes the same implementation. It needs no saved
profiles, NumPy, CUDA or measurement program. Parameters can be explicit hardware
declarations or values estimated by profiling. Unknowns are not replaced by zero.

The public `AnalyticalCosts` compute and transfer methods now delegate their
formulas to modeling. Existing component sum/max, fork/join, effective composition,
pipeline and KV storage prediction formulas have also moved to modeling.
Profiling retains their fitting, raw trace extraction and residual statistics.
The historical total-time Ridge diagnostic is retained and explicitly marked
as historical; it is not the new physical-parameter model.

Old calibration runners, Nsight analysis and MoE measurement conversion now live
under `profiling/llm_infer_profile`. Lazy compatibility entries preserve existing
modeling import paths. The new performance core never imports profiling.

## What modeling calculates

- FLOP work at a named effective FLOP/s rate, and byte traffic at a named byte/s rate.
- Explicit fixed overhead stages, dependency waits and resource serialization.
- Overlap between independent CPU, GPU, host and transfer resources.
- Remaining work across piecewise resource changes, including pause and recovery.
- Declared resident memory, live task reservations and external memory occupancy.

Multiple demands within a task explicitly use a roofline maximum. Serial stages
must be separate dependent tasks. Resource locks are exclusive; proportional
bandwidth sharing and operating-system scheduling are not inferred.

Memory occupancy consumes capacity. It does not automatically imply a bandwidth
or compute slowdown. Likewise, utilization percentages are not available FLOP/s.
Converting observed pressure into available service needs separately calibrated
evidence; this refactor does not establish that conversion.

The runnable example `examples/library/interpretable_performance.py` gives a
hand-calculable 16 ns critical path. Halving GPU compute service during execution
changes it to 23 ns. These integers verify model semantics, not hardware accuracy.

## What profiling calibrates

The optional `profiling.parameter_calibration` workflow keeps two checks separate:

1. **Parameter evidence:** timing boundary, units, source hashes, numerical
   verification, process independence and variability. The estimator takes the
   exact median of per-process medians, using profile observations only. Default
   evidence requires at least three independent runs. Failed evidence leaves
   the physical parameter unknown.
2. **Model prediction:** modeling evaluates the supplied graph and parameter
   bundle; profiling compares its output with independent completed-action or
   completed-iteration observations. The report includes parameter problems,
   applicability problems, model exceptions, residuals and relative error.

Accuracy acceptance requires graph and reference-boundary bindings established
in the profile context; a dynamic timeline also requires a matching binding.
UTC chronology is checked as collector-declared metadata, not an independently
issued collection barrier. Accuracy gates use process-balanced medians; worst-run
errors are reported separately. A residual alone does not prove a missing
mechanism, and heldout data are not silently reused for tuning.

## What the retained data can establish

The old resource campaign was reimported without running hardware or the old
Ridge predictor:

| Split | Verified bound raw files | Target PV samples | Steady references | Transition/outside samples | Offered states |
|---|---:|---:|---:|---:|---:|
| Profile | 127 | 3285 | 488 | 2797 | 16 |
| Heldout | 85 | 1828 | 476 | 1352 | 16 |

These are completed PV reference times, not identified GPU throughput, memory
bandwidth or exclusive host overhead. Transition samples remain accounted for;
they are not discarded as outliers. One foreground process per split also fails
the new independent-run requirement. No new physical parameter accuracy claim
is made from this campaign, and a new freeze cannot be backdated to turn these
old observations into fresh validation.

Historical sources were archived before modification. Verification checks their
original campaign bindings and reports changed current files separately. It does
not claim an archived runtime was executed. Existing published result JSON files
and raw measurements were not rewritten.

## Validation and remaining integration

The [machine-readable verification receipt](data/performance-boundary-refactor-20261018.json)
records regression counts, the archived-source check and retained-data import
summaries: **2,571 discovered, 2,563 passed, 8 existing optional checks skipped,
zero failures or errors**, without double-counting reruns. The library
distribution test builds a wheel and source archive,
rebuilds from that archive, installs in a fresh environment without profiles or
runtime dependencies, and executes both the physical example and Dense/MoE demos.

Existing full LLM workflows still use compiler-generated exact cost tables in
the tensor runtime. General lowering from every mapping transform to the finer
physical `Graph` is not automatic. A profiled bundle supplied to a standalone
physical graph does not silently recalibrate the full LLM convenience API.

The next hardware work is to collect matching active-compute, transfer and
exclusive-host measurements across independent runs; estimate scoped parameters;
then validate complete actions and chains on fresh data. This refactor repairs
ownership and provides that calibration path. It does not claim reduced hardware
prediction error.

See [the ownership guide](PERFORMANCE_BOUNDARIES.md) and
[the parameter calibration workflow](../profiling/parameter_calibration/README.md)
for API details and timing scopes.
