# Performance under externally available resources

The primary physical interface now consumes the resources available to the LLM,
not a request to start background work. Operating-system scheduling and unrelated
applications belong outside this simulator. Their effects enter through a finite
external supply trajectory. Modeling computes how the LLM's own actions use that
supply; optional profiling measures the scoped parameters.

This corrects the interface and parameter-construction boundary. It does **not**
establish new hardware accuracy. The previous 9.43% median, 33.56% P90 and failed
joint-load results remain results of the earlier request-pressure experiment.

## Inputs and responsibilities

| Quantity | Representation | Responsibility |
|---|---|---|
| Available CPU/GPU computation | Absolute `flop/s`, or explicitly scoped single-thread `thread_cpu_ns/s` | External scenario or independent measurement |
| Available memory/link/SSD bandwidth | Absolute `byte/s` for a named supply | External scenario or independent measurement |
| Available memory capacity | Byte budget already excluding background allocations | External scenario or independent measurement |
| Operator work and byte traffic | Explicit `WorkDemand` values in the graph | Modeling/graph construction, with an evidenced work convention |
| Shape/backend efficiency | `ServiceBinding`, exact fraction in [0,1] or unknown | Declared assumption or profiling estimate |
| Submission, synchronization, layout and resume | Explicit physical stages, fixed overhead or completion gates | Modeling; parameters supplied by profiling |
| LLM resource ordering and overlap | Dependencies, named locks and declared partitions | Modeling, as specified by the strategy |

An external rate is not `1 - utilization`. An owned background worker's ON/OFF
request, host-call duration or completion count is not an available FLOP rate.
Those signals cannot silently substitute for the required inputs.

`AvailableResources` accepts ordinary model-owned `ParameterBundle` and
`ResourceTimeline` objects. Rates and capacities are piecewise constant between
explicit events, with an explicit clock, origin, provenance and finite horizon.
Unknown is distinct from zero: unknown service rejects a prediction; a known
zero pauses the remaining physical work until supplied recovery.

## One law and one execution engine

For each physical action demand, the initial scoped law is:

```text
effective_rate[action, supply](t)
    = efficiency[action, supply] * external_available_rate[supply](t)

completed_work = integral(effective_rate(t), actual_start, actual_finish)
```

CPU, GPU and transfer demands use the same rule. The rule is a transparent
physical approximation, not a claim that every operator has state-independent
efficiency. Different shape, precision, backend or physical timing scope requires
applicable evidence; a generic efficiency must not be guessed from one total
layer time.

`evaluate_available_resources` lowers these declarations to ordinary physical
parameters and resource points, then calls the existing `performance.evaluate`.
There is no second profiling predictor. The generated inputs remain visible in
the result for inspection. Stage boundaries follow predicted progress and
dependencies, so every later stage sees the same external time axis at its
actual predicted start. Work already completed is never reset after a speed
change. No background-pressure multiplier is applied again.

An `ns` fixed overhead is elapsed time and is not scaled by available CPU rate.
Runtime work that must receive CPU service must instead be declared as physical
work. A completion gate waits for the modeled device task and does not add its
duration a second time. A blocked host can release the CPU execution lane;
polling occupancy requires an explicit lock or physical work specification.

Multiple demands in one task retain the common engine's explicit roofline
maximum assumption. Sequential mechanisms require separate dependent stages.
Named locks are exclusive; this interface does not invent fractional sharing.
Tasks spending the same supply must either have dependency ordering or a common
physical lock. Independent partitions require separately declared supplies.

Available memory budgets already exclude outside allocations. The graph charges
only its own resident storage and reservations. Nonzero graph `external_bytes`
or timeline `external_memory` is rejected here to prevent double subtraction.
One shared-memory domain must remain one domain rather than duplicate the same
budget for CPU and GPU. A live capacity violation is reported explicitly; no
implicit eviction policy is invented.

## Installed-library example

```text
python examples/library/available_resources.py
```

The [example](../examples/library/available_resources.py) requires no saved
profile, native executable, GPU or optional numerical package. Its declared
CPU/GPU/copy graph finishes in 11.5 ms with constant supply and 18.5 ms under the
declared supply trajectory:

| Stage | Dynamic start | Dynamic finish | Explanation |
|---|---:|---:|---|
| Host submission work | 0 ms | 3 ms | CPU supply falls halfway through |
| GPU computation | 3 ms | 13 ms | Zero GPU service until 7 ms, then work resumes |
| Other host work | 3 ms | 6.5 ms | Overlaps the device and its completion wait |
| Blocked host completion gate | 3 ms | 13 ms | Same device completion, no second GPU charge |
| Copy | 13 ms | 18 ms | Link bandwidth falls at 14 ms |
| Host resume | 18 ms | 18.5 ms | Begins after copy and other host work |

The timestamps above are model semantics, not measurements. The result's task
times are relative to the requested execution window; add `result.start_ns`
for absolute timestamps on the external clock. Completion outside the known
supply horizon is rejected rather than reported as an extrapolated forecast.

## Profiling and correlated observations

The new optional [available-service tooling](../profiling/available_service/README.md)
has two distinct roles:

1. Estimate a physical stage's efficiency from known work and independently
   supplied resource service over its measured interval, with exact clock,
   context, timing-scope and provenance checks. Failed or missing evidence
   produces an unknown parameter. Inclusive host wall time cannot become pure
   GPU throughput or efficiency by division.
2. Retain existing inclusive action/gap/tail measurements as correlated
   same-invocation vectors. Bounded chronological selection keeps each original
   vector together; no action is independently cherry-picked. These samples
   support diagnostic audits and explicit serial replay, not a newly calibrated
   physical model or an unseen-strategy accuracy claim.

The distinction addresses the observed aggregation problem:

```text
sum(median(each stage)) != median(sum(stages from the same invocation))
```

Preserving an original vector fixes the loss of its observed correlations; it
does not establish a new physical service law. The previous inclusive traces
do not contain all independent external-supply and active-stage observations
needed by the new physical estimator. Their missing requirements remain
explicit; no whole-layer offset, inferred utilization or invented zero fills
the gaps.

## Validation and preserved evidence

The new modeling tests use hand-derived physical work and external service
trajectories. The profiler tests separate parameter estimation from independent
evaluation and reject inclusive timing scope, unknown supply, chronology or
source problems. Distribution tests run the installed example from outside the
checkout, without profiles or optional dependencies.

These are software and contract checks. A new hardware-accuracy acceptance
requires new profile-only physical evidence, a frozen model and fresh heldout
measurements with independently established external supply. The old heldout
data has already been used for diagnosis and cannot become untouched acceptance
data for a revised model.

The [delivery receipt](data/available-resource-boundary-20261022.json) records
the checked source revisions, test-log hashes and offline diagnostic evidence:

| Check | Result |
|---|---|
| Modeling regression suite | 575 tests, seven skips, no failures |
| New physical available-resource tests, included above | 22 hand-derived tests pass, including a 1,100-stage dependency chain |
| Public library suite | 63 tests pass |
| Final source/wheel installation after the dependency-chain fix | Pass, including the profile-free example outside the checkout |
| Available-service profiling | 33 tests pass |
| Historical response contracts and snapshot preservation | 38 tests pass |
| Existing physical parameter-calibration contracts | 58 tests pass |

One profiling test freezes efficiency 1/2 using three declared profile fixtures,
then keeps that artifact unchanged while evaluating the same 10-flop operator
on a different supply trajectory. Service is 4 flop/ns until 2 ns, zero until
7 ns, then 2 flop/ns. Modeling predicts the hand-derived completion at 13 ns.
This checks transfer to a new supplied state trajectory without refitting; the
fixtures are not hardware measurements.

The offline migration checks all 14,515 original profile invocations and
exports 24 chronological correlated vectors across 12 run/state cohorts.
All selected vectors replay exactly through the existing serial adapter.
The export creates **zero physical bindings**, listing the independent supply,
physical work/boundaries, separate overhead/wait and attribution requirements
that the old records cannot establish. Exact source-vector replay is a
consistency check, not an independent accuracy result.

The snapshot audit preserves all 4,025 historical source entries: 474 via
manifest-bound archives and 3,551 at their original paths. Two current source
files differ from that recorded version, with the exact old bytes retained in
the archives. There are zero preservation failures. This check does not
re-extract old native measurements or rebuild the old frozen parameter file.

The old request-pressure model and all its failed results remain available as
diagnostics. Current source changes intentionally differ from the old source
hashes. The [snapshot audit](../profiling/service_response_snapshot.py) checks
the exact manifest-bound retained source bytes without relaxing the old loader:

```text
python -m profiling.service_response_snapshot --manifest CAMPAIGN/manifest.json --expected-manifest-sha256 RECORDED_SHA256 --output NEW_AUDIT.json
```

This proves source preservation only. It does not claim a parameter rebuild,
execute the archived evaluator or revalidate raw measurement extraction.

The existing full-LLM tensor runtime also supports declared reference-work
service fractions. It remains a separate timing contract; these absolute
physical rates do not silently reinterpret its macro prices or calibrate the
default analytical LLM workflow. General fine-stage lowering and hardware
qualification remain explicit integration work.

## New physical-stage measurements and frozen comparisons

The [2026-10-22 experiment report](PHYSICAL_STAGE_CALIBRATION_20261022.md)
records 12 profile processes, 24 independent heldout processes and observer
controls. The completed-GPU reference-response hypothesis changes bare-target
median error from 29.00% to 19.59%, but P90 worsens from 51.76% to 64.57%.
This is not a passed accuracy calibration. All frozen candidates and failed
qualifications remain visible.

Direct CUPTI now attributes kernel, copy and host API intervals on a common QPC
clock. Before-target reference measurements are independent of the target's
latency, but remain stale workload response estimates. They do not establish
the continuous neutral availability required by the physical contract above.
CPU physical service stays unknown; GPU reference rates stay diagnostic.
`performance.reference_components` consumes scoped response parameters through
the common modeling engine and does not replace the physical supply contract or
change the default LLM cost tables. See the
[reusable measurement workflow](PHYSICAL_STAGE_MEASUREMENT.md) for collection,
freeze, independent comparison and evidence-audit commands.
