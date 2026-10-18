# Physical parameter calibration

Performance calculations belong to `llm_infer_model.performance`. This optional
profiling package reads measurements, summarizes physical parameter evidence,
and compares independent measurements with predictions from that modeling API.
It has no total-latency regression, learned resource-response equation, DNN, or
second execution simulator.

## Responsibility boundary

| Layer | Owns |
|---|---|
| Modeling | Physical equations, stage composition, dependency/resource waits, remaining work under a rate trace, memory capacity constraints |
| Profiling | Raw values and file hashes, timing scopes, numerical validation receipts, exact process medians, variability, physical parameter estimation, independent residual checks |
| Mapping | Strategy actions and the resulting work/dependency/resource graph |

`ParameterObservation` records a scalar, unit, exact context, run/process/seed,
split, UTC process interval, source hash, evidence references, and timing scope.
Unknown values stay unknown. `load_observations` rehashes the referenced files;
it does not independently prove the collector's scalar extraction or instrument
semantics. `source_verified` in input JSON is ignored until files are checked.
Numerical verification is explicit rather than presumed.

Only these conversions are accepted:

| Physical input | Permitted timing scope | Meaning |
|---|---|---|
| `flop/s` | `device_compute_active` | Known work divided by a matching compute probe's active interval. An effective rate for that probe, not a claim of pure arithmetic or device peak throughput. |
| `byte/s` | `payload_transfer_active` | Known payload divided by payload-transfer time, with unrelated host/queue/synchronization work outside the interval. |
| Fixed `ns` | `host_exclusive_completed` | Matching exclusive completed host stage; excludes time already counted by another stage. |
| Capacity `byte` | `capacity_snapshot` | Capacity observation; never interpreted as bandwidth or a slowdown multiplier. |

A total PV/whole-action interval includes submission, device execution,
layout/copy work, and waiting. It remains a `reference_duration`, not an
automatically inferred FLOP rate, bandwidth, or host cost. Overlapping host and
device intervals cannot be separated by subtraction.

## Estimation and validation

`estimate_parameters` accepts profile observations from exactly one context.
It takes the **exact median of per-process medians**. A process with more samples
does not get more weight. Every observation remains in the frozen artifact;
there is no outlier filtering. Within-process and between-process CV and at
least three independent runs are required by default. A renamed run with the
same PID and seed is not independent. Different seeds permit legitimate later
PID reuse. Failed evidence produces an unknown physical parameter rather than
a fabricated zero or an aggregate-total fallback.

The frozen artifact records all observations, parameter evidence, source
digests, process statistics, applicability context, freeze time and a digest.
`load_frozen_parameters` reconstructs the statistics and uses model-owned
`ParameterBundle` validators. Digest integrity is not external authenticity;
`verify_sources=True` additionally checks current raw file bytes. Default
receipt loading alone does not claim current raw verification.

`evaluate_calibration` calls `llm_infer_model.performance.evaluate` for all
predicted times. It requires heldout run IDs and seeds disjoint from profile,
and heldout collection after the freeze. Raw profile/heldout sources are
rehashed by default. Scalar rows check declared UTC chronology; they do not
establish an independent collector barrier. The accuracy gate uses the
process-balanced median; worst-run error is separately reported, not silently
claimed to pass. Reports separately contain:

- parameter evidence issues;
- heldout measurement stability/boundary issues;
- model prediction or an explicit missing-parameter/capacity error;
- observed-minus-predicted total/task residual and relative error.

For an accepted calibration claim, the profile context must already bind the
physical graph's canonical JSON SHA-256 as `graph_sha256` and the completed
reference boundary as `reference_timing_scope`. Supplying a dynamic timeline
also requires its `timeline_sha256` in that context. Missing bindings permit
diagnostic prediction only; conflicting bindings reject the request. This
does not prevent using a general hardware parameter bundle for other graphs,
but a calibrated accuracy claim cannot be transferred automatically.
`reference_timing_scope="whole_action_completed"` lets a multi-stage graph
(host submission, layout, device work) represent one composite measured action.
Its graph total must be compared with that completed action, not just one GPU
stage. No default subtraction separates those overlapping times.

A residual's cause is **unresolved**. It does not by itself prove a missing
mechanism or a faulty instrument. Diagnose parameter stability and model stage
composition with matching controlled measurements before changing either.

## CLI

Run in a source checkout with the library installed, or with `modeling` and the
repository root on `PYTHONPATH`:

```text
python -m profiling.parameter_calibration estimate --observations raw-observations.json --context exact-context.json --output frozen-parameters.json
python -m profiling.parameter_calibration evaluate --parameters frozen-parameters.json --graph physical-work-graph.json --heldout heldout-observations.json --output validation.json
python -m profiling.parameter_calibration import-legacy --campaign profiling/resource_calibration/results/smoke-v1 --split profile --output-directory profiling/parameter_calibration/results/legacy-profile
```

All outputs are new files, written with LF line endings. No command launches
hardware or mutates an old campaign. `legacy.import_legacy_resource_campaign`
verifies raw job artifact hashes, reconstructs the target intervals and steady
state assignments, and preserves transition counts. The historical 16 offered
load/allocation states provide **measured PV references only**. One process per
split and overlapping whole-action timing cannot establish the missing physical
rate/host parameters or hardware prediction accuracy. Historical source hashes
identify the old code version; the importer does not claim those bytes match
the current working tree, and never imports or runs the old Ridge fit.
