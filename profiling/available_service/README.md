# Profiling against externally available resources

The [independent validation contract](VALIDATION.md) binds frozen parameters,
physical graph, fixed overhead and independent heldout sources before launch.
The native physical-stage collector and reference-service estimates retain
separate diagnostic qualifications; reference throughput is not automatically
an identified universal resource supply.

This optional source-checkout package measures **parameters**, not execution
time predictions. Modeling owns `AvailableResources.service_integral`,
`ServiceBinding` and the common graph evaluator. External trajectories describe
resources available to the LLM after other applications and operating-system
effects; background ON/OFF requests and total utilization are not those rates.

There are two deliberately separate inputs:

* Physical active-stage observations can establish a scoped efficiency against
  an independently supplied resource trajectory.
* Existing inclusive completed-action observations can preserve their actual
  within-invocation timing correlations for **diagnostic replay**. They cannot
  establish physical service or predict another resource trajectory by renaming
  wall duration as compute throughput.

## Physical stage efficiency

For measured work `W` and an explicitly established active-service boundary,
profiling asks modeling for the exact external-service integral over that
interval and estimates `efficiency = W / supplied_work`. It does not calculate
an execution duration or add a whole-layer correction. CPU, GPU and payload
transfer use this same contract with their explicit physical units.

```python
from profiling.available_service import freeze_efficiencies, load_efficiency_bindings

frozen = freeze_efficiencies(
    {"stages.json": "<actual SHA256 of the stage source>"},
    available_by_run,                       # run ID -> AvailableResources
    supply_source_files=supply_sources,      # run ID -> {path, sha256}
    context=exact_hardware_backend_shape_context,
    frozen_at_utc="<UTC time after all profile runs>",
    minimum_runs=3,
)
bindings = load_efficiency_bindings(frozen)  # rehashes and reconstructs sources
result = available_for_this_scenario.evaluate(graph, fixed_overhead_parameters, bindings)
```

The graph's fixed submission/resume costs are separately measured parameters.
Completion waiting is expressed by dependencies/completion gates; it must not
also be converted into useful compute work. Other bottlenecks must be controlled
or attributed before one resource efficiency can be established.

The stage JSON uses schema `available-service-stage-observations-1` with an
`observations` array. Every row explicitly contains:

| Category | Required fields |
|---|---|
| Work and binding | `target_parameter`, `supply_parameter`, exact `work`, `work_unit` |
| Boundary | `begin_ns`, `end_ns`, `timing_scope` |
| Resource alignment | `clock_id`, `clock_origin`, `supply_scope` |
| Exact context | `context` |
| Independent run identity | `observation_id`, `run_id`, `process_id`, `seed`, `split`, `started_at_utc`, `ended_at_utc` |
| Evidence | `overheads_excluded`, `other_resources_controlled`, `numerical_verified`, nonempty `evidence` |

Work is an integer or `{numerator, denominator}`; floating-point evidence is
rejected. `work_unit` and `timing_scope` must match one of:

| Work unit | Established boundary |
|---|---|
| `flop` | `device_compute_active` |
| `byte` | `payload_transfer_active` |
| `thread_cpu_ns` | `thread_active_service` |

These labels are evidence requirements, not a claim that the package can infer
physical activity from wall timestamps. The measurement producer must establish
the stated boundary. Inclusive action, stream-envelope and whole-iteration
timings are refused as physical-stage observations.

Each independent supply-source receipt points to JSON with exactly these fields:

```json
{
  "schema": "independent-available-resource-observation-1",
  "available_resources": {"...": "the exact AvailableResources.to_dict() payload"},
  "measurement_scope": "external_resources_available_to_llm",
  "derived_from_target_timing": false,
  "independence_evidence": ["the measurement protocol and independent source references"]
}
```

Profiling verifies the bytes, reconstructs that supplied trajectory, checks
clock/origin/scope and retains the independence protocol receipts. A provenance
tag alone is insufficient. Hashes and protocol receipts are not physical
remeasurement or proof of an honest producer. Independent hardware validation
remains necessary even when parameter evidence gates pass.

Three independent profile runs receive equal weight. Each run's physical
efficiency median is combined with the other runs' medians. This does not
construct a whole execution by adding marginal action durations. Sample counts,
within/between-run CV, every observation and all failed gates are retained.
Unknown work/service, unestablished attribution, missing independence evidence
or failed gates produce `ServiceBinding(efficiency=None)`. Estimates above one
are retained as failed candidates, never clipped. Heldout observations cannot
fit a parameter. Loading with `verify_sources=False` checks the artifact and
binding contract only; it does not re-establish current source bytes.

The hand-computed tests cover 1 flop/ns supply, a 5 ns zero-service interval,
then 2 flop/ns supply. The 0--20 ns integral is exactly 20 flops; a separately
measured 10-flop stage yields efficiency 1/2, and modeling completes that stage
at exactly 20 ns. This is software verification, not hardware calibration.

## Preserve correlated inclusive observations

Independently selected per-action medians need not sum to the whole-invocation
median. `freeze_coherent_ensemble` instead preserves entire real invocation
vectors: all original actions, disjoint preceding gaps and final tail stay
together with their observation identity. It never fits a total-duration offset.

```powershell
python -m profiling.available_service coherent --records "<profile-records.json>" --sha256 "<its exact SHA256>" --samples-per-run-state 2 --output "<new-ensemble.json>"
```

Output is write-once, UTF-8 and LF. This command is offline and starts no native
workload. Its selector uses chronological equal-width ordinal bin centers,
separately per run and offered-state cohort. It does not inspect durations to
choose a representative. Calling the Python API with
`samples_per_run_state=None` explicitly retains every eligible vector.

The compact artifact retains the complete source record-file hash, historical
record digests, cohort counts, selected original observation IDs and exact token
vectors. Every original invocation is checked for complete, ordered intervals.
Full original observations remain in the source. The artifact also shows the
observed whole-call median and the sum of marginal medians **as audit statistics**,
not whole-layer model parameters or a new accuracy test.

```python
from profiling.available_service.coherent import coherent_sample_parameters
from llm_infer_model.performance.serial_actions import evaluate_serial_actions

program, parameters = coherent_sample_parameters(ensemble, exact_sample_id)
replayed = evaluate_serial_actions(program, parameters)
```

The existing modeling serial adapter computes this diagnostic replay. It is
expected to reproduce its source vector; that is not independent acceptance.
Selecting an observed complete vector does not make it a predictor under a
different strategy or dynamic resource trajectory.

The old state-response records contain inclusive host actions and offered
background requests. Consequently the new artifact explicitly returns
`physical_bindings=[]` and lists the missing independent service trajectory,
physical work/boundaries and overhead/attribution evidence. It does not repair
those gaps by interpreting requests or measured foreground slowdown as supply.

Historical source code can change while an old record remains immutable.
`validate_coherent_ensemble` rehashes the record file and reconstructs the selected
vectors; it does **not** claim current native/source re-extraction. Original
source-verification receipts are retained as historical receipts. An archived
source audit is a separate preservation check, not a new hardware experiment.

Run focused checks with:

```powershell
python -m unittest discover -s profiling/available_service/tests -v
```
