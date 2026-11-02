# Independent available-service validation

`validation.py` is an offline evaluator. It starts no native workloads and fits
no parameter. All task timing is computed by modeling's
`evaluate_available_resources`; profiling supplies source checks and error
statistics. The native collector must establish the measurement boundaries,
independent supply protocol, numerical correctness and process lifetimes.

## Freeze, bind, launch, compare

```python
from profiling.available_service.validation import make_barrier, evaluate_available_service

# First save the physical-efficiency freezer's complete output to frozen.json.
barrier = make_barrier(
    "frozen.json",
    graph=graph,
    fixed_parameters=fixed_parameters,
    signature_id="exact-work-and-backend-signature",
    created_at_utc="2026-10-22T02:00:00+00:00",
    protocol={
        "minimum_runs": 3,
        "minimum_samples_per_run": 10,
        "median_relative_error_max": 0.20,
        "p90_relative_error_max": 0.20,
    },
)
# Persist barrier with an exclusive write BEFORE starting heldout processes.

# Collect new processes/seeds after that barrier, then compare without refitting.
result = evaluate_available_service(
    "frozen.json",
    heldout_source_files,   # {absolute reference JSON path: SHA256}
    available_by_run,      # {run ID: AvailableResources}
    supply_source_files,   # {run ID: {"path": absolute path, "sha256": SHA256}}
    graph,
    fixed_parameters,
    barrier=barrier,
)
```

The barrier pins the actual saved frozen-file bytes, frozen semantic artifact,
graph, fixed-parameter bundle, signature, context, profile identities and error
gates. Changing them after the barrier is rejected. Its clock chronology is
collector-declared UTC metadata, not independent proof that a collector actually
wrote the barrier before launching a process. The caller's orchestration must
enforce that order and retain its receipts.

Heldout run IDs, PIDs and seeds must be disjoint from profile identities; distinct
heldout runs also need distinct PIDs and seeds. Invocation intervals within one
run may not overlap. Concurrent work must be represented within the declared
joint graph rather than evaluated as independent overlapping references.

## Heldout reference source

Each source JSON has exactly:

```json
{"schema": "available-service-heldout-observations-1", "observations": []}
```

Every observation has exactly these fields:

| Fields | Type and meaning |
|---|---|
| `observation_id`, `run_id` | Nonempty unique observation ID and stable run ID |
| `process_id`, `seed` | Positive integer native PID and integer numeric seed |
| `split` | Literal `heldout` |
| `begin_ns`, `end_ns` | Integer timestamps on the external clock, `0 <= begin < end` |
| `clock_id`, `clock_origin` | Exact match to the run's supplied `AvailableResources` |
| `context` | Exact frozen physical hardware/backend/shape context object |
| `graph_sha256` | Canonical digest of the barrier-bound `Graph.to_dict()` |
| `signature_id` | Exact barrier-bound work/backend signature |
| `started_at_utc`, `ended_at_utc` | UTC run lifetime; start must be strictly after barrier |
| `reference_timing_scope` | Literal `whole_execution_completed` |
| `numerical_receipt` | `{"path": "absolute JSON path", "sha256": "actual SHA256"}` |
| `state` | Diagnostic grouping object or explicit `null`; never a rate predictor |
| `evidence` | Nonempty array of measurement/protocol source references |

The numerical receipt must contain a top-level boolean `passed`. Its bytes are
rehashed. Additional native verification details are retained in that file;
offline validation does not execute the numerical oracle again. A false result
retains the observation but prevents primary acceptance.

Use absolute source paths to avoid ambiguous working-directory resolution.
The runner hashes and parses source bytes; a JSON `source_verified` assertion
cannot replace those checks.

## Separate supply sources

Each supply receipt binds the same schema used by physical-efficiency profiling:

```json
{
  "schema": "independent-available-resource-observation-1",
  "available_resources": {"...": "exact AvailableResources.to_dict()"},
  "measurement_scope": "external_resources_available_to_llm",
  "derived_from_target_timing": false,
  "independence_evidence": ["independent producer and measurement protocol"]
}
```

The supplied object must reconstruct exactly and use the reference's clock and
origin. A primary forecast requires `source_kind="independent_measurement"`,
nonempty independence evidence and no derivation from target timing. A worker
request, utilization percentage or target slowdown cannot be relabeled as
available service.

The model receives only each observed invocation's start timestamp and the
external supply trajectory. It predicts every internal task boundary itself.
Reference end times are used only for error metrics and diagnostic crossing
classification. Results explicitly state `conditional_on_observed_start=true`;
they do not establish rolling prediction of all future invocation launches.

`by_supply_crossing` detects actual changes in supplied rates/capacities during
the measured call; redundant updates to identical values do not count as a
change. `state` labels produce additional error groups without feeding the model.

## Primary versus explicit diagnostic comparisons

The primary result uses qualified frozen efficiencies only. Missing/unknown
parameters or supply leave a prediction unknown and reduce reported coverage.
The runner never fills that hole with a measured duration or candidate value.

`diagnostic_candidates=True` enables a separate comparison using only
profile-frozen candidate efficiencies and explicitly permitted observed supply
replay. Invalid candidates outside [0,1] remain unknown. Diagnostic predictions
have separate fields and summaries; `accepted=false` and
`hardware_accuracy_qualified=false` remain explicit for that comparison. The
primary result remains unchanged even if the diagnostic has a small error.

All source observations are retained. Parameter issues outside the predeclared
graph are reported separately rather than silently disappearing; only targets
actually demanded by that graph are used to evaluate its parameter coverage.

The current efficiency freezer does not qualify separate fixed `ns` submission
or resume costs. Graphs using them can be compared descriptively, but the runner
reports `fixed_overhead_qualification_unestablished` and does not accept them as
hardware-qualified. An empty fixed bundle for an isolated physical-stage graph
avoids introducing an unmeasured fixed-overhead assumption. This does not prove
that a complete LLM has no overhead.

## Outputs and checks

Outputs include process-balanced median absolute relative error, pooled P90,
worst error, per-run counts, state and crossing groups, complete prediction
coverage, per-observation source references, profile and heldout evidence
failures, fixed-parameter qualification and a digest of the evaluation. All
thresholds are predeclared in the barrier. Dynamic observed-duration CV is not
misused as a stability gate across genuinely different supplied states.

Source hashes and declared protocol receipts are evidence integrity checks, not
physical remeasurement, authenticity verification or arbitrary-policy accuracy.

```text
python -B -m unittest discover -s profiling/available_service/tests -p test_validation.py -v
```

These tests use synthetic fixtures. They verify source/chronology guards,
candidate isolation, unknown coverage, clock matching, actual-change crossing
classification and delegation to modeling; they are not hardware results.
