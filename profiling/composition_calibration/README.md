# Offline composition calibration

This independent, stdlib-only workflow tests whether a small shared empirical correction improves the original sum/max component recipes for the retained FFN and post-RoPE Attention batch experiments. It does not run hardware, change the simulator core, alter historical artifacts, or establish GPU capacity.

Complete **profile** strategy totals are now training targets (`joint_chain_totals_used_for_calibration: true`). The fitted correction is effective composition overhead; its coefficients do not identify physical context-switch latency. Three global nonnegative coefficients cover declared stage boundaries, the hybrid FFN critical branch duration, and a preload event. Case/path offsets are absent. The feature recipe, coefficients, fitting diagnostics, signed residuals, source row hashes and exact component prices are saved in each frozen model.

## Run and verify

From `Simulator`, with Python 3.10+:

```powershell
python profiling/composition_calibration/run.py --initial profiling/results/20261017-batched-calibration --confirmation profiling/results/20261017-batched-confirmation --output profiling/results/20261018-composition-calibration
python profiling/composition_calibration/verify.py --experiment profiling/results/20261018-composition-calibration --output profiling/results/20261018-composition-calibration/verification.json
python -m unittest discover -s profiling/composition_calibration/tests -v
```

The output directory must be new. The runner first verifies archived source/build identities and deserializes only profile objects from the retained measurements array. It freezes both campaigns' full models and all four grouped-fold models to JSON, records their SHA256 digests in `freeze-ledger.json`, and only then deserializes heldout timings for scoring. Lexical scanning and hashing of original files are provenance operations. The verifier reconstructs the fitted models, folds, raw totals, normalization, original reports and new reports, and rejects changed sources, models, original artifacts, or report content.

These heldout data were inspected historically. The stage ordering makes this pipeline reusable and auditable, but the result is explicitly **retrospective**, with `fresh_blind_validation: false`. Saving a new model today does not mean it was frozen before the original heldout processes ran. Fresh validation requires a new campaign of independent processes and seeds after freezing the implementation and model.

## Evaluation contracts

- All ten primary paths per campaign retain original exact geometry, timing boundary and independent component calibration. Raw samples remain completed-work batch totals; prediction units are per-invocation means derived by the original rounding rule. All original samples and failures remain recorded.
- Original within-process CV, between-process CV, process/numerical coverage and relative-error definitions and thresholds are retained. The original freeze gate refers to the original component model's pre-collection freeze. Corrected predictions update only the error gate. `calibration_target_quality` additionally discloses the variability of profile chain totals newly used to fit coefficients, at those same CV thresholds. `composition_calibration_quality_accepted` also requires these calibration targets to be stable.
- Grouped leave-one-case-out fits coefficients on all repeats/seeds of three geometries and excludes every joint total of the fourth geometry. Independent component prices at the excluded geometry remain available. This tests transfer of shared coefficients to an independently component-profiled geometry; it does not predict an unprofiled shape.
- Initial-profile to confirmation-heldout transfer freezes **both initial component prices and initial coefficients**, with zero refitting. Compatibility requires identical physical binary/generated/native/runtime signatures, protocol, batch units, repetitions and child flags, plus disjoint seeds. Collection/CMake metadata revisions are disclosed separately.
- The strict historical current-source verifier remains unchanged. Initial evidence has a known archived/current collection and CMake metadata difference. This workflow therefore uses a separately labelled archived-evidence reconstruction: exact validation/normalization/recipe ASTs must match, archived snapshots must match the original inventories, and every physical/analysis asset must remain identical. It does not silently describe initial evidence as passing the strict current-source verifier.

`REPORT.md` and `report.json` include old/corrected predictions, features, per-family relative error, signed bias, four same-boundary resident rankings, selection regret and gate failures. The singleton preload path is not compared to resident paths with a different placement boundary. Ranking quality remains diagnostic if latency gates fail.

Four geometries and three shared coefficients provide limited evidence for generalization. Batch means do not certify isolated request jitter, device-kernel overlap, pretrained model accuracy, complete multi-layer inference, dynamic token-state behavior, or capacity. GPU private workspace is retained as `null`; `physical_capacity_validated` remains `false`. No corrected timing is an accepted physical capacity cost.

## Exact IR accounting

An explicit adapter applies the frozen correction to the same measured complete chain after rechecking the calibration artifact, original physical source evidence and exact geometry/protocol. It also reconstructs the fixed tensor IR through the common compiler/runtime and requires the existing binding to match that replay:

```powershell
python profiling/composition_calibration/bind_ir.py --calibration profiling/results/20261018-composition-calibration/confirmation-frozen-model.json --experiment profiling/results/20261017-batched-confirmation --bindings profiling/results/20261017-batch-ir-binding-confirmation --output profiling/results/20261018-composition-ir-accounting-verified
```

Use a new output directory for each run. `independently_replayed_makespan_ns` and the event timeline remain unchanged; `corrected_complete_chain_latency_ns` reports the effective correction separately. The existing runtime has no justified placement/resource semantics for this aggregate residual. The adapter does not create fake tensor work or reinterpret decision commands as switching delays. Arbitrary overlapping bundles cannot simply add this correction. The API `apply(binding, model, applicability_signature=...)` requires the signature to be independently reconstructed from verified measurement evidence; the CLI performs that verification.

The model also has a lower-level data-only interface:

```python
# From a source checkout; rows must follow the retained explicit batch schema.
from profiling.composition_calibration import model

frozen = model.fit(profile_rows)  # rejects heldout/unknown splits
case = frozen["profile_cases"][exact_case_id]
prices_us = frozen["profile_component_prices_us"][exact_case_id]
prediction = model.predict(case, "hybrid/resident", prices_us, frozen)
print(prediction["baseline_ns"], prediction["correction_ns"], prediction["predicted_ns"])
```

`fit` and `predict` do not access files or hardware. Callers must verify source evidence and bind protocol/geometry; the lower-level predictor alone does not enforce measured applicability for new hardware or shapes. This remains an optional research workflow outside the installed public library's default cost provider.

The [18 October evidence report](../../docs/COMPOSITION_CALIBRATION_20261018.md) includes the numerical results and generalization failures. Confirmation median error falls from 13.65% to 6.14%, but leave-one-geometry-out worst error reaches 34.85%, and original quality acceptance remains 0/10. These are retrospective diagnostics, not a calibration-complete claim.
