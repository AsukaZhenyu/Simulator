# Composition calibration evidence (18 October 2026)

A shared empirical composition correction reduces systematic underprediction on retained FFN and post-RoPE Attention experiments. This is retrospective evidence at four geometries. It does not establish physically identified switching costs, general inference accuracy, or accepted hardware timing.

## What changed

The original component estimate assumes that independently timed stages can be combined with sum/max recipes. The new model accounts for a complete chain as:

```text
estimated_chain_us = original_component_recipe_us
                   + a * critical_path_boundary_proxy
                   + b * hybrid_ffn_critical_branch_us
                   + c * preload_event
```

The three nonnegative coefficients are shared across every case and path. No per-case or per-path offset is fitted. The boundary proxy is 4 for five-stage GPU paths, 5 for hybrid resident paths, and 8 for FFN preload. It counts serial boundaries on the declared critical path, not OS context switches or every graph edge. The branch feature is the maximum of the two independently timed FFN branch sums; it is zero for other paths. The preload feature is one only for the three-weight FFN preload path.

Training uses each profile process's complete-chain median minus its own component-recipe estimate. A relative-error weighted nonnegative least-squares fit retains all signed residual targets, including any negative observations. Only profile data train coefficients. Historical heldout results were already inspected, so freezing models before the current offline scoring does not make this a fresh blind experiment.

## Retrospective results

Each campaign has 24 independent processes (three profile and three heldout repeats at four geometries), 31 retained samples per metric and 16 completed invocations per batch. The ten primary paths include GPU/hybrid resident FFN, hybrid FFN preload and GPU/hybrid resident Attention. Timings are per-invocation means within repeated completed-work batches, not isolated-request latency.

| Evaluation | Baseline median error | Corrected median error | Corrected worst error | Paths within 20% | Original quality accepted |
|---|---:|---:|---:|---:|---:|
| Initial | 11.89% | 4.77% | 10.72% | 10/10 | 0/10 |
| Confirmation | 13.65% | 6.14% | 7.98% | 10/10 | 0/10 |
| Initial, leave one geometry out | 11.89% | 4.26% | 30.17% | 9/10 | 0/10 |
| Confirmation, leave one geometry out | 13.65% | 8.18% | 34.85% | 8/10 | 0/10 |
| Initial model transferred to confirmation, zero refit | 8.11% | 2.06% | 14.35% | 10/10 | 0/10 |

The transfer row freezes both the initial component prices and the initial coefficients. Its baseline therefore differs from the confirmation campaign's own-profile baseline. Both campaigns executed the same physical binary/generated/native/runtime identities and explicit protocol, with disjoint seeds; metadata-only source differences are disclosed by the audit. This transfer is also retrospective.

Leave-one-geometry-out excludes all complete-chain targets, strategies and process repeats for that geometry from fitting the shared coefficients. It still uses independent component profiles at the excluded geometry. The 34.85% worst error is a material generalization limit, not a passed calibration.

Every mode retains four correct same-boundary resident rankings and zero measured selection regret. The old model already selected those four winners, so this experiment demonstrates improved latency estimates rather than better strategy selection.

## Confirmation paths

| Case | Strategy | Baseline error | Corrected error |
|---|---|---:|---:|
| ffn-h256-i512-q1-cut128 | gpu/resident | 8.44% | 0.00% |
| ffn-h256-i512-q1-cut128 | hybrid/resident | 23.72% | 7.86% |
| ffn-h256-i512-q1-cut128 | hybrid/preload | 27.63% | 0.88% |
| ffn-h512-i1024-q32-cut256 | gpu/resident | 14.34% | 7.98% |
| ffn-h512-i1024-q32-cut256 | hybrid/resident | 19.61% | 1.03% |
| ffn-h512-i1024-q32-cut256 | hybrid/preload | 16.17% | 1.12% |
| attention-h256-q32-hist0 | gpu/resident | 10.94% | 7.83% |
| attention-h256-q32-hist0 | hybrid/resident | 11.69% | 6.52% |
| attention-h256-q1-hist128 | gpu/resident | 9.17% | 5.75% |
| attention-h256-q1-hist128 | hybrid/resident | 12.97% | 7.44% |

## Parameters and their limits

| Effective coefficient | Initial | Confirmation | Unit |
|---|---:|---:|---|
| composition_boundaries | 16.052581 | 12.361055 | us/boundary |
| hybrid_ffn_critical_branch_us | 0.198694 | 0.198889 | dimensionless |
| preload_event | 36.390882 | 91.637668 | us/preload_event |

The preload coefficient changes substantially between campaigns. All three features are confounded with strategy structure and only four geometries are available. Neither full design-matrix rank nor lower heldout error identifies a physical context-switch constant. These coefficients must not become platform defaults.

## Integration and acceptance

The optional [exact IR accounting adapter](../profiling/composition_calibration/bind_ir.py) verifies the calibration model, original source evidence and exact physical signature, then reconstructs the fixed tensor IR through the existing compiler/runtime. It preserves `independently_replayed_makespan_ns` and the original event timeline. A separate `corrected_complete_chain_latency_ns` includes the effective correction. It does not insert invented tensor work or controller delays, and cannot be composed arbitrarily across overlapping bundles.

All original CV, process coverage, numerical validity and error thresholds remain unchanged. Newly used profile-chain targets also receive separate stability diagnostics. Accepted timing remains 0/10 because original component-level variability gates fail; lower relative error does not override them. Unknown CUDA workspace remains null and physical capacity validation remains false. No core runtime or default analytical estimate was replaced.

The new model/runner/adapter tests pass 26 synthetic tests. Neighboring original execution-calibration and batch contracts pass 54 and 15 tests. The offline verifier reconstructs every full/grouped model and result, confirms model freezing before offline heldout deserialization, and checks that original artifacts were unchanged. A separate adapter run accounts for all ten previously bound paths after reconstructing their fixed IR replay.

## Reproduce and inspect

- [Workflow and API instructions](../profiling/composition_calibration/README.md).
- [Portable aggregate results](data/composition-calibration-20261018.json), including predictions, errors, gates, rankings, coefficients and local-source artifact SHA256 values. These aggregates are not raw timing evidence and cannot independently re-audit native measurements.
- Local full report: `profiling/results/20261018-composition-calibration/REPORT.md` and `report.json`.
- Local source-verification result: `profiling/results/20261018-composition-calibration/verification.json`.
- Local exact IR accounting: `profiling/results/20261018-composition-ir-accounting-verified/REPORT.md`.

Raw timings, native build identities and source snapshots remain local retained evidence under `profiling/results/`; they are not required by library installation and are not included in this public aggregate. Source-report SHA256: `d0beac4cc57709a3ee4b515fcfd8b5982ac2f3ef0fb47a23c0ae5eb84db10eed`.

The next measurement should timestamp stages within real serial and hybrid chains, freezing this model before new independent processes/seeds. This would separate chain-order effects, branch start skew and concurrent slowdown, and determine whether a schedulable completion service is justified. Additional shape families and stable source measurements are required before applying the model to general Dense/MoE inference.
