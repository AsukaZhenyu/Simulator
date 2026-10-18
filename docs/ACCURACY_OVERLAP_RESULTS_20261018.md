# Copy/compute overlap accuracy results — 2026-10-18

The fresh overlap campaign completed all 108 processes without a recorded failed row. Of 72 case/path predictions, 70 have heldout median error <= 20%, but only 15 pass every original quality gate. The main remaining limitation is timing stability. The results support a shared-link service model over an independent-links ideal in this bounded experiment; they do not prove physical GPU kernel overlap or a general contention law.

This note aggregates the existing [report.json](../profiling/overlap_accuracy/results/formal-v1/report.json) using Python's standard library. No model was fitted, no hardware was run, and no frozen protocol or native source was changed for this analysis.

## Experiment and scope

There are 18 FFN cases: H256/512/1024, I=2H, q1/q32, and 1/16/64 MiB payloads. Every case has three independent profile processes and three independent heldout processes: 54 + 54 = 108 processes. Each case exercises four completed-work paths, giving 72 case/path results. Profile isolated costs and the compiled serial/overlap programs were frozen before heldout collection. All 72 results pass the declared-process, minimum-process, common-compiler/runtime, and freeze gates; no result has an `unknown_reason`.

The current FFN computes with resident current weights while asynchronous H2D copies write distinct next-payload buffers. Copy destinations do not overwrite current compute inputs. Payload checks and numerical validation are outside the timed totals. This is an FFN plus future-payload experiment; it does not establish an end-to-end sequence of future model layers.

Let A and B denote the independently profiled copy costs and C the independently profiled resident FFN cost. The fixed predictions are:

| Native path | Fixed cost hypothesis |
| --- | --- |
| `serial/one_copy_compute` | A + C |
| `overlap/one_copy_compute` | max(A, C) |
| `serial/two_copy_compute` | A + B + C |
| `overlap/two_copy_compute` | max(A + B, C), with a shared copy link |

Both paths are compiled through the same fixed tensor IR/compiler/runtime using profile costs. The report records `common_fixed_tensor_compiler_runtime_exercised=true` and `highlevel_ModelPlan_coverage=false`: this is a real fixed-program bridge, not coverage of the general high-level Model/Workload/ModelPlan compiler.

## Prediction accuracy and complete acceptance

Each error below compares a frozen prediction against the pooled heldout median of uninstrumented, synchronized completed-work invocations. Percentages aggregate one error per case/path. Acceptance requires within-process CV <= 0.15, CV of process medians <= 0.20, error <= 0.20, and every coverage/freeze gate.

| Path | Results | Median error | Worst error | Error <= 20% | Fully accepted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Serial, one copy | 18 | 3.523647% | 7.699208% | 18 / 18 | 4 / 18 |
| Overlap, one copy | 18 | 0.992034% | 17.806604% | 18 / 18 | 3 / 18 |
| Serial, two copies | 18 | 2.936090% | 9.385209% | 18 / 18 | 4 / 18 |
| Overlap, two copies | 18 | 2.501843% | 47.662711% | 16 / 18 | 4 / 18 |
| All paths | 72 | 2.501843% | 47.662711% | 70 / 72 | 15 / 72 |

The two error failures are two-copy overlap at `ffn-h1024-i2048-q32-copy1MiB` (20.705347%; prediction 487,900 ns versus observation 615,300 ns) and `ffn-h1024-i2048-q32-copy16MiB` (47.662711%; prediction 2,677,000 ns versus observation 5,114,900 ns). Both also fail within-process stability; the latter fails between-process stability as well. They remain failures.

| Original gate failure | Serial one | Overlap one | Serial two | Overlap two | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Within-process CV | 14 | 15 | 14 | 14 | 57 / 72 |
| Between-process CV | 0 | 1 | 0 | 1 | 2 / 72 |
| Heldout relative error | 0 | 0 | 0 | 2 | 2 / 72 |

At least one profile source fails within-process CV in 56 / 72 results; a heldout path fails it in 19 / 72. Their union is 57 / 72. No profile source group fails between-process CV; two heldout path groups fail it. The counts overlap and must not be added as distinct failures. Low median error does not waive a source or target CV gate.

## Two copy flows: shared link versus independent-links ideal

For the 18 two-copy overlap cases, the report also evaluates max(A, B, C), an ideal in which the two copies have independent links. It uses the same frozen isolated costs and introduces no heldout fit.

| Hypothesis | Median error | Worst error | Error <= 20% |
| --- | ---: | ---: | ---: |
| Shared link: max(A + B, C) | 2.501843% | 47.662711% | 16 / 18 |
| Independent-links ideal: max(A, B, C) | 49.152362% | 73.700366% | 5 / 18 |

The shared-link hypothesis has lower error in 12 cases and ties in six. All six 1 MiB cases tie because the predicted compute cost dominates both hypotheses. The ideal's median signed error is -49.152362%, indicating substantial underprediction across this set. Retaining the original coverage/stability gates and replacing only the error test with the ideal's error gives 0 / 18 accepted ideal predictions; this acceptance count is derived, not a separate report field.

| Payload per flow | Cases | Shared-link median / worst error | Ideal median / worst error | Shared / ideal errors <= 20% |
| --- | ---: | --- | --- | --- |
| 1 MiB | 6 | 9.609348% / 20.705347% | 9.609348% / 20.705347% | 5 / 5 |
| 16 MiB | 6 | 1.182567% / 47.662711% | 49.414564% / 73.700366% | 5 / 0 |
| 64 MiB | 6 | 1.196906% / 3.279474% | 49.386112% / 51.192893% | 6 / 0 |

Observed serial-total / overlap-total ratios are greater than one in 18 / 18 one-copy cases and 17 / 18 two-copy cases. Their medians are 1.284917 and 1.203238, respectively. The worst two-copy ratio is 0.649925, so overlap is not uniformly faster. These are descriptive comparisons of the observed medians, not a claim that every comparison passes the stability gates.

## Diagnostic and coverage limits

Host intervals and CUDA copy event durations do not establish that GPU compute kernels physically overlap a copy. The report explicitly records `kernel_overlap_proved=false` and `copy_events_prove_kernel_overlap=false`. A profiler kernel timeline would be needed for that claim. Host submission and completion intervals are not a measured operating-system context-switch cost.

Instrumentation controls and diagnostic traces are separate from the uninstrumented completed-work target. The report records `instrumentation_overhead_subtracted=false`; no timestamp cost is subtracted to manufacture a target. This summary uses the small aggregate report, not a fresh examination of diagnostic raw traces. The experiment tests the declared sum/max laws and one shared-link assumption; it does not validate arbitrary link contention, general high-level planning, private CUDA allocation accounting, or physical memory capacity (`physical_capacity_verified=false`). Standalone verifier receipts are reported separately by the comprehensive campaign report.

## Exact report input

- Path: `profiling/overlap_accuracy/results/formal-v1/report.json`.
- Schema: `overlap-accuracy-report-0.1`.
- SHA256: `4c1f49753fa64ab5c763ae6212e7b9307ae07e72023b09f290871a26fdd687ab`.
