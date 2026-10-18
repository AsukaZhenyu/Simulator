# Accuracy validation, 2026-10-18

This validation separates execution correctness, numerical correctness, latency
prediction, measurement stability and physical memory feasibility. Passing one
does not establish the others. The default library remains an uncalibrated
analytical simulator; none of these experiments silently replaces its costs.

The completed latency campaigns comprise **378 fresh native processes**: 189
profile processes and 189 heldout processes. Eight additional baseline/budget
trials check tracked capacity. Earlier rejected diagnostic campaigns remain
separate and are not counted as accepted evidence. Software regressions ran
**2,073 tests: 2,065 passed and eight optional tests skipped**; these software
results do not override failed hardware latency gates.

All seven offline protocol verifiers completed successfully: chain, high-level
strategy, overlap, CPU/CUDA workloads and CPU/CUDA tracked capacity. The
high-level verifier reconstructed 56 programs and checked 1,197 heldout output
ports. Original campaign inputs were unchanged. The
[portable result snapshot](data/accuracy-validation-20261018.json) and
[verification receipts](data/accuracy-offline-verification-20261018.json) retain
exact report hashes, failed latency gates and scope limits. Full raw records and
build artifacts remain local; these summaries are not a portable raw archive.

## Implemented coverage

| Question | Executable evidence | Boundary |
|---|---|---|
| Are composition costs observable? | Separate uninstrumented chains and timestamped stage/fork/join diagnostics | Host completed-work intervals; no CUDA kernel-overlap proof |
| Do corrections transfer across cases? | Independent components, profile-only shared coefficients, whole-case coefficient holdout | A case includes shape and partition; other partitions of the same base shape may remain in training, and excluded cases retain their own component measurements |
| Does the real executor follow the compiled strategy? | Actual public ModelPlan graph, operands, serial actions, measured-policy rerun and independent full-port checks | One complete F32 Dense invocation with supported AttentionStages/FFNPartition |
| Is prefetch overlap represented correctly? | Actual asynchronous H2D to distinct next-payload buffers during resident CUDA FFN; one/two streams | Common tensor compiler/runtime, not complete high-level multi-step replay |
| Do layers, KV and routes evolve correctly? | Real two-layer Dense/MoE/mixed activation traces with independent references | Supplied inputs/routes, not pretrained token generation |
| Are allocation lifetimes and budgets enforced? | Actual tracked buffer births/releases/reloads and peak minus/exact/plus one-byte trials | Tracked allocator budget; private CUDA scratch and physical-fit proof remain unknown |

The [suite guide](../profiling/accuracy_suite/README.md) links build commands,
interfaces, fixed protocols and software tests. Every campaign uses an exclusive
measurement lock. Collection is serial and does not modify user applications,
power settings or device clocks. Raw samples, failures and rejected protocols
are retained locally rather than replaced.

## Numerical precision defect found and repaired

The initial complete-strategy campaign completed 42 profile processes but
rejected all four H128/q16 plans at the independent Attention output check.
The original backend's custom MMF route used explicit TF32 instructions despite
the graph's `GGML_PREC_F32` request and `NVIDIA_TF32_OVERRIDE=0`. API flags alone
therefore did not establish arithmetic precision. The same failure affected
eligible FFN intermediate matmuls.

A reusable, pinned-version [private runtime builder](../profiling/native/highlevel_accuracy/build_f32_runtime.py)
copies the source and disables that route for a strict F32 request on regular
`MUL_MAT`. It recompiles one router translation unit and relinks 131 unchanged
supplied Release objects. This is an explicitly recorded partial rebuild, not
a clean rebuild or an observation of resolved operating-system loader paths.
The original source/build/DLLs remain intact. `MUL_MAT_ID` is outside this patch.

All four previously failing plans passed untimed independent checks afterward,
with maximum absolute error `8.94e-7` at the unchanged absolute/relative tolerance
`5e-5`. A new complete campaign collected costs and heldout evidence separately.
The rejected campaign, its exact source archive and every raw record remain local.

## Fixed acceptance rules

Profile sources and heldout totals must satisfy within-process CV <= 15% and
between-process median CV <= 20%. Pooled-median heldout prediction error must be
<= 20%, with at least three independent processes per split, disjoint numerical
seeds, complete numerical/process coverage and models frozen before heldout.
The workload evaluator additionally reports individual heldout-process errors.
No observations are filtered and no threshold is relaxed after inspection.

Ranking and selection regret require the same logical weights, external inputs,
KV, initial/final locations and matched process cohorts. Intervals resample
processes and then invocation samples within selected processes for the chain
study; high-level ranking intervals resample independent process medians.
Missing paths block survivor-subset selection;
unstable or overlapping comparisons remain descriptive.

## Completed native-chain experiment

The 138-process campaign covered 23 declared shape/partition cases: 14 FFN and
9 Attention cases. All native numerical, port, timing-contract, affinity and
source-binding checks passed. The models below were frozen before the first
heldout process. Full shared-coefficient and leave-one-case-out rows cover the
same 60 primary GPU/hybrid paths; the other rows cover all 143 declared paths.

| Model | Paths | Median error | Worst error | Error <= 20% | All latency gates passed |
|---|---:|---:|---:|---:|---:|
| Independent components, all paths | 143 | 6.93% | 52.26% | 102/143 | 7/143 |
| In-chain stages/gaps/fork/join | 143 | 3.94% | 24.27% | 141/143 | 0/143 |
| Independent components, primary paths | 60 | 10.96% | 52.26% | 35/60 | 0/60 |
| Shared coefficients, primary paths | 60 | 6.19% | 35.34% | 53/60 | 0/60 |
| Shared coefficients, whole-case holdout | 60 | 6.51% | 35.94% | 52/60 | 0/60 |

The in-chain model improves errors but every path fails its source stability
gate, including variable short host gaps. This does not erase that failed gate
or replace it with the lower total error. No general calibration claim follows.
The [chain results note](ACCURACY_CHAIN_RESULTS_20261018.md) contains source and
target gates, instrumentation limits, comparisons and exact report hashes.

Descriptive baseline rankings select the observed winner in 11/23 cases. Median
selection regret is 0.72%, but worst regret is 95.09%. No complete ranking passes
all latency-quality gates. A low average timing error therefore does not by
itself validate a strategy selector.

## Confirmed limits

The [high-level timing diagnosis](ACCURACY_HIGHLEVEL_DIAGNOSIS_20261018.md)
records substantial process-dependent source-price variation. Fourteen complete
strategy cases all passed execution/numerical checks but none passed latency
acceptance. Worst pooled-median error was 152.95%; the separate worst individual
process error was 161.51%. These are failure evidence, not accepted calibration.

The [multi-layer workload results](../profiling/workload_accuracy/RESULTS.md)
passed numerical, persistent-KV, causal expert-lifetime and public-model audits
in all 48 CPU/CUDA processes. CPU worst per-process error was 9.356%, CUDA
22.103%; both backends failed the original statistical acceptance gates. An
independent audit rejected 48 deliberate metadata/payload early-allocation
corruptions. Byte-exact tracked-budget trials passed on both backends.

These experiments do not establish quantized arithmetic accuracy, batch>1,
arbitrary parallel high-level transformations, full compiled multi-round native
replay, unified-memory hardware, SSD production paths, pretrained-model
generation or complete physical VRAM capacity. Unsupported scopes remain
explicit rather than inheriting an accuracy claim from smaller tests.

## Completed copy/compute overlap experiment

The 108-process experiment covered 18 geometries/payloads and 72 paths. Every
native numerical, copied-byte and distinct-target check passed. Overall median
prediction error was 2.50%, worst error 47.66%; 70/72 paths met the error threshold
and 15/72 passed all original gates. Source instability is still the dominant
reason for rejection.

For the 18 two-flow overlap cases, the common compiler/runtime's one-shared-link
model `max(A+B, compute)` had median error 2.50% and four fully accepted cases.
The ideal independent-link alternative `max(A, B, compute)` had median error
49.15% and no fully accepted cases under the same gates. This is evidence for
retaining shared transfer-resource constraints in this bounded experiment; it
does not prove simultaneous CUDA kernels or a general contention model.

The [overlap results note](ACCURACY_OVERLAP_RESULTS_20261018.md) gives all four
path families, payload sizes, failed gates, the alternative model comparison and
exact source report hash. The next-payload buffers are copied but not consumed
by a complete subsequent decoder step, so this is not a production-prefetch
speedup result or complete high-level multi-step strategy replay.
