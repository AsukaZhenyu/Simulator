# Accuracy validation suite

This source-checkout workflow separates numerical/strategy fidelity, cost-model
error, statistical acceptance and physical memory feasibility. The core library
still works without profiles, CUDA, model weights or third-party dependencies.
The optional high-level bridge needs NumPy:

```powershell
python -m pip install ".[profiling]"
```

Four complementary native workflows share `profiling/.measurement.lock`:

| Workflow | Actual execution and comparison |
|---|---|
| [Chain calibration](../chain_calibration/README.md) | CPU/GPU serial, joint graph and parallel FFN/Attention paths; stage spans, host gaps, fork/join, three separately frozen model families and whole-case coefficient holdout |
| [High-level strategy bridge](../highlevel_accuracy/README.md) | Actual Python ModelPlan graph/actions, complete F32 Dense macros and supported Attention/FFN transforms, exact measured leaf costs, independent full-port numerical checks |
| [Workload and memory](../workload_accuracy/README.md) | Real layers, evolving KV and changing/revisited experts across input-trace rounds; independent numerical reference, tracked backend allocation lifetimes and explicit budget boundaries |
| [Copy/compute overlap](../overlap_accuracy/README.md) | Real asynchronous H2D prefetch to separate next-payload buffers during resident CUDA FFN; one/two copy streams, shared-link sum/max predictions through the common tensor compiler/runtime |

The high-level compiler bridge and multi-round native workload evaluator have
different coverage. The latter audits matching public workload geometry,
topology, KV and trace visibility; it is not automatically proof that the entire
compiled multi-round strategy has been numerically replayed in the native
executor. Each report retains its explicit execution/placement boundary.

## Common acceptance and timing rules

All samples and failures are retained. Each split has at least three independent
processes and disjoint numerical seeds. Models and all grouped shape folds are
written and hashed before the first heldout process. Complete heldout timings
never fit their predictor. Sources, executables and relevant runtime libraries
are checked at process boundaries. Collection uses fixed protocols and does not
alter user applications, power policies or device clocks.

Original thresholds remain 15% within-process CV, 20% between-process median CV
and 20% relative prediction error. `evaluate.py` provides a pure stdlib common
evidence gate. It also reports strategy-selection regret at a common external
completion boundary, with intervals obtained by resampling independent process
medians. It does not treat individual timings as independent processes or batch
means as isolated-request tail latencies.

`highlevel_rankings.py` additionally compares complete compiled Dense strategies
only after checking logical weight contents, external hidden/position/KV data,
initial/final locations and matched numerical/process cohorts. It reconstructs
the same parent matrices from immutable FFN partitions. Missing or failed paths
prevent choosing a winner from a surviving subset; unstable rankings remain
descriptive. A fixed bootstrap seed exposes overlapping process-median intervals.

`trace_model.py` reconstructs explicitly profiled main/worker stage durations,
same-thread gaps, launch start offsets and join completion costs. A join waits
for both branches; the already elapsed wait is not charged a second time.
Diagnostics are host completed-work intervals, not CUDA kernel activity.
Instrumentation costs are separately observed and never silently subtracted.

`shape_models.py` reuses the frozen three-feature implementation for newly
collected single-invocation profiles, with an explicit repetitions=1 normalization
view. Full and entire-case holdout models freeze before collection of heldout
processes. Excluded geometries keep their independent component measurements:
this tests shared-coefficient transfer, not prediction of an unprofiled shape.
The excluded case includes its shape and partition ratio. Other partition cases
at the same underlying model shape may remain in coefficient training; this is
not leave-all-partitions-of-one-base-shape-out validation.

Memory ledgers check actual tracked requested/reserved buffers and allocation,
release and reload lifetimes. A byte-exact tracked-budget test is meaningful for
that ledger. Unknown private CUDA scratch and untracked process memory remain
unknown; neither free-device snapshots nor passing a tracked budget establish a
physical capacity bound for arbitrary execution.

## Tests and reports

```powershell
python -B -m unittest discover -s profiling/accuracy_suite/tests -v
python -B -m unittest discover -s profiling/chain_calibration/tests -v
python -B -m unittest discover -s profiling/tests -p test_highlevel_accuracy.py -v
python -B -m unittest discover -s profiling/workload_accuracy/tests -v
python -B -m unittest discover -s profiling/overlap_accuracy/tests -v
```

Native builds and hardware campaigns are explicit separate commands documented
in each workflow. Run them serially after software tests/builds finish; preserve
the entire output tree and its frozen source/runtime bindings. New output paths
are required for repeated or repaired campaigns. The comprehensive report
assembler records prediction errors, failed gates, numerical and memory scope,
regressions and unresolved coverage separately.

The report assembler does not replace each workflow's offline verifier. Run
those verifiers first and retain their commands, outputs and input hashes. The
high-level verifier is `verify_highlevel.py --experiment DIRECTORY`; the
workload's independent v2 expert-lifetime audit also checks both metadata and
payload allocations against route visibility barriers. Strict current-byte
verification is different from a portable archived-software replay.

The capacity verifier is read-only and never launches native work:

```powershell
python -B -m profiling.accuracy_suite.verify_capacity profiling/workload_accuracy/capacity-v2-cpu-20261018
python -B -m profiling.accuracy_suite.verify_capacity profiling/workload_accuracy/capacity-v2-cuda-20261018
```

It reconstructs the baseline and exact/plus-or-minus-one-byte budget cases,
commands, expected pre-allocation rejection, numerical and shape checks, known
F32 payload bytes, releases/reloads, both expert metadata and payload route
barriers, and the exact saved capacity report. Current build/source/runtime
hashes must match the saved binding. Run outside hardware timing campaigns
because this includes runtime/library hashing. Optional `--output NEW_RECEIPT`
creates a receipt without overwriting existing files. Passing verifies only the
tracked allocator budget; physical capacity, process peak and private CUDA
workspace remain unknown. The frozen cases, raw records and reports are never
rewritten.
