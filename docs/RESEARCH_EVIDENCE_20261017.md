# Research evidence report — 17 October 2026

This report separates **strategy semantics**, **measured-component replay**, **analytical design exploration**, and **hardware performance acceptance**. They answer different questions and their counts must not be combined into a single accuracy score.

The current evidence supports an installable simulator and a finite set of independently checked strategy mechanisms. It does **not** establish a fully calibrated Dense/MoE inference simulator, complete reproduction of published systems, globally optimal strategies, or a completed paper contribution.

## Evidence at a glance

| Question | Current evidence | What is established | What is not established |
|---|---|---|---|
| Does a declared strategy survive Python → IR → compiler → runtime? | 9/9 independent reference groups; 13/13 acceptance tests | Handwritten operand, byte, dependency, transfer, lifetime, state and visibility checks for finite cases | Exhaustive strategy coverage, numerical tensor correctness, physical performance accuracy |
| Can measured components be bound to the fixed tensor IR? | Historical 14/14 resident paths; separate repeated-execution confirmation 10/10 paths, including two preload paths | Exact fixed-IR/action/runtime binding with source and cost provenance; denominators remain separate | High-level `ModelPlan` semantic-transform calibration or full LLM calibration |
| Are those measured paths accepted as accurate performance predictions? | Historical 0/14 accepted; repeated-execution confirmation 0/10 accepted | Correct compiler replay preserves the source quality failures | Accepted latency/capacity calibration or measured speedup |
| Can the library support controlled hardware/strategy exploration? | 220 unique analytical attempts: 188 complete, 32 fail explicitly | Reusable finite-candidate, fixed-versus-reselected analysis over declared hardware inputs | Real-machine validation, a new search algorithm, global optimality or economic return |
| Has the policy dead-end diagnostic been rechecked after its reporting fix? | Separate 40-attempt smoke: 26 complete, 14 explicit policy dead-ends | Successful latencies, transfer bytes, peaks and completion sets match the same frozen-campaign points; full-220 artifacts remain unchanged | The 14 dead-ends do not prove global hardware infeasibility |
| Does batching/steady execution resolve measurement instability? | Confirmation: 24/24 native processes, 10 primary predictions, median error 13.65%, 0/10 accepted | All ten fail within-process CV; two also fail held-out error. Initial overlapping collection remains separate | No accepted calibration or causal claim that batching/interference explains differences |

Generated evidence is retained locally under `build/research-evidence/` and `profiling/results/`; those directories are not installed library dependencies or a claim that all raw data is publicly archived. The scripts and protocols are reusable. Every rerun must use a new output directory.

## 1. Independent strategy semantics

Runner: [experiments/strategy_fidelity.py](../experiments/strategy_fidelity.py). Protocol and limits: [STRATEGY_FIDELITY.md](STRATEGY_FIDELITY.md). Acceptance tests: [tests/test_strategy_fidelity.py](../tests/test_strategy_fidelity.py).

Current post-diagnostic evidence: [strategy-fidelity-current/REPORT.md](../build/research-evidence/strategy-fidelity-current/REPORT.md), [report.json](../build/research-evidence/strategy-fidelity-current/report.json), and the source/import manifests beside them. The previous source snapshot remains separately retained as [strategy-fidelity-final/report.json](../build/research-evidence/strategy-fidelity-final/report.json); its counts are not added to the current run.

The suite passed **nine reference groups**. These are not nine hardware measurements. A separate set of **13 acceptance tests** checks both the reference cases and whether the evidence checker rejects incorrect operands, foreign imports, changed source bindings and unsafe lifetime changes. The two counts describe different units and should not be added.

| Reference group | Independent expected property |
|---|---|
| Attention stages | Exact operand identities and F32 sizes for Q/K/V, scores, probabilities, context and appended KV; charged cross-device Q/K and probability copies |
| FFN partition | Exact interval coverage and weight-byte conservation; one normalization result feeding the branches; explicit partial SUM and residual inputs; charged CPU-partial return |
| Transform composition | Attention and FFN transforms in one graph, charged startup copies, dependency ordering and a shared cache budget |
| Multilayer phases and persistent KV | Two layers across prefill and two decode rounds share eight stable KV blocks; history grows consistently and cross-device phase changes move existing state |
| Placement, prefetch and cache | Scoped placement and cross-layer prefetch coexist, all transfers retain supplied nonzero costs, and in-flight/resident payloads respect declared cache accounting |
| MoE causality | Paired traces with different hidden future routes produce **23 identical public observation/command decisions** before the next route is revealed |
| Rejection behavior | Nine deliberate invalid variants reject partition gaps/overlaps, wrong merge structure, missing exact costs/transfers, unmatched selectors and unsupported device identities |
| Capacity admission | A direct 32-byte weight copy into one byte of VRAM rejects with `CAPACITY_EXCEEDED`; an unfinished GPU policy that cannot progress reports `no_progress` / `policy_dead_end`, without silently falling back to CPU |
| Device availability | A 10,000 ns initial CPU blackout adds exactly 10,000 ns to the matching CPU-only reference |

The expected geometry, algebra and operand identities are handwritten rather than generated by the compiler being checked. Successful traces additionally satisfy producer-before-consumer timing and protection of in-flight compute inputs in the executing storage domain. All service times in this suite are deliberately synthetic exact contracts.

The current retained run verified source integrity and bound **62 imported package modules** to the expected checkout paths and frozen bytes. The 13 acceptance tests also passed after the diagnostic edit. Later source edits do not retroactively update this evidence; a new run gets its own manifest.

These cases support the claim that specific execution/storage mechanisms can be composed and inspected. They do not prove that arbitrary strategies are expressible, that numerical model outputs are correct, or that an entire published system has been reproduced. Phase placement uses shared CPU/GPU resources, not independent disaggregated serving clusters. Physical unified memory and multi-GPU collectives remain outside this evidence.

## 2. Measured costs through the fixed tensor IR

Bridge: [profiling/execution_calibration/replay_ir.py](../profiling/execution_calibration/replay_ir.py). Detailed contract: [REPLAY_IR.md](../profiling/execution_calibration/REPLAY_IR.md). Frozen evidence: [20261017-ir-binding-final/REPORT.md](../profiling/results/20261017-ir-binding-final/REPORT.md) and [report.json](../profiling/results/20261017-ir-binding-final/report.json).

Python builds the exact materialized task/port graph and a fixed `strategy-ir-0.1` program. The existing compiler emits actions and waits; Modeling executes their dependencies, resources, transfers and state transitions. Each component cost points to an independent profile metric and its original raw-process SHA256. The component model was frozen before held-out totals were read for comparison. No held-out strategy total is inserted as an operator cost.

All **14 supported resident paths** replayed successfully, and their runtime makespans reproduce the frozen integer-nanosecond component expressions. Their median absolute relative error is **9.48%**, maximum **44.60%**, and **9/14** have error no greater than 20%. Nevertheless, **0/14 pass the complete original quality gates**, because acceptable mean error does not erase measurement-instability failures.

The supported physical boundaries include compact F32 GPU FFN, CPU/GPU intermediate-axis FFN with an explicit CPU sum, GPU post-RoPE Attention, and hybrid CPU QK/softmax followed by charged probability movement and GPU PV/output. Gate and up remain serialized within each native branch when that is the measured execution order.

Five preload paths are **unsupported**: their source measurement contains the aggregate duration of three ordered weight H2D transfers, while the runtime needs individually priced tensor copies. The bridge refuses to divide that aggregate or charge one copy while treating the other two as free. Supporting those paths requires new exact transfer evidence or an independently validated ordered-copy bundle action.

This is a **fixed tensor IR** replay. It does not calibrate the higher-level `ModelPlan` Attention/FFN semantic transforms: for example, the Dense transform includes normalization and residual operations that the compact native FFN excludes. It also does not calibrate a full Dense/MoE model, repeated decode or multi-layer execution.

CUDA total private workspace remains **unknown (`null`)**. Timing-only replay uses a recorded allocator lower bound and explicitly artificial, nonbinding capacities. Its peak allocations are not a physically validated memory budget, and `scenario.TIMING-ONLY.json` must not be installed as an accepted capacity profile. **Physical capacity acceptance remains false.**

## 3. Analytical hardware and strategy sensitivity

Runner: [experiments/hardware_sensitivity.py](../experiments/hardware_sensitivity.py). Protocol: [HARDWARE_SENSITIVITY.md](HARDWARE_SENSITIVITY.md). Tests: [tests/test_hardware_sensitivity.py](../tests/test_hardware_sensitivity.py).

Frozen full campaign: [hardware-sensitivity/REPORT.md](../build/research-evidence/hardware-sensitivity/REPORT.md), [protocol.json](../build/research-evidence/hardware-sensitivity/protocol.json), [runs.csv](../build/research-evidence/hardware-sensitivity/runs.csv), [selections.csv](../build/research-evidence/hardware-sensitivity/selections.csv), [elasticities.csv](../build/research-evidence/hardware-sensitivity/elasticities.csv), and [completion.json](../build/research-evidence/hardware-sensitivity/completion.json).

The campaign contains **220 unique attempts**, not 220 real hardware measurements: 110 Dense and 110 causal synthetic MoE attempts over 11 deduplicated hardware points and 10 unique strategy pairs. Dense completes 95/110 and MoE 93/110, totaling **188 complete executions and 32 explicit failures**. The seven protocol/aggregation/integration tests pass. An earlier 20-attempt Dense CLI smoke is separate and is not added to these denominators.

The models have two layers, H=64, FFN width 192, eight-token prefill and two decode steps. Declared CPU/GPU FLOP rates, VRAM capacity and DRAM–VRAM bandwidth are swept independently, with memory bandwidth, launch overhead and other parameters held fixed unless they are the swept quantity. Rates are fictional effective inputs; all costs are **uncalibrated analytical roofline estimates**, with explicitly idealized zero scratch.

Three families have **four declared candidates each**: execution only, storage only and joint. A single fixed reference is reported separately. The joint family is a predeclared diagonal subset, not a superset of the other families or the complete Cartesian product. Its results must not be advertised as inherently better because it is named “joint.” Optional full-16 enumeration is explicitly a different, unequal-budget comparison and was not used for the primary 220-attempt campaign.

Every candidate starts with cold DRAM weights and empty KV, runs the same trace, and must meet the same required hidden/KV DRAM goals. Optional retained weight copies can differ at termination; no artificial final weight flush is charged. For each family, one policy is selected at the centre and frozen; another is reselected from the same four candidates at each point. This is offline workload-conditioned analysis, not an adaptive algorithm that can predict future MoE routes.

Representative observations under these assumptions are:

| Observation | Evidence | Interpretation boundary |
|---|---|---|
| Execution and cache management interact | At 64 KiB, Dense FFN-on-CPU/other-tasks-on-GPU fails without eviction but completes with LRU at 0.168092 analytical ms and 63,236-byte peak VRAM | A declared strategy fails or completes; this does not prove globally infeasible hardware |
| Link bandwidth can change the selected execution placement | MoE execution-only selection switches from CPU at the centre to all GPU at 4× link bandwidth: frozen CPU 0.169964 ms versus reselected GPU 0.148544 ms | Finite-candidate analytical preference, not measured speedup or global optimum |
| Prefetch can become counterproductive under capacity pressure | MoE at 256 KiB: historical-frequency/prefetch takes 0.547981 ms and 3,302,912 H2D bytes; LRU without prefetch takes 0.478748 ms and 2,662,912 H2D bytes | The combined mechanisms and this synthetic route pattern are compared; the experiment does not isolate every cause or generalize to learned routers |
| More compute throughput may provide little benefit | GPU-compute curves are flat or nearly flat on these tiny memory/launch/transfer-limited analytical cases | A bottleneck observation for the declared model, not a recommendation against faster GPUs |

The storage-only family first completes at a **tested 256 KiB** point; its smaller tested points have no successful candidate. Families containing CPU-only placement complete at the smallest tested 64 KiB point without GPU weight residency. These are finite-grid observations, not exact continuous capacity thresholds.

The 32 failures in the frozen campaign use the earlier high-level dead-end diagnostic, including `ILLEGAL_ACTION: ADVANCE has no running task or future environment event`. They must be described as **no successful policy completion**, not 32 proofs of hardware infeasibility. The separate post-fix smoke below verifies the improved classification without rewriting the frozen campaign or adding its attempts to the original counts.

`elasticities.csv` retains finite-interval `log(T_high/T_low)/log(parameter_high/parameter_low)` for fixed reference, centre-frozen and reselected policies. Failed endpoints yield null; strategy switches and capacity boundaries are not differentiable performance curves. No price, energy or area model exists, so this evidence cannot identify the highest economic investment return.

Eight SVGs and eight standard Matplotlib PNGs were produced and visually checked. [MoE link sensitivity](../build/research-evidence/hardware-sensitivity/moe-transfer_bandwidth.png) is one example. Curves and captions explicitly say uncalibrated analytical; missing points are not plotted as zero.

## 4. Post-fix dead-end classification smoke

The separate [post-diagnostic smoke](../build/research-evidence/hardware-sensitivity-policy-dead-end-smoke/REPORT.md) completed **40 attempts: 26 successful executions and 14 policy dead-ends**. It uses Dense and MoE at the centre and 64 KiB, without rerunning or replacing the full 220-attempt campaign.

[comparison-to-frozen.json](../build/research-evidence/hardware-sensitivity-policy-dead-end-smoke/comparison-to-frozen.json) verifies that every corresponding success flag, successful makespan, H2D/D2H/SSD byte count and DRAM/VRAM peak matches the same point in the frozen campaign. There are no metric differences. All 14 failed cases now expose:

```text
status = invalid_mapping
success = false
termination_reason = policy_dead_end
error_code = no_progress
makespan_ns = null
compilation_error = null
```

The reason explicitly states that the current policy made no progress and this is **not a proof of global infeasibility**. A failed `ADVANCE` is not accepted as valid execution. Direct impossible copies retain their separate `CAPACITY_EXCEEDED` admission result; the diagnostic change does not create CPU fallback or a new scheduling policy.

Source manifests before and after the smoke match, and 60 imported package modules are recorded within the checkout. The [frozen-220 artifact hashes](../build/research-evidence/hardware-sensitivity-policy-dead-end-smoke/frozen220-artifact-hashes.json) verify that its entire retained artifact tree stayed byte-for-byte unchanged. Post-fix reruns of the 13 strategy acceptance tests and seven hardware-protocol tests also passed.

This is a reporting regression check, not another performance-calibration campaign. Its 40 attempts remain a separate denominator.

## 5. Batched hardware measurement and performance acceptance

Two separately retained repeated-execution campaigns completed. Their timing unit is the mean of **16 completed invocations**, derived from each saved batch-total sample, rather than an isolated request. The initial campaign used 24 native processes and produced ten primary predictions with median relative error **11.89%** and **0/10** accepted. Part of that collection overlapped software tests. It remains a separate protocol revision; its measurements are not replaced or pooled with confirmation data.

The confirmation evidence is [20261017-batched-confirmation/analysis/REPORT.md](../profiling/results/20261017-batched-confirmation/analysis/REPORT.md) and [report.json](../profiling/results/20261017-batched-confirmation/analysis/report.json). Four declared shapes use three profile processes with seed **163** and three held-out processes with seed **179** per shape, totaling **24/24 completed native processes**. Each metric retains **31 samples of 16-invocation batch totals**. Costs are frozen from independent profile components before held-out complete-strategy totals are read for comparison.

| Repeated-execution campaign | Native processes | Primary predictions | Median relative error | Maximum relative error | Accepted primary predictions |
|---|---:|---:|---:|---:|---:|
| Initial collection, partly overlapping software tests | 24/24 | 10 | 11.89% | | 0/10 |
| Separate confirmation, agent test workloads paused | 24/24 | 10 | 13.65% | 27.63% | 0/10 |

In confirmation, **all ten primary predictions fail the within-process CV gate**, and **two also exceed the 20% held-out error gate**. The original gates remain 15% within-process CV, 20% between-process median CV, and 20% prediction error. Six alternative dispatch-model predictions are retained separately from the ten primary predictions; they are not added to the primary denominator. Numerical execution checks pass for the declared compact F32 paths, but that does not make their timing predictions accepted.

Agent software tests did not overlap the confirmation collection. Ordinary desktop applications, power policies, and clocks were left unchanged and were not continuously observed or controlled. The confirmation must therefore not be described as an idle-host or interference-free benchmark. CMake/build binding and runner metadata checks were hardened between the initial and confirmation revisions, and the sampling seeds differ. Comparing their medians is not a controlled single-factor ablation and cannot establish a causal batching or host-interference improvement.

The confirmation [fixed-IR binding report](../profiling/results/20261017-batch-ir-binding-confirmation/REPORT.md) and [report.json](../profiling/results/20261017-batch-ir-binding-confirmation/report.json) are verified. Offline audit accepts the provenance of all 24 raw process records; **10/10 primary component-cost paths replay through the compiler and runtime**, comprising eight resident paths and two preload paths. Unlike the historical aggregate-copy experiment, this protocol independently prices gate, up, and down H2D copies, so no aggregate duration is divided and no preload path is unsupported. The bridge still accepts **0/10 latency predictions**, and physical capacity acceptance remains false. Fifteen batch/contract/tamper tests pass separately from these prediction counts.

Preload replay uses explicit absent logical GPU weight versions to require priced copies, while the native probe overwrites preallocated buffers. It does not validate the physical allocation/overwrite lifecycle. All paths retain unknown CUDA private workspace and use an explicitly nonbinding timing ledger; they are not accepted physical-capacity profiles. Fixed-IR replay also remains distinct from calibration of the higher-level `ModelPlan` transforms or full Dense/MoE inference.

Protocol and reproduction guidance are in [batched_calibration/README.md](../profiling/batched_calibration/README.md). The retained `suite.json`, `plan.json`, source/build/runtime hashes, raw batch totals, frozen component model, and standalone `verify_batch.py` audit bind the confirmation. Existing source assets must remain available and unchanged for strict local re-verification. Correct provenance and compiler execution do not override the failed performance gates.

## 6. Reproduction commands and provenance

Run from the repository root with Python 3.10 or newer. The explicit source paths ensure the semantic suite checks this checkout rather than a foreign installed library. Optional Matplotlib adds PNG rendering; SVGs need no plotting dependency.

```powershell
$env:PYTHONPATH = "src;mapping;modeling"
python -m unittest discover -s tests -p test_strategy_fidelity.py -v
python -m unittest discover -s tests -p test_hardware_sensitivity.py -v

# Choose a NEW output directory for every run.
python experiments/strategy_fidelity.py --out build/research-evidence/strategy-fidelity-rerun
python experiments/hardware_sensitivity.py --out build/research-evidence/hardware-sensitivity-rerun

# Fast driver check; not a replacement for the full campaign.
python experiments/hardware_sensitivity.py --quick --out build/research-evidence/hardware-sensitivity-quick-rerun

# Reuse an existing verified hardware experiment. This replays costs; it does not measure hardware.
python profiling/execution_calibration/replay_ir.py `
  --experiment profiling/results/20261017-execution-calibration/transforms-context-01 `
  --output profiling/results/fixed-ir-replay-rerun

python -B -m unittest discover -s profiling/execution_calibration/tests -p test_replay_ir.py -v
```

The semantic report records frozen source and imported-module hashes. The fixed-IR bridge records the frozen component model, physical signatures, raw profile-process hashes and runtime fingerprint. The analytical report records the complete declared hardware/model/route/candidate protocol, deterministic strategy/input/event hashes, and a separately named temporary-path-dependent compilation-audit digest. Its original captured JSONL/protocol are retained; the documented digest-field normalization changed no timing, feasibility or selection value.

These runs predate or postdate different diagnostic edits. Do not present them as one identical-source benchmark without checking their recorded manifests. A later message-label fix does not silently update an earlier retained result.

## 7. What remains before a complete research claim

The immediate technical gap is **performance calibration with matching high-level physical boundaries**, stable independent profiles and held-out validation. Unknown workspace must remain unknown until separately evidenced; accepted timing and accepted capacity are different claims.

The empirical gap is a representative workload/hardware matrix showing that the proposed execution/storage abstraction changes the correctness or usefulness of actual design decisions relative to clear alternatives. Tiny analytical cases demonstrate the workflow and interactions; they cannot carry that final claim alone.

The methodology gap is a precise statement of what the abstraction adds, its supported dimensions, its legal combinations and its exclusions, followed by controlled comparisons with reduced abstractions or missing mechanisms. Complete implementation of every historical paper is not required, but a mechanism-level description is not automatically a reproduction.

The current evidence is therefore suitable for strengthening a working manuscript with real semantic checks and explicitly labeled use cases. It is **not a claim that the paper is complete, that hardware accuracy has passed, or that a DAC/CCF-A result has been established**.
