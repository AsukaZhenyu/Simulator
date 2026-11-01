# Physical-stage calibration and fresh validation — 2026-10-22

**Conclusion: this is a completed diagnostic calibration/validation cycle, not a qualified instantaneous-availability model.** CUPTI materially changes the small workload. Before/after reference tasks provide stale workload response estimates; they do not identify a neutral, continuous resource-availability trajectory. All errors below remain diagnostic.

For the predeclared completed-GPU-response hypothesis, bare-target median APE changes from 29.00% (idle baseline) to 19.59%, but P90 worsens from 51.76% to 64.57%. Traced median/P90 are 12.38%/31.64%. This is a typical-call improvement, not an overall accuracy pass. A fresh bare CPU-contention call took 36.1308 ms while the frozen prediction was 3.904765 ms (89.19% error). All 24 heldout processes have CV above 10%; high-latency calls remain in every reported metric.

## Workload and independence

- One H=128, q=16, history=0 F32 dense layer; CPU/GPU partition, attention, FFN and transfers; one CPU thread on a declared core. This is a small controlled layer, not an accuracy result for a production-size LLM.
- Four offered-load conditions: idle, CPU replay, GPU replay, joint replay. These labels are not observed available-resource percentages.
- Profile: 12 fresh main processes, 32 full calls each. Observer controls: 9 processes, 32 calls each. Traced heldout: 12 processes / 384 calls. Bare-target heldout: 12 additional processes / 384 calls. Reference probes remain CUPTI-instrumented in the bare-target cohort and execute outside target timing.
- After two post-target constructor failures, remaining collection explicitly stops owned background actors immediately after main sampling and before after-reference construction. BEFORE and main work are unchanged. These AFTER references are unloaded recovery diagnostics, not same-load drift observations; the frozen parameter file remains unchanged.
- All five candidates were declared before heldout. Two reference-response hypotheses use only completed BEFORE references. AFTER references audit drift and never feed prediction. Different heldout numeric seeds and process identities are enforced.
- Two attempts (one profile, one heldout) completed main timing but their after-reference constructors timed out. The failed attempts are retained and excluded as units; infrastructure replacements with the same configuration are explicitly indexed. A prior two-call smoke export failure is also retained outside this campaign.
- Frozen parameter SHA256: `697253dba6adc23329519aa3450f930b35724886810a465889e1e9900ba88516`; saved barrier time: `2026-10-22T09:53:15.404615+00:00`.
- No DNN, heldout fitting, whole-layer residual offset, observer-overhead subtraction or rate clipping is used. Parameter means come from profiling; scoped component response and serial execution are calculated in modeling using the common engine.

## What the candidates mean

| Candidate | Frozen inputs and rule | Limitation |
|---|---|---|
| idle_component_baseline | Sum of disjoint per-action component means from idle profile processes | No response to resource changes |
| condition_component_baseline | Component means for each known offered-load label | Label-conditioned control; not a general availability model |
| reference_activity_candidate | Idle kernel activity scaled by BEFORE GPU reference activity rate; host components by completed BEFORE CPU reference response | Host remainder is not active CPU work; references are stale and workload dependent |
| reference_completed_gpu_candidate | Whole GPU COMPUTE host response scaled by completed BEFORE GPU reference response; host CPU components by completed BEFORE CPU reference response | Tests driver/queue response, without identifying its physical mechanism |
| observer_bare_idle_baseline | Original per-action inclusive means from the three bare idle controls | Untraced idle control only; no contention model |

Activity, API, and host columns overlap in raw CUPTI. The estimator instead forms an explicit disjoint partition: kernel activity, copy outside kernel, API outside device activity, remaining host wall, inter-action gaps, and tail. Their sum is checked against every original whole-call duration. Component means preserve mean accounting; a sum of marginal medians is not used.

## Fresh traced heldout

| Frozen candidate | Calls | Median APE | P90 APE | Worst APE |
|---|---:|---:|---:|---:|
| idle_component_baseline | 384 | 30.04% | 52.89% | 91.19% |
| condition_component_baseline | 384 | 37.75% | 100.30% | 180.09% |
| reference_activity_candidate | 384 | 35.99% | 98.72% | 145.67% |
| reference_completed_gpu_candidate | 384 | 12.38% | 31.64% | 89.85% |
| observer_bare_idle_baseline | 384 | 42.70% | 61.41% | 92.78% |

## Fresh bare-target heldout

| Frozen candidate | Calls | Median APE | P90 APE | Worst APE |
|---|---:|---:|---:|---:|
| idle_component_baseline | 384 | 29.00% | 51.76% | 91.41% |
| condition_component_baseline | 384 | 47.54% | 129.09% | 173.32% |
| reference_activity_candidate | 384 | 37.88% | 145.03% | 193.96% |
| reference_completed_gpu_candidate | 384 | 19.59% | 64.57% | 89.19% |
| observer_bare_idle_baseline | 384 | 39.46% | 60.49% | 92.96% |

APE = |prediction − measured whole-call time| / measured whole-call time. Each process contributes all 32 calls; no high-latency calls are discarded. Global percentiles are calculated across calls, not by averaging state percentiles. Traced and bare errors describe different timing protocols and must not be silently pooled.

## Per-state traced errors

| State | Candidate | Processes | Median APE | P90 APE | Worst APE |
|---|---|---:|---:|---:|---:|
| idle | idle_component_baseline | 3 | 9.21% | 21.40% | 39.18% |
| idle | condition_component_baseline | 3 | 9.21% | 21.40% | 39.18% |
| idle | reference_activity_candidate | 3 | 8.34% | 20.20% | 31.08% |
| idle | reference_completed_gpu_candidate | 3 | 14.99% | 24.08% | 34.10% |
| cpu | idle_component_baseline | 3 | 6.38% | 35.06% | 90.53% |
| cpu | condition_component_baseline | 3 | 81.94% | 98.95% | 110.66% |
| cpu | reference_activity_candidate | 3 | 93.88% | 119.44% | 145.67% |
| cpu | reference_completed_gpu_candidate | 3 | 8.69% | 32.47% | 89.85% |
| gpu | idle_component_baseline | 3 | 46.60% | 52.59% | 64.42% |
| gpu | condition_component_baseline | 3 | 8.26% | 27.59% | 44.56% |
| gpu | reference_activity_candidate | 3 | 41.42% | 50.54% | 59.26% |
| gpu | reference_completed_gpu_candidate | 3 | 12.64% | 23.07% | 41.57% |
| both | idle_component_baseline | 3 | 46.84% | 90.85% | 91.19% |
| both | condition_component_baseline | 3 | 87.89% | 130.20% | 180.09% |
| both | reference_activity_candidate | 3 | 18.88% | 81.86% | 82.87% |
| both | reference_completed_gpu_candidate | 3 | 14.17% | 83.98% | 85.71% |

## Per-state bare-target errors

| State | Candidate | Processes | Median APE | P90 APE | Worst APE |
|---|---|---:|---:|---:|---:|
| idle | idle_component_baseline | 3 | 10.79% | 25.32% | 32.68% |
| idle | condition_component_baseline | 3 | 10.79% | 25.32% | 32.68% |
| idle | reference_activity_candidate | 3 | 18.45% | 36.81% | 48.52% |
| idle | reference_completed_gpu_candidate | 3 | 23.98% | 66.42% | 87.78% |
| cpu | idle_component_baseline | 3 | 14.27% | 32.73% | 90.53% |
| cpu | condition_component_baseline | 3 | 103.71% | 139.51% | 149.99% |
| cpu | reference_activity_candidate | 3 | 139.58% | 179.66% | 193.96% |
| cpu | reference_completed_gpu_candidate | 3 | 24.32% | 49.85% | 89.19% |
| gpu | idle_component_baseline | 3 | 42.94% | 51.72% | 57.88% |
| gpu | condition_component_baseline | 3 | 9.84% | 37.51% | 56.93% |
| gpu | reference_activity_candidate | 3 | 37.92% | 47.13% | 53.82% |
| gpu | reference_completed_gpu_candidate | 3 | 13.30% | 24.92% | 34.00% |
| both | idle_component_baseline | 3 | 43.50% | 90.65% | 91.41% |
| both | condition_component_baseline | 3 | 99.70% | 148.11% | 173.32% |
| both | reference_activity_candidate | 3 | 32.91% | 79.80% | 81.63% |
| both | reference_completed_gpu_candidate | 3 | 14.53% | 82.55% | 85.16% |

## Measured physical accounting on traced heldout

| State | Kernel activity (ms) | Copy outside kernel (ms) | API outside device (ms) | Other host + gaps (ms) | Whole-call mean (ms) |
|---|---:|---:|---:|---:|---:|
| idle | 0.1262 | 0.1079 | 2.9439 | 0.5681 | 3.7461 |
| cpu | 0.1252 | 0.1059 | 5.2940 | 0.9357 | 6.4608 |
| gpu | 0.1185 | 0.1018 | 5.6869 | 0.6605 | 6.5676 |
| both | 0.1190 | 0.1018 | 10.4877 | 1.7191 | 12.4276 |

These are descriptive measured envelopes. API outside device activity is not identified active CPU work. Disjoint columns sum to whole-call mean; they are not a decomposition into independently additive physical service demands.

## Backend call multiplicity and launch-to-start timing

In this executor/shape, the Attention PV action (original action 14) issues 52 cudaLaunchKernel calls, 33 cudaMemcpyAsync calls and 2 cudaStreamSynchronize calls per invocation. The idle profile-0 mean PV host span is about 1.91 ms; launch APIs account for about 1.06 ms and memcpy APIs about 0.67 ms. This describes this concrete GGML execution path, not a universal Attention implementation or a fused production kernel.

| Profile-0 state | PV host span (ms) | Launch API union (ms) | Sync API union (ms) | Other runtime API union (ms) |
|---|---:|---:|---:|---:|
| idle | 1.9113 | 1.0630 | 0.0120 | 0.6842 |
| cpu | 3.9837 | 1.0415 | 0.0205 | 1.7414 |
| gpu | 2.3361 | 1.0706 | 0.3462 | 0.7802 |
| both | 3.4980 | 1.1346 | 1.3861 | 0.8270 |

The exact-correlation diagnostics find GPU-COMPUTE per-kernel launch-return-to-start median gaps of approximately 0.130/0.083/143.655/140.853 microseconds for idle/CPU/GPU/joint profile-0 conditions. These overlapping gaps must not be summed as serial costs. They establish delayed starts, not whether the cause is hardware admission, dependencies, driver behavior or OS scheduling. Kernel envelopes alone miss this behavior. The measured inclusive action cost already contains its launches and waits; adding another launch fee to that cost would double-count. A future physical lowering must replace inclusive terms with an explicit submission/admission/device/completion dependency graph, with independently identified parameters.

## Qualification and remaining identification gaps

The frozen observer check reports mean call overhead of 14.30% for CUPTI and 18.08% for CUPTI plus references, against a predeclared 5% neutrality gate. This gate fails. Paired per-process medians and actual background coverage are retained in `campaign-audit.json`.

- Low point error alone does not certify accuracy. Required checks also include independent processes, full coverage, numerical correctness, capture integrity, CV <= 10%, observer neutrality, and identified physical service. No candidate passes the complete hardware qualification.
- The CPU batch counter is quantized and includes query brackets. It can exceed the measured batch wall. The strict supply adapter therefore leaves CPU physical service unknown; it never clips to 100%. Completed CPU reference throughput is a separate, explicitly scoped diagnostic response measurement.
- A CUPTI kernel interval is a device activity envelope, not proof of SM-active or preemption-free service. Known GEMM work divided by that union gives reference-workload effective throughput, not universal available FLOP/s.
- References are outside the target interval. CPU and GPU sensors have different staleness; after-reference constructors and warmups also change device history. Within-call transitions and immediate recovery remain unidentified. The provided sample-held trajectory must retain unknown regions and its observed_diagnostic source kind.
- The analyzer trusts collection numerical/owned-process receipts. The independent reference adapter cross-checks oracle/raw hashes, physical process identity, native action identity, common clock and exact CUPTI correlation; its outputs are retained under `reference-adaptations/`.
- These data do not establish storage-bandwidth response, memory-pressure accuracy, another shape/dtype/CPU-thread count, a multi-layer model, or an unseen fluctuating workload. Previous campaign percentages use a different protocol and cannot be cited as a before/after improvement here.

## Next mechanisms to identify

1. Reduce and independently quantify tracing cost. Use sampled/separate diagnostic captures for physical explanation and bare whole-call runs for final accuracy; do not subtract a global profiler constant.
2. Measure CPU active runtime work separately from blocked/ready waiting at sufficient resolution. Use profile-only high-resolution scheduling evidence to identify parameters; do not inject the heldout target execution trace as its own external resource input.
3. Separate CUDA submission CPU work, device admission/queue delay, device execution and completion notification. GPU kernel-throughput corrections alone cannot account for dominant host/API waiting.
4. Validate reference observability, cadence, perturbation and hold validity with independent transitions before claiming a continuous r_i(t). Then freeze again and use a new heldout cohort; the current heldout must remain a final comparison for this frozen version.

## Evidence and reproduction

The independent campaign audit admits all 46 effective jobs, verifies 238 successful owned-process exits, and retains both failed attempts (all their processes exited, including the two forced constructor exits). It verifies 19 post-target cleanup chronologies and actual background coverage of every main call. The strict adapter additionally cross-checks 238 actor/program/clock/raw-file bindings, 238 saved independent numerical receipts covering 8,330 ports, and 46 program/oracle bindings. All 78 GPU reference samples are available as scoped diagnostics; all 78 CPU physical-service samples remain unknown. Software regression: 729 tests executed successfully, with 7 declared skips; this is not a hardware-accuracy assertion.

All JSON/CSV outputs use LF and are write-once. Raw data, failed attempts, source snapshots, binary build receipts, process ownership/cleanup, oracle receipts, frozen parameters and both heldout cohorts remain local under this campaign directory. They are not library prerequisites and generated results are ignored by Git.

See `docs/PHYSICAL_STAGE_MEASUREMENT.md` for reusable collection/build commands. Use `physical_stage_analysis.py` to repeat the frozen comparisons and `physical_campaign_audit.py` for independent evidence checks.
