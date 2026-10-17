# Analytical hardware and strategy sensitivity

This source-checkout experiment provides a reusable **analysis protocol**, not a hardware validation result. It calls the public `llm_simulator` API with explicitly declared, fictional device rates. It needs no saved profile, native executable, GPU, model weights, network access or plotting dependency.

Every output says **uncalibrated analytical**. A simulated latency difference is not a measured speedup, a global optimum, or evidence that a real machine should be bought or redesigned.

## Run

From the repository root, with Python 3.10 or newer:

```bash
python -m pip install -e .
python experiments/hardware_sensitivity.py --out build/research-evidence/hardware-sensitivity
```

The result directory must be new or empty. Existing evidence is never overwritten. Use `--models dense` to run Dense only; `--quick` runs the centre and one low-VRAM point as a pipeline check. `--retain-traces` saves the complete public-API audit artifacts for every attempt, including failed evaluations. The default compact outputs contain all metrics, deterministic declared-strategy/input/event digests and a separately named compilation-audit digest. The latter includes temporary input paths and is not a portable strategy identity. The protocol and source reproduce the finite experiment.

```bash
python -m unittest discover -s tests -p test_hardware_sensitivity.py -v
```

Unexpected Python exceptions stop the campaign and leave a partial `runs.jsonl` without `completion.json`. Known unsupported/invalid contracts are recorded as failed attempts. A valid evaluation that does not reach its goal cannot enter candidate selection.

## Controlled inputs

The two small models have two layers, hidden width 64, FFN width 192, four query heads and two KV heads. Both execute one eight-token prefill and two one-token decode rounds. Dense uses the materialized F32 macro contract. MoE has four experts, top-2 selection and a deliberately synthetic rotating-pair trace. The model owns that trace; the built-in policy gets only revealed routes and public history. This experiment does not evaluate a learned router or real workload distribution.

The centre platform declares CPU compute 100 GFLOP/s and memory bandwidth 20 GB/s; GPU compute 1 TFLOP/s and memory bandwidth 100 GB/s; DRAM 128 MiB; VRAM 2 MiB; a shared DRAM–VRAM link at 8 GB/s. CPU/GPU launch costs are 500/2,000 ns and transfer latency is 1,000 ns. MoE's contract additionally requires explicit SSD rates of 1/0.5 GB/s even though the default inventory starts in DRAM. No weights begin on SSD and no SSD performance claim follows.

The one-factor-at-a-time grids are:

| Axis | Tested values | Held fixed |
|---|---|---|
| CPU compute throughput | 0.5×, 1×, 2× | CPU memory bandwidth, overhead and all other inputs |
| GPU compute throughput | 0.5×, 1×, 2× | GPU memory bandwidth, overhead and all other inputs |
| DRAM–VRAM payload bandwidth | 0.25×, 1×, 4× | Transfer latency, compute and capacities |
| VRAM capacity | 64, 128, 256, 512, 2,048 KiB | All effective rates and policy parameters |

The centre is deduplicated across grids. Models, traces, costs other than the swept parameter and initial/required terminal conditions stay fixed. Every candidate gets a fresh runtime: all weights initially in DRAM, empty KV, mandatory hidden outputs and persistent KV in DRAM. Optional retained weight copies differ by strategy; there is no artificial final weight flush. The experiment therefore measures cold finite-workload completion rather than a steady-state service or a common post-flush state.

Costs use `max(FLOPs / compute_rate, declared_bytes / memory_bandwidth)` plus the declared overhead and exact payload transfer formula. `workspace_bytes=0` is an idealized scratch declaration, never a measurement. Flat compute-sweep curves may indicate the chosen graph is limited by memory, launches or transfers.

## Equal-budget finite candidate families

Execution choices:

| ID | Placement |
|---|---|
| e0 | All macros on GPU |
| e1 | All macros on CPU |
| e2 | FFN/expert FFN on CPU; remaining macros on GPU |
| e3 | Attention on CPU; remaining macros on GPU |

Storage choices all release dead intermediates at last use:

| ID | Weight cache | Cross-layer prefetch |
|---|---|---|
| s0 | No weight eviction; no explicit weight-cache budget | None |
| s1 | LRU, 192 KiB weight-payload budget | None |
| s2 | LRU, 1,536 KiB weight-payload budget | None |
| s3 | Historical-frequency eviction, 1,536 KiB budget | One next layer, at most two units; include static Dense weights |

Cache budgets are strategy parameters fixed across hardware points. VRAM is the actual capacity constraint and also contains activations/KV/workspace. A payload-cache budget is not total VRAM capacity.

| Family | Exactly four candidates |
|---|---|
| Execution only | e0-s0, e1-s0, e2-s0, e3-s0 |
| Storage only | e0-s0, e0-s1, e0-s2, e0-s3 |
| Joint | e0-s0, e1-s1, e2-s2, e3-s3 |

The fixed reference is e0-s0. It has no reselection budget because it is a single declared strategy. All three selectable families have equal budgets of four unique candidates. Duplicates across families are executed once and reused, without increasing any family’s candidate count. The joint family is a predeclared diagonal subset, **not a superset** of the other families and not the complete Cartesian product; it can be worse than an execution-only or storage-only family. Equal candidate counts alone do not make the spaces equally expressive or sampling unbiased.

`--exhaustive-joint` optionally adds the complete 16 combinations as `joint_full16_unequal_budget`. Its selection is a best among tested finite candidates. It is reported separately because its larger budget cannot substantiate an equal-budget advantage.

This harness enumerates candidates to support a controlled use case. It does not add a search algorithm to the simulator, compare arbitrary tensor partitions or implement distributed TP/PP, independent serving clusters, physical unified memory or numerical inference.

## Metrics and interpretation

Each attempt retains status, success, termination reason, errors, estimated makespan, H2D/D2H and SSD bytes, peak DRAM/VRAM, event counts and per-device compute service time. Failed partial-run latency is not treated as a completed inference time. Unknown and infeasible selection cells remain empty/null.

For each family, two policies are compared:

1. **Frozen at the centre:** choose the best successful candidate at the centre and replay it unchanged at every hardware point.
2. **Reselected:** choose the best successful candidate from the same four declared candidates independently at each hardware point.

The frozen-to-reselected latency ratio uses successful completions at both endpoints. No ratio is reported if the frozen strategy fails. Reselection is offline and workload-conditioned: it must not be described as a causal runtime algorithm that predicts future routes.

`elasticities.csv` reports the finite-interval dimensionless quantity

`log(T_high / T_low) / log(parameter_high / parameter_low)`.

A value near −1 means doubling that parameter halves the analytical latency over that interval. A value near zero means little effect under those inputs. It is not a derivative across a strategy switch or capacity discontinuity. Infeasible endpoints yield null. Compute throughput and memory bandwidth are swept separately, so this does not predict the effect of replacing a whole real processor.

`thresholds.json` reports only the first feasible tested VRAM point and the largest infeasible tested point within each candidate family. It does not claim the exact minimum feasible capacity. A CPU-only candidate can complete at tiny declared VRAM because it needs no GPU weight residency.

No prices, power or silicon area are modeled. A useful next hardware direction requires calibrated costs, representative workloads and explicit objectives; this experiment can expose bottlenecks under its assumptions but cannot rank economic investment returns.

## Artifacts

| File | Evidence |
|---|---|
| `protocol.json` | All fictional hardware values, complete model/workload/route definitions, candidate parameters, budgets and conditions |
| `runs.jsonl`, `runs.csv` | Every attempted model/point/candidate, including explicit failures |
| `selections.csv` | Fixed reference, frozen-at-centre and reselected outcomes, bytes and peaks |
| `elasticities.csv` | Dimensionless interval sensitivities, policy switches and null boundaries |
| `thresholds.json` | Finite-grid feasibility observations |
| `*.svg` | Dependency-free plots with missing failed/unsupported points and analytical labeling |
| `*.png` | Optional standard Matplotlib rendering when Matplotlib is installed; no effect on numerical results |
| `REPORT.md` | Automatic outcome summary and interpretation limits |
| `completion.json` | Completion marker, attempt counts and protocol digest |

Machine-readable numbers remain analytical even if semantic tests pass. These are suitable for a clearly labeled simulator demonstration; they cannot fill a paper's held-out measured accuracy or real-system speedup tables.
