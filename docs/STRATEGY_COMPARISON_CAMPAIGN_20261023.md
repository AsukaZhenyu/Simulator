# Expanded fixed-hardware strategy comparison

This protocol extends the [pilot metric definitions](STRATEGY_COMPARISON_PROTOCOL_20261023.md)
without changing modeling, profiling, existing measurements or their acceptance
thresholds. Inputs and the full candidate matrix are declared before execution.
All costs remain explicitly **uncalibrated analytical**. Results are conditional
simulator experiments, not measured speedups of a named device or model.

## Scope and complete finite matrix

One CPU, one GPU, separate DRAM/VRAM, constant resource budgets, one request,
two layers, H64/I192, F32. Dense has four execution choices and MoE has six.

| Workload ID | Prefill tokens | Decode steps | Purpose |
|---|---:|---:|---|
| short_prefill | 8 | 0 | Small prompt, no Decode |
| long_prefill | 128 | 0 | Larger prompt, no Decode |
| decode_heavy | 8 | 8 | Short prompt followed by repeated Decode |
| mixed | 128 | 8 | Larger prompt and repeated Decode |

Dense uses these four workloads. MoE crosses each with fixed hot experts,
rotating pairs, and a hotspot switch at the Prefill/Decode boundary. Every token
selects two of four experts. Routes are explicit synthetic model inputs; mapping
continues to see them only after router completion. Different patterns can
change both locality and per-expert token grouping/task count.

With zero Decode, the hotspot-switch trace equals the hot trace exactly. These
two declared scenarios are intentional equivalence controls. There are 16
scenario IDs, **14 distinct workload behaviors plus two controls**, not 16
independent route behaviors. Every one of the 24 matching candidate event traces
must agree in each control pair.

| Execution | Placement/rule |
|---|---|
| e0 | Macros on GPU |
| e1 | Macros on CPU |
| e2 | Dense FFN or MoE experts on CPU, other macros on GPU |
| e3 | Attention on CPU, other macros on GPU |
| e4, MoE only | Revealed expert regions choose by current residency |
| e5, MoE only | Revealed expert regions choose by current isolated compute/transfer path cost |

The adaptive expert rules are local causal rules, not global schedule optimizers.
Default source/control gates retain the model format's explicit CPU contract.

| Storage | Cache | Next-layer prefetch |
|---|---|---|
| s0 | LRU | Off |
| s1 | LRU | On |
| s2 | History frequency | Off |
| s3 | History frequency | On |

Every cell has the same 192 KiB weight-payload cache budget, last-use release,
512 KiB physical VRAM, DRAM 128 MiB, declared CPU 100 GFLOP/s and 20 GB/s,
GPU 1 TFLOP/s and 100 GB/s, and a shared 8 GB/s payload path. CPU/GPU launch
costs are 500/2000 ns; each declared transfer includes 1000 ns fixed latency.
Prefetch uses one next layer, at most two units and includes static Dense
weights. Zero scratch is an idealized explicit assumption, not a measurement.

All combinations run: `4*16 + 12*24 = 352`. This is complete enumeration of this
catalogue, not completeness of the IR or a new search algorithm. It does not
evaluate task partitions, request priority pipelines, phase placement, unified
memory, multi-GPU communication or numerical generation.

## Controlled conditions and retained outcomes

Each cell gets a fresh runtime, weights initially in DRAM, empty KV and the same
mandatory final hidden/KV goals in DRAM. Optional retained weight copies can
differ; no artificial final weight flush is added. Required transfers and
declared execution overheads are timed. Process startup, downloads, compilation
and actual host contention are not part of simulated latency. Harness wall time
is retained separately and is not inference performance.

Primary outcomes and paired interaction formulas follow the pilot protocol.
The full campaign also evaluates a precisely defined offline diagnostic:
choose execution under s0, choose storage under e0, compose those choices, and
compare its result with the best completed cell in the same finite catalogue.
This is a counterexample test for independent selection, not a newly proposed
online search method. Report ties, failures and zero/negative gains unchanged.

Intervals are unions/intersections of recorded event spans. Higher overlap is
not necessarily faster. A post-run lower-bound waste metric counts weight bytes
copied to GPU when that immutable weight ID has no actual GPU consumer anywhere
in the finite workload. It is neither exact prefetch usefulness nor demand hit
rate, and never feeds future information back into mapping.

## Reproduce

From an installed source checkout, or with source package paths configured:

```bash
python experiments/strategy_comparison_campaign.py --out build/strategy-comparison/full --workers 3
```

`--scenarios` accepts an explicit subset for debugging, recorded as a partial
matrix. Output must be new or empty. Unexpected Python failures stop the
campaign; supported model/compiler rejections remain recorded case outcomes.

`protocol.json` includes the entire model/hardware/workload/route/candidate
catalogue. Before/after source manifests cover modeling, mapping, the public
library, packaged data and experiment execution/analysis scripts. Existing
accuracy files are hashed without being used as analytical prices. Each case
retains generated exact prices, compiled IR, actual events, final operation and
tensor catalogues, decision identities and policy audit. Per-case file manifests
bind these artifacts. Only a final completion marker establishes a finished,
source-unchanged campaign.

`records.json`, `runs.csv` and `analysis.json` contain all results. Paired cache,
prefetch, rank-reversal, interaction and independent-selection CSVs preserve
every eligible comparison and explicit unavailable cells. Known deterministic
equivalence controls are not independent statistical samples. Exact tiny
ranking gaps must not be described as robust real-hardware ordering.

The reporting/plotting stage reads those frozen results. Standard Matplotlib
heatmaps show every declared cell; failures and missing cells are distinct
from zero latency. Charts and reports identify analytical scope and link back
to the complete data rather than selecting only positive outcomes.

## Audit and readable report

From PowerShell in a source checkout, use the same source imports for the
simulation and reporting commands:

```powershell
$env:PYTHONPATH = 'src;modeling;mapping;profiling'
python experiments/strategy_comparison_campaign.py --out build/strategy-comparison/full --workers 3
python experiments/strategy_comparison_audit.py build/strategy-comparison/full build/strategy-comparison/full-independent-audit.json
python experiments/strategy_comparison_report.py --campaign build/strategy-comparison/full --audit build/strategy-comparison/full-independent-audit.json --out build/strategy-comparison/report
```

Simulation and the independent audit need no plotting package or saved profile.
Reporting needs a Matplotlib environment (validated with Matplotlib 3.10.1).
The independent auditor imports no simulator, modeling, mapping or campaign
analysis helpers. It replays dependency readiness, copy/compute completion,
locks, releases, physical allocations, final goals, exact cost entries and
summary metrics. Its receipt binds the completed campaign and audit source.
This is an internal consistency check, not hardware accuracy validation.

The report emits self-contained Chinese HTML, companion Markdown, PNG/SVG
figures, all CSV result tables and hashed delivery files. Main aggregate
comparisons use 14 distinct inputs (304 runs); all 16 declared scenario IDs
and 352 runs remain visible, including 48 equivalence-control runs. Pairwise
and four-cell comparisons share runs and are not independent timing samples.
