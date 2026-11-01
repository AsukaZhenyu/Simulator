# Paper study 2: Can predictions select the better strategy?

Date: 2026-10-23. Final evidence: `build/dac-evidence-20261023/selection-v3/`.

The offline experiment is complete. **Current complete-layer strategy selection
is not yet validated.** The most recent steady-cost campaign measures one plan
for each of three distinct layer shapes. It supports scoped latency validation,
but comparing different shapes is not strategy selection. Older fixed-program
data supply 59 matched comparisons, of which seven copy/compute comparisons
retain all original accuracy/stability gates and have a separated observed winner.

This is a retrospective reanalysis of already inspected evidence, not a newly
preregistered experiment or new hardware measurement. No parameter was fitted,
no historical file changed, and no historical quality threshold was relaxed.

## Available evidence and matching

| Cohort | Scope | Selection groups | Matching and exclusions |
|---|---|---:|---|
| 22 October steady costs | 11 cases, 33 heldout launches; three full Dense shapes plus eight components | 0 complete-layer groups | Each layer shape has one plan. CPU/GPU component shapes do not form a complete-layer comparison and have different case seeds. |
| 18 October chain | 23 shape/partition cases; 143 path predictions | 23 | Five Attention or six FFN resident paths share generated F32 inputs, case, seed and native process. All immutable inputs start resident in both CPU/GPU memory; output finishes in DRAM. Fourteen preload paths have different placement contracts and remain in latency results, but are excluded from resident selection. |
| 18 October copy/compute overlap | 18 FFN/payload cases; 72 path predictions | 36 | Compare serial versus overlap separately for one-copy and two-copy work. Same resident FFN, same future payloads, same process and completed-copy/output boundary. One-copy is never compared to two-copy as equal work. |
| 18 October high-level compiler | 14 cases, 42 heldout processes | Not inferred | Preserve the prior 0/14 complete-gate result. Neighboring partition cuts are not treated as matched plans without complete initial data/placement identity proof. |

The chain and overlap executables are historical bounded protocols. Their
results do not validate the current general high-level compiler, production
weights, arbitrary MoE routes, multi-layer cache history or changing-resource
responses. Path measurement order inside a process was fixed; order and drift
effects cannot be separated after collection.

## Measurements and decision rules

The selected strategy minimizes the original frozen prediction. Prediction ties
break lexicographically. The observed winner minimizes the pooled heldout median
over all three launches. Every planned candidate remains in this comparison,
including failed stability cases. Selection regret is

`observed_median(selected) / min(observed_median(all_candidates)) - 1`.

The script independently reloads 246 historical chain/overlap processes,
checks their numerical/precision/seed/geometry/process records, reconstructs
profile medians and frozen sum/max predictions, and recomputes heldout medians
and original statistical gates. Heldout launches must bind the exact frozen
predictor hash. The 33 latest steady launches are checked against their raw
trial-average timings and launch identities. A 439-file input inventory is
verified before and after analysis; missing or altered evidence fails closed.

Uncertainty uses an exact empirical **paired process-cluster bootstrap**: all
27 ordered draws of three real launches. All timing observations belonging to a
drawn process are retained; strategies from the same process use the same draw.
This does not turn repeated invocations into independent processes. Intervals
are 95% nearest-rank empirical intervals; three clusters make them coarse.
An interval containing zero means the ordering is inconclusive, not that the
strategies are equivalent. Comparing the selected winner with itself cannot
establish a near tie: winner separation is checked against every alternative.

Full qualification retains the original profile-source, heldout repeatability,
error, numerical and coverage gates for **every candidate**. No survivor-only
reranking is performed. Bootstrap intervals never override a failed original
gate.

## Recomputed results

| Cohort / error unit | Observations | Median APE | P90 APE | Worst APE | Original full-gate results |
|---|---:|---:|---:|---:|---:|
| Steady: one error per heldout process mean | 33 | 2.793% | 9.599% | 12.082% | 7/11 cases |
| Chain: one error per case/path pooled median | 143 | 6.928% | 43.527% | 52.264% | 7/143 paths |
| Overlap: one error per case/path pooled median | 72 | 2.502% | 7.734% | 47.663% | 15/72 paths |

The error units differ and must not be pooled into a single accuracy number.
Steady repeats are timed in batches: these are not individual-call tails.
Steady accuracy gates pass in 9/11 cases; complete layers pass accuracy in 2/3
and all gates in 1/3. This remains the existing scoped result.

| Matched selection cohort | Groups | Exact observed winner | Median regret | P90 regret | Worst regret | Groups passing every original gate |
|---|---:|---:|---:|---:|---:|---:|
| Resident chain | 23 | 11/23 | 0.722% | 15.559% | 95.092% | 0/23 |
| Serial/overlap | 36 | 35/36 | 0% | 0% | 53.864% | 7/36 |

All seven fully qualified overlap groups also have a separated observed winner
under the coarse cluster interval. Three of the twelve chain mismatches and the
one overlap mismatch are statistically inconclusive under this resampling.
The remaining mismatches stay errors; low aggregate regret cannot erase them.

The largest chain mistake is `ffn-h256-i512-q8-cut256`. The frozen model selects
`hybrid/resident`; its heldout median is 552,500 ns versus CPU/resident's
283,200 ns. Regret is 95.092% (269,300 ns). Its process-cluster regret interval is
88.456%–101.292%. This is historical failed-quality evidence, not a prediction
from the latest steady model.

The largest overlap mistake is H1024/I2048/q32 with two 16 MiB flows. The selected
overlap median is 5,114,900 ns versus serial's 3,324,300 ns: 53.864% regret. Its
cluster regret interval is 0%–57.043%, and the signed difference interval crosses
zero. The unstable evidence does not establish that overlap is always worse.

## Consequence for the paper

The existing evidence supports bounded copy/compute selection and an honest
latency-validation section. It does **not** establish broad complete-layer
policy selection. This negative result locates a specific missing experiment,
rather than requiring a new model architecture or another broad calibration
campaign.

The next narrowly scoped hardware matrix can reuse one existing H512/q64/history0
F32 Dense shape, then compare CPU-only, GPU-only, CPU-Attention/GPU-FFN and one
CPU/GPU FFN split. All candidates must share the same weights, numerical seed,
precision, initial residency, final output and KV boundary. First check that the
native executor supports every compiled action. Freeze scoped profile prices
before three fresh heldout launches per plan, use a common disjoint heldout seed
and balance plan order. Retain the steady gates and report all candidates even
if one fails. This matrix is a proposed protocol, **not already executed data**.

## Reproduction and receipts

From the repository root, use a new output directory:

```powershell
python experiments/paper_selection_study.py prepare --output build/selection-new
python experiments/paper_selection_study.py run --output build/selection-new
python -m unittest discover -s tests -p test_paper_selection_study.py -v
```

The final directory contains `protocol.json`, `study-source.py`, `results.json`,
`completion.json`, and focused software-validation receipts. A completed offline
run exits successfully even when the scientific qualification is negative;
`current_whole_layer_strategy_selection_qualified` remains explicitly false.
Missing/changed evidence raises an error and cannot produce a completion receipt.

Version 1 failed closed because two historical native schemas use distinct F32
precision enum spellings. Its attempt and source are preserved. Version 2
completed; version 3 adds a correct winner-versus-alternatives uncertainty
distinction. No measurements, price models, candidates, exclusions or gates
changed between these postprocessing versions.
