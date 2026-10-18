# Chain accuracy results — 2026-10-18

The fresh chain campaign completed all 138 native processes without a recorded native failure. Median errors improved with in-chain observations and profile-only shared corrections, but the unchanged stability gates prevent a claim that chain accuracy is broadly calibrated. The independent-component baseline accepts 7 of 143 case/path results; the in-chain and shared models accept none under the complete gates below.

This note is a lightweight aggregation of four existing JSON reports. It does not refit a model, replay a native process, inspect new raw data, or change a protocol source.

## Scope and units

The campaign covers F32 compact FFN and post-RoPE Attention with completed-work CPU, GPU, hybrid, and joint paths. There are 23 declared cases: 14 FFN and 9 Attention. Each case has three independent profile processes and three independent heldout processes, giving 69 profile and 69 heldout processes. Each process records 51 invocations per measured path. The error target is the pooled heldout median of separate **uninstrumented** completed-work invocations; error is absolute relative error against that median.

| Family | Cases | Profile / heldout processes | Paths per case | Case/path results |
| --- | ---: | ---: | ---: | ---: |
| FFN | 14 | 42 / 42 | 7 | 98 |
| Attention | 9 | 27 / 27 | 5 | 45 |
| Total | 23 | 69 / 69 | — | 143 |

Both families include `cpu/resident`, `gpu/resident`, `hybrid/resident`, `cpu/joint_resident`, and `gpu/joint_resident`. FFN also includes `hybrid/preload` and `hybrid/joint_resident`. Preload is a distinct placement contract. The largest declared FFN is H2048/I5632/q1 with CPU cut 1408; the largest Attention is H2048/q1/history128. These are measured cases, not a general capacity validation.

Full acceptance requires numerical/process coverage, a model frozen before heldout collection, at least three processes per split, within-process CV <= 0.15, CV of process medians <= 0.20, and heldout median relative error <= 0.20. Every applicable source and heldout target must pass; a low median prediction error alone is insufficient. All numeric/process coverage and freeze gates passed in the baseline and trace reports.

## Prediction results

Percentages below aggregate one error per case/path, rather than one error per invocation. The in-chain model uses profile stage durations, measured gaps, branch launch offsets, and join completion behavior. It does not replace the independent baseline's evidence or gates.

| Model | Case/path scope | Median error | Worst error | Error <= 20% | Fully accepted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Independent components | 143 | 6.927711% | 52.264374% | 102 / 143 | 7 / 143 |
| In-chain host trace | 143 | 3.940763% | 24.267201% | 141 / 143 | 0 / 143 |
| Shared correction, full profile set | 60 | 6.186170% | 35.340137% | 53 / 60 | 0 / 60 |
| Shared correction, leave one whole case out | 60 | 6.505150% | 35.943179% | 52 / 60 | 0 / 60 |

The shared-model scope contains 42 FFN and 18 Attention results: only the primary `gpu/resident`, `hybrid/resident`, and FFN `hybrid/preload` strategies. It excludes CPU and joint paths. On those same 60 results, the independent baseline has median error 10.960826%, worst error 52.264374%, and 35 / 60 errors <= 20%. Shared models use profile complete-chain totals to calibrate a correction; they are not an independent-component-only prediction.

| Model | FFN median / worst / <= 20% | Attention median / worst / <= 20% |
| --- | --- | --- |
| Independent components | 11.674736% / 52.264374% / 57 of 98 | 3.539653% / 16.624063% / 45 of 45 |
| In-chain host trace | 4.895208% / 24.267201% / 96 of 98 | 3.168407% / 9.332516% / 45 of 45 |
| Shared full | 10.755727% / 35.340137% / 35 of 42 | 1.543226% / 6.343302% / 18 of 18 |
| Shared leave one case out | 11.019744% / 35.943179% / 34 of 42 | 1.448024% / 6.158821% / 18 of 18 |

The shared leave-one-case-out test excludes complete-chain calibration targets for one declared **shape-and-partition case ID**. It retains that case's independently measured components and may retain sibling partitions with the same base dimensions. It does not hold out every partition of a base geometry, nor establish prediction without profiling. All full and excluded-case coefficient models were frozen before heldout collection; heldout data were not used for fitting.

The largest baseline error is `ffn-h512-i1024-q32-cut128`, `hybrid/preload` (52.264374%). The same path is worst for shared full (35.340137%) and excluded-case transfer (35.943179%). The worst in-chain path is `ffn-h512-i1024-q1-cut128`, `hybrid/preload` (24.267201%).

## Stability is the dominant failing gate

These counts are case/path results with at least one failing source or heldout CV. Source and target failures can overlap.

| Gate failure | Independent components | In-chain host trace |
| --- | ---: | ---: |
| Profile source within-process CV | 131 / 143 | 143 / 143 |
| Heldout target within-process CV | 88 / 143 | 88 / 143 |
| Combined within-process gate | 136 / 143 | 143 / 143 |
| Profile source between-process CV | 28 / 143 | 138 / 143 |
| Heldout target between-process CV | 3 / 143 | 3 / 143 |
| Combined between-process gate | 31 / 143 | 138 / 143 |
| Heldout relative error | 41 / 143 | 2 / 143 |

The trace source gate retains every modeled stage, gap, launch, and tail cost, including small variable gaps; their CV failures remain failures. For the shared correction, 102 of 180 profile process/path calibration targets fail within-process CV and 1 of 60 groups fails between-process CV. All 60 shared paths also fail the original independent-source/heldout stability conjunction. The shared acceptance counts above are derived conservatively by preserving those original gates, adding the applicable full or excluded-case calibration-source gates, and applying the new model's error threshold. They are not an acceptance field supplied by `shape-report.json`.

The seven accepted baseline results are:

- `ffn-h256-i512-q8-cut64`: `cpu/joint_resident`.
- `ffn-h512-i1024-q32-cut128`: `cpu/joint_resident`.
- `ffn-h512-i1024-q32-cut512`: `cpu/resident`, `cpu/joint_resident`.
- `ffn-h512-i1024-q32-cut896`: `cpu/joint_resident`.
- `attention-h256-q8-hist512`: `cpu/joint_resident`.
- `attention-h512-q32-hist128`: `cpu/joint_resident`.

## Rankings and selection regret

The baseline report compares resident paths within each of 23 common geometry/placement boundaries. It records 248 / 300 correct median pair orders. Of 228 pairs with disjoint reported 95% median intervals, 209 have the correct order; overlapping intervals are inconclusive. No complete ranking passes all latency-quality gates.

The predicted winner matches the observed median winner in 11 / 23 groups. Median selection regret is 0.721649%; worst regret is 95.091808% (269,300 ns) at `ffn-h256-i512-q8-cut256`, where the prediction selects `hybrid/resident` and the observed winner is `cpu/resident`. These are descriptive medians, not validated deployment decisions. Observed winners include 13 CPU joint, 3 CPU serial, 5 GPU joint, 1 GPU serial, and 1 hybrid path, so this grid contains real changes in the winning strategy.

The predeclared near-crossing selector uses profile prediction ratios <= 1.25. Among its 101 pairs, 60 have correct median order; among 48 pairs with disjoint intervals, 34 are correct. Only one pair has accepted latency quality, and its intervals overlap (`ffn-h512-i1024-q32-cut512`, CPU serial versus CPU joint); it therefore does not establish a decisive ranking.

## Instrumentation limits

The diagnostic report has 286 case/path/split rows, each pooling 153 invocations across three processes. Profile instrumented-to-uninstrumented median ratios range from 0.844689 to 1.145390 (median across paths 1.002171). Heldout ratios range from 0.770161 to 1.132329 (median 0.998408). Ratios below one are possible because the timed populations are separate; they are not negative instrumentation costs. No-op trace outer-timer medians range from 300 to 1,800 ns in profile and 300 to 1,900 ns in heldout (median across paths 600 ns for either split). These measured controls are reported separately; no timestamp overhead is subtracted from a target or treated as ground truth.

For the 42 heldout parallel-path groups, median branch-start skew ranges from 9,000 to 17,900 ns, and median join wait ranges from 300 to 303,400 ns. Host completed-work bracket overlap, branch skew, and join wait describe observed host intervals. They do not prove physical GPU kernel overlap, identify an operating-system context switch cost, or validate arbitrary bundle composition. This campaign's preload finishes before compute; copy/compute overlap has a separate protocol. GPU workspace/private allocation and physical capacity remain unvalidated by these chain reports.

## Exact report inputs

All paths are relative to `profiling/chain_calibration/results/campaign-v1/`. SHA256 values bind the JSON inputs used for this note.

| Report | SHA256 |
| --- | --- |
| [analysis/report.json](../profiling/chain_calibration/results/campaign-v1/analysis/report.json) | `e4a1749d2a802a112719beb75f5b9b9d4ae5e7cd620574fdbf4b1817655925f3` |
| [trace-report.json](../profiling/chain_calibration/results/campaign-v1/trace-report.json) | `f8b21fdc437c5c067d9a49ac97094c5263cbe0e9fdd5150d6eca5eaa258c2027` |
| [shape-report.json](../profiling/chain_calibration/results/campaign-v1/shape-report.json) | `31af6fbdf9dec2caac956c0dd8e9846899a78fd8b87f3f23762dbaec4d235b22` |
| [diagnostic-summary.json](../profiling/chain_calibration/results/campaign-v1/diagnostic-summary.json) | `9c48ba93bebb39e5e7cc762a075f6c2165a14114bc0e449e56953c9491e85e6b` |
