# High-level v3 latency diagnosis, 18 October 2026

The complete saved v3 report has 14 cases and 42 heldout processes, with no
execution failures. Numerical, graph, action and copy checks passed, but **zero
cases satisfy the complete acceptance gates**. Every source ledger fails source
stability. Pooled absolute prediction errors range from 2.15% to 152.95%; the
report's separate maximum of 161.51% is a single-process error. Predictions exceed
the pooled observation in four cases and fall below it in ten.

This note uses saved JSON only. It performs no replay, numerical evaluation,
fitting, sample removal, new measurement or gate changes. It is a descriptive
diagnosis, not an independent protocol-verifier receipt or a causal attribution.

## Neighboring cuts

Times below are milliseconds. Signed error is `(prediction / observation) - 1`.
The observation is the saved median across all 93 heldout samples per case.

| Geometry / FFN cut | Frozen prediction | Pooled observation | Signed error | Heldout process medians |
|---|---:|---:|---:|---|
| H256 q32 / 252 | 15.0203 | 5.9381 | +152.95% | 5.7437, 6.0194, 5.9381 |
| H256 q32 / 256 | 5.1401 | 14.6835 | -64.99% | 15.1521, 5.3415, 14.6096 |
| H256 q32 / 260 | 5.1299 | 5.4456 | -5.80% | 5.9255, 5.3575, 5.3037 |
| H128 q16 / 124 | 2.9662 | 8.5162 | -65.17% | 8.7654, 3.1859, 8.6553 |
| H128 q16 / 128 | 2.5617 | 3.0174 | -15.10% | 8.5615, 3.0174, 2.6647 |
| H128 q16 / 132 | 2.6559 | 7.7324 | -65.65% | 8.1286, 7.7370, 2.6699 |

For each of these six cases, the saved prediction equals the sum of its frozen
component prices exactly. The largest price is GPU `attn.pv`, action 14. It
precedes the FFN partition; neighboring cuts have the same declared attention
geometry and device placement. Thus the following price variation is not a
change in the declared attention geometry caused by the FFN cut.

| Geometry / cut | Frozen PV price | Three source-process PV medians | Between-process CV |
|---|---:|---|---:|
| H256 / 252 | 10.1085 | 10.5519, 10.1612, 3.0298 | 0.4369 |
| H256 / 256 | 3.1104 | 3.1233, 3.3052, 2.6980 | 0.0836 |
| H256 / 260 | 2.9838 | 3.4572, 2.6952, 3.0208 | 0.1021 |
| H128 / 124 | 1.6264 | 1.6022, 5.5431, 1.5856 | 0.6397 |
| H128 / 128 | 1.5778 | 1.4408, 1.6143, 1.5780 | 0.0484 |
| H128 / 132 | 1.6252 | 5.1856, 1.6106, 1.5837 | 0.6056 |

PV alone accounts for 67.30% of H256/cut252's prediction and 70.83% / 72.04%
of its prediction increase over cut256 / cut260. Its first two source-process
PV CVs are only 0.0322 and 0.0747, but the third is 0.3523 and the process-median
CV is 0.4369. An individually consistent slow process therefore does not establish
a stable cross-process cost.

Other saved prices contribute: H256/cut252's 45 COPY components sum to 2.2893 ms,
versus 0.8753 / 0.8393 ms for its neighbors. CPU left-shard down projection costs
0.6549 ms versus 0.1229 / 0.2420 ms. The latter changes its actual matrix width,
so an alignment or kernel-shape effect remains possible; it accounts for only
0.5320 ms of the 9.8802 ms prediction increase over cut256. GPU attention output
also costs 0.2742 ms versus 0.1039 / 0.0998 ms. This broader co-variation argues
against attributing the full jump to one FFN partition shape.

H128's frozen PV prices are close despite one slow source process in cut124
and cut132. Their heldout cohorts contain two slow processes each; cut128 has
one. H256/cut252 shows the opposite pattern: two slow source PV processes but
three much faster whole-program heldout medians. These observations support the
**hypothesis** that process-dependent performance regimes and differences between
isolated-component repetition and complete mixed-action replay dominate the
large errors. The records do not establish a particular cause such as clocks,
power, WDDM scheduling, cache state or an application. Heldout data was not used
to choose a correction. Whole-program timing has no per-action attribution, so
it cannot prove that PV itself changed by the same amount during heldout.

Instability is widespread, not just one outlier: H256/cut252 has 53, 50 and 53
source components failing within-process CV out of 61; 37 fail between-process
CV. Even H256/cut260, whose pooled error is 5.80%, has 52, 47 and 49 within-process
failures and one between-process failure. The unchanged gates correctly prevent
calling these paths accurate or treating their apparent ordering as a validated
strategy-ranking result. Summing marginal component medians also need not equal
the median of a complete execution when costs vary together.

No native memory/capacity accuracy, complete CUDA scratch accounting, original
model-kernel reproduction or resolved runtime-loader-path claim follows from
these results.

## Exact saved inputs

Paths below are relative to `profiling/highlevel_accuracy/runs/representative-v3`.
These are SHA-256 hashes of file bytes, distinct from embedded canonical IDs.

| File | SHA-256 |
|---|---|
| `campaign.json` | `59851b9298cb1b36583013550103152737f0a0daa79789b28a7e11508dde95a0` |
| `all_profiles_frozen.json` | `c31f124c306c81455a77bc700bf619f6b12e2e2606d96b81239eed205e6ec1cd` |
| `report.json` | `e65f28bf3c71ce18545d8d0d1c52bcb93606dd8d314cdc5ec0e5a11a22670f36` |
| `h256-q32-history0-attention-ffn-cut252/frozen_profile.json` | `fcd6f9904ff6936269e26ceb6341465cd47ee96e0d42ce527d3e3086f7fe9e82` |
| `h256-q32-history0-attention-ffn-cut256/frozen_profile.json` | `94620402c57f47cd388755f35c585276c6f5e73264c8827db9bf9622b55b21e0` |
| `h256-q32-history0-attention-ffn-cut260/frozen_profile.json` | `0ff83d52745415c4ca2936b08c440b2e3b309e2a612ed6af004864f1f3ebef2a` |
| `h128-q16-history0-attention-ffn-cut124/frozen_profile.json` | `9471f678fbb9799c73685d9387ad0ee955c4950aadde8cb18d49de1da1fa45d3` |
| `h128-q16-history0-attention-ffn-cut128/frozen_profile.json` | `2e25515cd5d46ab0fa3c0ae31eb66017e5221208c2fb4ed42edeac951290943e` |
| `h128-q16-history0-attention-ffn-cut132/frozen_profile.json` | `b95e1bd4571299be8a0acd5f38eba7bbb69db26f78ad1c9d232257b6aeedc8f9` |

Campaign canonical ID: `ad0451b473dff24a817077291c806b899f967dcdfbcb83cabf28f89ab76560d2`.
Barrier canonical ID: `b70fb4eada9127c60930d02e0953d34f1fd79eeae474441a0052596fbe392e0f`.
All-profile freeze time: `2026-10-18T04:31:36.566074+00:00`.
