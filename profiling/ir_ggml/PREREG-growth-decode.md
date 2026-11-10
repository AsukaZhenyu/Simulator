# PREREG — a decode chain whose KV grows

Written before the campaign was run. Nothing in this file was chosen after seeing a
case's measured makespan.

## Question

`instantiate.py --decode-history chain` prices and executes round `r` of a decode chain
at `visible_history = T + r`: the KV a real generator would have. The frozen table prices
decode at `visible_history` 0/128/512/1024 and nowhere else, so with the frozen table
alone every round after the first is `no_price_at` — the model cannot say what a growing
chain costs. This campaign asks whether, once each visited depth is priced, the model's
per-action prediction (A1) and whole-program prediction (A2) still hold at every round,
or whether the growth itself is where the model breaks.

## Corpus and window

- Corpus: `cases-es` (the 800 declared policies), expanded to **8 chained decode rounds**
  by `expand_decode.py --rounds 8`, which adds rounds to every family and rewrites the
  inter-round dependencies into one chain. The archived pack and `cases-es/` are not
  modified.
- Model: `qwen3-0.6b`, `--tokens 64`, `--decode-history chain`. Round `r` is therefore at
  `visible_history = 64 + r`, `r = 0..7`, so the chain visits depths 64–71.
- Families: `dense` only (a dense model cannot execute a router/expert topology).
- `--cpu-mask 2,4`, as every published table declares.

## Prices for the window

`growth_prices.py` measures the depths the chain visits, on the same protocol as the
table it extends, and hands the result to `run.py --prices`. It never interpolates and
never borrows a neighbouring depth.

- Measured in **one probe process** (`--process-scope per_suite`), so whatever a process
  starts out with is a property of the process and cancels out of the ratios between the
  window's entries.
- The window is 27 CPU cells: `projection`/`attention`/`dense_ffn` at prefill `t=64,h=0`
  and decode `t=1` at each of h=64..71.
- Cells are cloned from the frozen export's own geometry; the tranche extends the depth
  axis only. A `(kind, phase)` the frozen table priced nowhere is refused, because
  pricing it here would give this campaign a placement the rest of the corpus does not
  have.

### Acceptance rule, fixed in advance

| rule | CPU (this table's protocol) | accelerator (its own protocol) |
|---|---|---|
| cv ≤ 0.10 | enters clean | enters clean |
| cv > 0.10 | refused, unless `--allow-unstable` carries it with the CPU export's own `high_observed_variability_cv_gt_0.1` flag | refused |

Carried by the first pass: **`projection` decode h=71, cv 0.148**, flagged in the price
entry. Every other cell entered clean.

The accelerator column of the window is **not** part of this campaign. Two passes over the
same 17 accelerator cells at 1 token gave, at the same key, medians that differ by up to
3.5× (attention h=66: 814 µs vs ~250 µs) with a cv of 0.05–0.30, i.e. the cells were
measured on different device tiers and no single tier could be held for a whole window.
The consequence is declared rather than worked around: only cases whose every action is
priced on the CPU are scored, and the number of cases excluded for an unpriced
accelerator action is reported as a coverage statement.

## Gates, fixed in advance

- **G1 — state.** The control family `dense-compute` must land in `[0.85, 1.20]`
  (the driver's existing rule). Outside it the tranche is a different observation band and
  its readings are not compared.
- **G2 — replicas.** A case whose replica wall-clock cv exceeds 5% is excluded, as in every
  published run.
- **G3 — coverage.** Among scored cases, no decode round may be reported `no_price_at`.
  One such action falsifies the claim that the window covers what the chain visits.
- **G4 — depth.** A1 is read **per round**, not only in aggregate. The model is claimed to
  hold at every round only if the median A1 of `attention` decode units lies inside
  `[0.85, 1.20]` at **each** of the 8 rounds. A pass at round 0 with a failure at rounds
  4–7 is a failure of the depth claim, and would be reported that way.
- **G5 — whole program.** A2 (measured makespan / priced makespan) is reported per case as
  a lower bound with the unpriced copy declaration, and its median is read in the same
  `[0.85, 1.20]` band. A2 is not a fitted number and is not adjusted to pass.

## What would count as a negative result

- Any round whose attention median A1 falls outside `[0.85, 1.20]` while the control
  family is in band: the price is a function of depth in a way this table does not capture.
- Any `no_price_at` among scored cases: the window is incomplete.
- A2 drifting systematically below 1 with the count of rounds: the chain's actions
  interleave in a way the sum of prices does not express (the composition effect this
  corpus already reports for prefill bridges).

All three are reported as findings, not discarded.

## Result — dense column (2026-11-10)

`qwen3-0.6b`, `--tokens 64 --rounds 8`, families `dense`, 672 cases at 3 replicas,
9 minutes of execution. Prices: 27 CPU cells in one probe process
(`C:/tmp/growth/qwen3-0.6b-t64-persuite.json`), one cell carried under
`--allow-unstable` (`projection` decode h=71, cv 0.148). Run document:
`C:/tmp/growth-results/all-dense-r8.json`; report: `growth_report.py` on it.

| gate | verdict | number |
|---|---|---|
| G1 state | PASS | control family median 1.134 over n=13 |
| G2 replicas | PASS | 160 of 672 cases excluded for replica cv > 5% |
| G3 coverage | PASS | no decode round `no_price_at`; 1,376 actions on the device this tranche does not price, declared outside scope |
| G4 depth | **FAIL** | attention decode out of band at h=64 (1.202) and h=66 (1.286); the other six rounds 1.066–1.156 |
| G5 whole program | PASS | A2 median 1.062 [0.936, 1.105] over n=14 fully priced cases |

Both misses have a mechanism, and neither is the growth:

- **h=64 is the first round, and the first round costs more than the second.** Per case,
  attention's measured time at h=64 over h=65 has median 1.057 and p90 1.717 (n=257).
  Round 0 is the first graph shape after the prefill and, in policies that release
  weights, the first re-materialization: an additive cost that a price keyed by
  `(stage, depth)` has nowhere to put. The same additive term explains the p10 of
  h=64 being 1.044 while its median is 1.202.
- **h=66 is a cell of the table, not a property of the chain.** The price at h=66 is
  158.2 µs, −12.2% from the window's own median, while the *measured* cost across the
  eight rounds moves only 6.8% (`growth_report.py`'s level-wander section). `dense_ffn`,
  which also reads no KV, shows the same shape: its prices scatter 17.0% across the
  window against a measured 5.4%, and it is out of band at four of eight rounds for that
  reason. Two stages reading no KV cannot depend on depth, so what varies across the
  window is the instrument's level at the moment each cell was taken.

What the model holds, then, is stated as the measurement it is: attention's *measured*
per-round cost is flat to 6.8% across a KV that grew from 64 to 71 tokens, its A1 is
inside `[0.85, 1.20]` at six of the eight rounds, and A2 for the fully priced subset is
1.062. What it does not hold is a per-depth claim at the resolution of a single round
whose price was taken minutes apart from its neighbours — exactly the scatter the
history-invariant stages make visible.

The `projection` column is reported and not scored: a projection node's measured time
carries its policy's weight movement, which the price does not, so its per-round median
A1 runs 1.50–4.11. The extreme is a single decode projection node in
`pipeline-transfer_constrained-049-demand`: 6,945.2 µs measured against a 199.5 µs price,
in a case whose measured copy time is 6.25 ms and whose copies carry no price. It is a
limiter of the corpus, not of the depth model, and it is left in the table rather than
filtered out — filtering it would raise the median by removing the cases that make the
limiter visible.

## Result — MoE column, the two price tranches (2026-11-10)

`olmoe-1b-7b`, `--tokens 64 --rounds 8`, the same window of eight depths, measured on both
devices so that a case placed either way has a price.

**CPU tranche: 27 of 27 priced.** One probe process for the whole window
(`--process-scope per_suite`, threads 2, `--cpu-mask 2,4`, warmup 10, samples 40), so a
level common to every entry cancels out of the ratios between them. Two cells came back
above the CPU rule: `attention` decode h=69 at cv 0.126 and h=66 at 0.101. Both were
re-measured alone in a second one-process run (`C:/tmp/growth2/retry-attn-olmoe-t64-dir`)
and came back at 0.025 and 0.089, so the export needed no `--allow-unstable` flag:
27 prices over 8 depths in `C:/tmp/growth-moe/olmoe-1b-7b-t64.json`.

**GPU tranche: refused, and the second attempt did not rescue it.** The first pass
(warmup 1000, samples 20, kv f16, graphs disabled — the frozen table's own protocol)
measured 27 cells and 10 of them exceeded the accelerator's cv rule, 0.106 to 0.223. The
ten were re-measured at twice the samples: two came back inside the rule (both
`projection`), eight did not, 0.101 to 0.285. Two independent facts say this is the
instrument and not the depth:

- **the same cell moves between runs by more than any depth effect.** `attention` decode
  h=69 was 266.7 µs in the first pass and 196.4 µs in the retry, −26%; h=71 was 186.7 then
  201.5 µs, +8%.
- **two of the three stages cannot depend on depth at all.** `projection` and `moe_ffn`
  read no KV, yet their cells scatter 285–325 µs and 283–346 µs across a window whose
  only variable is the KV length. The frozen table's own decode rows sit at cv 0.085–0.094
  at h=0/128/1024, so the published prices were already near this boundary; asking for
  eight depths *between* them multiplies the number of near-boundary cells.

Taken together: at one token per step the accelerator's per-stage price is 200–350 µs,
the launch floor this table has recorded before, and the spread of that floor across
neighbouring depths is comparable to the whole depth effect the chain is supposed to
expose. **So the gap the growth tranche closes is closed on the CPU only.** A decode
round placed on the accelerator still files `no_price_at`, and the receipt says so
instead of borrowing a neighbouring depth. One identity detail made the two GPU runs
comparable with the frozen table: the archived probe bytes
(`ggml-dense-layer-cuda-probe.pre-route.exe`, sha256 `d69d616b…`) are what the frozen
export was measured with, and the current build in that directory is a later one
(`7364bcfd…`), so the window ran on a copy of the archived binary rather than on
whatever is on disk.

### The first MoE corpus run is void, and the reason is the replica count

The corpus was first run at `--replicas 1` (800 cases, 1,750 s,
`C:/tmp/growth-results/all-moe-r8.json`). Its control family read **2.94**, so the state
gate fired and the run cannot be scored. The cause is the campaign's own replica rule,
not the corpus: `analyse` reads `replicas = range(1, len(wall_ns))` because **replica 0
pays the CUDA module load** — the comment has said so since the first campaign — and a
one-replica run leaves nothing to read but that replica. The arithmetic is visible in the
units: single prefill actions on the accelerator measured 104.8 ms against a 0.36 ms
price (290×), 21.4 ms against 0.31 ms (69×), while the same cases' CPU actions sat at
0.96–1.01. The 11-09 archive ran this corpus at **four** replicas, which is why its GPU
prefill units read 1.4–2.2.

Two control cases re-run at four replicas gave control 1.029; the 16 control cases gave
**1.164** over n=36 (4 of 16 excluded for replica cv > 5%). The corpus was then re-run at
`--replicas 4` in full (`C:/tmp/growth-results/all-moe-r8-rep4.json`), and that is the run
this column reports. `run.py` now refuses `--replicas 1` when any selected case has an
accelerator node, so the mistake cannot be made silently a second time.

## Result — MoE column, the corpus (2026-11-10)

`olmoe-1b-7b`, `--tokens 64 --rounds 8`, families `all` (800 cases), **4 replicas**,
2,070 s. Prices: `C:/tmp/growth-moe/olmoe-1b-7b-t64.json` (44 cells: 6 prefill, 38 decode;
19 of the window's 24 decode cells per device). Run document:
`C:/tmp/growth-results/all-moe-r8-rep4.json`; report: `growth_report.py` on it.

| gate | verdict | number |
|---|---|---|
| G1 state | PASS | control family median 1.093 over n=42 |
| G2 replicas | PASS | 68 of 800 cases excluded for replica cv > 5% |
| G3 coverage | PASS | no decode round `no_price_at` on the CPU; 1,488 actions on the device axis declared outside scope |
| G4 depth | PASS | attention decode 0.965–1.170 across the eight rounds |
| G5 whole program | PASS | A2 median 0.938 [0.918, 2.153] over n=45 fully priced cases |

- **The depth claim holds on the MoE column.** Attention decode, the unit that reads the
  KV, is inside `[0.85, 1.20]` at every round: 1.170 at h=64 and then 0.965–1.108. The
  first-round excess the dense column reported is here too — this round costs 1.322× the
  second measured on attention alone (n=532) — but on this corpus it stays under the band
  edge instead of crossing it.
- **`moe_ffn`, priced on the CPU and the executor's batched dispatch, holds at 0.857–1.093
  per round** (median 0.920 overall), i.e. the dispatch-shaped price built earlier this
  campaign is accurate across a growing KV as well.
- **A2 sits at 0.938, below 1**: among the 45 cases whose every action carries a price at
  its own shape, the frozen table over-predicts the makespan by 6.6% at the median. That
  is a *lower* bound — copies carry no price and the 1,488 unpriceable actions were
  excluded, both of which add measured time without adding predicted time — so the true
  ratio is above 0.938, not below it.
- **`projection` is reported and not scored**, exactly as on the dense column: its decode
  median is 1.199 and it is out of band at five of eight rounds, because a projection
  node's measured time carries its policy's weight movement, which the price does not.
  Its measured spread inside the window (96.1%) exceeds its price spread (73.0%), so here
  the scatter is the corpus, not the instrument.
- **Coverage, stated as the number it is**: 1,488 actions were placed on a device the
  tranche could not price (`no_price_at on gpu` 1,168; `split_across_machines` 320), and
  the measured copy time those cases pay has no table price (median 36.6 ms per case).
  Five of the window's 24 decode cells per device are missing on the accelerator because
  the GPU tranche refused them; a case placed there files a gap instead of borrowing the
  neighbouring depth, which is G3's whole content.

The level the table can resolve is the same finding as on the dense column: across the
window the *prices* wander more than the measurement does (attention price spread 33.3%
against a measured 23.3%), so a single round's A1 moves with the instrument's level at the
moment that cell was taken. What is claimed here is per-round accuracy inside the band,
not a per-depth curve.

## Where this result is cited

`paper-write/第八教学周/output/DAC_Draft_Week8_2026-10-30_IEEE_v69.docx`, §V-A
(`paragraph 5E000005`), one sentence: attention decode stays inside 0.85 to 1.20 at all
eight rounds on the MoE model and at six of eight on the dense one, the two misses being
the first-round additive cost and the h=66 cell whose price wandered. The paper does not
carry the accelerator column's refusal, nor A2 (0.938 MoE, 1.062 dense): both are reported
here and neither fits the six-page budget. `v68` is untouched and remains the previous
submission candidate.
