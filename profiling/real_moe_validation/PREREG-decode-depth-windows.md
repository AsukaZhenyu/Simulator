# PREREG — the two decode windows that follow a padded-KV depth

Written before either fresh cohort was measured. Nothing in the rule below was chosen
after seeing the fresh cohorts' timings; the evidence that motivated it is the two
finished cohorts named under *Why the rule exists*, and both are reported whatever the
fresh cohorts say.

## Why the rule exists

The model column prices a decode bridge window with one affine function per key:
`intercept + per_query·q + per_pair·q·KV + per_byte·payload`. `q` is 1 and the padded
KV depth is a coordinate on that line. On the dense model (`Qwen3-0.6B`, cohort
`accuracy-qwen3-20261111c`, placement `cpu-experts-gpu-other`) the measured cost of the
whole GPU bridge window of one decode forward, at the same key, is:

| request | decode forwards, in order | padded KV |
|---|---|---|
| `code-target29-examples0` | 2.130, **1.899**, 1.269, 1.268, 1.268, 1.267, 1.266 | 256 |
| `dialogue-row3-turns7` | 2.256, **2.012**, 1.332, 1.331, 1.336, 2.259, 2.032 | 1024 → 1280 |
| `code-target45-examples6` | 2.260, **2.206**, 1.574, 1.575, 1.577, 1.578, 1.580 | 1280 |
| `math-target38-examples6` | 2.250, **2.215**, 1.536, 1.534, 1.547, 2.264, 2.240 | 1024 → 1280 |

(ms; the bold entry is the second decode forward.) Two facts follow. First, the first two
decode forwards cost 40–70 % more than the same work two forwards later. Second, the
excess reappears **when the padded depth changes inside a generation** (`row3` at 2.259,
`math38` at 2.264, both on entering 1280), and it does not appear when a request starts in
a depth (`code-target45`, whose whole decode is at 1280, is flat at 1.575 after its first
two forwards). So the position that matters is *within a padded-KV depth*, not within the
request. `The OLMoE cohort of the same day shows the same shape (1.765, 2.274 at the start
of `dialogue-row59-turns7`, and 1.803, 2.208 on entering 1280).

The price key carries `history = decode:first | decode:continuation`, and `first` means the
request's first decode forward only. The second forward of a depth, and both forwards after
a depth change, therefore share one key with the steady state, and one affine price is
fitted across a class whose members differ by 70 %. On the dense cohort that price is
1.72 ms per window where the steady windows measure 1.44; `dialogue-row11-turns9` decode is
predicted 15 % high by that alone.

## The rule

For a decode forward, `history` names its position among the decode forwards **of the same
padded-KV depth**:

- `decode:first` — the first decode forward at that depth,
- `decode:settling` — the second decode forward at that depth,
- `decode:continuation` — every later one.

Prefill history is unchanged (`prefill:first` / `prefill:continuation`). This redefines
`decode:first`, which previously meant the request's first decode forward; a request whose
generation stays in one depth keeps exactly one `decode:first` and one `decode:settling`.

The rule is a statement about a measured position in a declared coordinate, not about a
mechanism. The leading mechanism hypothesis is that the device's execution graph is created
per padded depth and the first two forwards pay its creation; the probe refuses to run with
graphs disabled, so the hypothesis is reported and not claimed.

## The cohorts

Two fresh model-column cohorts are measured after this file, each with its own calibration
and its own held-out window, at held-out stratum **index 1** (index 0 is the stratum the
previous cohorts used on this device):

- `Qwen3-0.6B` (dense, no expert block), `--model-id qwen3-0.6b`, suite `suite-v1`.
- `OLMoE-1B-7B` (the published MoE model column), `--model-id olmoe-1b-7b`, shipped suite.

Each is run **twice** on the same tree and the same device: once with the previous
`history` rule (control) and once with the rule above (treatment). Both runs archive their
own executed sources in the cohort, so the two arms are distinguishable from the artifacts
alone. Only the analysis differs; the probe binary, the request files, the model file and
the measurement protocol are identical.

## Gates, fixed in advance

- **G1 endpoints.** A condition counts only when every declared repeat produced the same
  prompt/output token IDs and the same final full-logits hash as its timing-free inventory.
  A condition that fails is retained with its reason and excluded from the mean, as in every
  published run.
- **G2 metric.** A condition passes when its complete-request, completed-Prefill and
  post-Prefill decode mean errors are each within 10 % of its repeats' mean measured time.
  This is the published rule and it is not moved.
- **G3 the claim.** On each fresh cohort every complete condition passes, and no phase's
  mean error over complete conditions is worse than in that cohort's own control arm.
- **G4 the rule does its work.** For every key present in both arms, the `settling` and
  `first` prices must exceed the `continuation` price at the same coordinates; a rule that
  prices `settling` at or below steady would be fitting noise and the claim is withdrawn.
- **G5 coverage.** No window of a held-out inventory may be unpriced, and the count of
  windows outside the fitted coordinate range is reported per cohort (extrapolation is
  enabled for the model column, as before).

## What would count as a negative result

- Any complete condition failing G2 on a fresh cohort: the rule is reported as not
  sufficient, with the failing condition and its window-level cause.
- Any phase's mean error rising against the matched control: reported as a regression, and
  the previous rule remains the published one.
- A `settling` price at or below `continuation`: the two classes would be measuring the
  same thing and the rule is withdrawn.

All three are reported as findings, not discarded.

## Result

Measured 2026-11-11. Both fresh cohorts at held-out index 1, control and treatment run
back to back on the same tree and device. Equivalence of the shipped analysis with
the exploratory one was checked before the campaign: replaying cohort
`accuracy-qwen3-20261111c` with the shipped rule reproduces 3.013050% total (the
exploratory number) and with the previous rule 3.099483% (that cohort's published
number); the published OLMoE cohort replays 1.4111% and 1.4164%.

| cohort | complete | passed | mean total | mean prefill | mean decode | worst decode |
|---|---|---|---|---|---|---|
| Qwen3-0.6B control `accuracy-qwen3-20261111d` | 12/12 | 11 | 3.1396 % | 3.0497 % | 4.5854 % | 14.8940 % |
| Qwen3-0.6B treatment `accuracy-qwen3-20261111e` | 10/12 | 9 | 3.4204 % | 3.9133 % | 2.6018 % | 7.6403 % |
| OLMoE control `accuracy-20261111c` | 12/12 | 12 | 1.6721 % | 1.5305 % | 3.6679 % | 8.6244 % |
| OLMoE treatment `accuracy-20261111d` | 11/12 | 11 | 1.5420 % | 1.6921 % | 1.6800 % | 3.0564 % |

The decode error the rule was written for is removed on both models: the dense model's
only failing condition goes from 14.89 % to 7.64 %, and OLMoE's decode mean from 3.67 %
to 1.68 % with its worst condition from 8.62 % to 3.06 %. Measured per forward on the
dense model, `dialogue-row9-turns3` reads 2.249, **1.981**, 1.327, 1.325, 1.331, 1.332,
1.333 ms across its decode forwards: the old rule priced the second forward with the
steady state, the new rule prices it in its own class.

The gates, as declared:

- **G1 endpoints: fails on both treatment cohorts.** The dense treatment lost two
  conditions (four rows) and the OLMoE treatment one condition (two rows). These are not
  the same failure. The dense rows are *unpriced* — see G5. The OLMoE rows have identical
  prompt and output token IDs and a different final full-logits hash, the intermittent
  numerical variant already recorded for this device; the control arm of the same
  campaign had none.
- **G2 metric: fails on the dense cohort.** Nine of ten complete conditions pass. The
  failing one is `cpu math-target46-examples0` prefill at 10.18 %, where the control arm
  read 8.53 % on the same condition — a prefill condition the rule does not touch.
- **G3 the claim: fails on both cohorts**, on prefill only. Dense prefill mean 3.05 % →
  3.91 %, OLMoE 1.53 % → 1.69 %. Decode and, on OLMoE, total improve. Prefill keys are
  identical under both rules, so this is the spread between two calibration runs, not a
  consequence of the rule; the same-cohort replay above reproduces prefill to the last
  digit under both rules. The gate as written does not distinguish the two, and it is
  reported as failed.
- **G4 the rule does its work: holds on the dense model, fails on OLMoE.** The dense
  treatment has one bridge key and it orders 1.323 < 2.011 (settling) < 2.139 (first).
  OLMoE's decode bridge is split per layer, 28 windows of about 0.06 ms per forward, and
  16 of its 17 keys order the three classes within 1–2 % of each other with no consistent
  direction; the decode gain on OLMoE therefore does not come from the bridge keys, and
  where it comes from is not established by this gate.
- **G5 coverage: fails on the dense cohort.** Two conditions are unpriceable, both at the
  same window: the request's closing interval (`left_kind` 25 → `right_kind` 26,
  `left_cpu` `result_output` and `embd`). `code-target54-examples6` ends its generation on
  the forward that enters padded KV depth 1280, so that forward is `decode:first` and the
  closing interval inherits the class; the interval's identity exists once per request,
  and no calibration request ever produced it under `decode:first`. The identity is the
  same under the old rule with `decode:continuation`, which every calibration request
  carries, which is why the control arm prices it.

### Where the dense column's remaining prefill error lives

Not a consequence of the rule, and measured separately. On the dense control cohort the
`cpu` placement's prefill residual is almost entirely the KV-cache write windows
(`cache_k*`/`cache_v*`, `GGML_OP_SET_ROWS` = 42), which the classifier leaves
`other_compute` and therefore prices as an intercept only. For
`cpu math-target46-examples0` those windows measure 12.75 ms and are predicted 22.05 ms
(+72.9 %); every other prefill window of the same request is +1.89 %. Across the six `cpu`
conditions the family swings from −31 % to +73 % while the rest stays within ±3 %. The
cause is visible in the calibration samples themselves: one key of that family measures a
median of 0.45 ms with a minimum of 0.017 ms and a maximum of 3.70 ms, and the fitted
intercept is the mean of that spread, so the price is neither the cold nor the warm
response. The window is a row scatter whose cost depends on page residency of the cache
region, a state the probe does not declare; this file does not model it and no price was
changed on account of it.

### What follows

The rule is not sufficient as declared: it leaves a key unpriceable at the request's
closing interval. The successful part of the rule is kept and the closure defect is
corrected in `PREREG-decode-depth-windows-v2.md`. That file declares held-out index 1 with
both arms re-measured: the suite carries two held-out requests per domain/length cell, so
index 1 is the last stratum it has and the v2 cohorts are new measurements of a seen
stratum rather than a fresh one.


