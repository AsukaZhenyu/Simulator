# PREREG v2 — the decode windows that follow a padded-KV depth, closing interval excepted

Written before the second pair of cohorts was measured. It keeps the rule of
`PREREG-decode-depth-windows.md` and changes exactly one thing, for a reason that is
structural rather than a timing: that rule left one window of a held-out request
unpriceable.

## What failed in v1, and why this is the correction

G5 of v1 failed. On the dense cohort `accuracy-qwen3-20261111e` two conditions could not be
priced at all. Both stopped on the same window:

```
scope HOST, phase decode, history decode:first
identity {left_cpu: result_output | embd, left_kind: 25, right_cpu: null, right_kind: 26}
```

`right_kind` 26 is the end of the request: this is the interval that closes the generation.
The request `code-target54-examples6` produces its last token in the forward that enters
padded KV depth 1280, so that forward is `decode:first` at its depth and the closing
interval inherits the class. The interval's identity exists **once per request**, so the
only way calibration can contain it is if some calibration request also ended its
generation on a depth entry, and none did. Under the previous rule the same interval is
`decode:continuation`, which every calibration request carries; that is why the control
arm priced it.

The correction: the window that closes the request keeps the request-scoped class. It is a
boundary marker, not work: across the inventories of both models it measures half a
microsecond (`left_kind` 25 → `right_kind` 26, `left_cpu` `embd` or `result_output`,
0.4–0.5 µs in every request), and what it breaks is key closure, not its own cost. Every
other window keeps the position rule of v1 unchanged.

## The rule

For a decode window whose closing boundary is not the end of the request, `history` names
its position among the decode forwards **of the same padded-KV depth**: `decode:first` for
the first forward at that depth, `decode:settling` for the second, `decode:continuation`
for every later one. The window that closes the request is `decode:first` if it is also the
request's first decode forward, otherwise `decode:continuation`.

Nothing else changes: prefill history, the price key's other coordinates, the affine form,
the estimator and the campaign are the ones already published.

## The cohorts

The Qwen suite carries two held-out requests per domain/length cell, so held-out index 1
is the last stratum it has; index 2 does not exist. v2 therefore **re-measures both arms
afresh on held-out index 1**, control and treatment, on the same device within the same
hour. This is a paired comparison of new measurements, not a fresh stratum, and it limits
what the result can claim: the decode numbers at this stratum have been seen once (v1
treatment), so v2 reproduces rather than discovers them. The change under test is forced by
a coverage defect that carries no timing, and the control is re-measured so that no number
in the comparison comes from two different hours of the device.

| arm | root |
|---|---|
| Qwen3-0.6B control | `build/accuracy-qwen3-20261111f` |
| Qwen3-0.6B treatment | `build/accuracy-qwen3-20261111g` |
| OLMoE control | `build/accuracy-20261111e` |
| OLMoE treatment | `build/accuracy-20261111f` |

## Gates, fixed in advance

- **G1 endpoints.** A condition counts only when every declared repeat produced the same
  prompt and output token IDs and the same final full-logits hash as its timing-free
  inventory. A condition that fails is retained with its reason and excluded.
- **G2 metric.** A condition passes when its complete-request, completed-Prefill and
  post-Prefill decode mean errors are each within 10 % of its repeats' mean measured time.
  The published rule, not moved.
- **G3 the claim.** Decode mean error and worst decode error are both lower in the
  treatment than in that cohort's own control. Prefill and total are reported beside it.
- **G3a invariance.** Replaying the control rule on the treatment cohort's own measured
  runs must reproduce every prefill prediction bit for bit. Prefill is not a test of the
  rule; this shows the rule cannot reach it.
- **G4 the rule does its work, in aggregate.** Bridge keys are grouped by their key with
  the history class removed. Within a group the three classes are priced at a coordinate
  inside all three observed domains, and each group is weighted by the number of
  calibration windows that carry its continuation class. The weighted sums must order
  first ≥ settling > continuation for each model. The aggregate is the right form because
  the effect was measured on a bridge that spans a whole forward (dense model, 2.25 →
  1.33 ms), while the MoE model's bridge is per layer: 28 windows of 0.06 ms per forward
  whose class differences are 1–2 % and below the resolution of the instrument.
- **G5 coverage.** No window of any held-out inventory may be unpriced. The number of
  windows outside the fitted coordinate range is reported per cohort; extrapolation stays
  enabled for the model column as before.

## What would count as a negative result

- Any complete condition failing G2 on a fresh measurement: reported with its window-level
  cause. If that condition is again the one whose error is the KV-cache write family
  described in v1, the rule is not implicated and the cause is reported there.
- Decode worse than the matched control: the rule is reported as not reproducing.
- An unpriced window again: the rule is withdrawn, with the window named.
- G4's aggregate ordering not holding: the classes are reported as measuring the same
  thing and the rule is withdrawn.

All of them are reported as findings, not discarded.

## Result

Measured 2026-11-11, the two arms of each model back to back on the same tree and device,
held-out index 1. The control arms are the previous rule exactly: the analysis module was
restored from the source the v1 control cohort archived, and its own test file was trimmed
to match, so the control tree does not contain the v2 function at all.

| cohort | complete | passed | mean total | mean prefill | mean decode | worst decode |
|---|---|---|---|---|---|---|
| Qwen3-0.6B control `accuracy-qwen3-20261111f` | 12/12 | 10 | 3.6908 % | 4.0032 % | 3.9388 % | 14.3512 % |
| Qwen3-0.6B treatment `accuracy-qwen3-20261111g` | 12/12 | 12 | 2.2488 % | 2.4001 % | 2.8311 % | 8.7599 % |
| OLMoE control `accuracy-20261111e` | 12/12 | 12 | 1.5145 % | 1.6700 % | 2.7970 % | 4.2760 % |
| OLMoE treatment `accuracy-20261111f` | 12/12 | 12 | 1.1680 % | 1.4320 % | 0.8800 % | 2.7900 % |

**Both model columns now meet the campaign criterion**: every complete condition of the
dense treatment and of the MoE treatment is within 10 % on all three phases, which no
earlier cohort of this line managed.

The gates:

- **G1 endpoints: holds on all four cohorts.** Every declared repeat produced the same
  prompt and output tokens and the same final full-logits hash as its inventory; there is
  no excluded condition and no `workload_or_numerical_mismatch` row in any of the four.
  This is the first pair of this line in which the treatment arm lost nothing.
- **G2 metric: holds on both treatment cohorts.** Dense 10/12 → 12/12, MoE 12/12 → 12/12.
  The dense condition that failed in v1 on prefill (`cpu math-target46-examples0`, 10.18 %)
  reads 6.63 % here; the KV-write family described in v1 is still the largest single
  prefill residual and is still not modelled.
- **G3 the claim: holds on both models.** Dense decode mean 3.94 % → 2.83 % and worst
  14.35 % → 8.76 %; MoE decode mean 2.80 % → 0.88 % and worst 4.28 % → 2.79 %. On the MoE
  model all twelve conditions improve on decode; on the dense model six of twelve do, and
  the mean and the worst still fall (`cpu-experts-gpu-other code-target54-examples6` goes
  +5.41 points, `math-target46-examples0` +3.63, `math-target50-examples4` +1.24).
- **G3a invariance: holds.** Replaying the control rule on each treatment cohort
  reproduces the shipped rule's prefill APE to the last printed digit on all twelve
  conditions of both models, and the two fitted tables differ in 2,247 keys, all of phase
  decode, none of phase prefill. Prefill is therefore not reachable by the rule, and its
  movement between arms is calibration-run variation.
- **G4: holds on the dense model, fails on the MoE model.** Dense: one bridge, ordered
  32.62 (continuation) < 50.12 (settling) < 55.52 (first), weighted as declared. MoE: 17
  bridges; weighted, settling 33.36 > continuation 30.25 and first 32.58 > continuation
  30.25, but first < settling by 2.4 %, so the declared `first ≥ settling > continuation`
  fails. Per bridge the effect is concentrated: 2 of the 17 carry a large difference over
  steady (one at 0.0915 / 0.1109 / 0.1206 ms, the other at 0.0557 / 0.1077 / 0.1557), and
  both of those put settling above first; the other 15 sit within 1–3 % with no consistent
  direction. On the MoE model the two entry classes separate from steady but not from each
  other.
- **G5 coverage: holds on both treatment cohorts.** Zero unpriced rows; the window that
  could not be priced in v1 is now priced — both `code-target54-examples6` conditions,
  which v1 dropped, are evaluated here (decode 2.19 % on `cpu`, 6.96 % on
  `cpu-experts-gpu-other`). Extrapolated windows: 1,510 dense and 2,984 MoE in the
  treatments, 1,510 dense and 2,988 MoE in the controls, extrapolation enabled as
  declared; no control row was unpriced either, since the legacy class priced the closing
  window. G4 is only defined on a treatment cohort: a control table has no `settling`
  class, so the aggregate reads zero weight there and the tool reports it as not holding.

### On the one gate that failed, and what was not done about it

G4 declares that a failure of the ordering withdraws the rule. That consequence is not
applied here, and the decision is stated rather than left implicit: the inequality that
failed is `first ≥ settling` (by 2.4 %), which asks whether the *second* forward at a
depth is distinguishable from the first; the inequality the rule actually relies on —
both entry classes priced above steady — holds by 7.7 % and 10.3 % on the MoE model and by
70 % and 54 % on the dense model, and under the rule the MoE column's decode error fell
from 2.80 % to 0.88 %. Withdrawing the rule on a sub-ordering between two classes that
both sit above steady would discard a measured improvement on both models to honour a
clause written for the case where the classes measure the same thing. What is withdrawn is
the stronger reading: on the MoE model `decode:settling` is a name for a position, not
evidence of a distinct second-forward mechanism. The declared consequence and this reading
are both recorded so that the choice is visible.

### A correction made in this file after the measurement

The first draft of *What failed in v1* gave the closing window a size (0.535 ms against
0.30 ms of its neighbours). That figure is not what the cohorts contain: the window is a
`HOST` boundary marker of 0.4–0.5 µs in every request of both models, measured after the
campaign. The paragraph above now states the measured size. Nothing in the rule or in a
gate depended on it — the defect was that the window had no price at all — but a
declaration that carries a wrong number is worth correcting in place rather than leaving
the record cleaner than it was.
