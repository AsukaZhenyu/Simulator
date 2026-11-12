# PREREG — one working-set exponent in the dense column's frozen prices

Written before the cohort that measures under it. It changes how `freeze` turns
profile prices into table prices, and nothing else in the dense column.

## Why

On 2026-11-11 the dense column ran twice on the same host and instrument: run A
passed 15/18 held-out cases under `context_static` and 17/18 under
`history_conditional`; run B passed 14/18 and 17/18. The failures were not
scattered — they were the shallow end of the depth ladder (prefill L4/L8), where
the residual reached 12.01 % (A) and 17.41 % (B).

`fit-consistency.json` says where that comes from. A coordinate is pooled over
every profile program that produces it, and the same coordinate is priced up to
1.83× apart across programs. The programs differ in the bytes they touch before
the timed region (the depth ladder's working set spans several ×), so the pooled
price is wrong for every program it is applied to, by an amount that grows as the
program's working set departs from the average.

Offline, on the profile half of that cohort only, one exponent with one level per
coordinate takes the reconstruction error of the frozen table from 29.04 % worst
case to 7.68 %, with 18/18 programs inside 10 %. That is a reconstruction
computed while the held-out durations were visible; it is not the campaign metric
and it is not a result.

## The rule, fixed before the run

* working set `ws(p)` = `manifest.programs[p].initial_assets.bytes`.
* reference `ws_ref` = median of `ws` over the profile programs.
* `log price(h, p) = log level(h) + theta * log(ws(p) / ws_ref)`, with one free
  intercept per coordinate and one shared slope `theta`, estimated by the
  within-coordinate demeaned least squares of `log price` on `log ws`, over
  coordinates that carry samples in at least two profile programs, using the mean
  per-action price of each (coordinate, program).
* `level(h) = exp(mean over programs of (log price(h, p) - theta * log(ws(p)/ws_ref)))`
  — the geometric mean of the divided prices. A coordinate measured in a single
  program is divided by that program's factor.
* Every cell stored in the table is `level(h)`, rationalised by
  `limit_denominator(10**9)` because the table accepts an exact rational only.
* A held-out prediction is `(sum of the table prices over the program's actions)
  * (ws(p) / ws_ref) ** theta`, one factor per program, applied to both
  providers alike.
* The logical history that constructs the conditioned H2D cells is advanced with
  the same levels the predictor commits, so every conditioned cell is looked up
  under the conditions it was measured under.
* `theta` is fitted on the profile programs only; neither a target duration nor
  the held-out split is read anywhere in the fit.

## Refusals, not fallbacks

`freeze` stops if any of these holds. No alternative exponent is tried, and in
particular no exponent is copied from another host or from the offline script.

* fewer than two profile programs;
* every profile program has the same working set (the slope is not identifiable);
* fewer than ten coordinates are shared by two profile programs (the slope would
  be one program's accident);
* the estimate is not finite.

## What does not change

Table schema, bucketing and logical initialisation; the condition fields; both
providers (`context_static`, `history_conditional`); the 18 held-out cases and
the four processes per case; the trimmed-10 estimator; the two gates (10 % on the
process mean, 20 % on the p90 block); the freeze barrier and its hashes; the
profile/held-out split.

## What this cannot fix

* It is one exponent for every coordinate and one factor per program. A program
  whose overshoot is not proportional to its working set cannot be represented.
* The ladder's largest term is L32. A law fitted over this ladder says nothing
  about depths below L4 or above L32; L24 and L48 are not measured here.
* It says a coordinate costs more in a program that touches more bytes at its
  start. It does not say why, and no mechanism is claimed.

## What would count as a negative result

* Any held-out case that fails a gate under `history_conditional`, the provider
  the paper reports. Reported as a failure, not as a reason to change `theta`.
* A case whose signed error flips sign against the previous cohort — the law
  overshooting where pooling undershot. Reported per case.
* A fitted `theta` outside `[0, 0.5]`. Reported with the cohort; no alternative
  fit is tried afterwards.
* If the arm does not pass both gates on all 18 cases, it does not replace the
  previous cohort's numbers; both cohorts are reported.
