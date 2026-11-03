# Prediction-versus-measurement verdict (P1 / P2)

An offline, hardware-free judge for two claims that can be made from already
collected numbers:

* **P1 crossover** - where the predicted ranking of configurations flips
  between adjacent conditions, and whether the measured ranking flips there too.
* **P2 ranking fidelity** - pairwise agreement, Kendall's tau-b, top-1 hit rate
  and the relative regret of the predicted choice.

The tool never launches a GPU, benchmark or simulator. It reads one JSON file,
validates it strictly, and prints a human-readable report or `--json` output.
Only the Python standard library is used.

## Files

| File | Purpose |
|---|---|
| `schema.py` | input contract, strict validation, tolerance/ordering arithmetic |
| `p1.py` | crossover scan and verdict (`CONFIRMED` / `NOT_CONFIRMED` / `INCONCLUSIVE`) |
| `p2.py` | pairwise agreement, tau-b, top-1 and regret |
| `cli.py` | command line front end |
| `__init__.py` | public API |

## Input

```json
{
  "schema": "verdict-input-1",
  "title": "capacity crossover, cpu vs hybrid",
  "conditions": [
    {
      "label": "8GB",
      "value": 8,
      "unit": "GB",
      "configs": [
        {"name": "cpu", "predicted_latency_ns": 12000000, "measured_latency_ns": 12400000},
        {"name": "hybrid", "predicted_latency_ns": 9000000, "measured_latency_ns": 9600000}
      ]
    },
    {
      "label": "16GB",
      "value": 16,
      "unit": "GB",
      "configs": [
        {"name": "cpu", "predicted_latency_ns": 12000000, "measured_latency_ns": 12300000},
        {"name": "hybrid", "predicted_latency_ns": 13500000, "measured_latency_ns": 12900000}
      ]
    }
  ]
}
```

| Field | Required | Meaning |
|---|---|---|
| `schema` | no | if present, must be `"verdict-input-1"` |
| `title`, `notes` | no | strings copied into reports |
| `conditions` | yes | non-empty array, analysed **in array order** |
| `conditions[].label` | yes | unique non-empty string |
| `conditions[].value`, `.unit` | no | numeric coordinate / display unit |
| `conditions[].configs` | yes | array; may be empty (counted as an empty condition) |
| `conditions[].configs[].name` | yes | unique within its condition |
| `predicted_latency_ns`, `measured_latency_ns` | no | finite positive number, or `null`/absent for "not available" |

Unknown fields are rejected, with the exact JSON path in the error (for
example `conditions[1].configs[0].measured_latency`). Duplicate JSON keys,
`NaN`/`Infinity`, non-positive latencies, duplicate labels and duplicate
configuration names are also rejected. `configs: []` and one-configuration
conditions are valid because P2 must count them, not drop them.

### Ties and tolerance

Latency is lower-is-better. For two values `a`, `b` the tool uses the symmetric
relative gap

```
gap(a, b) = |a - b| / ((|a| + |b|) / 2)
```

and treats them as tied when `gap < tolerance` (default `0.02` = 2 %). A tie is
never treated as a flip. `--tol` changes the tolerance for both P1 and P2.

## P1 crossover

For every adjacent pair of conditions, every configuration pair present in both
is checked. A predicted crossover is an interval where the ordering of that
configuration pair is non-tied at both ends and changes direction.

The measured side is then classified. Every event carries both `relation` and
`reason_code`; the summary keeps `confirmed_same_interval` and
`confirmed_adjacent_interval` as separate counts, because only the same-interval
result is used as the main claim and the adjacent result is sensitivity-only.

| Relation | reason_code | Verdict | Meaning |
|---|---|---|---|
| `same_interval` | `same_interval` | `CONFIRMED` | measured ordering flips in the predicted interval |
| `adjacent_interval` | `adjacent_interval` | `CONFIRMED` (sensitivity only) | it flips one interval earlier or later |
| `other_interval` | `other_interval` | `NOT_CONFIRMED` | it flips only outside the predicted and adjacent intervals |
| `no_flip` | `coverage_unresolved` | `INCONCLUSIVE` | no measured flip occurs anywhere in the sampled range; the sweep may not cover the crossover, so it is neither confirmation nor contradiction |
| `missing` | `missing_values` | `INCONCLUSIVE` | a needed measured value or configuration is absent |
| `tie` | `tied_measurement` | `INCONCLUSIVE` | measured ordering is tied within tolerance |

`INCONCLUSIVE` is also returned when no predicted crossover exists at all
(`reason_code = "no_predicted_crossover"`): there is then nothing to confirm, so
this is not evidence against the prediction. The overall verdict is conservative
- any `NOT_CONFIRMED` event wins, otherwise any `INCONCLUSIVE` event wins,
otherwise every event is `CONFIRMED`. The overall `reason_code` is
`coverage_unresolved` only when every inconclusive event is coverage-unresolved;
it is `inconclusive` for a mixture of missing/tied cases.

Each event reports the values and the relative gaps at both endpoints on both
sides. Crossovers that reach a tie at one endpoint are listed under
`ambiguous_crossovers`; missing configuration/value pairs are listed under
`skipped_pairs` and `measured_skipped_pairs`.

## P2 ranking fidelity

For each condition, all unordered configuration pairs with both values present
are classified as `concordant`, `discordant`, `predicted_tie`, `measured_tie`,
`both_tie`, or `missing` (subdivided into missing predicted / missing measured /
both). Ties within tolerance are counted and excluded from the agreement
denominator:

```
pairwise agreement = concordant / (concordant + discordant)
```

Kendall's tau-b is reported twice: `kendall_tau_b` uses the tolerance-induced
ties; `kendall_tau_b_exact` uses only exact equalities. Both are `null` when
their denominator is zero. The top-level tau-b pools all within-condition pairs
(the report also gives per-condition values).

Top-1 and regret are computed only for conditions where **every** configuration
has both values and there are at least two configurations. A condition is then
called `evaluated`; other conditions are counted as `single_config`,
`incomplete` or `empty` and listed with their missing fields.

* **top-1 hit**: the predicted fastest configuration is also the measured
  fastest, with both fastest labels unique outside the tolerance. A top tie is
  counted as `predicted_tie` or `measured_tie`, not guessed.
* **regret**: `(measured[selected] - measured_best) / measured_best`, where
  `selected` is the uniquely predicted-fastest configuration. It is always
  non-negative. Mean, median and maximum over evaluated conditions are reported.

## Usage

From the repository root (PowerShell):

```powershell
modeling/.venv/Scripts/python.exe tools/verdict/cli.py input.json
modeling/.venv/Scripts/python.exe tools/verdict/cli.py input.json --json
modeling/.venv/Scripts/python.exe tools/verdict/cli.py input.json --tol 0.05 --kind p1
```

`--kind` is `p1`, `p2` or `both` (default). `--json` emits
`{"schema": "verdict-output-1", "title": ..., "tolerance": ..., "tie_rule": ...,
"tau_b_variants": {...}, "p1": ..., "p2": ...}` with `ensure_ascii=False`; tie
values and unavailable statistics are `null`, never `NaN`. The tolerance is
echoed in the envelope and in both analyses, and P2 always carries both
`kendall_tau_b` (tolerance ties) and `kendall_tau_b_exact` (exact ties), so the
preregistered 2 % value and its two tau conventions are recoverable from any
saved output. Exit code is 0 on a successful analysis and 2 on an unreadable or
invalid input (the field paths go to stderr).

The same API is importable (`tools/` must be on `sys.path`, which the test file
and the direct CLI invocation do automatically):

```python
import sys
sys.path.insert(0, "tools")

from verdict import analyze_crossover, analyze_ranking, load
document = load("input.json")
p1 = analyze_crossover(document, tolerance=0.02)
p2 = analyze_ranking(document, tolerance=0.02)
```

## Tests

The repository uses `unittest` (there is no pytest dependency):

```powershell
modeling/.venv/Scripts/python.exe -m unittest discover -s tests -p test_verdict.py -v
```

## Recorded judgement calls

1. **Same-interval is the main claim; adjacent-interval is sensitivity only.**
   Both remain `CONFIRMED` at event level, but the summary reports
   `confirmed_same_interval` and `confirmed_adjacent_interval` separately, and
   only the former should be cited in the main text.
2. **No measured flip anywhere in the sampled range is `coverage_unresolved` /
   `INCONCLUSIVE`, not `NOT_CONFIRMED`.** A sweep that may not cover the
   crossover can neither confirm nor contradict it. Only a measured flip
   *outside* the predicted and adjacent intervals (`other_interval`) is
   `NOT_CONFIRMED`.
3. **No predicted crossover is `INCONCLUSIVE`, not `NOT_CONFIRMED`.** The tool
   deliberately does not call an all-tie/no-flip prediction "wrong".
4. **Conservative aggregation.** One contradicted crossover makes the whole P1
   claim `NOT_CONFIRMED`; per-event verdicts are still available.
5. **Top-1 and regret exclude incomplete conditions.** Pairwise metrics still
   use every complete pair, but a condition with one missing value is not used
   for a whole-condition winner.
6. **Tau-b is reported with and without tolerance ties; the pooled number
   treats each within-condition pair as one observation.** Both variants and
   the tolerance are echoed in every saved output for preregistration.
7. **Relative gap is symmetric (`|a-b| / mean`).** A one-sided convention such
   as `|a-b| / b` would change tie membership near the threshold.
