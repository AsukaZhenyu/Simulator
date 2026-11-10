# Inputs the published model column was measured from

These three files are inputs, not results. They are here because the campaign
that produced the paper's OLMoE row reads them, and a second machine cannot
re-derive them without the first machine's tokenizer pass and capacity pilot.

| File | SHA-256 | What it is |
|---|---|---|
| `request-suite-v3.json` | `bac95b6d…` | the frozen eighteen-request corpus: three domains × two length windows × three targets, calibration/heldout split fixed before any timing |
| `RESIDENT_CAPACITY_POLICY-v2.json` | `6f36f9e5…` | the declared resident budget: model bytes plus a 1 GiB context workspace and a 512 MiB system margin, with the observed startup peak |
| `request-template-normal-SSLS.json` | `ee0fc8e3…` | the request template one tokenizer pass produced (threads, context, batch, model contract) |

`prepare_portable.py` reads them through `--suite`, `--capacity-policy` and
`--template` (these paths are the defaults) and rewrites only the three fields
that name a device's file: `model_path`, `model_sha256`, `model_bytes`.

The prompts are derived from the public GSM8K, MBPP and UltraChat snapshots that
`dataset.py` downloads; `request-suite-v3.json` retains each row's source ID and
hash. Nothing here contains a timing, a price or a prediction.
