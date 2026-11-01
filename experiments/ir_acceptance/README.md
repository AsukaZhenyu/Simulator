# Finite IR acceptance baseline

This source-checkout benchmark evaluates **existing** strategy representation;
it does not repair production features during the campaign. The versioned
`manifest.json` fixes 12 basic requirements, 12 legal combinations, 8 deliberately
invalid cases and 6 paired authoring tasks. Its SHA-256 is bound in `run.py`.
Known gaps remain in the denominator. The cases are a retrospective engineering
baseline, not blind held-out evidence or universal LLM-strategy coverage.

The [first baseline report](../../docs/IR_ACCEPTANCE_BASELINE_20261022.md)
records both shared-frontend coverage and mechanisms that require a retained
specialized adapter, rather than merging them into a claim of unified support.
The [composition completion report](../../docs/IR_COMPOSITION_COMPLETION_20261022.md)
records the subsequent implementation and reacceptance under the same manifest;
B06 and C07 now use the shared frontend and C08/C09 have independent positive
region oracles. Earlier reports are retained unchanged.

From the repository root:

```powershell
python experiments/ir_acceptance/run.py --out build/ir-acceptance-v1
```

Use a new or empty output directory. Core Python has no third-party dependencies.
The source checkout supplies explicitly synthetic fixtures and independent
handwritten references from `experiments/strategy_fidelity.py`; no model weights,
host GPU, profile download, hardware measurement or accuracy refit is used.

## Interpretation

- **Expression:** complete basic requirements divided by all 12 requirements.
  Partial support is reported, not awarded fractional credit.
- **Fidelity:** independently checked basic cases divided by the fixed 12.
  The completely expressed subset is reported separately, without hiding gaps.
- **Composition:** passing legal combinations divided by 12; independently
  correct rejection of specified bad declarations/actions divided by 8.
- **Authoring:** successful paired strategy edits with unchanged production
  bytes divided by 6. Each task retains baseline/changed Python snippets and
  physical line counts including task-specific strategy helpers. Input generation
  and independent checking code are separately identified. No fair low-level code
  baseline or human task study was established, so compression and human time
  remain unmeasured.

A declared option is not enough: positive cases require the specified actual
actions and invariants, including nonzero overlap where required. Compiler
output is never its own oracle. Unsupported is a coverage gap, not an invalid
strategy; a policy dead end does not prove global hardware infeasibility.

`report.json` and `REPORT.md` retain all statuses, reasons and fixed denominators.
Each case retains its expected contract and observed evidence. Before/after
source and reference hashes, actual imported package paths, the manifest hash,
and the accuracy freeze record protect attribution. The runner exits nonzero on
source drift or broken evidence integrity. Measured coverage gaps do not crash
the experiment; gate them explicitly from the report if used in CI.

The baseline is not an installed-library feature, a numerical LLM-output test,
a real-time accuracy campaign or an optimal-strategy searcher. Higher coverage
does not by itself establish a new paper contribution. Extend the manifest by
creating a new version; preserve old results and denominators.
