# Strategy representation: first implementation slice

Date: 2026-10-22. Mainline: expression and compiler fidelity. Performance
calibration evidence remains frozen; no native measurement or refit was run.

## What changed

The architecture stays **readable modular Python → coupled strategy IR → strict
compiler → one modeling runtime**. Execution and storage choices constrain each
other through dependencies, shared tensor identities, capacity and causal
information. This iteration improves the contracts of that existing architecture;
it does not establish a novel research abstraction by renaming existing fields.

1. `ModelPlan.declaration_document()`, `inspect()` and `explain()` expose choices
   before performance execution. Missing costs are reported separately from
   declaration checks. No prices are guessed, substituted or read by inspection.
   Full lowering and runtime feasibility remain unproved at this stage.
2. `prepare()` now inherits a module's layer selector. Contradictory selectors
   fail. Phase/request/invocation/kind scopes cannot silently select physical
   weights because persistent weights do not have those consumer identities.
3. `transfer()` selects CE/SM for static weight owners by layer/stage/expert/tensor.
   Explicit priority followed by true selector specialization resolves overlaps;
   conflicting incomparable winners fail. Cost estimates and actual copies use
   the same resolver. All modules retain the same cache budget and physical
   capacity ledger.
4. Whole-invocation `refine()/transform()/transform_file()` reject unsupported
   narrower module scopes instead of pretending to rewrite just that region.
5. Compiled `hierarchical-strategy-0.3` records scoped transfer declarations and
   matched provenance. Changed rules invalidate old checkpoints. The installed
   library exports the new types and accepts explicit service contracts, including
   caller-relative paths; these files remain source-bound checkpoint inputs.

## Read and run

See [the representation guide](STRATEGY_REPRESENTATION.md) for the two subspaces,
current dimensions, scope table and unresolved checks. The
[standalone example](../examples/library/strategy_ir.py) constructs its own
analytical inputs and service contract. It needs no saved profile or repository
fixture:

```powershell
python examples/library/strategy_ir.py --out build/strategy-ir-example
```

Use a new or empty output directory. Inspect `declaration.json`,
`explanation.txt`, `inspection.json`, then `execution/plan.json`,
`execution/decisions.json` and `execution/policy_audit.json`. The example's
synthetic duration is not a hardware result or a strategy speedup claim.

## Acceptance evidence

| Check | Result | Evidence scope |
|---|---|---|
| New inspection tests | 15/15 | Missing costs, invalid declarations/bindings, no cost read or runtime construction |
| New preparation-scope tests | 10/10 | Selector intersection/rejection, inherited rules, one physical copy/accounting |
| New scoped-transfer tests | 11/11 | Typed rules, precedence, estimate/action agreement, shared budget, replay |
| New frontend integration tests | 9/9 | Public declarations, actual L0=CE/L1=SM actions, unsupported region scopes |
| New installed-style public API tests | 2/2 | Generated analytical inputs, relative service path and source binding, no profile required |
| Frontend/inspection/preparation/phase/transfer suite | 79/79 | Includes the first four new suites plus existing frontend and phase regressions |
| Composition/compiler/block/unified/reconfiguration regressions | 70/70 | Existing suites; separate from the 11 new scoped-transfer tests |
| Public library regression | 21/21 | Existing library behavior |
| Analytical-input regression | 9/9 | Existing generated geometry/cost contracts |
| Offline distribution test | 1/1 | Build sdist and wheel, rebuild from sdist, install in fresh venv, run demos and new IR example outside checkout |
| Frozen evidence identity | 9/9 unchanged | SHA-256 checks against `data/research-direction-20261022.json` |

There are **47 new tests** in this slice. Rows are not additive: the 79-test
suite includes the four focused mapping suites. These checks establish the
listed semantics and packaging behavior, not real-system timing accuracy.
The public-library regression emitted temporary-directory cleanup warnings and
still passed; it was not a new performance experiment.

## Remaining expression work

- A fully cost-independent lowered graph: legacy refined-stage inputs still
  combine geometry and inline prices. Declaration inspection does not solve that
  complete separation or prove transformed port connections.
- Region transforms with explicit targets, read/write sets and shared-weight
  representation. Disjoint regions within one invocation cannot be enabled just
  by dropping the duplicate-family check.
- Scoped cache and resource-controller ownership. Physical budgets remain global;
  arbitrary per-module controllers or budgets are not available.
- Independently checked mechanism cases and comparisons under a common candidate
  set. Current Python choices do not reproduce every detail of published systems.

The next bounded extension should define a small region contract and its
independent action/accounting reference before adding more syntax. Search
algorithms and broader accuracy campaigns remain outside this implementation
slice.
