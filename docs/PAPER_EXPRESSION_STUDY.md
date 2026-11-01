# Paper study: finite strategy expression and compiler fidelity

This study asks whether existing strategy primitives can execute together through
one public Python declaration, one IR/compiler path, and one resource/state
ledger. It does not infer usability, universal coverage, hardware accuracy, or
novelty from a count of passing tests.

## Reproduce

From the repository with its normal Python import paths configured:

```powershell
python experiments/paper_expression_study.py --out build/dac-evidence-20261023/expression-new
python -m unittest discover -s tests -p test_paper_expression_study.py -v
```

The output must be new or empty. No saved profile, GPU, model weights, network
access, or measured timing input is required. Kernel and transfer prices are
explicit synthetic contracts. Workload geometry is small F32 H8/I16/GQA2:1.

## Frozen finite requirements

Eight requirements are written to a manifest before the first case executes.
Failure remains in its fixed denominator. Four are reference reruns and four
are new combinations of already implemented features. The new combinations are
author-selected scenario extensions, not blind heldout tests or new algorithms.

| ID | Mechanisms composed | Independent obligations |
|---|---|---|
| R01 | Unequal CPU/GPU FFN slices and SUM | Exact gate/up/down operands; 128/384-byte materialized leaves; 1536-byte total weights; 32-byte host partial delivered before merge. |
| R02 | CPU Attention/GPU FFN across requests, async progress, cross-layer prefetch | Exact arrivals/scopes/retirement, positive compute overlap, charged next-layer prefetch, one memory ledger. |
| R03 | Causal expert execution, shared weight cache, prefetch | Revealed route rows only, reused expert identity and 1536-byte payload, no redundant reload for retained expert, real prefetch. |
| R04 | GPU Prefill/CPU Decode, all-layer KV handoff, cache/prefetch | Each layer/token KV has one stable 32-byte identity; each required handoff occurs once; first Decode gate depends on final hidden and complete KV. |
| X01 | Attention stages, unequal FFN slices, startup preparation, CPU recovery | Operand and weight conservation, charged host-partial return, startup completed before compute, no CPU compute during declared initial blackout. |
| X02 | Request pipeline, async progress, prefetch, GPU recovery | Exact requests and dependencies, actual cross-request CPU/GPU overlap and prefetch, no GPU compute during initial zero-service interval. |
| X03 | MoE PD handoff, history ordering, prefetch, CPU recovery and private future routes | Two complete workloads share identical public observation/decision prefixes before their routes diverge; full KV handoff and revealed expert constraints hold in each. |
| X04 | Opposite request-local FFN slices using parent views in shared physical memory | Request0 splits4/12 and request1 splits12/4; both read three full512-byte parents, not invented small weights; 1536-byte FFN/2368-byte total weight inventory; zero-payload SHARE and no COPY; independently replayed pool accounting. |

R01 reuses the existing hand-derived fidelity reference. R03/R04 reuse the
serving acceptance recipes and checkers. R02 restates a previously exercised
pipeline/prefetch mechanism with the finite serving lifecycle checker. The
different mechanisms share the current library/compiler/runtime; no alternative
per-paper executor is supplied.

## What is checked independently

Expected shapes, operand identities, request sets, slice intervals, memory budgets
and route boundaries are declared before compilation. The compiler supplies the
observed graph and events only. Shared helper checkers are acknowledged and
source-bound rather than described as a second implementation of the compiler.

The added event checker validates charged physical COPY bytes, unique compute
charges, producer-before-consumer dependencies, complete operations, capacity
limits, protected live inputs, and initial service blackout behavior. Mutation
tests deliberately violate these conditions. Existing handwritten operand
checks distinguish equal-size but incorrect tensors; simple byte totals alone
would not establish semantic fidelity.

X04 additionally uses a separate standard-library physical-state replay checker.
It reconstructs allocation aliases, references, reservations, release, and peak
capacity from saved events rather than accepting the simulator's peak as truth.
This gives a concrete example of why computation slices and physical storage
identities are distinct concepts. It does not establish that prior tools cannot
express the same distinction.

## Authoring measurements and limits

Each case retains a source extract covering its recipe and selected helper
definitions. These extracts are not transitive dependency-complete executable
scripts. Nonblank/noncomment lines and AST statements are counted. Exact
synthetic-price literals, workload declarations and embedded checks are included;
these are extract-size observations, not complete authoring effort or pure
end-user strategy code counts. Backend edits and manual lowered-graph edits are zero for
these cases and the runtime source hashes are checked before/after execution.

There is no matched hand-written low-level baseline and no participant study.
Consequently no code-compression factor, reduced development time, or measured
ease-of-use advantage is claimed. A fair external-tool comparison remains open.

## Evidence and repeat policy

Each output contains the preregistered manifest and obligations, source hashes,
actual imported module paths/hashes, generated inputs, exact costs, compiled IR,
commands/events/state, expected properties, observations, and every result.
`summary.json` is the portable report input; `artifact-hashes.json` binds all
retained evidence. Experimental source files belonging to other parallel studies
are excluded from the execution freeze, while all runtime sources and imported
experiment helpers are included.

The first run, `expression-v1`, retains a 7/8 result: X02 incorrectly supplied a
nonexistent `gpu_available` constructor keyword in the experiment runner. The
existing API expresses GPU unavailability with rational service rate0/1. The
runner was repaired to use that API, keeping all eight requirements unchanged;
`expression-v2` reruns the same manifest. No library/compiler/modeling code was
changed to make the cases pass. Neither run is a blind holdout evaluation.

The retained `expression-v2` result is **8/8**, comprising 4/4 reference mechanisms
and 4/4 new finite combinations. All 140 frozen source files were unchanged and 77
imported core modules were bound to the checkout. The focused checker suite
passes 12/12 tests. X03 has 128 identical public decisions before the future routes
diverge. X04 checks 22 exact operand contracts and independently replays 105 events,
including 26 computes and 21 zero-payload SHAREs, with 0 COPY and 2768-byte physical
peak usage. Those small synthetic numbers are semantic witnesses, not measured
hardware performance.

The study is useful evidence for finite composability and semantic checking.
Novelty relative to other tools, arbitrary-shape coverage, numerical equivalence,
calibrated performance, and improved hardware/strategy choices require the
separate comparisons and timing/design studies.
