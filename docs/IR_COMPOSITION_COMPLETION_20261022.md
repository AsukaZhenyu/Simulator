# Unified request-region composition: implementation and reacceptance

Date: 2026-10-22. The three gaps in the
[initial acceptance baseline](IR_ACCEPTANCE_BASELINE_20261022.md) are implemented
through the common Python frontend, IR, composition compiler and modeling
runtime. The original 38 requirement IDs, meanings and denominators were kept.

## Before and after

| Metric | Initial baseline | After implementation |
|---|---:|---:|
| Complete expression through shared frontend | 11/12 | **12/12** |
| Basic compiler semantic checks | 12/12, including one specialized adapter | **12/12, all through the shared frontend** |
| Legal combinations | 9/12 | **12/12** |
| Specified invalid cases correctly rejected | 8/8 | **8/8** |
| Paired authoring tasks without backend edits during the task | 6/6 | **6/6** |
| Code compression / human completion time | Not measured | Still not measured |

The frozen manifest is `v1-20261022`, SHA-256
`cfea763f9c30406f23190610fbdbf8bdff8181a62afd2ab6779809b27e59a882`.
The initial campaign and its results remain intact. New positive adapters for
previously unsupported cases supply independent expected operands, byte counts,
placements and storage constraints; no requirement was removed or weakened.
Production features changed before the new campaign, and remained unchanged
during that campaign. These finite synthetic checks are not universal
expressiveness, hardware accuracy or a blind held-out study.

## The three completed cases

**C07: subbatch pipeline plus cross-layer prefetch.** The analytical Dense
builder can explicitly declare independent request kernels. One invocation may
have two requests while each kernel operates on one request. The exact-cost key
retains invocation batch size; physical signatures distinguish kernel batch 1
from invocation batch 2. The compiler neither divides a whole-batch time nor
duplicates a measured fused-batch duration. This new accounting contract is
synthetic-only; existing native/measured B1 boundaries stay restricted.

B06 now uses the same common frontend rather than `legacy_paper_strategy`.
C07 runs two layers and two requests in one global modeling session, with
positive-duration CPU Attention / GPU FFN overlap and charged next-layer
prefetch COPY actions. Dependencies, completion and the one shared cache budget
are checked independently. The transform-pass pipeline is not mistaken for an
execution pipeline.

**C08: local Attention refinement plus phase placement.** A request-scoped
`plan.transform()` records `execution-region-0.1` in the same execution pipeline:

```python
with plan.module("selected_attention", request_id="request0"):
    plan.transform("layer0-prefill", exact_attention_stages)
```

The case refines only request0's prefill Attention. Request1 keeps its original
projection, Attention and FFN macro wiring and receives no refined allocations.
Both requests use CPU decode with their original KV identities. The checker
verifies five selected-request operand contracts, four KV blocks totaling 128 B,
charged migration, actual phase devices and dependency completion.

**C09: different FFN partitions in disjoint request regions.** The same
invocation expresses CPU/GPU ranges 4/12 for request0 and 12/4 for request1.
Each inner FFN spec explicitly chooses `shared_parent_slice_views`. Full F32
parent matrices stay the physical weight assets, and operation parameters
record the selected columns / corresponding down-projection input rows.

```python
for request, cpu_width in (("request0", 4), ("request1", 12)):
    spec = exact_ffn_contract(cpu_width)  # Explicit stage/transfer prices.
    spec["weight_storage_contract"] = "shared_parent_slice_views"
    with plan.module(request, request_id=request):
        plan.transform("layer0-prefill", spec)
```

The benchmark supplies the exact contract builders; these snippets illustrate
the frontend rather than inventing missing prices. At H=8, I=16, the three
parent matrices total 1536 B. The accepted trace contains three GPU parent COPY
actions of 512 B each, not two independently allocated partition sets or free
runtime repacks. Each request's CPU partial incurs a 32 B return before merge.
All 22 operand contracts, slice parameters, intermediate bytes, stage devices
and the global weight inventory are checked.

The existing `materialized_partition_replaces_parent` contract remains available
for its original whole-invocation mode. The compiler never silently changes a
requested storage representation to make a region compile.

## Compiler invariants

The compiled frontend format is `hierarchical-strategy-0.4`. The inner
`execution-region-0.1` contains public `request_ids` and one supported Attention
or FFN transform. Region selection is explicit in the same serialized IR and
checkpoint fingerprint; it is not a side callback or separate runtime.

- Unknown requests and overlapping same-family regions fail. Distinct families
  can compose on one request; the same family can compose on disjoint requests.
- An unselected task's operation, task metadata, exact costs, allowed devices
  and input/output tensor descriptors must remain unchanged.
- Hidden outputs and KV block identities must remain unchanged.
- Shared-parent views have no separate storage allocation. Their consumer scope
  and spec index are metadata, not new physical tensor ownership.
- Same-plan checkpoint resume reproduces the event timeline; changing the region
  target invalidates the old checkpoint.
- MoE route revelation, shared capacity and engine arbitration remain owned by
  the existing modeling session. Aggregated MoE-row request-local FFN rewrites
  remain explicitly unsupported.

## Validation and retained evidence

The accepted full campaign is
`build/ir-compositions-20261022/acceptance-final-composed/`. It contains the verbatim
manifest, all 38 case results and raw execution artifacts, source manifests and
import identities. All source/reference bytes remained unchanged during the
run; all nine accuracy-freeze evidence hashes still match. No native measurement
or profiling fit was run.

Additional checks passed:

| Suite | Result |
|---|---:|
| New Dense independent-request tests | 8/8 |
| New shared-parent FFN tests, including independent numerical interpretation | 4/4 |
| New region parser/preservation/checkpoint tests | 7/7 |
| New common-path composition integration tests | 5/5 |
| Existing physical, cost and native-signature regressions | 109/109 |
| Existing public library / analytical input regressions | 21/21 and 9/9 |
| Attention, FFN and pipeline suites with the first six region tests | 53/53 |
| Frontend, inspection, preparation, phase and scoped-transfer regression suites | 83/83 |
| Public strategy-IR / independent-fidelity checks | 2/2 and 13/13 |
| Offline sdist/wheel rebuild, fresh-venv installation and examples | 1/1 |

Rows overlap and must not be added into a total test count. CI now includes the
new Dense-subbatch and request-region composition tests. The updated fixed
campaign and its summary are available in the
[machine-readable result](data/ir-composition-completion-20261022.json).

The fifth integration test additionally combines all three new capabilities in
one Plan/IR, beyond the fixed score denominators. Its retained evidence is
`build/ir-combinations-20261022/three-way-regression-v1/`: 35 operations complete,
52 producer dependencies pass, five cross-request CPU/GPU stage overlaps occur,
two charged next-layer weight copies each move 512 B, and peak cached weight
payload is 4480 B under one 8192 B budget. The IR contains exactly three region
declarations. This checks their joint use, not merely three separate entry paths.

Reproduce with a new or empty output directory:

```powershell
python experiments/ir_acceptance/run.py --out build/ir-composition-recheck
```

The results qualify the finite strategy semantics tested here. General dynamic
region rewriting, arbitrary MoE-row partition ownership, measured shared-view
kernel timings, fused batch execution and broad hardware-policy ranking still
need separate contracts and evidence.
