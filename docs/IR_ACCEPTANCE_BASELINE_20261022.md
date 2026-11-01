# Strategy expression and IR acceptance baseline

Date: 2026-10-22. This campaign measures the existing implementation against a
fixed finite manifest. No compiler, modeling, profiling or runtime feature was
changed to improve the scores. The performance-calibration freeze was preserved.

## Four results

| Dimension | Measured result | Interpretation |
|---|---|---|
| Expression | **11/12 (91.67%) through the shared composition frontend; 12/12 across all existing entry paths** | Subbatch execution pipelining is supported by the retained specialized adapter, not the common Python/composition path. |
| Compiler fidelity | **12/12 finite basic cases** | Independent device, operand, byte, dependency, lifetime, completion and causal checks passed; one case uses the specialized adapter. This is not a proof for arbitrary graphs. |
| Composition | **9/12 (75%) legal combinations**; **8/8 specified bad cases correctly rejected** | One combination is partial and two are unsupported. Gaps remain in the denominator and are not reclassified as illegal strategies. |
| Authoring cost | **6/6 tasks completed without production changes** | Changed strategy snippets use 4–28 nonblank/noncomment lines, median 6.5. No fair low-level-code baseline or human task study exists, so compression and human completion time remain unmeasured. |

There is no weighted overall score. A high score on one dimension cannot hide a
missing capability or a semantic error on another. In particular, authoring
success does not establish that the proposed code-compression target was met.

## Fixed requirements and evidence

The [manifest](../experiments/ir_acceptance/manifest.json) contains 12 basic
requirements, 12 legal combinations, 8 bad declarations/actions and 6 paired
authoring tasks. Its exact SHA-256 is:

`cfea763f9c30406f23190610fbdbf8bdff8181a62afd2ab6779809b27e59a882`

Requirements were frozen before writing the campaign adapters. They were chosen
with knowledge of the implementation, so this is a **retrospective engineering
baseline**, not a blind held-out study or coverage of all LLM strategies.

| Basic ID | Required behavior | Outcome |
|---|---|---|
| B01 | Different CPU/GPU module placement with charged data movement | Pass |
| B02 | GPU prefill, CPU decode, persistent KV and migration | Pass |
| B03 | Attention refinement with exact operands, bytes and stage devices | Pass |
| B04 | CPU/GPU FFN partition and correct partial-output merge | Pass |
| B05 | Multi-layer/multi-round dependencies without extra module barriers | Pass |
| B06 | Actual cross-subbatch CPU Attention / GPU FFN overlap | Pass through specialized adapter; shared frontend gap |
| B07 | Retain and reuse weights across invocations | Pass |
| B08 | Safely evict under an undersized shared cache | Pass |
| B09 | Charged next-layer prefetch during current-layer computation | Pass |
| B10 | Layer-scoped CE/SM choices on one shared service contract | Pass |
| B11 | MoE decisions respect route revelation and identical public prefixes | Pass |
| B12 | Current resource state changes causal placement; private future does not | Pass |

The oracle uses handwritten geometry/algebra and explicit synthetic costs,
including the independent references in `experiments/strategy_fidelity.py`.
Compiler output is only the observed side. Each case writes its expectation
before execution and retains inputs, declarations, actions, observations and
results. No real GPU, downloaded model weights or measured profile is required.

Representative stronger checks include:

- The H=8, I=16 FFN partition conserves 1536 B of gate/up/down weights; CPU/GPU
  intermediate activations are 16/48 B and each partial output is 32 B. Actual
  devices, merge inputs and the charged CPU partial return are checked.
- C06 executes expert0 on GPU, a privately selected expert1/2 on CPU, then expert0
  on GPU again. The packed expert asset `w:layer0:E0` is 1536 B, retains its
  physical identity, is copied to GPU once, remains ready at reuse, and is not
  evicted between uses. Both private-future traces have the same public prefix.
- C10 checks both preparation rule indices against their static layer owners,
  matches each issued preparation to a real COPY, and checks the physical CE/SM
  event resource rather than merely reading an enabled option.
- B06 and the raw advanced-runtime combinations require `goal_reached`; a valid
  submitted prefix is insufficient. B06 additionally requires every operation to
  be DONE.
- The resource-reclaim combination checks actual prefetch, safe eviction,
  charged controller decision/ACK, capacity and the public-prefix boundary under
  two different private restoration times.

## Three missing combinations

| ID | Missing capability | Observed boundary |
|---|---|---|
| C07 | Subbatch execution pipeline + cross-layer prefetch | The specialized pipeline works alone, but its multi-source bundle cannot bind to the common global scope and share that prefetch composition path. Partial support. |
| C08 | Request-local Attention refinement + phase placement | Transform inside a request selector is explicitly unsupported; the implemented transform target is a whole invocation. |
| C09 | Different FFN partitions in two disjoint requests of one invocation | The same whole-invocation restriction prevents local partition choices and their shared-weight ownership contract. |

C01–C06 and C10–C12 pass. An unsupported combination is a gap in expression,
not evidence that the intended strategy is mathematically illegal. No global
infeasibility or optimality claim follows from a policy dead end.

## Authoring tasks

| ID | Strategy edit | Baseline → changed lines | Actual difference checked |
|---|---|---:|---|
| U01 | Layer0 FFN GPU → CPU | 4 → 7 | Device changes and two charged 32 B boundary copies |
| U02 | CPU/GPU FFN split 4/12 → 12/4 | 28 → 28 | Leaf/activation dimensions change; total bytes and merge remain correct |
| U03 | Cache budget 8192 B → 4096 B | 4 → 4 | Safe weight eviction is exercised under the smaller budget |
| U04 | Disable → enable next-layer prefetch | 5 → 5 | Audited, charged next-layer copies overlap current-layer compute |
| U05 | Layer1 CE → SM | 7 → 7 | Actual COPY resource changes, with the same link contract |
| U06 | Fixed → current-cost resource response | 6 → 6 | CPU blackout changes placement response; a private restoration variant preserves the earlier public prefix |

Counts include task-specific strategy helpers. U02 conservatively embeds and
counts the transform/explicit-price construction helper. Input generation and
the independent evidence checker are identified separately. The long FFN case
also exposes remaining geometry/price coupling; it is not evidence that all
strategies are short or easy for a new user to write.

## Reproduce and inspect

```powershell
python experiments/ir_acceptance/run.py --out build/ir-acceptance-new-run
```

Use a new or empty output directory. See the
[benchmark guide](../experiments/ir_acceptance/README.md).

The accepted local campaign is `build/ir-acceptance-v1-20261022-strict/`:
`report.json`, `REPORT.md`, per-group results, verbatim manifest, per-case raw
artifacts, before/after source manifests and actual imported-module identities.
Sources/references remained unchanged during the run; all nine retained accuracy
evidence hashes matched the freeze record. The earlier campaign and adapter
development runs remain separately retained. The strict campaign supersedes
the first campaign after independent review strengthened the evidence checks;
the requirement manifest and production code did not change.

The [compact machine-readable result](data/ir-acceptance-baseline-20261022.json)
records the fixed denominators, results, artifact identities and scope. The
existing independent-reference acceptance tests also passed 13/13. Their count
is not added to strategy coverage.

These results establish the listed finite semantics and authoring behavior.
They do not establish numerical LLM correctness, hardware timing accuracy,
generalization to every shape, human readability, optimal search, or a new
research contribution merely from grouping existing mechanisms.
