# Independent strategy fidelity references

This suite checks whether concrete Python strategy choices survive IR lowering
and execute with the required dependencies, tensor bytes, transfers and lifetime
constraints. It is a small semantic reference suite, not a performance
calibration, tensor numerical correctness test, or reproduction of an entire
published system.

The oracle is handwritten geometry and algebra. It does not ask the compiler
to generate the expected graph, byte counts or action sequence. The observed
graph and events come from the actual public `Simulator` / `ModelPlan` pipeline.
All inputs are generated into the output directory. No old fixture, saved
profile, GPU, model weights, network access or optional dependency is needed.

## Run

From a source checkout, with Python 3.10 or newer:

```powershell
python experiments/strategy_fidelity.py --out build/research-evidence/strategy-fidelity
```

The output directory must be new or empty. To repeat a run, choose a new output
directory. Existing evidence is never replaced. The process returns nonzero if
any reference fails.

The normal source test setup can also run the 13 acceptance tests:

```powershell
$env:PYTHONPATH = "src;mapping;modeling"
python -m unittest discover -s tests -p test_strategy_fidelity.py -v
```

## Reference cases

| Case | Independent expected property |
|---|---|
| Attention boundary | F32 H=8, two query heads, one KV head, D=4, q=1, history=0. Q=32 B; K=V=16 B; scores/probabilities=8 B; context and appended KV=32 B. Exact operand identities verify QK uses Q/K, softmax consumes scores, PV consumes probabilities/K/V and owns the new KV block, and output consumes context, the output weight and original hidden residual. Q/K move GPU to CPU; probabilities move CPU to GPU. |
| FFN partition | I=16 is partitioned into [0,4) on CPU and [4,16) on GPU. Each gate/up/down matrix has 128 B and 384 B leaves; all six leaves total 1536 B. Exact operand identities verify both branches consume the one normalization output and their own gate/up/down weights, each activation consumes its matching gate/up pair, SUM consumes both partials, and residual consumes the merged output plus the original FFN input. The CPU partial needs a charged 32 B return before SUM. |
| Transform composition | Attention stages and FFN shards remain in one graph. Three explicitly selected GPU weight leaves each cost a 384 B startup copy; all finish before the first compute. Resident plus in-flight weight copies remain within the declared 4096 B cache budget. |
| Phase and KV persistence | Two layers, two prefill tokens, then two decode steps produce eight stable 32 B KV blocks, totaling 256 B. Attention histories are 0, 2, 3 for each layer. GPU prefill and CPU decode share the same KV identities; the four prefill blocks each incur a 32 B GPU-to-CPU copy. This is shared-resource phase placement, not independent P/D server instances. |
| Placement, prefetch and cache | Layer-0 FFN executes on CPU and layer-1 FFN on GPU while next-layer prefetch issues charged actions. Weight residency plus transfers respect the cache budget. Every copy consumes its supplied nonzero payload time. |
| MoE causality | Two traces share the first route to expert 0 but use different hidden second routes to expert 1 or 2. Public observations and commands must be identical through the second router launch. Expert tasks stay hidden until their routes are revealed. A command targeting an unrevealed expert is rejected. Static expert weight catalogues may remain public. |
| Invalid or unpriced transforms | Partition gaps, overlaps, incomplete coverage, changed SUM arity, absent exact shard/attention/transfer prices, unmatched selectors, and an unsupported GPU identifier are rejected. No interpolation or silent replacement is allowed. |
| Infeasible capacity | Directly copying the handwritten 32 B input-normalization weight into one byte of VRAM must be rejected with `CAPACITY_EXCEEDED`. The high-level GPU policy remains unsuccessful without a silent CPU fallback and reports `policy_dead_end` / `no_progress`. This diagnoses the current schedule and does not prove global infeasibility. |
| CPU service blackout | An initially unavailable CPU that resumes at 10000 ns adds exactly 10000 ns to a CPU-only reference execution. This models a service blackout, not operating-system threads. |

For successful executions, the suite additionally checks that every producer
finishes before its consumers start and that no in-flight compute input is
evicted from the storage domain used by that computation. Evicting an unused
backup in the other domain is permitted. These invariants supplement the fixed byte/algebra oracles;
they do not replace them.

Four additional tests check that the evidence checker rejects equal-sized but
incorrect Attention/FFN operands, foreign installed module imports, source
drift or an unbound module hash, and eviction of an executing input while
allowing release of its unused backup. These tests verify the checker itself
as well as successful reference executions.

## Synthetic service contracts

The hypothetical CPU/GPU rates and one-byte-per-nanosecond transfer link are
explicit declarations. Refined stages use independently supplied exact costs:
QK/softmax/PV/output use 11/13/17/19 ns; FFN norm/gate/up/activation/down/SUM/
residual use 7/11/13/17/19/23/29 ns. Scratch is explicitly declared as 4 B per
refined stage. These arbitrary small prices make missing costs and uncharged
actions visible. They are not estimates of real hardware or derived by dividing
a measured macro duration.

Changing partition widths, token counts, history, stage boundaries or merge
arity requires a matching cost contract. The analytical convenience API only
generates its declared macro costs; it does not silently invent refined-stage
measurements. This suite supplies the refined-stage contracts explicitly through
`plan.transform(...)`.

## Evidence artifacts

- `report.json`: machine-readable status, assumptions, results and limits.
- `REPORT.md`: short English run summary.
- `source-manifest-before.json`: runner and checkout Python source hashes frozen
  before executing any case, including modules that may be imported lazily.
- `source-manifest.json`: the successfully verified source manifest. A changed,
  added or removed source file prevents accepted report generation.
- `imported-modules-before.json` and `imported-modules-after.json`: actual module
  names, absolute source paths and hashes. Imported library, mapping and modeling
  modules must resolve inside their corresponding checkout directories and match
  the frozen bytes. An installed or foreign package cannot silently replace them.
- `<case>/expected.json`: handwritten expected properties, written before
  compilation.
- `<case>/result.json`: case status and observed values or a traceback.
- `<case>/inputs/` and `<case>/execution/`: generated costs, workload, IR,
  commands, events and final persistent state. Paired scenarios use child
  directories. Rejected cases retain their generated invalid inputs when
  applicable.

The latest local retained evidence is in
`build/research-evidence/strategy-fidelity-current/`: 9/9 reference groups passed,
the source did not change during the run, and all 62 imported package modules
were bound to the expected checkout paths. These are local development results;
the generated evidence directory is not required to use or rerun the suite.
The corresponding acceptance suite passes 13/13 tests. The five focused public
no-progress regressions also pass. Earlier local evidence,
including `strategy-fidelity-final/` from before the dead-end diagnostic change,
is retained separately and is not added to these counts.

The reusable runner covers nine finite reference groups; one rejection group
contains nine deliberately invalid variants. A passing run
does not establish arbitrary strategy-space coverage, full-paper reproduction,
hardware prediction accuracy, correctness of numerical outputs, physical unified
memory, multi-GPU collectives, or globally optimal scheduling. Those require
separate evidence. Do not merge these semantic pass counts with held-out timing
accuracy or claim a measured speedup from their synthetic makespans.

The one-byte-VRAM example distinguishes two diagnoses. A direct oversized COPY
is a concrete capacity rejection. If the high-level policy cannot prepare a
required input and requests ADVANCE while there is neither running work nor a
future event, the stateful runtime reports `no_progress` and the unified driver
reports `policy_dead_end`. The result remains unsuccessful with no completed
latency; it must not win a candidate comparison or be labeled proof that every
mapping on the hardware is infeasible. A deliberately idle policy receives the
same diagnosis even on hardware where the baseline strategy succeeds.

The public regressions in `tests/test_policy_no_progress.py` separately verify
that future CPU and memory restoration still allow waiting and completion,
malformed early computation retains `INPUT_NOT_READY`, and a failed GPU policy
never silently falls back to CPU. The legacy fixed-action evaluator retains its
existing `ILLEGAL_ACTION` code for an empty ADVANCE; the new structured code is
specific to stateful sessions with environment timelines.
