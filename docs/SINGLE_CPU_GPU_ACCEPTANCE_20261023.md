# Single-CPU/GPU request and expert scheduling: implementation and acceptance

Date: 2026-10-23. Expert scheduling, independent multi-request pipelines and an
explicit P/D state handoff now compose through the existing Python frontend,
IR, compiler and **one** modeling runtime. The scope remains one CPU and one
GPU. The [design and runnable example](SINGLE_CPU_GPU_REQUEST_FLOW_20261023.md)
explain the interfaces and their limitations.

## What changed

| Mechanism | Implemented behavior |
|---|---|
| Request lifecycle | Finite independent arrivals, unequal Prefill/Decode lengths, charged input/position materialization, retirement and real KV release |
| Request pipeline | No whole-batch round barrier; dependency-based CPU/GPU overlap; graph order, Decode-first or request round robin |
| P/D handoff | Final Prefill hidden state plus every layer's KV must reach the selected domain before first Decode; actual COPY/capacity charges and graph-level KV-reader device constraints |
| Expert scheduling | Order router-revealed regions by Token count or known history; choose CPU/GPU from public residency, prices, current service rates and capacity; preserve real request ownership |
| Unified expression | `hierarchical-strategy-0.5` records request pipelines and handoffs beside existing transforms, placement, cache, prefetch and transport choices |
| Portable use | Public `Request`, `RequestFlow`, `RequestPipeline` and `PhaseHandoff`; Dense/MoE examples work after offline installation with no generated profile |

`RequestFlow` is workload/model state. `RequestPipeline`, expert rules and
`PhaseHandoff` are mapping choices. New `global-execution-0.2` leaves remain
exactly priced B1 kernels; old `.1` fixed rounds and the previous B=2 independent
kernel contract remain available. This does not synthesize fused batching.

## Fixed acceptance results

| Set / group | Passed / fixed requirements |
|---|---:|
| Original IR set | **38/38** |
| New mechanisms | **6/6** |
| New combinations | **4/4** |
| New invalid-case checks | **4/4** |
| New causal-prefix contrasts | **2/2** |
| New extension total | **16/16** |

There are **54 fixed requirements in two separate manifests**. This count is
not a percentage of all inference strategies. The original IDs, meanings and
denominators were unchanged; the new 16 requirements were frozen before their
adapters were implemented. This is a known finite synthetic set, not a blind
held-out study.

- Original manifest SHA-256:
  `cfea763f9c30406f23190610fbdbf8bdff8181a62afd2ab6779809b27e59a882`.
- Extension manifest SHA-256:
  `bf7953057c9ed0575df2802b51e0fa3141ed0cd23b8d63058bab9c6471aa6745`.

Canonical accepted campaigns:

- `build/serving-20261023/prior-38-final-portable/`
- `build/serving-20261023/acceptance-final-portable-v2/`

Both have `source_unchanged=true`, source/import identities and all case
artifacts. All nine frozen accuracy-evidence hashes still match. Earlier
pre-freeze/source-drift and interrupted runs remain diagnostic artifacts and
are excluded from these accepted results. The
[machine-readable summary](data/single-cpu-gpu-acceptance-20261023.json)
binds the canonical reports and their retained evidence.

## Concrete semantic evidence

The new examples use two layers, two requests, H=8, I=16, F32 geometry and
explicitly chosen synthetic prices. Numbers below are simulation witnesses,
not hardware measurements or demonstrated speedups.

| Obligation | Observed evidence |
|---|---|
| Arrival and retirement | A request retires at 406668 ns; four KV blocks each release 32 B through real EVICT actions while the other request continues |
| Independent round progress | One request starts Decode while another is still in Prefill; their valid cross-round work overlaps for 644 ns |
| CPU/GPU pipeline | Seven positive-duration cross-request CPU-Attention/GPU-FFN overlap witnesses |
| Shared weight ownership | Exact ID-to-byte inventories: Dense has 18 tensors / 4736 B; MoE has 10 / 10944 B across two layers, without per-request weight clones |
| Complete P/D state | Each request hands over four KV blocks / 128 B from both layers; the gate has exactly final hidden plus those four KV operands |
| No early Decode | For request0, all KV are ready at 156356 ns; first Decode work starts at 156396 ns |
| Strong missing-KV negative | With Prefill complete and every other gate operand ready, withholding only `kv:layer1:request0:1` rejects the gate with `INPUT_NOT_READY` naming that exact block |
| Expert ordering | All 14 expert computations follow route revelation; four Prefill scopes select the two-row expert before the one-row expert, order `[1,0]` |
| Expert cache reuse | Each layer's E0 has four GPU consumers and one charged load |
| Storage composition | P/D plus bounded cache produces four actual next-layer prefetch COPY witnesses during current-layer work |
| Joint case | Arrivals, unequal Decode lengths, adaptive expert device selection, P/D handoff and bounded cache finish together; peak cached weights 4800 B under one 32768 B budget |
| Future arrival noninterference | Changed private arrival, prompt length and Decode length preserve all 90 public observation/decision prefix records, including memory |
| Future route noninterference | Changed future expert assignments preserve all 111 public prefix records before the differing route is revealed |

Both runs in each causal contrast reach their full goals. Expectations are
written before compilation and checked against observed operands, identities,
bytes, devices, dependencies and actual runtime actions. A successful prefix,
mere configuration field, or zero-duration overlap is insufficient.

## Verification and engineering corrections

On the final frozen source, focused mechanisms and public API checks pass
**36/36**: expert helper 7, request declarations 10, request modeling 14 and
public library 5. Receipts/logs are retained under
`build/serving-20261023/final-focused-v2/`. The profile-independent serving
unittest passes **2/2**, exercising all 16 fresh cases without reading ignored
native profiling results. These overlap the benchmark and are not added to its
54-requirement denominator.

Offline distribution verification also passes **1/1**: build wheel/sdist,
rebuild from the source archive, install into a fresh isolated venv, import the
installed modules and execute both new Dense/MoE request examples outside the
checkout without profiles or CUDA.

The new checks exposed and fixed real integration faults: lifecycle fields
were dropped by the composition observation wrapper; MoE groups retained a
batch pseudo-identity in single-request streams; initial position tensors
leaked future prompt lengths through memory; and runtime overrides could have
contradicted the first-Decode handoff domain. Default source/control gates now
have an explicit CPU contract, so JSON cost-key ordering cannot select their
device. Checkpoint replay and source changes are also checked.

## Boundaries and reproduction

The current P/D feature is a **single-machine phase/state handoff**, not a
disaggregated serving cluster. Multi-GPU communication, fused dynamic batches,
chunked Prefill, prefix/paged-KV sharing and speculative commit/rollback remain
outside this implementation. Next-token inputs are trace supplied; numerical
token generation is not executed.
Performance calibration was not rerun or changed; no new accuracy claim follows
from these semantic tests.

From a source checkout, choose new or empty output directories:

```bash
python experiments/ir_acceptance/run.py --out build/ir-regression-recheck
python experiments/serving_acceptance/run.py --out build/serving-recheck
python -m unittest discover -s tests -p test_serving_acceptance.py -v
python tests/test_distribution.py
```

The two research runners strictly verify the retained local calibration-evidence
freeze. Portable semantic tests and installed-library examples require no such
saved measurements. Future multiple-device extensions should retain request
semantics independent of device naming, explicit communication/data ownership,
and one real capacity/resource ledger; no extra GPU is implemented here.
