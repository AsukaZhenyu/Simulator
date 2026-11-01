# Strategy expression and IR

The current path is **modular Python choices → explicit IR → strict compiler →
the shared modeling runtime**. Modules organize choices and their provenance.
They do not add barriers, allocate independent devices, or create extra memory.
Accuracy experiments remain frozen. The examples below use declared analytical
or synthetic costs, not measurements of the host computer.

## The strategy space

A strategy is a compatible pair of execution choices `E` and storage/scheduling
choices `S`, subject to model dependencies, physical capacity, resources and causal
information. It is not an unconstrained Cartesian product. For example, moving
an FFN stage to the CPU changes the required input copies and which weight copy
can be reused; preloading a weight changes the remaining shared capacity.

| Subspace | Choices currently represented | Important boundary |
|---|---|---|
| Execution | Invocation or explicit request-region Attention/FFN refinements, partitions and stage devices | Disjoint same-family request regions compose; overlapping regions fail |
| Execution | Device bindings selected by layer, invocation, request, phase and task kind | Priority followed by selector specialization; ambiguity fails |
| Execution | Independent-request CPU/GPU pipelines, finite arrivals/retirement, Decode-first/round-robin priority and causal Python decisions | Explicit independent kernel accounting; no fused-batch measurement, multi-GPU collectives or independent P/D servers |
| Execution and state | Single-machine P/D placement and explicit all-layer KV/hidden handoff before first Decode | Actual charged COPY/ready dependencies; first KV readers obey the destination; later rounds can be placed separately |
| Scheduling | Router-revealed expert Token/history order and current-rate CPU/GPU path choices | No future routes or recovery forecasts; paper-owned queues reject conflicting priorities |
| Storage/scheduling | Global cache rule, tensor/expert grouping, shared budget and release rules | Module scopes do not create separate cache budgets |
| Storage/scheduling | Startup/spare weight preparation and existing causal next-layer prefetch | Physical ownership comes from the model catalogue, not future expert routes |
| Storage/scheduling | CE/SM choice for weights by layer, stage, expert or exact tensor identity | Both engines still use the same declared link and runtime |
| Storage/scheduling | Existing global resource-response controller | No per-module controller ownership or calibrated dynamic response claim |

The stage/task vocabulary is implemented by `modeling`. A macro is a packaging
boundary that a supported execution transform can refine. Neither the Python
syntax nor grouping the existing fields alone establishes a new paper-level
abstraction. Coverage and compiler fidelity require independent examples.

## A readable Python strategy

```python
from llm_simulator import Select

for layer, engine in (("layer0", "ce"), ("layer1", "sm")):
    with plan.module(layer, layer_id=layer):
        plan.place("dense_ffn", "gpu")
        plan.transfer(engine=engine)
        plan.prepare(destination="dram", trigger="spare")

with plan.module("decode", phase="decode"):
    plan.place("dense_ffn", "cpu",
               select=Select(layer_id="layer0"), priority=1)

print(plan.explain())
draft = plan.declaration_document()
inspection = plan.inspect()
compiled = plan.compile()       # Still requires exact applicable prices.
result = compiled.run()         # Same dependency/capacity/service evaluator.
```

Here, layer 0 uses CE for its weight copies and layer 1 uses SM. Both layers use
the same physical memory ledger and cache budget. The Decode override changes
layer 0 FFN execution; it does not assign Decode its own hardware or duplicate
shared weights. `transfer()` chooses an engine when a COPY is needed. It does
not force a copy or invent a new route.

Run the complete installed-library example from the repository root:

```powershell
python examples/library/strategy_ir.py --out build/strategy-ir-example
```

The output must be new or empty. No repository fixtures, downloaded models or
saved profiles are required. The example explicitly creates a payload-only
service contract and chooses analytical parameters. It writes:

- `declaration.json`: non-executable declared choices and scopes.
- `inspection.json` and `explanation.txt`: completed checks and unresolved checks.
- `execution/plan.json`: compiled hierarchical IR, including scoped transfer rules.
- `execution/decisions.json`, `events.json` and `policy_audit.json`: actual choices,
  actions and matched-rule provenance.

`hierarchical-strategy-0.5` is the compiled frontend format. Legacy
`unified-strategy-0.2/0.3` files keep their strict schemas. Scoped rules are typed
compiler inputs and are persisted in the hierarchical IR and audit, not added
silently to the legacy JSON. `ScopedTransportRule.document()/from_document()`
round-trip this rule contract. Loading arbitrary hierarchical Python callbacks
from a JSON file is not implemented; keep the source Python to reconstruct them.

## Inspection, compilation and execution are separate

| Stage | What it does | What a successful check establishes |
|---|---|---|
| `plan.inspect()` | Parses declared model/DAG and static macro bindings without opening price tables or creating a runtime | Only the checks listed in the report |
| `plan.compile()` | Applies supported transforms, binds exact costs, validates placements/services and source identities | A strictly compiled instance for the supplied workload and contracts |
| `compiled.run()` | Applies causal decisions using one modeling runtime | Whether that policy completes under that declared scenario |

Inspection reports **declarations**, **static bindings**, **cost coverage**,
**lowering** and **runtime feasibility** separately. A missing price file is
`cost_coverage=missing`; it does not make an otherwise checkable declaration
invalid. A file that merely exists is `present_unverified`, never evidence of
complete cost coverage. Inspection substitutes neither zero nor a guessed price.

Inspection currently checks finite layer declarations and static macro selectors.
It does not prove transformed ports, cross-invocation hidden/KV connections,
dynamic MoE tasks, or scoped transfer applicability. These remain explicitly
unproved until strict compilation. Consequently its overall `semantic_validity`
is `unknown`, `unsupported` or `invalid`, and `executable` is always `False`.
Do not read a declarations-only pass as a full semantic proof. Missing exact
prices continue to block strict compilation and execution.

For per-request arrivals, unequal Decode lengths, public request priorities,
causal expert selection and an explicit P/D boundary, see the
[single-CPU/GPU request-flow design](SINGLE_CPU_GPU_REQUEST_FLOW_20261023.md).
`RequestFlow` owns workload/state facts; `RequestPipeline` and `PhaseHandoff`
are recorded mapping choices in the same frontend IR. Both old fixed rounds
and new independent streams execute through the existing global runtime.

`declaration_document()` is a copy of the declared input, not an executable graph.
It retains explicit model/profile references. Legacy refined-stage documents
can still contain inline prices; complete separation of transform geometry and
price binding is a remaining task.

## Two kinds of scope

Execution tasks have phase/request/invocation labels. A physical weight can be
shared across those tasks and rounds, so it cannot acquire a new physical owner
every time a different request uses it.

| Operation | Inherited module selectors | Additional explicit selectors |
|---|---|---|
| `place()` | Layer, invocation, request, phase, kind | `Select(...)` intersection |
| `prepare()` | Layer only | Stage, expert, tensor ID set via `DataPreparation` fields |
| `transfer()` | Layer only | `TensorSelect(layer_id, stage, expert_id, tensor_id)` |
| `transform()/refine()` | Matching invocation and optional request scope, or an unfiltered grouping module | Typed `ExecutionRegionSpec` / serialized `execution-region-0.1`; whole-invocation transforms remain supported |

Contradictory selectors fail. A phase/request/kind selector around weight
preparation or scoped transfer fails explicitly rather than being ignored.
`prepare(DataPreparation(...))` remains supported; keyword fields are a readable
alternative, and the two forms cannot be mixed in one call.

`transform()` in a request module records an explicit region in the same
execution pipeline. Only the selected request's supported Attention or Dense FFN
region changes; unselected tasks, exact costs, allowed devices, tensors and
hidden/KV ports are checked for preservation. Two transforms of one family may
compose only on disjoint request sets. Unknown requests and overlapping targets
fail. Layer/phase/kind module selectors are not silently interpreted as graph
regions; select the invocation explicitly and use phase selectors for placement.

```python
with plan.module("selected_attention", request_id="request0"):
    plan.transform("layer0-prefill", exact_attention_stages)

for request, split in (("request0", 4), ("request1", 12)):
    spec = make_exact_ffn_spec(split)  # Explicit geometry and matching prices.
    spec["weight_storage_contract"] = "shared_parent_slice_views"
    with plan.module(request, request_id=request):
        plan.transform("layer0-prefill", spec)
```

`execution-region-0.1` contains `request_ids` and an inner `transform`. Its
selector names public static requests, not unrevealed MoE expert rows. Routed
MoE row-group region rewrites remain unsupported.

For local FFN partitions, `shared_parent_slice_views` is an explicit storage
choice: full F32 parent matrices remain the single physical assets; stage
`op_params=[start,end]` describes output-column / matching down-input-row reads.
Views allocate no second copy. A GPU COPY moves and charges the entire parent
matrix, even if that stage computes only a slice. For H=8, I=16 the three parents
total 1536 B, and their GPU copies are three 512 B transfers. This does not model
a free runtime repack or measured strided-view kernel performance. Independent
synthetic stage costs are still required. Existing
`materialized_partition_replaces_parent` keeps its original full-invocation
behavior; the compiler never silently substitutes it for shared-parent views.

The analytical Dense builder declares multi-request execution as independent
per-request kernels: cost-table `batch` retains invocation size, while physical
signatures identify kernel batch 1 and the invocation batch separately. CPU/GPU
overlap and prefetch share the existing slots and ledger. Measured/native B1
contracts are not automatically extended to fused or independent B2 timing.

## Scoped transfer precedence and shared resources

Scoped rules override the global role/stage engine fallback for matched weights.
Within scoped rules:

1. The highest explicit priority wins.
2. At equal priority, a strict specialization wins.
3. Incomparable rules prescribing different engines are ambiguous and fail.

Counting selector fields is not a tie-breaker. A tensor/expert rule and a layer
rule can overlap without either being a specialization of the other.

Static ownership is resolved from the public weight catalogue. A selector that
matches no weight fails compilation. Non-weight roles keep the existing global
role/stage rules; scoped activation/KV ownership is not implemented in this
iteration. CE/SM requires an explicit compatible service contract. Without
services, only an explicit `engine=None` is admissible.

The public `Simulator.compile(services_file=...)` resolves a relative path from
the caller's working directory. The service file is a source-bound external
input: retain it alongside the generated project when using checkpoints.

Estimates and emitted COPY commands call the same engine resolver. Cache
accounting includes in-flight weight copies and retains the existing global
budget. Choosing an engine does not grant a module extra capacity or bandwidth.
Rule declarations and matched provenance enter the audit. Changing rules changes
the compiled fingerprint and prevents replaying a checkpoint from another plan.

## Evidence and remaining work

Focused tests cover missing prices, invalid DAG/selectors, scope contradictions,
actual L0=CE/L1=SM actions, shared cache accounting and checkpoint rejection.
These are semantic and bookkeeping tests with declared costs, not new hardware
accuracy results or evidence of a better search algorithm.

Remaining work includes independently checked cost-free lowered graphs, region
types beyond public request-scoped Dense transformations, scoped cache or
resource-controller ownership, and mechanism comparisons under common candidate
sets. Multi-layer timing, real dynamic-resource accuracy and real-system policy
ranking retain their separately documented qualification limits.
