# Single-CPU/GPU request pipelines, experts and P/D handoff

The scope is one CPU, one GPU and their existing shared DRAM/VRAM/transport
ledger. All new declarations follow the same **Python -> IR -> composition
compiler -> global graph -> tensor runtime** path. This is a finite trace
simulator, not a production inference server or numerical language model.

## Responsibilities and representations

| Component | Owns | Does not supply |
|---|---|---|
| `RequestFlow` / modeling | Request arrivals, per-request lengths, model layers, exact token/KV dependencies, private MoE routes | A preferred execution device, a cache policy or future observations for mapping |
| Python / mapping | Public ready-work order, phase placement, explicit handoff, expert ordering/device rule, cache/prefetch/transport choices | Token generation, hidden future routes or another device/memory ledger |
| Compiler | Strict schemas and prices, scope/ownership checks, input producers, token gates and all-layer handoff operands | Guessed missing prices or free data movement |
| Modeling runtime | Clock, allocation, COPY/COMPUTE/EVICT, causality, dependencies, resource competition and retirement release candidates | OS task scheduling or a second P/D service cluster |

`hierarchical-strategy-0.5` records `request_pipeline` and `phase_handoffs`
beside the existing execution transforms and storage rules. Modeling accepts
`global-execution-0.2` request streams; legacy `global-execution-0.1` retains its
original fixed-round contract. A new frontend format does not alter the old
38-case acceptance manifest or its historical reports.

## Readable declarations

```python
from llm_simulator import (
    Request, RequestFlow, RequestPipeline, PhaseHandoff, PhasePlacement,
    Scheduling,
)

workload = RequestFlow(requests=(
    Request("interactive", prefill_tokens=2, decode_steps=2, arrival_ns=0),
    Request("background", prefill_tokens=3, decode_steps=1, arrival_ns=10000),
))

def configure(plan):
    plan.pipeline(RequestPipeline(ready_rule="decode_first"))
    PhasePlacement(
        prefill={"projection": "gpu", "attention": "gpu", "dense_ffn": "gpu"},
        decode={"projection": "cpu", "attention": "cpu", "dense_ffn": "cpu"},
    ).apply(plan)
    plan.handoff(PhaseHandoff(destination="dram"))

scheduling = Scheduling(default_device="gpu", release="last_use")
# simulator = Simulator(model=..., hardware=..., workload=workload,
#                       cost_model=AnalyticalCosts())
# simulator.compile(scheduling=scheduling, configure=configure, ...)
```

The complete profile-independent example supplies model geometry, all hardware
rates and capacities, and either Dense or explicit MoE routes:

```bash
python examples/library/request_pipeline.py --kind dense --out build/requests-dense
python examples/library/request_pipeline.py --kind moe --out build/requests-moe
```

Both require a new or empty output directory and work with the installed
library outside the repository. Neither downloads weights nor reads existing
profiles. The chosen analytical prices remain explicitly uncalibrated.

## Request lifecycle and execution pipeline

Each request has its own arrival, one complete Prefill and finite Decode steps.
The compiler validates B=1 leaf invocations and connects layers and rounds for
that request. It combines their graphs before constructing **one** runtime.
There is no per-request simulation and no whole-batch round barrier.

The model keeps future arrival boundaries in its private event queue. ADVANCE
can stop at an arrival during a long running operation without changing the
physical CPU/GPU state or exposing a future timer to the policy. Public
`arrived_requests` and `retired_requests` contain current/history information.

An explicit, positive-cost request-input producer materializes the first
hidden tensor and all layers' I32 positions after arrival. Future prompt inputs
are not reserved at time zero. Later token gates materialize supplied hidden
and position data after the request's own previous output. The illustrative
input/gate byte work is `(hidden_bytes_per_token + 4 * layers) * tokens`;
this does not execute embedding, sampling or actual token generation.
The format fixes input materialization and ordinary trace-token gates to CPU,
requiring explicit CPU prices. A first-Decode handoff gate instead uses its
declared destination and that device's exact price. JSON price-key order never
selects an executor, and an unused alternative price is not required.

Ready-work choices are:

- `graph_order`: preserve the declared public operation order.
- `decode_first`: stable priority for public Decode work, with dependencies and
  shared slots still deciding whether a command can run.
- `round_robin`: rotate public request priority after an actual compute command;
  preserve within-request order and retain priority state during replay.

These are independent-request pipelines. They do not fuse requests into a
larger GEMM or claim dynamic-batch throughput gains. The serialized accounting
contract is `independent_request_kernels`; requesting fused accounting fails.
Existing explicit B=2 contracts still distinguish invocation batch 2 from
kernel batch 1, and exact lookup rejects batch-axis substitutions.

## P/D handoff and storage lifetime

`PhasePlacement` specifies computation devices. `PhaseHandoff` separately
requires a complete first-Decode boundary: the Prefill final hidden output and
every Prefill KV block from every layer must be READY in `destination` before
the trace-token gate can run. They are real graph inputs; ordinary COPY actions
charge exact transfer prices, time, destination capacity and shared transport.

The first Decode KV consumers must agree with that destination. The graph
contract also constrains custom runtime device overrides. Later Decode rounds
may use separately declared placements and ordinary charged KV movement.
Request selectors expand explicitly; unknown, overlapping or Decode-less
handoffs fail. There is no KV clone, identity reset, network or extra P/D slot.

A request retires only after all its declared operations and final hidden
delivery finish. Its KV becomes a model-certified release candidate; the same
policy issues real EVICT actions. Final outputs remain declared goals, weights
remain shared/reusable, and tensor identities stay in the audit inventory.
Retirement is not a reset of another request's resources or state.

## Expert scheduling and combination rules

The same `Scheduling` object now exposes `expert_order`:

```python
scheduling = Scheduling(
    default_device="gpu", release="last_use",
    expert_order="token_count_desc",   # or history_frequency_desc / graph_order
    expert_rule="current_cost",        # or fixed / resident_first
)
```

Only router-revealed expert regions are candidates. Ordering uses distinct
revealed Token rows or explicitly known history; staged operations retain their
regional order. Request/phase priorities are stable outer priorities over this
expert ordering. One-request MoE leaves retain their real request identity,
rather than a batch pseudo-identity, in public scheduling labels.

Device selection uses public complete-region stage prices, actual tensor
payloads, residency, capacities and current CPU/GPU service rates. It estimates
isolated transfers and current-rate compute; it does not forecast recovery or
queue completion. The runtime arbitrates the actual contention. A region binds
its device only when a real preparation/compute command is returned.

Opaque MoE regions consume their declared full Attention/router tensors and
return outputs to declared combine consumers. No compact gather or proportional
COPY saving is invented. Explicit selected-row transforms retain their own
gather/scatter contracts. P/D placement can bind Attention/router/combine while
leaving expert execution adaptive across CPU/GPU.

Existing paper-specific queues remain available under their declared rules.
`hybrimoe_queues` owns its queue order, so combining it with a nondefault expert
or request order fails explicitly instead of silently ignoring one choice.

## Acceptance and remaining boundaries

The [serving acceptance set](../experiments/serving_acceptance/manifest.json)
adds 16 fixed requirements: six mechanisms, four combinations, four negative
cases and two causal contrasts. Expected operands, lifecycle obligations,
storage bytes and prefix boundaries are written before each run. The old
38-case set remains separate. Run both against a frozen implementation:

```bash
python experiments/ir_acceptance/run.py --out build/ir-regression
python experiments/serving_acceptance/run.py --out build/serving-acceptance
```

These finite synthetic checks concern expression, lowering, causality and
resource bookkeeping. They do not validate hardware timing or universal
strategy coverage. Performance calibration evidence remains frozen.

Unimplemented mechanisms include fused online batching, chunked Prefill,
prefix sharing/paged-KV ownership, speculative commit/rollback, distributed
communication and multiple GPUs. The current format deliberately rejects
unsupported contracts. Future device-group and communication extensions can
reuse task dependencies, explicit storage domains, transfer operands and one
resource ledger; no additional GPU or communication algorithm is claimed here.
