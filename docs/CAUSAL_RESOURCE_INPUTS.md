# External resource functions and inference-owned state

The intended resource input is a **declared external scenario**, not a forecast
of OS scheduling. It consists of CPU/GPU service budgets, memory budgets and
other applications' resource functions over time. Inference-owned work,
placements, live tensors, queues and completed actions are internal model state.

## Two valid budget conventions

Use exactly one convention for each resource:

1. **Shared pool minus external reservations.** Supply a pool capacity C(t)
   and independent reservations E(t) for other applications. The inference
   budget is C(t) - E(t), with feasibility checked against the declared pool.
2. **Net remaining budget.** Supply the already-remaining inference budget
   A(t). Other application reservations must not be subtracted again.

A planned demand is different from a reservation. If competing demands exceed
capacity, declare how the pool is shared, for example fixed quotas or weights.
This is an abstract resource-sharing rule; it does not require simulating OS
threads, time slices or dispatch decisions. Capacity and memory must still
remain feasible under that rule.

Compute service, memory capacity and transfer bandwidth are separate quantities.
For example, a VRAM reservation lowers the capacity available to the strategy;
it does not automatically imply a proportional reduction in GPU arithmetic
throughput. Required evictions and moves remain execution actions.

## Inference changes its own state

At an event time t:

1. Read the external budget at t and the current internally modeled state.
2. Determine which actions are legal and their available service rates.
3. Advance work until a completion, resource change or other modeled event.
4. Update progress, queues, tensor placement and owned memory.
5. Repeat from the new time and state.

In notation, x is the inference-owned state and u is the external scenario:

    r_i(t) = g_i(operation, shape, backend, x(t), u(t))
    x(t + delta) = F(x(t), action progress, completions, u)

The integral of r_i determines completion. The simulator does not need the
future realized total state of that inference execution as an input. A fixed
resource scenario is a conditional experiment: its result does not claim to
predict arbitrary real background applications.

## What can see the future

The **performance evaluator** may receive the complete externally declared
scenario to replay a known experiment. A **mapping policy** may see an entire
future reservation plan only if that plan is explicitly announced before its
decision. Otherwise it receives only current and past available observations.

Measured current state can include effects of earlier inference. That is valid
historical feedback when its measurement window has completed before the
decision. It is not automatically target-independent capacity.

Invalid shortcuts include:

- Using the average utilization measured over the current call to predict
  that same call online: the input includes observations unavailable at entry.
- Using policy A's realized total GPU utilization, frequency or memory trace
  unchanged as policy B's external environment: B can change those quantities.
- Subtracting measured total VRAM usage and then charging the inference's own
  tensors again: this double-counts owned occupancy.
- Treating a utilization percentage as a universal throughput reduction.

Same-policy replay using a measured realized trace can be useful as an explicit
diagnostic. It is not an online prediction or proof of counterfactual validity.
Frequency and temperature affected by the strategy require an internal response
model, a controlled fixed-state assumption, or a clearly bounded diagnostic.

## Current implementation boundary

`llm_simulator.steady_costs` exposes `ResourceInput`,
`validate_resource_input` and `validate_budget_contract`. They validate declared
provenance, information availability, accounting convention and policy scope.
They do not prove that a caller's independence declaration is physically true,
measure a new resource trace, or implement a utilization-to-throughput model.

`ResourceInput` contains provenance and visibility metadata, not numerical
resource-function samples. `validate_budget_contract` checks the declared
accounting convention; it does not subtract numeric reservations, reject a
negative computed budget, or execute a named demand-sharing rule. The numerical
`AvailableResources` interface takes **net budgets after external reservations
only**, before charging inference-owned tensors and workspace, and physical
service rates. Its common evaluator charges owned state internally. Inclusive
steady action wall-time prices cannot automatically be scaled by a FLOP/s
budget: a separately calibrated physical service binding is required. The
same-policy diagnostic guard also requires a matching policy for noninvariant
realized state, even when raw own-work counters are absent from that state.

The steady-cost calibration uses no current/future measured target-state
features. It freezes measured action prices before collecting heldout data.
GPU stream intervals remain overlapping diagnostics, not extra additive costs.
