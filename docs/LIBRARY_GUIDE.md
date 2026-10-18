# Python library guide

Distribution: `llm-infer-simulator`; public import: `llm_simulator`.
The convenience API wraps the existing compiler and common runtime.

## The workflow

1. `Model.dense(...)` / `Model.moe(...)`: geometry and layer count.
2. `Hardware(...)`: explicit capacities, compute rates and bandwidth.
3. `Workload(...)`: prompt length, finite Decode steps and routing input.
4. `AnalyticalCosts(...)`: explicit uncalibrated analytical estimates.
5. `Scheduling(...)` and `configure(plan)`: strategy declarations.
6. `Simulator.compile(...)`: generated inputs and immutable strategy IR.
7. `CompiledSimulation.run()`: the existing causal tensor runtime.

No saved profile, model weights, native executable, source-tree fixture or
GPU is required. Simulated devices are described by the caller.

```python
from llm_simulator import PhasePlacement, DataPreparation
from llm_simulator.demo import make_demo

simulator, scheduling = make_demo("dense")

def configure(plan):
    PhasePlacement(
        prefill={"projection": "gpu", "attention": "gpu"},
        decode={"projection": "cpu", "attention": "cpu"},
    ).apply(plan)
    plan.prepare(DataPreparation(layer_id="layer1", destination="vram", trigger="spare"))

with simulator.compile(scheduling=scheduling, configure=configure,
                       directory="phase-project") as compiled:
    result = compiled.run()
    assert result.success, result.summary()
    result.save("phase-results")
```

Modules are selector scopes, without extra execution barriers. Dependencies,
token gates, router visibility and capacity remain model constraints. Phase
placement is not a complete distributed/disaggregated serving deployment.

## Analytical cost assumptions

The provider derives macro work and declared traffic from the model geometry
and invocation shape, using:

`duration_ns = ceil(max(FLOPs / compute_rate, declared_bytes / memory_bandwidth) * 1e9) + launch_overhead_ns`

Transfers use `ceil(storage_bytes / link_bandwidth * 1e9) + transfer_latency_ns`.
Times remain positive integer nanoseconds. Exact shape entries are generated
on demand, with assumptions recorded in `library-inputs.json` and cost inputs.

- Rates are caller inputs; demo hardware is fictional. Nothing is calibrated.
- Fusion, cache reuse, kernel choice, quantization performance and concurrent
  bandwidth interference are not automatically inferred.
- `workspace_bytes` declares per-macro scratch. Zero is an idealized
  assumption, not a measured allocator fact. Weights, inputs, outputs and KV
  are separately accounted by the runtime.
- Dense CPU-only and hybrid paths retain their existing physical contracts.
  The Dense builder currently accepts one request; the underlying exact
  signature validator does not support fused multi-request batches.
- MoE uses opaque synthetic expert macros and evaluator-owned causal replay,
  not numerical routing or a measured expert kernel.
- MoE's existing `.6` contract requests DRAM↔VRAM and SSD read/write prices
  even when a policy uses only some routes. Supply transfer/read/write
  bandwidths; missing rates are rejected rather than guessed.
- This convenience builder starts all weights in DRAM, so the initial DRAM
  budget must fit the complete declared weights. SSD initial placements and
  custom inventories require explicit advanced `ModelPlan` model inputs.
  Dense `.3`/`.5` does not support SSD transfers.
- Compute is modeled by serial resource slots and explicit overlap rules,
  not arbitrary streams, distributed topologies or an operating system.

## MoE routes

Each token/request/layer needs `top_k` distinct expert IDs. The library does
not invent routing for a user's model. Identities are `request0`, `request1`,
... and `layer0`, `layer1`, ... .

```python
from llm_simulator import Route, Workload

routes = tuple(
    Route(request_id="request0", layer_id=f"layer{layer}",
          absolute_position=position, expert_ids=(0, 2))
    for layer in range(2) for position in range(5)
)
workload = Workload(prefill_tokens=3, decode_steps=2,
                    routes=routes, route_source="my-explicit-routing-trace")
```

Demo round-robin routes are labeled synthetic. Policy receives only revealed
routes and declared history. Full route files/compiled debug graphs are
private evaluator inputs, never policy observations.

## Device state changes

```python
from llm_simulator import EnvironmentEvent
from llm_simulator.demo import make_demo

simulator, scheduling = make_demo("dense", device="cpu")
events = (
    EnvironmentEvent(t_ns=0, cpu_available=False),
    EnvironmentEvent(t_ns=10000, cpu_available=True),
    EnvironmentEvent(t_ns=15000, vram_external_bytes=1024),
)
result = simulator.run(scheduling=scheduling, environment_events=events)
print(result.summary())
```

CPU unavailability pauses service under runtime rules. External memory is an
absolute allocation in addition to reserves/tensors. Infeasible allocations
remain failures. This builder supports `environment-trace-0.1`; reclaim with
explicit control costs uses `ModelPlan` and `environment-trace-0.2`.

CPU/GPU compute rates can also change while work is running, using exact
`cpu_service_rate_numerator`/`cpu_service_rate_denominator` or corresponding
`gpu_` fields. Remaining work survives slowdowns and pauses without restarting;
these rates are explicit inputs, not inferred from utilization. See
[device-state runtime semantics](DEVICE_STATE_RUNTIME.md) for examples, current
price estimates, checkpoint behavior and calibration limits.

## Exact costs and execution transforms

```python
from llm_simulator import ModelPlan, Scheduling

plan = ModelPlan(id="my-exact-plan", model_kind="global_m2",
    model_file="my-inputs/model.global.json",
    environment_file="my-inputs/environment.json",
    scheduling=Scheduling(default_device="gpu", release="last_use"))
# Exact transform inputs belong to the caller:
# plan.transform_file("layer0-prefill", "my-inputs/attention-stages.json")
# plan.transform_file("layer0-prefill", "my-inputs/ffn-partitions.json")
compiled = plan.compile()
run, session, policy, reconstructed_commands = compiled.run()
```

The analytical macro builder does not guess arbitrary new decomposition
costs. Existing Attention stages, FFN partitions, MoE selected rows and their
combinations retain exact validation. Missing stage prices, workspace,
transfer sizes and incompatible boundaries are rejected. See
[execution transforms](../mapping/docs/EXECUTION_TRANSFORMS_GUIDE.md) and
[Python frontend](../mapping/examples/python/README.md).

Saved-source `RuntimePolicy` subclasses can be installed in `configure(plan)`
using `plan.runtime_policy(MyPolicy)`. Callbacks get public observations.
Unsaved notebook/REPL classes are rejected because source binding is required.
Python callbacks are trusted caller code, not a security sandbox.

The legacy layer-calibration helper can also be used from an installed wheel
with caller-provided measurements and configurations:

```python
from llm_infer_model.calibration import calibrate_configs
calibrate_configs("my-measurements.json", "calibrated",
                  config_root="my-input-directory")
```

The explicit root anchors referenced configuration, Nsight and context files
and relative output directories. Omitting it preserves the old source-checkout
maintenance default; the installed `llm-sim` workflow never calls this helper
or loads those historical measurements implicitly.

## Results and source lifetime

`result.success` requires valid evaluation and reaching the goal. Invalid or
incomplete runs retain their reasons. No optimality or hardware speedup is
implied. `result.raw` exposes the unfiltered original result tuple.

`result.save(new_directory)` writes IR, events, states and audit documents,
`library-summary.json`, and a generated input snapshot. Checkpoints bind the
original source paths/bytes and runtime hashes. Keep the original project
and resume through the same compiled object; snapshots are not portable
executable archives of arbitrary Python policies.

Without `directory=`, the compiled object owns temporary files until close
or garbage collection. Results retain that object. A context manager closes
it deterministically. Closing an explicit directory never deletes the
caller's project. Inputs and results must be in separate directories.

```python
simulator, scheduling = make_demo("dense")
with simulator.compile(scheduling=scheduling, directory="checkpoint-project") as compiled:
    partial = compiled.run(max_decisions=2)
    resumed = compiled.run(checkpoint=partial.checkpoint())
    assert resumed.success
```
