# A Simulator for Large Language Model Inference Systems on Heterogeneous Hardware Platforms

**Simulator** is a Python research library for expressing, compiling and simulating LLM execution and storage strategies on heterogeneous hardware.

**Python strategy → IR → compiler → one tensor/resource runtime → timeline and memory accounting.**

An installable research prototype is currently available: it can simulate a declared CPU/GPU platform without downloading model weights, using an existing profile, or installing CUDA locally. The analysis produces **estimates that have not been calibrated against hardware**; passing semantic regression tests does not establish performance accuracy.

## Install

Python **3.10+**. The core has **no third-party runtime dependencies**.

```bash
python -m pip install "git+https://github.com/AsukaZhenyu/Simulator.git@main"
llm-sim --version
llm-sim demo --model dense --out dense-demo
llm-sim demo --model moe --out moe-demo
```

Alternatively, download the repository ZIP, extract it, and run `python -m pip install .` inside it. Downloadable wheels and source archives are available from [GitHub Releases](https://github.com/AsukaZhenyu/Simulator/releases). A wheel can be installed with `python -m pip install path/to/llm_infer_simulator-0.1.0a2-py3-none-any.whl`. This project has **not been published to PyPI**; a bare `pip install llm-infer-simulator` is not the documented installation path.

For development use `python -m pip install -e .`. There is one installation entry point; installing the two internal subprojects separately is unnecessary.

## First Python run

The demo declares fictional hardware parameters and an explicit analytical cost model. It never reads a saved profile or detects the host GPU.

```python
from llm_simulator.demo import make_demo

simulator, scheduling = make_demo("dense")  # or "moe": explicit synthetic routes
with simulator.compile(scheduling=scheduling, directory="my-project") as compiled:
    result = compiled.run()
    print(result.status, result.makespan_ms)
    result.save("my-results")
```

Use new or empty directories. CLI runs save generated inputs under `project/` and detailed results under `result/`: strategy IR, compiled graph, actions, timeline events, memory states, policy audit and cost assumptions. Keep the original project when using checkpoints; copied input snapshots are audit evidence, not portable checkpoint identities.

## Declare your own experiment

```python
from llm_simulator import (
    AnalyticalCosts, Cache, Device, Hardware, Model, Scheduling, Simulator, Workload,
)

model = Model.dense(layers=2, hidden_size=128, intermediate_size=384,
                    heads=4, kv_heads=2)
hardware = Hardware(
    cpu=Device(compute_flops_per_second=100e9,
               memory_bandwidth_bytes_per_second=20e9, launch_overhead_ns=500),
    gpu=Device(compute_flops_per_second=1e12,
               memory_bandwidth_bytes_per_second=100e9, launch_overhead_ns=2000),
    dram_capacity_bytes=2 * 1024**3, vram_capacity_bytes=128 * 1024**2,
    transfer_bandwidth_bytes_per_second=8e9, transfer_latency_ns=1000,
)
simulator = Simulator(model=model, hardware=hardware,
    workload=Workload(prefill_tokens=32, decode_steps=2),
    cost_model=AnalyticalCosts(workspace_bytes=0))
strategy = Scheduling(default_device="gpu", placements={"dense_ffn": "cpu"},
                      release="last_use", cache=Cache(rule="lru", budget_bytes=64 * 1024**2))
result = simulator.run(scheduling=strategy)
print(result.summary())
```

All hardware numbers above are **illustrative inputs**. `workspace_bytes=0` declares idealized scratch, not measured zero workspace. See the [Python API guide](docs/LIBRARY_GUIDE.md) for assumptions, MoE routes, phase placement, environment events and exact-cost inputs.

For one physical memory pool shared by CPU/GPU, use
`Hardware.shared_memory(cpu=..., gpu=..., capacity_bytes=...)` and inspect
`result.summary()["physical_memory_peaks_bytes"]`. Same-buffer `SHARE` moves zero
payload; compute memory traffic and explicit real copies still cost time. See
the [shared-memory guide](docs/SHARED_MEMORY.md) and
[runnable example](examples/library/shared_memory.py). Shared visibility is a
declared abstraction, not a calibrated Apple/Jetson/Spark profile.

## Inspect and express a strategy

Python module scopes organize execution placement and weight preparation/transfer
rules before compilation. `plan.explain()` shows declared choices and unresolved
checks; `plan.inspect()` can check declarations even when prices are missing.
It does not certify full semantic validity or make an unpriced plan executable.
Strict compilation and execution still require applicable cost contracts.

The [strategy expression and IR guide](docs/STRATEGY_REPRESENTATION.md) explains
the coupled execution and storage dimensions, conflict rules and current limits.
Run the [standalone example](examples/library/strategy_ir.py) with
`python examples/library/strategy_ir.py --out strategy-example` to inspect a
two-layer plan, use CE/SM for different layers, and execute it with one shared
memory ledger. The example needs no saved profile and makes no hardware
accuracy claim. Use a new or empty output directory.

For independently arriving requests, the [single-CPU/GPU request-flow guide](docs/SINGLE_CPU_GPU_REQUEST_FLOW_20261023.md)
adds public request/Token lifecycles, ready-work ordering, causal expert rules,
and an explicit all-layer P/D KV handoff in the same IR/compiler/runtime.
Run `python examples/library/request_pipeline.py --kind moe --out request-example`.
Kernels are independently priced B1 requests; this does not imply fused batching.

## Capabilities and boundaries

| Area | Current behavior |
|---|---|
| Strategy expression | Python modules/selectors, placement, cache, prefetch, transport, causal callbacks; existing exact Attention/FFN transforms |
| Workloads | Finite multi-layer Prefill→Decode traces and independent request arrivals/retirement, persistent weights/KV, Dense and trace-driven MoE |
| Hardware | Declared CPU/GPU compute and DRAM/VRAM capacity; Dense DRAM↔VRAM transfers, with SSD routes in the advanced/MoE contract |
| Dynamic state | External memory and CPU availability events; explicit resource reclaim through the advanced API |
| No-profile entry | On-demand estimates from declared geometry, rates and bandwidth; always labeled uncalibrated |
| Exact-cost entry | `ModelPlan` accepts supplied costs/contracts; missing shapes, workspace or routes are rejected without analytical substitution |
| Not provided | Numerical inference, actual token generation, a general distributed serving runtime or an optimal global searcher |

The convenience builder uses full-causal F32 geometry. Dense batch and decomposition support follow existing physical contracts; unsupported combinations fail explicitly. MoE routes are evaluator inputs, visible to policy after the router barrier. Phase placement expresses P/D assignments on this platform, without creating separate serving clusters or network links.

**The hardware calibration quality gate has not passed.** Do not interpret demo speedups as measured benefits or claim reproduction of entire published systems. [Release and preprint preparation](docs/OPEN_SOURCE_AND_PREPRINT.md) records the remaining evidence needed.

## Repository layout

| Directory | Responsibility |
|---|---|
| `src/llm_simulator/` | Public API, generated inputs, modeling facade and installed CLI |
| [mapping](mapping/README.md) | Strategy frontend, IR, execution transforms, compiler and causal policies |
| [modeling](modeling/README.md) | LLM state/graph expansion, common tensor runtime, capacity/resource accounting |
| [profiling](profiling/README.md) | Optional native measurement tools; not needed to install or use the simulator |
| [docs](docs/CODE_LAYOUT.md) | Technical guides and architecture |

Existing `tensor_mapping` and `llm_infer_model` imports remain available. Legacy measurement-maintenance commands that read repository fixtures are source-checkout tools, outside the installed `llm-sim` workflow.

## Validation

One command runs every suite that CI runs, after `python -m pip install -e ".[profiling]"`:

```bash
python tools/run_tests.py              # 36 suites
python tools/run_tests.py -k mapping   # one group
python tools/run_tests.py --list       # what exists
```

Individual suites are plain `unittest` targets:

```bash
python -m unittest discover -s tests -p test_library.py
python -m unittest discover -s tests -p test_analytical_inputs.py
python tests/test_distribution.py
python -B -m unittest discover -s modeling/tests
python -B -m unittest discover -s mapping/tests -t mapping
python -B -m unittest discover -s profiling/tests
```

Neither the runner nor any suite needs a GPU, CUDA, model weights or network
access. Setup from a fresh clone, the expected numbers, the meaning of each
suite and the second-machine checklist are in
[Reproduction and tests](docs/REPRODUCTION.md).

Distribution checks build wheel/sdist and run Dense/MoE in a fresh virtual environment outside the source tree, without profiles. CI checks the library and semantic regressions. Optional native/GGUF tests retain explicit prerequisites.

## Research experiments

The [initial IR baseline](docs/IR_ACCEPTANCE_BASELINE_20261022.md) recorded shared
expression at 11/12 and legal composition at 9/12. The
[request-region composition completion](docs/IR_COMPOSITION_COMPLETION_20261022.md)
raises both to 12/12 through the common frontend, compiler and runtime, with 8/8
invalid cases and 6/6 paired authoring tasks retained. The same fixed manifest
and old evidence are preserved. These are finite synthetic semantic checks, not
hardware performance results or universal strategy coverage. The
[reusable benchmark](experiments/ir_acceptance/README.md) records every case and
its independent expected behavior.

The [single-CPU/GPU request and expert extension](docs/SINGLE_CPU_GPU_ACCEPTANCE_20261023.md)
retains all 38 original passes and adds 16/16 fixed arrival, pipeline, P/D,
expert, negative and causal-prefix requirements. It includes exact shared-weight
inventories and real KV release/transfer checks, with unchanged frozen accuracy
evidence. See the [new benchmark](experiments/serving_acceptance/README.md) for
portable semantic testing and source-bound research runs.

The [expanded fixed-hardware comparison](docs/STRATEGY_COMPARISON_CAMPAIGN_20261023.md)
crosses Dense/MoE Prefill and Decode workloads, synthetic expert routes, execution
placement and same-budget cache/prefetch choices. Its
[352-run results](docs/STRATEGY_COMPARISON_RESULTS_20261023.md) retain every declared
cell, exact ties, negative outcomes and two equivalence controls (14 distinct
inputs plus two control scenario IDs). Reusable campaign, independent event
audit and readable-report commands are included. These are uncalibrated
analytical comparisons of a finite catalogue, not real-device speedups or a
new optimal search method.

The [research evidence report](docs/RESEARCH_EVIDENCE_20261017.md) separates independent strategy checks, measured-component replay, analytical design exploration and hardware acceptance. Reusable source-checkout experiments include [strategy fidelity](docs/STRATEGY_FIDELITY.md), [hardware sensitivity](docs/HARDWARE_SENSITIVITY.md) and [repeated completed-work calibration](profiling/batched_calibration/README.md). Analytical curves illustrate declared assumptions; successful compilation or replay does not establish hardware accuracy. A policy that cannot advance is reported as a policy dead end, without claiming that every mapping on the hardware is infeasible.

The [composition calibration study](docs/COMPOSITION_CALIBRATION_20261018.md) adds profile-only shared corrections and exact IR accounting for retained FFN/Attention chains. Confirmation median error decreases from 13.65% to 6.14%; held-out-geometry worst error is still 34.85% and quality acceptance remains 0/10. This retrospective, opt-in workflow preserves the original runtime timeline and does not replace the library's uncalibrated analytical defaults.

The [accuracy validation suite](profiling/accuracy_suite/README.md) adds complete compiled Dense strategy replay, execution-chain diagnostics, real multi-layer Dense/MoE traces, tracked allocator boundaries, and asynchronous copy/compute experiments. Optional source-checkout tooling uses `pip install ".[profiling]"`; the installed simulator still needs no profiles or CUDA. The [validation report](docs/ACCURACY_VALIDATION_20261018.md) separates numerical correctness, prediction error, measurement stability and physical capacity. The suite does not establish universal hardware accuracy or change the analytical defaults.

Performance predictions are computed in [modeling](docs/PERFORMANCE_BOUNDARIES.md), using physical work, dependencies, parameters and resource trajectories. The installed `llm_simulator.performance` API shares that same implementation; `AnalyticalCosts` also delegates its compute and transfer formulas to modeling. [Profiling parameter calibration](profiling/parameter_calibration/README.md) owns measurement sources, exact medians, quality gates and frozen parameter evidence, and calls modeling for predictions during independent validation. It reports parameter evidence and model residuals separately.

For new physical experiments, [external available resources](docs/AVAILABLE_RESOURCE_MODEL_20261022.md) are the input boundary: absolute CPU/GPU service, bandwidth and already-available memory budgets. `AvailableResources` and `ServiceBinding` lower scoped physical work to the same evaluator; OS scheduling stays outside the simulator. The [installed example](examples/library/available_resources.py) needs no profile or native hardware. [Optional profiling](profiling/available_service/README.md) estimates physical efficiency from independently evidenced service and preserves same-invocation correlations in older diagnostic traces. This boundary correction does not establish new hardware accuracy or silently recalibrate the default analytical LLM workflow.

The [complete single-layer calibration study](docs/SINGLE_LAYER_CALIBRATION_20261018.md) measures actions in their real CPU/GPU/copy order and validates frozen model-owned predictions with 42 fresh processes at 14 small F32 Dense strategy cases. Primary pooled-median error is 8.54% at the median and 18.39% at worst, versus an 82.61% worst isolated-component baseline error on the same heldout data. All 14 primary errors are within 20%, but **complete stability/compatibility acceptance remains 0/14**. The optional distribution candidate does not consistently improve the primary model, and the library's analytical defaults remain uncalibrated. [Reusable source-checkout commands](profiling/single_layer_calibration/README.md) preserve all failed gates and independent evidence.

The separate [CUDA interval diagnosis](docs/SINGLE_LAYER_CUDA_WAIT_DIAGNOSIS_20261018.md) attributes 808 instrumented PV invocations. In this small materialized-port configuration with CUDA graphs disabled, GPU activity remains nearly constant while host CUDA runtime API wall time dominates the observed action. This is evidence for explicit backend-call/layout boundaries, not a calibrated GPU throughput value or a universal synchronization penalty.

The [earlier total-duration response experiment](docs/UNIFIED_RESOURCE_RESULTS_20261018.md) is a historical diagnostic. Its Ridge predictor is not the primary performance model. Its raw resource trajectories, failed gates and source version are preserved; whole-PV medians cannot silently become pure GPU throughput or bandwidth parameters.

The earlier [state-aligned calibration workflow](profiling/state_calibration/README.md) and [device-state study](docs/DEVICE_STATE_CALIBRATION_20261018.md) remain diagnostic controls, including all failed gates; their two historical table families are not the target simulator interface. [Explicit CPU/GPU service rates](docs/DEVICE_STATE_RUNTIME.md) support mid-operation speed changes, pauses and recovery with exact remaining-work accounting; they do not infer an unmeasured slowdown from utilization or clocks.

The [continuous state-response study](docs/STATE_RESPONSE_CALIBRATION_20261021.md) separates model-owned progress and response delays from profiling-owned parameters and synchronized observations. Fresh-process validation **has not passed**: repeated CPU/GPU replay requests do not produce a reproducible contention state. The [reusable collector](profiling/state_response/README.md) preserves the failed gates, actual owned-process CPU activity, runtime receipts and independent heldout results. This experimental response path does not calibrate the default LLM workflow.

The subsequent [thread/runtime/recovery investigation](docs/THREAD_RUNTIME_RECOVERY_DIAGNOSIS_20261022.md) adds sixteen controlled runs, a bounded kernel trace and an owned main-thread placement intervention. Modeling now distinguishes completion gates, active thread CPU service and explicit same-core ISR/DPC interruptions. A separate three-profile/three-fresh-process validation covers 686 invocations: primary median relative error is 10.50%, P90 is 39.80%, and GPU-load subsets remain inaccurate. Full accuracy remains unqualified; these mechanisms do not yet supply a calibrated predictive law for driver interruptions or GPU contention.

The earlier [shared stateful service-response hypothesis](docs/UNIFIED_SERVICE_RESPONSE_20261022.md) maps external requests and residual competition proxies to CPU/GPU service rates, then delegates work progress to the common evaluator. [Its profiling workflow](profiling/service_response/README.md) freezes measured parameters before fresh-process validation. In that exact-context experiment, all 3,165 calls are predictable; the median of three process-level median relative errors is 9.43%, pooled P90 is 33.56%, and combined CPU/GPU competition remains inaccurate. The reservoir is a declared interference proxy, not an identified hardware queue; source stability and full accuracy have **not passed**. These request inputs are not independently measured available service, and the experiment remains diagnostic evidence.

## License and research use

Original code uses the [MIT License](LICENSE). Attributed third-party data retains its own license; see [public routing data](profiling/data/public-routes/README.md). This is an **alpha research library**, with no associated accepted paper or accuracy claim. Citation metadata: [CITATION.cff](CITATION.cff).
