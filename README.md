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

## Capabilities and boundaries

| Area | Current behavior |
|---|---|
| Strategy expression | Python modules/selectors, placement, cache, prefetch, transport, causal callbacks; existing exact Attention/FFN transforms |
| Workloads | Finite multi-layer Prefill→Decode traces, persistent weights/KV, Dense and trace-driven MoE |
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
| `src/llm_simulator/` | Public API, generated inputs, analytical provider and installed CLI |
| [mapping](mapping/README.md) | Strategy frontend, IR, execution transforms, compiler and causal policies |
| [modeling](modeling/README.md) | LLM state/graph expansion, common tensor runtime, capacity/resource accounting |
| [profiling](profiling/README.md) | Optional native measurement tools; not needed to install or use the simulator |
| [docs](docs/CODE_LAYOUT.md) | Technical guides and architecture |

Existing `tensor_mapping` and `llm_infer_model` imports remain available. Legacy measurement-maintenance commands that read repository fixtures are source-checkout tools, outside the installed `llm-sim` workflow.

## Validation

```bash
python -m unittest discover -s tests -p test_library.py
python -m unittest discover -s tests -p test_analytical_inputs.py
python tests/test_distribution.py
python -B -m unittest discover -s modeling/tests
python -B -m unittest discover -s mapping/tests -t mapping
python -B -m unittest discover -s profiling/tests
```

Distribution checks build wheel/sdist and run Dense/MoE in a fresh virtual environment outside the source tree, without profiles. CI checks the library and semantic regressions. Optional native/GGUF tests retain explicit prerequisites.

## Research experiments

The [research evidence report](docs/RESEARCH_EVIDENCE_20261017.md) separates independent strategy checks, measured-component replay, analytical design exploration and hardware acceptance. Reusable source-checkout experiments include [strategy fidelity](docs/STRATEGY_FIDELITY.md), [hardware sensitivity](docs/HARDWARE_SENSITIVITY.md) and [repeated completed-work calibration](profiling/batched_calibration/README.md). Analytical curves illustrate declared assumptions; successful compilation or replay does not establish hardware accuracy. A policy that cannot advance is reported as a policy dead end, without claiming that every mapping on the hardware is infeasible.

The [composition calibration study](docs/COMPOSITION_CALIBRATION_20261018.md) adds profile-only shared corrections and exact IR accounting for retained FFN/Attention chains. Confirmation median error decreases from 13.65% to 6.14%; held-out-geometry worst error is still 34.85% and quality acceptance remains 0/10. This retrospective, opt-in workflow preserves the original runtime timeline and does not replace the library's uncalibrated analytical defaults.

The [accuracy validation suite](profiling/accuracy_suite/README.md) adds complete compiled Dense strategy replay, execution-chain diagnostics, real multi-layer Dense/MoE traces, tracked allocator boundaries, and asynchronous copy/compute experiments. Optional source-checkout tooling uses `pip install ".[profiling]"`; the installed simulator still needs no profiles or CUDA. The [validation report](docs/ACCURACY_VALIDATION_20261018.md) separates numerical correctness, prediction error, measurement stability and physical capacity. The suite does not establish universal hardware accuracy or change the analytical defaults.

The primary state-dependent modeling target is [one execution-time functional over resource traces](docs/UNIFIED_RESOURCE_FUNCTIONAL.md): workload and hardware constants plus time-varying CPU/GPU pressure and DRAM/VRAM allocations enter one prediction path. The [resource-trajectory workflow](profiling/resource_calibration/README.md) collects numeric controlled inputs while a foreground process continues running. It separates offered background work from target-inclusive utilization counters. The [first implementation and hardware check](docs/UNIFIED_RESOURCE_RESULTS_20261018.md) records the working integration and the response model's remaining accuracy failures.

The earlier [state-aligned calibration workflow](profiling/state_calibration/README.md) and [device-state study](docs/DEVICE_STATE_CALIBRATION_20261018.md) remain diagnostic controls, including all failed gates; their two historical table families are not the target simulator interface. [Explicit CPU/GPU service rates](docs/DEVICE_STATE_RUNTIME.md) support mid-operation speed changes, pauses and recovery with exact remaining-work accounting; they do not infer an unmeasured slowdown from utilization or clocks.

## License and research use

Original code uses the [MIT License](LICENSE). Attributed third-party data retains its own license; see [public routing data](profiling/data/public-routes/README.md). This is an **alpha research library**, with no associated accepted paper or accuracy claim. Citation metadata: [CITATION.cff](CITATION.cff).
