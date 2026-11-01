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

All hardware numbers above are **illustrative inputs**. `workspace_bytes=0` declares idealized scratch, not measured zero workspace. MoE routes, phase placement, environment events and exact-cost inputs are documented in the module docstrings under `src/llm_simulator/` and exercised by the runnable files in [examples/library](examples/library/README.md).

For one physical memory pool shared by CPU/GPU, use
`Hardware.shared_memory(cpu=..., gpu=..., capacity_bytes=...)` and inspect
`result.summary()["physical_memory_peaks_bytes"]`. Same-buffer `SHARE` moves zero
payload; compute memory traffic and explicit real copies still cost time. The
[runnable example](examples/library/shared_memory.py) is the reference for this
mode. Shared visibility is a declared abstraction, not a calibrated
Apple/Jetson/Spark profile.

## Inspect and express a strategy

Python module scopes organize execution placement and weight preparation/transfer
rules before compilation. `plan.explain()` shows declared choices and unresolved
checks; `plan.inspect()` can check declarations even when prices are missing.
It does not certify full semantic validity or make an unpriced plan executable.
Strict compilation and execution still require applicable cost contracts.

The strategy frontend couples execution and storage dimensions; the conflict
rules and current limits are stated in the docstrings of
`mapping/tensor_mapping/python_frontend.py` and enforced by the `strategy-ir`
suite. Run the [standalone example](examples/library/strategy_ir.py) with
`python examples/library/strategy_ir.py --out strategy-example` to inspect a
two-layer plan, use CE/SM for different layers, and execute it with one shared
memory ledger. The example needs no saved profile and makes no hardware
accuracy claim. Use a new or empty output directory.

For independently arriving requests, the request-flow frontend in
`mapping/tensor_mapping` adds public request/Token lifecycles, ready-work
ordering, causal expert rules, and an explicit all-layer P/D KV handoff in the
same IR/compiler/runtime. Run
`python examples/library/request_pipeline.py --kind moe --out request-example`.
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

**The hardware calibration quality gate has not passed.** Do not interpret demo speedups as measured benefits or claim reproduction of entire published systems.

## Repository layout

| Directory | Responsibility |
|---|---|
| `src/llm_simulator/` | Public API, generated inputs, modeling facade and installed CLI |
| [mapping](mapping/README.md) | Strategy frontend, IR, execution transforms, compiler and causal policies |
| [modeling](modeling/README.md) | LLM state/graph expansion, common tensor runtime, capacity/resource accounting |
| `profiling/llm_infer_profile/` | Library package imported by modeling for calibration and provenance conversion |
| [docs](docs/CODE_LAYOUT.md) | Architecture notes and the reproduction guide |
| [tools](tools/run_tests.py) | Runs every suite, and writes a Markdown report with `--report` |

Existing `tensor_mapping` and `llm_infer_model` imports remain available.

## Validation

One command runs every suite that CI runs:

```bash
python -m pip install -e .
python tools/run_tests.py                        # 11 suites
python tools/run_tests.py -k mapping             # one group
python tools/run_tests.py --list                 # what exists
python tools/run_tests.py --report report.md     # Markdown report of the run
```

`--report` writes the environment (OS, CPU, GPU, Python, dependency versions),
the repository commit, one row per suite and the complete output of every
failing suite, so a run on another machine can be diagnosed without it.

Individual suites are plain `unittest` targets:

```bash
python -m unittest discover -s tests -p test_library.py
python -m unittest discover -s tests -p test_analytical_inputs.py
python -B -m unittest discover -s modeling/tests
python -B -m unittest discover -s mapping/tests -t mapping
```

Neither the runner nor any suite needs a GPU, CUDA, model weights or network
access. Setup from a fresh clone, the expected numbers, the meaning of each
suite and the second-machine checklist are in
[Reproduction and tests](docs/REPRODUCTION.md).

`tests/test_distribution.py` is deliberately not in the runner: it builds a wheel
and an sdist and runs the demo in a fresh virtual environment outside the source
tree.

## What is not in this repository

The measurement apparatus, campaign outputs and study documents behind the
paper's numbers are kept with the paper, not here. This repository ships the
simulator and the tests for it; it does not contain CUDA probes, recorded traces
or measured profiles.

## License

Original code uses the [MIT License](LICENSE). This is an **alpha research
library**, with no associated accepted paper or accuracy claim. Citation
metadata: [CITATION.cff](CITATION.cff).
