# Simulator 0.1.0a2

Unified Python research-library alpha. Python 3.10+, MIT-licensed
original code, no core third-party runtime dependencies.

## Included

- `llm_simulator`: public Model/Hardware/Workload/Simulator interface.
- Explicit on-demand analytical costs; no previously generated profile,
  model weights, GPU or CUDA installation is required to simulate a device.
- Finite multi-layer Dense and causal trace-driven MoE Prefill/Decode.
- Existing Python strategy IR/compiler, common tensor resource runtime,
  cache/prefetch/transport rules and advanced exact execution transforms.
- CLI examples, detailed results, checkpoint lifetime checks and standalone
  Python tutorials.
- Wheel/sdist isolated installation checks and semantic regression CI.

## Install

Download the `.whl` from this release and run:

```bash
python -m pip install llm_infer_simulator-0.1.0a2-py3-none-any.whl
llm-sim demo --model dense --out dense-demo
llm-sim demo --model moe --out moe-demo
```

Or install directly from the matching Git tag:

```bash
python -m pip install "git+https://github.com/AsukaZhenyu/Simulator.git@v0.1.0a2"
```

## Changes since 0.1.0a1

Legacy repository calibration accepts an explicit configuration root, so its
relative configuration, Nsight decomposition and context-validation inputs
remain usable when the runtime is installed from a wheel. Its regression
tests now supply that root explicitly. The original source-maintenance default
is retained; the public no-profile simulation API is unchanged.

## Research status

This is an alpha artifact, not a calibrated performance claim. Analytical
hardware numbers are caller declarations, and the example numbers are
fictional. The hardware calibration quality gate has not passed. Logical
test success is not evidence of numerical LLM correctness or real speedups.
Advanced decomposition requires explicit matching stage-cost contracts.
See `docs/LIBRARY_GUIDE.md` and `docs/OPEN_SOURCE_AND_PREPRINT.md` for limits
and remaining research work. Third-party routing data retains CC BY 4.0.
