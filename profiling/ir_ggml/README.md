# IR → ggml: compile a declared policy and time it

This directory holds the end-to-end apparatus behind the paper's Section V-A/B: an
**E/S policy corpus**, a compiler from a policy to an executable ggml graph, and a
driver that compares what the graph costs on this machine against the frozen price
table the search priced it with.

Everything here runs from a plain clone. What is *not* here is anything a clone cannot
contain: the frozen price exports ship in `prices/`, the corpus ships as a pack, and the
only external dependency — a ggml build — is fetched by a script.

## 1. Build the executor

`ggml-ir-runner` is a C++ program that allocates synthetic tensors for one declared
policy, runs it on ggml's CPU (and, when configured, CUDA) backend, and prints per-node
and whole-graph times. It links against llama.cpp's ggml at the pinned release.

```bash
python profiling/native/build_ir_runner.py --fetch          # clone llama.cpp b8705 + build ggml
python profiling/native/build_ir_runner.py --llama PATH     # or reuse an existing checkout
python profiling/native/build_ir_runner.py --fetch --no-cuda
```

The script prints the binary path and the two variables the driver reads:

```bash
IR_RUNNER=<...>/build-ir/Release/ggml-ir-runner.exe
IR_DLL_DIR=<...>/llama.cpp-b8705/build/bin/Release     # where ggml-*.dll / .so live
```

The tag is pinned (`b8705`, ggml 0.9.11) because a different ggml is a different
measurement. `--print` shows the commands without running them.

## 2. Unpack the E/S corpus

`corpus_es.json.gz` is the 800 declared policies: the power set over per-operator
placement, each with one copy of every weight, one cache budget and a release rule.
What varies is *where work runs and what is kept*; the work itself is identical in all
800. The pack is 90 KiB because those files are 90% shared structure.

```bash
python profiling/ir_ggml/corpus_es.py --list      # names + count
python profiling/ir_ggml/corpus_es.py --unpack profiling/ir_ggml/cases-es
python profiling/ir_ggml/corpus_es.py --verify    # every case against its recorded hash
```

`instantiate.py` unpacks the corpus on first use, so this step is optional.

## 3. Instantiate on a real geometry

The corpus carries a real schedule topology but a toy substrate (hidden 8, one layer,
byte-scale copies). `instantiate.py` keeps the topology and substitutes the model's real
geometry and the real byte count of every tensor it copies. Nothing here reads a price.

```bash
python profiling/ir_ggml/instantiate.py --model qwen3-0.6b --tokens 64
python profiling/ir_ggml/instantiate.py --model all --tokens 64
```

It writes `cases/<model>/<case>.txt` (the runner's case format) and
`cases/<model>/manifest.json` (which case came from which policy, at which shapes).
`--cpu-mask` sets the placement the case header declares; the default `2,4` is the
mask the published tables were measured with, and the driver refuses a run whose echo
differs from it. On a machine without logical CPUs 2 and 4 the placement does not
exist: re-instantiate with `--cpu-mask` for a mask this machine has, and read every
number as a different instrument — the price tables were measured under `2,4`.

Models: `qwen3-0.6b`, `qwen3-1.7b`, `qwen3-4b` (dense), `granite-3.0-3b-a800m`,
`qwen1.5-moe-a2.7b`, `olmoe-1b-7b` (MoE). Geometry and weights provenance are in
[`../multimodel/models.json`](../multimodel/models.json); the executor needs no weights
at all, it allocates synthetic tensors from the declared geometries and dtypes.

## 4. Run and compare

```bash
python profiling/ir_ggml/run.py --model qwen3-0.6b --tokens 64 --limit 20
python profiling/ir_ggml/run.py --model qwen3-0.6b --analyse-only
```

`run.py` executes each case, then reports two independent comparisons:

- **A1** — per priced action unit, measured time / the frozen table's price.
- **A2** — per case, measured makespan / the makespan the same schedule gets when every
  node carries its table price. Copies carry no table price, so they are *declared*
  unpriced and A2 is a lower bound; the report says so rather than dropping them.

Gates run before either number: replica CV > 5% excludes a case, and the control family
(`dense-compute`) must land in the observation band `[0.85, 1.20]` or the tranche is
declared a different band and its readings are not compared.

Environment: `IR_RUNNER`, `IR_DLL_DIR`, `IR_RESULTS` (default `results/`, gitignored).
The instantiator reads `IR_CASES` and `IR_PRICES` (defaults `cases-es/` and `prices/`).

## 5. The other corpus generator

`corpus_v4.py` generates an **edge-heterogeneous** corpus from its own axes — a block
stack of `D` layers with the first `g` pinned on the accelerator, which the 1-layer,
per-op-bit E/S space cannot express. It needs no input files:

```bash
python profiling/ir_ggml/corpus_v4.py --list
python profiling/ir_ggml/corpus_v4.py                  # -> cases-v4/, cases-v4-moe/
python profiling/ir_ggml/instantiate.py --cases cases-v4 --out cases-v4-out
```

## What is not reproducible from this repository

- **The price tables.** `prices/<model>/costs.json` are frozen exports from the
  measurement campaigns (`20261104-multimodel-*`). They are inputs to this apparatus,
  not outputs of it: they let a fresh clone check that the search's prices match this
  machine, but they were measured here, not there. On another device, A1 will read the
  device-to-device ratio, not 1.0 — that ratio *is* the portable number.
- **The measurement state.** The same binary and command can sit on two observation
  bands a day apart (`composing.variability`, `python -m composing table-state`). A
  single point is never comparable across sessions.
- **The corpus's generator.** `corpus_es.json.gz` was produced by
  `exact_fairness.stage1.py`, which prices policies with a MILP and depends on the
  `experiments/ir_value` tree. That tree is kept with the paper. `corpus_es.py --pack`
  rebuilds the pack from such an output; the pack itself is self-contained.
- **The llama.cpp C arm.** Comparing the same policy against real inference on GGUF
  weights needs a full llama.cpp build with `llama-bench`, plus 0.4--2.5 GiB per model.
  Weights are fetched by `python profiling/multimodel/fetch_models.py` (SHA-256 in
  `models.json`); the C arm itself is campaign code, not part of this package.
