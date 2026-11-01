# Reproducing the library and its tests

This document is the shortest path from a fresh clone to a green test run, on a
machine that has never seen this project. It is also the checklist to follow
when the same clone is run on a second machine.

Everything below runs **without a GPU, without CUDA, without model weights and
without network access after installation**. The measured latency numbers in the
documents under `docs/` and in the papers require the machine that recorded them;
that part is covered in
[Reproducing measured studies](#reproducing-measured-studies) and is *not*
reproducible from this repository alone.

## Requirements

| Item | Value |
|---|---|
| Python | 3.10 – 3.13 (CI runs 3.10, 3.12, 3.13) |
| OS | Linux, macOS or Windows; CI runs Ubuntu and Windows |
| Third-party runtime dependencies | none for the core; `numpy`, `scipy`, `jinja2` through the `profiling` extra |
| GPU, CUDA, drivers | not needed by any test suite |
| Disk | roughly 25 MB of tracked files plus git history; more if you regenerate results |

## Clean setup

```bash
git clone https://github.com/AsukaZhenyu/Simulator.git
cd Simulator
python -m venv .venv
```

Activate the environment, then install once:

```bash
# Windows (PowerShell or cmd)
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

python -m pip install -U pip
python -m pip install -e ".[profiling]"
```

`-e` (editable) is the intended setup. The suite runner also works from a bare
checkout, but the `llm-sim` command and the import check below need the install.
`[profiling]` adds the three wheels a few suites need — `numpy`, `scipy` (the
idle-fit and exact-certification code) and `jinja2` (chat template rendering in
the MoE validation). It does not install CUDA or any measurement tool.

Verify the install:

```bash
llm-sim --version
python -c "import llm_simulator, tensor_mapping, llm_infer_model, llm_infer_profile; print('ok')"
```

## Run every test suite

```bash
python tools/run_tests.py
```

That is the same set of suites the `regression` job in
[.github/workflows/ci.yml](../.github/workflows/ci.yml) runs, in one process
tree. A full pass ends with:

```
36/36 suites passed in ...
```

Useful options:

| Command | Effect |
|---|---|
| `python tools/run_tests.py --list` | print the 36 suites with their directories |
| `python tools/run_tests.py -k mapping` | run only suites whose name contains `mapping` |
| `python tools/run_tests.py --keep-going -q` | do not stop at the first failure; one line per suite |
| `python tools/run_tests.py -k public-library` | fastest end-to-end smoke test (about 30 s) |

Measured on a Windows 11 laptop, Python 3.12, no GPU: **11 minutes** wall clock for
all 36 suites. The `mapping` suite dominates at 278 s (1,206 tests), then
`serving-acceptance` at 108 s, `profiling` at 54 s and `accuracy-suite` at 51 s;
the remaining 32 suites are under 35 s each. The heavy suites are combinatorial
search and simulation rather than floating point, so a slower CPU stretches them
roughly proportionally.

### One suite at a time

Each suite is a plain `unittest` discovery target, so any runner works:

```bash
python -m unittest discover -s tests -p test_library.py -v
python -m unittest discover -s modeling/tests -v
python -m unittest discover -s mapping/tests -t mapping -v
python -m unittest discover -s profiling/tests -v
```

One further suite is deliberately **not** in the runner: `tests/test_distribution.py`
builds a wheel and an sdist and runs the demo in a fresh virtual environment
outside the source tree. It needs `setuptools` and `wheel` and, on a machine
without a package cache, network access:

```bash
python -m pip install "setuptools>=68" wheel
python -m unittest discover -s tests -p test_distribution.py -v
```

Run it after the runner when you want to check that packaging still works, not
before: it is slower and its failure means the distribution is broken, not the
simulator.

`tools/run_tests.py` puts the four source directories (`src`, `mapping`,
`modeling`, `profiling`) on `PYTHONPATH` for every suite, so it behaves the same
whether or not you ran the editable install. When you invoke `unittest`
yourself, either install first (then the commands above work as written) or
carry the same paths by hand:

```bash
PYTHONPATH=src:mapping:modeling:profiling python -m unittest discover -s modeling/tests -v
```

On Windows the separator is `;`.

## What each suite covers

| Group | Suites | What a failure means |
|---|---|---|
| Public API | `public-library`, `analytical-inputs`, `request-flow`, `serving-acceptance` | the installed facade, generated inputs or the request-level flow changed behaviour |
| Strategy semantics | `ir-region-acceptance`, `strategy-fidelity`, `policy-no-progress`, `dense-subbatches` | compiled strategy diverged from an independent reference, or a declared policy became a dead end |
| Design exploration | `hardware-sensitivity`, `performance-boundaries` | analytical sweep or the performance ownership boundary moved |
| Modeling | `modeling` | graph expansion, tensor runtime, capacity or cost binding changed |
| Mapping | `mapping` | policy frontend, IR, compiler or causal policy changed |
| Profiling contracts | the `profiling/*` and `native/*` suites | a measurement contract, frozen parameter set or provenance guard changed |
| Tools | `tools` | the portable calibration report stopped being reproducible |

The native suites read source files and provenance records; they do **not**
compile CUDA code and do not need a GPU.

## Run the simulator

```bash
llm-sim demo --model dense --out dense-demo
llm-sim demo --model moe --out moe-demo
```

Both demos declare fictional hardware and an explicit analytical cost model. They
are examples, not measurements; see
[Capabilities and boundaries](https://github.com/AsukaZhenyu/Simulator#capabilities-and-boundaries).

Library-level examples that run after installation, outside the source tree,
live in [examples/library](../examples/library/README.md).

## Reproducing measured studies

The measured numbers under `docs/` and in the accompanying papers come from a
recording machine and are **not** redistributed with this repository. What ships
here is the code that produced them: campaign drivers under
[experiments](../experiments/) and [profiling](../profiling/), plus the frozen
inputs that are small enough to belong in a repository
(`profiling/data/`, `modeling/tests/data/`).

The retained raw outputs are excluded on purpose:

| Path | Why excluded |
|---|---|
| `profiling/**/results/` | per-job measurement output, gigabytes, machine-specific |
| `modeling/outputs/`, `mapping/outputs/` | simulation output, regenerable from `configs/` and `examples/` |
| `docs/data/host-blackout-diagnosis-*.json` | 2.4 MB diagnostic dump that embeds per-job workstation paths |

To regenerate a study, run its driver on the target machine and keep the output
directory outside version control. Each study's own protocol document states its
inputs, freezing order and acceptance rule; start from
[profiling/README.md](../profiling/README.md) and
[docs/PERFORMANCE_BOUNDARIES.md](PERFORMANCE_BOUNDARIES.md).

Reproduction of a *measurement* is a different claim from reproduction of a
*test run*: the suites here check behaviour and contracts, not hardware accuracy.

## Running on a second machine

Use this order; each step is cheap and fails loudly.

1. `python -V` — confirm 3.10 or newer.
2. Fresh venv, `pip install -e ".[profiling]"`.
3. `llm-sim --version` and the four-module import check above.
4. `python tools/run_tests.py -k public-library` — about 30 s.
5. `python tools/run_tests.py --keep-going -q` — the full regression set.
6. `llm-sim demo --model dense --out dense-demo` — write to a **new** directory.
7. Only then, hardware collection: `nvidia-smi`, `nvcc --version`, and the
   profile the study needs. Paths, node names and link rates you record become
   part of the evidence, so keep the campaign directory intact.

Two machine-specific facts that change results and are worth writing down before
a campaign starts: the GPU driver/CUDA version, and the CPU pinning or thread
configuration. Neither is inferred by the model; both are declared inputs.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `No module named 'llm_simulator'` | the editable install did not run in the active interpreter; reinstall with the venv's `python -m pip` |
| `No module named 'tensor_mapping'` (or `llm_infer_model`) | running from a directory outside the clone without installing; install editable, or set `PYTHONPATH` to `src;mapping;modeling;profiling` (POSIX: `:`) |
| `No module named 'scipy'` or `'jinja2'` | the profile was installed without the extra; `python -m pip install -e ".[profiling]"` |
| a suite fails only under a bare `python` | the tests must run with the interpreter that has the packages; use the venv |
| demo writes into an existing directory | use new or empty directories; copied input snapshots are audit evidence, not checkpoints |
| line-ending churn in `git status` | the repository pins `text=auto eol=lf` in `.gitattributes`; check with `git ls-files --eol`, do not "fix" it by re-saving files |
