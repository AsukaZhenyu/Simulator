# Reproducing the library and its tests

This is the shortest path from a fresh clone to a green test run on a machine
that has never seen this project, and the checklist to follow when the same
clone is run on a second machine.

Every suite runs **without a GPU, without CUDA, without model weights and
without network access after installation**.

## Requirements

| Item | Value |
|---|---|
| Python | 3.10 – 3.13 (CI runs 3.10, 3.12, 3.13) |
| OS | Linux, macOS or Windows; CI runs Ubuntu and Windows |
| Third-party runtime dependencies | none for the simulator. `numpy`, `scipy` and `jinja2` come with the `profiling` extra, which only the measurement suites need; the `gguf` extra adds `gguf` and `numpy` for the GGUF import path |
| GPU, CUDA, drivers | not needed by any test suite |
| Network | only for `git clone` and `pip install` |

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
checkout without any install; the `llm-sim` command and the import check below
need it. `[profiling]` adds `numpy`, `scipy` and `jinja2` for the measurement
suites; it installs neither CUDA nor any measurement tool.

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
[.github/workflows/ci.yml](../.github/workflows/ci.yml) runs. A full pass ends
with:

```
34/34 suites passed in ...
```

Measured on a Windows 11 laptop with Python 3.12: **about 10 minutes** in total
for 3,928 tests (9 min 51 s in the working tree, 10 min 6 s from a `git archive`
export of the same commit). `mapping` dominates at about 278 s (1,206 tests),
then `profiling` at about 50 s (844 tests), `accuracy-suite` at 54 s,
`expert-recipes` at 39 s, `modeling` at 37 s and `public-library` at 32 s; the
other 28 suites finish under 15 s each. The heavy suites are combinatorial
search and simulation rather than floating point, so a slower CPU stretches them
roughly proportionally.

A fresh clone with only the `profiling` extra installed reports **25 skips**, and
none of them is a failure — each is a check whose optional prerequisite is
absent:

| Skips | Suite | Prerequisite |
|---|---|---|
| 4 | `mapping` | the compiled `ggml-export-workload`; those tests say explicitly that the export side of the GGML chain was **not** regenerated in this run. Build it per `mapping/ggml/README.md`, or point `TENSOR_MAPPING_GGML_EXPORTER` at a binary |
| 11 | `state-response` | `LLM_THREAD_HELPER_TEST_BINARY` — a separately built Windows thread helper |
| 2 | `profiling` | `HIGHLEVEL_PROBE`, the timed high-level probe binary, and the pinned external GGML checkout used by the F32 runtime comparison |
| 1 | `workload-accuracy` | previously collected native evidence, which is not redistributed |
| 5 | `modeling` | the optional `gguf` extra: `python -m pip install -e ".[gguf]"` |
| 2 | `modeling` | the Nsight artifact used by the nsys analysis tests |

The count depends on the machine, not on the commit: a developer tree that still
has the built exporter, the GGML checkout or the Nsight artifact reports fewer,
and installing the `gguf` extra removes five. No skip changes a verdict — but a
skip is not a pass: it marks a step the run did not perform.

| Command | Effect |
|---|---|
| `python tools/run_tests.py --list` | print the suites with their directories |
| `python tools/run_tests.py -k mapping` | run only suites whose name contains `mapping` |
| `python tools/run_tests.py --keep-going -q` | do not stop at the first failure; one line per suite |
| `python tools/run_tests.py --report report.md` | also write a Markdown report of the run |

### The report

`--report PATH` writes one Markdown file describing the run, so a failing run on
another machine can be diagnosed without access to that machine:

- the command, the time and the overall verdict;
- the repository remote, the commit and whether the tree was dirty;
- the host: OS, CPU, logical cores, GPU (from `nvidia-smi`, when present), Python
  and the executable path, and the versions of `numpy`, `PyYAML` and `gguf`;
- one row per suite: directory, result, wall time, tests run, tests skipped;
- for every suite that did not pass, its exact command, exit code and complete
  output in a fenced block;
- a reproduction block with the clone URL and commit.

The file is self-contained, so send it as-is rather than summarizing it.

### One suite at a time

Each suite is a plain `unittest` discovery target, so any runner works:

```bash
python -m unittest discover -s tests -p test_library.py -v
python -m unittest discover -s tests -p test_strategy_ir.py -v
python -m unittest discover -s modeling/tests -v
python -m unittest discover -s mapping/tests -t mapping -v
```

`tools/run_tests.py` puts the four source directories (`src`, `mapping`,
`modeling`, `profiling`) on `PYTHONPATH` for every suite, so it behaves the same
whether or not you ran the editable install. When you invoke `unittest`
yourself, either install first (then the commands above work as written) or
carry the same paths by hand:

```bash
PYTHONPATH=src:mapping:modeling:profiling python -m unittest discover -s modeling/tests -v
```

On Windows the separator is `;`.

One further suite is deliberately **not** in the runner:
`tests/test_distribution.py` builds a wheel and an sdist and then runs the demo
in a fresh virtual environment outside the source tree. It needs `setuptools`,
`wheel` and, on a machine without a package cache, network access:

```bash
python -m pip install "setuptools>=68" wheel
python -m unittest discover -s tests -p test_distribution.py -v
```

Run it after the runner when you want to check that packaging still works, not
before: it is slower, and its failure means the distribution is broken, not the
simulator.

## What each suite covers

| Group | Suites | What a failure means |
|---|---|---|
| Public API | `public-library`, `analytical-inputs` | the installed facade or the generated experiment inputs changed behaviour |
| Strategy expression | `strategy-ir`, `request-flow`, `shared-memory-facade` | the frontend, the IR, or the single-ledger CPU/GPU memory model diverged |
| Strategy semantics | `strategy-fidelity`, `policy-no-progress`, `dense-subbatches`, `expert-recipes` | a compiled strategy diverged from an independent reference, a declared policy became a dead end, or a batch/expert recipe changed |
| Modeling | `modeling` | graph expansion, the tensor runtime, capacity accounting or cost binding changed |
| Mapping | `mapping` | the policy frontend, IR, compiler or a causal policy changed |
| Measurement contracts | `profiling` and the `profiling/*` suites, plus the five `native-*` suites | a measurement contract, a frozen parameter set or a provenance guard changed |

The `native-*` suites read the probe sources and their provenance records; they
do **not** compile CUDA code and do not need a GPU. Re-running a measurement is
a separate, explicit step — see [profiling/README.md](../profiling/README.md).

None of these suites measures hardware. A green run means the simulator is
internally consistent with the references those suites encode; it says nothing
about how close a prediction is to a real machine.

## Run the simulator

```bash
llm-sim demo --model dense --out dense-demo
llm-sim demo --model moe --out moe-demo
```

Both demos declare fictional hardware and an explicit analytical cost model.
They are examples, not measurements. Write to a **new or empty** directory.

Library-level examples that run after installation live in
[examples/library](../examples/library/README.md).

## Running on a second machine

Use this order; each step is cheap and fails loudly.

1. `python -V` — confirm 3.10 or newer.
2. Fresh venv, `python -m pip install -e ".[profiling]"`.
3. `llm-sim --version` and the four-module import check above.
4. `python tools/run_tests.py -k public-library` — the fastest end-to-end signal.
5. `python tools/run_tests.py --keep-going --report report.md` — the full
   regression set, plus the Markdown report to send back.
6. `llm-sim demo --model dense --out dense-demo`.

If step 5 fails, keep `report.md` and the console log. The report already
carries the environment, the commit and each failing suite's full output, which
is everything needed to work out whether the cause is the machine or the code.

What a second machine should see: `34/34 suites passed`, roughly ten minutes,
and 25 skips if nothing is built there (the table above). A different skip count
is worth one line in the report — it says which optional prerequisite that
machine happens to have — but it is not a failure.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `No module named 'llm_simulator'` | the editable install did not run in the active interpreter; reinstall with the venv's `python -m pip` |
| `No module named 'tensor_mapping'` (or `llm_infer_model`) | running from outside the clone without installing; install editable, or set `PYTHONPATH` to `src;mapping;modeling;profiling` (POSIX: `:`) |
| `No module named 'scipy'` or `'jinja2'` | the environment was installed without the extra; `python -m pip install -e ".[profiling]"` |
| a suite fails only under a bare `python` | the tests must run with the interpreter that has the packages; use the venv, or use `tools/run_tests.py` |
| demo writes into an existing directory | use new or empty directories; copied input snapshots are audit evidence, not checkpoints |
| line-ending churn in `git status` | the repository pins `text=auto eol=lf` in `.gitattributes`; check with `git ls-files --eol`, do not "fix" it by re-saving files |
| `--report` path is surprising | a relative path resolves against the repository root, not the current directory |

## What this repository does not contain

The apparatus is here; the **results** are not. Retained raw measurements
(`profiling/**/results/`), the experiment drivers under `experiments/`, and the
study documents stay with the paper. A test run here reproduces the simulator's
behaviour and the measurement *contracts*, not a hardware measurement: re-running
a campaign means building the probes and running them on the target machine,
which is an explicit step described in
[profiling/README.md](../profiling/README.md).
