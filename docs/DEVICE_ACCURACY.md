# Measuring prediction accuracy on a new device

Section V-A of the paper reports prediction error per machine. The numbers in a
row belong to the machine that produced them: a different CPU, GPU, driver or
memory bandwidth is a different instrument, and the error it produces is the
thing worth reporting. Nothing in a table transfers between devices — only the
procedure does.

This document is that procedure. Run it on one device at a time, with nothing
else measuring on the machine.

## What is measured, and by which campaign

| Column | Campaign | What it compares | Roughly |
|---|---|---|---|
| Dense executor | `profiling.depth_validation` | per-action time of the composed F32 Dense executor against the history-conditioned price, on heldout processes | 108 short processes (36 profile, 72 heldout) |
| Real MoE end to end | `profiling.real_moe_validation` | whole-request latency of OLMoE-1B-7B Q4_K_M against the composed model | one model load, minutes, needs 4.2 GB of weights |
| Matched Dense strategies (V-B) | `profiling.dac_validation` | four placements x six workloads at eight fresh processes each, plus the decision-margin study | 192 heldout and 192 observer processes |

The first two rows are the two columns of Table II; the third regenerates Table V
and the selection claim beside it. All three are driven by
`tools/run_accuracy_campaign.py`.

Both campaigns hold out their own processes: a price is fitted on one set and
scored on another that the fit never saw. Neither reads a previous device's
result, and neither is allowed to reuse this device's heldout measurements.

## Which host can run which column

Every column here compares a measurement against a price, and the measurement
needs an accelerator the ggml build on this host computes on. Which accelerator
that is depends on the machine: CUDA on Windows, Linux and DGX Spark, Metal on a
Mac. Both are real backends here, and the dense probe builds for either from the
same pinned sources.

| On macOS | Status |
|---|---|
| `tools/run_tests.py`, `table-state`, `composed` evaluation, every `--dry-run` | runs; nothing touches a device |
| `python -m profiling.depth_validation.build_probe` | the Metal branch compiles (`PROBE_BACKEND` defaults to this host's backend), against headers in `profiling/native/backend/metal` and no CUDA toolkit |
| the dense column: `prepare`, `profile`, `freeze`, `heldout`, `evaluate` | the V-A row a Mac can produce; it has not been run on Apple hardware from this checkout |
| `python -m profiling.dac_validation.build_probe` | compiles for Metal, but see the strict-F32 note below |
| `profiling.real_moe_validation.linux_build` | refused by name -- the model column's observer is a hook inside `ggml-cuda.cu` |
| `c++ profiling/native/portable/selftest.cpp` (build it, below) | runs, and is the honest answer to "does the port work here" |

What has been exercised is everything except a Mac itself: the Metal compile
path for both probe projects is built by `tools/check-probe-backends.sh` on any
POSIX host (`PROBE_BACKEND=metal`, stopping at the link, since the ggml Metal
libraries are not there), the refusals have their own tests, and the portable
half has the `--self-test` below. A Mac run is what would turn the third row
from "can" into a measured table, and its receipt is the only thing that would.

The probes' `--self-test` runs on a Mac once the probe is built there: it is the
same binary as the measurement, and on Metal it starts.

### How a CUDA-shaped source compiles for Metal

The Executor fragments the two probes compile are hash-pinned: `generate_probe.py`
records their digests and the campaigns refuse a set that hashes to anything
else. They are also written in CUDA's vocabulary -- `cudaEventCreateWithFlags`,
`cudaEventElapsedTime`, `cudaSetDeviceFlags` and the rest -- because the column
they implement is a stream of timed events.

So the port is not an edit to those fragments. It is a toolkit-shaped directory,
`profiling/native/backend/metal/`, whose `include/cuda_runtime_api.h` answers
exactly those names in Metal's terms:

* a host-side clock (`std::chrono::steady_clock`) behind the event API, because
  the fragments only ever create, record and read elapsed time between events;
* `cudaSuccess` / `cudaError_t` / `cudaGetErrorString` / `cudaGetErrorName`, so
  the fragments' error checks keep their shape;
* device and flag calls that a unified-memory Mac has nothing to answer
  (`cudaSetDevice`, `cudaGetDeviceFlags`, `cudaSetDeviceFlags`) as successful
  no-ops -- there is one pool and one device, and a stream priority is a CUDA
  scheduling concept the column never reads back.

Anything that would fake an observation raises `portable_metal::unsupported` at
run time instead. `cudaMemGetInfo`, `cudaPointerGetAttributes` and
`cudaDeviceGetAttribute` return `cudaErrorNotSupported`, `cudaHostRegister` and
`cudaHostUnregister` raise, and `cudaEventElapsedTime` is real. A probe that
asked for device memory on a Mac would be answered "not supported", not with a
free/total pair that describes the host.

`ggml-cuda.h` is answered the same way: `ggml_backend_cuda_init` and
`ggml_backend_cuda_reg` call the Metal ones, and `ggml_backend_cuda_buffer_type`
returns `ggml_backend_dev_buffer_type(ggml_backend_get_device(backend))`.

CMake puts that directory first on the include path when
`-DPROBE_BACKEND=metal` and defaults `CUDA_ROOT` to it, which is why a Mac build
names no toolkit at all:

```bash
python -m profiling.depth_validation.build_probe \
    --build-dir build/depth-probe-mac --llama-root <llama.cpp b8705> \
    --ggml-build-dir <its build dir>
```

and the runtime directories the later `prepare` step names are the probe's own
directory and that ggml build's `lib/` (there is no toolkit and no CUPTI).

`PROBE_BACKEND` is a cache variable of the two probe projects and takes only
`cuda` or `metal`; a third value is refused at configure time with the value
quoted, not ignored.

A Mac cannot be borrowed for checking, so the closest reproducible substitute is
a script that configures and compiles both projects for both backends on a host
that has one of them:

```bash
bash tools/check-probe-backends.sh --llama-root <llama.cpp b8705> \
    --cuda-root <CUDA toolkit>          # omit on a host without CUDA
```

It stops at the link, which is expected: the ggml libraries are empty stubs. A
configure that gets through and a build that produces the probe's object file is
the pass, and an unknown `PROBE_BACKEND` must be refused. What it proves is
narrow and worth stating: no fragment names a CUDA call the compat headers do
not answer, the Metal branch needs no toolkit, CUPTI or CUDART, and the CUDA
branch still compiles unchanged.

Two capabilities stay CUDA-only by their own nature, and both refuse by name
rather than failing obscurely:

* the model column's observer is inserted into `ggml-cuda.cu` (`tap_v4`,
  `tap_v5`, `runtime_windows`), so there is nothing on a Metal host to insert it
  into -- `tools/run_accuracy_campaign.py` refuses that column with
  `this column's observer is the cuda backend` before building anything;
* the e2a column's strict-F32 runtime is a bypass router inside
  `src/ggml-cuda/ggml-cuda.cu` (`GGML_PREC_F32` skips the TF32 MMA path). No
  builder produces the Metal counterpart yet, so a Mac builds the e2a probe and
  still cannot run that column; `tools/device_check` says exactly that.

Windows and Linux run all three columns. The model column has two build routes
(MSVC on Windows, CMake on Linux); the dense and e2a columns have one portable
CMake recipe each, with the Windows, Linux and macOS halves already in their
`CMakeLists.txt`. DGX Spark (Grace + Blackwell, `aarch64`) is a Linux host: the
toolkit paths it needs are the `targets/aarch64-linux/lib` ones, which both
`CMakeLists.txt` already search, and its compute capability is not written down
here at all — `profiling.real_moe_validation.linux_build.compute_capability`
asks `nvidia-smi --query-gpu=compute_cap`, and
`resolved_architectures()` passes the answer as `-DCMAKE_CUDA_ARCHITECTURES` and
records the source of it in the receipt. A hand-written `121` would be one
device's number baked into every build.

### The three machines this paper measures on

Windows laptop, 5090 Ubuntu desktop, Mac mini M4. The Mac is a supported *build*
target and, since 2026-11-13, one of this paper's declared machines; it replaces
the DGX Spark in that role. Only one column can carry a number on it: both
portable probes configure and compile for Metal and the dense column runs end to
end, while the model column has no Metal observer and the e2a column still needs
a strict-F32 Metal route. Those two are out of scope on Apple silicon rather than
pending, so the paper's row carries dashes for them. Nothing is claimed of an
Apple platform beyond what a dense-column cohort would measure; the manuscript's
shared-pool branch says in its own words that it is "an idealized architectural
counterfactual, not a measured Apple or phone platform". The Metal paragraphs
below are the record of that refusal, not a to-do list.

That leaves one question per machine, and it is the same one on all three: does
the procedure that produced the published numbers reproduce on this box? Each
column answers it from its own cohort — `profile → freeze → heldout → evaluate`
inside the column — so a device does not need the laptop's table to run one; it
needs the probes built for its own backend, and, for e2a, a strict-F32 runtime
built on the host.

### The six machines, and what each one is for

One command answers this on any box, and it is the first thing to run on a device
that has not been used before:

```bash
python -m tools.device_check          # add --json for the receipt
```

It names the ggml backend this host builds for, whether the accelerator has its
own memory, which libraries a probe here will load, and a verdict per column with
the first missing piece spelled out. The reason it exists is that every one of
those differences used to surface as a linker error or, worse, a table that
looked plausible.

| Machine | Backend | Accelerator memory | What it is for |
|---|---|---|---|
| Windows laptop | CUDA | separate | all three columns; the published table's origin |
| Windows desktop | CUDA | separate | all three columns; a second GPU, a separate table |
| Linux laptop | CUDA | separate | all three columns; the same recipe as Windows |
| 5090 server | CUDA (`sm_120`) | separate | all three columns; Blackwell arch and bandwidth |
| Mac mini M4 (Apple silicon) | Metal | **shared** | one of the paper's three machines: the dense column only; the model and e2a columns have no Metal route |
| DGX Spark (GB10) | CUDA (`sm_121`) | **shared**, declared | a supported target; not a paper machine since 2026-11-13 |

The last two rows are the ones the code had to change for, and they are not the
same problem.

A Mac has no CUDA, so its two portable probes are built for Metal instead
(`PROBE_BACKEND=metal`, above). The dense column then runs there end to end, and
that is the V-A row a Mac produces; the model column does not, because its
observer is a hook compiled into the CUDA backend, and the e2a column does not
until its strict-F32 runtime has a Metal route. Both refusals name the missing
capability rather than a missing file.

What that Metal route would have to be is now written down rather than left as
"no counterpart". An F32 matmul on Metal is *not* already exact: for any shape
with more than 8 columns -- a prefill -- `ggml_metal_op_mul_mat` dispatches
`kernel_mul_mm_f32_f32`, which is instantiated with `half` operand tiles and
`simdgroup_half8x8` fragments, so both multiplicands are rounded to fp16 before
the multiply (the accumulation is `simdgroup_float8x8`). A one-token decode
takes `kernel_mul_mv_f32_f32`, `<float, float>`, and is exact. So the e2a
column's requirement on a Mac is a **new fp32-fragment matmul kernel plus a
dispatch switch** in `ggml-metal-ops.cpp`, not the one-line router patch the two
CUDA routes apply, and `build_f32_runtime_linux.py` refuses a Mac by name instead
of configuring a CUDA build and failing inside CMake where it reads as "install
CUDA". This is read out of the pinned b8705 tree (source lines, not a
measurement): **no Mac has run either route here**, and the kernel names are the
evidence a Mac owner would start from rather than a claim that the route works.

DGX Spark has CUDA on a unified pool, and that is the case the simulator already
models: `memory_mode: "shared"` replaces a transfer between DRAM and VRAM with
`SHARE`, an alias that occupies no link and transfers no bytes. Both directions are
enforced by the strategy compiler rather than assumed by whoever wrote the strategy:
on a shared pool a physical `COPY` on a route that only grants shared access is
refused (`test_shared_only_route_cannot_compile_into_physical_copy`), and on a machine
with two pools a `SHARE` is refused because there is no pool to alias inside
(`test_share_cannot_compile_where_there_is_no_shared_pool`). A table built for a
discrete machine is *not* the right table there, and the receipt for one is not
comparable with the other's. The distinction is also why the depth column's
`preblock_state` records `device_memory` as `not_queried` on a unified host: a
free/total pair from the accelerator would be the host's own pool reported twice.

What `profiling.host.device_memory()` answers for `cuda` on `aarch64` is **`unknown`**,
not `shared`, and the difference is deliberate. A Mac's accelerator is the SoC's own
GPU, so `metal` decides `shared` by itself. aarch64 does not: a Grace-Blackwell
*host* can carry a discrete card on PCIe (GB300 in a DGX Station), and a wrong
`shared` there prices every transfer at zero, which is the one error a cost model
cannot absorb. So the architecture refuses to answer, a caller that would price a
copy from it must refuse rather than guess, and the owner of a machine whose topology
they have checked states it once with `IR_DEVICE_MEMORY=shared` (or `separate`); from
then on it enters the receipt as a declared input, which is what it is. `shared` is
what a GB10 gets *after* that declaration or an observation, never from the arch.

No DGX Spark has run these columns here yet, and the table says what the code
will do there rather than what a GB10 measured. What *is* pinned on this side is
the part that decides it: `device_memory()` answers `unknown` for `cuda` on
`aarch64` unless `IR_DEVICE_MEMORY` says otherwise (a test with no device behind
it, `test_an_aarch64_cuda_device_is_not_decided_by_its_architecture`), the aarch64
toolkit and CUPTI directories are searched by both `CMakeLists.txt`
(`cupti_library_dir`'s test covers the layout), and the compute capability is
read from `nvidia-smi` at build time and recorded in the receipt rather than
being hard-coded from this device's `sm_89`. None of those has been exercised
on a GB10; the first run there is what turns that row into a measured one, and its
receipt is the only thing that would.

Two instruments are deliberately *not* on this list: `profiling/native/physical_stages`
(and the `available_service` campaign that drives it) and the `tap_campaign*`
diagnostics. They record the Windows QPC clock by name, their GO handshake and
audits compare against that name, and their Executor snapshot is a capture of the
machine that produced the published physical-stage evidence. They are not among
the three V-A columns, and their absence on Linux is scope, not a broken port.

One input of the e2a column is likewise Windows-only: its strict-F32 runtime
(`profiling/native/highlevel_accuracy/build-runtime-f32-v2`) ships as a set of
`.dll` files, and the builder that produced it (`build_f32_runtime.py`) drives
MSVC's `cmd.exe`/`cl.exe` and relinks the tree's Release objects. Running the e2a
column on a POSIX host therefore needs that runtime built there first:

```bash
python -m profiling.native.highlevel_accuracy.build_f32_runtime_linux \
    --source <llama.cpp b8705 tree> --output build/strict-f32-linux \
    --cuda-root <CUDA toolkit>
# then name the directory it wrote, which the dry run also tells you about:
bash run-accuracy-campaign.sh --e2a-runtime build/strict-f32-linux/build/bin
```

The patch is the same three lines (`GGML_PREC_F32` bypasses the TF32 MMA path) on
the same pinned bytes, and both builders verify the router's SHA-256 before they
touch anything. What differs is the scope each manifest claims: the Windows route
preserves every object it was given and rebuilds one translation unit, while this
route compiles the whole tree, so its manifest says `complete CMake build` and
records the compiler, toolkit, architecture and every library hash instead of
claiming the binary is the one that was measured. The dense and model columns
have no such dependency: their probes link ggml themselves.

The e2a column's report carries one sentence about the machine the numbers came
from. It is written from the campaign's own `native-reproducibility.json`, so a
run on this machine still produces the frozen sentence word for word, and a run
on another device gets a sentence about *its* build -- compiler and version as
the build manifest records them -- with the flags, CUDA SDK and compute
architecture left unclaimed, because that manifest does not record them.

## First hour on a rented machine

The checkout carries nothing that belongs to a device, so a fresh clone plus the
toolkit is the whole setup. In this order, because each step tells you whether
the next one is worth starting:

```bash
# 1. the two packages, from one command -- a second install of either breaks
#    the editable paths the profiling code imports through -- plus the
#    measurement extras (`pip install -e ".[profiling]"` also works, but it
#    pulls the root package in with them)
python3 -m venv modeling/.venv
modeling/.venv/bin/python -m pip install -e ./modeling -e ./mapping \
    numpy scipy jinja2 psutil gguf
#    If that sits for minutes without downloading, the index is the problem, not
#    pip: `--index-url https://pypi.tuna.tsinghua.edu.cn/simple` turns a
#    quarter-hour of index pages into a few seconds of wheels.

# 2. the port itself: no CUDA, no model, no ggml
c++ -std=c++17 -O2 -o /tmp/portable-selftest profiling/native/portable/selftest.cpp
/tmp/portable-selftest                     # wants "0 failure(s)"

# 3. the software side of every column, which needs no device at all
modeling/.venv/bin/python tools/run_tests.py --keep-going

# 4. what this host can actually measure (writes a preflight, measures nothing)
bash run-accuracy-campaign.sh --dry-run
#    On a unified-pool CUDA host (DGX Spark) the topology is declared, not inferred,
#    and the dry run above will not do it for you:
#      IR_DEVICE_MEMORY=shared python -m tools.device_check   # memory row must say shared
```

Step 4 is the honest gate: it prints the columns this device can run, the
toolkit it found, the binaries it is missing and the resident processes it saw.
A missing artifact is *reported* there, not raised -- a device that has not built
its probes yet is the normal case on a first run, and the list of what is missing
is the answer the dry run exists to give.

Two artifacts can also be present and still unusable, and the preflight refuses
those by name instead of starting a probe that would fail at its first `dlopen`:

* a **Windows build on a POSIX host** -- the common case when a shared folder
  (a VM mount, a network drive) exposes the tree that produced the published
  table to a Linux machine. It holds `*.dll` and this host loads `*.so*`, so the
  refusal says "artifacts belong to another platform" and lists the directory;
* a **Windows image under any name**, recognised by its first two bytes (`MZ`)
  rather than by its suffix, because a copied probe is usually renamed;
* a **half-built runtime** -- a directory that holds this platform's libraries
  but not the three the probe loads by name (`ggml-base`, `ggml-cpu`,
  `ggml-cuda`). A strict-F32 build that stopped part way leaves
  `libggml-base.so` and nothing else; the preflight refuses it, and so do both
  columns' `prepare` steps, before a cohort is built on top of it.

Neither is a verdict on the machine: both mean "build this here first", which is
the next section.

A real run refuses instead of reporting. On a Mac the refusal is per column and
names the capability, not the file: `--columns dense` starts, `--columns model`
stops with `REFUSED, this column's observer is the cuda backend`, and a Mac that
happens to have copies of the CUDA probes reads that refusal instead of
`preflight ok`, because having the files is not the same as being able to run
them.

## Before anything runs

1. **One campaign at a time.** Every campaign takes a machine-wide lock
   (`profiling.state_calibration.run.measurement_lock`) and the numbers assume it
   is the only load. Do not compile during a timing phase: a compile changes the
   observation band, and a band is not a drift you can correct afterwards.
2. **This is a laptop-class rule, not a formality.** `python -m composing
   table-state` reports the band a table belongs to (5 % tolerance). If a
   campaign's control family lands outside its band, the run says so; that is a
   different machine state, not an error to average away.
3. **Record the device.** Both manifests carry a hardware snapshot
   (`hardware_identity()`: OS, logical processors, NVML identity, core topology).
   Send the whole manifest, not a summary.

## One command for all columns

`run-accuracy-campaign.cmd` (at the repository root) runs the columns below in
order -- model, dense, then the E2A/V-B regeneration -- and stops at the first
phase that fails. Double-click it, or run it from a plain console:

    run-accuracy-campaign.cmd

On Linux or macOS use the shell wrapper of the same name, which calls the same
campaign and differs only in how it finds the interpreter:

    bash run-accuracy-campaign.sh [phases...]

It looks for `modeling/.venv/bin/python`, then a sibling `.venv/bin/python`, then
a bare `python3` -- and `PYTHON` wins over all three, which is what a rented box
or a WSL install needs, because there the venv is usually outside the checkout:

    PYTHON=$HOME/venv/bin/python bash run-accuracy-campaign.sh --dry-run

The chosen interpreter is printed, and one that cannot `import numpy` is refused
by name before anything starts: otherwise every column refuses for a reason that
names the probes rather than the interpreter, and the reader chases the wrong
file.

Sometimes the interpreter is the whole of the missing setup, and the fix is a
venv rather than an argument. A minimal Ubuntu or WSL image ships `python3`
without `pip` or `ensurepip`, so `python3 -m venv` fails and `sudo` may want a
password the campaign cannot answer; that is not a repository problem, and the
wrapper's `cannot import numpy` is the honest symptom. Once:

```bash
python3 get-pip.py --user --break-system-packages      # PEP 668 image, no pip
python3 -m pip install --user virtualenv
python3 -m virtualenv "$HOME/llm-infer-venv"
"$HOME/llm-infer-venv/bin/pip" install -e ./modeling -e ./mapping   # one command
PYTHON="$HOME/llm-infer-venv/bin/python" bash run-accuracy-campaign.sh --dry-run
```

Both packages go in one `pip install`, because `mapping` depends on
`llm-infer-model` and that package is on no index: named on its own, `-e
./mapping` stops at `Could not find a version that satisfies the requirement
llm-infer-model`, which reads like a network problem and is not one.

It is meant to be started on a machine with nothing else open: close the editor,
the browser and any assistant first. The wrapper waits ten seconds so that the
last of them can exit, then the script takes over. Nothing it does is retried and
nothing is averaged over a failure; a failed phase leaves its output on disk and
the run stops, so a finished run means every phase in `campaign.log` returned
zero.

Before each column it refuses to start unless all of these hold, and the refusal
is the point of having a script at all:

* `profiling/.measurement.lock` is free -- no other campaign is running;
* no process that *times* something is resident (a probe, `llama-server`, or one
  of this repository's `profiling.*` campaign modules). A probe that outlives its
  shell keeps the machine in a measured state after its parent is gone;
* the artifacts that column needs exist (weights, build receipt, CUDA runtime);
* free physical memory covers the column: about 6 GiB for the model column, 2 GiB
  for either native column. The five heaviest resident processes are written to
  `campaign.log` either way, so a run that is quoted later says what else was
  there.

Output goes to `build/accuracy-<date>/`: `campaign.log`, one directory per column,
`CAMPAIGN_SUMMARY.md` and `campaign-records.json` with every phase, its runtime
and its return code. Cohort directories are never reused -- a second run moves an
earlier one aside as `*-aborted-<stamp>` rather than inheriting its frozen prices.

Useful subsets:

    run-accuracy-campaign.cmd --dry-run                 # preflight only, measures nothing
                                                        # (writes build/accuracy-<date>-dryrun,
                                                        #  never a real cohort's log)
    run-accuracy-campaign.cmd --columns model           # Table II "Model" cell
    run-accuracy-campaign.cmd --columns dense,e2a       # Table II "Dense" cell and V-B

`--root <dir>` writes the columns somewhere other than `build/accuracy-<date>`.
Use it when an earlier directory holds something the new run must keep -- for
example a dense cohort used as a `--dense-cache`: the cache has to stay at the
absolute path recorded inside its own programs, and a run into the same root
would move it aside.

Six inputs belong to the machine rather than to the checkout, and every one has
a flag: `--cuda-dir` (the toolkit root; without it the newest toolkit under this
host's own install roots is used -- `C:/Program Files/NVIDIA GPU Computing
Toolkit/CUDA` on Windows, `/usr/local/cuda*` elsewhere), `--depth-binary`
and `--e2a-binary` (probes built on this device, §1 below), `--e2a-runtime` (the
strict-F32 runtime; the default is this device's `.dll` build, so a POSIX host
names the one it built), `--model` (the GGUF) and `--runtime-build` (the
runtime-window build tree). The shipped defaults name *this* device's build
products, so a second machine passes its own -- a new machine that does not fails
the preflight with the path it wanted, before anything runs, and a build that
exists but belongs to another platform is refused as such rather than started.
On a host with no toolkit at all the refusal names the first directory it looked
in rather than producing an empty path.

A toolkit is not a device, and the two refusals have different fixes. The toolkit
is what a *build* needs -- `nvcc` and CUPTI on the link line -- while a probe
computes on a device the driver reports, and a CI image or a container can hold
the first without the second. So `profiling.host.backend()` asks `nvidia-smi -L`
before it answers `cuda`, and a driver that answers with no GPU turns a toolkit
into `None`. `python -m tools.device_check` then prints `backend none (CPU
only)` followed by the sentence saying which half is missing (`a CUDA toolkit is
installed but the driver reports no GPU`). A driver that cannot be asked -- absent
from `PATH`, a nonzero exit, a timeout -- is *unknown*, not absent: only a driver
that answers with an empty list is evidence, because refusing a working GPU box
over its `PATH` would turn a diagnostic into a broken checkout.

`--heldout-index` picks the held-out request stratum. Index 0 is the published
cohort's and is the default; a device that has already run the published
campaign passes `1`. The run logs which index it used, because the two are
different requests and the difference belongs in the row's provenance.

The summary's headline lines are read from each column's own output file, never
recomputed: the Model cell from `CONDITION_MEAN_EVALUATION.json`, the Dense cell
from `summary.json`, the selection cell from the E2A `summary.json` (both how
many workloads picked the real winner and how many candidates qualified), and
the margin sweep from `decision-margin.json`.

## Checking the procedure without running it

`tools/run_tests.py` is the single test entry point, and it now covers the
software side of V-A and V-B: `real-moe-validation` (173 tests),
`dac-validation` (39) and `depth-validation` (17). None of them starts a probe or
touches hardware, so they can be run while anything else is open.

Two things that entry point does not buy:

* The Model column's tests compare a generated source against the frozen tap
  source of the 2026-10-27 build tree. That tree is a build product, archived
  outside the checkout, so three of the 173 skip on a fresh clone instead of
  failing -- a missing fixture is not a failing transform. A fourth
  (`test_nvcc_is_asked_through_the_toolkit_the_build_named`) skips on Windows
  and runs on a POSIX host: it is about the Linux build route.
* Eleven more skip until `LLAMA_B8705_ROOT` names a llama.cpp `b8705` checkout
  (`export LLAMA_B8705_ROOT=/path/to/llama.cpp-b8705`): nine in
  `real-moe-validation`, two in `dac-validation` (the strict-F32 receipt, whose
  route is the POSIX one). They are the tests that read the pinned tree's own
  sources and rebuild from it; the skip message says so by name, because the
  default is the path on the machine that produced the published build and no
  other host has it.
* The Dense column's `freeze` is covered only by the fit-consistency report's
  tests. `prepare_portable`, `programs` and the `profile`/`heldout`/`evaluate`
  phases have no unit tests: they are exercised by a real run, which is what
  `run-accuracy-campaign.cmd` does.

## Dense executor column

### 1. Build this device's probe

The probe links a pinned llama.cpp/ggml build, and on a CUDA host its runtime and
CUPTI. The pinned revision is llama.cpp `b8705` (`profiling/native/build_ir_runner.py
--fetch` can clone it; any matching build works if you pass its paths).

```bash
python -m profiling.depth_validation.build_probe \
    --build-dir build/depth-probe-<device> \
    --llama-root <llama.cpp b8705 checkout> \
    --ggml-build-dir <that checkout>/build \
    --cuda-root <CUDA toolkit> \
    --cupti-root <CUDA toolkit>/extras/CUPTI
```

`--cuda-root` and `--cupti-root` are required only when this host's backend is
CUDA; the driver asks `profiling.host.backend()` and refuses with the missing
flag's name if they are omitted there. On a Mac the same command without them
configures `-DPROBE_BACKEND=metal` against `profiling/native/backend/metal`,
which is where the fragments' CUDA event vocabulary comes from (see "How a
CUDA-shaped source compiles for Metal" above). `PROBE_BACKEND` is a cache
variable of the CMake project, so passing `-DPROBE_BACKEND` by hand to `cmake`
is the same choice, and the driver always passes its own.

It refuses to overwrite an existing build, verifies the five generated Executor
fragments against their recorded hashes, and writes `build-manifest.json` next to
the binary; the manifest records the backend and the memory mode, so two builds
of the same sources for two backends are distinguishable afterwards. On Linux
(and on a DGX Spark, which is a Linux host) the same command
builds with Ninja or Unix Makefiles: the binary lands at `build/depth-probe-<device>/depth-probe`
with no `Release/` directory, the shared libraries are `libggml-*.so` in the ggml
build's `bin/`, and `--cupti-root` is `<CUDA>/extras/CUPTI` (the CMakeLists
searches `targets/aarch64-linux/lib` as well, which is where an aarch64 toolkit
keeps `libcupti`). Compiling the fragments is not incidental:

```bash
python -m profiling.depth_validation.generate_sources --check
```

The depth probe is compiled against the **historical** highlevel executor, not
today's. Only that revision carries the binary initial-asset transport the depth
programs are initialised through, and it is the revision whose arithmetic the
published table used. `generate_sources.py --out` regenerates all five fragments
byte for byte from the pinned sources, and the build refuses a set that hashes to
anything else.

**The ggml build the probe links is a separate step**, and on Linux it has one
trap of its own: CMake has to find `nvcc`, and it looks on `PATH`, not under
`--cuda-root` (that argument is this repository's, not CMake's). A configure run
without the toolkit's `bin/` on `PATH` stops at

    CMake Error at ggml/src/ggml-cuda/CMakeLists.txt:58 (enable_language):
      No CMAKE_CUDA_COMPILER could be found.

which names neither `nvcc` nor the toolkit, so the fix is easy to miss:

```bash
export PATH=<CUDA toolkit>/bin:$PATH        # so CMake finds nvcc
export CUDACXX=<CUDA toolkit>/bin/nvcc      # optional, but removes the search
cmake -S <llama.cpp b8705 checkout> -B <ggml build dir> \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=<this device's compute capability> \
    -DCUDAToolkit_ROOT=<CUDA toolkit>
cmake --build <ggml build dir> --target ggml ggml-base ggml-cpu ggml-cuda -j <cores>
```

The four libraries land in `<ggml build dir>/bin/` as `libggml*.so` on Linux and
`ggml*.dll` on Windows; `--ggml-build-dir` is that directory's parent. Partial
builds are the other common way to lose an hour: `--target ggml-cuda` alone does
not build `ggml-base`, and the probe's `find_library` then fails by name
(`Could not find GGML_base`). Naming all four targets is what makes the snippet
above complete. A toolkit that is not under `/usr/local/cuda` is also where the
loader warning comes from later: `libggml-cuda.so` needs `libcudart.so.12` at
launch, so pass `--cuda-root` to `build_probe` as well (it puts the toolkit's
`targets/<arch>-linux/lib` in the probe's rpath) and keep it in the campaign's
runtime directories.

The same thing as a script, which does the `PATH`/`CUDACXX` export, detects the
compute capability from `nvidia-smi`, and then checks the four libraries are
really in `bin/`:

```bash
sh tools/build-ggml.sh --llama-root <llama.cpp b8705 checkout> \
    --build-dir <ggml build dir> --cuda-root <CUDA toolkit>
```

### 2. Prepare, without timing anything

```bash
python -m profiling.depth_validation.prepare_portable \
    --out build/depth-validation-<device>/formal \
    --binary build/depth-probe-<device>/Release/depth-probe.exe \
    --runtime-dir build/depth-probe-<device>/Release \
    --runtime-dir <CUDA toolkit>/bin \
    --runtime-dir <CUDA toolkit>/extras/CUPTI/lib64
```

The three paths are the Windows layout. On Linux the binary is
`build/depth-probe-<device>/depth-probe` (no `Release/`), its directory is the
first `--runtime-dir`, and CUPTI is wherever `profiling.host.cupti_library_dir`
would look — `extras/CUPTI/lib64` on x86_64, `targets/aarch64-linux/lib` on an
aarch64 toolkit. `python -c "from profiling import host; print(host.cupti_library_dir('<toolkit>'))"`
answers it for the device you are on. On a Mac the binary is
`build/depth-probe-<device>/depth-probe`, its directory and the ggml build's
`lib/` are the only runtime directories (there is no toolkit and no CUPTI to
name), and the column runs in its clock-only modes: the request check refuses
the `cupti` instrumentation on a Metal build at request construction, before a
cohort is built.

`prepare` is software only: it builds 18 program/strategy cohorts (prefill at 4,
8, 16 and 32 layers over 64 tokens; decode at 16 and 32 layers at history 128,
each under three placements), each with an independently reseeded heldout twin,
then declares every process before any of them runs — 36 profile processes (two
repeats) and 72 heldout ones (four repeats). The historical entry point read the
archived H2D manifest for three fields — the protocol constants, the runtime DLLs
and a hardware snapshot; `prepare_portable` supplies them from the arguments
above and this device, and leaves the constants exactly as they were.

The probe *links* its ggml import libraries out of the pinned tree but *loads*
the matching DLLs from that tree's `bin/<config>`, which is a different directory
(`ggml/src/<config>` holds the `.lib` files). A runtime list built only from the
directories on the command line therefore loads nothing and every process exits
with `STATUS_DLL_NOT_FOUND` (`0xC0000135`). `prepare_portable` reads
`GGML_BUILD_DIR` out of the `CMakeCache.txt` above the binary, appends
`bin/Release`, `bin/Debug` or `bin` — whichever exists — to the runtime list, and
records them under `derived_runtime_dirs` in the manifest so the receipt is not
silent about it.

`--cache <earlier cohort>` re-seeds the templates of an earlier preparation —
18 compiled programs, 36 with their heldout twins — instead of composing them
again (about 85 minutes saved on this machine, a few minutes spent re-seeding).
Only `inputs/*/program.json` is read from the cache, and only for the compiled
graph; the numerical initialisers are generated fresh from the new seed into the
new cohort, and no timing, profile or heldout result is read. The cache must stay
at the absolute path recorded inside its own `program.json` files, because
`validate_compact` re-hashes the initialiser asset through that recorded path and
the path itself is inside the program's identity hash. Renaming or moving the
cache breaks it; point `--cache` at it where it was written.

What is frozen at this point: the executable and all bound sources (hashes), the
request matrix, the process order, and the gates.

### 3. Measure

```bash
python -m profiling.depth_validation.campaign profile  --out build/depth-validation-<device>/formal
python -m profiling.depth_validation.campaign freeze   --out build/depth-validation-<device>/formal
python -m profiling.depth_validation.campaign heldout  --out build/depth-validation-<device>/formal
python -m profiling.depth_validation.campaign evaluate --out build/depth-validation-<device>/formal
```

- `profile` runs two repeats of every profile process, each 16 trials of 8
  replays, and refuses a process whose ports differ from the reference or whose
  copy bytes differ.
- `freeze` fits the history-conditioned response table and predicts the heldout
  programs with both providers, then writes a barrier over the frozen files. It
  refuses to run after any heldout process has started. Alongside the table it
  writes `fit-consistency.json`: for every semantic coordinate that more than one
  profile program contributes samples to, the ratio between the largest and
  smallest per-program mean. One table serves every placement, so a coordinate
  measured in two programs is averaged, and that average cannot be right for both
  if the two realized costs differ. A large `max_spread` therefore predicts a
  heldout miss before any target process runs; the CPU-only, all-GPU and hybrid
  placements share copy coordinates whose realized cost depends on what else is
  queued, and on the laptop that produced this document that spread reached
  27.2× (a 2 KiB host-to-device copy: about 21 µs in the all-GPU programs, about
  500 µs in the hybrid decode programs, 26 µs in the hybrid prefill ones), which
  the pooled table could not price.
- `heldout` re-checks the barrier, then runs four repeats per heldout program
  with bare instrumentation.
- `evaluate` scores: **process-mean maximum APE ≤ 10 %** and **trial-average p90
  APE ≤ 20 %** per case, for both `context_static` and `history_conditional`.

`summary.json` reports, per provider, how many cases pass. That count — not one
averaged error — is what a row in the table says.

### 4. What to send back

`build/depth-validation-<device>/formal/` as a whole (manifest, programs,
executions, `model-frozen.json`, `predictions-frozen.json`, `summary.json`), plus
`build-manifest.json` from the probe build. The manifest records the binary hash,
the fragment hashes and every bound source; a re-run that changes any of them is
refused by `check()` rather than silently producing a second instrument's numbers.

## Real MoE end-to-end column

### What a row is

One real model — OLMoE-1B-7B-0924-Instruct Q4_K_M, 16 layers, 64 experts, top-8,
4.2 GB of weights — served by the composed model under one declared policy, in
two placements (all-CPU and hybrid), on six held-out requests each, with two
fresh processes per placement. That is 12 conditions. A **condition** is one
placement and one request, and its error is

    |predicted / mean(measured over that condition's repeats) − 1|

not the mean of the per-launch errors; the two differ whenever the repeats
differ from each other. A condition counts only when every declared repeat
produced an identical token trace and final-logit hash; one whose endpoints
disagree is kept with its reason and excluded from the mean, never averaged in.

The column's number is the mean total error over the complete conditions, with
the worst condition in parentheses.

### 1. Build the instrument

The probe is not one build. It recompiles one translation unit at a time inside
a pinned llama.cpp `b8705` tree that has **already** been configured and built
Release/x64 with CUDA, because it reuses that tree's objects and refuses a build
whose inputs it cannot hash. Prepare that tree first, by whatever route the pinned
revision supports; then, with `LR` its root, `B` a scratch build directory,
`U` the CUDA toolkit and `MSVC` your Visual Studio `cl.exe`/`link.exe`:

```bash
# 1. CPU consumer tap -- recompiles ggml-cpu.c only
python -m profiling.real_moe_validation.tap_v4 prepare --out $B/tap-v4 --llama-root $LR
python -m profiling.real_moe_validation.tap_v4 build   --out $B/tap-v4 --cl $MSVC/cl --link $MSVC/link

# 2. GPU consumer tap -- recompiles ggml-cuda.cu only, on top of stage 1
python -m profiling.real_moe_validation.tap_v5 prepare --out $B/tap-v5 \
    --llama-root $LR --v4-build $B/tap-v4
python -m profiling.real_moe_validation.tap_v5 build   --out $B/tap-v5 \
    --v4-build $B/tap-v4 --nvcc $U/bin/nvcc --cl $MSVC/cl --link $MSVC/link

# 3. numerical driver -- the reference the endpoint checks compare against
python -m profiling.real_moe_validation.numerical_v6 prepare-build --out $B/numerical-v6 \
    --v5-build $B/tap-v5
python -m profiling.real_moe_validation.numerical_v6 build --out $B/numerical-v6 \
    --cl $MSVC/cl --link $MSVC/link

# 4. private runtime fix -- needs the operator's own build authorization
python -m profiling.real_moe_validation.mmid_runtime_fix prepare --out $B/mmid-fix \
    --observer-build $B/tap-v5 --numerical-build $B/numerical-v6
#   then write $B/mmid-fix/ROOT_BUILD_AUTHORIZATION.json yourself:
#   {"build_authorized": true, "prepared_sha256": "<sha256 of $B/mmid-fix/PREPARED.json>"}
python -m profiling.real_moe_validation.mmid_runtime_fix build --out $B/mmid-fix

# 5. the runtime-window probe itself
python -m profiling.real_moe_validation.runtime_windows prepare --out $B/runtime-windows \
    --llama-root $LR --observer-build $B/tap-v5 --fixed-build $B/mmid-fix
python -m profiling.real_moe_validation.runtime_windows build   --out $B/runtime-windows \
    --cl $MSVC/cl --link $MSVC/link
python -m profiling.real_moe_validation.runtime_windows finalize --out $B/runtime-windows \
    --link $MSVC/link
```

Every stage writes `PREPARED.json` before compiling and `BUILD_COMPLETE.json`
after, records each compiler/linker invocation, and refuses an `--out` that
already exists — a retry is a new instrument, not a repair. Stage 4 is gated on a
receipt this repository will not write for you: name the build yourself only
after reading `PREPARED.json`.

The measured executable is `$B/runtime-windows/bin/real-moe-runtime-window-probe.exe`.
**That chain is Windows/MSVC only**: it drives the pinned tree's own `.vcxproj`
compiler contracts and reconstructs each compiler command from `.tlog` files, so
it cannot run elsewhere.

#### On Linux: the same probe, built whole

Linux has no reason to reuse MSVC's incremental machinery, so this route patches
a private copy of the pinned tree and rebuilds it with CMake:

```bash
python -m profiling.real_moe_validation.linux_build prepare \
    --out $B/runtime-linux --llama-root $LR
python -m profiling.real_moe_validation.linux_build build \
    --out $B/runtime-linux --cuda-root $U \
    --cmake-argument -DCMAKE_CUDA_ARCHITECTURES=121     # optional: see below
```

Without `-DCMAKE_CUDA_ARCHITECTURES` the build asks `nvidia-smi` for the attached
GPU's compute capability and compiles for exactly that device, which is what a
single-GPU rented host wants; `BUILD_COMPLETE.json` records the value *and*
whether the caller named it or the host answered. Pass the flag to target
another device, or several (`-DCMAKE_CUDA_ARCHITECTURES=89;121`) for one binary
that runs on both -- a build whose toolkit default list omits the device starts
and then fails every kernel with "no kernel image is available".

It reuses the four anchor-checked source transforms unchanged, and adds only the
host's spelling of five things: the probe's `<windows.h>` becomes
`profiling/native/portable/platform.hpp`, `route_tap_pod.hpp`'s BCrypt digest
becomes the in-tree `sha256.hpp`, the injected hooks drop
`__declspec(dllexport)` (they are `extern "C"` and ggml does not hide symbols),
the receipt's `clock_id` becomes `PORTABLE_CLOCK_ID` -- `windows_qpc` on
Windows, `posix_clock_monotonic` on POSIX, because the shim's
`QueryPerformanceCounter` is CLOCK_MONOTONIC at a declared 1 ns tick there. A row
that said `windows_qpc` on a Linux host would be a false provenance line --
and `common/debug.cpp`, this repository's `ffn_moe_topk` diagnostic, gets
`#include <cstring>` for `strstr`/`memcpy`: MSVC's headers lend it, libstdc++'s
do not, and without the line the tree fails at 93 % with "'strstr' was not
declared in this scope". `ggml-cpu.c` is a C translation unit whose own
`<windows.h>` sits inside `#if defined(_WIN32)`, so it is edited for the export
only and its Windows block is left exactly as the tree has it.

Three things about the private copy and the link line are worth knowing before
the first build on a new host, because each of them fails in a way that reads
like a broken checkout:

* the copy skips `build/` and `build-*/` **directories**, `.git`, and a `models/`
  directory holding no C++ source (GGUF downloads). It keeps
  `cmake/build-info.cmake` and `src/models/*.cpp`: a name-based `build-*` filter
  drops the first (configure: "include could not find requested file") and a
  blanket `models` rule drops the second ("Cannot find source file:
  models/afmoe.cpp").
* `--cuda-root` is put at the front of `PATH` and named as `CUDACXX`, because a
  toolkit installed under `/usr/local/cuda-12.4` without a profile edit is not on
  the search path and CMake then reports "No CMAKE_CUDA_COMPILER could be found".
* the toolkit's `lib64` (or `targets/<arch>-linux/lib`) goes on the probe's
  rpath, because `libggml-cuda.so` needs `libcudart.so.12` and the probe would
  otherwise fail to start *after* a successful build. `LD_LIBRARY_PATH` is
  searched first, so a campaign's own `--runtime-dir` list is still the authority
  at run time.

The probe finds the libraries it observes through the shim, and that lookup is a
little more than a name translation. `GetModuleHandleA("ggml-cpu.dll")` becomes
`dlopen("libggml-cpu.so", RTLD_NOLOAD)` -- the query is "already loaded, never
load one", so the tap binds to the copy the process is executing. A library
opened by absolute path registers no basename for that query unless it carries a
`DT_SONAME`, so on Linux the shim also reads `/proc/self/maps` and re-asks with
the mapped path. It never falls back to *loading* the library: that would
succeed, bind the tap to a second copy, and turn into "the tap was never
registered" without any error.

The result is `$B/runtime-linux/bin/real-moe-runtime-window-probe` with its
libraries beside it, plus `PREPARED.json` and a `BUILD_COMPLETE.json` in the same
schema the Windows chain writes -- the campaign reads that file, not the route.
`build` also proves with `nm -D` that every symbol the probe binds by name is
exported by the library that has to carry it -- `simulator_runtime_window_set` in
`libggml-base.so`, `simulator_moe_tap_set` and
`simulator_runtime_cpu_window_set` in `libggml-cpu.so`, and the three v5 GPU
symbols in `libggml-cuda.so`. A build that produced the libraries but not the
observers would otherwise fail later as "the tap was never registered".
`BUILD_COMPLETE.json` also records the toolkit root, the `nvcc` banner and the
`CMAKE_CUDA_ARCHITECTURES` the CUDA objects were compiled for: a build for
`sm_89` launches no kernel on an `sm_121` device, and "no kernel image is
available" is otherwise the only trace of that mistake.

#### The affinity mask names the CPUs this process may use

The protocol asks every probe to pin itself to *every CPU this host offers*. On
Windows that is the machine's processor count; on Linux it is
`sched_getaffinity(0)`, which a cgroup cpuset, `taskset`, a scheduler
reservation, WSL's CPU limit or `docker --cpuset-cpus=24-47` can narrow --
sometimes to a non-contiguous set. Both the campaign
(`profiling.host.process_affinity_mask`) and the probes
(`GetProcessAffinityMask` in `platform.hpp`) read that set, so the mask one sends
is the mask the other applies. A mask built as "the low N bits" would name cores
`sched_setaffinity` refuses, and the run would stop with "cannot set owned
process mask" on a host that pins perfectly well. The pin for the main thread is
still `4` (CPU 2) wherever that CPU is usable; a container that starts at 24 is
refused by name, because it has no CPU 2 to pin.

The frozen `tap_native/real_moe_tap_probe.cpp` is never edited. The published
cohort's `native_source_sha256` names its bytes, so the platform edits are made
in the derived copy and the derived digest is what the new build records.

### 1b. Prove the port on the device before building the probe

Building any probe needs the toolkit, the ggml libraries and the pinned tree. The
two headers every probe includes first need none of that, and they are where a
silent port failure would live (a digest that is not SHA-256, an affinity mask
that names the wrong cores, a module lookup that opens a second copy of the
library it was supposed to bind). Build and run them first:

```bash
c++ -std=c++17 -O2 -o portable-selftest profiling/native/portable/selftest.cpp
./portable-selftest        # exit status is the number of failed checks
```

It prints one line per check and `0 failure(s)` when the host behaves. On macOS
it reports that the affinity protocol cannot be honoured (`PORTABLE_AFFINITY 0`)
rather than pretending a scheduling hint is an enforced mask, and it expects
`QueryThreadCycleTime` to report unavailability there.

### 2. Fetch and verify the weights

```bash
python -m profiling.multimodel.fetch_models --only olmoe-1b-7b --root <weights dir>
```

`prepare_portable` verifies the file against `profiling/multimodel/models.json`:
byte count first, then a streamed SHA-256. A mismatched or truncated download is
refused, not measured.

### 3. Prepare, without timing anything

```bash
python -m profiling.real_moe_validation.record_tests --out <d>/software-tests.json
python -m profiling.real_moe_validation.prepare_portable \
    --out <d>/formal --build $B/runtime-windows \
    --model <weights dir>/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf \
    --runtime-dir <CUDA toolkit>/bin --labels cpu,hybrid
```

`--build` names whichever route built the probe: `$B/runtime-windows` from the
MSVC chain above, `$B/runtime-linux` from the CMake one. Both write the same
`BUILD_COMPLETE.json`, and that file -- not the route -- is what
`prepare_portable` reads.

Two inputs the published campaign read from its own build tree are supplied
here instead: the numerical campaign's descriptor and the tokenizer's request
template. `prepare_portable` answers those two paths from

* the frozen request suite and capacity policy shipped in
  `profiling/real_moe_validation/published/` (hashes are the ones the published
  cohort froze), and
* this device's model receipt, which changes exactly three template fields
  (`model_path`, `model_sha256`, `model_bytes`) and nothing else.

The two host-memory guards are derived from the capacity policy's own budget
block, not typed in. `--native-guard-bytes` may **raise** the floor on a device
that needs more headroom; it refuses a lower value, because the guard exists to
stop the host thrashing mid-run.

`--labels cpu,hybrid` is not optional: the code's legacy default is three
placements, which would declare 18 conditions where the table has 12.

### 4. Measure

```bash
P=profiling.real_moe_validation
python -m $P.runtime_window_campaign freeze    --out <d>/formal --tests <d>/software-tests.json
python -m $P.authorize --out <d>/formal --user-authorization "<your sentence>" \
    --jobs cpu-calibration-profile,cpu-calibration-bare,cpu-inventory,\
hybrid-calibration-profile,hybrid-calibration-bare,hybrid-inventory
# then collect those six jobs, one at a time
python -m $P.runtime_window_predict calibrate --out <d>/formal
python -m $P.runtime_window_predict forecast  --out <d>/formal --allow-extrapolation
python -m $P.authorize --out <d>/formal --user-authorization "<your sentence>" \
    --jobs cpu-heldout-r0,cpu-heldout-r1,hybrid-heldout-r0,hybrid-heldout-r1
# then collect those four
python -m $P.runtime_window_predict evaluate --out <d>/formal
python -m $P.condition_mean --out <d>/formal
```

- `freeze` refuses to run without a passing software-test receipt that performed
  no hardware execution, and refuses after any heldout process has started.
- `authorize` will not invent a window: it takes the operator's own sentence and
  binds it to the current manifest, freeze barrier and (for the second window)
  the price and prediction freezes, so an edited cohort cannot inherit an
  earlier permission.
- `forecast` freezes prices and predictions **before** the heldout window opens.
- `condition_mean` writes `CONDITION_MEAN_EVALUATION.json`: the mean and worst
  error per metric, with each incomplete condition listed and its reason.

### 5. What to send back

`<d>/formal/` whole — `manifest.json`, `portable-preparation-receipt.json`,
`FREEZE_BARRIER.json`, `PRICE_FREEZE.json`, `PREDICTION_FREEZE.json`,
`INDEPENDENT_EVALUATION.json`, `CONDITION_MEAN_EVALUATION.json`, the request and
execution records — plus `$B/runtime-windows/BUILD_COMPLETE.json` and the
`BUILD_COMPLETE.json` of each upstream stage. `manifest.json` carries the device
snapshot, the binary hash and every bound source; it is the row.

Held-out request index 0 was consumed by the published campaign. A second run on
the same device uses `--heldout-index 1`, which the adapter checks is disjoint
from calibration and from the index-0 requests by both ID and exact prompt-token
sequence — a replication there is a new measurement, not a replay of the table.
