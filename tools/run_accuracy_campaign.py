#!/usr/bin/env python
"""Run the paper's measurement columns on this machine, one at a time.

This is the entry point for the numbers in Section V-A (the two accuracy
columns) and the Section V-B regeneration. It exists so that the campaigns can be
launched from a plain terminal with nothing else running -- no editor, no
browser, no assistant -- because every column here is a timing measurement and
every number in it belongs to the machine state that produced it.

    python tools/run_accuracy_campaign.py --columns model,dense,e2a

Columns
-------
  model   Real MoE end to end: OLMoE-1B-7B Q4_K_M against the composed model.
          Needs the 4.2 GB GGUF and about 6 GiB of free memory. This is the
          Table II "Model" cell.
  dense   Dense executor: per-action time of the composed F32 executor against
          the history-conditioned price, held out by process. This is the
          Table II "Dense" cell.
  e2a     Matched Dense strategies: four placements x six workloads, eight
          fresh processes per configuration. This is Table V and the selection
          claim beside it, plus the decision-margin study.

Nothing here is retried and nothing is averaged over a failure: a phase that
fails stops the run and leaves its output on disk for inspection. Cohorts are
written under `--root` and never reused, so a second run cannot inherit a first
run's frozen prices.

Before each column the script refuses to start if the measurement lock is held,
if a process that times something is still resident, or if free memory is below
what that column needs. Close everything else first; that refusal is the point.

`--dry-run` runs those checks without raising: it writes the log a real run
would, names every artifact the columns still need, and stops. On a device that
has just been rented or unpacked that is the first command to run.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:                # `tools/` is on sys.path, the repository is not
    sys.path.insert(0, str(ROOT))
from profiling import host

LOCK = ROOT / "profiling" / ".measurement.lock"


def default_python() -> Path:
    """The interpreter with this repository's packages installed, in either layout."""
    venv = ROOT / "modeling" / ".venv"
    return venv / ("Scripts/python.exe" if host.is_windows() else "bin/python")


def default_executable(directory: Path, stem: str) -> Path:
    """Where this platform's CMake generator leaves a probe.

    Visual Studio is multi-config and writes `Release/depth-probe.exe`; Ninja and
    Unix Makefiles are single-config and write `depth-probe` at the build root.
    """
    name = host.executable_name(stem)
    return directory / "Release" / name if host.is_windows() else directory / name


VENV_PYTHON = default_python()



def default_root(dry_run: bool) -> Path:
    """`build/accuracy-<date>`, and a separate name when nothing will be measured."""
    stamp = datetime.now().strftime("%Y%m%d")
    return ROOT / ("build/accuracy-" + stamp + ("-dryrun" if dry_run else ""))


def toolkit_root(args) -> Path:
    """Where this machine's CUDA toolkit is installed.

    Split out of `cuda_dirs` because two different questions are asked of the
    same root: which directory holds the executables and the CUDA runtime
    (`bin`), and -- on a layout that separates them -- which holds the
    libraries a probe links (`lib64`).
    """
    named = getattr(args, "cuda_dir", None)
    if named is not None:
        return Path(named)
    installed = host.newest_toolkit()
    roots = host.cuda_toolkit_roots()
    # Nothing installed: name the first place this platform would keep one,
    # so the preflight refusal says where it looked.
    return installed if installed is not None else Path(roots[0] if roots else "cuda")


def cuda_dirs(args) -> tuple[Path, Path]:
    """(runtime bin, CUPTI library dir) of this device's toolkit.

    The toolkit belongs to the machine, not to the repository: pinning it to one
    version meant every other machine failed the preflight with no way to say
    where its CUDA lives. `--cuda-dir` names one; otherwise the newest installed
    toolkit wins, so a device with v12.6 beside v12.4 uses v12.6.
    """
    root = toolkit_root(args)
    return root / "bin", host.cupti_library_dir(root)


def probe_toolkit_dirs(args) -> tuple[Path, ...]:
    """The toolkit directories a probe that links this host's backend needs.

    The dense and e2a probes are built for the host's own accelerator
    (`PROBE_BACKEND`): on CUDA that means the toolkit's `bin` (which also holds
    `libcudart`) and CUPTI; on Metal there is no toolkit at all, and asking for
    one would refuse a Mac for the absence of something its probe never links.
    An empty tuple is a statement, not a fallback: it says nothing outside the
    build belongs on this column's artifact and runtime lists.

    A layout that keeps its shared libraries away from `bin` contributes that
    directory too (`host.cuda_library_dir`), because a probe linking ggml-cuda
    asks the loader for cuBLAS before it runs an action and a missing directory
    is then one loader error repeated across every process of the column.
    """
    if host.backend() != "cuda":
        return ()
    directories = list(cuda_dirs(args))
    library = host.cuda_library_dir(toolkit_root(args))
    if library is not None and library not in directories:
        directories.append(library)
    return tuple(directories)

DEFAULT_MODEL = ROOT / "build/real-moe-validation-20261026/model-olmoe-v2/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf"
DEFAULT_RUNTIME_BUILD = ROOT / "build/real-moe-validation-20261028/runtime-windows-build-v8"
DEFAULT_DEPTH_BINARY = default_executable(ROOT / "profiling/depth_validation/native/build-local", "depth-probe")
DEFAULT_E2A_BINARY = default_executable(ROOT / "profiling/native/steady_costs/build-release-v2",
                                        "steady-costs-probe")
E2A_RUNTIME_F32 = ROOT / "profiling/native/highlevel_accuracy/build-runtime-f32-v2/bin"
# The published F32 runtime is a Windows build. On a POSIX host the same
# contract is rebuilt from the pinned tree by
# `profiling.native.highlevel_accuracy.build_f32_runtime_linux`, and the
# directory it wrote is named here instead.


def e2a_runtime(args) -> Path:
    named = getattr(args, "e2a_runtime", None)
    return Path(named) if named else E2A_RUNTIME_F32

MODEL_LABELS = ("cpu", "hybrid")
MODEL_CALIBRATION_JOBS = tuple(label + "-" + kind for label in MODEL_LABELS
                               for kind in ("calibration-profile", "calibration-bare", "inventory"))
MODEL_HELDOUT_JOBS = tuple(label + "-heldout-r" + repeat for label in MODEL_LABELS for repeat in ("0", "1"))


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def human(n: float) -> str:
    return f"{n / 2**30:.1f} GiB"


class Campaign:
    def __init__(self, root: Path, log_path: Path):
        self.root = root
        self.log_path = log_path
        self.started = time.monotonic()
        self.records = []

    def log(self, text: str) -> None:
        line = f"{utc()} {text}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line + "\n")

    def step(self, name: str, argv: list[str]) -> dict:
        """Run one phase as its own process; stop the campaign if it fails."""
        self.log(f"=== {name} ===")
        self.log("    " + " ".join(str(a) for a in argv))
        began = time.monotonic()
        with self.log_path.open("a", encoding="utf-8", newline="\n") as stream:
            done = subprocess.run([str(a) for a in argv], cwd=ROOT, stdout=stream,
                                  stderr=subprocess.STDOUT, env=environment())
        elapsed = time.monotonic() - began
        record = dict(phase=name, argv=[str(a) for a in argv], returncode=done.returncode,
                      seconds=round(elapsed, 1), finished_at_utc=utc())
        self.records.append(record)
        if done.returncode != 0:
            self.log(f"--- {name} FAILED rc={done.returncode} after {elapsed:.0f}s; stopping")
            raise SystemExit(f"{name} failed; see {self.log_path}")
        self.log(f"--- {name} ok ({elapsed:.0f}s)")
        return record


def environment() -> dict:
    env = dict(os.environ)
    parts = ["."]
    for name in ("src", "mapping", "modeling", "profiling"):
        parts.append(str(ROOT / name))
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def free_memory() -> tuple[float, list[tuple[str, float]]]:
    """Free physical memory and the heaviest processes, for the refusal message.

    The figure comes from `profiling.host`, which asks each platform in its own
    terms and needs nothing installed; `psutil` is only how the heaviest-process
    list is read. On a host without `psutil` that list is empty -- the number the
    refusal turns on is still this host's own, and the check that a measurement
    process is resident refuses by name rather than degrading.
    """
    available = host.available_memory_bytes()
    try:
        psutil = __import__("psutil")
    except ImportError:
        return available, []
    heavy = []
    for process in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            rss = process.info["memory_info"].rss
        except Exception:
            continue
        if rss and rss > 512 * 2**20:
            heavy.append((process.info["name"] or str(process.pid), rss))
    heavy.sort(key=lambda item: -item[1])
    return available, heavy


# Executables that time something, and the modules that drive them. A leftover
# process from an interrupted run keeps the machine in the wrong state even after
# its parent is gone -- that is not hypothetical: a probe outlived its shell and
# held the lock through a whole calibration block. Refuse rather than measure.
MEASUREMENT_EXECUTABLES = host.measurement_executables()
MEASUREMENT_MODULES = ("profiling.real_moe_validation.", "profiling.dac_validation.",
                       "profiling.depth_validation.", "profiling.state_calibration.",
                       "profiling.es_abstraction.")


class ResidentCheckUnavailable(RuntimeError):
    """The resident-process check cannot run on this host.

    Raised rather than swallowed: a measurement campaign must not quietly skip
    the check that says nothing else is measuring. It is a separate type so the
    preflight can turn it into a named refusal with the install line, instead of
    a traceback that says only `No module named 'psutil'`.
    """


def resident_measurements() -> list[tuple[int, str]]:
    try:
        psutil = __import__("psutil")
    except ImportError as error:
        raise ResidentCheckUnavailable(
            "psutil is not installed, so the check for a resident measurement "
            "cannot run; install it (`pip install psutil`) and start again") from error
    found = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (process.info["name"] or "").lower()
            cmdline = " ".join(process.info["cmdline"] or [])
        except Exception:
            continue
        if name in MEASUREMENT_EXECUTABLES or any(m in cmdline for m in MEASUREMENT_MODULES):
            found.append((process.info["pid"], f"{name} {cmdline[:120]}".strip()))
    return found


def preflight(campaign: Campaign, column: str, needed_gib: float, artifacts: list[Path],
              strict: bool = True, runtime_dirs: tuple[Path, ...] = (),
              backend: str | None = None) -> list[str]:
    """Everything that would stop this column, logged; raised unless `strict`.

    A dry run calls this with `strict=False`: on a device that has not built its
    probes yet, the whole point of the dry run is to print what is missing so it
    can be built. Refusing there would answer "you may not ask" to the one
    question -- what does this machine still need -- that a new host has.

    `backend` names the accelerator a column's observer is compiled into. The
    model column's taps are inserted into `ggml-cuda.cu`, so on a Metal host that
    column has no observer at all: it refuses by name rather than reporting a
    missing toolkit directory, which would suggest a toolkit would fix it.
    """
    problems: list[str] = []

    def refuse(summary: str, detail: str) -> None:
        campaign.log(f"preflight {column}: {summary}")
        problems.append(detail)
        if strict:
            raise SystemExit(detail)

    if backend is not None and host.backend() != backend:
        refuse("REFUSED, this column's observer is the " + backend + " backend",
               "this host's accelerator is " + str(host.backend()) + ", and the " + column
               + " column reads its measurements from observers compiled into the "
               + backend + " backend. No toolkit directory would change that; the "
               "column runs on a host with a " + backend + " device.")

    if LOCK.exists():
        refuse("REFUSED, the measurement lock is held",
               f"the measurement lock is held: {LOCK.read_text(encoding='utf-8', errors='replace')}\n"
               "Another campaign is running. Nothing is measured while that is true.")
    try:
        running = resident_measurements()
    except ResidentCheckUnavailable as error:
        refuse("REFUSED, the resident list cannot be checked", str(error))
        running = []
    if running:
        refuse("REFUSED, a measurement process is resident",
               "these processes time something and must be closed first:\n  "
               + "\n  ".join(f"pid {pid}  {text}" for pid, text in running))
    missing = [p for p in artifacts if not p.exists()]
    if missing:
        refuse("REFUSED, artifacts are missing",
               "missing artifacts for the " + column + " column:\n  "
               + "\n  ".join(str(p) for p in missing))
    # A file that exists is not a file this host can use: a Windows build on a
    # shared drive satisfies every existence check and cannot start a single
    # probe. The manifest is checked with the directory it describes, because
    # the libraries beside it are what the loader is asked for.
    foreign = []
    for artifact in artifacts:
        if not artifact.exists():
            continue
        for candidate in (artifact, artifact.parent) if artifact.name == "BUILD_COMPLETE.json" \
                else (artifact,):
            reason = host.foreign_artifact_reason(candidate)
            if reason is not None:
                foreign.append(f"{candidate}: {reason}")
    if foreign:
        refuse("REFUSED, artifacts belong to another platform",
               "these artifacts cannot be used on this host:\n  " + "\n  ".join(foreign))
    # A runtime directory can exist, hold this platform's libraries, and still be
    # the wrong size: a build that stopped part way has the first library and not
    # the others. The probe loads all of them by name at launch, so the column
    # would start, prepare a cohort, and then fail one phase in.
    held = [Path(directory) for directory in runtime_dirs if Path(directory).is_dir()]
    if held:
        # Deliberately not `profiling.dac_validation.e2a`: that module imports the
        # model package and numpy, and this process is the one that decides
        # whether the campaign may start at all.
        from profiling import runtime_libraries
        absent = runtime_libraries.missing(held)
        if absent:
            refuse("REFUSED, the runtime is incomplete",
                   "the " + column + " runtime does not hold " + ", ".join(absent)
                   + ":\n  " + "\n  ".join(str(directory) for directory in held)
                   + "\nThe probe built for this host's " + str(host.backend())
                     + " backend loads " + ", ".join(runtime_libraries.components())
                     + " at launch; a build that stopped part way holds some of them "
                       "and the process dies before it measures.")
    available, heavy = free_memory()
    need = needed_gib * 2**30
    line = f"preflight {column}: {human(available)} free, {needed_gib:.1f} GiB required"
    # Recorded either way: a run whose numbers are quoted later should say what
    # else was resident, in case the answer ever needs to be re-examined.
    for name, rss in heavy[:5]:
        line += f"\n    resident {human(rss):>9}  {name}"
    if available < need:
        refuse("REFUSED, memory", "free memory is below what this column needs; "
                                  "close other programs and run again")
    if not problems:
        campaign.log(line + "  ok")
    return problems


def fresh_cohort(output: Path, campaign: Campaign) -> Path:
    """Campaign directories are never reused; an interrupted one is moved aside."""
    if output.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        aside = output.with_name(output.name + "-aborted-" + stamp)
        output.rename(aside)
        campaign.log(f"moved an existing cohort aside: {aside.name}")
    return output


def fresh_file(output: Path, campaign: Campaign) -> Path:
    """Same rule for a receipt that has to live outside the cohort directory.

    `record_tests` creates its receipt exclusively, and `prepare` refuses an
    existing cohort directory, so the receipt cannot be written inside the cohort
    either: it is moved aside here and rewritten.
    """
    if output.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        aside = output.with_name(output.stem + "-aborted-" + stamp + output.suffix)
        output.rename(aside)
        campaign.log(f"moved an existing receipt aside: {aside.name}")
    return output


def require_python(campaign: Campaign, executable: Path) -> Path:
    if executable.exists():
        return executable
    campaign.log(f"{executable} not found; falling back to {sys.executable}")
    return Path(sys.executable)


def retire_authorization(campaign: Campaign, cohort: Path, role: str) -> None:
    """Keep the previous window's receipt under its own name.

    `authorize` creates `ROOT_AUTHORIZATION_RECEIPT.json` exclusively and its own
    docstring says to run it again for the held-out window, so the first receipt
    has to be moved before the second can be written. The published campaign keeps
    exactly this pair: `ROOT_AUTHORIZATION_CALIBRATION.json` beside the canonical
    `ROOT_AUTHORIZATION_RECEIPT.json`, and `collect` reads only the canonical one.
    """
    receipt = cohort / "ROOT_AUTHORIZATION_RECEIPT.json"
    if not receipt.is_file():
        return
    target = cohort / f"ROOT_AUTHORIZATION_{role}.json"
    if target.exists():
        raise SystemExit(f"{target} already exists; two windows of the same role in one cohort")
    receipt.rename(target)
    campaign.log(f"kept the {role.lower()} window's authorization as {target.name}")


# ---------------------------------------------------------------- columns

def column_model(campaign: Campaign, args) -> Path:
    model, build = Path(args.model), Path(args.runtime_build)
    cuda_bin, _ = cuda_dirs(args)
    preflight(campaign, "model", 6.0,
              [model, build / "BUILD_COMPLETE.json", cuda_bin], backend="cuda")
    out = fresh_cohort(campaign.root / "model" / "formal", campaign)
    python = require_python(campaign, VENV_PYTHON)
    tests = fresh_file(campaign.root / "model" / "software-tests.json", campaign)
    tests.parent.mkdir(parents=True, exist_ok=True)

    campaign.log("model: held-out request index {} (index 0 is the published cohort; a device "
                 "that already ran the published campaign passes --heldout-index 1)".format(
                     args.heldout_index))
    campaign.step("model:software tests", [python, "-m", "profiling.real_moe_validation.record_tests",
                                          "--out", tests])
    campaign.step("model:prepare", [python, "-m", "profiling.real_moe_validation.prepare_portable",
                                    "--out", out, "--build", build, "--model", model,
                                    "--runtime-dir", cuda_bin,
                                    "--heldout-index", args.heldout_index,
                                    "--labels", ",".join(MODEL_LABELS)])
    campaign.step("model:freeze", [python, "-m", "profiling.real_moe_validation.runtime_window_campaign",
                                   "freeze", "--out", out, "--tests", tests])
    campaign.step("model:authorize calibration", [python, "-m", "profiling.real_moe_validation.authorize",
                                                  "--out", out, "--user-authorization", args.authorization,
                                                  "--jobs", ",".join(MODEL_CALIBRATION_JOBS)])
    for job in MODEL_CALIBRATION_JOBS:
        campaign.step("model:collect " + job, [python, "-m", "profiling.real_moe_validation.runtime_window_campaign",
                                              "collect", "--out", out, "--job", job])
    campaign.step("model:calibrate", [python, "-m", "profiling.real_moe_validation.runtime_window_predict",
                                      "calibrate", "--out", out])
    campaign.step("model:forecast", [python, "-m", "profiling.real_moe_validation.runtime_window_predict",
                                     "forecast", "--out", out, "--allow-extrapolation"])
    # Prices and predictions are frozen from here on; the held-out window is a
    # second authorization and cannot overwrite the first one's receipt.
    retire_authorization(campaign, out, "CALIBRATION")
    campaign.step("model:authorize heldout", [python, "-m", "profiling.real_moe_validation.authorize",
                                              "--out", out, "--user-authorization", args.authorization,
                                              "--jobs", ",".join(MODEL_HELDOUT_JOBS)])
    for job in MODEL_HELDOUT_JOBS:
        campaign.step("model:collect " + job, [python, "-m", "profiling.real_moe_validation.runtime_window_campaign",
                                              "collect", "--out", out, "--job", job])
    campaign.step("model:evaluate", [python, "-m", "profiling.real_moe_validation.runtime_window_predict",
                                     "evaluate", "--out", out])
    campaign.step("model:condition mean", [python, "-m", "profiling.real_moe_validation.condition_mean",
                                           "--out", out])
    return out


def column_dense(campaign: Campaign, args) -> Path:
    binary = Path(args.depth_binary)
    toolkit = probe_toolkit_dirs(args)
    preflight(campaign, "dense", 2.0, [binary, *toolkit],
              runtime_dirs=runtime_dirs("dense", args))
    out = fresh_cohort(campaign.root / "dense" / "formal", campaign)
    python = require_python(campaign, VENV_PYTHON)
    step = [python, "-m", "profiling.depth_validation.prepare_portable",
            "--out", out, "--binary", binary,
            "--runtime-dir", binary.parent]
    for directory in toolkit:
        step += ["--runtime-dir", directory]
    # Preparing the cohort composes and re-seeds 18 programs and takes about an
    # hour and a half. `--cache` reuses the program templates of an earlier cohort
    # through `reseed_case` with this run's seed: no timing, profile or held-out
    # result is read, and the manifest records that the templates were reused.
    cache = Path(args.dense_cache) if args.dense_cache else None
    if cache is not None and (cache / "inputs").is_dir():
        step += ["--cache", cache]
    elif cache is not None:
        campaign.log(f"no dense template cache at {cache}; composing every program from scratch")
    campaign.step("dense:prepare", step)
    for phase in ("profile", "freeze", "heldout", "evaluate"):
        campaign.step("dense:" + phase, [python, "-m", "profiling.depth_validation.campaign",
                                         phase, "--out", out])
    return out


def column_e2a(campaign: Campaign, args) -> Path:
    binary = Path(args.e2a_binary)
    runtime = e2a_runtime(args)
    toolkit = probe_toolkit_dirs(args)
    preflight(campaign, "e2a", 2.0, [binary, runtime, *toolkit],
              runtime_dirs=runtime_dirs("e2a", args))
    prepared = fresh_cohort(campaign.root / "e2a" / "e2a-v1", campaign)
    out = fresh_cohort(campaign.root / "e2a" / "e2a-v2", campaign)
    (campaign.root / "e2a").mkdir(parents=True, exist_ok=True)
    python = require_python(campaign, VENV_PYTHON)
    prepare = [python, "-m", "profiling.dac_validation.prepare_portable",
               "--out", prepared, "--binary", binary,
               "--runtime-dir", runtime]
    for directory in toolkit:
        prepare += ["--runtime-dir", directory]
    campaign.step("e2a:prepare", prepare)
    campaign.step("e2a:refine", [python, "-m", "profiling.dac_validation.e2a", "refine-prethermal",
                                 "--prepared", prepared, "--out", out])
    # The barrier is taken before any held-out process exists: profile -> freeze ->
    # heldout -> observer -> evaluate. Nothing may be fitted after a target runs.
    campaign.step("e2a:profile", [python, "-m", "profiling.dac_validation.e2a", "profile", "--out", out])
    campaign.step("e2a:freeze", [python, "-m", "profiling.dac_validation.freeze_e2a", "--out", out])
    for phase in ("heldout", "observer", "evaluate"):
        campaign.step("e2a:" + phase, [python, "-m", "profiling.dac_validation.e2a", phase, "--out", out])
    campaign.step("e2a:analyze", [python, "-m", "profiling.dac_validation.analyze_e2a", "--out", out])
    campaign.step("e2a:native reproducibility", [python, "-m", "profiling.dac_validation.native_reproducibility",
                                                 "--out", out])
    campaign.step("e2a:plot", [python, "-m", "profiling.dac_validation.plot_e2a", "--out", out,
                               "--stem", "real_strategy_selection_v2"])
    campaign.step("e2a:report", [python, "-m", "profiling.dac_validation.report_e2a", "--out", out])
    margin = fresh_cohort(campaign.root / "e2a" / "decision-margin", campaign)
    campaign.step("e2a:decision margin", [python, "-m", "profiling.decision_margin.run",
                                          "--campaign", out, "--output", margin])
    return out


COLUMNS = dict(model=column_model, dense=column_dense, e2a=column_e2a)


def runtime_dirs(column: str, args) -> tuple[Path, ...]:
    """The directories whose union must hold the three libraries the probe loads.

    The e2a column names its runtime outright. The dense column does not: its
    probe records the ggml build it was linked against in its own `CMakeCache.txt`,
    and `prepare_portable` derives the runtime from that. When the derivation
    answers nothing -- a probe built outside CMake -- this returns nothing, and the
    preflight stays silent rather than inventing a directory to hold against it.
    """
    toolkit = probe_toolkit_dirs(args)
    if column == "e2a":
        return (e2a_runtime(args), *toolkit)
    if column == "dense":
        binary = Path(args.depth_binary)
        if not binary.is_file():
            return ()
        from profiling.depth_validation.runtime_dirs import ggml_runtime_dirs
        derived = tuple(Path(path) for path in ggml_runtime_dirs(binary))
        return (binary.parent, *toolkit, *derived) if derived else ()
    return ()


def requirements(column: str, args) -> tuple[float, list[Path]]:
    """Free memory and the artifacts a column needs, without touching hardware."""
    cuda_bin, _ = cuda_dirs(args)
    toolkit = probe_toolkit_dirs(args)
    if column == "model":
        return 6.0, [Path(args.model), Path(args.runtime_build) / "BUILD_COMPLETE.json", cuda_bin]
    if column == "dense":
        return 2.0, [Path(args.depth_binary), *toolkit]
    return 2.0, [Path(args.e2a_binary), e2a_runtime(args), *toolkit]


def headline(root: Path, produced: dict) -> list[str]:
    """Quote the paper-facing cell from each column's own output.

    Nothing here recomputes an error: each number is read from the file the
    campaign wrote, so the summary cannot disagree with the campaign it describes.
    """
    lines = []
    condition_mean = Path(produced["model"]) / "CONDITION_MEAN_EVALUATION.json" if "model" in produced else None
    if condition_mean is not None and condition_mean.is_file():
        result = json.loads(condition_mean.read_text(encoding="utf-8"))
        mean, worst = result.get("mean_APE", {}), result.get("worst_APE", {})
        lines.append("- Model cell: mean APE {:.4%}, worst {:.4%} over {} of {} declared "
                     "conditions (`{}`)".format(mean.get("total") or 0.0, worst.get("total") or 0.0,
                                                result.get("complete_endpoint_conditions"),
                                                result.get("declared_conditions"), condition_mean))
        if result.get("complete_endpoint_conditions") != result.get("declared_conditions"):
            lines.append("  - the conditions that did not complete are listed with a reason and "
                         "are excluded, not averaged in")
    dense = Path(produced["dense"]) / "summary.json" if "dense" in produced else None
    if dense is not None and dense.is_file():
        result = json.loads(dense.read_text(encoding="utf-8"))
        for provider, entry in sorted(result.get("providers", {}).items()):
            lines.append("- Dense cell, {}: {} of {} heldout cases pass (`{}`)".format(
                provider, entry.get("accuracy_passes"), entry.get("cases"), dense))
        worst = {}
        for case in result.get("cases", []):
            for provider, entry in case.get("providers", {}).items():
                value = entry.get("max_process_mean_ape")
                if value is not None:
                    worst[provider] = max(worst.get(provider, 0.0), value)
        for provider, value in sorted(worst.items()):
            lines.append("- Dense cell, {}: process-mean max APE {:.4%} over {} cases".format(
                provider, value, result.get("declared_cases")))
    e2a = Path(produced["e2a"]) / "summary.json" if "e2a" in produced else None
    if e2a is not None and e2a.is_file():
        result = json.loads(e2a.read_text(encoding="utf-8"))
        counts = result.get("counts", {})
        lines.append("- Matched Dense strategies: {} of {} workloads pick the real winner "
                     "with zero point regret; {} of {} candidate programs and {} of {} groups "
                     "are fully qualified (`{}`)".format(
                         counts.get("regret_point_passes"), counts.get("declared_groups"),
                         counts.get("qualified_programs"), counts.get("programs"),
                         counts.get("qualified_groups"), counts.get("declared_groups"), e2a))
        if counts.get("qualified_programs") != counts.get("programs"):
            lines.append("  - a candidate is qualified only when its mean, variability and "
                         "stamped-versus-bare observer checks all pass; the report keeps the "
                         "per-program failures rather than averaging over them")
    # The margin study is a sibling of the cohort, not a part of it: `column_e2a`
    # runs it into `root/e2a/decision-margin` so it can be recomputed without
    # touching the frozen cohort.
    margin = Path(produced["e2a"]).parent / "decision-margin" / "decision-margin.json" if "e2a" in produced else None
    if margin is not None and margin.is_file():
        sweep = json.loads(margin.read_text(encoding="utf-8")).get("sweep", {})
        per_error = sweep.get("per_error", {})
        if per_error:
            def flips(key):
                rows = per_error[key]
                total = sum(row["draws"] for row in rows)
                return sum(row["flips"] for row in rows) / total, total
            parts = []
            for key in sorted(per_error, key=float):
                rate, total = flips(key)
                parts.append("{:.0f}%: {:.3f}% of {}".format(float(key) * 100, rate * 100, total))
            worst = max(max(row["regret_max"] for row in rows) for rows in per_error.values())
            lines.append("- Decision margin under a uniform per-price error: {}; worst regret "
                         "{:.4%} (`{}`)".format("; ".join(parts), worst, margin))
        for name in ("E2A_REPORT.md", "postfreeze-diagnostics.json"):
            candidate = Path(produced["e2a"]) / name

            if candidate.is_file():
                lines.append(f"- Matched Dense strategies, detail: `{candidate}`")
    return lines


def summarize(campaign: Campaign, produced: dict) -> Path:
    lines = ["# Accuracy campaign", "",
             f"Started {campaign.started_at}, finished {utc()}.",
             f"Host `{socket.gethostname()}`; python `{sys.executable}`.", "",
             "Every column below was measured on a machine with nothing else running.",
             "The numbers belong to this machine's state; they do not transfer to another device.", ""]
    for column, path in produced.items():
        lines.append(f"## {column}")
        lines.append(f"Output: `{path}`")
        lines.append("")
    numbers = headline(campaign.root, produced)
    if numbers:
        lines.append("## Headline numbers")
        lines.append("")
        lines.append("Read from the phase outputs below, not recomputed.")
        lines.append("")
        lines.extend(numbers)
        lines.append("")
    lines.append("## Phases")
    lines.append("")
    lines.append("| phase | seconds | rc |")
    lines.append("|---|---|---|")
    for record in campaign.records:
        lines.append(f"| {record['phase']} | {record['seconds']} | {record['returncode']} |")
    summary = campaign.root / "CAMPAIGN_SUMMARY.md"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    (campaign.root / "campaign-records.json").write_text(
        json.dumps(dict(host=socket.gethostname(), started_at_utc=campaign.started_at,
                        finished_at_utc=utc(), phases=campaign.records, produced=
                        {k: str(v) for k, v in produced.items()}), indent=1),
        encoding="utf-8", newline="\n")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--columns", default="model,dense,e2a",
                        help="comma-separated subset of model,dense,e2a (order is ignored; "
                             "columns always run in the order model, dense, e2a)")
    parser.add_argument("--root", type=Path, help="output root; default build/accuracy-<date>")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--runtime-build", type=Path, default=DEFAULT_RUNTIME_BUILD)
    parser.add_argument("--depth-binary", type=Path, default=DEFAULT_DEPTH_BINARY)
    parser.add_argument("--dense-cache", type=Path,
                        help="an earlier dense cohort whose program templates may be re-seeded; "
                             "no timing, profile or held-out result is read from it")
    parser.add_argument("--e2a-binary", type=Path, default=DEFAULT_E2A_BINARY)
    parser.add_argument("--e2a-runtime", type=Path, default=None,
                        help="the strict-F32 runtime directory the e2a probe loads its "
                             "libraries from; the default is the published Windows build, "
                             "so a POSIX host names the one it built with "
                             "profiling.native.highlevel_accuracy.build_f32_runtime_linux")
    parser.add_argument("--cuda-dir", type=Path,
                        help="CUDA toolkit root; default is the newest installed toolkit "
                             "(Windows: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*; "
                             "Linux: /usr/local/cuda*, /opt/cuda, /usr/lib/cuda)")
    parser.add_argument("--heldout-index", default="0",
                        help="held-out request stratum; index 0 is the published cohort's, so a "
                             "device that already ran the published campaign passes 1")
    parser.add_argument("--authorization", default=(
        "standalone campaign: operator closed every other program on this machine and ran "
        "tools/run_accuracy_campaign.py, one native process at a time"))
    parser.add_argument("--dry-run", action="store_true", help="run the preflight only")
    args = parser.parse_args(argv)

    requested = [c.strip() for c in args.columns.split(",") if c.strip()]
    unknown = [c for c in requested if c not in COLUMNS]
    if unknown:
        raise SystemExit(f"unknown column(s): {', '.join(unknown)}; choose from {', '.join(COLUMNS)}")
    if not requested:
        raise SystemExit("no columns selected")

    # A dry run measures nothing, so it must not append to a real cohort's log:
    # on a machine that already ran today, the default root would be that
    # cohort's, and its log would end with "nothing was measured".
    root = (args.root or default_root(args.dry_run)).resolve()
    root.mkdir(parents=True, exist_ok=True)
    campaign = Campaign(root, root / "campaign.log")
    campaign.started_at = utc()
    campaign.log("")
    campaign.log("=" * 78)
    campaign.log(f"accuracy campaign on {socket.gethostname()} -> {root}")
    campaign.log("columns: " + ", ".join(c for c in COLUMNS if c in requested))
    if args.dry_run:
        campaign.log("dry run: preflight only")

    # Refuse on a host that cannot measure before spending an hour composing
    # programs, not three layers into a CMake build. A dry run reports the
    # verdict and continues, so `--dry-run` on a Mac says why rather than
    # looking like a broken checkout.
    host_problem = None
    try:
        host.require_measurement_host(args.cuda_dir)
    except host.UnsupportedHost as error:
        host_problem = str(error)
        campaign.log(f"this host cannot run a V-A column: {error}")
        if not args.dry_run:
            raise SystemExit(str(error))
    else:
        campaign.log("host: a CUDA toolkit is present")

    produced = {}
    for name in COLUMNS:                      # a fixed order, whatever the argument order was
        if name not in requested:
            continue
        campaign.log("")
        campaign.log(f"##### {name} #####")
        if args.dry_run:
            needed, artifacts = requirements(name, args)
            missing = preflight(campaign, name, needed, artifacts, strict=False,
                                runtime_dirs=runtime_dirs(name, args))
            # A host that cannot measure is a reason a column is not ready, and it
            # is the decisive one -- a Mac with every artifact present would
            # otherwise read "preflight ok" and learn nothing from it.
            if host_problem is not None:
                missing.insert(0, "this host cannot run a V-A column: " + host_problem)
            # A dry run on a device that has not built its probes yet is the
            # normal case -- it is how a new host learns what to build -- so the
            # missing list is the answer, not a failure. The detail lines are
            # printed too: "missing artifacts for the dense column:" on its own
            # names nothing, and the names are the whole answer on a first run.
            for problem in missing:
                lines = problem.splitlines() or [problem]
                campaign.log(f"dry run: the {name} column is not ready: {lines[0]}")
                for detail in lines[1:]:
                    campaign.log(f"dry run:   {detail.strip()}")
            continue
        produced[name] = COLUMNS[name](campaign, args)

    if args.dry_run:
        campaign.log("preflight finished; nothing was measured")
        return 0
    summary = summarize(campaign, produced)
    campaign.log("")
    campaign.log(f"campaign finished in {time.monotonic() - campaign.started:.0f}s")
    campaign.log(f"summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
