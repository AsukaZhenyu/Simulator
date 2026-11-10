"""What this machine can build, measure and test -- answered before any of it starts.

The V-A columns were written for one discrete-memory CUDA box. Since then the
repository has to run on six: two Windows machines, a Linux laptop, a 5090
server, an Apple-silicon Mac and a DGX Spark. The differences that actually bite
are few and none of them is speed:

  * which ggml backend computes here, and therefore which library a probe links
    (`ggml-cuda`, `ggml-metal`, or none on a CPU-only host);
  * whether the accelerator has memory of its own -- on a unified box a "copy to
    the GPU" is an alias, which the simulator spells SHARE;
  * whether a CUDA toolkit with CUPTI is installed, which a CUDA build's link line
    requires even in its clock-only modes;
  * whether a column's *observer* exists on this backend -- the model column's tap
    lives inside the llama.cpp CUDA backend, so a Metal host has no model column
    even though it builds the dense and e2a probes;
  * whether the pinned llama.cpp tree is present, which the real-MoE probe and
    the model column cannot be built without.

Each of those is asked of `profiling.host` and printed as a verdict rather than
left to fail three layers down in a build. The point is that one command on a
new machine says what it is good for, in the words of the thing that will later
refuse it.

Nothing here builds or measures. `--json` gives the same answers for a receipt.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from profiling import host
from profiling import runtime_libraries
from profiling.native.highlevel_accuracy.build_f32_runtime import METAL_STRICT_F32_KERNELS

ROOT = Path(__file__).resolve().parents[1]


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def toolkit() -> dict:
    """The CUDA pieces the link line needs, if this host has any."""
    if host.is_macos():
        return {"present": False,
                "reason": "macOS has had no CUDA since 10.13; the backend is Metal"}
    newest = host.newest_toolkit()
    if newest is None:
        return {"present": False, "reason": "no CUDA toolkit with a bin/ directory found"}
    cupti = host.cupti_library_dir(newest)
    return {"present": True, "root": str(newest), "bin": str(newest / "bin"),
            "cupti_dir": str(cupti),
            "cupti_present": bool(cupti.is_dir() and any(cupti.glob("*cupti*")))}


def probes() -> dict:
    """The three native probes, whether each is on disk, and what it would need."""
    from tools import run_accuracy_campaign as campaign

    named = {
        "model": (campaign.DEFAULT_MODEL, campaign.DEFAULT_RUNTIME_BUILD),
        "dense": (campaign.DEFAULT_DEPTH_BINARY, None),
        "e2a": (campaign.DEFAULT_E2A_BINARY, campaign.E2A_RUNTIME_F32),
    }
    answer = {}
    for column, (binary, runtime) in named.items():
        entry = {"binary": str(binary), "binary_present": Path(binary).is_file()}
        if runtime is not None:
            entry["runtime"] = str(runtime)
            entry["runtime_present"] = Path(runtime).is_dir()
            if entry["runtime_present"]:
                entry["runtime_missing_libraries"] = runtime_libraries.missing([Path(runtime)])
        answer[column] = entry
    return answer


def llama_tree() -> dict:
    """The pinned tree the real-MoE probe and the model column are built from."""
    import os

    named = os.environ.get("LLAMA_B8705_ROOT")
    if named:
        root = Path(named)
        return {"declared": "LLAMA_B8705_ROOT", "path": str(root), "present": root.is_dir()}
    return {"declared": None, "path": None, "present": False,
            "reason": "set LLAMA_B8705_ROOT to the b8705 checkout; the model column "
                      "and the real-MoE probe build from it"}


def verdicts(facts: dict) -> list[tuple[str, str, str]]:
    """One row per capability: what it is, whether it works here, and why not."""
    backend = facts["backend"]
    rows = []

    rows.append(("software tests", "yes",
                 "tools/run_tests.py; the suites that need a GPU skip themselves"))
    rows.append(("simulator + mapping", "yes",
                 "no device code: a cost table is numbers, not a machine"))

    if backend == "metal":
        dense, e2a, model = (facts["probes"][name] for name in ("dense", "e2a", "model"))

        def metal_column(entry, needs=(), ready_why="", absent_why=""):
            """A column's verdict on a Metal host, naming the first missing piece."""
            for ok, why in needs:
                if not ok:
                    return "not yet", why
            if not entry["binary_present"]:
                return "not yet", absent_why
            return "yes", ready_why

        rows.append(("V-A dense column", *metal_column(
            dense,
            ready_why="clock-only modes: the portable clock; a Metal build has no CUPTI "
                      "to query and refuses the cupti instrumentation by name",
            absent_why="the probe is not built on this host "
                       "(`profiling.depth_validation.build_probe`, PROBE_BACKEND=metal)")))
        rows.append(("V-A e2a column", *metal_column(
            e2a,
            needs=[(bool(e2a.get("runtime_present")),
                    "the strict-F32 runtime directory is absent, and this column's "
                    "requirement is not satisfiable by building one here: "
                    + METAL_STRICT_F32_KERNELS),
                   (not e2a.get("runtime_missing_libraries"),
                    "the strict-F32 runtime does not hold "
                    + ", ".join(e2a.get("runtime_missing_libraries") or []))],
            ready_why="strict-F32 runtime and probe are both built",
            absent_why="the probe is not built on this host "
                       "(`profiling.dac_validation.build_probe`, PROBE_BACKEND=metal)")))
        rows.append(("V-A model column", "no",
                     "its observer is inserted into the llama.cpp CUDA backend; a Metal "
                     "host has no tap to insert it into, so this column runs on a CUDA box"))
    elif backend == "cuda":
        kit = facts["toolkit"]
        dense, e2a, model = (facts["probes"][name] for name in ("dense", "e2a", "model"))

        def column(entry, needs=(), ready_why=""):
            """A column's verdict, naming the first thing that is missing."""
            if not kit["present"]:
                return "not yet", "no CUDA toolkit found"
            if not kit.get("cupti_present"):
                return "not yet", ("the toolkit is installed but CUPTI is not where "
                                   "the link line looks")
            for ok, why in needs:
                if not ok:
                    return "not yet", why
            if not entry["binary_present"]:
                return "not yet", "the probe is not built on this host"
            return "yes", ready_why

        rows.append(("V-A dense column", *column(
            dense, ready_why="clock-only modes: a portable clock, no CUPTI query "
                             "at run time")))
        rows.append(("V-A e2a column", *column(
            e2a,
            needs=[(bool(e2a.get("runtime_present")),
                    "the strict-F32 runtime directory is absent "
                    "(`build_f32_runtime_linux.py` / `build_f32_runtime.py`)"),
                   (not e2a.get("runtime_missing_libraries"),
                    "the strict-F32 runtime does not hold "
                    + ", ".join(e2a.get("runtime_missing_libraries") or []))],
            ready_why="strict-F32 runtime and probe are both built")))
        rows.append(("V-A model column", *column(
            model,
            needs=[(facts["llama_tree"]["present"],
                    "set LLAMA_B8705_ROOT to the b8705 checkout")],
            ready_why="builds from the pinned tree with the portable shim")))
    else:
        # A toolkit with no device is the one host that reads as ready and is
        # not: it compiles every probe here and cannot run one. The reason the
        # backend answered `None` is printed rather than left to be inferred.
        reason = facts.get("backend_reason")
        rows.append(("V-A columns", "no", reason or
                     "no accelerator this repository builds a probe against"))

    if facts["device_memory"] == "shared":
        rows.append(("transfer prices", "different",
                     "one pool: SHARE aliases an allocation at 0 bytes, and a "
                     "transfer price would describe a link this machine does not have"))
    elif facts["device_memory"] == "unknown":
        # An aarch64 host may carry a discrete card, so the architecture cannot
        # decide, and a wrong `shared` would price every copy at zero. The owner
        # checks the topology once and states it.
        rows.append(("transfer prices", "not decided",
                     "the architecture does not decide whether the accelerator has "
                     "its own pool; after checking, run with "
                     "IR_DEVICE_MEMORY=shared (SoC pool) or IR_DEVICE_MEMORY=separate "
                     "(card on PCIe)"))
    return rows


def collect() -> dict:
    facts = {
        "platform": sys.platform,
        "os": f"{host.__name__}: {sys.platform}",
        "arch": host.arch(),
        "backend": host.backend(),
        "backend_reason": host.backend_reason(),
        "device_memory": host.device_memory(),
        "online_cpus": host.online_cpus(),
        "python": sys.version.split()[0],
        "toolkit": toolkit(),
        "llama_tree": llama_tree(),
        "probes": probes(),
        "runtime_components": list(runtime_libraries.components()),
    }
    try:
        facts["device_name"] = host.accelerator_device_name()
    except host.UnsupportedHost as error:
        facts["device_name"] = None
        facts["device_name_reason"] = str(error)
    if host.backend() == "cuda":
        facts["toolkit"]["cuda_available"] = host.cuda_available()
    return facts


def render(facts: dict) -> str:
    lines = [
        "host        {} {}".format(facts["platform"], facts["arch"]),
        "backend     {}{}".format(
            facts["backend"] or "none (CPU only)",
            "  device " + facts["device_name"] if facts["device_name"] else ""),
    ]
    if not facts["backend"] and facts.get("backend_reason"):
        # "none (CPU only)" on a host with a toolkit installed is the answer that
        # needs a second sentence, or it reads as a broken install.
        lines.append("            " + facts["backend_reason"])
    lines += [
        "memory      {}".format(facts["device_memory"]),
        "libraries   {}".format(", ".join(facts["runtime_components"])),
        "python      {}".format(facts["python"]),
        "cpus        {}".format(facts["online_cpus"]),
    ]
    kit = facts["toolkit"]
    lines.append("cuda        " + ("{}  cupti {}".format(
        kit.get("root"), "yes" if kit.get("cupti_present") else "missing")
        if kit["present"] else "absent -- " + kit.get("reason", "")))
    tree = facts["llama_tree"]
    lines.append("llama.cpp   " + (tree["path"] if tree["present"]
                                   else "absent -- " + tree.get("reason", "")))
    for column, entry in facts["probes"].items():
        state = "on disk" if entry["binary_present"] else "not built"
        missing = entry.get("runtime_missing_libraries")
        if missing:
            state += ", runtime lacks " + ", ".join(missing)
        lines.append("{:<11} {}".format(column, state))
    lines.append("")
    lines.append("capability                     verdict   why")
    for what, verdict, why in verdicts(facts):
        lines.append("{:<30} {:<9} {}".format(what, verdict, why))
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="the same facts as a receipt")
    args = parser.parse_args(argv)
    facts = collect()
    print(json.dumps(facts, indent=2) if args.json else render(facts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
