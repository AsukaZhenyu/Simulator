#!/usr/bin/env python3
"""Dimension A (accuracy) over the re-instantiated 800 topologies.

For every case: run the compiled IR on the real machine, then compare

  A1  per priced action unit: measured action time vs the frozen table's price
  A2  per case: measured makespan vs the makespan the same schedule gets when
      every node carries its table price (copies carry no price -- the tables
      have no transfer entries -- so they are declared, not silently dropped)

Everything comes from one execution: the runner emits per-node times and the
wall clock of the same replica.

Usage::

    python -m profiling.ir_ggml.run --model qwen3-0.6b --tokens 64 [--replicas 2] [--limit N]
    python -m profiling.ir_ggml.run --model qwen3-0.6b --analyse-only

The three machines of an invocation are all overridable, because the binary, its DLLs
and the output directory are properties of *this* checkout, not of the campaign::

    IR_RUNNER=<...>/ggml-ir-runner[.exe]   the executor (build_ir_runner.py prints this)
    IR_DLL_DIR=<...>/build/bin/Release     where ggml-*.dll live; skipped if unset and
                                            the runner has no siblings to find them in
    IR_RESULTS=<dir>                       default: this package's results/
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIMULATOR = HERE.parents[1]
ROOT = SIMULATOR.parent
EXE = ".exe" if os.name == "nt" else ""


def _default_runner() -> Path:
    # Single-config generators (Linux/Mac default) put it at the tree root.
    out = SIMULATOR / "profiling" / "native" / "build-ir"
    for candidate in (out / "Release" / f"ggml-ir-runner{EXE}", out / f"ggml-ir-runner{EXE}"):
        if candidate.exists():
            return candidate
    return out / "Release" / f"ggml-ir-runner{EXE}"


def _default_dll_dir() -> Path:
    """Where this checkout's ggml shared libraries live.

    A fresh clone gets them from ``build_ir_runner.py --fetch`` (into ``native/_llama``);
    the campaign machine kept its own build under ``llama.cpp-implement/``.  Multi-config
    generators add ``Release/``.  Taking the first that exists means the fetch path needs
    no environment variables at all, and a campaign checkout still behaves as it did.
    """
    fetched = SIMULATOR / "profiling" / "native" / "_llama" / "build" / "bin"
    campaign = ROOT / "llama.cpp-implement" / "release-b8705" / "llama.cpp-b8705" / "build" / "bin"
    for base in (fetched, campaign):
        for candidate in (base / "Release", base):
            if candidate.is_dir():
                return candidate
    return campaign / "Release"


RUNNER = Path(os.environ.get("IR_RUNNER", _default_runner()))
DLL_DIR = Path(os.environ.get("IR_DLL_DIR", _default_dll_dir()))
RESULTS = Path(os.environ.get("IR_RESULTS", HERE / "results"))

sys.path.insert(0, str(HERE))
import instantiate as inst  # noqa: E402  (same directory)

# Declared before the campaign runs (`PREREG-design.md` §5): the control family's median A1
# must land inside this band, otherwise the tranche is on a different observation band and
# none of its readings are compared.  A whole day of this project already separates two
# bands by 8.7--10.6% (`measurement-band-state`), so the band is wide enough not to fire on
# that and narrow enough to catch a swept state.
STATE_BAND = (0.85, 1.20)



def environment() -> dict:
    env = dict(os.environ)
    if DLL_DIR.is_dir():
        env["PATH"] = str(DLL_DIR) + os.pathsep + env.get("PATH", "")
    # The executor synchronizes the CUDA backend after every node, so a backend that
    # starts capturing a replay graph on its second execution hits
    # "operation not permitted when stream is capturing" (5 cases failed that way in
    # the first attempt).  The frozen CUDA stage prices were measured with graphs
    # explicitly disabled; the same declaration applies here.
    env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    return env


def table_for(model: str, extensions: list[Path] = ()) -> dict:
    """(device, kind, phase, tokens, visible_history) -> ns.

    The tables are shape-conditioned in both directions: prefill entries exist at 16/64/256
    tokens with an empty history, decode entries at one token with a visible history from
    0/128/512/1024.  A node is priced at *its own* (phase, shape), never at the case
    header's, so a pd case's decode round cannot be charged a prefill price.

    **`device` is part of the key.**  Every frozen export is `device: cpu`, and the key used
    to drop it -- so a separately measured GPU table merged as an extension would have
    *overwritten* the CPU price at every shape the two share, and A1 would have drifted on
    a table that looked unchanged.  An entry without a `device` is refused rather than
    defaulted to `cpu`: the default is the defect.

    `extensions` add prices from a separate measured campaign (same idea as the transfer
    prices: the frozen export is never edited).  Each file holds either a list of entries
    or ``{"prices": [...]}`` with `device`, `kind`, `phase`, `tokens`, `visible_history`,
    `duration_ns`.
    """
    prices = {}

    def add(entries, source):
        for entry in entries:
            if "device" not in entry:
                raise SystemExit(f"{source}: entry {entry.get('kind')}/{entry.get('phase')} "
                                 f"has no `device`; defaulting it to cpu would overwrite the "
                                 f"cpu price at the same shape")
            key = (str(entry["device"]), entry["kind"], entry["phase"], int(entry["tokens"]),
                   int(entry.get("visible_history", 0)))
            prices[key] = float(entry["duration_ns"])

    document = json.loads((inst.EXPORTS[model] / "costs.json").read_text(encoding="utf-8"))
    add(document["entries"], str(inst.EXPORTS[model] / "costs.json"))
    for path in extensions:
        extra = json.loads(Path(path).read_text(encoding="utf-8"))
        add(extra["prices"] if isinstance(extra, dict) else extra, str(path))
    return prices


def read_case_file(path: Path) -> dict:
    header = {}
    nodes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        w = line.split()
        if w[0] == "node":
            params = {}
            for kv in " ".join(w[5:]).split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    params[k] = v
            nodes.append({"index": int(w[1]), "op": w[2], "machine": w[3],
                          "deps": [] if w[4] == "-" else [int(x) for x in w[4].split(",")],
                          "params": params})
        elif len(w) == 2:
            header[w[0]] = w[1]
    return {"header": header, "nodes": nodes}


def predicted_makespan(nodes: list[dict], price_of: dict, tokens: int) -> tuple[float, dict]:
    """Longest path with one serial queue per machine, lowest index first.

    This is the same policy the runner uses, so the comparison isolates the
    prices rather than the scheduling heuristic.
    """
    end = {}
    # One queue per machine *that the case actually uses*.  Listing the machines by hand
    # (`cpu/gpu/pcie`) is how the third tier went unnoticed until the first v4 case ran: an
    # `ssd` node is a real serial machine in the executor, so a hard-coded table raises
    # KeyError on exactly the family that measures it.
    machine_free = {n["machine"]: 0.0 for n in nodes}
    machine_free.setdefault("cpu", 0.0)
    machine_free.setdefault("gpu", 0.0)
    machine_free.setdefault("pcie", 0.0)
    pending = sorted(nodes, key=lambda n: n["index"])
    while pending:
        progressed = False
        for n in list(pending):
            if any(d not in end for d in n["deps"]):
                continue
            start = max([machine_free[n["machine"]]] + [end[d] for d in n["deps"]])
            duration = price_of.get(n["index"], 0.0)
            end[n["index"]] = start + duration
            machine_free[n["machine"]] = end[n["index"]]
            pending.remove(n)
            progressed = True
        if not progressed:
            raise RuntimeError("cyclic dependencies in case")
    return max(end.values()), end


def _failure_reasons(stdout: str) -> list[str]:
    """The `status` of every node that did not run, from the executor's own JSON.

    Empty when the executor never got as far as printing a document (a missing flag, a
    parse error), which is exactly the case where stderr is the only evidence left.
    """
    try:
        run = json.loads(stdout[stdout.index("{"):])
    except (ValueError, IndexError):
        return []
    return [str(n.get("status", "")) for n in run.get("nodes", []) if not n.get("ok", True)]


def analyse(case: dict, run: dict, prices: dict, tokens: int, manifest: dict,
            copy_prices: dict | None = None, gpu_node_overhead_ns: float = 0.0) -> dict:
    """A1 per priced unit and A2 for one case, from one replica.

    A priced unit is only comparable with the table when it is machine-atomic and
    on the table's device: every entry in the frozen tables is ``device: cpu``,
    and a unit whose sub-actions were split across machines has no single
    measured duration that the price is meant to describe.  Both cases are
    counted as ``unpriceable`` with their reason instead of silently dropped.
    """
    # replica 0 pays CUDA module load on the first GPU node, so it is a warmup and
    # is never a reading; one replica in total means there is no warmup to discard.
    replicas = list(range(1, len(run["wall_ns"]))) if len(run["wall_ns"]) > 1 else [0]
    def reading(node_index: int) -> float:
        return statistics.median(float(run["nodes"][node_index]["dur_ns"][r]) for r in replicas)

    def spread(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = statistics.fmean(values)
        return statistics.stdev(values) / mean if mean else 0.0

    # An *action* is the topology's own unit, not the table's kind.  The node ids keep
    # the request/phase/layer path, and the four attn.* sub-ops of one attention action
    # share everything up to their final segment; grouping by kind instead would fuse a
    # pd topology's prefill and decode actions into one "unit" and compare the sum of
    # two actions against a single price (exactly 2.0 on every pd/pipeline case).
    groups: dict[str, list[dict]] = {}
    for node, meta in zip(case["nodes"], manifest["nodes"]):
        if meta["priced_unit"] is None:
            continue
        # `g:<request>-<phase>-layer0:op:layer0:<request>:<tail>` -- the prefix names the
        # layer and the request, so it is shared by proj/attn/ffn.  Only the four attn.*
        # sub-ops of one attention belong to a single action.
        prefix, tail = meta["id"].rsplit(":", 1)
        if tail.startswith("attn"):
            action = prefix + ":attn"
        elif meta["op"] in ("router", "combine") or meta["op"].startswith("expert"):
            # The MoE joint block is *one* priced action: the router, the expert set and
            # the combine of a layer.  Its nodes differ in the segment before the tail
            # (`request0` vs `__moe_batch__`), so the attention rule above would split
            # them and charge the whole `moe_ffn` price to each -- three prices for one
            # action.  The layer path is everything up to `...:op:layer0`.
            action = ":".join(meta["id"].split(":")[:4]) + ":moe"
        else:
            action = meta["id"]
        groups.setdefault(action, []).append(
            {"index": node["index"], "machine": node["machine"], "id": meta["id"],
             "op": meta["op"], "unit": meta["priced_unit"],
             # the shape this action executed at, and the phase it belongs to
             "phase": "decode" if "-decode" in meta["id"] else "prefill",
             "tokens": meta.get("tokens", tokens), "history": meta.get("history", 0),
             "dur": reading(node["index"])})
    units: dict[str, dict] = {}
    unpriceable: dict[str, str] = {}
    price_of = {}
    # A case that re-quantizes its weights or its KV cache executes tensors the frozen
    # table never measured: the table holds exactly one signature per export, and its
    # price is that signature's.  Charging it anyway would read as accuracy, so the
    # action is named as unpriced with the signature it would have needed.
    signature_override = case["header"].get("price_signature_override")
    for action, members in groups.items():
        unit = members[0]["unit"]
        if signature_override:
            unpriceable[unit] = "no_price_for_signature:" + signature_override
            continue
        machines = {m["machine"] for m in members}
        measured = sum(m["dur"] for m in members)
        if len(machines) > 1:
            unpriceable[unit] = "split_across_machines:" + ",".join(sorted(machines))
            continue
        machine = machines.pop()
        shapes = {(m["phase"], m["tokens"], m["history"]) for m in members}
        if len(shapes) > 1:
            unpriceable[unit] = "shape_mixed_within_action:" + ",".join(
                sorted(f"{p}@{t}t/{h}h" for p, t, h in shapes))
            continue
        phase, node_tokens, node_history = shapes.pop()
        # The device is looked up, not assumed.  Before there were GPU prices this line read
        # `if machine != "cpu"`, which is right only while the tables are cpu-only: with a
        # gpu table present it would refuse a price that exists.
        key = (machine, unit, phase, node_tokens, node_history)
        if key not in prices:
            # two different facts, named separately: no price family for this device at
            # all, versus a family that does not cover this shape
            if not any(k[0] == machine and k[1] == unit for k in prices):
                unpriceable[unit] = "no_price_for_device:" + machine
            elif machine == "cpu":
                unpriceable[unit] = f"no_price_at:{phase}@{node_tokens}t/{node_history}h"
            else:
                unpriceable[unit] = (f"no_price_at:{machine}/{phase}"
                                     f"@{node_tokens}t/{node_history}h")
            continue
        # A shape the frozen table does not price is a *coverage* fact about the case, not
        # an accuracy reading: it is named above rather than fitted or substituted from a
        # neighbouring shape.  `--prices` can close one with a separately measured price.
        # a priced action is one action: an IR that splits it into qk/softmax/pv/output
        # still pays one attention price, so the charge sits on its first node
        #
        # A non-CPU unit additionally pays one device dispatch per plan node it expands to.
        # `ir_runner.cpp:1234-1237` brackets every node with two `ggml_backend_synchronize`
        # calls; a CPU backend synchronizes for free, a CUDA backend waits on the stream, and
        # `PREREG-gpu-prices.md` 4f measures that floor directly on a node whose own compute
        # is negligible (`attn.softmax` at 256 tokens: 71.4 us, 4% spread).  The node count
        # comes from the topology, not from a fit, and the default 0 reproduces every
        # published A1 bit for bit.
        per_node = gpu_node_overhead_ns if machine != "cpu" else 0.0
        overhead = per_node * len(members)
        for position, m in enumerate(sorted(members, key=lambda m: m["index"])):
            price_of[m["index"]] = (prices[key] if position == 0 else 0.0) + per_node
        units[action] = {"unit": unit, "phase": phase, "tokens": node_tokens,
                         "history": node_history,
                         "measured_ns": measured, "predicted_ns": prices[key] + overhead,
                         "price_ns": prices[key], "node_overhead_ns": overhead,
                         "nodes": [m["index"] for m in members],
                         "ratio": measured / (prices[key] + overhead)}
    # Copies.  Until `PREREG-gpu-prices.md`'s 4b they were charged nothing at all, which
    # is not a modelling choice: a plan with more weight traffic then looked cheaper than
    # the same plan without it, so a ranking over predicted makespans preferred the plan
    # that moves the most bytes.  The price is a lookup at (machine, direction, bytes);
    # `copy_prices is None` reproduces the old behaviour exactly, which is why every A1
    # already published stays valid and the switch has to be asked for.
    copy_unpriced: dict[int, str] = {}
    copy_charged: dict[int, float] = {}
    if copy_prices is not None:
        for node in case["nodes"]:
            if node["op"] != "copy":
                continue
            value, reason = instantiate.copy_price(copy_prices, node["machine"],
                                                   node.get("params") or {})
            if value is None:
                copy_unpriced[int(node["index"])] = reason
            else:
                copy_charged[int(node["index"])] = value
        price_of.update(copy_charged)

    predicted, _ = predicted_makespan(case["nodes"], price_of, tokens)
    unpriced = [n["index"] for n in case["nodes"] if n["index"] not in price_of]
    fully_priced = not unpriceable and all(n["index"] in price_of for n in case["nodes"])
    # A case is shape-pure when every node it contains is a prefill action; only then
    # does one header shape describe the whole execution.
    prefill_pure = not any("-decode" in m["id"] for m in manifest["nodes"])
    measured_wall = statistics.median(float(run["wall_ns"][r]) for r in replicas)
    wall_cv = spread([float(run["wall_ns"][r]) for r in replicas])
    measured_total = sum(reading(n["index"]) for n in case["nodes"])
    copy_ns = sum(reading(i) for i in unpriced
                  if run["nodes"][i]["op"] == "copy")
    return {
        "case": case["header"]["case"],
        "tokens": tokens,
        "replica_used": replicas,
        "wall_cv": wall_cv,
        # gate 3: a case whose own replicas disagree by more than 5% is not a
        # reading and is kept out of the accuracy buckets.
        "unstable": wall_cv > 0.05,
        "prefill_pure": prefill_pure,
        "units": units,
        "unpriceable": unpriceable,
        "measured_wall_ns": measured_wall,
        "predicted_makespan_ns": predicted,
        "makespan_ratio": measured_wall / predicted if predicted else None,
        "priced_action_ns": sum(u["measured_ns"] for u in units.values()),
        "measured_action_ns_total": measured_total,
        "unpriced_nodes": len(unpriced),
        "copy_charged": len(copy_charged),
        "copy_unpriced": dict(sorted(copy_unpriced.items())),
        "fully_priced": fully_priced,
        "measured_copy_ns": copy_ns,
        # The edge-tier side of the reading.  `evicted_bytes` is time the copy engine spent
        # making room, so a plan that does not fit is visible here rather than being folded
        # silently into a longer copy.  `uma_alias_nodes` counts copies that declared bytes
        # and moved none -- on a shared address space the transfer does not exist.
        "tiers": run.get("tiers"),
        "evicted_bytes": sum(int(n.get("evicted_bytes", 0)) for n in run.get("nodes", [])),
        "uma_alias_nodes": sum(1 for n in run.get("nodes", [])
                               if n.get("op") == "copy" and n.get("bytes", 0) > 0
                               and n.get("moved", n.get("bytes", 0)) == 0),
    }


def summarise(rows: list[dict], control_prefix: str = "dense-compute-") -> dict:
    out = {"cases": len(rows), "units": {}, "makespan": {}, "unpriceable": {}}
    reasons: dict[str, int] = {}
    for row in rows:
        for unit, reason in row.get("unpriceable", {}).items():
            head, _, rest = reason.partition(":")
            key = f"{unit}: {head}"
            # `no_price_at:gpu/decode@1t/64h` is a depth the *accelerator* table lacks, not
            # a depth the campaign's own window lacks; the reader separates the two axes
            # and the report counts the device axis as the declared coverage gap.
            if head == "no_price_at" and "/" in rest.split("@", 1)[0]:
                key += " on " + rest.split("@", 1)[0].split("/", 1)[0]
            reasons[key] = reasons.get(key, 0) + 1
    out["unpriceable"] = dict(sorted(reasons.items()))
    # gate 3 first: an unstable case is not a reading, so it never enters A1 either.
    out["unstable_cases"] = sum(1 for r in rows if r.get("unstable"))
    steady = [r for r in rows if not r.get("unstable")]
    buckets: dict[str, list[float]] = {}
    by_phase: dict[str, list[float]] = {}
    for row in steady:
        for slot in row["units"].values():
            buckets.setdefault(slot["unit"], []).append(slot["ratio"])
            by_phase.setdefault(f"{slot['unit']}:{slot['phase']}", []).append(slot["ratio"])
    for unit, values in sorted(buckets.items()):
        out["units"][unit] = {
            "n": len(values),
            "median": statistics.median(values),
            "p10": statistics.quantiles(values, n=10)[0] if len(values) > 9 else min(values),
            "p90": statistics.quantiles(values, n=10)[-1] if len(values) > 9 else max(values),
            "min": min(values), "max": max(values),
        }
    out["units_by_phase"] = {
        name: {"n": len(values), "median": statistics.median(values),
               "p10": statistics.quantiles(values, n=10)[0] if len(values) > 9 else min(values),
               "p90": statistics.quantiles(values, n=10)[-1] if len(values) > 9 else max(values),
               "min": min(values), "max": max(values)}
        for name, values in sorted(by_phase.items())}
    def bucket(values: list[float]) -> dict:
        if not values:
            return {"n": 0}
        return {"n": len(values), "median": statistics.median(values),
                "p10": statistics.quantiles(values, n=10)[0] if len(values) > 9 else min(values),
                "p90": statistics.quantiles(values, n=10)[-1] if len(values) > 9 else max(values),
                "min": min(values), "max": max(values)}

    # A2 is an accuracy statement only where every node of the case carries a price
    # AND one declared shape describes the whole case; elsewhere the gap is a coverage
    # fact, not a model error, and it is reported separately rather than pooled.
    pure = [r for r in steady if r.get("prefill_pure")]
    out["makespan"] = bucket([r["makespan_ratio"] for r in pure if r["makespan_ratio"]])
    out["makespan_fully_priced"] = bucket(
        [r["makespan_ratio"] for r in pure if r["makespan_ratio"] and r.get("fully_priced")])
    # The accuracy set of this campaign: a steady case whose every action carries a frozen
    # price at its *own* shape.  It is the only bucket a pd case can enter, and only when
    # the decode round's visible history is one the table prices.
    resolved = [r for r in steady if not r.get("unpriceable")]
    out["makespan_resolved"] = bucket([r["makespan_ratio"] for r in resolved
                                       if r["makespan_ratio"]])
    out["resolved_cases"] = len(resolved)
    out["makespan_shape_mixed"] = bucket(
        [r["makespan_ratio"] for r in steady if not r.get("prefill_pure") and r["makespan_ratio"]])
    out["shape_pure_cases"] = len(pure)
    out["fully_priced_cases"] = sum(1 for r in steady if r.get("fully_priced"))
    out["copy_ns_median"] = (statistics.median([r["measured_copy_ns"] for r in rows])
                             if rows else None)
    # Tranche-level state gate.  Gate 3 sees replica disagreement *inside* a case; it cannot
    # see a whole tranche running on a different observation band, which is a real failure
    # mode here (one sweep this session ran every CPU action at 0.42 of its price while
    # every replica agreed to <1%).  `dense-compute` is the control: pure CPU, no copies,
    # no GPU, three priced actions per case.  Its median A1 is the tranche's state reading;
    # outside the declared band the tranche is void, not explained.
    control = [u["ratio"] for r in steady if r["case"].startswith(control_prefix)
               for u in r["units"].values() if u["phase"] == "prefill"]
    out["state"] = {
        "band": list(STATE_BAND),
        "control_family": control_prefix,
        "n": len(control),
        "median": statistics.median(control) if control else None,
        "verdict": ("no_control_case" if not control else
                    "in_band" if STATE_BAND[0] <= statistics.median(control) <= STATE_BAND[1]
                    else "off_band:tranche_void"),
    }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen3-0.6b")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--families", choices=("all", "dense", "moe"), default="dense",
                        help="a moe-family topology needs router/expert/combine nodes, which a "
                             "dense model cannot execute and a dense table cannot price")
    parser.add_argument("--analyse-only", action="store_true")
    parser.add_argument("--cases-dir", type=Path, default=None,
                        help="where instantiate.py wrote this model's cases; defaults to "
                             "cases/<model> (the t=64 tranche) so other shapes can coexist")
    parser.add_argument("--out", default=None)
    parser.add_argument("--gguf", type=Path, default=None,
                        help="the model's own gguf file.  The executor then runs the file's "
                             "real weights -- every layer's own bytes when the case names its "
                             "layers (`instantiate.py --per-layer-weights`), layer 0's "
                             "otherwise -- instead of a deterministic pattern of the same "
                             "shape and dtype, and aborts if the file and the case disagree")
    parser.add_argument("--ssd-dir", default=None,
                        help="the third tier's file, for cases that declare ssd_bytes; one "
                             "file is reused and resized to each case's own budget")
    parser.add_argument("--copy-prices", type=Path, default=None,
                        help="the 4b copy lookup (`v4-transfer-lookup.json`).  Without it a "
                             "copy node is charged zero, exactly as every published A1 was "
                             "computed; with it, a copy whose (direction, bytes) has no "
                             "stable measured point is named unpriceable, never estimated")
    parser.add_argument("--prices", nargs="*", default=[],
                        help="extra measured price files (list or {\"prices\": [...]}) keyed by "
                             "kind/phase/tokens/visible_history; the frozen export is never "
                             "edited")
    parser.add_argument("--gpu-node-overhead-ns", type=float, default=0.0,
                        help="per plan node device-dispatch cost added to every non-CPU unit "
                             "(PREREG-gpu-prices.md 4g).  Measured on a node whose compute is "
                             "negligible, not fitted to A1; 0 reproduces the published runs")
    parser.add_argument("--control-prefix", default="dense-compute-",
                        help="the case name prefix whose pure-CPU prefill units are the "
                             "tranche's state reading.  The 800-policy corpus carries one "
                             "control family; a whole-model tranche carries its all-CPU arm, "
                             "which is the same control on a different graph")
    args = parser.parse_args(argv)

    case_dir = args.cases_dir if args.cases_dir else (HERE / "cases" / args.model)
    if case_dir.name != args.model:
        case_dir = case_dir / args.model
    manifest_path = case_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"run instantiate.py --model {args.model} --tokens {args.tokens} first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prices = table_for(args.model, [Path(p) for p in args.prices])
    copy_prices = inst.load_copy_prices(args.copy_prices) if args.copy_prices else None

    names = sorted(manifest["cases"])

    def has_moe(name: str) -> bool:
        return any(n["op"] in ("router", "expert", "combine")
                   for n in manifest["cases"][name]["nodes"])

    if args.families != "all":
        names = [n for n in names if has_moe(n) == (args.families == "moe")]
    # A model whose table prices no `moe_ffn` cannot execute a MoE topology: the executor
    # has no router weight to allocate ("weight role not allocated: router") and there is no
    # price for the action.  Asking for `all` on a dense model is therefore not something to
    # discover case by case -- the families that model cannot run are excluded up front and
    # counted in the output, so the corpus's size and the model's capability stay separate
    # numbers instead of turning into 125 error rows.
    model_is_moe = any(key[1] == "moe_ffn" for key in prices)
    not_applicable = 0
    if not model_is_moe:
        keep = [n for n in names if not has_moe(n)]
        not_applicable = len(names) - len(keep)
        names = keep
    if args.only:
        names = [n for n in names if any(n.startswith(p) for p in args.only)]
    if args.limit:
        names = names[: args.limit]

    out_path = Path(args.out) if args.out else (RESULTS / f"a1a2-{args.model}-t{args.tokens}.json")
    # Replica 0 pays the CUDA module load (see `analyse`), so it is a warmup and never a
    # reading.  A one-replica run of a corpus that reaches the accelerator therefore has
    # no reading left and measures every non-CPU unit through the load: the control family
    # of the MoE growth corpus read 2.94 this way with single prefill units 290x their
    # price, and 1.16 when the same corpus was re-run at `--replicas 4`.  A corpus with no
    # accelerator node has nothing to warm up, so that case is allowed.
    if not args.analyse_only and args.replicas < 2 and any(
            node.get("machine", "cpu") != "cpu"
            for name in names for node in manifest["cases"][name]["nodes"]):
        raise SystemExit(
            f"--replicas {args.replicas}: replica 0 pays the CUDA module load and is the "
            f"warmup, so a one-replica run of a case with accelerator nodes leaves no "
            f"reading; pass --replicas 2 at least (the published protocol uses 4)")
    if args.analyse_only:
        document = json.loads(out_path.read_text(encoding="utf-8"))
        rows = document["rows"]
    else:
        if not RUNNER.is_file():
            raise SystemExit(f"no executor at {RUNNER}\n"
                             f"build it with `python profiling/native/build_ir_runner.py --fetch`\n"
                             f"(or --llama PATH) and set IR_RUNNER to the path it prints")
        rows = []
        began = time.time()

        def flush() -> None:
            """Persist what has been measured so far.

            A long tranche is meant to run unattended: one bad case must not cost the
            cases that already ran, so the document is written every 25 cases and once
            more from the failure path.
            """
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(
                {"model": args.model, "tokens": args.tokens, "replicas": args.replicas,
                 "runner": str(RUNNER), "gguf": str(args.gguf) if args.gguf else None,
                 "weights": "real" if args.gguf else "pattern",
                 "rows": rows}, indent=1), encoding="utf-8")

        for i, name in enumerate(names):
            path = case_dir / f"{name}.txt"
            try:
                case = read_case_file(path)
                command = [str(RUNNER), "--case", str(path), "--replicas", str(args.replicas)]
                if args.gguf:
                    command += ["--gguf", str(args.gguf)]
                if args.ssd_dir:
                    command += ["--ssd-dir", str(args.ssd_dir)]
                done = subprocess.run(command, capture_output=True, text=True,
                                      env=environment())
                if done.returncode != 0:
                    # A plan that declares `decline` and does not fit is *supposed* to fail.
                    # That is the arm's whole content, so it is a result, not a defect: the
                    # prediction "this plan is infeasible" and the execution agreeing is
                    # exactly the comparison this case exists to make.
                    #
                    # The reason is read from the nodes, not from stderr: the executor marks
                    # the failing node and finishes the graph, so it exits 1 with an empty
                    # stderr and a text match on stderr never fired.  The check is now the
                    # whole set of failed nodes, which is also stricter -- one node failing
                    # for some other reason does not count as "infeasible as declared".
                    reasons = _failure_reasons(done.stdout)
                    if (case["header"].get("reclaim") == "decline" and reasons
                            and all("capacity exceeded" in r for r in reasons)):
                        rows.append({"case": name, "infeasible_as_declared":
                                     f"{len(reasons)} node(s): {sorted(set(reasons))[0][:180]}"})
                        print(f"  {name}: infeasible as declared ({len(reasons)} nodes)",
                              flush=True)
                        flush()
                        continue
                    detail = str(sorted(set(reasons))[:2]) if reasons else done.stderr.strip()[-400:]
                    rows.append({"case": name, "error": detail})
                    print(f"  {name}: FAILED {detail[:120]}", flush=True)
                    flush()
                    continue
                run = json.loads(done.stdout[done.stdout.index("{"):])
            except (OSError, ValueError) as exc:
                rows.append({"case": name, "error": f"{type(exc).__name__}: {exc}"})
                print(f"  {name}: FAILED {exc}", flush=True)
                flush()
                continue
            # gate 4: the instrument must echo the placement it says it used.
            want_mask = manifest.get("cpu_mask", "2,4")
            echoed = ",".join(str(x) for x in run.get("cpu_mask", []))
            if echoed != want_mask or run.get("thread_placement") != "explicit_strict_cpu_mask":
                rows.append({"case": name, "declared_mismatch":
                             f"cpu_mask echo {echoed!r} placement {run.get('thread_placement')!r}"})
                print(f"  {name}: declared_mismatch", flush=True)
                continue
            # gate 6: the case file the executor read must be the one the manifest prices.
            # A stale manifest would otherwise price action A with action B's shape and no
            # other gate would notice.  Checked on the whole (index, op, machine) line.
            m_nodes = manifest["cases"][name]["nodes"]
            wrote = [(n["index"], n["op"], n["machine"]) for n in case["nodes"]]
            priced = [(n["index"], n["op"], n["machine"]) for n in m_nodes]
            if wrote != priced:
                first = next((i for i, (a, b) in enumerate(zip(wrote, priced)) if a != b),
                             min(len(wrote), len(priced)))
                rows.append({"case": name, "manifest_mismatch":
                             f"node {first}: file {wrote[first:first + 1]} vs manifest "
                             f"{priced[first:first + 1]} ({len(wrote)} vs {len(priced)} nodes)"})
                print(f"  {name}: manifest_mismatch", flush=True)
                continue
            # gate 5: the shape every node says it ran must be the shape the driver prices
            # for that node.  Per node, not as a set: a pd case contains both shapes, so a
            # decode round running at the prefill shape would still leave the two sets
            # equal and the gate silent.
            want_shapes = {n["index"]: (n.get("tokens"), n.get("history")) for n in m_nodes}
            got_shapes = {n["index"]: (n.get("tokens"), n.get("history"))
                          for n in run.get("nodes", [])}
            built_shapes = {(s["tokens"], s["history"]) for s in run.get("shapes", [])}
            wrong = sorted(i for i in want_shapes
                           if got_shapes.get(i) != want_shapes[i])
            if wrong or not built_shapes <= set(want_shapes.values()):
                rows.append({"case": name, "shape_mismatch":
                             f"nodes {wrong[:5]} ran at "
                             f"{ {i: got_shapes.get(i) for i in wrong[:5]} }, built "
                             f"{sorted(built_shapes)}, priced {sorted(set(want_shapes.values()))}"})
                print(f"  {name}: shape_mismatch", flush=True)
                continue
            # gate 5b: the executor must report every node successful.  A node that failed
            # to execute has no duration, and a zero would read as a very fast action.
            if not run.get("all_ok", False):
                bad = [n["index"] for n in run.get("nodes", []) if not n.get("ok", True)]
                why = sorted({str(n.get("status", "")) for n in run.get("nodes", [])
                              if not n.get("ok", True)})[:2]
                rows.append({"case": name,
                             "node_failed": f"all_ok=false, nodes {bad[:8]}, why {why}"})
                print(f"  {name}: node_failed {bad[:5]} {why[:1]}", flush=True)
                continue
            rows.append(analyse(case, run, prices, args.tokens, manifest["cases"][name],
                                copy_prices, args.gpu_node_overhead_ns))
            if (i + 1) % 25 == 0:
                flush()
                print(f"  {i + 1}/{len(names)}  {time.time() - began:.0f}s", flush=True)
        flush()
        document = {"model": args.model, "tokens": args.tokens, "replicas": args.replicas,
                    "runner": str(RUNNER), "gguf": str(args.gguf) if args.gguf else None,
                    "weights": "real" if args.gguf else "pattern", "rows": rows}

    # A row is a reading only if it got far enough to analyse: every gate rejection carries
    # its own reason key instead, and pooling one into A1/A2 would count a rejected case as
    # a measurement.
    good = [r for r in rows if "replica_used" in r]
    failed = [r for r in rows if "replica_used" not in r]
    print(f"\n{len(good)} cases analysed, {len(failed)} rejected  -> {out_path}")
    if not_applicable:
        print(f"  {not_applicable} moe-family topologies not applicable to a dense model "
              f"(no router weights, no moe_ffn price) -- not counted as failures")
    if failed:
        for r in failed[:5]:
            reason = {k: v for k, v in r.items() if k != "case"}
            print(f"  REJECTED {r['case']}: {reason}")
    summary = summarise(good, args.control_prefix)
    print(f"  gate 3: {summary['unstable_cases']} of {len(good)} cases unstable "
          f"(replica CV > 5%), excluded from A1 and A2")
    state = summary["state"]
    if state["median"] is None:
        print(f"  state gate: {state['verdict']} (no {state['control_family']} cases in this "
              f"selection)")
    else:
        print(f"  state gate: control {state['control_family']} median A1 "
              f"{state['median']:.3f} over n={state['n']}, declared band "
              f"[{state['band'][0]:.2f}, {state['band'][1]:.2f}] -> {state['verdict']}")
    print("\nA1  measured / predicted, per priced action unit")
    for unit, row in summary["units"].items():
        print(f"  {unit:12s} n={row['n']:4d}  median {row['median']:.3f}  "
              f"p10 {row['p10']:.3f}  p90 {row['p90']:.3f}  [{row['min']:.3f}, {row['max']:.3f}]")
    print("  by phase")
    for name, row in summary["units_by_phase"].items():
        print(f"    {name:20s} n={row['n']:4d}  median {row['median']:.3f}  "
              f"p10 {row['p10']:.3f}  p90 {row['p90']:.3f}")
    print(f"  A2 uses the {summary['resolved_cases']} steady cases whose every action carries "
          f"a frozen price at its own shape")
    for label, key in (("prefill-only cases, LOWER BOUND (unpriced nodes cost real time)",
                        "makespan"),
                       ("every action priced at its own shape (still a lower bound: copies)",
                        "makespan_resolved"),
                       ("fully priced (unreachable: copies carry no price)", "makespan_fully_priced"),
                       ("mixed-shape cases, NOT an accuracy claim", "makespan_shape_mixed")):
        mk = summary[key]
        if not mk.get("n"):
            print(f"  A2  {label}: none")
            continue
        print(f"  A2  {label}: measured / predicted makespan  n={mk['n']}  "
              f"median {mk['median']:.3f}  p10 {mk['p10']:.3f}  p90 {mk['p90']:.3f}")
    if summary["unpriceable"]:
        print("  unpriceable actions: " +
              ", ".join(f"{k} x{v}" for k, v in summary["unpriceable"].items()))
    if summary["copy_ns_median"] is not None:
        print(f"  measured copy time, median {summary['copy_ns_median'] / 1e6:.3f} ms "
              f"(copies carry no table price)")
    document["summary"] = summary
    document["not_applicable_moe_cases"] = not_applicable
    out_path.write_text(json.dumps(document, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
