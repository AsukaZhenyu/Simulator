#!/usr/bin/env python3
"""The model column: a whole real model, measured through our own lowering and ggml alone.

What this replaces
------------------
The published model column timed *another engine's* request -- a language-model runtime
loading the gguf and reporting per-forward windows -- and then asked the simulator to
predict that engine.  Fitting one runtime's scheduler is not the claim the simulator
makes, and it made the project's measurement depend on that runtime's headers and
release cadence.  This column removes the engine from the loop: the plan is our IR,
lowered to ggml by `profiling/native/ir_runner.cpp`, executed against the model file's own
per-layer weight bytes, on the same machine.

What it measures
----------------
For every placement `g` (how many of the model's `L` blocks sit on the accelerator):

  * the whole stack, prefill at the declared token count and `--rounds` decode rounds;
  * A1 -- each priced action unit's measured time over its frozen table price, split by
    phase.  This is the accuracy claim: does the price of *this* action, at this shape,
    on this machine, describe what the action costs;
  * A2 -- the case's measured wall time over the sum of its actions' prices (the copies
    carry no table price and are declared, not silently dropped).

The control family is the all-CPU placement: pure CPU, no copies, no accelerator.  Its
median A1 is the tranche's state reading, exactly as `dense-compute-` is for the 800-policy
corpus -- a tranche outside the declared band is void, not explained.

Pre-registered acceptance
-------------------------
Declared before the run (`PREREG-model-column.md`):

  * state gate in band  -> the tranche is readable at all;
  * |1 - median A1| <= 10% for prefill attention, dense feed-forward and projection;
  * |1 - median A1| <= 10% for the same three in decode at the pinned history;
  * |1 - A2| <= 10% on the all-CPU case (the only one whose every action is priced at
    its own shape; a case with copies is a lower bound because copies carry no price).

Nothing here is tuned after a reading.  A miss is reported as a miss and the residual is
named by the unit and phase it lives in.

Usage::

    python -m profiling.ir_ggml.model_column --model qwen3-0.6b \
        --gguf ../llama.cpp-implement/validation-models/Qwen3-0.6B-Q4_K_M.gguf \
        --out build/model-ggml/column-qwen3 [--rounds 2] [--replicas 4]

    # analyse a tranche that already ran (no executor, no GPU)
    python -m profiling.ir_ggml.model_column --model qwen3-0.6b --out DIR --analyse-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import corpus_v4 as v4          # noqa: E402  (same directory)
import instantiate as inst      # noqa: E402
import run as runner            # noqa: E402

# Declared before the run; see the module docstring.  One band, applied to every claim
# this column makes -- a separate looser band for decode would be a tuning knob.
BAND = 0.10
# The frozen table prices decode at visible_history 0/128/512/1024.  A chain history
# (prefill length + round) lands between two grid points and would be `no_price_at`,
# which is a coverage gap reported as an accuracy residual if it is not pinned: the
# column therefore declares the grid point it measures at.
DECODE_HISTORY = 128


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def placements(layers: int) -> list[int]:
    """Every block off the accelerator through every block on it."""
    points = sorted({0, layers // 4, layers // 2, 3 * layers // 4, layers})
    return [g for g in points if 0 <= g <= layers]


def case_name(layers: int, g: int) -> str:
    return f"whole-d{layers}-g{g}"


def build_cases(spec: dict, rounds: int, out: Path) -> dict[str, str]:
    """Write one `nodes.json` per placement; return {name: graph size}."""
    layers = spec["layers"]
    moe = bool(spec.get("experts"))
    written = {}
    shutil.rmtree(out, ignore_errors=True)
    for g in placements(layers):
        name = case_name(layers, g)
        nodes = v4.topology(layers=layers, place=v4.residency(g, "first", layers),
                            moe=moe, storage="demand", rounds=rounds,
                            phases=("prefill", "decode"))
        directory = out / name
        directory.mkdir(parents=True)
        (directory / "nodes.json").write_text(json.dumps(nodes, indent=1), encoding="utf-8")
        written[name] = len(nodes)
    return written


def run_column(args) -> Path:
    spec = inst.load_model(args.model)
    layers = spec["layers"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = out / "cases-src"
    built = build_cases(spec, args.rounds, source)
    print(f"{args.model}: {len(built)} placements, {layers} blocks, "
          f"prefill {args.tokens} tokens + {args.rounds} decode rounds at h={args.decode_history}")

    cases = out / "cases"
    shutil.rmtree(cases, ignore_errors=True)
    (out / "column-input.json").write_text(json.dumps(
        {"model": args.model, "layers": layers, "tokens": args.tokens,
         "rounds": args.rounds, "decode_history": args.decode_history,
         "replicas": args.replicas, "gguf": str(args.gguf)}, indent=1), encoding="utf-8")
    code = inst.main(["--model", args.model, "--tokens", str(args.tokens),
                      "--decode-history", str(args.decode_history),
                      "--cases", str(source), "--out", str(cases),
                      "--per-layer-weights", "--gguf", str(args.gguf)])
    if code != 0:
        raise SystemExit(f"instantiate failed with {code}")

    document = out / "a1a2.json"
    families = "moe" if spec.get("experts") else "dense"
    code = runner.main(["--model", args.model, "--tokens", str(args.tokens),
                        "--replicas", str(args.replicas),
                        "--families", families,
                        "--cases-dir", str(cases / args.model),
                        "--gguf", str(args.gguf),
                        "--out", str(document),
                        "--control-prefix", case_name(layers, 0)])
    if code != 0:
        raise SystemExit(f"run failed with {code}")
    return document


def by_unit_nodes(good: list[dict]) -> dict:
    """A1 grouped by (unit, phase, how many plan nodes the unit is made of).

    The executor times one ggml graph per plan node, so a priced unit that the lowering
    splits into four nodes pays the graph's fixed cost four times.  On a prefill unit
    (milliseconds) that is invisible; on a decode unit (a fifth of a millisecond) it is
    the difference between one ratio and another, which is why the split is reported
    rather than pooled away.
    """
    buckets: dict[str, list[float]] = {}
    for row in good:
        for slot in row["units"].values():
            key = f"{slot['unit']}:{slot['phase']}:{len(slot['nodes'])}n"
            buckets.setdefault(key, []).append(slot["ratio"])
    return {key: {"n": len(v), "median": statistics.median(v)}
            for key, v in sorted(buckets.items())}


def claim(summary: dict) -> dict:
    """The pre-registered statement, one entry per declared cell."""
    cells = {}
    for unit, row in summary.get("units_by_phase", {}).items():
        error = abs(1.0 - row["median"])
        cells[unit] = {"n": row["n"], "median": row["median"], "error": error,
                       "within_band": error <= BAND}
    a2 = summary.get("makespan_resolved") or {}
    if a2.get("n"):
        cells["makespan"] = {"n": a2["n"], "median": a2["median"],
                             "error": abs(1.0 - a2["median"]),
                             "within_band": abs(1.0 - a2["median"]) <= BAND}
    state = summary.get("state", {})
    return {"band": BAND, "cells": cells,
            "state": state,
            "verdict": ("no_reading" if not cells else
                        "in_band" if state.get("verdict") == "in_band"
                        and all(c["within_band"] for c in cells.values())
                        else "off_band")}


def analyse(args) -> int:
    out = Path(args.out)
    document = out / "a1a2.json"
    if not document.is_file():
        raise SystemExit(f"{document} does not exist; run the column first")
    run_doc = json.loads(document.read_text(encoding="utf-8"))
    good = [r for r in run_doc["rows"] if "replica_used" in r]
    supplied = out / "column-input.json"
    declared = json.loads(supplied.read_text(encoding="utf-8")) if supplied.is_file() else {}
    summary = runner.summarise(good, case_name(inst.load_model(args.model)["layers"], 0))
    report = {
        "schema": "ir-ggml-model-column-1",
        "model": run_doc["model"],
        "gguf": ({"path": str(args.gguf), "sha256": sha256(Path(args.gguf)),
                  "bytes": Path(args.gguf).stat().st_size} if args.gguf else None),
        "runner": run_doc["runner"],
        "weights": run_doc.get("weights"),
        "tokens": run_doc["tokens"],
        "replicas": run_doc["replicas"],
        "decode_history": declared.get("decode_history", DECODE_HISTORY),
        "rounds": declared.get("rounds", args.rounds),
        "cases": {r["case"]: {k: r.get(k) for k in
                              ("measured_wall_ns", "predicted_makespan_ns", "makespan_ratio",
                               "fully_priced", "prefill_pure", "unstable", "replica_cv")}
                  for r in good},
        "a1": summary.get("units_by_phase", {}),
        "a2": {k: summary.get(k) for k in
               ("makespan", "makespan_resolved", "makespan_shape_mixed", "resolved_cases")},
        "unpriceable": summary.get("unpriceable", {}),
        "by_unit_nodes": by_unit_nodes(good),
        "claim": claim(summary),
    }
    (out / "column.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    lines = [f"# model column -- {report['model']}",
             "",
             f"weights: {report['weights']}  ({report['gguf']['sha256'][:16] if report['gguf'] else 'pattern'})",
             f"shape: prefill {report['tokens']} tokens, {report['rounds']} decode rounds "
             f"at visible history {report['decode_history']}, {report['replicas']} replicas",
             f"state gate: {report['claim']['state'].get('verdict')} "
             f"(control median {report['claim']['state'].get('median')})",
             "",
             "| unit | n | measured/predicted | error | band |",
             "|---|---|---|---|---|"]
    for unit, cell in sorted(report["claim"]["cells"].items()):
        lines.append(f"| {unit} | {cell['n']} | {cell['median']:.3f} | "
                     f"{cell['error']:.1%} | {'ok' if cell['within_band'] else 'MISS'} |")
    lines += ["", "A1 by the node count the lowering splits each unit into:", "",
              "| unit:phase | nodes | n | measured/predicted |", "|---|---|---|---|"]
    for key, cell in report["by_unit_nodes"].items():
        unit, phase, nodes = key.rsplit(":", 2)
        lines.append(f"| {unit}:{phase} | {nodes[0]} | {cell['n']} | {cell['median']:.3f} |")
    lines += ["", f"verdict: **{report['claim']['verdict']}** "
                  f"(declared band +/-{BAND:.0%})", ""]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("\n".join(lines))
    return 0 if report["claim"]["verdict"] == "in_band" else (1 if args.strict else 0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen3-0.6b")
    parser.add_argument("--gguf", type=Path, default=None,
                        help="the model's file; required unless --analyse-only")
    parser.add_argument("--out", type=Path, default=HERE / "results" / "model-column")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--decode-history", type=int, default=DECODE_HISTORY)
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--analyse-only", action="store_true")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 when the reading is outside the declared band.  A "
                             "campaign must not: a miss is the result it was built to "
                             "report, and stopping there would hide the cells that passed")
    args = parser.parse_args(argv)

    if args.decode_history != DECODE_HISTORY:
        raise SystemExit(f"--decode-history {args.decode_history}: the column's claim is "
                         f"registered at the frozen grid point h={DECODE_HISTORY}; a different "
                         f"history is a different, undeclared cell")
    if args.analyse_only:
        return analyse(args)
    if args.gguf is None:
        raise SystemExit("--gguf is required: the column measures the model's own bytes")
    run_column(args)
    return analyse(args)


if __name__ == "__main__":
    raise SystemExit(main())
