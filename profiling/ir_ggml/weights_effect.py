#!/usr/bin/env python3
"""Does running the model's own weights change the time?  Alternate the two and look.

Every published price was measured on a deterministic byte pattern of the model's shape and
dtype.  That is sound only if the values do not reach the timing, which is a claim about the
kernels, not an opinion: a matmul's cost is a function of its shape and type, and a quantized
block's cost does not depend on what is in it.  This script is the check.  It is *not* a
campaign: the criterion is written here, before the first run, and nothing is tuned after.

Method.  The session level moves between runs (see `20261109-fit-anchors`), so a block of
synthetic runs followed by a block of real runs would confound the weights with the level.
Alternate instead: synth, real, synth, real, ... in one session, `--replicas 5` each, and
compare the two groups' medians per node.  The alternation is the same instrument the
prefix-pair measurement uses, for the same reason.

Criterion.  For every node of every case, |median_real / median_synth - 1| <= 0.10.  A node
over that is a finding, not noise to be averaged away: it would mean the published table was
measured on something the real model does not do.

    python weights_effect.py --runner <ggml-ir-runner.exe> --gguf-model NAME=PATH [--pairs 3]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

TOLERANCE = 0.10   # written before the first run; do not change after seeing data
PAIRS = 3
REPLICAS = 5

HERE = Path(__file__).resolve().parent
CASES = {
    "dense": "qwen3-4b/dense-compute-000-demand.txt",
    "moe": "granite-3.0-3b-a800m/attention-compute-000-demand.txt",
}


def run(runner: Path, case: Path, gguf: Path | None, env: dict) -> dict:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out.json"
        cmd = [str(runner), "--case", str(case), "--replicas", str(REPLICAS), "--out", str(out)]
        if gguf:
            cmd += ["--gguf", str(gguf)]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise SystemExit(f"{case.name}: exit {proc.returncode}\n{proc.stderr[-800:]}")
        return json.loads(out.read_text(encoding="utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runner", type=Path, required=True)
    ap.add_argument("--gguf-model", action="append", default=[],
                    help="KIND=PATH, where KIND is a key of CASES (dense, moe)")
    ap.add_argument("--pairs", type=int, default=PAIRS)
    ap.add_argument("--cases", type=Path, default=HERE / "cases")
    args = ap.parse_args(argv)

    env = dict(os.environ)
    gguf_of = dict(x.split("=", 1) for x in args.gguf_model)
    worst = []
    for kind, rel in CASES.items():
        if kind not in gguf_of:
            print(f"{kind}: no --gguf-model given, skipped")
            continue
        case = args.cases / rel
        if not case.exists():
            print(f"{kind}: {case} missing, skipped")
            continue
        synth_runs, real_runs = [], []
        for _ in range(args.pairs):
            synth_runs.append(run(args.runner, case, None, env))
            real_runs.append(run(args.runner, case, Path(gguf_of[kind]), env))
        if not all(r["all_ok"] for r in synth_runs + real_runs):
            print(f"{kind}: a run reported all_ok=false, stopping")
            continue
        w = real_runs[0]["weights"]
        print(f"\n== {kind}  ({case.name})")
        print(f"   weights: real={w['real']} roles={w['roles_real']} bytes={w['bytes_real']}"
              f" experts {w['experts_loaded']}/{w['experts_in_file']}"
              + (f"  pattern: {w['pattern_roles']}" if w["pattern_roles"] else ""))
        print(f"   {args.pairs} alternating pairs x {REPLICAS} replicas")
        print(f"   {'op':16s}{'synth ms':>11s}{'real ms':>11s}{'ratio':>9s}{'verdict':>10s}")
        for i, node in enumerate(synth_runs[0]["nodes"]):
            s = statistics.median(d for run_ in synth_runs
                                  for d in run_["nodes"][i]["dur_ns"]) / 1e6
            r = statistics.median(d for run_ in real_runs
                                  for d in run_["nodes"][i]["dur_ns"]) / 1e6
            if s <= 0:
                continue
            ratio = r / s
            bad = abs(ratio - 1) > TOLERANCE
            print(f"   {node['op']:16s}{s:11.3f}{r:11.3f}{ratio:9.3f}"
                  f"{'  OVER' if bad else '  ok':>10s}")
            worst.append((abs(ratio - 1), kind, node["op"], ratio))
    if not worst:
        return 1
    worst.sort(reverse=True)
    print(f"\nworst |ratio-1| {worst[0][0]*100:.1f}% at {worst[0][1]}/{worst[0][2]}"
          f" (ratio {worst[0][3]:.3f}), tolerance {TOLERANCE*100:.0f}%")
    print("verdict: " + ("PASS -- the weights do not reach the timing"
                         if worst[0][0] <= TOLERANCE else "FAIL -- see the rows marked OVER"))
    return 0 if worst[0][0] <= TOLERANCE else 1


if __name__ == "__main__":
    sys.exit(main())
