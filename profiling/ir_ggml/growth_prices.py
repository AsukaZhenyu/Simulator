#!/usr/bin/env python3
"""Prices at the depths a growing decode chain actually visits.

Why this exists
---------------
`instantiate.py` already gives multi-round decode its physical history: with
``--decode-history chain`` (the default) round ``r`` of a request that prefilled
``T`` tokens is priced and executed at ``visible_history = T + r``.  That is a
chain of one-token steps whose KV grows by one token per step, and it is what a
real generator does.  What stopped the campaigns from running it is the table,
not the code: the frozen export prices decode at ``visible_history`` 0/128/512/
1024 and nowhere else, so the second round of a chain filed ``no_price_at`` and
was reported as unpriced rather than priced at a guessed depth.

This module closes that coverage gap the only way a price may be closed -- by
measuring the missing depths on the same devices and the same protocol as the
existing table, and handing the result to `run.run` as an extension table.  It
never interpolates and never borrows a neighbouring depth.  A cell whose repeats
disagree with themselves is refused, or -- for a CPU cell only, and only on an
explicit request -- carried with the quality flag the CPU protocol itself uses
for that case, exactly as `export_dense_layer` does.

The window
----------
A campaign instantiated with ``--tokens T`` and expanded to ``R`` decode rounds
visits depths ``T .. T+R-1``.  ``--rounds`` therefore names both the number of
decode rounds and the size of the window, and the two cannot drift apart.

Devices
-------
``attention`` reads the KV and is the only stage whose work depends on the
depth; ``projection`` and the feed-forward do not, but they are charged at the
node's own ``(tokens, history)`` key like everything else, so they are measured
at the same depths rather than copied from a shallower entry (copies across
depths are exactly the pooling decision `history_invariant_prices` owns, and it
is not this module's to make silently).

Usage::

    python profiling/ir_ggml/growth_prices.py suite --model qwen3-0.6b --tokens 64 --rounds 8 --out /tmp/g
    python profiling/run_dense_layer_measurements.py --suite /tmp/g/cpu-qwen3-0.6b-t64.json ...
    python profiling/run_dense_cuda_measurements.py --suite /tmp/g/cuda-qwen3-0.6b-t64.json ...
    python profiling/ir_ggml/growth_prices.py export --model qwen3-0.6b --tokens 64 --rounds 8 \\
        --cpu-run <cpu results dir> --cuda-run <cuda results dir> --out prices/qwen3-0.6b/growth-t64.json
    python profiling/ir_ggml/growth_prices.py check --model qwen3-0.6b --tokens 64 --rounds 8 \\
        --prices prices/qwen3-0.6b/growth-t64.json --frozen prices/qwen3-0.6b/gpu_costs.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gpu_prices  # noqa: E402  (same directory)
import instantiate as inst  # noqa: E402

# The stages a corpus case can place on either device, in the order they are written.
STAGES = ("projection", "attention", "dense_ffn")
# Each device is held to its own table's acceptance rule, not to the stricter of the two.
# The CPU export treats a cv above 0.1 as a quality flag that only an explicit validation
# export may carry (`export_dense_layer`); the accelerator export refuses the cell.
CPU_MAX_CV = 0.10
GPU_MAX_CV = gpu_prices.MAX_CV
CPU_UNSTABLE_FLAG = "high_observed_variability_cv_gt_0.1"
KV_DTYPE = gpu_prices.KV_DTYPE


def window(tokens: int, rounds: int) -> list[int]:
    """The depths a chain of `rounds` decode steps visits after a `tokens` prefill."""
    if tokens < 1 or rounds < 1:
        raise SystemExit(f"window needs tokens>=1 and rounds>=1, got {tokens}/{rounds}")
    return list(range(tokens, tokens + rounds))


def stage_configs(model: str) -> dict[str, dict]:
    """One entry per stage kind the model prices, at any depth.

    The configuration of a stage does not depend on the depth it is measured at --
    only the KV the attention reads does -- so the shallowest entry of each kind
    is the geometry the window's cells are cloned from.
    """
    entries = sorted(gpu_prices._entries(model),
                     key=lambda e: (e["kind"], e["visible_history"]))
    configs: dict[str, dict] = {}
    for entry in entries:
        configs.setdefault(entry["kind"], entry)
    missing = [kind for kind in STAGES if kind not in configs and kind != "dense_ffn"]
    if missing:
        raise SystemExit(f"{model}: the table prices no {missing}")
    return configs


def frozen_kinds(model: str, device: str) -> set[tuple[str, str]]:
    """The ``(kind, phase)`` pairs the frozen table already prices on `device`.

    A growth tranche extends the depth axis and nothing else.  Measuring a pair the
    frozen table priced nowhere would hand the growth campaign a placement the rest
    of the corpus does not have, and A2 would then be a comparison between two
    different strategy spaces rather than between a prediction and a measurement.
    """
    if device == "cpu":
        document = json.loads((inst.EXPORTS[model] / "costs.json").read_text(encoding="utf-8"))
        return {(entry["kind"], entry["phase"]) for entry in document["entries"]
                if entry.get("device") == "cpu"}
    path = inst.EXPORTS[model] / "gpu_costs.json"
    if not path.exists():
        raise SystemExit(f"{model}: no accelerator table to extend")
    document = json.loads(path.read_text(encoding="utf-8"))
    return {(price["kind"], price["phase"]) for price in document["prices"]}


def _order(kinds) -> list[str]:
    # Materialised first: `in` on a generator consumes it, so an exhausted generator
    # would silently drop every stage after the first.
    wanted = set(kinds)
    return [kind for kind in ("projection", "attention", "dense_ffn", "moe_ffn")
            if kind in wanted]


def prefill_cell(model: str, tokens: int, entry: dict) -> dict:
    """The campaign's own prefill shape, measured in the same session as the window.

    A growth campaign divides a measured makespan by a predicted one, and the
    prediction is a sum over prefill *and* decode actions.  Taking the prefill
    half from a table measured on another day would make A2 a statement about
    two sessions rather than about the model, so the same session prices it.
    """
    return {
        "id": f"growth_{model}_{entry['kind']}_prefill_t{tokens}_h0",
        "kind": entry["kind"], "phase": "prefill", "batch": 1,
        "tokens": tokens, "visible_history": 0,
        "config": dict(entry["physical_signature"]["config"]),
        "source": {"frozen_price": str(inst.EXPORTS[model] / "costs.json"),
                   "frozen_duration_ns": entry["duration_ns"]},
    }


def cpu_suite(model: str, tokens: int, rounds: int, prefill: bool = True) -> dict:
    """The CPU measurement suite for one window, geometry cloned from the frozen export."""
    configs = stage_configs(model)
    priced = frozen_kinds(model, "cpu")
    sparse = any(kind == "moe_ffn" for kind in configs)
    schema = "moe_layer_measurement_suite_v1" if sparse else "dense_layer_measurement_suite_v1"
    kinds = _order(kind for kind in configs if (kind, "decode") in priced)
    cases = []
    if prefill:
        cases += [prefill_cell(model, tokens, configs[kind]) for kind in kinds
                  if (kind, "prefill") in priced]
    for depth in window(tokens, rounds):
        for kind in kinds:
            entry = configs[kind]
            cases.append({
                "id": f"growth_{model}_{kind}_decode_t1_h{depth}",
                "kind": kind, "phase": "decode", "batch": 1,
                "tokens": 1, "visible_history": depth,
                "config": dict(entry["physical_signature"]["config"]),
                "source": {"frozen_price": str(inst.EXPORTS[model] / "costs.json"),
                           "cloned_from_depth": int(entry["visible_history"]),
                           "frozen_duration_ns": entry["duration_ns"]},
            })
    return {"schema_version": schema,
            "comment": f"growing-decode window h={tokens}..{tokens + rounds - 1} for {model}, "
                       f"t=1, with the campaign's own prefill shape at t={tokens}; "
                       f"geometry cloned from the frozen CPU export",
            "cases": cases}


def cuda_suite(model: str, tokens: int, rounds: int, prefill: bool = True) -> dict:
    """The accelerator suite for the same window, under the model's own envelope.

    A dense model's contract cannot express a routed block, so its sparse fields
    are projected away exactly as `gpu_prices.build_suite` does; a sparse model's
    contract declares the block, so its config is carried verbatim.  The two
    schemas are the two the CUDA runner already accepts.
    """
    configs = stage_configs(model)
    priced = frozen_kinds(model, "gpu")
    sparse = any(kind == "moe_ffn" for kind in configs)
    decodable = [kind for kind in configs if (kind, "decode") in priced]
    if sparse:
        kinds = _order(decodable)
        schema = "moe_layer_measurement_suite_v1"
    else:
        kinds = [kind for kind in _order(decodable) if kind != "moe_ffn"]
        schema = "dense_layer_measurement_suite_v1"
    cases, projections = [], {}

    def cell(kind: str, phase: str, cell_tokens: int, depth: int) -> dict:
        entry = configs[kind]
        signature = entry["physical_signature"]
        case_id = (f"growth_{model}_{kind}_decode_t1_h{depth}" if phase == "decode"
                   else f"growth_{model}_{kind}_prefill_t{cell_tokens}_h0")
        if sparse:
            config = dict(signature["config"])
        else:
            config, projection = gpu_prices.dense_config(model, kind, signature)
            if projection["dropped_keys"] or projection["dropped_weight_roles"]:
                projections[case_id] = projection
        return {
            "id": case_id, "kind": kind, "phase": phase, "batch": 1,
            "tokens": cell_tokens, "visible_history": depth, "kv_dtype": KV_DTYPE,
            "config": config,
            "source": {"frozen_price": str(inst.EXPORTS[model] / "costs.json"),
                       "cloned_from_depth": int(entry["visible_history"]),
                       "frozen_duration_ns": entry["duration_ns"]},
        }

    if prefill:
        cases += [cell(kind, "prefill", tokens, 0) for kind in kinds
                  if (kind, "prefill") in priced]
    for depth in window(tokens, rounds):
        cases += [cell(kind, "decode", 1, depth) for kind in kinds]
    document = {"schema_version": schema,
                "comment": f"growing-decode window h={tokens}..{tokens + rounds - 1} for "
                           f"{model}, t=1; cache {KV_DTYPE} as the CPU table declares",
                "cases": cases}
    if projections:
        document["config_projections"] = projections
    return document


def _rows(run_dir: Path) -> list[dict]:
    path = run_dir / "measurements.json"
    if not path.exists():
        raise SystemExit(f"{run_dir}: no measurements.json")
    return json.loads(path.read_text(encoding="utf-8"))


def export(model: str, tokens: int, rounds: int,
           cpu_runs: list[Path], cuda_runs: list[Path],
           allow_unstable: bool = False) -> dict:
    """Turn measured runs into the extension table `run.table_for` merges.

    A row that failed, or whose repeats disagree with themselves beyond its own
    device's rule, is refused with its reason: a depth is either measured or
    absent.  Several runs may be given per device and an earlier run wins a cell
    it already answered, so a cell the first pass refused can be re-measured on
    its own without re-measuring the window.  The frozen export is never edited,
    so the published 800 keep the table they were published with.
    """
    prices, refused, seen = [], [], set()
    for device, run_dirs in (("cpu", cpu_runs), ("gpu", cuda_runs)):
        priced = frozen_kinds(model, device) if run_dirs else set()
        for run_dir in run_dirs:
            for row in _rows(run_dir):
                case = row["case"]
                decode = case.get("phase") == "decode" and int(case["tokens"]) == 1
                prefill = case.get("phase") == "prefill" and int(case["visible_history"]) == 0
                if not (decode or prefill):
                    refused.append({"device": device, "case": case.get("id"),
                                    "reason": "not a one-token decode cell and not the "
                                              "campaign's own prefill shape"})
                    continue
                key = (device, case["kind"], case["phase"], int(case["tokens"]),
                       int(case["visible_history"]))
                if (case["kind"], case["phase"]) not in priced:
                    refused.append({"device": device, "key": str(key),
                                    "reason": f"the frozen table prices no {case['kind']}/"
                                              f"{case['phase']} on {device}; a growth tranche "
                                              f"extends the depth axis only"})
                    continue
                if row.get("status") != "measured":
                    refused.append({"device": device, "key": str(key),
                                    "reason": row.get("error", "not measured")})
                    continue
                # Asked before the quality rule: a cell an earlier run already priced is
                # skipped whatever this row's cv is, and reporting the cv here would name
                # the wrong reason for the very case the earlier-run rule exists for (a
                # cell the first pass refused is re-measured alone, and the first pass's
                # now-superseded row must not read as a second refusal).
                if key in seen:
                    refused.append({"device": device, "key": str(key),
                                    "reason": "already priced by an earlier run"})
                    continue
                summary = row["summary"]
                cv = summary.get("cv")
                if cv is None:
                    refused.append({"device": device, "key": str(key), "reason": "no cv"})
                    continue
                if device == "gpu" and cv > GPU_MAX_CV:
                    refused.append({"device": device, "key": str(key),
                                    "reason": f"cv {cv:.4f} > {GPU_MAX_CV}"})
                    continue
                flags = [CPU_UNSTABLE_FLAG] if device == "cpu" and cv > CPU_MAX_CV else []
                if flags and not allow_unstable:
                    refused.append({"device": device, "key": str(key),
                                    "reason": f"cv {cv:.4f} > {CPU_MAX_CV}; the CPU protocol "
                                              f"carries this only as an explicit validation "
                                              f"export (--allow-unstable)"})
                    continue
                seen.add(key)
                duration = summary["median_ns"]
                if device == "cpu":
                    # Native samples are integral nanoseconds; an even-count median can be
                    # half a nanosecond, rounded up once as the CPU export declares.
                    duration = max(1, math.ceil(duration))
                prices.append({
                    "device": device, "kind": case["kind"], "phase": case["phase"],
                    "batch": 1, "tokens": int(case["tokens"]),
                    "visible_history": int(case["visible_history"]),
                    "duration_ns": duration,
                    "provenance": "measured_macro",
                    "duration_conversion": ("ceil_observed_median_ns" if device == "cpu"
                                            else "observed_median_ns"),
                    "measurement": {"profiles": [str(run_dir / row["raw_output"])]
                                    if row.get("raw_output") else [],
                                    "samples": summary.get("samples"),
                                    "cv": cv,
                                    "quality_flags": flags,
                                    "cv_gt_0_1": bool(flags)},
                })

    if refused:
        for item in refused:
            print(f"refused {item}", file=sys.stderr)
    prices.sort(key=lambda p: (p["kind"], p["device"], p["phase"], p["visible_history"]))
    return {"schema": "ir-ggml-growth-prices/v1", "model": model, "kv_dtype": KV_DTYPE,
            "window": {"tokens": tokens, "rounds": rounds, "depths": window(tokens, rounds)},
            "runs": [str(p) for p in list(cpu_runs) + list(cuda_runs)],
            "prices": prices}


def check(model: str, tokens: int, rounds: int, prices_path: Path,
          frozen: Path | None) -> list[str]:
    """Every (device, stage, depth) the chain needs must be in the merged table."""
    import run as runner  # same directory; the lookup is the one the campaign will use

    extensions = [Path(prices_path)] + ([Path(frozen)] if frozen else [])
    table = runner.table_for(model, extensions)
    measured = {(p["device"], p["kind"], p["phase"], int(p["tokens"]),
                 int(p["visible_history"]))
                for p in json.loads(Path(prices_path).read_text(encoding="utf-8"))["prices"]}
    configs = stage_configs(model)
    kinds = [k for k in ("projection", "attention", "dense_ffn", "moe_ffn") if k in configs]
    devices = sorted({key[0] for key in table})
    problems = []
    for device in devices:
        for kind in kinds:
            for depth in window(tokens, rounds):
                if (device, kind, "decode", 1, depth) not in table:
                    problems.append(f"{device}/{kind}/decode@1t/{depth}h: no price")
                elif (device, kind, "decode", 1, depth) not in measured:
                    problems.append(f"{device}/{kind}/decode@1t/{depth}h: priced by the other "
                                    f"table, not by this tranche (mixed sessions)")
            # the campaign's prefill shape decides A2's other half; from the frozen
            # table it is a second session's number, which is a disclosure, not a fault
            if (device, kind, "prefill", tokens, 0) not in measured and \
                    (device, kind, "prefill", tokens, 0) in table:
                print(f"note: {device}/{kind}/prefill@t{tokens} comes from the frozen table "
                      f"(not this session)", file=sys.stderr)
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("suite", "export", "check"))
    parser.add_argument("--model", default="qwen3-0.6b")
    parser.add_argument("--tokens", type=int, default=64, help="prefill length of the campaign")
    parser.add_argument("--rounds", type=int, default=8, help="decode rounds == window size")
    parser.add_argument("--out", type=Path, default=None,
                        help="suite: a directory; export: a file")
    parser.add_argument("--cpu-run", type=Path, action="append", default=[],
                        help="a CPU measurement run; repeatable, earlier runs win a cell")
    parser.add_argument("--cuda-run", type=Path, action="append", default=[],
                        help="an accelerator measurement run; repeatable")
    parser.add_argument("--prices", type=Path, default=None)
    parser.add_argument("--frozen", type=Path, default=None,
                        help="a second table (e.g. the GPU extension) the campaign also loads")
    parser.add_argument("--allow-unstable", action="store_true",
                        help="export CPU cells above cv 0.1 carrying the CPU protocol's own "
                             "quality flag, instead of refusing them")
    args = parser.parse_args(argv)

    if args.action == "suite":
        if args.out is None:
            raise SystemExit("suite needs --out DIR")
        args.out.mkdir(parents=True, exist_ok=True)
        cpu = args.out / f"cpu-{args.model}-t{args.tokens}.json"
        cuda = args.out / f"cuda-{args.model}-t{args.tokens}.json"
        cpu.write_text(json.dumps(cpu_suite(args.model, args.tokens, args.rounds), indent=2) + "\n",
                       encoding="utf-8")
        cuda.write_text(json.dumps(cuda_suite(args.model, args.tokens, args.rounds), indent=2) + "\n",
                        encoding="utf-8")
        print(f"{cpu}\n{cuda}  depths {window(args.tokens, args.rounds)}")
        return 0
    if args.action == "export":
        if args.out is None:
            raise SystemExit("export needs --out FILE")
        document = export(args.model, args.tokens, args.rounds,
                          list(args.cpu_run), list(args.cuda_run),
                          allow_unstable=args.allow_unstable)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(f"{args.out}: {len(document['prices'])} prices over "
              f"{len(document['window']['depths'])} depths")
        return 0
    problems = check(args.model, args.tokens, args.rounds, args.prices, args.frozen)
    if problems:
        print(f"{len(problems)} missing price(s):", file=sys.stderr)
        for line in problems[:20]:
            print("  " + line, file=sys.stderr)
        return 1
    print("check: every device, stage and depth the chain visits carries a price")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
