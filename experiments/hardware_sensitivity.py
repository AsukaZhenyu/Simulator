"""Reproducible, uncalibrated hardware/strategy sensitivity through public APIs.

Install the repository first (``python -m pip install -e .``), then run this
file. No profiles, model weights, native probes or hardware detection are used.
The finite candidate sets are an experiment protocol, not a search algorithm.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
from html import escape
import json
import math
from pathlib import Path
import platform
import time

from llm_simulator import (
    __version__, AnalyticalCosts, Cache, Device, Hardware, InvalidInput, Model,
    Prefetch, Route, Scheduling, Simulator, Unsupported, Workload,
)


WARNING = "Uncalibrated analytical simulation; no hardware measurements or global-optimality claim."
GROUPS = {
    "execution_only": ("e0-s0", "e1-s0", "e2-s0", "e3-s0"),
    "storage_only": ("e0-s0", "e0-s1", "e0-s2", "e0-s3"),
    "joint": ("e0-s0", "e1-s1", "e2-s2", "e3-s3"),
}
BASELINE = "e0-s0"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def stable_hash(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class Point:
    id: str
    cpu_factor: float = 1.0
    gpu_factor: float = 1.0
    link_factor: float = 1.0
    vram_bytes: int = 2 * 1024**2


def experiment_points(quick=False):
    """One-factor-at-a-time grids; the centre is executed only once."""
    base = Point("base")
    grids = {
        "cpu_compute": [(0.5, replace(base, id="cpu-0.5", cpu_factor=0.5)),
                        (1.0, base), (2.0, replace(base, id="cpu-2", cpu_factor=2.0))],
        "gpu_compute": [(0.5, replace(base, id="gpu-0.5", gpu_factor=0.5)),
                        (1.0, base), (2.0, replace(base, id="gpu-2", gpu_factor=2.0))],
        "transfer_bandwidth": [(0.25, replace(base, id="link-0.25", link_factor=0.25)),
                               (1.0, base), (4.0, replace(base, id="link-4", link_factor=4.0))],
        "vram_capacity": [(n * 1024, replace(base, id=f"vram-{n}k", vram_bytes=n * 1024))
                          for n in (64, 128, 256, 512)] + [(base.vram_bytes, base)],
    }
    if quick:
        grids = {"vram_capacity": [grids["vram_capacity"][0],
                                   grids["vram_capacity"][-1]]}
    points = {point.id: point for grid in grids.values() for _, point in grid}
    # Centre first: frozen-at-centre policies can be reported immediately.
    return [base] + [p for key, p in sorted(points.items()) if key != "base"], grids


def make_inputs(kind, point):
    """Fictional rates and a finite small graph; not a named real machine."""
    model_args = dict(layers=2, hidden_size=64, intermediate_size=192,
                      heads=4, kv_heads=2)
    model = Model.dense(**model_args) if kind == "dense" else Model.moe(
        **model_args, experts=4, top_k=2)
    # A deterministic evaluator-owned trace. It is not router-quality evidence.
    routes = () if kind == "dense" else tuple(
        Route("request0", f"layer{layer}", position,
              ((position + layer) % 4, (position + layer + 1) % 4))
        for layer in range(2) for position in range(10)
    )
    workload = Workload(prefill_tokens=8, decode_steps=2, routes=routes,
        route_source="declared_synthetic_rotating_pairs_not_measured" if routes else None)
    hardware = Hardware(
        cpu=Device(100e9 * point.cpu_factor, 20e9, 500),
        gpu=Device(1e12 * point.gpu_factor, 100e9, 2000),
        dram_capacity_bytes=128 * 1024**2, vram_capacity_bytes=point.vram_bytes,
        transfer_bandwidth_bytes_per_second=8e9 * point.link_factor,
        ssd_read_bandwidth_bytes_per_second=1e9,
        ssd_write_bandwidth_bytes_per_second=500e6,
        transfer_latency_ns=1000)
    return model, hardware, workload


def strategy(kind, candidate):
    execution, storage = (int(part[1:]) for part in candidate.split("-"))
    ffn = "dense_ffn" if kind == "dense" else "expert_ffn"
    placements = ({}, {}, {ffn: "cpu"}, {"attention": "cpu"})[execution]
    default_device = "cpu" if execution == 1 else "gpu"
    # Budgets are fixed strategy parameters across hardware points. They do
    # not expand silently when the simulated VRAM capacity increases.
    caches = (Cache(rule="none"), Cache(rule="lru", budget_bytes=192 * 1024),
              Cache(rule="lru", budget_bytes=1536 * 1024),
              Cache(rule="history_frequency", budget_bytes=1536 * 1024))
    prefetch = Prefetch(next_layers=1, max_count=2, include_dense_weights=True) if storage == 3 else Prefetch()
    return Scheduling(default_device=default_device, placements=placements,
                      release="last_use", cache=caches[storage], prefetch=prefetch)


def candidate_catalogue(exhaustive=False):
    ids = set(candidate for members in GROUPS.values() for candidate in members)
    if exhaustive:
        ids.update(f"e{e}-s{s}" for e in range(4) for s in range(4))
    return sorted(ids)


def run_candidate(kind, point, candidate, output, retain_traces=False):
    model, hardware, workload = make_inputs(kind, point)
    scheduling = strategy(kind, candidate)
    record = dict(model_kind=kind, point_id=point.id, candidate_id=candidate,
                  calibrated=False, hardware_measurement=False,
                  numerical_llm_execution=False, cost_model="analytical_roofline",
                  point=asdict(point), exception_type=None, exception_message=None)
    record["strategy_sha256"] = stable_hash(dict(model_kind=kind, scheduling=asdict(scheduling)))
    record["declared_inputs_sha256"] = stable_hash(dict(model=asdict(model), hardware=asdict(hardware),
                                                       workload=asdict(workload), workspace_bytes=0))
    started = time.perf_counter()
    try:
        simulator = Simulator(model=model, hardware=hardware, workload=workload,
                              cost_model=AnalyticalCosts(workspace_bytes=0))
        # Every candidate gets a fresh runtime, cold DRAM weights, empty KV,
        # the same token trace and the same mandatory DRAM output/KV goals.
        with simulator.compile(scheduling=scheduling) as compiled:
            result = compiled.run()
            summary = result.summary()
            summary.pop("input_directory", None)  # temporary path is not an identity
            record.update(summary)
            record["reason"] = result.evaluation.reason
            # Generated-file paths belong to this particular compilation.
            # Preserve its digest without calling it a portable strategy ID.
            record["compiled_ir_audit_sha256"] = stable_hash(compiled.ir_document())
            record["event_sha256"] = stable_hash([asdict(event) for event in result.events])
            record["event_counts"] = dict(sorted(Counter(event.kind for event in result.events).items()))
            record["compute_ns_by_device"] = {
                device: sum(event.end_ns - event.start_ns for event in result.events
                            if event.kind == "COMPUTE" and event.device == device
                            and event.end_ns is not None and event.start_ns is not None)
                for device in ("cpu", "gpu")}
            if retain_traces:
                destination = output / "traces" / kind / point.id / candidate
                result.save(destination)
                record["trace_directory"] = str(destination.relative_to(output))
    except (Unsupported, InvalidInput, ValueError) as error:
        record.update(status="unsupported_or_invalid_input", success=False,
                      makespan_ns=None, makespan_ms=None,
                      exception_type=type(error).__name__, exception_message=str(error))
    record["experiment_wall_seconds"] = time.perf_counter() - started
    return record


def best_candidate(records, members):
    """Only complete valid runs enter selection; deterministic tie breaking."""
    valid = [row for row in records if row["candidate_id"] in members and row.get("success")]
    return min(valid, key=lambda row: (row["makespan_ns"], row["candidate_id"])) if valid else None


def derive_tables(records, grids, exhaustive=False):
    index = {(r["model_kind"], r["point_id"], r["candidate_id"]): r for r in records}
    kinds = sorted({r["model_kind"] for r in records})
    groups = dict(GROUPS)
    if exhaustive:
        groups["joint_full16_unequal_budget"] = tuple(candidate_catalogue(True))
    selections, elasticities, thresholds = [], [], []
    for kind in kinds:
        centre = [r for r in records if r["model_kind"] == kind and r["point_id"] == "base"]
        for group, members in groups.items():
            frozen = best_candidate(centre, members)
            for axis, grid in grids.items():
                axis_rows = []
                for value, point in grid:
                    available = [index[(kind, point.id, member)] for member in members]
                    selected = best_candidate(available, members)
                    fixed = index[(kind, point.id, BASELINE)]
                    frozen_here = None if frozen is None else index[(kind, point.id, frozen["candidate_id"])]
                    row = dict(model_kind=kind, axis=axis, parameter_value=value,
                        point_id=point.id, group=group, candidate_budget=len(members),
                        feasible_candidates=sum(bool(r.get("success")) for r in available),
                        selected_candidate=None if selected is None else selected["candidate_id"],
                        selected_makespan_ms=None if selected is None else selected["makespan_ms"],
                        fixed_reference_success=fixed.get("success", False),
                        fixed_reference_ms=fixed.get("makespan_ms") if fixed.get("success") else None,
                        frozen_at_centre_candidate=None if frozen is None else frozen["candidate_id"],
                        frozen_at_centre_ms=(frozen_here.get("makespan_ms") if frozen_here and frozen_here.get("success") else None),
                        analytical_only=True)
                    for metric in ("h2d_bytes", "d2h_bytes", "ssd_read_bytes", "ssd_write_bytes",
                                   "peak_dram_bytes", "peak_vram_bytes"):
                        row["selected_" + metric] = None if selected is None else selected.get(metric)
                    row["frozen_to_reselected_latency_ratio"] = (
                        row["frozen_at_centre_ms"] / row["selected_makespan_ms"]
                        if row["frozen_at_centre_ms"] is not None and row["selected_makespan_ms"] else None)
                    selections.append(row)
                    axis_rows.append(row)
                for left, right in zip(axis_rows, axis_rows[1:]):
                    for policy, key in (("fixed_reference", "fixed_reference_ms"),
                                        ("frozen_at_centre", "frozen_at_centre_ms"),
                                        ("reselected", "selected_makespan_ms")):
                        a, b = left[key], right[key]
                        elasticity = (math.log(b / a) / math.log(right["parameter_value"] / left["parameter_value"])
                                      if a and b else None)
                        elasticities.append(dict(model_kind=kind, axis=axis, group=group,
                            policy=policy, parameter_low=left["parameter_value"],
                            parameter_high=right["parameter_value"],
                            log_latency_elasticity=elasticity,
                            policy_switch=(left["selected_candidate"] != right["selected_candidate"]
                                           if policy == "reselected" else False),
                            interpretation="finite-grid analytical elasticity; null when either endpoint is infeasible"))
                if axis == "vram_capacity":
                    feasible = [r for r in axis_rows if r["selected_candidate"] is not None]
                    failures = [r for r in axis_rows if r["selected_candidate"] is None]
                    thresholds.append(dict(model_kind=kind, group=group,
                        first_feasible_tested_vram_bytes=min((r["parameter_value"] for r in feasible), default=None),
                        largest_infeasible_tested_vram_bytes=max((r["parameter_value"] for r in failures), default=None),
                        exact_threshold_known=False,
                        note="Only tested grid points; a CPU placement may need no GPU weight capacity. No continuous threshold is claimed."))
    return selections, elasticities, thresholds


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (json.dumps(value, sort_keys=True, ensure_ascii=False)
                                   if isinstance(value, (dict, list)) else value)
                             for key, value in row.items()})


def plot_svg(output, selections):
    """Dependency-free semantic plots; gaps denote infeasible/unknown, not zero."""
    colors = {"execution_only": "#1f77b4", "storage_only": "#d97706", "joint": "#14823b"}
    for kind in sorted({r["model_kind"] for r in selections}):
        for axis in sorted({r["axis"] for r in selections}):
            display_kind = "Dense" if kind == "dense" else "MoE"
            selected = [r for r in selections if r["model_kind"] == kind and r["axis"] == axis and r["group"] in GROUPS]
            values = sorted({r["parameter_value"] for r in selected})
            ys = [r["selected_makespan_ms"] for r in selected if r["selected_makespan_ms"] is not None]
            ys += [r["fixed_reference_ms"] for r in selected if r["fixed_reference_ms"] is not None]
            top = max(ys, default=1) * 1.12
            width, height, x0, y0, pw, ph = 760, 400, 80, 65, 640, 245
            def x(value):
                return x0 + pw * (math.log(value) - math.log(values[0])) / (math.log(values[-1]) - math.log(values[0]))
            def y(value):
                return y0 + ph - ph * value / top
            lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                     '<rect width="100%" height="100%" fill="white"/>',
                     f'<text x="20" y="23" font-size="17" font-family="sans-serif">{escape(display_kind+": "+axis.replace("_", " "))}</text>',
                     '<text x="20" y="44" font-size="11" font-family="sans-serif" fill="#9b1c1c">UNCALIBRATED ANALYTICAL ESTIMATES — no real hardware validation</text>']
            for step in range(5):
                value = top * step / 4
                lines += [f'<line x1="{x0}" y1="{y(value):.2f}" x2="{x0+pw}" y2="{y(value):.2f}" stroke="#ddd"/>',
                          f'<text x="{x0-8}" y="{y(value)+4:.2f}" text-anchor="end" font-size="11">{value:.3f}</text>']
            lines.append('<text x="20" y="185" transform="rotate(-90 20 185)" font-size="12">Analytical latency (ms)</text>')
            for value in values:
                label = f"{value / 1024:g} KiB" if axis == "vram_capacity" else f"{value:g}x"
                lines.append(f'<text x="{x(value):.2f}" y="330" text-anchor="middle" font-size="11">{label}</text>')
            series = [(group, "selected_makespan_ms", color) for group, color in colors.items()]
            series.append(("execution_only", "fixed_reference_ms", "#555555"))
            for group, key, color in series:
                rows = sorted((r for r in selected if r["group"] == group), key=lambda r: r["parameter_value"])
                previous = None
                for row in rows:
                    if row[key] is None:
                        previous = None
                        continue
                    current = (x(row["parameter_value"]), y(row[key]))
                    if previous:
                        dash = ' stroke-dasharray="7 4"' if group == "joint" or key == "fixed_reference_ms" else ""
                        lines.append(f'<line x1="{previous[0]:.2f}" y1="{previous[1]:.2f}" x2="{current[0]:.2f}" y2="{current[1]:.2f}" stroke="{color}" stroke-width="2"{dash}/>')
                    fill = "white" if group == "joint" else color
                    lines.append(f'<circle cx="{current[0]:.2f}" cy="{current[1]:.2f}" r="3.5" fill="{fill}" stroke="{color}"/>')
                    previous = current
            for i, (label, color) in enumerate(list(colors.items()) + [("fixed_reference", "#555555")]):
                xx = 25 + i * 180
                lines += [f'<rect x="{xx}" y="350" width="12" height="4" fill="{color}"/>',
                          f'<text x="{xx+17}" y="356" font-size="11">{escape(label)}</text>']
            lines += ['<text x="25" y="385" font-size="11">Budgets: 4 per family. Missing points: no successful completion. Joint is a declared four-point subset.</text>', '</svg>']
            (output / f"{kind}-{axis}.svg").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def plot_matplotlib(output, selections):
    """Optional standard scientific rendering; core use never requires it."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    colors = {"execution_only": "#1f77b4", "storage_only": "#d97706", "joint": "#14823b"}
    for kind in sorted({r["model_kind"] for r in selections}):
        for axis in sorted({r["axis"] for r in selections}):
            display_kind = "Dense" if kind == "dense" else "MoE"
            rows = [r for r in selections if r["model_kind"] == kind and r["axis"] == axis and r["group"] in GROUPS]
            figure, plot = plt.subplots(figsize=(8.0, 4.8), layout="constrained")
            markers = {"execution_only": "o", "storage_only": "s", "joint": "^"}
            for group, color in colors.items():
                series = sorted((r for r in rows if r["group"] == group), key=lambda r: r["parameter_value"])
                plot.plot([r["parameter_value"] for r in series],
                          [r["selected_makespan_ms"] if r["selected_makespan_ms"] is not None else math.nan for r in series],
                          marker=markers[group], linestyle="--" if group == "joint" else "-",
                          markerfacecolor="none" if group == "joint" else color,
                          label=group.replace("_", " "), color=color)
            series = sorted((r for r in rows if r["group"] == "execution_only"), key=lambda r: r["parameter_value"])
            plot.plot([r["parameter_value"] for r in series],
                      [r["fixed_reference_ms"] if r["fixed_reference_ms"] is not None else math.nan for r in series],
                      marker="x", linestyle="--", label="fixed reference", color="#555555")
            plot.set_xscale("log", base=2)
            xs = [r["parameter_value"] for r in series]
            plot.set_xticks(xs, [f"{v / 1024:g} KiB" if axis == "vram_capacity" else f"{v:g}x" for v in xs])
            plot.set_xlabel({"cpu_compute": "CPU compute throughput multiplier",
                             "gpu_compute": "GPU compute throughput multiplier",
                             "transfer_bandwidth": "DRAM–VRAM bandwidth multiplier",
                             "vram_capacity": "VRAM capacity"}[axis])
            plot.set_ylabel("Analytical latency (ms)")
            plot.set_ylim(bottom=0)
            plot.grid(alpha=0.25)
            plot.legend(fontsize=9, loc="best")
            plot.set_title(display_kind + ": equal four-candidate families", fontsize=12)
            figure.suptitle("UNCALIBRATED ANALYTICAL ESTIMATES", fontsize=11, color="#9b1c1c")
            figure.supxlabel("Missing points: no successful completion. Joint: declared subset, not a superset.", fontsize=9)
            figure.savefig(output / f"{kind}-{axis}.png", dpi=180)
            plt.close(figure)
    return True


def report(output, records, selections, thresholds):
    lines = ["# Analytical hardware sensitivity experiment", "", f"**{WARNING}**", "",
        "Three four-candidate families are selected independently at each hardware point. The fixed reference is one strategy, not a fourth search family. Joint uses a declared diagonal subset, not all 16 combinations; these sets are not nested.", "",
        "A frozen policy is selected at the centre once and replayed unchanged along each grid. Reselection uses the same four declared candidates. These are finite, offline workload-conditioned comparisons; there is no globally optimal or production-adaptive claim.", "",
        "## Execution outcomes", "", "| Model | Attempts | Goal reached | Explicit failures |", "|---|---:|---:|---:|"]
    for kind in sorted({r["model_kind"] for r in records}):
        rows = [r for r in records if r["model_kind"] == kind]
        successes = sum(bool(r.get("success")) for r in rows)
        lines.append(f"| {kind} | {len(rows)} | {successes} | {len(rows)-successes} |")
    lines += ["", "## Centre-point candidate selection", "",
              "| Model | Family | Candidate | Estimated latency (ms) | H2D bytes | Peak VRAM bytes |",
              "|---|---|---|---:|---:|---:|"]
    seen = set()
    for row in selections:
        key = (row["model_kind"], row["group"])
        if row["point_id"] == "base" and key not in seen:
            seen.add(key)
            latency = row["selected_makespan_ms"]
            lines.append(f"| {key[0]} | {key[1]} | {row['selected_candidate'] or 'none feasible'} | {latency if latency is not None else ''} | {row['selected_h2d_bytes'] if latency is not None else ''} | {row['selected_peak_vram_bytes'] if latency is not None else ''} |")
    lines += ["", "## Tested capacity boundaries", "",
        "| Model | Family | First feasible tested VRAM (bytes) | Exact threshold known? |",
        "|---|---|---:|---|"]
    for row in thresholds:
        lines.append(f"| {row['model_kind']} | {row['group']} | {row['first_feasible_tested_vram_bytes']} | no |")
    lines += ["", "## Interpretation limits", "",
        "- Rates and capacities are fictional declared inputs; graphs use F32, idealized zero workspace, one compute slot per device and a shared copy slot.",
        "- Dense uses complete materialized macro contracts. MoE uses opaque expert macros and a declared synthetic route trace; policy sees only revealed routes/history.",
        "- CPU/GPU compute sweeps hold memory bandwidth and launch cost fixed. Flat curves can reflect memory/launch/transfer bottlenecks rather than a broken sweep.",
        "- All candidates begin with cold weights in DRAM and empty KV, and use the same mandatory hidden/KV DRAM goals. Optional retained weight copies at termination differ by strategy; no artificial final flush is charged.",
        "- Failures stay visible; no CPU fallback, omitted error, interpolation through an infeasible point, or replacement of missing data with zero.",
        "- Feasible means the declared simulator goals were met; it does not establish numerical model correctness or measured performance accuracy.",
        "- Log elasticity describes a tested interval; policy switches and capacity boundaries are discontinuities, not derivatives. No prices/energy are modeled, so no economic return is inferred.",
        "- Wall time is harness execution time on the current host, not simulated inference latency or a reproducible hardware benchmark.", "",
        "See protocol.json for every candidate, input and assumption; runs.jsonl/runs.csv retain every attempted run. selections.csv compares frozen and reselected candidates; elasticities.csv and thresholds.json retain null/finite-grid limits.", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new or empty result directory")
    parser.add_argument("--models", nargs="+", choices=("dense", "moe"), default=["dense", "moe"])
    parser.add_argument("--quick", action="store_true", help="centre and one low-VRAM point only")
    parser.add_argument("--exhaustive-joint", action="store_true", help="separate 16-candidate comparison, explicitly unequal budget")
    parser.add_argument("--retain-traces", action="store_true", help="save each result's complete public-API audit artifact")
    args = parser.parse_args(argv)
    args.models = list(dict.fromkeys(args.models))
    output = args.out.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("--out must be new or empty; existing evidence is never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    points, grids = experiment_points(args.quick)
    candidates = candidate_catalogue(args.exhaustive_joint)
    protocol = dict(format_version="hardware-sensitivity-0.1", warning=WARNING,
        library_version=__version__, python_version=platform.python_version(),
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        hardware_detection=False, calibrated=False, measured_profiles_used=False,
        equal_candidate_budget=4, groups=GROUPS, fixed_reference=BASELINE,
        joint_coverage="predeclared four-point diagonal, not a superset of other families",
        exhaustive_joint_separate_unequal_budget=args.exhaustive_joint,
        points=[asdict(point) for point in points],
        grids={axis: [dict(value=value, point_id=point.id) for value, point in grid]
               for axis, grid in grids.items()},
        models={}, initial_condition="cold weights in DRAM; empty KV; same finite trace",
        terminal_condition="same mandatory hidden/KV DRAM goals; policy-owned optional weight copies may differ",
        scratch="AnalyticalCosts(workspace_bytes=0): simplified declaration, not measured zero")
    for kind in args.models:
        model, hardware, workload = make_inputs(kind, Point("base"))
        protocol["models"][kind] = dict(model=asdict(model), centre_hardware=asdict(hardware),
            workload=asdict(workload), candidates={c: asdict(strategy(kind, c)) for c in candidates})
    write_json(output / "protocol.json", protocol)
    records = []
    total = len(args.models) * len(points) * len(candidates)
    with (output / "runs.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for kind in args.models:
            for point in points:
                for candidate in candidates:
                    row = run_candidate(kind, point, candidate, output, args.retain_traces)
                    records.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
                    print(f"{len(records)}/{total} {kind} {point.id} {candidate}: {row['status']} {row.get('makespan_ms')} ms", flush=True)
    selections, elasticities, thresholds = derive_tables(records, grids, args.exhaustive_joint)
    write_csv(output / "runs.csv", records)
    write_csv(output / "selections.csv", selections)
    write_csv(output / "elasticities.csv", elasticities)
    write_json(output / "thresholds.json", thresholds)
    plot_svg(output, selections)
    png_rendered = plot_matplotlib(output, selections)
    report(output, records, selections, thresholds)
    write_json(output / "completion.json", dict(complete=True, attempted_runs=len(records),
        valid_runs=sum(bool(r.get("success")) for r in records),
        protocol_sha256=stable_hash(protocol), calibrated=False, hardware_measurement=False,
        optional_matplotlib_png_rendered=png_rendered))
    print(str(output / "REPORT.md"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
