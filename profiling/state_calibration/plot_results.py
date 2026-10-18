"""Plot every primary process median from the portable state evidence summary.

Requires matplotlib; no hardware execution or source-data mutation. Outputs
must be new paths. All 72 primary process records are shown, including rejected
ones; separate diagnostic/control jobs are explicitly outside this figure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


CASES = ("h128-q16", "h256-q32")
CONDITIONS = ("baseline", "cpu_replay_load", "gpu_replay_load")
MODES = ("isolated", "mixed")
SPLITS = ("profile", "heldout")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot(input_path: str | Path, png_path: str | Path, svg_path: str | Path) -> dict:
    input_path, png_path, svg_path = map(Path, (input_path, png_path, svg_path))
    for output in (png_path, svg_path):
        if output.exists():
            raise FileExistsError(f"new output path required: {output}")
    raw = input_path.read_bytes()
    input_sha = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw)
    if data.get("schema") != "device-state-evidence-summary-0.1":
        raise ValueError("unsupported portable evidence summary")
    primary = [row for row in data["jobs"] if row["split"] in SPLITS]
    expected = {(case, condition, mode, split, repeat) for case in CASES for condition in CONDITIONS
                for mode in MODES for split in SPLITS for repeat in range(3)}
    points = {}
    for row in primary:
        try:
            repeat = int(row["id"].split("-", 2)[1])
        except (ValueError, IndexError) as error:
            raise ValueError("process repeat must be explicit in the saved job ID") from error
        key = (row["case_id"], row["condition"], row["mode"], row["split"], repeat)
        value = row.get("target_median_ns")
        if key not in expected or key in points or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("duplicate, unsupported or invalid primary process record; no filtering performed")
        if row.get("status") != "measured" or row.get("numerical_passed") is not True or row.get("target_sample_count") != 768:
            raise ValueError("this figure requires the declared measured 768-invocation primary process records")
        points[key] = row
    if set(points) != expected or data["measurement_counts"]["primary_processes"] != len(points):
        raise ValueError("primary process coverage differs from all 72 declared launches")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator

    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 13, "axes.labelsize": 11.5, "xtick.labelsize": 11,
        "ytick.labelsize": 10.5, "svg.fonttype": "none", "svg.hashsalt": input_sha,
        "axes.spines.top": False, "axes.spines.right": False})
    colors = {"isolated": "#2166AC", "mixed": "#D66032"}
    offsets = {"isolated": -.22, "mixed": .22}
    marker_offsets = {"profile": (-.14, -.09, -.04), "heldout": (.04, .09, .14)}
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.5), sharey=True)
    fig.subplots_adjust(left=.085, right=.985, bottom=.235, top=.77, wspace=.12)
    for case_index, (case, axis) in enumerate(zip(CASES, axes)):
        for condition_index, condition in enumerate(CONDITIONS):
            for mode in MODES:
                for split in SPLITS:
                    for repeat in range(3):
                        row = points[(case, condition, mode, split, repeat)]
                        x = condition_index + offsets[mode] + marker_offsets[split][repeat]
                        axis.plot(x, row["target_median_ns"] / 1e6,
                            marker="o" if split == "profile" else "D", markersize=6.4,
                            markerfacecolor=colors[mode] if split == "profile" else "white",
                            markeredgecolor=colors[mode], markeredgewidth=1.2,
                            linestyle="None", zorder=3)
        axis.set_title(f"({'ab'[case_index]}) H = {128 if case_index == 0 else 256}, q = {16 if case_index == 0 else 32}", loc="left", pad=15, fontweight="bold")
        axis.set_xlim(-.55, 2.55)
        axis.set_ylim(0, 12.4)
        axis.set_xticks(range(3), ("Baseline", "Owned CPU\nreplay load", "Owned GPU\nreplay load"))
        axis.yaxis.set_major_locator(MultipleLocator(2))
        axis.grid(axis="y", color="#D8DEE7", linewidth=.7, zorder=0)
        axis.tick_params(axis="both", length=3.5, color="#627085")
        axis.set_axisbelow(True)
        for spine in ("left", "bottom"):
            axis.spines[spine].set_color("#8390A0")
    axes[0].set_ylabel("Completed PV action time per process (ms)", labelpad=10)
    fig.suptitle("PV action timings vary across contexts and controlled load conditions",
                 x=.085, y=.965, ha="left", fontsize=15.5, fontweight="bold")
    fig.text(.085, .912, "F32 Attention PV  |  CUDA graphs disabled  |  Legacy completed-action timing boundary", fontsize=11.5, color="#475569")
    handles = [Line2D([], [], color=colors[mode], linewidth=3, label=label) for mode, label in
               (("isolated", "Isolated action"), ("mixed", "Within full serial replay"))]
    handles.extend([Line2D([], [], color="#475569", marker="o", markersize=6, linestyle="None", label="Profile process"),
                    Line2D([], [], color="#475569", marker="D", markerfacecolor="white", markersize=6, linestyle="None", label="Heldout process")])
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .871), ncol=4,
               frameon=False, handlelength=1.7, columnspacing=2.1, fontsize=10.5)
    fig.text(.085, .116, "Every dot is shown: 3 profile + 3 heldout launches per context/condition; each dot summarizes 768 invocations.", fontsize=10.2, color="#334155")
    rejected = all(data["families"][family]["target"]["accepted"] == 0 for family in ("constant", "conditional"))
    if rejected:
        gate_note = "All 12 target timing cells fail the full accuracy/quality gates in both models. No causal device-speed claim."
    else:
        gate_note = "Acceptance counts remain those of the source report. These observations do not establish a causal device-speed model."
    fig.text(.085, .070, gate_note, fontsize=10.2, color="#9A3412")
    description = json.dumps({"input_sha256": input_sha, "plot_source_sha256": _sha(Path(__file__)),
        "primary_points": len(points), "profile_points": 36, "heldout_points": 36,
        "excluded_diagnostic_jobs": len(data["jobs"]) - len(primary), "outlier_removal": False,
        "bars_or_averaging_across_processes": False, "metric": "per-process completed PV host action median; ms",
        "full_llm_or_kernel_active_time": False, "quality_gate_results_unchanged": True}, sort_keys=True)
    for output in (png_path, svg_path):
        output.parent.mkdir(parents=True, exist_ok=True)
    title = "PV action timing across exact serial contexts and owned replay conditions"
    fig.savefig(svg_path, format="svg", metadata={"Title": title, "Description": description,
                "Creator": "Simulator state calibration plot_results.py", "Date": None})
    fig.savefig(png_path, format="png", dpi=300, metadata={"Title": title, "Description": description})
    plt.close(fig)
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != input_sha:
        raise ValueError("portable source summary changed during plotting")
    receipt = json.loads(description)
    receipt.update(input=str(input_path.resolve()), outputs={str(output.resolve()): _sha(output) for output in (png_path, svg_path)})
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--png", type=Path, required=True)
    parser.add_argument("--svg", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(plot(arguments.input, arguments.png, arguments.svg), indent=2))


if __name__ == "__main__":
    main()
