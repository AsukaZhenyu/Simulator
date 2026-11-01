"""Plot completed follow-up observations; never fit a performance parameter."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plot(v1, v2, multilayer, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    out.mkdir(parents=True, exist_ok=True)
    short, long, multi = read(v1), read(v2), read(multilayer)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    routes = ("dram->vram", "vram->dram")
    for index, (data, label, color) in enumerate(((short, "8-replay window", "#3575a7"),
                                                 (long, "256-replay window", "#df8c32"))):
        error = [100 * data["transfer"][route]["maximum_heldout_size_mean_ape"] for route in routes]
        stable = [data["transfer"][route]["process_cv_passes"] for route in routes]
        positions = np.arange(2) + (index - .5) * .32
        axes[0].bar(positions, error, .30, label=label, color=color)
        axes[1].bar(positions, stable, .30, label=label, color=color)
        for ax, values in ((axes[0], error), (axes[1], stable)):
            for x, value in zip(positions, values):
                ax.text(x, value, f"{value:.1f}" if ax is axes[0] else str(value), ha="center", va="bottom", fontsize=10)
    axes[0].axhline(10, color="#b42a29", linestyle="--", linewidth=1, label="10% error target")
    axes[0].set_ylabel("Maximum heldout size-mean APE (%)")
    axes[1].set_ylabel("Processes with trial CV at most 10%")
    axes[0].set_ylim(0, 1.5 * max(100 * data["transfer"][route]["maximum_heldout_size_mean_ape"]
                               for data in (short, long) for route in routes))
    axes[1].set_ylim(0, 34)
    for ax in axes:
        ax.set_xticks(range(2), ("H2D", "D2H"))
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9, loc="upper right")
    figure.suptitle("Independent transfer validation under two sampling windows")
    figure.supxlabel("Fresh calibration and heldout processes in each campaign; longer bursts may also change state.", fontsize=9)
    for extension in ("png", "svg"):
        figure.savefig(out / ("transfer-validation." + extension), dpi=200)
    plt.close(figure)

    figure, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    rows = short["observer"]["rows"]
    real = sorted(100 * abs(row["relative_difference"]) for row in rows if not row.get("sham") and row["status"] == "completed")
    sham = sorted(100 * abs(row["relative_difference"]) for row in rows if row.get("sham") and row["status"] == "completed")
    for values, label, color in ((real, "Within-process bare/stamped", "#3575a7"),
                                 (sham, "Within-process bare/bare control", "#46824b")):
        ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=label, color=color)
    ax.axvline(5, color="#b42a29", linestyle="--", label="5% contrast limit")
    ax.set(xlabel="Absolute difference between paired mean responses (%)", ylabel="Fraction of process contrasts",
           title="Observer consistency with adjacent ABBA / BAAB contrasts", ylim=(0, 1.03))
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=1, loc="lower right")
    ax.grid(alpha=.18)
    figure.supxlabel("New diagnostic binary and matched preconditioning; historical accuracy is not requalified.", fontsize=9)
    for extension in ("png", "svg"):
        figure.savefig(out / ("observer-consistency." + extension), dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(9.5, 7), layout="constrained")
    for ax, group in zip(axes.flat, multi["groups"]):
        candidates = group["candidates"]
        x = np.arange(len(candidates))
        measured = [row["measured_mean_ns"] / 1e6 for row in candidates]
        predicted = [row["predicted_ns"] / 1e6 for row in candidates]
        ax.bar(x - .17, measured, .32, color="#3575a7", label="Measured")
        ax.bar(x + .17, predicted, .32, color="white", edgecolor="#d8792d", hatch="//", label="Frozen prediction")
        for index, candidate in enumerate(candidates):
            values = [row["real_ns"] / 1e6 for row in multi["processes"]
                      if row["workload"] == group["workload"] and row["strategy"] == candidate["strategy"]
                      and row["status"] == "completed"]
            ax.scatter(np.repeat(index - .17, len(values)), values, s=16, color="#263743", alpha=.75)
        ax.set_xticks(x, ("S0\nCPU", "S1\nGPU", "S2\nCPU attention"))
        ax.set_ylabel("Replay-average latency (ms)")
        ax.set_title(group["workload"] + f"\nSelected S{group['predicted_winner']}; observed winner S{group['real_winner']}", fontsize=11)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    axes.flat[0].legend(frameon=False, fontsize=9)
    figure.suptitle("Connected multi-layer Dense execution on one CPU and one GPU")
    figure.supxlabel("Four fresh processes per recipe; serial F32, one round, supplied initial KV; complete observer matrix not rerun.", fontsize=9)
    for extension in ("png", "svg"):
        figure.savefig(out / ("multilayer-validation." + extension), dpi=200)
    plt.close(figure)
    receipt = dict(source_summaries=[dict(path=str(path.resolve()), sha256=sha(path)) for path in (v1, v2, multilayer)],
                   figures=[dict(path=str(path.resolve()), sha256=sha(path)) for path in sorted(out.glob("*.png"))],
                   parameter_fit_performed=False, plotting_script_sha256=sha(__file__))
    with (out / "plot-receipt.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("v1", "v2", "multilayer", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(plot(args.v1, args.v2, args.multilayer, args.out)))
