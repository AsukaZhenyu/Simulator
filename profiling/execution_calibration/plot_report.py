"""Standalone scientific figures from saved calibration evidence; no fitting."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    report = json.loads((args.experiment / "analysis/report.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    primary = [row for row in report["results"] if "@" not in row["strategy"] and row["observed_ns"] is not None]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    colors = {"gpu/resident": "#2457a5", "hybrid/resident": "#e58325", "hybrid/preload": "#23816b"}
    for strategy, color in colors.items():
        rows = [row for row in primary if row["strategy"] == strategy]
        x = np.array([row["observed_ns"] / 1000 for row in rows])
        y = np.array([row["predicted_ns"] / 1000 for row in rows])
        axes[0].scatter(x, y, label=strategy, color=color, s=45)
    bounds = [min(min(row["observed_ns"], row["predicted_ns"]) for row in primary)/1000*.7,
              max(max(row["observed_ns"], row["predicted_ns"]) for row in primary)/1000*1.4]
    axes[0].plot(bounds, bounds, color="black", linewidth=1, label="equal")
    axes[0].fill_between(bounds, np.array(bounds)*.8, np.array(bounds)*1.2, color="gray", alpha=.12, label="+/-20%")
    axes[0].set(xscale="log", yscale="log", xlabel="Measured heldout median (us)", ylabel="Component prediction (us)", title="Independent component prediction")
    axes[0].legend(fontsize=8)
    labels = [row["case_id"].replace("ffn-", "F ").replace("attention-", "A ") + "/" + row["strategy"].replace("hybrid/", "mix/").replace("gpu/", "G/") for row in primary]
    axes[1].barh(np.arange(len(primary)), [row["absolute_relative_error"]*100 for row in primary],
                 color=[colors[row["strategy"]] for row in primary])
    axes[1].set_yticks(np.arange(len(primary)), labels, fontsize=6.5)
    axes[1].invert_yaxis()
    axes[1].axvline(20, color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Absolute relative error (%)", title="Point errors; full quality gates reported separately")
    for axis in axes:
        axis.grid(axis="x", alpha=.15)
    figure.savefig(args.output / "prediction_error.png", dpi=180)
    figure.savefig(args.output / "prediction_error.pdf")
    plt.close(figure)

    ffns = [row for row in primary if row["case_id"].startswith("ffn-") and row["strategy"] == "hybrid/resident"]
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    x = np.arange(len(ffns))
    fixed = {row["case_id"]: row for row in report["results"] if row["strategy"] == "hybrid/resident@dispatch_model"}
    ax.bar(x-.25, [row["predicted_ns"]/1000 for row in ffns], width=.25, label="Ideal max components", color="#2457a5")
    ax.bar(x, [fixed[row["case_id"]]["predicted_ns"]/1000 for row in ffns], width=.25, label="+ independently measured handshake", color="#23816b")
    ax.bar(x+.25, [row["observed_ns"]/1000 for row in ffns], width=.25, label="Heldout mixed execution", color="#e58325")
    ax.set_xticks(x, [row["case_id"].replace("ffn-", "") for row in ffns], fontsize=8)
    ax.set(ylabel="Completed-work wall time (us)", title="Predeclared dispatch model versus ideal overlap")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.15)
    fig.savefig(args.output / "dispatch_model.png", dpi=180)
    fig.savefig(args.output / "dispatch_model.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    xs = np.arange(len(primary))
    max_cv = [max(s["cv"] for s in row["source_evidence"]+row["heldout_evidence"])*100 for row in primary]
    ax.bar(xs, max_cv, color=["#23816b" if row["calibrated"] else "#bd4f42" for row in primary])
    ax.axhline(15, color="black", linestyle="--", linewidth=1, label="Within-process CV gate (15%)")
    ax.set_xticks(xs, labels, rotation=70, ha="right", fontsize=6)
    ax.set(ylabel="Maximum within-process CV (%)", title="All component and heldout samples retained")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.15)
    fig.savefig(args.output / "measurement_stability.png", dpi=180)
    fig.savefig(args.output / "measurement_stability.pdf")
    plt.close(fig)
    print(json.dumps({"figures": str(args.output), "primary_predictions": len(primary)}))


if __name__ == "__main__":
    main()
