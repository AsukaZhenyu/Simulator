"""Run the decision-margin study over a frozen campaign.

    python -m profiling.decision_margin.run --campaign <e2a-v2> --output <dir>

Nothing is measured and no simulator is executed: the campaign's frozen prices,
its per-price repeats and the coordinate keys recorded in `predictions.json`
are enough to recompute a selection under a perturbed price table.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path

from profiling.decision_margin import margin as M


def infer_archive_root(campaign: Path) -> Path | None:
    parts = campaign.resolve().parts
    text = "\\".join(parts)
    marker = "simulator-archive\\build"
    if marker in text:
        head = text.split(marker, 1)[0]
        return Path(head) / "simulator-archive" / "build"
    return None


def check_reproduction(root: Path, archive_root: Path | None):
    """The stored keys must add up to the stored prediction, exactly."""
    frozen = M.read(root / "profile-frozen.json")
    exact = {entry["key"]: Fraction(entry["mean_ns"]) for entry in frozen["entries"]}
    worst = Fraction()
    for row in M.read(root / "predictions.json"):
        total = sum((exact[key] for key in row["keys"]), Fraction())
        stored = Fraction(row["prediction"]["duration_ns"])
        worst = max(worst, abs(total - stored) / stored)
    return worst


def observed_error(instances, prices):
    """Prediction error per program, so the perturbation range is not arbitrary."""
    rows = []
    for instance in instances:
        for program in instance.programs:
            predicted = sum(prices[key] for key in program.keys)
            measured = float(instance.measured_ns[program.strategy])
            rows.append(dict(program=program.label,
                             ape=abs(predicted - measured) / measured,
                             signed=(predicted - measured) / measured))
    ordered = sorted(row["ape"] for row in rows)
    return dict(rows=rows, observations=len(rows),
                median=ordered[len(ordered) // 2], p90=ordered[int(len(ordered) * 0.9)],
                maximum=ordered[-1])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epsilons", type=float, nargs="+",
                        default=(0.01, 0.02, 0.05, 0.10, 0.15, 0.20))
    options = parser.parse_args(argv)

    root = options.campaign.resolve()
    archive_root = infer_archive_root(root)
    residual = check_reproduction(root, archive_root)
    prices, samples, instances = M.load_campaign(root, archive_root)
    study = M.sweep(instances, prices, samples, options.epsilons, options.draws, options.seed)
    error = observed_error(instances, prices)

    base = []
    for instance in instances:
        best, runner_up, span, _ = instance.margins(prices)
        base.append(dict(workload=instance.workload, base_choice=best,
                         runner_up=runner_up, margin=span,
                         analytic_radius=span / (2.0 + span),
                         measured_best=min(instance.measured_ns, key=lambda s: instance.measured_ns[s]),
                         regret_at_base=instance.regret(best),
                         measured_ns={str(k): float(v) for k, v in sorted(instance.measured_ns.items())},
                         common_mode_holds=M.common_mode_cannot_flip(instance, prices, 0.2)))

    report = dict(
        schema="decision-margin-1",
        campaign=str(root),
        protocol=dict(draws=options.draws, seed=options.seed,
                      epsilons=list(options.epsilons),
                      perturbation="independent per price entry, uniform in [-eps, +eps]",
                      bootstrap="resample each price from its own measured repeats",
                      regret_basis="heldout process means; selection cost uses the same table"),
        reproduction_max_relative_residual=float(residual),
        decisions=base,
        observed_error=dict(median=error["median"], p90=error["p90"],
                            maximum=error["maximum"], observations=error["observations"]),
        sweep=study,
    )
    options.output.mkdir(parents=True, exist_ok=True)
    target = options.output / "decision-margin.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"reproduction residual: {float(residual):.3e}")
    print(f"observed prediction error: median {error['median']:.2%} "
          f"p90 {error['p90']:.2%} max {error['maximum']:.2%}")
    print(f"{'workload':<12}{'choice':>7}{'runner':>7}{'margin':>9}{'radius':>9}"
          f"{'boot flip':>10}")
    boot = {row["workload"]: row for row in study["bootstrap"]}
    for row in base:
        print(f"{row['workload']:<12}{row['base_choice']:>7}{str(row['runner_up']):>7}"
              f"{row['margin']:>8.2%}{row['analytic_radius']:>9.2%}"
              f"{boot[row['workload']]['flips']:>10}")
    for epsilon in sorted(study["per_error"]):
        rows = study["per_error"][epsilon]
        total = sum(row["flips"] for row in rows)
        worst = max((row["regret_max"] for row in rows), default=0.0)
        print(f"eps={float(epsilon):>5.0%}  flips {total}/{sum(row['draws'] for row in rows)}"
              f"  worst regret {worst:.2%}")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
