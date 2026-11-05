"""How far a selection sits from flipping, and what a flip would cost.

The evaluator's product is `argmin` over a strategy set.  A prediction error
does not have to change that `argmin`, and when it does the consequence can be
nil if the two candidates were nearly tied.  This module measures the three
quantities separately:

* **margin**   ``(T2 - T1) / T1`` for the two best predicted strategies.  It is
  arithmetic, not an experiment: an ordering survives independent per-price
  errors bounded by ``eps`` whenever ``eps <= (T2 - T1) / (T1 + T2)``.
* **flip**     whether a perturbed price table selects a different strategy.
* **regret**   ``(measured[selected] - min measured) / min measured``, i.e. what
  the flip actually cost, taken from the campaign's heldout processes.

Prices are perturbed at the **price-entry** level, never at the strategy total.
Two strategies usually share prices, so a shared error partly cancels; adding
independent noise to strategy totals would break that cancellation and
systematically overstate the flip rate.  A common-mode error applied to every
entry cannot change any ordering at all, which `common_mode_cannot_flip` states
as an executable check rather than a footnote.

The prediction of a program here is exactly the model's composition: a sum of
prices over that program's frozen action coordinates.  ``predictions.json``
records the coordinate key of every action, so a perturbed selection is a
recomputation over stored keys -- no simulator run and no re-measurement.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
import random
from typing import Iterable, Mapping, Sequence

# Campaigns were moved from `Simulator/build/...` to the local archive after the
# paper campaign; evidence receipts still carry the original absolute paths.
_ARCHIVE = Path("paper-write/simulator-archive/build")
_ORIGINAL = Path("Simulator/build")


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relocated(path: str, archive_root: Path | None = None) -> Path:
    """Point a receipt written under `Simulator/build` at the archived copy."""
    text = str(path)
    marker = str(_ORIGINAL).replace("/", "\\")
    if marker in text and archive_root is not None:
        tail = text.split(marker, 1)[1]
        return Path(archive_root) / tail.lstrip("\\/")
    return Path(text)


def trial_values(native: Mapping, action_index: int | None) -> list[Fraction]:
    """Every timed repeat of one action in one process, as ns per repeat."""
    groups = native.get("groups") or []
    trials = None
    if groups:
        for group in groups:
            if action_index is None or group.get("action_index") == action_index:
                trials = group.get("trials") or []
                break
    if trials is None:
        trials = native.get("trials") or []
    return [Fraction(int(t["wall_ns"]), int(t.get("repeats") or 1)) for t in trials if t.get("wall_ns") is not None]


@dataclass(frozen=True)
class Program:
    workload: str
    strategy: int
    keys: tuple[str, ...]

    @property
    def label(self):
        return f"{self.workload}/s{self.strategy}"


@dataclass(frozen=True)
class Instance:
    """One decision: a workload and the strategies the tool may choose among."""

    workload: str
    programs: tuple[Program, ...]
    measured_ns: Mapping[int, Fraction]

    def predicted(self, prices: Mapping[str, float]) -> dict[int, float]:
        return {p.strategy: sum(prices[key] for key in p.keys) for p in self.programs}

    def margins(self, prices: Mapping[str, float]):
        costs = self.predicted(prices)
        ordered = sorted(costs.items(), key=lambda item: (item[1], item[0]))
        best, first = ordered[0]
        second_value = ordered[1][1] if len(ordered) > 1 else first
        margin = (second_value - first) / first if first else 0.0
        return best, ordered[1][0] if len(ordered) > 1 else None, margin, costs

    def regret(self, strategy: int) -> float:
        values = self.measured_ns
        best = min(values.values())
        return float((values[strategy] - best) / best)


def load_campaign(root: Path, archive_root: Path | None = None) -> tuple[dict[str, float], dict[str, list[Fraction]], list[Instance]]:
    """Prices, per-price samples and the decision instances of a frozen campaign."""
    root = Path(root)
    frozen = read(root / "profile-frozen.json")
    prices: dict[str, float] = {}
    samples: dict[str, list[Fraction]] = {}
    for entry in frozen["entries"]:
        key = entry["key"]
        prices[key] = float(Fraction(entry["mean_ns"]))
        values: list[Fraction] = []
        for evidence in entry.get("evidence", []):
            path = relocated(evidence["native"]["path"], archive_root)
            if not path.exists():
                continue
            native = read(path)
            values.extend(trial_values(native, evidence.get("action_index")))
        samples[key] = values

    predictions = read(root / "predictions.json")
    by_workload: dict[str, list[Program]] = {}
    for row in predictions:
        by_workload.setdefault(row["workload"], []).append(
            Program(row["workload"], int(row["strategy"]), tuple(row["keys"])))
    measured = measured_means(root, archive_root)
    instances = [Instance(workload, tuple(sorted(programs, key=lambda p: p.strategy)),
                          measured[workload])
                 for workload, programs in sorted(by_workload.items())]
    return prices, samples, instances


def measured_means(root: Path, archive_root: Path | None = None) -> dict[str, dict[int, Fraction]]:
    """Process-mean completion time per program, averaged over the heldout rounds."""
    root = Path(root)
    rows: dict[str, list[Fraction]] = {}
    for execution in read(root / "heldout-executions.json"):
        if execution.get("status") != "completed":
            continue
        native = read(relocated(execution["native"]["path"], archive_root))
        process = [Fraction(int(t["wall_ns"]), int(t.get("repeats") or 1))
                   for t in native.get("trials", []) if t.get("wall_ns") is not None]
        if not process:
            continue
        program_key = execution["job"]["program_key"]
        rows.setdefault(program_key, []).append(sum(process, Fraction()) / len(process))
    result: dict[str, dict[int, Fraction]] = {}
    for program_key, process_means in rows.items():
        workload, strategy = program_key.rsplit("-s", 1)
        strategy = int(strategy.split("-")[0])
        result.setdefault(workload, {})[strategy] = sum(process_means, Fraction()) / len(process_means)
    return result


def bootstrap_prices(prices, samples, rng: random.Random) -> dict[str, float]:
    """Resample every price from its own measured repeats."""
    out = {}
    for key, value in prices.items():
        values = samples.get(key) or []
        if len(values) < 2:
            out[key] = value
            continue
        draw = [values[rng.randrange(len(values))] for _ in range(len(values))]
        out[key] = float(sum(draw, Fraction()) / len(draw))
    return out


def scaled_prices(prices, epsilon: float, rng: random.Random | None = None,
                  common: bool = False) -> dict[str, float]:
    """Independent per-price error bounded by `epsilon`; common mode shares one draw."""
    if common:
        factor = 1.0 + rng.uniform(-epsilon, epsilon) if rng else 1.0 + epsilon
        return {key: value * factor for key, value in prices.items()}
    if rng is None:
        return {key: value * (1.0 + epsilon) for key, value in prices.items()}
    return {key: value * (1.0 + rng.uniform(-epsilon, epsilon)) for key, value in prices.items()}


def common_mode_cannot_flip(instance: Instance, prices, epsilon: float) -> bool:
    """Every price times one factor multiplies every total by that same factor."""
    best, _, _, _ = instance.margins(prices)
    for sign in (-1.0, 1.0):
        scaled = {key: value * (1.0 + sign * epsilon) for key, value in prices.items()}
        if instance.margins(scaled)[0] != best:
            return False
    return True


@dataclass(frozen=True)
class FlipResult:
    workload: str
    base_choice: int
    runner_up: int | None
    margin: float
    analytic_radius: float
    flips: int
    draws: int
    regrets: tuple[float, ...]
    choices: Mapping[int, int]

    def as_dict(self):
        ordered = sorted(self.regrets)
        return dict(
            workload=self.workload, base_choice=self.base_choice,
            runner_up=self.runner_up, margin=self.margin,
            analytic_radius=self.analytic_radius,
            flips=self.flips, draws=self.draws,
            flip_rate=self.flips / self.draws if self.draws else 0.0,
            choice_counts={str(k): v for k, v in sorted(self.choices.items())},
            regret_median=(ordered[len(ordered) // 2] if ordered else 0.0),
            regret_max=(ordered[-1] if ordered else 0.0),
        )


def flip_study(instance: Instance, draw_prices, draws: int, rng: random.Random) -> FlipResult:
    best, runner_up, margin, _ = instance.margins(draw_prices[0])
    flips = 0
    regrets: list[float] = []
    choices: dict[int, int] = {}
    for prices in draw_prices:
        choice = instance.margins(prices)[0]
        choices[choice] = choices.get(choice, 0) + 1
        if choice != best:
            flips += 1
            regrets.append(instance.regret(choice))
    # An ordering survives per-price errors bounded by eps when
    # T1(1+eps) < T2(1-eps), i.e. eps < (T2-T1)/(T1+T2) = m/(2+m).
    return FlipResult(instance.workload, best, runner_up, margin,
                      margin / (2.0 + margin), flips, draws, tuple(regrets), choices)


def sweep(instances: Sequence[Instance], prices, samples, epsilons: Iterable[float],
          draws: int, seed: int) -> dict:
    """Flip rate against an error bound, for bootstrapped and per-price errors."""
    rng = random.Random(seed)
    per_error = {}
    for epsilon in epsilons:
        draws_prices = [scaled_prices(prices, epsilon, rng) for _ in range(draws)]
        per_error[epsilon] = [flip_study(i, draws_prices, draws, rng).as_dict() for i in instances]
    bootstrap = []
    draws_prices = [bootstrap_prices(prices, samples, rng) for _ in range(draws)]
    for instance in instances:
        bootstrap.append(flip_study(instance, draws_prices, draws, rng).as_dict())
    return dict(per_error={f"{e:.4f}": rows for e, rows in per_error.items()},
                bootstrap=bootstrap)
