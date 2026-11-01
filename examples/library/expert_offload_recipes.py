"""Compile core MoE policies; explicitly analytical, not local measurements.

The same expert_recipe(...) declaration can be passed to ModelPlan with measured
costs. This runnable demo deliberately needs no generated profile files.
"""
import argparse
import json
from pathlib import Path

from llm_simulator.demo import make_demo
from llm_simulator.expert_recipes import (
    EXPERT_RECIPE_NAMES, expert_recipe, expert_recipe_scope,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weight-budget-bytes", type=int, default=24 * 1024,
                        help="all GPU weight payload, not total VRAM (demo default: 24 KiB)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for name in EXPERT_RECIPE_NAMES:
        simulator, _ = make_demo("moe")
        scheduling = expert_recipe(name, gpu_weight_budget_bytes=args.weight_budget_bytes)
        with simulator.compile(scheduling=scheduling, directory=args.out / name / "project") as compiled:
            result = compiled.run()
            result.save(args.out / name / "result")
            rows.append({**expert_recipe_scope(name), **result.summary()})
    (args.out / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(rows, indent=2))
    return 0 if all(row["success"] for row in rows) else 3


if __name__ == "__main__":
    raise SystemExit(main())
