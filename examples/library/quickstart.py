"""Run after installing the package; no repository-relative data is loaded."""
import argparse
import json
from pathlib import Path

from llm_simulator.demo import make_demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("dense", "moe"), default="dense")
    parser.add_argument("--out", type=Path, default=Path("library-example"))
    args = parser.parse_args()
    simulator, scheduling = make_demo(args.model)
    with simulator.compile(scheduling=scheduling, directory=args.out / "project") as compiled:
        result = compiled.run()
        result.save(args.out / "result")
        print(json.dumps(result.summary(), indent=2))
        return 0 if result.success else 3


if __name__ == "__main__":
    raise SystemExit(main())
