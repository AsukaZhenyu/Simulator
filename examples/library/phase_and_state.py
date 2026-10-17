"""Combine P/D placement, prefetch and external resource changes."""
import argparse
import json
from pathlib import Path

from llm_simulator import DataPreparation, EnvironmentEvent, PhasePlacement
from llm_simulator.demo import make_demo


def configure(plan):
    PhasePlacement(prefill={"projection": "gpu", "attention": "gpu"},
                   decode={"projection": "cpu", "attention": "cpu"}).apply(plan)
    plan.prepare(DataPreparation(layer_id="layer1", destination="vram", trigger="spare"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("phase-state-example"))
    args = parser.parse_args()
    simulator, scheduling = make_demo("dense")
    events = (EnvironmentEvent(t_ns=0, cpu_available=False),
              EnvironmentEvent(t_ns=10000, cpu_available=True),
              EnvironmentEvent(t_ns=15000, vram_external_bytes=1024))
    with simulator.compile(scheduling=scheduling, configure=configure,
            directory=args.out / "project", environment_events=events) as compiled:
        result = compiled.run()
        result.save(args.out / "result")
        print(json.dumps(result.summary(), indent=2))
        return 0 if result.success else 3


if __name__ == "__main__":
    raise SystemExit(main())
