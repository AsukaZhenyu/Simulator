"""Installed command line; no repository path or saved profile is required."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from llm_infer_model.tensor.spec import InvalidInput, SpecError


def main(argv=None):
    from . import __version__
    parser = argparse.ArgumentParser(prog="llm-sim", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="run a Dense/MoE example with declared analytical costs")
    demo.add_argument("--model", choices=("dense", "moe"), default="dense")
    demo.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    demo.add_argument("--out", type=Path, help="new directory for persistent project and detailed results")
    args = parser.parse_args(argv)
    try:
        from .demo import make_demo
        simulator, scheduling = make_demo(args.model, device=args.device)
        output = args.out.resolve() if args.out else None
        if output is not None:
            if output.exists() and (not output.is_dir() or any(output.iterdir())):
                raise InvalidInput("--out must be new or empty")
            output.mkdir(parents=True, exist_ok=True)
        with simulator.compile(scheduling=scheduling,
                directory=output / "project" if output else None) as compiled:
            result = compiled.run()
            if output:
                result.save(output / "result")
            print(json.dumps(result.summary(), ensure_ascii=False, allow_nan=False))
            return 0 if result.success else 3
    except (SpecError, OSError, ValueError) as error:
        print(f"llm-sim: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
