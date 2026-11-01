"""Write-once offline profiling artifacts; never starts hardware workloads."""
import argparse
import json
from pathlib import Path

from .coherent import freeze_coherent_ensemble


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    coherent = commands.add_parser("coherent", help="preserve actual complete-invocation parameter vectors")
    coherent.add_argument("--records", required=True, type=Path)
    coherent.add_argument("--sha256", required=True)
    coherent.add_argument("--samples-per-run-state", required=True, type=int)
    coherent.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = freeze_coherent_ensemble(args.records, args.sha256,
        samples_per_run_state=args.samples_per_run_state)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status="written", output=str(args.output), samples=len(result["samples"]),
                         physical_bindings=len(result["physical_bindings"]), hardware_accuracy_qualified=False)))


if __name__ == "__main__":
    main()
