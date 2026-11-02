"""New-file-only offline state-response commands; collectors are explicit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT, ROOT / "modeling", ROOT / "profiling", ROOT / "mapping", ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from .analysis import evaluate_heldout, freeze_profile, import_legacy_campaign


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "collect":
        from .collect import main as collect_main
        return collect_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    legacy = sub.add_parser("import-legacy")
    legacy.add_argument("--campaign", required=True)
    legacy.add_argument("--split", choices=("profile", "heldout"), default="profile")
    legacy.add_argument("--out", required=True)
    freeze = sub.add_parser("freeze")
    for name in ("program", "records", "context", "states", "protocol", "out"):
        freeze.add_argument("--" + name, required=True)
    freeze.add_argument("--frozen-at-utc")
    evaluate = sub.add_parser("evaluate")
    for name in ("program", "frozen", "records", "out"):
        evaluate.add_argument("--" + name, required=True)
    evaluate.add_argument("--barrier")
    args = parser.parse_args(argv)
    if args.command == "import-legacy":
        result = import_legacy_campaign(args.campaign, split=args.split)
    elif args.command == "freeze":
        result = freeze_profile(read(args.program), read(args.records), context=read(args.context),
            required_states=read(args.states), protocol=read(args.protocol),
            frozen_at_utc=args.frozen_at_utc or datetime.now(timezone.utc).isoformat())
    else:
        result = evaluate_heldout(read(args.program), read(args.frozen), read(args.records),
            barrier=None if args.barrier is None else read(args.barrier))
    save(args.out, result)


if __name__ == "__main__":
    main()
