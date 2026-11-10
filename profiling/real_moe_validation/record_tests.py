"""Record a software-only test run as the freeze receipt the campaign accepts.

`runtime_window_campaign.freeze` refuses to run unless it is handed a receipt of
a passing software test run that performed no hardware execution. The receipt is
not a formality: it names the exact command, its exit status and the hashes of
the test sources, so a cohort cannot be frozen behind a test run that happened
somewhere else.

    python -m profiling.real_moe_validation.record_tests --out <dir>/software-tests.json

It runs the real-MoE software suites with `unittest`, captures stdout/stderr, and
writes the receipt. `--tests` overrides the module list.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import subprocess
import sys
from pathlib import Path

from .dataset import write
from .runtime_window_io import receipt

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TESTS = ("profiling.real_moe_validation.test_runtime_windows",
                 "profiling.real_moe_validation.test_runtime_window_io",
                 "profiling.real_moe_validation.test_runtime_window_predict")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tests", nargs="+", default=list(DEFAULT_TESTS))
    args = parser.parse_args(argv)

    command = [sys.executable, "-m", "unittest", *args.tests]
    started = datetime.now(timezone.utc).isoformat()
    done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                          env={**__import__("os").environ})
    sources = []
    for module in args.tests:
        path = ROOT / (module.replace(".", "/") + ".py")
        if path.is_file():
            sources.append(receipt(path))
    write(args.out, dict(schema="real-moe-software-test-receipt-1",
                         command=command, started_at_utc=started,
                         finished_at_utc=datetime.now(timezone.utc).isoformat(),
                         returncode=done.returncode, hardware_executed=False,
                         stdout=done.stdout, stderr=done.stderr,
                         source_receipts=sources))
    print(json.dumps(dict(out=str(args.out), returncode=done.returncode,
                          tests=list(args.tests))))
    return 0 if done.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
