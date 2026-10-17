"""Numerical/protocol smoke; never used as calibration or heldout evidence."""
import argparse
import copy
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_native as runner
import analysis as a


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("smoke output must be new")
    suite = runner.document(HERE / "transform_suite.json")
    suite["protocol"].update(samples=3, warmup=1, min_warmup_ms=0)
    cases = [suite["cases"][index] for index in (0, 4, 5, 6)]
    env, flags = runner.child_context(args.runtime_dir)
    lock = HERE.parent / ".measurement.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(str(os.getpid()) + "\n")
    try:
        args.output.mkdir(parents=True)
        binary_before = a.sha(args.binary)
        records = []
        for index, case in enumerate(cases):
            row = dict(case=case, seed=101, split="smoke", id=case["id"], repeat=0)
            command = runner.command(args.binary.resolve(), case, suite["protocol"], row["seed"])
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=240)
            record = dict(command=command, returncode=result.returncode, stdout=result.stdout,
                          stderr=result.stderr, observed_child_flags=flags)
            a.dump(args.output / (f"{index:02d}-" + case["id"] + ".json"), record)
            if result.returncode:
                raise ValueError("native smoke failed: " + case["id"])
            payload = __import__("json").loads(result.stdout)
            runner.validate_native(row, payload, suite)
            records.append(dict(case_id=case["id"], all_checks_passed=payload["all_checks_passed"],
                                metric_count=len(payload["wall_ns"])))
            print(case["id"], "passed", flush=True)
        assert binary_before == a.sha(args.binary)
        a.dump(args.output / "summary.json", dict(purpose="numerical_protocol_smoke_not_calibration",
            completed_utc=datetime.now(timezone.utc).isoformat(), binary_sha256=binary_before,
            all_passed=True, records=records))
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
