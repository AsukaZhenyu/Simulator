"""Offline service-response freeze/evaluation; no native actors are launched."""
import argparse
import json
from pathlib import Path

from profiling.state_calibration import run
from profiling.state_response import collect as common
from .analysis import freeze_profile, evaluate_frozen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--out", type=Path, required=True, help="source-bound campaign directory")
        command.add_argument("--output", type=Path, required=True, help="new artifact path; never overwritten")
        if name == "evaluate":
            command.add_argument("--frozen", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("output already exists; retain it and select a new path")
    manifest = common.load_manifest(args.out)
    if args.command == "freeze":
        if (args.out / "heldout-barrier.json").exists():
            raise ValueError("cannot refit after the heldout barrier")
        records = run.read(args.out / "profile-records.json")
        result = freeze_profile(records, protocol=manifest["protocol"]["service_response"],
                                frozen_at_utc=run.utc(), verify_sources=True)
        run.save(args.output, result)
        summary = dict(status="frozen", artifact_sha256=result["artifact_sha256"],
            model_parameters_available=result["model_parameters_available"],
            original_action_failed_cells=result["original_action_failed_cells"],
            worker_proxy_source_qualified=result["worker_proxy_source_qualified"],
            source_qualified=result["source_qualified"])
    else:
        frozen = run.read(args.frozen)
        barrier = run.read(args.out / "heldout-barrier.json")
        if run.sha(args.frozen) != barrier["frozen_file_sha256"]:
            raise ValueError("frozen file differs from the pre-launch heldout barrier")
        if run.sha(args.out / "manifest.json") != barrier["manifest_sha256"]:
            raise ValueError("manifest differs from the pre-launch heldout barrier")
        result = evaluate_frozen(frozen, run.read(args.out / "heldout-records.json"),
                                 barrier=barrier, verify_sources=True)
        run.save(args.output, result)
        summary = dict(status="evaluated", primary=result["chosen_model"],
                       primary_summary=result["primary_summary"], accepted=result["accepted"],
                       accuracy_qualified=result["accuracy_qualified"])
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
