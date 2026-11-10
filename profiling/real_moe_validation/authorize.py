"""Write the root authorization a hardware window requires, and nothing else.

`runtime_window_campaign.collect` will not launch a native process without a
receipt that names this exact manifest, this exact freeze barrier and the jobs
the window covers. The receipt is a human decision -- which window, on what
grounds -- so this script does not invent one: it requires the operator's own
sentence in `--user-authorization` and records the hashes that were current when
the sentence was written, so a later edit to the cohort cannot inherit it.

    python -m profiling.real_moe_validation.authorize --out <cohort> \\
        --user-authorization "run the inventory window now" \\
        --jobs cpu-inventory,hybrid-inventory

Run it again for the heldout jobs after the prediction freeze; the second receipt
also records the price and prediction freeze it was granted against. Because the
receipt is created exclusively, the caller moves the earlier window's receipt to
`ROOT_AUTHORIZATION_<ROLE>.json` first -- the published campaign keeps
`ROOT_AUTHORIZATION_CALIBRATION.json` beside the canonical receipt that way, and
`collect` reads only the canonical name.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import read, write
from .runtime_window_io import require, sha


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", required=True, help="comma-separated frozen job ids")
    parser.add_argument("--user-authorization", required=True,
                        help="the operator's own sentence granting this window")
    args = parser.parse_args(argv)

    out = Path(args.out).resolve()
    manifest = out / "manifest.json"
    barrier = out / "FREEZE_BARRIER.json"
    require(manifest.is_file(), "prepare the cohort before authorizing it")
    require(barrier.is_file(), "freeze the cohort before authorizing hardware")
    declared = read(manifest)
    jobs = [job for job in (x.strip() for x in args.jobs.split(",")) if job]
    require(jobs, "an authorization must name at least one job")
    for job in jobs:
        require(job in declared.get("requests", {}), f"{job} is not a frozen request of this cohort")

    receipt = dict(collection_authorized=True,
                   manifest_sha256=sha(manifest),
                   freeze_barrier_sha256=sha(barrier),
                   jobs=jobs,
                   user_authorization=args.user_authorization)
    for name, field in (("PRICE_FREEZE.json", "price_freeze_sha256"),
                        ("PREDICTION_FREEZE.json", "prediction_freeze_sha256")):
        path = out / name
        if path.is_file():
            receipt[field] = sha(path)
    write(out / "ROOT_AUTHORIZATION_RECEIPT.json", receipt)
    print(json.dumps(dict(out=str(out), jobs=jobs,
                          manifest_sha256=receipt["manifest_sha256"],
                          freeze_barrier_sha256=receipt["freeze_barrier_sha256"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
