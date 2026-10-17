"""Recompute a completed matched-mechanism experiment without hardware."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment as e
from binding import validate_ports, fnv_f32_bytes


def verify(directory):
    directory = Path(directory).resolve()
    manifest = e.document(directory / "manifest.json")
    suite = e.validate_suite(e.document(directory / "suite.json"))
    saved = e.document(directory / "measurements.json")
    expected = {}
    for split in ("profile", "heldout"):
        for case in suite["cases"]:
            for seed in suite[split + "_seeds"]:
                for repeat in range(suite["repeats"]):
                    rid = f"{split}-{case['id']}-s{seed}-r{repeat}"
                    expected[rid] = ("compute", case, seed, split, repeat)
        for case in suite["transfers"]:
            for repeat in range(suite["repeats"]):
                rid = f"{split}-{case['id']}-r{repeat}"
                expected[rid] = ("transfer", case, None, split, repeat)
    if not isinstance(saved, list) or len(saved) != len(expected) or {r["id"] for r in saved} != set(expected):
        raise ValueError("declared native row coverage mismatch")
    rebuilt = []
    for original in saved:
        row = dict(original)
        kind, case, seed, split, repeat = expected[row["id"]]
        if any(row[key] != value for key, value in (("kind", kind), ("case", case), ("seed", seed), ("split", split), ("repeat", repeat))):
            raise ValueError("saved row declaration mismatch")
        raw_file = directory / "raw" / (row["id"] + ".json")
        if e.sha(raw_file) != row["raw_sha256"]:
            raise ValueError("raw evidence hash mismatch")
        raw = e.document(raw_file)
        command = e.compute_command(row["command"][0], case, suite["compute_protocol"], seed) if kind == "compute" else e.transfer_command(row["command"][0], case)
        if raw["command"] != command or row["command"] != command:
            raise ValueError("actual native command mismatch")
        if row["status"] != "measured":
            # Failure evidence stays a failure; it cannot become a valid price.
            rebuilt.append(row)
            continue
        if raw["returncode"] != 0 or raw["observed_child_flags"] != manifest["observed_child_flags"]:
            raise ValueError("raw status/runtime flags mismatch")
        payload = json.loads(raw["stdout"])
        e.validate_native(row, payload, suite)
        metrics = payload["wall_ns"] if kind == "compute" else {"wall": payload["wall_ns"]}
        if row["metrics"] != metrics:
            raise ValueError("raw sample binding mismatch")
        if kind == "compute":
            validate_ports(case, payload)
            for key in ("ports", "workspaces", "checks"):
                if row[key] != payload[key]:
                    raise ValueError("raw numerical/workspace binding mismatch")
            if row["fixture_hashes"] != e.fixture_hashes(payload):
                raise ValueError("fixture hashes mismatch")
            for name, info in row["fixture_hashes"].items():
                path = directory / "fixtures" / (row["id"] + "-" + name + ".f32le")
                if path.stat().st_size != info["storage_bytes"] or e.sha(path) != info["sha256_f32_le"]:
                    raise ValueError("materialized physical fixture mismatch")
                # Native digest independently binds its pre-serialization F32
                # bytes to the reconstructed physical fixture.
                if payload["ports"][name]["input_fixture_fnv1a64"] != fnv_f32_bytes(path.read_bytes()):
                    raise ValueError("native physical F32 byte identity mismatch")
        rebuilt.append(row)
    model = e.fit(rebuilt)
    if model != e.document(directory / "prediction-model.json") or e.sha(directory / "prediction-model.json") != manifest["model_frozen_before_heldout_sha256"]:
        raise ValueError("profile-only frozen prediction model mismatch")
    report = e.evaluate(rebuilt, model, suite)
    if report != e.document(directory / "report.json"):
        raise ValueError("heldout statistics/quality report mismatch")
    if manifest["inventory_before"] != manifest["inventory_after"] or manifest["assets_unchanged"] is not True:
        raise ValueError("experiment source/binary/runtime/frozen inventory changed")
    # Verify the exact executed source snapshots; changing current source does
    # not rewrite the old experiment. Statistical code changes require a new run.
    current_file = str(Path(e.__file__).resolve())
    if current_file not in manifest["inventory_before"] or e.sha(directory / "source" / "experiment.py") != manifest["inventory_before"][current_file]:
        raise ValueError("executed experiment source snapshot mismatch")
    if e.sha(e.__file__) != manifest["inventory_before"][current_file]:
        raise ValueError("current statistical implementation differs from executed source")
    return {"schema": "matched-mechanisms-verification-0.1", "verified": True,
            "native_rows": len(saved), "numeric_execution_valid": report["numeric_execution_valid"],
            "metrics": len(report["results"]), "calibrated_metrics": sum(r["calibrated"] for r in report["results"]),
            "all_declared_metrics_calibrated": report["all_declared_metrics_calibrated"],
            "gpu_workspace_known": False, "general_arbitration_model_calibrated": False,
            "report_sha256": e.sha(directory / "report.json")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    try:
        if args.output.exists():
            raise ValueError("verification output must be new")
        result = verify(args.experiment)
        e.dump(args.output, result)
        print(e.canonical(result)); return 0
    except (OSError, ValueError, KeyError) as exc:
        print("matched verification: " + str(exc), file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
