"""Rebuild numerical, physical-signature and quality evidence without hardware."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from profiling.reconfiguration import run as r
from profiling.matched_mechanisms import verify as matched_verify


def verify(directory):
    directory = Path(directory).resolve()
    suite = r.validate_suite(r.matched.document(directory / "suite.json"))
    if r.matched.document(directory / "ffn-suite.json") != suite["ffn_suite"]:
        raise ValueError("executed FFN suite differs from declared suite")
    ffn = matched_verify.verify(directory / "ffn")
    manifest = r.matched.document(directory / "manifest.json")
    before, after = manifest["inventory_before"], manifest["inventory_after"]
    if before != after or manifest.get("assets_unchanged") is not True or manifest.get("status") != "completed":
        raise ValueError("source/native assets were changed or measurement is incomplete")
    for path, digest in before.items():
        source = Path(path)
        if source.parent == r.ROOT or source.parent == r.ROOT / "native":
            if r.matched.sha(directory / "source" / source.name) != digest:
                raise ValueError("executed reconfiguration source snapshot mismatch")
            if r.matched.sha(source) != digest:
                raise ValueError("current implementation differs from executed source; rerun required")
    rows = r.matched.document(directory / "allocator-measurements.json")
    expected = {f"{c['id']}-r{i}": (c, i) for c in suite["allocator_cases"] for i in range(suite["allocator_repeats"])}
    if len(rows) != len(expected) or {row["id"] for row in rows} != set(expected):
        raise ValueError("allocator coverage mismatch")
    for row in rows:
        case, repeat = expected[row["id"]]
        if row["case"] != case or row["repeat"] != repeat:
            raise ValueError("allocator row declaration mismatch")
        raw_path = directory / "raw" / (row["id"] + ".json")
        if r.matched.sha(raw_path) != row["raw_sha256"]:
            raise ValueError("raw allocator evidence hash mismatch")
        raw = r.matched.document(raw_path)
        if row["command"] != r.allocator_command(row["command"][0], case) or raw["command"] != row["command"]:
            raise ValueError("native allocator executed command mismatch")
        if row["status"] == "measured":
            if raw["returncode"] != 0:
                raise ValueError("measured allocator row has nonzero status")
            payload = r.validate_allocator(case, json.loads(raw["stdout"]))
            if row["payload"] != payload:
                raise ValueError("saved allocator payload differs from raw")
        elif row["status"] != "failed":
            raise ValueError("unknown allocator row status")
    report = r.summarize(directory)
    if report != r.matched.document(directory / "report.json"):
        raise ValueError("recomputed reconfiguration quality report mismatch")
    return {"schema": "reconfiguration-measurement-verification-0.1", "verified": True,
            "ffn_verification": ffn, "allocator_native_rows": len(rows),
            "ffn_pair_quality_passed": sum(c["both_latency_gates_passed"] for c in report["comparisons"]),
            "ffn_pair_count": len(report["comparisons"]),
            "all_input_nondegeneracy_checks_passed": report["all_input_nondegeneracy_checks_passed"],
            "allocator_api_metrics_stable": sum(c["stable_api_observation"] for c in report["allocator_api_metrics"]),
            "allocator_api_metrics_total": len(report["allocator_api_metrics"]),
            "gpu_capacity_proved": False, "mapper_control_cost_ns": None,
            "report_sha256": r.matched.sha(directory / "report.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("verification output must be new")
        result = verify(args.experiment)
        r.matched.dump(args.output, result)
        print(r.matched.canonical(result))
        return 0
    except (OSError, ValueError) as exc:
        print("reconfiguration verification: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
