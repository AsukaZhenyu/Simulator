"""Read a measured case with provenance; never fill unknown workspace with zero."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from profile_contract import validate_measurement


def query(profile_dir: Path, case_id: str, statistic: str = "median_ns") -> dict:
    if statistic not in {"median_ns", "mean_ns", "p10_ns", "p90_ns"}:
        raise ValueError("unsupported statistic")
    manifest = json.loads((profile_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = json.loads((profile_dir / "measurements.json").read_text(encoding="utf-8"))
    matched = [row for row in rows if row.get("case_id") == case_id]
    if len(matched) != 1:
        raise ValueError(f"expected exactly one measured case {case_id!r}; found {len(matched)}")
    row = matched[0]
    if row.get("status") != "measured":
        raise ValueError(f"case is not a successful measurement: {row.get('status')}")
    native = validate_measurement(row["native"])
    return {"case_id": case_id, "profile_id": manifest["profile_id"], "signature": row["signature"],
            "duration_ns": max(1, math.ceil(native["summary"][statistic])), "statistic": statistic,
            "summary": native["summary"], "workspace_bytes": native.get("workspace_bytes"),
            "workspace_source": native.get("workspace_source"), "raw_output": row["raw_output"],
            "quality_flags": row.get("quality_flags", []),
            "binding_rule": "match the exact backend, shape, dtype and CPU threads; this is not an M1/M2 cost file",
            "capacity_verified": False,
            "environment": manifest["environment"], "software": manifest["software"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--statistic", choices=["median_ns", "mean_ns", "p10_ns", "p90_ns"], default="median_ns")
    args = parser.parse_args()
    try:
        print(json.dumps(query(args.profile, args.case_id, args.statistic), ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
