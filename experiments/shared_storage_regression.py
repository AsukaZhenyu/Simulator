"""Compare every old separate-memory outcome and event hash with its rerun."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

IGNORED_METADATA_FIELDS = frozenset({
    "declared_inputs_sha256",  # New optional Hardware fields change serialization.
    "compiled_ir_audit_sha256",  # Runtime source identity is deliberately source-bound.
    "harness_wall_seconds",  # Host wall time is not a simulated result.
})


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_records(baseline, current, *, expected=352):
    def indexed(rows):
        result = {(r["scenario_id"], r["candidate_id"]): r for r in rows}
        if len(result) != len(rows):
            raise ValueError("duplicate scenario/candidate key")
        return result
    old, new = indexed(baseline), indexed(current)
    missing, extra = sorted(old.keys()-new.keys()), sorted(new.keys()-old.keys())
    mismatches, identical, events = [], 0, 0
    for key in sorted(old.keys() & new.keys()):
        differences = {name: {"old": value, "new": new[key].get(name)}
                       for name, value in old[key].items()
                       if name not in IGNORED_METADATA_FIELDS and new[key].get(name) != value}
        if differences:
            mismatches.append(dict(scenario_id=key[0], candidate_id=key[1], fields=differences))
        else:
            identical += 1
        events += old[key].get("event_sha256") == new[key].get("event_sha256")
    return dict(expected_cases=expected, baseline_cases=len(old), current_cases=len(new),
        compared_cases=len(old.keys() & new.keys()), identical_outcomes=identical,
        identical_event_hashes=events, metric_mismatches=mismatches, missing_cases=missing,
        extra_cases=extra, ignored_metadata_fields=sorted(IGNORED_METADATA_FIELDS),
        passed=(len(old) == len(new) == expected and not missing and not extra and not mismatches))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    comparison = compare_records(read(args.baseline/"records.json"), read(args.current/"records.json"))
    completion = read(args.current/"completion.json")
    before, after = (read(args.current/f"source-manifest-{suffix}.json") for suffix in ("before", "after"))
    original = read(args.baseline/"source-manifest-before.json")
    formulas = {name: digest for name, digest in original.items()
                if name.startswith("modeling/llm_infer_model/performance/")}
    comparison["formula_files"] = formulas
    comparison["formula_files_unchanged"] = bool(formulas) and all(after.get(name) == digest for name, digest in formulas.items())
    comparison["source_manifest_unchanged_during_campaign"] = before == after
    old_protocol, new_protocol = read(args.baseline/"protocol.json"), read(args.current/"protocol.json")
    comparison["frozen_accuracy_evidence_unchanged"] = (
        old_protocol["frozen_accuracy_evidence_before"] == new_protocol["frozen_accuracy_evidence_before"])
    # Old hardware values and workloads must still mean the same thing despite
    # the new optional dataclass keys and resulting input hash differences.
    declarations = []
    new_catalogue = {r["spec"]["id"]: r for r in new_protocol["catalogue"]["scenarios"]}
    for row in old_protocol["catalogue"]["scenarios"]:
        current = new_catalogue.get(row["spec"]["id"], {})
        same = all(current.get(key) == row[key] for key in ("spec", "model", "workload"))
        same = same and all(current.get("hardware", {}).get(k) == v for k, v in row["hardware"].items())
        same = same and current.get("hardware", {}).get("memory_mode", "separate") == "separate"
        declarations.append(dict(scenario_id=row["spec"]["id"], equivalent=same))
    comparison["declarations"] = declarations
    comparison["passed"] = comparison["passed"] and all((
        comparison["formula_files_unchanged"], comparison["source_manifest_unchanged_during_campaign"],
        comparison["frozen_accuracy_evidence_unchanged"], all(r["equivalent"] for r in declarations),
        completion.get("complete") is True, completion.get("goal_reached") == 352,
    ))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in comparison.items() if k not in ("formula_files", "declarations")}, indent=2))
    return 0 if comparison["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
