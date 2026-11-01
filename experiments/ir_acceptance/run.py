"""Run the finite IR baseline without changing production or calibration data.

python experiments/ir_acceptance/run.py --out build/ir-acceptance-v1
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT, ROOT / "src", ROOT / "mapping", ROOT / "modeling"):
    sys.path.insert(0, str(folder))

from experiments import strategy_fidelity as ref

MANIFEST_SHA256 = "cfea763f9c30406f23190610fbdbf8bdff8181a62afd2ab6779809b27e59a882"
COUNTS = dict(basic=12, combinations=12, negative=8, authoring=6)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_snapshot():
    paths = {ROOT / "experiments/strategy_fidelity.py"}
    for directory in ("src/llm_simulator", "mapping/tensor_mapping", "modeling/llm_infer_model",
                      "experiments/ir_acceptance", "mapping/tests", "modeling/tests", "tests", "examples/library"):
        paths.update((ROOT / directory).rglob("*.py"))
    for directory in ("mapping/examples", "modeling/tests/fixtures", "experiments/ir_acceptance"):
        paths.update((ROOT / directory).rglob("*.json"))
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(paths)}


def frozen_evidence():
    record_path = ROOT / "docs/data/research-direction-20261022.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    return [dict(path=row["path"], expected_sha256=row["sha256"], actual_sha256=sha(ROOT / row["path"]),
                 unchanged=sha(ROOT / row["path"]) == row["sha256"])
            for row in record["accuracy"]["evidence"]]


def ratio(numerator, denominator):
    return dict(numerator=numerator, denominator=denominator,
                percent=round(100 * numerator / denominator, 2) if denominator else None)


def summarize(groups, source_unchanged):
    basic, combo, negative, authoring = (groups[key] for key in COUNTS)
    complete = sum(row.get("expression_status") == "complete" for row in basic)
    legacy_only = [row["id"] for row in basic if row.get("expression_status") == "complete" and
                   (row.get("evidence", {}).get("unified_frontend_supported") is False or
                    row.get("evidence", {}).get("shared_composition_frontend_support") is False)]
    faithful = sum(row.get("fidelity_status") == "pass" for row in basic)
    supported = [row for row in basic if row.get("expression_status") == "complete"]
    conditional = sum(row.get("fidelity_status") == "pass" for row in supported)
    legal_combinations = sum(row["status"] == "pass" for row in combo)
    bad_rejected = sum(row["status"] == "pass" for row in negative)
    # All task successes must additionally be bound to unchanged production.
    no_backend_edit = sum(row["status"] == "pass" and
                          row.get("authoring", {}).get("production_files_changed") == 0
                          for row in authoring) if source_unchanged else 0
    paired_lines = []
    for row in authoring:
        value = row.get("authoring", {}).get("authored_nonblank_noncomment_lines")
        if isinstance(value, dict):
            paired_lines.append(dict(id=row["id"], **value))
        elif type(value) is int:
            paired_lines.append(dict(id=row["id"], changed=value))
    lines = [row["changed"] for row in paired_lines if type(row.get("changed")) is int]
    return dict(
        expression=ratio(complete, len(basic)),
        expression_shared_composition_frontend=ratio(complete - len(legacy_only), len(basic)),
        legacy_only_complete_requirements=legacy_only,
        fidelity_fixed_denominator=ratio(faithful, len(basic)),
        fidelity_among_completely_expressed=ratio(conditional, len(supported)),
        legal_composition=ratio(legal_combinations, len(combo)),
        invalid_case_detection=ratio(bad_rejected, len(negative)),
        authoring_success_without_backend_edits=ratio(no_backend_edit, len(authoring)),
        authoring_lines=dict(paired_values=paired_lines, changed_values=lines,
                             changed_median=statistics.median(lines) if lines else None,
                             scope="Task-authored nonblank/noncomment strategy lines, including task-specific helpers; see each artifact."),
        code_compression=dict(value=None, status="not_measured", reason="No fair paired handwritten low-level implementation baseline was established."),
        human_completion_time=dict(value=None, status="not_measured"),
        status_counts={key: dict(Counter(row["status"] for row in rows)) for key, rows in groups.items()},
    )


def markdown(report):
    def fraction(value):
        return f"{value['numerator']}/{value['denominator']} ({value['percent']}%)"
    metrics = report["metrics"]
    lines = ["# IR acceptance baseline", "", f"Manifest: `{report['manifest_version']}`.", "",
        "This is a retrospective finite baseline with synthetic prices, not a blind study, universal strategy coverage, or hardware accuracy validation.", "",
        "| Metric | Result |", "|---|---|",
        f"| Complete expression, all tested entry paths | {fraction(metrics['expression'])} |",
        f"| Complete expression, shared composition frontend | {fraction(metrics['expression_shared_composition_frontend'])} |",
        f"| Compiler fidelity / all fixed basic requirements | {fraction(metrics['fidelity_fixed_denominator'])} |",
        f"| Compiler fidelity / completely expressed subset | {fraction(metrics['fidelity_among_completely_expressed'])} |",
        f"| Legal composition | {fraction(metrics['legal_composition'])} |",
        f"| Invalid-case detection | {fraction(metrics['invalid_case_detection'])} |",
        f"| Authoring tasks without backend changes | {fraction(metrics['authoring_success_without_backend_edits'])} |",
        "| Code compression / human completion time | Not measured |", "",
        "All fixed denominators include gaps and unverified items. Unsupported does not mean illegal; policy dead end does not prove global infeasibility.", "",
        f"Complete requirements available only through a legacy/specialized entry: **{metrics['legacy_only_complete_requirements']}**. These do not establish support in the shared Python/composition frontend.", "",
        f"Production, benchmark and reference bytes unchanged during run: **{report['source_unchanged']}**.",
        f"Frozen calibration evidence unchanged: **{all(row['unchanged'] for row in report['frozen_evidence'])}**.", ""]
    for group, rows in report["groups"].items():
        lines += [f"## {group}", "", "| ID | Requirement | Status | Reason |", "|---|---|---|---|"]
        for row in rows:
            reason = row.get("reason", "").replace("\n", " ").replace("|", "/")
            lines.append(f"| {row['id']} | {row['name']} | {row['status']} | {reason} |")
        lines.append("")
    lines += ["## Interpretation", "",
        "A passing semantic case checks a particular declared strategy under finite inputs. It does not establish arbitrary shape support or all-policy correctness.",
        "The evidence includes declared inputs, independent expectations, execution artifacts and all failures/unsupported cases. Sources and manifest are bound by SHA-256.",
        "The runner does not modify production features or run performance measurements/refits. Each campaign evaluates source bytes frozen at its start; implementation work may occur between campaigns.", ""]
    return "\n".join(lines)


def run(output):
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("--out must name a new or empty directory")
    manifest_file = Path(__file__).with_name("manifest.json")
    ref.equal(sha(manifest_file), MANIFEST_SHA256, "frozen manifest identity")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    for group, count in COUNTS.items():
        ref.equal(len(manifest[group]), count, f"fixed {group} denominator")
    output.mkdir(parents=True, exist_ok=True)
    before = source_snapshot()
    frozen_before = frozen_evidence()
    ref.require(all(row["unchanged"] for row in frozen_before), "accuracy freeze was already changed before campaign")
    # Retain the exact preregistered bytes so the artifact's own hash matches
    # report.manifest_sha256, rather than reformatting equivalent JSON.
    (output / "manifest.json").write_bytes(manifest_file.read_bytes())
    ref.write_json(output / "source-manifest-before.json", before)
    ref.write_json(output / "frozen-evidence-before.json", frozen_before)

    from experiments.ir_acceptance import basic, combinations, negative, authoring
    modules = dict(basic=basic, combinations=combinations, negative=negative, authoring=authoring)
    groups = {}
    for name, module in modules.items():
        print(f"Running {name}: {COUNTS[name]} fixed requirements", flush=True)
        try:
            rows = module.run_group(output / name)
        except Exception as error:
            rows = [dict(id=requirement["id"], status="unverified", expression_status="unverified",
                         fidelity_status="not_tested", reason=f"Group runner failed: {type(error).__name__}: {error}",
                         evidence=dict(traceback=traceback.format_exc()), artifacts=str(output / name))
                    for requirement in manifest[name]]
        expected = {row["id"]: row for row in manifest[name]}
        ref.equal({row["id"] for row in rows}, set(expected), f"all fixed {name} cases reported")
        ref.equal(len(rows), len(expected), "no duplicate requirement IDs")
        for row in rows:
            row["name"] = expected[row["id"]]["name"]
            row["requirement"] = expected[row["id"]]["requirement"]
            ref.require(row["status"] in {"pass", "fail", "partial", "unsupported", "unverified"}, "unknown case status")
        groups[name] = rows
        ref.write_json(output / f"{name}-results.json", rows)
        print(f"{name}: {dict(Counter(row['status'] for row in rows))}", flush=True)

    after = source_snapshot()
    frozen_after = frozen_evidence()
    unchanged = before == after
    imported = ref.imported_sources()
    for row in imported:
        relative = Path(row["path"]).resolve().relative_to(ROOT).as_posix()
        ref.equal(before.get(relative), row["sha256"], "imported source bound to snapshot")
    ref.write_json(output / "source-manifest-after.json", after)
    ref.write_json(output / "imported-modules.json", imported)
    report = dict(schema="ir-acceptance-results-1", manifest_version=manifest["version"],
        manifest_sha256=MANIFEST_SHA256, recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        source_unchanged=unchanged, frozen_evidence=frozen_after,
        hardware_accuracy_claimed=False, blind_heldout_study=False, runner_modifies_production=False,
        implementation_may_differ_from_prior_campaigns=True,
        metrics=summarize(groups, unchanged), groups=groups,
        qualifications=["Finite declared strategy cases, not universal expressiveness.",
                        "Synthetic costs, not hardware performance evidence.",
                        "No matched low-level code baseline or human authoring study.",
                        "Unsupported and unverified cases remain in fixed denominators."])
    ref.write_json(output / "report.json", report)
    (output / "REPORT.md").write_text(markdown(report), encoding="utf-8", newline="\n")
    ref.require(unchanged, "source/reference bytes changed during campaign; results are not accepted")
    ref.require(all(row["unchanged"] for row in frozen_after), "frozen calibration evidence changed")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    options = parser.parse_args()
    report = run(options.out)
    print(json.dumps(report["metrics"], indent=2), flush=True)
    # Coverage gaps are a measured outcome, not a crashed experiment. CI/tests
    # can enforce explicit target gates using report.json without hiding them.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
