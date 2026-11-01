"""Run the frozen finite serving extension on one shared tensor runtime.

python experiments/serving_acceptance/run.py --out build/serving-acceptance-v1
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
for directory in (ROOT,ROOT/"src",ROOT/"mapping",ROOT/"modeling"):
    sys.path.insert(0,str(directory))

from experiments import strategy_fidelity as ref
from experiments.ir_acceptance.run import frozen_evidence
from llm_simulator import Unsupported

MANIFEST_SHA256="bf7953057c9ed0575df2802b51e0fa3141ed0cd23b8d63058bab9c6471aa6745"
PRIOR_MANIFEST_SHA256="cfea763f9c30406f23190610fbdbf8bdff8181a62afd2ab6779809b27e59a882"
COUNTS=dict(positive=6,combinations=4,negative=4,causal=2)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_snapshot():
    paths={ROOT/"experiments/strategy_fidelity.py"}
    for folder in ("src/llm_simulator","mapping/tensor_mapping","modeling/llm_infer_model",
                   "experiments/ir_acceptance","experiments/serving_acceptance","examples/library","tests","mapping/tests","modeling/tests"):
        paths.update((ROOT/folder).rglob("*.py"))
    for folder in ("experiments/ir_acceptance","experiments/serving_acceptance","mapping/examples","modeling/tests/fixtures"):
        paths.update((ROOT/folder).rglob("*.json"))
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}


def run(output):
    output=Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("--out must name a new or empty directory")
    manifest_path=Path(__file__).with_name("manifest.json")
    ref.equal(sha(manifest_path),MANIFEST_SHA256,"frozen serving requirements")
    ref.equal(sha(ROOT/"experiments/ir_acceptance/manifest.json"),PRIOR_MANIFEST_SHA256,"prior frozen 38-case requirements")
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    for group,count in COUNTS.items():ref.equal(len(manifest[group]),count,"fixed serving "+group+" denominator")
    before=source_snapshot();freeze_before=frozen_evidence()
    ref.require(all(r["unchanged"] for r in freeze_before),"performance evidence was changed before campaign")
    output.mkdir(parents=True,exist_ok=True)
    (output/"manifest.json").write_bytes(manifest_path.read_bytes())
    ref.write_json(output/"source-manifest-before.json",before)
    ref.write_json(output/"frozen-evidence-before.json",freeze_before)
    from experiments.serving_acceptance.cases import CASES
    expected_ids={r["id"] for g in COUNTS for r in manifest[g]}
    ref.equal(set(CASES),expected_ids,"adapter registration must exactly match all 16 fixed requirements")
    groups={}
    for group,count in COUNTS.items():
        print(f"Running {group}: {count} fixed requirements",flush=True)
        rows=[]
        for requirement in manifest[group]:
            identifier=requirement["id"];folder=output/group/identifier;folder.mkdir(parents=True)
            # Persist the independent requirement before any compiler/run work.
            ref.write_json(folder/"requirement.json",requirement)
            try:
                evidence=CASES[identifier](folder)
                status=evidence.pop("_status","pass")
                reason=evidence.pop("reason","All fixed semantic obligations checked.")
            except Unsupported as error:
                status="unsupported";reason=str(error)
                evidence=dict(exception=type(error).__name__,message=str(error),traceback=traceback.format_exc())
            except Exception as error:
                status="fail";reason=f"{type(error).__name__}: {error}"
                evidence=dict(exception=type(error).__name__,message=str(error),traceback=traceback.format_exc())
            ref.require(status in {"pass","fail","unsupported","unverified"},"unknown serving result status")
            row=dict(id=identifier,name=requirement["name"],requirement=requirement["requirement"],
                status=status,reason=reason,evidence=evidence,artifacts=str(folder))
            ref.write_json(folder/"result.json",row);rows.append(row)
            print(f"{identifier}: {status}",flush=True)
        groups[group]=rows;ref.write_json(output/(group+"-results.json"),rows)
    after=source_snapshot();freeze_after=frozen_evidence();imported=ref.imported_sources()
    for row in imported:
        relative=Path(row["path"]).resolve().relative_to(ROOT).as_posix()
        ref.equal(before.get(relative),row["sha256"],"imported source hash bound before campaign")
    report=dict(schema="serving-acceptance-results-1",manifest_version=manifest["version"],
        manifest_sha256=MANIFEST_SHA256,prior_manifest_sha256=PRIOR_MANIFEST_SHA256,
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),source_unchanged=before==after,
        frozen_evidence=freeze_after,hardware_accuracy_claimed=False,blind_heldout_study=False,
        groups=groups,metrics={group:dict(numerator=sum(r["status"]=="pass" for r in rows),
            denominator=COUNTS[group],status_counts=dict(Counter(r["status"] for r in rows)))
            for group,rows in groups.items()},qualifications=manifest["excluded"])
    ref.write_json(output/"source-manifest-after.json",after)
    ref.write_json(output/"imported-modules.json",imported)
    ref.write_json(output/"report.json",report)
    lines=["# Single-CPU/single-GPU serving semantics acceptance","",
        "This fixed extension uses synthetic prices. It does not establish hardware accuracy, universal strategy expressiveness, or multi-GPU support.","",
        "The original 38-case IR manifest is unchanged and evaluated separately.","",
        "| Group | Passed / fixed total |","|---|---|"]
    for group,metric in report["metrics"].items():lines.append(f"| {group} | {metric['numerator']}/{metric['denominator']} |")
    for group,rows in groups.items():
        lines.extend(["",f"## {group}","","| Case | Status | Reason |","|---|---|---|"])
        for row in rows:lines.append(f"| {row['id']} {row['name']} | {row['status']} | {row['reason'].replace('|','/').replace(chr(10),' ')} |")
    (output/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")
    ref.require(before==after,"source/reference bytes changed during campaign")
    ref.require(all(r["unchanged"] for r in freeze_after),"frozen performance evidence changed")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    report=run(parser.parse_args().out)
    print(json.dumps(report["metrics"],indent=2),flush=True)
