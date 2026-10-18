"""Replay overlap evidence, native contracts, frozen profile costs and fixed IR."""
import argparse
import json
from pathlib import Path
import run as protocol
import contract as c


def verify(directory):
    root=Path(directory).resolve();manifest=protocol.load(root/"manifest.json")
    suite=c.validate_suite(protocol.load(root/"suite.json"));plan=protocol.load(root/"plan.json");rows=protocol.load(root/"measurements.json")
    if protocol.jobs(suite)!=plan or c.sha(root/"plan.json")!=manifest["plan_sha256_before_collection"]:raise ValueError("frozen declaration/plan differs")
    if [r["id"]for r in rows]!=[r["id"]for r in plan]:raise ValueError("declared process/failure coverage differs")
    if manifest.get("assets_unchanged")is not True or manifest["inventory_before"]!=manifest["inventory_after"]:raise ValueError("native/runtime/source asset drift")
    protocol.runtime_manifest.verify_recorded(manifest.get("private_runtime_builds",[]),manifest["inventory_before"])
    for source,item in manifest["source_snapshots"].items():
        if c.sha(root/item["path"])!=item["sha256"]or manifest["inventory_before"][source]!=item["sha256"]:raise ValueError("source snapshot differs")
    model=protocol.load(root/"frozen-model.json")
    if c.sha(root/"frozen-model.json")!=manifest["model_frozen_before_heldout_sha256"]:raise ValueError("frozen model hash differs")
    probe_sha=next(value for path,value in manifest["inventory_before"].items()if Path(path).parent.name=="overlap_accuracy"and Path(path).name=="compute.cpp")
    by_id={job["id"]:job for job in plan}
    binary=Path(manifest["bound_binary"])
    if not binary.is_absolute()or str(binary)not in manifest["inventory_before"]:raise ValueError("bound executable missing from campaign inventory")
    for row in rows:
        job=by_id[row["id"]]
        if any(row.get(key)!=value for key,value in job.items()):raise ValueError("row immutable split/case/seed/repeat fields differ from frozen job")
        expected_command=protocol.command(binary,job["case"],suite["protocol"],job["seed"])
        rawpath=Path(row["raw_path"]).resolve()
        if not rawpath.is_relative_to(root)or c.sha(rawpath)!=row["raw_sha256"]:raise ValueError("raw path/hash differs")
        raw=protocol.load(rawpath)
        if raw["command"]!=expected_command or row["command"]!=expected_command:raise ValueError("row/raw command differs from independently regenerated planned command")
        runtime_hashes=protocol.runtime_manifest.hash_bindings(manifest.get("private_runtime_builds",[]))
        if raw.get("private_runtime_manifest_sha256_at_start",{})!=runtime_hashes or row.get("private_runtime_manifest_sha256_at_start",{})!=runtime_hashes:raise ValueError("per-process private runtime manifest binding differs")
        if row["split"]=="heldout":
            if raw["frozen_model_sha256_at_start"]!=manifest["model_frozen_before_heldout_sha256"]or row["frozen_model_sha256_at_start"]!=manifest["model_frozen_before_heldout_sha256"]or raw["started_utc"]<manifest["model_frozen_utc"]:raise ValueError("heldout predates/misses frozen model")
        if row["status"]=="measured":
            if type(raw.get("returncode"))is not int or raw["returncode"]!=0 or raw.get("timed_out")or raw.get("validation_error"):raise ValueError("measured row has unsuccessful raw process evidence")
            payload=json.loads(raw["stdout"]);c.validate_native(row["case"],suite["protocol"],row["seed"],payload,probe_sha)
            if row["metrics"]!=payload["wall_ns"]or row["diagnostic_host_and_copy_events"]!=payload["diagnostic_host_and_copy_events"]:raise ValueError("retained measurements differ from raw")
        elif not raw.get("validation_error"):raise ValueError("failed process lacks preserved reason")
    rebuilt=protocol.frozen_bundle([r for r in rows if r["split"]=="profile"],suite)
    if rebuilt!=model:raise ValueError("profile-only costs/common compiler programs do not replay")
    report=protocol.evaluate(rows,model,suite)
    if report!=protocol.load(root/"report.json")or c.sha(root/"report.json")!=manifest["report_sha256"]:raise ValueError("CV/error acceptance report does not replay")
    return dict(schema="overlap-evidence-verification-0.1",verified=True,processes=len(rows),failed_rows=report["failed_rows"],
        heldout_used_for_fitting=False,kernel_overlap_proved=False,physical_capacity_verified=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("directory",type=Path);args=parser.parse_args()
    print(json.dumps(verify(args.directory),indent=2))
