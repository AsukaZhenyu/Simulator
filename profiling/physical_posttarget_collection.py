"""Resume a frozen campaign with owned backgrounds stopped after the target.

Only post-target lifecycle changes: before references and foreground execute
the original declared workload/conditions. AFTER references run unloaded and
are audit-only, never same-load forward-resource observations. No fitting or
latency-based retry selection occurs here. Frozen collector sources are reused
unchanged through temporary, finally-restored orchestration hooks.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path

from profiling.available_service import campaign
from profiling.physical_job_index import effective_jobs

INDEX_SCHEMA = "physical-posttarget-job-index-1"
INDEX_NAME = "posttarget-job-index.json"
SCOPE = "unloaded_drift_only"


def _receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=campaign.run.sha(path))


def _write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _barrier(root, path, *, expected_sha=None):
    path = Path(path).resolve()
    campaign.require(expected_sha is None or campaign.run.sha(path) == expected_sha, "frozen barrier bytes changed")
    value = campaign.run.read(path)
    campaign.require(value.get("frozen_at_utc") and
        value.get("manifest_sha256") == campaign.run.sha(root / "manifest.json"), "barrier parent/freeze differs")
    frozen = Path(value["frozen_file"]).resolve()
    campaign.require(campaign.run.sha(frozen) == value["frozen_file_sha256"], "frozen parameters changed")
    return value, dict(path=str(frozen), sha256=value["frozen_file_sha256"])


def _validate_replacement(root, old, replacement):
    new = replacement["replacement_job"]
    campaign.require(old["split"] in {"heldout", "heldout_bare"} and
        new["id"] == old["id"] + "-retry1" and "-retry" not in old["id"] and
        {k:v for k,v in old.items() if k != "id"} == {k:v for k,v in new.items() if k != "id"},
        "only one same-workload heldout infrastructure retry is permitted")
    failed_path = Path(replacement["failed_execution_path"]).resolve()
    campaign.require(failed_path == root / "jobs" / old["id"] / "execution.json" and
        campaign.run.sha(failed_path) == replacement["failed_execution_sha256"], "retained failed execution binding differs")
    failed = campaign.run.read(failed_path)
    campaign.require(failed.get("status") == "failed" and failed.get("job") == old and
        "readiness deadline" in failed.get("error", "") and failed.get("cleanup_verified") is True,
        "replacement requires a retained, cleanup-verified infrastructure readiness failure")
    main = failed.get("completed", {}).get("main", {})
    campaign.require(main.get("completed_samples") == old["samples"] and
        all("before-" + kind in failed.get("completed", {}) for kind in ("cpu", "gpu")) and
        "after-gpu" not in failed.get("completed", {}), "replacement failure is not after the completed foreground")


def effective_posttarget_jobs(root, manifest):
    """Validate the append-only amendment and return its full effective job list."""
    root = Path(root).resolve()
    base = effective_jobs(root, manifest)
    path = root / INDEX_NAME
    if not path.exists():
        return base
    value = campaign.run.read(path)
    campaign.require(value.get("schema") == INDEX_SCHEMA and
        value.get("parent_manifest_sha256") == campaign.run.sha(root / "manifest.json") and
        value.get("posttarget_after_reference_scope") == SCOPE, "post-target index schema/parent/scope differs")
    frozen = value.get("frozen_file", {})
    campaign.require(set(frozen) == {"path", "sha256"} and campaign.run.sha(frozen["path"]) == frozen["sha256"],
        "post-target index frozen profile changed")
    sources = value.get("source_receipts")
    campaign.require(isinstance(sources, list) and sources and all(
        campaign.run.sha(r["path"]) == r["sha256"] for r in sources), "post-target source receipt changed")
    campaign.require(_receipt(__file__) in sources, "post-target collector source is not bound")
    replacements = value.get("replacements")
    campaign.require(isinstance(replacements, list) and len(replacements) <= 1, "only one new infrastructure retry is permitted")
    jobs = list(base)
    for replacement in replacements:
        matches = [j for j in jobs if j["id"] == replacement["original_id"]]
        campaign.require(len(matches) == 1, "replacement original is absent/ambiguous")
        old = matches[0]
        _validate_replacement(root, old, replacement)
        jobs[jobs.index(old)] = replacement["replacement_job"]
    campaign.require(jobs == value.get("jobs") and len({j["id"] for j in jobs}) == len(jobs),
        "full effective post-target job inventory differs")
    known = {j["id"] for j in jobs}
    original = value.get("original_completed_job_ids")
    changed = value.get("posttarget_cleanup_job_ids")
    campaign.require(isinstance(original, list) and len(set(original)) == len(original) and set(original) <= known and
        isinstance(changed, list) and len(set(changed)) == len(changed) and set(changed) <= known and
        not set(original) & set(changed), "post-target preserved/changed job inventories differ")
    campaign.require(set(changed) == {j["id"] for j in jobs if j["split"] in {"heldout", "heldout_bare"} and
        j["id"] not in original}, "post-target cleanup coverage differs")
    return jobs


def _prepare_index(root, manifest, barrier_path):
    barrier, frozen = _barrier(root, barrier_path)
    path = root / INDEX_NAME
    if path.exists():
        jobs = effective_posttarget_jobs(root, manifest)
        value = campaign.run.read(path)
        campaign.require(value["frozen_file"] == frozen, "post-target frozen binding differs")
        return value, jobs
    jobs, completed, replacements = list(effective_jobs(root, manifest)), [], []
    for old in list(jobs):
        execution_path = root / "jobs" / old["id"] / "execution.json"
        if not execution_path.exists():
            continue
        row = campaign.run.read(execution_path)
        campaign.require(row.get("job") == old, "existing execution job metadata differs")
        if row.get("status") == "completed":
            if old["split"] in {"heldout", "heldout_bare"}:
                completed.append(old["id"])
            continue
        campaign.require(old["split"] in {"heldout", "heldout_bare"}, "unresolved non-heldout failure cannot be replaced")
        replacement = dict(original_id=old["id"], replacement_job=dict(old, id=old["id"] + "-retry1"),
            failed_execution_path=str(execution_path), failed_execution_sha256=campaign.run.sha(execution_path))
        _validate_replacement(root, old, replacement)
        campaign.require(not (root / "jobs" / replacement["replacement_job"]["id"]).exists(), "replacement attempt already exists")
        replacements.append(replacement)
        jobs[jobs.index(old)] = replacement["replacement_job"]
    campaign.require(len(replacements) <= 1, "multiple new failures require explicit new protocol; no silent retries")
    analyzer_path = Path(__file__).with_name("physical_posttarget_analysis.py")
    campaign.require(analyzer_path.is_file(), "declared post-target analysis adapter is missing")
    sources = [_receipt(__file__), _receipt(analyzer_path), _receipt(root / "manifest.json"), _receipt(barrier_path)]
    for name in ("job-replacements.json", "bare-transfer-protocol.json"):
        if (root / name).exists():
            sources.append(_receipt(root / name))
    value = dict(schema=INDEX_SCHEMA, parent_manifest_sha256=campaign.run.sha(root / "manifest.json"),
        frozen_file=frozen, jobs=jobs, replacements=replacements, original_completed_job_ids=completed,
        posttarget_after_reference_scope=SCOPE,
        posttarget_cleanup_job_ids=[j["id"] for j in jobs if j["split"] in {"heldout", "heldout_bare"} and j["id"] not in completed],
        declared_at_utc=campaign.run.utc(), source_receipts=sources)
    _write_new(path, value)
    effective_posttarget_jobs(root, manifest)
    return value, jobs


@contextmanager
def _posttarget_hooks(root, job, index_path):
    """Restore all module hooks even when a constructor, STOP or sample fails."""
    original_child, original_wait, original_save = campaign.OwnedChild, campaign.wait_complete, campaign.run.save
    backgrounds = []
    state = {"receipt": None}
    directory = root / "jobs" / job["id"]

    class PostTargetChild(original_child):
        def __init__(self, name, *args, **kwargs):
            if name.startswith("after-"):
                campaign.require(state["receipt"] is not None and all(
                    b.record.get("process_exited_verified") is True and b.process.poll() is not None for b in backgrounds),
                    "AFTER construction requires verified post-target background exit")
            super().__init__(name, *args, **kwargs)
            if name.endswith("-worker"):
                backgrounds.append(self)

    def wait_complete(child, clock, timeout):
        result = original_wait(child, clock, timeout)
        if child.name == "main":
            begin = clock.now()
            receipts = []
            for background in backgrounds:
                receipt = background.stop(cleanup=False)
                receipts.append(dict(receipt))
                campaign.require(receipt.get("process_exited_verified") is True and
                    receipt.get("forced_termination") is False and receipt.get("returncode") == 0,
                    "post-target owned background failed normal STOP")
            path = directory / "posttarget-cleanup.json"
            _write_new(path, dict(schema="physical-posttarget-cleanup-1", job_id=job["id"],
                index=_receipt(index_path), clock=clock.metadata(), main_sampling_complete=result,
                main_completion_observed_tick=begin, background_stop_begin_tick=begin,
                background_stop_end_tick=clock.now(), background_processes=receipts,
                background_count=len(backgrounds), all_backgrounds_exited_before_after_construction=True,
                posttarget_after_reference_scope=SCOPE, target_sampling_altered=False,
                before_reference_conditions_altered=False, numerical_or_prediction_selection=False))
            state["receipt"] = _receipt(path)
        return result

    def save(path, value):
        if Path(path).resolve() == directory / "execution.json" and isinstance(value, dict):
            value.update(posttarget_cleanup_applied=state["receipt"] is not None,
                posttarget_after_reference_scope=SCOPE if state["receipt"] is not None else "cleanup_not_reached",
                posttarget_cleanup_receipt=state["receipt"])
        return original_save(path, value)

    campaign.OwnedChild, campaign.wait_complete, campaign.run.save = PostTargetChild, wait_complete, save
    try:
        yield
    finally:
        campaign.OwnedChild, campaign.wait_complete, campaign.run.save = original_child, original_wait, original_save


def collect_remaining(root, split, barrier):
    """Preserve completed jobs and collect only declared missing/replacement jobs."""
    root, barrier = Path(root).resolve(), Path(barrier).resolve()
    campaign.require(split in {"heldout", "heldout_bare"}, "only frozen heldout splits are accepted")
    manifest = campaign.run.read(root / "manifest.json")
    campaign.verify(manifest)
    barrier_sha = campaign.run.sha(barrier)
    rows = []
    with campaign.run.measurement_lock():
        index, jobs = _prepare_index(root, manifest, barrier)
        index_path = root / INDEX_NAME
        index_sha = campaign.run.sha(index_path)
        _write_new(root / ("posttarget-" + split + "-collection-started.json"), dict(
            schema="physical-posttarget-collection-started-1", started_at_utc=campaign.run.utc(), split=split,
            index=dict(path=str(index_path), sha256=index_sha), barrier=_receipt(barrier), used_for_calibration=False))
        for job in jobs:
            if job["split"] != split:
                continue
            campaign.verify(manifest)
            _barrier(root, barrier, expected_sha=barrier_sha)
            campaign.require(campaign.run.sha(index_path) == index_sha, "post-target job index changed during collection")
            path = root / "jobs" / job["id"] / "execution.json"
            if path.exists():
                row = campaign.run.read(path)
                campaign.require(row.get("status") == "completed" and row.get("job") == job,
                    "existing failed attempt retained; no automatic additional retry")
            else:
                campaign.require(job["id"] in index["posttarget_cleanup_job_ids"], "new job lacks predeclared post-target amendment")
                with _posttarget_hooks(root, job, index_path):
                    row = campaign.execute(root, manifest, job)
                campaign.require(row.get("posttarget_cleanup_applied") is True, "post-target cleanup receipt missing")
                print(json.dumps(dict(job=job["id"], status=row["status"],
                    posttarget_cleanup_applied=True, cleanup_verified=row["cleanup_verified"])), flush=True)
            rows.append(row)
        _write_new(root / ("posttarget-" + split + "-executions.json"), rows)
    return dict(status="completed", split=split, jobs=len(rows), replacements_retained=len(index["replacements"]),
        posttarget_after_reference_scope=SCOPE, frozen_parameters_changed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("heldout", "heldout_bare"))
    parser.add_argument("--barrier", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(collect_remaining(args.out, args.split, args.barrier)))


if __name__ == "__main__":
    main()
