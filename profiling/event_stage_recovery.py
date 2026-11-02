"""Append-only recovery of one retained Windows completion-file read failure.

This changes controller file observation only. It does not change target work,
timeouts, native timing or numerical acceptance. No latency selects a retry.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import time

from profiling.available_service import campaign
from profiling.event_stage_campaign import event_actors, verify
from profiling.physical_posttarget_collection import _posttarget_hooks

INDEX_NAME="event-recovery-index.json"


def receipt(path):
    return dict(path=str(Path(path).resolve()),sha256=campaign.run.sha(path))


def write_new(path,value):
    with Path(path).open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write("\n")


def event_effective_jobs(root,protocol):
    root=Path(root).resolve()
    jobs=list(protocol["jobs"])
    index_path=root/INDEX_NAME
    if not index_path.exists():
        return jobs
    value=campaign.run.read(index_path)
    campaign.require(value.get("schema")=="event-stage-read-recovery-1" and
        value["manifest_sha256"]==campaign.run.sha(root/"manifest.json") and
        value["event_protocol_sha256"]==campaign.run.sha(root/"event-protocol.json"),"recovery parent changed")
    campaign.require(len(value["replacements"])==1,"only the declared single file-observation retry is accepted")
    campaign.require(receipt(__file__) in value["source_receipts"] and
        receipt(Path(__file__).with_name("physical_posttarget_collection.py")) in value["source_receipts"],
        "recovery controller and actual post-target helper must be source-bound")
    for source in value["source_receipts"]:
        campaign.require(campaign.run.sha(source["path"])==source["sha256"],"recovery source changed")
    for replacement in value["replacements"]:
        old=next(j for j in jobs if j["id"]==replacement["original_id"])
        new=replacement["replacement_job"]
        campaign.require(old["split"]=="profile" and new["id"]==old["id"]+"-retry1" and
            {k:v for k,v in old.items() if k!="id"}=={k:v for k,v in new.items() if k!="id"},
            "replacement changes workload or timing protocol")
        campaign.require(Path(replacement["failed_execution"]["path"]).resolve()==root/"jobs"/old["id"]/"execution.json",
            "failed execution is outside the declared job")
        failed=campaign.run.read(replacement["failed_execution"]["path"])
        campaign.require(campaign.run.sha(replacement["failed_execution"]["path"])==replacement["failed_execution"]["sha256"] and
            failed.get("job")==old and failed.get("status")=="failed" and
            failed.get("error")=="PermissionError(13, 'Permission denied')" and failed.get("cleanup_verified") is True and
            all(p.get("returncode")==0 and p.get("forced_termination") is False and p.get("process_exited_verified") is True
                for p in failed["processes"]),"retained failure is not the declared cleanup-verified file read failure")
        jobs[jobs.index(old)]=new
    campaign.require(jobs==value["effective_jobs"],"effective recovery inventory changed")
    return jobs


def prepare_recovery(root):
    root=Path(root).resolve()
    parent,protocol=verify(root)
    if (root/INDEX_NAME).exists():
        return event_effective_jobs(root,protocol)
    replacements=[]
    jobs=list(protocol["jobs"])
    for job in list(jobs):
        path=root/"jobs"/job["id"]/"execution.json"
        if not path.exists():
            continue
        row=campaign.run.read(path)
        if row["status"]=="completed":
            continue
        campaign.require(job["split"]=="profile" and row.get("error")=="PermissionError(13, 'Permission denied')" and
            row.get("cleanup_verified") is True,"unresolved failure needs a separate reviewed protocol")
        replacement=dict(original_id=job["id"],replacement_job=dict(job,id=job["id"]+"-retry1"),
            failed_execution=receipt(path),selection_reason="controller completion-file PermissionError; no latency/error selection",
            all_failed_main_samples_excluded=True,failed_main_samples_requested=job["samples"],
            retained_files=[receipt(p) for p in sorted(path.parent.rglob("*")) if p.is_file()])
        replacements.append(replacement)
        jobs[jobs.index(job)]=replacement["replacement_job"]
    campaign.require(len(replacements)==1,"one retained infrastructure failure is required")
    helper=Path(__file__).with_name("physical_posttarget_collection.py")
    value=dict(schema="event-stage-read-recovery-1",declared_at_utc=campaign.run.utc(),
        manifest_sha256=campaign.run.sha(root/"manifest.json"),event_protocol_sha256=campaign.run.sha(root/"event-protocol.json"),
        replacements=replacements,effective_jobs=jobs,source_receipts=[receipt(__file__),receipt(helper)],
        read_policy="sampling-complete only: retry PermissionError/sharing errors until the original deadline; reject semantic/identity errors",
        native_or_target_timing_changed=False,parameter_fitting_during_collection=False)
    write_new(root/INDEX_NAME,value)
    return event_effective_jobs(root,protocol)


@contextmanager
def bounded_completion_reads(events):
    original=campaign.wait_complete

    def wait(child,clock,timeout):
        path=child.directory/"sampling-complete.json"
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            campaign.require(child.process.poll() is None,"owned actor exited before sampling-complete: "+child.name)
            if path.exists():
                try:
                    value=campaign.run.read(path)
                except (json.JSONDecodeError,PermissionError) as error:
                    events.append(dict(actor=child.name,path=str(path),error=repr(error),observed_at_utc=campaign.run.utc()))
                    time.sleep(.005)
                    continue
                except OSError as error:
                    if getattr(error,"winerror",None) not in {32,33}:
                        raise
                    events.append(dict(actor=child.name,path=str(path),error=repr(error),observed_at_utc=campaign.run.utc()))
                    time.sleep(.005)
                    continue
                campaign.require(value["process_id"]==child.process.pid,"sampling identity differs")
                return value
            time.sleep(.005)
        raise TimeoutError("sampling deadline: "+child.name)

    campaign.wait_complete=wait
    try:
        yield
    finally:
        campaign.wait_complete=original


def collect(root,split,barrier_path=None):
    root=Path(root).resolve()
    parent,protocol=verify(root)
    jobs=prepare_recovery(root)
    barrier=None
    if split=="heldout":
        campaign.require(barrier_path is not None,"heldout requires frozen barrier")
        barrier=campaign.run.read(barrier_path)
        campaign.require(barrier["event_protocol_sha256"]==campaign.run.sha(root/"event-protocol.json") and
            campaign.run.sha(barrier["frozen_file"])==barrier["frozen_file_sha256"],"heldout freeze differs")
    started=root/("heldout-started.json" if split=="heldout" else "profile-resume-started.json")
    write_new(started,dict(started_at_utc=campaign.run.utc(),split=split,
        event_protocol_sha256=campaign.run.sha(root/"event-protocol.json"),barrier=barrier,
        recovery_index=receipt(root/INDEX_NAME)))
    rows=[]
    with campaign.run.measurement_lock():
        for job in jobs:
            if job["split"]!=split:
                continue
            path=root/"jobs"/job["id"]/"execution.json"
            if path.exists():
                row=campaign.run.read(path)
                campaign.require(row["status"]=="completed","no silent retry of a retained replacement failure")
            else:
                verify(root)
                event_effective_jobs(root,protocol)
                if barrier:
                    campaign.require(campaign.run.sha(barrier["frozen_file"])==barrier["frozen_file_sha256"],"frozen parameters changed")
                manifest=deepcopy(parent)
                manifest["binary"]=protocol["event_binary"]
                manifest["protocol"].update(protocol["protocol_overrides"])
                events=[]
                try:
                    with bounded_completion_reads(events),event_actors(job),_posttarget_hooks(root,job,root/"event-protocol.json"):
                        row=campaign.execute(root,manifest,job)
                finally:
                    write_new(root/"jobs"/job["id"]/"completion-read-observations.json",dict(events=events,
                        read_policy_bound_by=receipt(root/INDEX_NAME),native_target_timing_changed=False))
                print(json.dumps(dict(job=job["id"],status=row["status"],cleanup_verified=row["cleanup_verified"])),flush=True)
            rows.append(row)
    write_new(root/(split+"-executions.json"),rows)
    return dict(status="collected",split=split,jobs=len(rows))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--split",choices=("profile","heldout"),required=True)
    parser.add_argument("--barrier",type=Path)
    args=parser.parse_args()
    print(json.dumps(collect(args.out,args.split,args.barrier)))
