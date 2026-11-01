"""Predeclared, fresh untraced transfer check for a traced physical campaign.

This is an additional heldout cohort, not an overhead-correction fit. The same
before-target reference sensors remain enabled; only target CUPTI is disabled.
No result is used to adjust the frozen model.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json

from profiling.available_service import campaign


def prepare(root):
    manifest_path = root / 'manifest.json'
    manifest = campaign.run.read(manifest_path)
    campaign.verify(manifest)
    jobs = []
    for original in manifest['jobs']:
        if original['split'] != 'heldout':
            continue
        job = dict(original, id=original['id']+'-bare', split='heldout_bare', instrumentation='bare')
        jobs.append(job)
    value = dict(schema='physical-bare-transfer-protocol-1', declared_at_utc=campaign.run.utc(),
                 parent_manifest_sha256=campaign.run.sha(manifest_path), jobs=jobs,
                 driver_path=str(Path(__file__).resolve()), driver_sha256=campaign.run.sha(__file__),
                 overhead_subtraction=False, used_for_calibration=False,
                 interpretation='CUPTI-trained model transfer to fresh bare target, with disjoint reference actors')
    campaign.run.save(root/'bare-transfer-protocol.json',value)
    return dict(status='declared', jobs=len(jobs))


def collect(root, barrier_path):
    protocol_path=root/'bare-transfer-protocol.json'
    protocol=campaign.run.read(protocol_path)
    manifest=campaign.run.read(root/'manifest.json')
    barrier=campaign.run.read(barrier_path)
    campaign.require(protocol['driver_sha256']==campaign.run.sha(__file__), 'transfer driver changed')
    campaign.require(protocol['parent_manifest_sha256']==barrier['manifest_sha256']==campaign.run.sha(root/'manifest.json'),
                     'transfer/parent/frozen barrier differ')
    campaign.require(campaign.run.sha(barrier['frozen_file'])==barrier['frozen_file_sha256'], 'frozen parameters changed')
    campaign.run.save(root/'heldout_bare-collection-started.json',dict(started_at_utc=campaign.run.utc(),
        barrier_path=str(barrier_path.resolve()),barrier_sha256=campaign.run.sha(barrier_path),
        protocol_sha256=campaign.run.sha(protocol_path)))
    rows=[]
    with campaign.run.measurement_lock():
        for job in protocol['jobs']:
            campaign.verify(manifest)
            campaign.require(campaign.run.sha(barrier['frozen_file'])==barrier['frozen_file_sha256'], 'frozen parameters changed')
            row=campaign.execute(root,manifest,job)
            rows.append(row)
            print(json.dumps(dict(job=job['id'],status=row['status'],cleanup_verified=row['cleanup_verified'])),flush=True)
    campaign.run.save(root/'heldout_bare-executions.json',rows)
    return dict(status='collected',jobs=len(rows),used_for_calibration=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','collect'))
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--barrier',type=Path)
    args=parser.parse_args()
    if args.mode=='collect':
        parser.error('--barrier is required for collect') if args.barrier is None else None
    print(json.dumps(prepare(args.out.resolve()) if args.mode=='prepare' else collect(args.out.resolve(),args.barrier)))


if __name__=='__main__':
    main()
