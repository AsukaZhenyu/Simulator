"""Explicit additions/replacements without rewriting a hardware manifest."""
from pathlib import Path
from profiling.available_service import campaign


def effective_jobs(root, manifest):
    root=Path(root)
    jobs=list(manifest['jobs'])
    path=root/'job-replacements.json'
    if path.exists():
        index=campaign.run.read(path)
        campaign.require(index['parent_manifest_sha256']==campaign.run.sha(root/'manifest.json'), 'replacement parent differs')
        for replacement in index['replacements']:
            old=next(j for j in jobs if j['id']==replacement['original_id'])
            new=replacement['replacement_job']
            campaign.require({k:v for k,v in old.items() if k!='id'}=={k:v for k,v in new.items() if k!='id'},
                             'retry changes workload or protocol')
            failure=campaign.run.read(replacement['failed_execution_path'])
            campaign.require(campaign.run.sha(replacement['failed_execution_path'])==replacement['failed_execution_sha256'] and
                             failure['status']=='failed' and 'readiness deadline' in failure.get('error',''),
                             'replacement requires retained infrastructure readiness failure')
            jobs[jobs.index(old)]=new
    path=root/'bare-transfer-protocol.json'
    if path.exists():
        bare=campaign.run.read(path)
        campaign.require(bare['parent_manifest_sha256']==campaign.run.sha(root/'manifest.json'), 'bare parent differs')
        jobs.extend(bare['jobs'])
    campaign.require(len({j['id'] for j in jobs})==len(jobs),'duplicate effective job id')
    return jobs


def resume_profile(root):
    root=Path(root).resolve()
    manifest=campaign.run.read(root/'manifest.json')
    campaign.verify(manifest)
    replacements=[]
    for job in manifest['jobs']:
        if job['split']!='profile':continue
        path=root/'jobs'/job['id']/'execution.json'
        if not path.exists():continue
        failed=campaign.run.read(path)
        if failed['status']=='completed':continue
        campaign.require('readiness deadline' in failed.get('error',''),'only readiness failures can be retried here')
        replacements.append(dict(original_id=job['id'],replacement_job=dict(job,id=job['id']+'-retry1'),
            failed_execution_path=str(path),failed_execution_sha256=campaign.run.sha(path),
            selection_reason='infrastructure readiness timeout; not latency or prediction error',
            failed_main_samples_excluded=job['samples']))
    campaign.run.save(root/'job-replacements.json',dict(schema='physical-job-replacements-1',
        declared_at_utc=campaign.run.utc(),parent_manifest_sha256=campaign.run.sha(root/'manifest.json'),
        replacements=replacements,maximum_retries=1,driver_sha256=campaign.run.sha(__file__)))
    rows=[]
    with campaign.run.measurement_lock():
        for job in effective_jobs(root,manifest):
            if job['split']!='profile':continue
            path=root/'jobs'/job['id']/'execution.json'
            if path.exists():
                row=campaign.run.read(path)
                campaign.require(row['status']=='completed','retained replacement failed; no silent retry')
            else:
                campaign.verify(manifest)
                row=campaign.execute(root,manifest,job)
                print(job['id'],row['status'],flush=True)
            rows.append(row)
    campaign.run.save(root/'profile-executions.json',rows)
    return dict(status='completed',jobs=len(rows),failed_attempts_retained=len(replacements))


if __name__=='__main__':
    import argparse,json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(resume_profile(args.out)))
