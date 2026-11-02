"""Acquire one explicitly declared public GGUF, with bounded resumable transfer."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

from .dataset import fetch, read, write

REPO='bartowski/OLMoE-1B-7B-0924-Instruct-GGUF'
FILE='OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf'
ORIGINAL='allenai/OLMoE-1B-7B-0924-Instruct'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def _acquire(out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    manifest_path=out/'acquisition-plan.json'
    if manifest_path.exists():
        m=read(manifest_path)
        if m['repository']!=REPO or m['filename']!=FILE:raise ValueError('resume identity differs')
    else:
        raw,_=fetch('https://huggingface.co/api/models/'+REPO+'/revision/main?blobs=true',4*1024*1024)
        info=json.loads(raw);file=next(s for s in info['siblings'] if s['rfilename']==FILE)
        lfs=file['lfs'];expected=lfs.get('sha256',lfs.get('oid'));size=lfs['size']
        if len(expected)!=64 or not 3*1024**3<size<5*1024**3:raise ValueError('unexpected publisher artifact identity')
        original_raw,_=fetch('https://huggingface.co/api/models/'+ORIGINAL+'/revision/main')
        original=json.loads(original_raw)
        files={}
        for name in ('config.json','tokenizer_config.json'):
            content,headers=fetch('https://huggingface.co/'+ORIGINAL+'/resolve/'+original['sha']+'/'+name)
            path=out/('original-'+name)
            with path.open('xb') as f:f.write(content)
            files[name]=dict(path=str(path),sha256=hashlib.sha256(content).hexdigest(),http=headers)
        config=read(out/'original-config.json');tokenizer=read(out/'original-tokenizer_config.json')
        contract=dict(architecture=config['model_type'],layers=config['num_hidden_layers'],
                      experts=config['num_experts'],top_k=config['num_experts_per_tok'])
        if not tokenizer.get('chat_template'):raise ValueError('official prompt template absent')
        m=dict(schema='real-moe-public-model-acquisition-1',repository=REPO,filename=FILE,
               revision=info['sha'],publisher_lfs_sha256=expected,bytes=size,
               publisher_info_sha256=hashlib.sha256(raw).hexdigest(),original_model=ORIGINAL,
               original_revision=original['sha'],original_files=files,model_contract=contract,
               url='https://huggingface.co/'+REPO+'/resolve/'+info['sha']+'/'+FILE,
               prompt_template_source='original tokenizer_config.json, checked against GGUF metadata',
               warm_resident_capacity_claimed=False)
        write(manifest_path,m)
    target=out/FILE;partial=out/(FILE+'.partial')
    if not partial.resolve().is_relative_to(out) or not target.resolve().is_relative_to(out):raise ValueError('download path escaped output directory')
    if target.exists():
        if target.stat().st_size!=m['bytes'] or sha(target)!=m['publisher_lfs_sha256']:raise ValueError('existing model identity differs')
        return dict(completed=True,model=str(target),bytes=m['bytes'],sha256=m['publisher_lfs_sha256'],reused_verified_file=True)
    start=partial.stat().st_size if partial.exists() else 0
    if start>m['bytes']:raise ValueError('owned partial exceeds declared artifact size')
    if start<m['bytes']:
        headers={'User-Agent':'llm-infer-simulator-public-model/1'}
        if start:headers['Range']=f'bytes={start}-'
        req=urllib.request.Request(m['url'],headers=headers)
        with urllib.request.urlopen(req,timeout=120) as response:
            if start and response.status!=206:raise ValueError('server refused safe resume range')
            if start and not response.headers.get('Content-Range','').startswith(f'bytes {start}-'):raise ValueError('resume offset differs')
            with partial.open('ab' if partial.exists() else 'xb') as f:
                total=start;last=total
                while total<m['bytes']:
                    block=response.read(min(8*1024*1024,m['bytes']-total))
                    if not block:break
                    f.write(block);total+=len(block)
                    if total-last>=128*1024*1024:
                        print(json.dumps(dict(downloaded_bytes=total,total_bytes=m['bytes'])),flush=True);last=total
                if total!=m['bytes']:raise ValueError(f'incomplete download retained for explicit resume: {total}/{m["bytes"]}')
    actual=sha(partial)
    if actual!=m['publisher_lfs_sha256']:raise ValueError('publisher SHA256 mismatch; partial retained, never promoted')
    partial.replace(target)
    proof=dict(schema='real-moe-public-model-acquisition-proof-1',completed=True,
               model=str(target),bytes=target.stat().st_size,sha256=actual,
               publisher_lfs_sha256=m['publisher_lfs_sha256'],sha256_matches=True,
               plan_sha256=sha(manifest_path),model_contract=m['model_contract'],performance_measured=False)
    if not (out/'acquisition-proof.json').exists():write(out/'acquisition-proof.json',proof)
    return proof


def _owned(out, operation, *args):
    """Only one writer owns an acquisition directory, including resume."""
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    lock=out/'acquisition.lock'
    with lock.open('x',encoding='utf-8') as stream:stream.write(str(os.getpid()))
    try:return operation(out,*args)
    finally:lock.unlink()


def acquire(out):return _owned(out,_acquire)


def _recover_duplicate(out, skip_begin, skip_bytes):
    """Recover a rejected owned partial only if the final publisher hash matches.

    This is an explicit repair operation, never an automatic relaxation of the
    content identity check. The original rejected bytes and repair evidence stay.
    """
    out=Path(out).resolve();m=read(out/'acquisition-plan.json')
    source=out/(FILE+'.partial');candidate=out/(FILE+'.recovered.partial');target=out/FILE
    if target.exists() or candidate.exists():raise ValueError('recovery requires new candidate and target')
    size=source.stat().st_size
    if skip_begin<0 or skip_bytes<=0 or skip_begin+skip_bytes>size or size-skip_bytes!=m['bytes']:
        raise ValueError('recovery interval cannot produce exact publisher size')
    original_sha=sha(source);h=hashlib.sha256()
    with source.open('rb') as src,candidate.open('xb') as dst:
        left=skip_begin
        while left:
            block=src.read(min(8*1024*1024,left))
            if not block:raise ValueError('source ended before recovery interval')
            dst.write(block);h.update(block);left-=len(block)
        src.seek(skip_begin+skip_bytes)
        for block in iter(lambda:src.read(8*1024*1024),b''):dst.write(block);h.update(block)
    if candidate.stat().st_size!=m['bytes'] or h.hexdigest()!=m['publisher_lfs_sha256'] or sha(candidate)!=m['publisher_lfs_sha256']:
        raise ValueError('recovery rejected; publisher identity was not restored')
    rejected=out/(FILE+'.rejected-'+original_sha)
    if rejected.exists():raise ValueError('rejected source preservation target already exists')
    source.replace(rejected);candidate.replace(target)
    proof=dict(schema='real-moe-public-model-acquisition-proof-1',completed=True,
        model=str(target),bytes=target.stat().st_size,sha256=h.hexdigest(),
        publisher_lfs_sha256=m['publisher_lfs_sha256'],sha256_matches=True,
        plan_sha256=sha(out/'acquisition-plan.json'),model_contract=m['model_contract'],performance_measured=False,
        acquisition_recovery=dict(reason='duplicate interval in overlapping resumed partial',
            rejected_source=str(rejected),rejected_sha256=original_sha,rejected_bytes=size,
            skipped_begin=skip_begin,skipped_bytes=skip_bytes,exact_publisher_hash_restored=True))
    write(out/'acquisition-proof.json',proof);return proof


def recover_duplicate(out, skip_begin, skip_bytes):
    return _owned(out,_recover_duplicate,skip_begin,skip_bytes)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--recover-skip-begin',type=int);p.add_argument('--recover-skip-bytes',type=int);a=p.parse_args()
    result=acquire(a.out) if a.recover_skip_begin is None else recover_duplicate(a.out,a.recover_skip_begin,a.recover_skip_bytes)
    print(json.dumps(result,ensure_ascii=False),flush=True)
