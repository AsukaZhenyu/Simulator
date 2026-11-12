"""Bounded public prompt snapshots and deterministic real-MoE request assembly.

No model inference occurs here. Corpus snapshots retain source URL, raw SHA256,
revision evidence and exact source IDs. Length selection uses actual tokenizer
results from the independent native vocab-only mode, not character estimates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,allow_nan=False,separators=(',',':'));f.write('\n')


def fetch(url, maximum=8*1024*1024):
    req=urllib.request.Request(url,headers={'User-Agent':'llm-infer-simulator-reproducible-corpus/1'})
    with urllib.request.urlopen(req,timeout=60) as response:
        content=response.read(maximum+1)
        if len(content)>maximum: raise ValueError('bounded download limit exceeded: '+url)
        return content,dict(url=response.geturl(),headers={k:v for k,v in response.headers.items()
            if k.lower() in ('etag','last-modified','content-length','x-revision','x-repo-commit')})


def github_snapshot(repository, filename):
    endpoint='https://api.github.com/repos/'+repository+'/commits?'+urllib.parse.urlencode(dict(path=filename,per_page=1))
    info,_=fetch(endpoint,2*1024*1024); commit=json.loads(info)[0]['sha']
    url='https://raw.githubusercontent.com/'+repository+'/'+commit+'/'+filename
    raw,headers=fetch(url)
    records=[json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    return raw,dict(repository=repository,filename=filename,revision=commit,
        revision_evidence_sha256=hashlib.sha256(info).hexdigest(),raw_sha256=hashlib.sha256(raw).hexdigest(),
        bytes=len(raw),http=headers,records=records)


def snapshot(out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    sources={}
    for name,repo,path in (('math','openai/grade-school-math','grade_school_math/data/test.jsonl'),
                           ('code','google-research/google-research','mbpp/mbpp.jsonl')):
        raw,source=github_snapshot(repo,path)
        target=out/(name+'-source.jsonl')
        with target.open('xb') as f:f.write(raw)
        source['local_raw_path']=str(target);sources[name]=source
    ds='HuggingFaceH4/ultrachat_200k'
    info_url='https://huggingface.co/api/datasets/'+ds+'/revision/main'
    info,_=fetch(info_url);revision=json.loads(info)['sha']
    url='https://datasets-server.huggingface.co/rows?'+urllib.parse.urlencode(dict(
        dataset=ds,config='default',split='test_sft',offset=0,length=100,revision=revision))
    raw,headers=fetch(url);data=json.loads(raw)
    again,_=fetch(info_url)
    if json.loads(again)['sha']!=revision:raise ValueError('dataset revision changed during API snapshot')
    target=out/'dialogue-source.json'
    with target.open('xb') as f:f.write(raw)
    sources['dialogue']=dict(dataset=ds,requested_revision=revision,split='test_sft',offset=0,length=100,
        local_raw_path=str(target),raw_sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),http=headers,
        revision_evidence_sha256=hashlib.sha256(info).hexdigest(),
        revision_association='API snapshot; requested revision retained, cache revision is not independently established',
        records=[dict(row_idx=r['row_idx'],**r['row']) for r in data['rows']])
    metadata={name:{k:v for k,v in source.items() if k!='records'} for name,source in sources.items()}
    write(out/'source-manifest.json',dict(schema='real-moe-public-prompt-snapshot-1',sources=metadata,
        no_routing_or_performance_labels=True,model_run=False))
    candidates=assemble(sources)
    write(out/'candidate-requests.json',dict(schema='real-moe-request-candidates-1',requests=candidates,
        snapshot_manifest_sha256=hashlib.sha256((out/'source-manifest.json').read_bytes()).hexdigest()))
    return dict(out=str(out),candidates=len(candidates),download_bytes=sum(s['bytes'] for s in sources.values()))


def chatml(text):
    return '<|im_start|>user\n'+text+'<|im_end|>\n<|im_start|>assistant\n'


def render_official(messages, config):
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True)
    return env.from_string(config['chat_template']).render(messages=messages,
        bos_token=config['bos_token'],eos_token=config['eos_token'],add_generation_prompt=True)


def assemble(sources, tokenizer_config=None):
    result=[]
    for domain in ('math','code'):
        records=sources[domain]['records']
        # Context examples are held separate from target questions. They are
        # meaningful few-shot demonstrations, never repeated-token padding.
        context_pool=records[:24];targets=records[24:54]
        for ordinal,target in enumerate(targets):
            target_id=target.get('task_id',24+ordinal)
            for n in (0,1,2,4,6,8):
                examples=[];ids=[]
                for j in range(n):
                    example=context_pool[(ordinal*3+j)%len(context_pool)];ids.append(example.get('task_id',(ordinal*3+j)%len(context_pool)))
                    if domain=='math':examples.append('Question: '+example['question']+'\nWorked solution: '+example['answer'])
                    else:examples.append('Task: '+example['text']+'\nExample solution:\n'+example['code']+'\nTests:\n'+'\n'.join(example['test_list']))
                if domain=='math':question='Solve the following problem. Explain the reasoning briefly.\n'+target['question']
                else:question='Write a Python solution for this task and explain it briefly.\n'+target['text']+'\nRequired tests:\n'+'\n'.join(target['test_list'])
                prompt=('Reference examples:\n\n'+'\n\n'.join(examples)+'\n\nTarget task:\n' if examples else '')+question
                row=dict(id=f'{domain}-target{target_id}-examples{n}',domain=domain,source_target_id=str(target_id),
                    context_source_ids=[str(x) for x in ids],prompt=chatml(prompt),template='explicit_chatml_user_assistant_v1',
                    messages=[dict(role='user',content=prompt)],
                    construction='disjoint-target few-shot context' if examples else 'target task and tests only',max_output_tokens=64)
                row['prompt_sha256']=hashlib.sha256(row['prompt'].encode()).hexdigest();result.append(row)
    for ordinal,target in enumerate(sources['dialogue']['records'][:60]):
        messages=target.get('messages',[])
        # Use actual prior conversation turns and end on a user message. No
        # response to the final user is included in the input.
        for turns in (1,3,5,7,9):
            chosen=messages[:turns]
            if len(chosen)!=turns or chosen[-1].get('role')!='user':continue
            prompt=''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n' for m in chosen)+'<|im_start|>assistant\n'
            row=dict(id=f'dialogue-row{target["row_idx"]}-turns{turns}',domain='dialogue',source_target_id=str(target['row_idx']),
                context_source_ids=[],prompt=prompt,messages=chosen,template='explicit_chatml_history_v1',construction='original conversation prefix',max_output_tokens=64)
            row['prompt_sha256']=hashlib.sha256(prompt.encode()).hexdigest();result.append(row)
    if tokenizer_config is not None:
        for row in result:
            row['prompt']=render_official(row['messages'],tokenizer_config)
            row['template']='official_checkpoint_jinja_trim_blocks_lstrip_blocks_v2'
            row['prompt_sha256']=hashlib.sha256(row['prompt'].encode()).hexdigest()
    return result


def rebuild(snapshot_dir, tokenizer_config, out, revision='olmoe-v2'):
    folder=Path(snapshot_dir);sources={}
    for domain in ('math','code'):
        records=[json.loads(line) for line in (folder/(domain+'-source.jsonl')).read_text(encoding='utf-8').splitlines() if line.strip()]
        sources[domain]=dict(records=records)
    data=read(folder/'dialogue-source.json')
    sources['dialogue']=dict(records=[dict(row_idx=r['row_idx'],**r['row']) for r in data['rows']])
    rows=assemble(sources,read(tokenizer_config))
    write(out,dict(schema='real-moe-request-candidates-2',requests=rows,
        snapshot_manifest_sha256=hashlib.sha256((folder/'source-manifest.json').read_bytes()).hexdigest(),
        tokenizer_config_sha256=hashlib.sha256(Path(tokenizer_config).read_bytes()).hexdigest(),
        no_routing_or_performance_labels=True,execution_revision=revision))
    return dict(out=str(out),candidates=len(rows),model_run=False)


def select(tokenized, out):
    data=read(tokenized)
    if data['mode']!='tokenize-only' or data.get('model_weights_loaded') is not False:
        raise ValueError('actual model vocab-only tokenizer evidence required')
    rows=data['tokenized_requests'];chosen=[];used=set()
    for domain in ('dialogue','code','math'):
        for length,(lo,hi) in (('short',(96,160)),('long',(896,1152))):
            target=128 if length=='short' else 1024
            eligible=[r for r in rows if r['domain']==domain and lo<=len(r['prompt_token_ids'])<=hi
                      and (domain,r['source_target_id']) not in used]
            eligible.sort(key=lambda r:(abs(len(r['prompt_token_ids'])-target),r['id']))
            group=[]
            for row in eligible:
                key=(domain,row['source_target_id'])
                if key in used:continue
                used.add(key);group.append(row)
                if len(group)==3:break
            if len(group)!=3:raise ValueError(f'unknown coverage: only {len(group)} actual-tokenizer rows for {domain}/{length}; do not pad silently')
            for i,row in enumerate(group):
                copied=dict(row,length_class=length,split='calibration' if i==0 else 'heldout',actual_prompt_tokens=len(row['prompt_token_ids']))
                chosen.append(copied)
    result=dict(schema='real-moe-frozen-request-suite-1',model_sha256=data['model_sha256_verified_by_parent'],
        tokenizer_evidence_sha256=hashlib.sha256(Path(tokenized).read_bytes()).hexdigest(),requests=chosen,
        counts=dict(total=18,calibration=6,heldout=12),
        sampling=dict(kind='greedy',max_output_tokens=64,stop_on_eog=True),
        no_performance_measurement_used_for_selection=True)
    result['suite_sha256']=digest(result);write(out,result);return result


def audit(suite_path, out):
    """Audit frozen workload coverage without reading any target timings."""
    import csv
    suite=read(suite_path);rows=suite['requests'];seen=set();cells={}
    if len(rows)!=18:raise ValueError('exact eighteen-request inventory required')
    for row in rows:
        key=(row['domain'],row['source_target_id'])
        if key in seen:raise ValueError('target reused across request windows')
        seen.add(key)
        if row['source_target_id'] in row['context_source_ids']:raise ValueError('target appears in its few-shot context')
        if hashlib.sha256(row['prompt'].encode()).hexdigest()!=row['prompt_sha256']:raise ValueError('prompt content hash differs')
        n=len(row['prompt_token_ids']);lo,hi={'short':(96,160),'long':(896,1152)}[row['length_class']]
        if not lo<=n<=hi or n!=row['actual_prompt_tokens']:raise ValueError('actual tokenizer length outside declared window')
        if not all(type(t) is int and t>=0 for t in row['prompt_token_ids']):raise ValueError('invalid actual token IDs')
        if 'output_token_ids' in row:raise ValueError('target outputs cannot enter frozen prompt selection')
        cells.setdefault((row['domain'],row['length_class']),[]).append(row['split'])
    expected={(d,l) for d in ('dialogue','code','math') for l in ('short','long')}
    if set(cells)!=expected or any(sorted(v)!=['calibration','heldout','heldout'] for v in cells.values()):raise ValueError('domain/window/split coverage differs')
    proof=dict(schema='real-moe-suite-audit-1',passed=True,requests=18,calibration=6,heldout=12,
        all_windows_checked=True,all_prompt_hashes_checked=True,no_target_reuse=True,
        no_target_performance_selection=suite['no_performance_measurement_used_for_selection'],
        suite_file_sha256=hashlib.sha256(Path(suite_path).read_bytes()).hexdigest(),
        model_sha256=suite['model_sha256'],performance_measured=False)
    write(out,proof)
    with Path(out).with_suffix('.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['id','domain','length_class','split','actual_prompt_tokens','source_target_id','prompt_sha256'],lineterminator='\n')
        writer.writeheader();writer.writerows({k:r[k] for k in writer.fieldnames} for r in rows)
    return proof


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('snapshot','select','rebuild','audit'));p.add_argument('--out',required=True);p.add_argument('--tokenized');p.add_argument('--suite')
    p.add_argument('--snapshot');p.add_argument('--tokenizer-config');p.add_argument('--revision',default='olmoe-v2')
    a=p.parse_args()
    if a.command=='snapshot':result=snapshot(a.out)
    elif a.command=='rebuild':result=rebuild(a.snapshot,a.tokenizer_config,a.out,a.revision)
    elif a.command=='audit':result=audit(a.suite,a.out)
    else:
        result=select(a.tokenized,a.out)
        result=dict(out=a.out,counts=result['counts'],suite_sha256=result['suite_sha256'])
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
