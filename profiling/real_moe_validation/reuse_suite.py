"""Reuse tokenizer-only evidence across execution revisions, with strict binding."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import shutil
from .dataset import digest,read,write
from .run import receipt,require


def tokenizer_body(path):
    source=Path(path).read_text(encoding='utf-8')
    begin=source.index('static std::vector<llama_token> tokenize(')
    end=source.index('static void fill_batch(',begin)
    return hashlib.sha256(source[begin:end].encode()).hexdigest()


def reuse(old_collection,new_collection):
    old,new=Path(old_collection).resolve(),Path(new_collection).resolve()
    first,last=read(old/'manifest.json'),read(new/'manifest.json')
    require(first['model']==last['model'] and first['model_contract']==last['model_contract'],'model identity differs across tokenizer reuse')
    require(first['official_tokenizer_config_sha256']==last['official_tokenizer_config_sha256'],'official prompt template/tokenizer config changed')
    require(first['runtime']==last['runtime'],'exact llama/GGML tokenizer runtime changed')
    old_source=next(s['archived']['path'] for s in first['sources'] if Path(s['source']['path']).name=='real_moe_probe.cpp')
    new_source=next(s['archived']['path'] for s in last['sources'] if Path(s['source']['path']).name=='real_moe_probe.cpp')
    source_hash=tokenizer_body(old_source)
    require(source_hash==tokenizer_body(new_source),'native tokenizer implementation or flags changed')
    tokenized=read(old/'jobs/tokenize-candidates/result.json');suite=read(old/'request-suite-v2.json')
    require(tokenized['mode']=='tokenize-only' and tokenized['model_weights_loaded'] is False,'reuse source must be actual tokenizer-only evidence')
    require(tokenized['model_sha256_verified_by_parent']==last['model']['sha256'] and tokenized['tokenization_add_special'] is False,'tokenizer/model boundary differs')
    require(suite['tokenizer_evidence_sha256']==receipt(old/'jobs/tokenize-candidates/result.json')['sha256'],'suite lost its tokenizer evidence binding')
    unhashed={k:v for k,v in suite.items() if k!='suite_sha256'}
    require(digest(unhashed)==suite['suite_sha256'],'frozen suite content changed')
    lookup={r['id']:r for r in tokenized['tokenized_requests']}
    for row in suite['requests']:
        source=lookup[row['id']]
        for key in ('prompt','prompt_sha256','prompt_token_ids','source_target_id','context_source_ids','messages'):
            require(row[key]==source[key],'frozen request differs from actual tokenizer output')
    target=new/'request-suite-v3.json';require(not target.exists(),'new suite path required');shutil.copy2(old/'request-suite-v2.json',target)
    proof=dict(schema='real-moe-tokenizer-reuse-proof-1',passed=True,source_suite=receipt(old/'request-suite-v2.json'),
        reused_suite=receipt(target),source_tokenizer=receipt(old/'jobs/tokenize-candidates/result.json'),
        old_manifest=receipt(old/'manifest.json'),new_manifest=receipt(new/'manifest.json'),
        tokenizer_function_sha256=source_hash,official_tokenizer_config_sha256=last['official_tokenizer_config_sha256'],
        exact_runtime_receipts_equal=True,requests_compared=18,no_inference_reused=True,
        new_execution_threadpool_policy=last['cpu_threadpool_policy'],source_tool=receipt(__file__))
    write(new/'tokenizer-reuse-proof.json',proof);return proof


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old',required=True);p.add_argument('--new',required=True);a=p.parse_args()
    result=reuse(a.old,a.new);print(__import__('json').dumps(dict(passed=result['passed'],requests_compared=18,reused_suite=result['reused_suite'])))
