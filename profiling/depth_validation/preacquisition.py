"""Amend two scope descriptions before the first acquisition, never prices.

The running preparation may already have imported its immutable campaign source.
This explicit receipt preserves its original manifest and changes only the two
named context fields, before any job directory, acquisition marker or freeze.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESPONSE = 'inclusive stamped full-context per-action profile mean'
ORIGINAL = 'PREACQUISITION_ORIGINAL.json'
RECEIPT = 'PREACQUISITION_SCOPE_AMENDMENT.json'
FORBIDDEN = ('jobs','profile-started.json','heldout-started.json',
             'profile-executions.json','heldout-executions.json',
             'model-frozen.json','predictions-frozen.json','frozen-barrier.json')


def require(ok,message):
    if not ok: raise ValueError(message)


def sha(data): return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value,separators=(',',':'),allow_nan=False)+'\n').encode('utf-8')


def no_acquisition(out):
    require(not any((out/name).exists() for name in FORBIDDEN),
            'pre-acquisition amendment forbidden after jobs, markers or frozen artifacts exist')


def amend(out):
    out = Path(out).resolve()
    require(out.is_relative_to((ROOT/'build').resolve()), 'amendment must stay inside this repository build directory')
    no_acquisition(out)
    require(not (out/ORIGINAL).exists() and not (out/RECEIPT).exists(),
            'scope amendment already exists; original evidence cannot be overwritten')
    path = out/'manifest.json'; original_bytes=path.read_bytes()
    original=json.loads(original_bytes)
    require(original.get('schema')=='dense-depth-validation-1' and original.get('pilot') is False,
            'only the formal unmeasured Dense-depth manifest can be amended')
    jobs=original.get('jobs')
    require(isinstance(jobs,list) and jobs, 'prepared job inventory required')
    for job in jobs:
        request=job['request']
        require(job['split'] in ('profile','heldout') and request['memory_mode']=='pageable'
                and request['mode']=='full'
                and request['instrumentation']==('stamped' if job['split']=='profile' else 'bare'),
                'actual planned acquisition does not match the proposed scope descriptions')
    context=original.get('context')
    require(isinstance(context,dict) and isinstance(context.get('response_context'),str),
            'original response context must be explicit')
    require('memory_mode' not in context or context['memory_mode']=='pageable',
            'cannot relabel a different memory mode as pageable')
    updated=deepcopy(original)
    updated['context']['response_context']=RESPONSE
    updated['context']['memory_mode']='pageable'
    expected=deepcopy(updated)
    expected['context']['response_context']=original['context']['response_context']
    if 'memory_mode' in original['context']:
        expected['context']['memory_mode']=original['context']['memory_mode']
    else: expected['context'].pop('memory_mode')
    require(expected==original,'amendment changed fields outside the two permitted scope descriptions')
    updated_bytes=encode(updated)
    differences=[dict(field='context.'+key,
        before=dict(present=key in context,value=context.get(key)),after=updated['context'][key])
        for key in ('response_context','memory_mode')
        if key not in context or context[key]!=updated['context'][key]]
    require(differences,'scope descriptions already match; no amendment is needed')
    temporary=out/'PREACQUISITION_CANDIDATE.tmp'
    require(not temporary.exists(),'unfinished pre-acquisition candidate already exists')
    with (out/ORIGINAL).open('xb') as stream: stream.write(original_bytes)
    with temporary.open('xb') as stream: stream.write(updated_bytes)
    no_acquisition(out)
    require(path.read_bytes()==original_bytes,'manifest changed while preparing the amendment')
    os.replace(temporary,path)
    record=dict(schema='dense-depth-preacquisition-scope-amendment-1',
        amended_at_utc=datetime.now(timezone.utc).isoformat(),
        original_manifest=dict(path=str(out/ORIGINAL),sha256=sha(original_bytes)),
        amended_manifest=dict(path=str(path),sha256=sha(updated_bytes)),
        changes=differences,hardware_acquisition_started=False,
        no_acquisition_scope='this Dense-depth campaign; no jobs, markers, results or freeze existed',
        case_matrix_changed=False,source_hash_bindings_changed=False,
        prediction_or_parameter_code_changed=False,prices_or_outcomes_read=False,
        tool=dict(path=str(Path(__file__).resolve()),sha256=sha(Path(__file__).read_bytes())))
    with (out/RECEIPT).open('xb') as stream: stream.write(encode(record))
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    print(json.dumps(amend(a.out)),flush=True)
