"""Run pinned ServingSim whole-request trace frontend with explicit fixtures.

These are deterministic synthetic profile fixtures, not actual vLLM timings.
The original model/architecture, lookup, placement and full-trace functions run
unchanged. The absent ASTRA binary is attempted and retained as a backend block.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def dump(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8",newline="\n")


def main(out):
    sys.path.insert(0,str(Path.cwd()))
    from serving.core import trace_generator as tg
    from serving.core.request import Batch, Request

    out=Path(out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    source=Path(tg.__file__)
    source_before=hashlib.sha256(source.read_bytes()).hexdigest()
    repo=source.parents[2]
    architecture=tg._load_architecture("llama")
    expected_block=(architecture['sequence']['pre_attn']+
                    architecture['sequence']['post_attn']+
                    architecture['sequence']['mlp_dense'])
    assert expected_block==['layernorm','qkv_proj','rotary_emb','attention',
                            'o_proj','layernorm','gate_up_proj','act_fn','down_proj']
    fixtures=out/'fixtures'
    tp=fixtures/'tp1'
    tp.mkdir(parents=True,exist_ok=True)
    dense_names=list(architecture['catalog']['dense'])
    rows=['layer,tokens,time_us']
    for index,name in enumerate(dense_names):
        for tokens in (1,16,64):
            rows.append(f'{name},{tokens},{(index+1)*3+tokens/8}')
    (tp/'dense.csv').write_text('\n'.join(rows)+'\n',encoding='utf-8',newline='\n')
    rows=['layer,sequences,time_us','lm_head,1,11','sampler,1,7']
    (tp/'per_sequence.csv').write_text('\n'.join(rows)+'\n',encoding='utf-8',newline='\n')
    rows=['prefill_chunk,kv_prefill,n_decode,kv_decode,time_us',
          '16,0,0,0,25','64,0,0,0,80','0,0,1,128,17']
    (tp/'attention.csv').write_text('\n'.join(rows)+'\n',encoding='utf-8',newline='\n')
    native_tables=tg._build_tp_tables(str(tp))
    cases=[]
    for layers in (2,4):
        for h in (128,256):
            config=dict(model_type='llama',hidden_size=h,intermediate_size=4*h,
                        num_hidden_layers=layers,num_attention_heads=h//64,
                        num_key_value_heads=h//64,head_dim=64,vocab_size=1024,
                        torch_dtype='float16')
            config_path=fixtures/f'model-h{h}-l{layers}.json'
            dump(config_path,config)
            # get_config accepts an absolute filename stem via os.path.join.
            # Only fixture/model inputs are created; no external repo edits.
            model=str(config_path.with_suffix(''))
            db=dict(meta={},architecture=architecture,variant='fp16',hardware='declared-fixture',
                    model=model,root=str(fixtures),available_tps=[1],tables={1:native_tables})
            tg._perf_db_cache[('declared-fixture',model,'fp16')]=db
            for phase,q,hist in [('prefill',16,0),('prefill',64,0),('decode',1,128)]:
                batch=Batch(0,model,q,hist,[q],[hist+q],int(phase=='prefill'),int(phase=='decode'),
                            [q] if phase=='prefill' else [],[0] if phase=='prefill' else [],
                            [hist] if phase=='decode' else [],0,0,pd_kv_send_tokens=q if phase=='prefill' else 0)
                batch.requests=[Request(0,model,max(q,hist),max(q,hist)+1,0,0)]
                for placement_name in ('resident_weights','ffn_remote_weights'):
                    placement=dict(default={'weights':'LOCAL','kv_loc':'LOCAL','kv_evict_loc':'REMOTE:0'},
                                   layer={},block=[])
                    if placement_name=='ffn_remote_weights':
                        placement['layer']={name:{'weights':'REMOTE:0'} for name in ('gate_up_proj','down_proj')}
                    trace,starts=tg._synthesize_trace('declared-fixture',model,config,1,1,1,1,
                          None,0,0,batch,q+hist,
                          placement,True,None,False,None,None,2,'fp16')
                    pd_trace=None
                    if phase=='prefill':
                        pd_trace,_=tg._synthesize_trace('declared-fixture',model,config,1,1,1,1,
                              'prefill',0,0,batch,q+hist,placement,True,None,False,None,None,2,'fp16')
                        assert all(int(r[9])==2*q*h*2 for r in pd_trace if r[0]=='qkv_proj')
                    expected_names=['embedding']+expected_block*layers+architecture['sequence']['head']
                    assert [r[0] for r in trace]==expected_names
                    assert len(trace)==4+9*layers
                    assert starts==[1+9*j for j in range(layers)]
                    assert trace[0][2]=='REMOTE:0' and trace[-1][6]=='REMOTE:0'
                    assert all(len(r)==11 for r in trace)
                    assert all(int(r[1])>0 and min(int(r[3]),int(r[5]),int(r[7]))>=0 for r in trace)
                    # Independent geometry formulas at public trace ports.
                    for r in trace:
                        if r[0]=='qkv_proj':
                            assert (int(r[3]),int(r[5]),int(r[7]))==(q*h*2,3*h*h*2,3*q*h*2)
                            assert int(r[9])==0
                        if r[0]=='gate_up_proj':
                            assert int(r[5])==2*h*(4*h)*2
                        if r[0] in ('gate_up_proj','down_proj'):
                            assert r[4]==('REMOTE:0' if placement_name=='ffn_remote_weights' else 'LOCAL')
                    case_id=f'h{h}-l{layers}-{phase}-q{q}-h{hist}-{placement_name}'
                    row=dict(id=case_id,layers=layers,hidden=h,phase=phase,tokens=q,history=hist,
                             placement=placement_name,rows=len(trace),block_starts=starts,
                             total_declared_compute_ns=sum(int(r[1]) for r in trace),
                             colocated_qkv_comm_bytes=sum(int(r[9]) for r in trace if r[0]=='qkv_proj'),
                             qkv_pd_bytes=sum(int(r[9]) for r in (pd_trace or []) if r[0]=='qkv_proj'),
                             additional_pd_trace_is_byte_witness_only=pd_trace is not None,
                             remote_weight_rows=sum(r[4]=='REMOTE:0' and int(r[5])>0 for r in trace),
                             complete_native_frontend=True,astra_backend_executed=False,
                             independent_timing_accuracy_claimed=False,status='pass')
                    dump(out/'cases'/case_id/'rows.json',trace)
                    dump(out/'cases'/case_id/'indexed-cols.json',tg.indexed_cols(trace))
                    dump(out/'cases'/case_id/'summary.json',row)
                    if pd_trace is not None:
                        dump(out/'cases'/case_id/'pd-byte-witness-rows.json',pd_trace)
                    cases.append(row)
    binary=repo/'astra-sim/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra'
    try:
        run=subprocess.run([str(binary),'--help'],capture_output=True,text=True,timeout=10)
        backend=dict(executable=str(binary),exists=binary.exists(),returncode=run.returncode,
                     stdout=run.stdout,stderr=run.stderr,status='available_but_workload_not_executed')
    except (FileNotFoundError,OSError) as error:
        backend=dict(executable=str(binary),exists=binary.exists(),error_type=type(error).__name__,
                     error=str(error),status='blocked_missing_backend_binary')
    except subprocess.TimeoutExpired as error:
        backend=dict(executable=str(binary),exists=binary.exists(),status='backend_help_timeout',error=str(error))
    source_after=hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_before==source_after
    summary=dict(schema='dac-followup-servingsim-whole-trace-1',cases=cases,
                 whole_frontend_traces=len(cases),native_rows=sum(r['rows'] for r in cases),
                 additional_pd_byte_trace_witnesses=sum(r['additional_pd_trace_is_byte_witness_only'] for r in cases),
                 source_unchanged=True,source_sha256=source_before,
                 fixture_profiles_are_synthetic=True,backend=backend,
                 native_backend_workload_timings=0,
                 port_shape_pd_placement_checks_passed=len(cases),
                 scope='Unmodified native complete Llama trace frontend; two/four layers, prefill/decode, resident/remote FFN weight placement',
                 cpu_execution_strategy_not_claimed=True,
                 limitation='Trace frontend execution does not replace missing ASTRA execution. Declared fixed profile fixtures do not validate performance or reproduce vLLM profiling.')
    dump(out/'summary.json',summary)
    print('FOLLOWUP_JSON='+json.dumps(dict(traces=len(cases),backend=backend['status'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    main(parser.parse_args().out)
