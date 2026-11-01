"""Read-only cross-check of saved native results and replayable strategy IR.

No comparison-builder/oracle helper is imported. Recomputes the native grouping,
macro costs, exact copy-count/byte formulas and native trace structure from the
retained receipts, then independently reloads each compiler input from disk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
for folder in (ROOT,ROOT/'modeling',ROOT/'mapping'):
    sys.path.insert(0,str(folder))
from tensor_mapping.strategy_compiler import compile_strategy_file,run_strategy


def audit(out):
    out=Path(out).resolve()
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    summary=read(out/'summary.json')
    native=read(out/'compass-native-whole-blocks.json')
    assert len(native['cases'])==8
    assert summary['external_core_changes']==0
    assert native['external_transformer_source_unchanged']
    assert not native['in_memory_adapter']['cost_formula_changed']
    groups={'projection':{'Q_proj','K_proj','V_proj'},
            'attention':{'Q_mul_K','A_softmax','A_mul_V','H_matmul0','layer_norm0'},
            'dense_ffn':{'H_matmul1','H_gelu','H_matmul2','layer_norm1'}}
    placements={'all_cpu':('cpu','cpu','cpu'),'all_gpu':('gpu','gpu','gpu'),
                'cpu_attention':('gpu','cpu','gpu'),'cpu_ffn':('gpu','gpu','cpu')}
    rows=0
    for case in native['cases']:
        assert len(case['operations'])==12
        assert len(case['native_metadata_operations'])==8
        assert not any(r['priced_by_native_whole_block'] or r['physical_zero_cost_claimed']
                       for r in case['native_metadata_operations'])
        total=sum(r['roofline_seconds']+r['overhead_seconds'] for r in case['operations'])
        assert math.isclose(total,case['native_whole_block_seconds'],rel_tol=1e-12,abs_tol=1e-18)
        macro={kind:sum(r['inclusive_seconds'] for r in case['operations'] if r['id'] in names)
               for kind,names in groups.items()}
        h,q=case['hidden'],case['tokens']
        expected_copies={'all_cpu':0,'all_gpu':8,'cpu_attention':11,'cpu_ffn':6}
        expected_bytes={'all_cpu':0,'all_gpu':24*h*h+4*h*q,
                        'cpu_attention':22*h*h+12*h*q,'cpu_ffn':8*h*h+4*h*q}
        for name,devices in placements.items():
            directory=out/'compass'/case['id']/name
            scenario=read(directory/'scenario.json')
            workload=read(directory/'workload.json')
            copies=read(directory/'copies.json')
            declared=read(directory/'summary.json')
            assert len(workload['operations'])==3
            assert len(copies)==expected_copies[name]
            assert sum(r['bytes'] for r in copies)==expected_bytes[name]
            tensors={r['id']:r for r in workload['tensors']}
            assert sum(r['storage_bytes'] for r in tensors.values() if r['role']=='weight')==24*h*h
            assert all(tensors[t]['storage_bytes']==2*h*q for t in ('X','Q','K','V','H0','Y'))
            assert tensors['W1']['ne']==[4*h,h,1,1] and tensors['W2']['ne']==[h,4*h,1,1]
            expected_compute=0
            for kind,device in zip(('projection','attention','dense_ffn'),devices):
                cost=math.ceil(macro[kind]*1e9*(4 if device=='cpu' else 1))
                assert scenario['costs']['compute'][kind][device]['duration_ns']==cost
                expected_compute+=cost
            expected_copy=sum(5000+math.ceil(r['bytes']/16) for r in copies)
            expected=expected_compute+expected_copy
            assert declared['engine_ns']==expected==declared['independent_serial_ns']
            replay=run_strategy(compile_strategy_file(directory/'strategy-ir.json'))
            assert replay.evaluation.status=='valid' and replay.evaluation.makespan_ns==expected
            assert replay.final_state is not None
            assert declared['extra_workspace_zero_is_scenario_assumption']
            assert not declared['numerical_kernel_equivalence_verified']
            rows+=1
    assert rows==32
    serving=read(out/'serving/summary.json')
    assert len(serving['cases'])==24
    expected_block=['layernorm','qkv_proj','rotary_emb','attention','o_proj','layernorm',
                    'gate_up_proj','act_fn','down_proj']
    trace_rows=0
    for case in serving['cases']:
        trace=read(out/'serving/cases'/case['id']/'rows.json')
        expected=['embedding']+expected_block*case['layers']+['final_layernorm','lm_head','sampler']
        assert [r[0] for r in trace]==expected
        assert len(trace)==4+9*case['layers']
        assert trace[0][2]=='REMOTE:0' and trace[-1][6]=='REMOTE:0'
        assert sum(int(r[1]) for r in trace)==case['total_declared_compute_ns']
        h,q=case['hidden'],case['tokens']
        for row in trace:
            if row[0]=='qkv_proj':
                assert [int(row[i]) for i in (3,5,7)]==[2*q*h,6*h*h,6*q*h]
                assert int(row[9])==0
            if row[0] in ('gate_up_proj','down_proj'):
                assert row[4]==('REMOTE:0' if case['placement']=='ffn_remote_weights' else 'LOCAL')
        trace_rows+=len(trace)
        if case['phase']=='prefill':
            pd_trace=read(out/'serving/cases'/case['id']/'pd-byte-witness-rows.json')
            assert [r[0] for r in pd_trace]==expected
            assert all(int(r[9])==4*q*h for r in pd_trace if r[0]=='qkv_proj')
            assert sum(int(r[9]) for r in pd_trace if r[0]=='qkv_proj')==case['qkv_pd_bytes']
    assert serving['fixture_profiles_are_synthetic']
    assert serving['native_backend_workload_timings']==0
    hashes=read(out/'artifact-hashes.json')
    for relative,expected in hashes.items():
        assert hashlib.sha256((out/relative).read_bytes()).hexdigest()==expected,relative
    result=dict(schema='dac-followup-external-audit-1',passed=True,
                native_whole_blocks_checked=8,actual_ir_file_replays=rows,
                native_whole_frontend_traces_checked=24,native_trace_rows=trace_rows,
                additional_pd_byte_trace_witnesses=16,
                bound_artifacts=len(hashes),independent_cost_accuracy_claim=False,
                complete_native_memory_accuracy_claim=False,
                real_backend_timing_for_servingsim=False)
    (out/'independent-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    audit(parser.parse_args().out)
