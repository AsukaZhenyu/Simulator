"""Frozen high-level paired strategy edits; no generated compiler inputs."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT, ROOT/'src', ROOT/'mapping', ROOT/'modeling'):
    sys.path.insert(0, str(folder))
from experiments import dac_validation_e1 as e1
from experiments.dac_validation_e1_oracle import declared_dag

OUT = ROOT/'build/coordination-metrics-20261030/physical'
GEOMETRIES = (('development', 8, 1), ('holdout', 16, 3))
PAIRS = (
 ('D01','Dense','projection placement',{'projection_device':'cpu'},{'projection_device':'gpu'}),
 ('D02','Dense','weight preparation',{'preparation':'demand'},{'preparation':'spare'}),
 ('A01','refinedAttention','QK and softmax placement',{'attention_front_device':'cpu'},{'attention_front_device':'gpu'}),
 ('A02','refinedAttention','weight preparation',{'preparation':'demand'},{'preparation':'spare'}),
 ('F01','FFNpartition','physical weight storage',{'ffn_storage':'materialized_partition_replaces_parent'},{'ffn_storage':'shared_parent_slice_views'}),
 ('F02','FFNpartition','weight preparation',{'preparation':'demand'},{'preparation':'spare'}),
 ('R01','requestpipeline','ready request ordering',{'pipeline':'graph_order'},{'pipeline':'decode_first'}),
 ('R02','requestpipeline','weight preparation',{'preparation':'demand'},{'preparation':'spare'}),
 ('P01','PDhandoff','explicit all-layer KV boundary',{'handoff':False},{'handoff':True}),
 ('P02','PDhandoff','prefill placement',{'prefill_device':'cpu'},{'prefill_device':'gpu'}),
 ('M01','causalMoE','current revealed expert placement',{'expert_device':'cpu'},{'expert_device':'gpu'}),
 ('M02','causalMoE','weight retention',{'cache_rule':'none'},{'cache_rule':'lru'}),
)

def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')

def declarations():
    for split,h,q in GEOMETRIES:
        for pair,family,change,left,right in PAIRS:
            for label,edit in (('A',left),('B',right)):
                s=e1.spec('A06' if family=='causalMoE' else 'A01',h,q)
                s.update(id=f'{pair}-{split}-{label}',layers=2 if family in {'requestpipeline','PDhandoff','causalMoE'} else 1,
                    requests=2 if family=='requestpipeline' else 1,
                    decode_steps=2 if family=='causalMoE' else 1 if family in {'requestpipeline','PDhandoff'} else 0,
                    attention_refined=family=='refinedAttention',ffn_refined=family=='FFNpartition',
                    handoff=family=='PDhandoff',prefetch=False,cache=True,
                    physical_capacity_bytes=1048576,cache_budget_bytes=262144,
                    memory_mode='separate',pattern='A03' if family=='FFNpartition' else 'A06' if family=='causalMoE' else 'A01')
                intent=dict(default_device='gpu',projection_device='cpu',attention_device='cpu',
                    attention_front_device='cpu',preparation='demand',pipeline='graph_order',
                    ffn_storage='materialized_partition_replaces_parent',expert_device='gpu',
                    expert_rule='fixed',cache_rule='lru',prefill_device='gpu',decode_device='cpu',
                    handoff=family=='PDhandoff',release='last_use',future_expert=1)
                intent.update(edit)
                s['handoff']=intent['handoff']
                if intent['ffn_storage']=='shared_parent_slice_views':s['pattern']='A12'
                yield dict(id=s['id'],pair=pair,family=family,split=split,variant=label,change=change,
                    workload=s,intent=intent,geometry=dict(hidden=h,intermediate=2*h,heads=2,kv_heads=1,
                    dtype='F32',prefill_tokens=q),hardware=dict(memory_mode='separate',dram_capacity_bytes=1048576,
                    vram_capacity_bytes=1048576,cache_budget_bytes=262144,h2d_ns_per_byte=1,d2h_ns_per_byte=1),
                    cost_contract=dict(kind='declared_analytical_not_measured',macro_workspace_bytes=16,
                    state_update_workspace_bytes=16,refined_stage_workspace_bytes=4),
                    routes=[asdict(r) for r in e1.routes_for(s)])

def freeze(output=OUT):
    output=Path(output)
    rows=list(declarations())
    document=dict(schema='coordination-physical-protocol-1',frozen_utc=datetime.now(timezone.utc).isoformat(),
        pair_count=12,geometries=[dict(split=x,hidden=h,q=q) for x,h,q in GEOMETRIES],
        paired_contrast_count=24,nominal_variant_conditions=48,
        invalid_controls=12,invalid_control_definition='Each development B variant with VRAM=1 byte; at least one mandatory GPU operand/output exceeds 1 byte. Capacity rejection separate from Unsupported/unpriced/compiler failure.',
        physical_metrics=['h2d_bytes','d2h_bytes','peak_dram_bytes','peak_vram_bytes'],
        delta_definition='B minus A; error=(observed_B-observed_A)-(independent_ledger_B-independent_ledger_A); zero-reference deltas use absolute bytes only.',
        independence='Expected shapes, operands and byte sizes from handwritten e1 algebra; physical ledger replays actual actions from allocation/reader lifetimes, not production summaries. Generated graph must match expected DAG exactly.',
        capacity_positive_reference='A conservative legal serial schedule exists: all declared weights plus all activations and workspace fit within each 1MiB pool; no cache capacity conflict. Existence is not inferred from successful production run.',
        numerical_scope='Two independent arithmetic implementations of complete declared Dense ports; synthetic opaque MoE ports and model-owned trace router IDs, not production MoE numerical kernels.',
        numerical_tolerance=dict(atol=5e-5,rtol=5e-5),
        timing='One compile and one simulation wall time per condition on current host, independently timed; exploratory implementation latency, no native inference or comparative runtime claim.',
        efficiency_counts='Record source/public operation and tensor counts, dynamic COMPUTE/COPY/EVICT/ADVANCE counts, and unique IR/action signatures. Nominal duplicate baselines remain identified; no inflated independent count.',
        no_core_changes=True,no_native_measurement=True,conditions=rows)
    for row in rows:
        path=output/'public-inputs'/row['id']
        write(path/'intent.json',row)
        write(path/'declared-workload.json',declared_dag(row['workload']))
    write(output/'physical_PROTOCOL.json',document)
    return document

if __name__=='__main__':
    freeze()
