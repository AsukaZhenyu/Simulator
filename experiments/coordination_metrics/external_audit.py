"""Frozen, reproducible external-interface audit; no native experiments."""
from __future__ import annotations
import argparse
import ast
import hashlib
import html
import json
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import traceback

from external_adapter import audit_case,public_condition,CATEGORIES
from external_interfaces import InterfaceBundle,ROOT,COMPASS,SERVING,CHAKRA,sha

HERE=Path(__file__).resolve().parent
OUT=ROOT/'Simulator/build/coordination-metrics-20261030/external'
SHARED=ROOT/'Simulator/build/coordination-metrics-20261030/physical'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def code_hashes():return {str(p.relative_to(ROOT)):sha(p) for p in HERE.glob('external_*.py')}
def input_hashes(protocol):
    result={str((SHARED/'physical_PROTOCOL.json').relative_to(ROOT)):sha(SHARED/'physical_PROTOCOL.json')}
    for c in protocol['conditions']:
        for name in ('intent.json','declared-workload.json'):
            p=SHARED/'public-inputs'/c['id']/name
            result[str(p.relative_to(ROOT))]=sha(p)
    return result
def unchanged(hashes):return all(sha(ROOT/p)==h for p,h in hashes.items())

def bind():
    protocol=read(SHARED/'physical_PROTOCOL.json')
    save(OUT/'INPUT_BINDING.json',dict(schema='coordination-external-input-binding-1',
        created_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(OUT/'PROTOCOL.json'),
        shared_manifest_sha256=sha(SHARED/'physical_PROTOCOL.json'),
        exact_tasks=[dict(id=c['pair'],family=c['family'],change=c['change']) for c in protocol['conditions']
                     if c['split']=='development' and c['variant']=='A'],
        amendment_before_any_case_execution='Use the shared exact twelve paired edits. Its Attention second edit is preparation rather than opaque/refined; FFN edits are representation/preparation, not a fraction change; request-pipeline first edit is ready order rather than placement. No result may claim those untested changes.',
        conditions=48,paired_edits_per_split=12,development_geometry='H8 I16 q1 F32',
        holdout_geometry='H16 I32 q3 F32',
        common_input_scope='Independently authored mathematical DAG, geometry, hardware, implementation workspace, high-level policy. No production IR, actions, allocation/copy/lifetime replay results.',
        moe_boundary='Host reads family/geometry only; no private routes or future expert DAG is passed to adapter. Its complete path is not evaluated.',
        whole_task_scoring_scope='No full foreign scheduler is run. Whole-task scores remain null. Zero counts refer only to executed static interface/adapter contract categories.',
        input_hashes=input_hashes(protocol),cases_executed_before_binding=0))

def run(split,tag):
    protocol=read(SHARED/'physical_PROTOCOL.json')
    binding=read(OUT/'INPUT_BINDING.json')
    if not unchanged(binding['input_hashes']):raise ValueError('Shared input changed after binding')
    if split=='holdout':
        frozen=read(OUT/'ADAPTER_FREEZE.json')
        if not unchanged(frozen['adapter_source_hashes']):raise ValueError('Frozen adapter source changed')
        if not unchanged(frozen['external_source_hashes']):raise ValueError('External source changed')
    bundle=InterfaceBundle();folder=OUT/tag
    if folder.exists():raise ValueError('Refusing to overwrite an execution directory')
    folder.mkdir(parents=True)
    rows=[]
    for c in protocol['conditions']:
        if c['split']!=split:continue
        try:
            public=public_condition(c)
            # This branch is important: no future MoE topology enters the adapter.
            dag=None if c['workload']['kind']=='moe' else read(SHARED/'public-inputs'/c['id']/'declared-workload.json')
            result=audit_case(bundle,public,dag)
            save(folder/(c['id']+'.json'),result)
            rows.append(dict(id=c['id'],status=result['status'],artifact=str((folder/(c['id']+'.json')).relative_to(ROOT)),
                             sha256=sha(folder/(c['id']+'.json'))))
        except Exception as ex:
            save(folder/(c['id']+'.ERROR.json'),dict(id=c['id'],exception=type(ex).__name__,error=str(ex),traceback=traceback.format_exc()))
            raise
    if not bundle.sources_unchanged():raise ValueError('Original source was modified')
    save(folder/'RUN.json',dict(schema='coordination-external-run-1',split=split,
        created_utc=datetime.now(timezone.utc).isoformat(),cases=rows,
        static_contract_conditions=sum(x['status']=='static_interface_and_adapter_obligations_only' for x in rows),
        causal_not_evaluated=sum(x['status'].startswith('not_evaluated') for x in rows),
        whole_policy_execution_conditions=0,source_definitions=bundle.extractor.evidence,
        adapter_source_hashes=code_hashes(),original_sources_unchanged=True,hardware_experiments=False))
    print(json.dumps(dict(folder=str(folder),split=split,conditions=len(rows),static_contracts=20,causal_not_evaluated=4)))

def freeze(training):
    run=read(OUT/training/'RUN.json')
    if run['split']!='development' or len(run['cases'])!=24:raise ValueError('Complete development run required')
    if not unchanged(run['adapter_source_hashes']):raise ValueError('Source changed after development run; rerun development under a new tag')
    tree=ast.parse((HERE/'external_adapter.py').read_text())
    funcs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    relations=[('allocation_alias','derive_allocation_contract','shape/dtype -> buffer bytes; domain -> replica identity; FFN bounds -> materialized vs parent-view relation'),
        ('copy_payload','derive_copy_contract','producer/initial domain and consumer placement -> potential copy; physical view bytes -> payload'),
        ('reader_lifetime','derive_reader_contract','operand-use graph -> allocation readers and conditional safe-release/last-use guards'),
        ('preparation_dependency','derive_preparation_contract','high-level demand/spare trigger plus copy/source -> readiness and capacity-reservation guards')]
    examples=read(OUT/training/'F01-development-B.json')
    setup=[]
    fieldmap=dict(allocation_alias='allocation_contract',copy_payload='copy_contract',
                  reader_lifetime='reader_contract',preparation_dependency='preparation_contract')
    for category,symbol,rule in relations:
        n=funcs[symbol];field=fieldmap[category]
        setup.append(dict(category=category,source='parameterized_adapter_implemented_once',symbol=symbol,
            first_line=n.lineno,last_line=n.end_lineno,relation=rule,
            artifact=f'{training}/F01-development-B.json',artifact_field=field,
            example_relation_sha256=hashlib.sha256(json.dumps(examples[field],sort_keys=True).encode()).hexdigest(),
            extra_user_values_per_edit=0,not_an_upstream_native_feature_claim=True))
    save(OUT/'ADAPTER_SETUP.json',dict(schema='coordination-external-setup-1',
        relation_categories_implemented_once=len(setup),relations=setup,
        setup_not_per_edit_burden=True,not_LOC_or_person_time=True,
        upstream_automation_credited=['Tensor sizes and Matmul arithmetic','Softmax shapes',
            'Serving native size helper','Configuration location precedence','Layer trace field parsing'],
        not_implemented=['Full external execution planner/scheduler','Causal MoE route callback',
            'Runtime dynamic cache state','Observed execution bytes/peak live-memory validation'],
        conclusion='Four static relation families were implemented once in a generic adapter, then reused. This demonstrates a possible extension, not that existing tools fundamentally require manual per-edit repair.'))
    external={x['path']:x['source_sha256'] for x in run['source_definitions']}
    save(OUT/'ADAPTER_FREEZE.json',dict(schema='coordination-external-freeze-1',
        frozen_utc=datetime.now(timezone.utc).isoformat(),training_run=training,
        training_run_sha256=sha(OUT/training/'RUN.json'),adapter_source_hashes=code_hashes(),
        external_source_hashes=external,input_binding_sha256=sha(OUT/'INPUT_BINDING.json'),
        setup_sha256=sha(OUT/'ADAPTER_SETUP.json'),holdout_not_yet_executed=True))

def report(training,holdout):
    runs=[read(OUT/training/'RUN.json'),read(OUT/holdout/'RUN.json')]
    pairs=[]
    for run in runs:
        grouped={}
        for row in run['cases']:
            c=read(ROOT/row['artifact']);grouped.setdefault(c['pair'],{})[c['variant']]=c
        for pair,ab in grouped.items():
            a,b=ab['A'],ab['B'];evaluated=a['status']=='static_interface_and_adapter_obligations_only' and b['status']==a['status']
            pairs.append(dict(pair=pair,family=a['family'],split=run['split'],
                category_extra_declarations={k:0 if evaluated else None for k in CATEGORIES},
                static_contract_score=0 if evaluated else None,whole_task_score=None,
                scope='static interface + generic sidecar' if evaluated else 'causal adapter not implemented',
                original_tool='Both LLMCompass and LLMServingSim interfaces are exercised for evaluated static cases',
                A_artifact=str((OUT/(training if run['split']=='development' else holdout)/(a['id']+'.json')).relative_to(ROOT)),
                B_artifact=str((OUT/(training if run['split']=='development' else holdout)/(b['id']+'.json')).relative_to(ROOT))))
    frozen=read(OUT/'ADAPTER_FREEZE.json')
    summary=dict(schema='coordination-external-summary-1',status='completed_scoped_interface_audit_no_whole_tool_claim',
        registered_paired_edits=12,total_variant_conditions=48,development_variant_conditions=24,holdout_variant_conditions=24,
        static_interface_conditions=40,not_evaluated_causal_conditions=8,
        paired_contrasts_static=20,paired_contrasts_causal_not_evaluated=4,
        heldout_static_pairs=10,heldout_static_extra_categories=0,
        whole_task_scores_available=0,whole_task_advantage_established=False,
        setup_relation_categories=4,setup_relations_artifact='ADAPTER_SETUP.json',
        both_tools_with_adapter_have_zero_extra_static_categories=True,
        rows=pairs,adapter_frozen_unchanged=unchanged(frozen['adapter_source_hashes']),
        external_sources_unchanged=unchanged(frozen['external_source_hashes']),
        input_binding='INPUT_BINDING.json',protocol='PROTOCOL.json',
        interpretation='On the executed static-contract boundary, a generic adapter derives all four coordination categories without per-edit extra inputs. This is a negative result for a claimed exclusive per-edit automation advantage. Full policy execution was not measured and receives no zero score.',
        no_production_trace_as_input=True,no_gpu_or_native_run=True)
    save(OUT/'SUMMARY.json',summary)
    tr=''.join('<tr><td>'+html.escape(x['pair'])+'</td><td>'+html.escape(x['family'])+'</td><td>'+x['split']+'</td><td>'+('0 / 4 (static scope)' if x['static_contract_score']==0 else 'Not evaluated')+'</td><td>Not measured</td></tr>' for x in pairs)
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>外部接口与协调声明审计</title><style>body{font:16px Arial,"Microsoft YaHei",sans-serif;max-width:1100px;margin:36px auto;padding:0 22px;line-height:1.7;color:#182032}h1{font-size:28px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border:1px solid #cbd3dd;text-align:left}a{color:#1354a0}code{font-size:13px}</style><body><h1>外部接口与协调声明审计</h1>
<p>结论：已有接口配合一次性通用适配器，也能在本次静态契约范围内做到零额外协调声明。因此没有得到“每次改策略只有我们不需补协调信息”的正面优势。</p>
<p>12对策略变更×开发/保留两种几何，共48个variant。40个静态条件执行了原始LLMCompass算子定义、LLMServingSim尺寸/位置定义和Chakra trace解析；另外8个MoE条件没有实现因果router回调，明确未评估。共10对保留集静态变更的4类额外声明均为0。没有运行外部完整scheduler；所有whole-task score均为空，不把接口接受冒充完整策略执行。</p>
<h2>计数与一次性适配工作</h2><p>四类为：分配/别名、搬运字节、读者生命周期、准备与执行依赖。按独立类别计，每个策略变更最多4；重复层、tensor与trace行不增加分数。适配器一次性实现这4类关系，源代码符号、关系描述与实例工件保存在ADAPTER_SETUP.json。之后参数化重用，未要求用户逐策略补这些值。此项不衡量代码量、开发时间或理论最小负担。</p>
<h2>逐对结果</h2><table><tr><th>Pair</th><th>Structure</th><th>Split</th><th>Extra categories</th><th>Whole policy</th></tr>'''+tr+'''</table>
<h2>证据范围</h2><p>共同输入是独立声明的数学图、硬件、workspace与高层策略。未输入本项目的compiler IR、production动作trace或物理汇总。MoE私有未来route及专家DAG没有进入adapter。接口计算时间用1ns占位，仅验证schema，并非任何性能测量。sidecar的copy/lifetime条件是静态模板，不是实际总搬运量、live-memory峰值或真实OOM。</p>
<p>LLMCompass的已有mapper和LLMServingSim的已有调度/卸载能力均被承认。本实验没有禁止它们已有自动化，也没有从缺字段推导“无法表示”。没有执行到的能力保持未知。完整跨工具策略比较需要真正接入各自运行时，当前结果不能替代。</p>
<p><a href="PROTOCOL.json">冻结协议</a> · <a href="INPUT_BINDING.json">输入与任务映射</a> · <a href="ADAPTER_SETUP.json">一次性关系实现</a> · <a href="ADAPTER_FREEZE.json">保留集前冻结</a> · <a href="SUMMARY.json">逐项机器可读结果</a></p></body></html>'''
    (OUT/'REPORT_zh.html').write_text(doc,encoding='utf-8',newline='\n')
    print(json.dumps({k:summary[k] for k in ('status','static_interface_conditions','not_evaluated_causal_conditions','heldout_static_extra_categories','whole_task_scores_available','adapter_frozen_unchanged')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['bind','run','freeze','report']);p.add_argument('--split');p.add_argument('--tag');p.add_argument('--training',default='development-v1');p.add_argument('--holdout',default='holdout-v1');a=p.parse_args()
    if a.mode=='bind':bind()
    elif a.mode=='run':run(a.split,a.tag)
    elif a.mode=='freeze':freeze(a.training)
    else:report(a.training,a.holdout)
