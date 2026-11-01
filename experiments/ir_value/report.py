"""Summarize the bounded analytical E/S campaign without completeness claims."""
from __future__ import annotations
import argparse
import csv
from html import escape
import json
from pathlib import Path
import statistics

def generate(directory):
    directory=Path(directory)
    report=json.loads((directory/'REPORT.json').read_text(encoding='utf-8'))
    rows=report['rows'];valid=[r for r in rows if r['status']=='pass']
    rejected=[r for r in rows if r['status']!='pass' and r.get('error_type')=='InvalidInput' and
        r.get('error')=='first Decode Attention placement conflicts with the phase handoff destination']
    unexpected=[r for r in rows if r['status']!='pass' and r not in rejected]
    table=[]
    for s in report['summary']:
        values=s['ablations'];joint=s['best_ns']
        pure=min(values[k]['duration_ns'] for k in ('cpu_only','gpu_only') if values[k])
        e_only=values['demand_only']['duration_ns']
        s_only=pure
        selected=next(r for r in valid if r['id']==s['best'])
        table.append(dict(scenario=s['scenario'],regime=s['regime'],conditions=s['conditions'],
            joint_best_ns=joint,execution_only_ns=e_only,storage_only_ns=s_only,
            single_device_best_ns=pure,
            joint_latency_reduction_vs_execution_only=1-joint/e_only,
            joint_latency_reduction_vs_better_isolated=1-joint/min(e_only,s_only),
            joint_speedup_vs_better_isolated=min(e_only,s_only)/joint,
            independently_generated_reference_best_ns=s['reference_family_best_ns'],
            selected_point_reference_ns=selected['reference_family_best_ns'],
            reference_family_selection_regret=selected['reference_family_best_ns']/s['reference_family_best_ns']-1,
            selected_lowering_signed_gap=joint/selected['reference_family_best_ns']-1,
            signed_reference_gap=s['reference_gap_fraction'],best_id=s['best']))
    payload=dict(schema='ir-value-paper-data-1',analytic_only=True,
        expression_space='Five fixed structures; scoped CPU/GPU placement; demand or spare static preparation; same high capacity and fixed cache. Not full E5/S4 completeness.',
        semantic_support=dict(syntactic_conditions=report['conditions'],admitted_and_passed=report['passed'],
            fixed_host_handoff_contract_rejections=len(rejected),unexpected_failures=len(unexpected),
            raw_failed_status_count=report['failed'],
            rejection_note='CurrentPhaseHandoff(destination=dram) contract requires firstDecodeAttentionCPU. GPUDecodeAttentionpoints are unsupported by thatfixedcontract, not classified as universallyphysicallyillegal.',
            source_integrity=report['imported_production_sources_unchanged'],
            max_abs_error=max(r['numerical']['max_absolute_error'] for r in valid),
            generated_copy_count=sum(r['automatically_derived_copies'] for r in valid),
            copy_count_per_condition_range=[min(r['automatically_derived_copies'] for r in valid),max(r['automatically_derived_copies'] for r in valid)],
            explicit_user_copy_commands=sum(r['explicit_user_copy_commands'] for r in valid),
            independently_checked_obligations=['operand and tensor identities','declared placement','producer and transfer dependencies',
                'physical capacity and lifetime','complete COPY inventory/payload','closed-loop hidden/KV numerical values','MoE current-route visibility'],
            interpretation='Counts qualify tested configurations; automatic COPY counts are explanatory, not a productivity/compression score.'),
        reference_support=dict(distinct_duration_range=[min(r['reference_distinct_durations'] for r in valid),max(r['reference_distinct_durations'] for r in valid)],
            conditions_with_distinct_reference_schedules=sum(r['reference_distinct_durations']>1 for r in valid),
            negative_signed_gaps=sum(r['reference_gap_fraction']<0 for r in valid),
            max_signed_gap=max(r['reference_gap_fraction'] for r in valid),
            min_signed_gap=min(r['reference_gap_fraction'] for r in valid),
            median_signed_gap=statistics.median(r['reference_gap_fraction'] for r in valid),
            note='Independent bounded16tie-order family is a best-known comparison, not exhaustive optimum. Signed gap combines remaining lowering/reference-family differences; not pure expression loss.'),
        table=table)
    if 'corrective_rerun' in report:payload['corrective_rerun']=report['corrective_rerun']
    if 'reference_correction' in report:payload['reference_correction']=report['reference_correction']
    exact_path=Path(report.get('exact_dense_certificates_path',directory/'exact-dense-certificates/REPORT.json'))
    if exact_path.exists():
        payload['exact_dense_optimum_certificates']=json.loads(exact_path.read_text(encoding='utf-8'))['summary']
    (directory/'paper-data.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    with (directory/'paper-table.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    html=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>策略表达价值实验</title><style>body{font-family:"Microsoft YaHei","Times New Roman",serif;max-width:1150px;margin:40px auto;line-height:1.8;color:#161616;background:white}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #ccc;padding:9px;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#f4f4f4}.note{background:#f0f6fa;border-left:3px solid #0072b2;padding:14px}code{font-size:13px}</style>',
        '<h1>策略表达价值：跨场景的执行与准备联合选择</h1>',
        '<p class="note">本实验使用显式分析型成本，不是新的本机耗时测量。它检验共同Python→IR→编译流程中，执行放置与数据准备是否保留有用的组合选择。没有把有限参考调度族称为全局最优或抽象完备性证明。</p>',
        '<h2>固定实验与对照</h2><p>Dense整块、Attention拆分、双请求流水、P/D交接、因果MoE五类结构；H=8、I=16，1层，F32。每类穷举已声明作用域内的CPU/GPU放置，与按需加载/空闲准备静态权重组合。在1和50 ns/B两组显式搬运成本下分别运行。容量1MiB，固定缓存，主实验不研究容量受限的替换策略。</p>',
        '<p>联合选择优化放置与准备；仅执行选择固定为按需加载后重新优化全部放置；仅存储选择在全CPU或全GPU两种固定执行模式内重新优化准备；表中以两者较好的结果作为对照。</p>',
        '<h2>结果</h2><table><tr><th>结构 / 搬运成本</th><th>联合最好 µs</th><th>仅执行 µs</th><th>仅存储 µs</th><th>相对较好独立选择的耗时降低</th><th>独立参考族 µs</th><th>有符号差距</th></tr>']
    for r in table:
        html.append('<tr><td>'+escape(r['scenario']+' / '+r['regime'])+'</td>'+''.join('<td>'+v+'</td>' for v in
            [f"{r['joint_best_ns']/1000:.3f}",f"{r['execution_only_ns']/1000:.3f}",f"{r['storage_only_ns']/1000:.3f}",
             f"{100*r['joint_latency_reduction_vs_better_isolated']:.2f}%",f"{r['independently_generated_reference_best_ns']/1000:.3f}",f"{100*r['signed_reference_gap']:.3f}%"])+ '</tr>')
    html.extend(['</table><h2>语义与自动承担的义务</h2>',
        f"<p>原始笛卡尔网格共{report['conditions']}个声明条件，{report['passed']}个完成全部独立检查，{len(rejected)}个被当前固定主存P/D交接契约拒绝，其他失败{len(unexpected)}个。固定交接目标DRAM要求首个Decode Attention在CPU，枚举中的GPU Decode Attention与这项接口契约冲突；不把这些点称为普遍物理非法，也不从原始记录中删除。数值最大绝对差为{payload['semantic_support']['max_abs_error']:.3g}。高层策略没有手写COPY命令；各成功条件实际推导{payload['semantic_support']['copy_count_per_condition_range'][0]}–{payload['semantic_support']['copy_count_per_condition_range'][1]}条搬运，并与独立操作数/设备账本逐项一致。</p>",
        '<p>检查包括操作数与张量身份、设备放置、生产依赖、搬运字节与顺序、物理容量/生命周期、最终hidden与KV数值，以及MoE路由公开后才能执行对应专家。高层规则显式选择需要提前准备的静态权重；并不宣称编译器自己找到了最佳预取策略。没有用生成动作数构造人为的高自动化分数。</p>',
        '<h2>参考空间与结论边界</h2><p>独立参考从手写工作负载DAG构造COPY集合，使用同一CPU/GPU/共享PCIe资源和同一成本。每个候选比较16种计算/搬运资源内的优先规则（ID顺序、逆序、短任务优先、长任务优先）。它允许与默认编译不同的动作时序；但没有穷举所有合法调度，因此差距不是严格的最优表达损失。</p>',
        '<p>另外对最小Dense空间给出精确证书：每个候选从独立DAG计算前驱最长路径/资源总工作量下界，所有候选下界均不小于独立可行方案的耗时。两个带宽条件的上下界分别同为140100和143236ns，故这两个固定Dense空间内的最优值已被证明，默认编译选优也达到它。这个结论只限该结构、固定COPY集合、持久副本、非抢占算子和1MiB容量，不扩展到所有IR策略。</p>',
        f"<p>默认编译单个方案相对于有限参考族的有符号差距范围为{100*payload['reference_support']['min_signed_gap']:.2f}%至{100*payload['reference_support']['max_signed_gap']:.2f}%。负差距表示16种参考优先规则未覆盖默认编译的更好时序。不能据此声称低层优化空间总是很小；策略选优后悔值在paper-data.json中单列。</p>",
        '<p>这组数据可以说明执行与静态准备组合的收益、共同编译路径能否忠实保留选择；不能证明所有推理策略均已覆盖，也不能证明人的开发效率提高或硬件实测加速。缓存、驱逐和专家CPU回退的实测交互由另外的MoE实验提供。</p>',
        '<p>完整原始声明、IR、事件、数值/物理检查与独立参考保存在每个条件目录；PROTOCOL.json在首个运行前写入，生产模块字节逐项校验。</p></html>'])
    if 'corrective_rerun' in report:
        html.insert(-1,'<h2>保留的实验修正记录</h2><p>首轮MoE独立参考把请求输入准备误写为32ns；公共RequestFlow契约实际为32字节hidden＋4字节位置元数据，共36ns。128个MoE条件因此触发成本断言；生产模型与测量参数没有修改。原失败记录和源代码保留，单独冻结修正协议后重跑这128个条件并全部完成。汇总对应928个目标声明，实际原始尝试1056个，两个协议/产物均在REPORT.json中可追溯。</p>')
    if 'reference_correction' in report:
        html.insert(-1,'<p>独立审查还发现，共享权重的按需搬运可以由任一已就绪消费者触发，初版参考错误地固定由字典中的第一个请求触发。修正为消费者就绪条件的逻辑OR后，重新冻结分析协议并重算参考调度；800条实际执行轨迹均满足修正后的前驱/可选前驱约束，实际执行、成本与数值结果不变。初版参考与全部修正产物均保留。本文有符号参考差距和后悔值使用修正版。</p>')
    (directory/'REPORT_zh.html').write_text(''.join(html),encoding='utf-8',newline='\n')
    print(json.dumps(payload,ensure_ascii=False,indent=2))
    return payload

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True)
    generate(parser.parse_args().directory)
