"""Readable Chinese report and English evidence summary from a frozen campaign.

Run with an existing Matplotlib-capable Python. No simulation, profiling or
parameter fitting occurs here. Figures read the retained analytical results.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import hashlib
from html import escape
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from experiments.strategy_comparison_analysis import analyze, plots

EXEC_ZH={'e0':'LLM 宏任务在 GPU','e1':'LLM 宏任务在 CPU',
         'e2':'FFN／专家在 CPU，其余在 GPU','e3':'Attention 在 CPU，其余在 GPU',
         'e4':'专家按当前驻留选择设备','e5':'专家按当前路径成本选择设备'}
STORE_ZH={'s0':'LRU，不预取','s1':'LRU，下一层预取',
          's2':'历史频率，不预取','s3':'历史频率，下一层预取'}
SHAPE_ZH={'short_prefill':'短 Prefill','long_prefill':'长 Prefill',
          'decode_heavy':'短输入＋多步 Decode','mixed':'长输入＋多步 Decode'}
ROUTE_ZH={None:'不适用','hot':'固定热点','rotating':'轮换专家','phase_shift':'P/D 热点切换'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()


def ns_ms(value):
    return '不可用' if value is None else f'{value/1e6:.6f}'


def ns_us(value):
    return '不可用' if value is None else f'{value/1e3:,.3f}'


def kib(value):
    return '不可用' if value is None else f'{value/1024:,.3f}'


def pct(value):
    return '不可用' if value is None else f'{100*value:+.2f}%'


def scenario_name(row):
    scenario=row['scenario_id']
    shape=next(k for k in SHAPE_ZH if f'-{k}' in scenario)
    route=ROUTE_ZH[row.get('route_pattern')]
    return f"{row['model_kind'].upper()}，{SHAPE_ZH[shape]}"+(f'，{route}' if route!='不适用' else '')


class Document:
    def __init__(self,title):
        self.title=title;self.md=[];self.html=[]
        self.h(1,title)

    def h(self,level,text):
        self.md.append('#'*level+' '+text+'\n')
        self.html.append(f'<h{level}>{escape(text)}</h{level}>')

    def p(self,text):
        self.md.append(text+'\n')
        self.html.append('<p>'+escape(text)+'</p>')

    def bullets(self,items):
        self.md.extend('- '+t for t in items);self.md.append('')
        self.html.append('<ul>'+''.join('<li>'+escape(t)+'</li>' for t in items)+'</ul>')

    def table(self,headers,rows):
        def cell(t):return str(t).replace('|','\\|').replace('\n',' / ')
        self.md.append('| '+' | '.join(map(cell,headers))+' |')
        self.md.append('| '+' | '.join('---' for _ in headers)+' |')
        self.md.extend('| '+' | '.join(map(cell,row))+' |' for row in rows);self.md.append('')
        self.html.append('<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+escape(str(t))+'</th>' for t in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(t))+'</td>' for t in row)+'</tr>' for row in rows)+'</tbody></table></div>')

    def image(self,path,caption):
        self.md.append(f'![{caption}](figures/{path.name})\n\n{caption}\n')
        uri='data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode('ascii')
        self.html.append(f'<figure><img src="{uri}" alt="{escape(caption,quote=True)}"><figcaption>{escape(caption)}</figcaption></figure>')

    def links(self,items):
        self.md.extend(f'- [{label}]({target})' for label,target in items);self.md.append('')
        self.html.append('<ul>'+''.join(f'<li><a href="{escape(target,quote=True)}">{escape(label)}</a></li>' for label,target in items)+'</ul>')

    def write(self,output):
        (output/'REPORT_zh.md').write_text('\n'.join(self.md),encoding='utf-8',newline='\n')
        css='''body{font-family:"Microsoft YaHei","Noto Sans CJK SC",Arial,sans-serif;color:#172333;line-height:1.8;max-width:1160px;margin:38px auto;padding:0 28px;background:#fff}h1{font-size:32px;color:#203864;border-bottom:3px solid #356ca5;padding-bottom:14px}h2{font-size:25px;color:#203864;margin-top:42px}h3{font-size:20px;color:#203864}p,li{font-size:16px}table{border-collapse:collapse;width:100%;font-size:14px;margin:16px 0}th,td{border:1px solid #d4dde6;padding:9px 12px;text-align:left;vertical-align:top}th{background:#e8f0f8;color:#203864}tbody tr:nth-child(even){background:#f8fafc}.table-wrap{overflow-x:auto}figure{margin:28px 0}img{display:block;max-width:100%;height:auto;margin:auto}figcaption{font-size:14px;color:#586879;margin-top:9px}a{color:#215c91}code{background:#f1f4f7;padding:2px 4px}@media print{body{max-width:none;margin:0;padding:0}h2{break-after:avoid}figure,table{break-inside:avoid}a{color:inherit}}'''
        html='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>'+escape(self.title)+'</title><style>'+css+'</style></head><body>'+''.join(self.html)+'</body></html>'
        (output/'REPORT_zh.html').write_text(html,encoding='utf-8',newline='\n')


def event_breakdown(campaign,row):
    folder=campaign/row['artifact_directory']
    events=json.loads((folder/'events.json').read_text(encoding='utf-8'))
    tensors={t['id']:t for t in json.loads((folder/'tensor-catalogue.json').read_text(encoding='utf-8'))}
    ops={o['id']:o for o in json.loads((folder/'operation-catalogue.json').read_text(encoding='utf-8'))}
    copies=defaultdict(lambda:dict(count=0,bytes=0,span_ns=0))
    compute=Counter()
    for event in events:
        if event['kind']=='COPY':
            kind='kv' if event['target_id'].startswith('kv:') else tensors[event['target_id']]['role']
            slot=copies[kind];slot['count']+=1;slot['bytes']+=event['bytes_transferred']
            slot['span_ns']+=event['end_ns']-event['start_ns']
        elif event['kind']=='COMPUTE':
            compute[ops[event['target_id']]['kind']]+=event['end_ns']-event['start_ns']
    return dict(copies=dict(copies),compute_by_kind_ns=dict(compute)),events,tensors


def timeline_figure(campaign,rows,output,stem,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    colors={'CPU compute':'#356ca5','GPU compute':'#2a8064','KV copy':'#cf8a31',
            'Weight copy':'#8d64ac','Other copy':'#8493a2'}
    fig,axes=plt.subplots(len(rows),1,figsize=(12,2.8*len(rows)),layout='constrained',squeeze=False)
    limit=max(r['completed_makespan_ns'] for r in rows)/1000
    for ax,row in zip(axes[:,0],rows):
        _,events,tensors=event_breakdown(campaign,row)
        spans=defaultdict(list)
        for e in events:
            if e['kind']=='COMPUTE':
                label='CPU compute' if e['device']=='cpu' else 'GPU compute'
            elif e['kind']=='COPY':
                label=('KV copy' if e['target_id'].startswith('kv:') else
                       'Weight copy' if tensors[e['target_id']]['role']=='weight' else 'Other copy')
            else:continue
            spans[label].append((e['start_ns']/1000,(e['end_ns']-e['start_ns'])/1000))
        for label,intervals in spans.items():
            lane=2 if label=='CPU compute' else 1 if label=='GPU compute' else 0
            ax.broken_barh(intervals,(lane-.3,.6),facecolors=colors[label],edgecolors='none')
        ax.set_yticks([0,1,2],['COPY','GPU','CPU']);ax.set_ylim(-.6,2.7);ax.set_xlim(0,limit*1.025)
        ax.axvline(row['completed_makespan_ns']/1000,color='#77292d',linestyle='--',linewidth=1)
        ax.set_title(f"{row['scenario_id']} / {row['candidate_id']}: {row['completed_makespan_ns']/1000:.3f} us",loc='left',fontsize=11)
        ax.grid(axis='x',alpha=.15);ax.set_xlabel('Simulated elapsed time (microseconds)')
    fig.suptitle(title+'\nUncalibrated analytical event spans, not hardware profiling',fontsize=13)
    fig.legend(handles=[Patch(facecolor=color,label=label) for label,color in colors.items()],loc='outside lower center',ncol=5,frameon=False,fontsize=9)
    for extension in ('png','svg'):
        fig.savefig(output/(stem+'.'+extension),dpi=170,facecolor='white')
    plt.close(fig)


def _aggregate_pairs(pairs):
    complete=[p for p in pairs if p['complete_pair']]
    return dict(total=len(pairs),complete=len(complete),
        faster=sum(p['latency_saving_ns']>0 for p in complete),
        tied=sum(p['latency_saving_ns']==0 for p in complete),
        slower=sum(p['latency_saving_ns']<0 for p in complete),
        best_saving=max((p['latency_saving_fraction'] for p in complete),default=None),
        worst_saving=min((p['latency_saving_fraction'] for p in complete),default=None))


def build(campaign,output,english_doc=None,summary_json=None,audit=None):
    campaign,output=Path(campaign).resolve(),Path(output).resolve()
    completion=json.loads((campaign/'completion.json').read_text(encoding='utf-8'))
    protocol=json.loads((campaign/'protocol.json').read_text(encoding='utf-8'))
    records=json.loads((campaign/'records.json').read_text(encoding='utf-8'))
    analysis=json.loads((campaign/'analysis.json').read_text(encoding='utf-8'))
    if not completion['complete'] or not completion['full_matrix'] or not completion['source_unchanged']:
        raise ValueError('a completed source-frozen full matrix is required')
    if sha(campaign/'records.json')!=completion['records_sha256'] or sha(campaign/'analysis.json')!=completion['analysis_sha256']:
        raise ValueError('retained results do not match completion hashes')
    if canonical(protocol)!=completion['protocol_sha256']:
        raise ValueError('protocol hash differs')
    audit_receipt=None
    if audit is not None:
        audit=Path(audit).resolve()
        audit_receipt=json.loads(audit.read_text(encoding='utf-8'))
        if not audit_receipt['passed'] or audit_receipt['attempts_checked']!=len(records):
            raise ValueError('independent audit did not pass this complete matrix')
        if audit_receipt['completed_campaign_sha256']!=sha(campaign/'completion.json'):
            raise ValueError('independent audit belongs to another campaign')
        if audit_receipt['records_sha256']!=sha(campaign/'records.json') or audit_receipt['analysis_sha256']!=sha(campaign/'analysis.json'):
            raise ValueError('independent audit results differ')
        if audit_receipt['independent_audit_script_sha256']!=sha(ROOT/'experiments/strategy_comparison_audit.py'):
            raise ValueError('independent audit source differs from its receipt')
    for entry in protocol['frozen_accuracy_evidence_before']['files']:
        path=ROOT/entry['path']
        actual=sha(path) if path.exists() else None
        if actual!=entry['sha256']:
            raise ValueError('existing accuracy evidence changed before reporting')
    output.mkdir(parents=True,exist_ok=True)
    figures=output/'figures';figures.mkdir(exist_ok=True)
    plots(records,analysis,figures)
    data=output/'data';data.mkdir(exist_ok=True)
    for name in ('runs.csv','comparisons.csv','prefetch_pairs.csv','cache_pairs.csv',
                 'execution_rank_reversals.csv','interactions.csv','one_shot_selections.csv',
                 'analysis.json','records.json','protocol.json','completion.json','equivalence-controls.json'):
        shutil.copyfile(campaign/name,data/name)
    if audit_receipt:
        shutil.copyfile(audit,data/'independent-audit.json')
        shutil.copyfile(ROOT/'experiments/strategy_comparison_audit.py',data/'independent-audit-script.py')
    controls=protocol['catalogue']['equivalence_controls']
    unique=lambda rows:[r for r in rows if r['scenario_id'] not in controls]
    distinct_analysis=analyze(unique(records))
    # Preserve all sixteen heatmaps, then replace aggregate plots with the
    # fourteen distinct inputs so equivalence controls do not inflate them.
    plots(unique(records),distinct_analysis,figures)
    (data/'analysis_distinct_inputs.json').write_text(json.dumps(distinct_analysis,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    scenarios=analysis['scenarios'];main=unique(scenarios)
    prefetch=_aggregate_pairs(unique(analysis['prefetch_pairs']))
    cache=_aggregate_pairs(unique(analysis['cache_pairs']))
    reversals=[r for r in unique(analysis['execution_rank_reversals']) if r['strict_reversal']]
    joint=[r for r in unique(analysis['interactions']) if r['complete_four_cells']]
    selections=unique(analysis['one_shot_selections'])
    regret=[s for s in selections if s['finite_catalogue_regret_fraction'] is not None]
    row_index={(r['scenario_id'],r['candidate_id']):r for r in records}
    def row(s,c):return row_index[(s,c)]
    adaptive_checks=[]
    for execution in ('e4','e5'):
        pairs=[(r,row(r['scenario_id'],'e2-'+r['candidate_id'].split('-')[1]))
               for r in unique(records) if r['model_kind']=='moe' and r['candidate_id'].startswith(execution+'-')]
        complete=[(a,b) for a,b in pairs if a['success'] and b['success']]
        adaptive_checks.append(dict(execution=execution,complete_pairs=len(complete),
            same_latency=sum(a['completed_makespan_ns']==b['completed_makespan_ns'] for a,b in complete),
            same_event_trace=sum(a['event_sha256']==b['event_sha256'] for a,b in complete)))
    long_gpu=row('dense-long_prefill','e0-s0');long_cpuatt=row('dense-long_prefill','e3-s0')
    b_gpu,_,_=event_breakdown(campaign,long_gpu)
    b_cpuatt,_,_=event_breakdown(campaign,long_cpuatt)
    timeline_figure(campaign,[long_gpu,long_cpuatt],figures,'timeline_kv_contract',
        'Long-Prefill example: transfer granularity changes the preferred placement')
    pf_before=row('dense-short_prefill','e0-s0');pf_after=row('dense-short_prefill','e0-s1')
    timeline_figure(campaign,[pf_before,pf_after],figures,'timeline_prefetch_tradeoff',
        'Same LRU budget: prefetch can increase both overlap and total work')
    summaries=dict(schema='strategy-comparison-readable-summary-0.1',scope='uncalibrated analytical simulation',
        campaign=str(campaign),completion=completion,distinct_scenarios=14,
        independent_audit=audit_receipt,
        equivalence_controls=controls,main_prefetch_pairs=prefetch,main_cache_pairs=cache,
        main_rank_reversal_count=len(reversals),
        main_scenarios_with_rank_reversal=len({r['scenario_id'] for r in reversals}),
        main_joint_cells=dict(complete=len(joint),positive=sum(r['joint_gain_over_better_single']>0 for r in joint),
            zero=sum(r['joint_gain_over_better_single']==0 for r in joint),
            negative=sum(r['joint_gain_over_better_single']<0 for r in joint)),
        one_shot=dict(complete=len(regret),positive_regret=sum(r['finite_catalogue_regret_fraction']>0 for r in regret),
            maximum_regret=max((r['finite_catalogue_regret_fraction'] for r in regret),default=None)),
        scenarios=scenarios,
        adaptive_rule_comparisons_to_e2=adaptive_checks,
        long_prefill_mechanism=dict(gpu=b_gpu,cpu_attention=b_cpuatt),
        report_script_sha256=sha(__file__))

    doc=Document('Dense／MoE 执行策略与存储调度组合实验报告')
    doc.p('日期：2026 年 10 月 23 日。阅读目标：弄清楚哪些候选策略在什么条件下更快，哪些代价导致了差异，以及本轮结果能支持什么结论。')
    doc.p('本轮使用固定假想硬件和显式解析成本，运行的是两层小型模型的时间／资源模拟。没有重新采集、拟合性能参数，也没有执行真实语言生成。以下“更快”均指本轮声明条件下的模拟结果，不是实机加速结论。')
    doc.h(2,'1. 先看结果概况')
    total=analysis['totals']
    doc.table(['项目','本轮完成情况'],[
        ['组合运行',f"{total['present_count']} / {total['expected_count']} 次已运行，{total['complete_count']} 次完成，{total['failure_count']} 次失败，{total['missing_count']} 次缺失"],
        ['场景','4 个 Dense＋12 个 MoE 声明场景，其中 14 种不同输入行为、2 个等价性对照'],
        ['主分析与对照',f"主分析包含 {distinct_analysis['totals']['present_count']} 次运行；另有 {total['present_count']-distinct_analysis['totals']['present_count']} 次等价条件对照运行"],
        ['每场景候选','Dense：4 类执行×4 类存储＝16；MoE：6 类执行×4 类存储＝24'],
        ['等价性检查',f"{completion['equivalence_control_pairs']} / {completion['equivalence_control_pairs']} 对通过，比较了真实事件与主要指标"],
        ['原准确性依据',f"{completion['accuracy_file_count']} 份文件哈希未变，执行源码前后未变"],
        ['时间口径','完整工作量完成时间；失败的局部耗时不参加排名'],
    ])
    doc.p(f"在排除两组重复条件对照后的 14 种输入中，打开预取的 {prefetch['complete']} 组可比较配对，有 {prefetch['faster']} 组变快、{prefetch['tied']} 组不变、{prefetch['slower']} 组变慢。这里没有预设‘预取必然有益’。")
    doc.p(f"其中最大耗时缩短为 {100*max(0,prefetch['best_saving']):.2f}%，最大耗时增加为 {100*max(0,-prefetch['worst_saving']):.2f}%。它们是不同候选配对的条件结果，不是重复实验的误差范围。")
    doc.p(f"改变存储规则后，{len({r['scenario_id'] for r in reversals})} 个不同输入场景中出现过执行策略的严格排名反转。不同配对共享同一批运行，反转次数不是独立统计样本；很小的差距也不是稳健性证明。")
    doc.p(f"把执行、存储分别选好后再直接组合，{len(regret)} 个可比较输入中有 {sum(s['finite_catalogue_regret_fraction']>0 for s in regret)} 个比本候选集最佳点更慢。最大相对差距为 {pct(max((s['finite_catalogue_regret_fraction'] for s in regret),default=None))}。这项检查针对一种明确的独立选择方法，并不证明所有分解方法都会失败。")

    doc.h(2,'2. 到底比较了什么')
    doc.table(['工作负载','Prefill Token','Decode 步数','直观含义'],[
        ['短 Prefill',8,0,'短输入，不继续生成'],['长 Prefill',128,0,'长输入，不继续生成'],
        ['Decode 为主',8,8,'短输入，继续做 8 步 Decode'],['混合负载',128,8,'长输入，再做 8 步 Decode']])
    doc.table(['MoE 路由','具体规则','它会改变什么'],[
        ['固定热点','每个 Token 都选择专家 0、1','专家集合稳定，可能有重复利用'],
        ['轮换专家','在 4 个专家中轮换相邻的两位','专家复用与每位专家的 Token 聚集都变化'],
        ['P/D 热点切换','Prefill 用 0、1，Decode 用 2、3','此前热门专家可能在阶段切换后失去价值']])
    doc.p('两组没有 Decode 的热点切换场景实际上没有发生切换，因此其路由与固定热点完全一致。本轮保留它们检查输入相同是否得到相同执行，不把它们当作额外的路由发现。')
    doc.table(['执行代号','意思'],[[e,t] for e,t in EXEC_ZH.items()])
    doc.p('e4、e5 仅适用于 MoE，属于根据已揭示路由和当前状态做出的局部规则，不是全局最优调度。e0、e1 指 LLM 宏任务的设备，轨迹输入／状态更新门仍可能按格式契约在 CPU 执行。')
    doc.table(['存储代号','规则'],[[s,t] for s,t in STORE_ZH.items()])
    doc.p('缓存规则回答“空间不够时先释放哪份权重”：LRU 优先释放较久没用的，历史频率规则优先保留历史使用更多的。预取回答“真正需要前要不要先搬”：本轮提前准备下一层的候选权重，它也会占用同一份预算、消耗搬运链路。')
    doc.p('四种存储规则的缓存预算都是 192 KiB，物理显存都是 512 KiB。预取深度是下一层，每个源层调用最多预取两个单元。比较 s0→s1 或 s2→s3 时，只改变预取；比较 s0→s2 或 s1→s3 时，只改变缓存淘汰规则。')
    doc.p('硬件参数固定：CPU 100 GFLOP/s、20 GB/s；GPU 1 TFLOP/s、100 GB/s；DRAM 128 MiB，CPU/GPU 间有效带宽 8 GB/s，每次搬运另加 1,000 ns。模型为两层 H64、FFN 宽度 192、F32；MoE 有 4 位专家、top-2。这些是明确给定的解析参数，不是某台电脑的实测参数。')
    doc.p('解析成本的直观算法：计算耗时＝max(运算量÷算力，估计访存量÷内存带宽)＋每个宏任务固定启动开销；搬运耗时＝字节数÷链路带宽＋每次固定延迟，均向上取整到纳秒。CPU／GPU 的宏任务启动开销分别设为 500／2,000 ns。这里的启动是每次任务执行的成本，不是启动 Python 程序。小图的固定开销与数据搬运可能压过 GPU 的计算优势，因此这些参数必须和结果一起看。')
    doc.p('每个候选从同样的冷状态开始：权重位于 DRAM，KV 为空。必须交付的最终 hidden 和 KV 都在 DRAM；可选权重副本不强制统一清空。初始搬运、运行中计算和搬运都计时，进程启动、模型下载、编译和实验程序本身的运行耗时不计入模拟延迟。')
    doc.p('实验使用三个进程并行运行独立候选，只是缩短实验程序的等待时间。每个候选仍有独立的单 CPU＋单 GPU 模型，不模拟多 GPU；电脑上这些实验进程的实际竞争不会写进模拟耗时。')
    doc.h(3,'指标怎么理解')
    doc.table(['指标','本轮如何计算','它回答的问题'],[
        ['是否完成','相同输入与最终交付目标均完成，才能比较完整耗时','是否真的做完了同一件事'],
        ['完整耗时／相对加速','最后目标完成的模拟时间；基线耗时÷候选耗时','哪个候选更快'],
        ['H2D／D2H／SSD 流量','累加实际 COPY 事件的字节，分方向统计','为了计算多搬了多少数据'],
        ['DRAM／VRAM 峰值','分别取模拟物理占用的峰值，含声明的中间量与工作区','是否可放下、用了多少存储'],
        ['重复权重加载','相同不可变权重沿同一路径第二次及以后加载的次数、字节','是否反复支付相同搬运成本'],
        ['CPU／GPU／COPY 区间','各类实际事件时间段的并集','模型里各引擎有多少时间承担工作'],
        ['计算／搬运重叠','计算区间与 COPY 区间的交集','有多少搬运时间与计算同时发生']])
    doc.p('这些指标分别报告，不合成一个任意的综合分。事件占用不是实测硬件利用率，重叠也不等于加速；缓存命中率、因果等待分类、TTFT／TPOT、能耗本轮没有对应测量边界，保持不可用。')

    doc.h(2,'3. 每种场景中，哪个候选耗时最短')
    table_rows=[]
    for s in scenarios:
        label=scenario_name(s)+('〔等价对照〕' if s['scenario_id'] in controls else '')
        table_rows.append([label,f"{s['prefill_tokens']} / {s['decode_steps']}",
            ', '.join(s['best_candidates']) or '无可完成候选',ns_ms(s['best_makespan_ns']),
            '不可用' if s['best_reference_speedup'] is None else f"{s['best_reference_speedup']:.3f}×"])
    doc.table(['场景','Prefill / Decode','最短耗时候选，含全部并列','模拟耗时 ms','相对 e0-s0'],table_rows)
    doc.p('“相对 e0-s0”是基线耗时除以候选耗时。它比较的是这份预先声明的有限候选集，不是与某篇论文的完整系统比，也不是全策略空间最优。并列只表示耗时完全相同，显存占用和搬运量仍可能不同。')
    doc.image(figures/'summary_speedup.png','图 1：14 种不同输入下，候选集最佳点相对固定 GPU 基线。图中排除了两组重复条件对照；数字来自解析模拟。')
    doc.h(3,'速度之外，流量和内存付出了什么代价')
    representative_ids=('dense-long_prefill','dense-decode_heavy','moe-decode_heavy-rotating','moe-mixed-phase_shift')
    representative=[]
    summary_index={s['scenario_id']:s for s in scenarios}
    for scenario_id in representative_ids:
        summary=summary_index[scenario_id]
        for candidate in dict.fromkeys(('e0-s0',summary['best_candidate'])):
            if candidate is not None:
                representative.append(row(scenario_id,candidate))
    doc.p('下面选取长输入、重复 Decode 和专家切换四个代表场景，同时展示基线与耗时最短候选。若有并列，仅用编号最小者展示代价；完整候选表保留了所有并列。KiB＝1,024 字节，各峰值单独取最大值，不要求发生在同一时刻。')
    doc.table(['场景／候选','耗时 ms','H2D KiB','D2H KiB','峰值 DRAM KiB','峰值 VRAM KiB','重复权重次数／KiB'],[
        [scenario_name(r)+' / '+r['candidate_id'],ns_ms(r['completed_makespan_ns']),
         kib(r['h2d_bytes']),kib(r['d2h_bytes']),kib(r['peak_dram_bytes']),kib(r['peak_vram_bytes']),
         str(r['immutable_weight_reload_count'])+' / '+kib(r['immutable_weight_reload_bytes'])] for r in representative])
    doc.table(['场景／候选','CPU 区间 μs','GPU 区间 μs','COPY 区间 μs','计算／COPY 重叠 μs','CPU／GPU 重叠 μs'],[
        [scenario_name(r)+' / '+r['candidate_id'],ns_us(r['cpu_event_span_ns']),ns_us(r['gpu_event_span_ns']),
         ns_us(r['copy_event_span_ns']),ns_us(r['copy_compute_event_overlap_ns']),ns_us(r['cpu_gpu_event_overlap_ns'])] for r in representative])
    no_compute_overlap=all(r['cpu_gpu_event_overlap_ns']==0 for r in records if r['success'])
    no_ssd=all(r['ssd_read_bytes']==r['ssd_write_bytes']==0 for r in records if r['success'])
    if no_compute_overlap:
        doc.p('本矩阵所有候选的 CPU／GPU 计算重叠均为零：它检验的是任务放在哪里，以及计算与搬运如何配合，还没有检验 CPU／GPU 同时分担张量计算的收益。')
    if no_ssd:
        doc.p('本矩阵 SSD 读写字节均为零：权重从 DRAM 开始，不使用 SSD。本轮没有提供 SSD 卸载的性能对比证据。')

    doc.h(2,'4. 为什么“重叠更多”有时反而更慢')
    delta_copy=pf_after['copy_event_span_ns']-pf_before['copy_event_span_ns']
    delta_overlap=pf_after['copy_compute_event_overlap_ns']-pf_before['copy_compute_event_overlap_ns']
    delta_compute=pf_after['compute_event_span_ns']-pf_before['compute_event_span_ns']
    delta_total=pf_after['completed_makespan_ns']-pf_before['completed_makespan_ns']
    doc.p('看一个最容易核对的例子：Dense 短 Prefill，执行方式保持 e0，只打开下一层预取，缓存预算不变。')
    doc.table(['指标，微秒','不预取 e0-s0','预取 e0-s1','变化'],[
        ['完整耗时',ns_us(pf_before['completed_makespan_ns']),ns_us(pf_after['completed_makespan_ns']),ns_us(delta_total)],
        ['计算区间并集',ns_us(pf_before['compute_event_span_ns']),ns_us(pf_after['compute_event_span_ns']),ns_us(delta_compute)],
        ['搬运区间并集',ns_us(pf_before['copy_event_span_ns']),ns_us(pf_after['copy_event_span_ns']),ns_us(delta_copy)],
        ['计算／搬运重叠',ns_us(pf_before['copy_compute_event_overlap_ns']),ns_us(pf_after['copy_compute_event_overlap_ns']),ns_us(delta_overlap)],
        ['H2D 字节',pf_before['h2d_bytes'],pf_after['h2d_bytes'],pf_after['h2d_bytes']-pf_before['h2d_bytes']],
        ['重复权重搬运字节',pf_before['immutable_weight_reload_bytes'],pf_after['immutable_weight_reload_bytes'],pf_after['immutable_weight_reload_bytes']-pf_before['immutable_weight_reload_bytes']]])
    doc.p(f"本例计算区间变化为 {ns_us(delta_compute)} 微秒。预取多产生 {ns_us(delta_copy)} 微秒搬运覆盖，只多隐藏了 {ns_us(delta_overlap)} 微秒；最后总时间变化 {ns_us(delta_total)} 微秒。重叠长度与额外工作量需要一起看，不能只看 GPU 或链路更忙了。")
    doc.image(figures/'timeline_prefetch_tradeoff.png','图 2：同一执行方式、同一缓存预算下的真实模拟事件。空白和重叠按事件区间绘制，不是硬件利用率采样。')
    cpu_off=row('dense-short_prefill','e1-s0');cpu_on=row('dense-short_prefill','e1-s1')
    doc.p(f"另一个更直观的情况是全 CPU 的 Dense 短 Prefill：两种存储规则的耗时都为 {ns_us(cpu_off['completed_makespan_ns'])} 微秒，但打开预取后多搬了 {cpu_on['h2d_bytes']-cpu_off['h2d_bytes']:,} 字节到 GPU，且没有 GPU 宏任务消费者。这说明 IR 可以表达这个组合，并不意味着这个组合值得采用。")
    doc.p('报告中的“没有 GPU 消费者的权重搬运”只是一种可直接确认的无效流量下界，不等于所有浪费，也不是缓存命中率。')
    doc.image(figures/'summary_prefetch_effect.png','图 3：同预算、同执行方式下打开预取的延迟变化。正值表示变快，负值表示变慢；每个点是一组候选配对，不是重复采样。')

    doc.h(2,'5. 长 Prefill 的排名，为什么必须连同模型假设一起看')
    kv=b_gpu['copies'].get('kv',dict(count=0,bytes=0,span_ns=0))
    kv_latency=protocol['catalogue']['scenarios'][0]['hardware']['transfer_latency_ns']*kv['count']
    doc.p('Dense 长 Prefill 中，把 Attention 放到 CPU 的候选比全部 GPU 更快。仅看结论容易误会：这并不是 CPU Attention 计算本身更快，而是当前数据交付方式产生了大量 KV 搬运。')
    doc.table(['指标，微秒，除特别标注','全部 GPU e0-s0','CPU Attention e3-s0'],[
        ['完整耗时',ns_us(long_gpu['completed_makespan_ns']),ns_us(long_cpuatt['completed_makespan_ns'])],
        ['Attention 计算时间',ns_us(b_gpu['compute_by_kind_ns'].get('attention',0)),ns_us(b_cpuatt['compute_by_kind_ns'].get('attention',0))],
        ['COPY 区间并集',ns_us(long_gpu['copy_event_span_ns']),ns_us(long_cpuatt['copy_event_span_ns'])],
        ['计算／COPY 重叠',ns_us(long_gpu['copy_compute_event_overlap_ns']),ns_us(long_cpuatt['copy_compute_event_overlap_ns'])],
        ['KV COPY 次数',kv['count'],b_cpuatt['copies'].get('kv',{}).get('count',0)],
        ['KV COPY 字节',kv['bytes'],b_cpuatt['copies'].get('kv',{}).get('bytes',0)]])
    doc.p(f"当前契约要求最终 KV 位于 DRAM。GPU Attention 生成的 KV 按 Token 小块分别搬回：本例 {kv['count']} 次、合计 {kv['bytes']:,} 字节，共用 {ns_us(kv['span_ns'])} 微秒，其中 {ns_us(kv_latency)} 微秒来自每次固定延迟，实际载荷部分只有 {ns_us(kv['span_ns']-kv_latency)} 微秒。CPU Attention 直接在 DRAM 生成 KV，因此避开了这些 COPY。")
    doc.p('因此，本例首先暴露的是 KV 搬运粒度、固定传输延迟与最终放置约束的联合作用。真实系统可能批量搬运连续 KV，不能直接把这份排名推广到真实 CPU/GPU。我们保留了原模型和冻结准确性实验，未为获得好结果调整固定延迟或重新拟合。')
    doc.image(figures/'timeline_kv_contract.png','图 4：长 Prefill 的计算与搬运时间轴。橙色是按显式 KV 张量身份识别的搬运，通用 role 字段会把 KV 与其他 output 合并，因此本图单独核对 KV ID。')
    doc.image(figures/'latency_dense-long_prefill.png','图 5：Dense 长 Prefill 的完整 4×4 候选矩阵。相同耗时的格子可能有不同搬运和内存代价。')

    doc.h(2,'6. 执行与存储是否值得联合考虑')
    doc.table(['配对检查，仅 14 种不同输入','可比较数量','变快／正收益','不变／零收益','变慢／负收益'],[
        ['打开预取',prefetch['complete'],prefetch['faster'],prefetch['tied'],prefetch['slower']],
        ['LRU 改为历史频率',cache['complete'],cache['faster'],cache['tied'],cache['slower']],
        ['联合改变，相对更好的单项改变',len(joint),sum(r['joint_gain_over_better_single']>0 for r in joint),sum(r['joint_gain_over_better_single']==0 for r in joint),sum(r['joint_gain_over_better_single']<0 for r in joint)]])
    if cache['faster']==0:
        doc.p(f"本轮没有观察到历史频率规则比 LRU 更快；{cache['tied']} 组相同、{cache['slower']} 组稍慢，最大增加 {100*max(0,-cache['worst_saving']):.2f}%。这份场景集尚未支持替换缓存规则的收益主张。")
    doc.p('联合收益的定义是：先分别只改执行、只改存储，再把两种改变组合起来，比较组合与两个单项中较快者的差距。正数说明该四格实验中组合更快，零或负数照实保留。不同四格可能重复使用候选，不能把计数解释成独立实验的成功概率。')
    if joint:
        witness=max(joint,key=lambda r:abs(r['nonadditive_time_saving_ns']))
        doc.h(3,'用四个格子区分“更快”和“相互影响”')
        doc.p(f"取非可加性绝对值最大的四格作解释：{scenario_name(witness)}，执行改变为 {witness['execution']}，存储改变为 {witness['storage']}。这是解释性例子，全部正负值均保留在结果表中。")
        doc.table(['格子','改变','模拟耗时 μs'],[
            [label,c['candidate_id'],ns_us(c['completed_makespan_ns'])]
            for label,c in zip(('T00：基线','T10：只改执行','T01：只改存储','T11：两者都改'),witness['cells'])])
        doc.p(f"联合收益＝[min(T10,T01)−T11]÷min(T10,T01)＝{pct(witness['joint_gain_over_better_single'])}；它问组合有没有比更好的单项还快。非可加节省＝T10＋T01−T00−T11＝{ns_us(witness['nonadditive_time_saving_ns'])} 微秒；它问两种改变是否相互加强或抵消。正值表示在这个时间差口径上相互加强，负值表示抵消，不能仅凭非可加性就认定组合是最优。")
    if reversals:
        chosen=max(reversals,key=lambda r:min(r['relative_gap_at_storage_a'],r['relative_gap_at_storage_b']))
        doc.h(3,'一个实际发生的执行排名反转')
        doc.p(f"场景：{scenario_name(chosen)}。比较 {chosen['execution_a']} 与 {chosen['execution_b']}，存储规则从 {chosen['storage_a']} 换到 {chosen['storage_b']}。")
        times=chosen['times_ns']
        doc.table(['存储规则',chosen['execution_a']+'，模拟 ms',chosen['execution_b']+'，模拟 ms','绝对差距，微秒'],[
            [chosen['storage_a'],ns_ms(times[0]),ns_ms(times[1]),ns_us(abs(times[0]-times[1]))],
            [chosen['storage_b'],ns_ms(times[2]),ns_ms(times[3]),ns_us(abs(times[2]-times[3]))]])
        doc.p(f"两侧相对较快者的差距分别为 {100*chosen['relative_gap_at_storage_a']:.2f}% 和 {100*chosen['relative_gap_at_storage_b']:.2f}%。这证明本候选集里存在条件依赖；它没有给出真实硬件的误差界或统计显著性。全部反转格子和差距保留在 CSV。")
    worst=max(regret,key=lambda r:r['finite_catalogue_regret_fraction']) if regret else None
    if worst:
        doc.h(3,'独立选择再组合的对照结果')
        doc.p('本轮另外固定了一种简单做法：在 s0 下挑执行策略，在 e0 下挑存储策略，然后直接组合。它只是一个明确的对照规则，使用同一候选表，离线做选择，不是新的在线搜索器。')
        doc.p('下表展示额外耗时比例最大的场景；若全部为零，则展示其中一个零损失例子。')
        doc.table(['场景','独立选择再组合','组合耗时 ms','候选集最佳','最佳耗时 ms','额外耗时比例'],[
            [scenario_name(worst),worst['composed_candidate'],ns_ms(worst['completed_makespan_ns']),
             ', '.join(worst['offline_catalogue_best_candidates']),ns_ms(worst['offline_catalogue_best_makespan_ns']),pct(worst['finite_catalogue_regret_fraction'])]])
        if worst['finite_catalogue_regret_fraction']==0:
            doc.p('本轮这项独立选择检查没有产生额外耗时，不能宣称它已经被本实验击败。局部排名反转和全局最佳选择受损是两种不同证据。')
        else:
            doc.p('该结果支持对这类独立决策保持警惕，但不能据此宣布联合搜索算法获胜；本轮仍是完整枚举有限表格。')

    doc.h(2,'7. MoE 路由结果应该怎样读')
    moe=[s for s in main if s['model_kind']=='moe']
    doc.table(['负载','路由','e0-s0 模拟 ms','e2-s0 模拟 ms','最短耗时候选','最佳模拟 ms'],[
        [next(SHAPE_ZH[k] for k in SHAPE_ZH if f'-{k}' in s['scenario_id']),ROUTE_ZH[s['route_pattern']],
         ns_ms(row(s['scenario_id'],'e0-s0')['completed_makespan_ns']),
         ns_ms(row(s['scenario_id'],'e2-s0')['completed_makespan_ns']),
         ', '.join(s['best_candidates']),ns_ms(s['best_makespan_ns'])] for s in moe])
    doc.p('固定热点、轮换和阶段切换不仅影响缓存，还会改变每个专家聚集到的 Token 数、专家任务数和启动次数。因此，两个路由场景的差异不能全部归因于“缓存命中率”。同一路由内部的候选比较使用完全相同的 trace；e4／e5 只读取已揭示任务与当前状态。')
    hot_decode=row('moe-decode_heavy-hot','e0-s0')
    shift_decode=row('moe-decode_heavy-phase_shift','e0-s0')
    doc.h(3,'重复加载减少，也不一定代表搬运减少')
    doc.table(['短输入＋8 步 Decode，e0-s0','固定热点','P/D 热点切换'],[
        ['完整耗时 μs',ns_us(hot_decode['completed_makespan_ns']),ns_us(shift_decode['completed_makespan_ns'])],
        ['H2D KiB',kib(hot_decode['h2d_bytes']),kib(shift_decode['h2d_bytes'])],
        ['重复权重加载 KiB',kib(hot_decode['immutable_weight_reload_bytes']),kib(shift_decode['immutable_weight_reload_bytes'])]])
    doc.p('“重复加载”以同一个权重张量为单位计数。热点切换后，新专家的首次加载不会计入重复加载；因此，即使这个指标减少，总搬运量和耗时也可能不变。判断存储策略不能只看重复加载一个数，更不能把它当成命中率。')
    doc.image(figures/'latency_moe-decode_heavy-rotating.png','图 6：MoE 短输入＋多步 Decode、轮换专家的 6×4 全矩阵，包含固定放置和两种局部自适应专家规则。')
    doc.image(figures/'latency_moe-mixed-phase_shift.png','图 7：MoE 长输入＋多步 Decode、P/D 热点切换的 6×4 全矩阵。横向比较存储规则，纵向比较执行规则。')
    doc.table(['局部专家规则，与 e2 比较','可比较配对，仅不同输入','耗时完全相同','事件轨迹完全相同'],[
        [a['execution']+'：'+EXEC_ZH[a['execution']],a['complete_pairs'],a['same_latency'],a['same_event_trace']]
        for a in adaptive_checks])
    doc.p('这项检查区分“规则可以写出来”和“规则在本轮产生了不同执行”。如果事件也相同，说明该组输入下局部规则落到了同一条动作路径，不能宣称它带来了自适应收益，也不能推断这些规则在所有输入上等价。')
    expert_row=row('moe-long_prefill-hot','e5-s0')
    expert_audit=json.loads((campaign/expert_row['artifact_directory']/'policy-audit.json').read_text(encoding='utf-8'))
    if expert_audit.get('region_bindings'):
        binding=expert_audit['region_bindings'][0]
        paths=binding.get('paths',[])
        doc.h(3,'为什么“按当前成本选设备”也可能没有选到更快的 GPU')
        doc.p('查看 MoE 长 Prefill、固定热点、e5-s0 的第一位专家。审计记录把设备路径成本和当前缓存预算可行性分开给出：')
        doc.table(['路径','独立完整路径估计，微秒','GPU 路径拟占预算，字节','缓存预算可行','物理容量可行'],[
            [p['device'].upper(),ns_us(p['total_ns']),
             p['weight_budget_footprint_bytes'] if p['device']=='gpu' else '不适用',
             str(p['budget_feasible']) if p['device']=='gpu' else '不适用',str(p['physical_feasible'])] for p in paths])
        doc.p('GPU 路径拟占预算是现有驻留权重加上准备加入的专家权重，不是此刻已经占用的空间；CPU 路径不使用这项 GPU 缓存预算。')
        doc.p(f"缓存预算为 196,608 字节。e5 先过滤当前预算和容量不可行的路径，再从剩下的路径比较局部成本；本例实际提交到 {binding['device'].upper()}。它没有在同一个选择中联合探索‘先驱逐旧权重，再把专家放到 GPU’。因此，更智能的名字并不等于更强的联合决策能力。")
        fixed=row('moe-long_prefill-hot','e0-s0')
        fixed_folder=campaign/fixed['artifact_directory']
        fixed_events=json.loads((fixed_folder/'events.json').read_text(encoding='utf-8'))
        released=next((e for e in fixed_events if e['kind']=='EVICT' and e['target_id']=='w:layer0:W_proj'),None)
        if released:
            doc.p(f"作为同条件对照，固定 GPU 候选 e0-s0 实际先释放了投影权重 {released['target_id']}，释放 {released['bytes_released']:,} 字节，然后合法地在 GPU 执行该专家。其完整模拟耗时 {ns_us(fixed['completed_makespan_ns'])} 微秒，而 e5-s0 为 {ns_us(expert_row['completed_makespan_ns'])} 微秒。这个差异可从实际驱逐、搬运、计算动作逐步核对。")
        doc.p('这个例子的 GPU 预估占用只比预算多 1 KiB，属于预算临界条件下的见证。它说明执行选择与存储动作之间的接缝需要检查，不说明局部自适应规则普遍更差。下一轮可以预先声明容量扫描，观察阈值两侧的决策；本轮没有修改预算来追求更好数值，也没有提出新的全局优化结果。')

    doc.h(2,'8. 这轮能得出什么，不能得出什么')
    doc.bullets([
        '已完成：同一 IR、编译器和 modeling 下，对所有声明候选按相同口径运行；能够追溯延迟、流量、内存、重复加载和事件重叠。',
        '已观察：策略的效果依赖工作负载、数据放置和另一类策略的选择。预取、更多重叠、局部自适应均不是无条件改进。',
        '重要模型边界：H64 是小图，固定传输开销和 KV 小块搬运很显著；零工作区也是理想化声明。不能把小图赢家当作真实大模型赢家。',
        '算法执行边界：Python 调度决策自身的运行开销未计入模拟预测，实验程序的墙钟耗时另行保留。',
        '尚未证明：真实模型和设备的加速、准确率进一步提升、比已有模拟器更准确、覆盖全部策略空间、或新搜索算法优于现有方法。',
        '尚未纳入本矩阵：Attention／FFN 内部拆分、多请求流水、P/D 执行放置、传输打包、CPU Attention＋自适应专家等模板内部组合，以及设备波动／统一内存。这里 P/D 热点切换只是路由输入变化，不是 P/D 放置策略。多 GPU 不在当前 DAC 范围内。',
        '下一步更有价值：挑出差异清楚的组合，加入真实尺寸与匹配的成本边界，再检查粒度假设变化后结论是否保留；不要为了正收益调参数，也不必重新打开整条校准主线。'])

    failed=[r for r in records if not r['success']]
    doc.h(2,'9. 完整性、失败与复现')
    if failed:
        doc.table(['场景','候选','状态／终止原因','错误'],[[r['scenario_id'],r['candidate_id'],str(r.get('status'))+' / '+str(r.get('termination_reason')),r.get('error_message') or r.get('compilation_error') or r.get('error_code')] for r in failed])
    else:
        doc.p('本轮 352 个声明候选均完成，没有把失败时间当作更快结果，也没有删除负收益配置。此结果只表明该矩阵在声明资源与成本下可执行，不代表性能模型已通过新的实测验收。')
    if audit_receipt:
        doc.p(f"独立审计另用标准库程序重放了 {audit_receipt['attempts_checked']} 组事件，不调用 simulator、modeling、mapping 或指标分析代码。它核对声明成本、计算依赖、张量搬运就绪、释放锁定、内存峰值、最终交付和汇总指标，均通过。该检查证明产物内部一致，不等同于真实硬件准确性验收。")
        doc.links([('独立事件重放与指标审计凭据','data/independent-audit.json'),
                   ('独立审计程序源码','data/independent-audit-script.py')])
    doc.p(f"执行源码清单哈希：{completion['execution_sources_sha256']}。协议哈希：{completion['protocol_sha256']}。原始输入、精确成本、IR、事件、策略审计和逐文件清单保留在 {campaign.as_posix()}。模拟运行程序墙钟耗时为 {completion['harness_wall_seconds']:.1f} 秒，它不是推理耗时。")
    doc.p("在仓库根目录用 PowerShell 复现：先运行 $env:PYTHONPATH = 'src;modeling;mapping;profiling'，再运行 python experiments/strategy_comparison_campaign.py --out build/strategy-comparison/reproduction --workers 3。输出目录必须新建或为空。报告和标准图可由已有 Matplotlib 环境运行 experiments/strategy_comparison_report.py 生成。重复确定性模拟用于检查复现，不是额外统计样本。")
    doc.links([('完整 352 行结果 CSV','data/runs.csv'),('全部指标与所有配对 JSON','data/analysis.json'),
        ('预取配对','data/prefetch_pairs.csv'),('缓存规则配对','data/cache_pairs.csv'),
        ('排名反转与差距','data/execution_rank_reversals.csv'),('联合收益','data/interactions.csv'),
        ('独立选择再组合','data/one_shot_selections.csv'),('预先冻结的协议','data/protocol.json'),
        ('完整性凭据','data/completion.json')])
    doc.h(2,'附录：全部场景热图')
    doc.p('每张热图都包含该场景的全部 16 或 24 个候选。灰色失败／缺失格不会画成零延迟。本轮控制场景也照常保留。热图数字做显示取整，是否并列按原始整数纳秒判断，CSV 保留精确数值。')
    doc.links([(scenario_name(s)+('〔等价对照〕' if s['scenario_id'] in controls else ''),
                'figures/latency_'+s['scenario_id']+'.png') for s in scenarios])
    doc.write(output)
    (output/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    english=['# Expanded analytical strategy comparison results','',
        'All results are uncalibrated analytical simulation, not real-device speedups. No performance measurement, fit or modeling mechanism changed.','',
        f"Completed {total['complete_count']} of {total['expected_count']} declared runs, with {total['failure_count']} failures and {total['missing_count']} missing cells. Sixteen scenario IDs contain 14 distinct input behaviors plus 2 equivalence controls. All {completion['equivalence_control_pairs']} matching control comparisons agree.",'',
        '| Scenario | Best completed candidates, exact ties retained | Simulated ms | Reference speedup |',
        '|---|---|---:|---:|']
    for s in scenarios:
        speed='unavailable' if s['best_reference_speedup'] is None else f"{s['best_reference_speedup']:.6f}"
        english.append(f"| {s['scenario_id']} | {', '.join(s['best_candidates'])} | {ns_ms(s['best_makespan_ns'])} | {speed} |")
    english+=['','## Main comparisons excluding equivalence controls','',
        f"Prefetch: {prefetch['complete']} complete pairs, {prefetch['faster']} faster, {prefetch['tied']} tied and {prefetch['slower']} slower. Cache-family change: {cache['faster']} faster, {cache['tied']} tied and {cache['slower']} slower out of {cache['complete']} complete pairs.",
        f"Strict execution-order reversal appears in {len({r['scenario_id'] for r in reversals})} distinct input scenarios; {len(reversals)} overlapping four-cell comparisons reverse. These are not independent statistical samples.",
        f"The predefined one-shot independent selector has positive finite-catalogue regret on {sum(s['finite_catalogue_regret_fraction']>0 for s in regret)} of {len(regret)} complete inputs; maximum {max((s['finite_catalogue_regret_fraction'] for s in regret),default=0):.6%}.",'',
        '## Important contract-dependent mechanism','',
        f"Dense Prefill128: e0-s0 takes {long_gpu['completed_makespan_ns']} ns; e3-s0 takes {long_cpuatt['completed_makespan_ns']} ns. The all-GPU case performs {kv['count']} token-granular KV copies totaling {kv['bytes']} B and {kv['span_ns']} ns. Of that, {kv_latency} ns is the declared per-copy fixed latency. CPU Attention avoids those final-DRAM copies. This is a granularity/terminal-goal effect, not evidence that CPU Attention is physically faster.",'',
        '## Evidence and reproduction','',
        '- Every candidate retains generated prices, compiled IR, exact events, final catalogues, decisions and policy audit; per-case file hashes are indexed.',
        f"- Execution source unchanged: {completion['execution_sources_sha256']}; protocol: {completion['protocol_sha256']}; {completion['accuracy_file_count']} frozen accuracy files unchanged.",
        '- Enumeration covers this declared candidate matrix only. No numerical generation, new search method, multi-GPU, physical UMA or real-device accuracy claim follows.',
        '- H64 geometry, zero declared scratch and token-granular transfer costs limit extrapolation. Near ties lack a performance-error confidence bound.',
        f"- Canonical campaign:{campaign.as_posix()}",
        f"- Readable Chinese report:{(output/'REPORT_zh.html').as_posix()}",
        '- Reproduce using experiments/strategy_comparison_campaign.py and the companion protocol document.','']
    english+=['## Representative resource costs','',
        'Each pair uses the same input and terminal goals. The lowest numeric candidate ID represents exact latency ties; all tied candidates remain in the data.','',
        '| Scenario / candidate | Simulated ns | H2D B | D2H B | Peak DRAM B | Peak VRAM B | Weight reload B | Compute/COPY overlap ns |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in representative:
        english.append(f"| {r['scenario_id']} / {r['candidate_id']} | {r['completed_makespan_ns']} | {r['h2d_bytes']} | {r['d2h_bytes']} | {r['peak_dram_bytes']} | {r['peak_vram_bytes']} | {r['immutable_weight_reload_bytes']} | {r['copy_compute_event_overlap_ns']} |")
    english+=['','## Adaptive expert rules and unexercised mechanisms','']
    for a in adaptive_checks:
        english.append(f"- {a['execution']} versus fixed CPU-expert e2, excluding equivalence controls: {a['same_latency']} of {a['complete_pairs']} pairs have identical latency; {a['same_event_trace']} have identical event hashes. This is not evidence of adaptive-policy benefit or universal policy equivalence.")
    if no_compute_overlap:
        english.append('- CPU/GPU compute overlap is zero in every run. This catalogue exercises placement and compute/transfer overlap, not concurrent tensor partitioning.')
    if no_ssd:
        english.append('- SSD traffic is zero in every run; no SSD-offload benefit is evaluated.')
    if audit_receipt:
        english.append(f"- An independent stdlib-only audit replays all {audit_receipt['attempts_checked']} cases, including exact costs, dependencies, allocation peaks and tensor goals. It checks internal consistency, not hardware prediction accuracy.")
    english.append('')
    if english_doc:
        Path(english_doc).write_text('\n'.join(english),encoding='utf-8',newline='\n')
    if summary_json:
        Path(summary_json).write_text(json.dumps(summaries,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    inventory={str(p.relative_to(output)).replace('\\','/'):sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='delivery.json'}
    (output/'delivery.json').write_text(json.dumps(dict(campaign_completion_sha256=sha(campaign/'completion.json'),
        records_sha256=completion['records_sha256'],analysis_sha256=completion['analysis_sha256'],
        report_script_sha256=sha(__file__),files=inventory),ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(report=(output/'REPORT_zh.html').as_posix(),runs=len(records),figures=len(list(figures.glob('*.png')))),ensure_ascii=False))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--english-doc',type=Path)
    parser.add_argument('--summary-json',type=Path)
    parser.add_argument('--audit',type=Path,help='optional independent completed-campaign audit receipt')
    args=parser.parse_args(argv)
    build(args.campaign,args.out,args.english_doc,args.summary_json,args.audit)


if __name__=='__main__':
    main()
