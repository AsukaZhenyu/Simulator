"""Render completed H2D/history evidence without inventing missing results.

Report generation is deliberately separate from native acquisition. All claims
derive from completed JSON evidence; the report fits no parameter and selects
no provider according to heldout performance.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
from html import escape
import json
import math
from pathlib import Path
import shutil
import statistics


LABELS={'old_hot':'旧独立热测基线','context_static':'同上下文静态基线','history_conditional':'历史条件化模型（预先指定主模型）'}
COLORS={'old_hot':'#9AABB8','context_static':'#78A2BF','history_conditional':'#183E64'}


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def receipt(path): return dict(path=str(Path(path).resolve()),sha256=sha(path))


def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False); stream.write('\n')


def require(condition,message):
    if not condition: raise ValueError(message)


def fmt(value,scale=1,digits=2):
    return '未知' if value is None else f'{value/scale:,.{digits}f}'


def pct(value): return '未知' if value is None else f'{100*value:.2f}%'


def short(key):
    parts=key.split('-'); layers=parts[0][1:]; strategy=next(p[1:] for p in parts if p.startswith('s') and p[1:].isdigit())
    return f'{layers}L S{strategy}'


def table(headers,rows):
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def avg(values):
    values=[v for v in values if v is not None]
    return statistics.mean(values) if values else None


def first_wup(row):
    values=[a for a in row['actions'] if a['focus_2MiB_H2D'] and (a['semantic'].get('target_id') or '').endswith(':W_up')]
    return min(values,key=lambda a:a['action_index']) if values else None


def phase_rows(diagnostics):
    groups=defaultdict(list)
    for row in diagnostics['rows']:
        if row['instrumentation']!='cupti': continue
        action=first_wup(row)
        if action is None or 'physical' not in action: continue
        groups[(row['program_key'],row['memory_mode'],row['mode'],row['history_intervention'])].append((row,action))
    result=[]
    for (key,memory,mode,intervention),items in sorted(groups.items()):
        physical=[a['physical'] for _,a in items]
        api_names=sorted({name for p in physical for name in p['api_union_ns_by_name']})
        result.append(dict(program_key=key,memory_mode=memory,mode=mode,intervention=intervention,
            processes=len(items),host_ns=avg(a['wall_ns']['mean'] for _,a in items),
            copy_call_ns=avg(a['copy_call_ns']['mean'] for _,a in items),
            gpu_fence_ns=avg(a['gpu_sync_ns']['mean'] for _,a in items),
            cpu_fence_ns=avg(a['cpu_sync_ns']['mean'] for _,a in items),
            device_ns=avg(p['device_memcpy_union_ns']['mean'] for p in physical),
            pre_device_ns=avg(p['pre_device_wall_ns']['mean'] for p in physical),
            post_device_ns=avg(p['post_device_wall_ns']['mean'] for p in physical),
            covered_invocations=sum(p['exact_payload_coverage_count'] for p in physical),
            expected_invocations=sum(p['expected_invocations'] for p in physical),
            all_capture_integrity=all(r['capture_integrity_pass'] for r,_ in items),
            api_ns={name:avg(p['api_union_ns_by_name'][name]['mean'] for p in physical if name in p['api_union_ns_by_name']) for name in api_names},
            raw_jobs=[r['job_id'] for r,_ in items]))
    return result


def clock_rows(clock):
    rows=[]
    fields=('sm_clock_mhz','memory_clock_mhz','pstate','temperature_c','current_pcie_generation','current_pcie_width')
    for row in clock.get('rows',[]):
        if not row.get('telemetry_enabled'): continue
        sensors={}
        for field in fields:
            values=[w['values'][field] for w in row.get('native_batch_observations',[]) if field in w['values'] and w['values'][field]['n']]
            count=sum(v['n'] for v in values)
            sensors[field]=dict(n=count,min=min((v['min'] for v in values),default=None),
                max=max((v['max'] for v in values),default=None),
                mean=sum(v['mean']*v['n'] for v in values)/count if count else None)
        halves={}
        for label,test in (('early',lambda t:t<8),('late',lambda t:t>=8)):
            windows=[w for w in row.get('native_batch_observations',[]) if test(w['trial'])]
            half_values={}
            for field in fields:
                values=[w['values'][field] for w in windows if field in w['values'] and w['values'][field]['n']]
                count=sum(v['n'] for v in values)
                half_values[field]=dict(n=count,min=min((v['min'] for v in values),default=None),
                    max=max((v['max'] for v in values),default=None),
                    mean=sum(v['mean']*v['n'] for v in values)/count if count else None)
            halves[label]=half_values
        result=dict(job=row['job'],program_key=row['program_key'],mode=row['mode'],status=row['status'],
            sensor_values=sensors,early_late_sensor_values=halves,
            replay_early_mean_ns=row.get('early_mean_ns'),replay_late_mean_ns=row.get('late_mean_ns'),
            replay_late_over_early_minus_one=row.get('late_over_early_minus_one'),
            sampling=None,unknown_reasons={})
        if row.get('telemetry') and Path(row['telemetry']['path']).exists():
            raw=read(row['telemetry']['path'])
            result['sampling']={k:raw.get(k) for k in ('requested_period_ms','sample_count','actual_begin_period_ns','errors','polling_thread_exited_verified')}
            unknown=defaultdict(set)
            for sample in raw.get('samples',[]):
                for field,message in sample['observation'].get('unknown',{}).items():
                    unknown[field].add(str(message))
            result['unknown_reasons']={k:sorted(v) for k,v in unknown.items()}
        rows.append(result)
    return rows


def make_plots(destination,summary,phases):
    """Scientific vector figures with one shared Times New Roman style."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_path=font_manager.findfont('Times New Roman',fallback_to_default=False)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'axes.labelsize':10,
                         'axes.titlesize':11,'legend.fontsize':9,'svg.fonttype':'none',
                         'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.dpi':160,'savefig.bbox':'tight'})
    folder=destination/'figures'; folder.mkdir()
    cases=summary['providers']['history_conditional']['cases']; labels=[short(c['program_key']) for c in cases]
    positions=list(range(len(cases))); width=.19
    fig,ax=plt.subplots(figsize=(8.0,3.5))
    ax.bar([x-1.5*width for x in positions],[c['observed_primary_mean_ns']/1e6 if c['observed_primary_mean_ns'] is not None else float('nan') for c in cases],width,label='Measured (bare)',color='#333333')
    names={'old_hot':'Isolated hot (older context)','context_static':'Matched-context static','history_conditional':'History-conditioned'}
    for j,provider in enumerate(('old_hot','context_static','history_conditional')):
        data=summary['providers'][provider]['cases']
        ax.bar([x+(j-.5)*width for x in positions],[c['predicted_mean_ns']/1e6 for c in data],width,label=names[provider],color=COLORS[provider])
    ax.set_xticks(positions,labels); ax.set_ylabel('Time per replay (ms)'); ax.set_xlabel('Complete Dense workload and placement')
    ax.legend(ncol=2,frameon=False); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(folder/'complete-dense-times.svg'); fig.savefig(folder/'complete-dense-times.png'); plt.close(fig)
    usable=[r for r in phases if r['intervention']=='none']
    keys=sorted({r['program_key'] for r in usable})
    fig,axes=plt.subplots(1,len(keys),figsize=(8.4,3.3),squeeze=False)
    for ax,key in zip(axes[0],keys):
        order=[(m,mode) for m in ('pageable','registered') for mode in ('target-hot','full')]
        data=[next((r for r in usable if r['program_key']==key and (r['memory_mode'],r['mode'])==pair),None) for pair in order]
        x=list(range(len(data)))
        for delta,field,color,label in ((-.18,'host_ns','#183E64','Host-completed action'),(.18,'device_ns','#78A2BF','Device memcpy activity')):
            ax.bar([i+delta for i in x],[r[field]/1e3 if r and r[field] is not None else float('nan') for r in data],.34,color=color,label=label)
        ax.set_xticks(x,['Pageable\nhot','Pageable\nfull','Registered\nhot','Registered\nfull'])
        ax.set_title(short(key)); ax.set_ylabel('2 MiB H2D time (µs)'); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    axes[0][0].legend(frameon=False,fontsize=8)
    fig.tight_layout(); fig.savefig(folder/'h2d-context-phases.svg'); fig.savefig(folder/'h2d-context-phases.png'); plt.close(fig)
    return dict(font='Times New Roman',resolved_font=font_path,figures=['complete-dense-times.svg','complete-dense-times.png','h2d-context-phases.svg','h2d-context-phases.png'],
                figure_scope='CUPTI host/device intervals overlap; side-by-side bars are not additive. Equal weighting of available process means.')


def render(out,clock_out,destination,idle_out=None):
    out=Path(out).resolve(); destination=Path(destination).resolve()
    require(not destination.exists(),'new report destination required; published evidence cannot be overwritten')
    summary=read(out/'summary.json'); diagnostics=read(out/'diagnostic-summary.json')
    model=read(out/'model-frozen.json'); predictions=read(out/'predictions-frozen.json')
    manifest=read(out/'manifest.json')
    require(summary['schema']=='h2d-history-independent-qualification-1','complete qualification summary required')
    require(summary['declared_native_processes']==48,'heldout denominator differs')
    require(diagnostics['declared']==52,'diagnostic denominator differs')
    require(set(summary['providers'])==set(LABELS),'three prespecified providers required')
    clock_path=Path(clock_out).resolve()/'summary.json' if clock_out else None
    clock=read(clock_path) if clock_path and clock_path.exists() else {}
    idle_path=Path(idle_out).resolve()/'idle-summary.json' if idle_out else None
    if idle_path: require(idle_path.is_file(),'requested idle report evidence is not complete')
    idle=read(idle_path) if idle_path and idle_path.exists() else {}
    if idle:
        require(idle['schema']=='h2d-history-idle-diagnostic-summary-1','idle evidence schema differs')
        idle_manifest=read(idle_path.parent/'manifest.json')
        require(idle_manifest['target_action']['bytes']==2097152,'idle diagnostic actual target byte count differs')
    phases=phase_rows(diagnostics); sensors=clock_rows(clock)
    destination.mkdir(parents=True); evidence=destination/'evidence'; evidence.mkdir()
    sources=[]
    names=['summary.json','diagnostic-summary.json','model-frozen.json','predictions-frozen.json','frozen-barrier.json',
           'manifest.json','case-metrics.csv','process-metrics.csv','block-errors.csv']
    for name in names:
        path=out/name; require(path.exists(),'required report evidence missing: '+name)
        shutil.copy2(path,evidence/name); sources.append(dict(source=receipt(path),published=receipt(evidence/name)))
    if clock:
        shutil.copy2(clock_path,evidence/'clock-summary.json')
        sources.append(dict(source=receipt(clock_path),published=receipt(evidence/'clock-summary.json')))
    idle_names=[]
    if idle:
        for source_name,published_name in (('idle-summary.json','idle-summary.json'),
                ('manifest.json','idle-manifest.json'),('diagnostic-summary.json','idle-diagnostic-summary.json'),
                ('diagnostic-executions.json','idle-executions.json')):
            source=idle_path.parent/source_name
            require(source.is_file(),'idle report evidence missing: '+source_name)
            shutil.copy2(source,evidence/published_name); idle_names.append(published_name)
            sources.append(dict(source=receipt(source),published=receipt(evidence/published_name)))
    write(evidence/'phase-means.json',phases); write(evidence/'clock-sensor-summary.json',sensors)
    plot=make_plots(destination,summary,phases)
    primary=summary['providers']['history_conditional']; static=summary['providers']['context_static']; old=summary['providers']['old_hot']
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>H2D 与连续执行状态：测量、建模及独立验收</title>',
        '<style>body{max-width:1180px;margin:36px auto;padding:0 24px;color:#202c36;font:16px/1.75 Georgia,"Times New Roman","Microsoft YaHei",serif;background:#fafbfd}h1,h2,h3{color:#173d63;line-height:1.35}h1{font-size:30px}h2{margin-top:38px;border-bottom:2px solid #b4c8da;padding-bottom:8px}p{margin:12px 0}.callout{padding:16px 22px;background:#eaf1f7;border-left:5px solid #173d63}.caution{background:#fff5df;border-left-color:#ab7622}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}th,td{border:1px solid #cdd8e1;padding:8px 10px;text-align:left;vertical-align:top}th{background:#e5edf4;white-space:nowrap}tbody tr:nth-child(even){background:#f2f5f8}.ok{color:#226641;font-weight:bold}.fail{color:#a43729;font-weight:bold}.small{font-size:13px;color:#526574}code,pre{font-family:Consolas,monospace}pre{padding:16px;background:#edf2f6;overflow:auto}.figure{margin:24px 0}.figure img{width:100%;height:auto;background:white}.caption{font-size:14px;color:#526574}a{color:#245e96}.formula{font-family:"Times New Roman",serif;font-size:19px;padding:12px 18px;background:#eef3f8}</style><body>',
        '<h1>H2D 与连续执行状态：测量、建模及独立验收</h1>',
        '<p class="small">证据生成于 '+escape(datetime.now(timezone.utc).isoformat())+'。本文报告实际产物，不把诊断假设、软件测试或模拟结果写成真实硬件验收。</p>',
        '<div class="callout"><b>预先指定主模型的完整验收：'+str(primary['fully_qualified_cases'])+'/6。</b> 同上下文静态基线为 '+str(static['fully_qualified_cases'])+'/6；旧独立热测基线为 '+str(old['fully_qualified_cases'])+'/6。前两者使用相同的独立分项校准数据，可用于检验历史条件化是否带来额外收益；旧基线的校准上下文不同，仅作历史参照。</div>',
        '<h2>1. 本轮实际完成了什么</h2>',
        '<p>测量程序位于 <code>profiling/h2d_history</code>，负责记录动作区间、内存类型、CPU 端点计数和 CUPTI 活动。逻辑状态更新与预测耗时位于 <code>modeling/llm_infer_model/performance</code>，不在采集程序中拟合整层修正系数。</p>']
    count_rows=[]
    for split,label in (('diagnostic','分项与干预诊断'),('profile','独立分项校准'),('heldout','完整策略独立验收')):
        declared=manifest['protocol']['counts'][split]
        if split=='diagnostic': completed=diagnostics['completed']
        elif split=='heldout': completed=summary['completed_native_processes']
        else:
            executions=read(out/'profile-executions.json'); completed=sum(e['status']=='completed' for e in executions)
        count_rows.append([label,declared,completed,declared-completed])
    if clock: count_rows.append(['被动时钟/链路诊断（单独证据）',clock['declared_processes'],clock['completed_processes'],clock['declared_processes']-clock['completed_processes']])
    if idle: count_rows.append(['受控空闲与 H2D 诊断（单独证据）',idle['declared'],idle['completed'],idle['declared']-idle['completed']])
    parts.append(table(['阶段','原定进程数','成功完成','未完成/失败'],count_rows))
    parts += ['<p>真实计算范围为 Dense，H=512、FFN 中间维度 1024、Prefill=64，1 层与 2 层，F32；同一台 CPU+单 GPU，GGML CPU 线程数为 1。S0 为全 CPU，S1 为全 GPU，S2 为 GPU Projection/FFN 与 CPU Attention。完整验收使用原有普通主机内存路径；注册内存只用于单独干预对照。</p>',
        '<p>每个区块先 reset，再完整执行 1 次不计时 prime；计时区间内固定输入重放 8 次。每个新进程采集 16 个区块。bare 与 stamped 使用相同区块数、重放数和准备方式，分属相邻的新进程。该采样契约与旧报告的 8 区块、observer 每轮 4 区块契约不同，不能将新结果追溯为旧契约已通过。</p>',
        '<h2>2. H2D 变慢发生在哪里</h2>',
        '<p>保持实际 Dense arena 与同一个 2 MiB <code>W_up</code>，对比独立重复搬运（hot）、完整计算流程中的搬运（full）、普通/注册内存，以及 full 流程中立即只读触碰源缓冲区。下表每格对已有各轮进程均值等权平均；2 层只取与 hot 目标一致的第一层 W_up。所有 CUPTI 结果均属于带观测器的诊断。</p>']
    phase_table=[]
    for row in phases:
        condition=('源缓冲区只读干预' if row['intervention']=='source-read' else row['mode'])
        phase_table.append([short(row['program_key']),row['memory_mode'],condition,row['processes'],
            fmt(row['host_ns'],1000),fmt(row['copy_call_ns'],1000),fmt(row['device_ns'],1000),
            fmt(row['pre_device_ns'],1000),fmt(row['post_device_ns'],1000),
            str(row['covered_invocations'])+'/'+str(row['expected_invocations'])])
    parts.append(table(['上下文','主机内存','执行条件','进程数','动作总耗时 μs','GGML copy μs','设备 MEMCPY μs','设备开始前 μs','设备完成后 μs','精确活动覆盖'],phase_table))
    parts += ['<div class="figure"><img src="figures/h2d-context-phases.svg" alt="Host response and device memcpy time under matched H2D conditions"><p class="caption">图 1. Host-completed action and device memcpy activity. 设备活动与主机响应有重叠，图中两种柱不能相加。不同条件来自独立新进程，跨进程变化与观测开销仍存在。</p></div>',
        '<p>GGML 的这条 H2D 路径包含 <code>cudaMemcpyAsync</code> 和内部 <code>cudaStreamSynchronize</code>，随后 Executor 还调用 backend 同步。因此 <code>copy_call_ns</code> 是包含内部同步的完整 GGML copy 响应；<code>submit_copy_api_ns</code> 只是它的别名，不是独立可加的一项。</p>',
        '<p>设备开始前/完成后的区间只说明时间位置，不能直接命名为 pageable staging、缓存未命中或操作系统等待。设备活动按显式 external-ID/correlation 关联，检查方向、字节数和是否位于动作区间；缺失或模糊的关联保持未知。主机 API 与设备区间重叠，不能重复计费。</p>']
    api_table=[]
    for row in phases:
        mem=avg(v for name,v in row['api_ns'].items() if ':cudaMemcpyAsync' in name)
        sync=avg(v for name,v in row['api_ns'].items() if ':cudaStreamSynchronize' in name)
        api_table.append([short(row['program_key']),row['memory_mode'],row['intervention'] if row['intervention']!='none' else row['mode'],fmt(mem,1000),fmt(sync,1000),fmt(row['gpu_fence_ns'],1000),fmt(row['cpu_fence_ns'],1000)])
    parts.append(table(['上下文','内存','条件','cudaMemcpyAsync API μs','StreamSynchronize API 并集 μs','额外 GPU fence μs','CPU fence μs'],api_table))
    parts.append('<p class="small">同名 API 时间先在各动作内部取区间并集；runtime 与 driver 不相加。表中 StreamSynchronize 是整个动作内该 API 的并集，不能据此把内部 PTDS 等待与额外 backend 等待完全分开。</p>')
    if idle:
        parts += ['<h3>受控空闲：只比较目标搬运，不比较含 Sleep 的整段时间</h3>',
            '<p>在两层混合方案的同一笔实际 2 MiB W_up H2D 前，请求 <code>Sleep(5)</code>；与无插入空闲的独立新进程对照。普通/注册内存各进行两轮，切换顺序。两组使用相同 CUPTI 观测器。Sleep 位于目标 H2D 的计时区间之外、整个区块之内，所以完整区块总耗时不作为性能准确性或慢化证据。</p>']
        idle_table=[]
        idle_by_id={r['job_id']:r for r in idle['rows']}
        for pair in idle['paired_contrasts']:
            if pair['status']!='complete':
                idle_table.append([pair['round'],pair['memory_mode'],'未完成','未知','未知','未知','未知','未知']); continue
            changes=pair['target_action_changes']; host=changes['host_wall_ns']; device=changes['device_memcpy_union_ns']; pre=changes['pre_device_wall_ns']
            a,b=idle_by_id[pair['no_idle_job']],idle_by_id[pair['idle_job']]
            coverage=str(a['exact_payload_coverage_count'])+'/'+str(a['expected_invocations'])+'；'+str(b['exact_payload_coverage_count'])+'/'+str(b['expected_invocations'])
            idle_table.append([pair['round'],pair['memory_mode'],fmt(host['no_idle'],1000)+' → '+fmt(host['idle'],1000),
                pct(host['relative']),fmt(device['no_idle'],1000)+' → '+fmt(device['idle'],1000),
                fmt(pre['no_idle'],1000)+' → '+fmt(pre['idle'],1000),coverage,
                '是' if pair['complete_activity_coverage_both'] else '否，物理区间仅作不完整描述'])
        parts.append(table(['轮次','内存','目标 H2D μs：无空闲 → 有空闲','目标变化','设备活动 μs：前 → 后',
                            '设备开始前 μs：前 → 后','2 MiB 精确覆盖：无/有空闲','两组完整活动覆盖'],idle_table))
        sleep_table=[]
        for row in idle['rows']:
            if row['history_intervention']!='idle': continue
            duration=row['actual_intervention_wall_ns']
            sleep_table.append([row['round'],row['memory_mode'],duration['n'],fmt(duration['mean'],1e6),
                                fmt(duration['min'],1e6)+' 至 '+fmt(duration['max'],1e6),
                                str(row['exact_payload_coverage_count'])+'/'+str(row['expected_invocations'])])
        parts.append(table(['轮次','内存','实际空闲次数','实际空闲均值 ms','实际空闲范围 ms','目标精确活动覆盖'],sleep_table))
        parts.append('<p>这项干预能够描述“插入空闲后，目标响应是否改变”，但 Sleep 同时可能改变 CPU 就绪、驱动准备和设备运行状态；本组没有同步时钟采样，不能唯一归因于 PCIe 节能、频率或任何单一机制。跨进程变化与 CUPTI 开销仍然存在。本组数据没有重新拟合已经冻结的模型。</p>')
    parts += ['<h2>3. 连续执行与观测开销</h2>',
        '<p>前后段比较使用每个进程前 8 个区块与后 8 个区块。CV 描述区块平均耗时的波动；CV 高本身不能证明随机性不可建模。线程/进程 CPU 时间、周期数和页错误只作端点诊断：没有据此区分 Running、Ready、Blocked，也不把周期数等同于有效 FLOPs。</p>']
    drift_groups=defaultdict(list)
    for row in diagnostics['rows']:
        if row['history_intervention']=='none' and row['memory_mode']=='pageable':
            drift_groups[(row['program_key'],row['mode'],row['instrumentation'])].append(row)
    drift_table=[]
    for (key,mode,instrument),rows in sorted(drift_groups.items()):
        drifts=[r['replay_trend']['late_over_early_minus_one'] for r in rows]
        drift_table.append([short(key),mode,instrument,len(rows),pct(avg(drifts)),pct(min(drifts)),pct(max(drifts)),pct(max(r['replay_time_ns']['cv'] for r in rows))])
    parts.append(table(['上下文','执行方式','观测器','进程数','平均后/前变化','最小变化','最大变化','最大区块 CV'],drift_table))
    observer_groups=defaultdict(list)
    for row in diagnostics['contrasts']:
        if row['comparison'] in ('stamped_minus_bare','cupti_minus_bare'):
            observer_groups[(row['program_key'],row['memory_mode'],row['mode'],row['comparison'])].append(row['replay_time_ratio_minus_one'])
    parts.append(table(['上下文','内存','执行方式','对照','进程对数','平均相对差','范围'],
        [[short(key),memory,mode,comparison,len(values),pct(avg(values)),pct(min(values))+' 至 '+pct(max(values))]
         for (key,memory,mode,comparison),values in sorted(observer_groups.items())]))
    parts += ['<p>上述 observer 对照来自不同新进程，因此包含观测器影响与进程间变化，不能视为纯粹的因果开销。CUPTI 诊断与 bare 验收保持分开；不使用带 CUPTI 的动作时间替代 bare 完整执行真值。</p>',
        '<h2>4. 状态演化与耗时计算</h2>',
        '<p>增加的是一个可选的、具有明确校准范围的逻辑执行历史模型。它从已声明的计算和搬运动作更新状态，不读取目标动作未来的实测时间、未来总利用率，也不用策略名、层号或程序 ID 选择价格。</p>',
        '<div class="formula">s<sub>k</sub> = snapshot(x<sub>k</sub>, a<sub>k</sub>)<br>T̂<sub>k</sub> = C<sub>frozen</sub>(semantic(a<sub>k</sub>), bucket(s<sub>k</sub>))<br>x<sub>k+1</sub> = F(x<sub>k</sub>, a<sub>k</sub>, T̂<sub>k</sub>),　t̂<sub>k+1</sub> = t̂<sub>k</sub> + T̂<sub>k</sub></div>',
        '<p>本轮真正用于条件化的字段只有 H2D 源张量上次逻辑访问后，其他已声明 DRAM 张量的不同数据量。把这段逻辑 footprint 用向下取 2 的幂分桶，零和未知分别保留。它反映可由动作序列计算的复用关系，<b>不是实测缓存容量、缓存命中率、真实页面集合或驱动 staging 状态</b>。没有分项校准数据的语义/状态组合报未知，不静默使用静态价格补齐。</p>',
        '<p>每个区块从新的逻辑历史开始，显式推进 1 次完整 prime，再预测 8 次重放。真实 reset 的写访问、物理缓存/时钟状态和上一计时区块的恢复没有被此初始化识别。当前模型也没有把跨区块的温度或频率漂移拟合进去。</p>',
        '<p>profiling 只提供独立 profile split 的逐动作区间及对应证据；完整区块耗时、诊断作业和 heldout 的逐动作耗时都不进入参数拟合。共 '+str(model['calibration_processes'])+' 个校准进程，'+str(len(model['entries']))+' 个静态/条件响应单元。参数、预测与模型源码在首个 heldout 进程之前冻结。</p>',
        '<h2>5. 完整策略的独立验收</h2>',
        '<p>主模型在采集 heldout 之前指定为历史条件化模型。对照一是使用相同新分项数据、取消 H2D 历史条件的静态模型；对照二是旧的独立热测模型，其硬件/运行时校准上下文与本轮不同，不能将差异全部归因于新增状态。</p>',
        '<div class="figure"><img src="figures/complete-dense-times.svg" alt="Measured and predicted complete Dense replay times"><p class="caption">图 2. Complete Dense replay time. 黑色是 bare 独立测量均值；三种预测均在本轮 heldout 采集前冻结。</p></div>']
    metric_rows=[]
    for provider in ('old_hot','context_static','history_conditional'):
        for case in summary['providers'][provider]['cases']:
            mark='<span class="ok">通过</span>' if case['fully_qualified'] else '<span class="fail">未通过</span>'
            metric_rows.append([LABELS[provider],short(case['program_key']),fmt(case['predicted_mean_ns'],1e6),fmt(case['observed_primary_mean_ns'],1e6),
                pct(case['maximum_primary_process_mean_ape']),pct(case['pooled_primary_block_ape_p90']),pct(case['maximum_primary_process_cv']),pct(case['maximum_paired_observer_difference']),mark])
    parts.append(table(['预测器','案例','预测 ms','实测 ms','最差进程均值 APE','区块 APE P90','最大 bare CV','最大 observer 差','原门槛'],metric_rows))
    parts += ['<p>原门槛保持不变：4 个 bare 进程的均值误差分别 ≤10%，64 个 bare 区块误差的 P90 ≤20%（<code>higher</code> 分位数），每个 bare 进程 CV ≤10%，4 对 bare/stamped 均值差分别 ≤5%。数值/拷贝正确性和进程完整性也必须成立。64 个区块不被当成 64 个独立进程或独立统计样本；该 P90 不是单次算子尾延迟。</p>',
        '<p>更严格的“所有 bare 与 stamped 进程 CV 均通过”只作辅助结果，不替代原门槛：</p>']
    parts.append(table(['预测器','原门槛通过','所有 8 进程 CV 的辅助门槛通过'],[[LABELS[p],str(v['fully_qualified_cases'])+'/6',str(v['qualified_conservative_all_process_cv_cases'])+'/6'] for p,v in summary['providers'].items()]))
    failures=[]
    for case in primary['cases']:
        if not case['fully_qualified']:
            failures.append([short(case['program_key']),escape('、'.join(case['failed_gates'])),str(case['completed_primary_processes'])+'/4',str(case['completed_observer_processes'])+'/4'])
    if failures: parts.append(table(['主模型未通过案例','具体未过门槛','bare 完成','observer 完成'],failures))
    else: parts.append('<p>主模型在本轮六个限定案例中通过原完整门槛；这一结论不外推到未测的形状、模型、设备和资源波动轨迹。</p>')
    drifts=[]
    for case in primary['cases']:
        for p in case['processes']:
            if p['instrumentation']=='bare': drifts.append([short(case['program_key']),p['round'],pct(p['late_over_early_minus_one']),pct(p['trial_cv']),pct(p['mean_ape'])])
    parts.append(table(['主模型案例','进程轮次','后 8 / 前 8 变化','区块 CV','进程均值 APE'],drifts))
    parts += ['<h2>6. 时钟与链路观测能说明什么</h2>']
    if clock:
        parts.append('<p>时钟/PCIe 诊断单独运行 '+str(clock['completed_processes'])+'/'+str(clock['declared_processes'])+' 个进程；没有把采集到的未来目标状态输入预测器，也没有拟合状态到服务速度的函数。</p>')
        sensor_table=[]
        for row in sensors:
            def cell(field):
                value=row['sensor_values'][field]
                return '未知（无计时区间内有效样本）' if not value['n'] else fmt(value['mean'])+' ['+fmt(value['min'])+', '+fmt(value['max'])+']; n='+str(value['n'])
            sensor_table.append([short(row['program_key']),row['mode'],row['status'],cell('sm_clock_mhz'),cell('memory_clock_mhz'),cell('pstate'),cell('temperature_c'),cell('current_pcie_generation'),cell('current_pcie_width')])
        parts.append(table(['上下文','执行方式','进程状态','SM MHz','显存 MHz','P-state','温度 °C','PCIe 代际','PCIe 宽度'],sensor_table))
        parts.append(table(['上下文','执行方式','开启观测/关闭观测的均值变化'],[[short(r['program_key']),r['mode'],pct(r['enabled_over_disabled_mean_minus_one'])] for r in clock['contrasts']]))
        early_late=[]
        for row in sensors:
            if row['mode']!='full': continue
            def change_cell(field):
                a,b=(row['early_late_sensor_values'][half][field] for half in ('early','late'))
                return fmt(a['mean'])+' → '+fmt(b['mean'])+'；n='+str(a['n'])+'/'+str(b['n'])
            early_late.append([short(row['program_key']),fmt(row['replay_early_mean_ns'],1e6)+' → '+fmt(row['replay_late_mean_ns'],1e6),
                pct(row['replay_late_over_early_minus_one']),change_cell('sm_clock_mhz'),change_cell('memory_clock_mhz'),
                change_cell('current_pcie_generation'),change_cell('current_pcie_width')])
        if early_late:
            parts.append('<h3>同一完整进程的前后段观测</h3>')
            parts.append(table(['上下文','完整重放 ms：前 8 → 后 8','耗时变化','SM MHz：前 → 后',
                                '显存 MHz：前 → 后','PCIe 代际样本均值：前 → 后','PCIe 宽度样本均值：前 → 后'],early_late))
            parts.append('<p>传感器前后均值按完整落在相应 8 个计时区块内的查询括号样本数加权，而非按动作时间插值。PCIe 代际是离散状态，表中小数是样本均值，不代表存在该小数代际的链路。耗时与观测时钟/链路的同步变化只是相关证据，尚不能证明因果方向，也不等于已获得每笔搬运的可用带宽。观测包含推理自身的影响，没有作为未来外部资源轨迹输入预测。</p>')
        unknown=[]
        for row in sensors:
            for field,reasons in row['unknown_reasons'].items():
                unknown.append([short(row['program_key']),row['mode'],escape(field),escape('; '.join(reasons))])
        if unknown: parts.append(table(['上下文','方式','未知传感器','真实返回原因'],unknown))
        parts.append('<p>采样周期、实际查询间隔和失败原因见机器可读证据。驱动传感器有自身刷新窗口；提高轮询频率不会自动获得微秒级服务率。一个查询不能反推出每笔 2 MiB 搬运的瞬时状态。每格只含一个新进程，开启/关闭观测的差异混合了观测开销与进程间变化。</p>')
    else:
        parts.append('<p>本次生成没有可读取的完整时钟/链路诊断 summary，因此时钟、温度与 PCIe 变化对误差的解释保持未知。</p>')
    parts += ['<h2>7. 已支持的结论与边界</h2>',
        '<ul><li>同一语义动作的独立热测均值不能自动代表完整流程里的响应；本轮直接测量了实际 arena 与执行上下文的差异。</li>',
        '<li>主机内存注册、执行历史与观测器影响被分别报告。某项变化可观测，不等于其底层物理原因已被唯一识别。</li>',
        '<li>逻辑历史状态由 modeling 推进；独立分项参数与采集证据由 profiling 提供。主模型相对同上下文静态基线的实际收益，以本轮完整验收和逐案例指标为准。</li>',
        '<li>缓存命中、页面准备、驱动 staging、CPU 就绪等待、频率/温度恢复仍未形成经过完整验证的物理分解。不能宣称已找到所有大误差来源，也不能以 CV 为由将剩余问题归为不可控随机性。</li>',
        '<li>本轮不构成任意外部资源波动、真实长序列多轮 LLM、MoE、多硬件平台或 UMA 的准确性认证。</li></ul>',
        '<h2>8. 可复核产物</h2><ul>']
    for name in names:
        parts.append('<li><a href="evidence/'+escape(name)+'">'+escape(name)+'</a></li>')
    if clock: parts.append('<li><a href="evidence/clock-summary.json">clock-summary.json</a></li>')
    for name in idle_names: parts.append('<li><a href="evidence/'+escape(name)+'">'+escape(name)+'</a></li>')
    parts += ['<li><a href="evidence/phase-means.json">phase-means.json：逐条件 W_up 相位均值</a></li>',
              '<li><a href="evidence/clock-sensor-summary.json">clock-sensor-summary.json：观测粒度与未知原因</a></li>',
              '<li><a href="evidence/report-receipt.json">report-receipt.json：源文件及发布产物 SHA256</a></li></ul>',
              '<p class="small">原始逐动作与 CUPTI 大文件保留在实验目录，报告中的 JSON 记录其绝对路径和 SHA256；不以截图替代原始证据。</p></body></html>']
    html=destination/'REPORT_zh.html'
    with html.open('x',encoding='utf-8',newline='\n') as stream: stream.write('\n'.join(parts)+'\n')
    outputs=[receipt(html)]+[receipt(destination/'figures'/name) for name in plot['figures']]
    report=dict(schema='h2d-history-report-receipt-1',generator=receipt(Path(__file__)),sources=sources,
        outputs=outputs,plot_style=plot,primary_provider_prespecified='history_conditional',
        primary_fully_qualified_cases=primary['fully_qualified_cases'],declared_cases=6,
        measured_results_invented=False,parameters_refitted=False,physical_cause_identified=False)
    write(evidence/'report-receipt.json',report)
    return dict(report=str(html),primary_passes=primary['fully_qualified_cases'],primary_cases=6,
                diagnostic_completed=diagnostics['completed'],clock_completed=clock.get('completed_processes'),
                idle_completed=idle.get('completed'))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--out',required=True)
    parser.add_argument('--clock'); parser.add_argument('--idle'); parser.add_argument('--destination',required=True); args=parser.parse_args()
    print(json.dumps(render(args.out,args.clock,args.destination,getattr(args,'idle',None)),ensure_ascii=False))


if __name__=='__main__': main()
