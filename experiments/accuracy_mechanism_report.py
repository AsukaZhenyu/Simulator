"""Evidence-backed repair report; does not collect, fit, or qualify prices."""
from __future__ import annotations
from datetime import datetime, timezone
from html import escape
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'paper-write/第八教学周/experiment-reports/accuracy-mechanism-repair-20261028'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def link(p,label=None):
    p=Path(p).resolve()
    return f'<a href="{escape(p.as_uri(),quote=True)}">{escape(label or p.name)}</a>'
def percent(v):return f'{100*v:.2f}%'
def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+escape(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'

def figure(summaries):
    """Standalone scientific plot with editable English text."""
    import io
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager
    font_manager.findfont('Times New Roman',fallback_to_default=False)
    colors=['#123b64','#5f93bb','#d88d37'];kinds=['static_affine','zero_initial_recovery','prepared_initial_recovery']
    with plt.rc_context({'font.family':'Times New Roman','font.size':13,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(1,2,figsize=(11.5,4.4))
        for ax,(key,title) in zip(axes,[('first_mean_APE','First-copy mean prediction error'),('burst_mean_APE','Burst mean prediction error')]):
            for i,(kind,label) in enumerate(zip(kinds,['Static affine','Idle recovery','Prepared-state recovery'])):
                values=[next(s for s in summaries if s['memory_mode']==m and s['kind']==kind)[key]*100 for m in ['registered','pageable']]
                bars=ax.bar([i*.24-.24,1+i*.24-.24],values,width=.21,color=colors[i],label=label,zorder=3)
                ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=11)
            ax.set(xticks=[0,1],xticklabels=['Registered','Pageable'],ylim=(0,52),ylabel='Mean absolute percentage error (%)',title=title)
            ax.grid(axis='y',color='#dfe5ec',zorder=0);ax.set_axisbelow(True)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.01))
        fig.tight_layout(rect=(0,.09,1,1));buf=io.StringIO();fig.savefig(buf,format='svg',metadata={'Date':None});fig.savefig(OUT/'h2d-independent.png',dpi=180);plt.close(fig)
    return buf.getvalue()

def timeline_figure(primary, paired_forecasts):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager
    font_manager.findfont('Times New Roman',fallback_to_default=False)
    samples=[('heldout-cpu_tail-r0','CPU-tail: r0'),
             ('heldout-separated-r1','Separated: r1 (remaining timing failure)')]
    with plt.rc_context({'font.family':'Times New Roman','font.size':12,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(1,2,figsize=(11.5,4.2))
        for ax,(job,title) in zip(axes,samples):
            row=next(r for r in primary['rows'] if r['job_id']==job and r['provider']=='memoryless_four_state_macro')
            old=next(r for r in paired_forecasts['rows'] if r['job_id']==job and r['variant']=='prepared_single')
            x=list(range(1,17));ms=lambda values:[v/1e6 for v in values]
            ax.plot(x,ms(row['measured_block_ns']),color='#151f29',marker='o',markersize=3,label='Measured')
            ax.plot(x,ms(old['prediction']['block_ns']),color='#d88d37',linestyle='--',label='Single-call calibration')
            ax.plot(x,ms(row['predicted_block_ns']),color='#123b64',marker='s',markersize=3,label='Continuous calibration')
            ax.set(title=title,xlabel='Work block (16 invocations)',ylabel='Block latency (ms)',xticks=[1,4,8,12,16],xlim=(.7,16.3),ylim=(0,None))
            ax.grid(axis='y',color='#dfe5ec');ax.set_axisbelow(True)
        axes[1].axvspan(7,8,color='#5f93bb',alpha=.12)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.01))
        fig.tight_layout(rect=(0,.1,1,1))
        fig.savefig(OUT/'dynamic-blocks.svg',metadata={'Date':None});fig.savefig(OUT/'dynamic-blocks.png',dpi=180);plt.close(fig)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    a=ROOT/'build/dynamic-diagnosis-20261028/h2d-v9-bare-v1'
    h=ROOT/'build/dynamic-diagnosis-20261028/h2d-v9-independent-diagnostic-v1'
    n=ROOT/'build/real-moe-validation-20261027/numerical-diagnostic-v6/NUMERICAL_DIAGNOSIS.json'
    old=ROOT/'build/real-moe-validation-20261027/frozen-v5b-reproduction-v1/REPRODUCTION_ANALYSIS.json'
    m=ROOT/'build/real-moe-validation-20261028/mmid-attachment-v1'
    b=ROOT/'build/dynamic-diagnosis-20261028/external-v8-stage-v2'
    independent_moe=ROOT/'build/real-moe-validation-20261028/mmid-fence-numerical-v2/NUMERICAL_EVALUATION.json'
    assert independent_moe.is_file() and read(independent_moe)['gate_passed'], 'numerical claims require completed independent evidence'
    result=read(h/'DIAGNOSTIC_EVALUATION.json');fit=read(a/'fit.json');support=read(a/'baseline-support.json')
    gate=read(b/'00_observer_gate-observer-gate.json')
    original=(m/'original.log').read_text();fixed=(m/'syncwarp.log').read_text()
    assert '4 hazards displayed (0 errors, 4 warnings)' in original
    assert '0 hazards displayed (0 errors, 0 warnings)' in fixed
    assert not read(a/'frozen-barrier.json')['profile_gate_passed']
    assert result['strict_qualified']==0 and not result['parameters_refitted'] and not result['prices_exported']
    assert not gate['passed']
    rows=result['rows'];physical={}
    for row in rows:physical[(row['job_id'],row['bytes'],row['requested_idle_ms'])]=row
    assert len(rows)==72 and len(physical)==24
    summary=dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
        numerical_v6_processes=read(n)['processes_completed'],old_binary_reproductions=read(old)['processes_completed'],
        historical_logit_outlier_cause_proven=False,mmid_standalone_reported_hazards=dict(original=4,syncwarp=0),
        H2D_independent_processes=6,H2D_independent_bursts=288,H2D_independent_copy_invocations=4608,
        H2D_provider_evaluations=72,H2D_unique_physical_cells=24,
        H2D_mean_passed=result['diagnostic_mean_accuracy_passed'],
        H2D_mean_and_stability_passed=result['diagnostic_mean_and_stability_passed'],
        H2D_qualified=0,H2D_models_exported=0,H2D_summaries=result['summaries'],
        fine_runtime_observer_pairs=sum(x['passed'] for x in gate['pairs']),fine_runtime_observer_declared=3,
        fine_runtime_observer_qualified=False)
    if independent_moe.exists():
        verified_moe=read(independent_moe)
        summary['MoE_independent_numerical']=dict(
            gate_passed=verified_moe['gate_passed'],completed_processes=verified_moe['completed_processes'],
            completed_request_observations=verified_moe['completed_request_observations'],
            comparisons=[{k:v for k,v in c.items() if k!='requests'} for c in verified_moe['comparisons']],
            route_audit=verified_moe['strict_fixed_replay_capture_audit']['totals'],
            historical_outlier_cause_proven=verified_moe['original_race_cause_of_old_logits_proven'],
            whole_model_performance_accuracy_evaluated=verified_moe['whole_model_performance_accuracy_evaluated'])
    macro_path=ROOT/'build/dynamic-diagnosis-20261028/external-bare-macro-v2/INDEPENDENT_EVALUATION.json'
    continuous_path=ROOT/'build/dynamic-diagnosis-20261028/external-continuous-macro-v10/INDEPENDENT_EVALUATION.json'
    paired_path=continuous_path.parent/'paired-price-ablation-v1/PAIRED_EVALUATION.json'
    if macro_path.exists():
        macro=read(macro_path)
        summary['bare_macro_independent']={k:macro[k] for k in
            ('providers','repeat_variability','primary_all_six_passed','thresholds','no_refit',
             'whole_LLM_dynamic_accuracy_validated','raw_CV_is_prediction_gate',
             'actual_start_or_internal_boundaries_used_in_prediction')}
    if continuous_path.exists():
        continuous=read(continuous_path)
        summary['continuous_lifecycle_independent']={k:continuous[k] for k in
            ('providers','repeat_variability','primary_all_six_passed','thresholds','no_refit',
             'whole_LLM_dynamic_accuracy_validated','heldout_schedules_and_processes_new',
             'heldout_mathematical_seed_previously_used')}
    if paired_path.exists():summary['paired_price_ablation']=read(paired_path)['summaries']
    capacity_path=ROOT/'build/accuracy-mechanism-20261028/MOE_CAPACITY.json'
    if capacity_path.exists():summary['MoE_current_capacity']=read(capacity_path)
    (OUT/'RESULTS.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
    (OUT/'h2d-independent.svg').write_text(figure(result['summaries']),encoding='utf-8',newline='\n')
    top=table(['部分','本轮证据','结论'],[
        ['混合模式数值','新完整向量诊断与原二进制复查均未复现旧异常；独立原语racecheck确认共享内存竞争','已修复一个明确隐患；旧logits异常的因果关系仍未知'],
        ['H2D','新profile 6进程；新尺寸/闲置/数值种子独立诊断6进程','首copy均值预测改善；基线漂移及稳定性未通过'],
        ['细粒度设备波动','完成响应阶段、亲和性/作用域、编译与计时边界；OFF观察器2/3对照通过','未获观察器资格，未将其价格用于性能预测'],
        ['新MoE模型验收','4个新进程、6个新请求×4种运行方式；完整50304维logits、图指纹及路由验收通过','数值与观察器一致性通过；未认证全模型性能准确性']])
    tr=[]
    for s in result['summaries']:
        tr.append([s['memory_mode'],s['kind'],percent(s['first_mean_APE']),percent(s['burst_mean_APE']),f"{s['diagnostic_mean_passed']}/12",'0/12'])
    evidence=[n,old,m/'PLAN.json',m/'original.log',m/'syncwarp.log',
        ROOT/'build/real-moe-validation-20261028/mmid-fence-runtime-v1/BUILD_COMPLETE.json',
        ROOT/'build/real-moe-validation-20261028/mmid-fence-numerical-v2/manifest.json',
        a/'manifest.json',a/'fit.json',a/'baseline-support.json',a/'frozen-barrier.json',
        h/'manifest.json',h/'frozen-diagnostic.json',h/'DIAGNOSTIC_EVALUATION.json',b/'00_observer_gate-observer-gate.json']
    if macro_path.exists():evidence.append(macro_path)
    if continuous_path.exists():evidence.extend([continuous_path,continuous_path.parent/'manifest.json',continuous_path.parent/'frozen-barrier.json'])
    if paired_path.exists():evidence.extend([paired_path,paired_path.parent/'FREEZE_BARRIER.json'])
    if continuous_path.exists():
        evidence.extend([continuous_path.parent/'LIFECYCLE_DIAGNOSIS.json',continuous_path.parent/'BLOCK_PHASE_DIAGNOSIS.json'])
    if independent_moe.exists():evidence.append(independent_moe)
    if capacity_path.exists():evidence.append(capacity_path)
    sources=[ROOT/'modeling/llm_infer_model/performance/transfer_recovery.py',ROOT/'modeling/llm_infer_model/performance/runtime_response.py',
        ROOT/'profiling/dynamic_validation/h2d_bare_v9.py',ROOT/'profiling/dynamic_validation/h2d_bare_v9_fit.py',
        ROOT/'profiling/dynamic_validation/h2d_v9_diagnostic_holdout.py',ROOT/'profiling/real_moe_validation/mmid_runtime_fix.py',
        ROOT/'profiling/real_moe_validation/mmid_numerical.py']
    if continuous_path.exists():sources.extend([ROOT/'profiling/dynamic_validation/continuous_macro_campaign.py',ROOT/'profiling/dynamic_validation/continuous_macro_fit.py',ROOT/'profiling/dynamic_validation/continuous_price_ablation.py',ROOT/'profiling/dynamic_validation/native/v10/continuous_probe.cpp'])
    extra=''
    if macro_path.exists():
        extra='<h2>5. 完整Attention宏任务的独立波动验收</h2><p>使用全部3份原OFF无回调对照，以及9份ON标定。稳态标定依据实际负载窗口包围完整计算：原25.864ms迟到记录保留原failed状态，在新的稳态验证中记录其适用范围；动态heldout仍保留原10ms入场要求，不按实测起点重新对齐预测。性能阈值保持进程均值误差≤10%、分块APE的P90≤20%。</p>'
        extra+=table(['预测器','通过/声明','平均进程APE','最大进程APE','最差进程分块P90'],
            [[s['provider'],f"{s['passed']}/{s['declared']}",percent(s['mean_process_APE']) if s['mean_process_APE'] is not None else '未知',percent(s['worst_process_APE']) if s['worst_process_APE'] is not None else '未知',percent(s['worst_block_APE_P90']) if s['worst_block_APE_P90'] is not None else '未知'] for s in macro['providers']])
        extra+='<p>四状态参数仅来自标定组件的耗时；静态对照只使用OFF价格，四状态模型按预定CPU/GPU负载计划在modeling内积分，沿自己的完成时间递推。两种预测都在6个新进程采集前冻结。原始CV单独报告：计划中的资源变化本来就可能让相邻块时间变化，不能直接将这种CV当成预测失败。</p>'
        extra+=table(['独立进程','预测器','均值APE','分块P90','验收'],
            [[r['job_id'],r['provider'],percent(r['mean_response_APE']) if r['status']=='evaluated' else '未知',percent(r['block_APE_P90']) if r['status']=='evaluated' else '未知','通过' if r['accuracy_passed'] else '未通过'] for r in macro['rows']])
        extra+='<p>适用范围仅为一个固定形状的完整Attention组件、已声明的GEMM竞争者和本机核绑定，不代表完整LLM在所有外部应用与设备状态下已准确。完整原始指标见'+link(macro_path)+'。</p>'
    if continuous_path.exists():
        extra+='<h2>连续标定后的新独立验收</h2><p>上一轮的失败集中于同一交错计划，三次重复总耗时CV约0.5%；CPU-only连续调用约为单次标价的1.82倍，循环间隙占比不足0.003%。据此修正profiling生命周期：12个新标定进程在各自竞争状态下保留一次预热，再测连续至少1秒的完整调用。按各进程调用均值、再对三个进程等权平均取得四状态价格；modeling的积分公式不变。所有原始调用、预热和间隙保留，超过1秒的单个调用不被当成重复稳定性证据。</p>'
        extra+='<p>这轮使用两个新负载计划、六个新进程。数学种子曾用于前一轮，未再声称首次未见；旧验证结果用于定位问题，未参与新价格拟合。标定额外状态内预热1次，独立目标没有这次额外调用，首个目标瞬态由独立验收检验。两次验收计划不同，不能只比较3/6与新通过数来归因。</p>'
        extra+=table(['新验收预测器','通过/声明','平均进程APE','最大进程APE','最差进程分块P90'],
            [[s['provider'],f"{s['passed']}/{s['declared']}",percent(s['mean_process_APE']) if s['mean_process_APE'] is not None else '未知',percent(s['worst_process_APE']) if s['worst_process_APE'] is not None else '未知',percent(s['worst_block_APE_P90']) if s['worst_block_APE_P90'] is not None else '未知'] for s in continuous['providers']])
        extra+=table(['新独立进程','预测器','均值APE','分块P90','验收'],
            [[r['job_id'],r['provider'],percent(r['mean_response_APE']) if r['status']=='evaluated' else '未知',percent(r['block_APE_P90']) if r['status']=='evaluated' else '未知','通过' if r['accuracy_passed'] else '未通过'] for r in continuous['rows']])
    if paired_path.exists():
        paired=read(paired_path)
        timeline_figure(continuous,read(paired_path.parent/'FROZEN_PREDICTIONS.json'))
        extra+='<h3>同进程标定、同目标数据的配对对照</h3><p>为分离价格估计口径的作用，额外在新验证开始前冻结了两个预测器：同一批12个标定进程的首次调用均值、及后续连续调用均值。两者使用相同g和同六份实测目标；单次价格作为生命周期不匹配的诊断对照，不被提升为合格成本。固定先后顺序仍不能区分所有物理原因。</p>'
        extra+=table(['价格来源','满足原误差阈值/声明','平均进程APE','最差分块P90'],
            [[s['variant'],f"{s['thresholds_met']}/{s['declared']}",percent(s['mean_process_APE']) if s['mean_process_APE'] is not None else '未知',percent(s['worst_block_APE_P90']) if s['worst_block_APE_P90'] is not None else '未知'] for s in paired['summaries']])
        extra+='<p>这仍是六个硬件目标上的多个预测器对照，不是额外十二次独立硬件实验。完整证据见'+link(continuous_path)+'、'+link(paired_path)+'。</p>'
        extra+='<img src="dynamic-blocks.svg" alt="Measured and predicted block latencies"><small>示例明确包含一个通过案例cpu_tail-r0、一个剩余失败案例separated-r1；完整六项均保留在上表。</small>'
        extra+='<p class="note">剩余两项失败的直接表现为慢段的逻辑块归属错位：在图中的第7块之前，预测已比实测滞后约35.33/67.05ms，1.5–1.75秒的脉冲因此进入预测第7块，而实际落在第8块（原始JSON索引为6/7，从0计数）。脉冲错位后，单块结束时间偏差放大至约299/363ms，不能把35–67ms当作最大时序误差。两块合计误差约2.28%/3.26%，仅用于解释误差放大，未合并时间块或修改原验收。CPU-only连续响应与标价之比已由旧诊断的1.82修正至0.9886；动作外间隙仍不足0.002%，尚不能把相位偏差归因于某个已识别的OS或队列机制。</p>'
    cvfirst=[r['first_measured_population_CV'] for r in physical.values()];cvburst=[r['burst_measured_population_CV'] for r in physical.values()]
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Accuracy mechanism repair — 2026-10-28</title>
<style>body{{max-width:1160px;margin:36px auto;padding:0 26px;color:#19304a;background:#f4f6f9;font:17px/1.75 "Times New Roman","Microsoft YaHei",serif}}h1,h2{{color:#123b64}}section{{background:white;padding:24px 30px;margin:20px 0;border:1px solid #d8e1eb;border-radius:8px}}table{{width:100%;border-collapse:collapse;font-size:15px;margin:16px 0}}th,td{{border:1px solid #cbd6e3;padding:9px;text-align:left;vertical-align:top}}th{{background:#eaf0f6}}.note{{border-left:5px solid #d88d37;padding:12px 20px;background:#fff5e7}}code,pre{{font:14px/1.6 Consolas,monospace;white-space:pre-wrap;word-break:break-word}}a{{color:#165d95}}img{{width:100%}}small{{color:#54677a}}</style>
<h1>混合数值、H2D与设备波动：机制修复与独立诊断</h1><small>2026-10-28 · 测量与拟合在profiling，状态演化与耗时计算在modeling。证据生成UTC：{escape(summary['generated_at_utc'])}</small>
<section><h2>1. 当前结论</h2><p class="note">本轮找到了一个可复核的GPU共享内存竞争，完成私有修复并通过新真实MoE数值验收；H2D恢复模型在独立数据上的首copy均值预测明显改善。但H2D基线与原始耗时仍不稳定，设备波动准确性仍受适用范围限制，不能称全部修好。</p>{top}</section>
<section><h2>2. 混合模式：区分历史异常与已证实的缺陷</h2><p>完整logits诊断完成6个新进程、24个请求、192个输出；保留原二进制与请求字节的复查完成6个进程、12个请求、96个输出。本轮均未重现旧异常，这只缩小了问题范围，不证明旧失败是随机性，也不构成修复证明。</p>
<p>独立索引原语测试覆盖42种token尾部、路由与stride组合、252次GPU启动，两版整数输出均与CPU参考一致。进一步运行Compute Sanitizer，原版在共享内存写入与跨lane读取之间报告4组RAW警告；仅增加一条全warp同步的版本为0。私有运行库只替换mmid对象，原DLL、其他对象及host驱动的来源哈希保持可查。</p>
<p>NVIDIA文档说明warp同步提供内存排序，寄存器shuffle/reduce本身不能替代共享内存同步。见<a href="https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/">NVIDIA warp primitives说明</a>与<a href="https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html">Compute Sanitizer文档</a>。这证明索引原语存在并已消除该检查器报告的同步隐患；尚不能把历史全模型logits异常归因于它。</p>
<p>沙箱内两次检查器附加超时已原样保留；沙箱外工具成功附加后才获得上述结果。未修改注册表或驱动配置。新真实MoE验收保留完整50304维logits，只有实际token/位置/序列输入前缀一致的行才能比较；生成轨迹分岔后不把不同输入的后续行混在一起。</p>
<p>已完成新真实MoE独立数值验收：6个未用于旧数值诊断的请求，分别运行原版生成、修复版生成、修复版回放、修复版采集，共4个进程、24个请求观测、192个输出。三组比较各48/48行完整logits逐字节一致、实际输入前缀一致、计算图前缀一致；这些比较复用同一批6个请求，不是144个独立请求。修复版回放与采集的严格审计还通过58/58个图指纹、918/918个必需专家层记录。未以teacher-forcing输出ID相同代替数值证据。</p>
<p>这批独立请求验证了修复与观察器一致性，仍未证明旧异常由该竞争导致，未覆盖所有硬件/图捕获路径，也没有把观察器包含的计时用于全模型性能准确性认证。</p></section>
<section><h2>3. H2D：模型、测量与独立结果</h2><p>计时沿用原始GGML完成式COPY边界，包括API提交及已有GPU/CPU完成等待。计时区间内采集器不额外插入CUDA事件、CUPTI或NVML查询；不把CUDA事件区间误称纯DMA，也不重复加入GPU内部时间。每个burst先执行完整准备图，随后请求闲置，再连续执行16次COPY；真实准备、闲置、每笔COPY及outer-loop区间均保留QPC时间戳。</p>
<p>modeling中的状态律为：基线b(B)=α+βB；最大额外响应A(B)、准备后额外响应S₀(B)均显式给出；q₀=S₀/A，闲置后q=1−(1−q₀)exp(−idle/τᵢ)，单笔T=ceil(b+Aq)，完成后q←q exp(−T/τᵣ)。初态与闲置共同占用同一容量，避免重复叠加惩罚。这是经验响应状态，尚未标定为某个物理缓存、电源或OS队列。</p>
<p>profile使用128KiB、512KiB、2MiB与0/1/5/20ms；独立诊断使用256KiB、1MiB与2/10ms，并更换数值种子、使用6个新进程。全部6个可识别候选的参数及预测在独立采集前冻结，未读取新目标耗时来修正参数、实际闲置超时或内部边界。profile共13,824次COPY，独立诊断共4,608次COPY、288个burst。</p>
<img src="h2d-independent.svg" alt="Independent H2D prediction errors">
<small>图中每个柱为同一预测器12个条件单元的均值APE的平均；不同预测器复用相同实测数据。</small>
{table(['内存','预测器','首copy均值MAPE','burst均值MAPE','两项均值≤20%','两项均值≤20%且原始CV≤10%'],tr)}
<p>72项预测比较复用了24个独立硬件条件单元，不能称72次独立硬件实验。静态预测通过2/24个均值条件；零初态与非零准备态恢复预测分别通过17/24。三种预测合计36/72，加入原始CV后0/72；24个物理单元本身也全部未通过CV门槛。首copy CV为{percent(min(cvfirst))}–{percent(max(cvfirst))}，burst CV为{percent(min(cvburst))}–{percent(max(cvburst))}。</p>
<p>原始CV衡量实测样本的离散程度，不等于逐样本预测误差；它本身既不能证明模型必然遗漏机制，也不能证明噪声不可建模。本轮同样输入状态下基线仍漂移，因此当前状态描述不足以支持稳定的条件预测，严格门槛未通过。</p><p>准备态增量没有稳定的独立收益：两种恢复模型均通过17/24，部分burst误差反而略高。因此不能把更复杂模型作为已经验证优于简单模型的结论。每16次COPY的循环间隙均值仅约0.65–1.08µs，解释不了主要偏差。</p>
<p class="note">同一基线早晚稳定性仅6/18、跨进程一致性仅1/6通过。原正式profile支持门槛保持失败，原正式heldout未启动、正式成本导出0项。额外独立诊断用于检验冻结模型的泛化，不会把失败门槛改成通过。</p></section>
<section><h2>4. 设备波动：统一响应框架与观察器边界</h2><p>modeling把一次完整响应分成互不重叠的API响应和host间隙，各自以实测四状态响应为参数，沿预测器自己的完成时间递推。输入是预先给定的CPU/GPU竞争计划、已验证的物理核关系及历史状态，不把目标自己的实测耗时当作未来输入；不模拟操作系统调度，也不将inclusive API响应解释为CPU有效工作或再加一份内部GPU耗时。</p>
<p>新profile程序在所有初始化完成后选择共同的未来epoch，再按固定计划启动其他计算。CPU目标与竞争者的实际亲和性、GPU竞争提交线程位置及物理核映射均验证。状态标价只接受完全落在同一预定状态中的响应，不按“启动时是什么状态”给跨边界响应填价格。实际负载窗口包围组件，只证明竞争者处于提供负载的阶段，不代表组件期间取得固定CPU份额或设备微架构状态恒定。</p>
{table(['OFF对照','无回调 / 有回调','相对差','观察器门槛'],[[p['pair_id'],f"{p['observed_target_ns']['stamped']/1e6:.3f} / {p['observed_target_ns']['runtime_callbacks']/1e6:.3f} ms",percent(p['relative_difference']),'通过' if p['passed'] else '未通过'] for p in gate['pairs']])}
<p>3对照仅2对通过5%门槛，不能把未获资格的细粒度数据作为正式成本；也不能仅凭一个失败对照断言回调必然造成全部差异。该分支的ON采集与性能验收保持未启动。完整Attention宏任务可直接测量，另用无回调的粗粒度响应检验相同竞争问题，避免将细粒度观察器限制误当作所有性能建模均不可行。</p>{extra}</section>
<section><h2>6. 如何理解剩余误差</h2><p>已经有证据支持“闲置和恢复状态对首copy很重要”，但当前输入状态还不能解释持续变化的响应基线。没有证据证明剩余CV不可建模；宿主等待/调度、驱动/队列、链路及电源状态仍需区分。实际定时等待超过请求值也被记录，但没有把它自动认定为主要原因，更没有用实测超时回填预测。</p><p>按用户授权关闭非必要应用并回收闲置UI页面后，内存达到原门槛，已完成上述真实MoE验收。未终止Codex后端或网络/系统核心进程，未降低原有容量、resident峰值或可用内存看门狗门槛；未以小原语通过替代真实MoE验收。</p></section>
<section><h2>7. 代码与证据</h2><p>计算机制：</p><ul>{''.join('<li>'+link(p)+'</li>' for p in sources[:2])}</ul><p>采集、拟合与验收：</p><ul>{''.join('<li>'+link(p)+'</li>' for p in sources[2:])}</ul>
{table(['证据','SHA256'],[[link(p),f'<code>{sha(p)}</code>'] for p in evidence])}<p>原报告与失败样本均保留。此报告不改变既有Dense小图/多层固定输入验收的适用范围，不宣称全模型、多设备、全部策略已获性能准确性验证。</p></section></html>'''
    (OUT/'REPORT_zh.html').write_text(html,encoding='utf-8',newline='\n')
    index=dict(generated_at_utc=summary['generated_at_utc'],generator=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),files=[dict(path=str(p),sha256=sha(p)) for p in evidence+sources],outputs=[dict(name=p.name,sha256=sha(p)) for p in OUT.iterdir() if p.is_file() and p.name!='FILES_INDEX.json'])
    (OUT/'FILES_INDEX.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps(dict(report=str(OUT/'REPORT_zh.html'),results=str(OUT/'RESULTS.json')),ensure_ascii=False))

if __name__=='__main__':main()
