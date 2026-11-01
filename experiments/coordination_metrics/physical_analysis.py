"""Summarize frozen physical-accounting results without running new cases."""
from collections import defaultdict
import hashlib
import html
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.coordination_metrics.physical_protocol import OUT,write

def main():
    data=json.loads((OUT/'physical_RESULTS.json').read_text(encoding='utf-8'))
    protocol=json.loads((OUT/'physical_PROTOCOL.json').read_text(encoding='utf-8'))
    conditions=data['conditions'];aliases=defaultdict(list)
    for row in conditions:aliases[row['action_signature']].append(row['id'])
    nonzero=[r for r in data['contrasts'] if any(r.get('reference_delta',{}).values())]
    timings={}
    for field in ('compile_wall_ns','evaluate_wall_ns'):
        vals=[r[field]/1e6 for r in conditions]
        timings[field.replace('_ns','_ms')]=dict(min=min(vals),median=statistics.median(vals),max=max(vals))
    metrics=dict(nonzero_physical_contrasts=len(nonzero),zero_physical_contrasts=len(data['contrasts'])-len(nonzero),
        max_absolute_differential_byte_error=max(abs(v) for r in data['contrasts'] for v in r['delta_error_bytes'].values()),
        max_absolute_per_condition_byte_error=max(abs(v) for r in conditions for v in r['absolute_byte_error'].values()),
        numerical_ports_including_reused_baselines=sum(r['numerical']['checked_logical_ports'] for r in conditions),
        max_absolute_numerical_error=max(r['numerical']['max_absolute_error'] for r in conditions),
        causal_expert_visibility_checks=sum(r.get('causal_checks',0) for r in conditions),
        declared_operations_range=[min(r['declared_operations'] for r in conditions),max(r['declared_operations'] for r in conditions)],
        declared_tensors_range=[min(r['declared_tensors'] for r in conditions),max(r['declared_tensors'] for r in conditions)],
        dynamic_events_range=[min(sum(r['event_counts'].values()) for r in conditions),max(sum(r['event_counts'].values()) for r in conditions)],
        timing=timings,canonical_action_groups=list(aliases.values()),
        negative_rejection_diagnostic='All 12 independently impossible VRAM=1B controls ended policy_dead_end/no_progress. They were not accepted, but the API did not give a specific capacity rejection. No Unsupported exception was recoded as physics-invalid.',
        zero_effect_scope='D02/R02 preparation changes action order but not totals/peaks in these ample-memory cases. R01 ordering and M02 cache none/LRU additionally collapse to the same observed action trace. These are preserved invariance/alias cases, not optimized or retuned to manufacture a delta.',
        compiler_space_scope='This checks preservation and physical accounting of supported policies. It does not bound the compiler optimization gap; does not prove few remaining lower-level optimizations; does not establish novelty over WriteScope/Timeloop or other systems.',
        hardware_scope='All costs are declared analytical prices. No native runtime, OS effects, device fluctuation prediction, DMA accuracy, real MoE router arithmetic or UMA accuracy is tested.')
    write(OUT/'physical_ANALYSIS.json',metrics)
    table=[]
    for row in data['contrasts']:
        d=row['reference_delta'];table.append('<tr><td>'+html.escape(row['pair'])+'</td><td>'+html.escape(row['family'])+'</td><td>'+row['split']+'</td>'+''.join('<td>'+str(d[k])+'</td>' for k in ('h2d_bytes','d2h_bytes','peak_dram_bytes','peak_vram_bytes'))+'<td>0</td></tr>')
    text=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>策略变更的物理代价守恒</title><style>
body{{max-width:1120px;margin:40px auto;padding:0 24px;background:#fff;color:#111;font:17px/1.65 Georgia,"Times New Roman",serif}}h1,h2{{line-height:1.2}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:7px 9px;border-bottom:1px solid #ccc;text-align:right}}th:first-child,td:first-child{{text-align:left}}.note{{padding:15px;border:1px solid #aaa;background:#f7f7f7}}code{{font-size:14px}}</style>
<h1>指标2：策略变更后，物理代价是否被忠实保留？</h1>
<p>已按运行前固定的12对策略变更，在两组模型尺寸上执行24组成对对照。48个名义条件全部完成；共有32种不同的实际动作序列，重复基线没有当作新的独立行为。</p>
<p class="note"><b>主要结果：</b>H2D、D2H、DRAM峰值和VRAM峰值的每条件误差与成对差分误差均为0字节。16对至少一个物理指标发生变化，8对为零变化。这里验证的是有限范围的策略表达、编译与物理记账；不是实际硬件耗时准确率，也不是策略性能提升。</p>
<h2>如何独立验证</h2><p>模型数学图来自独立手写的张量维度、权重形状和算子输入输出关系，不从编译结果反推期望。先核对实际图与独立图，再按实际动作时间线重算物理账本：初始驻留、搬运期间两个端点、待写输出空间、仍在被读取的张量、工作区和释放时点。账本函数不能读取生产汇总。</p>
<p>冻结尺寸为 development H=8、I=16、q=1；holdout H=16、I=32、q=3。两者均F32、2个query头/1个KV头、CPU与GPU独立内存各1MiB，缓存预算256KiB。宏任务和请求gate工作区明确设为16B；细化Attention/FFN任务为4B。请求流水与P/D为2层，MoE为2层、3专家、top-1；专家未来路由由模型持有，仅当前router完成后向策略开放。</p>
<h2>差分结果</h2><p>表中数字均为 B−A，单位字节。最后一列为四项差分误差绝对值的最大值；零变化仍保留。</p><table><thead><tr><th>Pair</th><th>Family</th><th>Geometry</th><th>ΔH2D</th><th>ΔD2H</th><th>ΔDRAM peak</th><th>ΔVRAM peak</th><th>Max error</th></tr></thead><tbody>{''.join(table)}</tbody></table>
<p>例如F01仅将实体FFN权重分片改为读取完整父张量的切片视图，H2D与VRAM峰值分别增加384B（development）和1536B（holdout）。A02与F02改变准备时机，总搬运量不变，但峰值VRAM会增加。P01显式增加所有层KV交接的边界，总搬运量不变，活跃DRAM峰值改变。</p>
<h2>哪些结论不能扩大</h2><p>D02/R02的预备加载改变动作顺序，但在当前宽松内存条件下没有改变这四项物理指标。R01的请求排序和M02的缓存规则进一步成为相同动作轨迹；因此本实验不能宣称这两种变更已经展示出性能价值，也不能拿48次运行假装48个独立策略行为。</p>
<h2>可行性和数值</h2><p>48个正例均有独立的保守可行证据：把全部声明张量和工作区放入任一1MiB池都能容纳，存在合法串行执行。另12个控制将VRAM改为1B；必需GPU操作数或输出都大于1B，独立证明不可行，全部未被接受。实际诊断均为policy_dead_end/no_progress，因此只能说没有错误接受，不能说编译器给出了精确容量错误原因。</p>
<p>独立标量数学实现与对实际动作执行NumPy回放，比较了{metrics['numerical_ports_including_reused_baselines']}个逻辑端口（含重复基线），最大绝对误差{metrics['max_absolute_numerical_error']:.3g}。Dense覆盖声明的RMSNorm、RoPE、GQA、残差和SwiGLU；MoE是明确规定的合成不透明算子语义，不等同于生产模型数值验证。共{metrics['causal_expert_visibility_checks']}次专家可见性检查没有发现提前执行未来专家。</p>
<h2>探索开销</h2><p>每条件4–35个算子、17–86个张量、27–185条动态事件。单次编译中位数{timings['compile_wall_ms']['median']:.1f}ms（{timings['compile_wall_ms']['min']:.1f}–{timings['compile_wall_ms']['max']:.1f}ms），单次模拟中位数{timings['evaluate_wall_ms']['median']:.1f}ms（{timings['evaluate_wall_ms']['min']:.1f}–{timings['evaluate_wall_ms']['max']:.1f}ms）。这是当前主机含输入生成/写盘的单次实现开销，不是稳定性能基准，也未对外部模拟器速度做比较。</p>
<h2>证据与适用范围</h2><p>结果：<a href="physical_RESULTS.json">physical_RESULTS.json</a>；协议：<a href="physical_PROTOCOL.json">physical_PROTOCOL.json</a>；摘要：<a href="physical_ANALYSIS.json">physical_ANALYSIS.json</a>。各条件下保存intent、独立数学图、实际IR/图/动作、独立账本及数值输出。运行前后核心与实验源码哈希一致。没有改动公共核心代码，没有真机测量。</p>
<p>本组数据适合支持“在所选策略变更上，用户只改高层执行/存储选择，编译模拟仍精确保持物理语义”的有限声明。它不证明抽象全覆盖，不证明编译器已经接近最优，也不单独证明论文创新性。</p></html>'''
    (OUT/'physical_REPORT_zh.html').write_text(text,encoding='utf-8',newline='\n')
    write(OUT/'analysis-source.json',dict(path=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))

if __name__=='__main__':main()
