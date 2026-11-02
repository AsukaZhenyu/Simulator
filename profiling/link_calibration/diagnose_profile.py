"""Profile-only descriptive/LOCO diagnostics; never exports/freeze costs.

All raw samples remain, including slow samples and boundary-transition rows.
Only the preregistered qualified snapshot scope enters the diagnostic regression.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import statistics
import numpy as np
from .campaign import read,write,sha,receipt,require,classify,FIELDS

def describe(values):
    values=list(values)
    if not values:return dict(count=0)
    mean=statistics.mean(values)
    return dict(count=len(values),mean=mean,median=statistics.median(values),cv=statistics.pstdev(values)/mean if mean else None,
                minimum=min(values),maximum=max(values),p10=float(np.percentile(values,10)),p90=float(np.percentile(values,90)))
def matrix(rows,stateful):
    if stateful:return np.asarray([[1.,r['bytes']/1048576. if r['state_key'][0]==3 else 0.,r['bytes']/1048576. if r['state_key'][0]==4 else 0.] for r in rows])
    return np.asarray([[1.,r['bytes']/1048576.] for r in rows])
def nonnegative_fit(rows,stateful):
    x=matrix(rows,stateful);y=np.asarray([r['suffix_mean_us'] for r in rows]);best=None
    for mask in range(1,1<<x.shape[1]):
        columns=[i for i in range(x.shape[1]) if mask&(1<<i)];beta=np.zeros(x.shape[1]);candidate=np.linalg.lstsq(x[:,columns],y,rcond=None)[0]
        if np.any(candidate<0):continue
        beta[columns]=candidate;loss=float(np.sum((x@beta-y)**2))
        if best is None or loss<best[0]:best=loss,beta
    require(best is not None,'nonnegative diagnostic fit unavailable');return best[1]
def errors(rows,beta,stateful):
    y=np.asarray([r['suffix_mean_us'] for r in rows]);pred=matrix(rows,stateful)@beta;ape=np.abs(pred-y)/y
    return dict(count=len(rows),MAPE=float(np.mean(ape)),median_APE=float(np.median(ape)),p90_APE=float(np.percentile(ape,90)),maximum_APE=float(np.max(ape)),
                mean_absolute_error_us=float(np.mean(np.abs(pred-y))),within_10pct=int(np.sum(ape<=.1)),within_20pct=int(np.sum(ape<=.2)))
def fit_report(rows,stateful):
    beta=nonnegative_fit(rows,stateful);folds=[];predictions=[]
    for context in sorted({r['context_id'] for r in rows}):
        train=[r for r in rows if r['context_id']!=context];test=[r for r in rows if r['context_id']==context];b=nonnegative_fit(train,stateful)
        folds.append(dict(omitted_context=context,train_bursts=len(train),test_bursts=len(test),diagnostic_coefficients=b.tolist(),metrics=errors(test,b,stateful)))
        for row,p in zip(test,matrix(test,stateful)@b):predictions.append(dict(context_id=context,state_key=row['state_key'],bytes=row['bytes'],actual_us=row['suffix_mean_us'],candidate_us=float(p),APE=abs(float(p)-row['suffix_mean_us'])/row['suffix_mean_us']))
    apes=[r['APE'] for r in predictions]
    names=['shared_completed_boundary_fixed_us','Gen3_completed_response_us_per_MiB','Gen4_completed_response_us_per_MiB'] if stateful else ['shared_completed_boundary_fixed_us','common_completed_response_us_per_MiB']
    return dict(status='candidate_unqualified_profile_only',parameters=dict(zip(names,beta.tolist())),parameter_count=len(beta),
                coefficient_estimator='nonnegative least squares; enumerate active subsets, no theoretical link ratios',
                each_qualified_burst_one_equal_weight=True,in_sample=errors(rows,beta,stateful),leave_one_context_out=folds,
                LOCO_aggregate=dict(count=len(apes),MAPE=statistics.mean(apes),median_APE=statistics.median(apes),p90_APE=float(np.percentile(apes,90)),maximum_APE=max(apes),within_20pct=sum(a<=.2 for a in apes)),
                LOCO_predictions=predictions,LOCO_is_independent_heldout=False,cost_exported=False,model_frozen=False,physical_DMA_rate_identified=False,intercept_is_identified_CPU_work=False)

def analyze(campaign,out):
    campaign,out=Path(campaign).resolve(),Path(out).resolve();require(not out.exists(),'new profile-only analysis directory required')
    summary=read(campaign/'profile-summary.json');executions=read(campaign/'profile-executions.json');groups=defaultdict(list);controls=defaultdict(list);qualified=[];unqualified=[];sources=[]
    for e in executions:
        require(e['status']=='completed','this diagnostic expects all three retained completed contexts');require(sha(e['native']['path'])==e['native']['sha256'],'raw native changed');n=read(e['native']['path']);sources.append(e['native']);hz=n['clock']['frequency_hz']
        for index,row in enumerate(n['bursts']):
            suffix=[x['wall_ns']/1000 for x in row['copies'][1:]];gap=(row['burst_begin_tick']-row['prime_end_tick'])*1e6/hz
            item=dict(context_id=e['context_id'],burst_index=index,trial=row['trial'],size_slot=row['size_slot'],bytes=row['bytes'],suffix_mean_us=statistics.mean(suffix),suffix_copy_us=suffix,first_copy_us=row['first_copy_ns']/1000,prime_to_burst_gap_us=gap,
                      raw_native=e['native'],control=row['control'])
            controls[(e['context_id'],row['bytes'],row['control'])].append(item);c=classify(row)
            if c['qualified']:
                item['state_key']=list(c['prior_key']);qualified.append(item);groups[(tuple(c['prior_key']),row['bytes'],e['context_id'])].append(item)
            elif row['control']=='state_queries':
                item.update(prior_key=list(c['prior_key']) if c['prior_key'] is not None else None,reason=c['reason'],pcie_before=row['pcie_before'],pcie_after=row['pcie_after']);unqualified.append(item)
    require({tuple(r['state_key'][1:]) for r in qualified}=={(4,8,8)} and {r['state_key'][0] for r in qualified}=={3,4},'three-parameter diagnostic scope differs; do not pool different link widths/capabilities')
    per_context=[];per_state=defaultdict(list)
    for (key,size,context),rows in sorted(groups.items()):
        flat=[x for r in rows for x in r['suffix_copy_us']];entry=dict(state_key=list(key),bytes=size,context_id=context,qualified_bursts=len(rows),suffix_copy_us=describe(flat),burst_suffix_mean_us=describe(r['suffix_mean_us'] for r in rows),first_copy_us=describe(r['first_copy_us'] for r in rows),prime_to_burst_gap_us=describe(r['prime_to_burst_gap_us'] for r in rows))
        per_context.append(entry);per_state[(key,size)].append(entry)
    spreads=[]
    for (key,size),rows in sorted(per_state.items()):
        means=[r['suffix_copy_us']['mean'] for r in rows];spreads.append(dict(state_key=list(key),bytes=size,contexts=len(rows),context_means_us=[dict(context_id=r['context_id'],mean_us=r['suffix_copy_us']['mean']) for r in rows],context_mean_distribution=describe(means),max_to_min_context_mean_ratio=max(means)/min(means)))
    comparisons=[]
    for context,size,_ in sorted(k for k in controls if k[2]=='bare'):
        a=controls[(context,size,'bare')];b=controls[(context,size,'state_queries')];ma=statistics.mean(r['suffix_mean_us'] for r in a);mb=statistics.mean(r['suffix_mean_us'] for r in b)
        comparisons.append(dict(context_id=context,bytes=size,bare_suffix_copy_mean_us=ma,queried_all_suffix_copy_mean_us=mb,query_over_bare_ratio=mb/ma,query_minus_bare_percent=(mb/ma-1)*100,
                                bare_gap_us=describe(r['prime_to_burst_gap_us'] for r in a),query_gap_us=describe(r['prime_to_burst_gap_us'] for r in b),
                                interpretation='different observed inter-burst gaps and unknown bare state; not an identified pure query-overhead effect'))
    fixed_models=dict(common_state_agnostic_2parameter=fit_report(qualified,False),measured_state_3parameter=fit_report(qualified,True))
    equal_cells=[dict(context_id=r['context_id'],state_key=r['state_key'],bytes=r['bytes'],suffix_mean_us=r['suffix_copy_us']['mean']) for r in per_context]
    fixed_models['state_3parameter_equal_context_size_cell_weight_sensitivity']=fit_report(equal_cells,True)
    examples=[]
    for gen in (3,4):
        rows=[r for r in qualified if r['bytes']==2097152 and r['state_key'][0]==gen]
        for label,r in (('minimum_mean',min(rows,key=lambda r:r['suffix_mean_us'])),('maximum_mean',max(rows,key=lambda r:r['suffix_mean_us']))):examples.append(dict(selection=label,**r,suffix_copy_distribution=describe(r['suffix_copy_us'])))
    report=dict(schema='link-copy-size-profile-only-diagnosis-1',status='candidate_unqualified',source_campaign=receipt(campaign/'manifest.json'),source_summary=receipt(campaign/'profile-summary.json'),source_executions=receipt(campaign/'profile-executions.json'),raw_sources=sources,analysis_tool=receipt(__file__),
                acquisition_contexts=3,declared_profile_bursts=216,bare_control_bursts=108,queried_bursts=108,qualified_boundary_state_bursts=len(qualified),unqualified_query_bursts=unqualified,
                coverage_passed=summary['exact_state_coverage']['coverage_passed'],exact_state_coverage=summary['exact_state_coverage'],state_key_fields=list(FIELDS),per_state_size_context=per_context,inter_context_spread=spreads,query_vs_bare=comparisons,diagnostic_models=fixed_models,same_state_counterexamples=examples,
                all_slow_samples_retained=True,post_state_defines_predictor_key=False,post_state_only_qualifies_scope=True,model_frozen=False,cost_table_exported=False,heldout_collected=False,extra_hardware_samples_collected=False,
                conclusions=dict(observed_generation_width_alone_sufficient=False,physical_PCIe_cause_excluded=False,
                    reason='Large residual spread remains for equal pre/post generation/width. Boundary queries cannot prove internal link constancy or isolate host/driver/device waiting.',
                    recommendation='Do not freeze/export the three-parameter candidate or start heldout from this unsupported cohort. Keep PCIe state as a useful conditioning/qualification variable, and diagnose residual completed-boundary mechanisms.'))
    out.mkdir(parents=True);write(out/'DIAGNOSIS.json',report)
    md=['# H2D 多尺寸与 PCIe 状态：仅校准集诊断','',
        '**结论：当前观测到的 PCIe 代际/宽度有解释价值，但单独使用它们还不够。不能据此冻结性能模型。**','',
        f'3个新进程均完成。108组查询 burst 中107组前后状态一致，1组状态变化保留为不支持；另有108组无查询对照。所有慢样本保留。Gen3/128 KiB 只有2个进程，其余五格为3个，原覆盖门槛未通过。','',
        '| 状态 | 尺寸 | 进程 | 后15次均值（µs/次） | 单次CV | burst均值CV |','|---|---:|---|---:|---:|---:|']
    for r in per_context:md.append(f"| Gen{r['state_key'][0]} ×{r['state_key'][2]} | {r['bytes']//1024} KiB | {r['context_id']} | {r['suffix_copy_us']['mean']:.2f} | {r['suffix_copy_us']['cv']:.1%} | {r['burst_suffix_mean_us']['cv']:.1%} |")
    md+=['','同一Gen4×8、同为2MiB、burst前后状态读数一致时，后15次均值仍有明显不同：','']
    for r in examples:
        if r['state_key'][0]==4:md.append(f"- {r['selection']}：{r['suffix_mean_us']:.2f} µs/次，15次范围 {min(r['suffix_copy_us']):.1f}–{max(r['suffix_copy_us']):.1f} µs（{r['context_id']}，trial {r['trial']}）。")
    md+=['','这说明“测到同一个链路状态”还不能唯一决定完成时间。它也不能反过来证明PCIe物理变化无关：两端读数无法确认整个burst内部状态恒定，主机提交、同步返回、驱动或设备等待也尚未分开。','',
         '查询条件本身也不同：bare的prime结束到burst开始约0.2–0.3 µs，带查询的均值约107–261 µs。虽然requested idle都是0，但实际间隙不相同。因此query/bare差异不是已识别的“纯查询开销”，不能把两组直接合并。','',
         '| 仅校准集候选 | 参数数 | 拟合集MAPE | 留一进程MAPE | 留一进程P90 APE |','|---|---:|---:|---:|---:|']
    for name,r in fixed_models.items():md.append(f"| {name} | {r['parameter_count']} | {r['in_sample']['MAPE']:.1%} | {r['LOCO_aggregate']['MAPE']:.1%} | {r['LOCO_aggregate']['p90_APE']:.1%} |")
    md+=['','三参数候选为“共享固定完成边界残差＋Gen3字节斜率＋Gen4字节斜率”，直接从本批数据拟合，未使用理论代际比例。固定项不等于纯CPU工作，斜率不等于DMA峰值。留一进程仍是已观察校准集内部诊断，不能替代新输入、新尺寸的独立验收。','',
         '本轮不补采、不运行heldout、不降低覆盖门槛、不导出成本表、不冻结模型。建议保留PCIe状态作为条件变量，先解释相同状态下的剩余差异，再考虑新的独立验收。',
         '',f"完整数值与原始证据：[DIAGNOSIS.json](<{(out/'DIAGNOSIS.json').as_posix()}>)",'']
    (out/'DIAGNOSIS_zh.md').write_text('\n'.join(md),encoding='utf-8',newline='\n')
    return dict(diagnosis=receipt(out/'DIAGNOSIS.json'),explanation=receipt(out/'DIAGNOSIS_zh.md'),fit_metrics={k:r['LOCO_aggregate'] for k,r in fixed_models.items()})
def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(__import__('json').dumps(analyze(a.campaign,a.out),ensure_ascii=False))
if __name__=='__main__':main()
