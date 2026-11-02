"""English, Times New Roman figures from immutable scored dynamic evidence.

Reporting performs no fitting and runs no native workload. Run plotting only
outside a physical measurement window. SVG, PDF and 300dpi PNG share a style.
"""
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
import statistics

from profiling.dynamic_validation import external


COLORS={'static':'#666666','current_idle':'#D8802B','memoryless':'#D8802B','stateful':'#1E628B'}


def figure_setup():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    from matplotlib import pyplot as plt
    font=font_manager.findfont('Times New Roman',fallback_to_default=False)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'axes.labelsize':11,
        'axes.titlesize':11,'legend.fontsize':8,'lines.linewidth':1.4,'axes.linewidth':.7,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,
        'svg.fonttype':'none','savefig.dpi':300})
    return plt,font


def save(fig,out,name):
    paths=[]
    for suffix in ('svg','pdf','png'):
        path=out/(name+'.'+suffix);fig.savefig(path,bbox_inches='tight');paths.append(external.receipt(path))
    return paths


def idle_plot(path,out,plt):
    path=Path(path).resolve();manifest=external.read(path/'manifest.json')
    original=external.read(path/'summary.json');comparison=external.read(path/'provider-comparison.json')
    frozen=external.read(path/'predictions-frozen.json');barrier=external.read(path/'frozen-barrier.json')
    external.require(external.sha(path/'manifest.json')==barrier['manifest']['sha256'],'idle report plan changed')
    for name,sha in barrier['files'].items():external.require(external.sha(path/name)==sha,'idle report frozen artifact changed')
    fig,axes=plt.subplots(2,2,figsize=(8.2,5.2),sharex=True)
    for column,memory in enumerate(('pageable','registered')):
        records=[r for r in original['process_conditions'] if r['status']=='completed' and r['job']['memory_mode']==memory]
        idle=sorted({r['requested_idle_ms'] for r in records})
        for row,(observed,label) in enumerate((('observed_ordinal_mean_ns','First copy response (µs)'),('observed_burst_mean_ns','Mean response in 16-copy burst (µs)'))):
            ax=axes[row,column]
            for process in range(3):
                data=[next((r for r in records if r['job']['round']==process and r['requested_idle_ms']==x),None) for x in idle]
                ys=[(r[observed][0] if row==0 else r[observed])/1000 if r is not None else float('nan') for r in data]
                ax.plot(idle,ys,color='#8C8C8C',marker='o',markersize=3,alpha=.65,label='Measured process '+str(process+1))
            for provider,rows in frozen['providers'].items():
                values=[]
                for x in idle:
                    v=[r['predicted_ns'] for r in sorted(rows,key=lambda a:a['ordinal']) if r['memory_mode']==memory and r['requested_idle_ms']==x]
                    values.append((v[0] if row==0 else statistics.mean(v))/1000)
                ax.plot(idle,values,color=COLORS[provider],linestyle='--' if provider=='static' else '-',label=provider.replace('_',' ').title())
            ax.set_title(memory.title()+' host memory');ax.set_xlabel('Requested idle (ms)');ax.set_ylabel(label);ax.grid(alpha=.18)
    axes[0,0].legend(loc='best',ncol=2);fig.tight_layout()
    name=path.name+'-idle-responses';artifacts=save(fig,out,name);plt.close(fig)
    precision=path/'heldout-controller-precision.json'
    controller_status={}
    for split in ('profile','heldout'):
        source=path/(split+'-controller-precision.json')
        if source.exists():
            value=external.read(source)
            controller_status[split]=dict(qualified_processes=value['qualified_processes'],declared=value['declared'],
                completed=value['completed'],evidence=external.receipt(source))
    review=path/'ROOT_CONTROLLER_REVIEW.json'
    return dict(campaign=str(path),manifest=external.receipt(path/'manifest.json'),frozen=external.receipt(path/'frozen-barrier.json'),
        summary=external.receipt(path/'summary.json'),comparison=external.receipt(path/'provider-comparison.json'),
        controller_precision=external.receipt(precision) if precision.exists() else None,
        controller_precision_status=controller_status,controller_review=external.receipt(review) if review.exists() else None,
        response_scope='requested-plan completed-response forecast only; physical recovery tau is not identified',
        declared_cases=8,burst_passed=original['burst_mean_qualified_cases'],first_passed=original['first_response_accuracy_cases'],
        providers={p:{k:r[k] for k in ('burst_mean_qualified_cases','first_response_accuracy_cases')} for p,r in comparison['providers'].items()},
        figure=name+'.png',artifacts=artifacts,physical_cause_identified=False,raw_cv_is_prediction_gate=False)


def external_plots(path,out,plt):
    path=Path(path).resolve();m=external.read(path/'manifest.json');external.check(m)
    evaluation=external.read(path/'evaluation.json');predictions=external.read(path/'predictions-frozen.json')
    barrier=external.read(path/'frozen-barrier.json');external.require(external.sha(path/'manifest.json')==barrier['manifest']['sha256'],'external report plan changed')
    for name,sha in barrier['files'].items():external.require(external.sha(path/name)==sha,'external report frozen artifact changed')
    executions=external.read(path/'heldout-executions.json');plots=[]
    for workload in external.WORKLOADS:
        fig,axes=plt.subplots(2,2,figsize=(9.2,5.7))
        for ax,scenario in zip(axes.flat,external.PLANS):
            selected=[e for e in executions if e['job']['workload']==workload and e['job']['scenario']==scenario]
            for process,e in enumerate(selected):
                if e['status']!='completed':continue
                external.require(external.sha(e['native']['path'])==e['native']['sha256'],'external report raw changed')
                n=external.read(e['native']['path']);epoch=n['target_expected_origin_tick'];hz=n['clock']['frequency_hz']
                xs=[((b['begin_tick']+b['end_tick'])/2-epoch)/hz for b in n['blocks']]
                ys=[b['wall_ns']/16/1000 for b in n['blocks']]
                ax.plot(xs,ys,color='#9A9A9A',alpha=.65,linewidth=.9,label='Measured process '+str(process+1))
            # Curves use each provider's own predicted time axis. Observed block
            # starts are never attached to a model curve, even for illustration.
            for provider,rows in predictions['providers'].items():
                if not selected:continue
                job=selected[0]['job'];blocks=rows[job['id']]['block_ns'];cumulative=0;xs=[]
                for value in blocks:xs.append((cumulative+value/2)/1e9);cumulative+=value
                ax.plot(xs,[v/16/1000 for v in blocks],color=COLORS[provider],linestyle='--' if provider=='static' else '-',label=provider.title())
            if selected:
                request=selected[0]['job']['request']
                for device,color in (('cpu','#DC9C4B'),('gpu','#4489A6')):
                    for first,last in request[device+'_actor']['intervals_ms']:
                        ax.axvspan(max(0,first/1000),last/1000,color=color,alpha=.045)
            ax.set_title(scenario.replace('_',' ').title());ax.set_xlabel('Time from planned epoch (s)');ax.set_ylabel('Response per forward (µs)');ax.grid(alpha=.18)
        axes.flat[0].legend(loc='best',ncol=2);fig.suptitle(workload.replace('_',' ').title()+' — fixed-work block responses')
        fig.tight_layout();name=path.name+'-'+workload+'-responses';artifacts=save(fig,out,name);plt.close(fig)
        plots.append(dict(figure=name+'.png',artifacts=artifacts))
    return dict(campaign=str(path),evaluation=external.receipt(path/'evaluation.json'),frozen=external.receipt(path/'frozen-barrier.json'),
        primary_provider=evaluation['primary_provider'],providers={p:dict(passed=r['passed'],declared=r['declared']) for p,r in evaluation['providers'].items()},
        unsupported_providers=predictions.get('unsupported_providers',{}),figures=plots,
        own_predicted_timeline_used=True,whole_target_used_for_fit=False,
        scope=evaluation['claim_scope'],raw_cv_is_prediction_gate=False)


def controller_plot(paths,out,plt):
    fig,axes=plt.subplots(1,2,figsize=(8.2,3.2));observations=[]
    for path in paths:
        path=Path(path);summary=external.read(path/'summary.json')
        for ax,memory in zip(axes,('pageable','registered')):
            selected=[r for r in summary['process_conditions'] if r['status']=='completed' and r['job']['memory_mode']==memory]
            idles=sorted({r['requested_idle_ms'] for r in selected});medians=[];lows=[];highs=[]
            for idle in idles:
                values=sorted(v/1000000 for r in selected if r['requested_idle_ms']==idle for v in r['actual_idle_ns'])
                median=statistics.median(values);low=values[math.floor(.1*(len(values)-1))];high=values[math.ceil(.9*(len(values)-1))]
                medians.append(median);lows.append(median-low);highs.append(high-median)
                observations.append(dict(campaign=path.name,memory_mode=memory,requested_idle_ms=idle,
                    actual_median_ms=median,actual_p10_ms=low,actual_p90_ms=high,all_rows_retained=True))
            ax.errorbar(idles,medians,yerr=[lows,highs],marker='o',markersize=3,capsize=2,label=path.name)
            if idles:ax.plot([min(idles),max(idles)],[min(idles),max(idles)],color='#666666',linestyle=':',linewidth=.8)
            ax.set_title(memory.title()+' host memory');ax.set_xlabel('Requested idle (ms)');ax.set_ylabel('Actual idle (ms)');ax.grid(alpha=.18)
    axes[0].legend(loc='best');fig.tight_layout();name='idle-controller-actuation';artifacts=save(fig,out,name);plt.close(fig)
    return dict(figure=name+'.png',artifacts=artifacts,observations=observations,intervals='median and10th–90th percentiles of achieved idle, all independent processes')


def report(out,idles,external_path=None):
    out=Path(out).resolve();external.require(not out.exists(),'new report output directory required');out.mkdir(parents=True)
    plt,font=figure_setup();idle=[idle_plot(path,out,plt) for path in idles]
    controllers=controller_plot(idles,out,plt)
    competition=None
    if external_path:
        external_path=Path(external_path)
        if (external_path/'evaluation.json').is_file():competition=external_plots(external_path,out,plt)
        elif (external_path/'coverage-failure.json').is_file():
            from profiling.dynamic_validation.coverage_failure_plot import plot
            competition=plot(external_path/'coverage-failure.json',out)
        else:raise ValueError('external campaign has neither evaluated results nor a sealed coverage-failure record')
    sources=[external.receipt(Path(__file__))]
    if competition and competition.get('status')=='coverage_failed':
        sources.append(external.receipt(Path(__file__).with_name('coverage_failure_plot.py')))
    result=dict(schema='dynamic-response-report-1',font=font,idle=idle,controllers=controllers,external=competition,
        fitting_performed=False,new_measurements_performed=False,raw_failed_cases_preserved=True,
        sources=sources)
    external.write(out/'REPORT_DATA.json',result)
    escape=lambda value:html.escape(str(value))
    sections=['<h1>Dynamic-resource response validation</h1><p>Predictions use predeclared requested timelines and their own recursively predicted completion times. The unit is an inclusive completed invocation; actor work is not physical available FLOPs. Raw variability and prediction accuracy are reported separately.</p>']
    sections.append('<h2>Requested and achieved controller timing</h2><img src="'+controllers['figure']+'"><p>Points show achieved median idle and10th–90th percentiles, including all three processes. The dotted line is exact actuation. Actual completed idle is retained as a controller-adherence diagnostic; it is not inserted into whole-ahead forecasts.</p>')
    for r in idle:
        rows=''.join('<tr><td>'+escape(p)+'</td><td>'+str(v['burst_mean_qualified_cases'])+'/8</td><td>'+str(v['first_response_accuracy_cases'])+'/8</td></tr>' for p,v in r['providers'].items())
        sections.append('<h2>'+escape(Path(r['campaign']).name)+'</h2><table><tr><th>Frozen provider</th><th>Burst mean and tail gates</th><th>First-response mean gate</th></tr>'+rows+'</table><p>Each case requires all three independent processes. Burst gates: process mean APE ≤10% and burst APE P90 ≤20%; first-response mean APE ≤10%. A raw CV threshold is not an accuracy gate.</p><img src="'+r['figure']+'"><p>All original processes and unsuccessful outcomes remain in the evidence. The scalar response proxy does not identify the underlying physical cause of slow regimes.</p>')
        if r['controller_precision_status']:
            for split,precision in r['controller_precision_status'].items():
                sections.append('<p>Controller precision ('+split+'): '+str(precision['qualified_processes'])+'/'+str(precision['declared'])+' processes qualified under the original median-overshoot ≤250 µs and P90-overshoot ≤1 ms limits. All slow waits remain in the data; these precision gates were not relaxed.</p>')
            sections.append('<p><b>Interpretation boundary:</b> this cohort evaluates requested-plan completed-response forecasts. It does not identify a physical recovery time constant, even though an owned high-resolution timer was used.</p>')
    if competition:
        if competition.get('status')=='coverage_failed':
            failure=external.read(competition['coverage_failure']['path'])
            sections.append('<h2>External competition: profile support gate failed</h2><p>'+escape(competition['scope'])+'</p>')
            sections.append('<p>Profile processes completed: '+str(failure['completed_profile_processes'])+'/'+str(failure['declared_profile_processes'])+
                '. Exact response cells supported: '+str(failure['supported_response_cells'])+'/'+str(failure['declared_response_cells'])+
                '; complete four-state semantic coordinates: '+str(failure['fully_supported_semantics'])+'/'+str(failure['semantic_coordinates'])+'.</p>')
            sections.append('<p>GPU Attention has no completed steady joint-state calibration response. Hybrid exact-state coverage is complete; the whole comparison stopped at its preregistered support gate. No model was fitted, no predictions were frozen, and none of the '+str(failure['declared_heldout_processes'])+' independent target predictions was generated or evaluated. These are not numeric prediction failures.</p>')
            sections.append('<p>'+str(failure['overlapping_calls'])+' Attention calls overlap planned joint intervals: '+str(failure['release_spanning_calls'])+' span a CPU release, and '+str(failure['marginal_start_overlap_calls'])+' marginal call overlaps the joint start. All are retained, including the call starting just before GPU activation. Actual actor starts are diagnostic markers only.</p>')
            for f in competition['figures']:sections.append('<img src="'+f['figure']+'">')
            sections.append('<p>Bars are inclusive completed Executor.action responses. Cross-boundary responses are not steady joint-state prices or pure GPU kernel times. Completion following CPU OFF is a temporal association; these observations do not identify a deadlock, operating-system mechanism or physical cause. Missing costs remain unknown.</p>')
        else:
            rows=''.join('<tr><td>'+escape(p)+'</td><td>'+str(v['passed'])+'/'+str(v['declared'])+'</td></tr>' for p,v in competition['providers'].items())
            sections.append('<h2>Continuous Dense execution under owned actor plans</h2><p>'+escape(competition['scope'])+'</p><table><tr><th>Frozen provider</th><th>Independent process gates</th></tr>'+rows+'</table>')
            for provider,status in competition['unsupported_providers'].items():sections.append('<p>'+escape(provider)+' unsupported: '+escape('; '.join(status.get('reasons',[])))+'</p>')
            for f in competition['figures']:sections.append('<img src="'+f['figure']+'">')
            sections.append('<p>Shading shows requested actor pulses. Model curves use model-owned time axes; measured curves use their separately recorded clocks. No curve is re-anchored to future target boundaries. Process gates: total APE ≤10% and fixed-work block APE P90 ≤20%.</p>')
    document='<!doctype html><meta charset="utf-8"><title>Dynamic-resource response validation</title><style>body{max-width:1100px;margin:36px auto;padding:0 24px;color:#162B3D;font:17px/1.6 Georgia,serif}h1,h2{color:#173D63}table{border-collapse:collapse}td,th{border:1px solid #cad3db;padding:8px 16px;text-align:left}img{display:block;max-width:100%;margin:24px auto}p{max-width:1000px}</style>'+''.join(sections)
    (out/'REPORT_en.html').write_text(document,encoding='utf-8',newline='\n')
    return dict(html=str(out/'REPORT_en.html'),data=str(out/'REPORT_DATA.json'),font=font)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--idle',action='append',required=True)
    p.add_argument('--external');a=p.parse_args();print(json.dumps(report(a.out,a.idle,a.external),ensure_ascii=False),flush=True)
