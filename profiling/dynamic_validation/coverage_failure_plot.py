"""Report-only figure layout for an immutable profile coverage-failure record.

The extractor and sealed record remain unchanged. This renderer preserves all
coordinates and moves only the marginal-overlap label above its marker.
"""
from pathlib import Path
from profiling.dynamic_validation import external


def plot(record_path,out):
    """Plot only the sealed failure record; never reopen measurement outcomes."""
    from profiling.dynamic_validation.report import figure_setup,save
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    record_path=Path(record_path).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    record=external.read(record_path);external.require(record['schema']=='dynamic-external-coverage-failure-1','coverage failure schema differs')
    plt,font=figure_setup();fig,axes=plt.subplots(len(record['processes']),1,figsize=(10.0,7.0),sharex=True,squeeze=False)
    for ax,process in zip(axes.flat,record['processes']):
        for interval in process['planned_joint_intervals']:ax.axvspan(max(-2.2,interval['begin_ns']/1e9),interval['end_ns']/1e9,color='#DFE5EC',alpha=.6)
        for device,y,color in (('cpu',3,'#B1764D'),('gpu',2,'#387CA0')):
            for interval in process['actors'][device]:
                a,b=interval['planned_begin_ns']/1e9,interval['planned_end_ns']/1e9
                ax.broken_barh([(a,b-a)],(y-.20,.40),facecolors=color)
                ax.plot(interval['actual_first_relative_ns']/1e9,y,marker='v',color='#111111',markersize=4)
                if device=='cpu':ax.axvline(b,color='#555555',linestyle=':',linewidth=.75)
        for call in (r for r in record['affected_calls'] if r['job_id']==process['job_id']):
            a,b=call['begin_relative_ns']/1e9,call['end_relative_ns']/1e9
            if call['release_spanning']:
                ax.broken_barh([(a,b-a)],(.80,.40),facecolors='#A94442')
                ax.plot(b,1,marker='o',color='#6E1D1D',markersize=3)
                ax.text((a+b)/2,1,f"{call['wall_ns']/1e9:.3f} s",ha='center',va='center',color='white',fontsize=8)
            else:
                ax.plot(b,1,marker='D',color='#6E569A',markersize=4)
                ax.annotate(f"{sum(r['overlap_ns'] for r in call['joint_intersections'])/1e6:.3f} ms joint overlap",(b,1),xytext=(-45,25),textcoords='offset points',fontsize=8,
                    arrowprops=dict(arrowstyle='-',linewidth=.5,color='#6E569A'))
        ax.set_yticks([1,2,3],['Attention response','GPU offered (planned)','CPU offered (planned)'])
        ax.set_ylim(.35,3.55);ax.set_title('Profile round '+str(process['round'])+' — observed inclusive GPU Attention completion')
        ax.grid(axis='x',alpha=.18)
    axes[-1,0].set_xlim(-2.2,17.2);axes[-1,0].set_xticks([-2,0,2,4,6,8,10,12,14,16]);axes[-1,0].set_xlabel('Time from planned target epoch (s)')
    handles=[Patch(color='#B1764D',label='Planned CPU pulse'),Patch(color='#387CA0',label='Planned GPU pulse'),
        Patch(color='#A94442',label='Release-spanning response'),Line2D([0],[0],marker='D',color='#6E569A',linestyle='None',label='Marginal boundary overlap'),
        Line2D([0],[0],marker='v',color='#111111',linestyle='None',label='Actual actor first call')]
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.55,-.01),fontsize=8)
    fig.tight_layout(rect=(0,.08,1,1));name='external-coverage-failure-timeline';artifacts=save(fig,out,name);plt.close(fig)
    figures=[dict(figure=name+'.png',artifacts=artifacts)]
    selected=[r for r in record['affected_calls'] if r['release_spanning']]
    fig,ax=plt.subplots(figsize=(7.5,3.5))
    for index,call in enumerate(selected):
        interval=next(i for i in call['joint_intersections'] if i['spans_planned_cpu_off'])
        completion=interval['completion_minus_planned_cpu_off_ns']/1e6
        actual_stop=completion-interval['completion_minus_actual_cpu_last_completion_ns']/1e6
        ax.plot([actual_stop,completion],[index,index],color='#BFC8D0',linewidth=1)
        ax.plot(actual_stop,index,marker='x',color='#222222',label='Actual CPU last completion' if index==0 else None)
        ax.plot(completion,index,marker='o',color='#A94442',label='Attention completion' if index==0 else None)
    ax.axvline(0,color='#555555',linestyle=':',linewidth=.9)
    ax.set_yticks(range(len(selected)),['Round '+str(r['round'])+' / replay '+str(r['global_replay']) for r in selected]);ax.invert_yaxis()
    ax.set_xlabel('Completion offset from planned CPU OFF (ms)');ax.set_title('Six release-spanning responses; temporal association, not causal identification')
    ax.grid(axis='x',alpha=.18);ax.legend(loc='best');fig.tight_layout()
    name='external-coverage-release-offsets';artifacts=save(fig,out,name);plt.close(fig);figures.append(dict(figure=name+'.png',artifacts=artifacts))
    return dict(status='coverage_failed',coverage_failure=external.receipt(record_path),providers={},unsupported_providers={},figures=figures,
        scope=record['scope'],font=font)
