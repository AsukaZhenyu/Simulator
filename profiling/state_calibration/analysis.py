"""Describe retained state experiments without fitting, filtering or attribution claims."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def render(root,output):
    root=Path(root).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('new report directory required')
    manifest=read(root/'manifest.json');rows=[]
    for split in ('profile','heldout','diagnostic'):
        path=root/(split+'-rows.json')
        if path.exists():rows.extend(read(path))
    report=read(root/'report.json')if(root/'report.json').exists()else None
    controls=[]
    for mode in ('isolated','mixed'):
        for feature in ('telemetry','instrumentation'):
            for repeat in sorted({r['job']['repeat']for r in rows if r['job']['split']=='diagnostic'}):
                cohort=[r for r in rows if r['job']['split']=='diagnostic'and r['job']['repeat']==repeat and r['job']['mode']==mode and r.get('status')=='measured']
                if feature=='telemetry':
                    a=[r for r in cohort if not r['job']['telemetry']and r['job']['instrumentation']=='bare'];b=[r for r in cohort if r['job']['telemetry']and r['job']['instrumentation']=='bare']
                else:
                    a=[r for r in cohort if not r['job']['telemetry']and r['job']['instrumentation']=='bare'];b=[r for r in cohort if not r['job']['telemetry']and r['job']['instrumentation']=='stamped']
                if len(a)==len(b)==1:
                    controls.append(dict(mode=mode,repeat=repeat,feature=feature,ratio=b[0]['summary']['iteration_median_ns']/a[0]['summary']['iteration_median_ns'],
                        baseline_job=a[0]['job']['id'],observed_job=b[0]['job']['id'],scope='descriptive matched-cohort median ratio, independent processes; no overhead subtraction'))
    compact=[];boundary_diagnostics=[]
    for row in rows:
        summary={k:v for k,v in row.get('summary',{}).items()if not k.endswith('_samples_ns')}
        compact.append(dict(job=row['job'],status=row.get('status'),summary=summary,error=row.get('error'),
            affinity=row.get('affinity'),background=row.get('background')))
        if row['job']['id'].startswith('boundary-')and row.get('status')=='measured':
            native=read(root/'jobs'/row['job']['id']/'native.json')
            spans=[d for block in native['blocks']for d in block.get('target_diagnostics',[])]
            def median_field(key):
                values=[d[key]for d in spans if d.get(key)is not None]
                return statistics.median(values)if values else None
            boundary_diagnostics.append(dict(job_id=row['job']['id'],mode=row['job']['mode'],repeat=row['job']['repeat'],
                execution_boundary=row['job']['execution_boundary'],target_median_ns=summary.get('target_median_ns'),
                target_cv=summary.get('target_cv'),host_enqueue_median_ns=median_field('host_enqueue_ns'),
                host_sync_median_ns=median_field('host_sync_ns'),stream_span_median_ns=median_field('stream_span_ns'),
                observed_sm_clock_mhz=summary.get('device_state',{}).get('gpu.sm_clock_mhz'),
                scope='CUDA stream span includes submission gaps, not pure kernel activity; host/stream spans are not additive'))
    doc=dict(schema='state-calibration-description-0.1',campaign_manifest_sha256=sha(root/'manifest.json'),
        jobs=compact,controls=controls,boundary_diagnostics=boundary_diagnostics,evaluation=report,all_samples_retained=True,heldout_used_for_fitting=False,
        device_state_causal_attribution_proven=False,universal_device_multiplier_calibrated=False,
        native_capacity_proven=False,unknown_private_cuda_workspace_bytes=None,
        scope='Two previously failing F32 Attention PV shapes; fixed serial high-level actions; declared owned replay interference; no clock/power changes',
        sources={p.name:sha(p)for p in [root/'manifest.json',root/'profile-rows.json',root/'heldout-rows.json',root/'diagnostic-rows.json',root/'frozen-model.json',root/'report.json']if p.exists()})
    lines=['# State-aligned calibration experiment','',
        'This report separates measured state association, controlled execution regimes, and a causal explanation of the earlier error. It does not assume they are equivalent. No samples are removed and no measurement overhead is subtracted.','',
        f"Declared primary native jobs: {sum(j['split']!='diagnostic'for j in manifest['jobs'])}; retained primary jobs: {sum(r['job']['split']!='diagnostic'for r in rows)}; diagnostic controls: {sum(r['job']['split']=='diagnostic'for r in rows)}.",'',
        'The same profile numerical matrices are reused across conditions. Fresh heldout processes use disjoint numerical seeds. Models freeze before any heldout. Every process retains all 256 invocations per block; block medians are not substituted for invocation CV.','',
        '| Split | Case | Mode | Condition | Target ms | Whole invocation ms | Target CV | SM MHz | CPU busy % | Numerical | Aligned block quality |',
        '|---|---|---|---|---:|---:|---:|---:|---:|---|---|']
    def number(value,scale=1):return 'unknown'if value is None else f'{value/scale:.4f}'
    for r in compact:
        j,s=r['job'],r['summary'];state=s.get('device_state',{})
        if j['split']=='diagnostic':continue
        lines.append(f"| {j['split']} | {j['case_id']} | {j['mode']} | {j['condition']} | {number(s.get('target_median_ns'),1e6)} | {number(s.get('iteration_median_ns'),1e6)} | {number(s.get('target_cv'))} | {number(state.get('gpu.sm_clock_mhz'))} | {number(state.get('cpu.busy_percent'))} | {s.get('native_numeric_passed',False)} | {s.get('alignment_model_ready',False)} |")
    lines+=['','## Profile-only cost models','',
        'The constant model pools all observed named conditions within the exact operator/execution context. The conditional model separates the same profile data by the verified controlled condition. An unseen context or condition is unknown; neither model interpolates or derives throughput from utilization. Exported service-rate ratios apply only to this exact operation/context.','']
    if report:
        lines+=['| Model | Cells | Known errors | Median error | Worst error | Fully accepted |','|---|---:|---:|---:|---:|---:|']
        for family,data in report['families'].items():
            lines.append(f"| {family} | {data['cell_count']} | {data['known_error_count']} | {number(data['median_relative_error_known_only'])} | {number(data['worst_relative_error_known_only'])} | {data['accepted_count']} |")
        lines+=['','Errors are fractions (0.20 = 20%). A lower error does not override failed numerical, sample coverage, source/target stability or telemetry alignment gates.','',
                '| Model | Case | Mode | Condition | Metric | Relative error | Accepted | Failed gates |','|---|---|---|---|---|---:|---|---|']
        for family,data in report['families'].items():
            for c in data['cells']:
                lines.append(f"| {family} | {c['context']['case_id']} | {c['context']['mode']} | {c['condition']} | {c['metric']} | {number(c['relative_error'])} | {c['accepted']} | {', '.join(c['gate_failures'])} |")
    lines+=['','## Observation controls','','| Mode | Feature | Repeat | Observed / baseline invocation median |','|---|---|---:|---:|']
    for c in controls:lines.append(f"| {c['mode']} | {c['feature']} | {c['repeat']} | {c['ratio']:.4f} |")
    lines+=['','These controls are separate fresh processes, so the ratios include process variation. They do not isolate pure sampler/timestamp overhead. Differences remain in the results; no ratio is subtracted from any timing.','',
        '## Submission, synchronization and CUDA stream diagnostics','',
        '| Mode | Boundary | Repeat | Target host ms | Submit host ms | Synchronize host ms | CUDA stream ms | SM MHz |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for b in boundary_diagnostics:
        lines.append(f"| {b['mode']} | {b['execution_boundary']} | {b['repeat']} | {number(b['target_median_ns'],1e6)} | {number(b['host_enqueue_median_ns'],1e6)} | {number(b['host_sync_median_ns'],1e6)} | {number(b['stream_span_median_ns'],1e6)} | {number(b['observed_sm_clock_mhz'])} |")
    lines+=['','Alternate boundaries change only the selected target; other actions retain the original boundary. Legacy internals are unknown, not zero. CUDA stream intervals include gaps in command submission and omit waiting before the start event. They are not pure kernel-active time, and the host/device columns cannot be summed as independent components. Matched asynchronous event-free controls expose event instrumentation effects.','',
        '## What the observations establish','',
        'QPC/CLOCK_MONOTONIC ticks align native intervals and bounded telemetry queries. Individual short actions may contain no valid telemetry query; their state is explicitly unknown. Sustained blocks support descriptive association. NVML utilization is rolling activity including the measured workload, memory utilization is not bandwidth saturation, and power can be a one-second average.','',
        'Controlled background records prove owned replay activity over an invocation envelope; they do not prove continuous GPU occupancy, actual kernel concurrency or a unique explanation involving frequency, WDDM, thermals or caches. Native inputs/outputs are independently checked outside timing.','',
        'On Windows the main process and CPU replay are restricted to logical-CPU mask 3, and the GPU replay host is restricted to mask 12. CPU busy percentage is machine-wide; a low aggregate value does not exclude contention on those selected logical CPUs. Both backgrounds retain a full initialized executor, including CUDA allocations.','',
        'A state-dependent service runtime can replay supplied exact rate changes while preserving remaining work and causal visibility. These experiments do not calibrate one universal CPU/GPU rate for every operator, arbitrary external workloads, quantized inference, SSD, unified memory, physical VRAM limits or full distributed/multiround production inference.']
    output.mkdir(parents=True)
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    (output/'report.json').write_text(json.dumps(doc,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(report=str(output/'REPORT.md'),rows=len(rows),failed=sum(r.get('status')!='measured'for r in rows))))
    return doc

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();render(a.directory,a.output)
