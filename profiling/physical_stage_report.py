"""Render frozen, unselected physical calibration comparisons as a local report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from fractions import Fraction


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def q90(values):
    values=sorted(values)
    index=(len(values)-1)*.9
    left=int(index)
    return values[left]+(values[min(left+1,len(values)-1)]-values[left])*(index-left)


def error_table(evaluation):
    lines=['| Frozen candidate | Calls | Median APE | P90 APE | Worst APE |',
           '|---|---:|---:|---:|---:|']
    for candidate in ('idle_component_baseline','condition_component_baseline',
                      'reference_activity_candidate','reference_completed_gpu_candidate','observer_bare_idle_baseline'):
        errors=[]
        for run in evaluation['runs']:
            prediction=run['predictions'].get(candidate,{})
            if not prediction.get('known'):continue
            predicted=prediction['predicted_ns']
            errors.extend(abs(predicted-actual)/actual for actual in run['observed_ns'])
        lines.append(f'| {candidate} | {len(errors)} | '+
                     ('Unknown | Unknown | Unknown |' if not errors else
                      f'{statistics.median(errors):.2%} | {q90(errors):.2%} | {max(errors):.2%} |'))
    return '\n'.join(lines)


def state_table(evaluation):
    lines=['| State | Candidate | Processes | Median APE | P90 APE | Worst APE |',
           '|---|---|---:|---:|---:|---:|']
    for state in ('idle','cpu','gpu','both'):
        for row in evaluation['summaries']:
            if row['condition']!=state:continue
            if row['candidate']=='observer_bare_idle_baseline':continue
            s=row['per_call_relative_error']
            lines.append(f"| {state} | {row['candidate']} | {row['processes']} | {s['median']:.2%} | {s['p90']:.2%} | {s['worst']:.2%} |")
    return '\n'.join(lines)


def physical_table(evaluation):
    def number(value):
        return Fraction(value['numerator'],value['denominator']) if isinstance(value,dict) else Fraction(value)
    lines=['| State | Kernel activity (ms) | Copy outside kernel (ms) | API outside device (ms) | Other host + gaps (ms) | Whole-call mean (ms) |',
           '|---|---:|---:|---:|---:|---:|']
    for state in ('idle','cpu','gpu','both'):
        rows=[r['mean_components_ns'] for r in evaluation['physical_breakdown'] if r['condition']==state]
        if not rows:continue
        means={k:sum(number(r[k]) for r in rows)/len(rows) for k in rows[0]}
        kernel=means['kernel_activity'];copy=means['copy_activity_outside_kernel'];api=means['host_api_outside_device']
        total=sum(means.values());other=total-kernel-copy-api
        lines.append('| '+state+' | '+' | '.join(f'{float(x)/1e6:.4f}' for x in (kernel,copy,api,other,total))+' |')
    return '\n'.join(lines)


def runtime_table(root):
    summary=read(root/'cuda-waits/summary.json')
    lines=['| Profile-0 state | PV host span (ms) | Launch API union (ms) | Sync API union (ms) | Other runtime API union (ms) |',
           '|---|---:|---:|---:|---:|']
    for state in ('idle','cpu','gpu','both'):
        job=next(j for j in summary['jobs'] if j['job_id']=='profile-0-'+state)
        m=next(a['metrics'] for a in job['per_action'] if a['action_index']==14)
        fields=('per_action_host_span','per_action_launch_API_union','per_action_sync_API_union','per_action_other_runtime_API_union')
        lines.append('| '+state+' | '+' | '.join(f"{m[k]['mean_ns']/1e6:.4f}" for k in fields)+' |')
    return '\n'.join(lines)


def render(root):
    root=Path(root)
    frozen=read(root/'frozen-components.json')
    traced=read(root/'heldout-evaluation.json')
    bare=read(root/'heldout-bare-evaluation.json')
    barrier=read(root/'frozen-barrier.json')
    lines=[
        '# Physical-stage calibration and fresh validation — 2026-10-22',
        '',
        '**Conclusion: this is a completed diagnostic calibration/validation cycle, not a qualified instantaneous-availability model.** '
        'CUPTI materially changes the small workload. Before/after reference tasks provide stale workload response estimates; '
        'they do not identify a neutral, continuous resource-availability trajectory. All errors below remain diagnostic.',
        '',
        'For the predeclared completed-GPU-response hypothesis, bare-target median APE changes from 29.00% (idle baseline) to 19.59%, '
        'but P90 worsens from 51.76% to 64.57%. Traced median/P90 are 12.38%/31.64%. This is a typical-call improvement, '
        'not an overall accuracy pass. A fresh bare CPU-contention call took 36.1308 ms while the frozen prediction was 3.904765 ms (89.19% error). '
        'All 24 heldout processes have CV above 10%; high-latency calls remain in every reported metric.',
        '',
        '## Workload and independence',
        '',
        '- One H=128, q=16, history=0 F32 dense layer; CPU/GPU partition, attention, FFN and transfers; one CPU thread on a declared core. '
        'This is a small controlled layer, not an accuracy result for a production-size LLM.',
        '- Four offered-load conditions: idle, CPU replay, GPU replay, joint replay. These labels are not observed available-resource percentages.',
        '- Profile: 12 fresh main processes, 32 full calls each. Observer controls: 9 processes, 32 calls each. '
        'Traced heldout: 12 processes / 384 calls. Bare-target heldout: 12 additional processes / 384 calls. '
        'Reference probes remain CUPTI-instrumented in the bare-target cohort and execute outside target timing.',
        '- After two post-target constructor failures, remaining collection explicitly stops owned background actors immediately after main sampling and before after-reference construction. '
        'BEFORE and main work are unchanged. These AFTER references are unloaded recovery diagnostics, not same-load drift observations; the frozen parameter file remains unchanged. ',
        '- All five candidates were declared before heldout. Two reference-response hypotheses use only completed BEFORE references. '
        'AFTER references audit drift and never feed prediction. Different heldout numeric seeds and process identities are enforced.',
        '- Two attempts (one profile, one heldout) completed main timing but their after-reference constructors timed out. '
        'The failed attempts are retained and excluded as units; infrastructure replacements with the same configuration are explicitly indexed. '
        'A prior two-call smoke export failure is also retained outside this campaign.',
        f"- Frozen parameter SHA256: `{barrier['frozen_file_sha256']}`; saved barrier time: `{barrier['frozen_at_utc']}`.",
        '- No DNN, heldout fitting, whole-layer residual offset, observer-overhead subtraction or rate clipping is used. '
        'Parameter means come from profiling; scoped component response and serial execution are calculated in modeling using the common engine.',
        '',
        '## What the candidates mean',
        '',
        '| Candidate | Frozen inputs and rule | Limitation |',
        '|---|---|---|',
        '| idle_component_baseline | Sum of disjoint per-action component means from idle profile processes | No response to resource changes |',
        '| condition_component_baseline | Component means for each known offered-load label | Label-conditioned control; not a general availability model |',
        '| reference_activity_candidate | Idle kernel activity scaled by BEFORE GPU reference activity rate; host components by completed BEFORE CPU reference response | Host remainder is not active CPU work; references are stale and workload dependent |',
        '| reference_completed_gpu_candidate | Whole GPU COMPUTE host response scaled by completed BEFORE GPU reference response; host CPU components by completed BEFORE CPU reference response | Tests driver/queue response, without identifying its physical mechanism |',
        '| observer_bare_idle_baseline | Original per-action inclusive means from the three bare idle controls | Untraced idle control only; no contention model |',
        '',
        'Activity, API, and host columns overlap in raw CUPTI. The estimator instead forms an explicit disjoint partition: '
        'kernel activity, copy outside kernel, API outside device activity, remaining host wall, inter-action gaps, and tail. '
        'Their sum is checked against every original whole-call duration. Component means preserve mean accounting; a sum of marginal medians is not used.',
        '',
        '## Fresh traced heldout',
        '',
        error_table(traced),
        '',
        '## Fresh bare-target heldout',
        '',
        error_table(bare),
        '',
        'APE = |prediction − measured whole-call time| / measured whole-call time. '
        'Each process contributes all 32 calls; no high-latency calls are discarded. '
        'Global percentiles are calculated across calls, not by averaging state percentiles. '
        'Traced and bare errors describe different timing protocols and must not be silently pooled.',
        '',
        '## Per-state traced errors',
        '',
        state_table(traced),
        '',
        '## Per-state bare-target errors',
        '',
        state_table(bare),
        '',
        '## Measured physical accounting on traced heldout',
        '',
        physical_table(traced),
        '',
        'These are descriptive measured envelopes. API outside device activity is not identified active CPU work. '
        'Disjoint columns sum to whole-call mean; they are not a decomposition into independently additive physical service demands.',
        '',
        '## Backend call multiplicity and launch-to-start timing',
        '',
        'In this executor/shape, the Attention PV action (original action 14) issues 52 cudaLaunchKernel calls, '
        '33 cudaMemcpyAsync calls and 2 cudaStreamSynchronize calls per invocation. The idle profile-0 mean PV host span is about 1.91 ms; '
        'launch APIs account for about 1.06 ms and memcpy APIs about 0.67 ms. This describes this concrete GGML execution path, '
        'not a universal Attention implementation or a fused production kernel.',
        '',
        runtime_table(root),
        '',
        'The exact-correlation diagnostics find GPU-COMPUTE per-kernel launch-return-to-start median gaps of approximately '
        '0.130/0.083/143.655/140.853 microseconds for idle/CPU/GPU/joint profile-0 conditions. These overlapping gaps must not be summed '
        'as serial costs. They establish delayed starts, not whether the cause is hardware admission, dependencies, driver behavior or OS scheduling. '
        'Kernel envelopes alone miss this behavior. The measured inclusive action cost already contains its launches and waits; '
        'adding another launch fee to that cost would double-count. A future physical lowering must replace inclusive terms with an explicit '
        'submission/admission/device/completion dependency graph, with independently identified parameters.',
        '',
        '## Qualification and remaining identification gaps',
        '',
        f"The frozen observer check reports mean call overhead of {frozen['observer']['relative_observer_overheads']['cupti']:.2%} "
        f"for CUPTI and {frozen['observer']['relative_observer_overheads']['cupti_references']:.2%} for CUPTI plus references, "
        'against a predeclared 5% neutrality gate. This gate fails. Paired per-process medians and actual background coverage are retained in `campaign-audit.json`.',
        '',
        '- Low point error alone does not certify accuracy. Required checks also include independent processes, full coverage, numerical correctness, '
        'capture integrity, CV <= 10%, observer neutrality, and identified physical service. No candidate passes the complete hardware qualification.',
        '- The CPU batch counter is quantized and includes query brackets. It can exceed the measured batch wall. '
        'The strict supply adapter therefore leaves CPU physical service unknown; it never clips to 100%. '
        'Completed CPU reference throughput is a separate, explicitly scoped diagnostic response measurement.',
        '- A CUPTI kernel interval is a device activity envelope, not proof of SM-active or preemption-free service. '
        'Known GEMM work divided by that union gives reference-workload effective throughput, not universal available FLOP/s.',
        '- References are outside the target interval. CPU and GPU sensors have different staleness; after-reference constructors and warmups '
        'also change device history. Within-call transitions and immediate recovery remain unidentified. '
        'The provided sample-held trajectory must retain unknown regions and its observed_diagnostic source kind.',
        '- The analyzer trusts collection numerical/owned-process receipts. The independent reference adapter cross-checks oracle/raw hashes, '
        'physical process identity, native action identity, common clock and exact CUPTI correlation; its outputs are retained under `reference-adaptations/`.',
        '- These data do not establish storage-bandwidth response, memory-pressure accuracy, another shape/dtype/CPU-thread count, a multi-layer model, '
        'or an unseen fluctuating workload. Previous campaign percentages use a different protocol and cannot be cited as a before/after improvement here.',
        '',
        '## Next mechanisms to identify',
        '',
        '1. Reduce and independently quantify tracing cost. Use sampled/separate diagnostic captures for physical explanation and '
        'bare whole-call runs for final accuracy; do not subtract a global profiler constant.',
        '2. Measure CPU active runtime work separately from blocked/ready waiting at sufficient resolution. '
        'Use profile-only high-resolution scheduling evidence to identify parameters; do not inject the heldout target execution trace as its own external resource input.',
        '3. Separate CUDA submission CPU work, device admission/queue delay, device execution and completion notification. '
        'GPU kernel-throughput corrections alone cannot account for dominant host/API waiting.',
        '4. Validate reference observability, cadence, perturbation and hold validity with independent transitions before claiming a continuous r_i(t). '
        'Then freeze again and use a new heldout cohort; the current heldout must remain a final comparison for this frozen version.',
        '',
        '## Evidence and reproduction',
        '',
        'The independent campaign audit admits all 46 effective jobs, verifies 238 successful owned-process exits, and retains both failed attempts '
        '(all their processes exited, including the two forced constructor exits). It verifies 19 post-target cleanup chronologies and actual '
        'background coverage of every main call. The strict adapter additionally cross-checks 238 actor/program/clock/raw-file bindings, '
        '238 saved independent numerical receipts covering 8,330 ports, and 46 program/oracle bindings. '
        'All 78 GPU reference samples are available as scoped diagnostics; all 78 CPU physical-service samples remain unknown. '
        'Software regression: 729 tests executed successfully, with 7 declared skips; this is not a hardware-accuracy assertion.',
        '',
        'All JSON/CSV outputs use LF and are write-once. Raw data, failed attempts, source snapshots, binary build receipts, '
        'process ownership/cleanup, oracle receipts, frozen parameters and both heldout cohorts remain local under this campaign directory. '
        'They are not library prerequisites and generated results are ignored by Git.',
        '',
        'See `docs/PHYSICAL_STAGE_MEASUREMENT.md` for reusable collection/build commands. '
        'Use `physical_stage_analysis.py` to repeat the frozen comparisons and `physical_campaign_audit.py` for independent evidence checks.',
        '',
    ]
    return '\n'.join(lines)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('campaign',type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    with a.output.open('x',encoding='utf-8',newline='\n') as f:f.write(render(a.campaign))
