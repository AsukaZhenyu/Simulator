"""Independent complete-program gates for frozen H2D/history predictions.

Only bare complete-program timings determine accuracy. Paired stamped processes
test the observer and supply diagnostics, never replacement ground truth. Failed
native processes remain in declared denominators. No parameter fitting occurs.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


QUALITY = dict(max_process_mean_ape=.10,max_trial_average_p90_ape=.20,
               max_process_trial_cv=.10,max_observer_relative_mean_difference=.05)
PROVIDERS = ('old_hot','context_static','history_conditional')


def require(condition,message):
    if not condition: raise ValueError(message)


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def receipt(path): return dict(path=str(Path(path).resolve()),sha256=sha(path))


def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,separators=(',',':'),allow_nan=False); stream.write('\n')


def csv_new(path,rows):
    require(bool(rows),'empty CSV is not an evidence table')
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def rational(value):
    require(not isinstance(value,bool),'boolean prediction rejected')
    parsed=Fraction(str(value))
    require(parsed>0,'positive predicted response required')
    return parsed


def quantile(values,q):
    values=sorted(values); require(bool(values),'empty quantile')
    require(0<=q<=1,'quantile outside range')
    # Same order-statistic convention as e2a.quantile(method='higher').
    return values[math.ceil((len(values)-1)*q)]


def cv(values):
    return statistics.stdev(values)/statistics.mean(values) if len(values)>1 else None


def numeric_equal(actual,expected):
    if isinstance(expected,list):
        return isinstance(actual,list) and len(actual)==len(expected) and all(numeric_equal(a,b) for a,b in zip(actual,expected))
    if isinstance(actual,(dict,list,bool)) or not isinstance(actual,(int,float)): return False
    return math.isfinite(actual) and math.isfinite(expected) and abs(actual-expected)<=5e-5+5e-5*abs(expected)


def process_record(out,manifest,job,barrier):
    folder=out/'jobs'/job['id']; execution=read(folder/'execution.json')
    require(execution['job']==job,'declared/executed job differs')
    require(execution['barrier']==barrier,'heldout process did not bind frozen barrier')
    require(datetime.fromisoformat(execution['started_at_utc'])>=datetime.fromisoformat(barrier['frozen_at_utc']),
            'heldout process started before parameter freeze')
    match=re.match(r'heldout-r(\d+)-',job['id']); require(match is not None,'heldout round absent')
    result=dict(job_id=job['id'],program_key=job['program_key'],round=int(match.group(1)),
                instrumentation=job['request']['instrumentation'],status=execution['status'],
                execution=receipt(folder/'execution.json'),native=None,process_id=execution.get('process_id'),
                mean_ns=None,trial_cv=None,trial_mean_replay_ns=[],early_mean_ns=None,late_mean_ns=None,
                late_over_early_minus_one=None,numeric_oracle_passed=False,
                native_repeat_consistency_passed=False,error=execution.get('error'))
    if execution['status']!='completed': return result
    raw=folder/'result.json'; require(sha(raw)==execution['native']['sha256'],'native evidence changed')
    native=read(raw); req=job['request']; binding=manifest['programs'][job['program_key']]
    for name in ('program','reference'): require(sha(binding[name]['path'])==binding[name]['sha256'],'input evidence changed')
    program,oracle=read(binding['program']['path']),read(binding['reference']['path'])
    require(execution['process_exited_verified'] and not execution['forced_termination'] and execution['returncode']==0,'native process exit differs')
    require(native['schema']=='h2d-history-native-1' and native['request']==req,'native/request differs')
    require(native['process_id']==execution['process_id'] and native['source_sha256']==req['source_sha256']
            and native['caller_source_sha256_match'],'native source/PID differs')
    require(native['program_sha256']==program['program_sha256'] and native['actions']==program['actions'],'native mathematical action sequence differs')
    require(native['mode']=='full' and native['memory_mode']=='pageable' and native['history_intervention']=='none',
            'heldout target is not unmodified complete pageable Dense')
    require(native['instrumentation']==req['instrumentation'],'native observer type differs')
    affinity=native['main_thread_affinity']; process=native['process_affinity']
    require(affinity['requested_mask']==affinity['observed_mask']==affinity['final_mask']==4 and affinity['restored'],'main thread context differs')
    require(process['requested_mask']==process['observed_mask']==process['final_mask']==4294967295
            and process['enforced'] and process['applied_before_backend_initialization'],'owned process context differs')
    preparation=native['preparation_contract']
    require(preparation['reset_before_batch'] and preparation['untimed_complete_prime_passes']==1
            and preparation['timed_replays_per_batch']==8 and preparation['fixed_input_replay']
            and preparation['blocks_per_trial']==1,'heldout replay preparation differs')
    require(req['sampling']['repeats']==8 and req['sampling']['trials']==16,'declared heldout sample count differs')
    require(native['arena_registration']['restored'] and native['arena_registration']['mode']=='pageable','heldout memory ownership differs')
    proof=native['verification_native']
    numeric=bool(execution.get('full_oracle_passed') and proof['copy_values_equal'] and set(proof['ports'])==set(oracle)
                 and all(numeric_equal(proof['ports'][tid],values) for tid,values in oracle.items()))
    repeated=bool(native['native_repeat_consistency']['passed'])
    require(numeric and repeated,'independent numeric/copy/replay oracle failed')
    require(not native['diagnostic_state_used_as_future_input'] and not native['whole_program_time_used_as_parameter'],
            'native collection used future-target state or parameter fitting')
    trials=native['trials']; require(len(trials)==16,'heldout block count differs')
    hz=native['clock']['frequency_hz']; values=[]
    for index,t in enumerate(trials):
        require(t['trial']==index and t['block']==0 and t['repeats']==8,'heldout chronological block identity differs')
        require(t['end_tick']>t['begin_tick'] and t['wall_ns']>0,'heldout QPC order differs')
        require(abs((t['end_tick']-t['begin_tick'])*1e9/hz-t['wall_ns'])<=1,'heldout QPC ledger differs')
        expected=0 if req['instrumentation']=='bare' else len(program['actions'])*8
        require(len(t['action_intervals'])==expected,'observer invocation count differs')
        values.append(t['wall_ns']/8)
    early,late=statistics.mean(values[:8]),statistics.mean(values[8:])
    result.update(native=receipt(raw),mean_ns=statistics.mean(values),trial_cv=cv(values),
                  trial_mean_replay_ns=values,early_mean_ns=early,late_mean_ns=late,
                  late_over_early_minus_one=late/early-1,numeric_oracle_passed=numeric,
                  native_repeat_consistency_passed=repeated)
    return result


def case_metrics(provider,key,prediction,processes,quality=QUALITY):
    require(len(processes)==8,'case must retain all eight declared native processes')
    primary=[r for r in processes if r['instrumentation']=='bare']
    observer=[r for r in processes if r['instrumentation']=='stamped']
    require(len(primary)==len(observer)==4,'case primary/observer denominator differs')
    expected=[float(rational(v)) for v in prediction['trial_mean_replay_ns']]
    require(len(expected)==16,'sixteen independently frozen block predictions required')
    predicted_mean=float(rational(prediction['mean_replay_ns']))
    require(abs(statistics.mean(expected)-predicted_mean)<=max(1e-6,predicted_mean*1e-12),
            'frozen process prediction differs from its block mean')
    process_rows=[]; blocks=[]
    for r in processes:
        row={k:v for k,v in r.items() if k!='trial_mean_replay_ns'}
        row.update(provider=provider,predicted_mean_ns=predicted_mean,mean_ape=None,signed_mean_error_ns=None,
                   absolute_error_mean_ns=None,root_mean_square_error_ns=None,block_ape_p90=None)
        if r['status']=='completed':
            actual=r['trial_mean_replay_ns']; errors=[p-a for p,a in zip(expected,actual)]
            apes=[abs(error)/a for error,a in zip(errors,actual)]
            row.update(mean_ape=abs(predicted_mean-r['mean_ns'])/r['mean_ns'],
                       signed_mean_error_ns=predicted_mean-r['mean_ns'],
                       absolute_error_mean_ns=statistics.mean(abs(e) for e in errors),
                       root_mean_square_error_ns=math.sqrt(statistics.mean(e*e for e in errors)),
                       block_ape_p90=quantile(apes,.9))
            if r['instrumentation']=='bare':
                blocks.extend(dict(provider=provider,program_key=key,job_id=r['job_id'],round=r['round'],trial=t,
                                   predicted_mean_replay_ns=p,actual_mean_replay_ns=a,
                                   signed_error_ns=e,absolute_percentage_error=ape)
                              for t,(p,a,e,ape) in enumerate(zip(expected,actual,errors,apes)))
        process_rows.append(row)
    pairs=[]
    for round_id in range(4):
        a=next(r for r in primary if r['round']==round_id); b=next(r for r in observer if r['round']==round_id)
        ok=a['status']==b['status']=='completed'
        pairs.append(dict(round=round_id,primary_job=a['job_id'],observer_job=b['job_id'],
                          primary_mean_ns=a['mean_ns'],observer_mean_ns=b['mean_ns'],
                          relative_mean_difference=b['mean_ns']/a['mean_ns']-1 if ok else None,
                          passed=bool(ok and abs(b['mean_ns']/a['mean_ns']-1)<=quality['max_observer_relative_mean_difference'])))
    complete=all(r['status']=='completed' for r in processes)
    primary_rows=[r for r in process_rows if r['instrumentation']=='bare' and r['status']=='completed']
    max_ape=max((r['mean_ape'] for r in primary_rows),default=None)
    pooled=quantile([b['absolute_percentage_error'] for b in blocks],.9) if blocks else None
    primary_cv=all(r['status']=='completed' and r['trial_cv']<=quality['max_process_trial_cv'] for r in primary)
    observer_cv=all(r['status']=='completed' and r['trial_cv']<=quality['max_process_trial_cv'] for r in observer)
    gates=dict(all_declared_processes_completed=complete,
               numerical_oracles=all(r['numeric_oracle_passed'] and r['native_repeat_consistency_passed'] for r in processes),
               each_primary_process_mean_ape=bool(len(primary_rows)==4 and max_ape<=quality['max_process_mean_ape']),
               pooled_primary_block_p90_ape=bool(len(blocks)==64 and pooled<=quality['max_trial_average_p90_ape']),
               each_primary_process_cv=primary_cv,each_observer_process_cv=observer_cv,
               each_paired_observer_relative_mean_difference=all(p['passed'] for p in pairs))
    original_fields=[k for k in gates if k!='each_observer_process_cv']
    result=dict(provider=provider,program_key=key,predicted_mean_ns=predicted_mean,
        prediction_scope=prediction.get('context_comparability'),
        declared_primary_processes=4,declared_observer_processes=4,
        completed_primary_processes=sum(r['status']=='completed' for r in primary),
        completed_observer_processes=sum(r['status']=='completed' for r in observer),
        observed_primary_mean_ns=statistics.mean(r['mean_ns'] for r in primary_rows) if primary_rows else None,
        maximum_primary_process_mean_ape=max_ape,pooled_primary_block_ape_p90=pooled,
        maximum_primary_process_cv=max((r['trial_cv'] for r in primary if r['status']=='completed'),default=None),
        maximum_observer_process_cv=max((r['trial_cv'] for r in observer if r['status']=='completed'),default=None),
        maximum_paired_observer_difference=max((abs(p['relative_mean_difference']) for p in pairs if p['relative_mean_difference'] is not None),default=None),
        gates=gates,fully_qualified=all(gates[k] for k in original_fields),
        qualified_original_primary_cv_gate=all(gates[k] for k in original_fields),
        qualified_conservative_all_process_cv_gate=all(gates.values()),
        failed_gates=[k for k in original_fields if not gates[k]],
        failed_conservative_gates=[k for k,v in gates.items() if not v],observer_pairs=pairs,
        processes=process_rows)
    return result,blocks


def evaluate(out):
    out=Path(out).resolve(); m=read(out/'manifest.json'); frozen=read(out/'predictions-frozen.json')
    barrier=read(out/'frozen-barrier.json')
    require(barrier.get('frozen_at_utc') is not None,'freeze timestamp required for independent acceptance')
    for name,value in barrier['files'].items(): require(sha(out/name)==value,'frozen file changed: '+name)
    require(barrier['files'].get('predictions-frozen.json')==sha(out/'predictions-frozen.json'),'predictions not bound by frozen barrier')
    require(frozen['target_time_used_for_fit'] is False and frozen['future_target_state_used'] is False,'frozen predictions leak heldout information')
    require(set(frozen['providers'])==set(PROVIDERS),'all three prespecified providers must remain')
    require(m['protocol']['quality']==QUALITY,'acceptance thresholds changed')
    started=read(out/'heldout-started.json')
    require(started['barrier']==barrier and datetime.fromisoformat(started['at'])>=datetime.fromisoformat(barrier['frozen_at_utc']),
            'heldout phase preceded freeze')
    jobs=[j for j in m['jobs'] if j['split']=='heldout']
    require(len(jobs)==48 and all((out/'jobs'/j['id']/'execution.json').is_file() for j in jobs),
            'all 48 heldout execution records required; failed processes retained')
    cases=sorted({j['program_key'] for j in jobs}); require(len(cases)==6,'six prespecified heldout cases required')
    processes=[process_record(out,m,j,barrier) for j in jobs]
    providers={}; all_blocks=[]; case_csv=[]; process_csv=[]
    for provider in PROVIDERS:
        require(set(frozen['providers'][provider])==set(cases),'prediction case inventory differs')
        results=[]
        for key in cases:
            selected=[r for r in processes if r['program_key']==key]
            result,blocks=case_metrics(provider,key,frozen['providers'][provider][key],selected)
            results.append(result); all_blocks.extend(blocks)
            case_csv.append({k:result[k] for k in ('provider','program_key','declared_primary_processes','completed_primary_processes',
                'declared_observer_processes','completed_observer_processes','predicted_mean_ns','observed_primary_mean_ns',
                'maximum_primary_process_mean_ape','pooled_primary_block_ape_p90','maximum_primary_process_cv',
                'maximum_observer_process_cv','maximum_paired_observer_difference','qualified_original_primary_cv_gate',
                'qualified_conservative_all_process_cv_gate')})
            for row in result['processes']:
                process_csv.append({k:row[k] for k in ('provider','program_key','job_id','round','instrumentation','status','process_id',
                    'predicted_mean_ns','mean_ns','mean_ape','trial_cv','early_mean_ns','late_mean_ns','late_over_early_minus_one',
                    'signed_mean_error_ns','absolute_error_mean_ns','root_mean_square_error_ns','block_ape_p90','numeric_oracle_passed')})
        providers[provider]=dict(declared_cases=6,fully_qualified_cases=sum(r['fully_qualified'] for r in results),
            qualified_original_primary_cv_cases=sum(r['qualified_original_primary_cv_gate'] for r in results),
            qualified_conservative_all_process_cv_cases=sum(r['qualified_conservative_all_process_cv_gate'] for r in results),cases=results)
    csv_new(out/'case-metrics.csv',case_csv); csv_new(out/'process-metrics.csv',process_csv)
    if all_blocks: csv_new(out/'block-errors.csv',all_blocks)
    sources=[receipt(out/name) for name in ('manifest.json','predictions-frozen.json','frozen-barrier.json','heldout-started.json')]
    sources += [receipt(out/'jobs'/j['id']/'execution.json') for j in jobs]
    result=dict(schema='h2d-history-independent-qualification-1',quality=QUALITY,
        evaluation_source=receipt(Path(__file__)),sources=sources,
        frozen_files_verified=True,heldout_started_after_freeze=True,
        declared_native_processes=48,completed_native_processes=sum(r['status']=='completed' for r in processes),
        failed_processes=[r for r in processes if r['status']!='completed'],providers=providers,
        output_tables=[receipt(out/name) for name in ('case-metrics.csv','process-metrics.csv')]
                      +([receipt(out/'block-errors.csv')] if all_blocks else []),
        scope=dict(target='fixed-input warm complete Dense; one/two layers, H512/P64, F32, one GGML CPU thread',
            hardware='recorded CPU plus one GPU; original pageable memory path',
            preparation='reset plus one untimed complete prime, 8 replays/block, 16 blocks/process',
            process_mean_accuracy_applies_to='four bare independent processes per case',
            pooled_p90_scope='64 block means from four fresh processes; blocks are not assumed independent; not single-invocation tail',
            pooled_p90_quantile_method='higher; same as prior e2a.quantile',
            observer_scope='four bare/stamped fresh-process pairs with matching trial/replay preparation',
            original_cv_scope='four bare process CVs; observer mean gate remains mandatory',
            conservative_cv_scope='all eight bare/stamped process CVs',
            parameter_fitting_or_model_selection_performed=False,
            residual_CV_used=False,physical_cause_identified=False,
            arbitrary_external_device_states_or_UMA_qualified=False))
    write(out/'summary.json',result)
    return dict(out=str(out/'summary.json'),completed=result['completed_native_processes'],
                original_primary_cv_passes={p:v['qualified_original_primary_cv_cases'] for p,v in providers.items()},
                conservative_all_cv_passes={p:v['qualified_conservative_all_process_cv_cases'] for p,v in providers.items()})


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True); a=p.parse_args()
    print(json.dumps(evaluate(a.out)))


if __name__=='__main__': main()
