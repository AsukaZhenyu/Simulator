"""Freeze stage prices, forecast timing-free inventories, then score bare runs.

Only calibration-profile jobs supply coefficients. Heldout inventory jobs supply
the actual operator/copy workload, with all elapsed values removed before the
forecast. Fresh unrecorded runs are the independent timing references.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
from fractions import Fraction
from pathlib import Path

from .dataset import read,write
from .run import receipt,sha,require
from .runtime_window_analysis import extract_windows,fit_prices,predict_inventory,price_fit_diagnostics
from .runtime_window_io import LABELS,LEGACY_LABELS,LABEL_PLACEMENT
from llm_infer_model.performance.moe_execution import RuntimeWindowPrice


def encoded(value):
    if isinstance(value,Fraction):return dict(numerator=value.numerator,denominator=value.denominator)
    if isinstance(value,dict):return {k:encoded(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [encoded(v) for v in value]
    return value

def load_job(out,job):
    out=Path(out);execution=read(out/'jobs'/job/'execution.json')
    require(execution['status']=='completed','missing complete native job '+job)
    native=read(execution['native']['path'])
    require(sha(execution['native']['path'])==execution['native']['sha256'],'native output content changed')
    return native,execution

def calibrate(out):
    out=Path(out).resolve();m=read(out/'manifest.json');ledgers=[];evidence=[];records=[]
    policy=m.get('feature_policy','mechanism_v1')
    labels=tuple(m.get('labels',LEGACY_LABELS))
    for label in labels:
        native,execution=load_job(out,label+'-calibration-profile')
        require(native['request']['runtime_window_observer'] is True,'calibration must expose actual scoped windows')
        require(all(r['split']=='calibration' for r in native['request']['requests']),'heldout values cannot train prices')
        evidence.append(str(out/'jobs'/(label+'-calibration-profile')/'result.json'))
        for row in native['results']:
            ledger=extract_windows(row,placement=LABEL_PLACEMENT[label],frequency_hz=native['clock']['frequency_hz'],feature_policy=policy)
            ledger['source_process_id']=native['process_id']
            ledgers.append(ledger)
            records.append(dict(placement=LABEL_PLACEMENT[label],request_id=row['id'],window_count=len(ledger['windows']),CPU_node_windows=ledger['CPU_node_windows'],expert_layers=ledger['expert_layers'],clock_conservation=ledger['clock_conservation_passed'],scopes=dict(Counter(w['inventory']['scope'] for w in ledger['windows'])),
                work_kinds=dict(Counter(w['inventory']['work_spec']['kind'] for w in ledger['windows']))))
    prices=fit_prices(ledgers,evidence,policy)
    diagnostics=price_fit_diagnostics(ledgers,prices)
    calibration=[]
    for ledger in ledgers:
        prediction=predict_inventory(ledger['workload_inventory'],prices)
        calibration.append(dict(request_id=ledger['source_request_id'],predicted_ns=prediction['total_ns'],measured_ns=ledger['total_ns'],APE=abs(float(prediction['total_ns'])/float(ledger['total_ns'])-1)))
    result=dict(schema='real-moe-window-price-catalog-1',created_at_utc=datetime.now(timezone.utc).isoformat(),manifest=receipt(out/'manifest.json'),
        prices=[p.to_dict() for p in prices],price_count=len(prices),evidence=[receipt(p) for p in evidence],window_coverage=records,
        per_price_fit_diagnostics=[dict(key=p.key,**diagnostics[p.key]) for p in prices],
        fit=dict(method='mechanism-constrained nonnegative affine active-set least squares',candidate_terms=['intercept','actual local query rows','actual attention query rows * actual padded KV','observed H2D bytes'],
            feature_policy=policy,intercept_policy='per_key',structural_rules=receipt(Path(__file__).resolve().parents[2]/'modeling/llm_infer_model/performance/window_work.py'),
            forbidden_KV_proxies=['ordinary matrix','FFN/expert work','unclassified HOST','sampler','cache','OS/available compute state'],
            unknown_work_retains_conservative_shape_guards=True,
            whole_model_multiplier=False,heldout_times_read=False,old_heldout_diagnostic_times_used=False,calibration_processes=len(labels),calibration_requests=len(calibration)),calibration_fit_diagnostics=calibration,
        price_scope='inclusive CPU-node / natural-runtime / host response windows; SAMPLER is partitioned residual tail, not the whole sampling call; not independent device work',
        calibration_fit_diagnostics_are_independent_accuracy=False)
    write(out/'WINDOW_PRICES.json',encoded(result))
    write(out/'PRICE_FREEZE.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),prices=receipt(out/'WINDOW_PRICES.json'),calibration_sources=result['evidence'],whole_model_multiplier_fitted=False,heldout_TIMING_used=False,
        feature_policy=policy,structural_rules=result['fit']['structural_rules'],rules_selected_from_work_metadata_not_target_errors=True))
    return dict(prices=receipt(out/'WINDOW_PRICES.json'),price_count=len(prices),hardware_executed=False)

def forecast(out,allow_extrapolation=False):
    out=Path(out).resolve();frozen=read(out/'PRICE_FREEZE.json')
    require(sha(frozen['prices']['path'])==frozen['prices']['sha256'],'frozen calibration prices changed')
    table=read(frozen['prices']['path']);prices=tuple(RuntimeWindowPrice.from_dict(p) for p in table['prices'])
    policy=table['fit']['feature_policy']
    labels=tuple(read(out/'manifest.json').get('labels',LEGACY_LABELS))
    forecasts=[];inventories=[]
    for label in labels:
        native,execution=load_job(out,label+'-inventory')
        require(all(r['split']=='heldout' for r in native['request']['requests']),'separate heldout structural workload required')
        for row in native['results']:
            ledger=extract_windows(row,placement=LABEL_PLACEMENT[label],frequency_hz=native['clock']['frequency_hz'],feature_policy=policy)
            inventory=dict(schema='real-moe-timing-free-workload-1',placement=LABEL_PLACEMENT[label],request_id=row['id'],
                prompt_token_ids=row['prompt_token_ids'],generated_token_ids=row['output_token_ids'],final_full_logits_sha256=row['final_full_logits_sha256'],
                windows=ledger['workload_inventory'],source_native=execution['native'],durations_retained=False,source_elapsed_values_used_for_prices=False)
            path=out/'workload-inventories'/(label+'-'+row['id']+'.json');path.parent.mkdir(exist_ok=True);write(path,inventory);inventories.append(receipt(path))
            record=dict(placement=inventory['placement'],request_id=row['id'],inventory=receipt(path),status='unsupported')
            try:
                prediction=predict_inventory(read(path)['windows'],prices,allow_extrapolation=allow_extrapolation)
                record.update(status='predicted',predicted_total_ns=prediction['total_ns'],predicted_prefill_ns=prediction['phase_ns']['prefill'],predicted_decode_with_sampling_ns=prediction['phase_ns']['decode'],
                    window_count=len(inventory['windows']),extrapolated_windows=sum(e['extrapolated'] for e in prediction['compiled']['windows']))
                write(out/'workload-inventories'/(label+'-'+row['id']+'-compiled.json'),encoded(prediction['compiled']))
            except Exception as error:record['reason']=repr(error)
            forecasts.append(record)
    result=dict(schema='real-moe-window-forecasts-1',created_at_utc=datetime.now(timezone.utc).isoformat(),price_freeze=receipt(out/'PRICE_FREEZE.json'),
        source_prices=frozen['prices'],workload_inventories=inventories,forecasts=forecasts,allow_extrapolation=allow_extrapolation,
        heldout_elapsed_values_used_for_predictions=False,heldout_routes_and_operator_shapes_are_modeling_workload=True,mapping_receives_future_route_trace=False)
    write(out/'PREDICTIONS.json',encoded(result));write(out/'PREDICTION_FREEZE.json',dict(predictions=receipt(out/'PREDICTIONS.json'),prices=frozen['prices'],inventories=inventories,
        manifest=receipt(out/'manifest.json'),price_freeze=receipt(out/'PRICE_FREEZE.json'),
        created_at_utc=datetime.now(timezone.utc).isoformat(),bare_validation_jobs_not_read=True,source_prediction=receipt(Path(__file__))))
    return dict(predictions=receipt(out/'PREDICTIONS.json'),predicted=sum(r['status']=='predicted' for r in forecasts),declared=len(forecasts),hardware_executed=False)

def evaluate(out):
    out=Path(out).resolve();freeze=read(out/'PREDICTION_FREEZE.json');require(sha(freeze['predictions']['path'])==freeze['predictions']['sha256'],'predictions changed after freeze')
    predictions=read(freeze['predictions']['path']);lookup={(p['placement'],p['request_id']):p for p in predictions['forecasts']};rows=[]
    labels=tuple(read(out/'manifest.json').get('labels',LEGACY_LABELS))
    for repeat in (0,1):
        for label in labels:
            native,execution=load_job(out,label+'-heldout-r'+str(repeat))
            require(native['request']['runtime_window_observer'] is False,'bare validation must disable window observation')
            for row in native['results']:
                p=lookup[LABEL_PLACEMENT[label],row['id']];base=dict(placement=LABEL_PLACEMENT[label],request_id=row['id'],repeat=repeat,status='unsupported',reference=execution['native'])
                if p['status']!='predicted':base['reason']=p.get('reason');rows.append(base);continue
                require(p['inventory'] in freeze['inventories'],'forecast inventory not bound in prediction freeze')
                require(sha(p['inventory']['path'])==p['inventory']['sha256'],'frozen workload inventory changed')
                require(execution.get('prediction_freeze',{}).get('sha256')==sha(out/'PREDICTION_FREEZE.json'),'bare run did not bind this prediction freeze before launch')
                inventory=read(p['inventory']['path']);same_work=(row['prompt_token_ids']==inventory['prompt_token_ids'] and row['output_token_ids']==inventory['generated_token_ids'])
                same_final=row['final_full_logits_sha256']==inventory['final_full_logits_sha256']
                if not(same_work and same_final):base.update(status='workload_or_numerical_mismatch',actual_prefix_equal=same_work,final_full_logits_equal=same_final);rows.append(base);continue
                truth=dict(total=row['whole_request_wall_ns'],prefill=row['prefill_completed_ns'],decode_with_sampling=row['whole_request_wall_ns']-row['prefill_completed_ns'])
                forecast=dict(total=p['predicted_total_ns'],prefill=p['predicted_prefill_ns'],decode_with_sampling=p['predicted_decode_with_sampling_ns'])
                def number(v):return Fraction(v['numerator'],v['denominator']) if isinstance(v,dict) else Fraction(v)
                ape={k:abs(float(number(forecast[k]))/truth[k]-1) for k in truth}
                base.update(status='evaluated',measured_ns=truth,predicted_ns=forecast,APE=ape,mean_time_target_passed=all(v<=0.1 for v in ape.values()),actual_prefix_equal=True,final_full_logits_equal=True,extrapolated_windows=p['extrapolated_windows'])
                rows.append(base)
    scored=[r for r in rows if r['status']=='evaluated']
    declared=12*len(labels)
    result=dict(schema='real-moe-independent-runtime-window-evaluation-1',predictions=freeze['predictions'],rows=rows,declared=declared,evaluated=len(scored),
        all_declared_evaluated=len(scored)==declared,mean_APE={k:sum(r['APE'][k] for r in scored)/len(scored) if scored else None for k in ('total','prefill','decode_with_sampling')},
        worst_APE={k:max((r['APE'][k] for r in scored),default=None) for k in ('total','prefill','decode_with_sampling')},
        passed=sum(r['mean_time_target_passed'] for r in scored),target_APE=0.1,all_passed=len(scored)==declared and all(r['mean_time_target_passed'] for r in scored),
        parameters_refitted=False,whole_model_multiplier_fitted=False,bare_measurement_truth=True,route_and_operator_structure_driven_prediction=True,
        numerical_gate='same prompt/output token IDs and final full logits within each placement; does not establish all intermediate logits/routes identical',
        scope='real pinned OLMoE Q4_K_M; original normal output pruning; request warmup excluded; CPU-only, CPU-experts/GPU-other and CPU-attention/GPU-other actual runtime placements',
        limitations=['One local CPU/GPU and finite six-request heldout corpus','Runtime bridge prices preserve this implementation lifecycle, not isolated device kernels','No strategy ranking with different generated token trajectories','Extrapolated coordinates, if explicitly enabled, are reported per forecast'])
    write(out/'INDEPENDENT_EVALUATION.json',encoded(result));return result

def old_case_posthoc(out,baseline):
    """Known-case diagnosis after price freeze; never independent validation."""
    out,baseline=Path(out).resolve(),Path(baseline).resolve()
    frozen_receipt=receipt(out/'PRICE_FREEZE.json');frozen=read(frozen_receipt['path'])
    require(sha(frozen['prices']['path'])==frozen['prices']['sha256'],'frozen calibration prices changed')
    prices=tuple(RuntimeWindowPrice.from_dict(p) for p in read(frozen['prices']['path'])['prices'])
    previous=read(baseline/'PREDICTIONS.json')
    old_lookup={(r['placement'],r['request_id']):r for r in previous['forecasts']}
    reports=[];sources=[]
    def number(value):return Fraction(value['numerator'],value['denominator']) if isinstance(value,dict) else Fraction(value)
    for label in ('cpu','hybrid'):
        native,execution=load_job(baseline,label+'-inventory');sources.append(execution['native'])
        repeats=[]
        for repeat in (0,1):
            bare,run=load_job(baseline,label+'-heldout-r'+str(repeat));sources.append(run['native'])
            repeats.append({r['id']:r for r in bare['results']})
        for row in native['results']:
            placement=native['request']['placement'];old=old_lookup[placement,row['id']]
            ledger=extract_windows(row,placement=placement,frequency_hz=native['clock']['frequency_hz'])
            inventory=ledger['workload_inventory']
            require(all('observed_ns' not in value for value in inventory),'posthoc work inventory retained measured costs')
            record=dict(placement=placement,request_id=row['id'],status='unsupported',independent_validation=False,
                old_window_times_used_to_change_prices=False,numerical_gate='unchanged prompt/output tokens and full final logits',
                old_status=old['status'])
            try:
                prediction=predict_inventory(inventory,prices,allow_extrapolation=True)
                proposed=dict(total=prediction['total_ns'],prefill=prediction['phase_ns']['prefill'],decode_with_sampling=prediction['phase_ns']['decode'])
                record.update(proposed_prediction_ns=proposed,explicit_extrapolated_windows=sum(w['extrapolated'] for w in prediction['compiled']['windows']))
            except Exception as error:
                record['reason']=repr(error);reports.append(record);continue
            paired=[r[row['id']] for r in repeats]
            matched=[r['prompt_token_ids']==row['prompt_token_ids'] and r['output_token_ids']==row['output_token_ids'] and r['final_full_logits_sha256']==row['final_full_logits_sha256'] for r in paired]
            record['endpoint_matches']=matched
            if not all(matched):
                record.update(status='retained_original_numerical_mismatch',excluded_from_performance_accuracy=True)
                reports.append(record);continue
            truth={name:sum(Fraction((r['whole_request_wall_ns'] if name=='total' else r['prefill_completed_ns'] if name=='prefill' else r['whole_request_wall_ns']-r['prefill_completed_ns'])) for r in paired)/2 for name in proposed}
            original=dict(total=number(old['predicted_total_ns']),prefill=number(old['predicted_prefill_ns']),decode_with_sampling=number(old['predicted_decode_with_sampling_ns'])) if old['status']=='predicted' else None
            record.update(status='posthoc_compared',old_prediction_ns=original,mean_old_bare_ns=truth,
                proposed_APE={name:abs(float(proposed[name]/truth[name])-1) for name in truth},
                old_APE={name:abs(float(original[name]/truth[name])-1) for name in truth} if original else None)
            reports.append(record)
    require(receipt(out/'PRICE_FREEZE.json')==frozen_receipt and sha(frozen['prices']['path'])==frozen['prices']['sha256'],'posthoc operation changed frozen prices')
    result=dict(schema='real-moe-old-case-posthoc-1',price_freeze=frozen_receipt,baseline_predictions=receipt(baseline/'PREDICTIONS.json'),
        original_evidence=sources,rows=reports,known_cases=12,independent_validation=False,parameters_refitted=False,
        purpose='diagnose an already-observed failure set; never use this comparison to select features or adjust the frozen model',
        original_numerical_failure_preserved=True,new_independent_validation='disjoint fixed heldout-index=1 requests and fresh bare processes')
    write(out/'OLD_CASE_POSTHOC.json',encoded(result));return result

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('calibrate','forecast','evaluate','posthoc'));p.add_argument('--out',required=True);p.add_argument('--allow-extrapolation',action='store_true');p.add_argument('--baseline');a=p.parse_args()
    if a.command=='posthoc':require(a.baseline is not None,'posthoc baseline required')
    result=calibrate(a.out) if a.command=='calibrate' else forecast(a.out,a.allow_extrapolation) if a.command=='forecast' else old_case_posthoc(a.out,a.baseline) if a.command=='posthoc' else evaluate(a.out)
    print(__import__('json').dumps(encoded(result),ensure_ascii=False))
if __name__=='__main__':main()
