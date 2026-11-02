"""Observed route statistics and private Decode-only M2 route fragments.

The command accepts completed real native collection receipts only. Software
fixtures may exercise the pure transforms, but are never written as empirical
results. Normal Prefill pruning is retained; no missing routes are imputed.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path

from llm_infer_model.m2.moe_spec import parse_route_trace
from .dataset import digest,read,write
from .postcollection import audit_files
from .run import receipt,require


def observed_rows(native):
    """Flatten only values actually returned by the native routing callback."""
    metadata={r['id']:r for r in native['request']['requests']};rows=[]
    for result in native['results']:
        source=metadata[result['id']]
        for route in result['routes']:
            require(len(route['absolute_positions'])==len(route['expert_ids']),'observed positions/experts differ')
            for position,ids in zip(route['absolute_positions'],route['expert_ids']):
                rows.append(dict(request_id=result['id'],domain=source.get('domain'),length_class=source.get('length_class'),
                    split=source.get('split'),phase=route['phase'],forward_index=route['forward_index'],layer=route['layer'],
                    absolute_position=position,expert_ids=list(ids)))
    return rows


def statistics(native):
    rows=observed_rows(native);experts=native['observed_model_contract']['experts'];top_k=native['observed_model_contract']['top_k']
    frequencies=defaultdict(Counter);denominators=Counter();groups={};decode=defaultdict(list)
    for row in rows:
        key=(row['request_id'],row['phase'],row['layer']);denominators[key]+=1;frequencies[key].update(row['expert_ids'])
        groups[row['request_id']]=dict(domain=row['domain'],length_class=row['length_class'],split=row['split'])
        if row['phase']=='decode':decode[(row['request_id'],row['layer'])].append(row)
    matrix=[];category=defaultdict(lambda:dict(rows=0,counts=Counter()))
    for (request_id,phase,layer),counts in sorted(frequencies.items()):
        n=denominators[(request_id,phase,layer)];meta=groups[request_id]
        matrix.append(dict(request_id=request_id,phase=phase,layer=layer,**meta,observed_token_rows=n,
            expert_selection_count=n*top_k,counts=[counts[e] for e in range(experts)],
            activation_fraction=[counts[e]/n for e in range(experts)],
            selection_share=[counts[e]/(n*top_k) for e in range(experts)],
            denominator='actual observed token-layer rows, not total prompt length'))
        c=category[(meta['domain'],meta['length_class'],meta['split'],phase,layer)]
        c['rows']+=n;c['counts'].update(counts)
    reuse=[];totals=defaultdict(lambda:dict(pairs=0,overlap=0,gaps=0))
    for (request_id,layer),events in sorted(decode.items()):
        events.sort(key=lambda r:r['absolute_position']);overlap=0;pairs=0;gaps=0
        for a,b in zip(events,events[1:]):
            if b['absolute_position']!=a['absolute_position']+1:gaps+=1;continue
            pairs+=1;overlap+=len(set(a['expert_ids'])&set(b['expert_ids']))
        meta=groups[request_id];key=(meta['domain'],meta['length_class'],meta['split'])
        totals[key]['pairs']+=pairs;totals[key]['overlap']+=overlap;totals[key]['gaps']+=gaps
        reuse.append(dict(request_id=request_id,layer=layer,**meta,adjacent_pairs=pairs,
            selected_expert_overlap_sum=overlap,reuse_fraction=overlap/(top_k*pairs) if pairs else None,
            skipped_nonadjacent_gaps=gaps,definition='|top8(t) intersect top8(t-1)| / 8; same request and layer only'))
    category_frequency=[]
    for (domain,length,split,phase,layer),value in sorted(category.items(),key=lambda x:str(x[0])):
        n=value['rows'];counts=value['counts']
        category_frequency.append(dict(domain=domain,length_class=length,split=split,phase=phase,layer=layer,
            observed_token_rows=n,counts=[counts[e] for e in range(experts)],activation_fraction=[counts[e]/n for e in range(experts)]))
    grouped_reuse=[]
    for (domain,length,split),value in sorted(totals.items(),key=lambda x:str(x[0])):
        grouped_reuse.append(dict(domain=domain,length_class=length,split=split,**value,
            reuse_fraction=value['overlap']/(top_k*value['pairs']) if value['pairs'] else None))
    return dict(schema='real-moe-observed-route-statistics-1',model_sha256=native['model_sha256_verified_by_parent'],
        placement=native['request']['placement'],observed_rows=len(rows),frequencies=matrix,
        frequencies_by_category=category_frequency,decode_adjacent_reuse=reuse,decode_reuse_by_category=grouped_reuse,
        absent_category_labels_are_unknown=True,unobserved_scores_imputed=False,prefill_pruning_imputed=False,
        observed_rows_only=True,performance_accuracy_claimed=False,
        interpretation='researcher-only summary of recorded routes; full future statistics are not policy input')


def decode_fragments(native,*,source_evidence,layer_prefix='olmoe16.layer'):
    """Keep a Decode-only private container; never invent full Prefill coverage."""
    layers=native['observed_model_contract']['layers'];top_k=native['observed_model_contract']['top_k'];experts=native['observed_model_contract']['experts']
    require(layers==16 and top_k==8 and experts==64,'reviewed OLMoE route geometry required')
    rows=observed_rows(native);by_event=defaultdict(dict)
    for row in rows:
        if row['phase']!='decode':continue
        event=(row['request_id'],row['forward_index'],row['absolute_position'])
        require(row['layer'] not in by_event[event],'duplicate real layer route; do not merge')
        by_event[event][row['layer']]=row
    events=[];fragments=[]
    for (request_id,forward,position),observed in sorted(by_event.items()):
        require(set(observed)==set(range(layers)),'complete sixteen-layer Decode route required')
        entries=[]
        for layer in range(layers):
            row=observed[layer];ids=row['expert_ids']
            require(len(ids)==8 and len(set(ids))==8 and all(type(e) is int and 0<=e<64 for e in ids),'invalid observed expert selection')
            entry=dict(request_id=request_id,layer_id=layer_prefix+str(layer),absolute_position=position,expert_ids=ids)
            parsed=parse_route_trace([entry],where='real_olmoe_decode_fragment.route_trace')
            require(len(parsed)==1 and list(parsed[0].expert_ids)==ids,'M2 route parser changed actual expert IDs')
            fragment=dict(schema_version='m2_moe_route_fragment_v1',phase='decode',
                request=dict(request_id=request_id,tokens=1,position_start=position),route_trace=[entry],
                route_source='observed_native_inference:'+native['model_sha256_verified_by_parent']+f':forward={forward}:layer={layer}',
                information_mode='causal_replay',provenance=dict(source_evidence=source_evidence,source_layer=layer,
                    m2_layer_id=entry['layer_id'],model=native['observed_model_contract'],
                    actual_forward_index=forward,expert_ids_remapped=False,top_k_truncated=False,scores_observed=False,
                    initial_history_required=True,performance_costs_provided=False,quantized_layer_or_kernel_binding_provided=False))
            fragments.append(fragment);entries.append(entry)
        events.append(dict(request_id=request_id,forward_index=forward,absolute_position=position,route_trace=entries,
            disclosure='modeling-private; each layer row becomes visible only after its corresponding router completes'))
    return dict(schema='real-moe-modeling-private-decode-1',model_sha256=native['model_sha256_verified_by_parent'],
        model_contract=native['observed_model_contract'],placement=native['request']['placement'],source_evidence=source_evidence,
        events=events,fragments=fragments,complete_layers_per_nonempty_decode_event=16,
        decode_coverage='complete_observed_decode_events' if events else 'empty_no_decode_forward_observed',
        old_full_prefill_bundle_compatible=False,
        old_bundle_reason='moe_inference_routes_v1 requires every layer/token including Prefill; normal final-layer pruning does not satisfy it',
        fragment_route_parser_compatible=True,complete_quantized_model_binding_supported=False,
        prefill_exported_as_full_m2_invocation=False,prefill_missing_rows_imputed=False,
        visibility_mode='causal_replay',privileged_input=True,policy_must_receive='CausalM2Session.observe or GlobalM2Session.observe only',
        forbidden_policy_input='this full container, fragments for unopened invocations, or future expert-frequency summaries',
        existing_model_visibility='m2.causal._MoEVisibility and m2.global_execution._GlobalVisibility gate routes on router OP_DONE',
        costs_provided=False,hardware_accuracy_certified=False)


def m2_invocation_patch(fragment):
    """Privileged driver patch, matching the existing invocation_slice output.

    The caller must supply the matching layer, real history/state, hardware and
    qualified costs. This function creates none of those missing components.
    Policies receive only the session observation after router completion.
    """
    require(fragment['schema_version']=='m2_moe_route_fragment_v1' and fragment['phase']=='decode'
        and fragment['information_mode']=='causal_replay','causal Decode fragment required')
    request=fragment['request'];rows=fragment['route_trace']
    require(request['tokens']==1 and len(rows)==1 and rows[0]['request_id']==request['request_id']
        and rows[0]['absolute_position']==request['position_start'],'one actual Decode token required')
    parse_route_trace(rows,where='private_m2_invocation_patch.route_trace')
    return dict(phase='decode',requests=[dict(request)],route_trace=[dict(rows[0])],
        route_source=fragment['route_source'],information_mode='causal_replay')


def completed_native(path):
    path=Path(path).resolve();native=read(path);execution=path.parent/'execution.json';record=read(execution)
    require(record.get('status')=='completed' and record.get('native')==receipt(path)
        and record.get('process_id')==native.get('process_id'),'actual completed native acquisition receipt required')
    require(native.get('mode')=='trace' and native.get('model_weights_loaded') is True,'observed real-model trace required; tokenizer/fixtures are not empirical input')
    return native,dict(native=receipt(path),execution=receipt(execution))


def analyze(trace_path,out,bare_path):
    native,evidence=completed_native(trace_path);out=Path(out).resolve();require(not out.exists(),'new route analysis output directory required');out.mkdir(parents=True)
    audit=audit_files(trace_path,out/'shape-routing-audit.json',bare_path)
    if audit['status']!='passed':
        status=dict(schema='real-moe-route-analysis-status-1',status=audit['status'],source=evidence,
            reason='full route statistics/export withheld because the independent shape/provenance gate did not pass',
            synthetic_data_written=False,raw_modified=False,whole_model_accuracy_certified=False)
        write(out/'STATUS.json',status);return status
    evidence['shape_routing_audit']=receipt(out/'shape-routing-audit.json')
    summary=statistics(native);write(out/'observed-route-statistics.json',summary)
    private=decode_fragments(native,source_evidence=evidence);write(out/'modeling-private/decode-routes.json',private)
    with (out/'layer-expert-frequency.csv').open('x',encoding='utf-8',newline='') as stream:
        fields=['request_id','domain','length_class','split','phase','layer','expert','count','observed_token_rows','activation_fraction','selection_share']
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for row in summary['frequencies']:
            for expert,count in enumerate(row['counts']):
                writer.writerow({**{k:row[k] for k in fields if k in row},'expert':expert,'count':count,
                    'activation_fraction':row['activation_fraction'][expert],'selection_share':row['selection_share'][expert]})
    with (out/'decode-adjacent-reuse.csv').open('x',encoding='utf-8',newline='') as stream:
        fields=['request_id','domain','length_class','split','layer','adjacent_pairs','selected_expert_overlap_sum','reuse_fraction','skipped_nonadjacent_gaps']
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in summary['decode_adjacent_reuse'])
    status=dict(schema='real-moe-route-analysis-status-1',status='passed',source=evidence,
        statistics=receipt(out/'observed-route-statistics.json'),private_decode=receipt(out/'modeling-private/decode-routes.json'),
        decode_events=len(private['events']),m2_decode_fragments=len(private['fragments']),
        empirical_input_only=True,prefill_missing_rows_imputed=False,whole_model_accuracy_certified=False)
    write(out/'STATUS.json',status);return status


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--trace',required=True);p.add_argument('--bare',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    result=analyze(a.trace,a.out,a.bare);print(json.dumps(dict(out=a.out,status=result['status'],decode_events=result.get('decode_events',0))))
    if result['status']!='passed':raise SystemExit(1 if result['status']=='failed' else 2)
