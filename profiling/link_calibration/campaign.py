"""Size-separated registered/0-idle H2D observations with causal PCIe state.

Profiling records exact completed copy boundaries and never fits a link rate.
Model-side parameter selection/fitting and final reporting belong to the caller.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import shutil
import statistics
import subprocess
from profiling.dynamic_validation import idle_campaign as old

ROOT=Path(__file__).resolve().parents[2]
read,write,sha,receipt,require=old.read,old.write,old.sha,old.receipt,old.require
SIZES={'profile':(128*1024,512*1024,2*1024**2),'heldout':(256*1024,1024**2)}
FIELDS=('pcie_current_gen','pcie_max_gen','pcie_current_width','pcie_max_width')

def balanced_orders(size_count,round_id,trials=12):
    require(size_count in (2,3) and trials==12 and type(round_id) is int and 0<=round_id<3,'fixed three-process/twelve-trial plan required')
    rows=[]
    for trial in range(trials):
        row=[(trial+round_id+i)%size_count for i in range(size_count)]
        if (trial//size_count+round_id)%2:row.reverse()
        rows.append(row)
    require(all(Counter(row[slot] for row in rows)==Counter({i:trials//size_count for i in range(size_count)}) for slot in range(size_count)),'size positions not balanced')
    return rows
def query_orders(size_count,round_id):
    return [('bare','state_queries') if (trial+slot+round_id)%2==0 else ('state_queries','bare') for trial in range(12) for slot in range(size_count)]
def state_key(sample):
    if not isinstance(sample,dict) or sample.get('status')!='ok':return None
    values=sample.get('values',{})
    if not all(type(values.get(k)) is int and values[k]>0 for k in FIELDS):return None
    return tuple(values[k] for k in FIELDS)
def query_extent(sample):
    if not isinstance(sample,dict):return None
    queries=sample.get('queries',[])
    if len(queries)!=4 or not all(q.get('qpc_valid') is True and type(q.get('qpc_begin')) is int and type(q.get('qpc_end')) is int and q['qpc_begin']<=q['qpc_end'] for q in queries):return None
    return min(q['qpc_begin'] for q in queries),max(q['qpc_end'] for q in queries)
def classify(row):
    """Post state can reject a training scope; it never defines the input key."""
    before,after=row.get('pcie_before'),row.get('pcie_after');key=state_key(before)
    if row['control']=='bare':return dict(prior_key=None,qualified=False,reason='bare control; state not queried')
    a,b=query_extent(before),query_extent(after)
    if a is not None:require(a[1]<=row['burst_begin_tick'],'pre-state query used a future target interval')
    if b is not None:require(b[0]>=row['burst_end_tick'],'post-state query overlaps target interval')
    if key is None or a is None or b is None:return dict(prior_key=key,qualified=False,reason='unknown PCIe fields or query clocks')
    if state_key(after)!=key:return dict(prior_key=key,qualified=False,reason='state changed across burst; pure-state price unsupported')
    return dict(prior_key=key,qualified=True,reason='pre/post boundary values agree; within-burst constancy not directly measured')
def coverage(records):
    cells=defaultdict(set);all_keys=set();unknown=0
    for record in records:
        for row in record.get('bursts',[]):
            c=classify(row)
            if not c['qualified']:
                if row['control']=='state_queries':unknown+=1
                continue
            key=tuple(c['prior_key']);all_keys.add(key);cells[(key,row['bytes'])].add(record['context_id'])
    complete=[key for key in sorted(all_keys) if all(len(cells[(key,size)])>=3 for size in SIZES['profile'])]
    expected=all(any(key[0]==gen for key in complete) for gen in (3,4))
    return dict(schema='link-copy-size-state-coverage-1',coverage_passed=expected,
        completed_exact_state_keys=[list(k) for k in complete],expected_current_generations=[3,4],
        required_independent_contexts_per_state_size_cell=3,unqualified_or_unknown_query_bursts=unknown,
        cells=[dict(state_key=list(k),bytes=size,contexts=sorted(cells[(k,size)]),independent_contexts=len(cells[(k,size)])) for k in sorted(all_keys) for size in SIZES['profile']],
        key_fields=list(FIELDS),other_observed_state_keys_not_imputed=[list(k) for k in sorted(all_keys) if k not in complete],
        missing_cells_stop_without_extra_sampling=not expected,post_state_is_predictor_input=False,physical_effective_bandwidth_measured=False)

def validate_freeze(out,barrier):
    out=Path(out)
    require(barrier.get('schema')=='link-copy-cost-freeze-1' and barrier.get('manifest',{}).get('sha256')==sha(out/'manifest.json'),'root model freeze must bind the current manifest')
    files=barrier.get('files');require(isinstance(files,dict) and bool(files),'nonempty explicit frozen model files required')
    require(barrier.get('predictor_parameters_frozen') is True and isinstance(barrier.get('model_scope'),str) and bool(barrier['model_scope']),'root must explicitly declare frozen parameters and conditional model scope')
    summary_path=out/'profile-summary.json';require(barrier.get('profile_summary',{}).get('sha256')==sha(summary_path),'root freeze must bind completed profile support review')
    summary=read(summary_path);require(summary.get('declared_contexts')==3 and summary.get('completed_contexts')==3 and summary.get('exact_state_coverage',{}).get('coverage_passed') is True,'three fresh calibration contexts and complete Gen3/Gen4 size-state support required before heldout')
    for name,h in files.items():
        path=Path(name);require(not path.is_absolute() and '..' not in path.parts,'frozen model file must be relative within this campaign')
        require(sha(out/path)==h,'frozen model file changed')

def prepare(out,binary,nvml_dll):
    out,binary=Path(out).resolve(),Path(binary).resolve();require(not out.exists(),'new link calibration directory required')
    parent_path=ROOT/'build/dynamic-validation-20261026/idle-v2b/manifest.json';parent=read(parent_path)
    require(binary.is_file(),'explicit compiled link-copy-probe required');out.mkdir(parents=True)
    source=Path(__file__).parent/'native/link_probe.cpp';programs={}
    for split in SIZES:
        binding=parent['programs'][split]
        for k in ('program','reference'):require(sha(binding[k]['path'])==binding[k]['sha256'],'retained mathematical buffer/reference changed')
        programs[split]=binding
    require(programs['profile']['program_sha256']!=programs['heldout']['program_sha256'],'independent heldout mathematical seed required')
    paths=[Path(__file__),source,source.parent/'pcie_nvml.hpp',source.parent/'CMakeLists.txt',source.parent/'cmake/CMakeLists.txt',ROOT/'profiling/native/steady_costs/probe.cpp',ROOT/'profiling/native/steady_costs/components.hpp',ROOT/'profiling/dynamic_validation/idle_campaign.py',ROOT/'profiling/state_calibration/run.py',ROOT/'profiling/dac_validation/e2a.py']
    paths+=list((ROOT/'profiling/native/steady_costs/build-release-v1').glob('*.inc'));sources=[]
    for path in paths:
        target=out/'executed_sources'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);sources.append(dict(source=receipt(path),archived=receipt(target)))
    exe=out/'executed_binary'/binary.name;exe.parent.mkdir();shutil.copy2(binary,exe)
    dll=Path(nvml_dll).resolve();nvml_binding=receipt(dll) if dll.is_file() else dict(path=str(dll),sha256=None,status='unavailable_before_collection')
    jobs=[]
    for split,sizes in SIZES.items():
        for index in range(3):
            sampling=old.e2a.request('action',trials=12,target=programs[split]['target_action_index']);sampling['repeats']=16
            r=dict(schema='link-copy-size-request-1',source_sha256=sha(source),split=split,sampling=sampling,expected_process_mask=4294967295,memory_mode='registered',idle_ms=0,
                   sizes_bytes=list(sizes),orders=balanced_orders(len(sizes),index),query_orders=query_orders(len(sizes),index),nvml_dll_path=str(dll))
            jobs.append(dict(id=f'{split}-r{index}',split=split,round=index,request=r))
    manifest=dict(schema='link-copy-size-campaign-1',prepared_at_utc=old.e2a.utc(),parent_manifest=receipt(parent_path),binary=receipt(exe),sources=sources,
        runtime=parent['runtime'],runtime_dirs=parent['runtime_dirs'],nvml_module=nvml_binding,programs=programs,jobs=jobs,
        protocol=dict(fresh_profile_contexts=3,fresh_heldout_contexts=3,trials=12,copies_per_burst=16,bare_and_statequery_controls=True,
            split_first_and_suffix15=True,primary_sized_COPY_uses_unchanged_Executor_action=True,poison_oracle_before_formal_warmup=True,
            source_target_arena_bytes=2097152,registered=True,idle_ms=0,no_CUPTI=True,no_CUDA_events=True,heldout_requires_frozen_model=True,
            states_required_current_generations=[3,4],min_contexts_per_exact_state_size_cell=3,retries=0,missing_cells_stop=True,
            post_state_used_for_prediction=False,theoretical_link_gen_ratio_used=False,whole_LLM_accuracy_claimed=False,physical_effective_bandwidth_claimed=False))
    write(out/'manifest.json',manifest);return dict(manifest=receipt(out/'manifest.json'),profile_processes=3,heldout_processes=3,hardware_executed=False)

def verify(n,r,binding,pid):
    require(n['schema']=='link-copy-size-native-1' and n['request']==r and n['process_id']==pid and n['source_sha256']==r['source_sha256'],'native identity/request mismatch')
    require(n['program_sha256']==binding['program_sha256'] and n['original_target_action']==binding['target_action'] and n['original_arena_bytes']==2097152,'original target/arena changed')
    require(n['arena_registration']['during']['registered_host'] and n['arena_registration']['restored'] and n['source_pointer_kind']['registered_host'],'registered memory proof missing')
    require(n['all_sizes_share_original_buffers_context_and_registration'] and n['full_original_untimed_prime_before_every_burst'],'original arena/conditioning changed')
    require(n['cuda_events_used'] is False and n['cupti_used'] is False and n['queries_inside_copy_timer'] is False and n['post_state_used_as_prediction_input'] is False,'bare/causality contract violated')
    require(n['native_repeat_consistency']['passed'] and n['verification_native']['copy_values_equal'],'full retained mathematical execution failed')
    import numpy as np
    oracle=read(binding['reference']['path']);ports=n['verification_native']['ports'];require(set(oracle)==set(ports),'independent numerical port inventory differs')
    for key,values in oracle.items():require(np.allclose(np.asarray(ports[key]),np.asarray(values),rtol=5e-5,atol=5e-5),'independent retained port mismatch '+key)
    require({x['bytes'] for x in n['slice_copy_oracles']}==set(r['sizes_bytes']),'slice qualification oracle inventory missing')
    for x in n['slice_copy_oracles']:require(x['selected_range_initially_different'] and x['selected_range_equal_after_copy'] and x['outside_range_still_poison'] and x['full_arena_bytes_checked']==2097152,'vacuous or failed slice oracle')
    rows=n['bursts'];count=12*len(r['sizes_bytes'])*2;require(len(rows)==count,'missing planned bare/query bursts');hz=n['clock']['frequency_hz'];ledger=0
    for index,row in enumerate(rows):
        trial,slot_control=divmod(index,len(r['sizes_bytes'])*2);slot,control=divmod(slot_control,2);size_index=r['orders'][trial][slot]
        require((row['trial'],row['size_slot'],row['control_slot'],row['size_index'],row['bytes'],row['control'])==(trial,slot,control,size_index,r['sizes_bytes'][size_index],r['query_orders'][trial*len(r['sizes_bytes'])+slot][control]),'balanced size/control order differs')
        require(row['prime_begin_tick']<=row['prime_end_tick']<=row['burst_begin_tick']<row['burst_end_tick'] and len(row['copies'])==16,'copy clock/inventory differs')
        last=row['burst_begin_tick'];suffix=0
        for ordinal,c in enumerate(row['copies']):
            require(c['ordinal']==ordinal and last<=c['begin_tick']<c['end_tick']<=row['burst_end_tick'],'copy order/clock differs')
            require(abs((c['end_tick']-c['begin_tick'])*1e9/hz-c['wall_ns'])<=1,'QPC duration differs');last=c['end_tick'];suffix+=c['wall_ns'] if ordinal else 0;ledger+=1
        require(row['first_copy_ns']==row['copies'][0]['wall_ns'] and row['suffix_15_sum_ns']==suffix,'first/suffix partition differs');classify(row)
    return dict(passed=True,copy_intervals=ledger,bare_completed_boundaries=True,independent_value_oracle=True,performance_accuracy_certified=False)

def collect(out,split):
    out=Path(out).resolve();m=read(out/'manifest.json');require(split in SIZES,'unknown split')
    for s in m['sources']:
        for k in ('source','archived'):require(sha(s[k]['path'])==s[k]['sha256'],'frozen source changed')
    for r in [m['binary'],*m['runtime']]:require(sha(r['path'])==r['sha256'],'runtime/binary changed')
    for binding in m['programs'].values():
        for name in ('program','reference'):require(sha(binding[name]['path'])==binding[name]['sha256'],'retained mathematical input/reference changed before acquisition')
    if split=='heldout':
        validate_freeze(out,read(out/'frozen-barrier.json'))
    marker=out/(split+'-started.json');require(not marker.exists(),'no retries/repeated split acquisition');write(marker,dict(started_at_utc=old.e2a.utc(),split=split))
    records=[]
    with old.run.measurement_lock():
        for job in (j for j in m['jobs'] if j['split']==split):
            folder=out/'jobs'/job['id'];require(not folder.exists(),'job exists; no retry');folder.mkdir(parents=True);write(folder/'request.json',job['request'])
            binding=m['programs'][split];command=[m['binary']['path'],binding['program']['path'],str(folder/'request.json'),str(folder/'result.json')]
            record=dict(schema='link-copy-size-execution-1',job_id=job['id'],context_id=job['id'],split=split,status='failed',request=receipt(folder/'request.json'),command=command)
            handles=[];p=None
            try:
                env=old.run.child_environment(m['runtime_dirs']);env['OPENBLAS_NUM_THREADS']='1';p,a,b,affinity=old.run.launch(command,env,folder/'stdout.txt',folder/'stderr.txt',4294967295);handles=[a,b];record.update(process_id=p.pid,process_affinity=affinity)
                code=p.wait(timeout=180);record['returncode']=code;require(code==0,'native failed; no retry/substitution');n=read(folder/'result.json');record['verification']=verify(n,job['request'],binding,p.pid)
                actual=n['nvml'].get('actual_module_path')
                if actual:
                    actual_receipt=receipt(actual);require(m['nvml_module'].get('sha256')==actual_receipt['sha256'],'actual NVML module differs from frozen receipt');record['actual_nvml_module']=actual_receipt;record['actual_nvml_metadata']=n['nvml']
                record.update(status='completed',native=receipt(folder/'result.json'))
            except BaseException as error:
                record['error']=repr(error)
                if p is not None and p.poll() is None:
                    p.terminate()
                    try:p.wait(timeout=10)
                    except subprocess.TimeoutExpired:p.kill();p.wait(timeout=10)
            finally:
                for f in handles:f.close()
                record['process_exited_verified']=p is None or p.poll() is not None;write(folder/'execution.json',record);records.append(record)
            print(json.dumps(dict(job_id=job['id'],status=record['status'],error=record.get('error'))),flush=True)
    write(out/(split+'-executions.json'),records);return dict(declared=3,completed=sum(r['status']=='completed' for r in records),records=str(out/(split+'-executions.json')))
def summarize(out,split):
    out=Path(out).resolve();m=read(out/'manifest.json');records=[];rows=[]
    for e in read(out/(split+'-executions.json')):
        if e['status']!='completed':continue
        require(sha(e['native']['path'])==e['native']['sha256'],'raw native changed');n=read(e['native']['path']);records.append(dict(context_id=e['context_id'],bursts=n['bursts']))
        for control in ('bare','state_queries'):
            for size in SIZES[split]:
                samples=[b for b in n['bursts'] if b['control']==control and b['bytes']==size]
                for part in ('first','suffix15'):
                    values=[b['first_copy_ns'] if part=='first' else b['suffix_15_sum_ns']/15 for b in samples]
                    gaps=[(b['burst_begin_tick']-b['prime_end_tick'])*1e9/n['clock']['frequency_hz'] for b in samples]
                    rows.append(dict(context_id=e['context_id'],bytes=size,control=control,part=part,bursts=len(values),mean_ns=statistics.mean(values),cv=statistics.pstdev(values)/statistics.mean(values),slow_samples_removed=False,observed_prime_end_to_burst_begin_ns=gaps,requested_zero_idle_is_exact_zero_gap=False))
    report=dict(schema='link-copy-size-profile-summary-1',split=split,declared_contexts=3,completed_contexts=len(records),rows=rows,exact_state_coverage=coverage(records) if split=='profile' else None,
        no_parameters_fitted=True,no_theoretical_link_rate=True,only_registered_zero_idle=True,whole_LLM_accuracy_certified=False)
    write(out/(split+'-summary.json'),report);return report
def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','profile','heldout','summarize-profile','summarize-heldout'));p.add_argument('--out',required=True);p.add_argument('--binary');p.add_argument('--nvml-dll',default='C:/Windows/System32/nvml.dll');a=p.parse_args()
    if a.command=='prepare':r=prepare(a.out,a.binary,a.nvml_dll)
    elif a.command.startswith('summarize-'):r=summarize(a.out,a.command.split('-',1)[1])
    else:r=collect(a.out,a.command)
    print(json.dumps(r,ensure_ascii=False))
if __name__=='__main__':main()
