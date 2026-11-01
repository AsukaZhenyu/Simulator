"""Independent allocation ledger: no simulator imports and no summaries."""
from collections import Counter

def check(value,message):
    if not value:raise AssertionError(message)

def replay(declaration, expected, events, makespan):
    tensors=expected['tensors'];ops=expected['operations']
    allocations={};ready=set();running=[];completed=set();now=0
    peak=Counter();payload=Counter();counts=Counter();samples=[]
    for tid,tensor in tensors.items():
        for location in tensor['initial_locations']:
            allocations[tid,location]=tensor['storage_bytes'];ready.add((tid,location))
    def record():
        usage={loc:sum(size for (tid,l),size in allocations.items() if l==loc)+
            sum(task['workspace'] for task in running if task['location']==loc) for loc in ('dram','vram')}
        for loc,size in usage.items():
            check(size<=declaration['hardware'][loc+'_capacity_bytes'],f'{loc} exceeds physical capacity')
            peak[loc]=max(peak[loc],size)
        samples.append(dict(t_ns=now,**usage))
    def finish(t):
        nonlocal running,now
        check(t>=now,'trace time reversed')
        remaining=[]
        for task in running:
            if task['end']>t:remaining.append(task);continue
            ready.update(task['outputs'])
            if task['operation']:completed.add(task['operation'])
        running=remaining;now=t
    record()
    for e in events:
        finish(e['t_ns']);kind=e['kind'];counts[kind]+=1;tid=e.get('target_id')
        if kind=='ADVANCE':pass
        elif kind=='EVICT':
            key=(tid,e['location'])
            check(key in ready,'eviction of absent or pending allocation')
            check(all(key not in task['inputs'] for task in running),'eviction before final reader completes')
            size=allocations.pop(key);ready.remove(key)
            check(e.get('bytes_released')==size,'release byte count differs from declared tensor bytes')
        elif kind in {'COPY','COMPUTE'}:
            if kind=='COPY':
                inputs=[(tid,e['src'])];outputs=[(tid,e['dst'])];location=e['dst'];workspace=0;operation=None
                byte_count=tensors[tid]['storage_bytes']
                check(e['bytes_transferred']==byte_count,'copy payload differs from independent shape bytes')
                payload[e['src']+'>'+e['dst']]+=byte_count
            else:
                operation=tid;op=ops[tid];location='dram' if e['device']=='cpu' else 'vram'
                inputs=[(t,location) for t in op['inputs']]
                outputs=[(t,location) for t in [op['output'],*op['extra_outputs']]]
                refined=op['semantic_op'].startswith(('ffn_','attention_'))
                workspace=4 if refined else 16
                check(e.get('workspace_bytes')==workspace,'workspace differs from preregistered contract')
                check(operation not in completed and all(task['operation']!=operation for task in running),'duplicate compute')
            check(all(key in ready for key in inputs),'read before tensor is ready at executing endpoint')
            check(all(key not in allocations for key in outputs),'overwrite of live allocation')
            for key in outputs:allocations[key]=tensors[key[0]]['storage_bytes']
            running.append(dict(end=e['end_ns'],inputs=inputs,outputs=outputs,workspace=workspace,location=location,operation=operation))
        else:raise AssertionError('Unsupported physical action in separate-pool protocol: '+kind)
        record()
    finish(makespan);record()
    check(not running and len(completed)==len(ops),'unfinished independent physical replay')
    check(all(any((tid,loc) in ready for loc in ('dram','vram')) for tid in expected['terminal_hidden']),'missing terminal hidden output')
    return dict(h2d_bytes=payload['dram>vram'],d2h_bytes=payload['vram>dram'],
        peak_dram_bytes=peak['dram'],peak_vram_bytes=peak['vram'],event_counts=dict(counts),
        completed_operations=len(completed),samples=samples,
        contract='Initial and pending output allocations, both transfer endpoints, live-reader leases, exact declared workspace; no allocator fragmentation or undocumented runtime allocations claimed.')
