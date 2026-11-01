"""Decode actual ServingSim ETs and compare a matched target event interval.

This imports the native protocol DAG into modeling's event evaluator with the
same declared compute prices and analytical memory formula. It is a common-
cost semantics check, not independent hardware accuracy or a high-level IR port.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
for folder in (ROOT,ROOT/'modeling'):
    sys.path.insert(0,str(folder))


def check(build,run_id,attempt):
    build=Path(build).resolve()
    sys.path.insert(0,str(build/'site'))
    from chakra.schema.protobuf.et_def_pb2 import GlobalMetadata,Node,NodeType
    from chakra.src.third_party.utils.protolib import decodeMessage
    from llm_infer_model.performance.graph import Graph,Task
    from llm_infer_model.performance.parameters import Parameter,ParameterBundle
    from llm_infer_model.performance.evaluator import evaluate
    import google.protobuf

    inputs=build/'ServingSim/astra-sim/inputs/runs'/run_id
    out=build/attempt
    run=json.loads((out/'summary.json').read_text())
    assert run['status']=='completed' and run['returncode']==0
    files=[]
    for path in inputs.rglob('*.et'):
        with path.open('rb') as stream:
            metadata=GlobalMetadata();assert decodeMessage(stream,metadata)
            nodes=[]
            while True:
                node=Node()
                if not decodeMessage(stream,node):break
                attrs={}
                for item in node.attr:
                    if item.name=='tensor_size':value=int(item.uint64_val)
                    elif item.name in ('tensor_loc','tensor_device','tensor_channel'):value=int(item.uint32_val)
                    elif item.name=='is_cpu_op':value=bool(item.bool_val)
                    elif item.name=='num_ops':value=int(item.int64_val)
                    else:
                        fields={field.name:value for field,value in item.ListFields() if field.name!='name'}
                        value=fields
                    attrs[item.name]=value
                nodes.append(dict(id=int(node.id),name=node.name,type=NodeType.Name(node.type),
                                  duration_field=int(node.duration_micros),
                                  data_deps=list(node.data_deps),ctrl_deps=list(node.ctrl_deps),attrs=attrs))
        files.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),nodes=nodes))
    target=next(row for row in files if 'instance0_batch0' in row['path'])
    bootstrap=next(row for row in files if '/event_handler/' in row['path'])
    assert len(bootstrap['nodes'])==1 and bootstrap['nodes'][0]['type']=='COMP_NODE'
    bootstrap_duration=bootstrap['nodes'][0]['duration_field']
    assert bootstrap_duration>0
    memory=json.loads((inputs/'memory/memory_expansion.json').read_text())['remote_mem']
    assert memory['memory-type']=='PER_NODE_MEMORY_EXPANSION' and memory['num-devices']==1
    tasks=[];parameters=[];cost_rows=[]
    for node in target['nodes']:
        if node['type']=='COMP_NODE':
            duration=node['duration_field']
            assert duration>0 and not node['attrs'].get('is_cpu_op',False)
            resource='single_gpu'
        elif node['type'] in ('MEM_LOAD_NODE','MEM_STORE_NODE'):
            assert node['attrs']['tensor_loc']==2
            size=node['attrs']['tensor_size']
            duration=memory['mem-latency']+int(size/memory['mem-bw'])
            resource='single_remote_memory_queue'
        else:
            raise RuntimeError('Unmodeled ET node: '+node['type'])
        key='node-'+str(node['id'])
        parameters.append(Parameter(key,duration,'ns','declared',(target['sha256'],),
                                    'Exact native ServingSim input prices/memory formula; no real-hardware measurement'))
        tasks.append(Task(key,resource,depends_on=tuple('node-'+str(x) for x in sorted(set(node['data_deps']+node['ctrl_deps']))),fixed_parameter=key))
        cost_rows.append(dict(node_id=node['id'],type=node['type'],duration_ns=duration,
                              tensor_bytes=node['attrs'].get('tensor_size'),resource=resource))
    graph=Graph(tuple(tasks))
    evaluation=evaluate(graph,ParameterBundle(tuple(parameters)))
    event_key='native-first-arrival-event'
    event_parameter=Parameter(event_key,bootstrap_duration,'ns','declared',
                              (bootstrap['sha256'],),'Native event_handler ET duration, not a fitted constant')
    whole_graph=Graph((Task(event_key,'native_frontend_event',fixed_parameter=event_key),
                       *(replace(task,depends_on=(event_key,)) if not task.depends_on else task for task in tasks)))
    whole_evaluation=evaluate(whole_graph,ParameterBundle((event_parameter,*parameters)))
    text=(out/'stdout.txt').read_text()
    iterations=[(int(i),int(t.replace(',',''))) for i,t in re.findall(r'iteration (\d+) finished, ([0-9,]+) cycles',text)]
    assert len(iterations)==2 and [r[0] for r in iterations]==[0,1]
    start,end=iterations[0][1],iterations[1][1]
    assert start==bootstrap_duration
    target_elapsed=end-start
    difference=evaluation.duration_ns-target_elapsed
    summary=dict(schema='dac-followup-actual-backend-comparison-1',status='pass' if difference==0 else 'difference_retained',
                 actual_complete_cli_jobs=1,actual_native_backend_target_timings=1,
                 native_backend='Pinned ASTRA analytical C++17 Release',
                 native_cli='Pinned LLMServingSim scheduler/trace/Chakra/controller',
                 fixture_profiles=True,real_hardware_accuracy_claimed=False,
                 native_cpp_binary_sha256=run['native_cpp_binary_sha256'],
                 native_target_start_ns=start,native_target_end_ns=end,
                 native_target_elapsed_ns=target_elapsed,ours_modeling_event_dag_ns=evaluation.duration_ns,
                 difference_ns=difference,common_component_and_memory_costs=True,
                 cost_refit_from_target=False,explicit_bootstrap_constant_added=False,
                 bootstrap_interval_outside_comparison=True,bootstrap_tick_is_observed_not_a_universal_constant=True,
                 bootstrap_et_duration_ns=bootstrap_duration,
                 bootstrap_node=bootstrap['nodes'][0],
                 start_boundary_basis='Native router clamps first arrival to at least 1 ns; explicit event_handler ET has that duration',
                 native_protocol_total_ns=end,
                 ours_imported_protocol_total_ns=whole_evaluation.duration_ns,
                 whole_protocol_difference_ns=whole_evaluation.duration_ns-end,
                 whole_protocol_import_contains_explicit_event_task=True,
                 node_types=dict(Counter(n['type'] for n in target['nodes'])),
                 native_compute_ns=sum(r['duration_ns'] for r in cost_rows if r['type']=='COMP_NODE'),
                 memory_nodes=[r for r in cost_rows if r['type']!='COMP_NODE'],
                 protobuf_python=google.protobuf.__version__,
                 target_et_sha256=target['sha256'],highlevel_model_dense_frontend_coverage=False,
                 strategy_ir_compiler_comparison=False,physical_allocator_accuracy_claimed=False,
                 interpretation='Actual whole request and native backend run; same ET dependencies and fixed cost contract align in modeling event evaluator. Startup tick excluded by independently reported target boundary. No independent measured performance or broad strategy coverage claim.')
    for name,value in [('decoded-ets.json',files),('imported-event-graph.json',graph.to_dict()),
                       ('imported-event-evaluation.json',evaluation.to_dict()),('node-costs.json',cost_rows),
                       ('imported-whole-protocol-graph.json',whole_graph.to_dict()),
                       ('imported-whole-protocol-evaluation.json',whole_evaluation.to_dict()),
                       ('backend-comparison.json',summary)]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(summary))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--build',required=True)
    parser.add_argument('--run-id',default='whole-l2-p16-v2')
    parser.add_argument('--attempt',default='whole-cli-002')
    args=parser.parse_args()
    check(args.build,args.run_id,args.attempt)
