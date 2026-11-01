"""Profile-only local runtime response labels; never fit whole-target latency.

Four states describe this exact offered-actor protocol and affinity scope.
They are not measured global available FLOPS. Cross-boundary observations are
retained separately and cannot become a steady-state average by their start.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction

from .runtime_stage_records import (coordinate_sha256, core_scope, index_callbacks,
                                    ordered_stages, require)

STATES = ((0,0),(1,0),(0,1),(1,1))


def _declared_state(value):
    require(isinstance(value, (tuple, list)) and len(value)==2
            and all(type(x) is int and x in (0,1) for x in value),
            "explicit declared CPU/GPU offered-state bits required")
    return tuple(value)


def offered_state(native, begin_tick, end_tick):
    require(begin_tick <= end_tick, "reversed stage time")
    hz = native["clock"]["frequency_hz"];origin = native["target_expected_origin_tick"]
    begin = Fraction((begin_tick-origin)*1000, hz);end = Fraction((end_tick-origin)*1000, hz)
    state, crossings = [], []
    for device in ("cpu", "gpu"):
        pulses = native["request"][device+"_actor"]["intervals_ms"]
        bit = any(a <= begin < b for a,b in pulses)
        state.append(int(bit))
        for a,b in pulses:
            for boundary in (a,b):
                if begin < boundary < end:
                    crossings.append(dict(device=device, planned_boundary_ms=boundary))
        # Ending exactly at a release is consistent with the preceding state.
        if end>begin:
            tail = any(a < end <= b for a,b in pulses)
            require(crossings or bool(bit)==bool(tail), "unclassified state boundary")
    return dict(state=None if crossings else tuple(state), crossings=crossings,
                state_semantics="offered owned actor schedule, not measured available capacity",
                observed_target_times_used_for="profile stage labels only; prediction must use its own recursive clock")


def full_scope(native, binding):
    """Caller supplies verified collector/runtime/topology identities, not labels."""
    require(isinstance(binding, dict), "verified source/runtime/topology binding required")
    def sha(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
    for key in ('native_sha256','execution_sha256','runtime_binding_sha256','context_sha256','topology_sha256'):
        require(sha(binding.get(key)), "verified SHA binding missing: "+key)
    require(type(native.get('process_id')) is int and native['process_id']>0 and binding.get('process_id')==native['process_id'], "verified process identity missing")
    creation=binding.get('process_creation_time_100ns');require(type(creation) is int and creation>0, "verified process creation time required for fresh-context independence")
    placement=core_scope(native)
    for field in ('target_physical_cores','CPU_actor_physical_cores','GPU_actor_physical_cores'):
        values=binding.get(field)
        require(isinstance(values,(tuple,list)) and bool(values) and len(values)==len(set(values)) and all(type(v) is int and v>=0 for v in values), "verified physical core mapping missing: "+field)
        placement[field]=tuple(values)
    placement['topology_sha256']=binding['topology_sha256']
    gpu_mask=native['actors']['gpu']['thread_affinity']['observed_mask']
    require(type(gpu_mask) is int and gpu_mask>0, "actual GPU actor submission affinity missing")
    placement['GPU_actor_submission_thread_mask']=gpu_mask
    actors={}
    for device in ('cpu','gpu'):
        spec=native['request'][device+'_actor']
        require(isinstance(spec.get('component'),dict) and spec.get('device')==device, "explicit actor workload missing")
        actors[device]={k:v for k,v in spec.items() if k!='intervals_ms'}
    scope=dict(context_sha256=binding['context_sha256'],runtime_binding_sha256=binding['runtime_binding_sha256'],
               native_source_sha256=native['source_sha256'],callback_source_sha256=native['request']['callback_source_sha256'],
               placement=placement,actor_workload_sha256=coordinate_sha256(actors),actor_workloads=actors,
               init_policy=native['init_policy'],boundary='inclusive_host_response_partition')
    identity=(native['process_id'],creation)
    evidence=dict(native_sha256=binding['native_sha256'],execution_sha256=binding['execution_sha256'],
                  program_sha256=native['program_sha256'],process_id=native['process_id'],process_creation_time_100ns=creation)
    return scope,identity,evidence


def profile_observations(native, *, context_id, dataset_role, stream_roles, expected_API_counts, source_binding=None):
    require(dataset_role == "profile", "heldout action times cannot enter calibration")
    require(isinstance(context_id, str) and context_id, "fresh profile context identity required")
    require(native.get("schema") in ("dynamic-external-callback-native-6", "dynamic-external-stage-native-8"), "explicit supported native stage schema required")
    constant_state = native.get('schema')=='dynamic-external-stage-native-8'
    declared = None
    if constant_state:
        require(native.get('request',{}).get('split')=='profile' and native.get('profile_window_contained') is True
                and native.get('target_entry_controller_passed') is True,
                'complete constant-state profile window and fixed-entry controller required')
        declared = _declared_state(native['request'].get('offered_state'))
    require(native.get("instrumentation") == "runtime_callbacks" and native.get("performance_parameters_fitted") is False,
            "actual callback profile observation required")
    callbacks = native["runtime_callbacks"];indexed = index_callbacks(callbacks);hz = native["clock"]["frequency_hz"]
    attention = native["attention_action_index"];scope,independence,evidence=full_scope(native,source_binding);seen = set();sequence = None
    observations, transitions = [], []
    for block in native["blocks"]:
        for phase in block["actions"]:
            if phase["action_index"] != attention: continue
            pid = phase["callback_phase_id"];require(pid not in seen, "target phase repeated");seen.add(pid)
            if constant_state:
                phase_state = offered_state(native, phase['begin_tick'], phase['end_tick'])
                require(not phase_state['crossings'] and phase_state['state']==declared,
                        'complete v8 profile action must remain in its explicitly declared state')
            selected = indexed.get(pid, [])
            counts = Counter(callbacks["metadata"][x["metadata_index"]]["name"] for x in selected)
            require(dict(counts) == expected_API_counts, "actual per-action API coordinate count changed")
            stages = ordered_stages(callbacks, dict(phase_id=pid, begin_tick=phase["begin_tick"], end_tick=phase["end_tick"]),
                                    hz, stream_roles=stream_roles, phase_records=selected)["stages"]
            signature = [(x["coordinate_sha256"], x["role"]) for x in stages]
            if sequence is None: sequence=signature
            require(signature == sequence, "shape/runtime API+gap sequence changed; separate explicit program scope required")
            for stage in stages:
                state = offered_state(native, stage["begin_tick"], stage["end_tick"])
                if constant_state:
                    require(not state['crossings'] and state['state']==declared,
                            'v8 API/gap observation crosses or differs from the declared steady state')
                record = dict(context_id=context_id, source_phase_id=pid, coordinate_sha256=stage["coordinate_sha256"],
                              role=stage["role"], coordinate=stage["coordinate"], state=state["state"],
                              response_ns=Fraction((stage["end_tick"]-stage["begin_tick"])*10**9,hz),
                              begin_tick=stage["begin_tick"], end_tick=stage["end_tick"], scope=scope)
                if state["crossings"]:
                    transitions.append(dict(record, planned_crossings=state["crossings"]))
                else: observations.append(record)
    require(seen == set(indexed), "callback phases and target action inventory differ")
    require(len(seen) == native["continuous_replays"], "complete continuous replay inventory required")
    return dict(dataset_role="profile", context_id=context_id, scope=scope, stage_sequence=sequence,
                state_protocol='constant_state_v8' if constant_state else 'mixed_state_v6',
                declared_state=declared,
                independence_identity=independence,source_evidence=evidence,
                observations=observations, cross_state_observations=transitions,
                whole_target_response_used_for_fitting=False, actual_phase_boundaries_are_prediction_reanchors=False,
                coordinate_sequence_sha256=coordinate_sha256(sequence), parameters_qualified=False)


def aggregate_cells(datasets, *, minimum_contexts=3, minimum_samples_per_context=10):
    """Transparent local arithmetic means, still unqualified without observer gate.

This produces data for RuntimeCell; no end-to-end duration or target identity
enters a cell's value. No clipping, outlier removal or mean compensation.
"""
    require(datasets and type(minimum_contexts) is int and minimum_contexts>=3, "at least three fresh contexts required")
    require(type(minimum_samples_per_context) is int and minimum_samples_per_context>=1, "positive local sample support required")
    first=datasets[0];scope=first["scope"];sequence=first["stage_sequence"]
    protocol=first.get('state_protocol','mixed_state_v6')
    require(protocol in ('mixed_state_v6','constant_state_v8'), 'unsupported state acquisition protocol')
    require(all(d.get('state_protocol','mixed_state_v6')==protocol for d in datasets),
            'constant-state and mixed-state acquisition protocols cannot be pooled')
    contexts=[d["context_id"] for d in datasets];require(len(contexts)==len(set(contexts)), "duplicate calibration context")
    independence=[tuple(d['independence_identity']) for d in datasets]
    require(len(independence)==len(set(independence)), "same native process cannot count as multiple fresh contexts")
    require(len({d['source_evidence']['native_sha256'] for d in datasets})==len(datasets), "same native record cannot be relabeled as independent contexts")
    # v8 intentionally acquires three fresh processes for each constant state.
    # Their state membership comes only from a declared and validated protocol,
    # never from which measured rows happened to be present.
    expected = {state:list(contexts) for state in STATES}
    if protocol=='constant_state_v8':
        expected = {state:[] for state in STATES}
        for dataset in datasets:
            state = _declared_state(dataset.get('declared_state'))
            require(not dataset['cross_state_observations'],
                    'constant-state v8 profile cannot contain transition observations')
            expected[state].append(dataset['context_id'])
        require(all(len(expected[state])>=minimum_contexts for state in STATES),
                'all four declared state groups require at least three independent contexts')
    cells=defaultdict(list)
    for dataset in datasets:
        require(dataset["dataset_role"]=="profile" and dataset["scope"]==scope and dataset["stage_sequence"]==sequence,
                "profile/affinity/semantic scopes cannot be pooled")
        require(dataset["whole_target_response_used_for_fitting"] is False, "whole-target compensation prohibited")
        for row in dataset["observations"]:
            require(row["scope"]==scope and row["context_id"]==dataset["context_id"] and row["state"] is not None,
                    "observation scope/state differs")
            state = _declared_state(row['state'])
            if protocol=='constant_state_v8':
                require(state==_declared_state(dataset['declared_state']),
                        'observation differs from its declared constant-state context')
            cells[(row["coordinate_sha256"],row["role"],state)].append(row)
    required={(coordinate,role,state) for coordinate,role in sequence for state in STATES}
    result=[]
    for coordinate,role,state in sorted(required):
        rows=cells[coordinate,role,state];counts=Counter(x["context_id"] for x in rows)
        expected_contexts=expected[state]
        supported=len(counts)>=minimum_contexts and all(counts[c]>=minimum_samples_per_context for c in expected_contexts)
        total=sum((x["response_ns"] for x in rows),Fraction(0))
        result.append(dict(coordinate_sha256=coordinate,role=role,state=state,
                           mean_ns=total/len(rows) if supported else None,
                           observations=len(rows),counts_by_context=dict(counts),supported=supported,
                           expected_contexts=list(expected_contexts),
                           missing_contexts=[c for c in expected_contexts if counts[c]<minimum_samples_per_context]))
    return dict(schema="profile-local-runtime-response-cells-1",scope=scope,cells=result,
                state_protocol=protocol,
                expected_contexts_by_state=[dict(state=state,context_ids=expected[state]) for state in STATES],
                declared_cells=len(required),supported_cells=sum(x["supported"] for x in result),
                all_cells_supported=all(x["supported"] for x in result),
                transition_observations_retained=sum(len(d["cross_state_observations"]) for d in datasets),
                qualified=False,observer_gate_still_required=True,
                whole_target_duration_used=False,heldout_data_used=False,clipping_or_outlier_removal=False)


def to_model_objects(cell_record, stage_sequence, *, profile_evidence, program_evidence):
    """Bridge to the existing modeling engine; qualification remains external.

Hash the full verified performance scope, including workload/runtime provenance
and GPU actor placement, rather than dropping these to two CPU masks.
"""
    from llm_infer_model.performance.runtime_response import (
        HostPlacement, RuntimeCell, RuntimeProfile, RuntimeProgram, RuntimeScope, RuntimeStage)
    require(cell_record.get('schema')=='profile-local-runtime-response-cells-1', 'local phase cell schema required')
    require(cell_record.get('qualified') is False and cell_record.get('whole_target_duration_used') is False
            and cell_record.get('heldout_data_used') is False, 'profiling bridge cannot silently certify or fit heldout totals')
    scope=cell_record['scope'];placement=scope['placement']
    host=HostPlacement(placement['target_thread_mask'],placement['CPU_actor_thread_mask'],
                       tuple(placement['target_physical_cores']),tuple(placement['CPU_actor_physical_cores']),placement['topology_sha256'])
    model_scope=RuntimeScope(coordinate_sha256(scope),host,boundary=scope['boundary'])
    require(type(profile_evidence) is tuple and profile_evidence and type(program_evidence) is tuple and program_evidence,
            'explicit profile and program evidence tuples required')
    stages=tuple(RuntimeStage(f'{role}_{index:04d}',coordinate,role) for index,(coordinate,role) in enumerate(stage_sequence))
    prices=tuple(RuntimeCell(row['coordinate_sha256'],row['role'],tuple(row['state']),row['mean_ns'],profile_evidence)
                 for row in cell_record['cells'])
    return RuntimeProgram(model_scope,stages,program_evidence),RuntimeProfile(model_scope,prices,profile_evidence,qualified=False)
