"""Whole-block external comparisons with explicit common-cost boundaries.

The imported block is the external implementation's native symbolic graph, not
Model.dense's Qwen-like block. Three declared macro contracts package its 12
priced operators. Fixed strategy IR -> compiler -> tensor engine is exercised;
no new numerical kernel, performance accuracy or external superiority is claimed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "modeling", ROOT / "mapping", ROOT / "src"):
    sys.path.insert(0, str(folder))

from llm_infer_model.tensor.spec import (Architecture, ComputeCost, DTYPE_F16,
    M1Costs, M1Goal, M1Mapspace, MapperConfig, Operation, Origin, Scenario,
    Tensor, TransferCost, Workload, ggml_contiguous_layout, _canonical_workload)
from llm_infer_model.tensor.engine import (ACTION_ADVANCE, ACTION_COMPUTE,
    ACTION_COPY, ACTION_EVICT, initial_state, is_goal, transition, used_bytes)
from tensor_mapping.schedule import (ScheduleFacts, SchedulePolicy,
    make_schedule_context, plan_schedule)
from tensor_mapping.strategy_ir import (Instruction, Region, StrategyIR,
    strategy_to_document)
from tensor_mapping.strategy_compiler import (compile_strategy, run_strategy,
    compiled_document, compile_strategy_file)
from tensor_mapping.artifacts import events_document, states_document

PINS = {"LLMCompass":"2e015fd2ee750e6cad8d7152df52551e5b41ef20",
        "LLMServingSim":"a4053bc1161872420e1e0607cb3409ef659b828e"}
DEFAULT_EXTERNAL = ROOT / "build/dac-validation-20261025/e1-external-v2"
PLACEMENTS = {
    "all_cpu":{"projection":"cpu", "attention":"cpu", "dense_ffn":"cpu"},
    "all_gpu":{"projection":"gpu", "attention":"gpu", "dense_ffn":"gpu"},
    "cpu_attention":{"projection":"gpu", "attention":"cpu", "dense_ffn":"gpu"},
    "cpu_ffn":{"projection":"gpu", "attention":"gpu", "dense_ffn":"cpu"},
}


def validate_native_case(case):
    groups={'projection':['Q_proj','K_proj','V_proj'],
            'attention':['Q_mul_K','A_softmax','A_mul_V','H_matmul0','layer_norm0'],
            'dense_ffn':['H_matmul1','H_gelu','H_matmul2','layer_norm1']}
    rows=case['operations']
    if any(type(case[key]) is not int or case[key]<1 for key in ('hidden','tokens','heads','intermediate')) or case['hidden']%case['heads']:
        raise ValueError('Invalid native block geometry')
    if case['dtype']!='fp16' or case['intermediate']!=4*case['hidden']:
        raise ValueError('Native case does not match declared common FP16 GeLU block')
    if len(rows)!=12 or len({r['id'] for r in rows})!=12:
        raise ValueError('Missing or duplicate native operator contract')
    for macro,names in groups.items():
        if {r['id'] for r in rows if r['macro']==macro}!=set(names):
            raise ValueError('Incomplete native macro operator coverage: '+macro)
    if any(type(r.get('inclusive_seconds')) not in (int,float) or not math.isfinite(r['inclusive_seconds']) or r['inclusive_seconds']<=0 for r in rows):
        raise ValueError('Unknown or invalid native operator cost')
    h,q,n=case['hidden'],case['tokens'],case['heads']
    d=h//n
    shapes={
        **{name:([[1,q,h],[h,h]],[1,q,h]) for name in ('Q_proj','K_proj','V_proj','H_matmul0')},
        'Q_mul_K':([[1,n,q,d],[1,n,d,q]],[1,n,q,q]),
        'A_softmax':([[1,n,q,q]],[1,n,q,q]),
        'A_mul_V':([[1,n,q,q],[1,n,q,d]],[1,n,q,d]),
        'layer_norm0':([[1,q,h]],[1,q,h]),
        'H_matmul1':([[1,q,h],[h,4*h]],[1,q,4*h]),
        'H_gelu':([[1,q,4*h]],[1,q,4*h]),
        'H_matmul2':([[1,q,4*h],[4*h,h]],[1,q,h]),
        'layer_norm1':([[1,q,h]],[1,q,h]),
    }
    for row in rows:
        expected_inputs,expected_output=shapes[row['id']]
        if row['input_shapes']!=expected_inputs or row['output_shape']!=expected_output:
            raise ValueError('Native shape contract differs: '+row['id'])
        if row['logical_input_bytes']!=[math.prod(s)*2 for s in expected_inputs] or row['logical_output_bytes']!=math.prod(expected_output)*2:
            raise ValueError('Native byte contract differs: '+row['id'])
    if 'native_metadata_operations' in case:
        expected_metadata={
            **{name:([1,q,h],[1,q,n,d],None) for name in ('Q_reshape','K_reshape','V_reshape')},
            'Q_transpose':([1,q,n,d],[1,n,q,d],[0,2,1,3]),
            'K_transpose':([1,q,n,d],[1,n,d,q],[0,2,3,1]),
            'V_transpose':([1,q,n,d],[1,n,q,d],[0,2,1,3]),
            'H_transpose':([1,n,q,d],[1,q,n,d],[0,2,1,3]),
            'H_reshape':([1,q,n,d],[1,q,h],None),
        }
        metadata=case['native_metadata_operations']
        if len(metadata)!=8 or {r['id'] for r in metadata}!=set(expected_metadata):
            raise ValueError('Incomplete native reshape/transpose contract')
        for row in metadata:
            if (row['input_shape'],row['output_shape'],row['permutation'])!=expected_metadata[row['id']]:
                raise ValueError('Native metadata shape contract differs: '+row['id'])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")


def command(argv, cwd, env=None, timeout=120):
    started = datetime.now(timezone.utc).isoformat()
    try:
        result = subprocess.run([str(x) for x in argv], cwd=cwd, env=env,
                                capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=timeout)
        return dict(argv=[str(x) for x in argv], cwd=str(cwd), started_utc=started,
                    returncode=result.returncode, stdout=result.stdout,
                    stderr=result.stderr, timeout=False)
    except subprocess.TimeoutExpired as error:
        return dict(argv=[str(x) for x in argv], cwd=str(cwd), started_utc=started,
                    returncode=None, stdout=str(error.stdout), stderr=str(error.stderr), timeout=True)


def tensor(name, ne, role, initial=()):
    nb, nbytes = ggml_contiguous_layout(DTYPE_F16, ne)
    return Tensor(name, name, role, DTYPE_F16, tuple(ne), nb, nbytes, tuple(initial))


def import_block(case, *, initial_location="dram"):
    """Package exact external operator groups at their declared typed ports.

    No op is renamed GeLU->SwiGLU, LayerNorm->RMSNorm, or MHA->GQA.
    Existing macro labels are packaging categories; detailed native equations,
    layouts and exclusions are explicit in the retained external signature.
    """
    validate_native_case(case)
    h, q, i = case["hidden"], case["tokens"], case["intermediate"]
    tensors = [tensor("X", (h,q,1,1), "input", (initial_location,))]
    for name, shape in [("Wq",(h,h,1,1)),("Wk",(h,h,1,1)),("Wv",(h,h,1,1)),
                        ("W0",(h,h,1,1)),("W1",(i,h,1,1)),("W2",(h,i,1,1))]:
        tensors.append(tensor(name, shape, "weight", (initial_location,)))
    for name in ("Q","K","V","H0","Y"):
        tensors.append(tensor(name,(h,q,1,1),"output" if name=="Y" else "intermediate"))
    operations = (
        Operation("projection","projection","EXTERNAL_COMPASS_QKV",None,(),("X","Wq","Wk","Wv"),"Q",("K","V")),
        Operation("attention","attention","EXTERNAL_COMPASS_MHA_OUTPUT_LAYERNORM",None,(),("Q","K","V","W0"),"H0"),
        Operation("dense_ffn","dense_ffn","EXTERNAL_COMPASS_GELU_FFN_LAYERNORM",None,(),("H0","W1","W2"),"Y"),
    )
    return Workload(case["id"], Origin("synthetic_layer", sample="Exact pinned external Compass macro import; not measured kernel certification"),
                    tuple(tensors), operations, ("Y",))


def make_scenario(case, *, resident_gpu=False, capacity=1<<30):
    workload = import_block(case, initial_location="vram" if resident_gpu else "dram")
    macros = {op.id:sum(row["inclusive_seconds"] for row in case["operations"] if row["macro"]==op.id)
              for op in workload.operations}
    # CPU is deliberately a declared 4x response counterfactual. No measured
    # Ryzen/A100 or real CPU/GPU hardware equivalence is inferred.
    compute = {name:{device:ComputeCost(max(1,math.ceil(seconds*1e9*factor)),0)
                           for device,factor in (("cpu",4),("gpu",1))}
               for name,seconds in macros.items()}
    ids = tuple(t.id for t in workload.tensors)
    transfers = {route:{t.id:TransferCost(copy_ns(t.storage_bytes)) for t in workload.tensors}
                 for route in ("dram_to_vram","vram_to_dram")}
    return Scenario("0.2",case["id"],workload,
                    Architecture(vram_capacity_bytes=capacity,dram_capacity_bytes=capacity),
                    M1Costs("Declared native Compass component roofline+overhead; CPU response factor4; copy5us+bytes/16GBps; not local hardware",compute,transfers),
                    M1Mapspace({op.id:("cpu","gpu") for op in workload.operations},
                               {r:ids for r in transfers},True,False,False),
                    MapperConfig("uniform_cost"),M1Goal({"Y":"vram" if resident_gpu else "dram"}))


def copy_ns(size):
    return 5000 + math.ceil(size * 1e9 / 16_000_000_000)


def scenario_document(scenario):
    arch=scenario.architecture
    return dict(schema_version='0.2',id=scenario.id,workload_file='workload.json',
                architecture=dict(dram_capacity_bytes=arch.dram_capacity_bytes,
                    vram_capacity_bytes=arch.vram_capacity_bytes,
                    dram_runtime_reserved_bytes=0,vram_runtime_reserved_bytes=0,
                    dram_alignment_bytes=1,vram_alignment_bytes=1,
                    cpu_compute_slots=1,gpu_compute_slots=1,pcie_copy_slots=1,ssd_io_slots=1),
                costs=asdict(scenario.costs),mapspace=asdict(scenario.mapspace),
                goal=asdict(scenario.goal),mapper=asdict(scenario.mapper))


def compile_schedule(scenario, placement):
    leaves = scenario.workload.leaf_tensors
    weights = frozenset(t.id for t in scenario.workload.tensors if t.role=="weight")
    sources = tuple((t, scenario.workload.tensor_by_id[t].initial_locations[0]) for t in sorted(weights))
    facts = ScheduleFacts(tuple(op.id for op in scenario.workload.operations), weights,sources,
                          frozenset(leaves-weights))
    context = make_schedule_context(scenario,facts,SchedulePolicy(0,False),placement)
    scheduled = plan_schedule(context,max_actions=500)
    if not scheduled.complete:
        raise RuntimeError("External block schedule did not complete: " + scheduled.termination_reason)
    instructions = []
    for action in scheduled.actions:
        if action.kind == ACTION_COMPUTE:
            instructions.append(Instruction("compute",operation_id=action.target_id))
        elif action.kind == ACTION_COPY:
            instructions.append(Instruction("transfer",tensor_id=action.target_id,src=action.src,dst=action.dst))
        elif action.kind == ACTION_EVICT:
            instructions.append(Instruction("release",tensor_id=action.target_id,location=action.location))
        elif action.kind == ACTION_ADVANCE:
            # No copy/compute or CPU/GPU overlap is allowed, so exactly one
            # completion is outstanding. Drain is the same wait in this scope.
            instructions.append(Instruction("drain"))
        else:
            raise RuntimeError("Unaccounted schedule action: " + str(action))
    ir = StrategyIR("strategy-ir-0.1",scenario.id,"scenario.json","offline_fixed",
                    (Region("external_block","declared_external_dense_block",tuple(placement)),),
                    placement,tuple(instructions),{})
    compiled = compile_strategy(ir,scenario)
    result = run_strategy(compiled)
    if result.evaluation.status != "valid":
        raise RuntimeError("External import failed actual IR compiler/engine: " + str(result.evaluation))
    return ir,compiled,result,scheduled.actions


def independent_serial(case, placement, *, resident_gpu=False):
    """Separate mathematical recipe/copy ledger, never reads planned actions."""
    scenario = make_scenario(case,resident_gpu=resident_gpu)
    locations = {t.id:set(t.initial_locations) for t in scenario.workload.tensors}
    expected = []
    duration = 0
    for op in scenario.workload.operations:
        device = placement[op.id]
        domain = "dram" if device=="cpu" else "vram"
        for name in op.inputs:
            if domain not in locations[name]:
                source = "dram" if "dram" in locations[name] else "vram"
                size=scenario.workload.tensor_by_id[name].storage_bytes
                duration+=copy_ns(size)
                expected.append(dict(tensor=name,src=source,dst=domain,bytes=size))
                locations[name].add(domain)
        duration+=scenario.costs.compute[op.id][device].duration_ns
        for name in op.outputs:
            locations[name].add(domain)
    goal="vram" if resident_gpu else "dram"
    if goal not in locations["Y"]:
        source="vram" if goal=="dram" else "dram"
        size=scenario.workload.tensor_by_id["Y"].storage_bytes
        duration+=copy_ns(size)
        expected.append(dict(tensor="Y",src=source,dst=goal,bytes=size))
    return duration,expected


def compare_compass(payload, out):
    comparisons=[]
    controls=[]
    for case in payload["cases"]:
        directory=out/"compass"/case["id"]
        macro_signature={kind:[row for row in case["operations"] if row["macro"]==kind]
                         for kind in ("projection","attention","dense_ffn")}
        dump(directory/"native-operator-signature.json",macro_signature)
        for name,placement in PLACEMENTS.items():
            scenario=make_scenario(case)
            ir,compiled,result,actions=compile_schedule(scenario,placement)
            expected,copies=independent_serial(case,placement)
            actual=[dict(tensor=a.target_id,src=a.src,dst=a.dst,
                         bytes=scenario.workload.tensor_by_id[a.target_id].storage_bytes)
                    for a in actions if a.kind==ACTION_COPY]
            # Copy order between independent leaves may differ; the payload
            # multiset, placement, dependencies and total must still match.
            freeze=lambda rows:Counter((r['tensor'],r['src'],r['dst'],r['bytes']) for r in rows)
            assert freeze(copies)==freeze(actual)
            assert result.evaluation.makespan_ns==expected
            leaf={t.id:t for t in scenario.workload.tensors}
            assert all(leaf[key].storage_bytes==2*case['hidden']*case['tokens'] for key in ('X','Q','K','V','H0','Y'))
            row=dict(case=case['id'],strategy=name,placement=placement,
                     engine_ns=result.evaluation.makespan_ns,independent_serial_ns=expected,
                     copy_count=len(actual),copy_bytes=sum(r['bytes'] for r in actual),
                     peak_dram_bytes=result.evaluation.peak_dram_bytes,
                     peak_vram_bytes=result.evaluation.peak_vram_bytes,
                     peak_scope='Visible external macro-port allocations only; internal score/probability/layout storage and backend workspace not measured',
                     extra_workspace_zero_is_scenario_assumption=True,
                     cpu_response_factor_is_counterfactual=4,
                     copy_contract_is_counterfactual='5us fixed + bytes/16GBps',
                     declared_pool_capacity_bytes=1<<30,
                     actual_strategy_ir_compiler=True,numerical_kernel_equivalence_verified=False,
                     status='pass')
            item=directory/name
            workload_document=_canonical_workload(scenario.workload)
            workload_document.update(schema_version='0.2',origin=asdict(scenario.workload.origin))
            dump(item/'workload.json',workload_document)
            dump(item/'scenario.json',scenario_document(scenario))
            dump(item/'strategy-ir.json',strategy_to_document(ir))
            file_replay=run_strategy(compile_strategy_file(item/'strategy-ir.json'))
            assert file_replay.evaluation.status=='valid' and file_replay.evaluation.makespan_ns==expected
            dump(item/'compiled.json',compiled_document(compiled))
            dump(item/'copies.json',actual)
            dump(item/'summary.json',row)
            dump(item/'events.json',events_document(mode='model',evaluation=result.evaluation,scenario=scenario))
            dump(item/'states.json',states_document(mode='model',evaluation=result.evaluation,scenario=scenario))
            comparisons.append(row)
        scenario=make_scenario(case,resident_gpu=True)
        _,_,result,_=compile_schedule(scenario,PLACEMENTS['all_gpu'])
        error=result.evaluation.makespan_ns-case['native_whole_block_seconds']*1e9
        assert -1e-7<=error<3.000001
        controls.append(dict(case=case['id'],native_seconds=case['native_whole_block_seconds'],
                             ours_ns=result.evaluation.makespan_ns,rounding_difference_ns=error,
                             allowed_integer_rounding_ns=3,scope='All weights/input/output resident; three priced macro costs from exact12nativeoperators',status='pass'))
    return dict(native_whole_blocks=len(payload['cases']),strategy_comparisons=len(comparisons),
                comparisons=comparisons,resident_whole_block_controls=controls,
                kernel_math_scope=payload['model_scope'],
                original_tp1_native_failure=payload['original_native_tp1_failure'],
                compatibility_adapter=payload['in_memory_adapter'],
                import_adapter=payload['unused_gpu_decorator_import_adapter'],
                original_import_failure=payload['original_import_failure'],
                declared_macro_contracts=True,operator_level_lowering_not_claimed=True,
                highlevel_model_dense_frontend_coverage=False,
                independent_accuracy_claim=False,performance_superiority_claim=False)


def run(args):
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    for name in ('dac_followup_external.py','dac_followup_external_compass.py',
                 'dac_followup_external_serving.py','dac_followup_external_audit.py'):
        target=out/'code'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/'experiments'/name).read_bytes())
    external=args.external.resolve()
    source_receipts={}
    for name,pin in PINS.items():
        source=external/name
        head=command(['git','rev-parse','HEAD'],source)
        status=command(['git','status','--porcelain','--untracked-files=no'],source)
        if head['stdout'].strip()!=pin or status['stdout'].strip():
            raise RuntimeError('External pin/core status changed: '+name)
        source_receipts[name]=dict(head=head,status=status)
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONIOENCODING='utf-8')
    env['PYTHONPATH']=str(external/'deps')+os.pathsep+env.get('PYTHONPATH','')
    payload=out/'compass-native-whole-blocks.json'
    compass_run=command([args.python,ROOT/'experiments/dac_followup_external_compass.py','--out',payload],external/'LLMCompass',env)
    dump(out/'compass-process-receipt.json',compass_run)
    if compass_run['returncode']!=0:
        raise RuntimeError('Native whole Compass block attempt failed; inspect retained receipt')
    data=json.loads(payload.read_text())
    compass=compare_compass(data,out)
    serving_run=command([args.python,ROOT/'experiments/dac_followup_external_serving.py',
                         '--out',out/'serving'],external/'LLMServingSim',env)
    dump(out/'serving-process-receipt.json',serving_run)
    if serving_run['returncode']!=0:
        raise RuntimeError('Native ServingSim whole frontend attempt failed; inspect retained receipt')
    serving=json.loads((out/'serving/summary.json').read_text())
    for name in source_receipts:
        source_receipts[name]['post_status']=command(['git','status','--porcelain','--untracked-files=no'],external/name)
        assert not source_receipts[name]['post_status']['stdout'].strip()
    summary=dict(schema='dac-followup-external-1',compass=compass,servingsim=serving,
                 actual_external_resident_whole_cost_comparisons=len(compass['resident_whole_block_controls']),
                 ours_ir_vs_independent_copy_ledger_checks=len(compass['comparisons']),
                 serving_frontend_whole_traces=serving['whole_frontend_traces'],
                 serving_astra_workload_timings=serving['native_backend_workload_timings'],
                 source_receipts=source_receipts,
                 primary_sources=['https://github.com/PrincetonUniversity/LLMCompass/tree/'+PINS['LLMCompass'],
                                  'https://github.com/casys-kaist/LLMServingSim/tree/'+PINS['LLMServingSim']],
                 external_core_changes=0,
                 interpretation='Same exact external block and shared component costs. Backend legality/composition comparison, not independent measured accuracy, kernel mapping superiority or complete paper reproduction.')
    dump(out/'summary.json',summary)
    write_report(summary,out)
    dump(out/'artifact-hashes.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print(json.dumps(dict(native_whole_blocks=compass['native_whole_blocks'],actual_ir_strategy_comparisons=compass['strategy_comparisons'])))


def write_report(summary,out):
    compass,serving=summary['compass'],summary['servingsim']
    controls=compass['resident_whole_block_controls']
    maximum=max(abs(r['rounding_difference_ns']) for r in controls)
    table=['| Native block | Compass roofline us | Our resident IR us | Rounding ns |',
           '|---|---:|---:|---:|']
    for row in controls:
        table.append(f"| {row['case']} | {row['native_seconds']*1e6:.6f} | {row['ours_ns']/1000:.6f} | {row['rounding_difference_ns']:.6f} |")
    control_table='\n'.join(table)
    text=f'''# External comparison follow-up

This campaign runs the pinned external implementation rather than using source inspection as execution evidence. It adds **{len(controls)} native whole-block cost comparisons**, **{len(compass['comparisons'])} simulator-IR versus independent copy-ledger checks**, and **{serving['whole_frontend_traces']} complete native ServingSim traces**. These are separate denominators.

## Exact common block

The Compass comparison uses its native initialization block: Q/K/V projections, MHA score/softmax/value aggregation, output projection, two LayerNorms and a GeLU FFN. It does not substitute the project's Qwen-like block, GQA, RMSNorm, SwiGLU, RoPE, residual addition or an explicit causal-mask cost. FP16 logical-port geometry is H128/H256 with Q16/Q64, on the pinned A100/MI210 specifications. These are specified hardware models, not local GPU measurements. The native MI210 GeLU cost path internally uses its FP32 vector-unit datatype; each operator receipt records this distinction without changing its native formula. Logical-port byte checks do not certify internal kernel traffic or scratch.

Twelve priced native operators are packaged in three declared macro contracts at Q/K/V, H0 and Y ports. The actual strategy-ir-0.1 parser, compiler and tensor engine validate operand availability, physical macro-port allocations, copies, waits and final goals. Saved workload/scenario/IR files are compiled and replayed from disk as well. This is a custom external-macro import; it does not establish the high-level Model.dense frontend's coverage of a different mathematical block.

The eight resident controls have the same input/weights/output locations and native component costs. The maximum whole-block difference is **{maximum:.9f} ns**, within the declared three integer-rounding steps. Agreement checks backend composition under common prices; it is neither independent timing accuracy nor a speedup over Compass. The native whole-block call is roofline_model, not kernel mapping compile_and_simulate.

{control_table}

## Explicit adaptation and gaps

The original TP=1 Compass whole-block call is attempted first and its failure retained. A one-assignment in-memory adapter initializes the null-collective diagnostic local to zero, leaving all costs unchanged; the native source tree stays unmodified. The exact original/adapted function source and hashes are retained. The eight reshape/transpose logical steps, including permutations and input/output shapes, are retained as well as the twelve priced operators. Their behavior follows the native block cost contract, which does not separately charge those operations in its whole-block roofline sum; no physically zero transpose-cost claim follows.

The original module import is also attempted and retained. If the installed PyTorch rejects Python 3.12 torch.compile decorators, a narrowly allowlisted import adapter leaves the two unused GPU helper functions undecorated and immediately restores torch.compile. Those helpers are never executed in this roofline-only campaign. This is a declared environment compatibility adaptation, not a replacement timing or numerical kernel.

The 32 placement checks share declared CPU response factor 4, COPY cost 5 us plus bytes/16 GB/s, and 1 GiB per physical pool. These pool budgets are scenario inputs rather than the commercial GPUs' memory capacities; the native whole-block cost model does not certify the paired physical-allocation ledger. They compare our actual IR against a separately derived serial demand-copy ledger. They do not show the performance of real CPU/GPU strategies in the external runtime. The reported peaks cover visible macro-port allocations; score/probability/layout buffers and native backend workspace are unmeasured and an extra zero workspace is an explicit scenario assumption. No complete-block memory-accuracy claim follows.

## ServingSim native frontend

The original Llama architecture catalog, profile-table builders/lookups, full trace generator, placement resolver and shape accounting execute for two/four layers, H128/H256, Prefill16/64 and Decode1/history128. Resident and remote FFN-weight placements are both exercised. All fixture profiles are deliberately synthetic declared layer costs. The 24 co-located traces include embedding, every block and final normalization/head/sampler. Sixteen additional Prefill trace calls check P/D KV communication bytes; these are byte-accounting witnesses without a configured paired decode peer, not full P/D runs. QKV bytes and FFN weight locations are independently checked. The case file preserves every native row and block boundary.

ASTRA backend status: **{serving['backend']['status']}**. Actual native ASTRA workload timing count is **0**. Native frontend traces are not end-to-end ServingSim performance results, do not implement a replacement ASTRA backend, and do not validate vLLM profiles or request scheduling. The missing backend is an environment/setup limit, not proof that ServingSim cannot express this workload.

## Meaning for the paper

The original external whole-workload count was zero. A bounded exact common-cost comparison now exists for Compass, and real complete trace generation is established for ServingSim. Neither result proves broader IR expressivity, human-productivity superiority, or better measured accuracy than these simulators. The next stronger comparison requires matched numerical and storage boundaries plus a runnable external backend; the receipt deliberately keeps that gap visible.

Primary sources: [LLMCompass](https://github.com/PrincetonUniversity/LLMCompass/tree/{PINS['LLMCompass']}) and [LLMServingSim](https://github.com/casys-kaist/LLMServingSim/tree/{PINS['LLMServingSim']}).
'''
    (out/'REPORT.md').write_text(text,encoding='utf-8',newline='\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--external',type=Path,default=DEFAULT_EXTERNAL)
    parser.add_argument('--python',type=Path,default=Path(sys.executable))
    args=parser.parse_args()
    lock=ROOT/'profiling/.measurement.lock'
    token=json.dumps(dict(pid=os.getpid(),protocol='external_comparison_cpu_simulation',
                          started=datetime.now(timezone.utc).isoformat()))
    try:
        fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    except FileExistsError as error:
        raise RuntimeError('Wait for the exclusive native measurement lease before external CPU simulation') from error
    with os.fdopen(fd,'w',encoding='utf-8',newline='\n') as stream:
        stream.write(token)
    try:
        run(args)
    finally:
        if lock.exists() and lock.read_text(encoding='utf-8')==token:
            lock.unlink()
