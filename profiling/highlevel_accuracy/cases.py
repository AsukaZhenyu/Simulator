"""Build new complete physical ModelPlans for bridge validation.

The one-nanosecond prices are compile-time placeholders only. A frozen native
component ledger is mandatory for every measured prediction/comparison.
"""
from pathlib import Path
import json
import math

from llm_infer_model.m2.dense_contract import validate_config, weight_shapes, stage_weight_roles, signature
from llm_infer_model.m2.dense_cuda_contract import signature as cuda_signature
from llm_infer_model.m2.costs import canonical_contract_hash
from llm_infer_model.m2.attention_stages import KINDS as ATTENTION_KINDS
from tensor_mapping.python_frontend import ModelPlan, Scheduling

from .bridge import SerialExecutionPolicy, _require


def build_case(directory, *, hidden=16, intermediate=32, heads=4, kv_heads=2,
               head_dim=4, tokens=1, history=3, threads=2,
               devices=("cpu", "gpu", "cpu"), attention_stages=False,
               attention_devices=("cpu", "cpu", "gpu", "gpu"), ffn_cut=None,
               ffn_devices=("cpu", "gpu"), norm_device="gpu", merge_device="gpu",
               attention_gate="sigmoid", qk_norm="rms", reuse=False):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    _require(reuse or not any(directory.iterdir()), "case directory must be new or empty")
    _require(len(devices)==3 and set(devices)<= {"cpu","gpu"}, "exact three macro placements required")
    _require(type(tokens) is int and tokens>0 and type(history) is int and history>=0, "positive token / nonnegative history required")
    phase = "decode" if tokens==1 and history>0 else "prefill"
    config=dict(hidden=hidden, intermediate=intermediate, heads=heads, kv_heads=kv_heads,
        head_dim=head_dim, rope_dims=head_dim, rope_base=10000.0, norm_epsilon=1e-6,
        rope_mode="normal", attention_gate=attention_gate, qk_norm=qk_norm)
    config["weight_dtypes"]={role:"f32" for role in weight_shapes(config)}
    config=validate_config(config)
    shapes=weight_shapes(config)
    layer=dict(format_version="m2-layer-0.5", id="highlevel-dense", provenance="synthetic_layer",
        hidden_size=hidden, hidden_bytes_per_token=4*hidden, kv_bytes_per_token=8*kv_heads*head_dim,
        attention="full_causal", ffn="dense",
        weights=[dict(id="W_"+role,name="W_"+role,dtype="f32",shape=list(shape),storage_bytes=4*math.prod(shape),initial_location="dram") for role,shape in shapes.items()],
        tasks=[dict(id=name,kind=kind,depends_on=deps) for name,kind,deps in (("proj","projection",[]),("attn","attention",["proj"]),("ffn","dense_ffn",["attn"]))],
        physical=dict(contract="ggml_hybrid_dense_materialized_macros_v1", scope="complete_declared_decoder_block",
            input_boundary="hidden_activation", port_layout="materialized_stage_ports",kv_allocation="logical_blocks",
            activation_dtype="f32",kv_dtype="f32",config=config,threads=threads,cuda_device=0,cuda_graphs="disabled",
            weight_groups={stage:{r:"W_"+r for r in stage_weight_roles(config,stage)} for stage in ("projection","attention","dense_ffn")},excluded_operations=[]))
    entries=[]
    for kind in ("projection","attention","dense_ffn"):
        for device in ("cpu","gpu"):
            contract=(signature(config,kind,phase,tokens,history,threads) if device=="cpu" else cuda_signature(config,kind,phase,tokens,history,0))
            entries.append(dict(kind=kind,device=device,phase=phase,batch=1,tokens=tokens,visible_history=history,
                duration_ns=1,workspace_bytes=0,provenance="synthetic_layer",accounting="per_request",
                physical_signature=contract,contract_hash=canonical_contract_hash(contract)))
    sizes={4*math.prod(s) for s in shapes.values()} | {4*tokens,4*hidden*tokens,
        4*head_dim*heads*tokens,4*head_dim*kv_heads*tokens,8*kv_heads*head_dim,
        4*(history+tokens)*tokens*heads,4*intermediate*tokens}
    if ffn_cut is not None:
        _require(type(ffn_cut) is int and 0<ffn_cut<intermediate, "FFN cut must partition intermediate axis")
        for width in (ffn_cut,intermediate-ffn_cut):sizes.update((4*hidden*width,4*width*tokens))
    transfers=[dict(route=route,storage_bytes=size,duration_ns=1,provenance="synthetic_transfer") for route in ("dram_to_vram","vram_to_dram") for size in sorted(sizes)]
    costs=dict(format_version="m2-layer-0.5",source="compile_placeholders_never_measured_accuracy",entries=entries,transfer_entries=transfers)
    invocation=dict(format_version="m2-layer-0.5",id="dense-invocation",layer=layer["id"],layer_file="layer.json",costs_file="costs.json",
        phase=phase,devices=["cpu","gpu"],hardware=dict(dram_capacity_bytes=1<<30,vram_capacity_bytes=1<<30,
            dram_alignment_bytes=1,vram_alignment_bytes=1,dram_runtime_reserved_bytes=0,vram_runtime_reserved_bytes=0),
        requests=[dict(request_id="r0",position_start=history,tokens=tokens,input_location="dram",output_location="dram")],
        state_in=dict(requests=[dict(request_id="r0",next_position=history,version=1,state_location="dram",
            blocks=[dict(id=f"kv:highlevel-dense:r0:{i}",first_position=i,tokens=1,storage_bytes=8*kv_heads*head_dim,locations=["dram"]) for i in range(history)])]))
    global_model=dict(format_version="global-execution-0.1",id="highlevel-accuracy-dense",rounds=[dict(id="one-round",input_mode="initial_supplied_hidden",gate_cost=None,
        invocations=[dict(id="dense",invocation_file="invocation.json",execution_file=None)])])
    for name,value in (("layer.json",layer),("costs.json",costs),("invocation.json",invocation),("global.json",global_model),("environment.json",dict(format_version="environment-trace-0.1",events=[]))):
        content=json.dumps(value,indent=2)+"\n"
        if reuse:
            _require((directory/name).read_text(encoding="utf-8")==content, "saved case source changed: "+name)
        else:
            (directory/name).write_text(content,encoding="utf-8")
    plan=ModelPlan(id="highlevel-accuracy-plan",model_kind="global_m2",model_file=directory/"global.json",environment_file=directory/"environment.json",
        max_decisions=100000,scheduling=Scheduling(placements=dict(zip(("projection","attention","dense_ffn"),devices)),release="last_use"))
    if attention_stages:
        spec=dict(format_version="attention-stages-0.1",variant="qk_softmax_pv",provenance="synthetic_execution",
            geometry={key:config[key] for key in ("hidden","heads","kv_heads","head_dim","attention_gate")},frontend=dict(contract="existing_materialized_post_rope_qkv"),
            entries=[dict(kind=kind,device=device,phase=phase,batch=1,tokens=tokens,visible_history=history,duration_ns=1,workspace_bytes=0) for kind in sorted(ATTENTION_KINDS) for device in ("cpu","gpu")],
            transfer_entries=[{k:v for k,v in row.items() if k!="provenance"} for row in transfers])
        plan.transform("dense",spec)
        for kind,device in zip(("attention_qk","attention_softmax","attention_pv","attention_output"),attention_devices):plan.place(kind,device)
    if ffn_cut is not None:
        from llm_infer_model.m2.ffn_execution import KINDS
        spec=dict(format_version="ffn-execution-0.1",target_kinds=["dense_ffn"],intermediate_size=intermediate,
            shards=[dict(id="left",start=0,end=ffn_cut,device=ffn_devices[0]),dict(id="right",start=ffn_cut,end=intermediate,device=ffn_devices[1])],
            merge_input_count=2,preprocess_device=norm_device,merge_device=merge_device,provenance="synthetic_execution",weight_storage_contract="materialized_partition_replaces_parent",
            entries=[dict(kind=kind,device=device,phase=phase,batch=1,tokens=tokens,visible_history=history,hidden_size=hidden,intermediate_size=width,duration_ns=1,workspace_bytes=0)
                for kind in sorted(KINDS) for device in ("cpu","gpu") for width in sorted({intermediate,ffn_cut,intermediate-ffn_cut})],
            transfer_entries=[{k:v for k,v in row.items() if k!="provenance"} for row in transfers])
        plan.transform("dense",spec)
    plan.runtime_policy(SerialExecutionPolicy)
    return plan
