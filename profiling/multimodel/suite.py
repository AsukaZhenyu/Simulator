# -*- coding: utf-8 -*-
"""Turn a declared model into the two documents the dense probe consumes.

The measurement suites that ship with the repository were written by hand for
one geometry (``cases/dense_layer_qwen35_geometry.json`` hard-codes
2560/9216/16/4/256).  Nothing generic produced them, which is why "measure
another model" was a manual edit rather than a command.

This module closes that gap without inventing anything:

* geometry, weight shapes/dtypes/bytes and the presence of Q/K norm or an
  attention gate are read from the model's own tensor directory;
* every document it emits is re-parsed by ``m2.dense_contract``, so a suite the
  probe or the exporter would reject cannot be written in the first place;
* nothing about timing is produced here.  A suite is a list of shapes to
  measure, not a measurement.

Two values are *not* model metadata and are stated as defaults rather than
discovered: ``rope_dims`` falls back to ``head_dim`` when the GGUF publishes no
``rope.dimension_count`` (recorded in ``defaults``), and ``threads`` is whatever
the caller asks for, because a thread count belongs to the machine.
"""
from __future__ import annotations

from typing import Any

from llm_infer_model.m2 import dense_contract, moe_contract
from profiling.model_swap.build_model import _tensor_role, metadata_from_gguf

# GGUF tensor role -> the contract's weight role.  The contract declares exactly
# these, so an unlisted role is an error rather than an extra weight nobody
# charges for.
ROLES = {
    "attn_norm": "input_norm",
    "attn_q": "q",
    "attn_k": "k",
    "attn_v": "v",
    "attn_output": "output",
    "ffn_norm": "post_attention_norm",
    "ffn_gate": "gate",
    "ffn_up": "up",
    "ffn_down": "down",
    "attn_q_norm": "q_norm",
    "attn_k_norm": "k_norm",
    "attn_gate": "gate_proj",
}

# The same mapping for a sparse block: the attention-side roles are identical,
# the feed-forward is four expert tensors plus the shared expert when the
# checkpoint has one.
MOE_ROLES = {
    "attn_norm": "input_norm",
    "attn_q": "q",
    "attn_k": "k",
    "attn_v": "v",
    "attn_output": "output",
    "ffn_norm": "post_attention_norm",
    "ffn_gate_inp": "router",
    "ffn_gate_exps": "gate_exps",
    "ffn_up_exps": "up_exps",
    "ffn_down_exps": "down_exps",
    "ffn_gate_shexp": "gate_shexp",
    "ffn_up_shexp": "up_shexp",
    "ffn_down_shexp": "down_shexp",
    "ffn_gate_inp_shexp": "gate_inp_shexp",
    "attn_q_norm": "q_norm",
    "attn_k_norm": "k_norm",
    "attn_gate": "gate_proj",
}

STAGES = ("projection", "attention", "dense_ffn", "block")
# Stages outside the declared block that a complete token step still runs.
BOUNDARY_STAGES = ("embedding", "head")
WORD_DTYPES = ("f32", "f16", "q4_k", "q6_k")
# The contract stores K and V as f16; a descriptor that declared f32 would claim
# twice the cache bytes the measured block allocates.
KV_ELEMENT_BYTES = 2


def _representative_layer(meta: dict[str, Any], index: int | None) -> int:
    if index is not None:
        if not 0 <= index < meta["block_count"]:
            raise ValueError(f"layer {index} outside 0..{meta['block_count'] - 1}")
        return index
    for position, layer in enumerate(meta["layers"]):
        if layer["attention"] == "full_causal":
            return position
    raise ValueError("model has no full_causal layer; the dense contract cannot describe it")


def boundary(meta: dict[str, Any]) -> dict[str, Any]:
    """The two vocabulary-wide matrices a token step touches outside the block.

    Their types come from the GGUF tensor directory, not from the quantization
    label: a Q4_K_M file stores the embedding and the LM head in different types,
    and a tied checkpoint has no separate ``output.weight`` at all.
    """
    vocab = meta.get("vocab_size")
    if not vocab:
        raise ValueError("model publishes no token_embd vocabulary size")
    tensors = {t.get("name"): t for t in (meta.get("global_tensors") or [])}
    token = tensors.get("token_embd.weight")
    if token is None:
        raise ValueError("model has no token_embd.weight")
    head = tensors.get("output.weight")
    tied = head is None
    head = token if tied else head
    for role, tensor in (("token_embd", token), ("lm_head", head)):
        if tensor["dtype"] not in WORD_DTYPES:
            raise ValueError(
                f"{role} dtype {tensor['dtype']!r} is not one of {list(WORD_DTYPES)}; "
                "the declared boundary prices exactly these four types"
            )
        expected = (meta["hidden_size"], int(vocab))
        if tuple(tensor["shape"] or ()) != expected:
            raise ValueError(f"{role} is {tensor['shape']} but the geometry implies {list(expected)}")
    return {"vocab": int(vocab), "token_embd_dtype": token["dtype"],
            "lm_head_dtype": head["dtype"], "tied_embeddings": tied}


def describe(meta: dict[str, Any], *, layer_index: int | None = None) -> dict[str, Any]:
    """Derive the contract's config and weights from one decoder layer."""
    index = _representative_layer(meta, layer_index)
    layer = meta["layers"][index]
    # A Qwen2-MoE block stores one bias vector per Q/K/V beside each projection;
    # they are a different operand, never a stand-in for the matrix itself.
    tensors, biases = {}, {}
    for tensor in layer["tensors"]:
        role = _tensor_role(tensor["name"])
        if tensor["name"].endswith(".bias"):
            biases.setdefault(role, tensor)
        elif tensor["name"].endswith(".weight"):
            tensors.setdefault(role, tensor)

    fields, defaults = _attention_fields(meta, tensors)
    if layer["ffn"] == "moe":
        config = {**fields, **_moe_fields(meta, biases), "weight_dtypes": {}}
        contract, roles = moe_contract, MOE_ROLES
        sources = {mapped: name for name, mapped in roles.items()}
        sources.update({"q_bias": "attn_q", "k_bias": "attn_k", "v_bias": "attn_v"})
    else:
        if biases:
            raise ValueError(
                "this dense checkpoint stores attention biases "
                f"({sorted(biases)}); the dense contract prices no bias add, so the "
                "block would silently omit an operation")
        intermediate = meta["intermediate_size"]
        if intermediate is None:
            raise ValueError("model publishes no dense feed-forward width")
        config = {**fields, "intermediate": intermediate, "weight_dtypes": {}}
        contract, roles = dense_contract, ROLES

    # The contract re-derives every shape from the geometry; a mismatch here
    # means the GGUF and the declared metadata disagree about the model.
    shapes = contract.weight_shapes(config)
    weights = []
    for role, shape in shapes.items():
        if contract is moe_contract:
            source = sources.get(role)
            tensor = (biases if role in moe_contract.BIAS_ROLES else tensors).get(source)
        else:
            source = next((name for name, mapped in roles.items() if mapped == role), None)
            tensor = tensors.get(source) if source else None
        if tensor is None:
            raise ValueError(f"layer is missing the tensor for weight role {role!r}")
        config["weight_dtypes"][role] = tensor["dtype"]
        if tuple(tensor["shape"]) != tuple(shape):
            raise ValueError(
                f"W_{role}: tensor is {tensor['shape']} but the geometry implies {list(shape)}"
            )
        weights.append({
            "id": "W_" + role,
            "name": "W_" + role,
            "dtype": tensor["dtype"],
            "shape": list(tensor["shape"]),
            "storage_bytes": int(tensor["storage_bytes"]),
            "initial_location": "dram",
        })
    contract.validate_config(config)

    return {"layer_index": index, "config": config, "weights": weights,
            "boundary": boundary(meta), "defaults": defaults,
            "contract": contract, "ffn": layer["ffn"]}


def _attention_fields(meta: dict[str, Any], tensors: dict[str, Any]) -> tuple[dict, list[str]]:
    """Everything the two contracts declare identically about one block."""
    has_q_norm = "attn_q_norm" in tensors
    has_k_norm = "attn_k_norm" in tensors
    if has_q_norm != has_k_norm:
        raise ValueError("layer has only one of attn_q_norm / attn_k_norm")
    head_dim = meta["head_dim"]
    qk_norm = "none"
    if has_q_norm:
        widths = {tuple(tensors[role]["shape"]) for role in ("attn_q_norm", "attn_k_norm")}
        if len(widths) != 1:
            raise ValueError(f"attn_q_norm and attn_k_norm have different widths: {sorted(widths)}")
        width = widths.pop()
        if width == (head_dim,):
            qk_norm = "rms"
        elif width == (meta["hidden_size"],):
            # OLMoE normalizes the head-concatenated projection, not each head.
            qk_norm = "rms_hidden"
        else:
            raise ValueError(
                f"QK norm gain is {list(width)}; the contract declares either one gain "
                f"per head ({head_dim}) or one per hidden unit ({meta['hidden_size']})")
    rope_dims, defaults = meta["rope_dims"], []
    if rope_dims is None:
        rope_dims = head_dim
        defaults.append(
            f"rope_dims defaulted to head_dim ({head_dim}): the GGUF publishes no "
            "rope.dimension_count and the architecture applies rotary to the whole head"
        )
    if meta["rope_base"] is None:
        defaults.append("rope_base defaulted to 10000: the GGUF publishes no rope.freq_base")
    return {
        "hidden": meta["hidden_size"],
        "heads": meta["heads"],
        "kv_heads": meta["kv_heads"],
        "head_dim": head_dim,
        "attention_gate": "sigmoid" if "attn_gate" in tensors else "none",
        "qk_norm": qk_norm,
        "rope_mode": "normal",
        "rope_dims": rope_dims,
        "rope_base": float(meta["rope_base"] or 10000.0),
        "norm_epsilon": float(meta["norm_epsilon"]),
    }, defaults


def _moe_fields(meta: dict[str, Any], biases: dict[str, Any]) -> dict[str, Any]:
    experts, top_k = meta["experts"], meta["top_k"]
    width = meta["expert_intermediate_size"]
    if not experts or not top_k or not width:
        raise ValueError("model declares a sparse feed-forward without experts/top-k/width")
    shared = meta["shared_expert_intermediate_size"] or 0
    routing = moe_contract.routing_for(meta["architecture"])
    # The probe adds all three biases or none; a checkpoint with only some of
    # them would need a graph this contract does not declare.
    has_bias = {role for role in ("attn_q", "attn_k", "attn_v") if role in biases}
    if has_bias and has_bias != {"attn_q", "attn_k", "attn_v"}:
        raise ValueError(f"layer stores only some Q/K/V biases: {sorted(has_bias)}")
    return {"intermediate": int(width), "experts": int(experts), "top_k": int(top_k),
            "shared_intermediate": int(shared), "normalize_topk": routing["normalize_topk"],
            "attention_bias": bool(has_bias)}


def layer_document(meta: dict[str, Any], description: dict[str, Any], *,
                   model_id: str, threads: int, residency: str = "resident",
                   cpu_ids: list[int] | None = None) -> dict[str, Any]:
    """One decoder block, in the format its contract declares.

    ``cpu_ids`` is the placement the layer is *priced under*, not a command
    detail: when it is declared, the desktop signature names it, so a table
    measured on other logical CPUs cannot be claimed by this layer (R13).  It is
    omitted when undeclared so every earlier descriptor keeps its bytes.
    """
    config = description["config"]
    contract = description.get("contract", dense_contract)
    sparse = description.get("ffn") == "moe"
    hidden, head_dim = config["hidden"], config["head_dim"]
    stages = ("projection", "attention", "moe_ffn" if sparse else "dense_ffn")
    groups: dict[str, dict[str, str]] = {stage: {} for stage in stages}
    for stage in groups:
        for role in contract.stage_weight_roles(config, stage):
            groups[stage][role] = "W_" + role
    document = {
        # A measured sparse block is not an m2-layer mapping input: the mapping's
        # MoE path (m2-layer-0.6) is a synthetic boundary with its own semantics.
        "format_version": "moe-layer-0.1" if sparse else "m2-layer-0.3",
        "id": model_id,
        "provenance": "synthetic_layer",
        "hidden_size": hidden,
        # F32 activation, f16 K/V: the contract's declaration, not a GGUF
        # observation. The cache element type is the contract's, so the declared
        # per-token bytes are the ones the measured block actually allocates.
        "hidden_bytes_per_token": hidden * 4,
        "kv_bytes_per_token": 2 * config["kv_heads"] * head_dim * KV_ELEMENT_BYTES,
        "attention": "full_causal",
        "ffn": "moe" if sparse else "dense",
        "weights": description["weights"],
        "tasks": [
            {"id": stage, "kind": stage, "depends_on": [] if not index else [stages[index - 1]]}
            for index, stage in enumerate(stages)
        ],
        "physical": {
            "contract": contract.CONTRACT,
            "scope": "complete_declared_decoder_block",
            "input_boundary": "hidden_activation",
            "port_layout": "materialized_stage_ports",
            "kv_allocation": "logical_blocks",
            "activation_dtype": "f32",
            "kv_dtype": contract.KV_DTYPE,
            "config": config,
            "threads": threads,
            # Declared so the layer document and the table it is exported from
            # agree on the protocol that priced it; omitted for the default
            # resident regime so existing descriptors stay byte-identical.
            **({"residency": residency} if residency != "resident" else {}),
            **({"cpu_ids": list(cpu_ids)} if cpu_ids else {}),
            "weight_groups": groups,
            "excluded_operations": [],
        },
        "comment": (
            f"Generated from the {meta['architecture']} tensor directory of the declared "
            f"checkpoint (layer {description['layer_index']}). Weights are generated "
            "deterministically at these shapes and dtypes; this is not the pretrained "
            "execution path."
        ),
    }
    if description["defaults"]:
        document["comment"] += " Defaults applied: " + "; ".join(description["defaults"]) + "."
    return document


def history_of(shape: dict[str, Any]) -> int:
    return int(shape.get("visible_history", 0))


def suite_document(description: dict[str, Any], shapes: dict[str, Any], *,
                   model_id: str, comment: str | None = None) -> dict[str, Any]:
    config = description["config"]
    contract = description.get("contract", dense_contract)
    stages = contract.STAGES
    cases = []
    for phase in ("prefill", "decode"):
        for shape in shapes[phase]:
            tokens = int(shape["tokens"])
            history = int(shape.get("visible_history", 0))
            for stage in stages:
                cases.append({
                    "id": f"{model_id}-{stage}-{phase}-t{tokens}-h{history}",
                    "kind": stage,
                    "phase": phase,
                    "batch": int(shapes.get("batch", 1)),
                    "tokens": tokens,
                    "visible_history": history,
                    "config": dict(config),
                })
    document = {"schema_version": contract.SUITE_SCHEMA, "cases": cases}
    if description.get("boundary"):
        word = description["boundary"]
        config = {**config, "vocab": word["vocab"],
                  "token_embd_dtype": word["token_embd_dtype"],
                  "lm_head_dtype": word["lm_head_dtype"],
                  "weight_dtypes": {**config["weight_dtypes"], "final_norm": "f32",
                                    "lm_head": word["lm_head_dtype"],
                                    "token_embd": word["token_embd_dtype"]}}
        # A boundary stage has no cache, so it is measured once per token count
        # at history 0; its price does not depend on the conversation length.
        for phase in ("prefill", "decode"):
            for shape in shapes[phase]:
                tokens = int(shape["tokens"])
                if history_of(shape):
                    continue
                for stage in BOUNDARY_STAGES:
                    document["cases"].append({
                        "id": f"{model_id}-{stage}-{phase}-t{tokens}-h0",
                        "kind": stage, "phase": phase, "batch": int(shapes.get("batch", 1)),
                        "tokens": tokens, "visible_history": 0, "config": dict(config),
                    })
    if comment:
        document["comment"] = comment
    return document


def validate(suite: dict[str, Any], layer: dict[str, Any], threads: int) -> None:
    """Run the probe's own parser and the exporter's geometry check."""
    physical = layer["physical"]
    contract = moe_contract if physical["contract"] == moe_contract.CONTRACT else dense_contract
    contract.build_plan(suite, threads)
    contract.validate_config(physical["config"])
    if physical["threads"] != threads:
        raise ValueError("layer threads differ from the requested protocol")
    residency = physical.get("residency", "resident")
    if residency not in dense_contract.RESIDENCY_REGIMES:
        raise ValueError(f"layer residency {residency!r} is not a declared regime")
    declared = {weight["id"]: tuple(weight["shape"]) for weight in layer["weights"]}
    for role, shape in contract.weight_shapes(physical["config"]).items():
        if declared.get("W_" + role) != tuple(shape):
            raise ValueError(f"weight {role} does not match the physical geometry")
