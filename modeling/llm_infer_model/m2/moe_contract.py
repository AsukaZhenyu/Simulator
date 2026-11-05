"""One explicit CPU mixture-of-experts decoder-block contract.

The dense contract describes a block whose feed-forward is a SwiGLU MLP.  A
sparse block differs only there: the same norms, projections and causal GQA
attention run first, and then a router selects ``top_k`` of ``experts`` small
MLPs whose outputs are combined with the router weights.  That is why this
module delegates every attention-side declaration to ``dense_contract`` instead
of restating it: the two contracts must not drift on the part they share.

What is *not* delegated is the sparse feed-forward, because its weights are real
checkpoint tensors with their own storage types.  The published Q4_K_M files
store the expert gate and up projections as Q4_K, the down projection as Q6_K
(Granite) or Q8_0 (Qwen1.5-MoE), and the router as F32; a contract that priced
one uniform type would charge a different number of bytes than the model reads.

Declared variants rather than discovered facts:

* ``normalize_topk`` (renormalize the selected router weights) and the shared
  expert's presence are properties of the architecture's graph builder in
  llama.cpp.  They are not GGUF metadata, so the value comes from an explicit
  per-architecture table and an unlisted architecture is an error.
* Every stage boundary is a materialized copy, exactly as in the dense
  contract, so the sparse stage's cost can be measured on its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

from ..tensor.spec import InvalidInput, Unsupported, ggml_contiguous_layout
from . import dense_contract
from .dense_contract import (BOUNDARY_STAGES, DTYPES as _DENSE_DTYPES, EXCLUDED, KV_DTYPE,
                             PRICE_POLICIES, PROCESS_SCOPES, RESIDENCY_REGIMES, SCOPES,
                             declared_process_scope, float32_parameter, integer,
                             signature_price_policy, signature_process_scope, signature_residency)

CONTRACT = "ggml_cpu_moe_decoder_block_v1"
SUITE_SCHEMA = "moe_layer_measurement_suite_v1"
PROFILE_SCHEMA = "moe_layer_measurement_profile_v1"
# The probe's layer descriptor declares its own format: a measured sparse block
# is *not* an m2-layer mapping input (the mapping's MoE path is m2-layer-0.6).
LAYER_FORMAT_VERSION = "moe-layer-0.1"
STAGES = ("projection", "attention", "moe_ffn", "block")
DTYPES = {**_DENSE_DTYPES, "q8_0": "GGML_TYPE_Q8_0"}
VARIANTS = {stage: f"ggml_cpu_moe_materialized_{stage}_v1"
            for stage in STAGES + tuple(BOUNDARY_STAGES)}
WORD_KEYS = {"vocab", "token_embd_dtype", "lm_head_dtype"}
CONFIG_KEYS = {"hidden", "intermediate", "heads", "kv_heads", "head_dim",
               "attention_gate", "qk_norm", "rope_mode", "rope_dims", "rope_base",
               "norm_epsilon", "experts", "top_k", "shared_intermediate", "normalize_topk",
               "attention_bias", "weight_dtypes"}
# Roles that are always F32: the two norms, the router and the shared-expert
# gate are single rows or a [hidden, experts] matrix that llama.cpp keeps in F32.
F32_ROLES = ("input_norm", "post_attention_norm", "router", "gate_inp_shexp", "final_norm",
             "q_bias", "k_bias", "v_bias")
# The Q/K/V bias vectors a Qwen2-MoE checkpoint stores (granite stores none).
BIAS_ROLES = ("q_bias", "k_bias", "v_bias")

# The sparse feed-forward as llama.cpp builds it.  The list is what the probe
# performs and what the exported price covers; an operation missing here would be
# timed but not declared.
MOE_OPERATIONS = ("post_attention_RMSNorm", "post_attention_norm_scale", "router_projection",
                  "router_softmax", "top_k_selection", "top_k_weight_gather",
                  "expert_gate_projection", "expert_up_projection", "SiLU_gate",
                  "gate_times_up", "expert_down_projection", "expert_weighting")
SHARED_OPERATIONS = ("shared_expert_gate_inp", "shared_sigmoid", "shared_expert_gate_projection",
                     "shared_expert_up_projection", "SiLU_gate_shared", "gate_times_up_shared",
                     "shared_expert_down_projection", "shared_gate_scale")

# Architecture -> declared routing variant.  Confirmed against the graph builder
# of the pinned llama.cpp release (b8705): Granite renormalizes the selected
# weights (`norm_w = true`), Qwen2MoE and OLMoE do not; all three softmax over
# every expert and then select the top k.
ROUTING = {"granitemoe": {"normalize_topk": True},
           "qwen2moe": {"normalize_topk": False},
           "olmoe": {"normalize_topk": False}}


def routing_for(architecture: str) -> dict[str, Any]:
    if architecture not in ROUTING:
        raise Unsupported(f"no declared MoE routing variant for architecture {architecture!r}; "
                          f"known: {sorted(ROUTING)}")
    return dict(ROUTING[architecture])


def _dense_view(config: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a sparse config the dense contract already validates.

    ``intermediate`` is the expert width here, so it is passed through only
    because the dense helpers require the key; nothing this view feeds uses it
    for a dense matrix.
    """
    view = {key: config[key] for key in ("hidden", "intermediate", "heads", "kv_heads", "head_dim",
                                         "attention_gate", "qk_norm", "rope_mode", "rope_dims",
                                         "rope_base", "norm_epsilon")}
    if "vocab" in config:
        view["vocab"] = config["vocab"]
    return view


def weight_shapes(config: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    h, i, d = config["hidden"], config["intermediate"], config["head_dim"]
    experts = config["experts"]
    q, kv = config["heads"] * d, config["kv_heads"] * d
    result = {"input_norm": (h,), "q": (h, q), "k": (h, kv), "v": (h, kv),
              "output": (q, h), "post_attention_norm": (h,),
              "router": (h, experts),
              # One expert is one dense matrix: gate and up reduce over the
              # hidden axis, down reduces over the expert width.  These are the
              # stored GGUF shapes, in the stored order.
              "gate_exps": (h, i, experts), "up_exps": (h, i, experts),
              "down_exps": (i, h, experts)}
    if config["attention_bias"]:
        result.update(q_bias=(q,), k_bias=(kv,), v_bias=(kv,))
    if config["shared_intermediate"]:
        s = config["shared_intermediate"]
        result.update(gate_shexp=(h, s), up_shexp=(h, s), down_shexp=(s, h),
                      gate_inp_shexp=(h,))
    if config["attention_gate"] == "sigmoid":
        result["gate_proj"] = (h, q)
    if config["qk_norm"] == "rms":
        result.update(q_norm=(d,), k_norm=(d,))
    elif config["qk_norm"] == "rms_hidden":
        result.update(q_norm=(h,), k_norm=(h,))
    if "vocab" in config:
        result.update(final_norm=(h,), lm_head=(h, config["vocab"]),
                      token_embd=(h, config["vocab"]))
    return result


def stage_weight_roles(config: Mapping[str, Any], kind: str) -> tuple[str, ...]:
    if kind in ("embedding", "head"):
        return dense_contract.stage_weight_roles(config, kind)
    if kind in ("projection", "attention"):
        roles = list(dense_contract.stage_weight_roles(_dense_view(config), kind))
        if kind == "projection" and config["attention_bias"]:
            roles += BIAS_ROLES
        return tuple(roles)
    if kind == "moe_ffn":
        roles = ["post_attention_norm", "router", "gate_exps", "up_exps", "down_exps"]
        if config["shared_intermediate"]:
            roles += ["gate_shexp", "up_shexp", "down_shexp", "gate_inp_shexp"]
        return tuple(roles)
    if kind == "block":
        return tuple(role for stage in STAGES[:3] for role in stage_weight_roles(config, stage))
    raise Unsupported(f"unsupported sparse stage {kind!r}")


def validate_config(raw: Any) -> dict[str, Any]:
    allowed = CONFIG_KEYS | WORD_KEYS
    if not isinstance(raw, dict) or not (CONFIG_KEYS <= set(raw) <= allowed):
        raise InvalidInput(f"sparse config must declare exactly {sorted(CONFIG_KEYS)}")
    word = WORD_KEYS & set(raw)
    if word and word != WORD_KEYS:
        raise InvalidInput(f"vocabulary-wide stages must declare both {sorted(WORD_KEYS)}")
    config = {key: integer(raw[key], f"config.{key}") for key in
              ("hidden", "intermediate", "heads", "kv_heads", "head_dim", "rope_dims",
               "experts", "top_k")}
    if config["top_k"] > config["experts"]:
        raise InvalidInput("config.top_k must not exceed config.experts")
    shared = raw["shared_intermediate"]
    config["shared_intermediate"] = integer(shared, "config.shared_intermediate", 0)
    for key in ("normalize_topk", "attention_bias"):
        if not isinstance(raw[key], bool):
            raise InvalidInput(f"config.{key} must be a boolean")
        config[key] = raw[key]
    if word:
        config["vocab"] = integer(raw["vocab"], "config.vocab")
        for key in sorted(WORD_KEYS - {"vocab"}):
            if raw[key] not in ("f32", "f16", "q4_k", "q6_k"):
                raise InvalidInput(f"unsupported config.{key}: {raw[key]!r}")
            config[key] = raw[key]
    if config["heads"] % config["kv_heads"]:
        raise InvalidInput("config.heads must be divisible by kv_heads")
    if config["rope_dims"] % 2 or config["rope_dims"] > config["head_dim"]:
        raise InvalidInput("rope_dims must be positive, even and <= head_dim")
    for key, values in (("attention_gate", ("none", "sigmoid")),
                        ("qk_norm", ("none", "rms", "rms_hidden"))):
        if raw[key] not in values:
            raise InvalidInput(f"config.{key} must be one of {values}")
        config[key] = raw[key]
    if raw["rope_mode"] != "normal":
        raise Unsupported("this sparse contract supports normal adjacent-pair RoPE only")
    config["rope_mode"] = "normal"
    config["rope_base"] = float32_parameter(raw["rope_base"], "config.rope_base", 1.0)
    config["norm_epsilon"] = float32_parameter(raw["norm_epsilon"], "config.norm_epsilon")
    shapes = weight_shapes(config)
    dtypes = raw["weight_dtypes"]
    if not isinstance(dtypes, dict) or set(dtypes) != set(shapes):
        raise InvalidInput(f"weight_dtypes must declare exactly {sorted(shapes)}")
    config["weight_dtypes"] = {}
    for role, shape in shapes.items():
        dtype = dtypes[role]
        if dtype not in DTYPES or dtype == "i32":
            raise InvalidInput(f"unsupported weight dtype for {role}: {dtype!r}")
        if role in F32_ROLES and dtype != "f32":
            raise Unsupported(f"{role} is an F32 parameter of this contract")
        if role.endswith("_norm") and dtype != "f32":
            raise Unsupported("RMSNorm gamma vectors in this contract must be f32")
        # The router is the only [hidden, experts] matrix and the shared-expert
        # gate the only [hidden] one; both are F32 above, so the block alignment
        # check below only ever constrains the expert planes.
        ggml_contiguous_layout(DTYPES[dtype], (*shape, *([1] * (4 - len(shape)))))
        config["weight_dtypes"][role] = dtype
    if word:
        for key, role in (("token_embd_dtype", "token_embd"), ("lm_head_dtype", "lm_head")):
            if config["weight_dtypes"][role] != config[key]:
                raise InvalidInput(f"config.weight_dtypes.{role} must equal config.{key}")
    return config


def validate_case(raw: Any) -> dict[str, Any]:
    required = {"id", "kind", "phase", "batch", "tokens", "visible_history", "config"}
    optional = {"threads", "comment", "source", "metadata"}
    if not isinstance(raw, dict) or not required.issubset(raw) or set(raw) - required - optional:
        raise InvalidInput("sparse case has missing or unsupported fields")
    if not isinstance(raw["id"], str) or not raw["id"]:
        raise InvalidInput("sparse case id must be nonempty")
    if raw["kind"] not in STAGES + tuple(BOUNDARY_STAGES) or raw["phase"] not in ("prefill",
                                                                                 "decode"):
        raise InvalidInput("sparse case must select an explicit stage and phase")
    if integer(raw["batch"], "case.batch") != 1:
        raise Unsupported("sparse layer measurement supports batch=1 only")
    tokens = integer(raw["tokens"], "case.tokens")
    history = integer(raw["visible_history"], "case.visible_history", 0)
    if raw["phase"] == "decode" and tokens != 1:
        raise InvalidInput("decode requires one new token")
    if history + tokens > 0x7fffffff:
        raise Unsupported("absolute positions must fit the GGML I32 RoPE/mask interface")
    result = {**raw, "config": validate_config(raw["config"])}
    if raw["kind"] in BOUNDARY_STAGES:
        if history:
            raise InvalidInput("embedding/head cases must declare visible_history=0")
        if "vocab" not in result["config"]:
            raise InvalidInput("embedding/head cases must declare config.vocab and config.word_dtype")
    elif "vocab" in result["config"]:
        raise InvalidInput("decoder-block stages must not declare config.vocab")
    if "threads" in raw:
        result["threads"] = integer(raw["threads"], "case.threads")
    return result


def build_plan(suite: Any, threads: int) -> list[dict[str, Any]]:
    integer(threads, "threads")
    if not isinstance(suite, dict) or suite.get("schema_version") != SUITE_SCHEMA:
        raise InvalidInput(f"sparse suite schema_version must be {SUITE_SCHEMA}")
    if set(suite) - {"schema_version", "cases", "comment"}:
        raise InvalidInput("sparse suite contains unsupported fields")
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise InvalidInput("sparse suite must have cases")
    result = []
    for raw in cases:
        case = validate_case(raw)
        if "threads" in case and case["threads"] != threads:
            raise InvalidInput("case threads differ from requested measurement protocol")
        result.append({**case, "threads": threads})
    if len({case["id"] for case in result}) != len(result):
        raise InvalidInput("sparse case ids must be unique")
    return result


def operations(config: Mapping[str, Any], kind: str) -> list[str]:
    if kind == "projection":
        result = dense_contract.operations(_dense_view(config), kind)
        if config["attention_bias"]:
            # The three bias adds sit between the projections and RoPE, which is
            # where llama.cpp applies them.
            at = result.index("V_projection") + 1
            result[at:at] = ["Q_bias_add", "K_bias_add", "V_bias_add"]
        return result
    if kind in ("attention", "embedding", "head"):
        return dense_contract.operations(config if kind in ("embedding", "head")
                                         else _dense_view(config), kind)
    if kind == "moe_ffn":
        result = list(MOE_OPERATIONS)
        if config["normalize_topk"]:
            result.insert(6, "top_k_weight_normalize")
        result += ["expert_accumulate"] * (config["top_k"] - 1)
        if config["shared_intermediate"]:
            result += list(SHARED_OPERATIONS)
        return result + ["ffn_residual_add", "materialize_hidden"]
    if kind == "block":
        return [op for stage in STAGES[:3] for op in operations(config, stage)]
    raise Unsupported(f"unsupported sparse stage {kind!r}")


def ports(config: Mapping[str, Any], kind: str, tokens: int) -> tuple[dict, dict]:
    if kind == "moe_ffn":
        hidden = {"dtype": "f32", "ne": (config["hidden"], tokens, 1, 1)}
        return {"attention_hidden": hidden}, {"hidden_out": hidden}
    return dense_contract.ports(_dense_view(config) if kind != "embedding" and kind != "head"
                                else config, kind, tokens)


def history_invariant_kinds() -> tuple[str, ...]:
    """The sparse block's history-invariant stages: projection and moe_ffn."""
    return dense_contract.history_invariant_kinds(STAGES)


def signature(config: dict[str, Any], kind: str, phase: str, tokens: int,
              visible_history: int, threads: int, batch: int = 1,
              residency: str | None = None,
              process_scope: str | None = None,
              cpu_ids: list[int] | None = None,
              price_policy: str | None = None) -> dict[str, Any]:
    case = validate_case(dict(id="signature", kind=kind, phase=phase, batch=batch,
                              tokens=tokens, visible_history=visible_history, config=config,
                              threads=threads))
    config = case["config"]
    result = {"contract": CONTRACT, "scope": SCOPES.get(kind, "complete_declared_decoder_block"),
            "kind": kind,
            "device": "cpu", "backend": "CPU", "implementation_variant": VARIANTS[kind],
            "config": config, "phase": phase, "batch": batch, "tokens": tokens,
            "visible_history": visible_history, "threads_requested": threads,
            "threads_applied": threads, "thread_management": "persistent_ggml_threadpool",
            "activation_dtype": "f32", "kv_dtype": KV_DTYPE, "position_dtype": "i32",
            "numeric_parameter_semantics": "GGML_float32_parameters",
            "timing_scope": "single_completed_cpu_graph_wall_ns", "setup_compute_repetitions": 1,
            "stage_boundary_policy": "timed_materialized_copies_in_isolated_and_joint_graphs",
            "input_lineage": "same_seed_complete_upstream_preparation_outside_timing",
            "operations": operations(config, kind), "excluded_operations": list(EXCLUDED),
            "original_model_implementation_reproduced": False}
    # Same rule as the dense contract: a declared cold price must not answer to
    # the same signature as a resident one, and the default stays byte-identical
    # so tables measured under the old regime keep binding.
    if residency is not None:
        result["residency_regime"] = RESIDENCY_REGIMES[residency]
    # And the same for the process scope the table was measured in.
    if process_scope is not None:
        result["measurement_process_scope"] = PROCESS_SCOPES[process_scope]
    # And for the placement the workers were pinned to: a sparse block is just
    # as sensitive to which logical CPUs ran it as a dense one, and the rule is
    # the dense contract's own so the two cannot drift.
    if cpu_ids is not None:
        result["cpu_mask"] = dense_contract.signature_cpu_ids(cpu_ids, threads)
        result["thread_placement"] = dense_contract.THREAD_PLACEMENTS["pinned"]
    # And for the estimator behind a history-invariant decode price: the dense
    # contract owns the rule so a sparse and a dense block cannot spell the same
    # policy two ways.
    if price_policy is not None:
        result["history_invariant_price_policy"] = PRICE_POLICIES[price_policy]
    return result


def resident_accounting(config: dict[str, Any], kind: str, tokens: int, history: int,
                        kv_element_bytes: int = 2) -> dict[str, Any]:
    config = validate_config(config)
    integer(tokens, "tokens")
    integer(history, "history", 0)
    shapes = weight_shapes(config)
    weight_bytes = {}
    for role in stage_weight_roles(config, kind):
        shape = shapes[role]
        _, size = ggml_contiguous_layout(DTYPES[config["weight_dtypes"][role]],
                                         (*shape, *([1] * (4 - len(shape)))))
        weight_bytes[role] = size
    inputs, outputs = ports(config, kind, tokens)

    def sizes(values):
        return {name: math.prod(port["ne"]) * 4 for name, port in values.items()}

    input_sizes, output_sizes = sizes(inputs), sizes(outputs)
    kv = 2 * kv_element_bytes * config["kv_heads"] * config["head_dim"] if kind in (
        "attention", "block") else 0
    result = {"weight_roles_bytes": weight_bytes, "input_ports_bytes": input_sizes,
              "output_ports_bytes": output_sizes, "weight_bytes": sum(weight_bytes.values()),
              "input_bytes": sum(input_sizes.values()), "output_bytes": sum(output_sizes.values()),
              "kv_history_bytes": history * kv, "kv_new_bytes": tokens * kv,
              "kv_cache_bytes": (history + tokens) * kv}
    result["resident_logical_bytes"] = (result["weight_bytes"] + result["input_bytes"] +
                                        result["output_bytes"] + result["kv_cache_bytes"])
    return result


def native_arguments(case: dict[str, Any], warmup: int, samples: int, seed: int,
                     evict_bytes: int = 0) -> list[str]:
    case = validate_case(case)
    integer(warmup, "warmup", 0)
    integer(samples, "samples", 3)
    integer(seed, "seed", 0)
    integer(evict_bytes, "evict_bytes", 0)
    if samples > 10000 or seed > 0xffffffff or evict_bytes > (1 << 34):
        raise InvalidInput("samples, seed or evict_bytes exceed native protocol range")
    if "threads" not in case:
        raise InvalidInput("native invocation requires explicit threads")
    config = case["config"]
    args = ["--stage", case["kind"]]
    for key in ("hidden", "intermediate", "heads", "kv_heads", "head_dim"):
        args += ["--" + key.replace("_", "-"), str(config[key])]
    args += ["--q", str(case["tokens"]), "--history", str(case["visible_history"]),
             "--threads", str(case["threads"]), "--warmup", str(warmup),
             "--samples", str(samples), "--seed", str(seed)]
    for key, flag in (("attention_gate", "attention-gate"), ("qk_norm", "qk-norm"),
                      ("rope_mode", "rope-mode"), ("rope_dims", "rope-dims"),
                      ("rope_base", "rope-base"), ("norm_epsilon", "norm-eps")):
        args += ["--" + flag, str(config[key])]
    args += ["--experts", str(config["experts"]), "--top-k", str(config["top_k"]),
             "--shared-intermediate", str(config["shared_intermediate"]),
             "--normalize-topk", "1" if config["normalize_topk"] else "0",
             "--attention-bias", "1" if config["attention_bias"] else "0"]
    flags = {"q": "q-dtype", "k": "k-dtype", "v": "v-dtype", "gate_proj": "gate-proj-dtype",
             "output": "o-dtype", "gate_exps": "gate-exps-dtype", "up_exps": "up-exps-dtype",
             "down_exps": "down-exps-dtype", "gate_shexp": "shexp-gate-dtype",
             "up_shexp": "shexp-up-dtype", "down_shexp": "shexp-down-dtype"}
    for role, flag in flags.items():
        if role in config["weight_dtypes"]:
            args += ["--" + flag, config["weight_dtypes"][role]]
    args += ["--kv-dtype", KV_DTYPE]
    if evict_bytes:
        args += ["--evict-bytes", str(evict_bytes)]
    if "vocab" in config:
        args += ["--vocab", str(config["vocab"])]
        for role, flag in (("lm_head", "lm-head-dtype"), ("token_embd", "token-embd-dtype")):
            args += ["--" + flag, config["weight_dtypes"][role]]
    return args


# --------------------------------------------------------------------------
# The probe's layer descriptor.
#
# The exporter has to answer one question about a layer document: "does the
# geometry in this file agree, byte for byte, with the weights the probe will
# allocate?"  For a dense block that question is answered by m2.spec, whose
# m2-layer-0.3 parser also carries mapping semantics a measured sparse block
# does not have.  This parser answers only the probe's question.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SparsePhysical:
    config: dict[str, Any]
    threads: int
    contract: str = CONTRACT
    scope: str = "complete_declared_decoder_block"
    input_boundary: str = "hidden_activation"
    port_layout: str = "materialized_stage_ports"
    kv_allocation: str = "logical_blocks"
    activation_dtype: str = "f32"
    kv_dtype: str = KV_DTYPE
    weight_groups: dict[str, dict[str, str]] = field(default_factory=dict)
    excluded_operations: tuple[str, ...] = ()
    residency: str = "resident"
    # Whether the table's entries were priced in one probe process or one
    # process each. Like the regime it is a protocol, not a geometry: a price is
    # only valid under the scope that produced it.
    process_scope: str = "per_case"
    # The logical CPUs the measured workers were pinned to, or None when the OS
    # chose. Part of the signature when declared, for the same reason the
    # regime and the scope are: two threads on two SMT siblings price the same
    # block differently from two threads on two separate cores.
    cpu_ids: tuple[int, ...] | None = None


@dataclass(frozen=True)
class SparseLayer:
    id: str
    physical: SparsePhysical
    weights: tuple[dict[str, Any], ...] = ()


_LAYER_KEYS = {"format_version", "id", "provenance", "hidden_size",
               "hidden_bytes_per_token", "kv_bytes_per_token", "attention", "ffn",
               "weights", "tasks", "physical", "comment"}
_PHYSICAL_KEYS = {"contract", "scope", "input_boundary", "port_layout", "kv_allocation",
                  "activation_dtype", "kv_dtype", "config", "threads", "weight_groups",
                  "excluded_operations"}
_WEIGHT_KEYS = {"id", "name", "dtype", "shape", "storage_bytes", "initial_location"}


def _exact_keys(raw: Any, keys: set[str], where: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != keys:
        found = sorted(set(raw)) if isinstance(raw, dict) else type(raw).__name__
        raise InvalidInput(f"{where}: field set must be exactly {sorted(keys)}, found {found}")
    return raw


def load_layer_document(raw: Any) -> SparseLayer:
    """Parse the probe's ``moe-layer-0.1`` descriptor and verify its geometry.

    Every check here is a byte-level agreement between the declared config, the
    declared weight list and the GGML storage layout the probe allocates; a
    layer whose numbers disagree with itself can never become a cost entry.
    """
    if not isinstance(raw, dict) or not _LAYER_KEYS - {"comment"} <= set(raw) <= _LAYER_KEYS:
        raise InvalidInput(f"sparse layer fields must be exactly {sorted(_LAYER_KEYS)} "
                           "(comment optional)")
    block = raw
    if block["format_version"] != LAYER_FORMAT_VERSION:
        raise InvalidInput(f"expected {LAYER_FORMAT_VERSION!r}, got {block['format_version']!r}")
    if block["provenance"] != "synthetic_layer":
        raise Unsupported("a measured sparse block is declared synthetic, never a pretrained path")
    if block["attention"] != "full_causal" or block["ffn"] != "moe":
        raise Unsupported("this sparse contract describes full_causal + moe only")
    if not isinstance(block["id"], str) or not block["id"]:
        raise InvalidInput("sparse layer id must be nonempty")
    raw_physical = block["physical"]
    if not isinstance(raw_physical, dict):
        raise InvalidInput("sparse layer physical must be an object")
    declared_residency = raw_physical.get("residency", "resident")
    if declared_residency not in RESIDENCY_REGIMES:
        raise InvalidInput(f"sparse layer physical.residency must be one of "
                           f"{sorted(RESIDENCY_REGIMES)}")
    declared_scope = raw_physical.get("process_scope", "per_case")
    if declared_scope not in PROCESS_SCOPES:
        raise InvalidInput(f"sparse layer physical.process_scope must be one of "
                           f"{sorted(PROCESS_SCOPES)}")
    physical = _exact_keys({k: v for k, v in raw_physical.items()
                            if k not in ("residency", "process_scope", "cpu_ids")},
                           _PHYSICAL_KEYS, "sparse layer physical")
    if physical["contract"] != CONTRACT:
        raise InvalidInput(f"sparse layer physical.contract must be {CONTRACT!r}")
    for key, value in (("scope", "complete_declared_decoder_block"),
                       ("input_boundary", "hidden_activation"),
                       ("port_layout", "materialized_stage_ports"),
                       ("kv_allocation", "logical_blocks"),
                       ("activation_dtype", "f32"), ("kv_dtype", KV_DTYPE)):
        if physical[key] != value:
            raise Unsupported(f"sparse layer physical.{key} must be {value!r}")
    if physical["excluded_operations"]:
        raise Unsupported("a sparse layer descriptor may not exclude block operations")
    config = validate_config(physical["config"])
    threads = integer(physical["threads"], "physical.threads", 1)
    declared_cpu_ids = dense_contract.signature_cpu_ids(raw_physical.get("cpu_ids"), threads)

    hidden, head_dim = config["hidden"], config["head_dim"]
    if integer(block["hidden_size"], "hidden_size") != hidden:
        raise InvalidInput("hidden_size disagrees with physical.config.hidden")
    if integer(block["hidden_bytes_per_token"], "hidden_bytes_per_token") != hidden * 4:
        raise InvalidInput("hidden_bytes_per_token must be hidden * 4 (f32 activation)")
    if integer(block["kv_bytes_per_token"], "kv_bytes_per_token") != 2 * config["kv_heads"] * head_dim * 2:
        raise InvalidInput("kv_bytes_per_token must be two f16 caches per kv head")

    shapes = weight_shapes(config)
    groups = physical["weight_groups"]
    if not isinstance(groups, dict) or set(groups) != set(STAGES[:3]):
        raise InvalidInput(f"weight_groups must name exactly {list(STAGES[:3])}")
    for stage in STAGES[:3]:
        roles = stage_weight_roles(config, stage)
        if groups[stage] != {role: "W_" + role for role in roles}:
            raise InvalidInput(f"weight_groups.{stage} disagrees with the declared roles")

    weights = {}
    for index, item in enumerate(block["weights"] if isinstance(block["weights"], list) else []):
        item = _exact_keys(item, _WEIGHT_KEYS, f"weights[{index}]")
        role = item["id"][2:] if isinstance(item["id"], str) and item["id"].startswith("W_") else None
        if role not in shapes or item["id"] in weights:
            raise InvalidInput(f"weights[{index}].id {item['id']!r} is not one declared role")
        item_id = item["id"]
        if item["name"] != item_id:
            raise InvalidInput(f"weights[{index}].name must equal its id")
        declared = tuple(integer(v, f"weights[{index}].shape", 1) for v in item["shape"])
        if declared != tuple(shapes[role]):
            raise InvalidInput(f"{item_id}: shape {list(declared)} != geometry {list(shapes[role])}")
        if item["dtype"] != config["weight_dtypes"][role]:
            raise InvalidInput(f"{item_id}: dtype disagrees with physical.config.weight_dtypes")
        _, expected_bytes = ggml_contiguous_layout(DTYPES[item["dtype"]],
                                                   (*declared, *([1] * (4 - len(declared)))))
        if integer(item["storage_bytes"], f"weights[{index}].storage_bytes", 1) != expected_bytes:
            raise InvalidInput(f"{item_id}: storage_bytes disagrees with its GGML layout")
        if item["initial_location"] != "dram":
            raise Unsupported(f"{item_id}: this contract measures resident DRAM weights")
        weights[item_id] = dict(item)
    expected = {role for stage in STAGES[:3] for role in stage_weight_roles(config, stage)}
    if set(weights) != {"W_" + role for role in expected}:
        raise InvalidInput("the declared weights are not exactly the sparse block's weights")

    tasks = block["tasks"]
    if not isinstance(tasks, list) or [task.get("id") for task in tasks] != list(STAGES[:3]):
        raise InvalidInput(f"tasks must list {list(STAGES[:3])} in order")
    for index, stage in enumerate(STAGES[:3]):
        if tasks[index].get("kind") != stage:
            raise InvalidInput(f"tasks[{index}].kind must be {stage!r}")
        if tasks[index].get("depends_on") != ([] if index == 0 else [STAGES[index - 1]]):
            raise InvalidInput(f"tasks[{index}].depends_on must follow the declared chain")
    return SparseLayer(id=block["id"], weights=tuple(weights.values()),
                       physical=SparsePhysical(config=config, threads=threads,
                                               weight_groups=groups,
                                               residency=declared_residency,
                                               process_scope=declared_scope,
                                               cpu_ids=(tuple(declared_cpu_ids)
                                                        if declared_cpu_ids else None)))
