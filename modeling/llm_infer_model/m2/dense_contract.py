"""One explicit CPU decoder-block contract shared by modeling and profiling.

Separate Q/K/V/(gate) matrices, RMSNorm, normal adjacent-pair RoPE, causal GQA,
O projection, SwiGLU and both residuals. All stage ports are materialized even
in the joint graph. This specifies a complete variant, not a model runtime.

Version 2 changes the KV cache element type to f16 and stores V transposed, the
way llama.cpp does. Both are visibility fixes rather than accuracy knobs: with an
f32 cache the declared block streamed twice the attention bytes a real f16 run
does, and with a plain V layout the probe paid a full-cache transpose copy per
step that no real implementation performs.
"""
from __future__ import annotations

import math
import struct
from typing import Any, Mapping

from ..tensor.spec import InvalidInput, Unsupported, ggml_contiguous_layout

CONTRACT = "ggml_cpu_dense_decoder_block_v2"
SUITE_SCHEMA = "dense_layer_measurement_suite_v1"
NATIVE_SCHEMA = "ggml_dense_layer_probe/1"
STAGES = ("projection", "attention", "dense_ffn", "block")
# Stages outside the declared decoder block that a complete token step still
# executes: the token-embedding lookup before the first block and the final
# RMSNorm plus LM head after the last one. They carry a vocabulary-wide matrix,
# so they are the only stages that declare ``vocab``/``word_dtype``.
BOUNDARY_STAGES = ("embedding", "head")
DTYPES = {"f32": "GGML_TYPE_F32", "f16": "GGML_TYPE_F16",
          "q4_k": "GGML_TYPE_Q4_K", "q6_k": "GGML_TYPE_Q6_K", "i32": "GGML_TYPE_I32"}
VARIANTS = {stage: f"ggml_cpu_dense_materialized_{stage}_v2"
            for stage in STAGES + BOUNDARY_STAGES}
SCOPES = {"embedding": "resident_repeated_declared_token_embedding",
          "head": "resident_repeated_declared_final_norm_and_lm_head"}
# Element type of the K and V caches. f16 is what llama-bench can run, so it is
# what the declared block must price; the probe asserts this exact value.
KV_DTYPE = "f16"
# How the timed sample finds its own operands. "resident" is the historical
# protocol: consecutive samples share whatever the previous one left in the
# caches, which prices a block that is the only tenant of the machine.
# "streaming" declares that the operands were evicted before each sample, which
# prices the same block inside a real token step where the KV history of a long
# context no longer fits in cache. The two are different protocols for the same
# physical task, so their prices never share a signature.
RESIDENCY_REGIMES = {"resident": "resident_warm_cache_shared_across_samples",
                     "streaming": "streamed_cold_cache_before_each_sample"}


# Which process a price was measured in. "per_case" is the historical protocol:
# every case gets its own probe process, so each entry of a table carries
# whatever machine state that process happened to start from, and the entry-to-
# entry ratios inherit that spread. "per_suite" measures the whole table inside
# one process in the declared order, so a state common to every entry cancels
# out of the ratios between them. Neither is more correct in absolute terms;
# they are different protocols for the same physical task, so prices from the
# two never share a signature.
PROCESS_SCOPES = {"per_case": "one_probe_process_per_case",
                  "per_suite": "one_probe_process_per_suite_in_declared_case_order"}


def signature_process_scope(declared: str | None) -> str | None:
    """Map a declared process scope to the signature argument, or None.

    None is what keeps the per-case signature byte-identical to every table
    measured before the scope was declarable; "per_case" is accepted as its
    explicit spelling and normalises to the same absent key.
    """
    if declared is None or declared == "per_case":
        return None
    if declared not in PROCESS_SCOPES:
        raise InvalidInput(f"process scope must be one of {sorted(PROCESS_SCOPES)}")
    return declared


def declared_process_scope(value: str | None) -> str | None:
    """Inverse of :data:`PROCESS_SCOPES`: the name a manifest's long form names.

    Manifests written before the scope was declarable carry no key at all, which
    is the per-case protocol.
    """
    if value is None or value == PROCESS_SCOPES["per_case"]:
        return None
    for name, text in PROCESS_SCOPES.items():
        if text == value:
            return name
    raise InvalidInput(f"process scope must be one of {sorted(PROCESS_SCOPES.values())}")


# How a decode price was estimated for a stage that reads only the current
# token. Projection and the feed-forward see one token and no history, so the
# work at visible_history 128, 512 and 1024 is the same work: those entries are
# replicates of one physical task, not three tasks. "own_shape" is the
# historical estimator -- each entry keeps the median of its own measurement --
# and stays absent from the signature, so every table exported before the
# policy existed keeps binding. "median_of_history_replicates" pools the
# replicates and is emitted, because a pooled price is not the price of the one
# shape it is filed under and must never answer to that shape's signature.
PRICE_POLICIES = {"own_shape": "each_shape_keeps_its_own_measured_median",
                  "median_of_history_replicates": "one_pooled_median_over_history_replicates"}


def signature_price_policy(declared: str | None) -> str | None:
    """Map a declared price policy to the signature argument, or None.

    None is what keeps the historical own-shape signature byte-identical;
    "own_shape" is accepted as its explicit spelling and normalises to the same
    absent key.
    """
    if declared is None or declared == "own_shape":
        return None
    if declared not in PRICE_POLICIES:
        raise InvalidInput(f"price policy must be one of {sorted(PRICE_POLICIES)}")
    return declared


def declared_price_policy(value: str | None) -> str | None:
    """Inverse of :data:`PRICE_POLICIES`: the short name a long form names."""
    if value is None or value == PRICE_POLICIES["own_shape"]:
        return None
    for name, text in PRICE_POLICIES.items():
        if text == value:
            return name
    raise InvalidInput(f"price policy must be one of {sorted(PRICE_POLICIES.values())}")


def history_invariant_kinds(stages: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """The declared stages whose decode work does not depend on visible history.

    Derived from a contract's own stage list rather than listed by hand: the
    projection and the feed-forward (whichever feed-forward that contract
    declares) read one token, attention reads the history.
    """
    return tuple(stage for stage in (STAGES if stages is None else stages)[:3] if stage != "attention")


def signature_residency(declared: str | None) -> str | None:
    """Map a declared regime to the signature argument, or None for the default.

    None is what keeps the resident signature byte-identical to every table
    measured before the regime was declarable; "resident" is accepted as its
    explicit spelling and normalises to the same absent key.
    """
    if declared is None or declared == "resident":
        return None
    if declared not in RESIDENCY_REGIMES:
        raise InvalidInput(f"residency must be one of {sorted(RESIDENCY_REGIMES)}")
    return declared


# Which logical CPUs the timed workers were allowed to run on. Not a geometry
# and not a thread count: the same two threads on the two SMT siblings of one
# physical core and on two cores of different CCDs price the same shape
# differently (1.30x on the machine that produced these tables), so a pinned
# placement is part of the physical task. Emitted only when a run declared it,
# which keeps every table measured before the dimension existed -- including
# the pinned campaigns that recorded the mask as a command detail only --
# byte-identical.
THREAD_PLACEMENTS = {"unpinned": "os_scheduler_choice_of_logical_cpus",
                     "pinned": "explicit_strict_cpu_mask"}


def signature_cpu_ids(declared: Any, threads: int) -> list[int] | None:
    """The logical-CPU mask that enters the physical signature, or None.

    None is what keeps an unpinned signature identical to every table measured
    before placement was declarable. A declared mask must name exactly one
    distinct logical CPU per thread, sorted and inside the single processor
    group the probe supports. Those are the same rules the runner's command
    parser applies, so a signature can never describe a placement the probe
    could not have applied.
    """
    integer(threads, "threads", 1)
    if declared is None:
        return None
    if isinstance(declared, (str, bytes)) or not isinstance(declared, (list, tuple)):
        raise InvalidInput("cpu ids must be a list of nonnegative logical CPU ids")
    ids = []
    for value in declared:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise InvalidInput("cpu id must be a nonnegative integer")
        ids.append(value)
    if len(set(ids)) != len(ids):
        raise InvalidInput("cpu ids repeat a logical CPU")
    if len(ids) != threads:
        raise InvalidInput(f"{len(ids)} cpu ids for {threads} threads: "
                           "every worker needs exactly one")
    if ids != sorted(ids):
        raise InvalidInput("cpu ids must be sorted")
    if any(cpu >= 64 for cpu in ids):
        raise Unsupported("cpu ids exceed the single processor group the probe supports")
    return ids


CONFIG_KEYS = {"hidden", "intermediate", "heads", "kv_heads", "head_dim",
               "attention_gate", "qk_norm", "rope_mode", "rope_dims", "rope_base",
               "norm_epsilon", "weight_dtypes"}
# Some checkpoints store a bias on the query/key/value projections (Qwen2-MoE does). It is
# a different graph from the unbiased one, so it is part of the config when declared. The
# key is optional and a declared ``false`` normalizes to absent: both describe the same
# graph and therefore keep one signature.
OPTIONAL_KEYS = {"attention_bias"}
BIAS_ROLES = ("q_bias", "k_bias", "v_bias")
# Declared only by the two boundary stages. The two vocabulary-wide matrices are
# declared separately because real checkpoints quantize them separately: a
# Q4_K_M file stores the token embedding as q4_k and the LM head as q6_k.
WORD_KEYS = {"vocab", "token_embd_dtype", "lm_head_dtype"}
EXCLUDED = ("allocation", "quantization", "io", "external_concurrency")


def integer(value: Any, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InvalidInput(f"{label} must be an integer >= {minimum}")
    return value


def float32_parameter(value: Any, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise InvalidInput(f"{label} must be a finite numeric parameter")
    try:
        applied = struct.unpack("f", struct.pack("f", value))[0]
    except (OverflowError, struct.error) as error:
        raise InvalidInput(f"{label} does not fit GGML float32") from error
    if not math.isfinite(applied) or applied <= minimum:
        raise InvalidInput(f"{label} must remain > {minimum} after GGML float32 conversion")
    return applied


def weight_shapes(config: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    h, i, d = config["hidden"], config["intermediate"], config["head_dim"]
    q, kv = config["heads"] * d, config["kv_heads"] * d
    result = {"input_norm": (h,), "q": (h, q), "k": (h, kv), "v": (h, kv),
              "output": (q, h), "post_attention_norm": (h,),
              "gate": (h, i), "up": (h, i), "down": (i, h)}
    if config["attention_gate"] == "sigmoid":
        result["gate_proj"] = (h, q)
    # "rms" normalizes each head on its own; "rms_hidden" normalizes the whole
    # head-concatenated projection, so its gain vector is the hidden width
    # (OLMoE stores attn_q_norm as [hidden] and applies it before the reshape).
    if config["qk_norm"] == "rms":
        result.update(q_norm=(d,), k_norm=(d,))
    elif config["qk_norm"] == "rms_hidden":
        result.update(q_norm=(h,), k_norm=(h,))
    # A projection bias is one row per output channel: it broadcasts over the token axis.
    if config.get("attention_bias"):
        result.update(q_bias=(q,), k_bias=(kv,), v_bias=(kv,))
    if "vocab" in config:
        result.update(final_norm=(h,), lm_head=(h, config["vocab"]),
                      token_embd=(h, config["vocab"]))
    return result


def stage_weight_roles(config: Mapping[str, Any], kind: str) -> tuple[str, ...]:
    if kind == "projection":
        roles = ["input_norm", "q", "k", "v"]
        if config.get("attention_bias"):
            roles += list(BIAS_ROLES)
        if config["attention_gate"] == "sigmoid":
            roles.append("gate_proj")
        if config["qk_norm"] != "none":
            roles += ["q_norm", "k_norm"]
        return tuple(roles)
    if kind == "attention":
        return ("output",)
    if kind == "dense_ffn":
        return ("post_attention_norm", "gate", "up", "down")
    if kind == "block":
        return tuple(role for stage in STAGES[:3] for role in stage_weight_roles(config, stage))
    if kind == "head":
        return ("final_norm", "lm_head")
    if kind == "embedding":
        return ("token_embd",)
    raise Unsupported(f"unsupported dense stage {kind!r}")


def validate_config(raw: Any) -> dict[str, Any]:
    allowed = CONFIG_KEYS | WORD_KEYS | OPTIONAL_KEYS
    if not isinstance(raw, dict) or not (CONFIG_KEYS <= set(raw) <= allowed):
        raise InvalidInput(f"dense config must declare exactly {sorted(CONFIG_KEYS)}")
    word = WORD_KEYS & set(raw)
    if word and word != WORD_KEYS:
        raise InvalidInput(f"vocabulary-wide stages must declare both {sorted(WORD_KEYS)}")
    config = {key: integer(raw[key], f"config.{key}") for key in
              ("hidden", "intermediate", "heads", "kv_heads", "head_dim", "rope_dims")}
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
    for key, allowed in (("attention_gate", ("none", "sigmoid")),
                         ("qk_norm", ("none", "rms", "rms_hidden"))):
        if raw[key] not in allowed:
            raise InvalidInput(f"config.{key} must be one of {allowed}")
        config[key] = raw[key]
    if raw["rope_mode"] != "normal":
        raise Unsupported("this dense contract supports normal adjacent-pair RoPE only; no fallback")
    config["rope_mode"] = "normal"
    if "attention_bias" in raw:
        if not isinstance(raw["attention_bias"], bool):
            raise InvalidInput("config.attention_bias must be a boolean")
        # Absent and false describe the same graph; declaring true is what changes it.
        if raw["attention_bias"]:
            config["attention_bias"] = True
    config["rope_base"] = float32_parameter(raw["rope_base"], "config.rope_base", 1.0)
    config["norm_epsilon"] = float32_parameter(raw["norm_epsilon"], "config.norm_epsilon")
    shapes = weight_shapes(config)
    dtypes = raw["weight_dtypes"]
    # A case may declare only the roles its stages use.  This is what lets a model whose
    # feed-forward is sparse -- and therefore has no dense gate/up/down at all -- still
    # declare its projection and attention weights.  `validate_case` then requires the
    # declared set to cover exactly the stage the case names, so a case cannot be measured
    # while silently missing a weight its stage touches.
    if not isinstance(dtypes, dict) or not dtypes or not set(dtypes) <= set(shapes):
        raise InvalidInput(f"weight_dtypes must declare a nonempty subset of {sorted(shapes)}")
    # The subset may only drop roles that a stage can choose not to run.  A role the config
    # turns on -- the sigmoid gate, QK norm, the vocabulary matrices, the projection bias --
    # is part of every graph that declares it, so it cannot be left undeclared.
    implied = set()
    if config["attention_gate"] == "sigmoid":
        implied.add("gate_proj")
    if config["qk_norm"] != "none":
        implied |= {"q_norm", "k_norm"}
    if config.get("attention_bias"):
        implied |= set(BIAS_ROLES)
    if word:
        implied |= {"token_embd", "lm_head"}
    if implied - set(dtypes):
        raise InvalidInput(f"weight_dtypes omits {sorted(implied - set(dtypes))}, "
                           f"which the declared config turns on")
    config["weight_dtypes"] = {}
    for role in dtypes:
        shape = shapes[role]
        dtype = dtypes[role]
        if dtype not in ("f32", "f16", "q4_k", "q6_k"):
            raise InvalidInput(f"unsupported weight dtype for {role}: {dtype!r}")
        if role.endswith("norm") and dtype != "f32":
            raise Unsupported("RMSNorm gamma vectors in this contract must be f32")
        ggml_contiguous_layout(DTYPES[dtype], (*shape, *([1] * (4-len(shape)))))
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
        raise InvalidInput("dense case has missing or unsupported fields")
    if not isinstance(raw["id"], str) or not raw["id"]:
        raise InvalidInput("dense case id must be nonempty")
    if raw["kind"] not in STAGES + BOUNDARY_STAGES or raw["phase"] not in ("prefill", "decode"):
        raise InvalidInput("dense case must select an explicit stage and phase")
    if integer(raw["batch"], "case.batch") != 1:
        raise Unsupported("dense layer measurement supports batch=1 only")
    tokens = integer(raw["tokens"], "case.tokens")
    history = integer(raw["visible_history"], "case.visible_history", 0)
    if raw["phase"] == "decode" and tokens != 1:
        raise InvalidInput("decode requires one new token")
    if history + tokens > 0x7fffffff:
        raise Unsupported("absolute positions must fit the GGML I32 RoPE/mask interface")
    result = {**raw, "config": validate_config(raw["config"])}
    # A boundary stage has no cache, so a nonzero history would only fragment its
    # price key without changing the work performed.
    if raw["kind"] in BOUNDARY_STAGES:
        if history:
            raise InvalidInput("embedding/head cases must declare visible_history=0")
        if "vocab" not in result["config"]:
            raise InvalidInput("embedding/head cases must declare config.vocab and config.word_dtype")
    elif "vocab" in result["config"]:
        raise InvalidInput("decoder-block stages must not declare config.vocab")
    # The declared subset must cover the stage the case names: a case that lacks one of
    # its own weights would otherwise measure a different graph under this stage's name.
    needed = set(stage_weight_roles(result["config"], raw["kind"]))
    missing = needed - set(result["config"]["weight_dtypes"])
    if missing:
        raise InvalidInput(f"{raw['kind']} case declares no weight dtype for {sorted(missing)}")
    if "threads" in raw:
        result["threads"] = integer(raw["threads"], "case.threads")
    for key in ("comment", "source"):
        if key in raw and not isinstance(raw[key], (str, dict)):
            raise InvalidInput(f"case.{key} must be descriptive string/object")
    if "metadata" in raw and not isinstance(raw["metadata"], dict):
        raise InvalidInput("case.metadata must be descriptive object")
    return result


def build_plan(suite: Any, threads: int) -> list[dict[str, Any]]:
    integer(threads, "threads")
    if not isinstance(suite, dict) or suite.get("schema_version") != SUITE_SCHEMA:
        raise InvalidInput(f"dense suite schema_version must be {SUITE_SCHEMA}")
    if set(suite) - {"schema_version", "cases", "comment"}:
        raise InvalidInput("dense suite contains unsupported fields")
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise InvalidInput("dense suite must have cases")
    result = []
    for raw in cases:
        case = validate_case(raw)
        if "threads" in case and case["threads"] != threads:
            raise InvalidInput("case threads differ from requested measurement protocol")
        result.append({**case, "threads": threads})
    if len({case["id"] for case in result}) != len(result):
        raise InvalidInput("dense case ids must be unique")
    return result


def operations(config: Mapping[str, Any], kind: str) -> list[str]:
    if kind == "projection":
        result = ["input_RMSNorm", "input_norm_scale", "Q_projection", "K_projection", "V_projection"]
        if config.get("attention_bias"):
            result += ["Q_bias_add", "K_bias_add", "V_bias_add"]
        if config["attention_gate"] == "sigmoid":
            result += ["gate_projection"]
        if config["qk_norm"] == "rms":
            result += ["Q_per_head_RMSNorm", "Q_norm_scale", "K_per_head_RMSNorm", "K_norm_scale"]
        elif config["qk_norm"] == "rms_hidden":
            # One normalization over the whole projection output, before the
            # head split, then RoPE per head -- OLMoE's order.
            result += ["Q_RMSNorm", "Q_norm_scale", "K_RMSNorm", "K_norm_scale"]
        result += ["RoPE_Q", "RoPE_K", "materialize_Q", "materialize_K", "materialize_V"]
        if config["attention_gate"] == "sigmoid":
            result += ["materialize_gate"]
        return result
    if kind == "attention":
        result = ["append_new_KV", "QK_transpose", "scale", "causal_mask", "softmax", "PV"]
        if config["attention_gate"] == "sigmoid":
            result += ["sigmoid_gate", "attention_times_gate"]
        return result + ["output_projection", "attention_residual_add", "materialize_attention_hidden"]
    if kind == "dense_ffn":
        return ["post_attention_RMSNorm", "post_attention_norm_scale", "gate_projection", "up_projection",
                "SiLU_gate", "gate_times_up", "down_projection", "ffn_residual_add", "materialize_hidden"]
    if kind == "embedding":
        return ["token_embedding_lookup", "materialize_hidden"]
    if kind == "head":
        return ["final_RMSNorm", "final_norm_scale", "lm_head_projection", "materialize_logits"]
    if kind == "block":
        return [op for stage in STAGES[:3] for op in operations(config, stage)]
    raise Unsupported(f"unsupported dense stage {kind!r}")


def signature(config: dict[str, Any], kind: str, phase: str, tokens: int,
              visible_history: int, threads: int, batch: int = 1,
              residency: str | None = None,
              process_scope: str | None = None,
              cpu_ids: list[int] | None = None,
              price_policy: str | None = None) -> dict[str, Any]:
    case = validate_case(dict(id="signature", kind=kind, phase=phase, batch=batch,
        tokens=tokens, visible_history=visible_history, config=config, threads=threads))
    config = case["config"]
    result = {"contract": CONTRACT, "scope": SCOPES.get(kind, "complete_declared_decoder_block"),
            "kind": kind,
            "device": "cpu", "backend": "CPU", "implementation_variant": VARIANTS[kind],
            "config": config, "phase": phase, "batch": batch, "tokens": tokens,
            "visible_history": visible_history, "threads_requested": threads, "threads_applied": threads,
             "thread_management": "persistent_ggml_threadpool",
            "activation_dtype": "f32", "kv_dtype": KV_DTYPE, "position_dtype": "i32",
            "numeric_parameter_semantics": "GGML_float32_parameters",
            "timing_scope": "single_completed_cpu_graph_wall_ns", "setup_compute_repetitions": 1,
            "stage_boundary_policy": "timed_materialized_copies_in_isolated_and_joint_graphs",
            "input_lineage": "same_seed_complete_upstream_preparation_outside_timing",
            "operations": operations(config, kind), "excluded_operations": list(EXCLUDED),
            "original_model_implementation_reproduced": False}
    # Residency is a protocol, not a geometry: the same shape priced with its
    # operands resident and priced with them evicted must not answer to one
    # signature. Declared only when it differs from the default resident regime,
    # so every table measured before the regime existed keeps its identity.
    if residency is not None:
        result["residency_regime"] = RESIDENCY_REGIMES[residency]
    # Same rule for the process scope: a table measured inside one process is a
    # different protocol from one measured case by case, so the two never answer
    # to one signature. Only the non-default spelling is emitted.
    if process_scope is not None:
        result["measurement_process_scope"] = PROCESS_SCOPES[process_scope]
    # And the same for the placement the workers were pinned to. An unpinned run
    # leaves the mask to the OS and emits nothing, so it keeps the historical
    # signature; a pinned one declares both the mask and the fact that it was
    # strict, and neither can then answer to the other's price.
    if cpu_ids is not None:
        result["cpu_mask"] = signature_cpu_ids(cpu_ids, threads)
        result["thread_placement"] = THREAD_PLACEMENTS["pinned"]
    # And the same for how the price was estimated. A pooled median is still a
    # measurement of this task's work, but it is not this shape's own reading,
    # so it declares which estimator produced it and the two never share a
    # signature. The default emits nothing, keeping old tables byte-identical.
    if price_policy is not None:
        result["history_invariant_price_policy"] = PRICE_POLICIES[price_policy]
    return result


def ports(config: Mapping[str, Any], kind: str, tokens: int) -> tuple[dict, dict]:
    h, d, nh, nk = config["hidden"], config["head_dim"], config["heads"], config["kv_heads"]
    hidden = {"dtype": "f32", "ne": (h, tokens, 1, 1)}
    positions = {"dtype": "i32", "ne": (tokens, 1, 1, 1)}
    projection = {"q": {"dtype": "f32", "ne": (d, tokens, nh, 1)},
                  "k": {"dtype": "f32", "ne": (d, tokens, nk, 1)},
                  "v": {"dtype": "f32", "ne": (d, tokens, nk, 1)}}
    if config["attention_gate"] == "sigmoid":
        projection["gate"] = {"dtype": "f32", "ne": (d*nh, tokens, 1, 1)}
    if kind == "projection":
        return {"hidden": hidden, "positions": positions}, projection
    if kind == "attention":
        return {"hidden": hidden, **projection}, {"attention_hidden": hidden}
    if kind == "dense_ffn":
        return {"attention_hidden": hidden}, {"hidden_out": hidden}
    if kind == "block":
        return {"hidden": hidden, "positions": positions}, {**projection, "attention_hidden": hidden, "hidden_out": hidden}
    if kind == "embedding":
        return {"token_ids": {"dtype": "i32", "ne": (tokens, 1, 1, 1)}}, {"hidden_out": hidden}
    if kind == "head":
        return {"hidden": hidden}, {"logits": {"dtype": "f32", "ne": (config["vocab"], tokens, 1, 1)}}
    raise Unsupported(f"unsupported dense stage {kind!r}")


def resident_accounting(config: dict[str, Any], kind: str, tokens: int, history: int,
                        kv_element_bytes: int = 2) -> dict[str, Any]:
    """Logical resident bytes for one declared stage or block.

    ``kv_element_bytes`` is per cache element *per cache*: the K and the V cache
    are counted separately, so the per-token figure is twice this value times
    ``kv_heads * head_dim``. The CPU contract stores both caches as f16; the CUDA
    envelope still declares an f32 cache and passes 4.
    """
    config = validate_config(config)
    integer(tokens, "tokens"); integer(history, "history", 0)
    if kv_element_bytes not in (2, 4):
        raise InvalidInput("kv_element_bytes must be 2 (f16) or 4 (f32)")
    shapes = weight_shapes(config)
    weight_bytes = {}
    for role in stage_weight_roles(config, kind):
        shape = shapes[role]
        _, size = ggml_contiguous_layout(DTYPES[config["weight_dtypes"][role]], (*shape, *([1]*(4-len(shape)))))
        weight_bytes[role] = size
    inputs, outputs = ports(config, kind, tokens)
    def sizes(values):
        return {name: math.prod(port["ne"]) * 4 for name, port in values.items()}
    input_sizes, output_sizes = sizes(inputs), sizes(outputs)
    kv = 2 * kv_element_bytes * config["kv_heads"] * config["head_dim"] if kind in ("attention", "block") else 0
    result = {"weight_roles_bytes": weight_bytes, "input_ports_bytes": input_sizes,
              "output_ports_bytes": output_sizes, "weight_bytes": sum(weight_bytes.values()),
              "input_bytes": sum(input_sizes.values()), "output_bytes": sum(output_sizes.values()),
              "kv_history_bytes": history*kv, "kv_new_bytes": tokens*kv, "kv_cache_bytes": (history+tokens)*kv}
    result["resident_logical_bytes"] = result["weight_bytes"]+result["input_bytes"]+result["output_bytes"]+result["kv_cache_bytes"]
    return result


def native_arguments(case: dict[str, Any], warmup: int, samples: int, seed: int,
                     evict_bytes: int = 0) -> list[str]:
    case = validate_case(case)
    integer(warmup, "warmup", 0); integer(samples, "samples", 3); integer(seed, "seed", 0)
    integer(evict_bytes, "evict_bytes", 0)
    if samples > 10000 or seed > 0xffffffff or evict_bytes > (1 << 34):
        raise InvalidInput("samples, seed or evict_bytes exceed native protocol range")
    if "threads" not in case:
        raise InvalidInput("native invocation requires explicit threads")
    config = case["config"]
    args = ["--stage", case["kind"]]
    for key in ("hidden", "intermediate", "heads", "kv_heads", "head_dim"):
        args += ["--"+key.replace("_", "-"), str(config[key])]
    args += ["--q", str(case["tokens"]), "--history", str(case["visible_history"]),
             "--threads", str(case["threads"]), "--warmup", str(warmup), "--samples", str(samples), "--seed", str(seed)]
    for key, flag in (("attention_gate", "attention-gate"), ("qk_norm", "qk-norm"),
                      ("rope_mode", "rope-mode"), ("rope_dims", "rope-dims"),
                      ("rope_base", "rope-base"), ("norm_epsilon", "norm-eps")):
        args += ["--"+flag, str(config[key])]
    if config.get("attention_bias"):
        args += ["--attention-bias", "1"]
    flags = {"q": "q-dtype", "k": "k-dtype", "v": "v-dtype", "gate_proj": "gate-proj-dtype",
             "output": "o-dtype", "gate": "ffn-gate-dtype", "up": "up-dtype", "down": "down-dtype",
             "q_bias": "q-bias-dtype", "k_bias": "k-bias-dtype", "v_bias": "v-bias-dtype"}
    for role, flag in flags.items():
        if role in config["weight_dtypes"]:
            args += ["--"+flag, config["weight_dtypes"][role]]
    args += ["--kv-dtype", KV_DTYPE]
    # The residency regime is part of the protocol, not of the geometry: a price
    # taken with the operands evicted is not the price of the resident block. It
    # is emitted only when declared so a zero-eviction command stays exactly what
    # earlier campaigns ran.
    if evict_bytes:
        args += ["--evict-bytes", str(evict_bytes)]
    if "vocab" in config:
        args += ["--vocab", str(config["vocab"])]
        for role, flag in (("lm_head", "lm-head-dtype"), ("token_embd", "token-embd-dtype")):
            args += ["--"+flag, config["weight_dtypes"][role]]
    return args
