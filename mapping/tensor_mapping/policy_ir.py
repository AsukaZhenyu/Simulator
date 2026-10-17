"""Small, explicit causal policy configuration; not a general DSL or search."""
from __future__ import annotations

from dataclasses import dataclass
import math

from llm_infer_model.m2.causal import RevealedRoute
from llm_infer_model.tensor.spec import InvalidInput, Unsupported

POLICY_VERSION = "causal-policy-0.1"
POLICY_VERSION_2 = "causal-policy-0.2"
HISTORY_VERSION = "causal-history-0.1"
HISTORY_VERSION_2 = "causal-history-0.2"
OPERATION_KINDS = frozenset((
    "projection", "attention", "attention_core", "attention_output", "router", "dense_ffn", "expert_ffn",
    "shared_expert_ffn", "combine", "gather_rows", "scatter_rows",
    "expert_gate", "expert_up", "expert_swiglu", "expert_down", "combine_rows",
    "ffn_norm", "ffn_gate", "ffn_up", "ffn_swiglu", "ffn_down", "ffn_sum", "ffn_residual",
    "attention_qk", "attention_softmax", "attention_pv", "attention_projection",
))
EXPERT_COMPUTE_KINDS = frozenset((
    "expert_ffn", "shared_expert_ffn", "expert_gate", "expert_up",
    "expert_swiglu", "expert_down",
    "ffn_gate", "ffn_up", "ffn_swiglu", "ffn_down", "ffn_sum",
))


@dataclass(frozen=True)
class PolicyIR:
    placements: tuple[tuple[str, str], ...]
    expert_device_rule: str
    cache_rule: str
    prefetch_enabled: bool
    prefetch_max_count: int
    cache_budget_bytes: int
    release_rule: str = "none"
    format_version: str = POLICY_VERSION
    request_placements: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = ()
    mrs_alpha: float | None = None
    mrs_top_p: int | None = None


def _object(value, keys, where):
    if not isinstance(value, dict):
        raise InvalidInput(f"{where} must be an object")
    if set(value) != set(keys):
        raise InvalidInput(f"{where} requires exactly {sorted(keys)}")
    return value


def _integer(value, where):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InvalidInput(f"{where} must be a non-negative integer")
    return value


def _choice(value, choices, where):
    if not isinstance(value, str) or value not in choices:
        raise Unsupported(f"{where} must be one of {sorted(choices)}")
    return value


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def parse_policy(document) -> PolicyIR:
    if not isinstance(document, dict):
        raise InvalidInput("policy must be an object")
    if "format_version" not in document:
        raise InvalidInput("policy requires format_version")
    version = document.get("format_version")
    if version not in (POLICY_VERSION, POLICY_VERSION_2):
        raise Unsupported(f"policy.format_version must be {POLICY_VERSION} or {POLICY_VERSION_2}")
    fields = {"format_version", "placements", "experts", "cache", "prefetch", "release"}
    if version == POLICY_VERSION_2:
        fields.add("request_placements")
    _object(document, fields, "policy")
    placements = document["placements"]
    if not isinstance(placements, dict) or not placements:
        raise InvalidInput("policy.placements must be a nonempty object")
    for kind, device in placements.items():
        kinds = OPERATION_KINDS if version == POLICY_VERSION_2 else OPERATION_KINDS - {"attention_core", "attention_output"}
        _choice(kind, kinds, "policy.placements kind")
        _choice(device, ("cpu", "gpu"), f"policy.placements.{kind}")
    experts = _object(document["experts"], ("device_rule",), "policy.experts")
    rules = {"fixed", "resident_first", "current_cost"}
    if version == POLICY_VERSION_2:
        rules.update(("fiddler_paths", "hybrimoe_queues"))
    rule = _choice(experts["device_rule"], rules,
                   "policy.experts.device_rule")
    cache = document["cache"]
    if not isinstance(cache, dict):
        raise InvalidInput("policy.cache must be an object")
    cache_rules = {"none", "lru", "history_frequency"}
    if version == POLICY_VERSION_2:
        cache_rules.add("mrs")
    cache_rule = _choice(cache.get("rule"), cache_rules, "policy.cache.rule")
    _object(cache, ("rule", "alpha", "top_p") if cache_rule == "mrs" else ("rule",), "policy.cache")
    alpha = top_p = None
    if cache_rule == "mrs":
        alpha = cache["alpha"]
        if not _finite_number(alpha) or not 0 < alpha <= 1:
            raise InvalidInput("policy.cache.alpha must be a finite number in (0,1]")
        alpha = float(alpha)
        top_p = _integer(cache["top_p"], "policy.cache.top_p")
        if not top_p:
            raise InvalidInput("policy.cache.top_p must be positive")
    release = _object(document["release"], ("rule",), "policy.release")
    release_rule = _choice(release["rule"], ("none", "last_use"), "policy.release.rule")
    prefetch = _object(document["prefetch"], ("enabled", "max_count", "cache_budget_bytes"), "policy.prefetch")
    if not isinstance(prefetch["enabled"], bool):
        raise InvalidInput("policy.prefetch.enabled must be boolean")
    count = _integer(prefetch["max_count"], "policy.prefetch.max_count")
    budget = _integer(prefetch["cache_budget_bytes"], "policy.prefetch.cache_budget_bytes")
    if prefetch["enabled"] and (not count or not budget):
        raise InvalidInput("enabled prefetch requires positive max_count and cache_budget_bytes")
    if not prefetch["enabled"] and (count or budget):
        raise InvalidInput("disabled prefetch requires max_count=0 and cache_budget_bytes=0")
    request_placements = []
    if version == POLICY_VERSION_2:
        overrides = document["request_placements"]
        if not isinstance(overrides, dict):
            raise InvalidInput("policy.request_placements must be an explicit object")
        for request, bindings in overrides.items():
            if not isinstance(request, str) or not request:
                raise InvalidInput("policy.request_placements request ids must be nonempty strings")
            if not isinstance(bindings, dict) or not bindings:
                raise InvalidInput(f"policy.request_placements.{request} must be nonempty")
            for kind, device in bindings.items():
                _choice(kind, OPERATION_KINDS, f"policy.request_placements.{request} kind")
                _choice(device, ("cpu", "gpu"), f"policy.request_placements.{request}.{kind}")
            request_placements.append((request, tuple(sorted(bindings.items()))))
    return PolicyIR(tuple(sorted(placements.items())), rule, cache_rule,
                    prefetch["enabled"], count, budget, release_rule, version,
                    tuple(sorted(request_placements)), alpha, top_p)


def policy_document(policy: PolicyIR) -> dict:
    if not isinstance(policy, PolicyIR):
        raise InvalidInput("policy_document requires PolicyIR")
    if not isinstance(policy.placements, tuple) or any(
            not isinstance(pair, tuple) or len(pair) != 2 or
            not all(isinstance(value, str) for value in pair) for pair in policy.placements):
        raise InvalidInput("PolicyIR.placements must contain frozen string pairs")
    if len(dict(policy.placements)) != len(policy.placements):
        raise InvalidInput("PolicyIR.placements contains duplicate kinds")
    cache = {"rule": policy.cache_rule}
    if policy.cache_rule == "mrs":
        cache.update(alpha=policy.mrs_alpha, top_p=policy.mrs_top_p)
    elif policy.mrs_alpha is not None or policy.mrs_top_p is not None:
        raise InvalidInput("MRS parameters are only valid for cache.rule=mrs")
    document = {"format_version": policy.format_version, "placements": dict(policy.placements),
            "experts": {"device_rule": policy.expert_device_rule},
            "cache": cache,
            "release": {"rule": policy.release_rule},
            "prefetch": {"enabled": policy.prefetch_enabled, "max_count": policy.prefetch_max_count,
                         "cache_budget_bytes": policy.cache_budget_bytes}}
    if policy.format_version == POLICY_VERSION_2:
        if not isinstance(policy.request_placements, tuple):
            raise InvalidInput("PolicyIR.request_placements must be frozen pairs")
        requests = {}
        for request, bindings in policy.request_placements:
            if not isinstance(request, str) or request in requests or not isinstance(bindings, tuple):
                raise InvalidInput("PolicyIR.request_placements contains invalid/duplicate request")
            if any(not isinstance(pair, tuple) or len(pair) != 2 or
                   not all(isinstance(value, str) for value in pair) for pair in bindings):
                raise InvalidInput("PolicyIR request bindings must be frozen string pairs")
            if len(dict(bindings)) != len(bindings):
                raise InvalidInput("PolicyIR request bindings contain duplicate kinds")
            requests[request] = dict(bindings)
        document["request_placements"] = requests
    elif policy.request_placements:
        raise InvalidInput("request_placements requires causal-policy-0.2")
    return document


def history_document(entries: tuple[RevealedRoute, ...]) -> dict:
    if not isinstance(entries, tuple) or any(not isinstance(entry, RevealedRoute) for entry in entries):
        raise InvalidInput("history must contain frozen RevealedRoute values")
    version = HISTORY_VERSION_2 if any(getattr(entry, "router_scores_scope", "not_reported") != "not_reported"
                                      or getattr(entry, "router_scores", ()) for entry in entries) else HISTORY_VERSION
    rows = [
        {"request_id": entry.request_id, "layer_id": entry.layer_id,
         "absolute_position": entry.absolute_position, "expert_ids": list(entry.expert_ids),
         "scores": None if entry.scores is None else list(entry.scores)} for entry in entries]
    if version == HISTORY_VERSION_2:
        for row, entry in zip(rows, entries):
            row.update(router_scores_scope=getattr(entry, "router_scores_scope", "not_reported"),
                       router_scores=[{"expert_id": expert, "score": score}
                                      for expert, score in getattr(entry, "router_scores", ())])
    return {"format_version": version, "entries": rows}


def parse_history(document) -> tuple[RevealedRoute, ...]:
    _object(document, ("format_version", "entries"), "history")
    version = document["format_version"]
    if version not in (HISTORY_VERSION, HISTORY_VERSION_2):
        raise Unsupported(f"history.format_version must be {HISTORY_VERSION} or {HISTORY_VERSION_2}")
    if not isinstance(document["entries"], list):
        raise InvalidInput("history.entries must be an array")
    entries, seen = [], set()
    for index, entry in enumerate(document["entries"]):
        label = f"history.entries[{index}]"
        fields = {"request_id", "layer_id", "absolute_position", "expert_ids", "scores"}
        if version == HISTORY_VERSION_2:
            fields.update(("router_scores", "router_scores_scope"))
        _object(entry, fields, label)
        for name in ("request_id", "layer_id"):
            if not isinstance(entry[name], str) or not entry[name]:
                raise InvalidInput(f"{label}.{name} must be a nonempty string")
        position = _integer(entry["absolute_position"], f"{label}.absolute_position")
        experts = entry["expert_ids"]
        if not isinstance(experts, list) or not experts:
            raise InvalidInput(f"{label}.expert_ids must be a nonempty array")
        for expert in experts:
            _integer(expert, f"{label}.expert_ids")
        if len(set(experts)) != len(experts):
            raise InvalidInput(f"{label}.expert_ids contains duplicates")
        scores = entry["scores"]
        if scores is not None:
            if not isinstance(scores, list) or len(scores) != len(experts):
                raise InvalidInput(f"{label}.scores must match expert_ids or be null")
            if any(not _finite_number(score) for score in scores):
                raise InvalidInput(f"{label}.scores must be finite numbers")
            scores = tuple(float(score) for score in scores)
        key = (entry["request_id"], entry["layer_id"], position)
        if key in seen:
            raise InvalidInput(f"{label} duplicates a revealed-history key")
        seen.add(key)
        kwargs = {}
        if version == HISTORY_VERSION_2:
            scope = _choice(entry["router_scores_scope"], ("not_reported", "all_experts"),
                            f"{label}.router_scores_scope")
            vector = entry["router_scores"]
            if not isinstance(vector, list):
                raise InvalidInput(f"{label}.router_scores must be an array")
            checked = []
            for offset, score_entry in enumerate(vector):
                _object(score_entry, ("expert_id", "score"), f"{label}.router_scores[{offset}]")
                expert = _integer(score_entry["expert_id"], f"{label}.router_scores expert_id")
                value = score_entry["score"]
                if not _finite_number(value):
                    raise InvalidInput(f"{label}.router_scores must contain finite scores")
                checked.append((expert, float(value)))
            ids = [expert for expert, _ in checked]
            if len(ids) != len(set(ids)):
                raise InvalidInput(f"{label}.router_scores contains duplicate experts")
            if scope == "not_reported" and checked:
                raise InvalidInput(f"{label}.not_reported cannot contain router scores")
            if scope == "all_experts":
                if not checked or set(ids) != set(range(len(checked))):
                    raise InvalidInput(f"{label}.all_experts requires a complete contiguous expert vector")
                if any(not 0 <= value <= 1 for _, value in checked) or not math.isclose(
                        sum(value for _, value in checked), 1.0, rel_tol=0, abs_tol=1e-6):
                    raise InvalidInput(f"{label}.all_experts probabilities must be in [0,1] and sum to one")
                if any(expert not in ids for expert in experts):
                    raise InvalidInput(f"{label}.activated expert absent from complete router scores")
                chosen = [value for expert, value in checked if expert in experts]
                others = [value for expert, value in checked if expert not in experts]
                if others and min(chosen) < max(others) - 1e-12:
                    raise InvalidInput(f"{label}.selected experts contradict full router top-k")
                if scores is not None and any(score != dict(checked)[expert]
                                              for expert, score in zip(experts, scores)):
                    raise InvalidInput(f"{label}.selected scores disagree with all-expert scores")
            kwargs = {"router_scores_scope": scope, "router_scores": tuple(sorted(checked))}
        entries.append(RevealedRoute(key[0], key[1], position, tuple(experts), scores, **kwargs))
    # EMA replay must preserve the declared chronological row order. Legacy
    # frequency-only history keeps its previous canonical sorting behavior.
    return (tuple(entries) if version == HISTORY_VERSION_2 else
            tuple(sorted(entries, key=lambda entry: (entry.request_id, entry.layer_id, entry.absolute_position))))
