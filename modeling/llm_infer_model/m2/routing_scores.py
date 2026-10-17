"""Private full router-score evidence, revealed only after each Router finishes.

This is an explicit evaluation input, not a prediction or a measured kernel.
Missing full scores stay unknown; selected top-k scores never fill the rest.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

from ..tensor.spec import InvalidInput, Unsupported
from .causal import CausalM2Session


@dataclass(frozen=True)
class RouterScoreRow:
    request_id: str
    layer_id: str
    absolute_position: int
    scores: tuple[tuple[int, float], ...]


def parse_routing_scores(document) -> tuple[RouterScoreRow, ...]:
    if not isinstance(document, dict) or set(document) != {"format_version", "provenance", "entries"}:
        raise InvalidInput("routing scores require format_version, provenance, entries")
    if document["format_version"] != "routing-scores-0.1":
        raise Unsupported("routing scores require routing-scores-0.1")
    if document["provenance"] not in ("synthetic_router_probabilities", "recorded_router_probabilities"):
        raise Unsupported("routing score provenance must explicitly identify synthetic or recorded probabilities")
    if not isinstance(document["entries"], list):
        raise InvalidInput("routing scores entries must be an array")
    entries, seen = [], set()
    for row in document["entries"]:
        if not isinstance(row, dict) or set(row) != {"request_id", "layer_id", "absolute_position", "router_scores"}:
            raise InvalidInput("routing score entry has unknown or missing fields")
        if any(not isinstance(row[k], str) or not row[k] for k in ("request_id", "layer_id")):
            raise InvalidInput("routing scores need nonempty request/layer IDs")
        if type(row["absolute_position"]) is not int or row["absolute_position"] < 0:
            raise InvalidInput("routing score position must be a nonnegative integer")
        vector = row["router_scores"]
        if not isinstance(vector, list) or not vector:
            raise InvalidInput("full router scores must be a nonempty explicit vector")
        scores, ids = [], set()
        for item in vector:
            if not isinstance(item, dict) or set(item) != {"expert_id", "score"}:
                raise InvalidInput("router score must contain exactly expert_id and score")
            expert, score = item["expert_id"], item["score"]
            if type(expert) is not int or expert < 0 or expert in ids:
                raise InvalidInput("router expert ID must be nonnegative and unique")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise InvalidInput("router probability must be numeric")
            try:
                valid = math.isfinite(score) and 0 <= score <= 1
            except (OverflowError, TypeError):
                valid = False
            if not valid:
                raise InvalidInput("router probability must be finite and in [0,1]")
            ids.add(expert); scores.append((expert, float(score)))
        if not math.isclose(sum(s for _, s in scores), 1.0, rel_tol=0, abs_tol=1e-6):
            raise InvalidInput("full router probabilities must sum to one")
        key = (row["request_id"], row["layer_id"], row["absolute_position"])
        if key in seen:
            raise InvalidInput("duplicate routing score position")
        seen.add(key)
        entries.append(RouterScoreRow(*key, tuple(sorted(scores))))
    return tuple(sorted(entries, key=lambda row: (row.request_id, row.layer_id, row.absolute_position)))


def validate_routing_score_rows(routes, expert_counts, routing_scores):
    """Validate private score evidence without making it policy-visible.

    ``routes`` may span layers and rounds. An empty sidecar leaves all full
    vectors unknown; a supplied sidecar must cover every private route exactly.
    The runtime only reveals validated rows through completed Router history.
    """
    if not isinstance(routing_scores, tuple) or any(not isinstance(row, RouterScoreRow) for row in routing_scores):
        raise InvalidInput("routing_scores must be a tuple of parsed RouterScoreRow evidence")
    for row in routing_scores:
        if any(not isinstance(value,str) or not value for value in (row.request_id,row.layer_id)) \
                or type(row.absolute_position) is not int or row.absolute_position<0:
            raise InvalidInput("typed routing rows require nonempty request/layer IDs and nonnegative integer positions")
        if not isinstance(row.scores,tuple) or any(not isinstance(item,tuple) or len(item)!=2 for item in row.scores):
            raise InvalidInput("typed router scores require explicit tuple expert/probability pairs")
    actual = {(r.request_id, r.layer_id, r.absolute_position): r for r in routes}
    supplied = {(r.request_id, r.layer_id, r.absolute_position): r for r in routing_scores}
    if len(actual) != len(routes):
        raise InvalidInput("private route positions must be unique across the execution")
    if len(supplied) != len(routing_scores) or (supplied and set(supplied) != set(actual)):
        raise InvalidInput("routing scores must cover exactly the current private route positions")
    for key, row in supplied.items():
        count = expert_counts.get(row.layer_id)
        if count is None or tuple(i for i, _ in row.scores) != tuple(range(count)):
            raise InvalidInput("routing score vector must cover every catalogue expert exactly")
        # Typed rows may be supplied directly, so retain the parser's numeric
        # contract rather than trusting construction outside that parser.
        def valid_probability(score):
            try:return not isinstance(score,bool) and isinstance(score,(int,float)) and math.isfinite(score) and 0<=score<=1
            except (OverflowError,TypeError):return False
        if any(type(expert) is not int or not valid_probability(score) for expert, score in row.scores) \
                or not math.isclose(sum(score for _, score in row.scores), 1.0, rel_tol=0, abs_tol=1e-6):
            raise InvalidInput("full router probabilities must be finite in [0,1] and sum to one")
        trace = actual[key]
        selected = [score for expert, score in row.scores if expert in trace.expert_ids]
        unselected = [score for expert, score in row.scores if expert not in trace.expert_ids]
        if unselected and min(selected) < max(unselected) - 1e-12:
            raise InvalidInput("full router probabilities contradict actual selected top-k experts")
        if trace.scores is not None and any(
                not math.isclose(score, dict(row.scores)[expert], rel_tol=0, abs_tol=1e-6)
                for expert, score in zip(trace.expert_ids, trace.scores)):
            raise InvalidInput("full router probabilities contradict declared selected scores")
    return supplied


def reveal_routing_scores(observation, score_rows):
    """Enrich already-revealed Router history; never append future positions."""
    rows = []
    for row in observation.route_history:
        evidence = score_rows.get((row.request_id, row.layer_id, row.absolute_position))
        if evidence is not None:
            row = replace(row, router_scores=evidence.scores, router_scores_scope="all_experts")
        rows.append(row)
    return replace(observation, route_history=tuple(rows))


class ScoredCausalSession(CausalM2Session):
    def __init__(self, compiled, *, routing_scores=(), **kwargs):
        if compiled.layer.ffn != "moe":
            if routing_scores:
                raise Unsupported("Dense invocations have no MoE router-score evidence")
        self._score_rows = validate_routing_score_rows(compiled.invocation.route_trace,
            {compiled.layer.id: compiled.layer.ffn_params["experts"]} if compiled.layer.ffn == "moe" else {},
            routing_scores)
        super().__init__(compiled, **kwargs)

    def observe(self):
        return reveal_routing_scores(super().observe(), self._score_rows)
