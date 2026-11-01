"""Small provenance/visibility contracts for resource inputs, not an OS model.

External CPU/GPU service and memory budgets may be declared functions of time.
The simulator may read a whole independent scenario while a policy sees only
announced/observed information. Own occupancy belongs to model state. Realized
total utilization is not automatically exogenous available capacity.
"""
from __future__ import annotations

from dataclasses import dataclass

from .parameters import InvalidModel,integer,nonempty


QUANTITIES=frozenset({"pool_capacity","available_service","memory_budget","offered_background_load",
    "total_utilization","frequency","temperature","owned_memory","owned_work","response_time"})
SOURCE_KINDS=frozenset({"declared_external","measured","model_internal"})
CONSUMERS=frozenset({"model_scenario","online_policy","same_policy_diagnostic"})
ROLES=frozenset({"exogenous_input","feedback","internal_accounting","descriptive_replay"})


@dataclass(frozen=True)
class ResourceInput:
    id: str
    quantity: str
    source_kind: str
    begin_ns: int
    end_ns: int
    available_at_ns: int
    clock_id: str
    clock_origin: str
    evidence: tuple[str,...]
    includes_own_work: bool
    policy_invariant: bool
    source_policy: str | None = None

    def __post_init__(self):
        for name in ("id","clock_id","clock_origin"):
            nonempty(getattr(self,name),name)
        if self.quantity not in QUANTITIES or self.source_kind not in SOURCE_KINDS:
            raise InvalidModel("unknown resource quantity or source kind")
        for name in ("begin_ns","end_ns","available_at_ns"):
            integer(getattr(self,name),name)
        if self.end_ns<self.begin_ns:
            raise InvalidModel("resource observation window is reversed")
        if self.source_kind=="measured" and self.available_at_ns<self.end_ns:
            raise InvalidModel("a measured future window cannot have been observed earlier")
        if type(self.evidence) is not tuple or not self.evidence:
            raise InvalidModel("resource input evidence must be explicit")
        for item in self.evidence:
            nonempty(item,"resource evidence")
        if type(self.includes_own_work) is not bool or type(self.policy_invariant) is not bool:
            raise InvalidModel("target effects and invariance must be explicit booleans")
        if self.source_policy is not None:
            nonempty(self.source_policy,"source policy")
        if self.includes_own_work:
            nonempty(self.source_policy,"target-dependent source policy")
            if self.policy_invariant:
                raise InvalidModel("target-dependent realized state cannot claim policy invariance")


def validate_resource_input(observation: ResourceInput,*,role: str,consumer: str,cutoff_ns: int,
                            clock_id: str,clock_origin: str,evaluation_policy: str):
    """Validate declared causality; never infer capacity from utilization.

    Announced future external scenarios are legitimate policy inputs if their
    availability time precedes the cutoff. Independent future traces may also
    be replayed by the MODEL without claiming online prediction. A measured
    trace containing policy A's own effects cannot be replayed as policy B's
    external environment. Past measured feedback is allowed, including state
    affected by earlier policies, but is not relabeled independent capacity.
    """
    if not isinstance(observation,ResourceInput) or role not in ROLES or consumer not in CONSUMERS:
        raise InvalidModel("explicit resource input, consumer and role required")
    integer(cutoff_ns,"decision cutoff")
    nonempty(evaluation_policy,"evaluation policy")
    if (observation.clock_id,observation.clock_origin)!=(clock_id,clock_origin):
        raise InvalidModel("resource input clock differs")
    visible=observation.available_at_ns<=cutoff_ns
    if consumer=="online_policy" and not visible:
        raise InvalidModel("online policy cannot read unannounced or unobserved future inputs")
    if role=="exogenous_input":
        if observation.source_kind=="model_internal" or observation.includes_own_work or not observation.policy_invariant:
            raise InvalidModel("exogenous input needs a target-independent declared scenario or justified invariant trace")
        if observation.quantity in {"total_utilization","owned_memory","owned_work","response_time"}:
            raise InvalidModel("total utilization/own response is not exogenous available capacity")
    elif role=="feedback":
        if not visible or observation.end_ns>cutoff_ns:
            raise InvalidModel("feedback must be completed/available by the decision cutoff")
        if observation.source_kind=="model_internal":
            raise InvalidModel("internal accounting is not a measured feedback source")
    elif role=="internal_accounting":
        if (observation.source_kind!="model_internal" or observation.quantity not in {"owned_memory","owned_work"} or
            observation.source_policy!=evaluation_policy or not observation.includes_own_work):
            raise InvalidModel("own resource accounting must be current-policy internal model state")
    else:
        if consumer!="same_policy_diagnostic":
            raise InvalidModel("oracle realized-state replay must be explicitly diagnostic")
        if observation.source_policy!=evaluation_policy and (observation.includes_own_work or not observation.policy_invariant):
            raise InvalidModel("policy A realized trace cannot supply policy B counterfactual state without policy invariance")
    return dict(input_id=observation.id,role=role,consumer=consumer,cutoff_ns=cutoff_ns,
        visible_at_cutoff=visible,future_window=observation.end_ns>cutoff_ns,
        source_policy=observation.source_policy,evaluation_policy=evaluation_policy,
        contains_target_effects=observation.includes_own_work,
        declared_policy_invariance=observation.policy_invariant,physical_independence_proven=False,
        total_utilization_converted_to_capacity=False,
        online_prediction_claimed=consumer=="online_policy",
        conditional_diagnostic=consumer=="same_policy_diagnostic")


def validate_budget_contract(*,semantics: str,other_application_kind: str,
                             external_reservations_charged_separately: bool,
                             oversubscribed: bool=False,sharing_rule: str | None=None):
    """Reject double subtraction and unspecified oversubscribed demand sharing.

    Reservations are direct external claims on the pool. Offered demand needs
    a declared arbitration rule if it exceeds the pool; this is resource-share
    policy, not an OS scheduler implementation. An already-net budget cannot
    be charged for other-application reservations a second time.
    """
    if semantics not in {"pool_capacity_minus_other_reservations","net_remaining_budget"}:
        raise InvalidModel("explicit pool-versus-net budget semantics required")
    if type(external_reservations_charged_separately) is not bool or type(oversubscribed) is not bool:
        raise InvalidModel("budget accounting flags must be booleans")
    if semantics=="net_remaining_budget":
        if other_application_kind!="already_net" or external_reservations_charged_separately:
            raise InvalidModel("already-net available budget cannot subtract other applications twice")
    else:
        if other_application_kind not in {"reservation","demand"} or not external_reservations_charged_separately:
            raise InvalidModel("pool accounting requires explicit external reservations/demand accounting")
        if other_application_kind=="demand" and oversubscribed and not sharing_rule:
            raise InvalidModel("oversubscribed offered demand needs an explicit sharing/arbitration rule")
    if sharing_rule is not None:
        nonempty(sharing_rule,"sharing rule")
    return dict(semantics=semantics,other_application_kind=other_application_kind,
        external_reservations_charged_separately=external_reservations_charged_separately,
        oversubscribed=oversubscribed,sharing_rule=sharing_rule,OS_scheduler_simulated=False)
