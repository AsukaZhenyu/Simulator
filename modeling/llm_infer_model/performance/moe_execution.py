"""Serial composition of disjoint, completed host windows of real MoE execution.

Profiling supplies nonnegative affine response coefficients and their observed
domain. These responses include the work, submissions and waits inside each
window; they are not identified CPU/GPU work. No device or copy time is added.
Only frozen structural inputs enter prediction. This module performs no fitting.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .evaluator import Evaluation, TaskExecution, evaluate
from .graph import Graph, ResourceTimeline, Task, WorkDemand
from .parameters import (INSTANTANEOUS_RATE, InvalidModel, Parameter,
                         ParameterBundle, UnknownParameter, decode_fraction,
                         encode_fraction, integer, nonempty, rational)
from .window_work import WindowWorkSpec, QUERY, ATTENTION, PAYLOAD


SCOPES = frozenset({"CPU_NODE", "RUNTIME_BRIDGE", "HOST", "SAMPLER"})
PHASES = frozenset({"prefill", "decode", "request"})
BOUNDARY = "disjoint inclusive completed host window"
LEGACY_POLICY = "legacy_affine_v1"
MECHANISM_POLICY = "mechanism_v1"
# v2 adds one declared cross-window dependency: a bridge that carries the
# output of a host-resident attention stage may be described by that stage's
# actual attention work. Everything else is unchanged from v1.
MECHANISM_POLICY_V2 = "mechanism_v2"
MECHANISM_POLICIES = (MECHANISM_POLICY, MECHANISM_POLICY_V2)


def _strings(values, name, *, allow_empty=False):
    if type(values) is not tuple or (not values and not allow_empty):
        raise InvalidModel(name + " requires an explicit tuple")
    for value in values:
        nonempty(value, name)
    if len(set(values)) != len(values):
        raise InvalidModel("duplicate " + name)


def _scope(value):
    if not isinstance(value, str) or value not in SCOPES:
        raise InvalidModel("unknown completed-window scope")


def _observed(values, name):
    if type(values) is not tuple or not values:
        raise InvalidModel(name + " requires observed integer coordinates")
    for value in values:
        integer(value, name)
    if len(set(values)) != len(values):
        raise InvalidModel("duplicate " + name)


@dataclass(frozen=True)
class RuntimeWindowAction:
    id: str
    key: str
    scope: str
    query_tokens: int
    kv_padded: int
    history: str
    phase: str
    layer: int | None = None
    op: str | None = None
    covered_nodes: tuple[str, ...] = ()
    forward_index: int = 0
    global_operation: bool = False
    payload_bytes: int = 0
    work_spec: WindowWorkSpec | None = None

    def __post_init__(self):
        nonempty(self.id, "window occurrence id")
        nonempty(self.key, "exact window price key")
        _scope(self.scope)
        integer(self.query_tokens, "query tokens")
        integer(self.kv_padded, "padded KV tokens")
        integer(self.forward_index, "forward occurrence index")
        integer(self.payload_bytes, "window payload bytes")
        nonempty(self.history, "runtime history coordinate")
        if not isinstance(self.phase, str) or self.phase not in PHASES:
            raise InvalidModel("explicit prefill/decode/request phase required")
        if self.layer is not None:
            integer(self.layer, "actual node layer")
        if self.op is not None:
            nonempty(self.op, "actual node operation")
        if type(self.global_operation) is not bool:
            raise InvalidModel("explicit global-operation flag required")
        if self.global_operation and self.layer is not None:
            raise InvalidModel("a global node cannot have a fabricated layer")
        _strings(self.covered_nodes, "covered node occurrences", allow_empty=True)
        if self.scope == "CPU_NODE":
            if ((self.layer is None and not self.global_operation)
                    or self.op is None or len(self.covered_nodes) != 1
                    or self.query_tokens < 1 or self.phase == "request"):
                raise InvalidModel("CPU_NODE requires one actual node, layer or explicit global flag, op and forward phase")
            if self.op.upper() in {"FORWARD", "CPU_FORWARD", "LAYER", "WHOLE_MODEL", "PREFILL", "DECODE"}:
                raise InvalidModel("a whole forward or layer cannot be priced as one CPU_NODE")
        if self.scope == "RUNTIME_BRIDGE" and not self.covered_nodes:
            raise InvalidModel("RUNTIME_BRIDGE requires its explicitly covered node occurrences")
        if self.work_spec is not None:
            if not isinstance(self.work_spec, WindowWorkSpec) or self.work_spec.scope != self.scope:
                raise InvalidModel("actual work declaration must match the window scope")
            self.work_spec.validate_coordinates(self.query_tokens, self.kv_padded, self.payload_bytes)

    def to_dict(self):
        return dict(id=self.id, key=self.key, scope=self.scope,
                    query_tokens=self.query_tokens, kv_padded=self.kv_padded,
                    history=self.history, phase=self.phase, layer=self.layer,
                    op=self.op, covered_nodes=list(self.covered_nodes),
                    forward_index=self.forward_index, global_operation=self.global_operation,
                    payload_bytes=self.payload_bytes,
                    work_spec=self.work_spec.to_dict() if self.work_spec is not None else None)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise InvalidModel("window action must be an explicit object")
        obj = dict(value)
        obj["covered_nodes"] = tuple(obj.get("covered_nodes", ()))
        if obj.get("work_spec") is not None:
            obj["work_spec"] = WindowWorkSpec.from_dict(obj["work_spec"])
        try:
            return cls(**obj)
        except TypeError as error:
            raise InvalidModel("invalid completed-window action schema") from error


@dataclass(frozen=True)
class RuntimeWindowEstimate:
    action: RuntimeWindowAction
    mean_ns: Fraction
    interpolated: bool
    extrapolated: bool
    out_of_domain: tuple[str, ...]
    feature_policy: str = LEGACY_POLICY

    def to_dict(self):
        return dict(action=self.action.to_dict(), mean_ns=encode_fraction(self.mean_ns),
                    interpolated=self.interpolated, extrapolated=self.extrapolated,
                    out_of_domain=list(self.out_of_domain),
                    timing_boundary=BOUNDARY, physical_work_identified=False,
                    feature_policy=self.feature_policy,
                    mechanism_constraints_applied=self.feature_policy in MECHANISM_POLICIES)


@dataclass(frozen=True)
class RuntimeWindowPrice:
    key: str
    scope: str
    intercept_ns: int | Fraction
    per_query_token_ns: int | Fraction
    per_attention_pair_ns: int | Fraction
    observed_query_tokens: tuple[int, ...]
    observed_kv_padded: tuple[int, ...]
    history_domain: tuple[str, ...]
    evidence: tuple[str, ...]
    zero_duration_approximation: bool = False
    per_payload_byte_ns: int | Fraction = 0
    observed_payload_bytes: tuple[int, ...] = (0,)
    work_spec: WindowWorkSpec | None = None
    feature_policy: str = LEGACY_POLICY

    def __post_init__(self):
        nonempty(self.key, "exact window price key")
        _scope(self.scope)
        for name in ("intercept_ns", "per_query_token_ns", "per_attention_pair_ns", "per_payload_byte_ns"):
            object.__setattr__(self, name, rational(getattr(self, name), name))
        _observed(self.observed_query_tokens, "observed query tokens")
        _observed(self.observed_kv_padded, "observed padded KV tokens")
        _observed(self.observed_payload_bytes, "observed payload bytes")
        _strings(self.history_domain, "observed runtime histories")
        _strings(self.evidence, "independent calibration evidence")
        if type(self.zero_duration_approximation) is not bool:
            raise InvalidModel("explicit zero-duration approximation flag required")
        if not isinstance(self.feature_policy, str) or self.feature_policy not in {LEGACY_POLICY, *MECHANISM_POLICIES}:
            raise InvalidModel("unknown completed-response feature policy")
        if self.feature_policy in MECHANISM_POLICIES:
            if not isinstance(self.work_spec, WindowWorkSpec) or self.work_spec.scope != self.scope:
                raise InvalidModel("mechanism pricing requires actual work evidence matching its scope")
            allowed = self.work_spec.allowed_features
            for feature, coefficient in ((QUERY, self.per_query_token_ns),
                                         (ATTENTION, self.per_attention_pair_ns),
                                         (PAYLOAD, self.per_payload_byte_ns)):
                if coefficient and feature not in allowed:
                    raise InvalidModel("coefficient has no declared work mechanism: " + feature)
        elif self.work_spec is not None:
            raise InvalidModel("legacy coefficients cannot claim mechanism-constrained work evidence")

    def estimate(self, action, *, allow_extrapolation=False):
        if not isinstance(action, RuntimeWindowAction):
            raise InvalidModel("explicit RuntimeWindowAction required")
        if type(allow_extrapolation) is not bool:
            raise InvalidModel("explicit extrapolation policy required")
        if (action.key, action.scope) != (self.key, self.scope):
            raise UnknownParameter("no exact matching window key and scope")
        axes = ("query_tokens", "kv_padded", "payload_bytes")
        if self.feature_policy in MECHANISM_POLICIES:
            if action.work_spec is None:
                raise InvalidModel("mechanism-constrained estimate requires actual window work")
            if action.work_spec.structural_signature() != self.work_spec.structural_signature():
                raise UnknownParameter("actual work mechanism differs from the calibrated window")
            axes = self.work_spec.domain_axes
        outside = []
        for name, value, observed in (("query_tokens", action.query_tokens, self.observed_query_tokens),
                                      ("kv_padded", action.kv_padded, self.observed_kv_padded),
                                      ("payload_bytes", action.payload_bytes, self.observed_payload_bytes)):
            if name in axes and (value < min(observed) or value > max(observed)):
                outside.append(name)
        if action.history not in self.history_domain:
            outside.append("history")
        if outside and not allow_extrapolation:
            raise UnknownParameter("window outside observed affine domain: " + ", ".join(outside))
        mean = (self.intercept_ns + self.per_query_token_ns * action.query_tokens
                + self.per_attention_pair_ns * action.query_tokens * action.kv_padded
                + self.per_payload_byte_ns * action.payload_bytes)
        if mean == 0 and not self.zero_duration_approximation:
            raise InvalidModel("zero response needs an explicit resolution-limited approximation")
        if mean != 0 and self.zero_duration_approximation:
            raise InvalidModel("zero-duration approximation cannot label a positive response")
        interpolated = (not outside and any(name in axes and value not in observed for name, value, observed in
            (("query_tokens", action.query_tokens, self.observed_query_tokens),
             ("kv_padded", action.kv_padded, self.observed_kv_padded),
             ("payload_bytes", action.payload_bytes, self.observed_payload_bytes))))
        return RuntimeWindowEstimate(action, mean, interpolated, bool(outside), tuple(outside), self.feature_policy)

    def to_dict(self):
        return dict(key=self.key, scope=self.scope,
                    **{name: encode_fraction(getattr(self, name)) for name in
                       ("intercept_ns", "per_query_token_ns", "per_attention_pair_ns", "per_payload_byte_ns")},
                    observed_query_tokens=list(self.observed_query_tokens),
                    observed_kv_padded=list(self.observed_kv_padded),
                    observed_payload_bytes=list(self.observed_payload_bytes),
                    history_domain=list(self.history_domain), evidence=list(self.evidence),
                    zero_duration_approximation=self.zero_duration_approximation,
                    work_spec=self.work_spec.to_dict() if self.work_spec is not None else None,
                    feature_policy=self.feature_policy)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise InvalidModel("window price must be an explicit object")
        obj = dict(value)
        for name in ("intercept_ns", "per_query_token_ns", "per_attention_pair_ns"):
            obj[name] = decode_fraction(obj[name], name)
        if "per_payload_byte_ns" in obj:
            obj["per_payload_byte_ns"] = decode_fraction(obj["per_payload_byte_ns"], "per_payload_byte_ns")
        for name in ("observed_query_tokens", "observed_kv_padded", "history_domain", "evidence"):
            obj[name] = tuple(obj[name])
        if "observed_payload_bytes" in obj:
            obj["observed_payload_bytes"] = tuple(obj["observed_payload_bytes"])
        if obj.get("work_spec") is not None:
            obj["work_spec"] = WindowWorkSpec.from_dict(obj["work_spec"])
        try:
            return cls(**obj)
        except TypeError as error:
            raise InvalidModel("invalid completed-window price schema") from error


@dataclass(frozen=True)
class CompiledRuntimeWindows:
    tasks: tuple[Task, ...]
    parameters: ParameterBundle
    estimates: tuple[RuntimeWindowEstimate, ...]

    def __post_init__(self):
        # This exact serial shape permits linear static evaluation. Validate it
        # here without the general Graph's repeated dependency-closure scans.
        if (type(self.tasks) is not tuple or not self.tasks
                or any(not isinstance(t, Task) for t in self.tasks)):
            raise InvalidModel("nonempty serial runtime-window task tuple required")
        if not isinstance(self.parameters, ParameterBundle):
            raise InvalidModel("runtime-window ParameterBundle required")
        if (type(self.estimates) is not tuple or len(self.estimates) != len(self.tasks)
                or any(not isinstance(e, RuntimeWindowEstimate) for e in self.estimates)
                or len(self.parameters.parameters) != len(self.tasks)):
            raise InvalidModel("one estimate and response parameter per runtime window required")
        if len({task.id for task in self.tasks}) != len(self.tasks):
            raise InvalidModel("duplicate runtime-window task id")
        previous = ()
        for task, parameter, estimate in zip(self.tasks, self.parameters.parameters, self.estimates):
            mean = rational(estimate.mean_ns, "complete window response")
            rate = INSTANTANEOUS_RATE if mean == 0 else Fraction(1_000_000_000) / mean
            if (task.id != estimate.action.id or task.depends_on != previous
                    or task.resource != "moe_completed_host_response"
                    or task.fixed_parameter is not None or task.extra_resources
                    or task.memory or task.waits_for
                    or task.demands != (WorkDemand(parameter.name, 1, "invocation"),)
                    or parameter.unit != "invocation/s" or parameter.value != rate):
                raise InvalidModel("runtime-window task is not the declared inclusive serial response")
            previous = (task.id,)

    @property
    def graph(self):
        """Build the general Graph only for callers that explicitly need it."""
        return Graph(self.tasks)

    def evaluate(self, timeline: ResourceTimeline | None = None):
        """Integrate explicit inclusive-response service rates on the common clock.

        A supplied timeline is a declared response-rate scenario; no CPU/GPU
        utilization, physical service or environmental recovery is inferred.
        With no timeline, each dependent invocation completes at ceil(mean_ns),
        exactly as the general evaluator; explicit zero responses finish at the
        same clock tick. This path never scans unready downstream tasks.
        """
        if timeline is not None:
            return evaluate(self.graph, self.parameters, timeline)
        time, records, events = 0, [], []
        for task, estimate in zip(self.tasks, self.estimates):
            start = time
            mean = estimate.mean_ns
            time += (mean.numerator + mean.denominator - 1) // mean.denominator
            records.append(TaskExecution(task.id, task.resource, start, start, time))
            events.extend(((start, "start", task.id), (time, "finish", task.id)))
        return Evaluation(time, tuple(records), (), tuple(events))

    def to_dict(self):
        return dict(schema_version="moe-completed-host-windows-1",
                    graph=dict(tasks=[task.to_dict() for task in self.tasks], memory_domains=[]),
                    parameters=self.parameters.to_dict(),
                    windows=[item.to_dict() for item in self.estimates],
                    sum_window_means_ns=encode_fraction(sum((e.mean_ns for e in self.estimates), Fraction())),
                    any_extrapolation=any(e.extrapolated for e in self.estimates),
                    feature_policies=sorted({e.feature_policy for e in self.estimates}),
                    legacy_response_windows=sum(e.feature_policy == LEGACY_POLICY for e in self.estimates),
                    boundary=BOUNDARY, device_or_copy_diagnostics_added=False,
                    physical_work_identified=False, target_timings_consumed=False,
                    hardware_accuracy_qualified=False)


def compile_runtime_windows(actions, prices, *, allow_extrapolation=False):
    """Compose an explicit partition; every covered node is charged at most once.

    Covered node names identify occurrences within (phase, forward_index).
    Pricing coefficients remain supplied data. Only per-window completions are
    rounded by the evaluator; coefficients and service progress stay rational.
    """
    if (type(actions) is not tuple or not actions
            or any(not isinstance(a, RuntimeWindowAction) for a in actions)):
        raise InvalidModel("ordered nonempty RuntimeWindowAction tuple required")
    if (type(prices) is not tuple or not prices
            or any(not isinstance(p, RuntimeWindowPrice) for p in prices)):
        raise InvalidModel("nonempty RuntimeWindowPrice tuple required")
    if len({a.id for a in actions}) != len(actions):
        raise InvalidModel("duplicate window occurrence id")
    if len({p.key for p in prices}) != len(prices):
        raise InvalidModel("duplicate exact window price key")
    table = {p.key: p for p in prices}
    tasks, parameters, estimates, covered = [], [], [], set()
    previous = ()
    for index, action in enumerate(actions):
        nodes = {(action.phase, action.forward_index, node) for node in action.covered_nodes}
        if covered.intersection(nodes):
            raise InvalidModel("overlapping window coverage would charge an actual node twice")
        covered.update(nodes)
        price = table.get(action.key)
        if price is None:
            raise UnknownParameter("missing exact completed-window price: " + action.key)
        estimate = price.estimate(action, allow_extrapolation=allow_extrapolation)
        estimates.append(estimate)
        name = "moe_completed_window_response:" + str(index)
        rate = (INSTANTANEOUS_RATE if estimate.mean_ns == 0 else
                Fraction(1_000_000_000) / estimate.mean_ns)
        parameters.append(Parameter(name, rate, "invocation/s", "profiled", price.evidence,
                                    BOUNDARY + "; " + action.scope + "; " + action.key))
        tasks.append(Task(action.id, "moe_completed_host_response",
                          (WorkDemand(name, 1, "invocation"),), previous))
        previous = (action.id,)
    return CompiledRuntimeWindows(tuple(tasks), ParameterBundle(tuple(parameters)), tuple(estimates))
