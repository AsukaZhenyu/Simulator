"""Causal, fixed-parameter phase observation from completed host responses.

The observer does not fit a period/width, read a profiler, simulate OS scheduling
or identify neutral available CPU capacity. It maintains an explicitly uncertain
periodic-pause hypothesis. A completed PAST target call may update its phase;
the current/future target's outcome cannot enter an earlier prediction. CPU and
GPU timing still belongs to the common Graph evaluator, not this observer.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from types import MappingProxyType
from typing import Mapping

from .parameters import InvalidModel, encode_fraction, integer, nonempty, rational


def _signed_integer(value, label):
    if type(value) is not int:
        raise InvalidModel(label + " must be an explicit integer timestamp")
    return value


def _evidence(value):
    if type(value) is not tuple or not value:
        raise InvalidModel("explicit phase-hypothesis evidence required")
    for text in value:
        nonempty(text, "phase evidence")


def _encoded(value):
    if isinstance(value, Fraction):
        return encode_fraction(value)
    if isinstance(value, dict):
        return {k:_encoded(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return [_encoded(v) for v in value]
    return value


@dataclass(frozen=True)
class HostPauseParameters:
    period_ns: int | None
    blackout_ns: int | None
    min_excess_ns: int | None
    base_action_ns: Mapping[str, int | Fraction | None]
    jitter_ns: int | None
    evidence: tuple[str, ...]
    scope: str

    def __post_init__(self):
        for key in ("period_ns", "blackout_ns", "min_excess_ns", "jitter_ns"):
            if getattr(self,key) is not None:
                integer(getattr(self,key),key,1 if key in {"period_ns","min_excess_ns"} else 0)
        if self.period_ns is not None and self.blackout_ns is not None and self.blackout_ns > self.period_ns:
            raise InvalidModel("blackout width exceeds frozen period")
        if not isinstance(self.base_action_ns,Mapping) or not self.base_action_ns:
            raise InvalidModel("explicit nonempty action baseline mapping required")
        copied={}
        for name,value in self.base_action_ns.items():
            nonempty(name,"action baseline id")
            copied[name]=None if value is None else rational(value,"action baseline duration")
        object.__setattr__(self,"base_action_ns",MappingProxyType(copied))
        _evidence(self.evidence)
        nonempty(self.scope,"phase-observer scope")

    def to_dict(self):
        return _encoded(dict(period_ns=self.period_ns,blackout_ns=self.blackout_ns,
            min_excess_ns=self.min_excess_ns,base_action_ns=dict(self.base_action_ns),
            jitter_ns=self.jitter_ns,evidence=self.evidence,scope=self.scope))


@dataclass(frozen=True)
class PauseEndAnchor:
    lower_ns: int | Fraction
    upper_ns: int | Fraction
    cutoff_ns: int
    clock_id: str
    clock_origin: str
    evidence: tuple[str, ...]
    source_kind: str = "before_reference_pause_end"

    def __post_init__(self):
        for key in ("lower_ns","upper_ns"):
            value=getattr(self,key)
            if type(value) not in {int,Fraction}:
                raise InvalidModel("anchor bounds require explicit exact timestamps")
            object.__setattr__(self,key,Fraction(value))
        if self.lower_ns > self.upper_ns:
            raise InvalidModel("reversed pause-end uncertainty bounds")
        _signed_integer(self.cutoff_ns,"anchor observation cutoff")
        nonempty(self.clock_id,"anchor clock")
        nonempty(self.clock_origin,"anchor clock origin")
        _evidence(self.evidence)
        if self.source_kind not in {"before_reference_pause_end","completed_target_response_pause_end"}:
            raise InvalidModel("unknown phase anchor provenance")


@dataclass(frozen=True)
class CompletedHostAction:
    action_id: str
    begin_ns: int
    end_ns: int

    def __post_init__(self):
        nonempty(self.action_id,"completed action id")
        _signed_integer(self.begin_ns,"action begin")
        _signed_integer(self.end_ns,"action end")
        if self.begin_ns > self.end_ns:
            raise InvalidModel("action completion precedes its beginning")


@dataclass(frozen=True)
class HostPausePrediction:
    at_ns: int
    call_id: str | None
    mode: str
    period_ns: int | None
    blackout_ns: int | None
    phase_ns: int | None
    relative_phase_ns: int | None
    exact_phase_ns: Fraction | None
    phase_quantization_ns: Fraction | None
    uncertainty_ns: Fraction | None
    phase_intervals_ns: tuple[tuple[Fraction,Fraction], ...]
    anchor_bounds_ns: tuple[Fraction,Fraction] | None
    cutoff_ns: int | None
    history_cutoff_ns: int | None
    clock_id: str
    clock_origin: str
    source_kind: str
    reason: str | None

    @property
    def known(self):
        return self.phase_ns is not None

    def to_dict(self):
        from dataclasses import asdict
        return _encoded(dict(asdict(self),known=self.known,
            hardware_accuracy_qualified=False,independent_instantaneous_availability=False,
            frozen_period_or_width_updated=False,
            qualification="phase hypothesis from before references or completed past responses only"))


@dataclass(frozen=True)
class HostPauseObservation:
    call_id: str
    observed_at_ns: int
    qualifying_action_ids: tuple[str, ...]
    unknown_baseline_action_ids: tuple[str, ...]
    anchor_updated: bool
    phase_ambiguous: bool
    reason: str


class HostPauseObserver:
    """One sequential caller's causal phase state; no parameter estimation.

    ``predict(...,call_id=...)`` registers one in-flight call. Its completed
    action records may be supplied only afterwards with an explicit as-of time.
    ``predict(at_ns)`` without an ID is a read-only snapshot and cannot authorize
    a later observation. Multiple qualifying actions make that observation's
    phase ambiguous rather than selecting the largest/best-fitting pause.
    """

    def __init__(self, parameters: HostPauseParameters, *, initial_anchor=None,
                 clock_id: str, clock_origin: str, mode="causal_history"):
        if not isinstance(parameters,HostPauseParameters):
            raise InvalidModel("frozen HostPauseParameters required")
        nonempty(clock_id,"observer clock")
        nonempty(clock_origin,"observer clock origin")
        if mode not in {"causal_history","before_only"}:
            raise InvalidModel("phase observer mode must be causal_history or before_only")
        if initial_anchor is not None:
            if not isinstance(initial_anchor,PauseEndAnchor) or initial_anchor.source_kind != "before_reference_pause_end":
                raise InvalidModel("initial phase anchor must come from a BEFORE reference, not a target response")
            if (initial_anchor.clock_id,initial_anchor.clock_origin)!=(clock_id,clock_origin):
                raise InvalidModel("initial anchor clock/origin differs")
        self.parameters=parameters
        self.clock_id,self.clock_origin,self.mode=clock_id,clock_origin,mode
        self._anchor=initial_anchor
        self._pending=None
        self._seen=set()
        self._history_cutoff=None
        self._last_prediction_at=None
        self._last_call_end=None
        self._ambiguous=False

    def _clock(self,clock_id,clock_origin):
        if clock_id is not None and clock_id!=self.clock_id or clock_origin is not None and clock_origin!=self.clock_origin:
            raise InvalidModel("observer clock/origin differs; timestamps are never realigned implicitly")

    def predict(self, at_ns, *, call_id=None, clock_id=None, clock_origin=None):
        _signed_integer(at_ns,"prediction timestamp")
        self._clock(clock_id,clock_origin)
        if self._history_cutoff is not None and self._history_cutoff>=at_ns:
            raise InvalidModel("prediction requires observations available strictly BEFORE its timestamp")
        if self._last_prediction_at is not None and at_ns<self._last_prediction_at:
            raise InvalidModel("nonmonotonic prediction timestamps")
        anchor=self._anchor
        if anchor is not None and anchor.cutoff_ns>=at_ns:
            raise InvalidModel("BEFORE/response anchor is not available before prediction")
        if call_id is not None:
            nonempty(call_id,"prediction call id")
            if self._pending is not None:
                raise InvalidModel("a call is still in flight; complete it before predicting another")
            if call_id in self._seen:
                raise InvalidModel("duplicate prediction/observation call id")
            self._pending=(call_id,at_ns)
            self._seen.add(call_id)
        self._last_prediction_at=at_ns
        period,width=self.parameters.period_ns,self.parameters.blackout_ns
        phase=relative=exact=rounding=uncertainty=None
        intervals=()
        reason=None
        if period is None or width is None:
            reason="frozen_period_or_width_unknown"
        elif self._ambiguous:
            reason="multiple_pause_candidates_phase_ambiguous"
        elif anchor is None:
            reason="no_completed_before_reference_or_past_response_anchor"
        elif anchor.upper_ns-anchor.lower_ns>=period:
            reason="anchor_uncertainty_spans_a_period"
        else:
            center=(anchor.lower_ns+anchor.upper_ns)/2
            exact=(center-width)%period
            phase=math.floor(exact)
            relative=(phase-at_ns)%period
            rounding=exact-phase
            uncertainty=(anchor.upper_ns-anchor.lower_ns)/2+rounding
            start=(anchor.lower_ns-width)%period
            end=start+(anchor.upper_ns-anchor.lower_ns)
            intervals=((start,end),) if end<period else ((start,Fraction(period)),(Fraction(0),end-period))
        return HostPausePrediction(at_ns,call_id,self.mode,period,width,phase,relative,exact,rounding,uncertainty,
            intervals,None if anchor is None else (anchor.lower_ns,anchor.upper_ns),
            None if anchor is None else anchor.cutoff_ns,self._history_cutoff,self.clock_id,self.clock_origin,
            "observed_response_phase_hypothesis" if anchor is not None and anchor.source_kind=="completed_target_response_pause_end"
            else "before_reference_phase_hypothesis",reason)

    def observe_completed(self, call_id, begin_ns, end_ns, actions, *, observed_at_ns,
                          clock_id, clock_origin):
        nonempty(clock_id,"observation clock")
        nonempty(clock_origin,"observation clock origin")
        self._clock(clock_id,clock_origin)
        for value,label in ((begin_ns,"call begin"),(end_ns,"call end"),(observed_at_ns,"observation as-of")):
            _signed_integer(value,label)
        if self._pending!=(call_id,begin_ns):
            raise InvalidModel("only the registered current call can be completed, once")
        if not begin_ns<=end_ns<=observed_at_ns:
            raise InvalidModel("in-flight/future call timing cannot be observed as completed")
        if self._history_cutoff is not None and observed_at_ns<self._history_cutoff:
            raise InvalidModel("completed observations must be monotonic")
        if self._last_call_end is not None and begin_ns<self._last_call_end:
            raise InvalidModel("completed calls overlap in one sequential observer")
        if type(actions) is not tuple or any(not isinstance(a,CompletedHostAction) for a in actions):
            raise InvalidModel("completed action tuple required")
        cursor=begin_ns
        for action in actions:
            if not cursor<=action.begin_ns<=action.end_ns<=end_ns:
                raise InvalidModel("actions overlap, are reordered or lie outside completed call")
            cursor=action.end_ns
        if len({a.action_id for a in actions})!=len(actions):
            raise InvalidModel("duplicate action observations within a call")
        candidates=[]
        unknown=[]
        p=self.parameters
        for action in actions:
            base=p.base_action_ns.get(action.action_id)
            if base is None:
                unknown.append(action.action_id)
            elif p.min_excess_ns is not None and action.end_ns-action.begin_ns-base>=p.min_excess_ns:
                candidates.append((action,base))
        updated=False
        reason="before_only_history_ignored" if self.mode=="before_only" else "no_single_action_excess"
        if self.mode=="causal_history":
            if unknown:
                self._ambiguous=True
                reason="action_baseline_unknown_no_pause_anchor_inferred"
            elif len(candidates)>1:
                self._ambiguous=True
                reason="multiple_action_excesses_no_unique_pause_anchor"
            elif len(candidates)==1 and p.jitter_ns is not None:
                action,base=candidates[0]
                # Exact center=end-base/2, radius=base/2+jitter. No fitted
                # whole-call residual or measured current-target phase input.
                self._anchor=PauseEndAnchor(action.end_ns-base-p.jitter_ns,action.end_ns+p.jitter_ns,
                    observed_at_ns,self.clock_id,self.clock_origin,
                    p.evidence+("completed past call:"+call_id,"action:"+action.action_id),
                    "completed_target_response_pause_end")
                self._ambiguous=False
                updated=True
                reason="single_completed_action_excess_phase_update"
            elif len(candidates)==1:
                self._ambiguous=True
                reason="frozen_jitter_unknown_no_pause_anchor_inferred"
        self._history_cutoff=observed_at_ns
        self._last_call_end=end_ns
        self._pending=None
        return HostPauseObservation(call_id,observed_at_ns,tuple(a.action_id for a,_ in candidates),tuple(unknown),
                                    updated,self._ambiguous,reason)
