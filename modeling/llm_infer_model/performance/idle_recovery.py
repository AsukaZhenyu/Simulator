"""Optional causal response hypothesis for a declared idle/copy-burst protocol.

This is a bounded, interpretable *completed-response* model, not an assertion
that the scalar state is a real OS queue, cache, clock or PCIe power state. No
parameters are provided by default. Selecting/estimating them belongs to
profiling and requires independent validation before an accuracy claim.

The readiness-debt proxy fills while the scoped resource is idle and drains
during predicted completed target invocations. The caller must supply an
initial proxy state. A zero state after an untimed prime is one testable
initialization hypothesis, not a default or a measured physical reset.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .parameters import InvalidModel,UnknownParameter,integer


def _number(value,name,*,positive=False):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise InvalidModel(name+' must be an explicit finite number')
    if value<0 or positive and value<=0:
        raise InvalidModel(name+' is outside its nonnegative/positive domain')
    return float(value)


@dataclass(frozen=True)
class IdleRecoveryScope:
    sha256: str
    route: str
    payload_bytes: int
    memory_mode: str
    maximum_requested_idle_ns: int

    def __post_init__(self):
        if not isinstance(self.sha256,str) or len(self.sha256)!=64 or any(c not in '0123456789abcdef' for c in self.sha256):
            raise InvalidModel('explicit runtime/preparation scope SHA256 required')
        if self.route not in ('dram->vram','vram->dram'):
            raise InvalidModel('explicit host-device route required')
        integer(self.payload_bytes,'payload bytes',1)
        integer(self.maximum_requested_idle_ns,'maximum requested idle ns',1)
        if self.memory_mode not in ('pageable','registered'):
            raise InvalidModel('explicit host-memory mode required')


@dataclass(frozen=True)
class IdleRecoveryParameters:
    steady_response_ns: float
    maximum_extra_response_ns: float
    idle_fill_tau_ns: float
    active_recovery_tau_ns: float
    evidence: tuple[str,...]

    def __post_init__(self):
        for name in ('steady_response_ns','maximum_extra_response_ns','idle_fill_tau_ns','active_recovery_tau_ns'):
            object.__setattr__(self,name,_number(getattr(self,name),name,
                positive=name in ('steady_response_ns','idle_fill_tau_ns','active_recovery_tau_ns')))
        if type(self.evidence) is not tuple or not self.evidence or any(not isinstance(p,str) or not p.strip() for p in self.evidence):
            raise InvalidModel('independent parameter evidence required; no guessed defaults')


@dataclass(frozen=True)
class ReadinessDebt:
    value: float

    def __post_init__(self):
        value=_number(self.value,'readiness-debt proxy')
        if value>1:raise InvalidModel('readiness-debt proxy must be within zero and one')
        object.__setattr__(self,'value',value)


@dataclass(frozen=True)
class IdleRecoveryResponse:
    scope: IdleRecoveryScope
    parameters: IdleRecoveryParameters

    def _coordinate(self,*,scope_sha256,route,payload_bytes,memory_mode):
        if (scope_sha256,route,payload_bytes,memory_mode)!=(self.scope.sha256,self.scope.route,
                self.scope.payload_bytes,self.scope.memory_mode):
            raise UnknownParameter('unmeasured idle/recovery coordinate; no geometry or memory fallback')

    def after_idle(self,state:ReadinessDebt,requested_idle_ns:int)->ReadinessDebt:
        if not isinstance(state,ReadinessDebt):raise InvalidModel('explicit readiness state required')
        integer(requested_idle_ns,'requested idle ns')
        if requested_idle_ns>self.scope.maximum_requested_idle_ns:
            raise UnknownParameter('requested idle exceeds the declared scope')
        filled=-math.expm1(-requested_idle_ns/self.parameters.idle_fill_tau_ns)
        return ReadinessDebt(min(1.,max(0.,state.value+(1-state.value)*filled)))

    def complete(self,state:ReadinessDebt,*,scope_sha256,route,payload_bytes,memory_mode):
        self._coordinate(scope_sha256=scope_sha256,route=route,payload_bytes=payload_bytes,memory_mode=memory_mode)
        if not isinstance(state,ReadinessDebt):raise InvalidModel('explicit readiness state required')
        p=self.parameters
        predicted_ns=math.ceil(p.steady_response_ns+p.maximum_extra_response_ns*state.value)
        remaining=state.value*math.exp(-predicted_ns/p.active_recovery_tau_ns)
        return predicted_ns,ReadinessDebt(remaining)

    def continuous_burst(self,requested_idle_ns:int,copies:int,*,initial_state:ReadinessDebt):
        """Unrounded form of the same law for named-parameter estimation.

        This avoids a one-nanosecond quantizer flattening numerical derivatives.
        It still advances exclusively by its own predicted completed responses.
        Final forecast artifacts use predict_burst's integer-nanosecond clock.
        """
        integer(copies,'burst copies',1)
        state=self.after_idle(initial_state,requested_idle_ns);values=[];p=self.parameters
        for _ in range(copies):
            value=p.steady_response_ns+p.maximum_extra_response_ns*state.value
            values.append(value);state=ReadinessDebt(state.value*math.exp(-value/p.active_recovery_tau_ns))
        return values

    def predict_burst(self,requested_idle_ns:int,copies:int,*,initial_state:ReadinessDebt):
        integer(copies,'burst copies',1)
        if copies>100000:raise InvalidModel('burst exceeds declared safety envelope')
        state=self.after_idle(initial_state,requested_idle_ns);rows=[];elapsed=0
        for ordinal in range(copies):
            before=state
            duration,state=self.complete(state,scope_sha256=self.scope.sha256,route=self.scope.route,
                payload_bytes=self.scope.payload_bytes,memory_mode=self.scope.memory_mode)
            rows.append(dict(ordinal=ordinal,predicted_ns=duration,begin_predicted_ns=elapsed,
                             state_before=before.value,state_after=state.value))
            elapsed+=duration
        return dict(rows=rows,duration_ns=elapsed,mean_response_ns=elapsed/copies,
            state_final=state.value,requested_idle_ns=requested_idle_ns,
            initialization='explicit supplied proxy state; a full prime is not a measured physical reset',
            target_outcome_used=False,future_observed_state_used=False,
            physical_mechanism_identified=False,hardware_accuracy_qualified=False)

    @classmethod
    def from_profile(cls,profile):
        if profile.get('schema')!='idle-recovery-response-1':raise InvalidModel('idle-response profile schema differs')
        if profile.get('response_unit')!='inclusive completed invocation ns':raise InvalidModel('completed response cannot be relabelled physical throughput')
        if profile.get('future_target_state_used') is not False or profile.get('whole_Dense_target_used_for_fit') is not False:
            raise InvalidModel('profile does not establish target-blind causal parameter origin')
        try:
            scope=IdleRecoveryScope(**profile['scope'])
            p=dict(profile['parameters']);p['evidence']=tuple(p['evidence'])
            parameters=IdleRecoveryParameters(**p)
        except (KeyError,TypeError) as error:
            raise InvalidModel('complete explicit idle-response parameters required') from error
        return cls(scope,parameters)


@dataclass(frozen=True)
class ConstantIdleResponse:
    """Same-calibration static baseline; one fitted completed-response value."""
    scope: IdleRecoveryScope
    response_ns: float
    evidence: tuple[str,...]

    def __post_init__(self):
        object.__setattr__(self,'response_ns',_number(self.response_ns,'constant completed response',positive=True))
        if not self.evidence:raise InvalidModel('constant response evidence required')

    def predict_burst(self,requested_idle_ns:int,copies:int):
        integer(requested_idle_ns,'requested idle ns');integer(copies,'copies',1)
        if requested_idle_ns>self.scope.maximum_requested_idle_ns:raise UnknownParameter('idle outside constant baseline scope')
        value=math.ceil(self.response_ns)
        return dict(rows=[dict(ordinal=i,predicted_ns=value,begin_predicted_ns=i*value) for i in range(copies)],
                    duration_ns=value*copies,target_outcome_used=False)

    def continuous_burst(self,requested_idle_ns:int,copies:int):
        self.predict_burst(requested_idle_ns,1);integer(copies,'copies',1)
        return [self.response_ns]*copies


@dataclass(frozen=True)
class CurrentIdleResponse:
    """Only the immediately preceding requested idle; no prime debt/carry-over.

The first target follows the requested idle; later targets have zero declared
idle. This three-parameter baseline cannot use burst ordinal as a fitted key.
"""
    scope: IdleRecoveryScope
    steady_response_ns: float
    maximum_extra_response_ns: float
    idle_fill_tau_ns: float
    evidence: tuple[str,...]

    def __post_init__(self):
        for field in ('steady_response_ns','maximum_extra_response_ns','idle_fill_tau_ns'):
            object.__setattr__(self,field,_number(getattr(self,field),field,positive=field!='maximum_extra_response_ns'))
        if not self.evidence:raise InvalidModel('current-idle response evidence required')

    def predict_burst(self,requested_idle_ns:int,copies:int):
        integer(requested_idle_ns,'requested idle ns');integer(copies,'copies',1)
        if requested_idle_ns>self.scope.maximum_requested_idle_ns:raise UnknownParameter('idle outside current-idle scope')
        first=math.ceil(self.steady_response_ns+self.maximum_extra_response_ns*(-math.expm1(-requested_idle_ns/self.idle_fill_tau_ns)))
        later=math.ceil(self.steady_response_ns);rows=[];at=0
        for ordinal in range(copies):
            value=first if ordinal==0 else later;rows.append(dict(ordinal=ordinal,predicted_ns=value,begin_predicted_ns=at));at+=value
        return dict(rows=rows,duration_ns=at,target_outcome_used=False)

    def continuous_burst(self,requested_idle_ns:int,copies:int):
        self.predict_burst(requested_idle_ns,1);integer(copies,'copies',1)
        first=self.steady_response_ns+self.maximum_extra_response_ns*(-math.expm1(-requested_idle_ns/self.idle_fill_tau_ns))
        return [first]+[self.steady_response_ns]*(copies-1)
