"""Completed COPY cost with an explicit first-response impulse.

The sustained response and the first-response difference are measured completed
host responses, not identified DMA, clock or scheduler states. Byte curves are
continuous piecewise-linear parameters. Idle/preparation coordinates are exact:
the model does not invent a monotone physical recovery law between unmeasured
idle values. Estimation and independent qualification belong to profiling.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

from .parameters import InvalidModel,UnknownParameter,integer

def _sha(value,label):
    if not isinstance(value,str) or len(value)!=64 or any(x not in '0123456789abcdef' for x in value):
        raise InvalidModel(label+' requires explicit SHA256 provenance')

def _evidence(value):
    if type(value) is not tuple or not value or any(not isinstance(v,str) or not v.strip() for v in value):
        raise InvalidModel('nonempty explicit profiling evidence required')

@dataclass(frozen=True)
class PiecewiseCompletedResponse:
    byte_knots:tuple[int,...]
    response_ns:tuple[float,...]
    evidence:tuple[str,...]
    signed:bool=False

    def __post_init__(self):
        if type(self.byte_knots) is not tuple or type(self.response_ns) is not tuple or len(self.byte_knots)<2 or len(self.byte_knots)!=len(self.response_ns):
            raise InvalidModel('matching explicit byte and response knots required')
        for n in self.byte_knots:integer(n,'byte knot',1)
        if any(a>=b for a,b in zip(self.byte_knots,self.byte_knots[1:])):
            raise InvalidModel('byte knots must be strictly increasing')
        if type(self.signed) is not bool:raise InvalidModel('explicit signed-impulse flag required')
        for value in self.response_ns:
            if type(value) not in (int,float) or not math.isfinite(value) or (not self.signed and value<=0):
                raise InvalidModel('finite positive complete responses or explicit signed impulses required')
        _evidence(self.evidence)

    def at(self,size_bytes):
        integer(size_bytes,'COPY bytes',1)
        if not self.byte_knots[0]<=size_bytes<=self.byte_knots[-1]:
            raise UnknownParameter('COPY size outside the measured byte envelope')
        for i,(lo,hi) in enumerate(zip(self.byte_knots,self.byte_knots[1:])):
            if lo<=size_bytes<=hi:
                ratio=(size_bytes-lo)/(hi-lo)
                return self.response_ns[i]+ratio*(self.response_ns[i+1]-self.response_ns[i])
        raise UnknownParameter('unpriced byte coordinate')

@dataclass(frozen=True)
class FirstResponseImpulse:
    requested_idle_ns:int
    response:PiecewiseCompletedResponse

    def __post_init__(self):
        integer(self.requested_idle_ns,'declared idle ns')
        if not isinstance(self.response,PiecewiseCompletedResponse) or not self.response.signed:
            raise InvalidModel('explicit signed first-response difference required')

@dataclass(frozen=True)
class CompletedTransferLifecycle:
    context_sha256:str
    preparation_sha256:str
    route:str
    memory_mode:str
    sustained:PiecewiseCompletedResponse
    first_impulses:tuple[FirstResponseImpulse,...]
    maximum_copies:int
    first_qualified:bool
    window_qualified:bool
    qualified_byte_coordinates:tuple[int,...]
    qualified_window_lengths:tuple[int,...]

    def __post_init__(self):
        _sha(self.context_sha256,'runtime context');_sha(self.preparation_sha256,'preparation contract')
        if self.route not in ('dram->vram','vram->dram') or self.memory_mode not in ('registered','pageable'):
            raise InvalidModel('explicit transfer route and observed host category required')
        if not isinstance(self.sustained,PiecewiseCompletedResponse) or self.sustained.signed:
            raise InvalidModel('positive sustained completed-response curve required')
        if type(self.first_impulses) is not tuple or not self.first_impulses:
            raise InvalidModel('explicit measured first-response idle coordinates required')
        idles=[]
        for impulse in self.first_impulses:
            if not isinstance(impulse,FirstResponseImpulse):raise InvalidModel('typed first response required')
            if impulse.response.byte_knots!=self.sustained.byte_knots:raise InvalidModel('first/sustained byte inventories differ')
            idles.append(impulse.requested_idle_ns)
            if any(a+b<=0 for a,b in zip(self.sustained.response_ns,impulse.response.response_ns)):
                raise InvalidModel('total first completed response must remain positive')
        if len(set(idles))!=len(idles):raise InvalidModel('duplicate idle coordinate')
        integer(self.maximum_copies,'maximum COPY window',1)
        if type(self.first_qualified) is not bool or type(self.window_qualified) is not bool:
            raise InvalidModel('independent first/window qualification flags required')
        for field in (self.qualified_byte_coordinates,self.qualified_window_lengths):
            if type(field) is not tuple:raise InvalidModel('explicit independently validated coordinate tuples required')
            for value in field:integer(value,'qualified coordinate',1)
        if any(not self.sustained.byte_knots[0]<=n<=self.sustained.byte_knots[-1] for n in self.qualified_byte_coordinates):raise InvalidModel('qualified byte coordinate outside scope')
        if any(n>self.maximum_copies for n in self.qualified_window_lengths):raise InvalidModel('qualified window outside scope')

    def _check(self,size_bytes,requested_idle_ns,context_sha256,preparation_sha256):
        if context_sha256!=self.context_sha256 or preparation_sha256!=self.preparation_sha256:
            raise UnknownParameter('unmeasured runtime or preparation; no warm-state substitution')
        integer(requested_idle_ns,'declared idle ns')
        impulse=next((x for x in self.first_impulses if x.requested_idle_ns==requested_idle_ns),None)
        if impulse is None:raise UnknownParameter('unmeasured idle coordinate; no implicit recovery interpolation')
        baseline=self.sustained.at(size_bytes)
        return baseline,baseline+impulse.response.at(size_bytes)

    def first_response_ns(self,size_bytes,requested_idle_ns,*,context_sha256,preparation_sha256,require_qualified=True):
        base,first=self._check(size_bytes,requested_idle_ns,context_sha256,preparation_sha256)
        if require_qualified and (not self.first_qualified or size_bytes not in self.qualified_byte_coordinates):
            raise UnknownParameter('first COPY response lacks qualification at this coordinate')
        return math.ceil(first)

    def window_ns(self,size_bytes,requested_idle_ns,copies,*,context_sha256,preparation_sha256,require_qualified=True):
        integer(copies,'COPY window',1)
        if copies>self.maximum_copies:raise UnknownParameter('COPY count outside declared lifecycle envelope')
        base,first=self._check(size_bytes,requested_idle_ns,context_sha256,preparation_sha256)
        if require_qualified and (not self.window_qualified or size_bytes not in self.qualified_byte_coordinates or copies not in self.qualified_window_lengths):
            raise UnknownParameter('complete window lacks qualification at this coordinate')
        return math.ceil(first)+(copies-1)*math.ceil(base)

    def describe(self):
        return dict(boundary='inclusive host-completed COPY response',equation='T(N,b,i)=b_m(b)+p_m(b,i)+(N-1)*b_m(b)',
                    first_depends_on_future_window=False,physical_power_queue_OS_cause_identified=False,
                    idle_interpolation=False,context_sha256=self.context_sha256,preparation_sha256=self.preparation_sha256,
                    first_qualified=self.first_qualified,window_qualified=self.window_qualified)

    @classmethod
    def from_profile(cls,profile):
        if profile.get('schema')!='completed-transfer-lifecycle-1':raise InvalidModel('completed lifecycle profile schema differs')
        if profile.get('fit_uses_whole_LLM_target') is not False or profile.get('future_target_observations_used') is not False:
            raise InvalidModel('target-blind parameter provenance required')
        try:
            def curve(value):
                return PiecewiseCompletedResponse(tuple(value['byte_knots']),tuple(value['response_ns']),tuple(value['evidence']),value.get('signed',False))
            return cls(profile['context_sha256'],profile['preparation_sha256'],profile['route'],profile['memory_mode'],curve(profile['sustained']),
                       tuple(FirstResponseImpulse(r['requested_idle_ns'],curve(r['response'])) for r in profile['first_impulses']),
                       profile['maximum_copies'],profile['first_qualified'],profile['window_qualified'],
                       tuple(profile['qualified_byte_coordinates']),tuple(profile['qualified_window_lengths']))
        except (KeyError,TypeError) as error:raise InvalidModel('complete explicit lifecycle parameters required') from error
