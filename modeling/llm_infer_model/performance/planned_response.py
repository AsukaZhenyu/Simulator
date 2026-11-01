"""Scoped planned offered-work responses, evaluated by the common event engine.

The task unit is one inclusive completed invocation. Offered CPU/GPU work bits
are not FLOPs or utilization. Exact response cells and an optional globally
shared empirical regime-transition time are supplied by profiling. This module
contains no fitting, native measurement or future target observations.
"""
from __future__ import annotations

from bisect import bisect_right
from fractions import Fraction
import math

from .evaluator import evaluate
from .graph import Graph,ResourcePoint,ResourceTimeline,Task,WorkDemand
from .parameters import (INSTANTANEOUS_RATE,InvalidModel,Parameter,ParameterBundle,
    UnknownParameter,integer)
from .service_response import (BackgroundReservoir,ServiceInterference,
    ServiceRequestPoint,ServiceRequestTrajectory,ServiceResponseModel,
    ServiceResponseParameter,build_service_response)

STATES=((0,0),(1,0),(0,1),(1,1))


class PlannedCompletedResponse:
    """Serial completed actions under a fixed planned actor timeline.

``prices`` is keyed by normalized semantic digest, never layer/strategy IDs.
The optional stateful law uses four empirical offered-state reservoirs in the
existing g framework. It retains a bounded transition proxy; it does not claim
literal OS queues, physical compute service or identified power states.
"""
    def __init__(self,context_sha256,prices,semantic_order,cpu_intervals_ns,gpu_intervals_ns,
                 *,provider,evidence,tau_ns=None,origin_ns=-2200000000,horizon_ns=300000000000):
        if provider not in ('static','memoryless','stateful'):raise InvalidModel('unknown planned response provider')
        if not isinstance(context_sha256,str) or len(context_sha256)!=64:raise InvalidModel('response context SHA256 required')
        if type(semantic_order) is not tuple or not semantic_order:raise InvalidModel('explicit ordered semantic coordinates required')
        if type(evidence) is not tuple or not evidence:raise InvalidModel('profile response evidence required')
        if type(origin_ns) is not int or origin_ns>0:raise InvalidModel('nonpositive planned history origin required')
        integer(horizon_ns,'response horizon',1)
        self.provider=provider;self.origin_ns=origin_ns;self.horizon_ns=horizon_ns;self.order=semantic_order;self.evidence=evidence
        self.keys=tuple(dict.fromkeys(semantic_order));self.prices={};self.kinds={};self._graphs={}
        for key in self.keys:
            if key not in prices:raise UnknownParameter('unknown exact semantic response: '+key)
            row=prices[key];self.kinds[key]=row['kind'];values={}
            for state in STATES:
                value=row['states'].get(state)
                if value is None:raise UnknownParameter('unknown exact offered-state response: '+key+str(state))
                value=Fraction(value)
                if value<0:raise InvalidModel('negative inclusive completed response')
                values[state]=value
            self.prices[key]=values
        self.intervals=[]
        for intervals in (cpu_intervals_ns,gpu_intervals_ns):
            result=[];previous=origin_ns
            for pair in intervals:
                if len(pair)!=2 or any(type(x) is not int for x in pair):raise InvalidModel('integer planned intervals required')
                a,b=pair
                if not origin_ns<=a<b<=horizon_ns or a<previous:raise InvalidModel('planned actor interval order/envelope differs')
                result.append((a,b));previous=b
            self.intervals.append(tuple(result))
        self.edges=tuple(sorted({origin_ns,*[x for intervals in self.intervals for pair in intervals for x in pair]}))
        self.names={key:'response:'+key for key in self.keys}
        self.bundle=ParameterBundle(tuple(Parameter(self.names[key],self._rate(self.prices[key][(0,0)]),'invocation/s','profiled',evidence,
            'inclusive completed action response under the declared owned actor protocol') for key in self.keys))
        self.points=tuple(ResourcePoint(t-origin_ns,{self.names[key]:self._rate(self.prices[key][self.state_at(t)]) for key in self.keys}) for t in self.edges)
        self.point_times=tuple(p.at_ns for p in self.points)
        self.service=None
        if provider=='stateful':
            integer(tau_ns,'explicit identified empirical transition ns',1)
            self.service=self._stateful(context_sha256,tau_ns)

    @staticmethod
    def _rate(value):return INSTANTANEOUS_RATE if value==0 else Fraction(1000000000)/value

    def state_at(self,relative_ns):
        return tuple(int(any(a<=relative_ns<b for a,b in intervals)) for intervals in self.intervals)

    def _stateful(self,context,tau):
        # Four explicit response regimes cover joint CPU/GPU effects. A fastest
        # measured reference keeps empirical coefficients nonnegative; it is
        # not a hardware peak-throughput parameter.
        competitors=tuple(BackgroundReservoir('mode'+str(i),'capacity','arrival','drain','activity','backlog') for i in range(4))
        specifications=(('capacity',1,'normalized_work'),('arrival',Fraction(2,tau),'normalized_work/ns'),
            ('drain',Fraction(1,tau),'normalized_work/ns'),('activity',0,'dimensionless'),('backlog',1,'dimensionless'))
        params=[ServiceResponseParameter(name,value,unit,'profiled' if name in ('arrival','drain') else 'declared',self.evidence,
            'bounded empirical offered-response regime; not a physical actor queue') for name,value,unit in specifications]
        base=[];links=[];control_names=set()
        for key in self.keys:
            name=self.names[key];values=self.prices[key]
            if self.kinds[key] in ('ADVANCE','EVICT'):
                control_names.add(name);base.append(self.bundle.get(name));continue
            fastest=min(values.values())
            if fastest<=0:raise UnknownParameter('subclock-zero compute/copy response cannot identify stateful recovery: '+key)
            base.append(Parameter(name,Fraction(1000000000)/fastest,'invocation/s','profiled',self.evidence,
                'inverse fastest measured inclusive response; not physical peak service'))
            for i,state in enumerate(STATES):
                coefficient='coefficient:'+key+':'+str(i)
                params.append(ServiceResponseParameter(coefficient,values[state]/fastest-1,'dimensionless','profiled',self.evidence,
                    'exact four-state profile price relative to fastest measured response'))
                links.append(ServiceInterference(name,'mode'+str(i),coefficient))
        if not links:raise UnknownParameter('no measured compute/copy response can identify a stateful transition')
        model=ServiceResponseModel(context,tuple(params),competitors,tuple(links),
            ('four empirical offered-state response regimes; not physical OS/CUDA queues',
             'one profile-identified transition time shared by CPU/GPU and all action semantics',
             'initial response regime explicitly equals the declared plan state at history origin'),False)
        initial_state=self.state_at(self.origin_ns)
        requests=lambda state:{'mode'+str(i):int(s==state) for i,s in enumerate(STATES)}
        initial=requests(initial_state)
        trajectory=ServiceRequestTrajectory(initial,{key:Fraction(value) for key,value in initial.items()},
            tuple(ServiceRequestPoint(t-self.origin_ns,requests(self.state_at(t))) for t in self.edges[1:]),
            'predeclared_scenario',self.evidence,'planned-nanosecond-clock','offered-response:'+context)
        control=ResourceTimeline(tuple(ResourcePoint(p.at_ns,{name:value for name,value in p.rates.items() if name in control_names}) for p in self.points))
        return build_service_response(model,ParameterBundle(tuple(base)),trajectory,end_ns=self.horizon_ns-self.origin_ns,
            max_step_ns=max(1,min(1000000,tau//32)),max_points=200000,baseline_timeline=control)

    def _graph(self,order):
        if order not in self._graphs:
            for key in order:
                if key not in self.names:raise UnknownParameter('unprofiled exact action coordinate: '+key)
            self._graphs[order]=Graph(tuple(Task('action:'+str(i),'inclusive_completed_action',
                (WorkDemand(self.names[key],1,'invocation'),),() if i==0 else ('action:'+str(i-1),)) for i,key in enumerate(order)))
        return self._graphs[order]

    def evaluate(self,*,start_ns=0,semantic_order=None):
        if type(start_ns) is not int or not self.origin_ns<=start_ns<=self.horizon_ns:raise UnknownParameter('response entry outside scoped planned timeline')
        order=self.order if semantic_order is None else semantic_order
        if type(order) is not tuple or not order:raise InvalidModel('nonempty ordered response coordinates required')
        graph=self._graph(order)
        if self.provider=='static':return evaluate(graph,self.bundle)
        shifted=start_ns-self.origin_ns
        if self.provider=='stateful':return self.service.evaluate_serial_response(tuple(self.names[key] for key in order),start_ns=shifted)
        bound=sum(math.ceil(max(self.prices[key].values())) for key in order)+len(order)
        end=min(self.horizon_ns-self.origin_ns,shifted+bound)
        left=bisect_right(self.point_times,shifted);right=bisect_right(self.point_times,end)
        initial=ResourcePoint(0,self.points[left-1].rates)
        future=tuple(ResourcePoint(p.at_ns-shifted,p.rates) for p in self.points[left:right])
        result=evaluate(graph,self.bundle,ResourceTimeline((initial,*future)))
        if start_ns+result.duration_ns>self.horizon_ns:raise UnknownParameter('prediction exceeds response horizon')
        return result

    def predict_replays(self,*,replays=2048,block_replays=16,start_ns=0):
        integer(replays,'continuous replays',1);integer(block_replays,'replays per block',1)
        if replays%block_replays:raise InvalidModel('whole fixed-work report blocks required')
        entry=start_ns;blocks=[];block_begin=entry
        static_result=self.evaluate(start_ns=start_ns) if self.provider=='static' else None
        for index in range(replays):
            result=static_result if static_result is not None else self.evaluate(start_ns=entry)
            entry+=result.duration_ns
            if entry>self.horizon_ns:raise UnknownParameter('prediction exceeds response horizon')
            if (index+1)%block_replays==0:blocks.append(entry-block_begin);block_begin=entry
        return dict(block_ns=blocks,total_ns=entry-start_ns,planned_entry_ns=start_ns,
            future_target_state_used=False,observed_target_boundaries_used=False,
            response_unit='inclusive completed invocation ns',physical_FLOPs_or_utilization_claimed=False,
            calculation_provider='llm_infer_model.performance.evaluate',
            response_regime_history='bounded empirical proxy' if self.provider=='stateful' else 'memoryless',
            own_predicted_completion_clock=True)
