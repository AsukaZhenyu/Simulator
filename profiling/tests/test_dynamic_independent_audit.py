"""Independent small tests of forecast clock and unknown-response boundaries."""
from copy import deepcopy
from unittest import TestCase

from llm_infer_model.performance.planned_response import PlannedCompletedResponse, STATES
from llm_infer_model.performance.parameters import UnknownParameter
from profiling.dynamic_validation.external_fit import identify_tau, TAUS_NS


KEY = 'a'*64


def response(provider='memoryless', *, cpu=((40,160),), gpu=(), states=None, order=None):
    values = states or {(0,0):100,(1,0):200,(0,1):200,(1,1):400}
    return PlannedCompletedResponse('b'*64,{KEY:dict(kind='COMPUTE',states=values)},
        order or (KEY,),cpu,gpu,provider=provider,evidence=('independent synthetic software audit',),
        tau_ns=100 if provider=='stateful' else None, origin_ns=0,horizon_ns=100000)


class IndependentDynamicAuditTests(TestCase):
    def test_integrates_action_across_edge_and_recurses_on_predicted_finish(self):
        model=response(order=(KEY,KEY))
        evaluation=model.evaluate()
        self.assertEqual([(r.start_ns,r.end_ns) for r in evaluation.tasks],[(0,160),(160,260)])
        predicted=model.predict_replays(replays=2,block_replays=1)
        self.assertEqual(predicted['block_ns'],[260,200])
        self.assertEqual(predicted['total_ns'],460)

    def test_joint_state_uses_measured_joint_price(self):
        joint=response(cpu=((0,1000),),gpu=((0,1000),))
        self.assertEqual(joint.evaluate().duration_ns,400)

    def test_future_plan_and_other_device_are_causal(self):
        a=response(cpu=((5000,6000),),gpu=()).predict_replays(replays=3,block_replays=1)
        b=response(cpu=(),gpu=((9000,10000),)).predict_replays(replays=3,block_replays=1)
        self.assertEqual(a['block_ns'],[100,100,100])
        self.assertEqual(a['block_ns'],b['block_ns'])
        self.assertFalse(a['future_target_state_used'])
        self.assertFalse(a['observed_target_boundaries_used'])

    def test_stateful_common_engine_retains_constant_solution(self):
        values={state:137 for state in STATES}
        result=response('stateful',states=values).predict_replays(replays=4,block_replays=2)
        self.assertEqual(result['block_ns'],[274,274])
        self.assertEqual(result['total_ns'],548)

    def test_unknown_joint_state_is_not_product_or_off_off_fallback(self):
        values={(0,0):100,(1,0):200,(0,1):200}
        with self.assertRaises(UnknownParameter):response(states=values)

    def test_unstable_tau_cannot_be_declared_identified(self):
        losses={};counts={}
        for r,minimum in enumerate((100000,1000000,100000000)):
            for i in range(6):
                key=f'round{r}-process{i}'
                losses[key]={0:100.,**{tau:1.+abs(tau-minimum)/minimum for tau in TAUS_NS}}
                counts[key]=dict(round=r,strata=100)
        result=identify_tau(losses,counts)
        self.assertEqual(result['status'],'unsupported')
        self.assertIsNone(result['tau_ns'])
