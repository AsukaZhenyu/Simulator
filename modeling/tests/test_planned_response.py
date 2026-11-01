from copy import deepcopy
from unittest import TestCase

from llm_infer_model.performance.planned_response import PlannedCompletedResponse,STATES
from llm_infer_model.performance.parameters import InvalidModel,UnknownParameter


def model(provider='memoryless',*,cpu=((50,150),),gpu=(),prices=None,tau=None):
    if prices is None:prices={'a'*64:dict(kind='COMPUTE',states={(0,0):100,(1,0):200,(0,1):200,(1,1):300})}
    return PlannedCompletedResponse('b'*64,prices,('a'*64,),cpu,gpu,provider=provider,evidence=('declared test data',),
        tau_ns=tau,origin_ns=0,horizon_ns=100000)


class PlannedResponseTests(TestCase):
    def test_own_completion_time_controls_next_replay(self):
        predicted=model().predict_replays(replays=2,block_replays=1)
        self.assertEqual(predicted['block_ns'],[150,100]);self.assertEqual(predicted['total_ns'],250)
        self.assertTrue(predicted['own_predicted_completion_clock'])
        self.assertFalse(predicted['observed_target_boundaries_used'])

    def test_static_baseline_uses_exact_off_off_price(self):
        self.assertEqual(model('static').predict_replays(replays=2,block_replays=1)['block_ns'],[100,100])

    def test_cpu_and_gpu_use_identical_calculation_framework(self):
        cpu=model(cpu=((50,150),),gpu=()).predict_replays(replays=4,block_replays=1)
        gpu=model(cpu=(),gpu=((50,150),)).predict_replays(replays=4,block_replays=1)
        self.assertEqual(cpu['block_ns'],gpu['block_ns'])

    def test_no_actual_target_boundary_argument_exists(self):
        with self.assertRaises(TypeError):model().predict_replays(replays=2,block_replays=1,actual_starts=[0,100])

    def test_unknown_state_price_is_not_inferred(self):
        prices={'a'*64:dict(kind='COMPUTE',states={s:100 for s in STATES})};prices['a'*64]['states'].pop((1,1))
        with self.assertRaisesRegex(UnknownParameter,'offered-state'):model(prices=prices)

    def test_changed_semantic_coordinate_is_rejected(self):
        with self.assertRaisesRegex(UnknownParameter,'exact action coordinate'):model().evaluate(semantic_order=('c'*64,))

    def test_stateful_common_g_reduces_to_constant_when_all_cells_equal(self):
        prices={'a'*64:dict(kind='COMPUTE',states={s:100 for s in STATES})}
        stateful=model('stateful',prices=prices,tau=100)
        predicted=stateful.predict_replays(replays=3,block_replays=1)
        self.assertEqual(predicted['block_ns'],[100,100,100])
        self.assertEqual(predicted['calculation_provider'],'llm_infer_model.performance.evaluate')

    def test_stateful_requires_explicit_tau_and_positive_compute_response(self):
        with self.assertRaises(InvalidModel):model('stateful')
        prices={'a'*64:dict(kind='COMPUTE',states={s:0 for s in STATES})}
        with self.assertRaisesRegex(UnknownParameter,'subclock-zero'):model('stateful',prices=prices,tau=100)

    def test_fixed_future_plan_changes_after_finish_cannot_change_past(self):
        a=model(cpu=((1000,2000),)).predict_replays(replays=2,block_replays=1)
        b=model(cpu=((9000,10000),)).predict_replays(replays=2,block_replays=1)
        self.assertEqual(a['block_ns'],b['block_ns'])
