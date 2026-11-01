"""Model-owned causal state evolution, without fitted default coefficients."""
from unittest import TestCase

from llm_infer_model.performance.idle_recovery import (
    IdleRecoveryParameters,IdleRecoveryResponse,IdleRecoveryScope,ReadinessDebt,
)
from llm_infer_model.performance.parameters import InvalidModel,UnknownParameter


class IdleRecoveryTests(TestCase):
    def model(self,extra=100000):
        return IdleRecoveryResponse(IdleRecoveryScope('a'*64,'dram->vram',2097152,'pageable',20000000),
            IdleRecoveryParameters(250000,extra,2000000,500000,('explicit test-only coefficients; not hardware calibration',)))

    def test_zero_idle_and_zero_debt_are_steady(self):
        r=self.model().predict_burst(0,16,initial_state=ReadinessDebt(0))
        self.assertEqual(r['duration_ns'],16*250000)
        self.assertTrue(all(a['state_before']==0 for a in r['rows']))

    def test_idle_fills_but_repeated_work_recovers(self):
        r=self.model().predict_burst(10000000,16,initial_state=ReadinessDebt(0))
        values=[a['predicted_ns'] for a in r['rows']]
        self.assertGreater(values[0],values[-1]);self.assertEqual(values,sorted(values,reverse=True))

    def test_longer_idle_is_monotonic_and_bounded(self):
        model=self.model();values=[model.predict_burst(t,1,initial_state=ReadinessDebt(0))['duration_ns']
                                   for t in (0,1000000,5000000,20000000)]
        self.assertEqual(values,sorted(values));self.assertLessEqual(max(values),350000)

    def test_explicit_zero_amplitude_means_no_effect(self):
        r=self.model(extra=0).predict_burst(20000000,16,initial_state=ReadinessDebt(0))
        self.assertEqual(r['duration_ns'],16*250000)

    def test_completion_uses_own_prediction_for_clock_and_decay(self):
        model=self.model();s=ReadinessDebt(.7);duration,next_state=model.complete(s,
            scope_sha256='a'*64,route='dram->vram',payload_bytes=2097152,memory_mode='pageable')
        self.assertEqual(duration,320000);self.assertLess(next_state.value,s.value)
        row=model.predict_burst(0,2,initial_state=s)['rows']
        self.assertEqual(row[1]['begin_predicted_ns'],row[0]['predicted_ns'])

    def test_unknown_coordinate_and_extrapolation_rejected(self):
        model=self.model()
        with self.assertRaises(UnknownParameter):model.after_idle(ReadinessDebt(0),20000001)
        with self.assertRaises(UnknownParameter):model.complete(ReadinessDebt(0),scope_sha256='a'*64,
            route='dram->vram',payload_bytes=1048576,memory_mode='pageable')

    def test_unknown_or_nonfinite_parameters_are_not_defaults(self):
        for value in (None,float('nan'),float('inf'),-1,True):
            with self.assertRaises(InvalidModel):IdleRecoveryParameters(value,1,2,3,('test',))
        with self.assertRaises(InvalidModel):IdleRecoveryParameters(1,2,3,4,())

    def test_proxy_is_not_a_utilization_or_probability_claim(self):
        for value in (-.1,1.1):
            with self.assertRaises(InvalidModel):ReadinessDebt(value)
        r=self.model().predict_burst(1,1,initial_state=ReadinessDebt(0))
        self.assertFalse(r['physical_mechanism_identified']);self.assertFalse(r['future_observed_state_used'])
