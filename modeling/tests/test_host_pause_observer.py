"""Causality and exact uncertainty tests; no timing data or fitting."""
from dataclasses import replace
from fractions import Fraction
import unittest

from llm_infer_model.performance.host_pause_observer import (
    CompletedHostAction, HostPauseObserver, HostPauseParameters, PauseEndAnchor,
)
from llm_infer_model.performance.parameters import InvalidModel


EVIDENCE=('synthetic frozen response-phase declaration',)


def params(**kwargs):
    p=dict(period_ns=100,blackout_ns=20,min_excess_ns=20,
           base_action_ns={'a':10,'b':10},jitter_ns=3,evidence=EVIDENCE,scope='diagnostic phase hypothesis')
    p.update(kwargs)
    return HostPauseParameters(**p)


def anchor(**kwargs):
    p=dict(lower_ns=90,upper_ns=110,cutoff_ns=150,clock_id='qpc_ns',clock_origin='test_zero',evidence=EVIDENCE)
    p.update(kwargs)
    return PauseEndAnchor(**p)


def observer(*,initial=True,mode='causal_history',parameters=None):
    return HostPauseObserver(parameters or params(),initial_anchor=anchor() if initial else None,
                             clock_id='qpc_ns',clock_origin='test_zero',mode=mode)


def complete(o, call='first', begin=200, end=250, actions=None, at=None, **kwargs):
    return o.observe_completed(call,begin,end,
        (CompletedHostAction('a',begin,end),) if actions is None else actions,
        observed_at_ns=end if at is None else at,clock_id=kwargs.get('clock_id','qpc_ns'),
        clock_origin=kwargs.get('clock_origin','test_zero'))


class HostPauseObserverTests(unittest.TestCase):
    def test_before_reference_anchor_and_relative_phase(self):
        o=observer()
        p=o.predict(151)
        self.assertTrue(p.known)
        self.assertEqual((p.phase_ns,p.relative_phase_ns,p.cutoff_ns),(80,29,150))
        self.assertEqual(p.uncertainty_ns,10)
        self.assertEqual(p.source_kind,'before_reference_phase_hypothesis')

    def test_no_anchor_is_unknown_not_zero(self):
        p=observer(initial=False).predict(200)
        self.assertIsNone(p.phase_ns)
        self.assertFalse(p.known)

    def test_initial_anchor_must_precede_prediction(self):
        with self.assertRaises(InvalidModel):observer().predict(150)

    def test_completed_single_action_updates_only_phase(self):
        frozen=params()
        o=observer(parameters=frozen)
        old=o.predict(200,call_id='first')
        old_dict=old.to_dict()
        update=complete(o,at=251)
        p=o.predict(260)
        self.assertTrue(update.anchor_updated)
        self.assertEqual(p.anchor_bounds_ns,(237,253))
        self.assertEqual((p.phase_ns,p.uncertainty_ns,p.cutoff_ns),(25,8,251))
        self.assertEqual(p.source_kind,'observed_response_phase_hypothesis')
        self.assertEqual(old.to_dict(),old_dict)
        self.assertIs(o.parameters,frozen)
        self.assertEqual((o.parameters.period_ns,o.parameters.blackout_ns),(100,20))

    def test_fixed_before_only_ignores_target_history(self):
        o=observer(mode='before_only')
        before=o.predict(200,call_id='first')
        update=complete(o)
        after=o.predict(260)
        self.assertFalse(update.anchor_updated)
        self.assertEqual(before.phase_ns,after.phase_ns)
        self.assertEqual(after.cutoff_ns,150)

    def test_inflight_and_future_observation_rejected_without_consuming_call(self):
        o=observer()
        o.predict(200,call_id='first')
        with self.assertRaises(InvalidModel):complete(o,at=249)
        with self.assertRaises(InvalidModel):o.predict(210,call_id='second')
        self.assertTrue(complete(o).anchor_updated)

    def test_unregistered_wrong_and_duplicate_call_rejected(self):
        o=observer()
        with self.assertRaises(InvalidModel):complete(o)
        o.predict(200,call_id='first')
        with self.assertRaises(InvalidModel):complete(o,call='wrong')
        complete(o)
        with self.assertRaises(InvalidModel):complete(o)
        with self.assertRaises(InvalidModel):o.predict(260,call_id='first')

    def test_clock_origin_and_missing_observation_clock_rejected(self):
        o=observer()
        with self.assertRaises(InvalidModel):o.predict(200,clock_id='other')
        o.predict(200,call_id='first')
        for fields in ({'clock_id':'other'},{'clock_origin':'foreign'},{'clock_id':None}):
            with self.subTest(fields=fields),self.assertRaises(InvalidModel):complete(o,**fields)

    def test_observations_cannot_be_reused_before_cutoff(self):
        o=observer()
        o.predict(200,call_id='first')
        complete(o)
        with self.assertRaises(InvalidModel):o.predict(250)
        with self.assertRaises(InvalidModel):o.predict(249)
        self.assertTrue(o.predict(251).known)

    def test_multiple_excesses_are_ambiguous_not_largest_selected(self):
        o=observer()
        o.predict(200,call_id='first')
        update=complete(o,end=280,actions=(CompletedHostAction('a',200,240),CompletedHostAction('b',240,280)))
        self.assertFalse(update.anchor_updated)
        self.assertTrue(update.phase_ambiguous)
        self.assertIsNone(o.predict(281,call_id='second').phase_ns)
        complete(o,call='second',begin=281,end=321)
        self.assertTrue(o.predict(322).known)

    def test_unknown_action_baseline_remains_unknown(self):
        o=observer(parameters=params(base_action_ns={'a':None}))
        o.predict(200,call_id='first')
        update=complete(o)
        self.assertEqual(update.unknown_baseline_action_ids,('a',))
        self.assertIsNone(o.predict(251).phase_ns)

    def test_unknown_period_or_jitter_does_not_get_fitted(self):
        p=observer(parameters=params(period_ns=None)).predict(200)
        self.assertIsNone(p.phase_ns)
        o=observer(parameters=params(jitter_ns=None))
        o.predict(200,call_id='first')
        complete(o)
        self.assertIsNone(o.predict(251).phase_ns)
        self.assertIsNone(o.parameters.jitter_ns)

    def test_period_wide_uncertainty_has_no_unique_phase(self):
        o=HostPauseObserver(params(),initial_anchor=anchor(lower_ns=0,upper_ns=100),clock_id='qpc_ns',clock_origin='test_zero')
        self.assertIsNone(o.predict(200).phase_ns)

    def test_initial_target_response_cannot_masquerade_as_before_reference(self):
        with self.assertRaises(InvalidModel):
            HostPauseObserver(params(),initial_anchor=anchor(source_kind='completed_target_response_pause_end'),clock_id='qpc_ns',clock_origin='test_zero')

    def test_current_future_timings_do_not_change_earlier_prediction(self):
        left,right=observer(),observer()
        l0=left.predict(200,call_id='first');r0=right.predict(200,call_id='first')
        self.assertEqual(l0,r0)
        complete(left,end=250)
        complete(right,end=290)
        self.assertEqual(l0,r0)
        self.assertNotEqual(left.predict(300).phase_ns,right.predict(300).phase_ns)

    def test_exact_odd_baseline_and_explicit_floor_uncertainty(self):
        o=observer(initial=False,parameters=params(base_action_ns={'a':5},jitter_ns=2))
        o.predict(1,call_id='first')
        complete(o,begin=1,end=41)
        p=o.predict(42)
        self.assertEqual(p.anchor_bounds_ns,(34,43))
        self.assertEqual((p.exact_phase_ns,p.phase_ns,p.phase_quantization_ns),(Fraction(37,2),18,Fraction(1,2)))
        self.assertEqual(p.uncertainty_ns,5)

    def test_phase_uncertainty_wraps_instead_of_discarding_one_side(self):
        o=HostPauseObserver(params(blackout_ns=5),initial_anchor=anchor(lower_ns=0,upper_ns=10,cutoff_ns=10),
                            clock_id='qpc_ns',clock_origin='test_zero')
        p=o.predict(11)
        self.assertEqual(p.phase_ns,0)
        self.assertEqual(p.phase_intervals_ns,((95,100),(0,5)))

    def test_invalid_action_order_outside_bounds_and_duplicate_ids(self):
        for acts in ((CompletedHostAction('a',199,220),),
                     (CompletedHostAction('a',200,230),CompletedHostAction('b',220,250)),
                     (CompletedHostAction('a',200,220),CompletedHostAction('a',220,250))):
            o=observer();o.predict(200,call_id='first')
            with self.subTest(acts=acts),self.assertRaises(InvalidModel):complete(o,actions=acts)


if __name__=='__main__':unittest.main()
