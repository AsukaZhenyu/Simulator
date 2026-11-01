from dataclasses import replace
import unittest

from llm_infer_model.performance.causal_inputs import (
    ResourceInput,validate_resource_input,validate_budget_contract,
)
from llm_infer_model.performance.parameters import InvalidModel


def external(**changes):
    fields=dict(id='external',quantity='available_service',source_kind='declared_external',
        begin_ns=100,end_ns=200,available_at_ns=0,clock_id='scenario',clock_origin='zero',
        evidence=('declared external CPU/GPU resource function',),includes_own_work=False,policy_invariant=True)
    fields.update(changes)
    return ResourceInput(**fields)


def validate(value,**changes):
    fields=dict(role='exogenous_input',consumer='model_scenario',cutoff_ns=50,
        clock_id='scenario',clock_origin='zero',evaluation_policy='B')
    fields.update(changes)
    return validate_resource_input(value,**fields)


class CausalInputTests(unittest.TestCase):
    def test_model_can_read_entire_declared_external_function(self):
        r=validate(external())
        self.assertTrue(r['future_window'])
        self.assertFalse(r['online_prediction_claimed'])

    def test_announced_future_budget_is_visible_to_policy(self):
        self.assertTrue(validate(external(),consumer='online_policy')['visible_at_cutoff'])
        with self.assertRaisesRegex(InvalidModel,'unannounced'):
            validate(external(available_at_ns=60),consumer='online_policy')

    def test_independent_measured_future_replay_is_not_online_prediction(self):
        value=external(source_kind='measured',available_at_ns=210)
        self.assertFalse(validate(value)['online_prediction_claimed'])
        with self.assertRaises(InvalidModel):validate(value,consumer='online_policy')
        with self.assertRaises(InvalidModel):external(source_kind='measured',available_at_ns=0)

    def total(self):
        return external(quantity='total_utilization',source_kind='measured',begin_ns=10,end_ns=40,
            available_at_ns=45,includes_own_work=True,policy_invariant=False,source_policy='A')

    def test_past_total_state_is_legitimate_feedback_not_capacity(self):
        value=self.total()
        self.assertTrue(validate(value,role='feedback',consumer='online_policy')['contains_target_effects'])
        with self.assertRaises(InvalidModel):validate(value,role='exogenous_input')
        with self.assertRaises(InvalidModel):validate(replace(value,end_ns=60,available_at_ns=65),role='feedback')

    def test_counterfactual_replay_cannot_reuse_own_work_trace(self):
        value=self.total()
        with self.assertRaisesRegex(InvalidModel,'counterfactual'):
            validate(value,role='descriptive_replay',consumer='same_policy_diagnostic')
        r=validate(value,role='descriptive_replay',consumer='same_policy_diagnostic',evaluation_policy='A')
        self.assertTrue(r['conditional_diagnostic'])
        self.assertFalse(r['online_prediction_claimed'])

    def test_own_memory_is_model_state_not_external_load(self):
        value=external(quantity='owned_memory',source_kind='model_internal',includes_own_work=True,
            policy_invariant=False,source_policy='B')
        self.assertFalse(validate(value,role='internal_accounting')['total_utilization_converted_to_capacity'])
        with self.assertRaises(InvalidModel):validate(value)
        with self.assertRaises(InvalidModel):validate(value,role='internal_accounting',evaluation_policy='A')

    def test_noninvariant_state_requires_matching_policy_even_without_raw_own_work(self):
        value=replace(self.total(),quantity='frequency',includes_own_work=False)
        for source_policy in ('A',None):
            with self.subTest(source_policy=source_policy),self.assertRaisesRegex(InvalidModel,'counterfactual'):
                validate(replace(value,source_policy=source_policy),role='descriptive_replay',consumer='same_policy_diagnostic')
        self.assertTrue(validate(value,role='descriptive_replay',consumer='same_policy_diagnostic',
            evaluation_policy='A')['conditional_diagnostic'])
        independent=replace(value,policy_invariant=True)
        self.assertTrue(validate(independent,role='descriptive_replay',consumer='same_policy_diagnostic')['declared_policy_invariance'])

    def test_target_effects_cannot_claim_invariance_and_clocks_must_match(self):
        with self.assertRaisesRegex(InvalidModel,'invariance'):
            external(includes_own_work=True,policy_invariant=True,source_policy='A')
        with self.assertRaisesRegex(InvalidModel,'clock'):
            validate(external(),clock_origin='other')

    def test_net_budget_rejects_double_charge(self):
        r=validate_budget_contract(semantics='net_remaining_budget',other_application_kind='already_net',
            external_reservations_charged_separately=False)
        self.assertFalse(r['OS_scheduler_simulated'])
        with self.assertRaisesRegex(InvalidModel,'twice'):
            validate_budget_contract(semantics='net_remaining_budget',other_application_kind='reservation',
                external_reservations_charged_separately=True)

    def test_external_offered_demand_needs_rule_when_oversubscribed(self):
        fields=dict(semantics='pool_capacity_minus_other_reservations',other_application_kind='demand',
            external_reservations_charged_separately=True,oversubscribed=True)
        with self.assertRaisesRegex(InvalidModel,'sharing'):
            validate_budget_contract(**fields)
        self.assertEqual(validate_budget_contract(**fields,sharing_rule='declared proportional resource sharing')['sharing_rule'],
            'declared proportional resource sharing')


if __name__=='__main__':unittest.main()
