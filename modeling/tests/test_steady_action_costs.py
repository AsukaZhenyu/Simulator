from dataclasses import replace
from fractions import Fraction
import unittest

from llm_infer_model.performance.graph import ResourcePoint,ResourceTimeline
from llm_infer_model.performance.parameters import InvalidModel,UnknownParameter
from llm_infer_model.performance.steady_action_costs import (
    SteadyActionKey,SteadyActionCost,SteadyActionTable,compile_steady_actions,
)


def key(**changes):
    fields=dict(action_signature='exact-action-v1',backend='cuda:0',
        shape_signature='f32:16x512x512',execution_scope='hardware/backend/thread/boundary/steady-v1')
    fields.update(changes)
    return SteadyActionKey(**fields)


def cost(mean=10,**changes):
    fields=dict(key=key(),mean_ns=mean,evidence=('profile.json:sha256:exact',),sample_count=12,process_ids=(101,102,103))
    fields.update(changes)
    return SteadyActionCost(**fields)


class SteadyActionTests(unittest.TestCase):
    def test_serial_prices_are_computed_by_common_graph(self):
        a,b=key(),key(action_signature='second')
        table=SteadyActionTable((cost(),cost(7,key=b)))
        compiled=compile_steady_actions((a,b,a),table)
        self.assertEqual(compiled.evaluate().duration_ns,27)
        self.assertEqual(len(compiled.parameters.parameters),2)
        self.assertEqual(len(compiled.graph.tasks),3)
        self.assertTrue(all(p.unit=='invocation/s' for p in compiled.parameters.parameters))

    def test_exact_rational_is_preserved_until_common_completion_rounding(self):
        c=cost(Fraction(7,2))
        self.assertEqual(c.parameter().value,Fraction(2_000_000_000,7))
        result=compile_steady_actions((c.key,c.key),SteadyActionTable((c,))).evaluate()
        self.assertEqual(result.duration_ns,8)

    def test_backend_shape_scope_and_action_must_all_match(self):
        table=SteadyActionTable((cost(),))
        for field in ('backend','shape_signature','execution_scope','action_signature'):
            with self.subTest(field=field),self.assertRaises(UnknownParameter):
                table.lookup(replace(key(),**{field:'other'}))

    def test_device_diagnostics_are_never_added_or_subtracted(self):
        c=cost(10,device_interval_ns=1000,device_interval_scope='separate async envelope',
            device_interval_evidence=('events.json:sha256:exact',))
        result=compile_steady_actions((c.key,),SteadyActionTable((c,))).evaluate()
        self.assertEqual(result.duration_ns,10)
        self.assertFalse(c.to_dict()['device_interval_is_additional_latency'])

    def test_unknown_mean_remains_unknown(self):
        c=cost(None)
        self.assertIsNone(c.parameter().value)
        with self.assertRaises(UnknownParameter):
            compile_steady_actions((c.key,),SteadyActionTable((c,))).evaluate()

    def test_zero_requires_explicit_resolution_limited_approximation(self):
        with self.assertRaises(InvalidModel):cost(0)
        c=cost(0,zero_duration_approximation=True)
        self.assertEqual(compile_steady_actions((c.key,),SteadyActionTable((c,))).evaluate().duration_ns,0)
        with self.assertRaises(InvalidModel):cost(10,zero_duration_approximation=True)

    def test_inclusive_invocation_rejects_additional_device_demand(self):
        from llm_infer_model.performance.graph import Task,WorkDemand
        with self.assertRaisesRegex(InvalidModel,'inclusive invocation'):
            Task('bad','gpu',(WorkDemand('response',1,'invocation'),WorkDemand('gpu',1,'flop')))

    def test_profile_provenance_is_required(self):
        for values in (dict(evidence=()),dict(process_ids=()),dict(process_ids=(101,101)),dict(sample_count=0)):
            with self.subTest(values=values),self.assertRaises(InvalidModel):cost(**values)
        with self.assertRaises(InvalidModel):cost(device_interval_ns=3)

    def test_no_float_or_duplicate_key_prices(self):
        with self.assertRaises(InvalidModel):cost(1.5)
        with self.assertRaises(InvalidModel):SteadyActionTable((cost(),cost(20)))

    def test_explicit_response_timeline_does_not_infer_physical_slowdown(self):
        c=cost(10);compiled=compile_steady_actions((c.key,),SteadyActionTable((c,)))
        name=c.key.parameter_name
        timeline=ResourceTimeline((ResourcePoint(0,{name:Fraction(50_000_000)}),))
        self.assertEqual(compiled.evaluate(timeline).duration_ns,20)
        self.assertFalse(compiled.to_dict()['dynamic_available_resource_function_inferred'])


if __name__=='__main__':unittest.main()
