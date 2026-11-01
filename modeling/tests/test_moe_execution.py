"""Hand-computed runtime response contracts; no hardware or target labels."""
from dataclasses import replace
from fractions import Fraction
import json
from random import Random
import unittest
from unittest.mock import patch

from llm_infer_model.performance.evaluator import evaluate
from llm_infer_model.performance.graph import ResourcePoint, ResourceTimeline
from llm_infer_model.performance.moe_execution import (
    RuntimeWindowAction, RuntimeWindowPrice, compile_runtime_windows,
)
from llm_infer_model.performance.parameters import InvalidModel, UnknownParameter


def action(**kwargs):
    values = dict(id="cpu0", key="cpu-mmid-decode-layer0", scope="CPU_NODE",
                  query_tokens=1, kv_padded=32, history="persistent-multilayer",
                  phase="decode", layer=0, op="MUL_MAT_ID", covered_nodes=("gate-0",))
    return RuntimeWindowAction(**dict(values, **kwargs))


def price(**kwargs):
    values = dict(key="cpu-mmid-decode-layer0", scope="CPU_NODE", intercept_ns=3,
                  per_query_token_ns=2, per_attention_pair_ns=Fraction(1, 2),
                  observed_query_tokens=(1,), observed_kv_padded=(32, 64),
                  history_domain=("persistent-multilayer",), evidence=("independent-calibration.json#sha256",))
    return RuntimeWindowPrice(**dict(values, **kwargs))


class MoEExecutionTests(unittest.TestCase):
    def test_affine_formula_exact_json_roundtrip_and_no_fitting(self):
        p = price()
        self.assertEqual(p.estimate(action()).mean_ns, 21)
        self.assertEqual(RuntimeWindowPrice.from_dict(json.loads(json.dumps(p.to_dict()))), p)
        self.assertEqual(RuntimeWindowAction.from_dict(action().to_dict()), action())

    def test_negative_unknown_and_float_coefficients_are_rejected(self):
        for bad in (-1, None, 0.5, True):
            with self.subTest(value=bad), self.assertRaises(InvalidModel):
                price(intercept_ns=bad)
        with self.assertRaises(InvalidModel):
            action(scope="DEVICE_GPU")
        with self.assertRaises(UnknownParameter):
            price().estimate(action(key="other-runtime"))

    def test_domain_interpolation_and_explicit_extrapolation(self):
        self.assertTrue(price().estimate(action(kv_padded=48)).interpolated)
        unseen = action(kv_padded=96, history="cold")
        with self.assertRaises(UnknownParameter):
            price().estimate(unseen)
        estimate = price().estimate(unseen, allow_extrapolation=True)
        self.assertEqual(estimate.out_of_domain, ("kv_padded", "history"))
        self.assertTrue(estimate.extrapolated)
        self.assertFalse(estimate.interpolated)

    def test_payload_axis_has_exact_cost_domain_and_json(self):
        p = price(per_payload_byte_ns=Fraction(1, 4), observed_payload_bytes=(0, 64))
        a = action(payload_bytes=32)
        self.assertEqual(p.estimate(a).mean_ns, 29)
        self.assertTrue(p.estimate(a).interpolated)
        self.assertEqual(RuntimeWindowPrice.from_dict(json.loads(json.dumps(p.to_dict()))), p)
        self.assertEqual(RuntimeWindowAction.from_dict(a.to_dict()), a)
        with self.assertRaises(UnknownParameter):
            p.estimate(action(payload_bytes=96))
        self.assertEqual(p.estimate(action(payload_bytes=96), allow_extrapolation=True).out_of_domain,
                         ("payload_bytes",))
        for bad in (-1, 0.5, True):
            with self.subTest(value=bad), self.assertRaises(InvalidModel):
                action(payload_bytes=bad)
            with self.subTest(value=bad), self.assertRaises(InvalidModel):
                price(per_payload_byte_ns=bad)
        with self.assertRaises(InvalidModel):
            price(observed_payload_bytes=(-1,))

    def test_whole_forward_cannot_be_a_cpu_node_and_bridge_requires_coverage(self):
        for changes in (dict(layer=None), dict(op=None), dict(op="FORWARD"),
                        dict(covered_nodes=tuple("layer" + str(i) for i in range(16)))):
            with self.subTest(changes=changes), self.assertRaises(InvalidModel):
                action(**changes)
        with self.assertRaises(InvalidModel):
            action(scope="RUNTIME_BRIDGE", covered_nodes=())
        global_node = action(layer=None, global_operation=True, op="GET_ROWS", covered_nodes=("embedding",))
        self.assertTrue(global_node.global_operation)
        with self.assertRaises(InvalidModel):
            action(global_operation=True)

    def test_closed_windows_compose_and_internal_work_is_not_added(self):
        a = action()
        b = action(id="bridge", key="bridge", scope="RUNTIME_BRIDGE", layer=None,
                   op=None, covered_nodes=("attention-1", "router-1"))
        p = price(key="bridge", scope="RUNTIME_BRIDGE", intercept_ns=7,
                  per_query_token_ns=0, per_attention_pair_ns=0)
        compiled = compile_runtime_windows((a, b), (price(), p))
        self.assertEqual(compiled.evaluate().duration_ns, 28)
        self.assertEqual(len(compiled.graph.tasks), 2)
        self.assertEqual(compiled.graph.tasks[1].depends_on, (a.id,))
        self.assertFalse(compiled.to_dict()["device_or_copy_diagnostics_added"])
        with self.assertRaises(InvalidModel):
            compile_runtime_windows((a, replace(b, covered_nodes=("gate-0",))), (price(), p))
        repeated_forward = replace(a, id="next-forward", forward_index=1)
        self.assertEqual(compile_runtime_windows((a, repeated_forward), (price(),)).evaluate().duration_ns, 42)

    def test_explicit_response_rate_pause_keeps_progress(self):
        compiled = compile_runtime_windows((action(),), (price(),))
        name = compiled.parameters.parameters[0].name
        timeline = ResourceTimeline((ResourcePoint(7, {name: 0}),
                                     ResourcePoint(17, {name: Fraction(1_000_000_000, 21)})))
        self.assertEqual(compiled.evaluate(timeline).duration_ns, 31)

    def test_static_chain_matches_common_engine_records_and_events_exactly(self):
        rng = Random(8173)
        cases = [(Fraction(0), Fraction(0), Fraction(0)),
                 (Fraction(3, 2), Fraction(3, 2), Fraction(0), Fraction(1, 3))]
        cases.extend(tuple(Fraction(rng.randrange(32), rng.randrange(1, 14))
                           for _ in range(rng.randrange(1, 13))) for _ in range(40))
        for means in cases:
            with self.subTest(means=means):
                actions = tuple(action(id="window" + str(i), key="key" + str(i),
                                       phase="prefill" if i % 2 else "decode", forward_index=i)
                                for i in range(len(means)))
                prices = tuple(price(key=a.key, intercept_ns=mean,
                                     per_query_token_ns=0, per_attention_pair_ns=0,
                                     zero_duration_approximation=mean == 0)
                               for a, mean in zip(actions, means))
                compiled = compile_runtime_windows(actions, prices)
                expected = evaluate(compiled.graph, compiled.parameters)
                self.assertEqual(compiled.evaluate(), expected)
                self.assertEqual(compiled.to_dict()["graph"], compiled.graph.to_dict())
        # Every window rounds independently, rather than rounding their sum.
        self.assertEqual(sum(-(-v.numerator // v.denominator) for v in cases[1]), 5)

    def test_large_static_chain_and_export_do_not_build_general_graph(self):
        actions = tuple(action(id="node" + str(i), forward_index=i) for i in range(1000))
        with patch("llm_infer_model.performance.moe_execution.Graph",
                   side_effect=AssertionError("static windows must not validate a general Graph")):
            compiled = compile_runtime_windows(actions, (price(),))
            result = compiled.evaluate()
            exported = compiled.to_dict()
        self.assertEqual(result.duration_ns, 21_000)
        self.assertEqual(len(result.tasks), 1000)
        self.assertEqual(len(result.events), 2000)
        self.assertEqual(len(exported["graph"]["tasks"]), 1000)

    def test_explicit_timeline_still_uses_general_engine(self):
        compiled = compile_runtime_windows((action(),), (price(),))
        timeline = ResourceTimeline()
        with patch("llm_infer_model.performance.moe_execution.evaluate", wraps=evaluate) as engine:
            result = compiled.evaluate(timeline)
        engine.assert_called_once()
        self.assertEqual(result, evaluate(compiled.graph, compiled.parameters, timeline))

    def test_fast_path_rejects_a_chain_with_changed_dependency_or_rate(self):
        compiled = compile_runtime_windows((action(),), (price(),))
        with self.assertRaises(InvalidModel):
            replace(compiled, tasks=(replace(compiled.tasks[0], depends_on=("unknown",)),))
        parameter = compiled.parameters.parameters[0]
        with self.assertRaises(InvalidModel):
            replace(compiled, parameters=replace(compiled.parameters,
                                                parameters=(replace(parameter, value=1),)))

    def test_unknown_prices_and_implicit_zero_do_not_become_free_work(self):
        with self.assertRaises(UnknownParameter):
            compile_runtime_windows((action(),), (price(key="different"),))
        zero = price(intercept_ns=0, per_query_token_ns=0, per_attention_pair_ns=0)
        with self.assertRaises(InvalidModel):
            zero.estimate(action())
        explicit = replace(zero, zero_duration_approximation=True)
        self.assertEqual(compile_runtime_windows((action(),), (explicit,)).evaluate().duration_ns, 0)


if __name__ == "__main__":
    unittest.main()
