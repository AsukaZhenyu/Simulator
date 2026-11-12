"""Mechanism permissions use actual shapes; no target timing or hardware."""
from dataclasses import replace
from fractions import Fraction
import json
import unittest

from llm_infer_model.performance.moe_execution import (
    RuntimeWindowAction, RuntimeWindowPrice, LEGACY_POLICY, MECHANISM_POLICY,
    MECHANISM_POLICY_V2, compile_runtime_windows,
)
from llm_infer_model.performance.parameters import InvalidModel, UnknownParameter
from llm_infer_model.performance.window_work import WindowWorkSpec, CPU_ATTENTION_STAGE


def matrix(q=1, experts=8):
    return dict(name="ffn_moe_up-0", op=30, ne=[1024, experts, q, 1], type=0,
                weight_ne=[2048, 1024, 64, 1], weight_type=12)


def attention(q=3, k=256):
    return dict(name="kq-0", op=29, ne=[k, q, 16, 1], type=0,
                weight_ne=[128, k, 16, 1], weight_type=1)


def spec(node=None, **kwargs):
    return WindowWorkSpec.from_metadata("CPU_NODE", cpu_node=matrix() if node is None else node,
                                        copy_calls=0, **kwargs)


def action(work=None, **kwargs):
    s = spec() if work is None else work
    values = dict(id="w0", key="key", scope=s.scope, query_tokens=1,
        kv_padded=256, history="decode:continuation", phase="decode",
        layer=0 if s.scope == "CPU_NODE" else None,
        op="MUL_MAT_ID" if s.scope == "CPU_NODE" else None,
        covered_nodes=("actual-node0",) if s.scope in {"CPU_NODE", "RUNTIME_BRIDGE"} else (), work_spec=s)
    return RuntimeWindowAction(**dict(values, **kwargs))


def price(work=None, **kwargs):
    s = spec() if work is None else work
    values = dict(key="key", scope=s.scope, intercept_ns=3, per_query_token_ns=2,
        per_attention_pair_ns=0, observed_query_tokens=(1, 4), observed_kv_padded=(256,),
        history_domain=("decode:continuation",), evidence=("independent-profile#sha256",),
        feature_policy=MECHANISM_POLICY, work_spec=s)
    return RuntimeWindowPrice(**dict(values, **kwargs))


class WindowWorkTests(unittest.TestCase):
    def test_matrix_disallows_kv_and_retains_static_expert_geometry(self):
        s = spec()
        self.assertEqual(s.kind, "matrix")
        self.assertEqual(s.allowed_features, ("intercept", "query_tokens"))
        self.assertIn('"weight_ne":[2048,1024,64,1]', s.structural_signature())
        self.assertIn('"fixed_output_axes":[8,1]', s.structural_signature())
        with self.assertRaises(InvalidModel):
            price(per_attention_pair_ns=1)
        # A matrix response has no KV mechanism, so unrelated KV is not a
        # spurious interpolation/extrapolation axis.
        estimate = price().estimate(action(kv_padded=16384))
        self.assertEqual(estimate.mean_ns, 5)
        self.assertFalse(estimate.interpolated or estimate.extrapolated)

    def test_dynamic_rows_interpolate_but_expert_and_weight_shapes_do_not(self):
        s = spec(matrix(q=2))
        self.assertEqual(s.structural_signature(), spec().structural_signature())
        self.assertTrue(price().estimate(action(s, query_tokens=2)).interpolated)
        with self.assertRaises(InvalidModel):
            action(s, query_tokens=1)
        with self.assertRaises(UnknownParameter):
            price().estimate(action(spec(matrix(experts=4))))
        changed = matrix(); changed["weight_ne"][0] = 4096
        with self.assertRaises(UnknownParameter):
            price().estimate(action(spec(changed)))

    def test_attention_uses_actual_pairs_and_checks_geometry(self):
        s = spec(attention())
        p = price(s, per_attention_pair_ns=Fraction(1, 2))
        self.assertEqual(p.estimate(action(s, query_tokens=3)).mean_ns, 393)
        for changes in (dict(query_tokens=2), dict(kv_padded=128)):
            with self.assertRaises(InvalidModel):
                action(s, query_tokens=3, **changes) if "query_tokens" not in changes else action(s, **changes)
        another = spec(attention(q=2, k=512))
        self.assertEqual(s.structural_signature(), another.structural_signature())
        with self.assertRaises(UnknownParameter):
            p.estimate(action(another, query_tokens=2, kv_padded=512))

    def test_attention_static_head_geometry_is_part_of_the_work_signature(self):
        p = price(spec(attention()), per_attention_pair_ns=1)
        for field, axis, value in (("ne", 2, 8), ("weight_ne", 0, 64)):
            node = attention(); node[field][axis] = value
            with self.subTest(field=field, axis=axis), self.assertRaises(UnknownParameter):
                p.estimate(action(spec(node), query_tokens=3))
        renamed = matrix(); renamed["name"] = "kq-0"
        self.assertEqual(spec(renamed).kind, "matrix")
        with self.assertRaises(InvalidModel):
            price(spec(renamed), per_attention_pair_ns=1)

    def test_pointwise_pruned_q_is_actual_local_axis(self):
        for node, axis in ((dict(name="ffn_moe_swiglu-15", op=95, ne=[1024, 8, 1, 1], type=0,
                            weight_ne=[1024, 8, 1, 1], weight_type=0), 2),
                           (dict(name="norm-15", op=25, ne=[2048, 1, 1, 1], type=0,
                            weight_ne=[2048, 1, 1, 1], weight_type=0), 1)):
            with self.subTest(node=node["name"]):
                s = spec(node)
                self.assertEqual(s.kind, "pointwise")
                self.assertEqual(s.operators[0].query_axis, axis)
                with self.assertRaises(InvalidModel):
                    action(s, query_tokens=125)
                with self.assertRaises(InvalidModel):
                    price(s, per_attention_pair_ns=1)
                self.assertEqual(price(s).estimate(action(s)).mean_ns, 5)

    def test_a_per_head_norm_prices_on_its_rows_not_its_own_prompt_length(self):
        def per_head(rows, heads=16):
            return dict(name="norm-0", op=25, ne=[128, heads, rows, 1], type=0,
                        weight_ne=[128, heads, rows, 1], weight_type=0)

        def layer(rows):
            return dict(name="norm-0", op=25, ne=[1024, rows, 1, 1], type=0,
                        weight_ne=[1024, rows, 1, 1], weight_type=0)

        # The heads occupy axis 1 and the prompt chunk occupies axis 2. The
        # query axis is the chunk; pinning axis 1 instead would put each
        # request's own length in the signature, so no other request could be
        # priced from it.
        node = per_head(126)
        work = spec(node)
        self.assertEqual(work.operators[0].query_axis, 2)
        self.assertEqual(work.query_basis, "actual_pointwise_rows")
        self.assertEqual(json.loads(work.structural_signature())["operators"][0]
                         ["fixed_output_axes"], [128, 16, 1])
        self.assertEqual(work.structural_signature(), spec(per_head(245)).structural_signature())
        self.assertNotEqual(work.structural_signature(), spec(per_head(126, heads=8)).structural_signature())
        with self.assertRaises(InvalidModel):
            work.validate_coordinates(16, 256, 0)

        # An ordinary layer norm and a one-token per-head norm keep the row
        # axis they had before this rule: previously frozen keys must
        # reproduce byte for byte.
        self.assertEqual(json.loads(spec(layer(126)).structural_signature())["operators"][0]
                         ["fixed_output_axes"], [1024, 1, 1])
        self.assertEqual(spec(layer(126)).operators[0].query_axis, 1)
        self.assertEqual(json.loads(spec(per_head(1)).structural_signature())["operators"][0]
                         ["fixed_output_axes"], [128, 1, 1])
        self.assertEqual(spec(per_head(1)).operators[0].query_axis, 1)

    def test_unknown_host_is_not_declared_shape_independent(self):
        s = WindowWorkSpec.from_metadata("HOST", copy_calls=0)
        self.assertEqual(s.kind, "host_unclassified")
        self.assertTrue(s.unknown_mechanisms)
        self.assertEqual(s.allowed_features, ("intercept",))
        self.assertEqual(s.domain_axes, ("query_tokens", "kv_padded"))
        with self.assertRaises(InvalidModel):
            price(s)
        p = price(s, per_query_token_ns=0)
        with self.assertRaises(UnknownParameter):
            p.estimate(action(s, kv_padded=1024))

    def test_mask_preparation_needs_evidence_not_just_host_scope(self):
        with self.assertRaises(InvalidModel):
            WindowWorkSpec.from_metadata("HOST", copy_calls=0, host_stage="attention_mask_prepare")
        s = WindowWorkSpec.from_metadata("HOST", copy_calls=0,
            host_stage="attention_mask_prepare", host_stage_evidence=("actual-stage-event#1",))
        p = price(s, per_query_token_ns=0, per_attention_pair_ns=1)
        self.assertEqual(p.estimate(action(s)).mean_ns, 259)
        with self.assertRaises(InvalidModel):
            price(s, per_payload_byte_ns=1)

    def test_host_and_bridge_transfers_require_observed_copy(self):
        host = WindowWorkSpec.from_metadata("HOST", copy_calls=2)
        self.assertIn("payload_bytes", host.allowed_features)
        p = price(host, per_query_token_ns=0, per_payload_byte_ns=2, observed_payload_bytes=(64,))
        self.assertEqual(p.estimate(action(host, payload_bytes=64)).mean_ns, 131)
        plain = WindowWorkSpec.from_metadata("HOST", copy_calls=0)
        with self.assertRaises(InvalidModel):
            action(plain, payload_bytes=64)
        with self.assertRaises(InvalidModel):
            price(plain, per_query_token_ns=0, per_payload_byte_ns=1)
        with self.assertRaises(InvalidModel):
            WindowWorkSpec.from_metadata("SAMPLER", copy_calls=1)

    def test_bridge_attention_and_matrix_only_permissions_are_distinct(self):
        a = WindowWorkSpec.from_metadata("RUNTIME_BRIDGE", copy_calls=1,
            gpu_nodes=[matrix(q=3), attention()])
        self.assertIn("query_tokens_times_KV", a.allowed_features)
        self.assertIn("payload_bytes", a.allowed_features)
        action(a, query_tokens=3, payload_bytes=64)
        p = price(a, per_attention_pair_ns=1, per_payload_byte_ns=2, observed_payload_bytes=(64,))
        self.assertEqual(p.estimate(action(a, query_tokens=3, payload_bytes=64)).mean_ns, 905)
        matrix_only = WindowWorkSpec.from_metadata("RUNTIME_BRIDGE", copy_calls=1, gpu_nodes=[matrix()])
        with self.assertRaises(InvalidModel):
            price(matrix_only, per_attention_pair_ns=1)
        with self.assertRaises(InvalidModel):
            action(matrix_only, query_tokens=3)

    def test_mixed_rows_without_attention_do_not_fake_a_single_query(self):
        a = WindowWorkSpec.from_metadata("RUNTIME_BRIDGE", copy_calls=0,
            gpu_nodes=[matrix(q=1), dict(matrix(q=128), name="ffn_moe_up-1")])
        self.assertEqual(a.query_basis, "unidentified")
        self.assertNotIn("query_tokens", a.allowed_features)
        self.assertTrue(a.unknown_mechanisms)
        with self.assertRaises(InvalidModel):
            price(a)

    def test_real_pruned_matrix_nodes_retain_coverage_without_fake_query_work(self):
        node = matrix(q=0)
        s = WindowWorkSpec.from_metadata("RUNTIME_BRIDGE", copy_calls=0, gpu_nodes=[node])
        self.assertEqual(s.operators[0].role, "pruned_zero_output")
        self.assertIsNone(s.operators[0].query_axis)
        self.assertEqual(s.allowed_features, ("intercept",))
        self.assertIn('"zero_output_shape":[1024,8,0,1]', s.structural_signature())
        self.assertNotEqual(s.structural_signature(),
            WindowWorkSpec.from_metadata("RUNTIME_BRIDGE", copy_calls=0, gpu_nodes=[matrix()]).structural_signature())
        p = price(s, per_query_token_ns=0)
        a = action(s)
        compiled = compile_runtime_windows((a,), (p,))
        self.assertEqual(compiled.evaluate().duration_ns, 3)
        self.assertEqual(compiled.estimates[0].action.covered_nodes, ("actual-node0",))
        with self.assertRaises(InvalidModel):
            spec(node)
        bad = dict(node, weight_ne=[0, 0, 0, 0])
        with self.assertRaises(InvalidModel):
            WindowWorkSpec.from_metadata("RUNTIME_BRIDGE", copy_calls=0, gpu_nodes=[bad])

    def test_real_pruned_pointwise_nodes_do_not_change_live_bridge_query_basis(self):
        norm = dict(name="norm-15", op=25, ne=[2048, 0, 1, 1], type=0,
                    weight_ne=[2048, 0, 1, 1], weight_type=0)
        swiglu = dict(name="ffn_moe_swiglu-15", op=95, ne=[1024, 8, 0, 1], type=0,
                      weight_ne=[1024, 8, 0, 1], weight_type=0)
        s = WindowWorkSpec.from_metadata("RUNTIME_BRIDGE", copy_calls=0,
            gpu_nodes=[matrix(q=3), norm, swiglu])
        self.assertEqual([n.role for n in s.operators], ["matrix", "pruned_zero_output", "pruned_zero_output"])
        self.assertEqual(s.query_basis, "actual_matrix_rows")
        s.validate_coordinates(3, 256, 0)
        self.assertNotIn("query_tokens_times_KV", s.allowed_features)

    def test_json_roundtrip_and_forged_schema_are_rejected(self):
        s = spec()
        self.assertEqual(WindowWorkSpec.from_dict(json.loads(json.dumps(s.to_dict()))), s)
        self.assertEqual(RuntimeWindowAction.from_dict(json.loads(json.dumps(action().to_dict()))), action())
        self.assertEqual(RuntimeWindowPrice.from_dict(json.loads(json.dumps(price().to_dict()))), price())
        for changes in (dict(schema="unknown"), dict(kind="attention_core"),
                        dict(allowed_features=["intercept", "query_tokens_times_KV"]), dict(extra="ignored")):
            value = dict(s.to_dict(), **changes)
            with self.subTest(changes=changes), self.assertRaises(InvalidModel):
                WindowWorkSpec.from_dict(value)
        for bad in (True, -1, 0.5):
            with self.subTest(copy_calls=bad), self.assertRaises(InvalidModel):
                WindowWorkSpec.from_metadata("HOST", copy_calls=bad)
        missing = matrix(); missing.pop("weight_ne")
        with self.assertRaises(InvalidModel):
            spec(missing)
        empty = matrix(); empty["weight_ne"] = [0, 0, 0, 0]
        with self.assertRaises(InvalidModel):
            spec(empty)

    def test_bridge_dependency_needs_an_actual_producer_transfer_and_evidence(self):
        # The permission is for a transfer that carries the output of a host
        # attention stage, so a bare flag, a missing copy, or another scope
        # must all be refused before any coefficient exists.
        bridge = dict(scope="RUNTIME_BRIDGE", gpu_nodes=[matrix()])
        with self.assertRaises(InvalidModel):
            WindowWorkSpec.from_metadata(bridge["scope"], copy_calls=1, gpu_nodes=bridge["gpu_nodes"],
                dependency="cpu_attention_stage")
        with self.assertRaises(InvalidModel):
            WindowWorkSpec.from_metadata(bridge["scope"], copy_calls=0, gpu_nodes=bridge["gpu_nodes"],
                dependency="cpu_attention_stage", dependency_evidence=("actual_producer:kqv-0",))
        with self.assertRaises(InvalidModel):
            WindowWorkSpec.from_metadata(bridge["scope"], copy_calls=1, gpu_nodes=bridge["gpu_nodes"],
                dependency="unknown_stage", dependency_evidence=("actual_producer:kqv-0",))
        with self.assertRaises(InvalidModel):
            WindowWorkSpec.from_metadata(bridge["scope"], copy_calls=1, gpu_nodes=bridge["gpu_nodes"],
                dependency_evidence=("actual_producer:kqv-0",))
        with self.assertRaises(InvalidModel):
            WindowWorkSpec.from_metadata("HOST", copy_calls=1, dependency="cpu_attention_stage",
                dependency_evidence=("actual_producer:kqv-0",))

    def test_bridge_dependency_permits_attention_work_it_is_serialized_behind(self):
        s = WindowWorkSpec.from_metadata("RUNTIME_BRIDGE", copy_calls=1, gpu_nodes=[matrix(q=3)],
            dependency=CPU_ATTENTION_STAGE, dependency_evidence=("actual_producer:node_32",))
        self.assertEqual(s.kind, "mixed_bridge")
        self.assertIn("query_tokens_times_KV", s.allowed_features)
        self.assertIn("kv_padded", s.domain_axes)
        self.assertIn("dependency", s.structural_signature())
        self.assertTrue(any("serialized behind" in m for m in s.unknown_mechanisms))
        p = price(s, per_query_token_ns=0, per_attention_pair_ns=1, observed_payload_bytes=(64,))
        self.assertEqual(p.estimate(action(s, query_tokens=3, kv_padded=256, payload_bytes=64)).mean_ns, 771)
        # Without the declaration the same bridge has no KV mechanism at all,
        # and neither signature may collide with the declared one.
        plain = WindowWorkSpec.from_metadata("RUNTIME_BRIDGE", copy_calls=1, gpu_nodes=[matrix(q=3)])
        self.assertNotIn("query_tokens_times_KV", plain.allowed_features)
        self.assertNotEqual(plain.structural_signature(), s.structural_signature())
        self.assertNotIn('"dependency"', plain.structural_signature())
        self.assertEqual(WindowWorkSpec.from_dict(json.loads(json.dumps(s.to_dict()))), s)
        # Previously frozen mechanism_v1 declarations have no dependency keys
        # and must keep parsing to exactly the same structure.
        frozen = plain.to_dict(); frozen.pop("dependency"); frozen.pop("dependency_evidence")
        self.assertEqual(WindowWorkSpec.from_dict(frozen), plain)

    def test_legacy_affine_is_preserved_and_explicitly_labeled(self):
        legacy = price(work_spec=None, feature_policy=LEGACY_POLICY, per_attention_pair_ns=1)
        old = legacy.to_dict(); old.pop("feature_policy"); old.pop("work_spec")
        restored = RuntimeWindowPrice.from_dict(old)
        self.assertEqual(restored.feature_policy, LEGACY_POLICY)
        self.assertEqual(restored.estimate(action()).mean_ns, 261)
        compiled = compile_runtime_windows((action(),), (restored,))
        self.assertEqual(compiled.to_dict()["legacy_response_windows"], 1)
        self.assertEqual(compiled.to_dict()["windows"][0]["feature_policy"], LEGACY_POLICY)
        self.assertFalse(compiled.to_dict()["windows"][0]["mechanism_constraints_applied"])
        with self.assertRaises(InvalidModel):
            price(feature_policy="unknown")
        with self.assertRaises(InvalidModel):
            price(work_spec=None)


if __name__ == "__main__":
    unittest.main()
