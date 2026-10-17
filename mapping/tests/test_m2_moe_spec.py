"""MoE 0.6 input/state gates; all geometry and costs are explicitly synthetic."""
from __future__ import annotations

import copy
import dataclasses
import unittest

from llm_infer_model.m2.spec import (
    FORMAT_VERSION, MOE_FORMAT_VERSION, STATE_CACHE_FORMAT_VERSION,
    StateIn, StateOut, WeightStateRef, parse_layer_document,
    parse_invocation_document, parse_state_document, parse_state_in,
    state_in_from_state_out, to_state_document, to_state_in_document,
    validate_invocation,
)
from llm_infer_model.tensor.spec import InvalidInput, Unsupported


def moe_documents():
    weights = [
        {"id": "projection_w", "storage_bytes": 32, "initial_location": "dram"},
        {"id": "router_w", "storage_bytes": 16, "initial_location": "dram"},
        *({"id": f"expert_{i}", "storage_bytes": 16, "initial_location": "ssd"} for i in range(4)),
        {"id": "shared_w", "storage_bytes": 8, "initial_location": "ssd"},
    ]
    layer = {
        "format_version": MOE_FORMAT_VERSION, "id": "moe", "provenance": "synthetic_layer",
        "hidden_size": 4, "hidden_bytes_per_token": 16, "kv_bytes_per_token": 8,
        "attention": "full_causal", "attention_params": {}, "ffn": "moe",
        "ffn_params": {"experts": 4, "top_k": 2, "expert_weight_bytes": 16,
                       "expert_weights": [f"expert_{i}" for i in range(4)],
                       "shared_expert_bytes": 8, "shared_expert_weight": "shared_w"},
        "weights": weights,
        "tasks": [
            {"id": "proj", "kind": "projection", "depends_on": [], "weight": "projection_w"},
            {"id": "attn", "kind": "attention", "depends_on": ["proj"]},
            {"id": "route", "kind": "router", "depends_on": ["attn"], "weight": "router_w"},
            {"id": "expert", "kind": "expert_ffn", "depends_on": ["route"]},
            {"id": "combine", "kind": "combine", "depends_on": ["expert"]},
        ],
        "workspace_bytes": 0,
    }
    invocation = {
        "format_version": MOE_FORMAT_VERSION, "id": "prefill", "layer": "moe", "phase": "prefill",
        "hardware": {"dram_capacity_bytes": 4096, "vram_capacity_bytes": 4096},
        "requests": [
            {"request_id": "r0", "position_start": 0, "tokens": 2, "input_location": "dram", "output_location": "dram"},
            {"request_id": "r1", "position_start": 0, "tokens": 1, "input_location": "dram", "output_location": "dram"},
        ],
        "state_in": {"requests": [
            {"request_id": "r0", "next_position": 0, "blocks": [], "state_location": "dram"},
            {"request_id": "r1", "next_position": 0, "blocks": [], "state_location": "dram"},
        ], "weight_copies": [
            {"id": w["id"], "version": 3, "storage_bytes": w["storage_bytes"], "locations": [w["initial_location"]]}
            for w in weights
        ]},
        "devices": ["cpu", "gpu"], "costs": {},
        "route_source": "test:declared-routing-trace", "information_mode": "causal_replay",
        "route_trace": [
            {"request_id": "r0", "layer_id": "moe", "absolute_position": 0, "expert_ids": [0, 2], "scores": [0.6, 0.4]},
            {"request_id": "r0", "layer_id": "moe", "absolute_position": 1, "expert_ids": [1, 2]},
            {"request_id": "r1", "layer_id": "moe", "absolute_position": 0, "expert_ids": [0, 3]},
        ],
    }
    return layer, invocation


class MoESpecTests(unittest.TestCase):
    def accepted(self, mutate=None):
        layer_raw, invocation_raw = moe_documents()
        if mutate:
            mutate(layer_raw, invocation_raw)
        layer, invocation = parse_layer_document(layer_raw), parse_invocation_document(invocation_raw)
        validate_invocation(layer, invocation)
        return layer, invocation

    def refused(self, mutate, needle, error=InvalidInput):
        with self.assertRaises(error) as caught:
            self.accepted(mutate)
        self.assertIn(needle, str(caught.exception))
        self.accepted()

    def test_ragged_prefill_batch_and_top_k_trace_are_explicit(self):
        layer, invocation = self.accepted()
        self.assertEqual((invocation.batch, [r.tokens for r in invocation.requests]), (2, [2, 1]))
        self.assertEqual(invocation.route_trace[0].expert_ids, (0, 2))
        self.assertEqual(invocation.route_trace[0].scores, (0.6, 0.4))
        self.assertEqual(layer.ffn_params["expert_weights"], [f"expert_{i}" for i in range(4)])
        self.assertEqual(len(invocation.state_in.weight_copies), len(layer.weights))

    def test_initial_cache_source_is_bound_to_the_explicit_invocation_layer(self):
        layer, invocation = self.accepted()
        self.assertEqual(invocation.state_in.layer, layer.id)

    def test_cross_layer_weight_snapshot_is_refused_even_with_matching_weight_ids(self):
        self.refused(lambda l, i: i["state_in"].update(layer="different_layer"), "belongs to layer")

    def test_internal_expert_batch_namespace_is_not_a_request_id(self):
        def mutate(layer, invocation):
            invocation["requests"][0]["request_id"] = "__moe_batch__"
            invocation["state_in"]["requests"][0]["request_id"] = "__moe_batch__"
            for route in invocation["route_trace"]:
                if route["request_id"] == "r0":
                    route["request_id"] = "__moe_batch__"
        self.refused(mutate, "reserved for internal expert groups")

    def test_oracle_mode_is_separately_declared(self):
        _, invocation = self.accepted(lambda l, i: i.update(information_mode="offline_oracle"))
        self.assertEqual(invocation.information_mode, "offline_oracle")

    def test_moe_provenance_must_explicitly_be_synthetic(self):
        for value in (None, "measured_macro"):
            with self.subTest(value=value):
                def mutate(layer, invocation):
                    if value is None:
                        del layer["provenance"]
                    else:
                        layer["provenance"] = value
                self.refused(mutate, "synthetic_layer")

    def test_non_moe_ffn_and_attention_are_refused(self):
        self.refused(lambda l, i: l.update(attention="sliding_window", attention_params={"window": 4}), "full_causal + moe", Unsupported)
        self.refused(lambda l, i: l.update(ffn="dense", ffn_params={}), "full_causal + moe", Unsupported)

    def test_all_byte_vectors_must_have_whole_f32_elements(self):
        for field in ("hidden_bytes_per_token", "kv_bytes_per_token"):
            with self.subTest(field=field):
                self.refused(lambda l, i: l.update({field: 15}), "multiple of 4")
        self.refused(lambda l, i: l["weights"][0].update(storage_bytes=31), "multiple of 4")
        self.refused(lambda l, i: l["ffn_params"].update(expert_weight_bytes=15), "multiple of 4")

    def test_hidden_vector_bytes_must_match_declared_hidden_size(self):
        self.refused(lambda l, i: l.update(hidden_bytes_per_token=32), "equal hidden_size * 4")

    def test_weight_initial_location_must_be_explicit(self):
        self.refused(lambda l, i: l["weights"][2].pop("initial_location"), "initial_location must be explicit")

    def test_initial_ssd_and_current_multi_copy_locations_are_independent(self):
        _, invocation = self.accepted(lambda l, i: i["state_in"]["weight_copies"][2].update(locations=["ssd", "vram", "dram"]))
        self.assertEqual(invocation.state_in.weight_copies[2].locations, ("ssd", "vram", "dram"))

    def test_weight_snapshot_is_never_inferred(self):
        self.refused(lambda l, i: i["state_in"].pop("weight_copies"), "must be explicit")
        self.refused(lambda l, i: i["state_in"].update(weight_copies=[]), "exactly cover")

    def test_snapshot_unknown_missing_or_duplicate_weight_is_refused(self):
        self.refused(lambda l, i: i["state_in"]["weight_copies"][0].update(id="wrong"), "exactly cover")
        self.refused(lambda l, i: i["state_in"]["weight_copies"].pop(), "exactly cover")
        self.refused(lambda l, i: i["state_in"]["weight_copies"].append(copy.deepcopy(i["state_in"]["weight_copies"][0])), "duplicate weight")

    def test_weight_snapshot_bytes_must_match_inventory(self):
        self.refused(lambda l, i: i["state_in"]["weight_copies"][2].update(storage_bytes=32), "differs from layer inventory")

    def test_snapshot_version_is_a_positive_integer(self):
        for bad in (0, -1, True, 1.5):
            with self.subTest(version=bad):
                self.refused(lambda l, i: i["state_in"]["weight_copies"][0].update(version=bad), "version")

    def test_snapshot_locations_are_required_unique_and_known(self):
        for locations in ([], ["ssd", "ssd"], ["unknown"]):
            with self.subTest(locations=locations):
                self.refused(lambda l, i: i["state_in"]["weight_copies"][0].update(locations=locations), "locations")

    def test_expert_binding_ids_are_exact_distinct_and_known(self):
        self.refused(lambda l, i: l["ffn_params"].pop("expert_weights"), "expert_weights")
        self.refused(lambda l, i: l["ffn_params"].update(expert_weights=["expert_0"]), "one weight ID per expert")
        self.refused(lambda l, i: l["ffn_params"].update(expert_weights=["expert_0"] * 4), "must not repeat")
        self.refused(lambda l, i: l["ffn_params"].update(expert_weights=["expert_0", "expert_1", "expert_2", "unknown"]), "没有权重")

    def test_expert_bytes_are_checked_against_each_inventory_record(self):
        self.refused(lambda l, i: l["weights"][2].update(storage_bytes=32), "equal expert_weight_bytes")

    def test_shared_expert_requires_both_explicit_id_and_bytes(self):
        self.refused(lambda l, i: l["ffn_params"].pop("shared_expert_weight"), "declared together")
        self.refused(lambda l, i: l["ffn_params"].pop("shared_expert_bytes"), "declared together")
        self.refused(lambda l, i: l["ffn_params"].update(shared_expert_weight="expert_0"), "independent weight ID")
        self.refused(lambda l, i: l["ffn_params"].update(shared_expert_bytes=16), "equal shared_expert_bytes")

    def test_layer_without_shared_expert_is_valid(self):
        def mutate(layer, invocation):
            layer["ffn_params"].pop("shared_expert_weight")
            layer["ffn_params"].pop("shared_expert_bytes")
            layer["weights"] = [w for w in layer["weights"] if w["id"] != "shared_w"]
            invocation["state_in"]["weight_copies"] = [w for w in invocation["state_in"]["weight_copies"] if w["id"] != "shared_w"]
        self.accepted(mutate)

    def test_optional_projection_router_weights_are_explicit_if_present(self):
        def mutate(layer, invocation):
            for task in layer["tasks"]:
                task.pop("weight", None)
            keep = {f"expert_{i}" for i in range(4)} | {"shared_w"}
            layer["weights"] = [w for w in layer["weights"] if w["id"] in keep]
            invocation["state_in"]["weight_copies"] = [w for w in invocation["state_in"]["weight_copies"] if w["id"] in keep]
        self.accepted(mutate)

    def test_chain_requires_each_of_five_classes_once_and_exact_edges(self):
        self.refused(lambda l, i: l["tasks"][2].update(kind="attention"), "exactly projection")
        self.refused(lambda l, i: l["tasks"][2].update(depends_on=["proj"]), "five-stage chain")
        self.refused(lambda l, i: l["tasks"][0].update(depends_on=["combine"]), "有环")

    def test_expert_template_cannot_bind_one_weight(self):
        self.refused(lambda l, i: l["tasks"][3].update(weight="expert_0"), "must not bind a single weight")

    def test_layer_inventory_does_not_hide_unused_or_reused_weights(self):
        self.refused(lambda l, i: l["weights"].append({"id": "unused", "storage_bytes": 4, "initial_location": "ssd"}), "undeclared or unused")
        self.refused(lambda l, i: l["tasks"][2].update(weight="projection_w"), "independent from expert")

    def test_router_source_and_information_mode_are_required(self):
        for key in ("route_source", "information_mode", "route_trace"):
            with self.subTest(key=key):
                self.refused(lambda l, i: i.pop(key), key)
        self.refused(lambda l, i: i.update(route_source=""), "route_source")
        self.refused(lambda l, i: i.update(route_source="  "), "must not be blank")
        self.refused(lambda l, i: i.update(information_mode="guess"), "information_mode")

    def test_trace_must_exactly_cover_the_current_call(self):
        self.refused(lambda l, i: i["route_trace"].pop(), "exactly cover")
        self.refused(lambda l, i: i["route_trace"].append(copy.deepcopy(i["route_trace"][0])), "duplicates token")
        for key, value in (("request_id", "wrong"), ("layer_id", "wrong"), ("absolute_position", 99)):
            with self.subTest(key=key):
                self.refused(lambda l, i: i["route_trace"][0].update({key: value}), "outside this invocation")

    def test_programmatic_route_position_requires_a_nonnegative_integer(self):
        layer, invocation = self.accepted()
        for bad in (False, True, 0.0, 1.0, -1):
            with self.subTest(position=bad):
                route = dataclasses.replace(invocation.route_trace[0], absolute_position=bad)
                changed = dataclasses.replace(invocation, route_trace=(route, *invocation.route_trace[1:]))
                with self.assertRaisesRegex(InvalidInput, "absolute_position"):
                    validate_invocation(layer, changed)

    def test_trace_experts_must_match_top_k_and_be_in_range(self):
        for ids in ([0], [0, 0], [0, 4], [0, -1], [0, True]):
            with self.subTest(ids=ids):
                self.refused(lambda l, i: i["route_trace"][0].update(expert_ids=ids, scores=[0.5] * len(ids)), "expert")

    def test_route_scores_are_optional_finite_metadata_with_matching_length(self):
        for scores in ([0.5], [float("nan"), 0.5], [float("inf"), 0.5], [True, 0.5], ["0.5", 0.5]):
            with self.subTest(scores=scores):
                self.refused(lambda l, i: i["route_trace"][0].update(scores=scores), "scores")
        self.accepted(lambda l, i: i["route_trace"][0].update(scores=[-2.0, 4.0]))

    def test_absolute_route_position_tracks_continuous_kv_history(self):
        def mutate(layer, invocation):
            invocation["phase"] = "decode"
            for request, state in zip(invocation["requests"], invocation["state_in"]["requests"]):
                request.update(position_start=3, tokens=1)
                state.update(next_position=3, blocks=[{"id": f"kv:moe:{request['request_id']}:0", "first_position": 0, "tokens": 3,
                                                       "storage_bytes": 24, "locations": ["ssd"]}])
            invocation["route_trace"] = [{"request_id": r["request_id"], "layer_id": "moe", "absolute_position": 3, "expert_ids": [1, 2]}
                                          for r in invocation["requests"]]
        _, invocation = self.accepted(mutate)
        self.assertEqual([r.absolute_position for r in invocation.route_trace], [3, 3])

    def test_history_position_bytes_continuity_and_request_coverage_are_checked(self):
        self.refused(lambda l, i: i["requests"][0].update(position_start=1), "differs from next_position")
        self.refused(lambda l, i: i["state_in"]["requests"].pop(), "exactly cover invocation requests")
        def corrupt_history(layer, invocation):
            invocation["state_in"]["requests"][0].update(next_position=1, blocks=[{"id": "gap", "first_position": 1, "tokens": 1, "storage_bytes": 8, "locations": ["dram"]}])
            invocation["requests"][0].update(position_start=1)
        self.refused(corrupt_history, "不连续")

    def test_decode_refuses_more_than_one_token(self):
        self.refused(lambda l, i: i.update(phase="decode"), "q_i=1")

    def test_ssd_input_output_and_state_location_are_explicitly_supported_for_moe(self):
        _, invocation = self.accepted(lambda l, i: (
            i["requests"][0].update(input_location="ssd", output_location="ssd"),
            i["state_in"]["requests"][0].update(state_location="ssd")))
        self.assertEqual(invocation.requests[0].output_location, "ssd")
        for target, key in (("requests", "input_location"), ("requests", "output_location")):
            self.refused(lambda l, i: i[target][0].pop(key), "must be explicit")
        self.refused(lambda l, i: i["state_in"]["requests"][0].pop("state_location"), "must be explicit")

    def test_unknown_trace_fields_are_not_silently_discarded(self):
        self.refused(lambda l, i: i["route_trace"][0].update(predicted_experts=[1, 2]), "未知字段")

    def test_route_fields_cannot_leak_into_old_boundaries(self):
        _, invocation = moe_documents()
        invocation["format_version"] = FORMAT_VERSION
        with self.assertRaisesRegex(InvalidInput, "route declarations require"):
            parse_invocation_document(invocation)


class MoEStateTests(unittest.TestCase):
    def sample_state_out(self):
        raw_layer, raw_invocation = moe_documents()
        history = parse_invocation_document(raw_invocation).state_in
        return StateOut(layer="moe", version=7, requests=history.requests,
                        weight_copies=history.weight_copies)

    def test_state_out_to_state_in_preserves_all_versions_bytes_and_copies(self):
        state = self.sample_state_out()
        successor = state_in_from_state_out(state)
        self.assertEqual(successor.weight_copies, state.weight_copies)
        self.assertEqual(successor.layer, state.layer)
        self.assertEqual(successor.weight_copies[0].version, 3)
        embedded = to_state_in_document(successor, layer="moe")
        self.assertEqual(embedded["layer"], state.layer)
        self.assertEqual(parse_state_in(embedded), successor)

    def test_new_standalone_state_document_round_trips_cache_in_both_directions(self):
        state = self.sample_state_out()
        document = to_state_document(state)
        self.assertEqual(document["format_version"], STATE_CACHE_FORMAT_VERSION)
        loaded = parse_state_document(document)
        self.assertEqual(loaded.as_state_out(), state)
        self.assertEqual(loaded.as_state_in(), state_in_from_state_out(state))
        self.assertEqual(to_state_document(loaded.as_state_out()), document)

    def test_cache_snapshot_cannot_be_relabelled_during_serialization(self):
        state = state_in_from_state_out(self.sample_state_out())
        with self.assertRaisesRegex(InvalidInput, "cannot relabel"):
            to_state_document(state, layer="different_layer")
        self.assertEqual(to_state_document(state)["layer"], "moe")

    def test_new_state_document_preserves_ssd_state_goal(self):
        state = self.sample_state_out()
        state = dataclasses.replace(state, requests=(dataclasses.replace(state.requests[0], state_location="ssd"),))
        self.assertEqual(parse_state_document(to_state_document(state)).as_state_out(), state)

    def test_legacy_standalone_state_with_cache_field_is_refused(self):
        document = to_state_document(self.sample_state_out())
        document["format_version"] = FORMAT_VERSION
        with self.assertRaisesRegex(InvalidInput, "legacy state must not drop"):
            parse_state_document(document)

    def test_new_standalone_state_requires_explicit_cache_field(self):
        document = to_state_document(self.sample_state_out())
        document.pop("weight_copies")
        with self.assertRaisesRegex(InvalidInput, "must be explicit"):
            parse_state_document(document)

    def test_cache_free_legacy_state_serialization_is_unchanged(self):
        old = StateIn()
        document = to_state_document(old, layer="legacy")
        self.assertEqual(document["format_version"], FORMAT_VERSION)
        self.assertNotIn("weight_copies", document)
        self.assertEqual(to_state_in_document(old, layer="legacy"), {"requests": []})
        self.assertEqual(parse_state_document(document).as_state_in(), old)

    def test_legacy_execution_cannot_discard_programmatic_cache_snapshot(self):
        from tests.test_m2 import M2Case
        compiled = M2Case().compile()
        with_cache = dataclasses.replace(compiled.state_in, weight_copies=(WeightStateRef("x", 1, 4, ("dram",)),))
        with self.assertRaisesRegex(Unsupported, "must not discard"):
            validate_invocation(compiled.layer, compiled.invocation, with_cache)


if __name__ == "__main__":
    unittest.main()
