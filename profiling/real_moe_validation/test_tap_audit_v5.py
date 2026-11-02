"""Small synthetic v5 audit fixtures; no model/device/runtime execution."""
import copy
import unittest

from .postcollection import CONTRACT, expected_forwards
from .tap_audit_v5 import normal_equivalence, pair_audit


def fixture(prompt=7, outputs=2, backend="CUDA0", stages=("up", "gate", "down")):
    tokens, emitted = list(range(prompt)), [17] * outputs
    forwards = expected_forwards(prompt, outputs)
    graphs, routes, copies, decisions = [], [], [], []
    offset = 0
    for forward in forwards:
        experts = []
        for layer in range(16):
            positions = list(range(forward["position"], forward["position"] + forward["tokens"])) if layer < 15 else (
                [forward["position"] + forward["tokens"] - 1] if forward["output_rows"] else [])
            if not positions:
                continue
            for stage in stages:
                experts.append(dict(name=f"ffn_moe_{stage}-{layer}",backend=backend,weight_type="q4_K",weight_buffer="CPU_Mapped"))
                if backend == "CUDA0":
                    size = len(positions) * 8 * 4
                    copies.append(dict(forward_index=forward["index"],layer=layer,stage=("up", "gate", "down").index(stage),rows=len(positions),slots=8,
                                       source_nb0=4,source_nb1=256,pinned_destination_offset_bytes=offset,bytes=size,
                                       actual_consumer_stream_opaque_id=123,same_stream_before_consumer=True))
                    offset += size
            route = dict(forward_index=forward["index"],layer=layer,phase=forward["phase"],absolute_positions=positions,
                         expert_ids=[list(range(8)) for _ in positions],consumer_backend=backend,actual_stages_observed=list(stages),
                         all_three_stages_observed=set(stages) == {"up", "gate", "down"},observed_stage_routes_equal=True,stage_count=len(stages))
            if backend == "CUDA0":
                route["observation_transport"] = "same_stream_async_D2H_before_consumer"
            routes.append(route)
        graphs.append(dict(forward_index=forward["index"],node_count=200,scheduler_splits=3,scheduler_copies=1,
                           graph_partition_fingerprint_sha256="d" * 64,expert_nodes=experts))
        if backend == "CUDA0":
            decisions.append(dict(forward_index=forward["index"],partition_nodes=200,expert_nodes=len(experts),USE_CUDA_GRAPH_compiled=True,
                                  use_cuda_graph=False,cuda_graph_update_required=False,execution="direct",device_completion_observed_here=False))
    logits = [dict(output_index=index,full_f32_logits_sha256="e" * 64,top_logits=[dict(token=17,logit=2.0),dict(token=18,logit=1.0)],
                   top1_top2_margin=1.0,greedy_token=17,vocabulary_size=100) for index in range(outputs)]
    row = dict(id="sample",prompt_token_ids=tokens,output_token_ids=emitted,sampled_greedy_token_ids=emitted.copy(),teacher_forced=True,
               forward_inventory=forwards,forward_count=len(forwards),decode_forward_count=outputs - 1,logit_observations=logits,graph_observations=graphs,
               routes=routes,gpu_routes=copy.deepcopy(routes) if backend == "CUDA0" else [],gpu_copy_inventory=copies,
               gpu_capture_error_code=0,gpu_capture_status="observed",gpu_observer_copy_count=len(copies),gpu_pinned_capacity_bytes=offset,
               cuda_graph_decisions=decisions,observer_extra_flush_ns=0,observer_completion_boundary="existing llama_synchronize only",
               observer=dict(cpu_consumer_tap=True,step_logit_copy=True,graph_metadata=True),timing_usable_as_bare_reference=False,
               trace_timing_is_performance_truth=False)
    request = dict(schema="real-moe-tap-request-5",source_sha256="b" * 64,model_sha256="c" * 64,mode="capture",model_contract=CONTRACT,
                   batch_size=256,context_size=2048,threads=16,expected_process_mask=4294967295,placement="cpu-experts-gpu-other",
                   observe_logits=True,observe_graph=True,cuda_graph_policy="runtime_default_enabled",
                   requests=[dict(id="sample",prompt_token_ids=tokens,output_token_ids=emitted,max_output_tokens=outputs)])
    capture = dict(schema="real-moe-tap-native-5",source_sha256="b" * 64,model_sha256_verified_by_parent="c" * 64,request=request,mode="capture",
                   results=[row],scheduler_callback_enabled=False,trace_callback_enabled=False,model_weights_loaded=True,observed_model_contract=CONTRACT,
                   threadpool_policy="explicit_persistent_owned",graph_or_allocator_modified=False,cpu_consumer_tap_enabled=True,
                   tap_hook_available=True,gpu_tap_hook_available=True,actual_placement=dict(buffers={"CPU_Mapped":100}),
                   observed_runtime_flags=dict(NVIDIA_TF32_OVERRIDE="0",GGML_CUDA_DISABLE_GRAPHS=None))
    normal = copy.deepcopy(capture)
    normal["mode"] = normal["request"]["mode"] = "generate"
    normal["cpu_consumer_tap_enabled"] = False
    normal["request"]["requests"][0].pop("output_token_ids")
    normal["results"][0].update(teacher_forced=False,routes=[],gpu_routes=[],gpu_copy_inventory=[],gpu_observer_copy_count=0,gpu_pinned_capacity_bytes=0)
    normal["results"][0]["observer"]["cpu_consumer_tap"] = False
    return normal, capture


def original_normal(private):
    original = copy.deepcopy(private)
    original["schema"] = "real-moe-tap-native-4"
    original["request"]["schema"] = "real-moe-tap-request-4"
    original["source_sha256"] = original["request"]["source_sha256"] = "a" * 64
    for row in original["results"]:
        for key in list(row):
            if key.startswith("gpu_") or key in ("cuda_graph_decisions", "observer_extra_flush_ns", "observer_completion_boundary"):
                row.pop(key)
    return original


class V5ConsumerAudit(unittest.TestCase):
    def test_direct_GPU_pair_passes_exact_copy_receipts(self):
        result = pair_audit(*fixture())
        self.assertTrue(result["paired"], result)
        self.assertEqual(result["totals"]["greedy_token_matches"], 2)
        self.assertFalse(result["hidden_bare_routes_proven_identical"])
        self.assertFalse(result["whole_model_accuracy_certified"])

    def test_CPU_pair_uses_actual_CPU_routes_without_GPU_fabrication(self):
        result = pair_audit(*fixture(backend="CPU"))
        self.assertTrue(result["paired"], result)

    def test_cross_version_normal_allows_only_declared_schema_source_differences(self):
        private, _ = fixture()
        original = original_normal(private)
        saved = copy.deepcopy((original, private))
        result = normal_equivalence(original, private)
        self.assertTrue(result["equivalent"], result)
        self.assertEqual((original, private), saved)
        self.assertFalse(result["modes_rewritten"])
        private["request"]["threads"] = 8
        self.assertEqual(normal_equivalence(original, private)["binding_status"], "failed")

    def test_normal_comparator_never_relabels_capture(self):
        private, capture = fixture()
        result = normal_equivalence(original_normal(private), capture)
        self.assertFalse(result["equivalent"])

    def test_full_logits_and_sampled_tokens_not_forced_IDs_decide_numerics(self):
        normal, capture = fixture()
        capture["results"][0]["sampled_greedy_token_ids"][0] = 18
        capture["results"][0]["logit_observations"][1]["full_f32_logits_sha256"] = "f" * 64
        self.assertEqual(pair_audit(normal, capture)["numerical_status"], "failed")

    def test_partitions_are_actual_and_not_inferred_from_CPU_weights(self):
        normal, capture = fixture()
        capture["results"][0]["graph_observations"][0]["scheduler_copies"] = 9
        self.assertEqual(pair_audit(normal, capture)["partition_status"], "failed")

    def test_graph_replay_remains_unsupported_without_cached_route_reuse(self):
        normal, capture = fixture()
        for data in (normal, capture):
            event = data["results"][0]["cuda_graph_decisions"][0]
            event.update(use_cuda_graph=True,execution="replay_launch")
        capture["results"][0].update(gpu_capture_error_code=10,gpu_capture_status="unsupported")
        result = pair_audit(normal, capture)
        self.assertEqual(result["status"], "unsupported", result)
        self.assertFalse(result["paired"])
        self.assertFalse(result["graph_replay_routes_filled_from_cache"])

    def test_missing_actual_graph_decisions_are_unsupported(self):
        normal, capture = fixture()
        for data in (normal, capture):
            data["results"][0]["cuda_graph_decisions"] = []
        self.assertEqual(pair_audit(normal, capture)["partition_status"], "unsupported")

    def test_missing_copy_stage_receipt_fails_even_with_complete_route_ids(self):
        normal, capture = fixture()
        row = capture["results"][0]
        row["gpu_copy_inventory"].pop()
        row["gpu_observer_copy_count"] -= 1
        self.assertEqual(pair_audit(normal, capture)["structural_status"], "failed")

    def test_wrong_stream_pitch_or_bytes_fail(self):
        for key, value in (("actual_consumer_stream_opaque_id", 0), ("source_nb0", 8), ("source_nb1", 16), ("bytes", 0)):
            with self.subTest(key=key):
                normal, capture = fixture()
                capture["results"][0]["gpu_copy_inventory"][0][key] = value
                self.assertEqual(pair_audit(normal, capture)["structural_status"], "failed")

    def test_duplicate_overlapping_or_noncontiguous_pinned_slots_fail(self):
        for offset in (0, 4, 99999999):
            normal, capture = fixture()
            capture["results"][0]["gpu_copy_inventory"][1]["pinned_destination_offset_bytes"] = offset
            self.assertEqual(pair_audit(normal, capture)["structural_status"], "failed")

    def test_real_two_stage_expert_graph_is_not_assumed_three(self):
        result = pair_audit(*fixture(stages=("up", "down")))
        self.assertTrue(result["paired"], result)

    def test_native_stage_ID_equality_assertion_is_required(self):
        normal, capture = fixture()
        capture["results"][0]["routes"][0]["observed_stage_routes_equal"] = False
        capture["results"][0]["gpu_routes"][0]["observed_stage_routes_equal"] = False
        self.assertEqual(pair_audit(normal, capture)["structural_status"], "failed")

    def test_cross_backend_stages_are_explicitly_unsupported(self):
        normal, capture = fixture(backend="CPU")
        for data in (normal, capture):
            data["results"][0]["graph_observations"][0]["expert_nodes"][0]["backend"] = "CUDA0"
        # Explicit graph evidence is missing for the newly declared GPU stage;
        # no route is invented from the CPU weight-buffer label.
        result = pair_audit(normal, capture)
        self.assertEqual(result["partition_status"], "unsupported", result)

    def test_prefill_tail_and_pruned_final_layer_are_exact(self):
        result = pair_audit(*fixture(prompt=259,outputs=1,backend="CPU"))
        self.assertTrue(result["paired"], result)
        normal, capture = fixture(prompt=259,outputs=1,backend="CPU")
        wrong = copy.deepcopy(capture["results"][0]["routes"][0])
        wrong["layer"] = 15
        capture["results"][0]["routes"].append(wrong)
        self.assertEqual(pair_audit(normal, capture)["structural_status"], "failed")

    def test_missing_GPU_consumer_route_is_unsupported_not_filled(self):
        normal, capture = fixture()
        row = capture["results"][0]
        removed = row["routes"].pop()
        row["gpu_routes"].pop()
        row["gpu_copy_inventory"] = [x for x in row["gpu_copy_inventory"] if (x["forward_index"],x["layer"]) != (removed["forward_index"],removed["layer"])]
        row["gpu_observer_copy_count"] = len(row["gpu_copy_inventory"])
        result = pair_audit(normal, capture)
        self.assertEqual(result["status"], "unsupported", result)

    def test_observer_completion_boundary_cannot_add_flush(self):
        normal, capture = fixture()
        capture["results"][0]["observer_extra_flush_ns"] = 1
        self.assertEqual(pair_audit(normal, capture)["binding_status"], "failed")


if __name__ == "__main__":
    unittest.main()
