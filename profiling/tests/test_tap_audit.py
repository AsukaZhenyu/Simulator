"""Small synthetic records test audit logic; never execute a model or device."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from profiling.real_moe_validation.postcollection import CONTRACT, expected_forwards
from profiling.real_moe_validation.tap_audit import audit_files, normal_runtime_equivalence, pair_audit


def fixture(prompt=259, outputs=3, stages=("up", "gate", "down")):
    tokens, generated = list(range(prompt)), [17] * outputs
    forwards = expected_forwards(prompt, outputs)
    logits = [dict(output_index=i,full_f32_logits_sha256="c" * 64,top_logits=[dict(token=17,logit=2.0),dict(token=18,logit=1.5)],
                   top1_top2_margin=.5,greedy_token=17) for i in range(outputs)]
    graphs, routes = [], []
    for forward in forwards:
        experts = []
        for layer in range(16):
            positions = list(range(forward["position"], forward["position"] + forward["tokens"])) if layer < 15 else (
                [forward["position"] + forward["tokens"] - 1] if forward["output_rows"] else [])
            if not positions:
                continue
            for stage in stages:
                experts.append(dict(name=f"ffn_moe_{stage}-{layer}",backend="CPU",weight_buffer="CPU_Mapped",weight_type="q4_K"))
            routes.append(dict(forward_index=forward["index"],layer=layer,phase=forward["phase"],absolute_positions=positions,
                               expert_ids=[list(range(8)) for _ in positions],consumer_backend="CPU",actual_stages_observed=list(stages),
                               all_three_stages_observed=set(stages) == {"up", "gate", "down"},observed_stage_routes_equal=True,stage_count=len(stages)))
        graphs.append(dict(forward_index=forward["index"],node_count=120,scheduler_splits=3,scheduler_copies=1,
                           graph_partition_fingerprint_sha256="d" * 64,expert_nodes=experts))
    row = dict(id="request",prompt_token_ids=tokens,output_token_ids=generated,teacher_forced=True,
               sampled_greedy_token_ids=generated.copy(),logit_observations=logits,graph_observations=graphs,
               forward_inventory=forwards,forward_count=len(forwards),decode_forward_count=outputs - 1,routes=routes,
               observer=dict(cpu_consumer_tap=True,step_logit_copy=True,graph_metadata=True),
               timing_usable_as_bare_reference=False,trace_timing_is_performance_truth=False)
    request = dict(mode="capture",batch_size=256,model_contract=CONTRACT,observe_logits=True,observe_graph=True,
                   placement="cpu",threads=16,context_size=2048,requests=[dict(id="request",prompt_token_ids=tokens,max_output_tokens=outputs,output_token_ids=generated)])
    captured = dict(schema="real-moe-tap-native-4",mode="capture",request=request,process_id=100,results=[row],
                    scheduler_callback_enabled=False,trace_callback_enabled=False,cpu_consumer_tap_enabled=True,tap_hook_available=True,
                    model_weights_loaded=True,observed_model_contract=CONTRACT,threadpool_policy="explicit_persistent_owned",
                    graph_or_allocator_modified=False,model_sha256_verified_by_parent="a" * 64,source_sha256="b" * 64,
                    actual_placement=dict(weight_buffers={"CPU_Mapped":100}))
    reference = copy.deepcopy(captured)
    reference["mode"] = reference["request"]["mode"] = "generate"
    reference["cpu_consumer_tap_enabled"] = False
    reference["request"]["requests"][0].pop("output_token_ids")
    reference["results"][0]["teacher_forced"] = False
    reference["results"][0]["routes"] = []
    reference["results"][0]["observer"]["cpu_consumer_tap"] = False
    return reference, captured


class ConsumerTapAudit(unittest.TestCase):
    def test_normal_runtime_gate_compares_generate_without_mutating_modes(self):
        normal, _ = fixture(prompt=3, outputs=2)
        candidate = copy.deepcopy(normal)
        snapshot = copy.deepcopy((normal, candidate))
        report = normal_runtime_equivalence(normal, candidate)
        self.assertEqual(report["status"], "passed", report)
        self.assertEqual(report["denominator"], dict(requests=1,output_steps=2,graph_forwards=2))
        self.assertEqual((normal, candidate), snapshot)
        self.assertFalse(report["modes_rewritten"])

    def test_normal_runtime_gate_rejects_full_logit_or_partition_change(self):
        normal, _ = fixture(prompt=3, outputs=1)
        candidate = copy.deepcopy(normal)
        candidate["results"][0]["logit_observations"][0]["full_f32_logits_sha256"] = "e" * 64
        candidate["results"][0]["graph_observations"][0]["scheduler_splits"] = 9
        report = normal_runtime_equivalence(normal, candidate)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(any("full F32 logits" in x for x in report["failures"]))
        self.assertTrue(any("scheduler_splits" in x for x in report["failures"]))

    def test_normal_runtime_gate_rejects_capture_mode_instead_of_relabeling(self):
        normal, capture = fixture(prompt=3, outputs=1)
        report = normal_runtime_equivalence(normal, capture)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(any("actual generate mode" in x for x in report["failures"]))

    def test_complete_pair_passes_without_certifying_hidden_routes_or_accuracy(self):
        report = pair_audit(*fixture())
        self.assertTrue(report["paired"], report)
        self.assertEqual(report["counts"]["greedy_token_matches"], 3)
        self.assertFalse(report["hidden_bare_routes_proven_identical"])
        self.assertFalse(report["whole_model_accuracy_certified"])
        self.assertFalse(report["capture_timing_is_bare_truth"])

    def test_forced_ids_do_not_hide_sampled_greedy_difference(self):
        normal, capture = fixture()
        capture["results"][0]["sampled_greedy_token_ids"][1] = 18
        report = pair_audit(normal, capture)
        self.assertEqual(report["numerical_status"], "failed")
        self.assertFalse(report["paired"])

    def test_matching_top_logits_do_not_replace_full_logit_hashes(self):
        normal, capture = fixture()
        capture["results"][0]["logit_observations"][0]["full_f32_logits_sha256"] = "e" * 64
        self.assertEqual(pair_audit(normal, capture)["numerical_status"], "failed")

    def test_missing_step_hash_evidence_cannot_pass(self):
        normal, capture = fixture()
        capture["results"][0]["logit_observations"] = []
        self.assertEqual(pair_audit(normal, capture)["numerical_status"], "unknown")

    def test_observed_margin_and_greedy_are_checked(self):
        normal, capture = fixture()
        capture["results"][0]["logit_observations"][0]["top1_top2_margin"] = 9
        self.assertEqual(pair_audit(normal, capture)["numerical_status"], "failed")

    def test_graph_partition_or_copy_changes_fail_separately(self):
        normal, capture = fixture()
        capture["results"][0]["graph_observations"][0]["scheduler_copies"] = 2
        report = pair_audit(normal, capture)
        self.assertEqual(report["partition_status"], "failed")
        self.assertEqual(report["numerical_status"], "passed")

    def test_missing_last_layer_decode_route_fails(self):
        normal, capture = fixture()
        capture["results"][0]["routes"].pop()
        self.assertEqual(pair_audit(normal, capture)["structural_status"], "failed")

    def test_prefill_pruned_layer_does_not_get_fabricated_routes(self):
        normal, capture = fixture()
        fabricated = copy.deepcopy(capture["results"][0]["routes"][0])
        fabricated["layer"] = 15
        capture["results"][0]["routes"].append(fabricated)
        self.assertEqual(pair_audit(normal, capture)["structural_status"], "failed")

    def test_top8_ids_and_absolute_positions_are_validated(self):
        normal, capture = fixture()
        capture["results"][0]["routes"][0]["expert_ids"][0] = [0] * 8
        capture["results"][0]["routes"][1]["absolute_positions"][0] += 1
        self.assertEqual(pair_audit(normal, capture)["structural_status"], "failed")

    def test_actual_two_stage_graph_is_not_forced_to_have_three(self):
        report = pair_audit(*fixture(stages=("up", "down")))
        self.assertTrue(report["paired"], report)

    def test_missing_consumed_stage_fails_against_actual_graph(self):
        normal, capture = fixture()
        route = capture["results"][0]["routes"][0]
        route["actual_stages_observed"] = ["up", "down"]
        route["all_three_stages_observed"] = False
        route["stage_count"] = 2
        self.assertEqual(pair_audit(normal, capture)["structural_status"], "failed")

    def test_three_stage_flag_is_not_accepted_on_trust(self):
        normal, capture = fixture(stages=("up", "down"))
        capture["results"][0]["routes"][0]["all_three_stages_observed"] = True
        self.assertEqual(pair_audit(normal, capture)["structural_status"], "failed")

    def test_gpu_expert_consumer_is_explicitly_unsupported(self):
        normal, capture = fixture()
        for native in (normal, capture):
            native["results"][0]["graph_observations"][0]["expert_nodes"][0]["backend"] = "CUDA0"
        report = pair_audit(normal, capture)
        self.assertEqual(report["partition_status"], "unsupported")
        self.assertFalse(report["paired"])

    def test_unknown_expert_weight_binding_is_unsupported(self):
        normal, capture = fixture()
        for native in (normal, capture):
            native["results"][0]["graph_observations"][0]["expert_nodes"][0]["weight_buffer"] = "unknown"
        self.assertEqual(pair_audit(normal, capture)["partition_status"], "unsupported")

    def test_observer_run_is_never_promoted_to_bare_timing(self):
        normal, capture = fixture()
        normal["results"][0]["timing_usable_as_bare_reference"] = True
        self.assertEqual(pair_audit(normal, capture)["binding_status"], "failed")

    def test_backend_parameters_and_model_sha_are_bound(self):
        normal, capture = fixture()
        capture["request"]["threads"] = 8
        capture["model_sha256_verified_by_parent"] = "f" * 64
        self.assertEqual(pair_audit(normal, capture)["binding_status"], "failed")

    def test_scheduler_callback_cannot_silently_reenter(self):
        normal, capture = fixture()
        capture["scheduler_callback_enabled"] = True
        self.assertEqual(pair_audit(normal, capture)["binding_status"], "failed")

    def test_file_analysis_requires_completed_content_bound_receipts(self):
        normal, capture = fixture(prompt=3, outputs=1)
        workspace_build = Path(__file__).resolve().parents[2] / "build"
        workspace_build.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workspace_build, prefix="tap-audit-test-") as temporary:
            self.assertTrue(Path(temporary).resolve().is_relative_to(workspace_build.resolve()))
            paths = []
            for index, data in enumerate((normal, capture)):
                directory = Path(temporary) / str(index)
                directory.mkdir()
                path = directory / "result.json"
                raw = json.dumps(data).encode()
                path.write_bytes(raw)
                receipt = dict(status="completed",process_id=data["process_id"],native=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()))
                (directory / "execution.json").write_text(json.dumps(receipt))
                paths.append(path)
            report = audit_files(*paths)
            self.assertTrue(report["paired"], report)
            receipt["status"] = "failed"
            (paths[1].parent / "execution.json").write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):
                audit_files(*paths)


if __name__ == "__main__":
    unittest.main()
