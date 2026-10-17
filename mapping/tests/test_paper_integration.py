"""Drivers preserve the common dependency ledger and bind exact source bytes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from llm_infer_model.tensor.spec import InvalidInput, Unsupported, ROUTE_ENDS
from llm_infer_model.m2.routing_scores import ScoredCausalSession
from tensor_mapping.block_strategy import compile_blocks
from tensor_mapping.strategy_compiler import run_strategy
from tensor_mapping.service_program import (
    compile_service_program_file, run_service_program, write_service_result,
)
from tensor_mapping.paper_strategy import (
    compile_paper_strategy_file, run_paper_strategy, write_paper_result,
)
from tensor_mapping.causal import compile_decision
from tensor_mapping.policy_executor import DeclarativePolicy

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
PAPERS = EXAMPLES / "papers"


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def blocks(order="layer_major"):
    return {"format_version": "block-execution-0.1", "id": "hand-blocks",
        "provenance": "synthetic_block_template", "generations": 2,
        "initial_kv_positions": 1,
        "layers": [{"id": f"L{i}", "weight_storage_bytes": 64, "weight_home": "ssd",
                    "duration_ns": 100, "workspace_bytes": 0} for i in range(2)],
        "microbatches": [{"id": f"B{i}", "hidden_storage_bytes": 16,
                          "kv_storage_bytes": 8} for i in range(2)],
        "activation_home": "ssd", "kv_home": "ssd", "order": order,
        "weight_release": "after_group", "hardware": {
            "dram_capacity_bytes": 1024, "vram_capacity_bytes": 256,
            "dram_runtime_reserved_bytes": 0, "vram_runtime_reserved_bytes": 0,
            "dram_alignment_bytes": 1, "vram_alignment_bytes": 1,
            "cpu_compute_slots": 1, "gpu_compute_slots": 1,
            "pcie_copy_slots": 1, "ssd_io_slots": 1},
        "transfer_entries": [{"route": route, "storage_bytes": size, "duration_ns": size}
                             for route in ROUTE_ENDS for size in (8, 16, 64)]}


def service_fixture(directory):
    directory = Path(directory)
    scene = json.loads((EXAMPLES / "strategy/ssd-prefetch.scenario.json").read_text())
    scene["workload_file"] = str((EXAMPLES / "strategy/ssd-prefetch.graph.json").resolve())
    dump(directory / "scene.json", scene)
    dump(directory / "environment.json", {"format_version": "environment-trace-0.1", "events": []})
    dump(directory / "services.json", {"format_version": "tensor-services-0.1", "pcie": {
        "bandwidth_bytes_per_second": 10**9, "ce_bytes_per_second": 10**9,
        "sm_bytes_per_second": 10**9, "arbitration": "max_min_fair",
        "timing_scope": "payload_only_no_launch_latency"}, "slots": [
        {"id": "weights", "domain": "vram", "capacity_bytes": 64,
         "members": ["W:L0", "W:L1"]}]})
    program = {"format_version": "service-program-0.1", "id": "hand-services",
        "scenario_file": "scene.json", "services_file": "services.json",
        "environment_file": "environment.json", "information_mode": "offline_fixed",
        "pcie_cost_source": "services_payload_rates", "program": [
        {"kind": "compute", "operation_id": "layer0", "device": "gpu"},
        {"kind": "read_ssd", "tensor_id": "W:L1"},
        {"kind": "wait", "operations": ["layer0"],
         "copies": [{"tensor_id": "W:L1", "location": "dram"}]},
        {"kind": "release", "tensor_id": "W:L0", "location": "vram"},
        {"kind": "transfer", "tensor_id": "W:L1", "src": "dram", "dst": "vram", "engine": "ce"},
        {"kind": "wait", "operations": [], "copies": [{"tensor_id": "W:L1", "location": "vram"}]},
        {"kind": "compute", "operation_id": "layer1", "device": "gpu"},
        {"kind": "wait", "operations": ["layer1"], "copies": []}]}
    path = directory / "program.json"
    dump(path, program)
    return path, program


def paper_fixture(directory, paper, **changes):
    document = json.loads((PAPERS / f"{paper}.strategy.json").read_text())
    document["invocation_files"] = [str((PAPERS / item).resolve()) for item in document["invocation_files"]]
    for field in ("environment_file", "execution_file", "policy_file", "history_file", "routing_scores_file"):
        if document[field] is not None:
            document[field] = str((PAPERS / document[field]).resolve())
    document.update(changes)
    path = Path(directory) / "strategy.json"
    dump(path, document)
    return path


class BlockIntegrationTests(unittest.TestCase):
    def test_layer_weight_reuse_changes_only_explicit_transfer_traffic(self):
        results = []
        for order in ("layer_major", "batch_major"):
            result = run_strategy(compile_blocks(blocks(order)).compiled)
            self.assertEqual(result.evaluation.status, "valid")
            results.append(result.evaluation)
        # Four layer groups vs eight batch/layer groups load one 64-byte weight.
        self.assertEqual([result.h2d_bytes for result in results], [480, 736])
        self.assertEqual([result.makespan_ns for result in results], [1856, 2112])
        for result in results:
            self.assertEqual((result.d2h_bytes, result.ssd_read_bytes, result.ssd_write_bytes), (192, 192, 192))
            self.assertEqual(sum(event.kind == "COMPUTE" for event in result.events), 8)

    def test_generation_inputs_preserve_layer_chain_and_all_past_kv_versions(self):
        compiled = compile_blocks(blocks())
        operations = {row["id"]: row for row in compiled.workload["operations"]}
        first_next = operations["op:g1:L0:B0"]
        self.assertEqual(first_next["inputs"], ["h:g0:L1:B0", "w:L0", "kv:L0:B0:0", "kv:L0:B0:1"])
        self.assertEqual(operations["op:g1:L1:B0"]["inputs"],
                         ["h:g1:L0:B0", "w:L1", "kv:L1:B0:0", "kv:L1:B0:1"])
        self.assertEqual(set(compiled.scenario["goal"]["output_locations"]),
            {f"kv:L{layer}:B{batch}:{position}" for layer in range(2) for batch in range(2) for position in (1, 2)} |
            {"h:g1:L1:B0", "h:g1:L1:B1"})

    def test_capacity_rejection_stays_failure_instead_of_inferred_eviction(self):
        document = blocks()
        document["hardware"]["vram_capacity_bytes"] = 100
        result = run_strategy(compile_blocks(document).compiled)
        self.assertEqual(result.evaluation.status, "invalid_mapping")
        self.assertEqual(result.evaluation.error_code, "CAPACITY_EXCEEDED")


class ServiceProgramIntegrationTests(unittest.TestCase):
    def test_slot_reuse_waits_for_reader_and_uses_payload_service_price(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = service_fixture(directory)
            compiled = compile_service_program_file(path)
            session = run_service_program(compiled)
            result = session.result()
            self.assertEqual(result.evaluation.status, "valid")
            # SSD2ms overlaps GPU5ms; CE64B at1B/ns then GPU2ms.
            self.assertEqual(result.evaluation.makespan_ns, 7_000_064)
            self.assertEqual((result.evaluation.h2d_bytes, result.evaluation.ssd_read_bytes), (64, 64))
            services = session.service_document()
            self.assertEqual(services["slots"][0]["owner"], "W:L1")
            self.assertEqual(services["transfer_engines"][0]["source_index"], 4)
            self.assertEqual(result.action_sources[-1], 7)

    def test_early_slot_release_cannot_bypass_running_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            path, document = service_fixture(directory)
            document["program"].insert(1, document["program"].pop(3))
            dump(path, document)
            session = run_service_program(compile_service_program_file(path))
            self.assertEqual(session.result().evaluation.error_code, "TENSOR_IN_USE")
            self.assertEqual(session.service_document()["slots"][0]["owner"], "W:L0")
            self.assertEqual(session.state.op("layer0"), "RUNNING")

    def test_completed_goal_does_not_ignore_remaining_instruction(self):
        with tempfile.TemporaryDirectory() as directory:
            path, document = service_fixture(directory)
            document["program"].append({"kind": "release", "tensor_id": "x", "location": "vram"})
            dump(path, document)
            result = run_service_program(compile_service_program_file(path)).result()
            self.assertEqual(result.evaluation.status, "invalid_mapping")
            self.assertEqual(result.evaluation.error_code, "actions_remain_after_goal")
            self.assertEqual(result.action_sources[-1], 8)

    def test_unsatisfiable_wait_preserves_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path, document = service_fixture(directory)
            document["program"].insert(0, {"kind": "wait", "operations": ["layer1"], "copies": []})
            dump(path, document)
            result = run_service_program(compile_service_program_file(path)).result()
            self.assertEqual(result.evaluation.error_code, "INPUT_NOT_READY")
            self.assertEqual(result.evaluation.actions, ())

    def test_changed_source_refuses_run_and_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = service_fixture(directory)
            compiled = compile_service_program_file(path)
            session = run_service_program(compiled)
            source = Path(directory) / "services.json"
            source.write_bytes(source.read_bytes() + b"\n")
            with self.assertRaisesRegex(InvalidInput, "input changed"):
                run_service_program(compiled)
            with self.assertRaisesRegex(InvalidInput, "input changed"):
                write_service_result(Path(directory) / "output", compiled, session)
            self.assertFalse((Path(directory) / "output").exists())

    def test_written_events_sources_and_common_memory_are_reviewable(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = service_fixture(directory)
            compiled = compile_service_program_file(path)
            session = run_service_program(compiled)
            output = Path(directory) / "output"
            write_service_result(output, compiled, session)
            summary = json.loads((output / "summary.json").read_bytes())
            self.assertFalse(summary["paper_strategy_reproduced"])
            self.assertFalse(summary["physical_calibration_claimed"])
            self.assertEqual(summary["pcie_timing_scope"], "payload_only_no_launch_latency")
            for source in summary["sources"]:
                self.assertEqual(hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest(), source["sha256"])
            self.assertTrue(all(b"\r" not in child.read_bytes() for child in output.glob("*.json")))


class PaperDriverIntegrationTests(unittest.TestCase):
    def test_paper_examples_compile_execute_and_actions_match_current_observation(self):
        for paper in ("fiddler", "hybrimoe", "neo"):
            with self.subTest(paper=paper):
                compiled = compile_paper_strategy_file(PAPERS / f"{paper}.strategy.json")
                run, session, policy = run_paper_strategy(compiled)
                self.assertEqual(run.evaluation.status, "valid")
                self.assertFalse(policy.audit_document()["claims"]["paper_strategy_reproduced"])
                for record in run.decisions:
                    self.assertEqual(record.action, compile_decision(record.decision, record.observation))
                    self.assertFalse(hasattr(record.observation, "route_trace"))
                    self.assertFalse(hasattr(record.observation, "sources"))

    def test_paper_output_binds_inputs_and_labels_full_graph_private(self):
        with tempfile.TemporaryDirectory() as directory:
            compiled = compile_paper_strategy_file(PAPERS / "hybrimoe.strategy.json")
            result = run_paper_strategy(compiled)
            output = Path(directory) / "result"
            write_paper_result(output, compiled, result)
            graph = json.loads((output / "compiled_graph.json").read_bytes())
            self.assertEqual(graph["visibility_scope"], "private_debug_not_policy_input")
            summary = json.loads((output / "summary.json").read_bytes())
            self.assertFalse(summary["hardware_measurement"])
            self.assertFalse(summary["physical_calibration_claimed"])
            for source in summary["sources"]:
                self.assertEqual(hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest(), source["sha256"])
            self.assertTrue(all(b"\r" not in child.read_bytes() for child in output.glob("*.json")))

    def test_neo_execution_paths_cannot_claim_different_source_invocations(self):
        with tempfile.TemporaryDirectory() as directory:
            execution = json.loads((PAPERS / "neo.execution.json").read_text())
            execution["invocation_files"] = [str((PAPERS / item).resolve()) for item in execution["invocation_files"]]
            execution["invocation_files"][0] = str(Path(directory) / "unread-invocation.json")
            path = Path(directory) / "execution.json"
            dump(path, execution)
            with self.assertRaisesRegex(InvalidInput, "source paths"):
                compile_paper_strategy_file(paper_fixture(directory, "neo", execution_file=str(path)))

    def test_neo_typo_override_is_rejected_instead_of_silently_ignored(self):
        for request, kinds in (("r_typo", {"attention_core": "cpu"}),
                               ("r_cpu", {"expert_ffn": "cpu"})):
            with self.subTest(request=request, kinds=kinds), tempfile.TemporaryDirectory() as directory:
                policy = json.loads((PAPERS / "neo.policy.json").read_text())
                policy["request_placements"][request] = kinds
                path = Path(directory) / "policy.json"
                dump(path, policy)
                with self.assertRaisesRegex(InvalidInput, "unknown request or absent task kind"):
                    compile_paper_strategy_file(paper_fixture(directory, "neo", policy_file=str(path)))

    def test_paper_label_and_device_rule_cannot_disagree(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = json.loads((PAPERS / "fiddler.policy.json").read_text())
            policy["experts"]["device_rule"] = "fixed"
            path = Path(directory) / "policy.json"
            dump(path, policy)
            with self.assertRaisesRegex(InvalidInput, "mechanism label"):
                compile_paper_strategy_file(paper_fixture(directory, "fiddler", policy_file=str(path)))

    def test_mrs_requires_full_scores_at_compile_time(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(Unsupported, "full expert probabilities"):
                compile_paper_strategy_file(paper_fixture(directory, "hybrimoe", routing_scores_file=None))

    def test_private_full_probabilities_do_not_change_policy_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            scores = json.loads((PAPERS / "router.scores.json").read_text())
            # Preserve selected E0/E1 and normalization; alter only hidden E2/E3.
            scores["entries"][0]["router_scores"][2]["score"] = 0.10
            scores["entries"][0]["router_scores"][3]["score"] = 0.10
            score_path = Path(directory) / "scores.json"
            dump(score_path, scores)
            first = compile_paper_strategy_file(PAPERS / "hybrimoe.strategy.json")
            second = compile_paper_strategy_file(paper_fixture(directory, "hybrimoe", routing_scores_file=str(score_path)))
            compiled = (first, second)
            sessions = [ScoredCausalSession(item.binding, execution_spec=item.execution_spec,
                            routing_scores=item.routing_scores, environment_events=item.environment) for item in compiled]
            policies = [DeclarativePolicy(item.policy, history=item.history) for item in compiled]
            prefix_length = 0
            while not sessions[0].observe().route_history:
                observations = [session.observe() for session in sessions]
                self.assertEqual(observations[0], observations[1])
                decisions = [policy(observation) for policy, observation in zip(policies, observations)]
                self.assertEqual(decisions[0], decisions[1])
                self.assertEqual(policies[0].audit_document(), policies[1].audit_document())
                for session, observation, decision in zip(sessions, observations, decisions):
                    action = compile_decision(decision, observation)
                    step = session.advance() if action.kind == "ADVANCE" else session.submit(action)
                    self.assertTrue(step.ok)
                prefix_length += 1
                self.assertLess(prefix_length, 100)
            self.assertGreater(prefix_length, 0)
            self.assertNotEqual(sessions[0].observe().route_history, sessions[1].observe().route_history)


if __name__ == "__main__":
    unittest.main()
