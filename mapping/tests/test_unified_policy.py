"""Public service policy contracts and continuous checkpoint reconstruction."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from llm_infer_model.m2.causal import DevicePrice, PublicOperation, TransferPrice
from llm_infer_model.tensor.environment import RunningObservation
from llm_infer_model.tensor.services import PublicServices, PublicFixedSlot
from llm_infer_model.tensor.spec import COPY_ABSENT, COPY_READY, InvalidInput
from tensor_mapping.policy_ir import parse_policy, parse_history
from tensor_mapping.unified_policy import UnifiedPolicy
from tensor_mapping.unified_strategy import (compile_unified_strategy_file, run_unified_strategy,
    checkpoint_document, write_unified_result)
from .test_policy_executor import small_observation
from .test_policy_ir import policy_document_fixture, history_fixture

EXAMPLES = Path(__file__).resolve().parents[1] / "examples/unified"


def transport():
    return {"transfer_engines": {"weight": "ce", "input": "sm", "intermediate": "sm", "output": "sm"},
            "slot_reuse": "release_unlocked_backed_copy", "boundary_release": "disabled", "fragment_cache_unit": "tensor"}


def public_prefetch_observation(*, old_owner=True, locked=False):
    base = small_observation(free_vram=1024)
    tensors = []
    for tensor in base.tensors:
        copies = tensor.copies
        if tensor.id == "w0":
            copies = (("dram", COPY_READY), ("vram", COPY_ABSENT), ("ssd", COPY_READY))
        elif tensor.id == "w3" and not old_owner:
            copies = (("dram", COPY_ABSENT), ("vram", COPY_ABSENT), ("ssd", COPY_READY))
        prices = tuple(replace(price, duration_ns=None,
            engine_durations_ns=(("ce", tensor.storage_bytes), ("sm", tensor.storage_bytes)),
            timing_scope="isolated_payload_service_subject_to_contention")
            if {price.src, price.dst} == {"dram", "vram"} else price for price in tensor.transfers)
        allocation = (("dram", tensor.storage_bytes), ("vram", 0 if tensor.id in ("w0", "w3") else tensor.storage_bytes))
        tensors.append(replace(tensor, copies=copies, transfers=prices, allocation_bytes=allocation))
    operations, running = (), ()
    if locked:
        operations = (PublicOperation("reader", "projection", "RUNNING", ("w3",), ("y",),
                                     (DevicePrice("gpu", 100, 0, 0),)),)
        running = (RunningObservation("gpu_compute", "reader", "gpu", None),)
    services = PublicServices(10**9, 10**9, 10**9, "max_min_fair", "payload_only_no_launch_latency",
        (PublicFixedSlot("slot", "vram", 32, ("w0", "w3"), "w3" if old_owner else None),), ())
    return replace(base, operations=operations, running=running, tensors=tuple(tensors), services=services)


def prefetch_policy(budget=64):
    document = policy_document_fixture(prefetch=True)
    document["prefetch"]["cache_budget_bytes"] = budget
    history = parse_history(history_fixture((0,), layer="layer"))
    return UnifiedPolicy(parse_policy(document), transport=transport(), history=history)


class UnifiedServicePolicyTests(unittest.TestCase):
    def test_locked_slot_proposal_waits_then_releases_and_retries_prefetch(self):
        policy = prefetch_policy()
        decision = policy(public_prefetch_observation(locked=True))
        self.assertEqual(decision.kind, "ADVANCE")
        self.assertEqual(policy.audit_document()["predictions"], [])
        decision = policy(public_prefetch_observation())
        self.assertEqual((decision.kind, decision.target_id, decision.location), ("EVICT", "w3", "vram"))
        self.assertEqual(policy.audit_document()["predictions"], [])
        decision = policy(public_prefetch_observation(old_owner=False))
        self.assertEqual((decision.kind, decision.target_id, decision.transfer_engine), ("COPY", "w0", "ce"))
        self.assertEqual(policy.audit_document()["predictions"][0]["tensor_id"], "w0")
        self.assertEqual(policy.audit_document()["transport_decisions"][0]["reason"], "wait_for_locked_or_live_slot")

    def test_zero_incremental_slot_allocation_does_not_hide_cache_payload(self):
        observed = public_prefetch_observation(old_owner=False)
        blocked = prefetch_policy(31)
        self.assertEqual(blocked(observed).kind, "ADVANCE")
        self.assertEqual(blocked.audit_document()["predictions"], [])
        allowed = prefetch_policy(32)
        self.assertEqual(allowed(observed).kind, "COPY")
        self.assertEqual(allowed.audit_document()["cache_budget_scope"], "materialized_payload_bytes")

    def test_engine_mutex_and_overlap_contract_are_used_before_submission(self):
        observed = public_prefetch_observation(old_owner=False)
        gpu = replace(observed, running=(RunningObservation("gpu_compute", "work", "gpu", None),))
        self.assertTrue(UnifiedPolicy._resource_available(gpu, "pcie_ce"))
        self.assertFalse(UnifiedPolicy._resource_available(gpu, "pcie_sm"))
        sm = replace(observed, running=(RunningObservation("pcie_sm", "x", None, "dram_to_vram"),))
        self.assertFalse(UnifiedPolicy._resource_available(sm, "gpu_compute", compute=True))
        self.assertTrue(UnifiedPolicy._resource_available(sm, "cpu_compute", compute=True))
        self.assertFalse(UnifiedPolicy._resource_available(replace(sm, allow_copy_compute_overlap=False),
                                                          "cpu_compute", compute=True))

    def test_missing_explicit_service_engine_is_not_inferred(self):
        bindings = transport()
        bindings["transfer_engines"]["weight"] = None
        policy = UnifiedPolicy(parse_policy(policy_document_fixture()), transport=bindings)
        with self.assertRaisesRegex(InvalidInput, "explicitly bind"):
            policy(public_prefetch_observation(old_owner=False))


class UnifiedStrategyIntegrationTests(unittest.TestCase):
    def test_fragment_strategy_composes_explicit_slot_reuse_and_cpu_gpu(self):
        compiled = compile_unified_strategy_file(EXAMPLES / "fragment.strategy.json")
        run, session, policy, replayed = run_unified_strategy(compiled)
        self.assertEqual(run.evaluation.status, "valid", run.evaluation.reason)
        self.assertEqual(replayed, 0)
        decisions = [(record.decision.kind, record.decision.target_id, record.decision.device)
                     for record in run.decisions]
        self.assertIn(("EVICT", "w:moe-small:W_proj", None), decisions)
        placed = [(op.kind, record.decision.device) for record in run.decisions if record.decision.kind == "COMPUTE"
                  for op in record.observation.operations if op.id == record.decision.target_id]
        self.assertTrue(all(device == "gpu" for kind, device in placed if kind in ("expert_gate", "expert_up")))
        self.assertTrue(all(device == "cpu" for kind, device in placed if kind == "expert_down"))
        self.assertTrue(any(row["engine"] == "ce" for row in session.service_document()["transfer_engines"]))
        self.assertTrue(any(row["engine"] == "sm" for row in session.service_document()["transfer_engines"]))
        self.assertEqual(policy.audit_document()["transport_decisions"][0]["reason"], "explicit_slot_owner_release")
        self.assertEqual(len(session.persistent_state_document()["weight_fragments"]), 12)

    def _temporary_strategy(self, directory, *, fractional_environment=False):
        document = json.loads((EXAMPLES / "fragment.strategy.json").read_text(encoding="utf-8"))
        for key in ("model_file", "execution_file", "policy_file", "services_file", "environment_file"):
            document[key] = str((EXAMPLES / document[key]).resolve())
        if fractional_environment:
            environment = Path(directory) / "environment.json"
            environment.write_text(json.dumps({"format_version": "environment-trace-0.1", "events": [
                {"t_ns": 17, "vram_external_bytes": 16}, {"t_ns": 117, "vram_external_bytes": 0}]}), encoding="utf-8", newline="\n")
            document["environment_file"] = str(environment)
        manifest = Path(directory) / "strategy.json"
        manifest.write_text(json.dumps(document), encoding="utf-8", newline="\n")
        return manifest

    def test_checkpoint_reconstructs_fractional_inflight_service_and_same_final_run(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._temporary_strategy(directory, fractional_environment=True)
            compiled = compile_unified_strategy_file(manifest)
            partial = run_unified_strategy(compiled, max_decisions=3)
            self.assertEqual(partial[0].termination_reason, "decision_budget_exhausted")
            self.assertEqual(partial[1].observe().t_ns, 17)
            flows = partial[1].observe().services.flows
            self.assertEqual(len(flows), 2)
            self.assertTrue(all(Fraction(flow.remaining_bytes_numerator, flow.remaining_bytes_denominator)
                                == Fraction(47, 2) for flow in flows))
            checkpoint = checkpoint_document(compiled, partial)
            write_unified_result(Path(directory) / "partial", compiled, partial)
            inventory = json.loads((Path(directory) / "partial/persistent_state.json").read_text(encoding="utf-8"))
            self.assertIsNone(inventory["kv_state"])
            self.assertTrue(inventory["running"])
            resumed = run_unified_strategy(compiled, checkpoint=checkpoint)
            reference = run_unified_strategy(compiled)
            self.assertEqual(resumed[0].evaluation.status, "valid")
            self.assertEqual(resumed[3], 3)
            self.assertEqual(resumed[0].runtime_result, reference[0].runtime_result)
            self.assertEqual(resumed[1].service_document(), reference[1].service_document())
            self.assertEqual(resumed[1].persistent_state_document(), reference[1].persistent_state_document())

    def test_checkpoint_refuses_runtime_source_or_command_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._temporary_strategy(directory)
            compiled = compile_unified_strategy_file(manifest)
            partial = run_unified_strategy(compiled, max_decisions=2)
            checkpoint = checkpoint_document(compiled, partial)
            with patch("tensor_mapping.unified_strategy.runtime_fingerprint", return_value="changed"):
                with self.assertRaisesRegex(InvalidInput, "runtime changed"):
                    run_unified_strategy(compiled, checkpoint=checkpoint)
            corrupted = deepcopy(checkpoint)
            corrupted["commands"][0]["decision"]["transfer_engine"] = "sm"
            with self.assertRaisesRegex(InvalidInput, "decision/observation differs"):
                run_unified_strategy(compiled, checkpoint=corrupted)
            manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8", newline="\n")
            with self.assertRaisesRegex(InvalidInput, "input changed"):
                run_unified_strategy(compiled, checkpoint=checkpoint)

    def test_cli_outputs_resumable_partial_then_completes(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._temporary_strategy(directory)
            partial, resumed = Path(directory) / "partial", Path(directory) / "resumed"
            first = subprocess.run([sys.executable, "-m", "tensor_mapping.unified_strategy", str(manifest),
                "--max-decisions", "2", "--out", str(partial)], capture_output=True, text=True)
            self.assertEqual(first.returncode, 3, first.stderr)
            self.assertTrue((partial / "checkpoint.json").is_file())
            second = subprocess.run([sys.executable, "-m", "tensor_mapping.unified_strategy", str(manifest),
                "--resume", str(partial / "checkpoint.json"), "--out", str(resumed)], capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            summary = json.loads((resumed / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "valid")
            self.assertEqual(summary["checkpoint_replayed_commands"], 2)
            self.assertFalse(summary["physical_calibration_claimed"])

    def test_unused_request_override_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._temporary_strategy(directory)
            document = json.loads(manifest.read_text(encoding="utf-8"))
            policy = json.loads((EXAMPLES / "fragment.policy.json").read_text(encoding="utf-8"))
            policy["request_placements"] = {"r-typo": {"expert_gate": "gpu"}}
            policy_path = Path(directory) / "policy.json"
            policy_path.write_text(json.dumps(policy), encoding="utf-8", newline="\n")
            document["policy_file"] = str(policy_path)
            manifest.write_text(json.dumps(document), encoding="utf-8", newline="\n")
            with self.assertRaisesRegex(InvalidInput, "request"):
                compile_unified_strategy_file(manifest)

    def test_global_dense_two_layers_prefill_decode_runs_through_declared_token_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._temporary_strategy(directory)
            document = json.loads(manifest.read_text(encoding="utf-8"))
            fixture = Path(__file__).resolve().parents[2] / "modeling/tests/fixtures/global_execution/dense.global.json"
            document.update(model_file=str(fixture), execution_file=None, services_file=None)
            document["transport"]["transfer_engines"] = {role: None for role in ("weight", "input", "intermediate", "output")}
            policy = policy_document_fixture()
            policy["placements"]["dense_ffn"] = "cpu"
            policy_path = Path(directory) / "policy.json"
            policy_path.write_text(json.dumps(policy), encoding="utf-8", newline="\n")
            document["policy_file"] = str(policy_path)
            manifest.write_text(json.dumps(document), encoding="utf-8", newline="\n")
            compiled = compile_unified_strategy_file(manifest)
            result = run_unified_strategy(compiled)
            self.assertEqual(result[0].evaluation.status, "valid", result[0].compilation_error)
            self.assertEqual(result[0].evaluation.makespan_ns, 44_007)
            self.assertEqual(len(result[1].observe().completed_invocations), 4)
            self.assertEqual(len(result[1].persistent_state_document()["kv_ids"]), 8)
            gates = [record for record in result[0].decisions if record.decision.kind == "COMPUTE"
                     and ":trace_next:" in record.decision.target_id]
            self.assertEqual(len(gates), 1)
            self.assertEqual(gates[0].decision.device, "cpu")


if __name__ == "__main__":
    unittest.main()
