"""Single-document block policies exercise the common driver and checkpoints."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from llm_infer_model.tensor.spec import COPY_ABSENT, COPY_READY, InvalidInput, Unsupported
from tensor_mapping.unified_strategy import (checkpoint_document, compile_unified_strategy_file,
    run_unified_strategy, write_unified_result)


ROOT = Path(__file__).resolve().parents[1]
STRATEGY = ROOT / "examples/composition/block.strategy.json"
LEGACY = ROOT / "examples/papers/flexgen-layer-major.blocks.json"


def snapshot(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BlockCompositionIRTests(unittest.TestCase):
    def setUp(self):
        self.legacy_hash = snapshot(LEGACY)
        (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "outputs")
        self.output = Path(self.temp.name)

    def tearDown(self):
        self.assertEqual(snapshot(LEGACY), self.legacy_hash)
        self.temp.cleanup()

    def test_inline_mechanisms_run_block_and_charge_both_ssd_directions(self):
        compiled = compile_unified_strategy_file(STRATEGY)
        self.assertIsNotNone(compiled.composition)
        self.assertTrue(compiled.document["model"]["override_model_policy"])
        self.assertEqual(compiled.document["execution"], [])
        result = run_unified_strategy(compiled)
        run, session, policy, replayed = result
        self.assertEqual(run.evaluation.status, "valid")
        self.assertEqual(replayed, 0)
        self.assertEqual([row.action.target_id for row in run.decisions if row.action.kind == "COMPUTE"],
                         [task.operation_id for task in compiled.model.tasks])
        # 2 initial hidden ports + 4 initial KV ports + 2 immutable weights.
        self.assertEqual(run.evaluation.ssd_read_bytes, 2*16 + 4*16 + 2*64)
        # 8 generated KV ports + 2 final hidden ports.
        self.assertEqual(run.evaluation.ssd_write_bytes, 8*16 + 2*16)
        self.assertLessEqual(run.evaluation.peak_vram_bytes, 256)
        self.assertGreater(run.evaluation.h2d_bytes, 0)
        self.assertGreater(run.evaluation.d2h_bytes, 0)
        self.assertEqual({row["engine"] for row in session.service_document()["transfer_engines"]}, {"ce", "sm"})
        catalogue = {row[0] for row in session.observe().weight_catalogue}
        for record in run.decisions:
            resident = sum(tensor.storage_bytes for tensor in record.observation.tensors
                if tensor.id in catalogue and tensor.status_at("vram") != COPY_ABSENT)
            self.assertLessEqual(resident, compiled.composition.cache_budget_bytes)
        audit = policy.audit_document()
        prefetches = [row for row in audit["decisions"] if row["reason"] == "cross_layer_public_history_prefetch"]
        self.assertTrue(prefetches)
        self.assertEqual({row["evidence"]["tensor"] for row in prefetches}, {"w:L1"})
        tasks = {task.operation_id: task for task in compiled.model.tasks}
        for prefetch in prefetches:
            evidence = prefetch["evidence"]
            source = tasks[evidence["source_invocation_id"]]
            self.assertEqual(evidence["max_count_scope"], "per_active_source_invocation")
            self.assertEqual((source.layer_id, evidence["layer_id"]), ("L0", "L1"))
            # The same weight can be prefetched again in a later generation;
            # it must precede that generation's target, not the entire run's
            # first L1 block. The source must be current public scope evidence.
            target_time = next(row.observation.t_ns for row in run.decisions
                if row.action.kind == "COMPUTE" and tasks[row.action.target_id].layer_id == "L1"
                and tasks[row.action.target_id].generation == source.generation)
            self.assertLess(prefetch["t_ns"], target_time)
            self.assertTrue(any(row.observation.t_ns == prefetch["t_ns"]
                and (source.invocation_id, source.layer_id) in row.observation.active_invocations
                for row in run.decisions))
        self.assertTrue(any(row["reason"] == "release_recoverable_copy_keep_home" for row in audit["cache_decisions"]))
        write_unified_result(self.output / "full", compiled, result)
        summary = json.loads((self.output / "full/summary.json").read_bytes())
        self.assertTrue(summary["composed_mechanisms"])
        self.assertTrue(summary["one_clock_and_inventory"])
        self.assertFalse(summary["paper_strategy_reproduced"])
        self.assertFalse(summary["search_performed"])
        self.assertFalse(summary["physical_calibration_claimed"])
        inventory = session.persistent_state_document()
        copies = {row["tensor_id"]:row["copies"] for row in inventory["tensor_inventory"]}
        self.assertEqual(copies["kv:L0:B0:1"]["ssd"], COPY_READY)
        self.assertEqual(copies["kv:L0:B0:1"]["dram"], COPY_READY)

    def test_checkpoint_inside_two_transfers_recovers_same_absolute_execution(self):
        compiled = compile_unified_strategy_file(STRATEGY)
        full = run_unified_strategy(compiled)
        partial = run_unified_strategy(compiled, max_decisions=15)
        self.assertEqual(partial[0].termination_reason, "decision_budget_exhausted")
        self.assertEqual({row.resource for row in partial[1].observe().running}, {"pcie_sm", "ssd_io"})
        checkpoint = checkpoint_document(compiled, partial)
        resumed = run_unified_strategy(compiled, checkpoint=checkpoint)
        self.assertEqual(resumed[3], 15)
        self.assertEqual(resumed[0].evaluation, full[0].evaluation)
        self.assertEqual(resumed[1].persistent_state_document(), full[1].persistent_state_document())
        self.assertEqual(resumed[1].service_document(), full[1].service_document())
        self.assertEqual(resumed[2].audit_document(), full[2].audit_document())

    def test_legacy_override_is_explicit_and_cpu_cost_is_never_invented(self):
        raw = json.loads(STRATEGY.read_bytes())
        for name, value in (("model", False), ("placement", "cpu")):
            document = deepcopy(raw)
            for field in ("services_file", "environment_file"):
                document[field] = str((STRATEGY.parent / document[field]).resolve())
            document["model"]["file"] = str((STRATEGY.parent / document["model"]["file"]).resolve())
            if name == "model":
                document["model"]["override_model_policy"] = value
            else:
                document["policy"]["placements"]["dense_ffn"] = value
            path = self.output / (name + ".json")
            path.write_text(json.dumps(document)+"\n", encoding="utf-8", newline="\n")
            with self.assertRaises(InvalidInput if name == "model" else Unsupported):
                compile_unified_strategy_file(path)

    def test_cli_partial_resume_and_full_use_same_entry_point(self):
        def invoke(name, extra=(), expected=0):
            args = [sys.executable, "-m", "tensor_mapping.unified_strategy", str(STRATEGY),
                    "--out", str(self.output / name), *extra]
            process = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(process.returncode, expected, process.stdout + process.stderr)
        invoke("partial", ("--max-decisions", "15"), expected=3)
        invoke("resumed", ("--resume", str(self.output / "partial/checkpoint.json")))
        invoke("full")
        for filename in ("execution.json", "events.json", "states.json", "persistent_state.json",
                         "services.json", "policy_audit.json"):
            self.assertEqual((self.output / "resumed" / filename).read_bytes(),
                             (self.output / "full" / filename).read_bytes())
        summary = json.loads((self.output / "resumed/summary.json").read_bytes())
        self.assertEqual(summary["checkpoint_replayed_commands"], 15)


if __name__ == "__main__":
    unittest.main()
