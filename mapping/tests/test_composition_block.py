"""Block traversal shares causal commands, live state and service arbitration."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

from llm_infer_model.tensor.engine import Action
from llm_infer_model.tensor.environment import EnvironmentEvent, CODE_INFORMATION_NOT_KNOWN
from llm_infer_model.tensor.runtime import ExecutionCommand, dispatch_command
from llm_infer_model.tensor.services import parse_services
from llm_infer_model.tensor.spec import COPY_READY, InvalidInput, OP_DONE, Unsupported
from tensor_mapping.causal import Decision, SerialDemandPolicy, compile_command, run_causal
from tensor_mapping.composition_block import create_session, load_block_model_file


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/papers/flexgen-layer-major.blocks.json"


def service_contract():
    return parse_services({"format_version": "tensor-services-0.1", "pcie": {
        "bandwidth_bytes_per_second": 10**9, "ce_bytes_per_second": 10**9,
        "sm_bytes_per_second": 10**9, "arbitration": "max_min_fair",
        "timing_scope": "payload_only_no_launch_latency"}, "slots": []})


class BlockCompositionTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "outputs")
        self.path = Path(self.temp.name) / "block.json"
        self.raw = json.loads(EXAMPLE.read_bytes())
        self.raw["hardware"]["vram_capacity_bytes"] = 4096
        self.raw["hardware"]["dram_capacity_bytes"] = 4096
        self.write()

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        self.path.write_text(json.dumps(self.raw, indent=2) + "\n", encoding="utf-8", newline="\n")

    def test_static_catalogue_and_block_metadata_are_public_without_fixed_program(self):
        model = load_block_model_file(self.path)
        session = create_session(model)
        observation = session.observe()
        self.assertEqual(observation.layer_order, ("L0", "L1"))
        self.assertEqual(observation.weight_catalogue,
            (("w:L0", "L0", "dense_ffn", None), ("w:L1", "L1", "dense_ffn", None)))
        self.assertEqual(observation.active_invocations, (("op:g0:L0:B0", "L0"),))
        self.assertEqual([op.id for op in observation.operations], ["op:g0:L0:B0"])
        graph = session.execution_document()
        self.assertFalse(graph["boundary"]["block_execution"]["legacy_fixed_program_executed"])
        self.assertEqual(graph["scenario"]["mapspace"]["allowed_devices"]["op:g0:L0:B0"], ["gpu"])
        self.assertEqual(len(graph["tasks"]), 8)
        self.assertEqual(session.result().evaluation.actions, ())

    def test_block_adapter_exposes_current_rate_and_builtin_policy_waits(self):
        model = load_block_model_file(self.path)
        session = create_session(model, environment_events=(
            EnvironmentEvent(0, gpu_service_rate_numerator=0, gpu_service_rate_denominator=1),
            EnvironmentEvent(100000, gpu_service_rate_numerator=1, gpu_service_rate_denominator=2)))
        initial = session.observe()
        self.assertEqual(initial.gpu_service_rate_numerator, 0)
        self.assertTrue(all(price.current_duration_ns is None
                            for operation in initial.operations for price in operation.prices))
        run = run_causal(SerialDemandPolicy("gpu"), session, max_decisions=1000)
        self.assertEqual(run.evaluation.status, "valid")
        self.assertTrue(all(row.start_ns >= 100000 for row in run.evaluation.events if row.kind == "COMPUTE"))
        self.assertTrue(all(price.current_duration_ns == price.duration_ns * 2
                            for operation in session.observe().operations for price in operation.prices))
        self.assertEqual(session.persistent_state_document()["compute_progress"], [])

    def test_traversal_is_enforced_by_modeling_not_only_policy(self):
        for order, second in (("layer_major", "op:g0:L0:B1"), ("batch_major", "op:g0:L1:B0")):
            self.raw["order"] = order
            self.write()
            model = load_block_model_file(self.path)
            session = create_session(model)
            self.assertEqual(model.tasks[1].operation_id, second)
            with self.assertRaisesRegex(Unsupported, "not been revealed"):
                compile_command(Decision("COMPUTE", second, device="gpu"), session.observe())
            rejected = session.submit_command(ExecutionCommand(Action("COMPUTE", second, device="gpu")))
            self.assertFalse(rejected.ok)
            self.assertEqual(rejected.error_code, CODE_INFORMATION_NOT_KNOWN)

    def test_common_runtime_executes_all_blocks_with_one_clock_and_immutable_kv_ids(self):
        model = load_block_model_file(self.path)
        session = create_session(model, services=service_contract())
        demand = SerialDemandPolicy("gpu")
        def policy(observation):
            decision = demand(observation)
            if decision.kind == "COPY":
                role = next(t.role for t in observation.tensors if t.id == decision.target_id)
                return replace(decision, transfer_engine="ce" if role == "weight" else "sm")
            return decision
        run = run_causal(policy, session, max_decisions=1000)
        self.assertEqual(run.evaluation.status, "valid")
        computations = [record.action.target_id for record in run.decisions if record.action.kind == "COMPUTE"]
        self.assertEqual(computations, [task.operation_id for task in model.tasks])
        times = [record.observation.t_ns for record in run.decisions]
        self.assertEqual(times, sorted(times))
        inventory = session.persistent_state_document()
        self.assertGreater(inventory["t_ns"], 0)
        self.assertEqual(len(inventory["completed_blocks"]), 8)
        copies = {row["tensor_id"]:row["copies"] for row in inventory["tensor_inventory"]}
        for layer in ("L0", "L1"):
            for batch in ("B0", "B1"):
                for pos in (1, 2):
                    self.assertEqual(copies[f"kv:{layer}:{batch}:{pos}"]["ssd"], COPY_READY)
        self.assertFalse(inventory["resumable_checkpoint"])
        self.assertTrue(session.service_document()["transfer_engines"])
        self.assertEqual({row["engine"] for row in session.service_document()["transfer_engines"]}, {"ce", "sm"})
        # Writes preserve the already-known limitation instead of inventing a
        # complete bounded-buffer implementation while adapting the interface.
        self.assertEqual(copies["kv:L0:B0:1"]["dram"], COPY_READY)

    def test_future_layer_prefetch_overlaps_current_compute_in_same_runtime(self):
        session = create_session(load_block_model_file(self.path), services=service_contract())
        demand = SerialDemandPolicy("gpu")
        while not any(row.resource == "gpu_compute" for row in session.observe().running):
            observed = session.observe()
            decision = demand(observed)
            if decision.kind == "COPY":
                decision = replace(decision, transfer_engine="ce")
            self.assertTrue(dispatch_command(session, compile_command(decision, observed)).ok)
        compute_time = session.observe().t_ns
        self.assertTrue(dispatch_command(session, compile_command(Decision("READ_SSD", "w:L1"),
                                                                    session.observe())).ok)
        resources = {row.resource for row in session.observe().running}
        self.assertEqual(resources, {"gpu_compute", "ssd_io"})
        self.assertEqual(session.observe().active_invocations, (("op:g0:L0:B0", "L0"),))
        self.assertEqual(session.observe().t_ns, compute_time)
        while any(row.resource == "ssd_io" for row in session.observe().running):
            self.assertTrue(session.advance().ok)
        tensor = next(t for t in session.observe().tensors if t.id == "w:L1")
        self.assertEqual(tensor.status_at("dram"), COPY_READY)
        self.assertEqual(session.result().evaluation.ssd_read_bytes, 160)

    def test_recoverable_kv_is_distinct_from_dead_intermediate(self):
        session = create_session(load_block_model_file(self.path))
        run = run_causal(SerialDemandPolicy("gpu"), session, max_decisions=1000)
        self.assertEqual(run.evaluation.status, "valid")
        observed = session.observe()
        self.assertIn(("kv:L0:B0:1", "vram"), observed.recoverable_release_candidates)
        self.assertNotIn(("kv:L0:B0:1", "vram"), observed.release_candidates)
        self.assertIn(("h:g0:L0:B0", "vram"), observed.release_candidates)

    def test_source_binding_and_environment_remain_shared_and_causal(self):
        model = load_block_model_file(self.path)
        a = create_session(model, environment_events=(EnvironmentEvent(500, vram_external_bytes=16),))
        b = create_session(model, environment_events=(EnvironmentEvent(700, vram_external_bytes=32),))
        self.assertEqual(a.observe(), b.observe())
        self.assertTrue(a.advance().ok)
        self.assertEqual(a.observe().t_ns, 500)
        self.assertEqual(a.observe().memory[1].external_bytes, 16)
        self.assertEqual(b.observe().t_ns, 0)
        self.raw["generations"] = 3
        self.write()
        with self.assertRaisesRegex(InvalidInput, "source changed"):
            create_session(model)


if __name__ == "__main__":
    unittest.main()
