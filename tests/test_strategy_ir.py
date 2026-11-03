"""Installed-library strategy expression, without repository fixtures/profiles."""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest

from llm_simulator import (
    Cache, InvalidInput, ModelPlan, Scheduling, Select, Simulator, StrategyInspection,
    TensorSelect, Transport,
)
from llm_simulator.demo import make_demo


def caller_tempdir():
    """A temporary directory on the caller's volume.

    One test below asserts that a relative public path resolves against the
    caller's working directory. Windows has no relative path across volumes,
    so when the system temp directory sits on another drive (a checkout on
    ``G:`` with ``TEMP`` on ``C:``) no relative path can name it. The caller's
    own directory can always be addressed relatively; a run that cannot write
    there falls back to the system temp directory.
    """
    try:
        return tempfile.TemporaryDirectory(dir=os.getcwd(), prefix=".tmp-strategy-ir-")
    except OSError:
        return tempfile.TemporaryDirectory()


class StrategyIRLibraryTests(unittest.TestCase):
    def test_public_scoped_strategy_drives_the_shared_runtime(self):
        original, _ = make_demo("dense")
        simulator = Simulator(model=original.model, workload=original.workload,
            hardware=replace(original.hardware, transfer_bandwidth_bytes_per_second=1e9,
                             transfer_latency_ns=0), cost_model=original.cost_model)
        services = dict(format_version="tensor-services-0.1", pcie=dict(
            bandwidth_bytes_per_second=1_000_000_000, ce_bytes_per_second=1_000_000_000,
            sm_bytes_per_second=500_000_000,
            arbitration="max_min_fair", timing_scope="payload_only_no_launch_latency"), slots=[])
        with caller_tempdir() as folder:
            service_file = Path(folder) / "services.json"
            service_file.write_text(json.dumps(services), encoding="utf-8", newline="\n")
            reports = []
            def configure(plan):
                for layer, engine in (("layer0", "ce"), ("layer1", "sm")):
                    with plan.module(layer, layer_id=layer):
                        plan.place("dense_ffn", "gpu")
                        plan.transfer(engine=engine)
                        plan.prepare(destination="dram", trigger="spare")
                plan.place("dense_ffn", "cpu", select=Select(layer_id="layer0", phase="decode"), priority=1)
                reports.append(plan.inspect())
            schedule = Scheduling(default_device="gpu", release="last_use",
                                  cache=Cache(rule="lru", budget_bytes=1024 * 1024),
                                  transport=Transport(engine="ce"))
            # Relative public paths refer to the caller, not generated inputs.
            try:
                public = Path(os.path.relpath(service_file))
            except ValueError:            # system temp on another volume
                public = service_file
            with simulator.compile(scheduling=schedule, configure=configure,
                                   services_file=public) as compiled:
                self.assertIsInstance(reports[0], StrategyInspection)
                self.assertFalse(reports[0].executable)
                result = compiled.run()
                self.assertTrue(result.success, result.summary())
                by_weight = {tid: layer for tid, layer, _, _ in result.raw[0].decisions[0].observation.weight_catalogue}
                seen = set()
                for row in result.raw[0].decisions:
                    choice = row.decision
                    if choice.kind == "COPY" and choice.dst == "vram" and choice.target_id in by_weight:
                        layer = by_weight[choice.target_id]
                        self.assertEqual(choice.transfer_engine, "ce" if layer == "layer0" else "sm")
                        seen.add(layer)
                self.assertEqual(seen, {"layer0", "layer1"})
                self.assertEqual(compiled.ir_document()["format_version"], "hierarchical-strategy-0.5")
                self.assertEqual(len(compiled.ir_document()["scoped_transport"]), 2)
                service_file.write_bytes(service_file.read_bytes() + b"\n")
                with self.assertRaises(InvalidInput):
                    compiled.run()

    def test_unpriced_public_model_keeps_inspection_separate_from_execution(self):
        simulator, schedule = make_demo("dense")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs = simulator.cost_model.build_inputs(root, simulator.model, simulator.hardware, simulator.workload)
            # Only this test's generated input points at an absent measurement.
            # Nothing is removed, approximated, or changed in retained profiles.
            for path in root.glob("*.json"):
                value = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(value, dict) and "costs_file" in value:
                    value["costs_file"] = "not-measured.json"
                    path.write_text(json.dumps(value), encoding="utf-8", newline="\n")
            plan = ModelPlan(id="unpriced-public", model_file=inputs["model_file"],
                             model_kind=inputs["model_kind"], environment_file=inputs["environment_file"],
                             scheduling=schedule, base_dir=root)
            report = plan.inspect()
            self.assertEqual(report.declarations.status, "valid")
            self.assertEqual(report.cost_coverage.status, "missing")
            self.assertFalse(report.executable)
            self.assertIn("Cost coverage: missing", plan.explain())
            with self.assertRaises(FileNotFoundError):
                plan.compile()


if __name__ == "__main__":
    unittest.main()
