"""Finite coverage, accounting, paired controls and retained case evidence."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.hardware_system_campaign import (
    CaseSpec, audit_coverage, declared_jobs, execution_source_manifest,
    physical_cache_budget, run_case, verify_controls,
)
from experiments.hardware_system_protocol import NPUProxyCosts, candidate_spec
from llm_simulator import AnalyticalCosts, Device, Hardware, Model, Workload


class HardwareSystemCampaignTests(unittest.TestCase):
    def test_declared_full_factorial_is_480_with_two_paired_proxies(self):
        jobs = declared_jobs(proxy_system_ids=("core_ultra258v", "snapdragon8elite_phone"))
        self.assertEqual(len(jobs), 480)
        self.assertEqual(len({job.key for job in jobs}), 480)
        self.assertEqual(sum(job.accelerator == "gpu" for job in jobs), 288)
        self.assertEqual(sum(job.accelerator == "npu_proxy" for job in jobs), 192)
        for system in ("core_ultra258v", "snapdragon8elite_phone"):
            self.assertEqual(sum(job.system_id == system for job in jobs), 128)

    def test_selection_rejects_unknown_duplicate_or_unpaired_cases(self):
        for kwargs in (dict(system_ids=[]), dict(system_ids=["unknown"]),
                dict(system_ids=["m4pro_mini", "m4pro_mini"]),
                dict(system_ids=["m4pro_mini"], proxy_system_ids=["core_ultra258v"]),
                dict(proxy_system_ids=["rtx5090_desktop"]),
                dict(workload_ids=["unknown"]), dict(candidates=["e9-s0"]),
                dict(candidates=["e0-s0", "e0-s0"])):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                declared_jobs(**kwargs)

    def test_proxy_variants_have_stable_safe_paths(self):
        spec = CaseSpec("x", "workload", "e0-s0", "npu_proxy", .5)
        self.assertEqual(spec.variant_id, "npu-proxy-a0p5")
        self.assertEqual(spec.key, ("x", "npu-proxy-a0p5", "workload", "e0-s0"))
        for args in (("../escape", "w", "e0-s0"),
                ("x", "w", "e0-s0", "gpu", 1),
                ("x", "w", "e0-s0", "npu_proxy", True),
                ("x", "w", "e0-s0", "npu_proxy", .7)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                CaseSpec(*args)

    def control_rows(self):
        base = dict(system_id="system", variant_id="gpu", workload_id="w",
                    candidate_id="e0-s0", accelerator="gpu", npu_time_multiplier=None,
                    success=False, completed_makespan_ns=None, status="capacity_error",
                    event_sha256="identical")
        proxy = dict(base, accelerator="npu_proxy", npu_time_multiplier=1,
                     variant_id="npu-proxy-a1")
        return [base, proxy]

    def test_controls_accept_matching_failures_but_reject_different_events(self):
        rows = self.control_rows()
        self.assertTrue(verify_controls(rows)[0]["equivalent"])
        rows[1]["event_sha256"] = "different"
        with self.assertRaises(ValueError):
            verify_controls(rows)
        with self.assertRaises(ValueError):
            verify_controls(rows[1:])

    def test_all_cpu_is_control_at_any_proxy_multiplier(self):
        rows = self.control_rows()
        for row in rows:
            row["candidate_id"] = "e1-s1"
        rows[1].update(npu_time_multiplier=2, variant_id="npu-proxy-a2")
        self.assertEqual(verify_controls(rows)[0]["control"], "all_cpu_independent_of_proxy")

    def test_coverage_rejects_duplicates_missing_and_changed_shared_inputs(self):
        jobs = [CaseSpec("system", "w", "e0-s0"), CaseSpec("system", "w", "e1-s0")]
        rows = [dict(system_id=job.system_id, variant_id=job.variant_id,
            workload_id=job.workload_id, candidate_id=job.candidate_id,
            success=False, completed_makespan_ns=None, workload_hardware_sha256="same") for job in jobs]
        audit_coverage(rows, jobs)
        for bad in (rows[:1], rows + [rows[0]]):
            with self.assertRaises(ValueError):
                audit_coverage(bad, jobs)
        changed = deepcopy(rows)
        changed[1]["workload_hardware_sha256"] = "different"
        with self.assertRaises(ValueError):
            audit_coverage(changed, jobs)

    def test_execution_manifest_includes_proxy_and_all_formula_sources(self):
        manifest = execution_source_manifest()
        for name in ("experiments/hardware_system_protocol.py",
                "experiments/hardware_system_catalogue.py",
                "experiments/hardware_system_campaign.py",
                "src/llm_simulator/analytical.py",
                "modeling/llm_infer_model/performance/analytical.py"):
            self.assertIn(name, manifest)
            self.assertEqual(len(manifest[name]), 64)

    def tiny_inputs(self, case):
        model = Model.dense(layers=2, hidden_size=16, intermediate_size=32,
                            heads=4, kv_heads=2)
        if case.system_id == "tiny-separate":
            hardware = Hardware(cpu=Device(1e9, 1e9, 5), gpu=Device(2e9, 2e9, 10),
                dram_capacity_bytes=1024**2, vram_capacity_bytes=512*1024,
                transfer_bandwidth_bytes_per_second=1e9)
        else:
            hardware = Hardware.shared_memory(cpu=Device(1e9, 1e9, 5),
                gpu=Device(2e9, 2e9, 10), capacity_bytes=1024 ** 2,
                runtime_reserved_bytes=64*1024)
        workload = Workload(prefill_tokens=1)
        costs = (AnalyticalCosts() if case.accelerator == "gpu" else
                 NPUProxyCosts(alpha=case.npu_time_multiplier))
        return model, hardware, workload, candidate_spec("dense", case.candidate_id,
            cache_budget_bytes=physical_cache_budget(hardware)), costs

    def test_cache_binding_uses_available_physical_capacity(self):
        for system in ("tiny-shared", "tiny-separate"):
            case = CaseSpec(system, "w", "e0-s1")
            _, hardware, _, scheduling, _ = self.tiny_inputs(case)
            expected = 512*1024 if system == "tiny-separate" else 960*1024
            self.assertEqual(physical_cache_budget(hardware), expected)
            self.assertEqual(scheduling.cache.budget_bytes, expected)

    def test_public_runtime_retains_exact_inputs_events_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            cases = [CaseSpec(system, "tiny-workload", candidate, accelerator, alpha)
                     for system in ("tiny-shared", "tiny-separate")
                     for candidate in ("e0-s0", "e0-s1")
                     for accelerator, alpha in (("gpu", None), ("npu_proxy", 1))]
            with patch("experiments.hardware_system_campaign._materialize", side_effect=self.tiny_inputs):
                rows = [run_case(case, output) for case in cases]
            self.assertTrue(all(row["success"] for row in rows))
            self.assertEqual(len(verify_controls(rows)), 4)
            self.assertTrue(all(control["equivalent"] for control in verify_controls(rows)))
            for row in rows:
                if row["memory_mode"] == "shared":
                    self.assertEqual(row["h2d_bytes"], 0)
                    self.assertEqual(row["d2h_bytes"], 0)
                    self.assertGreater(row["share_count"], 0)
                else:
                    self.assertGreater(row["h2d_bytes"], 0)
                    self.assertEqual(row["share_count"], 0)
                folder = output / row["artifact_directory"]
                hashes = json.loads((folder / "artifacts.json").read_text(encoding="utf-8"))
                for required in ("declarations.json", "events.json", "compiled-ir.json",
                        "operation-catalogue.json", "tensor-catalogue.json", "decision-log.json"):
                    self.assertIn(required, hashes)
                for filename, digest in hashes.items():
                    self.assertEqual(hashlib.sha256((folder / filename).read_bytes()).hexdigest(), digest)
                self.assertNotIn(b"\r\n", (folder / "summary.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
