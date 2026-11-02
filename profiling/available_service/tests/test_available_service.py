from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
for directory in (ROOT, ROOT / "modeling"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from llm_infer_model.performance import Parameter, ParameterBundle, Graph, Task, WorkDemand, ResourcePoint, ResourceTimeline
from llm_infer_model.performance.available_resources import AvailableResources, evaluate_available_resources
from profiling.available_service import freeze_efficiencies, load_efficiency_bindings
from profiling.available_service.calibration import digest, encoded


class AvailableServiceCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="available-service-test-", dir=ROOT / "build")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.context = {"hardware": "declared fixture", "shape": [2, 3], "precision": "F32"}
        self.sources, self.available, self.supply_sources = {}, {}, {}
        self.observations = []
        for i in range(3):
            run = "profile-" + str(i)
            available = AvailableResources(ParameterBundle((Parameter("available", 1_000_000_000,
                "flop/s", "declared", (), "fixture"),)),
                ResourceTimeline((ResourcePoint(10, {"available": 0}), ResourcePoint(15, {"available": 2_000_000_000}))),
                100, "fixture-ns", "start", "independent_measurement", ("independent fixture receipt",), "fixture supply")
            self.available[run] = available
            self.observations.extend([dict(observation_id=run+":"+str(j), target_parameter="operator", supply_parameter="available",
                work=10, work_unit="flop", begin_ns=0, end_ns=20, timing_scope="device_compute_active",
                clock_id="fixture-ns", clock_origin="start", supply_scope="fixture supply", context=self.context,
                run_id=run, process_id=100+i, seed=10+i, split="profile",
                started_at_utc="2026-10-22T00:00:00Z", ended_at_utc="2026-10-22T00:01:00Z",
                overheads_excluded=True, other_resources_controlled=True, numerical_verified=True,
                evidence=["physical active interval; fixture only"]) for j in range(2)])
        self.write_sources()
        self.write_supplies()

    def write_sources(self):
        p = self.path / "stages.json"
        p.write_text(json.dumps(dict(schema="available-service-stage-observations-1", observations=self.observations)),
                     encoding="utf-8", newline="\n")
        self.sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()}

    def write_supplies(self, *, derived=False):
        for run, available in self.available.items():
            p = self.path / (run + "-supply.json")
            p.write_text(json.dumps(dict(schema="independent-available-resource-observation-1",
                available_resources=available.to_dict(), measurement_scope="external_resources_available_to_llm",
                derived_from_target_timing=derived, independence_evidence=["declared independent fixture protocol"])),
                encoding="utf-8", newline="\n")
            self.supply_sources[run] = dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())

    def freeze(self, **kwargs):
        return freeze_efficiencies(self.sources, self.available, context=self.context,
            frozen_at_utc="2026-10-22T01:00:00Z", supply_source_files=self.supply_sources, **kwargs)

    def test_exact_integral_with_rate_changes_and_pause(self):
        frozen = self.freeze()
        self.assertEqual(frozen["observation_estimates"][0]["available_service_integral"], encoded(Fraction(20)))
        self.assertEqual(frozen["binding_parameters"][0]["efficiency"], encoded(Fraction(1, 2)))
        self.assertTrue(frozen["parameter_cells"][0]["accepted"])
        self.assertFalse(frozen["hardware_accuracy_qualified"])
        self.assertFalse(frozen["whole_layer_residual_fitted"])

    def test_model_computes_duration_using_measured_efficiency(self):
        bindings = load_efficiency_bindings(self.freeze())
        graph = Graph((Task("op", "gpu", (WorkDemand("operator", 10, "flop"),)),))
        result = evaluate_available_resources(graph, ParameterBundle(()), self.available["profile-0"], bindings)
        self.assertEqual(result.duration_ns, 20)
        self.assertFalse(self.freeze()["performance_predictions_computed_here"])

    def test_frozen_efficiency_transfers_same_work_to_new_declared_trajectory(self):
        frozen = self.freeze()
        original = deepcopy(frozen)
        bindings = load_efficiency_bindings(frozen)
        self.assertEqual(bindings[0].efficiency, Fraction(1, 2))
        scenario = AvailableResources(
            ParameterBundle((Parameter("available", 4_000_000_000, "flop/s", "declared", (),
                                       "new declared resource trajectory; same operator context"),)),
            ResourceTimeline((ResourcePoint(2, {"available": 0}),
                              ResourcePoint(7, {"available": 2_000_000_000}))),
            100, "fixture-ns", "start", "declared_scenario",
            ("independent hand-derived scenario; no hardware accuracy claim",), "fixture supply")
        self.assertNotEqual(scenario.timeline, self.available["profile-0"].timeline)
        self.assertEqual(scenario.source_kind, "declared_scenario")
        graph = Graph((Task("op", "gpu", (WorkDemand("operator", 10, "flop"),)),))
        # eta=1/2: first 2 ns deliver 4 flops, 2..7 delivers zero,
        # then 6 remaining flops require 6 ns at 1 flop/ns: finish at 13 ns.
        result = evaluate_available_resources(graph, ParameterBundle(()), scenario, bindings)
        self.assertEqual(result.duration_ns, 13)
        self.assertEqual(frozen, original)
        self.assertEqual(bindings, load_efficiency_bindings(frozen))

    def test_inclusive_wall_scope_rejected(self):
        self.observations[0]["timing_scope"] = "whole_action_completed"
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "inclusive completed"):
            self.freeze()

    def test_declared_source_tag_is_not_independent_measurement(self):
        for run, available in list(self.available.items()):
            d = available.to_dict(); d["source_kind"] = "declared_scenario"
            self.available[run] = AvailableResources.from_dict(d)
        self.write_supplies()
        frozen = self.freeze()
        self.assertIsNone(frozen["binding_parameters"][0]["efficiency"])
        self.assertEqual(frozen["parameter_cells"][0]["candidate_efficiency"], encoded(Fraction(1, 2)))

    def test_missing_supply_source_receipt_cannot_qualify(self):
        self.supply_sources = {}
        frozen = self.freeze()
        self.assertIsNone(load_efficiency_bindings(frozen)[0].efficiency)
        self.assertIn("independent_supply_source_receipt_missing", frozen["observation_estimates"][0]["evidence_issues"])

    def test_supply_derived_from_target_time_cannot_qualify(self):
        self.write_supplies(derived=True)
        frozen = self.freeze()
        self.assertIsNone(load_efficiency_bindings(frozen)[0].efficiency)

    def test_unknown_physical_work_remains_unknown(self):
        self.observations[0]["work"] = None
        self.write_sources()
        frozen = self.freeze()
        self.assertIsNone(frozen["observation_estimates"][0]["candidate_efficiency"])
        self.assertIsNone(load_efficiency_bindings(frozen)[0].efficiency)

    def test_efficiency_above_one_is_not_clipped(self):
        for row in self.observations:
            row["work"] = 40
        self.write_sources()
        frozen = self.freeze()
        self.assertEqual(frozen["parameter_cells"][0]["candidate_efficiency"], encoded(Fraction(2)))
        self.assertIsNone(load_efficiency_bindings(frozen)[0].efficiency)

    def test_fixed_overhead_not_removed_stays_unqualified(self):
        self.observations[0]["overheads_excluded"] = False
        self.write_sources()
        self.assertIsNone(load_efficiency_bindings(self.freeze())[0].efficiency)

    def test_other_resource_attribution_unknown_stays_unqualified(self):
        self.observations[0]["other_resources_controlled"] = None
        self.write_sources()
        self.assertIsNone(load_efficiency_bindings(self.freeze())[0].efficiency)

    def test_numerics_unknown_stays_unqualified(self):
        self.observations[0]["numerical_verified"] = None
        self.write_sources()
        self.assertIsNone(load_efficiency_bindings(self.freeze())[0].efficiency)

    def test_unknown_external_rate_is_not_zero(self):
        for run, available in list(self.available.items()):
            d=available.to_dict(); d["supply"]["parameters"][0]["value"] = None
            self.available[run] = AvailableResources.from_dict(d)
        self.write_supplies()
        self.assertIsNone(load_efficiency_bindings(self.freeze())[0].efficiency)

    def test_clock_origin_and_context_must_match(self):
        for field, value in (("clock_id", "other"), ("clock_origin", "other"), ("supply_scope", "other"),
                             ("context", {"hardware": "other"})):
            with self.subTest(field=field):
                original = self.observations[0][field]
                self.observations[0][field] = value; self.write_sources()
                with self.assertRaises(ValueError): self.freeze()
                self.observations[0][field] = original
        self.write_sources()

    def test_heldout_cannot_fit(self):
        self.observations[0]["split"] = "heldout"; self.write_sources()
        with self.assertRaisesRegex(ValueError, "heldout"):
            self.freeze()

    def test_source_change_detected_on_load(self):
        frozen = self.freeze()
        Path(next(iter(self.sources))).write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA"):
            load_efficiency_bindings(frozen)

    def test_supply_source_change_detected_on_load(self):
        frozen = self.freeze()
        Path(self.supply_sources["profile-0"]["path"]).write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA"):
            load_efficiency_bindings(frozen)

    def test_correct_sha_with_utilization_scope_is_rejected(self):
        receipt = self.supply_sources["profile-0"]
        p = Path(receipt["path"]); d = json.loads(p.read_text(encoding="utf-8"))
        d["measurement_scope"] = "total_gpu_utilization"
        p.write_text(json.dumps(d), encoding="utf-8", newline="\n")
        receipt["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "not background requests or utilization"):
            self.freeze()

    def test_correct_sha_with_different_supply_payload_is_rejected(self):
        receipt = self.supply_sources["profile-0"]
        p = Path(receipt["path"]); d = json.loads(p.read_text(encoding="utf-8"))
        d["available_resources"]["valid_until_ns"] += 1
        p.write_text(json.dumps(d), encoding="utf-8", newline="\n")
        receipt["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "does not reconstruct"):
            self.freeze()

    def test_resigned_edited_binding_does_not_reconstruct(self):
        frozen = self.freeze()
        frozen["binding_parameters"][0]["efficiency"] = encoded(Fraction(9, 10))
        frozen["artifact_sha256"] = digest({k: v for k, v in frozen.items() if k != "artifact_sha256"})
        with self.assertRaisesRegex(ValueError, "reconstruct"):
            load_efficiency_bindings(frozen)

    def test_multiple_runs_cannot_be_one_renamed_process(self):
        for row in self.observations:
            row["process_id"] = 100
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "overlapping PID"):
            self.freeze()

    def test_sample_gates_preserve_unknown_instead_of_relaxing(self):
        frozen = self.freeze(minimum_samples_per_run=3)
        self.assertIsNone(load_efficiency_bindings(frozen)[0].efficiency)
        self.assertEqual(len(frozen["observation_estimates"]), 6)

    def test_float_work_rejected(self):
        self.observations[0]["work"] = 10.0; self.write_sources()
        with self.assertRaisesRegex(ValueError, "floats"):
            self.freeze()


if __name__ == "__main__":
    unittest.main()
