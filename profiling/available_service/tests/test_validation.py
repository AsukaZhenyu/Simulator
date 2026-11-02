"""Synthetic protocol checks, not hardware measurements or accuracy evidence."""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
for directory in (ROOT, ROOT / "modeling"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import test_available_service as fixtures
from llm_infer_model.performance import Graph, Task, WorkDemand, Parameter, ParameterBundle, ResourcePoint
from llm_infer_model.performance.available_resources import AvailableResources
from profiling.available_service.validation import make_barrier, evaluate_available_service
from profiling.available_service.calibration import digest


class AvailableServiceValidationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.AvailableServiceCalibrationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.directory = self.fixture.path
        self.frozen = self.fixture.freeze()
        self.frozen_path = self.directory / "frozen.json"
        self.write(self.frozen_path, self.frozen)
        self.graph = Graph((Task("op", "gpu", (WorkDemand("operator", 10, "flop"),)),))
        self.fixed = ParameterBundle(())
        self.barrier = self.make_barrier()
        self.available = {}
        self.supplies = {}
        self.references = []
        self.numeric = self.directory / "numeric.json"
        self.numeric_receipt = self.write(self.numeric, {"passed": True, "scope": "synthetic oracle fixture"})
        for i in range(3):
            run = "heldout-" + str(i)
            # A different clock-labelled observation stream, supplied separately
            # from the target references. No target timing is used to build it.
            self.available[run] = AvailableResources.from_dict(self.fixture.available["profile-0"].to_dict())
            self.references.append(dict(observation_id=run+":0", run_id=run, process_id=200+i,
                seed=20+i, split="heldout", begin_ns=0, end_ns=20,
                clock_id="fixture-ns", clock_origin="start", context=self.fixture.context,
                graph_sha256=digest(self.graph.to_dict()), signature_id="same-10-flop-op",
                started_at_utc="2026-10-22T03:00:00Z", ended_at_utc="2026-10-22T04:00:00Z",
                reference_timing_scope="whole_execution_completed", numerical_receipt=self.numeric_receipt,
                state={"external_supply_fixture": "dynamic"}, evidence=["synthetic exact interval"]))
        self.write_references()
        self.write_supplies()

    def write(self, path, value):
        path.write_text(json.dumps(value), encoding="utf-8", newline="\n")
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def write_references(self):
        receipt = self.write(self.directory / "heldout.json", dict(
            schema="available-service-heldout-observations-1", observations=self.references))
        self.reference_sources = {receipt["path"]: receipt["sha256"]}

    def write_supplies(self, derived=False):
        for run, available in self.available.items():
            self.supplies[run] = self.write(self.directory / (run + "-supply.json"), dict(
                schema="independent-available-resource-observation-1", available_resources=available.to_dict(),
                measurement_scope="external_resources_available_to_llm", derived_from_target_timing=derived,
                independence_evidence=["synthetic independent source fixture only"]))

    def make_barrier(self):
        return make_barrier(self.frozen_path, graph=self.graph, fixed_parameters=self.fixed,
            signature_id="same-10-flop-op", created_at_utc="2026-10-22T02:00:00Z",
            protocol=dict(minimum_runs=3, minimum_samples_per_run=1,
                          median_relative_error_max=.2, p90_relative_error_max=.2))

    def evaluate(self, **kwargs):
        return evaluate_available_service(self.frozen, self.reference_sources, self.available,
            self.supplies, self.graph, self.fixed, barrier=self.barrier, **kwargs)

    def test_independent_source_protocol_calls_modeling_without_refit(self):
        original = deepcopy(self.frozen)
        result = self.evaluate()
        self.assertEqual(result["primary_summary"]["predictable_count"], 3)
        self.assertEqual(result["primary_summary"]["process_balanced_median_absolute_relative_error"], 0)
        self.assertTrue(result["accepted"])
        self.assertTrue(result["conditional_on_observed_start"])
        self.assertFalse(result["heldout_used_for_fit"])
        self.assertEqual(self.frozen, original)
        self.assertTrue(all(p["predicted_ns"] == 20 for p in result["predictions"]))

    def test_barrier_precedes_actual_profile_freeze_rejected(self):
        with self.assertRaisesRegex(ValueError, "precedes"):
            make_barrier(self.frozen_path, graph=self.graph, fixed_parameters=self.fixed,
                signature_id="op", created_at_utc="2026-10-21T00:00:00Z")

    def test_changed_saved_frozen_file_rejected(self):
        self.frozen_path.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            self.evaluate()

    def test_changed_graph_or_fixed_parameters_after_barrier_rejected(self):
        self.graph = Graph((Task("op", "gpu", (WorkDemand("operator", 11, "flop"),)),))
        with self.assertRaisesRegex(ValueError, "graph differs"):
            self.evaluate()

    def test_heldout_starts_before_barrier_rejected(self):
        self.references[0]["started_at_utc"] = "2026-10-22T01:30:00Z"
        self.write_references()
        with self.assertRaisesRegex(ValueError, "after frozen barrier"):
            self.evaluate()

    def test_profile_run_seed_or_pid_reuse_rejected(self):
        for key, value in (("run_id", "profile-0"), ("seed", 10), ("process_id", 100)):
            original = self.references[0][key]
            self.references[0][key] = value
            self.write_references()
            with self.assertRaisesRegex(ValueError, "identity reused"):
                self.evaluate()
            self.references[0][key] = original

    def test_reference_clock_mismatch_rejected(self):
        self.references[0]["clock_origin"] = "different-origin"
        self.write_references()
        with self.assertRaisesRegex(ValueError, "clocks/origins"):
            self.evaluate()

    def test_reference_source_tampering_rejected(self):
        (self.directory / "heldout.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            self.evaluate()

    def test_duplicate_observation_or_overlapping_invocations_rejected(self):
        repeated = deepcopy(self.references[0])
        self.references.append(repeated)
        self.write_references()
        with self.assertRaisesRegex(ValueError, "duplicate heldout observation"):
            self.evaluate()
        repeated["observation_id"] += ":renamed"
        self.write_references()
        with self.assertRaisesRegex(ValueError, "invocations overlap"):
            self.evaluate()

    def test_distinct_heldout_run_names_cannot_reuse_pid(self):
        self.references[1]["process_id"] = self.references[0]["process_id"]
        self.write_references()
        with self.assertRaisesRegex(ValueError, "distinct PIDs"):
            self.evaluate()

    def test_supply_receipt_tampering_rejected(self):
        Path(self.supplies["heldout-0"]["path"]).write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            self.evaluate()

    def test_unknown_binding_primary_does_not_use_candidate_silently(self):
        for observation in self.fixture.observations:
            observation["overheads_excluded"] = False
        self.fixture.write_sources()
        self.frozen = self.fixture.freeze()
        self.write(self.frozen_path, self.frozen)
        self.barrier = self.make_barrier()
        result = self.evaluate(diagnostic_candidates=True)
        self.assertEqual(result["primary_summary"]["predictable_count"], 0)
        self.assertEqual(result["diagnostic_comparison"]["summary"]["predictable_count"], 3)
        self.assertTrue(all(p["predicted_ns"] is None and p["diagnostic_predicted_ns"] == 20
                            for p in result["predictions"]))
        self.assertFalse(result["accepted"])
        self.assertFalse(result["diagnostic_comparison"]["accepted"])

    def test_observed_diagnostic_supply_never_primary_acceptance(self):
        for run, available in list(self.available.items()):
            document = available.to_dict()
            document["source_kind"] = "observed_diagnostic"
            self.available[run] = AvailableResources.from_dict(document)
        self.write_supplies()
        result = self.evaluate(diagnostic_candidates=True)
        self.assertEqual(result["primary_summary"]["predictable_count"], 0)
        self.assertEqual(result["diagnostic_comparison"]["summary"]["predictable_count"], 3)
        self.assertFalse(result["accepted"])

    def test_unrelated_unknown_parameter_does_not_block_predeclared_graph(self):
        extra = []
        for row in self.fixture.observations:
            other = deepcopy(row)
            other["observation_id"] += ":unused"
            other["target_parameter"] = "unused-operator"
            other["work"] = None
            extra.append(other)
        self.fixture.observations.extend(extra)
        self.fixture.write_sources()
        self.frozen = self.fixture.freeze()
        self.write(self.frozen_path, self.frozen)
        self.barrier = self.make_barrier()
        result = self.evaluate()
        self.assertTrue(result["accepted"])
        self.assertEqual(result["primary_summary"]["predictable_count"], 3)
        self.assertEqual(result["parameter_evidence_issues"], [])
        self.assertEqual(len(result["unused_profile_parameter_evidence_issues"]), 1)

    def test_target_derived_supply_never_primary_acceptance(self):
        self.write_supplies(derived=True)
        result = self.evaluate(diagnostic_candidates=True)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["primary_summary"]["predictable_count"], 0)
        self.assertTrue(all("target_derived" in p["model_problem"] for p in result["predictions"]))

    def test_unknown_horizon_retains_observation_and_coverage_failure(self):
        for run, available in list(self.available.items()):
            document = available.to_dict()
            document["valid_until_ns"] = 19
            self.available[run] = AvailableResources.from_dict(document)
        self.write_supplies()
        result = self.evaluate()
        self.assertEqual(result["primary_summary"]["sample_count"], 3)
        self.assertEqual(result["primary_summary"]["predictable_count"], 0)
        self.assertFalse(result["accepted"])

    def test_fixed_overhead_qualification_reported_explicitly(self):
        self.fixed = ParameterBundle((Parameter("host", 2, "ns", "declared", (), "fixture overhead"),))
        self.graph = Graph((Task("host", "host", fixed_parameter="host"),
            Task("op", "gpu", (WorkDemand("operator", 10, "flop"),), ("host",))))
        self.barrier = self.make_barrier()
        for row in self.references:
            row["graph_sha256"] = digest(self.graph.to_dict())
            row["end_ns"] = 21
        self.write_references()
        result = self.evaluate()
        self.assertFalse(result["fixed_parameter_qualification"]["all_required_fixed_parameters_qualified"])
        self.assertIn("fixed_overhead_qualification_unestablished", result["qualification_issues"])
        self.assertFalse(result["accepted"])

    def test_bad_numerical_receipt_retained_unaccepted(self):
        self.numeric_receipt = self.write(self.numeric, {"passed": False})
        for row in self.references:
            row["numerical_receipt"] = self.numeric_receipt
        self.write_references()
        result = self.evaluate()
        self.assertEqual(result["primary_summary"]["sample_count"], 3)
        self.assertFalse(result["accepted"])

    def test_error_state_and_crossing_metrics_retained(self):
        for i, row in enumerate(self.references):
            row["end_ns"] = 40
            row["state"] = {"state": i}
        self.write_references()
        result = self.evaluate()
        self.assertEqual(result["primary_summary"]["pooled_p90_absolute_relative_error"], .5)
        self.assertEqual(len(result["by_state"]), 3)
        self.assertEqual(result["by_supply_crossing"]["true"]["sample_count"], 3)
        self.assertFalse(result["accepted"])

    def test_identical_supply_update_is_not_a_state_crossing(self):
        for run, available in list(self.available.items()):
            document = available.to_dict()
            document["timeline"]["points"].append(ResourcePoint(35, {"available": 2_000_000_000}).to_dict())
            self.available[run] = AvailableResources.from_dict(document)
        for row in self.references:
            row["begin_ns"], row["end_ns"] = 30, 40
        self.write_references()
        self.write_supplies()
        result = self.evaluate()
        self.assertTrue(result["accepted"])
        self.assertEqual(result["by_supply_crossing"]["false"]["sample_count"], 3)


if __name__ == "__main__":
    unittest.main()
