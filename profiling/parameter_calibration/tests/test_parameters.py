"""Synthetic evidence guards; no GPU, fitting library, or native process."""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from llm_infer_model import performance as modeling
from profiling.parameter_calibration.observations import (
    ParameterObservation, SCHEMA, canonical, derive_observation, digest,
    load_observations)
from profiling.parameter_calibration.parameters import estimate_parameters, load_frozen_parameters
from profiling.parameter_calibration.evaluator import evaluate_calibration
from profiling.parameter_calibration.__main__ import main

CONTEXT = {"device": "synthetic_gpu", "backend": "synthetic", "dtype": "f32", "shape": [8, 8]}


def observation(run=0, sample=0, value=1000, **updates):
    row = dict(observation_id=f"profile-{run}-{sample}", parameter_name="gpu_rate",
        kind="compute_throughput", value=Fraction(value) if value is not None else None,
        unit="flop/s", timing_scope="device_compute_active", context_json=canonical(CONTEXT).decode(),
        run_id=f"profile-{run}", process_id=1000 + run, seed=run,
        split="profile", started_at_utc="2026-10-18T00:00:00+00:00",
        ended_at_utc="2026-10-18T00:00:01+00:00", source_path="synthetic-source.json",
        source_sha256="a" * 64, evidence=("synthetic fixture, no hardware claim",),
        source_verified=True, numerical_passed=True)
    row.update(updates)
    return ParameterObservation(**row)


def freeze(rows=None, **kwargs):
    context = kwargs.pop("context", CONTEXT)
    return estimate_parameters(rows if rows is not None else [observation(i) for i in range(3)],
        context=context, frozen_at_utc="2026-10-18T00:00:10+00:00", **kwargs)


def heldout(run=0, value=1_000_000_000, **updates):
    return observation(run, value=value, observation_id=f"heldout-{run}",
        run_id=f"heldout-{run}", process_id=2000 + run, seed=100 + run,
        split="heldout", kind="reference_duration", parameter_name="graph_reference_ns",
        unit="ns", timing_scope="whole_iteration_completed",
        started_at_utc="2026-10-18T00:00:11+00:00",
        ended_at_utc="2026-10-18T00:00:12+00:00", **updates)


def graph():
    return modeling.Graph((modeling.Task("compute", "gpu", (modeling.WorkDemand("gpu_rate", 1000, "flop"),)),))


CONTEXT.update(graph_sha256=digest(graph().to_dict()), reference_timing_scope="whole_iteration_completed")


class ObservationTests(unittest.TestCase):
    def test_roundtrip_preserves_exact_fraction_and_context(self):
        row = observation(value=Fraction(7, 3))
        restored = ParameterObservation.from_dict(row.to_dict())
        self.assertEqual(restored.value, Fraction(7, 3))
        self.assertEqual(restored.context, CONTEXT)
        self.assertFalse(restored.source_verified)

    def test_context_is_copied_out(self):
        row = observation()
        context = row.context
        context["device"] = "other"
        self.assertEqual(row.context, CONTEXT)

    def test_boolean_and_float_not_raw_numeric_evidence(self):
        for value in (True, 1.5, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(observation(), value=value)

    def test_total_pv_cannot_be_compute_throughput(self):
        with self.assertRaisesRegex(ValueError, "total/overlapping"):
            replace(observation(), timing_scope="whole_action_completed")

    def test_stream_elapsed_cannot_be_transfer_bandwidth(self):
        with self.assertRaises(ValueError):
            replace(observation(), kind="transfer_bandwidth", unit="byte/s",
                    timing_scope="overlapping_device_stream_interval")

    def test_host_overhead_requires_exclusive_completed_scope(self):
        with self.assertRaises(ValueError):
            replace(observation(), kind="host_fixed_overhead", unit="ns",
                    timing_scope="whole_action_completed")

    def test_units_are_not_guessed(self):
        with self.assertRaises(ValueError):
            replace(observation(), unit="GFLOPS")

    def test_compute_rate_derivation_exact(self):
        metadata = observation().to_dict()
        for key in ("value", "context"):
            metadata.pop(key)
        metadata["context_json"] = canonical(CONTEXT).decode()
        metadata["evidence"] = tuple(metadata["evidence"])
        result = derive_observation(work=7, duration_ns=3, **metadata)
        self.assertEqual(result.value, Fraction(7_000_000_000, 3))

    def test_capacity_not_slowdown(self):
        row = replace(observation(), kind="capacity", unit="byte", timing_scope="capacity_snapshot", value=0)
        self.assertEqual(row.value, 0)
        with self.assertRaises(ValueError):
            replace(row, unit="byte/s")

    def test_empty_or_non_utc_provenance_rejected(self):
        for updates in ({"evidence": ()}, {"source_sha256": "bad"}, {"context_json": "{}"},
                        {"started_at_utc": "2026-10-18T00:00:00"},
                        {"started_at_utc": "2026-10-18T08:00:00+08:00"},
                        {"process_id": True}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                replace(observation(), **updates)

    def test_ended_before_started_rejected(self):
        with self.assertRaises(ValueError):
            replace(observation(), ended_at_utc="2026-10-17T00:00:00Z")


class EstimationTests(unittest.TestCase):
    def test_bundle_uses_model_contract(self):
        document = freeze()
        bundle = load_frozen_parameters(document)
        self.assertIsInstance(bundle, modeling.ParameterBundle)
        self.assertEqual(bundle.require("gpu_rate", "flop/s"), 1000)
        self.assertFalse(document["performance_predictions_computed_here"])

    def test_exact_process_balanced_not_pooled_median(self):
        rows = [observation(0, i, 10) for i in range(100)]
        rows += [observation(1, value=20), observation(2, value=30)]
        document = freeze(rows, between_process_cv_max=1.0)
        cell = document["parameter_cells"][0]
        self.assertEqual(cell["process_balanced_median_exact"], {"numerator": 20, "denominator": 1})
        self.assertEqual(len(document["profile_observations"]), 102)
        self.assertEqual(cell["outliers_removed"], 0)

    def test_single_run_is_unknown(self):
        document = freeze([observation()])
        self.assertIn("insufficient_independent_runs", document["parameter_cells"][0]["evidence_issues"])
        with self.assertRaises(modeling.UnknownParameter):
            load_frozen_parameters(document).require("gpu_rate", "flop/s")

    def test_failed_cv_keeps_all_observations(self):
        rows = [observation(i, j, (1, 1000)[j]) for i in range(3) for j in range(2)]
        document = freeze(rows)
        self.assertEqual(len(document["profile_observations"]), 6)
        self.assertFalse(document["parameter_cells"][0]["accepted"])

    def test_unverified_source_or_numeric_is_unknown(self):
        for updates in ({"source_verified": False}, {"numerical_passed": None}, {"value": None}):
            with self.subTest(updates=updates):
                document = freeze([replace(observation(i), **updates) for i in range(3)])
                self.assertFalse(document["parameter_cells"][0]["accepted"])

    def test_heldout_never_estimates_parameter(self):
        with self.assertRaisesRegex(ValueError, "heldout/diagnostic"):
            freeze([heldout()])

    def test_context_mix_rejected(self):
        with self.assertRaisesRegex(ValueError, "context mismatch"):
            freeze([replace(observation(), context_json='{"device":"other"}')])

    def test_profile_after_freeze_rejected(self):
        with self.assertRaisesRegex(ValueError, "after the freeze"):
            freeze([replace(observation(), ended_at_utc="2026-10-18T00:01:00Z")])

    def test_renamed_run_not_independent(self):
        rows = [observation(0), replace(observation(1), process_id=1000, seed=0)]
        with self.assertRaisesRegex(ValueError, "not independent"):
            freeze(rows)

    def test_later_pid_reuse_with_distinct_seed_is_allowed(self):
        rows = [replace(observation(i), process_id=1000,
                        started_at_utc=f"2026-10-18T00:00:0{i * 2}Z",
                        ended_at_utc=f"2026-10-18T00:00:0{i * 2 + 1}Z") for i in range(3)]
        self.assertTrue(freeze(rows)["parameter_cells"][0]["accepted"])

    def test_same_pid_overlapping_lifetimes_not_independent_even_different_seeds(self):
        with self.assertRaisesRegex(ValueError, "overlapping run lifetimes"):
            freeze([replace(observation(i), process_id=1000) for i in range(3)])

    def test_reference_duration_never_enters_physical_bundle(self):
        rows = [replace(observation(i), kind="reference_duration", unit="ns", timing_scope="whole_action_completed") for i in range(3)]
        document = freeze(rows)
        self.assertEqual(load_frozen_parameters(document).parameters, ())
        self.assertTrue(document["parameter_cells"][0]["reference_only"])

    def test_digest_tamper_rejected(self):
        document = freeze()
        document["profile_seeds"] = [999]
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            load_frozen_parameters(document)

    def test_rehashed_statistic_tamper_still_rejected(self):
        document = freeze()
        document["parameter_cells"][0]["accepted"] = False
        document.pop("artifact_sha256")
        document["artifact_sha256"] = digest(document)
        with self.assertRaisesRegex(ValueError, "does not reconstruct"):
            load_frozen_parameters(document)

    def test_zero_host_overhead_valid_not_fabricated(self):
        rows = [replace(observation(i), parameter_name="host", kind="host_fixed_overhead", value=0,
                        unit="ns", timing_scope="host_exclusive_completed") for i in range(3)]
        self.assertEqual(load_frozen_parameters(freeze(rows)).require("host", "ns"), 0)


class EvaluationTests(unittest.TestCase):
    def test_calls_model_owned_evaluator_and_exact_match(self):
        with mock.patch.object(modeling, "evaluate", wraps=modeling.evaluate) as spy:
            result = evaluate_calibration(graph(), freeze(), [heldout(i) for i in range(3)], verify_sources=False)
        spy.assert_called_once()
        self.assertEqual(result["predicted_ns"], 1_000_000_000)
        self.assertEqual(result["relative_error"], 0)
        self.assertTrue(result["accepted"])

    def test_large_residual_is_not_automatically_missing_mechanism(self):
        result = evaluate_calibration(graph(), freeze(), [heldout(i, value=2_000_000_000) for i in range(3)], verify_sources=False)
        self.assertEqual(result["relative_error"], .5)
        self.assertEqual(result["residual_cause"], "unresolved")
        self.assertFalse(result["mechanism_missing_proven"])
        self.assertFalse(result["accepted"])

    def test_unknown_parameter_is_explicit_model_problem(self):
        result = evaluate_calibration(graph(), freeze([observation()]), [heldout(i) for i in range(3)], verify_sources=False)
        self.assertEqual(result["model_problem"]["type"], "UnknownParameter")
        self.assertIsNone(result["predicted_ns"])
        self.assertTrue(result["parameter_evidence_issues"])

    def test_leaked_seed_or_run_id_rejected(self):
        for updates in ({"seed": 0}, {"run_id": "profile-0"}):
            with self.subTest(updates=updates), self.assertRaisesRegex(ValueError, "disjoint"):
                evaluate_calibration(graph(), freeze(), [replace(heldout(), **updates)], verify_sources=False)

    def test_heldout_before_freeze_rejected(self):
        with self.assertRaisesRegex(ValueError, "before parameter freeze"):
            evaluate_calibration(graph(), freeze(), [replace(heldout(), started_at_utc="2026-10-18T00:00:01Z")], verify_sources=False)

    def test_action_time_cannot_compare_to_whole_graph(self):
        with self.assertRaisesRegex(ValueError, "matching completed"):
            evaluate_calibration(graph(), freeze(), [replace(heldout(), timing_scope="whole_action_completed")], verify_sources=False)

    def test_one_heldout_run_not_accepted(self):
        result = evaluate_calibration(graph(), freeze(), [heldout()], verify_sources=False)
        self.assertFalse(result["accepted"])
        self.assertIn("insufficient_independent_heldout_runs", result["measurement_evidence_issues"])

    def test_task_comparison_uses_model_task_duration(self):
        context = dict(CONTEXT, reference_timing_scope="whole_action_completed")
        profiles = [replace(observation(i), context_json=canonical(context).decode()) for i in range(3)]
        rows = [replace(heldout(i), timing_scope="whole_action_completed", context_json=canonical(context).decode()) for i in range(3)]
        result = evaluate_calibration(graph(), freeze(profiles, context=context), rows, task_id="compute", verify_sources=False)
        self.assertEqual(result["predicted_ns"], 1_000_000_000)
        self.assertTrue(result["accepted"])

    def test_multistage_action_compares_completed_action_to_model_graph(self):
        rows = [replace(observation(i), observation_id=f"host-{i}", parameter_name="host", kind="host_fixed_overhead",
            unit="ns", timing_scope="host_exclusive_completed", value=100) for i in range(3)]
        action = modeling.Graph((modeling.Task("host", "cpu", fixed_parameter="host"),
            modeling.Task("gpu", "gpu", (modeling.WorkDemand("gpu_rate", 1000, "flop"),), depends_on=("host",))))
        context = dict(CONTEXT, graph_sha256=digest(action.to_dict()), reference_timing_scope="whole_action_completed")
        profiles = [replace(row, context_json=canonical(context).decode()) for row in rows + [observation(i) for i in range(3)]]
        frozen = freeze(profiles, context=context)
        refs = [replace(heldout(i, value=1_000_000_100), timing_scope="whole_action_completed", context_json=canonical(context).decode()) for i in range(3)]
        result = evaluate_calibration(action, frozen, refs, verify_sources=False,
                                      reference_timing_scope="whole_action_completed")
        self.assertEqual(result["predicted_ns"], 1_000_000_100)
        self.assertTrue(result["accepted"])

    def test_explicit_action_boundary_rejects_iteration_reference(self):
        with self.assertRaisesRegex(ValueError, "timing scope disagrees"):
            evaluate_calibration(graph(), freeze(), [heldout()], verify_sources=False,
                                 reference_timing_scope="whole_action_completed")

    def test_unbound_graph_prediction_is_diagnostic_only(self):
        context = {key: value for key, value in CONTEXT.items() if key != "graph_sha256"}
        profiles = [replace(observation(i), context_json=canonical(context).decode()) for i in range(3)]
        refs = [replace(heldout(i), context_json=canonical(context).decode()) for i in range(3)]
        result = evaluate_calibration(graph(), freeze(profiles, context=context), refs, verify_sources=False)
        self.assertIsNotNone(result["predicted_ns"])
        self.assertFalse(result["graph_applicability_verified"])
        self.assertFalse(result["accepted"])

    def test_changed_graph_after_freeze_rejected(self):
        other = modeling.Graph((modeling.Task("compute", "gpu", (modeling.WorkDemand("gpu_rate", 500, "flop"),)),))
        with self.assertRaisesRegex(ValueError, "requested graph"):
            evaluate_calibration(other, freeze(), [heldout()], verify_sources=False)

    def test_overlapping_heldout_pid_reuse_rejected(self):
        rows = [replace(heldout(i), process_id=2000) for i in range(3)]
        with self.assertRaisesRegex(ValueError, "overlapping run lifetimes"):
            evaluate_calibration(graph(), freeze(), rows, verify_sources=False)


class SourceAndCliTests(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[1] / "results"
        base.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "raw.json"
        self.source.write_text('{"label":"synthetic measurements"}\n', encoding="utf-8", newline="\n")
        self.sha = hashlib.sha256(self.source.read_bytes()).hexdigest()

    def source_rows(self):
        return [replace(observation(i), source_path=str(self.source), source_sha256=self.sha) for i in range(3)]

    def write_observations(self, rows):
        path = self.root / "observations.json"
        path.write_text(json.dumps(dict(schema=SCHEMA, observations=[row.to_dict() for row in rows])), encoding="utf-8")
        return path

    def test_source_loader_rehashes_and_ignores_json_verified_assertion(self):
        path = self.write_observations(self.source_rows())
        self.assertTrue(all(row.source_verified for row in load_observations(path)))
        self.source.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_observations(path)

    def test_frozen_current_source_recheck(self):
        document = freeze(self.source_rows())
        load_frozen_parameters(document, verify_sources=True)
        self.source.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source bytes changed"):
            load_frozen_parameters(document, verify_sources=True)
        # Receipt-only loading is explicitly different from raw source checking.
        load_frozen_parameters(document, verify_sources=False)

    def test_current_heldout_source_rechecked_by_evaluator(self):
        frozen = freeze(self.source_rows())
        rows = [replace(heldout(i), source_path=str(self.source), source_sha256=self.sha) for i in range(3)]
        result = evaluate_calibration(graph(), frozen, rows)
        self.assertTrue(result["current_raw_sources_verified"])
        self.assertTrue(result["accepted"])

    def test_cli_estimate_writes_lf_and_refuses_overwrite(self):
        observations = self.write_observations(self.source_rows())
        context = self.root / "context.json"
        context.write_text(json.dumps(CONTEXT), encoding="utf-8")
        output = self.root / "frozen.json"
        args = ["estimate", "--observations", str(observations), "--context", str(context),
                "--output", str(output), "--frozen-at-utc", "2026-10-18T00:00:10Z"]
        self.assertEqual(main(args), 0)
        self.assertNotIn(b"\r", output.read_bytes())
        load_frozen_parameters(output, verify_sources=True)
        with self.assertRaises(FileExistsError):
            main(args)

    def test_duplicate_observations_rejected(self):
        rows = self.source_rows()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_observations(self.write_observations([rows[0], rows[0]]))


if __name__ == "__main__":
    unittest.main()
