from copy import deepcopy
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

from profiling.available_service import freeze_coherent_ensemble, validate_coherent_ensemble
from profiling.available_service.coherent import coherent_sample_parameters
from profiling.available_service.calibration import digest, encoded
from profiling.service_response.evidence import token_inventory
from llm_infer_model.performance.serial_actions import evaluate_serial_actions


class CoherentEnsembleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="coherent-test-", dir=ROOT / "build")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "records.json"
        actions = [dict(kind="COMPUTE"), dict(kind="COMPUTE")]
        self.context = {"hardware": "fixture", "shape": [2, 2]}
        self.records = []
        for run in range(3):
            invocations = []
            for i, (a, b) in enumerate(((1, 9), (9, 1), (9, 9))):
                begin = i * 30
                invocations.append(dict(observation_id="i:"+str(i), begin_ns=begin, end_ns=begin+a+b,
                                        actions=[[0, begin, begin+a], [1, begin+a, begin+a+b]]))
            record = dict(schema="service-response-run-1", run_id="profile-"+str(run), split="profile",
                seed=run, process_id=100+run, started_at_utc="2026-10-22T00:00:00Z",
                ended_at_utc="2026-10-22T00:01:00Z", context=self.context, context_sha256=digest(self.context),
                contract_sha256="a"*64, action_inventory=actions, token_inventory=token_inventory(actions),
                instrumentation="stamped", invocations=invocations,
                calibration_segments=[dict(begin_ns=0, end_ns=100, requests={"cpu": 0, "gpu": 0},
                    host_memory_bytes=0, device_memory_bytes=0)], numerical_verified=True,
                all_samples_retained=True, outliers_removed=0, source_verified=True)
            record["record_sha256"] = digest(record)
            self.records.append(record)
        self.write()

    def write(self):
        for row in self.records:
            row["record_sha256"] = digest({k: v for k, v in row.items() if k != "record_sha256"})
        self.path.write_text(json.dumps(self.records), encoding="utf-8", newline="\n")
        self.sha = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def freeze(self, count=1):
        return freeze_coherent_ensemble(self.path, self.sha, samples_per_run_state=count)

    def test_preserves_one_real_vector_and_replays_existing_model(self):
        frozen = self.freeze()
        self.assertEqual(len(frozen["samples"]), 3)
        sample = frozen["samples"][0]
        self.assertEqual(sample["observation_id"], "i:1")
        program, bundle = coherent_sample_parameters(frozen, sample["sample_id"])
        self.assertEqual(evaluate_serial_actions(program, bundle).duration_ns, 10)
        self.assertEqual(frozen["cohorts"][0]["measured_whole_call_median_ns"], encoded(10))
        self.assertEqual(frozen["cohorts"][0]["sum_marginal_token_medians_ns"], encoded(18))

    def test_full_ensemble_keeps_correlations_and_zero_gaps(self):
        frozen = self.freeze(None)
        self.assertEqual(len(frozen["samples"]), 9)
        self.assertEqual([s["measured_complete_call_ns"] for s in frozen["samples"][:3]], [10, 10, 18])
        self.assertTrue(any(t["duration_ns"] == 0 for t in frozen["samples"][0]["tokens"]))
        self.assertEqual(frozen["physical_bindings"], [])
        self.assertFalse(frozen["external_available_service_inferred"])
        self.assertFalse(frozen["hardware_accuracy_qualified"])
        self.assertFalse(frozen["current_raw_source_reextraction_performed"])

    def test_selection_uses_order_not_duration(self):
        before = [s["observation_id"] for s in self.freeze()["samples"]]
        for row in self.records:
            row["invocations"][1]["actions"][0][2] -= 2
            row["invocations"][1]["actions"][1][1] -= 2
        self.write()
        self.assertEqual(before, [s["observation_id"] for s in self.freeze()["samples"]])

    def test_one_entire_calibration_window_required(self):
        for row in self.records:
            row["calibration_segments"][0]["end_ns"] = 39
        self.write()
        frozen = self.freeze(None)
        self.assertEqual(frozen["eligible_complete_invocations"], 3)
        self.assertEqual(frozen["excluded_invocations_without_one_complete_known_calibration_window"], 6)

    def test_changed_record_file_detected(self):
        frozen = self.freeze()
        self.path.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA"):
            validate_coherent_ensemble(frozen)

    def test_resigned_vector_does_not_reconstruct(self):
        frozen = self.freeze()
        frozen["samples"][0]["tokens"][1]["duration_ns"] += 1
        frozen["artifact_sha256"] = digest({k: v for k, v in frozen.items() if k != "artifact_sha256"})
        with self.assertRaisesRegex(ValueError, "reconstruct"):
            validate_coherent_ensemble(frozen)

    def test_invalid_action_intervals_rejected(self):
        self.records[0]["invocations"][0]["actions"][1][1] = 0
        self.write()
        with self.assertRaisesRegex(ValueError, "bounds"):
            self.freeze()

    def test_heldout_records_cannot_freeze(self):
        self.records[0]["split"] = "heldout"; self.write()
        with self.assertRaisesRegex(ValueError, "schema/split"):
            self.freeze()

    def test_renamed_pid_seed_is_not_independent(self):
        self.records[1]["process_id"] = self.records[0]["process_id"]
        self.records[1]["seed"] = self.records[0]["seed"]
        self.write()
        with self.assertRaisesRegex(ValueError, "PID/seed"):
            self.freeze()

    def test_same_pid_overlapping_lifetime_is_not_independent(self):
        self.records[1]["process_id"] = self.records[0]["process_id"]; self.write()
        with self.assertRaisesRegex(ValueError, "overlapping PID"):
            self.freeze()


if __name__ == "__main__":
    unittest.main()
