"""Software-only synthetic fixtures for the optional prepared transfer provider."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from llm_infer_model.performance.prepared_transfer import (
    COMPLETION_SEMANTIC, MEASUREMENT_API, PreparedTransferModel, transfer_coordinate,
)
from profiling.dac_accuracy_repair.fit import fit_training, freeze_training_profile


class PreparedTransferTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.contract = dict(route="vram->dram", memory_mode="pageable",
            measurement_api=MEASUREMENT_API, completion_semantic=COMPLETION_SEMANTIC,
            scope_sha256="a" * 64, preparation_contract=dict(reset_before_batch=False,
                untimed_transfer_prime_passes=1, timed_replays_per_batch=256,
                warmup_iterations_min=20, warmup_ms_min=500, fixed_input_replay=True,
                equal_warmup_per_memory_mode_visit=True))
        source = self.root / "software-fixture.txt"
        source.write_text("synthetic unit-test provenance; no native acquisition\n", encoding="utf-8", newline="\n")
        self.source = self.receipt(source)
        self.provenance = dict(hardware={"synthetic_unit_test": True}, binary=self.source,
            runtime=[self.source], sources=[self.source])
        self.rows = []
        for size in (4, 256, 4096, 65536, 1048576, 4194304, 16777216, 67108864):
            raw = self.root / (str(size) + ".json")
            raw.write_text(json.dumps(dict(synthetic=True, bytes=size)) + "\n", encoding="utf-8", newline="\n")
            self.rows.append(dict(split="calibration", bytes=size, mean_ns=2000 + .15 * size,
                contract=copy.deepcopy(self.contract), raw=self.receipt(raw),
                host_pointer=dict(status="observed", registered_host=False, unregistered_pageable=True)))

    @staticmethod
    def receipt(path):
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def fit(self, rows=None, **kwargs):
        return fit_training(self.rows if rows is None else rows, contract=self.contract,
            provenance=self.provenance, **kwargs)

    def test_training_fit_interpolates_known_affine_fixture_without_qualification(self):
        profile = self.fit()
        model = PreparedTransferModel.from_profile(profile)
        got = model.prediction(16384, contract=self.contract, allow_unqualified=True)
        self.assertAlmostEqual(got["duration_ns"], 2000 + .15 * 16384, places=5)
        self.assertFalse(got["qualified_for_requested_contract"])
        with self.assertRaisesRegex(ValueError, "qualification"):
            model.prediction(16384, contract=self.contract)

    def test_every_contract_dimension_is_checked(self):
        model = PreparedTransferModel.from_profile(self.fit())
        changes = [dict(route="dram->vram"), dict(memory_mode="registered"),
            dict(scope_sha256="b" * 64), dict(measurement_api="cuda_event_elapsed"),
            dict(completion_semantic="enqueue_only")]
        for change in changes:
            target = dict(self.contract, **change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                model.prediction(1024, contract=target, allow_unqualified=True)
        target = copy.deepcopy(self.contract)
        target["preparation_contract"]["untimed_transfer_prime_passes"] = 0
        with self.assertRaises(ValueError):
            model.prediction(1024, contract=target, allow_unqualified=True)

    def test_batch_mean_to_single_call_requires_explicit_unqualified_assumption(self):
        model = PreparedTransferModel.from_profile(self.fit())
        target = copy.deepcopy(self.contract)
        target["preparation_contract"]["timed_replays_per_batch"] = 1
        with self.assertRaises(ValueError):
            model.prediction(1024, contract=target, allow_unqualified=True)
        got = model.prediction(1024, contract=target, allow_unqualified=True,
            allow_batch_mean_as_single_call=True)
        self.assertEqual(len(got["scope_assumptions"]), 1)
        self.assertFalse(got["exact_contract_match"])
        self.assertFalse(got["qualified_for_requested_contract"])
        target["preparation_contract"]["reset_before_batch"] = True
        with self.assertRaises(ValueError):
            model.prediction(1024, contract=target, allow_unqualified=True,
                allow_batch_mean_as_single_call=True)

    def test_heldout_rows_and_missing_pointer_evidence_cannot_fit(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["split"] = "heldout"
        with self.assertRaisesRegex(ValueError, "calibration"):
            self.fit(rows)
        rows = copy.deepcopy(self.rows)
        rows[0]["host_pointer"] = {"status": "unknown"}
        with self.assertRaisesRegex(ValueError, "host-pointer"):
            self.fit(rows)

    def test_mixed_or_contradictory_contracts_are_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["contract"]["memory_mode"] = "registered"
        with self.assertRaisesRegex(ValueError, "mix"):
            self.fit(rows)
        rows = copy.deepcopy(self.rows)
        rows[0]["memory_mode"] = "registered"
        with self.assertRaisesRegex(ValueError, "contradicts"):
            self.fit(rows)

    def test_changed_raw_or_runtime_cannot_be_frozen(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["raw"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "changed"):
            self.fit(rows)
        self.provenance["runtime"][0] = dict(self.source, sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "changed"):
            self.fit()

    def test_diagnostics_and_aliases_are_never_added_to_the_price(self):
        baseline = self.fit()["parameters"]
        rows = copy.deepcopy(self.rows)
        for row in rows:
            row.update(gpu_event_mean_ns=1e15, copy_call_ns=row["mean_ns"],
                submit_copy_api_ns=row["mean_ns"], gpu_sync_ns=1e15)
        fitted = self.fit(rows)
        self.assertEqual(fitted["parameters"], baseline)
        self.assertFalse(fitted["diagnostic_phase_or_event_cost_added"])

    def test_unknown_or_unmeasured_size_and_frozen_mutation_are_rejected(self):
        profile = self.fit()
        model = PreparedTransferModel.from_profile(profile)
        profile["parameters"]["fixed_ns"] = 1e15
        self.assertLess(model.profile["parameters"]["fixed_ns"], 1e15)
        for size in (0, True, 67108865):
            with self.subTest(size=size), self.assertRaises(ValueError):
                model.prediction(size, contract=self.contract, allow_unqualified=True)
        target = copy.deepcopy(self.contract)
        target["preparation_contract"]["warmup_ms_min"] = None
        with self.assertRaises(ValueError):
            transfer_coordinate(1024, contract=target)

    def test_piecewise_response_is_continuous_and_negative_terms_are_rejected(self):
        profile = self.fit()
        profile["parameters"] = dict(fixed_ns=2000., small_ns_per_byte=.1,
            large_ns_per_byte=.2, breakpoint_bytes=65536)
        model = PreparedTransferModel.from_profile(profile)
        response = lambda n: model.prediction(n, contract=self.contract,
            allow_unqualified=True)["duration_ns"]
        self.assertAlmostEqual(response(65536) - response(65535), .1)
        self.assertAlmostEqual(response(65537) - response(65536), .2)
        for key, value in (("fixed_ns", -1), ("small_ns_per_byte", True),
                           ("large_ns_per_byte", float("inf")), ("breakpoint_bytes", 4)):
            bad = copy.deepcopy(profile)
            bad["parameters"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                PreparedTransferModel.from_profile(bad)

    def test_unsupported_hinge_without_two_sided_loo_evidence_is_not_admitted(self):
        fitted = self.fit(breakpoints=(4, 67108864, 16384))
        self.assertEqual(fitted["requested_breakpoints"], [4, 67108864, 16384])
        self.assertEqual(fitted["admitted_breakpoints"], [16384])
        self.assertEqual({m["breakpoint_bytes"] for m in fitted["candidate_selection"]["candidates"]},
            {None, 16384})

    def test_freeze_is_write_once_lf_and_contains_only_calibration_provenance(self):
        path = self.root / "frozen.json"
        frozen = freeze_training_profile(path, self.rows, contract=self.contract,
            provenance=self.provenance)
        original = path.read_bytes()
        self.assertNotIn(b"\r\n", original)
        self.assertFalse(frozen["heldout_rows_used"])
        self.assertEqual(len(frozen["calibration_rows"]), 8)
        self.assertTrue(frozen["candidate_selection"]["candidates"])
        with self.assertRaises(FileExistsError):
            freeze_training_profile(path, self.rows, contract=self.contract,
                provenance=self.provenance)
        self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
