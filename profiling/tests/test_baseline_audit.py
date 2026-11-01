"""Independent arithmetic and malformed-inventory checks for baseline gating."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from profiling.dynamic_validation.baseline_audit import audit_native, inspect_campaign


def fixture(early=100, late=100):
    rows = []
    for trial in range(16):
        for idle in (0, 1, 5, 20):
            tail = early if trial < 8 else late
            rows.append(dict(trial=trial, requested_idle_ms=idle, actual_idle_ns=idle * 10**6,
                copies=[dict(ordinal=i, wall_ns=1000 if i == 0 else tail) for i in range(16)]))
    return dict(request=dict(memory_mode="registered", idle_ms=[0, 1, 5, 20],
                              sampling=dict(trials=16, repeats=16)), sequences=rows)


def audit(value, **kw):
    return audit_native(value, tail_first_ordinal=kw.get("tail_first_ordinal", 1),
                        split_trial_count=8, relative_limit=kw.get("relative_limit", .1))


class BaselineAuditTests(unittest.TestCase):
    def test_constant_tail_does_not_confuse_first_response_with_base(self):
        result = audit(fixture())
        self.assertTrue(result["fixed_baseline_support"])
        self.assertEqual(result["all_suffix_mean_ns"], 100)
        self.assertEqual(result["zero_idle_trials"][0]["first_copy_ns"], 1000)
        self.assertEqual(result["trial_mean_cv_population"], 0)
        self.assertFalse(result["suffix_is_proven_steady"])

    def test_late_shift_is_retained_and_rejected_in_both_directions(self):
        for early, late, change in ((100, 200, 1), (200, 100, -.5)):
            with self.subTest(early=early):
                result = audit(fixture(early, late))
                self.assertFalse(result["fixed_baseline_support"])
                self.assertEqual(result["late_over_early_minus_one"], change)
                self.assertEqual(result["observed_trials"], 16)
                self.assertEqual(result["samples_removed"], 0)

    def test_missing_or_duplicate_conditions_cannot_disappear(self):
        value = fixture()
        value["sequences"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            audit(value)
        value = fixture()
        value["sequences"].append(copy.deepcopy(value["sequences"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            audit(value)

    def test_malformed_copy_inventory_is_rejected(self):
        value = fixture()
        value["sequences"][0]["copies"][2]["ordinal"] = 1
        with self.assertRaisesRegex(ValueError, "ordinal"):
            audit(value)

    def test_no_empty_tail_or_invalid_support_limit(self):
        for first, limit in ((16, .1), (0, .1), (1, 0), (1, float("nan")), (1, True)):
            with self.subTest(first=first, limit=limit), self.assertRaises(ValueError):
                audit(fixture(), tail_first_ordinal=first, relative_limit=limit)

    def test_heldout_cannot_be_used_as_calibration(self):
        with self.assertRaisesRegex(ValueError, "heldout"):
            inspect_campaign("unused", split="heldout", role="calibration_support",
                             tail_first_ordinal=1, split_trial_count=8, relative_limit=.1)

    def test_estimator_refuses_a_drifting_profile_before_optimization(self):
        from profiling.dynamic_validation import idle_fit
        repository = Path(__file__).resolve().parents[2]
        scratch = (repository / "build" / "test-baseline-audit").resolve()
        self.assertTrue(scratch.is_relative_to(repository))
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            base = Path(directory)
            self.assertTrue(base.resolve().is_relative_to(scratch))
            jobs, executions = [], []
            for memory in ("pageable", "registered"):
                for round_index in range(3):
                    native = fixture(100, 200)
                    native["request"]["memory_mode"] = memory
                    path = base / (memory + str(round_index) + ".json")
                    path.write_text(json.dumps(native), encoding="utf-8", newline="\n")
                    job = dict(id=memory + str(round_index), split="profile", memory_mode=memory,
                               request=native["request"])
                    jobs.append(job)
                    executions.append(dict(job=job, status="completed", native=dict(path=str(path),
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
            jobs.append(dict(id="independent", split="heldout"))
            for name, value in (("manifest.json", dict(jobs=jobs)),
                                ("profile-executions.json", executions),
                                ("profile-observations.json", dict(completed=6, declared=6))):
                (base / name).write_text(json.dumps(value), encoding="utf-8", newline="\n")
            # The fixture has no native DLLs: bypass only their provenance
            # check, not any measurement arithmetic or baseline decision.
            with patch.object(idle_fit.campaign, "check"), patch.object(idle_fit, "fit_hypothesis") as optimizer:
                with self.assertRaisesRegex(ValueError, "fixed-baseline hypothesis unsupported"):
                    idle_fit.fit_and_freeze(base)
                optimizer.assert_not_called()
            refusal = json.loads((base / "fit-refusal.json").read_text())
            self.assertFalse(refusal["predictions_frozen"])
            self.assertFalse(refusal["performance_parameters_fitted"])
            self.assertEqual(refusal["samples_removed"], 0)
            self.assertFalse((base / "frozen-barrier.json").exists())


if __name__ == "__main__":
    unittest.main()
