"""Synthetic bookkeeping/protocol tests; no hardware accuracy is asserted."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from single_layer_calibration import analysis


def program(seed=10):
    return {"program_sha256": analysis.sha({"seed": seed}), "seed": seed,
            "identity": {"graph": "same graph", "physical": "same F32 shape", "initial_values": str(seed)},
            "actions": [{"kind": "COMPUTE", "target_id": "projection", "device": "cpu"},
                        {"kind": "ADVANCE"},
                        {"kind": "COPY", "target_id": "projection.out", "src": "dram", "dst": "vram"},
                        {"kind": "EVICT", "target_id": "projection.out", "location": "dram"}]}


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[3] / "build/single-layer-analysis-tests"
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.number = 0
        self.program = program()
        self.oracle = patch.object(analysis, "verify_native", return_value={"all_ports_passed": True})
        self.oracle.start()
        self.addCleanup(self.oracle.stop)

    def record(self, mode, *, split="profile", seed=None, durations=None, isolated=None,
               gaps=None, tail=None, totals=None, poll="default", sample_count=3):
        self.number += 1
        seed = (10 if split == "profile" else 100 + self.number) if seed is None else seed
        source_program = program(seed)
        count = sample_count
        durations = durations or [[10, 3, 20, 2] for _ in range(count)]
        gaps = gaps or [[1, 1, 1, 1] for _ in range(count)]
        tail = [2] * count if tail is None else tail
        native = {"format_version": "single-layer-calibration-native-v1", "mode": mode,
                  "process_id": self.number, "program_sha256": source_program["program_sha256"],
                  "actions": deepcopy(source_program["actions"]), "samples": count, "warmup": 20,
                  "components": [], "iterations": [], "total_samples_ns": [],
                  "native_clock": {"clock_id": "synthetic-ns", "frequency_hz": 10**9,
                                   "cross_process_monotonic_clock": True},
                  "threadpool": {"requested_poll": poll, "effective_params": {"n_threads": 2, "poll": 50 if poll == "default" else 0}},
                  "execution_contract": "prebuilt_synchronized_GGML_actions;reset_excluded;preallocated_trace_buffer_v1",
                  "verification_native": {"observed_runtime_flags": {"NVIDIA_TF32_OVERRIDE": "0"}},
                  "synthetic_test_fixture": True}
        if mode == "isolated-components":
            for index, default in ((0, 8), (2, 20)):
                values = [default] * count if isolated is None else isolated[index]
                intervals = [{"sample": number, "begin_tick": number * 10000,
                              "end_tick": number * 10000 + value, "elapsed_ns": value}
                             for number, value in enumerate(values)]
                native["components"].append({"action_index": index, "action": source_program["actions"][index],
                    "samples_ns": values, "intervals": intervals, "workspace_bytes": 0})
        else:
            for number in range(count):
                base = number * 10000
                position, intervals = 0, []
                for index, duration in enumerate(durations[number]):
                    position += gaps[number][index]
                    begin = position
                    position += duration
                    intervals.append({"action_index": index, "kind": source_program["actions"][index]["kind"],
                                      "begin_tick": base + begin, "end_tick": base + position,
                                      "begin_offset_ns": begin, "end_offset_ns": position, "elapsed_ns": duration})
                elapsed = position + tail[number]
                if totals is not None:
                    elapsed = totals[number]
                native["total_samples_ns"].append(elapsed)
                native["iterations"].append({"iteration": number, "begin_tick": base,
                    "end_tick": base + elapsed, "elapsed_ns": elapsed,
                    "action_intervals": intervals if mode == "context-trace" else []})
        path = self.root / f"raw-{self.number}.json"
        path.write_text(json.dumps(native), encoding="utf-8", newline="\n")
        start = datetime(2026, 10, 18, 0, 0 if split == "profile" else 2, tzinfo=timezone.utc) + timedelta(seconds=self.number * 2)
        return analysis.make_run_record(source_program, native, run_id=f"run-{self.number}", split=split,
            seed=seed, started_at_utc=start.isoformat(), completed_at_utc=(start + timedelta(seconds=1)).isoformat(),
            source_path=path, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), software_sha256="a" * 64)

    def profiles(self, **changes):
        return [self.record(mode, **changes) for mode in analysis.MODES for _ in range(3)]

    def freeze(self, rows=None):
        return analysis.freeze_profile(self.program, rows or self.profiles(), frozen_at_utc="2026-10-18T00:01:00+00:00")

    def heldout(self, **changes):
        return [self.record("bare-replay", split="heldout", **changes) for _ in range(3)]

    def test_three_models_use_model_owned_evaluator(self):
        original = analysis.evaluate_serial_actions
        with patch.object(analysis, "evaluate_serial_actions", wraps=original) as evaluate:
            frozen = self.freeze()
            report = analysis.evaluate_heldout(self.program, frozen, self.heldout())
        self.assertEqual(evaluate.call_count, 6)
        self.assertEqual([row["prediction"]["duration_ns"] for row in frozen["models"].values()], [28, 30, 41])
        self.assertTrue(report["accepted"])
        self.assertFalse(report["models"]["isolated_components"]["accepted"])
        self.assertEqual(report["models"]["context_actions_and_gaps"]["absolute_relative_error"], 0)

    def test_zero_cost_controls_are_explicit_baseline_declarations(self):
        frozen = self.freeze()
        old = frozen["models"]["isolated_components"]
        values = {row["name"]: row for row in old["parameters"]["parameters"]}
        self.assertEqual(values["action:1:duration"]["origin"], "declared")
        self.assertEqual(values["action:1:duration"]["value"], {"numerator": 0, "denominator": 1})
        self.assertEqual(values["action:0:duration"]["origin"], "profiled")

    def test_bare_total_never_becomes_performance_parameter(self):
        rows = self.profiles()
        first = self.freeze(rows)
        for row in rows:
            if row["mode"] == "bare-replay":
                for iteration in row["native"]["iterations"]:
                    iteration["elapsed_ns"] = 1000
                    iteration["end_tick"] = iteration["begin_tick"] + 1000
                row["native"]["total_samples_ns"] = [1000] * 3
                self.rewrite_source(row)
        second = self.freeze(rows)
        for name in analysis.MODEL_NAMES:
            self.assertEqual(first["models"][name]["parameters"], second["models"][name]["parameters"])
        self.assertFalse(second["instrumentation_control"]["passed"])

    def rewrite_source(self, record):
        path = Path(record["source_path"])
        path.write_text(json.dumps(record["native"]), encoding="utf-8", newline="\n")
        record["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    def test_equal_process_weight_not_pooled_sample_weight(self):
        rows = self.profiles()
        for index, value in enumerate((10, 20, 100)):
            replacement = self.record("isolated-components", sample_count=31 if index == 2 else 3,
                                      isolated={0: [value] * (31 if index == 2 else 3), 2: [20] * (31 if index == 2 else 3)})
            rows[index] = replacement
        frozen = self.freeze(rows)
        self.assertEqual(frozen["action_statistics"]["isolated"]["0"]["value_ns"], {"numerator": 20, "denominator": 1})

    def test_gap_cv_is_reported_without_relabelling_original_gate(self):
        rows = []
        for mode in analysis.MODES:
            for _ in range(3):
                rows.append(self.record(mode, gaps=[[0] * 4, [10] * 4, [0] * 4], tail=[0, 0, 0],
                                        totals=[35] * 3 if mode == "bare-replay" else None))
        frozen = self.freeze(rows)
        self.assertFalse(frozen["gap_statistics"]["0"]["within_process_cv_passed"])
        self.assertEqual(frozen["gap_statistics"]["0"]["acceptance_scope"], "diagnostic_not_original_source_criterion")
        report = analysis.evaluate_heldout(self.program, frozen, self.heldout(totals=[35] * 3))
        self.assertTrue(report["accepted"])
        self.assertFalse(report["models"]["context_actions_and_gaps"]["gap_quality_claimed"])

    def test_unstable_context_compute_rejects_primary_model(self):
        rows = self.profiles(durations=[[1, 3, 20, 2], [10, 3, 20, 2], [100, 3, 20, 2]])
        frozen = self.freeze(rows)
        report = analysis.evaluate_heldout(self.program, frozen, self.heldout())
        self.assertFalse(report["accepted"])
        self.assertTrue(frozen["dominant_unstable_sources"])
        self.assertTrue(all(row["cause"] == "unresolved" for row in frozen["dominant_unstable_sources"]))

    def test_missing_mode_or_independent_process_rejected(self):
        with self.assertRaisesRegex(ValueError, ">=3"):
            self.freeze(self.profiles()[:-1])

    def test_duplicate_run_rejected(self):
        rows = self.profiles()
        rows[-1]["run_id"] = rows[0]["run_id"]
        with self.assertRaisesRegex(ValueError, "duplicate run ID"):
            self.freeze(rows)

    def test_renamed_same_native_process_not_independent(self):
        rows = self.profiles()
        replacement = deepcopy(rows[-1])
        replacement["run_id"] = "renamed"
        rows.append(replacement)
        with self.assertRaisesRegex(ValueError, "not independent"):
            self.freeze(rows)

    def test_polling_contracts_cannot_be_pooled(self):
        rows = self.profiles()
        rows[-1] = self.record("bare-replay", poll="0")
        with self.assertRaisesRegex(ValueError, "contracts cannot be mixed"):
            self.freeze(rows)

    def test_numerical_oracle_is_actually_called(self):
        with patch.object(analysis, "verify_native", side_effect=ValueError("independent numerical mismatch")):
            with self.assertRaisesRegex(ValueError, "numerical mismatch"):
                self.record("context-trace")

    def test_native_action_inventory_is_checked(self):
        row = self.record("context-trace")
        row["native"]["actions"][0]["target_id"] = "different"
        self.rewrite_source(row)
        with self.assertRaisesRegex(ValueError, "action inventory"):
            self.freeze([row] + self.profiles())

    def test_native_offset_gap_overlap_is_rejected(self):
        row = self.record("context-trace")
        row["native"]["iterations"][0]["action_intervals"][1]["begin_offset_ns"] = 0
        self.rewrite_source(row)
        with self.assertRaisesRegex(ValueError, "offsets overlap"):
            self.freeze([row] + self.profiles())

    def test_raw_bytes_changed_after_measurement_rejected(self):
        rows = self.profiles()
        Path(rows[0]["source_path"]).write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source bytes changed"):
            self.freeze(rows)

    def test_raw_hash_cannot_self_assert_different_native(self):
        rows = self.profiles()
        rows[0]["native"]["warmup"] = 99
        with self.assertRaisesRegex(ValueError, "hashed raw JSON"):
            self.freeze(rows)

    def test_freeze_must_follow_every_profile_process(self):
        rows = self.profiles()
        with self.assertRaisesRegex(ValueError, "after freeze"):
            analysis.freeze_profile(self.program, rows, frozen_at_utc="2026-10-18T00:00:00+00:00")

    def test_profile_records_cannot_be_used_as_heldout(self):
        frozen = self.freeze()
        with self.assertRaisesRegex(ValueError, "only verified heldout"):
            analysis.evaluate_heldout(self.program, frozen, self.profiles())

    def test_heldout_seed_must_be_new(self):
        frozen = self.freeze()
        with self.assertRaisesRegex(ValueError, "seed reused"):
            analysis.evaluate_heldout(self.program, frozen, self.heldout(seed=10))

    def test_heldout_must_begin_after_freeze(self):
        frozen = self.freeze()
        rows = self.heldout()
        rows[0]["started_at_utc"] = "2026-10-18T00:00:00+00:00"
        with self.assertRaisesRegex(ValueError, "before parameter freeze"):
            analysis.evaluate_heldout(self.program, frozen, rows)

    def test_heldout_is_bare_not_instrumented_trace(self):
        frozen = self.freeze()
        rows = [self.record("context-trace", split="heldout") for _ in range(3)]
        with self.assertRaisesRegex(ValueError, "bare whole-program"):
            analysis.evaluate_heldout(self.program, frozen, rows)

    def test_heldout_software_or_poll_change_rejected(self):
        frozen = self.freeze()
        rows = self.heldout()
        for row in rows:
            row["software_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "software inventory differs"):
            analysis.evaluate_heldout(self.program, frozen, rows)
        with self.assertRaisesRegex(ValueError, "execution scope differs"):
            analysis.evaluate_heldout(self.program, frozen, self.heldout(poll="0"))

    def test_frozen_parameter_mutation_rejected(self):
        frozen = self.freeze()
        frozen["models"]["context_components"]["parameters"]["parameters"][0]["value"]["numerator"] += 1
        with self.assertRaisesRegex(ValueError, "frozen parameter artifact changed"):
            analysis.evaluate_heldout(self.program, frozen, self.heldout())

    def test_source_verification_opt_out_cannot_accept(self):
        frozen = self.freeze()
        report = analysis.evaluate_heldout(self.program, frozen, self.heldout(), verify_sources=False)
        self.assertFalse(report["accepted"])
        self.assertFalse(report["raw_sources_verified"])

    def test_heldout_error_threshold_is_original_twenty_percent(self):
        frozen = self.freeze()
        report = analysis.evaluate_heldout(self.program, frozen, self.heldout(totals=[100] * 3))
        self.assertFalse(report["accepted"])
        self.assertAlmostEqual(report["models"]["context_actions_and_gaps"]["absolute_relative_error"], .59)

    def test_new_numerical_seeds_keep_same_physical_contract(self):
        frozen = self.freeze()
        rows = self.heldout()
        self.assertNotEqual(rows[0]["program_sha256"], self.program["program_sha256"])
        self.assertTrue(analysis.evaluate_heldout(self.program, frozen, rows)["accepted"])

    def test_exact_median_and_zero_cv(self):
        self.assertEqual(analysis.median([1, 2]), analysis.Fraction(3, 2))
        self.assertEqual(analysis.cv([0, 0, 0]), 0)


if __name__ == "__main__":
    unittest.main()
