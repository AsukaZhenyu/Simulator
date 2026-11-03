from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from profiling.storage import ssd_measure as ssd


class RecordValidationTests(unittest.TestCase):
    def good_record(self, direction="READ_FILE", samples=None, mechanism="buffered"):
        values = list(samples if samples is not None else [100, 110, 120])
        flush = [5, 5, 5] if direction == "WRITE_FILE" else None
        return ssd.build_record(
            direction, 4096, values, cache_bypass=mechanism, warmup=0,
            random_offset=False, backend="test", flush_samples=flush,
            sample_offsets=[0] * len(values),
        )

    def test_good_records_are_accepted(self):
        for direction in ("READ_FILE", "WRITE_FILE"):
            ssd.validate_record(self.good_record(direction))

    def test_bad_route_is_rejected(self):
        record = self.good_record()
        record["route"] = {"source": "vram", "destination": "dram"}
        with self.assertRaises(ssd.SsdMeasureError):
            ssd.validate_record(record)

    def test_two_samples_are_rejected(self):
        record = self.good_record()
        record["samples_ns"] = record["samples_ns"][:2]
        record["sample_count"] = 2
        with self.assertRaises(ssd.SsdMeasureError):
            ssd.validate_record(record)

    def test_wrong_duration_is_rejected(self):
        record = self.good_record()
        record["duration_ns"] += 1
        with self.assertRaises(ssd.SsdMeasureError):
            ssd.validate_record(record)

    def test_sample_count_mismatch_is_rejected(self):
        record = self.good_record()
        record["sample_count"] += 1
        with self.assertRaises(ssd.SsdMeasureError):
            ssd.validate_record(record)

    def test_buffered_record_cannot_claim_verified_bandwidth(self):
        record = self.good_record()
        self.assertFalse(record["physical_ssd_bandwidth_verified"])
        record["physical_ssd_bandwidth_verified"] = True
        with self.assertRaises(ssd.SsdMeasureError):
            ssd.validate_record(record)

    def test_unbuffered_record_must_claim_verified_bandwidth(self):
        record = self.good_record()
        record["cache_bypass"] = "unbuffered"
        with self.assertRaises(ssd.SsdMeasureError):
            ssd.validate_record(record)

    def test_read_record_cannot_report_flush(self):
        record = self.good_record("READ_FILE")
        record["flush_samples_ns"] = [1, 1, 1]
        record["flush_ns"] = 1
        with self.assertRaises(ssd.SsdMeasureError):
            ssd.validate_record(record)

    def test_inconsistent_statistics_are_rejected(self):
        record = self.good_record()
        record["statistics"]["median_ns"] = 1
        with self.assertRaises(ssd.SsdMeasureError):
            ssd.validate_record(record)


class DarwinMechanismTests(unittest.TestCase):
    """Darwin has no O_DIRECT and ignores FADV_DONTNEED; it must not claim either."""

    def test_darwin_selects_f_nocache_per_handle(self):
        with mock.patch.object(ssd, "_darwin_nocache_available", return_value=True), \
                mock.patch.object(ssd, "_posix_pair",
                                  return_value=("r", "w", "f_nocache", "darwin-f_nocache")) as pair:
            _, _, mechanism, label = ssd.open_backends(Path("ignored"), 4096, "required")
        self.assertEqual((mechanism, label), ("f_nocache", "darwin-f_nocache"))
        pair.assert_called_once_with(Path("ignored"), 4096, direct=False, nocache=True)

    def test_f_nocache_is_a_verified_bypass(self):
        record = ssd.build_record("READ_FILE", 4096, [100, 110, 120],
                                  cache_bypass="f_nocache", warmup=0,
                                  random_offset=False, backend="darwin-f_nocache")
        self.assertTrue(record["physical_ssd_bandwidth_verified"])
        ssd.validate_record(record)

    def test_darwin_is_not_offered_where_it_does_not_exist(self):
        with mock.patch.object(ssd.sys, "platform", "win32"):
            self.assertFalse(ssd._darwin_nocache_available())


class FitTests(unittest.TestCase):
    def test_known_slope_is_recovered(self):
        fit = ssd._least_squares_fit([(1000, 1100), (2000, 1200)])
        self.assertEqual(fit["n_points"], 2)
        self.assertTrue(fit["is_descriptive"])
        self.assertAlmostEqual(fit["slope_ns_per_byte"], 0.1)
        self.assertAlmostEqual(fit["latency_ns"], 1000.0)
        self.assertAlmostEqual(fit["bytes_per_ns"], 10.0)
        for size in ("1000", "2000"):
            self.assertAlmostEqual(fit["relative_residuals"][size], 0.0)

    def test_single_size_has_no_fit_but_keeps_residual_keys(self):
        fit = ssd._least_squares_fit([(4096, 100)])
        self.assertEqual(fit["n_points"], 1)
        self.assertIsNone(fit["latency_ns"])
        self.assertIsNone(fit["bytes_per_ns"])
        self.assertEqual(set(fit["relative_residuals"]), {"4096"})
        self.assertIsNone(fit["relative_residuals"]["4096"])


class MeasurementTests(unittest.TestCase):
    def test_unaligned_size_is_rejected_before_any_io(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ssd.SsdMeasureError):
                ssd.measure([4095], Path(temporary), samples=3, warmup=0,
                            cache_bypass="required", device_probe=False)

    def test_off_mode_measurement_and_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            document = ssd.measure(
                [4096, 65536], Path(temporary), samples=3, warmup=0,
                cache_bypass="off", random_offset=True, seed=7,
                platform_label="unit-test", device_probe=False,
            )
        self.assertEqual(document["schema_version"], "ssd_storage_measurements_v1")
        self.assertTrue(document["all_measured"])
        self.assertEqual(document["platform"]["label"], "unit-test")
        self.assertIsNone(document["storage_device"])
        self.assertEqual(len(document["measurements"]), 4)
        for record in document["measurements"]:
            self.assertEqual(record["status"], "measured")
            ssd.validate_record(record)
            self.assertEqual(record["cache_bypass"], "buffered")
            self.assertFalse(record["physical_ssd_bandwidth_verified"])
            self.assertEqual(record["sample_count"], 3)
            self.assertEqual(len(record["samples_ns"]), 3)
            self.assertEqual(record["duration_ns"],
                             math.ceil(record["statistics"]["median_ns"]))
            if record["direction"] == "WRITE_FILE":
                self.assertEqual(len(record["flush_samples_ns"]), 3)
                self.assertIsNotNone(record["flush_ns"])
            else:
                self.assertIsNone(record["flush_ns"])
        summary = document["summary"]
        self.assertTrue(summary["fit_is_descriptive_not_used_for_lookup"])
        self.assertEqual(summary["measurement_counts"], {"measured": 4, "failed": 0})
        for direction in ("READ_FILE", "WRITE_FILE"):
            block = summary["directions"][direction]
            self.assertEqual(len(block["sizes"]), 2)
            for row in block["sizes"]:
                for key in ("median_ns", "p10_ns", "p90_ns", "cv"):
                    self.assertIn(key, row)
            fit = block["fit"]
            self.assertTrue(fit["is_descriptive"])
            self.assertEqual(fit["n_points"], 2)
            self.assertEqual(set(fit["relative_residuals"]), {"4096", "65536"})


class CliTests(unittest.TestCase):
    def test_cli_off_mode_smoke(self):
        repository = Path(__file__).resolve().parents[3]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = (
            str(repository) + os.pathsep + environment.get("PYTHONPATH", ""))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "measurements.json"
            completed = subprocess.run(
                [sys.executable, "-m", "profiling.storage.ssd_measure",
                 "--sizes-bytes", "4096",
                 "--directory", temporary,
                 "--output", str(output),
                 "--samples", "3", "--warmup", "0",
                 "--cache-bypass", "off", "--platform", "cli-test",
                 "--seed", "1", "--device-probe", "off"],
                cwd=repository, env=environment, capture_output=True,
                text=True, timeout=120,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            document = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(document["schema_version"], "ssd_storage_measurements_v1")
        self.assertEqual(document["platform"]["label"], "cli-test")
        self.assertEqual(document["protocol"]["cache_bypass_requested"], "off")
        self.assertEqual(len(document["measurements"]), 2)
        self.assertEqual([record["direction"] for record in document["measurements"]],
                         ["READ_FILE", "WRITE_FILE"])
        for record in document["measurements"]:
            ssd.validate_record(record)


if __name__ == "__main__":
    unittest.main()
