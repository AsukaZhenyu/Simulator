from __future__ import annotations
import copy
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from profiling import moe_transfer as transfer

class TransferContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def record(self, direction="READ_FILE", samples=None):
        sample_values = samples if samples is not None else [100, 100, 100]
        record = transfer._base(direction, 64, "os-buffered-file", None)
        record.update(direction=direction, timing_scope="os_buffered_file_io_wall",
                      cache_policy="reused_file_os_cache_not_flushed", warmup=0,
                      sample_count=len(sample_values), copy_validation={"passed": True},
                      file_io={"os_buffered": True, "fsync_in_timing": direction == "WRITE_FILE"})
        return transfer._finish(record, sample_values)

    def native(self):
        return {"status": "measured", "direction": "H2D", "size_bytes": 64,
                "host_memory": "pinned", "warmup": 2, "samples": 3, "device_index": 0,
                "validation": "all_bytes_exact", "timing_scope": "single_transfer_wall",
                "ops_per_sample": 1, "wall_ns": [100, 100, 100], "device_ns": [90, 90, 90],
                "cache_policy": "reused buffers", "device": "fake", "synchronization": "sync"}

    def cuda(self, native):
        probe = self.directory / "probe.exe"
        probe.write_text("fake probe", encoding="utf-8")
        result = subprocess.CompletedProcess([], 0, json.dumps(native), "")
        with patch.object(transfer.subprocess, "run", return_value=result):
            return transfer.cuda_record(probe, "H2D", 64, "pinned", 2, 3, 0, [], 5)

    def test_stable_raw_statistics(self):
        record = self.record()
        self.assertEqual(record["duration_ns"], 100)
        self.assertEqual(record["quality_flags"], [])
        transfer.validate_record(record)

    def test_median_rounds_up(self):
        record = self.record(samples=[100, 101, 102, 103])
        self.assertEqual(record["duration_ns"], 102)

    def test_no_outliers_dropped(self):
        record = self.record(samples=[1, 1, 10000])
        self.assertEqual(record["samples_ns"], [1, 1, 10000])
        self.assertEqual(record["quality_flags"], ["cv_above_0.1"])

    def test_raw_statistics_must_match(self):
        record = self.record()
        record["statistics"]["median_ns"] = 200
        with self.assertRaises(transfer.TransferError):
            transfer.validate_record(record)

    def test_sample_count_must_match(self):
        record = self.record()
        record["sample_count"] = 4
        with self.assertRaises(transfer.TransferError):
            transfer.validate_record(record)

    def test_raw_duration_must_match(self):
        record = self.record()
        record["duration_ns"] = 1
        with self.assertRaises(transfer.TransferError):
            transfer.validate_record(record)

    def test_route_must_match(self):
        record = self.record()
        record["route"]["source"] = "vram"
        with self.assertRaises(transfer.TransferError):
            transfer.validate_record(record)

    def test_copy_validation_required(self):
        record = self.record()
        record["copy_validation"]["passed"] = False
        with self.assertRaises(transfer.TransferError):
            transfer.validate_record(record)

    def test_physical_ssd_claim_rejected(self):
        record = self.record()
        record["physical_ssd_bandwidth_verified"] = True
        with self.assertRaises(transfer.TransferError):
            transfer.validate_record(record)

    def test_cache_policy_required(self):
        record = self.record()
        record["cache_policy"] = "cold"
        with self.assertRaises(transfer.TransferError):
            transfer.validate_record(record)

    def test_file_write_fsync_required(self):
        record = self.record("WRITE_FILE")
        record["file_io"]["fsync_in_timing"] = False
        with self.assertRaises(transfer.TransferError):
            transfer.validate_record(record)

    def test_buffered_io_opt_in(self):
        record = self.record()
        with self.assertRaises(transfer.TransferError):
            transfer.transfer_cost_entry(record)
        entry = transfer.transfer_cost_entry(record, allow_os_buffered_file=True)
        self.assertFalse(entry["physical_ssd_bandwidth_verified"])
        self.assertEqual(entry["source_kind"], "measured_transfer")
        self.assertEqual(entry["backend"], "os-buffered-file")
        self.assertIsNone(entry["workspace_bytes"])
        self.assertFalse(entry["capacity_feasibility_verified"])
        self.assertEqual(entry["statistics"], record["statistics"])

    def test_unstable_validation_only(self):
        record = self.record(samples=[100, 100, 10000])
        with self.assertRaises(transfer.TransferError):
            transfer.transfer_cost_entry(record, allow_os_buffered_file=True)
        entry = transfer.transfer_cost_entry(record, allow_unstable=True, allow_os_buffered_file=True)
        self.assertTrue(entry["validation_only"])

    def test_no_interpolation(self):
        with self.assertRaises(transfer.TransferError):
            transfer.lookup_transfer([self.record()], "ssd", "dram", 65, host_memory=None,
                                     allow_os_buffered_file=True)

    def test_no_ambiguous_duplicate(self):
        record = self.record()
        with self.assertRaises(transfer.TransferError):
            transfer.lookup_transfer([record, copy.deepcopy(record)], "ssd", "dram", 64,
                                     host_memory=None, allow_os_buffered_file=True)

    def test_failed_is_not_a_cost(self):
        record = self.record()
        record["status"] = "unsupported"
        with self.assertRaises(transfer.TransferError):
            transfer.transfer_cost_entry(record, allow_os_buffered_file=True)

    def test_valid_cuda(self):
        record, raw = self.cuda(self.native())
        self.assertEqual(record["status"], "measured")
        self.assertEqual(raw["native"]["wall_ns"], record["samples_ns"])
        self.assertIsNone(record["workspace_bytes"])
        transfer.validate_record(record)

    def test_cuda_binds_all_requested_axes(self):
        for field, value in (("direction", "D2H"), ("size_bytes", 65), ("host_memory", "pageable"),
                             ("warmup", 1), ("samples", 4), ("device_index", 1),
                             ("validation", "unchecked"), ("timing_scope", "kernel"),
                             ("ops_per_sample", 2), ("wall_ns", [100, 100])):
            with self.subTest(field=field):
                native = self.native()
                native[field] = value
                record, raw = self.cuda(native)
                self.assertEqual(record["status"], "failed")
                self.assertIn("stdout", raw)

    def test_cuda_missing_probe_is_unsupported(self):
        record, raw = transfer.cuda_record(self.directory / "missing", "H2D", 64, "pinned",
                                          2, 3, 0, [], 5)
        self.assertEqual(record["status"], "unsupported")
        self.assertIsNone(raw["stdout"])

    def test_cuda_timeout_preserves_output(self):
        probe = self.directory / "probe"
        probe.write_text("fake", encoding="utf-8")
        with patch.object(transfer.subprocess, "run", side_effect=subprocess.TimeoutExpired([], 1, b"partial", b"error")):
            record, raw = transfer.cuda_record(probe, "H2D", 64, "pinned", 2, 3, 0, [], 1)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(raw["stdout"], "partial")
        self.assertEqual(raw["stderr"], "error")

    def test_file_probe_validates_and_removes_own_file(self):
        for direction in ("READ_FILE", "WRITE_FILE"):
            with self.subTest(direction=direction):
                path = self.directory / (direction + ".bin")
                record, raw = transfer.file_record(path, direction, 1025, 1, 3)
                transfer.validate_record(record)
                self.assertFalse(path.exists())
                self.assertEqual(len(raw["python_probe"]["samples_ns"]), 3)
                self.assertEqual(record["file_io"]["physical_storage_type"], "unknown")

    def test_file_probe_does_not_overwrite_existing_file(self):
        path = self.directory / "existing.bin"
        path.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            transfer.file_record(path, "READ_FILE", 64, 0, 3)
        self.assertEqual(path.read_bytes(), b"keep")

    def test_explicit_lf(self):
        path = self.directory / "artifact.json"
        transfer.dump(path, {"unicode": "测量"})
        self.assertNotIn(b"\r\n", path.read_bytes())

    def test_cli_required_axes(self):
        with self.assertRaises(SystemExit):
            transfer.main([])

    def test_cli_cuda_requires_memory_policy(self):
        with self.assertRaises(SystemExit):
            transfer.main(["--sizes", "64", "--backends", "cuda", "--warmup", "0",
                           "--samples", "3", "--output", str(self.directory / "result")])

    def test_cli_refuses_nonempty_output(self):
        output = self.directory / "result"
        output.mkdir()
        (output / "keep").write_text("original", encoding="utf-8")
        with self.assertRaises(SystemExit):
            transfer.main(["--sizes", "64", "--backends", "os-buffered-file", "--io-directory", str(self.directory),
                           "--warmup", "0", "--samples", "3", "--output", str(output)])
        self.assertEqual((output / "keep").read_text(), "original")

    def test_sizes_reject_missing_duplicate_negative_bool(self):
        for value in ("", "64,64", "0", "-1", "True", "1073741825"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                transfer._sizes(value)

    def profile(self):
        directory = self.directory / "profile"
        directory.mkdir()
        record = self.record()
        record["copy_validation"] = {"passed": True}
        record["file_io"] = {"os_buffered": True, "fsync_in_timing": False}
        transfer.dump(directory / "manifest.json", {"schema_version": transfer.SCHEMA_VERSION,
                      "protocol": {"sizes_bytes": [64], "backends": ["os-buffered-file"], "warmup": 0, "samples": 3}})
        transfer.dump(directory / "raw.json", {"returncode": 0, "python_probe": {
            "samples_ns": record["samples_ns"], "copy_validation": record["copy_validation"], "file_io": record["file_io"]}})
        record["provenance"] = {"raw_output": "raw.json", "raw_sha256": transfer.sha256_file(directory / "raw.json"),
                                "manifest_sha256": transfer.sha256_file(directory / "manifest.json")}
        transfer.dump(directory / "measurements.json", {"schema_version": transfer.SCHEMA_VERSION,
                      "records": [record], "all_succeeded": True, "calibrated_moe_model": False})
        return directory

    def test_profile_loader_binds_raw_artifacts(self):
        profile = self.profile()
        document = transfer.load_transfer_profile(profile)
        self.assertEqual(len(document["records"]), 1)

    def test_profile_loader_rejects_raw_tamper(self):
        profile = self.profile()
        (profile / "raw.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(transfer.TransferError):
            transfer.load_transfer_profile(profile)

    def test_profile_loader_rejects_manifest_tamper(self):
        profile = self.profile()
        (profile / "manifest.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(transfer.TransferError):
            transfer.load_transfer_profile(profile)

    def test_profile_loader_rejects_forged_summary_with_raw_retained(self):
        profile = self.profile()
        document = json.loads((profile / "measurements.json").read_text(encoding="utf-8"))
        record = document["records"][0]
        record["samples_ns"] = [200, 200, 200]
        transfer._finish(record, record["samples_ns"])
        transfer.dump(profile / "measurements.json", document)
        with self.assertRaises(transfer.TransferError):
            transfer.load_transfer_profile(profile)

    def test_profile_loader_rejects_path_escape(self):
        profile = self.profile()
        document = json.loads((profile / "measurements.json").read_text(encoding="utf-8"))
        document["records"][0]["provenance"]["raw_output"] = "../escaped.json"
        transfer.dump(profile / "measurements.json", document)
        with self.assertRaises(transfer.TransferError):
            transfer.load_transfer_profile(profile)


if __name__ == "__main__":
    unittest.main()
