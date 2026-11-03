"""A phone or a Mac reports observations; this process does the arithmetic.

The device cannot be trusted to compute ``statistics``: it runs on another
machine, in another language, and a document that carries derived values has
already made a claim nobody here can check. So the importer takes raw
nanosecond samples, the offsets the device actually used, and the bypass API it
actually called, and runs them through the same ``build_record`` the Windows
apparatus uses.

Every test below pins one thing the importer refuses to fill in.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from profiling.storage import import_device_measurements as importer
from profiling.storage import transfer_table as export

SIZE = 65536
BACKING = SIZE + 4096
READS = [30_000, 31_000, 29_000]
WRITES = [100_000, 102_000, 101_000]
FLUSHES = [50_000, 51_000, 49_000]


def observation(size: int = SIZE, *, backing: int = BACKING, reads=READS,
                writes=WRITES, flushes=FLUSHES, offsets=None) -> dict:
    record = {
        "size_bytes": size,
        "backing_file_bytes": backing,
        "read_samples_ns": list(reads),
        "write_samples_ns": list(writes),
        "write_flush_samples_ns": list(flushes),
    }
    if offsets is not None:
        record["read_offsets_bytes"] = list(offsets)
        record["write_offsets_bytes"] = list(offsets)
    return record


def document(**overrides) -> dict:
    raw = {
        "schema_version": importer.DEVICE_SCHEMA_VERSION,
        "started_at": "2026-11-04T01:00:00+00:00",
        "finished_at": "2026-11-04T01:02:00+00:00",
        "platform_label": "iphone-test",
        "platform_api": "F_NOCACHE",
        "system": "iOS",
        "release": "18.2",
        "machine": "iPhone17,1",
        "device": {"model": "iPhone17,1"},
        "thermal_state": "nominal",
        "battery_level": 0.91,
        "directory": "/var/mobile/Documents/ssd-measure",
        "protocol": {
            "sizes_bytes": [SIZE],
            "samples": 3,
            "warmup": 3,
            "alignment_bytes": 4096,
            "random_offset": True,
        },
        "observations": [observation()],
    }
    raw.update(overrides)
    return raw


class TheDeviceOnlySuppliesObservations(unittest.TestCase):
    """Derived values come from this process, never from the device."""

    def test_the_import_computes_statistics_from_raw_samples(self):
        imported = importer.import_document(document())
        read, write = imported["measurements"]
        self.assertEqual(read["direction"], "READ_FILE")
        self.assertEqual(read["samples_ns"], READS)
        self.assertEqual(read["statistics"]["median_ns"], 30_000)
        self.assertEqual(read["duration_ns"], 30_000)
        self.assertEqual(write["flush_samples_ns"], FLUSHES)
        self.assertEqual(write["flush_ns"], 50_000)
        # The flush never enters the write duration.
        self.assertEqual(write["duration_ns"], 101_000)

    def test_the_achieved_api_is_what_licenses_the_bypass_label(self):
        imported = importer.import_document(document())
        for record in imported["measurements"]:
            self.assertEqual(record["cache_bypass"], "f_nocache")
            self.assertTrue(record["physical_ssd_bandwidth_verified"])
        self.assertEqual(imported["platform"]["platform_api"], "F_NOCACHE")

    def test_a_global_nocache_fd_is_the_same_mechanism(self):
        imported = importer.import_document(document(platform_api="F_GLOBAL_NOCACHE"))
        self.assertEqual(imported["measurements"][0]["cache_bypass"], "f_nocache")

    def test_the_imported_document_names_where_it_came_from(self):
        imported = importer.import_document(document())
        self.assertEqual(imported["protocol"]["imported_from"],
                         importer.DEVICE_SCHEMA_VERSION)
        self.assertEqual(imported["thermal_state"], "nominal")
        self.assertEqual(imported["battery_level"], 0.91)


class TheImporterRefusesToFillGaps(unittest.TestCase):
    """A missing field is an error, not a zero."""

    def test_an_unknown_schema_is_not_guessed(self):
        with self.assertRaisesRegex(importer.DeviceDocumentError, "schema_version"):
            importer.import_document(document(schema_version="ssd_measurements_v9"))

    def test_a_document_that_names_no_bypass_api_cannot_claim_one(self):
        with self.assertRaisesRegex(importer.DeviceDocumentError, "platform_api"):
            importer.import_document(document(platform_api="none"))
        raw = document()
        del raw["platform_api"]
        with self.assertRaisesRegex(importer.DeviceDocumentError, "platform_api"):
            importer.import_document(raw)

    def test_an_unaligned_size_is_refused(self):
        raw = document()
        raw["protocol"]["sizes_bytes"] = [SIZE + 1]
        raw["observations"] = [observation(size=SIZE + 1)]
        with self.assertRaisesRegex(importer.DeviceDocumentError, "multiples of 4096"):
            importer.import_document(raw)

    def test_fewer_than_three_samples_is_not_a_measurement(self):
        raw = document()
        raw["observations"] = [observation(reads=[1, 2])]
        with self.assertRaisesRegex(importer.DeviceDocumentError, "at least 3"):
            importer.import_document(raw)

    def test_a_write_without_one_flush_per_sample_is_refused(self):
        raw = document()
        raw["observations"] = [observation(flushes=[50_000])]
        with self.assertRaisesRegex(importer.DeviceDocumentError,
                                    "one F_FULLFSYNC per write"):
            importer.import_document(raw)

    def test_a_declared_size_that_was_not_measured_is_refused(self):
        raw = document()
        raw["protocol"]["sizes_bytes"] = [SIZE, 131072]
        with self.assertRaisesRegex(importer.DeviceDocumentError, "never measured"):
            importer.import_document(raw)

    def test_a_measured_size_that_was_not_declared_is_refused(self):
        raw = document()
        raw["protocol"]["sizes_bytes"] = [131072]
        with self.assertRaisesRegex(importer.DeviceDocumentError, "does not declare"):
            importer.import_document(raw)

    def test_a_repeated_size_is_refused(self):
        raw = document()
        raw["observations"] = [observation(), observation()]
        with self.assertRaisesRegex(importer.DeviceDocumentError, "repeats"):
            importer.import_document(raw)

    def test_an_offset_outside_the_declared_file_is_refused(self):
        raw = document()
        raw["observations"] = [observation(offsets=[BACKING - SIZE + 4096] * 3)]
        with self.assertRaisesRegex(importer.DeviceDocumentError, "leaves the declared"):
            importer.import_document(raw)

    def test_an_unaligned_offset_is_refused(self):
        raw = document()
        raw["observations"] = [observation(offsets=[4097] * 3)]
        with self.assertRaisesRegex(importer.DeviceDocumentError, "not a multiple of 4096"):
            importer.import_document(raw)

    def test_an_extra_field_nobody_reads_is_refused(self):
        raw = document()
        raw["observations"] = [dict(observation(), statistics={"median_ns": 1})]
        with self.assertRaisesRegex(importer.DeviceDocumentError, "does not understand"):
            importer.import_document(raw)

    def test_a_battery_level_outside_zero_to_one_is_refused(self):
        with self.assertRaisesRegex(importer.DeviceDocumentError, "battery_level"):
            importer.import_document(document(battery_level=91))


class TheImportedDocumentReachesTheCostTable(unittest.TestCase):
    """The point is not a valid JSON file; the kernel has to accept the table."""

    def test_the_device_records_export_into_a_loadable_cost_table(self):
        imported = importer.import_document(document())
        base = Path(__file__).resolve().parents[3] / "mapping/examples/m2" \
            / "storage-interface-synthetic.costs.json"
        table = export.build_cost_table(
            json.loads(base.read_text(encoding="utf-8")), imported,
            require_sizes=[SIZE],
        )
        entries = table["transfer_entries"]
        read = [entry for entry in entries
                if entry["route"] == "ssd_to_dram" and entry["storage_bytes"] == SIZE]
        write = [entry for entry in entries
                 if entry["route"] == "dram_to_ssd" and entry["storage_bytes"] == SIZE]
        self.assertEqual(len(read), 1)
        self.assertEqual(len(write), 1)
        self.assertEqual(read[0]["duration_ns"], 30_000)
        self.assertEqual(write[0]["duration_ns"], 101_000)
        self.assertEqual(read[0]["provenance"], "measured_transfer")
        self.assertEqual(read[0]["cache_policy"], "f_nocache")

    def test_the_command_line_writes_a_document_the_exporter_reads(self):
        source = document()
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            observations = folder / "iphone.json"
            output = folder / "iphone.measurements.json"
            observations.write_text(json.dumps(source), encoding="utf-8")
            code = importer.main(["--observations", str(observations),
                                  "--output", str(output)])
            self.assertEqual(code, 0)
            loaded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(loaded["measurements"]), 2)
            self.assertTrue(output.exists())
            # Refuses to overwrite without --force.
            with self.assertRaises(SystemExit):
                importer.main(["--observations", str(observations),
                               "--output", str(output)])


if __name__ == "__main__":
    unittest.main()
