"""Measured SSD records become a loadable ``m2-layer-0.7`` cost table.

Nothing here touches a disk: records are built with ``build_record`` so the
exporter sees the same validated shape the measurement tool writes. The last
class loads the exported table through the real parser, because "the exporter
produced JSON" is not the claim -- "the kernel will accept it" is.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from profiling.storage import ssd_measure as ssd
from profiling.storage import transfer_table as export

try:  # the contract test needs the model; the exporter itself must not
    from llm_infer_model.m2.costs import canonical_contract_hash, parse_cost_table
    from llm_infer_model.tensor.spec import Unsupported
except ImportError:  # pragma: no cover - only when run without the model package
    canonical_contract_hash = parse_cost_table = Unsupported = None


SIGNATURE = {"contract": "test_macro_boundary", "device": "cpu", "tokens": 3}


def measured_document(sizes=(4096, 8192), mechanism="buffered", flush_ns=300,
                      write_ns=2000):
    records = []
    for size in sizes:
        records.append(ssd.build_record(
            "READ_FILE", size, [1000, 1000, 1000], cache_bypass=mechanism, warmup=0,
            random_offset=True, backend="test", sample_offsets=[0, 4096, 8192]))
        records.append(ssd.build_record(
            "WRITE_FILE", size, [write_ns, write_ns, write_ns], cache_bypass=mechanism,
            warmup=0, random_offset=True, backend="test",
            flush_samples=[flush_ns, flush_ns, flush_ns], sample_offsets=[0, 4096, 8192]))
    return {
        "schema_version": ssd.SCHEMA_VERSION,
        "started_at": "2026-11-03T00:00:00+00:00",
        "finished_at": "2026-11-03T00:01:00+00:00",
        "platform": {"label": "unit-test-laptop"},
        "storage_device": None,
        "protocol": {"sizes_bytes": list(sizes), "samples": 3, "warmup": 0},
        "all_measured": True,
        "measurements": records,
    }


def base_table(version="m2-layer-0.5", transfer_entries=None):
    entries = transfer_entries if transfer_entries is not None else [
        {"route": "dram_to_vram", "storage_bytes": 4096, "duration_ns": 20,
         "provenance": "synthetic_transfer"}]
    return {
        "format_version": version,
        "source": "unit-base",
        "entries": [{
            "kind": "projection", "device": "cpu", "phase": "prefill", "batch": 1,
            "tokens": 3, "visible_history": 0, "duration_ns": 100,
            "workspace_bytes": 0, "provenance": "synthetic_layer",
            "physical_signature": dict(SIGNATURE),
            "contract_hash": (canonical_contract_hash(SIGNATURE)
                              if canonical_contract_hash else "0" * 64),
        }],
        "transfer_entries": entries,
    }


class OneRecordOneKey(unittest.TestCase):
    def test_every_measured_record_becomes_one_exact_entry(self):
        entries = export.transfer_entries(measured_document())
        self.assertEqual([(e["route"], e["storage_bytes"], e["duration_ns"]) for e in entries],
                         [("dram_to_ssd", 4096, 2000), ("dram_to_ssd", 8192, 2000),
                          ("ssd_to_dram", 4096, 1000), ("ssd_to_dram", 8192, 1000)])
        for entry in entries:
            self.assertEqual(entry["provenance"], "measured_transfer")

    def test_an_unmeasured_size_is_absent_rather_than_borrowed(self):
        entries = export.transfer_entries(measured_document(sizes=(4096,)))
        self.assertNotIn(8192, {entry["storage_bytes"] for entry in entries})
        self.assertEqual(export.missing_sizes(entries, [4096, 8192]),
                         ["ssd_to_dram/8192", "dram_to_ssd/8192"])

    def test_requiring_an_unmeasured_size_refuses_to_export(self):
        with self.assertRaisesRegex(export.TransferTableError, "8192"):
            export.build_cost_table(base_table(), measured_document(sizes=(4096,)),
                                    require_sizes=[4096, 8192])

    def test_the_measured_cache_mechanism_is_carried_verbatim(self):
        entries = export.transfer_entries(measured_document(mechanism="unbuffered"))
        self.assertEqual({entry["cache_policy"] for entry in entries}, {"unbuffered"})


class FlushIsNamedNotBuried(unittest.TestCase):
    def test_excluding_flush_keeps_the_write_as_measured(self):
        entries = export.transfer_entries(measured_document(flush_ns=300), flush="exclude")
        write = next(entry for entry in entries if entry["route"] == "dram_to_ssd")
        self.assertEqual(write["duration_ns"], 2000)
        self.assertEqual(write["timing_scope"], export.WRITE_SCOPE_EXCLUDING_FLUSH)

    def test_including_flush_says_it_is_a_sum_of_medians(self):
        entries = export.transfer_entries(measured_document(flush_ns=300), flush="include")
        write = next(entry for entry in entries if entry["route"] == "dram_to_ssd")
        read = next(entry for entry in entries if entry["route"] == "ssd_to_dram")
        self.assertEqual(write["duration_ns"], 2300)
        self.assertEqual(write["timing_scope"], export.WRITE_SCOPE_WITH_FLUSH)
        self.assertEqual(read["duration_ns"], 1000)  # reads have no flush
        self.assertEqual(read["timing_scope"], export.READ_SCOPE)


class TheDocumentMustBeComplete(unittest.TestCase):
    def test_a_failed_record_stops_the_export(self):
        document = measured_document()
        document["measurements"].append({
            "id": "read_file_ssd_12288", "status": "failed", "direction": "READ_FILE",
            "route": {"source": "ssd", "destination": "dram"}, "size_bytes": 12288,
            "reason": "Access is denied",
        })
        with self.assertRaisesRegex(export.TransferTableError, "read_file_ssd_12288"):
            export.transfer_entries(document)

    def test_a_foreign_schema_is_refused(self):
        document = copy.deepcopy(measured_document())
        document["schema_version"] = "ssd_storage_measurements_v0"
        with self.assertRaises(export.TransferTableError):
            export.transfer_entries(document)

    def test_two_records_for_one_route_and_size_are_refused(self):
        document = measured_document()
        document["measurements"].append(copy.deepcopy(document["measurements"][0]))
        with self.assertRaisesRegex(export.TransferTableError, "duplicate"):
            export.transfer_entries(document)


class MergingRespectsTheBase(unittest.TestCase):
    def test_compute_entries_survive_and_the_version_is_promoted(self):
        table = export.build_cost_table(base_table(), measured_document())
        self.assertEqual(table["format_version"], "m2-layer-0.7")
        self.assertEqual(table["entries"][0]["kind"], "projection")
        self.assertEqual(table["source"], "unit-base+ssd-measurement:unit-test-laptop")

    def test_a_point_six_expert_table_cannot_be_relabelled_as_dense(self):
        with self.assertRaisesRegex(export.TransferTableError, "m2-layer-0.6"):
            export.build_cost_table(base_table(version="m2-layer-0.6"), measured_document())

    def test_an_existing_price_is_not_silently_overwritten(self):
        occupied = base_table(transfer_entries=[
            {"route": "ssd_to_dram", "storage_bytes": 4096, "duration_ns": 1,
             "provenance": "synthetic_transfer"}])
        with self.assertRaisesRegex(export.TransferTableError, "replace"):
            export.build_cost_table(occupied, measured_document())

    def test_explicit_replace_prices_the_move_once(self):
        occupied = base_table(transfer_entries=[
            {"route": "ssd_to_dram", "storage_bytes": 4096, "duration_ns": 1,
             "provenance": "synthetic_transfer"}])
        table = export.build_cost_table(occupied, measured_document(), replace=True)
        keys = [(entry["route"], entry["storage_bytes"]) for entry in table["transfer_entries"]]
        self.assertEqual(keys.count(("ssd_to_dram", 4096)), 1)
        measured = next(entry for entry in table["transfer_entries"]
                        if (entry["route"], entry["storage_bytes"]) == ("ssd_to_dram", 4096))
        self.assertEqual(measured["duration_ns"], 1000)


class TheResultLoads(unittest.TestCase):
    @unittest.skipIf(parse_cost_table is None, "llm_infer_model is not importable")
    def test_a_bypassing_measurement_keeps_a_full_claim(self):
        document = measured_document(mechanism="unbuffered")
        table = parse_cost_table(export.build_cost_table(base_table(), document))
        self.assertFalse(table.validation_only)
        entry = table.lookup_transfer("ssd_to_dram", 4096)
        self.assertEqual((entry.duration_ns, entry.cache_policy),
                         (1000, "unbuffered"))
        self.assertEqual(entry.timing_scope, export.READ_SCOPE)
        self.assertEqual(table.lookup_transfer("dram_to_vram", 4096).duration_ns, 20)

    @unittest.skipIf(parse_cost_table is None, "llm_infer_model is not importable")
    def test_a_buffered_measurement_is_forced_to_validation_only(self):
        table = parse_cost_table(export.build_cost_table(base_table(), measured_document()))
        self.assertTrue(table.validation_only)
        self.assertEqual(table.lookup_transfer("dram_to_ssd", 8192).cache_policy, "buffered")

    @unittest.skipIf(parse_cost_table is None, "llm_infer_model is not importable")
    def test_an_exported_size_cannot_stand_in_for_a_neighbour(self):
        table = parse_cost_table(export.build_cost_table(base_table(), measured_document()))
        with self.assertRaises(Unsupported):
            table.lookup_transfer("ssd_to_dram", 8193)


class TheWriteRegimeIsASecondCampaign(unittest.TestCase):
    """A write has two prices; the table must say when the second starts."""

    def sustained(self, sizes=(4096, 8192), **kwargs):
        return measured_document(sizes=sizes, write_ns=11000, **kwargs)

    def build(self, base=None, document=None, sustained=None, after=1 << 30, **kwargs):
        return export.build_cost_table(
            base or base_table(), document or measured_document(mechanism="unbuffered"),
            sustained_document=self.sustained() if sustained is None else sustained,
            sustained_after_bytes=after, **kwargs)

    def test_the_second_price_lands_on_the_write_entries_only(self):
        table = self.build()
        writes = {entry["storage_bytes"]: entry for entry in table["transfer_entries"]
                  if entry["route"] == "dram_to_ssd"}
        reads = [entry for entry in table["transfer_entries"]
                 if entry["route"] == "ssd_to_dram"]
        self.assertEqual({size: entry["sustained_duration_ns"]
                          for size, entry in writes.items()},
                         {4096: 11000, 8192: 11000})
        self.assertEqual([entry["duration_ns"] for entry in reads], [1000, 1000])
        self.assertTrue(all("sustained_duration_ns" not in entry for entry in reads))
        self.assertEqual(table["storage_write_regime"],
                         {"after_cumulative_bytes": 1 << 30})

    def test_the_threshold_has_to_be_supplied_by_the_campaign(self):
        with self.assertRaisesRegex(export.TransferTableError, "sustained-after-bytes"):
            export.build_cost_table(base_table(), measured_document(),
                                    sustained_document=self.sustained())

    def test_a_size_missing_from_the_sustained_campaign_is_refused(self):
        with self.assertRaisesRegex(export.TransferTableError, "8192"):
            self.build(sustained=self.sustained(sizes=(4096,)))

    def test_a_sustained_campaign_that_measured_more_sizes_is_refused(self):
        with self.assertRaisesRegex(export.TransferTableError, "16384"):
            self.build(document=measured_document(sizes=(4096,)),
                       sustained=self.sustained(sizes=(4096, 16384)))

    def test_a_sustained_price_faster_than_the_burst_is_refused(self):
        with self.assertRaisesRegex(export.TransferTableError, "swapped"):
            self.build(sustained=measured_document(write_ns=500))

    def test_a_threshold_without_a_second_campaign_changes_nothing(self):
        with self.assertRaisesRegex(export.TransferTableError, "sustained-measurements"):
            export.build_cost_table(base_table(), measured_document(),
                                    sustained_after_bytes=1 << 30)

    def test_a_burst_only_export_cannot_sit_on_a_regimed_base(self):
        table = self.build()
        with self.assertRaisesRegex(export.TransferTableError, "regime"):
            export.build_cost_table(table, measured_document())

    @unittest.skipIf(parse_cost_table is None, "llm_infer_model is not importable")
    def test_the_exported_regime_loads_and_selects_the_second_price(self):
        table = parse_cost_table(self.build())
        self.assertEqual(table.storage_write_regime.after_cumulative_bytes, 1 << 30)
        entry = table.lookup_transfer("dram_to_ssd", 4096)
        self.assertEqual((entry.duration_ns, entry.sustained_duration_ns),
                         (2000, 11000))
        self.assertIsNone(table.lookup_transfer("ssd_to_dram", 4096).sustained_duration_ns)

    def test_the_comment_says_which_campaign_each_half_came_from(self):
        comment = json.loads(self.build()["comment"])
        self.assertTrue(comment["write_regime_exported"])
        self.assertEqual(comment["sustained_after_cumulative_bytes"], 1 << 30)
        self.assertIn("garbage collection", comment["sustained_reads_not_used"])


class CliTests(unittest.TestCase):
    def test_cli_writes_a_table_and_never_overwrites(self):
        repository = Path(__file__).resolve().parents[3]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = (
            str(repository) + os.pathsep + environment.get("PYTHONPATH", ""))
        with tempfile.TemporaryDirectory() as temporary:
            measurements = Path(temporary) / "measurements.json"
            measurements.write_text(json.dumps(measured_document()), encoding="utf-8")
            base = Path(temporary) / "base.costs.json"
            base.write_text(json.dumps(base_table()), encoding="utf-8")
            output = Path(temporary) / "measured.costs.json"
            command = [sys.executable, "-m", "profiling.storage.transfer_table",
                       "--measurements", str(measurements), "--base", str(base),
                       "--output", str(output)]
            completed = subprocess.run(command, cwd=repository, env=environment,
                                       capture_output=True, text=True, timeout=120)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            table = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(table["format_version"], "m2-layer-0.7")
            self.assertTrue(table["validation_only"])
            again = subprocess.run(command, cwd=repository, env=environment,
                                   capture_output=True, text=True, timeout=120)
            self.assertEqual(again.returncode, 2)
            self.assertIn("must not exist", again.stderr)


if __name__ == "__main__":
    unittest.main()
