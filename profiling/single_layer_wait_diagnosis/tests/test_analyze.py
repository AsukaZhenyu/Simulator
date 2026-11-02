"""Synthetic SQLite interval/correlation checks; no GPU or simulator required."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "analyze.py"
SPEC = importlib.util.spec_from_file_location("wait_analyze", SOURCE)
analyze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyze)


class AnalyzeTests(unittest.TestCase):
    def setUp(self):
        self.connection = sqlite3.connect(":memory:")
        self.addCleanup(self.connection.close)
        c = self.connection
        c.executescript("""
        CREATE TABLE StringIds(id INTEGER,value TEXT);
        CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,category INTEGER,text TEXT,textId INTEGER,globalTid INTEGER,endGlobalTid INTEGER,eventType INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER,returnValue INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,globalPid INTEGER,correlationId INTEGER,contextId INTEGER,streamId INTEGER,shortName INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,globalPid INTEGER,correlationId INTEGER,contextId INTEGER,streamId INTEGER,bytes INTEGER,copyKind INTEGER);
        CREATE TABLE ANALYSIS_FILE(globalPid INTEGER,content TEXT);
        CREATE TABLE TARGET_INFO_CUDA_CONTEXT_INFO(processId INTEGER,contextId INTEGER);
        CREATE TABLE DIAGNOSTIC_EVENT(severity INTEGER,text TEXT);
        CREATE TABLE META_DATA_EXPORT(name TEXT,value TEXT);
        """)
        self.pid = (1 << 48) + (42 << 24)
        self.tid = self.pid + 7
        self.native = {"format_version": "single-layer-context-pair-native-v2", "pair_kind": "context_vs_isolated",
            "process_id": 42, "target_action_index": 14, "target_action": {"kind": "COMPUTE", "device": "gpu", "target_id": "PV"},
            "warmup_per_block": 1, "samples_per_block": 3, "block_order": "ABBA",
            "blocks": [{"block_index": i, "label": label,
                        "raw": {"mode": "isolated-components" if label == "A" else "context-trace"}} for i, label in enumerate("ABBA")],
            "diagnostic_only": True, "usable_for_parameter_fit": False, "usable_for_heldout_acceptance": False,
            "nvtx_diagnostic": {"enabled": True, "target_range_count_per_block": 5,
                "target_timed_range_count_per_block": 3, "target_validation_range_count_per_block": 1,
                "target_warmup_range_count_per_block": 1, "categories": {"block": 1, "gpu_compute": 11},
                "block_range_names": ["pair_block_" + str(i) for i in range(4)],
                "usable_for_parameter_fit": False, "usable_for_heldout_acceptance": False}}
        c.executemany("INSERT INTO StringIds VALUES(?,?)", [(1, "cudaLaunchKernel_v7000"), (2, "cudaMemcpyAsync_v3020"), (3, "cudaStreamSynchronize_v3020")])
        c.execute("INSERT INTO ANALYSIS_FILE VALUES(?,?)", (self.pid, json.dumps({"process_id": 42, "status": "ok"})))
        c.execute("INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO VALUES(42,1)")
        c.execute("INSERT INTO META_DATA_EXPORT VALUES('EXPORT_SCHEMA_VERSION','synthetic')")
        for block in range(4):
            first = 1000 + block * 10000
            c.execute("INSERT INTO NVTX_EVENTS VALUES(?,?,1,?,NULL,?,NULL,59)", (first, first + 9999, "pair_block_" + str(block), self.tid))
            for index in range(5):
                start = first + 100 + index * 100
                c.execute("INSERT INTO NVTX_EVENTS VALUES(?,?,11,'PV',NULL,?,NULL,59)", (start, start + 80, self.tid))
                correlation = (block * 5 + index) * 10
                for offset, left, right, name in ((1, 1, 10, 1), (2, 12, 18, 2), (3, 40, 50, 3), (4, 52, 55, 3)):
                    c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?,?,0)", (start + left, start + right, self.tid, correlation + offset, name))
                c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,1,1,1)", (start + 15, start + 25, self.pid, correlation + 1))
                c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(?,?,?,?,1,1,4,1)", (start + 28, start + 35, self.pid, correlation + 2))

    def analyze(self):
        return analyze.analyze_trace(self.connection, self.native)

    def test_only_last_samples_selected_and_cpu_gpu_not_added(self):
        result = self.analyze()
        self.assertEqual(len(result["blocks"]), 4)
        first = result["blocks"][0]
        self.assertEqual(first["target_ranges_total"], 5)
        self.assertEqual(first["warmup_ranges_excluded"], 1)
        self.assertEqual(len(first["invocations"]), 3)
        self.assertEqual(first["statistics"]["host_nvtx_span_ns"]["p50"], 80)
        self.assertEqual(first["statistics"]["correlated_gpu_activity_union_ns"]["p50"], 17)
        self.assertEqual(first["statistics"]["first_sync_api_ns"]["p50"], 10)
        self.assertEqual(first["statistics"]["second_sync_api_ns"]["p50"], 3)

    def test_interval_union_handles_overlap_without_double_counting(self):
        self.assertEqual(analyze.union_length([(1, 5), (3, 8), (8, 10), (20, 20)]), 9)
        self.assertEqual(analyze.union_length([]), 0)
        with self.assertRaises(ValueError):
            analyze.union_length([(5, 3)])

    def test_missing_target_range_rejected(self):
        self.connection.execute("DELETE FROM NVTX_EVENTS WHERE category=11 AND start=(SELECT MIN(start) FROM NVTX_EVENTS WHERE category=11)")
        with self.assertRaisesRegex(ValueError, "missing/extra target"):
            self.analyze()

    def test_wrong_target_thread_rejected(self):
        self.connection.execute("UPDATE NVTX_EVENTS SET globalTid=globalTid+1 WHERE category=11 AND start=(SELECT MIN(start) FROM NVTX_EVENTS WHERE category=11)")
        with self.assertRaisesRegex(ValueError, "thread/parent"):
            self.analyze()

    def test_missing_kernel_correlation_rejected(self):
        self.connection.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE correlationId=21")
        with self.assertRaisesRegex(ValueError, "missing correlated GPU"):
            self.analyze()

    def test_wrong_gpu_process_or_context_rejected(self):
        self.connection.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET contextId=99 WHERE correlationId=21")
        with self.assertRaisesRegex(ValueError, "process/context"):
            self.analyze()

    def test_gpu_outside_completed_boundary_rejected(self):
        self.connection.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end=end+100 WHERE correlationId=21")
        with self.assertRaisesRegex(ValueError, "escapes completed"):
            self.analyze()

    def test_duplicate_runtime_correlation_rejected(self):
        self.connection.execute("UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET correlationId=21 WHERE correlationId=22")
        with self.assertRaisesRegex(ValueError, "ambiguous runtime"):
            self.analyze()

    def test_missing_second_sync_rejected(self):
        self.connection.execute("DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId=24")
        with self.assertRaisesRegex(ValueError, "exactly two"):
            self.analyze()

    def test_profiler_dropped_events_never_silently_accepted(self):
        self.connection.execute("INSERT INTO DIAGNOSTIC_EVENT VALUES(2,'Dropped events due to buffer overflow')")
        with self.assertRaisesRegex(ValueError, "lost/dropped"):
            self.analyze()

    def test_signed_queue_difference_preserves_overlap(self):
        self.connection.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET start=start-10 WHERE correlationId=21")
        result = self.analyze()
        first = result["blocks"][0]["invocations"][0]
        self.assertIn(-5, first["post_api_return_to_gpu_start_ns"])

    def test_quantiles_retain_worst_observation(self):
        values = [1] * 99 + [1000]
        result = analyze.stats(values)
        self.assertEqual(result["max"], 1000)
        self.assertEqual(result["n"], 100)
        self.assertEqual(result["p99"], 1)

    def test_compact_export_retains_quantiles_warnings_and_full_hash(self):
        process = self.analyze()
        report = {"schema": "full", "reports": [process], "diagnostic_only": True,
                  "limits": ["profiler perturbation"], "source_files": [{"path": "trace.sqlite", "sha256": "a" * 64}]}
        parent = Path(__file__).resolve().parents[3] / "build/wait-compact-tests"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as folder:
            path = Path(folder) / "full.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            compact = analyze.compact_report(report, path)
            self.assertEqual(compact["full_decomposition_evidence"]["sha256"], analyze.sha(path))
        self.assertEqual(compact["limits"], report["limits"])
        self.assertEqual(compact["reports"][0]["blocks"][0]["statistics"], process["blocks"][0]["statistics"])
        self.assertNotIn("invocations", compact["reports"][0]["blocks"][0])
        self.assertEqual(compact["reports"][0]["blocks"][0]["full_invocation_count"], 3)
        self.assertTrue(compact["all_block_quantiles_retained"])


if __name__ == "__main__":
    unittest.main()
