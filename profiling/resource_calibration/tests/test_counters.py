"""Deterministic fixtures only; no GPU/CPU workloads or external processes."""
import copy
import ctypes
import json
from pathlib import Path
import tempfile
import threading
import unittest

from profiling.resource_calibration import counters as c


class Clock:
    frequency_hz = 1000

    def __init__(self, tick=0):
        self.tick = tick

    def now(self):
        self.tick += 1
        return self.tick

    def metadata(self):
        return {"clock_id": "synthetic", "frequency_hz": self.frequency_hz}


class Function:
    def __init__(self, call):
        self.call = call

    def __call__(self, *arguments):
        return self.call(*arguments)


class Kernel:
    def __init__(self, host_ok=True, times_ok=True, memory_ok=True):
        self.host_ok, self.times_ok, self.memory_ok = host_ok, times_ok, memory_ok
        self.handles = []
        self.GlobalMemoryStatusEx = Function(self.host)
        self.GetProcessTimes = Function(self.times)
        self.K32GetProcessMemoryInfo = Function(self.memory)

    def host(self, pointer):
        self.host_size = pointer._obj.length
        pointer._obj.total_phys, pointer._obj.avail_phys = 4096, 2048
        return self.host_ok

    def times(self, handle, created, exited, system, user):
        self.handles.append(handle)
        for target, value in ((created, (1 << 33) + 1), (exited, 0), (system, 10_000_000), (user, 20_000_000)):
            target._obj.low, target._obj.high = value & 0xffffffff, value >> 32
        return self.times_ok

    def memory(self, handle, pointer, size):
        self.handles.append(handle)
        self.memory_size, self.cb_size = size, pointer._obj.cb
        pointer._obj.working_set, pointer._obj.private_usage = 1024, 3072
        return self.memory_ok


def pair():
    clock = Clock()
    left = c.read_snapshot(123, clock=clock, windows=True, kernel=Kernel())
    right = copy.deepcopy(left)
    for key in ("query_begin_tick", "query_end_tick"):
        right[key] += 1000
    for key in ("query_begin_tick", "query_end_tick", "cpu_counter_begin_tick", "cpu_counter_end_tick"):
        right["process"][key] += 1000
    right["process"]["values"]["kernel_time_100ns"] += 5_000_000
    right["process"]["values"]["user_time_100ns"] += 15_000_000
    return left, right


class SnapshotTests(unittest.TestCase):
    def test_reads_only_passed_handle_and_distinct_memory_quantities(self):
        kernel = Kernel()
        row = c.read_snapshot(123, clock=Clock(), windows=True, kernel=kernel)
        self.assertEqual(kernel.handles, [123, 123])
        self.assertEqual(row["host_memory"]["values"], {"total_physical_bytes": 4096, "available_physical_bytes": 2048})
        self.assertEqual(row["process"]["values"]["creation_time_100ns"], (1 << 33) + 1)
        self.assertEqual(row["process"]["values"]["working_set_bytes"], 1024)
        self.assertEqual(row["process"]["values"]["private_commit_bytes"], 3072)
        self.assertEqual(kernel.host_size, ctypes.sizeof(c._MemoryStatus))
        self.assertEqual(kernel.memory_size, ctypes.sizeof(c._ProcessMemory))
        self.assertEqual(kernel.cb_size, kernel.memory_size)
        self.assertLess(row["query_begin_tick"], row["process"]["cpu_counter_begin_tick"])
        self.assertLess(row["process"]["cpu_counter_end_tick"], row["query_end_tick"])
        json.dumps(row, allow_nan=False)

    def test_no_handle_queries_host_only(self):
        kernel = Kernel()
        row = c.read_snapshot(clock=Clock(), windows=True, kernel=kernel)
        self.assertEqual(kernel.handles, [])
        self.assertEqual(row["process"]["values"], {})
        self.assertEqual(set(row["process"]["unknown"]), set(c.PROCESS_FIELDS))
        self.assertIsNone(row["process"]["handle_identity"])

    def test_ctypes_void_pointer_handle(self):
        row = c.read_snapshot(ctypes.c_void_p(321), clock=Clock(), windows=True, kernel=Kernel())
        self.assertEqual(row["process"]["handle_identity"], 321)

    def test_invalid_handles_reject_before_query(self):
        for handle in (True, False, 0, -1, "123", 1.2, object(), ctypes.c_void_p()):
            with self.subTest(handle=handle), self.assertRaises(ValueError):
                c.read_snapshot(handle, clock=Clock(), windows=True, kernel=Kernel())

    def test_nonwindows_explicit_unknown(self):
        row = c.read_snapshot(123, clock=Clock(), windows=False)
        self.assertEqual(row["host_memory"]["status"], "unavailable")
        self.assertEqual(set(row["host_memory"]["unknown"]), set(c.MEMORY_FIELDS))
        self.assertEqual(set(row["process"]["unknown"]), set(c.PROCESS_FIELDS))

    def test_failed_host_does_not_erase_process(self):
        row = c.read_snapshot(123, clock=Clock(), windows=True, kernel=Kernel(host_ok=False))
        self.assertEqual(row["host_memory"]["values"], {})
        self.assertEqual(row["process"]["status"], "ok")

    def test_failed_cpu_preserves_memory_and_no_cpu_zero(self):
        row = c.read_snapshot(123, clock=Clock(), windows=True, kernel=Kernel(times_ok=False))
        self.assertEqual(row["process"]["status"], "partial")
        self.assertNotIn("kernel_time_100ns", row["process"]["values"])
        self.assertEqual(row["process"]["values"]["working_set_bytes"], 1024)
        self.assertIn("kernel_time_100ns", row["process"]["unknown"])

    def test_failed_memory_preserves_cpu(self):
        row = c.read_snapshot(123, clock=Clock(), windows=True, kernel=Kernel(memory_ok=False))
        self.assertEqual(row["process"]["status"], "partial")
        self.assertEqual(row["process"]["values"]["kernel_time_100ns"], 10_000_000)
        self.assertEqual(set(row["process"]["unknown"]), set(c.PROCESS_FIELDS[3:]))

    def test_missing_api_is_unknown(self):
        row = c.read_snapshot(123, clock=Clock(), windows=True, kernel=object())
        self.assertEqual(row["host_memory"]["status"], "unavailable")
        self.assertEqual(row["process"]["status"], "unavailable")

    def test_invalid_host_total_not_silently_accepted(self):
        kernel = Kernel()
        kernel.GlobalMemoryStatusEx = Function(lambda pointer: 1)
        row = c.read_snapshot(123, clock=Clock(), windows=True, kernel=kernel)
        self.assertEqual(row["host_memory"]["values"], {})


class DeltaTests(unittest.TestCase):
    def test_exact_core_equivalents_can_exceed_one(self):
        row = c.delta(*pair())
        self.assertEqual(row["values"]["core_equivalents"], 2)
        self.assertEqual(row["exact_core_equivalents"], {"numerator": 2, "denominator": 1})
        self.assertEqual(row["values"]["cpu_time_100ns"], 20_000_000)
        self.assertIn("Polling/spinning", row["normalization"])

    def test_uses_cpu_query_end_not_later_memory_end(self):
        left, right = pair()
        right["query_end_tick"] += 99
        right["process"]["query_end_tick"] += 99
        self.assertEqual(c.delta(left, right)["values"]["core_equivalents"], 2)

    def test_counter_reset_is_unknown(self):
        left, right = pair()
        right["process"]["values"]["kernel_time_100ns"] = 0
        row = c.delta(left, right)
        self.assertEqual(row["values"], {})
        self.assertIsNone(row["exact_core_equivalents"])

    def test_missing_counter_is_unknown(self):
        left, right = pair()
        del right["process"]["values"]["user_time_100ns"]
        self.assertEqual(c.delta(left, right)["status"], "unavailable")

    def test_boolean_or_negative_counter_is_unknown(self):
        for value in (True, -1, 1.5):
            left, right = pair()
            right["process"]["values"]["user_time_100ns"] = value
            self.assertEqual(c.delta(left, right)["values"], {})

    def test_nonpositive_interval_unknown(self):
        left, right = pair()
        right["process"]["cpu_counter_end_tick"] = left["process"]["cpu_counter_end_tick"]
        self.assertEqual(c.delta(left, right)["values"], {})

    def test_process_or_clock_change_rejected(self):
        for section, key, value in (("clock", "frequency_hz", 2000), ("clock", "clock_id", "other"),
                                    ("process", "handle_identity", 999)):
            left, right = pair()
            right[section][key] = value
            with self.assertRaises(ValueError):
                c.delta(left, right)
        left, right = pair()
        right["process"]["values"]["creation_time_100ns"] += 1
        with self.assertRaises(ValueError):
            c.delta(left, right)

    def test_schema_and_frequency_rejected(self):
        left, right = pair()
        left["schema"] = "other"
        with self.assertRaises(ValueError):
            c.delta(left, right)
        left, right = pair()
        left["clock"]["frequency_hz"] = right["clock"]["frequency_hz"] = 0
        with self.assertRaises(ValueError):
            c.delta(left, right)

    def test_no_process_is_not_zero_activity(self):
        left = c.read_snapshot(clock=Clock(), windows=False)
        right = copy.deepcopy(left)
        right["query_end_tick"] += 1000
        self.assertEqual(c.delta(left, right)["values"], {})


class SamplerTests(unittest.TestCase):
    def test_invalid_interval_or_labels(self):
        for value in (True, 0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                c.ProcessCounterSampler(interval_ms=value, clock=Clock())
        for handles in ({"host": 1}, {"": 1}, {1: 1}, {"main": None}):
            with self.assertRaises(ValueError):
                c.ProcessCounterSampler(handles, clock=Clock())

    def test_raw_collection_and_handle_update(self):
        clock = Clock()
        kernel = Kernel()
        reader = lambda handle, clock: c.read_snapshot(handle, clock=clock, windows=True, kernel=kernel)
        sampler = c.ProcessCounterSampler({"main": 123}, clock=clock, reader=reader)
        first = sampler._collect(0)
        sampler.records.append(first)
        second = sampler._collect(50)
        self.assertEqual(second["activities"]["main"]["values"]["core_equivalents"], 0)
        self.assertIn("host", first["snapshots"])
        self.assertEqual(first["activities"]["main"]["values"], {})
        sampler.set_handles({"main": 456})
        third = sampler._collect(100)
        self.assertEqual(third["activities"]["main"]["values"], {})
        self.assertEqual(third["snapshots"]["main"]["process"]["handle_identity"], 456)
        self.assertFalse(sampler.metadata()["handles_opened_or_closed"])

    def test_provider_failure_kept_as_unknown(self):
        def fail(handle, clock):
            raise OSError("fixture")
        sampler = c.ProcessCounterSampler({"main": 123}, clock=Clock(), reader=fail)
        row = sampler._collect(0)
        self.assertEqual(row["snapshots"]["main"]["status"], "error")
        self.assertIn("fixture", row["snapshots"]["main"]["unknown"]["provider"])

    def test_start_stop_and_exclusive_lf_output(self):
        ready = threading.Event()
        def reader(handle, clock):
            ready.set()
            return c.read_snapshot(handle, clock=clock, windows=False)
        sampler = c.ProcessCounterSampler(clock=Clock(), reader=reader).start()
        self.assertTrue(ready.wait(timeout=2))
        sampler.stop()
        self.assertTrue(sampler.metadata()["stopped"])
        self.assertIsNone(sampler.metadata()["thread_error"])
        with self.assertRaises(RuntimeError):
            sampler.start()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counters.jsonl"
            sampler.write_jsonl(path)
            self.assertNotIn(b"\r\n", path.read_bytes())
            self.assertEqual(len(path.read_text().splitlines()), len(sampler.records))
            with self.assertRaises(FileExistsError):
                sampler.write_jsonl(path)

    def test_stop_before_start_rejects(self):
        with self.assertRaises(RuntimeError):
            c.ProcessCounterSampler(clock=Clock()).stop()


if __name__ == "__main__":
    unittest.main()
