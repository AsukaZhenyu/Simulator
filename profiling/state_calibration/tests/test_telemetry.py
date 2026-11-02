"""Deterministic contracts; no GPU, NVML installation or hardware workload."""
import ctypes
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest

from profiling.state_calibration import telemetry as t


class FakeClock:
    frequency_hz = 1000
    clock_id = "synthetic_test_clock"

    def __init__(self, tick=0, step=1):
        self.tick, self.step = tick, step

    def now(self):
        value = self.tick
        self.tick += self.step
        return value

    def metadata(self):
        return {"clock_id": self.clock_id, "frequency_hz": self.frequency_hz}


class Provider:
    def __init__(self, read=None):
        self._read = read
        self.closed = False

    def read(self):
        return self._read() if self._read else {"status": "ok", "values": {"sm_clock_mhz": 1200}, "unknown": {}}

    def metadata(self):
        return {"provider": "synthetic", "closed": self.closed}

    def close(self):
        self.closed = True


class Function:
    def __init__(self, function):
        self.function = function

    def __call__(self, *arguments):
        return self.function(*arguments)


class FakeNVML:
    def __init__(self, power_supported=True, init_status=0):
        self.shutdowns = 0
        self.nvmlInit_v2 = Function(lambda: init_status)
        self.nvmlShutdown = Function(self.shutdown)
        self.nvmlDeviceGetHandleByIndex_v2 = Function(self.handle)
        for name in ("nvmlDeviceGetUUID", "nvmlDeviceGetName", "nvmlSystemGetNVMLVersion", "nvmlSystemGetDriverVersion"):
            setattr(self, name, Function(self.string))
        self.nvmlDeviceGetPerformanceState = Function(lambda handle, pointer: self.scalar(pointer, 2))
        self.nvmlDeviceGetClockInfo = Function(lambda handle, kind, pointer: self.scalar(pointer, 1000 + kind))
        self.nvmlDeviceGetTemperature = Function(lambda handle, sensor, pointer: self.scalar(pointer, 65))
        self.nvmlDeviceGetPowerUsage = Function(lambda handle, pointer: self.scalar(pointer, 45000) if power_supported else 3)
        self.nvmlDeviceGetUtilizationRates = Function(self.utilization)
        self.nvmlDeviceGetMemoryInfo = Function(self.memory)

    def handle(self, index, pointer):
        pointer._obj.value = 0x1234
        return 0

    def string(self, *arguments):
        arguments[-2].value = b"SYNTHETIC"
        return 0

    def scalar(self, pointer, value):
        pointer._obj.value = value
        return 0

    def utilization(self, handle, pointer):
        pointer._obj.gpu, pointer._obj.memory = 75, 30
        return 0

    def memory(self, handle, pointer):
        pointer._obj.total, pointer._obj.free, pointer._obj.used = 10000, 6000, 4000
        return 0

    def shutdown(self):
        self.shutdowns += 1
        return 0


class FakeWaitKernel:
    def __init__(self, timer=101, event=202, wait_result=1, set_timer_result=1):
        self.calls = []
        self.timer, self.event = timer, event
        self.wait_result, self.set_timer_result = wait_result, set_timer_result
        self.CreateWaitableTimerExW = Function(self.create_timer)
        self.CreateEventW = Function(self.create_event)
        self.SetWaitableTimer = Function(self.set_timer)
        self.WaitForMultipleObjects = Function(self.wait_multiple)
        self.SetEvent = Function(self.set_event)
        self.CloseHandle = Function(self.close_handle)

    def create_timer(self, security, name, flags, access):
        self.calls.append(("create_timer", security, name, flags, access))
        return self.timer

    def create_event(self, security, manual_reset, initial, name):
        self.calls.append(("create_event", manual_reset, initial))
        return self.event

    def set_timer(self, handle, due, period, callback, argument, resume):
        self.calls.append(("set_timer", handle, due._obj.value, period, callback, argument, resume))
        return self.set_timer_result

    def wait_multiple(self, count, handles, all_handles, timeout):
        self.calls.append(("wait", count, list(handles), all_handles, timeout))
        return self.wait_result

    def set_event(self, handle):
        self.calls.append(("set_event", handle))
        return 1

    def close_handle(self, handle):
        self.calls.append(("close", handle))
        return 1

def sample(begin, end=None, value=1200, cpu_interval=None):
    row = {"query_begin_tick": begin, "query_end_tick": begin + 1 if end is None else end,
           "gpu": {"status": "ok", "values": {"sm_clock_mhz": value}, "unknown": {}}}
    if cpu_interval is not None:
        row["cpu"] = {"status": "ok", "values": {"busy_percent": 25}, "unknown": {},
                      "interval_begin_tick": cpu_interval[0], "interval_end_tick": cpu_interval[1]}
    return row


class HighResolutionWaitTests(unittest.TestCase):
    def test_private_timer_flags_relative_due_and_interruptible_wait(self):
        kernel = FakeWaitKernel()
        waiter = t._WindowsHighResolutionWaiter(kernel)
        self.assertFalse(waiter.wait(.00125))
        self.assertIn(("create_timer", None, None, 2, 0x00100002), kernel.calls)
        self.assertIn(("create_event", True, False), kernel.calls)
        timer = next(item for item in kernel.calls if item[0] == "set_timer")
        self.assertEqual(timer, ("set_timer", 101, -12500, 0, None, None, False))
        wait = next(item for item in kernel.calls if item[0] == "wait")
        self.assertEqual(wait, ("wait", 2, [202, 101], False, 0xFFFFFFFF))
        self.assertTrue(waiter.metadata()["high_resolution_supported"])
        self.assertFalse(waiter.metadata()["global_timer_settings_modified"])
        self.assertFalse(waiter.metadata()["busy_spin"])
        self.assertFalse(waiter.metadata()["cadence_guaranteed"])
        waiter.close()

    def test_stop_event_has_priority_and_is_explicitly_signalled(self):
        kernel = FakeWaitKernel(wait_result=0)
        waiter = t._WindowsHighResolutionWaiter(kernel)
        waiter.interrupt()
        self.assertIn(("set_event", 202), kernel.calls)
        self.assertTrue(waiter.wait(.01))
        waiter.close()

    def test_close_is_idempotent_and_releases_both_owned_handles(self):
        kernel = FakeWaitKernel()
        waiter = t._WindowsHighResolutionWaiter(kernel)
        waiter.close(); waiter.close()
        self.assertEqual([item for item in kernel.calls if item[0] == "close"], [("close", 202), ("close", 101)])
        with self.assertRaises(RuntimeError):
            waiter.wait(.01)

    def test_unavailable_high_resolution_timer_fallback_is_explicit(self):
        stop = threading.Event()
        waiter = t._make_waiter(stop, windows=True, kernel=FakeWaitKernel(timer=None))
        self.assertEqual(waiter.metadata()["actual"], "threading.Event.wait")
        self.assertFalse(waiter.metadata()["high_resolution_supported"])
        self.assertIn("HIGH_RESOLUTION", waiter.metadata()["fallback_reason"])
        waiter.interrupt()
        self.assertTrue(waiter.wait(1))

    def test_missing_api_fallback_and_partial_creation_cleanup(self):
        waiter = t._make_waiter(threading.Event(), windows=True, kernel=object())
        self.assertIn("AttributeError", waiter.metadata()["fallback_reason"])
        kernel = FakeWaitKernel(event=None)
        waiter = t._make_waiter(threading.Event(), windows=True, kernel=kernel)
        self.assertFalse(waiter.metadata()["high_resolution_supported"])
        self.assertIn(("close", 101), kernel.calls)

    def test_nonwindows_fallback_never_claims_high_resolution(self):
        waiter = t._make_waiter(threading.Event(), windows=False)
        self.assertEqual(waiter.metadata()["actual"], "threading.Event.wait")
        self.assertFalse(waiter.metadata()["high_resolution_supported"])
        self.assertIn("not applicable", waiter.metadata()["fallback_reason"])

    def test_wait_failure_raises_instead_of_silent_midrun_fallback(self):
        for kernel in (FakeWaitKernel(set_timer_result=0), FakeWaitKernel(wait_result=0xFFFFFFFF)):
            waiter = t._WindowsHighResolutionWaiter(kernel)
            with self.assertRaises(OSError):
                waiter.wait(.01)
            self.assertEqual(waiter.metadata()["actual"], "CreateWaitableTimerExW(HIGH_RESOLUTION)+WaitForMultipleObjects(stop,timer)")
            waiter.close()

    def test_invalid_wait_delay_rejected_and_zero_avoids_os_call(self):
        kernel = FakeWaitKernel()
        waiter = t._WindowsHighResolutionWaiter(kernel)
        for delay in (-1, float("inf"), float("nan"), 1e20):
            with self.assertRaises(ValueError):
                waiter.wait(delay)
        before = len(kernel.calls)
        self.assertFalse(waiter.wait(0))
        self.assertEqual(len(kernel.calls), before)
        waiter.close()

    def test_injected_wait_policy_lifecycle_and_metadata(self):
        class WaitPolicy:
            def __init__(self):
                self.closed = self.interrupted = False
                self.calls = []
            def wait(self, seconds):
                self.calls.append(seconds)
                self.sampler._stop.set()
            def interrupt(self):
                self.interrupted = True
            def close(self):
                self.closed = True
            def metadata(self):
                return {"actual": "synthetic", "high_resolution_supported": False}
        policy = WaitPolicy()
        sampler = t.DeviceTelemetrySampler(clock=FakeClock(), providers={"gpu": Provider()}, wait_policy=policy)
        policy.sampler = sampler
        sampler.start()
        sampler._thread.join(1)
        sampler.stop()
        self.assertTrue(policy.interrupted and policy.closed)
        self.assertEqual(len(policy.calls), 1)
        self.assertIsNone(sampler.metadata()["thread_error"])
        self.assertEqual(sampler.metadata()["wait_policy"]["actual"], "synthetic")


class NVMLTests(unittest.TestCase):
    def test_direct_queries_units_identity_and_repeated_struct_reads(self):
        library = FakeNVML()
        provider = t.NVMLProvider(1, library=library)
        for _ in range(3):
            row = provider.read()
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["values"]["power_w"], 45)
            self.assertEqual(row["values"]["memory_clock_mhz"], 1002)
            self.assertEqual(row["values"]["memory_used_bytes"], 4000)
            self.assertEqual(row["values"]["gpu_util_percent"], 75)
        self.assertEqual(provider.metadata()["device_uuid"], "SYNTHETIC")
        self.assertFalse(provider.metadata()["subprocess_per_sample"])
        provider.close()
        provider.close()
        self.assertEqual(library.shutdowns, 1)

    def test_unsupported_field_is_unknown_not_zero(self):
        provider = t.NVMLProvider(library=FakeNVML(power_supported=False))
        row = provider.read()
        self.assertEqual(row["status"], "partial")
        self.assertNotIn("power_w", row["values"])
        self.assertIn("NVML status 3", row["unknown"]["power_w"])
        self.assertIn("temperature_c", row["values"])
        provider.close()

    def test_initialization_failure_has_every_field_unknown(self):
        provider = t.NVMLProvider(library=FakeNVML(init_status=9))
        self.assertFalse(provider.metadata()["available"])
        self.assertEqual(provider.read()["values"], {})
        self.assertEqual(set(provider.read()["unknown"]), set(t.NVMLProvider.FIELDS))
        self.assertIn("NVML status 9", provider.metadata()["error"])

    def test_identity_query_failure_is_explicit(self):
        library = FakeNVML()
        library.nvmlDeviceGetUUID = Function(lambda *arguments: 3)
        provider = t.NVMLProvider(library=library)
        self.assertIsNone(provider.metadata()["device_uuid"])
        self.assertIn("nvmlDeviceGetUUID", provider.metadata()["identity_unknown"])
        provider.close()

    def test_invalid_index_rejected(self):
        for index in (-1, True, "0"):
            with self.assertRaises(ValueError):
                t.NVMLProvider(index, library=FakeNVML())


class CPUCounterTests(unittest.TestCase):
    def test_interval_busy_and_initial_unknown(self):
        counters = iter([(100, 60), (200, 90)])
        provider = t.CPUBusyProvider(FakeClock(), lambda: next(counters))
        first = provider.read()
        self.assertIsNone(first["interval_begin_tick"])
        self.assertEqual(first["values"], {})
        second = provider.read()
        self.assertEqual(second["values"]["busy_percent"], 70)
        self.assertEqual(second["interval_begin_tick"], first["interval_end_tick"])
        self.assertGreater(second["interval_end_tick"], second["interval_begin_tick"])

    def test_reset_or_zero_interval_remains_unknown(self):
        counters = iter([(100, 60), (100, 60), (50, 20), (80, 100)])
        provider = t.CPUBusyProvider(FakeClock(), lambda: next(counters))
        provider.read()
        self.assertEqual(provider.read()["values"], {})
        self.assertEqual(provider.read()["values"], {})
        self.assertIn("invalid OS CPU counters", provider.read()["unknown"]["busy_percent"])

    def test_os_reader_failure_is_not_low_cpu_activity(self):
        def fail():
            raise OSError("counter unavailable")
        row = t.CPUBusyProvider(FakeClock(), fail).read()
        self.assertEqual(row["values"], {})
        self.assertIn("counter unavailable", row["unknown"]["busy_percent"])


class SamplerTests(unittest.TestCase):
    def test_query_bounds_deadlines_and_missed_cadence(self):
        clock = FakeClock(100, step=10)
        sampler = t.DeviceTelemetrySampler(clock=clock, interval_ms=10, providers={"gpu": Provider()})
        row = sampler._collect(90)
        self.assertEqual(row["query_begin_tick"], 100)
        self.assertEqual(row["query_end_tick"], 130)
        self.assertEqual(row["lateness_ticks"], 10)
        self.assertEqual(row["missed_periods"], 4)
        self.assertLessEqual(row["query_begin_tick"], row["gpu"]["provider_query_begin_tick"])
        self.assertLessEqual(row["gpu"]["provider_query_end_tick"], row["query_end_tick"])

    def test_provider_failure_is_retained(self):
        def fail():
            raise OSError("driver query failed")
        sampler = t.DeviceTelemetrySampler(clock=FakeClock(), providers={"gpu": Provider(fail)})
        row = sampler._collect(0)
        self.assertEqual(row["gpu"]["status"], "error")
        self.assertIn("driver query failed", row["gpu"]["unknown"]["provider"])

    def test_invalid_provider_contract_is_retained(self):
        sampler = t.DeviceTelemetrySampler(clock=FakeClock(), providers={"gpu": Provider(lambda: {"values": []})})
        self.assertEqual(sampler._collect(0)["gpu"]["status"], "error")

    def test_invalid_intervals_rejected(self):
        for interval in (0, -1, True, float("nan"), float("inf"), "10"):
            with self.assertRaises(ValueError):
                t.DeviceTelemetrySampler(clock=FakeClock(), interval_ms=interval, providers={})

    def test_thread_lifecycle_jsonl_freeze_and_provenance(self):
        calls = []
        sampler = None
        done = threading.Event()
        def read():
            calls.append(1)
            if len(calls) >= 3:
                sampler._stop.set()
                done.set()
            return {"status": "ok", "values": {"sm_clock_mhz": 1234}, "unknown": {}}
        provider = Provider(read)
        sampler = t.DeviceTelemetrySampler(clock=FakeClock(), providers={"gpu": provider})
        with self.assertRaises(RuntimeError):
            sampler.stop()
        with self.assertRaises(RuntimeError):
            sampler.write_jsonl("unused.jsonl")
        sampler.start()
        self.assertTrue(done.wait(2))
        sampler.stop()
        self.assertTrue(provider.closed)
        self.assertEqual(len(sampler.records), 3)
        self.assertEqual(sampler.stop(), sampler.records)
        with self.assertRaises(RuntimeError):
            sampler.start()
        self.assertIsNone(sampler.metadata()["thread_error"])
        self.assertEqual(len(sampler.metadata()["source_sha256"]), 64)
        root = Path(__file__).parent.resolve()
        with tempfile.TemporaryDirectory(dir=root, prefix="synthetic-telemetry-") as temporary:
            temporary = Path(temporary).resolve()
            self.assertTrue(temporary.is_relative_to(root))
            path = temporary / "raw.jsonl"
            binding = sampler.write_jsonl(path)
            self.assertNotIn(b"\r\n", path.read_bytes())
            self.assertEqual(binding["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(len([json.loads(line) for line in path.read_text().splitlines()]), 3)
            with self.assertRaises(FileExistsError):
                sampler.write_jsonl(path)


class AlignmentTests(unittest.TestCase):
    clock = FakeClock().metadata()

    def align(self, begin, end, records, **options):
        return t.align_intervals([{"action_index": 7, "begin_tick": begin, "end_tick": end}], records, self.clock, **options)[0]

    def test_sustained_block_descriptive_quality(self):
        row = self.align(0, 50, [sample(tick, value=1000 + tick) for tick in (5, 15, 25, 35, 45)])
        self.assertTrue(row["quality"]["model_ready"])
        self.assertEqual(row["action_index"], 7)
        self.assertEqual(row["observed_fields"]["gpu.sm_clock_mhz"]["median"], 1025)
        self.assertEqual(row["observed_fields"]["gpu.sm_clock_mhz"]["n"], 5)
        self.assertFalse(row["interpolation_performed"])

    def test_short_action_has_no_nearest_state_or_interpolation(self):
        row = self.align(7, 8, [sample(5), sample(15)])
        self.assertEqual(row["observed_fields"], {})
        self.assertFalse(row["quality"]["model_ready"])
        self.assertIn("action_shorter_than_three_sampling_periods", row["quality"]["reasons"])

    def test_boundary_sample_never_enters_summary(self):
        row = self.align(10, 40, [sample(8, 12, 9999), sample(15), sample(25), sample(38, 42, 8888)])
        self.assertEqual(row["boundary_ambiguous_sample_indices"], [0, 3])
        self.assertEqual(row["contained_sample_indices"], [1, 2])
        self.assertEqual(row["observed_fields"]["gpu.sm_clock_mhz"]["max"], 1200)

    def test_cpu_counter_interval_must_be_contained(self):
        row = self.align(10, 60, [sample(15, cpu_interval=(5, 16)), sample(25, cpu_interval=(16, 26)), sample(35, cpu_interval=(26, 36)), sample(45, cpu_interval=(36, 46)), sample(55, cpu_interval=(46, 56))])
        self.assertEqual(len(row["excluded_rolling_intervals"]), 1)
        self.assertEqual(row["observed_fields"]["cpu.busy_percent"]["n"], 4)

    def test_long_sampling_gaps_fail_quality(self):
        row = self.align(0, 100, [sample(5), sample(15), sample(55), sample(65), sample(95)])
        self.assertFalse(row["quality"]["model_ready"])
        self.assertIn("sampling_gap_exceeds_2_5_periods", row["quality"]["reasons"])

    def test_unobserved_edges_fail_quality(self):
        row = self.align(0, 100, [sample(40), sample(50), sample(60)])
        self.assertIn("unobserved_action_edge_exceeds_2_5_periods", row["quality"]["reasons"])

    def test_slow_observer_fails_quality(self):
        row = self.align(0, 60, [sample(1, 15), sample(20, 34), sample(40, 54)])
        self.assertIn("observer_query_exceeds_sampling_period", row["quality"]["reasons"])

    def test_unknown_values_not_imputed(self):
        rows = [sample(tick) for tick in (5, 15, 25, 35)]
        for row in rows:
            row["gpu"] = {"values": {}, "unknown": {"power_w": "unsupported"}, "status": "unavailable"}
        row = self.align(0, 40, rows)
        self.assertEqual(row["observed_fields"], {})
        self.assertEqual(row["unknown_counts"]["gpu.power_w"], 4)
        self.assertFalse(row["quality"]["model_ready"])

    def test_nonfinite_telemetry_rejected(self):
        for value in (float("inf"), float("nan"), True, "unknown"):
            with self.assertRaises(ValueError):
                self.align(0, 50, [sample(5, value=value)])

    def test_clock_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            self.align(0, 50, [], action_clock={"clock_id": "windows_qpc", "frequency_hz": 1000})
        with self.assertRaises(ValueError):
            self.align(0, 50, [], action_clock={"clock_id": self.clock["clock_id"], "frequency_hz": 2000})

    def test_invalid_action_and_order_rejected(self):
        for begin, end in ((0, 0), (10, 9), (True, 10), (-1, 10)):
            with self.assertRaises(ValueError):
                self.align(begin, end, [])
        for rows in ([sample(5, 4)], [sample(10), sample(5)], [sample(5, 15), sample(10, 20)]):
            with self.assertRaises(ValueError):
                self.align(0, 50, rows)


if __name__ == "__main__":
    unittest.main()
