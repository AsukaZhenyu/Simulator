"""Injected Win32 fixtures only; no child launches, native calls or CPU work."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profiling.state_response import thread_affinity as a
from profiling.state_response.worker_activity import OwnedActor


class Clock:
    def __init__(self): self.tick = 0
    def now(self):
        self.tick += 1
        return self.tick
    def metadata(self): return dict(clock_id="fixture", frequency_hz=10_000_000)


class Function:
    def __init__(self, kernel, name): self.kernel, self.name = kernel, name
    def __call__(self, *args):
        self.kernel.calls.append((self.name, args))
        if self.name == self.kernel.fail: return 0
        return getattr(self.kernel, "call_" + self.name)(*args)


class Kernel:
    def __init__(self):
        self.calls, self.fail = [], None
        self.pid, self.tid, self.owner_pid = 99, 7, 99
        self.process_birth, self.thread_birth = 100, 200
        self.process_active = self.thread_active = a.STILL_ACTIVE
        self.process_mask, self.system_mask, self.groups = 5, 255, [0]
        self.thread_mask, self.thread_group = 5, 0
        self.closed = []
    def __getattr__(self, name):
        if not name.startswith("call_") and hasattr(type(self), "call_" + name):
            function = Function(self, name)
            setattr(self, name, function)
            return function
        raise AttributeError(name)
    def call_GetProcessId(self, handle): return self.pid
    def call_GetProcessIdOfThread(self, handle): return self.owner_pid
    def call_GetThreadId(self, handle): return self.tid
    def call_GetExitCodeProcess(self, handle, value):
        value._obj.value = self.process_active
        return 1
    def call_GetExitCodeThread(self, handle, value):
        value._obj.value = self.thread_active
        return 1
    def call_GetProcessTimes(self, handle, birth, end, kernel, user):
        birth._obj.low = self.process_birth
        return 1
    def call_GetThreadTimes(self, handle, birth, end, kernel, user):
        birth._obj.low = self.thread_birth
        return 1
    def call_GetProcessAffinityMask(self, handle, process, system):
        process._obj.value, system._obj.value = self.process_mask, self.system_mask
        return 1
    def call_GetProcessGroupAffinity(self, handle, count, groups):
        count._obj.value = len(self.groups)
        for index, group in enumerate(self.groups): groups[index] = group
        return 1
    def call_OpenThread(self, access, inherit, tid): return 456
    def call_GetThreadGroupAffinity(self, handle, value):
        value._obj.mask, value._obj.group = self.thread_mask, self.thread_group
        return 1
    def call_SetThreadAffinityMask(self, handle, mask):
        previous, self.thread_mask = self.thread_mask, mask
        return previous
    def call_CloseHandle(self, handle):
        self.closed.append(handle)
        return 1


class OwnedThreadAffinityTests(unittest.TestCase):
    def setUp(self):
        self.actor, self.kernel = OwnedActor("main", 99, 123), Kernel()
    def pin(self, **kwargs):
        return a.pin_owned_main_thread(self.actor, 7, 4, clock=Clock(),
                                       windows=True, kernel=self.kernel, **kwargs)
    def operations(self): return [name for name, _ in self.kernel.calls]

    def test_explicit_pin_observe_and_close_only_owned_thread_handle(self):
        control = self.pin()
        self.assertTrue(control.receipt["set_succeeded"])
        self.assertEqual(control.receipt["previous_affinity_mask"], 5)
        self.assertEqual(control.receipt["owned_thread_identity"]["creation_time_100ns"], 200)
        self.assertEqual(control.receipt["observations"][0]["observed_affinity_mask"], 4)
        self.assertEqual(control.observe()["observed_affinity_mask"], 4)
        open_args = next(args for name, args in self.kernel.calls if name == "OpenThread")
        self.assertEqual(open_args, (a.THREAD_SET_INFORMATION | a.THREAD_QUERY_LIMITED_INFORMATION, False, 7))
        self.assertEqual(self.kernel.closed, [])
        control.close(); control.close()
        self.assertEqual(self.kernel.closed, [456])
        self.assertEqual(self.kernel.thread_mask, 4)  # No silent restoration.
        self.assertFalse(control.receipt["borrowed_process_handle_closed"])
        self.assertFalse(control.receipt["system_settings_changed"])
        self.assertFalse(control.receipt["observed_cpu_residency_claimed"])
        with self.assertRaisesRegex(a.AffinityControlError, "closed"): control.observe()

    def test_non_windows_is_explicitly_unsupported_without_kernel_calls(self):
        with self.assertRaises(a.UnsupportedAffinityControl):
            a.pin_owned_main_thread(self.actor, 7, 4, windows=False, clock=Clock(), kernel=self.kernel)
        self.assertEqual(self.kernel.calls, [])

    def test_no_implicit_mask_or_thread_id_and_strict_unsigned_ranges(self):
        for tid, mask in ((None, 1), (True, 1), (0, 1), (2**32, 1),
                          (7, None), (7, True), (7, 0), (7, -1), (7, 2**64)):
            with self.subTest(tid=tid, mask=mask), self.assertRaises(ValueError):
                a.pin_owned_main_thread(self.actor, tid, mask, windows=True, clock=Clock(), kernel=self.kernel)
        self.assertEqual(self.kernel.calls, [])

    def test_foreign_process_handle_rejected_before_opening_thread(self):
        self.kernel.pid = 100
        with self.assertRaisesRegex(a.AffinityControlError, "borrowed process handle PID"):
            self.pin()
        self.assertNotIn("OpenThread", self.operations())
        self.assertEqual(self.kernel.closed, [])

    def test_foreign_thread_rejected_before_querying_its_times_or_setting_affinity(self):
        self.kernel.owner_pid = 100
        with self.assertRaisesRegex(a.AffinityControlError, "does not belong"):
            self.pin()
        self.assertNotIn("GetThreadTimes", self.operations())
        self.assertNotIn("SetThreadAffinityMask", self.operations())
        self.assertEqual(self.kernel.closed, [456])

    def test_wrong_thread_id_is_rejected_and_handle_closed(self):
        self.kernel.tid = 8
        with self.assertRaisesRegex(a.AffinityControlError, "READY thread id"): self.pin()
        self.assertNotIn("SetThreadAffinityMask", self.operations())
        self.assertEqual(self.kernel.closed, [456])

    def test_mask_must_be_subset_of_observed_process_mask(self):
        with self.assertRaisesRegex(a.AffinityControlError, "subset"):
            a.pin_owned_main_thread(self.actor, 7, 2, windows=True, clock=Clock(), kernel=self.kernel)
        self.assertNotIn("OpenThread", self.operations())

    def test_multi_group_or_nonzero_group_is_not_guessed(self):
        for groups in ([1], [0, 1], []):
            self.kernel.groups = groups
            with self.subTest(groups=groups), self.assertRaises(a.AffinityControlError): self.pin()
        self.kernel.groups, self.kernel.thread_group = [0], 1
        with self.assertRaisesRegex(a.AffinityControlError, "outside processor group 0"): self.pin()
        self.assertNotIn("SetThreadAffinityMask", self.operations())

    def test_inactive_process_or_thread_is_rejected(self):
        self.kernel.process_active = 0
        with self.assertRaisesRegex(a.AffinityControlError, "no longer active"): self.pin()
        self.assertNotIn("OpenThread", self.operations())
        self.kernel.process_active, self.kernel.thread_active = a.STILL_ACTIVE, 0
        with self.assertRaisesRegex(a.AffinityControlError, "no longer active"): self.pin()
        self.assertEqual(self.kernel.closed, [456])

    def test_process_and_thread_creation_time_are_rechecked_after_dut(self):
        control = self.pin()
        self.kernel.thread_birth = 201
        with self.assertRaisesRegex(a.AffinityControlError, "thread lifetime identity changed"): control.observe()
        self.kernel.thread_birth, self.kernel.process_birth = 200, 101
        with self.assertRaisesRegex(a.AffinityControlError, "process lifetime identity changed"): control.observe()
        control.close()

    def test_actual_affinity_change_is_failure_with_observed_receipt(self):
        control = self.pin()
        self.kernel.thread_mask = 1
        with self.assertRaisesRegex(a.AffinityControlError, "differs from explicit") as caught:
            control.observe()
        self.assertEqual(caught.exception.receipt["last_observed_affinity"], {"group": 0, "mask": 1})
        control.close()

    def test_query_failures_never_produce_a_verified_or_guessed_mask(self):
        for operation in ("GetProcessTimes", "GetProcessAffinityMask", "GetProcessGroupAffinity",
                          "OpenThread", "GetThreadTimes", "GetThreadGroupAffinity"):
            self.kernel = Kernel()
            self.kernel.fail = operation
            with self.subTest(operation=operation), self.assertRaises(a.AffinityControlError) as caught:
                self.pin()
            self.assertFalse(caught.exception.receipt["set_succeeded"])
            self.assertNotIn("SetThreadAffinityMask", self.operations())
            self.assertEqual(self.kernel.closed, [] if operation in (
                "GetProcessTimes", "GetProcessAffinityMask", "GetProcessGroupAffinity", "OpenThread") else [456])

    def test_failed_set_retains_failure_and_releases_thread_handle(self):
        self.kernel.fail = "SetThreadAffinityMask"
        with self.assertRaises(a.AffinityControlError) as caught: self.pin()
        self.assertFalse(caught.exception.receipt["set_succeeded"])
        self.assertEqual(self.kernel.closed, [456])

    def test_post_set_query_failure_retains_successful_mutation(self):
        original = self.kernel.call_SetThreadAffinityMask
        def setter(handle, mask):
            value = original(handle, mask)
            self.kernel.fail = "GetThreadGroupAffinity"
            return value
        self.kernel.call_SetThreadAffinityMask = setter
        with self.assertRaises(a.AffinityControlError) as caught: self.pin()
        self.assertTrue(caught.exception.receipt["set_succeeded"])
        self.assertEqual(self.kernel.thread_mask, 4)
        self.assertEqual(self.kernel.closed, [456])

    def test_context_manager_closes_even_when_caller_fails(self):
        with self.assertRaisesRegex(ValueError, "DUT"):
            with self.pin(): raise ValueError("DUT failed")
        self.assertEqual(self.kernel.closed, [456])

    def test_close_failure_is_explicit_and_owned_handle_can_be_retried(self):
        control = self.pin()
        self.kernel.fail = "CloseHandle"
        with self.assertRaises(a.AffinityControlError): control.close()
        self.assertIsNotNone(control.handle)
        self.kernel.fail = None
        control.close()
        self.assertEqual(self.kernel.closed, [456])


if __name__ == "__main__": unittest.main()
