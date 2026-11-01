"""Explicit affinity control of one thread in a caller-owned Windows child.

Only the READY-reported thread is changed. The caller supplies its existing
owned-process handle; ownership is a launch precondition, not inferred from a
PID. The newly opened thread handle is retained across the DUT to bind its
lifetime. There are no process/system affinity, priority, ideal-CPU, interrupt,
privilege or timer changes, and no performance interpretation is inferred.
"""
from __future__ import annotations

import ctypes
import os

from profiling.resource_calibration.counters import _handle_value
from profiling.state_calibration.telemetry import NativeClock
from .worker_activity import OwnedActor

THREAD_SET_INFORMATION = 0x0020
THREAD_QUERY_LIMITED_INFORMATION = 0x0800
STILL_ACTIVE = 259
SCHEMA = "owned-main-thread-affinity-1"


class AffinityControlError(RuntimeError):
    """Failure with a retained receipt, including any successful mutation."""

    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


class UnsupportedAffinityControl(AffinityControlError):
    pass


class _FileTime(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]

    def value(self):
        return self.low | (self.high << 32)


class _GroupAffinity(ctypes.Structure):
    _fields_ = [("mask", ctypes.c_size_t), ("group", ctypes.c_uint16),
                ("reserved", ctypes.c_uint16 * 3)]


def _bind(kernel, name, arguments, result):
    function = getattr(kernel, name)
    function.argtypes, function.restype = arguments, result
    return function


class OwnedMainThreadAffinity:
    """Hold exactly one owned-thread handle; close does not restore affinity.

    Call ``observe`` after the DUT, before stopping the owned child, to verify
    that the same live thread still has the requested group-0 affinity. Masks
    constrain eligible CPUs; neither residency nor continuous scheduling is
    established. Closing never closes the borrowed process handle.
    """

    def __init__(self, actor, main_thread_id, mask, *, clock=None, windows=None, kernel=None):
        if not isinstance(actor, OwnedActor):
            raise ValueError("caller-owned actor with borrowed process handle required")
        if type(main_thread_id) is not int or not 1 <= main_thread_id <= 0xFFFFFFFF:
            raise ValueError("explicit positive READY-reported native thread id required")
        if type(mask) is not int or not 1 <= mask <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("explicit nonzero unsigned 64-bit affinity mask required")
        self.actor, self.main_thread_id, self.mask = actor, main_thread_id, mask
        self.handle = None
        self.clock = clock or NativeClock()
        self.receipt = dict(schema=SCHEMA, actor_id=actor.actor_id, process_id=actor.process_id,
            main_thread_id=main_thread_id, requested_affinity_mask=mask, requested_group=0,
            clock=self.clock.metadata(), set_succeeded=False, thread_handle_closed=False,
            borrowed_process_handle_closed=False, affinity_restored_on_close=False,
            system_settings_changed=False, observed_cpu_residency_claimed=False,
            ownership_scope="caller-owned launch handle and READY thread id; no ownership inference from PID",
            observations=[])
        windows = os.name == "nt" if windows is None else windows
        if not windows or ctypes.sizeof(ctypes.c_void_p) != 8:
            raise UnsupportedAffinityControl("owned affinity control requires x64 Windows", self.receipt)
        self.kernel = kernel if kernel is not None else ctypes.WinDLL("kernel32", use_last_error=True)
        self._functions = {}
        self.receipt["pin_begin_tick"] = self.clock.now()
        try:
            identity = self._process_identity()
            self._process_creation = identity["creation_time_100ns"]
            self.receipt["owned_process_identity"] = identity
            process_mask, system_mask, groups = self._process_affinity()
            self.receipt.update(process_affinity_mask=process_mask,
                                system_affinity_mask=system_mask, process_groups=groups)
            self._require(groups == [0], "only an explicitly verified group-0 owned process is supported")
            self._require(mask & process_mask == mask, "requested mask is not a subset of owned process affinity")
            open_thread = self._api("OpenThread", [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32], ctypes.c_void_p)
            opened = open_thread(THREAD_SET_INFORMATION | THREAD_QUERY_LIMITED_INFORMATION,
                                 False, main_thread_id)
            self._require(bool(opened), self._failure("OpenThread"))
            self.handle = opened
            before = self._thread_identity()
            self._thread_creation = before["creation_time_100ns"]
            self.receipt["owned_thread_identity"] = before
            self.receipt["affinity_before_set"] = self._thread_affinity()
            self._require(self.receipt["affinity_before_set"]["group"] == 0,
                          "owned thread is outside processor group 0")
            setter = self._api("SetThreadAffinityMask", [ctypes.c_void_p, ctypes.c_size_t], ctypes.c_size_t)
            previous = setter(self.handle, mask)
            self._require(previous != 0, self._failure("SetThreadAffinityMask"))
            self.receipt.update(set_succeeded=True, previous_affinity_mask=int(previous))
            self.observe()
            self.receipt["pin_end_tick"] = self.clock.now()
        except BaseException as error:
            self.receipt["error"] = type(error).__name__ + ": " + str(error)
            try:
                self.close()
            except Exception as close_error:
                self.receipt["cleanup_error"] = str(close_error)
            if isinstance(error, AffinityControlError):
                raise
            raise AffinityControlError(str(error), self.receipt) from error

    def _api(self, name, arguments, result):
        if name not in self._functions:
            self._functions[name] = _bind(self.kernel, name, arguments, result)
        return self._functions[name]

    @staticmethod
    def _failure(operation):
        code = getattr(ctypes, "get_last_error", lambda: 0)()
        return f"{operation} failed (Windows error {code})"

    def _require(self, condition, message):
        if not condition:
            raise AffinityControlError(message, self.receipt)

    def _times(self, api_name, handle):
        creation, exit_time, kernel, user = (_FileTime() for _ in range(4))
        query = self._api(api_name, [ctypes.c_void_p] + [ctypes.POINTER(_FileTime)] * 4, ctypes.c_int)
        self._require(query(handle, ctypes.byref(creation), ctypes.byref(exit_time),
                            ctypes.byref(kernel), ctypes.byref(user)), self._failure(api_name))
        self._require(creation.value() > 0 and exit_time.value() == 0,
                      "owned process/thread has no live creation-time identity")
        return creation.value()

    def _active(self, api_name, handle):
        value = ctypes.c_uint32()
        query = self._api(api_name, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)], ctypes.c_int)
        self._require(query(handle, ctypes.byref(value)), self._failure(api_name))
        self._require(value.value == STILL_ACTIVE, "owned process/thread is no longer active")

    def _process_identity(self):
        handle = _handle_value(self.actor.process_handle)
        query = self._api("GetProcessId", [ctypes.c_void_p], ctypes.c_uint32)
        self._require(query(handle) == self.actor.process_id, "borrowed process handle PID differs from owned actor")
        self._active("GetExitCodeProcess", handle)
        return dict(status="verified", process_id=self.actor.process_id,
                    creation_time_100ns=self._times("GetProcessTimes", handle))

    def _thread_identity(self):
        owner = self._api("GetProcessIdOfThread", [ctypes.c_void_p], ctypes.c_uint32)
        self._require(owner(self.handle) == self.actor.process_id, "thread does not belong to owned actor PID")
        query = self._api("GetThreadId", [ctypes.c_void_p], ctypes.c_uint32)
        self._require(query(self.handle) == self.main_thread_id, "opened thread id differs from READY thread id")
        self._active("GetExitCodeThread", self.handle)
        return dict(status="verified", process_id=self.actor.process_id, thread_id=self.main_thread_id,
                    creation_time_100ns=self._times("GetThreadTimes", self.handle))

    def _process_affinity(self):
        handle = _handle_value(self.actor.process_handle)
        process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
        query = self._api("GetProcessAffinityMask", [ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)], ctypes.c_int)
        self._require(query(handle, ctypes.byref(process_mask), ctypes.byref(system_mask)),
                      self._failure("GetProcessAffinityMask"))
        self._require(process_mask.value > 0 and system_mask.value > 0 and
                      process_mask.value & system_mask.value == process_mask.value,
                      "owned process affinity inventory is unknown or inconsistent")
        count = ctypes.c_uint16(64)
        groups = (ctypes.c_uint16 * 64)()
        query_groups = self._api("GetProcessGroupAffinity", [ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16)], ctypes.c_int)
        self._require(query_groups(handle, ctypes.byref(count), groups), self._failure("GetProcessGroupAffinity"))
        self._require(1 <= count.value <= 64, "invalid owned process group inventory")
        return process_mask.value, system_mask.value, list(groups[:count.value])

    def _thread_affinity(self):
        affinity = _GroupAffinity()
        query = self._api("GetThreadGroupAffinity", [ctypes.c_void_p, ctypes.POINTER(_GroupAffinity)], ctypes.c_int)
        self._require(query(self.handle, ctypes.byref(affinity)), self._failure("GetThreadGroupAffinity"))
        return dict(group=int(affinity.group), mask=int(affinity.mask))

    def observe(self):
        """Verify current mask and the retained live process/thread identities."""
        self._require(self.handle is not None, "owned thread affinity handle is closed")
        begin = self.clock.now()
        try:
            process = self._process_identity()
            self._require(process["creation_time_100ns"] == self._process_creation,
                          "owned process lifetime identity changed")
            thread = self._thread_identity()
            self._require(thread["creation_time_100ns"] == self._thread_creation,
                          "owned thread lifetime identity changed")
            affinity = self._thread_affinity()
            self.receipt["last_observed_affinity"] = affinity
            self._require(affinity == dict(group=0, mask=self.mask),
                          "observed owned thread affinity differs from explicit requested mask")
            end = self.clock.now()
            self._require(end >= begin, "affinity observation clock moved backwards")
            row = dict(query_begin_tick=begin, query_end_tick=end,
                       process_identity=process, thread_identity=thread,
                       observed_group=affinity["group"], observed_affinity_mask=affinity["mask"],
                       status="verified")
            self.receipt["observations"].append(row)
            return row
        except BaseException as error:
            self.receipt["observation_error"] = type(error).__name__ + ": " + str(error)
            raise

    def close(self):
        if self.handle is not None:
            close = self._api("CloseHandle", [ctypes.c_void_p], ctypes.c_int)
            self._require(close(self.handle), self._failure("CloseHandle"))
            self.handle = None
            self.receipt["thread_handle_closed"] = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def pin_owned_main_thread(actor, main_thread_id, mask, *, clock=None, windows=None, kernel=None):
    """Pin exactly the caller-supplied READY thread; no default mask or CPU guess."""
    return OwnedMainThreadAffinity(actor, main_thread_id, mask,
                                    clock=clock, windows=windows, kernel=kernel)
