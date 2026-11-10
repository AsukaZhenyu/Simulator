"""What the instrumentation asks of a host, and what it refuses to pretend.

The three V-A columns were written on one 32-CPU Windows machine, so three of
their constants were properties of that machine rather than of the protocol: the
process affinity mask, the shared-library suffix and the virtualenv layout. These
tests pin both halves of the replacement -- the published machine's answer does
not change, and a host that cannot honour the protocol is refused with the
reason named instead of failing later inside a build.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import re
import unittest
from pathlib import Path
from unittest import mock

from profiling import host


class MaskTests(unittest.TestCase):
    def test_the_mask_names_every_cpu_this_process_may_use(self):
        self.assertEqual(host.process_affinity_mask(),
                         sum(1 << cpu for cpu in host.usable_cpus()))

    def test_the_published_machine_gets_the_value_it_always_had(self):
        # 32 CPUs, all usable: this is the only reason the constant ever read
        # `4294967295`, and on that machine the derived value is byte-identical.
        with mock.patch.object(host, "usable_cpus", return_value=tuple(range(32))):
            self.assertEqual(host.process_affinity_mask(), 4294967295)

    def test_a_twenty_core_device_gets_its_own_mask(self):
        with mock.patch.object(host, "usable_cpus", return_value=tuple(range(20))):
            self.assertEqual(host.process_affinity_mask(), 1048575)

    def test_a_container_cpuset_names_its_own_cores_not_the_low_bits(self):
        # `docker --cpuset-cpus=24-47` on a 64-CPU machine. "The low 24 bits"
        # would name cores this process cannot be pinned to, `sched_setaffinity`
        # would answer EINVAL, and the probe would report a failed pin on a host
        # that pins fine.
        with mock.patch.object(host, "usable_cpus", return_value=tuple(range(24, 48))):
            self.assertEqual(host.process_affinity_mask(),
                             ((1 << 48) - 1) ^ ((1 << 24) - 1))

    def test_a_host_wider_than_one_dword_ptr_is_refused_not_truncated(self):
        wide = tuple(range(host.MAX_MASK_BITS + 1))
        with mock.patch.object(host, "usable_cpus", return_value=wide):
            with self.assertRaises(host.UnsupportedHost) as raised:
                host.process_affinity_mask()
        self.assertIn("64", str(raised.exception))

    @unittest.skipUnless(host.MAIN_THREAD_CPU in host.usable_cpus(),
                         "this host offers no CPU %d to pin" % host.MAIN_THREAD_CPU)
    def test_the_main_thread_pin_is_the_published_constant(self):
        # The protocol wrote `4`, which is bit 2 -- not "CPU 4". Written as an
        # index here so a host without that core can say so, but the value a
        # receipt carries has to come back as `4` on every host that has CPU 2.
        self.assertEqual(host.main_thread_mask(), 4)

    def test_the_main_thread_pin_is_four_on_any_host_with_the_core(self):
        with mock.patch.object(host, "usable_cpus", return_value=tuple(range(32))):
            self.assertEqual(host.main_thread_mask(), 4)

    def test_the_main_thread_pin_requires_the_core_it_names(self):
        with mock.patch.object(host, "usable_cpus", return_value=(0, 1)):
            with self.assertRaises(host.UnsupportedHost) as raised:
                host.main_thread_mask()
        self.assertIn("CPU %d" % host.MAIN_THREAD_CPU, str(raised.exception))
        with mock.patch.object(host, "usable_cpus", return_value=(0, 1, 2, 3)):
            self.assertEqual(host.main_thread_mask(), 1 << host.MAIN_THREAD_CPU)

    def test_a_pinned_container_without_core_two_is_refused(self):
        # The count can be large and still not contain the pinned core: a
        # 24-core cpuset starting at 24 passes every "enough cores" check.
        with mock.patch.object(host, "usable_cpus", return_value=tuple(range(24, 48))):
            with self.assertRaises(host.UnsupportedHost) as raised:
                host.main_thread_mask()
        self.assertIn("CPUs 24-47", str(raised.exception))

    def test_the_probe_and_the_campaign_ask_the_same_affinity_question(self):
        # The campaign sends a mask and the probe refuses any mask that is not
        # the set it read itself; if the two read different sources -- one the
        # machine's CPU count, one the process's allowed set -- a restricted host
        # gets a mask the probe cannot apply.
        header = Path(host.__file__).resolve().parents[1] / "profiling/native/portable/platform.hpp"
        text = header.read_text(encoding="utf-8")
        body = text[text.index("inline BOOL GetProcessAffinityMask"):
                    text.index("inline BOOL SetProcessAffinityMask")]
        self.assertIn("sched_getaffinity", body)
        self.assertNotIn("portable_online_cpus", body)


class NamingTests(unittest.TestCase):
    def platform(self, name: str):
        return mock.patch.multiple(
            host, is_windows=mock.Mock(return_value=name == "win32"),
            is_macos=mock.Mock(return_value=name == "darwin"))

    def test_the_loader_variable_matches_the_library_suffix(self):
        for name, pattern, variable in (("win32", "*.dll", "PATH"),
                                        ("darwin", "*.dylib", "DYLD_LIBRARY_PATH"),
                                        ("linux", "*.so*", "LD_LIBRARY_PATH")):
            with self.platform(name):
                self.assertEqual(host.shared_library_pattern(), pattern, name)
                self.assertEqual(host.shared_library_variable(), variable, name)

    def test_the_suffix_is_applied_to_the_glob_not_the_other_way_round(self):
        # `*.so*` has to match `libggml.so` and `libggml.so.1`; `*.dll` must not.
        with self.platform("linux"):
            self.assertTrue(Path("libggml-base.so").match(host.shared_library_pattern()))
            self.assertTrue(Path("libggml-base.so.1").match(host.shared_library_pattern()))
            self.assertFalse(Path("ggml-base.dll").match(host.shared_library_pattern()))

    def test_an_executable_is_spelled_for_this_host(self):
        with self.platform("win32"):
            self.assertEqual(host.executable_name("depth-probe"), "depth-probe.exe")
        with self.platform("darwin"):
            self.assertEqual(host.executable_name("depth-probe"), "depth-probe")

    def test_every_measurement_executable_carries_the_suffix_psutil_reports(self):
        for name in ("win32", "linux"):
            with self.platform(name):
                found = host.measurement_executables()
            self.assertTrue(found)
            for executable in found:
                self.assertEqual(executable.endswith(".exe"), name == "win32", executable)
            self.assertIn(("depth-probe.exe" if name == "win32" else "depth-probe"), found)

    def test_the_clock_a_receipt_names_is_the_one_the_probe_stamped(self):
        # The probe writes `PORTABLE_CLOCK_ID` from platform.hpp; a reader that
        # compared against the published machine's `windows_qpc` would refuse
        # every Linux row. Both spellings are read out of that header here, so
        # the two cannot drift apart silently.
        header = Path(host.__file__).resolve().parents[1] / "profiling/native/portable/platform.hpp"
        text = header.read_text(encoding="utf-8")
        names = re.findall(r'#define PORTABLE_CLOCK_ID "([^"]+)"', text)
        with self.platform("win32"):
            self.assertEqual(host.measurement_clock_id(), "windows_qpc")
        self.assertEqual(sorted(names), sorted({"windows_qpc", "posix_clock_monotonic"}))
        self.assertIn(host.measurement_clock_id(), names)


class MemoryTests(unittest.TestCase):
    """The campaign preflight's one question, answered in each host's own terms."""

    def test_this_host_answers_with_a_positive_byte_count(self):
        self.assertIsInstance(host.available_memory_bytes(), int)
        self.assertGreater(host.available_memory_bytes(), 0)

    def test_linux_reads_the_kernels_own_estimate_not_free_pages(self):
        # MemFree would understate it by the reclaimable page cache, and the
        # preflight would then refuse runs the machine could comfortably do.
        pages = mock.mock_open(read_data="MemTotal: 1000 kB\nMemFree: 100 kB\n"
                                          "MemAvailable:  4096 kB\n")
        with mock.patch.object(host, "is_windows", return_value=False), \
             mock.patch.object(host, "is_linux", return_value=True), \
             mock.patch("builtins.open", pages):
            self.assertEqual(host.available_memory_bytes(), 4096 * 1024)

    def test_a_kernel_without_memavailable_is_a_refusal_not_a_guess(self):
        pages = mock.mock_open(read_data="MemTotal: 1000 kB\nMemFree: 100 kB\n")
        with mock.patch.object(host, "is_windows", return_value=False), \
             mock.patch.object(host, "is_linux", return_value=True), \
             mock.patch("builtins.open", pages):
            with self.assertRaises(OSError):
                host.available_memory_bytes()

    def test_the_three_platform_predicates_are_one_choice(self):
        self.assertEqual(sum((host.is_windows(), host.is_macos(), host.is_linux())), 1)


class ForeignArtifactTests(unittest.TestCase):
    """An artifact that exists and still cannot run here.

    The published build directory is a Windows tree; a Linux host that reaches it
    through a shared folder passes every "does the file exist" check and then
    fails inside a probe at its first dlopen. The preflight asks this seam instead.
    """

    def platform(self, name: str):
        return mock.patch.multiple(
            host, is_windows=mock.Mock(return_value=name == "win32"),
            is_macos=mock.Mock(return_value=name == "darwin"))

    def test_a_windows_image_is_named_on_a_posix_host(self):
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "depth-probe"
            image.write_bytes(b"MZ\x90\x00rest of a PE")
            with self.platform("linux"):
                reason = host.foreign_artifact_reason(image)
        self.assertIn("Windows executable", reason)

    def test_a_native_binary_is_not_questioned(self):
        with tempfile.TemporaryDirectory() as folder:
            native = Path(folder) / "depth-probe"
            native.write_bytes(b"\x7fELF\x02\x01\x01\x00")
            with self.platform("linux"):
                self.assertIsNone(host.foreign_artifact_reason(native))
            with self.platform("win32"):
                self.assertIsNone(host.foreign_artifact_reason(native))

    def test_a_build_of_another_platform_is_named_by_its_libraries(self):
        with tempfile.TemporaryDirectory() as folder:
            build = Path(folder) / "runtime-windows-build"
            build.mkdir()
            (build / "ggml-base.dll").write_bytes(b"MZ\x90\x00")
            (build / "BUILD_COMPLETE.json").write_text("{}", encoding="utf-8")
            with self.platform("linux"):
                reason = host.foreign_artifact_reason(build)
        self.assertIn(".dll", reason)
        self.assertIn("*.so*", reason)

    def test_a_build_with_this_hosts_libraries_is_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            build = Path(folder) / "runtime"
            build.mkdir()
            (build / "libggml-base.so.0").write_bytes(b"\x7fELF")
            with self.platform("linux"):
                self.assertIsNone(host.foreign_artifact_reason(build))

    def test_an_empty_directory_is_missing_not_foreign(self):
        # "Missing" and "built elsewhere" are different answers with different
        # fixes, and the preflight says which one applies.
        with tempfile.TemporaryDirectory() as folder:
            empty = Path(folder) / "runtime"
            empty.mkdir()
            with self.platform("linux"):
                self.assertIsNone(host.foreign_artifact_reason(empty))
            self.assertIsNone(host.foreign_artifact_reason(Path(folder) / "absent"))


class ToolkitTests(unittest.TestCase):
    def test_cuda_is_never_available_on_macos(self):
        with mock.patch.object(host, "is_windows", return_value=False), \
             mock.patch.object(host, "is_macos", return_value=True):
            self.assertFalse(host.cuda_available())
            self.assertEqual(host.cuda_toolkit_roots(), [])

    def test_a_toolkit_root_is_a_directory_with_a_bin(self):
        with tempfile.TemporaryDirectory(prefix="cuda-") as folder:
            self.assertFalse(host.cuda_available(folder))
            (Path(folder) / "bin").mkdir()
            self.assertTrue(host.cuda_available(folder))

    def test_a_versioned_install_is_found_under_its_parent(self):
        with tempfile.TemporaryDirectory(prefix="cuda-") as folder:
            (Path(folder) / "v12.6" / "bin").mkdir(parents=True)
            with mock.patch.object(host, "is_macos", return_value=False), \
                 mock.patch.object(host, "cuda_toolkit_roots", return_value=[folder]):
                self.assertTrue(host.cuda_available())

    def test_cupti_is_found_in_each_of_the_two_toolkit_layouts(self):
        # x86_64 keeps CUPTI under extras/CUPTI/lib64; an aarch64 install
        # (Jetson Orin, and DGX Spark's Grace part) also ships it under
        # targets/aarch64-linux/lib, which is the layout no Windows box has.
        for layout in (("extras", "CUPTI", "lib64"),
                       ("targets", "aarch64-linux", "lib")):
            with tempfile.TemporaryDirectory(prefix="cuda-") as folder:
                directory = Path(folder).joinpath(*layout)
                directory.mkdir(parents=True)
                (directory / "libcupti.so").write_bytes(b"\x7fELF")
                self.assertEqual(host.cupti_library_dir(folder), directory, layout)


class DeviceDetectionTests(unittest.TestCase):
    """A toolkit says what can be built; the driver says what can compute.

    The two come apart on a host this repository itself builds on -- a WSL
    distribution or a CI image with the CUDA toolkit installed for its link line
    and no GPU passed through. That host compiles every probe here and can run
    none, and answering `cuda` from the toolkit alone made a campaign describe
    it as missing artifacts rather than as having no device.
    """

    def backend_on(self, devices, toolkit=True):
        with mock.patch.object(host, "is_macos", return_value=False), \
             mock.patch.object(host, "cuda_available", return_value=toolkit), \
             mock.patch.object(host, "nvidia_smi_devices", return_value=devices):
            return host.backend()

    def test_a_driver_that_lists_a_gpu_is_a_cuda_host(self):
        self.assertEqual(self.backend_on(["GPU 0: NVIDIA GeForce RTX 5090 (UUID: x)"]), "cuda")

    def test_a_driver_that_answers_with_no_gpu_is_not_one(self):
        self.assertIsNone(self.backend_on([]))

    def test_a_host_that_cannot_be_asked_is_not_refused_on_a_guess(self):
        # `None` from the driver is "unknown", not "none": a working GPU box
        # whose `nvidia-smi` is missing or erroring must not read as CPU-only.
        self.assertEqual(self.backend_on(None), "cuda")

    def test_no_toolkit_means_no_backend_before_the_driver_is_asked(self):
        self.assertIsNone(self.backend_on(["GPU 0: unreachable"], toolkit=False))

    def test_the_reason_separates_a_missing_toolkit_from_a_missing_device(self):
        with mock.patch.object(host, "is_macos", return_value=False), \
             mock.patch.object(host, "cuda_available", return_value=False):
            reason = host.backend_reason()
        self.assertIn("nothing to build a probe against", reason)
        with mock.patch.object(host, "is_macos", return_value=False), \
             mock.patch.object(host, "cuda_available", return_value=True), \
             mock.patch.object(host, "nvidia_smi_devices", return_value=[]):
            reason = host.backend_reason()
        self.assertIn("no GPU", reason)
        self.assertIn("cannot run one", reason)

    def test_a_device_less_host_is_refused_by_capability_not_by_path(self):
        with mock.patch.object(host, "is_macos", return_value=False), \
             mock.patch.object(host, "cuda_available", return_value=True), \
             mock.patch.object(host, "nvidia_smi_devices", return_value=[]):
            with self.assertRaises(host.UnsupportedHost) as raised:
                host.require_measurement_host()
        message = str(raised.exception)
        self.assertIn("no GPU", message)
        self.assertNotIn("Pass --cuda-dir", message)


class DriverQueryTests(unittest.TestCase):
    def test_only_gpu_lines_are_devices(self):
        done = subprocess.CompletedProcess([], 0, "GPU 0: A \n\nGPU 1: B", "")
        with mock.patch.object(host, "nvidia_smi_path", return_value="/usr/bin/nvidia-smi"), \
             mock.patch.object(host.subprocess, "run", return_value=done):
            self.assertEqual(host.nvidia_smi_devices(), ["GPU 0: A", "GPU 1: B"])

    def test_an_absent_failing_or_erroring_driver_is_unknown(self):
        with mock.patch.object(host, "nvidia_smi_path", return_value=None):
            self.assertIsNone(host.nvidia_smi_devices())
        with mock.patch.object(host, "nvidia_smi_path", return_value="/x/nvidia-smi"), \
             mock.patch.object(host.subprocess, "run",
                               return_value=subprocess.CompletedProcess([], 9, "", "mismatch")):
            self.assertIsNone(host.nvidia_smi_devices())
        with mock.patch.object(host, "nvidia_smi_path", return_value="/x/nvidia-smi"), \
             mock.patch.object(host.subprocess, "run", side_effect=OSError("gone")):
            self.assertIsNone(host.nvidia_smi_devices())


class RefusalTests(unittest.TestCase):
    def test_a_mac_is_a_measurement_host_because_metal_is_a_backend(self):
        # The refusal that used to stand here said no macOS release has CUDA and
        # therefore no probe builds. That was true of a port that linked CUDA; the
        # probes now link the host's own backend, and Metal is one.
        with mock.patch.object(host, "is_macos", return_value=True), \
             mock.patch.object(host, "arch", return_value="aarch64"):
            self.assertEqual(host.backend(), "metal")
            host.require_measurement_host()               # must not raise

    def test_a_host_without_a_toolkit_is_refused_before_anything_starts(self):
        with mock.patch.object(host, "is_macos", return_value=False), \
             mock.patch.object(host, "cuda_available", return_value=False):
            with self.assertRaises(host.UnsupportedHost) as raised:
                host.require_measurement_host()
        message = str(raised.exception)
        self.assertIn("--cuda-dir", message)
        # And it names what is missing as a capability, not as a broken checkout.
        self.assertIn("GPU placement", message)

    def test_a_named_toolkit_is_accepted(self):
        with tempfile.TemporaryDirectory(prefix="cuda-") as folder:
            (Path(folder) / "bin").mkdir()
            with mock.patch.object(host, "is_macos", return_value=False), \
                 mock.patch.object(host, "nvidia_smi_devices",
                                   return_value=["GPU 0: a device this test invents"]):
                host.require_measurement_host(folder)      # must not raise


class UnifiedMemoryTests(unittest.TestCase):
    """Which machines have one memory pool, decided from the host's shape."""

    def test_a_mac_accelerator_is_the_socs_own_gpu(self):
        self.assertTrue(host.unified_memory("metal", "aarch64"))
        self.assertTrue(host.unified_memory("metal", "x86_64"))   # an Intel Mac

    def test_an_aarch64_cuda_device_is_not_decided_by_its_architecture(self):
        # Jetson Orin and GB10 in a DGX Spark put the GPU on the package, but a
        # Grace-Blackwell host can carry a discrete GB300 over PCIe. Nothing here
        # has observed either, so the answer is unknown -- never "shared", which
        # would price every copy at zero.
        self.assertIsNone(host.unified_memory("cuda", "aarch64"))

    def test_a_discrete_card_is_on_the_other_side_of_pcie(self):
        self.assertFalse(host.unified_memory("cuda", "x86_64"))

    def test_no_accelerator_has_nothing_to_share(self):
        # Answering "shared" here would make every copy free on a CPU-only host.
        self.assertFalse(host.unified_memory(None, "x86_64"))
        self.assertFalse(host.unified_memory(None, "aarch64"))

    def test_the_hosts_answer_follows_its_backend_and_architecture(self):
        with mock.patch.dict("os.environ", {}, clear=False), \
             mock.patch.object(host, "backend", return_value="cuda"), \
             mock.patch.object(host, "arch", return_value="aarch64"):
            os.environ.pop("IR_DEVICE_MEMORY", None)
            self.assertEqual(host.device_memory(), "unknown")
        with mock.patch.object(host, "backend", return_value="cuda"), \
             mock.patch.object(host, "arch", return_value="x86_64"):
            self.assertEqual(host.device_memory(), "separate")
        with mock.patch.object(host, "backend", return_value="metal"), \
             mock.patch.object(host, "arch", return_value="aarch64"):
            self.assertEqual(host.device_memory(), "shared")

    def test_an_owner_may_declare_what_the_architecture_cannot_decide(self):
        # A GB10's pool is real, but it is a fact about that machine and not about
        # `aarch64`, so it enters by declaration -- and only where the architecture
        # left the question open.
        with mock.patch.object(host, "backend", return_value="cuda"), \
             mock.patch.object(host, "arch", return_value="aarch64"), \
             mock.patch.dict(os.environ, {"IR_DEVICE_MEMORY": "shared"}):
            self.assertEqual(host.device_memory(), "shared")
        with mock.patch.object(host, "backend", return_value="cuda"), \
             mock.patch.object(host, "arch", return_value="x86_64"), \
             mock.patch.dict(os.environ, {"IR_DEVICE_MEMORY": "shared"}):
            # A discrete card is decided by the host's shape; a stray variable must
            # not be able to price its copies at zero.
            self.assertEqual(host.device_memory(), "separate")
        with mock.patch.object(host, "backend", return_value="cuda"), \
             mock.patch.object(host, "arch", return_value="aarch64"), \
             mock.patch.dict(os.environ, {"IR_DEVICE_MEMORY": "maybe"}):
            self.assertEqual(host.device_memory(), "unknown")


if __name__ == "__main__":
    unittest.main()
