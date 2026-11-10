"""The one command a new device runs before anything else.

Its job is to turn "this box is not the box the table came from" into named
answers, so the case that matters most is the one no single test machine can be:
a Mac with Metal and one shared memory pool, and a CUDA box whose toolkit is
installed but whose CUPTI is somewhere else.
"""
from __future__ import annotations

import unittest

from tools import device_check as subject


def facts(**overrides):
    base = {
        "platform": "linux", "arch": "x86_64", "backend": "cuda",
        "device_memory": "separate", "online_cpus": 32, "python": "3.12",
        "device_name": "CUDA0",
        "toolkit": {"present": True, "root": "/usr/local/cuda-12.4",
                    "bin": "/usr/local/cuda-12.4/bin", "cupti_dir": "/cupti",
                    "cupti_present": True},
        "llama_tree": {"declared": "LLAMA_B8705_ROOT", "path": "/llama", "present": True},
        "probes": {
            "dense": {"binary": "/b/depth-probe", "binary_present": True},
            "e2a": {"binary": "/b/steady-costs-probe", "binary_present": True,
                    "runtime": "/r", "runtime_present": True,
                    "runtime_missing_libraries": []},
            "model": {"binary": "/b/real-moe-probe", "binary_present": True},
        },
        "runtime_components": ["ggml-base", "ggml-cpu", "ggml-cuda"],
    }
    base.update(overrides)
    return base


def verdict_of(rows, name):
    for what, verdict, why in rows:
        if what == name:
            return verdict, why
    raise AssertionError(name + " is not in " + str(rows))


class VerdictTests(unittest.TestCase):
    def test_a_complete_cuda_box_can_run_all_three_columns(self):
        rows = subject.verdicts(facts())
        for name in ("V-A dense column", "V-A e2a column", "V-A model column"):
            self.assertEqual(verdict_of(rows, name)[0], "yes", name)

    def test_a_missing_cupti_is_named_rather_than_left_to_the_linker(self):
        kit = dict(facts()["toolkit"], cupti_present=False)
        rows = subject.verdicts(facts(toolkit=kit))
        self.assertEqual(verdict_of(rows, "V-A dense column")[0], "not yet")
        self.assertIn("CUPTI", verdict_of(rows, "V-A dense column")[1])

    def test_a_half_built_f32_runtime_names_the_libraries_it_lacks(self):
        broken = dict(facts()["probes"]["e2a"],
                      runtime_missing_libraries=["ggml-cuda"])
        probes = dict(facts()["probes"], e2a=broken)
        verdict, why = verdict_of(subject.verdicts(facts(probes=probes)), "V-A e2a column")
        self.assertEqual(verdict, "not yet")
        self.assertIn("ggml-cuda", why)

    def test_a_mac_builds_the_dense_and_e2a_probes_and_says_why_not_the_model_one(self):
        mac = facts(platform="darwin", arch="aarch64", backend="metal",
                    device_memory="shared", device_name="Metal0",
                    toolkit={"present": False, "reason": "no CUDA on macOS"},
                    llama_tree={"declared": None, "path": None, "present": False},
                    runtime_components=["ggml-base", "ggml-cpu", "ggml-metal"])
        rows = subject.verdicts(mac)
        self.assertEqual(verdict_of(rows, "software tests")[0], "yes")
        self.assertEqual(verdict_of(rows, "simulator + mapping")[0], "yes")
        # Both probes are built for Metal and neither asks for a toolkit: the
        # dense column's clock-only modes never enter CUPTI, and the e2a column's
        # events are refused by the request check rather than linked.
        for name in ("V-A dense column", "V-A e2a column"):
            self.assertEqual(verdict_of(rows, name)[0], "yes", name)
        verdict, why = verdict_of(rows, "V-A model column")
        self.assertEqual(verdict, "no")
        # The reason is the missing observer, not a missing toolkit -- naming a
        # toolkit would send someone looking for one that would not help.
        self.assertIn("CUDA backend", why)

    def test_a_mac_without_a_built_probe_says_so_rather_than_claiming_yes(self):
        mac = facts(platform="darwin", arch="aarch64", backend="metal",
                    device_memory="shared", device_name="Metal0",
                    toolkit={"present": False, "reason": "no CUDA on macOS"},
                    llama_tree={"declared": None, "path": None, "present": False},
                    probes={column: {"binary": "/b/" + column, "binary_present": False,
                                     "runtime": None, "runtime_present": False}
                            for column in ("dense", "e2a", "model")})
        rows = subject.verdicts(mac)
        verdict, why = verdict_of(rows, "V-A dense column")
        self.assertEqual(verdict, "not yet")
        self.assertIn("PROBE_BACKEND=metal", why)
        # e2a needs its strict-F32 runtime before any probe has something to load, and
        # that runtime is not something a Mac can build: the reason names the kernel that
        # would have to change, not a missing file.
        verdict, why = verdict_of(rows, "V-A e2a column")
        self.assertEqual(verdict, "not yet")
        self.assertIn("strict-F32 runtime", why)
        self.assertIn("kernel_mul_mm_f32_f32", why)

    def test_a_mac_reports_a_half_built_metal_runtime_by_the_missing_library(self):
        mac = facts(platform="darwin", arch="aarch64", backend="metal",
                    device_memory="shared", device_name="Metal0",
                    toolkit={"present": False, "reason": "no CUDA on macOS"},
                    llama_tree={"declared": None, "path": None, "present": False},
                    probes={"dense": {"binary": "/b/depth-probe", "binary_present": True},
                            "e2a": {"binary": "/b/steady-costs-probe",
                                    "binary_present": True, "runtime": "/r",
                                    "runtime_present": True,
                                    "runtime_missing_libraries": ["ggml-metal"]},
                            "model": {"binary": "/b/real-moe-probe",
                                      "binary_present": False}})
        verdict, why = verdict_of(subject.verdicts(mac), "V-A e2a column")
        self.assertEqual(verdict, "not yet")
        self.assertIn("ggml-metal", why)

    def test_a_shared_memory_box_is_told_its_transfers_are_different(self):
        spark = facts(arch="aarch64", device_memory="shared")
        verdict, why = verdict_of(subject.verdicts(spark), "transfer prices")
        self.assertEqual(verdict, "different")
        self.assertIn("SHARE", why)

    def test_an_undecided_pool_asks_the_owner_rather_than_assuming_one(self):
        # An aarch64 CUDA host is either a SoC pool or a card on PCIe, and the
        # report must not print a verdict for a topology nobody observed.
        verdict, why = verdict_of(subject.verdicts(
            facts(arch="aarch64", device_memory="unknown")), "transfer prices")
        self.assertEqual(verdict, "not decided")
        self.assertIn("IR_DEVICE_MEMORY", why)

    def test_a_cpu_only_host_still_runs_the_software_side(self):
        rows = subject.verdicts(facts(backend=None, device_name=None,
                                      runtime_components=["ggml-base", "ggml-cpu"]))
        self.assertEqual(verdict_of(rows, "software tests")[0], "yes")
        self.assertEqual(verdict_of(rows, "V-A columns")[0], "no")

    def test_a_toolkit_with_no_device_says_so_rather_than_nothing_installed(self):
        # The host this repository builds on but cannot measure on: a CUDA
        # toolkit for the link line, and a driver that reports no GPU. It is not
        # a CPU-only box and not a missing toolkit, and the verdict says which.
        reason = ("a CUDA toolkit is installed but the driver reports no GPU: "
                  "the toolkit can build a probe, this host cannot run one")
        verdict, why = verdict_of(subject.verdicts(facts(
            backend=None, device_name=None, backend_reason=reason)), "V-A columns")
        self.assertEqual(verdict, "no")
        self.assertIn("no GPU", why)


class RenderTests(unittest.TestCase):
    def test_the_report_names_the_backend_the_memory_mode_and_the_libraries(self):
        text = subject.render(facts())
        self.assertIn("backend     cuda", text)
        self.assertIn("memory      separate", text)
        self.assertIn("ggml-cuda", text)

    def test_a_host_with_no_accelerator_says_so_instead_of_printing_None(self):
        text = subject.render(facts(backend=None, device_name=None))
        self.assertIn("none (CPU only)", text)
        self.assertNotIn("None", text.split("capability")[0].split("device")[0])

    def test_a_toolkit_with_no_device_gets_a_second_sentence(self):
        text = subject.render(facts(
            backend=None, device_name=None,
            backend_reason="a CUDA toolkit is installed but the driver reports no GPU"))
        self.assertIn("driver reports no GPU", text)


if __name__ == "__main__":
    unittest.main()
