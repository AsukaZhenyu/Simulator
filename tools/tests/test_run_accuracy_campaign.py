# -*- coding: utf-8 -*-
"""The standalone runner must refuse to start, and never start something twice.

`tools/run_accuracy_campaign.py` is launched by hand on an otherwise idle machine
and then runs for hours without supervision, so its refusals are the only thing
standing between a contaminated machine state and a published number. These tests
pin each refusal, and pin the property that a second run never reuses a first
run's cohort: an inherited frozen price would look like a successful campaign.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import run_accuracy_campaign as subject
from profiling import host
from profiling import runtime_libraries


class Refusal(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.campaign = subject.Campaign(self.root, self.root / "campaign.log")
        self.artifact = self.root / "present.txt"
        self.artifact.write_text("x", encoding="utf-8")

    def tearDown(self):
        self.folder.cleanup()

    def refuse(self, reasons, needed=0.0, artifacts=None):
        free = (10 * 2**30, [])
        with mock.patch.object(subject, "LOCK", self.root / "held.lock"), \
             mock.patch.object(subject, "free_memory", return_value=free), \
             mock.patch.object(subject, "resident_measurements", return_value=reasons):
            return subject.preflight(self.campaign, "probe", needed,
                                     artifacts if artifacts is not None else [self.artifact])

    def test_a_held_lock_stops_the_run(self):
        (self.root / "held.lock").write_text(json.dumps(dict(pid=42)), encoding="utf-8")
        with self.assertRaises(SystemExit) as raised:
            self.refuse([])
        self.assertIn("lock", str(raised.exception))

    def test_a_resident_measurement_process_stops_the_run(self):
        # The exact failure this check exists for: a probe outlived its shell.
        with self.assertRaises(SystemExit) as raised:
            self.refuse([(4242, "real-moe-runtime-window-probe.exe")])
        self.assertIn("4242", str(raised.exception))

    def test_a_missing_artifact_stops_the_run(self):
        with self.assertRaises(SystemExit) as raised:
            self.refuse([], artifacts=[self.root / "absent.gguf"])
        self.assertIn("absent.gguf", str(raised.exception))

    def test_short_memory_stops_the_run(self):
        free = (1 * 2**30, [("editor.exe", 2 * 2**30)])
        with mock.patch.object(subject, "LOCK", self.root / "held.lock"), \
             mock.patch.object(subject, "free_memory", return_value=free), \
             mock.patch.object(subject, "resident_measurements", return_value=[]):
            with self.assertRaises(SystemExit) as raised:
                subject.preflight(self.campaign, "model", 6.0, [self.artifact])
        self.assertIn("memory", str(raised.exception))

    def test_a_clean_machine_passes(self):
        self.refuse([])
        self.assertTrue((self.root / "campaign.log").exists())

    def test_a_windows_build_on_a_shared_drive_is_refused_by_its_platform(self):
        # The published runtime tree lives on the Windows side of a shared
        # folder. A Linux host sees every file exist and cannot load one of
        # them; the refusal names the platform rather than starting a probe.
        windows = self.root / "runtime-windows-build"
        windows.mkdir()
        (windows / "ggml-base.dll").write_bytes(b"MZ\x90\x00")
        (windows / "BUILD_COMPLETE.json").write_text("{}", encoding="utf-8")
        free = (10 * 2**30, [])
        with mock.patch.multiple(host, is_windows=mock.Mock(return_value=False),
                                 is_macos=mock.Mock(return_value=False)), \
             mock.patch.object(subject, "LOCK", self.root / "held.lock"), \
             mock.patch.object(subject, "free_memory", return_value=free), \
             mock.patch.object(subject, "resident_measurements", return_value=[]):
            with self.assertRaises(SystemExit) as raised:
                subject.preflight(self.campaign, "model", 6.0,
                                  [windows / "BUILD_COMPLETE.json"])
        detail = str(raised.exception)
        self.assertIn("another platform", detail)
        self.assertIn(".dll", detail)

    def test_a_half_built_runtime_is_refused_before_a_cohort_is_prepared(self):
        # The strict-F32 runtime on a POSIX host is a full CMake build; one that
        # stopped part way leaves `libggml-base.so` in `build/bin` and nothing
        # else. Every existence check passes, and the e2a column then dies one
        # phase in, on a cohort that has to be prepared again from scratch.
        runtime = self.root / "runtime"
        runtime.mkdir()
        name = {"*.dll": "ggml-base.dll", "*.dylib": "libggml-base.dylib",
                "*.so*": "libggml-base.so"}[host.shared_library_pattern()]
        (runtime / name).write_bytes(b"")
        free = (10 * 2**30, [])
        with mock.patch.object(subject, "LOCK", self.root / "held.lock"), \
             mock.patch.object(subject, "free_memory", return_value=free), \
             mock.patch.object(subject, "resident_measurements", return_value=[]):
            with self.assertRaises(SystemExit) as raised:
                subject.preflight(self.campaign, "e2a", 2.0, [runtime],
                                  runtime_dirs=(runtime,))
        detail = str(raised.exception)
        self.assertIn("does not hold", detail)
        # The third library is the accelerator's, and which one that is differs
        # by device; naming `ggml-cuda` here would make this case a CUDA test
        # wearing a portable name.
        for absent in ("ggml-cpu", runtime_libraries.components()[-1]):
            self.assertIn(absent, detail)
        self.assertIn("runtime is incomplete",
                      (self.root / "campaign.log").read_text(encoding="utf-8"))

    def test_a_runtime_split_across_directories_is_not_called_incomplete(self):
        # What the e2a column actually passes: the runtime, the toolkit's bin and
        # CUPTI's lib separately. Only the union has to hold the components.
        runtime, toolkit = self.root / "runtime", self.root / "toolkit"
        runtime.mkdir()
        toolkit.mkdir()
        names = {"*.dll": "{}.dll", "*.dylib": "lib{}.dylib",
                 "*.so*": "lib{}.so"}[host.shared_library_pattern()]
        base, *elsewhere = runtime_libraries.components()
        (runtime / names.format(base)).write_bytes(b"")
        for component in elsewhere:
            (toolkit / names.format(component)).write_bytes(b"")
        free = (10 * 2**30, [])
        with mock.patch.object(subject, "LOCK", self.root / "held.lock"), \
             mock.patch.object(subject, "free_memory", return_value=free), \
             mock.patch.object(subject, "resident_measurements", return_value=[]):
            subject.preflight(self.campaign, "e2a", 2.0, [runtime, toolkit],
                              runtime_dirs=(runtime, toolkit))

    def test_a_dry_run_names_what_is_missing_instead_of_refusing(self):
        # The first command on a rented or freshly unpacked device: nothing is
        # built yet, and "missing artifact" is the answer it came for.
        free = (10 * 2**30, [])
        with mock.patch.object(subject, "LOCK", self.root / "held.lock"), \
             mock.patch.object(subject, "free_memory", return_value=free), \
             mock.patch.object(subject, "resident_measurements", return_value=[]):
            problems = subject.preflight(self.campaign, "model", 6.0,
                                         [self.root / "absent.gguf"], strict=False)
        self.assertEqual(len(problems), 1)
        self.assertIn("absent.gguf", problems[0])
        self.assertIn("artifact", (self.root / "campaign.log").read_text(encoding="utf-8"))

    def test_a_dry_run_reports_every_reason_not_just_the_first(self):
        free = (1 * 2**30, [])
        with mock.patch.object(subject, "LOCK", self.root / "held.lock"), \
             mock.patch.object(subject, "free_memory", return_value=free), \
             mock.patch.object(subject, "resident_measurements",
                               return_value=[(4242, "depth-probe")]):
            problems = subject.preflight(self.campaign, "dense", 2.0,
                                         [self.root / "absent-probe"], strict=False)
        self.assertEqual(len(problems), 3)          # resident, missing, memory

    def test_a_host_without_psutil_is_refused_by_name_not_by_traceback(self):
        # `psutil` is in the first-hour install line, but a host that skipped it
        # must be told which check cannot run and how to fix it -- the resident
        # check is a safety gate, so it is not silently skipped.
        with mock.patch.dict(sys.modules, {"psutil": None}):
            with self.assertRaises(subject.ResidentCheckUnavailable) as raised:
                subject.resident_measurements()
        self.assertIn("pip install psutil", str(raised.exception))
        free = (10 * 2**30, [])
        with mock.patch.object(subject, "LOCK", self.root / "held.lock"), \
             mock.patch.object(subject, "free_memory", return_value=free), \
             mock.patch.object(subject, "resident_measurements",
                               side_effect=subject.ResidentCheckUnavailable("install psutil")):
            with self.assertRaises(SystemExit) as raised:
                subject.preflight(self.campaign, "model", 6.0, [self.artifact])
        self.assertIn("psutil", str(raised.exception))

    def test_a_dry_run_keeps_going_when_the_resident_check_cannot_run(self):
        free = (10 * 2**30, [])
        with mock.patch.object(subject, "LOCK", self.root / "held.lock"), \
             mock.patch.object(subject, "free_memory", return_value=free), \
             mock.patch.object(subject, "resident_measurements",
                               side_effect=subject.ResidentCheckUnavailable("install psutil")):
            problems = subject.preflight(self.campaign, "model", 6.0, [self.artifact],
                                         strict=False)
        self.assertEqual(len(problems), 1)
        self.assertIn("psutil", problems[0])


class Step(unittest.TestCase):
    """Every phase is a subprocess with Path arguments; the log must accept them."""

    def test_a_step_with_path_arguments_is_logged_and_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            campaign = subject.Campaign(root, root / "campaign.log")
            record = campaign.step("noop", [sys.executable, "-c", "import sys; sys.exit(0)",
                                            root / "some" / "artifact.json"])
            self.assertEqual(record["returncode"], 0)
            self.assertIn("artifact.json", (root / "campaign.log").read_text(encoding="utf-8"))

    def test_a_failed_step_stops_the_campaign(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            campaign = subject.Campaign(root, root / "campaign.log")
            with self.assertRaises(SystemExit):
                campaign.step("boom", [sys.executable, "-c", "import sys; sys.exit(3)"])
            self.assertEqual(campaign.records[-1]["returncode"], 3)


class Authorization(unittest.TestCase):
    """Two hardware windows, two receipts; `collect` reads only the canonical one."""

    def test_the_calibration_receipt_is_kept_under_its_own_name(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            campaign = subject.Campaign(root, root / "campaign.log")
            receipt = root / "ROOT_AUTHORIZATION_RECEIPT.json"
            receipt.write_text(json.dumps(dict(jobs=["cpu-calibration-profile"])), encoding="utf-8")
            subject.retire_authorization(campaign, root, "CALIBRATION")
            self.assertFalse(receipt.exists())
            kept = json.loads((root / "ROOT_AUTHORIZATION_CALIBRATION.json").read_text(encoding="utf-8"))
            self.assertEqual(kept["jobs"], ["cpu-calibration-profile"])

    def test_nothing_happens_when_no_receipt_exists_yet(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            campaign = subject.Campaign(root, root / "campaign.log")
            subject.retire_authorization(campaign, root, "CALIBRATION")
            self.assertEqual(sorted(p.name for p in root.iterdir()), [])

    def test_a_second_window_of_the_same_role_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            campaign = subject.Campaign(root, root / "campaign.log")
            (root / "ROOT_AUTHORIZATION_CALIBRATION.json").write_text("{}", encoding="utf-8")
            (root / "ROOT_AUTHORIZATION_RECEIPT.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(SystemExit):
                subject.retire_authorization(campaign, root, "CALIBRATION")
            self.assertTrue((root / "ROOT_AUTHORIZATION_RECEIPT.json").exists())


class Reuse(unittest.TestCase):
    def test_an_existing_cohort_is_moved_aside_not_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            campaign = subject.Campaign(root, root / "campaign.log")
            cohort = root / "model" / "formal"
            cohort.mkdir(parents=True)
            (cohort / "PRICE_FREEZE.json").write_text("{}", encoding="utf-8")
            returned = subject.fresh_cohort(cohort, campaign)
            self.assertEqual(returned, cohort)
            self.assertFalse(cohort.exists())
            moved = list(cohort.parent.glob("formal-aborted-*"))
            self.assertEqual(len(moved), 1)
            self.assertTrue((moved[0] / "PRICE_FREEZE.json").exists())

    def test_a_new_cohort_is_returned_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            campaign = subject.Campaign(root, root / "campaign.log")
            cohort = root / "dense" / "formal"
            self.assertEqual(subject.fresh_cohort(cohort, campaign), cohort)
            self.assertFalse(cohort.exists())

    def test_an_existing_receipt_is_moved_aside_not_overwritten(self):
        # `record_tests` creates its file exclusively and lives outside the
        # cohort, so a second run must move the first one away.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            campaign = subject.Campaign(root, root / "campaign.log")
            receipt = root / "model" / "software-tests.json"
            receipt.parent.mkdir(parents=True)
            receipt.write_text("{}", encoding="utf-8")
            self.assertEqual(subject.fresh_file(receipt, campaign), receipt)
            self.assertFalse(receipt.exists())
            moved = list(receipt.parent.glob("software-tests-aborted-*.json"))
            self.assertEqual(len(moved), 1)

    def test_a_new_receipt_is_returned_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            campaign = subject.Campaign(root, root / "campaign.log")
            receipt = root / "model" / "software-tests.json"
            self.assertEqual(subject.fresh_file(receipt, campaign), receipt)
            self.assertFalse(receipt.exists())


class AHostThatCannotMeasure(unittest.TestCase):
    """A host with no accelerator ggml can compute on, and what a dry run owes it.

    The refusal has to be a written verdict in the log and a zero exit, not a
    traceback three layers into a column: a CPU-only box runs exactly one command
    of this campaign, and "this host cannot run a V-A column" is the whole answer
    it owes the operator. Nothing in this test needs such a host -- `host` is the
    seam. (A Mac is not this case any more: it has Metal, and its dense column
    runs.)
    """

    def run_dry(self, folder: str, **patches) -> int:
        root = Path(folder)
        with mock.patch.object(subject.host, "require_measurement_host",
                               side_effect=subject.host.UnsupportedHost(
                                   "no accelerator ggml can compute on was found")), \
             mock.patch.object(subject, "LOCK", root / "held.lock"), \
             mock.patch.object(subject, "free_memory", return_value=(10 * 2**30, [])), \
             mock.patch.object(subject, "resident_measurements", return_value=[]), \
             mock.patch.object(sys, "argv", ["run_accuracy_campaign", "--dry-run",
                                             "--root", str(root)]):
            return subject.main()

    def test_a_cpu_only_host_gets_a_verdict_and_a_zero_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(0, self.run_dry(folder))
            log = (Path(folder) / "campaign.log").read_text(encoding="utf-8")
        log = " ".join(log.split())
        self.assertIn("this host cannot run a V-A column", log)
        self.assertIn("no accelerator ggml can compute on was found", log)
        # And every column still says what it would have needed, so the operator
        # learns what the measurement requires rather than only that it cannot.
        for name in ("model", "dense", "e2a"):
            self.assertIn(f"the {name} column is not ready", log)
        self.assertIn("preflight finished; nothing was measured", log)


class ADryRunThatNamesWhatToBuild(unittest.TestCase):
    """The dry run's answer on a new host is the list of files to build.

    A heading with nothing under it ("missing artifacts for the dense column:")
    names no file, and the file names are the whole answer a first run exists to
    give -- so the detail lines are printed too, not only the first line.
    """

    def test_the_missing_artifact_paths_are_printed_not_only_the_heading(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            absent = root / "depth-probe-absent"
            with mock.patch.object(subject.host, "require_measurement_host",
                                   return_value=None), \
                 mock.patch.object(subject, "LOCK", root / "held.lock"), \
                 mock.patch.object(subject, "free_memory", return_value=(10 * 2**30, [])), \
                 mock.patch.object(subject, "resident_measurements", return_value=[]), \
                 mock.patch.object(subject, "requirements",
                                   return_value=(1.0, [absent])), \
                 mock.patch.object(sys, "argv", ["run_accuracy_campaign", "--dry-run",
                                                 "--root", str(root)]):
                self.assertEqual(0, subject.main())
            log = (root / "campaign.log").read_text(encoding="utf-8")
        self.assertIn("missing artifacts for the model column:", log)
        self.assertIn(str(absent), log)


class Requirements(unittest.TestCase):
    def test_every_column_declares_what_it_needs(self):
        args = mock.Mock(model=subject.DEFAULT_MODEL,
                         runtime_build=subject.DEFAULT_RUNTIME_BUILD,
                         depth_binary=subject.DEFAULT_DEPTH_BINARY,
                         e2a_binary=subject.DEFAULT_E2A_BINARY,
                         e2a_runtime=None,
                         cuda_dir=None)
        for name in ("model", "dense", "e2a"):
            needed, artifacts = subject.requirements(name, args)
            self.assertGreater(needed, 0, name)
            self.assertTrue(artifacts, name)
            self.assertTrue(any("python" not in str(a) for a in artifacts), name)

    def test_the_runtime_directories_are_the_ones_the_probes_load(self):
        args = mock.Mock(e2a_binary=subject.DEFAULT_E2A_BINARY,
                         e2a_runtime=None, cuda_dir=Path("D:/CUDA/v12.4"),
                         depth_binary=Path("D:/nowhere/depth-probe.exe"))
        self.assertEqual(subject.runtime_dirs("e2a", args),
                         (subject.E2A_RUNTIME_F32, Path("D:/CUDA/v12.4/bin"),
                          Path("D:/CUDA/v12.4/extras/CUPTI/lib64")))
        # No probe on disk means no CMakeCache to derive a ggml build from, so
        # the dense column names nothing and the preflight has nothing to hold
        # against it -- rather than inventing a directory.
        self.assertEqual(subject.runtime_dirs("dense", args), ())
        self.assertEqual(subject.runtime_dirs("model", args), ())

    def test_a_dry_run_does_not_default_into_a_real_cohorts_directory(self):
        self.assertNotEqual(subject.default_root(True), subject.default_root(False))
        self.assertTrue(str(subject.default_root(True)).endswith("-dryrun"))

    def test_a_named_toolkit_is_used_as_named(self):
        binary, cupti = subject.cuda_dirs(mock.Mock(cuda_dir=Path("D:/CUDA/v12.4")))
        self.assertEqual(binary, Path("D:/CUDA/v12.4/bin"))
        self.assertEqual(cupti, Path("D:/CUDA/v12.4/extras/CUPTI/lib64"))

    def test_the_newest_installed_toolkit_wins_when_none_is_named(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("v12.4", "v12.6", "v12.10"):
                (root / name / "bin").mkdir(parents=True)
            (root / "v99.0").mkdir()  # no bin/: not a toolkit, whatever its name
            # The search roots belong to the host, not to the driver: a device
            # that keeps its toolkit elsewhere answers through `host`.
            with mock.patch.object(subject.host, "cuda_toolkit_roots", return_value=[str(root)]):
                binary, _ = subject.cuda_dirs(mock.Mock(cuda_dir=None))
            self.assertEqual(binary, root / "v12.10" / "bin")

    def test_a_layout_that_splits_its_libraries_names_that_directory_too(self):
        # `bin` holds nvcc and the CUDA runtime on Linux while cuBLAS sits in
        # `lib64`; a probe linking ggml-cuda asks the loader for it before its
        # first action, so leaving the directory out fails every process of the
        # column with one loader error rather than one bad measurement.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "bin").mkdir()
            (root / "extras" / "CUPTI" / "lib64").mkdir(parents=True)
            (root / "extras" / "CUPTI" / "lib64" / "libcupti.so").write_bytes(b"")
            (root / "lib64").mkdir()
            (root / "lib64" / "libcublas.so.12").write_bytes(b"")
            with mock.patch.object(subject.host, "backend", return_value="cuda"), \
                 mock.patch.object(subject.host, "is_windows", return_value=False):
                directories = subject.probe_toolkit_dirs(mock.Mock(cuda_dir=root))
            self.assertEqual(directories, (root / "bin",
                                           root / "extras" / "CUPTI" / "lib64",
                                           root / "lib64"))

    def test_a_layout_that_keeps_libraries_in_bin_names_it_once(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "bin").mkdir()
            (root / "extras" / "CUPTI" / "lib64").mkdir(parents=True)
            (root / "extras" / "CUPTI" / "lib64" / "libcupti.so").write_bytes(b"")
            with mock.patch.object(subject.host, "backend", return_value="cuda"):
                directories = subject.probe_toolkit_dirs(mock.Mock(cuda_dir=root))
            self.assertEqual(directories, (root / "bin", root / "extras" / "CUPTI" / "lib64"))

    def test_the_environment_makes_every_source_root_importable(self):
        env = subject.environment()
        parts = env["PYTHONPATH"].split(subject.os.pathsep)
        for name in ("src", "mapping", "modeling", "profiling"):
            self.assertIn(str(subject.ROOT / name), parts)


class Headline(unittest.TestCase):
    """The summary quotes the campaign's own files; it never recomputes an error."""

    def test_the_model_cell_is_read_from_the_condition_mean_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "CONDITION_MEAN_EVALUATION.json").write_text(json.dumps(dict(
                declared_conditions=12, complete_endpoint_conditions=12,
                mean_APE={"total": 0.029012}, worst_APE={"total": 0.080218})), encoding="utf-8")
            text = "\n".join(subject.headline(root, {"model": root}))
            self.assertIn("2.9012%", text)
            self.assertIn("8.0218%", text)
            self.assertIn("12 of 12", text)

    def test_incomplete_conditions_are_reported_not_hidden(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "CONDITION_MEAN_EVALUATION.json").write_text(json.dumps(dict(
                declared_conditions=12, complete_endpoint_conditions=10,
                mean_APE={"total": 0.03}, worst_APE={"total": 0.09})), encoding="utf-8")
            text = "\n".join(subject.headline(root, {"model": root}))
            self.assertIn("10 of 12", text)
            self.assertIn("excluded, not averaged", text)

    def test_the_dense_cell_reports_the_worst_case_and_the_pass_count(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "summary.json").write_text(json.dumps(dict(
                declared_cases=18,
                providers={"context_static": dict(accuracy_passes=18, cases=18),
                           "history_conditional": dict(accuracy_passes=18, cases=18)},
                cases=[dict(providers={"context_static": dict(max_process_mean_ape=0.074),
                                       "history_conditional": dict(max_process_mean_ape=0.067)}),
                       dict(providers={"context_static": dict(max_process_mean_ape=0.051),
                                       "history_conditional": dict(max_process_mean_ape=0.033)})])),
                encoding="utf-8")
            text = "\n".join(subject.headline(root, {"dense": root}))
            self.assertIn("history_conditional: process-mean max APE 6.7000%", text)
            self.assertIn("18 of 18", text)

    def test_the_strategy_cell_reports_selection_and_qualification(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "summary.json").write_text(json.dumps(dict(counts=dict(
                declared_groups=6, regret_point_passes=6, programs=24,
                qualified_programs=20, qualified_groups=3))), encoding="utf-8")
            text = "\n".join(subject.headline(root, {"e2a": root}))
            self.assertIn("6 of 6 workloads pick the real winner", text)
            self.assertIn("20 of 24 candidate programs", text)
            self.assertIn("3 of 6 groups", text)
            self.assertIn("observer", text)

    def test_the_decision_margin_sweep_is_quoted_from_its_own_file(self):
        # The margin study is a sibling of the cohort, not a member of it.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cohort = root / "e2a-v2"
            cohort.mkdir()
            (root / "decision-margin").mkdir()
            (root / "decision-margin" / "decision-margin.json").write_text(json.dumps(dict(sweep=dict(
                per_error={"0.0100": [dict(draws=2000, flips=0, regret_max=0.0)],
                           "0.2000": [dict(draws=2000, flips=264, regret_max=0.3367)]}))),
                encoding="utf-8")
            text = "\n".join(subject.headline(root, {"e2a": cohort}))
            self.assertIn("1%: 0.000% of 2000", text)
            self.assertIn("20%: 13.200% of 2000", text)
            self.assertIn("33.6700%", text)

    def test_a_missing_column_contributes_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(subject.headline(Path(folder), {}), [])


if __name__ == "__main__":
    unittest.main()
