"""A campaign must not price a filesystem instead of a disk.

The measurement tool is honest about the mechanism it achieved, but it cannot
tell whether ``--directory`` is a real local disk. On a laptop that is the
user's own choice; on a rented UMA box the working directory is often an
overlay, a tmpfs or a network mount, and numbers taken there belong to no
device. These tests pin every refusal and every warning the preflight emits,
using synthetic reports so they do not need an overlay filesystem to exist.
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from profiling.storage import preflight

GOOD_REPORT = {
    "schema_version": "ssd_measurement_preflight_v1",
    "directory": "/mnt/nvme/ssd-measure",
    "platform": "Linux",
    "machine": "x86_64",
    "python": "3.13.0",
    "mount": {"source": "/dev/nvme0n1p2", "fstype": "ext4",
              "mount_point": "/mnt/nvme", "options": "rw",
              "filesystem_class": "local", "probe": "findmnt"},
    "free_bytes": 64 << 30,
    "bypass": {"mechanism": "unbuffered", "backend": "posix-o_direct",
               "error": None},
    "source": {"commit": "bb0ce9c", "dirty": False, "error": None},
}


def report(**overrides) -> dict:
    document = json.loads(json.dumps(GOOD_REPORT))
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(document.get(key), dict):
            document[key].update(value)
        else:
            document[key] = value
    return document


class TheFilesystemTypeIsClassified(unittest.TestCase):
    def test_virtual_filesystems(self):
        for name in ("overlay", "overlayfs", "tmpfs", "ramfs", "nfs",
                     "nfs4", "cifs", "smb3", "9p", "virtiofs", "fuse.sshfs",
                     "fuse.gcsfuse"):
            with self.subTest(name=name):
                self.assertEqual(preflight.classify_filesystem(name), "virtual")

    def test_local_filesystems(self):
        for name in ("ext4", "xfs", "btrfs", "apfs", "ntfs", "exfat", "zfs"):
            with self.subTest(name=name):
                self.assertEqual(preflight.classify_filesystem(name), "local")

    def test_missing_name_is_unknown_not_local(self):
        # Unknown must not masquerade as local: it downgrades to a warning so
        # somebody has to say which device the numbers describe.
        self.assertEqual(preflight.classify_filesystem(None), "unknown")
        self.assertEqual(preflight.classify_filesystem("  "), "unknown")


class AGoodDirectoryIsAccepted(unittest.TestCase):
    def test_ok_report(self):
        result = preflight.verdict(report())
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["refusals"], [])
        self.assertEqual(result["mechanism"], "unbuffered")
        self.assertEqual(result["commit"], "bb0ce9c")


class VirtualFilesystemsAreRefused(unittest.TestCase):
    def test_overlay_refuses(self):
        result = preflight.verdict(report(mount={
            "fstype": "overlay", "source": "overlay", "filesystem_class": "virtual"}))
        self.assertEqual(result["status"], "refuse")
        self.assertIn("overlay", result["refusals"][0])
        self.assertIn("not a local disk", result["refusals"][0])

    def test_a_network_mount_refuses(self):
        result = preflight.verdict(report(mount={
            "fstype": "nfs4", "source": "10.0.0.5:/data",
            "filesystem_class": "virtual"}))
        self.assertEqual(result["status"], "refuse")

    def test_allowing_non_local_downgrades_to_warning(self):
        # For a case where the virtual filesystem *is* the declared target, the
        # operator can say so; the verdict then warns instead of refusing, but
        # it still records the choice.
        result = preflight.verdict(
            report(mount={"fstype": "overlay", "filesystem_class": "virtual"}),
            allow_non_local=True)
        self.assertEqual(result["status"], "warn")
        self.assertEqual(len(result["warnings"]), 1)

    def test_unknown_filesystem_warns_without_refusing(self):
        result = preflight.verdict(report(mount={
            "fstype": None, "source": None, "filesystem_class": "unknown"}))
        self.assertEqual(result["status"], "warn")
        self.assertTrue(any("could not identify" in text
                            for text in result["warnings"]))

    def test_removable_media_warns(self):
        result = preflight.verdict(report(mount={
            "fstype": "exfat", "filesystem_class": "removable"}))
        self.assertEqual(result["status"], "warn")
        self.assertTrue(any("removable" in text for text in result["warnings"]))


class TheBypassMustBeReal(unittest.TestCase):
    def test_buffered_is_refused(self):
        result = preflight.verdict(report(bypass={
            "mechanism": "buffered", "backend": "buffered", "error": None}))
        self.assertEqual(result["status"], "refuse")
        self.assertTrue(any("page cache" in text for text in result["refusals"]))

    def test_no_bypass_at_all_is_refused(self):
        result = preflight.verdict(report(bypass={
            "mechanism": None, "backend": None, "error": "CacheBypassError: nope"}))
        self.assertEqual(result["status"], "refuse")
        self.assertIn("no cache bypass available", result["refusals"][0])
        self.assertIn("CacheBypassError", result["refusals"][0])

    def test_fadvise_fallback_warns_because_it_is_best_effort(self):
        result = preflight.verdict(report(bypass={
            "mechanism": "fadvise_dontneed", "backend": "posix-fadvise_dontneed",
            "error": None}))
        self.assertEqual(result["status"], "warn")
        self.assertTrue(any("best-effort" in text for text in result["warnings"]))

    def test_darwin_warns_that_nocache_is_per_handle(self):
        result = preflight.verdict(report(bypass={
            "mechanism": "f_nocache", "backend": "darwin-f_nocache",
            "error": None}))
        self.assertEqual(result["status"], "warn")
        self.assertTrue(any("per-handle" in text for text in result["warnings"]))


class SpaceAndProvenanceAreChecked(unittest.TestCase):
    def test_below_the_floor_refuses(self):
        result = preflight.verdict(report(free_bytes=1024))
        self.assertEqual(result["status"], "refuse")
        self.assertIn("below the", result["refusals"][0])

    def test_exactly_at_the_floor_is_fine(self):
        result = preflight.verdict(report(free_bytes=preflight.DEFAULT_MIN_FREE_BYTES))
        self.assertEqual(result["status"], "ok")

    def test_unknown_free_space_only_warns(self):
        result = preflight.verdict(report(free_bytes=None))
        self.assertEqual(result["status"], "warn")

    def test_commit_mismatch_refuses(self):
        result = preflight.verdict(report(), expected_commit="deadbee")
        self.assertEqual(result["status"], "refuse")
        self.assertIn("does not match", result["refusals"][0])

    def test_matching_commit_passes(self):
        result = preflight.verdict(report(), expected_commit="bb0ce9c")
        self.assertEqual(result["status"], "ok")

    def test_a_short_sha_matches_the_full_one(self):
        # The operator compares the short SHA printed by `git rev-parse --short
        # HEAD` on two machines; refusing that would make the check unusable.
        full = "bb0ce9c79b2f003e18eb81f1e1ff04d5094274fc"
        self.assertTrue(preflight.commits_match(full, "bb0ce9c"))
        self.assertTrue(preflight.commits_match("bb0ce9c", full))
        self.assertTrue(preflight.commits_match(full.upper(), "bb0ce9c"))
        self.assertFalse(preflight.commits_match(full, "e6c62d5"))

    def test_a_one_character_prefix_is_not_a_match(self):
        self.assertFalse(preflight.commits_match(
            "bb0ce9c79b2f003e18eb81f1e1ff04d5094274fc", "b"))

    def test_no_expected_commit_means_no_comparison(self):
        result = preflight.verdict(report(source={"commit": None}))
        self.assertEqual(result["status"], "ok")

    def test_dirty_tree_warns(self):
        result = preflight.verdict(report(source={"dirty": True}))
        self.assertEqual(result["status"], "warn")
        self.assertTrue(any("dirty" in text for text in result["warnings"]))


class TheProbeRunsHere(unittest.TestCase):
    def test_probe_identifies_this_directory_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            before = set(directory.iterdir())
            document = preflight.probe(directory)
            after = set(directory.iterdir())

        self.assertEqual(document["schema_version"], "ssd_measurement_preflight_v1")
        self.assertEqual(Path(document["directory"]), Path(name))
        self.assertIn("mount", document)
        self.assertIn("filesystem_class", document["mount"])
        self.assertIsNotNone(document["source"]["commit"])
        # The trial file must not survive, or the next campaign's free space
        # check sees a leftover.
        self.assertEqual(before, after)
        self.assertTrue(document["bypass"]["mechanism"] in
                        ("unbuffered", "fadvise_dontneed", "f_nocache"),
                        document["bypass"])


class TheCliReportsThroughItsExitCode(unittest.TestCase):
    def _run(self, *arguments):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = preflight.main(list(arguments))
        return code, buffer.getvalue()

    def test_a_good_directory_exits_zero(self):
        with tempfile.TemporaryDirectory() as name:
            code, output = self._run("--directory", name, "--min-free-bytes", "0")
        self.assertEqual(code, 0, output)
        self.assertIn("VERDICT", output)

    def test_a_refusal_exits_two_and_says_why(self):
        with tempfile.TemporaryDirectory() as name:
            code, output = self._run("--directory", name, "--min-free-bytes",
                                     str(1 << 62))
        self.assertEqual(code, 2)
        self.assertIn("REFUSE", output)
        self.assertIn("below the", output)

    def test_json_mode_carries_the_report_and_verdict(self):
        with tempfile.TemporaryDirectory() as name:
            code, output = self._run("--directory", name, "--min-free-bytes", "0",
                                     "--json")
        self.assertEqual(code, 0)
        document = json.loads(output)
        # Not necessarily "ok": running the suite leaves this tree dirty, and a
        # dirty tree is exactly the warning the verdict is supposed to raise.
        self.assertIn(document["verdict"]["status"], ("ok", "warn"))
        self.assertEqual(document["report"]["schema_version"],
                         "ssd_measurement_preflight_v1")

    def test_a_missing_directory_is_not_a_verdict(self):
        with contextlib.redirect_stderr(io.StringIO()):
            code = preflight.main(["--directory", str(
                Path(tempfile.gettempdir()) / "definitely-not-here-preflight")])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
