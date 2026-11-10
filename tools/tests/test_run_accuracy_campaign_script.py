"""The shell wrapper that a POSIX host starts the campaign with.

Two things can go wrong before the campaign itself runs, and both look like
something else in the log:

* the wrapper picks a bare `python3` that is not the venv, and every column
  refuses because that interpreter has no numpy -- a reader chases the probes;
* the log never says which interpreter was used, so the same command on two
  machines cannot be compared.

`PYTHON` is the override, and the interpreter is printed. These tests run the
script with a stub interpreter, so nothing is measured and no package is needed.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "run-accuracy-campaign.sh"


def stub(directory: Path, body: str, name: str = "python-stub") -> Path:
    """An executable that stands in for the interpreter."""
    path = directory / name
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


@unittest.skipIf(sys.platform == "win32", "the wrapper under test is the POSIX one")
class CampaignWrapperTests(unittest.TestCase):
    def run_wrapper(self, interpreter: Path, *args: str) -> subprocess.CompletedProcess[str]:
        environment = dict(os.environ, PYTHON=str(interpreter))
        return subprocess.run(
            ["sh", str(SCRIPT), *args],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=environment, cwd=str(REPO), timeout=120,
        )

    def test_the_named_interpreter_runs_the_campaign_and_is_printed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            directory = Path(raw_dir)
            log = directory / "argv.log"
            interpreter = stub(directory, f'echo "$@" >> "{log}"\nexit 0\n')
            completed = self.run_wrapper(interpreter, "--dry-run")
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(f"interpreter: {interpreter}", completed.stdout)
            calls = log.read_text(encoding="utf-8").splitlines()
        # The wrapper asks the interpreter whether it can import numpy before it
        # starts the campaign, rather than letting the campaign refuse per column.
        self.assertEqual(calls[0], "-c import numpy")
        self.assertEqual(calls[1], "tools/run_accuracy_campaign.py --dry-run")

    def test_an_interpreter_without_numpy_is_refused_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            directory = Path(raw_dir)
            interpreter = stub(directory, 'case "$1" in\n  -c) exit 1 ;;\nesac\nexit 0\n')
            completed = self.run_wrapper(interpreter, "--dry-run")
        self.assertEqual(completed.returncode, 3)
        self.assertIn("this interpreter has no numpy", completed.stderr)
        self.assertIn(str(interpreter), completed.stderr)
        self.assertIn("PYTHON=", completed.stderr)
        # Nothing was started: a wrong interpreter is not a campaign failure.
        self.assertNotIn("Campaign", completed.stdout)

    def test_the_wrapper_passes_its_arguments_through_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            directory = Path(raw_dir)
            log = directory / "argv.log"
            interpreter = stub(directory, f'echo "$@" >> "{log}"\nexit 0\n')
            self.run_wrapper(interpreter, "model", "e2a")
            calls = log.read_text(encoding="utf-8").splitlines()
        self.assertEqual(calls[-1], "tools/run_accuracy_campaign.py model e2a")

    def test_a_checkout_local_dot_venv_is_found_without_being_named(self) -> None:
        # The layout `python3 -m venv .venv` produces at the repository root.
        # The wrapper is copied so the repository itself is never written to.
        with tempfile.TemporaryDirectory() as raw_dir:
            checkout = Path(raw_dir)
            (checkout / ".venv" / "bin").mkdir(parents=True)
            log = checkout / "argv.log"
            stub(checkout / ".venv" / "bin",
                 f'echo "$@" >> "{log}"\nexit 0\n', name="python")
            script = checkout / SCRIPT.name
            script.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
            completed = subprocess.run(
                ["sh", str(script), "--dry-run"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                env={k: v for k, v in os.environ.items() if k != "PYTHON"},
                cwd=str(checkout), timeout=120)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            # Printed as the relative candidate the wrapper found, which is also
            # what makes the log readable next to the checkout it came from.
            self.assertIn("interpreter: .venv/bin/python", completed.stdout)
            self.assertTrue(log.is_file())

    def test_a_named_interpreter_that_does_not_exist_is_refused_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            absent = str(Path(raw_dir) / "no-such-venv" / "bin" / "python")
            completed = subprocess.run(
                ["sh", str(SCRIPT), "--dry-run"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                env=dict(os.environ, PYTHON=absent), cwd=str(REPO), timeout=120)
        self.assertEqual(completed.returncode, 2)
        self.assertIn(f"PYTHON={absent} is not an executable file", completed.stderr)


if __name__ == "__main__":
    unittest.main()
