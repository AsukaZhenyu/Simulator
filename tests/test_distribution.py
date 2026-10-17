"""Verify an offline installation, with neither the checkout nor saved profiles.

The build interpreter needs setuptools >= 68, wheel and pip. The installed
runtime has no third-party dependencies. Run this file with unittest; it builds
both distribution formats in a temporary project, then tests a fresh venv from
outside that project. No package index is contacted by the test.
"""

from __future__ import annotations

import configparser
import email
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    "llm_simulator": "src/llm_simulator",
    "tensor_mapping": "mapping/tensor_mapping",
    "llm_infer_model": "modeling/llm_infer_model",
}
BUILD_AVAILABLE = all(importlib.util.find_spec(name) for name in ("setuptools", "wheel", "pip"))


@unittest.skipUnless(BUILD_AVAILABLE, "offline distribution test needs setuptools, wheel and pip")
class DistributionTests(unittest.TestCase):
    def test_sdist_and_wheel_work_without_checkout_or_profiles(self) -> None:
        with tempfile.TemporaryDirectory(prefix="llm-simulator-distribution-") as directory:
            temporary = Path(directory)
            project = temporary / "project"
            project.mkdir()
            for name in ("pyproject.toml", "README.md", "LICENSE", "MANIFEST.in"):
                shutil.copy2(ROOT / name, project / name)
            for source in PACKAGES.values():
                shutil.copytree(
                    ROOT / source,
                    project / source,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
                )
            source_assets = [
                "CITATION.cff", "docs/LIBRARY_GUIDE.md",
                "docs/OPEN_SOURCE_AND_PREPRINT.md", "docs/RELEASE_0_1.md",
                "tests/test_distribution.py", "tests/test_library.py",
                "tests/test_analytical_inputs.py",
            ]
            examples = ROOT / "examples/library"
            if examples.is_dir():
                source_assets.extend(
                    path.relative_to(ROOT).as_posix()
                    for path in examples.rglob("*")
                    if path.is_file() and path.suffix in (".py", ".md")
                )
            copied_assets = []
            for source in source_assets:
                if (ROOT / source).is_file():
                    target = project / source
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(ROOT / source, target)
                    copied_assets.append(source)

            distribution = temporary / "dist"
            distribution.mkdir()
            self._run(
                [sys.executable, "-I", "-c",
                 "from setuptools.build_meta import build_sdist, build_wheel; import sys; "
                 "output=sys.argv[1]; build_sdist(output); build_wheel(output)", str(distribution)],
                cwd=project,
            )
            wheels = list(distribution.glob("*.whl"))
            archives = list(distribution.glob("*.tar.gz"))
            self.assertEqual(len(wheels), 1)
            self.assertEqual(len(archives), 1)
            with zipfile.ZipFile(wheels[0]) as archive:
                members = set(archive.namelist())
                metadata_path = next(name for name in members if name.endswith(".dist-info/METADATA"))
                metadata = email.message_from_bytes(archive.read(metadata_path))
                scripts = configparser.ConfigParser()
                scripts.read_string(archive.read(metadata_path.replace("METADATA", "entry_points.txt")).decode())
            self.assertEqual(metadata["Name"], "llm-infer-simulator")
            self.assertEqual(metadata["Version"], "0.1.0a1")
            self.assertIn("MIT License", metadata["License"])
            self.assertIn("llm_simulator/py.typed", members)
            self.assertTrue(any(name.endswith("/LICENSE") and ".dist-info/" in name for name in members))
            # Every declared runtime requirement must be gated by an optional extra.
            self.assertTrue(all("extra ==" in row for row in metadata.get_all("Requires-Dist", [])))
            self.assertEqual(scripts["console_scripts"]["llm-sim"], "llm_simulator.cli:main")
            self.assertEqual(set(scripts["console_scripts"]), {"llm-sim"})

            expected_modules: set[str] = set()
            for name, source in PACKAGES.items():
                for path in (ROOT / source).rglob("*.py"):
                    expected_modules.add(name + "/" + path.relative_to(ROOT / source).as_posix())
            self.assertTrue(expected_modules <= members, sorted(expected_modules - members))
            data = ROOT / "src/llm_simulator/data"
            if data.is_dir():
                expected_data = {
                    "llm_simulator/data/" + path.relative_to(data).as_posix()
                    for path in data.rglob("*")
                    if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo")
                }
                self.assertTrue(expected_data <= members, sorted(expected_data - members))
            for member in members:
                parts = set(member.lower().split("/"))
                self.assertFalse(parts & {"profiling", "results", "outputs", "reference", "reference-pdf", "__pycache__"}, member)
                self.assertFalse(member.lower().endswith((".gguf", ".safetensors", ".pt", ".pth", ".bin", ".pdf")), member)
                self.assertNotIn(member.split("/")[0], {"tests", "docs", "examples"})

            extracted = temporary / "sdist"
            extracted.mkdir()
            with tarfile.open(archives[0]) as archive:
                source_members = {name.partition("/")[2] for name in archive.getnames()}
                self.assertTrue(set(copied_assets) <= source_members, sorted(set(copied_assets) - source_members))
                # Check every path before extracting a locally built archive.
                for member in archive.getmembers():
                    target = (extracted / member.name).resolve()
                    self.assertTrue(target.is_relative_to(extracted.resolve()), member.name)
                    self.assertFalse(member.issym() or member.islnk(), member.name)
                if hasattr(tarfile, "data_filter"):
                    archive.extractall(extracted, filter="data")
                else:
                    archive.extractall(extracted)
            source_roots = list(extracted.iterdir())
            self.assertEqual(len(source_roots), 1)
            rebuilt = temporary / "rebuilt"
            rebuilt.mkdir()
            self._run(
                [sys.executable, "-I", "-c",
                 "from setuptools.build_meta import build_wheel; import sys; build_wheel(sys.argv[1])",
                 str(rebuilt)],
                cwd=source_roots[0],
            )
            rebuilt_wheels = list(rebuilt.glob("*.whl"))
            self.assertEqual(len(rebuilt_wheels), 1)
            with zipfile.ZipFile(rebuilt_wheels[0]) as archive:
                self.assertEqual(members, set(archive.namelist()))

            environment = temporary / "venv"
            self._run([sys.executable, "-I", "-m", "venv", str(environment)], cwd=temporary)
            scripts_dir = environment / ("Scripts" if os.name == "nt" else "bin")
            interpreter = scripts_dir / ("python.exe" if os.name == "nt" else "python")
            self._run(
                [str(interpreter), "-I", "-m", "pip", "install", "--no-index", "--no-deps",
                 "--no-compile", "--disable-pip-version-check", str(rebuilt_wheels[0])],
                cwd=temporary,
            )
            # A fresh venv has neither old editable packages nor a PYTHONPATH escape.
            imports = self._run(
                [str(interpreter), "-I", "-c",
                 "import importlib, importlib.metadata, json, pathlib, sys; "
                 "root=pathlib.Path(sys.prefix).resolve(); names=json.loads(sys.argv[1]); "
                 "origins={name: str(pathlib.Path(importlib.import_module(name).__file__).resolve()) for name in names}; "
                 "assert all(pathlib.Path(path).is_relative_to(root) for path in origins.values()), origins; "
                 "assert importlib.metadata.version('llm-infer-simulator')=='0.1.0a1'; "
                 "assert importlib.util.find_spec('numpy') is None; "
                 "assert importlib.util.find_spec('gguf') is None; "
                 "print(json.dumps(origins, sort_keys=True))",
                 json.dumps([
                     "llm_simulator", "tensor_mapping", "llm_infer_model",
                     "tensor_mapping.python_frontend", "tensor_mapping.unified_strategy",
                     "llm_infer_model.tensor.program", "llm_infer_model.m2.expand",
                     "llm_infer_model.m2.global_execution", "llm_infer_model.components",
                 ])],
                cwd=temporary,
            )
            self.assertIn("llm_simulator", json.loads(imports.stdout))
            executable = scripts_dir / ("llm-sim.exe" if os.name == "nt" else "llm-sim")
            help_result = self._run([str(executable), "--help"], cwd=temporary)
            self.assertIn("usage:", help_result.stdout)
            version_result = self._run([str(executable), "--version"], cwd=temporary)
            self.assertIn("0.1.0a1", version_result.stdout)
            for model in ("dense", "moe"):
                output = temporary / (model + "-simulation")
                demo = self._run([str(executable), "demo", "--model", model, "--out", str(output)], cwd=temporary)
                summary = json.loads(demo.stdout.strip().splitlines()[-1])
                self.assertEqual(summary["status"], "valid", summary)
                self.assertIs(summary["success"], True)
                self.assertIs(summary["calibrated"], False)
                self.assertIs(summary["hardware_measurement"], False)
                self.assertEqual(summary["cost_model"], "analytical_roofline")
                self.assertTrue(output.is_dir(), model)
                saved = output / "result/library-summary.json"
                self.assertEqual(json.loads(saved.read_text(encoding="utf-8")), summary)

    def _run(self, command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        environment["PIP_CONFIG_FILE"] = os.devnull
        environment["PIP_NO_INDEX"] = "1"
        environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        completed = subprocess.run(
            command, cwd=cwd, env=environment, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=240,
        )
        self.assertEqual(completed.returncode, 0, f"command={command}\nstdout={completed.stdout}\nstderr={completed.stderr}")
        return completed


if __name__ == "__main__":
    unittest.main()
