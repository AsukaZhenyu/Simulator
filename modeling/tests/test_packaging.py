"""Build and install the runtime wheel offline, without source-path imports.

This test needs the project's declared build backend (setuptools >= 68) and
wheel installed in the test interpreter. It never downloads dependencies.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

MODELING_ROOT = Path(__file__).resolve().parents[1]
BUILD_AVAILABLE = all(importlib.util.find_spec(name) for name in ("setuptools", "wheel", "pip"))


@unittest.skipUnless(BUILD_AVAILABLE, "offline packaging test needs setuptools, wheel and pip")
class InstalledPackageTests(unittest.TestCase):
    def test_installed_wheel_contains_and_imports_all_runtime_subpackages(self) -> None:
        with tempfile.TemporaryDirectory(prefix="llm-infer-wheel-") as directory:
            temporary = Path(directory)
            project = temporary / "project"
            project.mkdir()
            shutil.copy2(MODELING_ROOT / "pyproject.toml", project / "pyproject.toml")
            shutil.copytree(
                MODELING_ROOT / "llm_infer_model",
                project / "llm_infer_model",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            wheel_dir = temporary / "wheels"
            wheel_dir.mkdir()
            self._run(
                [sys.executable, "-I", "-c",
                 "from setuptools.build_meta import build_wheel; "
                 "import sys; build_wheel(sys.argv[1])", str(wheel_dir)],
                cwd=project,
            )
            wheels = list(wheel_dir.glob("*.whl"))
            self.assertEqual(len(wheels), 1)
            with zipfile.ZipFile(wheels[0]) as wheel:
                members = set(wheel.namelist())
            modules = (
                "llm_infer_model",
                "llm_infer_model.tensor",
                "llm_infer_model.m2",
                "llm_infer_model.components",
            )
            for module in modules:
                with self.subTest(module=module):
                    self.assertIn(module.replace(".", "/") + "/__init__.py", members)
            self.assertIn("llm_infer_model/units.py", members)
            modules += ("llm_infer_model.units",)
            for module_file in ("physical.py", "full_physical.py", "dense_contract.py",
                                "cuda_physical.py", "dense_cuda_contract.py", "hybrid_physical.py", "moe.py", "moe_spec.py", "moe_ffn_contract.py", "moe_measured.py", "__main__.py"):
                self.assertIn("llm_infer_model/m2/" + module_file, members)
            modules += ("llm_infer_model.m2.physical", "llm_infer_model.m2.full_physical",
                        "llm_infer_model.m2.dense_contract", "llm_infer_model.m2.cuda_physical",
                        "llm_infer_model.m2.dense_cuda_contract", "llm_infer_model.m2.hybrid_physical",
                        "llm_infer_model.m2.moe", "llm_infer_model.m2.moe_spec",
                        "llm_infer_model.m2.moe_ffn_contract", "llm_infer_model.m2.moe_measured",
                        "llm_infer_model.m2.__main__")
            install_dir = temporary / "installed"
            self._run(
                [sys.executable, "-I", "-m", "pip", "install", "--no-index",
                 "--no-deps", "--no-compile", "--disable-pip-version-check",
                 "--no-warn-script-location", "--target", str(install_dir), str(wheels[0])],
                cwd=temporary,
            )
            # -I removes cwd/PYTHONPATH. Every imported origin must be in the install target.
            code = (
                "import importlib, json, pathlib, sys; "
                "target=pathlib.Path(sys.argv[1]).resolve(); sys.path.insert(0, str(target)); "
                "names=json.loads(sys.argv[2]); "
                "origins={name: str(pathlib.Path(importlib.import_module(name).__file__).resolve()) "
                "for name in names}; "
                "assert all(pathlib.Path(path).is_relative_to(target) for path in origins.values()), origins; "
                "from llm_infer_model.m2 import CUDA_FORMAT_VERSION, CudaDensePhysical; "
                "assert CUDA_FORMAT_VERSION=='m2-layer-0.4'; "
                "assert 'threads' not in CudaDensePhysical.__dataclass_fields__; "
                "from llm_infer_model.m2 import HYBRID_FORMAT_VERSION, HybridDensePhysical; "
                "assert HYBRID_FORMAT_VERSION=='m2-layer-0.5'; "
                "assert {'threads','cuda_device'}<=HybridDensePhysical.__dataclass_fields__.keys(); "
                "from llm_infer_model.m2 import MOE_FORMAT_VERSION, STATE_CACHE_FORMAT_VERSION, WeightStateRef; "
                "assert MOE_FORMAT_VERSION=='m2-layer-0.6'; "
                "assert STATE_CACHE_FORMAT_VERSION=='m2-state-0.2'; "
                "assert 'locations' in WeightStateRef.__dataclass_fields__; "
                "from llm_infer_model.components import SlidingWindow, MoEFFN; "
                "assert SlidingWindow(kv_bytes_per_token=64,window=4).state(history=8,tokens=3).read_positions==6; "
                "assert MoEFFN(experts=2,top_k=2,expert_weight_bytes=8).plan(tokens=1,expert_tokens={0:1,1:1}).active_weight_bytes==16; "
                "from llm_infer_model.units import seconds_to_ns; "
                "from llm_infer_model.tensor.layer_costs import seconds_to_ns as legacy_seconds_to_ns; "
                "assert seconds_to_ns is legacy_seconds_to_ns; "
                "assert seconds_to_ns(1.5e-9, 'duration')==2; "
                "print(json.dumps(origins, sort_keys=True))"
            )
            completed = self._run(
                [sys.executable, "-I", "-c", code, str(install_dir), json.dumps(modules)],
                cwd=temporary,
            )
            self.assertEqual(set(json.loads(completed.stdout)), set(modules))

    def _run(self, command: list[str], *, cwd: Path) -> subprocess.CompletedProcess:
        completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=120)
        self.assertEqual(completed.returncode, 0, f"stdout={completed.stdout}\nstderr={completed.stderr}")
        return completed


if __name__ == "__main__":
    unittest.main()
