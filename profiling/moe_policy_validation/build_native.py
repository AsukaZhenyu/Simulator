"""Build the additive probe against explicit existing GGML libraries.

No runtime download, library modification, or automatic CPU fallback occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--llama-root", type=Path, required=True)
    p.add_argument("--ggml-build", type=Path, required=True)
    p.add_argument("--runtime-dir", type=Path, action="append", required=True)
    a = p.parse_args()
    out = a.out.resolve()
    if out.exists():
        raise ValueError("new build directory required")
    source = Path(__file__).parent / "native"
    out.mkdir(parents=True)
    commands = [
        ["cmake", "-S", str(source.resolve()), "-B", str(out), "-G", "Visual Studio 17 2022", "-A", "x64",
         "-DLLAMA_ROOT=" + str(a.llama_root.resolve()), "-DGGML_BUILD_DIR=" + str(a.ggml_build.resolve()), "-DCMAKE_TRY_COMPILE_CONFIGURATION=Release"],
        ["cmake", "--build", str(out), "--config", "Release"],
    ]
    for index, command in enumerate(commands):
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240)
        (out / f"build-{index}.json").write_text(json.dumps(dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr), indent=2), encoding="utf-8", newline="\n")
        if result.returncode:
            raise RuntimeError(f"build step {index} failed; retained log in {out}")
    binary = out / "Release/moe-expert-policy-probe.exe"
    runtimes = [receipt(f) for folder in a.runtime_dir for f in sorted(folder.glob("*.dll"))]
    manifest = dict(schema="moe-expert-policy-build-1", binary=receipt(binary), sources=[receipt(x) for x in sorted(source.iterdir()) if x.is_file()],
                    runtime_dirs=[str(x.resolve()) for x in a.runtime_dir], runtimes=runtimes,
                    compile_cache=receipt(out / "CMakeCache.txt"), mathematical_weight_dtype="F32", GPU_required=True)
    (out / "BUILD_COMPLETE.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(dict(binary=str(binary), manifest=str(out / "BUILD_COMPLETE.json"))))


if __name__ == "__main__":
    main()
