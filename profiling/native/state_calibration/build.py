"""Build the state-aligned diagnostic without changing historical GGML sources.

All compiler, SDK, source, library, and runtime paths are explicit. A fresh output
directory is mandatory. The generated historical helper changes only main's
name; its original exact bytes and runtime assets are recorded in the manifest.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess


HELPER_SHA256 = "ceeebc81c00a4fb760923c7b1286a7470558250488f93119c7d628dfed0872eb"
EVENT_ABI_FILES = {
    "ggml/src/ggml-backend-impl.h": "470f4ede95a1a66edb7259981c1e8c98e2827de765323fe1672badf8f99f4546",
    "ggml/src/ggml-backend.cpp": "20e449bd92cec4c0c16ab91dc103ee6871ac6c45f22fd70f1d43bca33f171500",
    "ggml/src/ggml-cuda/ggml-cuda.cu": "65389d299a774b8a777ac07b1a6558a46322638b3f99d337452d3f6f3d0cd50b",
}


def execution_boundaries(content: str | None = None) -> tuple[str, ...]:
    """Read the same explicit boundary declarations used by the C++ parser."""
    if content is None:
        content = Path(__file__).with_name("execution_boundaries.def").read_text(encoding="utf-8")
    names = []
    for line in content.splitlines():
        match = re.fullmatch(r"STATE_BOUNDARY\(([a-z][a-z0-9_]*)\)", line)
        if not match:
            raise ValueError("invalid native execution boundary declaration")
        names.append(match.group(1))
    if not names or len(names) != len(set(names)):
        raise ValueError("empty or duplicate native execution boundaries")
    return tuple(names)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generated_helper(content: bytes) -> bytes:
    """Replace exactly the main definition, never a wider text substitution."""
    if hashlib.sha256(content).hexdigest() != HELPER_SHA256:
        raise ValueError("immutable historical helper bytes differ")
    marker = b"int main(int argc,char **argv)"
    if content.count(marker) != 1:
        raise ValueError("ambiguous historical main definition")
    return content.replace(marker, b"static int state_historical_main(int argc,char **argv)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("compiler", "llama-root", "ggml-build", "msvc-root", "sdk-root", "runtime-dir", "cuda-root", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--sdk-version", required=True)
    args = parser.parse_args()
    for key in ("compiler", "llama_root", "ggml_build", "msvc_root", "sdk_root", "runtime_dir", "cuda_root", "output"):
        setattr(args, key, getattr(args, key).resolve())
    here = Path(__file__).resolve().parent
    helper = here.parent / "highlevel_accuracy" / "highlevel_probe.cpp"
    runtime_assets = sorted(args.runtime_dir.glob("*.dll"))
    if not runtime_assets:
        raise ValueError("explicit runtime directory contains no DLLs")
    required_runtime = {"ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll"}
    if not required_runtime.issubset({path.name for path in runtime_assets}):
        raise ValueError("explicit runtime directory lacks GGML backend DLLs")
    if args.output.exists():
        raise ValueError("new build directory required")
    content = generated_helper(helper.read_bytes())
    for name, digest in EVENT_ABI_FILES.items():
        if sha(args.llama_root / name) != digest:
            raise ValueError("pinned GGML event ABI/callback implementation differs: " + name)
    boundaries = execution_boundaries()
    includes = [args.llama_root / "ggml/include", args.llama_root / "ggml/src", args.llama_root / "vendor", args.msvc_root / "include", args.cuda_root / "include",
                *(args.sdk_root / "Include" / args.sdk_version / name for name in ("ucrt", "um", "shared")), args.output]
    libraries = [args.ggml_build / "ggml/src/Release/ggml-base.lib",
                 args.ggml_build / "ggml/src/Release/ggml-cpu.lib",
                 args.ggml_build / "ggml/src/ggml-cuda/Release/ggml-cuda.lib", args.cuda_root / "lib/x64/cudart.lib"]
    library_dirs = [args.msvc_root / "lib/x64", *(args.sdk_root / "Lib" / args.sdk_version / name / "x64" for name in ("ucrt", "um"))]
    all_required = [args.compiler, here / "compute.cpp", helper, *libraries, *runtime_assets,
                    args.llama_root / "vendor/nlohmann/json.hpp"]
    if any(not path.is_file() for path in all_required) or any(not path.is_dir() for path in [*includes[:-1], *library_dirs]):
        raise ValueError("an explicit required compiler/source/header/library/runtime path is missing")
    args.output.mkdir(parents=True)
    generated = args.output / "highlevel_generated.inc"
    generated.write_bytes(content)
    (args.output / "highlevel_original.cpp").write_bytes(helper.read_bytes())
    binary = args.output / "state-calibration-probe.exe"
    command = [str(args.compiler), "/nologo", "/EHsc", "/std:c++17", "/O2", "/MD", "/W4", "/DHIGHLEVEL_CUDA=1", "/DGGML_SHARED", "/DGGML_BACKEND_SHARED",
               *("/I" + str(path) for path in includes), "/Fe:" + str(binary), "/Fo:" + str(args.output / "compute.obj"), str(here / "compute.cpp"),
               "/link", "/INCREMENTAL:NO", *("/LIBPATH:" + str(path) for path in library_dirs), *(str(path) for path in libraries)]
    source_files = [here / "compute.cpp", here / "execution_boundary.h", here / "execution_boundaries.def", Path(__file__).resolve(), helper, generated, args.output / "highlevel_original.cpp",
                    *(args.llama_root / name for name in EVENT_ABI_FILES)]
    header_files = [*sorted((args.llama_root / "ggml/include").glob("*.h")), args.llama_root / "vendor/nlohmann/json.hpp",
                    *sorted((args.cuda_root / "include").rglob("*.h")), *sorted((args.cuda_root / "include").rglob("*.hpp"))]
    bound = {str(path): sha(path) for path in [*source_files, *header_files, *libraries, *runtime_assets]}
    runtime_manifests = sorted(args.runtime_dir.glob("*manifest*.json")) + sorted(args.runtime_dir.parent.glob("*runtime*.json"))
    bound.update({str(path): sha(path) for path in runtime_manifests})
    started = datetime.now(timezone.utc).isoformat()
    process = subprocess.run(command, cwd=args.output, capture_output=True)
    record = dict(format_version="state-calibration-native-build-v1", command=command, started_utc=started,
                  finished_utc=datetime.now(timezone.utc).isoformat(), returncode=process.returncode,
                  stdout=process.stdout.decode(errors="replace"), stderr=process.stderr.decode(errors="replace"),
                  compiler_sha256=sha(args.compiler), bound_files=bound, immutable_helper_sha256=HELPER_SHA256,
                  execution_boundaries=boundaries, event_abi_pinned_files=EVENT_ABI_FILES,
                  event_access="timing-enabled cudaEvent_t wrapped in pinned ggml_backend_event; public callback records actual GGML stream; no CUDA context-layout access",
                  runtime_dir=str(args.runtime_dir), runtime_loader_resolution_proven=False,
                  bound_inputs_unchanged=all(sha(Path(path)) == digest for path, digest in bound.items()))
    if process.returncode == 0:
        record["binary"] = str(binary)
        record["binary_sha256"] = sha(binary)
    (args.output / "build-manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(record["stdout"])
    print(record["stderr"])
    print(binary)
    return process.returncode if record["bound_inputs_unchanged"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
