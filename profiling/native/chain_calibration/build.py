"""Direct MSVC release build, with exact commands, output and source hashes.

Equivalent to CMake's generated include transformation. This route avoids a
machine's broken debug-PDB compiler probes; it does not alter the toolchain.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("compiler", "ggml-source", "ggml-build", "cuda-root", "msvc-root", "sdk-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sdk-version", required=True)
    args = parser.parse_args()
    for name in ("compiler", "ggml_source", "ggml_build", "cuda_root", "msvc_root", "sdk_root"):
        setattr(args, name, getattr(args, name).resolve())
    here = Path(__file__).resolve().parent
    out = args.output.resolve()
    if out.exists():
        raise ValueError("build output must be a new directory")
    context = here.parent / "execution_transforms_dual" / "compute_context.cpp"
    matched = here.parent / "matched_mechanisms" / "compute.cpp"
    if sha(context) != "37ed5525074d2653050daa433a7b938a6f7f751a1afdeaac17a8c21cc2173cd4" or sha(matched) != "2556bed47f9f45ffef7884f5f58b452ec69a5c036b98a1e137afc9c55085c327":
        raise ValueError("pinned historical sources changed")
    original = context.read_bytes()
    marker = b"int main(int argc,char**argv)"
    if original.count(marker) != 1:
        raise ValueError("historical terminal entry point ambiguous")
    out.mkdir(parents=True)
    generated = out / "context_generated.inc"
    generated.write_bytes(original.replace(marker, b"static int chain_context_legacy_main(int argc,char**argv)"))
    binding = dict(context_sha256=sha(context), matched_sha256=sha(matched), probe_sha256=sha(here / "compute.cpp"),
                   cmake_sha256=sha(here / "CMakeLists.txt"), generated_include_sha256=sha(generated), transformation="rename_terminal_main_only")
    (out / "source-binding.json").write_text(json.dumps(binding, indent=2) + "\n", encoding="utf-8")
    binary = out / "chain-calibration-probe.exe"
    includes = [args.ggml_source / "include", args.cuda_root / "include", args.msvc_root / "include",
                *(args.sdk_root / "Include" / args.sdk_version / p for p in ("ucrt", "um", "shared")),
                out, context.parent]
    library_paths = [args.msvc_root / "lib" / "x64", *(args.sdk_root / "Lib" / args.sdk_version / p / "x64" for p in ("ucrt", "um"))]
    libraries = [args.ggml_build / "ggml" / "src" / "Release" / "ggml-base.lib",
                 args.ggml_build / "ggml" / "src" / "Release" / "ggml-cpu.lib",
                 args.ggml_build / "ggml" / "src" / "ggml-cuda" / "Release" / "ggml-cuda.lib",
                 args.cuda_root / "lib" / "x64" / "cudart.lib"]
    command = [str(args.compiler), "/nologo", "/EHsc", "/std:c++17", "/O2", "/MD", "/W4",
               "/DMATCHED_HAVE_CUDA", "/DGGML_SHARED", "/DGGML_BACKEND_SHARED",
               '/DCHAIN_CONTEXT_SHA="' + binding["context_sha256"] + '"',
               '/DCHAIN_MATCHED_SHA="' + binding["matched_sha256"] + '"',
               '/DCHAIN_PROBE_SHA="' + binding["probe_sha256"] + '"',
               *("/I" + str(p) for p in includes), "/Fe:" + str(binary), "/Fo:" + str(out / "compute.obj"),
               str(here / "compute.cpp"), "/link", "/INCREMENTAL:NO", *("/LIBPATH:" + str(p) for p in library_paths),
               *(str(p) for p in libraries)]
    proc = subprocess.run(command, cwd=out, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    record = dict(command=command, started_utc=datetime.now(timezone.utc).isoformat(), returncode=proc.returncode,
                  stdout=proc.stdout.decode("utf-8", errors="replace"), stderr=proc.stderr.decode("utf-8", errors="replace"),
                  source_binding=binding, build_script_sha256=sha(__file__),
                  linked_libraries={str(p.resolve()): sha(p) for p in libraries})
    if proc.returncode == 0:
        record["binary_sha256"] = sha(binary)
    (out / "build-record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(record["stdout"])
    print(record["stderr"])
    print(binary)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
