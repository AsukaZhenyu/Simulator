"""Build a private precision-correct CUDA router, preserving the supplied build.

This is deliberately a partial rebuild: compile the one patched translation
unit and relink byte-for-byte copies of the other existing Release objects.
The manifest records that scope and every source/object/library/command hash.
It does not attest how the supplied objects were originally compiled.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

PINNED_ROUTER_SHA256 = "65389d299a774b8a777ac07b1a6558a46322638b3f99d337452d3f6f3d0cd50b"
PATCH_OLD = """    bool use_mul_mat_f     = !ggml_is_quantized(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;"""
PATCH_NEW = """    // Strict F32 requests must bypass MMF's explicit TF32 MMA instructions.
    bool use_mul_mat_f     = !ggml_is_quantized(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && ggml_get_op_params_i32(dst, 0) != GGML_PREC_F32;"""


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def patch_router(raw):
    if hashlib.sha256(raw).hexdigest() != PINNED_ROUTER_SHA256:
        raise ValueError("unsupported GGML router bytes; review and pin a new upstream version")
    text = raw.decode("utf-8")
    if text.count(PATCH_OLD) != 1 or "mma" not in text:
        raise ValueError("precision router patch context differs")
    return text.replace(PATCH_OLD, PATCH_NEW).encode("utf-8")


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def environment(devcmd):
    # Read a process-local compiler environment; never change the user's PATH.
    result = subprocess.run(
        f'cmd.exe /d /s /c ""{devcmd}" -no_logo -arch=x64 -host_arch=x64 >nul && set"',
        capture_output=True, text=True, check=True)
    return {key: value for line in result.stdout.splitlines() if "=" in line and not line.startswith("=")
            for key, value in [line.split("=", 1)]}


def run(command, cwd, env, destination):
    started = datetime.now(timezone.utc).isoformat()
    try:
        result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=600)
        record = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        def decoded(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
        record = dict(returncode=None, stdout=decoded(getattr(error, "stdout", "")),
                      stderr=decoded(getattr(error, "stderr", "")), exception=repr(error))
    record.update(command=command, cwd=str(cwd), started_at=started,
                  completed_at=datetime.now(timezone.utc).isoformat())
    write(destination, record)
    if record["returncode"] != 0:
        raise RuntimeError(f"private runtime build failed; preserved {destination}: {record['stderr'][-3000:]}")
    return record


def build(args):
    if os.name != "nt":
        raise ValueError("this bounded object-reuse builder supports Windows MSVC/CUDA Release builds only")
    original = args.ggml_source.resolve()
    supplied = args.ggml_build.resolve()
    output = args.output.resolve()
    if output.exists() or output == original or original in output.parents or supplied in output.parents:
        raise ValueError("output must be a NEW private directory outside supplied source/build")
    router = original / "src/ggml-cuda/ggml-cuda.cu"
    patched = patch_router(router.read_bytes())  # Fail before creating output.
    project = supplied / "ggml/src/ggml-cuda/ggml-cuda.vcxproj"
    ns = {"m": "http://schemas.microsoft.com/developer/msbuild/2003"}
    release = [g for g in ET.parse(project).getroot().findall("m:ItemDefinitionGroup", ns)
               if "'Release|x64'" in g.get("Condition", "")]
    if len(release) != 1:
        raise ValueError("one x64 Release project definition required")
    cu = release[0].find("m:CudaCompile", ns)
    options = cu.findtext("m:AdditionalOptions", namespaces=ns)
    match = re.search(r"--generate-code=arch=compute_(\d+),code=\[compute_\1,sm_\1\]", options)
    if not match or cu.findtext("m:Runtime", namespaces=ns) != "MD" or cu.findtext("m:Optimization", namespaces=ns) != "O2":
        raise ValueError("unsupported supplied CUDA Release compilation contract")
    object_dir = project.parent / "ggml-cuda.dir/Release"
    objects = sorted(object_dir.glob("*.obj"))
    if not objects or sum(p.name.lower() == "ggml-cuda.obj" for p in objects) != 1:
        raise ValueError("complete supplied object inventory including router required")
    # Prove that this is precisely the object set from the supplied link record.
    log = object_dir / "ggml-cuda.tlog/link.command.1.tlog"
    raw = log.read_bytes()
    linked = raw.decode("utf-16" if raw[:2] == b"\xff\xfe" else "utf-16-le").splitlines()[1:]
    linked_names = re.findall(r'"[^"\r\n]*[\\/]([^"\\/]+\.OBJ)"', "\n".join(linked), re.I)
    if sorted(p.name.lower() for p in objects) != sorted(name.lower() for name in linked_names):
        raise ValueError("supplied link record/object inventory differs")
    output.mkdir(parents=True)
    source = output / "source/ggml"
    shutil.copytree(original, source, ignore=shutil.ignore_patterns(".git", "build", "__pycache__"))
    source_before = {str(p.resolve()): digest(p) for p in sorted(original.rglob("*"))
                     if p.is_file() and ".git" not in p.parts and "build" not in p.parts and "__pycache__" not in p.parts}
    private_router = source / "src/ggml-cuda/ggml-cuda.cu"
    private_router.write_bytes(patched)
    diff = "".join(difflib.unified_diff(router.read_text().splitlines(True), patched.decode().splitlines(True),
                                     fromfile="original/ggml-cuda.cu", tofile="private/ggml-cuda.cu"))
    (output / "strict-f32-router.patch").write_text(diff, encoding="utf-8", newline="\n")
    objout = output / "objects"
    objout.mkdir()
    reused = {}
    for path in objects:
        if path.name.lower() != "ggml-cuda.obj":
            target = objout / path.name
            shutil.copy2(path, target)
            reused[str(target)] = dict(original=str(path), sha256=digest(path))
    bindir = output / "bin"
    bindir.mkdir()
    original_runtime = supplied / "bin/Release"
    copied_runtime = {}
    for path in sorted(original_runtime.glob("*.dll")):
        if path.name.lower() != "ggml-cuda.dll":
            target = bindir / path.name
            shutil.copy2(path, target)
            copied_runtime[str(target)] = dict(original=str(path), sha256=digest(path))
    env = environment(args.vs_dev_cmd.resolve())
    nvcc = args.cuda_root.resolve() / "bin/nvcc.exe"
    compiler_dir = Path(env["VCToolsInstallDir"]) / "bin/Hostx64/x64"
    compile_command = [str(nvcc), "-c", str(private_router), "-o", str(objout / "ggml-cuda.obj"),
        "-ccbin", str(compiler_dir), "-m64", "-O2", "--use_fast_math", "-std=c++17", "-extended-lambda",
        f"--generate-code=arch=compute_{match[1]},code=[compute_{match[1]},sm_{match[1]}]",
        "-Xcompiler=/EHsc,/Ob2,/MD,/Zc:preprocessor"]
    for path in (source / "src", source / "include", args.cuda_root.resolve() / "include"):
        compile_command += ["-I", str(path)]
    for definition in cu.findtext("m:Defines", namespaces=ns).split(";"):
        if definition and "%(" not in definition:
            compile_command += ["-D", definition]
    run(compile_command, output, env, output / "compile-execution.json")
    cuda_lib = args.cuda_root.resolve() / "lib/x64"
    libraries = [supplied / "ggml/src/Release/ggml-base.lib"] + [cuda_lib / (name + ".lib")
        for name in ("cudart", "cublas", "cuda", "cublasLt", "cudadevrt", "cudart_static")]
    system_names = ("kernel32", "user32", "gdi32", "winspool", "shell32", "ole32", "oleaut32", "uuid", "comdlg32", "advapi32")
    system_libraries = []
    for name in system_names:
        candidates = [Path(folder) / (name + ".lib") for folder in env["LIB"].split(";") if folder]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise ValueError("system import library not found: " + name)
        system_libraries.append(path)
    libraries += system_libraries
    response = output / "link.rsp"
    arguments = ["/DLL", "/NOLOGO", "/INCREMENTAL:NO", "/MACHINE:X64", "/DYNAMICBASE", "/NXCOMPAT", "/SUBSYSTEM:CONSOLE",
        "/MANIFEST:EMBED", f'/OUT:"{bindir / "ggml-cuda.dll"}"', f'/IMPLIB:"{output / "ggml-cuda.lib"}"']
    arguments += [f'"{p}"' for p in sorted(objout.glob("*.obj"))] + [f'"{p}"' for p in libraries]
    response.write_text("\n".join(arguments) + "\n", encoding="utf-8")
    run([str(compiler_dir / "link.exe"), "@" + str(response)], output, env, output / "link-execution.json")
    if any(digest(path) != value for path, value in source_before.items()):
        raise ValueError("supplied GGML source changed during private build")
    if any(digest(row["original"]) != row["sha256"] or digest(path) != row["sha256"] for path, row in reused.items()):
        raise ValueError("supplied/copied object changed during private build")
    inputs = {str(p): digest(p) for p in sorted(source.rglob("*")) if p.is_file()}
    inputs.update({str(p): digest(p) for p in sorted(objout.glob("*.obj"))})
    inputs.update({str(p): digest(p) for p in libraries})
    inputs.update({str(Path(__file__).resolve()): digest(__file__), str(project): digest(project), str(log): digest(log),
                   str(nvcc): digest(nvcc), str(compiler_dir / "cl.exe"): digest(compiler_dir / "cl.exe"),
                   str(compiler_dir / "link.exe"): digest(compiler_dir / "link.exe")})
    record = dict(format_version="ggml-private-strict-f32-runtime-v1", created_at=datetime.now(timezone.utc).isoformat(),
        build_scope="ONE patched router translation unit compiled; remaining supplied Release objects copied unchanged; not a clean GGML rebuild",
        precision_scope="regular GGML_OP_MUL_MAT with requested GGML_PREC_F32 bypasses custom MMF TF32; MUL_MAT_ID unchanged; no loader-path observation",
        original_source_sha256=source_before, inputs_sha256=inputs, reused_objects=reused,
        copied_runtime=copied_runtime, supplied_project_options=options,
        cuda_architecture=match[1], patch_hunks=1, original_router_sha256=PINNED_ROUTER_SHA256,
        patched_router_sha256=digest(private_router),
        output_sha256={str(p): digest(p) for p in [*sorted(bindir.glob("*.dll")), output / "ggml-cuda.lib"]},
        evidence_sha256={str(output / name): digest(output / name) for name in
            ("strict-f32-router.patch", "compile-execution.json", "link-execution.json", "link.rsp")})
    write(bindir / "strict-f32-runtime.json", record)
    print(json.dumps(dict(runtime_dir=str(bindir), manifest=str(bindir / "strict-f32-runtime.json"),
                          manifest_sha256=digest(bindir / "strict-f32-runtime.json"), scope=record["build_scope"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ggml-source", required=True, type=Path)
    parser.add_argument("--ggml-build", required=True, type=Path, help="existing llama.cpp build root")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cuda-root", required=True, type=Path)
    parser.add_argument("--vs-dev-cmd", required=True, type=Path)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
