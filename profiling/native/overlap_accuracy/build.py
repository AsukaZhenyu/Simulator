"""Explicit direct MSVC release build; no historical or GGML sources changed."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("compiler","ggml-source","ggml-build","cuda-root","msvc-root","sdk-root","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--sdk-version",required=True)
    args=parser.parse_args()
    for key in ("compiler","ggml_source","ggml_build","cuda_root","msvc_root","sdk_root","output"):
        setattr(args,key,getattr(args,key).resolve())
    here=Path(__file__).resolve().parent;out=args.output
    if out.exists():raise ValueError("new build directory required")
    context=here.parent/"execution_transforms_dual/compute_context.cpp"
    matched=here.parent/"matched_mechanisms/compute.cpp"
    if sha(context)!="37ed5525074d2653050daa433a7b938a6f7f751a1afdeaac17a8c21cc2173cd4" or sha(matched)!="2556bed47f9f45ffef7884f5f58b452ec69a5c036b98a1e137afc9c55085c327":
        raise ValueError("immutable helper source differs")
    content=context.read_bytes();marker=b"int main(int argc,char**argv)"
    if content.count(marker)!=1:raise ValueError("ambiguous legacy main")
    out.mkdir(parents=True)
    generated=out/"context_generated.inc"
    generated.write_bytes(content.replace(marker,b"static int overlap_context_legacy_main(int argc,char**argv)"))
    binding=dict(probe_sha256=sha(here/"compute.cpp"),context_sha256=sha(context),matched_sha256=sha(matched),
                 generated_include_sha256=sha(generated),cmake_sha256=sha(here/"CMakeLists.txt"),build_script_sha256=sha(__file__))
    includes=[args.ggml_source/"include",args.cuda_root/"include",args.msvc_root/"include",
              *(args.sdk_root/"Include"/args.sdk_version/p for p in ("ucrt","um","shared")),out,context.parent]
    libpaths=[args.msvc_root/"lib/x64",*(args.sdk_root/"Lib"/args.sdk_version/p/"x64" for p in ("ucrt","um"))]
    libs=[args.ggml_build/"ggml/src/Release/ggml-base.lib",args.ggml_build/"ggml/src/Release/ggml-cpu.lib",
          args.ggml_build/"ggml/src/ggml-cuda/Release/ggml-cuda.lib",args.cuda_root/"lib/x64/cudart.lib"]
    binary=out/"overlap-accuracy-probe.exe"
    command=[str(args.compiler),"/nologo","/EHsc","/std:c++17","/O2","/MD","/W4","/DMATCHED_HAVE_CUDA","/DGGML_SHARED","/DGGML_BACKEND_SHARED",
        '/DOVERLAP_PROBE_SHA="'+binding["probe_sha256"]+'"','/DOVERLAP_CONTEXT_SHA="'+binding["context_sha256"]+'"',
        '/DOVERLAP_MATCHED_SHA="'+binding["matched_sha256"]+'"',*("/I"+str(p) for p in includes),
        "/Fe:"+str(binary),"/Fo:"+str(out/"compute.obj"),str(here/"compute.cpp"),"/link","/INCREMENTAL:NO",
        *("/LIBPATH:"+str(p) for p in libpaths),*(str(p) for p in libs)]
    started=datetime.now(timezone.utc).isoformat()
    proc=subprocess.run(command,cwd=out,capture_output=True)
    record=dict(schema="overlap-native-build-0.1",command=command,started_utc=started,returncode=proc.returncode,
        stdout=proc.stdout.decode(errors="replace"),stderr=proc.stderr.decode(errors="replace"),sources=binding,
        linked_libraries={str(p):sha(p) for p in libs},ggml_headers={str(p):sha(p) for p in args.ggml_source.joinpath("include").glob("*.h")})
    if proc.returncode==0:record["binary_sha256"]=sha(binary)
    (out/"build-record.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(record["stdout"]);print(record["stderr"]);print(binary)
    return proc.returncode


if __name__=="__main__":raise SystemExit(main())
