"""Prepare the fixed E2A matrix from explicit binaries, without old profiles.

The historical experiment entry point reuses a validated local build descriptor.
This adapter supplies only binary/runtime paths explicitly; no prior measurement
manifest, frozen profile or target result is required on a fresh machine.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json

from . import e2a
from profiling import host


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--binary",type=Path,required=True)
    parser.add_argument("--runtime-dir",type=Path,action="append",required=True)
    args=parser.parse_args()
    binary=args.binary.resolve();runtime_dirs=[str(p.resolve()) for p in args.runtime_dir]
    # Every refusal names the path it is about: on a rented machine the paths
    # are typed, and "explicit runtime directory missing" alone does not say
    # which of several `--runtime-dir` values was the misspelling.
    e2a.require(binary.is_file(),"explicit binary is not a file: "+str(binary))
    e2a.require((binary.parent/"build-manifest.json").is_file(),
        "explicit source-bound native build required: no build-manifest.json beside "+str(binary))
    for directory in runtime_dirs:
        e2a.require(Path(directory).is_dir(),"explicit runtime directory missing: "+directory)
    # Same refusal as the depth column's adapter: a runtime list that holds no
    # library of this platform prepares a cohort whose every process exits before
    # it measures anything. The glob is the host's own, so a Linux host checks
    # for `lib*.so` and never passes because a `.dll` list happened to be right
    # on the machine the table came from.
    libraries=[path for directory in runtime_dirs
               for path in Path(directory).glob(host.shared_library_pattern())]
    e2a.require(libraries,"the runtime directories hold no "+host.shared_library_pattern()
        +" files; the probe would not load")
    # Holding *a* library is not holding the three the probe loads by name. A
    # strict-F32 build that stopped part way has `libggml-base.so` and nothing
    # else, which passes the check above; the process then exits on an unresolved
    # symbol, one phase into a cohort that can no longer be reused.
    e2a.require_components(runtime_dirs)
    descriptor=dict(binary=str(binary),runtime_dirs=runtime_dirs)
    original_read=e2a.read;legacy_descriptor=(e2a.ROOT/"profiling/steady_costs/results/steady-v1-f32/manifest.json").resolve()
    def explicit_descriptor_read(path):
        if Path(path).resolve()==legacy_descriptor:return descriptor
        return original_read(path)
    e2a.read=explicit_descriptor_read
    try:result=e2a.prepare(args.out)
    finally:e2a.read=original_read
    e2a.write(args.out/"portable-preparation-receipt.json",dict(schema="dac-e2a-explicit-build-preparation-1",
        source=e2a.receipt(__file__),fixed_matrix_builder=e2a.receipt(e2a.__file__),explicit_build_descriptor=descriptor,
        prior_profile_or_result_read=False,prepared_manifest=e2a.receipt(args.out/"manifest.json")))
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
