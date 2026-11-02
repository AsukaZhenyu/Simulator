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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--binary",type=Path,required=True)
    parser.add_argument("--runtime-dir",type=Path,action="append",required=True)
    args=parser.parse_args()
    binary=args.binary.resolve();runtime_dirs=[str(p.resolve()) for p in args.runtime_dir]
    e2a.require(binary.is_file() and (binary.parent/"build-manifest.json").is_file(),"explicit source-bound native build required")
    for directory in runtime_dirs:e2a.require(Path(directory).is_dir(),"explicit runtime directory missing")
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
