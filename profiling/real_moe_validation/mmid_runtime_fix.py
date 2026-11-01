"""Prepare one private MMID ordering change; never mutate a frozen runtime.

Preparation only records source and commands. Building requires a separate
root authorization bound to PREPARED.json. No inference or CUDA initialization
is performed here. The v5 observer object and v6 native driver are reused.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import difflib
import json
import os
from pathlib import Path
import re
import shutil

from .dataset import read, write
from .mmid_primitive import ANCHOR, FENCE
from .tap_v4 import receipt, run_record, sha


def fence_source(text: str) -> str:
    """The entire source differs by exactly one full-warp ordering operation."""
    if text.count(ANCHOR) != 1 or '__syncwarp(' in text:
        raise ValueError('pinned single unfenced MMID source required')
    result = text.replace(ANCHOR, ANCHOR.rstrip('\n') + '\n' + FENCE + '\n', 1)
    if result.replace(FENCE, '', 1) != text:
        raise ValueError('MMID change exceeds one fence')
    if result.index(FENCE) > result.index('    if (threadIdx.x != 0)'):
        raise ValueError('all participating warp lanes must reach the fence')
    return result


def compile_command(command: list[str], *, old_source: Path, old_object: Path,
                    old_include: Path, source: Path, object_file: Path,
                    include: Path) -> list[str]:
    """Reuse the verified v5 compile contract, removing only its observer flag."""
    before = list(command)
    exact = {str(old_source): str(source), str(old_object): str(object_file),
             '-I' + str(old_include): '-I' + str(include)}
    for value in exact:
        if before.count(value) != 1:
            raise ValueError('pinned CUDA compile argument missing/duplicated: ' + value)
    flag = '-DSIMULATOR_GPU_TAP_BUILD'
    if before.count(flag) != 1:
        raise ValueError('one v5 main-object observer define required')
    result = [exact.get(x, x) for x in before if x != flag]
    for required in ('-O2', '--use_fast_math', '-cudart', 'static', '-std=c++17',
                     '--generate-code=arch=compute_89,code=[compute_89,sm_89]',
                     '-DGGML_CUDA_USE_GRAPHS'):
        if required not in result:
            raise ValueError('pinned arch89/math/graph compile contract differs')
    return result


def link_response(text: str, *, old_object: Path, new_object: Path,
                  old_dll: Path, new_dll: Path, old_implib: Path,
                  new_implib: Path) -> str:
    """Swap exactly one object; retain the v5 link inventory and dependencies."""
    result = text
    for before, after in ((old_object, new_object), (old_dll, new_dll),
                          (old_implib, new_implib)):
        if result.count(str(before)) != 1:
            raise ValueError('one exact CUDA link substitution required: ' + str(before))
        result = result.replace(str(before), str(after), 1)
    old_objects = re.findall(r'^"([^"\r\n]+\.obj)"$', text, re.M | re.I)
    new_objects = re.findall(r'^"([^"\r\n]+\.obj)"$', result, re.M | re.I)
    if len(old_objects) < 2 or len(new_objects) != len(old_objects):
        raise ValueError('complete multi-object CUDA link inventory required')
    if [str(new_object) if x == str(old_object) else x for x in old_objects] != new_objects:
        raise ValueError('an unrelated CUDA object changed')
    return result


def verify_receipts(records):
    for record in records:
        if sha(record['path']) != record['sha256']:
            raise ValueError('frozen input changed: ' + record['path'])


def prepare(out, observer_build, numerical_build):
    out = Path(out).resolve()
    old = Path(observer_build).resolve()
    driver = Path(numerical_build).resolve()
    if out.exists():
        raise ValueError('new private MMID candidate directory required')
    parent = read(old / 'PREPARED.json')
    complete = read(old / 'BUILD_COMPLETE.json')
    numerical = read(driver / 'BUILD_COMPLETE.json')
    if complete.get('compiled') is not True or numerical.get('compiled') is not True:
        raise ValueError('successful frozen observer and numerical driver builds required')
    root = Path(parent['llama_root'])
    original = root / 'ggml/src/ggml-cuda/mmid.cu'
    old_object = next((Path(r['path']) for r in parent['reused_cuda_objects']
                       if Path(r['path']).name.lower() == 'mmid.obj'), None)
    if old_object is None:
        raise ValueError('original MMID object missing from v5 inventory')
    source = original.read_text(encoding='utf-8')
    patched = fence_source(source)
    for name in ('source', 'objects', 'bin', 'lib'):
        (out / name).mkdir(parents=True)
    (out / 'source/mmid.cu').write_text(patched, encoding='utf-8', newline='\n')
    (out / 'mmid-fence.patch').write_text(''.join(difflib.unified_diff(
        source.splitlines(True), patched.splitlines(True),
        fromfile='frozen/mmid.cu', tofile='private/mmid.cu')),
        encoding='utf-8', newline='\n')
    command = compile_command(read(old / 'CUDA_COMPILE.json')['command'],
        old_source=old / 'source/ggml-cuda.cu', old_object=old / 'objects/ggml-cuda.obj',
        old_include=old / 'source', source=out / 'source/mmid.cu',
        object_file=out / 'objects/mmid.obj', include=out / 'source')
    response = link_response((old / 'cuda-link.rsp').read_text(encoding='utf-8'),
        old_object=old_object, new_object=out / 'objects/mmid.obj',
        old_dll=old / 'bin/ggml-cuda.dll', new_dll=out / 'bin/ggml-cuda.dll',
        old_implib=old / 'lib/ggml-cuda.lib', new_implib=out / 'lib/ggml-cuda.lib')
    (out / 'cuda-link.rsp').write_text(response, encoding='utf-8', newline='\n')
    reused = [r for r in parent['reused_cuda_objects'] if Path(r['path']) != old_object]
    reused.append(receipt(old / 'objects/ggml-cuda.obj'))
    tool_paths = [Path(command[0]), Path(command[command.index('-ccbin') + 1]) / 'cl.exe',
                  Path(read(old / 'CUDA_LINK.json')['command'][0])]
    frozen = [receipt(original), receipt(root / 'ggml/src/ggml-cuda/common.cuh'),
              receipt(root / 'build/ggml/src/ggml-cuda/ggml-cuda.vcxproj'),
              receipt(old / 'BUILD_COMPLETE.json'), receipt(old / 'CUDA_COMPILE.json'),
              receipt(old / 'cuda-link.rsp'), receipt(old_object),
              receipt(driver / 'BUILD_COMPLETE.json'), numerical['binary'],
              *complete['binaries'], *reused, *[receipt(p) for p in tool_paths]]
    # The dependencies below are all quoted input paths in the original response;
    # Windows system library names still resolve through the unchanged build env.
    for value in re.findall(r'^"([^"\r\n]+\.lib)"$', response, re.M | re.I):
        if Path(value).is_absolute():
            frozen.append(receipt(value))
    verify_receipts(frozen)
    result = dict(schema='real-moe-mmid-private-runtime-prepared-1',
        prepared_at_utc=datetime.now(timezone.utc).isoformat(),
        observer_build=str(old), numerical_build=str(driver),
        original_mmid_source=receipt(original), replaced_original_object=receipt(old_object),
        frozen_inputs=frozen, reused_cuda_objects=reused,
        sources=[receipt(out / 'source/mmid.cu'), receipt(out / 'cuda-link.rsp'),
                 receipt(out / 'mmid-fence.patch'), receipt(__file__)],
        compile_command=command,
        link_command=[read(old / 'CUDA_LINK.json')['command'][0], '@' + str(out / 'cuda-link.rsp')],
        reused_numerical_driver=numerical['binary'],
        source_changes=['one fullmask __syncwarp before cross-lane shared reads'],
        compiler_math_flags_changed=False, graph_or_fusion_flags_changed=False,
        observer_main_object_rebuilt=False, CPU_or_llama_rebuilt=False,
        compiled=False, hardware_executed=False,
        old_logit_failure_cause_proven=False, timings_qualified_as_bare=False)
    write(out / 'PREPARED.json', result)
    return result


def build(out, env=None):
    out = Path(out).resolve()
    prepared = read(out / 'PREPARED.json')
    authorization = read(out / 'ROOT_BUILD_AUTHORIZATION.json')
    if authorization.get('build_authorized') is not True or authorization.get('prepared_sha256') != sha(out / 'PREPARED.json'):
        raise ValueError('separate prepared-bound root build authorization required')
    if (out / 'BUILD_COMPLETE.json').exists() or (out / 'CUDA_COMPILE.json').exists():
        raise ValueError('candidate build already attempted; no overwrite/retry')
    verify_receipts(prepared['frozen_inputs'] + prepared['sources'])
    run_record(out, 'CUDA_COMPILE', prepared['compile_command'], dict(os.environ) if env is None else env)
    run_record(out, 'CUDA_LINK', prepared['link_command'], dict(os.environ) if env is None else env)
    old = Path(prepared['observer_build'])
    for name in ('ggml-base.dll', 'ggml-cpu.dll', 'ggml.dll', 'llama.dll'):
        shutil.copy2(old / 'bin' / name, out / 'bin' / name)
    native = prepared['reused_numerical_driver']
    shutil.copy2(native['path'], out / 'bin' / Path(native['path']).name)
    verify_receipts(prepared['frozen_inputs'] + prepared['sources'])
    result = dict(schema='real-moe-mmid-private-runtime-build-complete-1',
        prepared=receipt(out / 'PREPARED.json'), authorization=receipt(out / 'ROOT_BUILD_AUTHORIZATION.json'),
        binaries=[receipt(p) for p in sorted((out / 'bin').iterdir())],
        execution_receipts=[receipt(out / (name + '.json')) for name in ('CUDA_COMPILE', 'CUDA_LINK')],
        original_inputs_unchanged=True, compiled=True, hardware_executed=False,
        original_logit_failure_cause_proven=False, timings_qualified_as_bare=False)
    write(out / 'BUILD_COMPLETE.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'build'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--observer-build')
    parser.add_argument('--numerical-build')
    args = parser.parse_args()
    result = prepare(args.out, args.observer_build, args.numerical_build) if args.command == 'prepare' else build(args.out)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
