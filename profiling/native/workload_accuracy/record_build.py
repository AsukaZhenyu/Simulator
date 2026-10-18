"""Post-build binding; never changes the existing GGML tree."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('binary', 'source-dir', 'ggml-source', 'ggml-build', 'compiler', 'compiler-version'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    binary, source = Path(args.binary).resolve(), Path(args.source_dir).resolve()
    sources = {str(p): sha(p) for p in sorted(source.glob('*')) if p.suffix in ('.cpp', '.py', '.txt')}
    headers = {str(p): sha(p) for p in sorted((Path(args.ggml_source)/'include').glob('*.h'))}
    runtime = {str(p.resolve()): sha(p) for p in sorted((Path(args.ggml_build)/'bin'/'Release').glob('ggml*.dll'))}
    libraries = {str(p.resolve()): sha(p) for p in Path(args.ggml_build).glob('ggml/src/**/Release/ggml*.lib')}
    data = dict(schema='workload_accuracy_build/1', binary=str(binary), binary_sha256=sha(binary),
                sources=sources, ggml_headers=headers, runtime=runtime, link_libraries=libraries,
                compiler=args.compiler, compiler_version=args.compiler_version,
                cmake_cache_sha256=sha(binary.parent.parent/'CMakeCache.txt'))
    binary.with_suffix('.build.json').write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
