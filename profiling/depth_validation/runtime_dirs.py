"""The GGML library directories a depth probe was linked against.

Kept apart from `prepare_portable` because the campaign's preflight asks this
question in its own process, before it has built the environment its child
steps run in -- and `prepare_portable` imports the campaign module, numpy and
the model package on the way in.
"""
from __future__ import annotations

from pathlib import Path


def ggml_runtime_dirs(binary: Path) -> list[Path]:
    """Directories holding the GGML libraries the probe imports.

    The probe links the pinned llama.cpp tree's import libraries, and CMake puts
    those under `ggml/src/<config>` while the libraries of the same build go to
    `bin/<config>` (a `.dll` beside its `.lib` on Windows, a `.so` on Linux). So
    the directory that held the import library does not hold the runtime library,
    and a runtime list built from the probe's own directory loads nothing: the
    child exits with `STATUS_DLL_NOT_FOUND` (0xC0000135) on Windows and with a
    loader error naming the missing `.so` on Linux, and every process in the
    profile fails. The probe's own build tree names the GGML build, so the
    library directory is derived from it rather than guessed or hard-coded.
    """
    binary = Path(binary)
    for parent in binary.parents:
        cache = parent / 'CMakeCache.txt'
        if not cache.is_file():
            continue
        for line in cache.read_text(encoding='utf-8', errors='replace').splitlines():
            if line.startswith('GGML_BUILD_DIR:'):
                build = Path(line.split('=', 1)[1].strip())
                candidates = [build / 'bin' / 'Release', build / 'bin' / 'Debug',
                              build / 'bin', build / 'lib', build / 'ggml' / 'src']
                return [d for d in candidates if d.is_dir()]
        return []
    return []
