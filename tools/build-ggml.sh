#!/usr/bin/env sh
# Build the ggml libraries that the depth and e2a probes link, on this host.
#
#   tools/build-ggml.sh --llama-root <b8705 checkout> --build-dir <dir> \
#       [--cuda-root <toolkit>] [--arch <compute capability>] [--jobs <n>]
#
# Why a script for four CMake lines: on Linux, CMake finds `nvcc` on PATH and
# nowhere else. A toolkit under /usr/local/cuda-12.4 without a profile edit is
# not on PATH, and the configure then stops with "No CMAKE_CUDA_COMPILER could
# be found." -- a message that names neither nvcc nor the toolkit. The other
# common half-hour is naming one target: `--target ggml-cuda` alone does not
# build `ggml-base`, and the probe's find_library then fails as
# "Could not find GGML_base". This script does both correctly.
#
# The result is four shared libraries in <build-dir>/bin/. Pass their parent as
# --ggml-build-dir to profiling.depth_validation.build_probe and
# profiling.dac_validation.build_probe.
set -eu

usage() {
    sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
}

LLAMA_ROOT=""
BUILD_DIR=""
CUDA_ROOT=""
ARCH=""
JOBS=""

while [ $# -gt 0 ]; do
    case "$1" in
        --llama-root) LLAMA_ROOT="$2"; shift 2 ;;
        --build-dir)  BUILD_DIR="$2";  shift 2 ;;
        --cuda-root)  CUDA_ROOT="$2";  shift 2 ;;
        --arch)       ARCH="$2";       shift 2 ;;
        --jobs)       JOBS="$2";       shift 2 ;;
        -h|--help)    usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [ -z "$LLAMA_ROOT" ] || [ -z "$BUILD_DIR" ]; then
    echo "--llama-root and --build-dir are required" >&2
    usage >&2
    exit 2
fi
if [ ! -f "$LLAMA_ROOT/ggml/include/ggml.h" ]; then
    echo "$LLAMA_ROOT does not look like a llama.cpp checkout (no ggml/include/ggml.h)" >&2
    exit 2
fi

# Toolkit: named, or the one whose nvcc is on PATH, or the conventional place.
if [ -z "$CUDA_ROOT" ]; then
    if command -v nvcc >/dev/null 2>&1; then
        CUDA_ROOT=$(dirname "$(dirname "$(command -v nvcc)")")
    elif [ -d /usr/local/cuda ]; then
        CUDA_ROOT=/usr/local/cuda
    else
        echo "no CUDA toolkit found: pass --cuda-root <toolkit>" >&2
        exit 2
    fi
fi
if [ ! -x "$CUDA_ROOT/bin/nvcc" ]; then
    echo "$CUDA_ROOT holds no bin/nvcc; pass --cuda-root <toolkit>" >&2
    exit 2
fi

# Both of these are what make the configure find the compiler at all.
PATH="$CUDA_ROOT/bin:$PATH"
CUDACXX="$CUDA_ROOT/bin/nvcc"
export PATH CUDACXX

# The device answers the architecture question unless the caller already did.
if [ -z "$ARCH" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
               | head -n 1 | tr -d '. ')
    fi
fi
if [ -z "$ARCH" ]; then
    echo "note: no compute capability detected; CMake will use the toolkit's own list" >&2
fi

if [ -z "$JOBS" ]; then
    JOBS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
fi

set -- -S "$LLAMA_ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF \
    -DCUDAToolkit_ROOT="$CUDA_ROOT"
if [ -n "$ARCH" ]; then
    set -- "$@" "-DCMAKE_CUDA_ARCHITECTURES=$ARCH"
fi

echo "toolkit: $CUDA_ROOT"
echo "arch:    ${ARCH:-cmake default}"
cmake "$@"
# All four: the probes link ggml, ggml-base, ggml-cpu and ggml-cuda.
cmake --build "$BUILD_DIR" --target ggml ggml-base ggml-cpu ggml-cuda -j "$JOBS"

missing=""
for name in libggml.so libggml-base.so libggml-cpu.so libggml-cuda.so; do
    if [ ! -e "$BUILD_DIR/bin/$name" ]; then
        missing="$missing $name"
    fi
done
if [ -n "$missing" ]; then
    echo "the build reported success but $BUILD_DIR/bin lacks:$missing" >&2
    echo "(the probes' find_library needs the unversioned names in bin/)" >&2
    exit 1
fi
echo "ggml libraries are in $BUILD_DIR/bin; pass --ggml-build-dir $BUILD_DIR"
