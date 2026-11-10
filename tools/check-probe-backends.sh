#!/usr/bin/env bash
# Configure and compile both probe projects for both backends.
#
# Why this exists: a Mac cannot be borrowed for CI, and the probe fragments are
# hash-pinned CUDA-shaped sources. The narrow question a Linux or Windows box can
# still answer is whether `-DPROBE_BACKEND=metal` is a *complete* choice -- that
# the project needs no CUDA toolkit, no CUPTI and no CUDART on that path, that
# every CUDA name the fragments use is answered by
# profiling/native/backend/metal/include, and that the cuda path still compiles
# unchanged. This script asks exactly that.
#
# It stops at the link, which is expected to fail: the ggml libraries are empty
# stubs, since this host has one backend's libraries at most. A configure that
# reaches "Generating done" and a build that produces the probe's object file is
# a pass.
#
#   bash tools/check-probe-backends.sh --llama-root <llama.cpp b8705> \
#       [--cuda-root <CUDA toolkit>] [--work <scratch dir>] [--jobs N]
#
# --cuda-root is only needed for the cuda pass; without it that pass is skipped
# and the metal pass alone runs. POSIX only (the stub libraries are `.so`).
set -euo pipefail

usage() {
  sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-2}"
}

here=$(cd "$(dirname "$0")/.." && pwd)
llama_root=""
cuda_root=""
work=""
jobs=""
while [ $# -gt 0 ]; do
  case "$1" in
    --llama-root) llama_root=$2; shift 2 ;;
    --cuda-root) cuda_root=$2; shift 2 ;;
    --work) work=$2; shift 2 ;;
    --jobs) jobs=$2; shift 2 ;;
    -h|--help) usage 0 ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done
[ -n "$llama_root" ] || { echo "--llama-root is required" >&2; usage; }
[ -d "$llama_root/ggml/include" ] || {
  echo "$llama_root holds no ggml/include; pass the llama.cpp tree the probes build against" >&2
  exit 2
}
[ -n "$work" ] || work=$(mktemp -d "${TMPDIR:-/tmp}/probe-backends-XXXXXX")
[ -n "$jobs" ] || jobs=$( (command -v nproc >/dev/null && nproc) || echo 4 )

projects=(
  "depth:profiling/depth_validation/native:depth-probe"
  "e2a:profiling/native/steady_costs:steady-costs-probe"
)

stubs="$work/stubs"
rm -rf "$stubs"
mkdir -p "$stubs/ggml/src/ggml-metal" "$stubs/ggml/src/ggml-cuda" "$stubs/bin"
for name in base cpu metal cuda; do
  : > "$stubs/ggml/src/libggml-$name.so"
done

status=0
report() { printf '\n== %s ==\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*"; status=1; }

objects_in() {
  find "$1" \( -name '*.o' -o -name '*.obj' \) 2>/dev/null | head -1
}

check_backend() {
  local backend=$1
  local cuda_args=()
  local depth_args=()
  if [ "$backend" = "cuda" ]; then
    [ -n "$cuda_root" ] || { report "$backend skipped: no --cuda-root"; return 0; }
    cuda_args=(-DCUDA_ROOT="$cuda_root")
    # CUPTI lives under extras/CUPTI on every toolkit this repository has used.
    [ -d "$cuda_root/extras/CUPTI" ] && depth_args=(-DCUPTI_ROOT="$cuda_root/extras/CUPTI")
  fi
  local row name source target tree
  for row in "${projects[@]}"; do
    IFS=: read -r name source target <<<"$row"
    tree="$work/$name-$backend"
    report "$name probe, $backend"
    local extra=("${cuda_args[@]+"${cuda_args[@]}"}")
    [ "$name" = "depth" ] && extra+=("${depth_args[@]+"${depth_args[@]}"}")
    # Each project names its own third input: the dense column compiles the
    # shipped generated fragments, the e2a column loads a runtime directory.
    case "$name" in
      depth) extra+=(-DSTEADY_GENERATED_DIR="$here/profiling/depth_validation/generated") ;;
      e2a) extra+=(-DRUNTIME_DIR="$stubs/ggml/src") ;;
    esac
    if ! cmake -S "$here/$source" -B "$tree" \
        -DPROBE_BACKEND="$backend" \
        -DLLAMA_ROOT="$llama_root" \
        -DGGML_BUILD_DIR="$stubs" \
        -DCMAKE_BUILD_TYPE=Release \
        "${extra[@]+"${extra[@]}"}" >"$tree.configure.log" 2>&1; then
      tail -20 "$tree.configure.log"
      fail "$name did not configure for $backend"
      continue
    fi
    cmake --build "$tree" --target "$target" -j "$jobs" >"$tree.build.log" 2>&1 || true
    # The link against stub libraries always fails; that is the expected end of
    # this check, not the subject. What is asked is whether the translation unit
    # compiled, so compiler diagnostics are counted and linker ones are not.
    local failures
    failures=$(grep -E 'error:' "$tree.build.log" \
               | grep -vE 'collect2:|ld returned|undefined reference|ld: ' || true)
    if [ -n "$failures" ]; then
      printf '%s\n' "$failures" | head -10
      fail "$name did not compile for $backend"
    elif [ -z "$(objects_in "$tree")" ]; then
      tail -20 "$tree.build.log"
      fail "$name produced no object file for $backend"
    else
      printf 'ok: configured without a toolkit failure and compiled %s\n' "$target"
    fi
  done
}

check_backend metal
check_backend cuda

report "an unknown backend must be refused, not ignored"
if cmake -S "$here/profiling/depth_validation/native" -B "$work/refused" \
    -DPROBE_BACKEND=vulkan -DLLAMA_ROOT="$llama_root" -DGGML_BUILD_DIR="$stubs" \
    >"$work/refused.log" 2>&1; then
  fail "PROBE_BACKEND=vulkan was accepted"
elif grep -q "PROBE_BACKEND is 'vulkan'" "$work/refused.log"; then
  printf 'ok: refused with the value quoted\n'
else
  tail -10 "$work/refused.log"
  fail "the refusal did not name the value"
fi

if [ "$status" -eq 0 ]; then
  printf '\nall probe-backend checks passed\n'
else
  printf '\nsee %s for the full logs\n' "$work"
fi
exit "$status"
