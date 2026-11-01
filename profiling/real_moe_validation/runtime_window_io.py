"""Small collector-only helpers, with bounded complete-file hashing.

Importing a timing collector must not initialize NumPy/BLAS worker pools.
Every hash still covers the complete file; one reusable 1 MiB buffer bounds
temporary allocation even for the real model and large CUDA DLL.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path


# Request placements to their campaign job-label prefix. The third placement
# keeps attention on the host; its label never reuses 'hybrid', so an earlier
# two-placement cohort can never be silently overwritten.
#
# The fourth placement is storage offload: expert weights stay in host memory and
# the CUDA scheduler copies them up for evaluation instead of computing on the
# host. The three original placements compute where the tensor lives; this one
# separates residency from execution, which is what the storage-staging systems
# in the related work actually do. The shipped observer binary accepts only the
# three original placement strings, so its request file carries the hybrid string
# while the analysis key carries this label; the difference is the scheduler's
# op-offload threshold, declared per campaign and recorded on every execution.
PLACEMENT_LABELS = (('cpu', 'cpu'), ('cpu-experts-gpu-other', 'hybrid'),
                    ('cpu-attn-gpu-other', 'attn'), ('cpu-experts-upload', 'upload'))
LABELS = tuple(label for _, label in PLACEMENT_LABELS)
# Cohorts prepared before the storage-offload placement existed declare no label
# list, and must keep reproducing exactly the placements they were frozen on.
LEGACY_LABELS = ('cpu', 'hybrid', 'attn')
PROBE_PLACEMENT = {'cpu-experts-upload': 'cpu-experts-gpu-other'}
LABEL_PLACEMENT = dict((label, placement) for placement, label in PLACEMENT_LABELS)
LABEL_FOR_PLACEMENT = dict((placement, label) for placement, label in PLACEMENT_LABELS)
# Placement -> extra child environment. 'upload' forces the CUDA scheduler's
# host-resident op offload at every batch size (default threshold is 32, which
# already uploads during Prefill but computes on the host during Decode).
PLACEMENT_ENVIRONMENT = {'upload': {'GGML_OP_OFFLOAD_MIN_BATCH': '1'}}

# Runtime policy is a declared coordinate, never an ambient shell choice. A cohort
# that measures several placements must use one policy for all of them, otherwise
# the comparison mixes placement with runtime configuration. The observer binary
# applies the declared policy to its own process, so no shell variable is exported.
GRAPH_POLICIES = ('runtime_default_enabled', 'runtime_graphs_disabled_declared')


def require(ok, text):
    if not ok:
        raise ValueError(text)


def file_sha(path):
    digest = hashlib.sha256()
    block = bytearray(1024 * 1024)
    view = memoryview(block)
    with Path(path).open('rb', buffering=0) as stream:
        while True:
            count = stream.readinto(block)
            if not count:
                break
            digest.update(view[:count])
    return digest.hexdigest()


sha = file_sha


def receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha(path))


def child_environment(runtime_dirs, request):
    require(request['cuda_graph_policy'] in GRAPH_POLICIES
        and request['tf32_override'] == '0', 'unknown explicit native flags')
    environment = dict(os.environ)
    environment['PATH'] = os.pathsep.join([*map(str, runtime_dirs), environment.get('PATH', '')])
    environment.pop('GGML_CUDA_DISABLE_GRAPHS', None)
    environment['NVIDIA_TF32_OVERRIDE'] = '0'
    return environment


class _Memory(ctypes.Structure):
    _fields_ = [('length', ctypes.c_uint32), ('load', ctypes.c_uint32)] + [
        (name, ctypes.c_uint64) for name in (
            'total', 'available', 'page_total', 'page_available',
            'virtual_total', 'virtual_available', 'ext')]


def available():
    state = _Memory()
    state.length = ctypes.sizeof(state)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
        raise OSError('GlobalMemoryStatusEx failed')
    return state.available
