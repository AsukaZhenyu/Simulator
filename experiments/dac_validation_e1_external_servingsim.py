"""Tiny native LLMServingSim checks; run with the pinned source as cwd."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from serving.core.config_builder import get_device
from serving.core.block_pool import BlockPool, Device


placement = {
    "default": {"weights":"CPU:0", "kv_loc":"NPU:0", "kv_evict_loc":"CPU:0"},
    "block": [{"weights":"NPU:0", "kv_loc":"CPU:0"}],
    "layer": {"q_proj":{"weights":"CXL:0", "kv_loc":"CXL:0", "kv_evict_loc":"CXL:0"}},
}
cases = [
    (9, "absent", "weights", "CPU:0"),
    (0, "absent", "weights", "NPU:0"),
    (0, "q_proj", "weights", "CXL:0"),
    (0, "q_proj", "kv_loc", "CXL:0"),
    (0, "q_proj", "kv_evict_loc", "CXL:0"),
    (0, "absent", "kv_evict_loc", "CPU:0"),
]
results = []
for block, layer, kind, expected in cases:
    actual = get_device(placement, block, layer, kind)
    assert actual == expected, (block, layer, kind, actual, expected)
    results.append(dict(block=block, layer=layer, kind=kind, expected=expected, actual=actual))
try:
    get_device(placement, 0, "q_proj", "invalid")
except ValueError as error:
    invalid_kind_rejected = str(error)
else:
    raise AssertionError("Native invalid-kind validation did not reject")

# Native physical pool API, 3 blocks of 32 B, no fabricated timing callbacks.
pool = BlockPool(Device.CPU, num_blocks=3, block_size=1, bytes_per_block=32,
                 enable_caching=False)
blocks = pool.get_new_blocks(2)
assert pool.get_num_free_blocks() == 1
assert pool.used_bytes() == 64
pool.touch(blocks[:1])
pool.free_blocks(blocks)
assert pool.get_num_free_blocks() == 2
assert pool.used_bytes() == 32
pool.free_blocks(blocks[:1])
assert pool.get_num_free_blocks() == 3
assert pool.used_bytes() == 0
try:
    pool.get_new_blocks(4)
except (ValueError, RuntimeError) as error:
    overflow_rejected = str(error)
else:
    raise AssertionError("Native capacity check did not reject")

print("E1_NATIVE_JSON=" + json.dumps({
    "scope":"native placement resolver and per-tier capacity/refcount component",
    "placement_input":placement, "placement_results":results,
    "invalid_kind_rejected":invalid_kind_rejected,
    "pool_input":{"device":"CPU", "num_blocks":3, "block_size":1, "bytes_per_block":32, "enable_caching":False},
    "pool_checks":{"allocate_two_bytes":64, "shared_after_one_release_bytes":32,
                   "released_all_bytes":0, "overflow_rejected":overflow_rejected},
    "placement_checks_passed":len(results), "whole_server_executed":False,
    "core_modifications":0,
}))
