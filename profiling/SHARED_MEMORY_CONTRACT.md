# Shared-memory profiling boundaries

Shared storage changes allocation identity and physical capacity accounting.
It does not make compute memory traffic free or turn an old CPU/CUDA copy
measurement into a shared-buffer measurement.

`llm_infer_profile.memory_contract.validate_memory_context()` validates explicit
metadata for future probes. It collects no data and chooses no vendor backend.

```python
from llm_infer_profile.memory_contract import validate_memory_context

context = validate_memory_context(dict(
    memory_mode="shared",
    allocation_kind="shared_buffer",
    timing_boundary="shared_visibility",
    timing_source="unknown",
    duration_ns=None,
    bytes_transferred=0,
))
```

- `memory_mode`: physical storage arrangement, `separate` or `shared`.
- `allocation_kind`: `shared_buffer` means the same allocation identity;
  `separate_buffers` also applies to a real copy inside a unified physical pool.
- `timing_boundary`: `actual_copy`, `shared_visibility`, `synchronize`, or
  `compute`. A synchronization measurement can be nonzero with zero copy bytes.
- `timing_source`: `measured`, `declared`, or `unknown`. Unknown duration stays
  `None`; measured duration requires an explicit `evidence` path or identifier.

Existing probe results retain their original allocation and timing boundary.
Historical records without this metadata do not prove shared-buffer handoff
latency. The simulator's current zero-time `SHARE` is an explicitly declared
visibility abstraction, not a measurement. A user can represent a measured
handoff/synchronization cost with an explicitly priced dependent task; the
library does not infer or subtract it.

CPU/GPU compute formulas and existing profiles are unchanged. They continue to
price the declared effective memory bandwidth. Shared CPU/GPU memory-bandwidth
contention is not inferred from isolated compute times; a separately declared
resource/service contract or new measurements are required before claiming
concurrent unified-memory accuracy. Do not charge an observed contention cost
again if it is already inside a measured task duration.
