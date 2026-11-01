# Shared physical memory

The same CPU/GPU compute formulas and tensor runtime support two storage
arrangements: `separate` (the existing default) and `shared`. Shared mode adds
allocation identity and capacity semantics; it does not simulate a new compute
architecture or select a real-platform profile.

## Public Python API

```python
from llm_simulator import (
    AnalyticalCosts, Cache, Device, Hardware, Model, Scheduling, Simulator, Workload,
)

hardware = Hardware.shared_memory(
    cpu=Device(100e9, 20e9, launch_overhead_ns=500),
    gpu=Device(1e12, 100e9, launch_overhead_ns=2000),
    capacity_bytes=64 * 1024**2,
    runtime_reserved_bytes=0,
    alignment_bytes=1,
)
simulator = Simulator(
    model=Model.dense(layers=2, hidden_size=16, intermediate_size=32,
                      heads=4, kv_heads=2),
    hardware=hardware,
    workload=Workload(prefill_tokens=3, decode_steps=2),
    cost_model=AnalyticalCosts(workspace_bytes=0),
)
with simulator.compile(scheduling=Scheduling(
    default_device="gpu", release="last_use",
    cache=Cache(rule="lru", budget_bytes=1024**2),
)) as compiled:
    result = compiled.run()
    assert result.success, result.summary()
    print(result.summary()["physical_memory_peaks_bytes"])
```

These parameters are fictional, uncalibrated inputs. Run the complete example:

```bash
python examples/library/shared_memory.py
```

The ordinary `Hardware(cpu=..., gpu=..., dram_capacity_bytes=...,
vram_capacity_bytes=..., transfer_bandwidth_bytes_per_second=...)` constructor
continues to select separate storage. Its compute costs, transfer prices and
capacity constraints retain their original meaning.

## Physical pool and logical access views

| Property | Separate storage | Shared storage |
|---|---|---|
| Physical pools | DRAM and VRAM | One shared pool |
| CPU/GPU locations in IR | `dram`, `vram` | The same names are access views |
| Same tensor accessible by both devices | Independent allocations | May reference one allocation |
| Expose an existing shared allocation | Not available | `SHARE`, zero payload movement |
| Explicit `COPY` | Independent destination allocation | Still an independent destination allocation |
| Capacity peak | Each pool reported independently | `peak_shared_bytes`, counted once |

The compatibility `peak_dram_bytes` and `peak_vram_bytes` fields both expose
the shared pool usage in shared mode. **Do not add them.** Use
`physical_memory_peaks_bytes={"shared": ...}` for physical capacity reporting.
State artifacts preserve allocation references so equal views and independent
copies can be distinguished during inspection and checkpoint replay.

Examples: one 10 GiB allocation shared by CPU/GPU fits a 16 GiB pool; two
distinct 10 GiB allocations do not. A real copy inside the pool consumes a new
allocation, even though both devices can access the pool. Workspace and static
runtime reservation also consume this one physical capacity.

## Data preparation, lifetime, and cache rules

The compiler keeps the existing high-level preparation and placement API. In
shared mode it prepares a missing device view with a `SHARE` action when the
source is an accessible, completed allocation. Producer dependencies, P/D
handoff completion, in-flight readers, format transforms and source validity
remain enforced. Shared visibility does not bypass a producer or manufacture a
recoverable backup.

Dropping one view can release **zero physical bytes** when another view or
reader still retains the allocation. Reclaim decisions use actual released
bytes. An SSD snapshot or independently allocated copy can be a backup; another
view of the same allocation is not an independent backup.

The existing GPU weight-cache budget is retained as an explicit **GPU access
view payload quota**. It is a strategy constraint, separate from shared physical
capacity. Removing a cached GPU view may change this quota while freeing no
physical memory. Prefetching already shared weights creates no H2D traffic;
prefetch from SSD and explicitly requested real copies retain their costs.

Shared capacity fluctuations use `EnvironmentEvent(shared_external_bytes=...)`
or the corresponding advanced environment trace field. External usage is charged
once to the physical pool. Do not independently charge one application to both
DRAM and VRAM aliases. Existing separate-mode external-usage inputs are retained.

## Costs and measurement boundaries

The computation formula is unchanged. Effective compute throughput and memory
bandwidth still determine the duration of each declared macro; memory accesses
inside a shared pool are not free. A real copy retains its positive transfer
price and needs an explicit route contract. No infinite-bandwidth value or fake
zero-duration COPY entry is required for shared visibility.

`SHARE` is a zero-time visibility abstraction, not a measured platform-specific
synchronization cost. An explicit priced dependent task can represent a known
handoff or conversion overhead. Missing native measurements stay unknown.
See the [profiling memory contract](../profiling/SHARED_MEMORY_CONTRACT.md).

This implementation does not automatically infer concurrent CPU/GPU shared
memory-bandwidth contention, cache coherence traffic, page migration, NUMA,
page faults or vendor runtime behavior. Existing declared service and resource
inputs are not a new hardware calibration. It does not establish accuracy on
Apple Silicon, Jetson, Spark, an APU or a mobile SoC, and adds no NPU backend.

## Verification and evidence

Focused checks cover allocation identity, a shared capacity limit, real copies,
workspace, reader locks, eviction/reclaim, KV/P/D handoff, request lifetimes,
checkpoint replay, services, external-memory events and public API/artifacts.
The original 352 strategy combinations are rerun against frozen baseline
outcomes and event hashes. A separate 104-combination matrix exercises shared
Dense/MoE Prefill/Decode paths. The dated report records actual completion and
audit status; this guide does not imply those campaigns passed merely because
the code is present.

```bash
python experiments/shared_storage_report.py --require-complete
```

The report generator reads `build/shared-storage-20261023/` and writes the
Chinese report under the workspace's `paper-write/第七教学周/experiment-reports/`.
Missing evidence is reported as unknown, never as a passing check.

The [2026-10-23 acceptance receipt](data/shared-storage-acceptance-20261023.json)
records the completed verification: all 352 separate-memory outcomes and event
hashes match the frozen baseline; all 104 shared-memory combinations pass an
independent replay audit; all 70 focused checks pass. Full modeling, mapping
and public API suites also pass, with eight optional checks skipped. Counts in
the receipt include repeated execution of focused tests inside the full suites.
These results establish the declared execution/storage semantics, not measured
accuracy on a particular unified-memory device.
