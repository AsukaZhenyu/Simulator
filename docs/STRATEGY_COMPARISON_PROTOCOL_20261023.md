# Strategy comparison protocol

Date: 2026-10-23. Scope: one CPU and one GPU with explicit DRAM/VRAM.
The first campaign is a finite analytical pilot. It verifies the comparison
protocol and exposes candidate behavior; it is not a measured speedup study.
Existing accuracy evidence and modeling/profiling parameters remain frozen.

## Questions and primary outcomes

1. Does the declared strategy complete the same workload and output goals?
2. Among completed runs, which strategy takes less simulated elapsed time?
3. What changes in physical traffic, capacity pressure and execution overlap?
4. Does changing storage alter the benefit or ordering of execution choices?

Report these outcomes separately. Do not merge them into an arbitrary weighted
score. A failure, missing cost or unsupported strategy must remain visible.

| Metric | Definition | Interpretation |
|---|---|---|
| Completion | Valid evaluation and `goal_reached` | Required before latency ranking |
| Complete-work latency | Makespan of the complete declared workload, ns | Primary performance outcome |
| Reference speedup | `T_reference / T_candidate` | Only when both complete with positive time |
| Transfer traffic | H2D, D2H, SSD reads/writes, plus actual COPY traffic by tensor role | Physical bytes, not parameter size estimates |
| Peak physical memory | Peak DRAM and VRAM separately | Includes modeled weights, KV, activations and declared scratch |
| Weight reloads | Repeated copies of the same immutable weight ID on the same path | Diagnose reuse without inventing cache hits |
| Timeline coverage | Union of CPU, GPU and transfer event intervals | Modeled event occupation, not measured device utilization |
| Timeline overlap | CPU/GPU intersection and compute/transfer intersection | Explains permitted overlap, not automatically a speedup |

Compute/COPY interval durations must not be added to reconstruct latency when
they overlap. Uncovered time is not an attributed operating-system or scheduling
wait. Weight evictions are distinct from freeing dead KV or intermediates.

Unsupported/invalid input, modeled capacity rejection, unfinished execution and
decision-budget exhaustion are different statuses. Partial execution time must
never rank ahead of a completed run. Completion fractions use the predeclared
scenario denominator, with each failure reason retained.

## Paired execution/storage comparisons

Use one fixed hardware configuration, model, trace, resource state, initial
placement and final goal set. Declare execution choices before running:

| ID | Execution placement |
|---|---|
| e0 | GPU macros |
| e1 | CPU macros |
| e2 | CPU FFN/expert FFN, remaining macros on GPU |
| e3 | CPU Attention, remaining macros on GPU |

For each execution choice, use the same LRU weight-payload budget, release rule
and physical capacities. Change only next-layer prefetch:

| ID | Storage rule |
|---|---|
| s0 | Bounded LRU cache, no next-layer prefetch |
| s1 | Same bounded LRU cache, next-layer prefetch enabled |

All eight combinations run. This pilot isolates prefetch as one storage
dimension; it does not establish coverage of all cache/transport policies or
evaluate task partitioning, request priorities or adaptive experts yet.
Enumerating a finite catalogue is not a new search algorithm. Report the best
**among the declared candidates**.

For each nonreference execution `e`, compare the four measured simulation cells:

`T00=T(e0,s0)`, `T10=T(e,s0)`, `T01=T(e0,s1)`, `T11=T(e,s1)`.

- Joint gain relative to the better single change:
  `(min(T10,T01)-T11) / min(T10,T01)`.
- Nonadditive time saving: `T10+T01-T00-T11`. Positive means the joint reduction
  exceeds the sum of the two separate reductions on this scale; negative means
  antagonism. Neither sign is assumed in advance.
- Execution ordering reversal: whether `e` versus `e0` changes strict order
  between `s0` and `s1`. Ties are not counted as reversals.

Compute these only when all four runs complete. They quantify interaction in
these particular cells; a larger catalogue or budget does not demonstrate a
better search algorithm.

## First pilot and subsequent extensions

The executable pilot reuses the existing analytical builder: two layers, H64,
I192, F32, Prefill8 plus Decode2. MoE uses four experts, top2 and a declared
rotating-pair trace. Hardware is fixed: CPU100 GFLOP/s and20 GB/s, GPU1 TFLOP/s
and100 GB/s, a shared8 GB/s DRAM/VRAM path, DRAM128 MiB, VRAM512 KiB and a
192 KiB weight-cache budget. Launch/transfer overheads are the predeclared
analytical inputs in `experiments/hardware_sensitivity.py`. Zero scratch is an
explicit idealized declaration, not a measured fact. No weights begin on SSD.
All weights begin in DRAM, KV is empty, and mandatory hidden/KV goals are in DRAM.
Optional terminal weight copies may differ; no artificial final flush is added.
The timer includes modeled required transfers and execution overheads, and
excludes process startup, model download and profiling.

Run from the source checkout:

```bash
python experiments/strategy_comparison.py --out build/strategy-comparison/pilot
```

Every attempted candidate retains generated inputs, compiled IR, exact events,
summary, hashes and costs. The protocol is saved before execution. A missing
completion marker means an incomplete campaign. Repeating a deterministic
simulation is a reproducibility check, not independent statistical evidence.

Subsequent predeclared experiments should add Prefill-heavy and Decode-heavy
workloads, high/low expert reuse, same-budget cache/eviction choices, adaptive
expert placement, partitions and request priorities in separate controlled
comparisons. These extensions must be declared before interpreting their results.
Representative measured costs can replace analytical inputs only where the
exact shape, backend and timing contract are covered; do not interpolate or
claim that prior single-layer calibration validates all new strategies.

## Metrics deferred until their boundaries are implemented

- First modeled output/Prefill completion and consecutive Decode-output gaps
  need explicit per-request output boundaries. They are proxies, not validated
  serving TTFT/TPOT: sampling, LM head and token delivery are not executed.
- Per-request completion distributions and finite-trace throughput are useful
  after adding multiple arrivals. A tiny deterministic trace does not support
  a meaningful P99 or a production serving-capacity claim.
- Demand cache hit rate requires recording residency at the first demand,
  before any demand load. READY inputs at compute start do not establish hits.
- Critical-path wait attribution needs causal readiness/resource records.
  ADVANCE counts and copy-only time do not uniquely identify transfer stalls.
- Energy and economic benefit require separate power/cost models and are not
  inferred from latency or event occupation.
