# Controlled strategy and hardware design study

This study uses the existing public compiler and runtime to compare finite,
predeclared strategy candidates. It is an analytical design experiment, not
native inference, a named-device benchmark, or proof of a new search algorithm.

## Frozen questions and scope

1. How do VRAM capacity and host/device bandwidth change the preferred candidate?
2. Does a policy chosen at the reference point remain preferable elsewhere?
3. What is gained by considering execution and storage decisions together?
4. How do starting weight locations and the finite generation length change the
   result?

Two F32 layers have hidden size 6144, intermediate size 16384, 24 query heads,
and 6 KV heads. Dense and MoE use the same geometry. MoE has four experts and
top-2 routing. Its two model-owned traces are hot experts (0,1), and experts
rotating with token position and layer. The policy sees a route only after the
router completes. Prefill always has eight tokens; decode inputs are trace
supplied, not sampled model outputs.

An anonymous CPU+GPU platform has 32 GiB DRAM, CPU effective compute/memory rates
0.6 TFLOP/s and 50 GB/s, GPU rates 8 TFLOP/s and 300 GB/s, CPU/GPU launch costs
500/5000 ns and link latency 1000 ns. These are declared assumptions, not
calibration parameters. Per-task extra workspace is explicitly assumed zero.
Storage uses binary GiB; bandwidth uses decimal GB/s.

## Matrix and candidates

The primary matrix is VRAM `{2,4,12}` GiB × link `{2,8,24}` GB/s, decode D1,
three workloads, and two initial states. The generation-length extension uses
the fixed reference point 12 GiB / 8 GB/s with D8 and D32. No parameter is tuned
after inspecting outcomes.

Dense has eight candidates and each MoE trace has ten:

| ID | Execution |
|---|---|
| e0 | All macro tasks on GPU |
| e1 | All macro tasks on CPU |
| e2 | Dense FFN / MoE experts on CPU, other tasks on GPU |
| e3 | Attention on CPU, other tasks on GPU |
| e4 | MoE only: causal current-cost expert device choice, other tasks on GPU |

Each execution choice combines with s0 (LRU, no prefetch) and s1 (same LRU plus
next-layer prefetch of up to two weights, including Dense weights). Both receive
the same physical VRAM capacity as cache budget; there is no smaller artificial
cache quota. Runtime activation/workspace capacity checks still apply.

This totals 504 primary cases plus 112 generation-length cases. Failed or
initially infeasible cases remain in the declared denominator.

Task partitioning is excluded from this performance campaign: the analytical
facade does not provide independent split-stage cost contracts. Its expression
and semantic correctness belong to the separate IR study. This study does not
pretend that halving a task automatically halves runtime or weight transfers.

## Initial and final state

Cold starts with immutable weights and input in DRAM, and empty KV. The resident
condition places all immutable weights in VRAM at time zero, input in DRAM and
empty KV. Actual VRAM allocations are charged before execution; a case whose
weights exceed VRAM is explicitly infeasible. GPU placement preparation is
outside this declared initial condition. No cold cost is subtracted afterward.

The current Dense layer input has a single initial weight location. Consequently
the resident condition has no free DRAM duplicate: CPU consumers must obtain a
DRAM copy. This is an initial-placement comparison, not a steady-state benchmark
or a policy-specific warmed cache. Every candidate in a group starts identically.

All candidates must deliver the same hidden and KV views to DRAM. Optional cached
weight views may differ; final costs do not force an artificial weight eviction.
The reported latency is finite P8 plus D rounds including required final copies,
not TTFT, TPOT, or sampled tokens/second.

More precisely, intermediate-layer hidden values have no mandatory DRAM barrier.
The CPU-priced trace-next-token gate consumes the previous round's final-layer
hidden, so that hidden must reach DRAM at each round boundary. All KV blocks have
final global DRAM goals; policies can copy them earlier, but their completion is
not a separate compulsory barrier before every following layer. A policy whose
terminal KV resides only on GPU is outside this experiment's boundary. MoE also
declares immutable weight goals at their original backing locations; Dense has
no corresponding explicit weight goal. Each family is compared within its own
fixed contract, not against the other family as an equivalent workload.

## Decision comparisons

The best candidate among all declared feasible pairs is the finite reference.
Execution-only fixes s0; storage-only fixes e0. Stagewise selection first chooses
execution under s0, then chooses storage for that execution, reusing the shared
s0 result. Candidate counts are reported. Joint selection cannot be worse than
its subsets by construction; the experiment quantifies the magnitude and cause
of any gain, not algorithmic superiority.

A second comparison freezes the best candidate selected at 12 GiB / 8 GB/s and
evaluates that same candidate at other points. Infeasible initial states remain
infeasible. No missing latency is imputed.

## Evidence and reproduction

`experiments/paper_design_study.py` generates declarations, exact input cost
tables, compiled IR, causal events, operation/tensor catalogues, policy audits,
metrics, artifact hashes, source manifests, and analysis JSON/CSV. An artifact
audit recomputes latency and transfer payloads directly from saved events and
checks initial weight locations, graph weight bytes, physical peak capacity,
file hashes, matrix coverage and identical within-group workload/hardware.

The runner's audit checks recorded peak capacity and event-derived metrics. The
separate `paper_design_independent_audit.py` reconstructs physical allocations,
live operands, dependencies and final goals from saved catalogues and events.
Its receipt distinguishes those independent checks from runtime success.

Run with the repository Python path configured:

```text
python experiments/paper_design_study.py --mode pilot --workers 2 --out build/dac-evidence-20261023/design-pilot-v1
python experiments/paper_design_study.py --mode main --workers 2 --out build/dac-evidence-20261023/design-main-v1
python experiments/paper_design_study.py --mode long --workers 2 --out build/dac-evidence-20261023/design-long-v1
```

The pilot is an implementation/runtime check and is not pooled with the final
matrix. Output directories must be new. Runtime sources and frozen accuracy
evidence are hashed before and after each complete run. Deterministic cases have
no statistical confidence intervals; native accuracy claims come only from the
separate calibration evidence.
