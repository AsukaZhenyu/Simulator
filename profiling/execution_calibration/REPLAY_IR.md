# Exact measured components through the fixed tensor IR

`replay_ir.py` is an offline bridge from a completed, verified
`execution-transform-context-probe` experiment to the existing tensor compiler
and Modeling runtime. It runs no hardware and performs no search or fitting.

```powershell
python profiling/execution_calibration/replay_ir.py `
  --experiment <completed-context-experiment-directory> `
  --output <new-replay-directory>
```

For the distinct repeated completed-work batch protocol, use an explicit flag:

```powershell
python profiling/execution_calibration/replay_ir.py --batch `
  --experiment <completed-batch-experiment-directory> `
  --output <new-replay-directory> `
  --host-interference-note "Describe the actual collection conditions here."
```

Batch mode first calls `../batched_calibration/verify_batch.py`. Every physical
signature records the batch repetition count, raw-total unit, explicit
normalization and derived per-invocation-mean unit. Repeated means cannot be
used to claim low isolated-request jitter. If no host note is supplied, the
report explicitly says host activity was not controlled or fully observed.

The source experiment must retain its native raw records, source snapshots,
manifest, pre-heldout frozen component model, and original statistical report.
The existing context verifier runs first. Missing evidence, an altered source,
a changed command, a mismatched materialization contract, an absent exact
component, or a changed profile/heldout binding causes rejection. Original
measurement evidence is hashed again after replay and is never rewritten.

## What is executed

Python builds an explicit materialized tensor graph and a `strategy-ir-0.1`
fixed program. `tensor_mapping.strategy_compiler.compile_strategy` lowers that
IR to actions and waits; `run_strategy` executes them using Modeling's existing
resource, dependency, transfer and state rules.

The supported paths have the exact native resident boundary:

- GPU compact F32 FFN: gate, up, SwiGLU, down, and output D2H.
- CPU/GPU intermediate-axis FFN: two independent sequential stage chains,
  followed by GPU partial-output D2H and a CPU sum. CPU and GPU branch starts
  and stage completion events are explicit. There is no norm or residual.
- GPU post-RoPE Attention: QK, causal softmax, PV with KV append outputs,
  sigmoid gate/O projection/residual, then output D2H.
- Hybrid Attention: CPU QK/softmax, an explicit probability H2D, and the same
  GPU PV/output/D2H chain.

Gate and up are serialized within each branch because that is what the native
probe executes. The fixed program does not exploit an extra overlap merely
because the mathematical graph would allow it. Parallel completion ordering is
derived only from the frozen component prices, and the actual runtime still
checks dependencies, device slots and movement readiness.

Every compute or copy cost points to the exact profile metric and the original
profile-process raw SHA256. No strategy total or heldout duration can be used as
a component price. Half-nanosecond medians are rounded upward to the integer
nanoseconds required by the runtime; the report preserves both values. The
runtime makespan must equal the predeclared component expression evaluated with
those same integer prices.

## Explicit exclusions

This bridge exercises the **fixed tensor IR compiler and runtime**. It does not
exercise `ModelPlan`'s AttentionStages or FFNPartition semantic lowering. Those
transforms currently have different operation boundaries: the Dense transform
includes normalization and residual stages that the compact native FFN omits.
Passing a fixed-IR replay must not be reported as validating those transforms,
the complete Python frontend, or a full Dense/MoE/multilayer/multiround LLM.

The older isolated preload probe measures three ordered weight transfers as one
aggregate component. The runtime represents individual tensor copies. The
bridge therefore marks those preload paths **unsupported** instead of dividing the
aggregate timing, inventing a contiguous packed tensor, or charging one copy
while treating the other two as free. To support them, independently measure
the gate/up/down H2D actions or introduce and independently validate a grouped
ordered-copy action.

The new batch probe independently measures each gate/up/down H2D operation.
With `--batch`, the bridge replays three COPY actions in that native order and
charges their separate frozen component prices before starting the CPU/GPU
branches. The aggregate preload metric remains validation data, never a source
from which per-weight prices are divided.

There is a deliberate, recorded lifecycle distinction: the native probe copies
into already allocated GPU buffers on every invocation; the immutable runtime
represents a fresh required logical weight version whose GPU copy starts absent.
This enables a forced charged COPY without inventing an overwrite instruction.
Native allocation is outside the timing boundary. The replay checks exact
per-weight payloads, costs and ordering; it does not validate physical buffer
reuse, allocation/overwrite lifecycle or memory capacity. Each preload binding
records `preload_buffer_lifecycle_equivalence_validated: false`.

The ideal resident expression is replayed. The separate empty-worker handshake
model remains a diagnostic alternative; it is not reinterpreted as a physical
compute operation or fitted into this replay.

## Timing acceptance is separate from capacity acceptance

The measured CUDA `workspace_bytes` remains `null` in every physical signature
and binding. The current runtime requires an integer workspace reservation.
For this **timing-only experiment**, it receives the recorded known
GGML-allocator-plus-cplan lower bound, with artificial, nonbinding `2**60`-byte
DRAM and VRAM capacities. These values are explicitly assumptions, not measured
total workspace or the hardware's memory budget. The simulator's resulting
allocation peaks are not physically validated.

`scenario.TIMING-ONLY.json` is diagnostic evidence and **must not** be installed
as an accepted memory-feasibility cost profile. Physical capacity acceptance
always remains false. A more complete performance interface would keep timing
and partial/unknown workspace contracts separate without using a mandatory
integer cost field; this bridge does not silently change that core contract.

Original CV, cross-process stability and heldout-error quality gates remain
unchanged. Correct compiler execution does not make an unstable hardware
measurement acceptable. The command returns success when the audit and replay
complete; `report.json` separately reports latency acceptance, unsupported
paths and the unvalidated capacity scope.

## Evidence and tests

Each path emits the physical binding, workload, timing-only scenario, original
strategy, lowered program, action sequence, state snapshots and event timeline.
The root report records the frozen component-model SHA256, all preserved source
hashes and the simulator implementation fingerprint.

```powershell
python -B -m unittest discover `
  -s profiling/execution_calibration/tests -p test_replay_ir.py -v
```

These hardware-independent tests check parallel branch timing, explicit join
and transfer costs, Attention KV ports, unknown workspace preservation, and
rejection of missing components, validation-set costs, wrong profile sources,
changed physical contracts and unsupported aggregate copies.
