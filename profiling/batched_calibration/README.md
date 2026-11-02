# Repeated completed-work timing and offline verification

This protocol measures generated F32 compact FFN and post-RoPE Attention paths.
Each native timing sample is the **total** wall time of an explicit batch of
completed invocations. The runner retains those totals and separately derives
`round(batch_total_ns / batch_repetitions)` for the repeated-execution model.
These are batch means, not isolated-request latency samples.

`suite.json` declares shapes, batch repetitions, samples, independent profile
and heldout seeds, process repeats and quality gates before execution. No heldout
strategy total is used as a component price. The native generator/build record
binds the original graph/math/precision contract, added timing wrapper and
independent gate/up/down H2D probes to the executable hash.

Run from the Simulator checkout, supplying the actual matching binary/runtime
paths. A native build record and source manifest must exist beside the binary.

```powershell
$env:GGML_CALIBRATION_GPU_DISPATCH_CPU_ID = '2'
python profiling/batched_calibration/run.py `
  --suite profiling/batched_calibration/suite.json `
  --binary <execution-transform-batch-probe.exe> `
  --runtime-dir <matching-GGML-runtime-directory> `
  --runtime-dir <CUDA-runtime-directory> `
  --timeout-s 240 --output <new-experiment-directory>
```

The shared profiling lock prevents simultaneous native measurement campaigns.
It does not prevent unrelated applications or software tests from consuming
resources. Record known host interference explicitly; do not call a campaign
host-idle merely because the lock was acquired. The protocol does not close
user applications, lock device clocks or change the power plan.

## Verify without executing hardware

```powershell
python profiling/batched_calibration/verify_batch.py `
  --experiment <completed-experiment-directory> `
  --output <new-verification-file.json>
```

Verification checks raw-record SHA256, command/geometry/environment binding,
native numeric/precision/affinity/workspace declarations, untouched batch totals,
exact normalization, independent-process coverage, source snapshots and current
source/binary/runtime hashes, generated-source/build binding, and the model hash
frozen before each heldout process. It reconstructs the profile-only costs and
all original statistical quality gates. A small error alone does not pass.

Bound assets must remain available and unchanged for this strict local audit.
Changing the implementation or moving/removing those assets requires an explicit
new provenance workflow; the verifier never silently replaces missing evidence.
The initial protocol records an environment snapshot before collection; it does
not establish a continuously observed host state or a post-run device snapshot.

To replay the verified prices through the tensor IR compiler and Modeling:

```powershell
python profiling/execution_calibration/replay_ir.py --batch `
  --experiment <completed-experiment-directory> `
  --output <new-binding-directory> `
  --host-interference-note "Actual collection conditions."
```

See [REPLAY_IR.md](../execution_calibration/REPLAY_IR.md) for exact scope. In
particular, preload uses explicit logical versions to require copies while the
native probe overwrites preallocated buffers. The replay does not validate that
physical allocation lifecycle, higher-level ModelPlan semantic transforms or
full LLM inference. CUDA private workspace remains unknown and capacity
acceptance remains false throughout.

Hardware-independent contract and archive-tamper tests:

```powershell
python -B -m unittest discover -s profiling/batched_calibration/tests -v
```
