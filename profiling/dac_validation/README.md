# Independent real-device strategy selection

`e2a.py` implements the E2A protocol in
[`DAC_VALIDATION_EXPERIMENT_DESIGN_20261024.md`](../../docs/DAC_VALIDATION_EXPERIMENT_DESIGN_20261024.md).
It creates a new write-once campaign; historical prices and measurements remain
unchanged. Supply the repository, `modeling`, `mapping`, and `profiling` on
`PYTHONPATH`; use one BLAS thread during preparation and analysis.

On a fresh machine, provide an explicitly built native binary, its adjacent
source-bound `build-manifest.json`, and runtime directories. No previous profile
or measurement result is needed:

```powershell
python -m profiling.dac_validation.prepare_portable --out build/my-preparation --binary PATH/steady-costs-probe.exe --runtime-dir PATH/strict-f32-runtime --runtime-dir PATH/CUDA/bin
python -m profiling.dac_validation.e2a refine-prethermal --prepared build/my-preparation --out build/my-campaign
```

The fixed Windows/QPC/CUDA study still requires the native build documented in
[`native/steady_costs`](../native/steady_costs/README.md); it does not claim a CPU
or NPU substitution when CUDA is unavailable.

```powershell
python -m profiling.dac_validation.e2a prepare --out build/dac-validation-20261025/e2a-v1
python -m profiling.dac_validation.e2a refine-prethermal --prepared build/dac-validation-20261025/e2a-v1 --out build/dac-validation-20261025/e2a-v2
python -m profiling.dac_validation.e2a profile --out build/dac-validation-20261025/e2a-v2
python -m profiling.dac_validation.freeze_e2a --out build/dac-validation-20261025/e2a-v2
python -m profiling.dac_validation.e2a heldout --out build/dac-validation-20261025/e2a-v2
python -m profiling.dac_validation.e2a observer --out build/dac-validation-20261025/e2a-v2
python -m profiling.dac_validation.e2a evaluate --out build/dac-validation-20261025/e2a-v2
```

The only measured parameters supplied to the primary predictor are arithmetic
means of isolated native actions. The preparation-only v1 inventory is refined
before any native acquisition: each unique semantic primitive is measured in
three fresh processes using `action`, with at least 500ms and 20 iterations of
active primitive warmup. This matches the target's minimum active warmup budget;
equality of hot-isolated and mixed working-set effects remains a heldout-tested
assumption. Full-graph initialization is outside timing. Warmup envelopes are
retained but are not used as parameters. Parent weights and
the numeric seed are matched across candidates. Profile and target seeds differ.

The modeling coordinate contains operation/configuration, operand geometry and
layout, backend, precision, threads, binary, runtime and hardware identity. It
contains no strategy/program identity, action ordinal or data values. Exact
coordinates are required; unknown costs raise. Reusing these prices assesses
**composition** at measured shapes, without claiming unseen-shape extrapolation.

Inclusive native actions contain dispatch and completion synchronization. No
independently unidentifiable transition correction is added. The remaining
assumption (that isolated hot action responses compose sufficiently accurately)
is explicitly tested by the target data. A failed accuracy gate is retained.

This executor is serial, has preallocated backing, and treats EVICT as a logical
control. Half-width CPU/GPU FFN parts execute in sequence. It does not validate
concurrent tensor parallelism, actual memory eviction, asynchronous prefetch,
multi-round token generation, or real unified-memory coherence. KV transfers
follow the existing token-block boundary; transfer coalescing is fixed and may
affect results. CUDA private workspace remains unknown.

Collection obeys `profiling/.measurement.lock`. Each process has its native
request/result, stdout, stderr and execution receipt. A failed process is retained
and collection continues; resuming never reruns a completed or failed job.
Selection uses eight paired fresh-process rounds and 10,000 paired round
bootstraps. Mean, P90, CV and observer gates retain the original thresholds.
`summary.json`, `process_metrics.csv`, `trial_metrics.csv` and
`selection_metrics.csv` are the primary paper-facing outputs.

After evaluation, additive postprocessing does not modify the frozen model or
prices:

```powershell
python -m profiling.dac_validation.analyze_e2a --out build/dac-validation-20261025/e2a-v2
python -m profiling.dac_validation.plot_e2a --out build/dac-validation-20261025/e2a-v2 --stem real_strategy_selection_v2
python -m profiling.dac_validation.report_e2a --out build/dac-validation-20261025/e2a-v2
```

The completed 2026-10-25 campaign retained all 471 calibration, 192 heldout and
192 observer processes. Frozen predictions selected the real arithmetic-mean
winner in 6/6 groups, but only 13/24 candidate programs and 0/6 entire groups
passed every quality gate. Seventeen of 96 observer pairs exceeded the 5% limit;
two heldout process means exceeded 10% APE and two process CVs exceeded 10%.
These failures are part of the result. Do not describe the campaign as fully
qualified merely because all 24 aggregate program means had APE below 6%.
`E2A_REPORT.md` and `delivery-manifest.json` bind the exact scope and outputs.
