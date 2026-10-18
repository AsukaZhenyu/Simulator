# CUDA API and GPU interval attribution

This optional stdlib source-checkout analyzer reads an operation-scoped
NVTX/CUDA capture. It does not predict latency or fit parameters. Use the
[separate native probe](../native/single_layer_wait_diagnosis/README.md), explicit
runtime paths and a new output directory. Preserve profiler failures, raw
numerical/interval checks and the Nsight SQLite export.

```powershell
python profiling/single_layer_wait_diagnosis/analyze.py `
  --campaign PATH-TO-CAPTURE `
  --out NEW-FULL.json --compact-out NEW-COMPACT.json
python -m unittest discover -s profiling/single_layer_wait_diagnosis/tests -v
```

Exact target counts, block boundaries, threads, process/context identities and
launch/copy correlations must pass before attribution. GPU activity uses an
interval union; CUDA API and GPU intervals overlap and cannot be added as
separate costs. Native QPC timestamps are not mapped to Nsight's clock.

The [diagnostic report](../../docs/SINGLE_LAYER_CUDA_WAIT_DIAGNOSIS_20261018.md)
retains profiler compatibility warnings, API coverage limits and all sample
tails. Profiler-instrumented observations are excluded from hardware acceptance
and parameter fitting. The full report remains local measurement evidence;
the compact export binds its source hashes and summarizes all blocks.
