# Unified stateful service-response profiling

This preserves the earlier request-pressure diagnostic hypothesis and its failed
hardware qualification. New physical experiments should use the
[external available-service workflow](../available_service/README.md), which
does not infer LLM-available compute from worker requests.

Optional source-checkout tools measure parameters for the model-owned
`performance.service_response` law. CPU and GPU competitors use the same schema,
estimation rules and event evaluator. There is no DNN, separate profiling
predictor, total-layer fit or heldout refit.

Read the model and limitations note, `docs/UNIFIED_SERVICE_RESPONSE_20261022.md` (kept with the paper),
before interpreting output. This first hardware adapter predicts exact-context
serial inclusive action responses, not pure CPU/GPU throughput. Its normalized
residual-work state is a declared interference proxy, not a measured GPU queue.
Baseline-fixed control/gap/tail prices and observer neutrality remain explicit
qualification issues.

## Reusable sequence

Supply a source-bound compatible native probe and matching runtime paths.
Preparation inventories hardware/sources and declares all six jobs without
launching timed work. The shown masks are specific to the recorded host; supply
valid explicit logical-CPU masks on another computer.

```powershell
$campaign = "profiling/service_response/results/my-new-campaign"
python -m profiling.service_response.campaign prepare `
  --out $campaign --binary "EXPLICIT_NATIVE_PROBE" `
  --runtime-dir "EXPLICIT_GGML_RUNTIME_DIR" `
  --runtime-dir "EXPLICIT_CUDA_RUNTIME_DIR" `
  --cuda-runtime "EXPLICIT_CUDART_DLL" --seed 2026102201 `
  --main-affinity-mask 5 --main-thread-affinity-mask 4 `
  --cpu-worker-affinity-mask 5 --gpu-worker-affinity-mask 48

python -m profiling.service_response.campaign collect --out $campaign --split profile
python -m profiling.service_response freeze --out $campaign --output "$campaign/frozen.json"
python -m profiling.service_response.campaign collect `
  --out $campaign --split heldout --frozen "$campaign/frozen.json"
python -m profiling.service_response evaluate --out $campaign `
  --frozen "$campaign/frozen.json" --output "$campaign/evaluation.json"
```

Profile processes include short pulses, long on states and off-recovery periods.
Only isolated CPU-only/GPU-only intervals estimate response coefficients and
worker proxies. Combined work is validation-only. Each exact action and process
keeps its own samples before equally weighted process medians are formed.
Unknown values, signed differences, sample-count failures and CV remain visible.

Parameters, sources and the complete protocol are frozen before three new bare
heldout processes. Heldout trajectories change seeds, order and dwell times.
The barrier binds the frozen file and campaign manifest. Observed internal
action times, worker completions and total telemetry are not predictor inputs.
Predictions condition on each observed invocation start; one causal residual
history is maintained for the whole run rather than reset per invocation.

Every attempt retains its raw native results, process cleanup, numerical checks,
actual main-thread endpoint masks and source hashes. A timed failure is not
retried or removed. The only allowed startup retry is before GO with all owned
process exits verified. Existing result directories and output artifacts are
never overwritten. Historical state-response evidence remains untouched.

The collector uses the same repository measurement lock as other profilers.
Do not run timed probes concurrently with one another. Kernel tracing and
administrator access are not required for this workflow.

## Offline APIs and checks

```python
from profiling.service_response import (
    extract_job, freeze_profile, make_barrier, evaluate_frozen,
)
```

Extraction reads native evidence incrementally and keeps a small run record;
freezes do not embed the large original normalized records. Provenance and
reconstruction checks stay separate from empirical accuracy qualification.
Independent numerical checks are source-bound collector receipts, not a claim
that offline extraction re-executed the oracle.

```text
python -B -m unittest discover -s profiling/service_response/tests -v
```

These tests verify contracts and hand-computable behavior. They do not establish
hardware accuracy, physical queue identification or arbitrary-policy transfer.
