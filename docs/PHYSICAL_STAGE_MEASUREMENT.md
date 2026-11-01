# Reusable physical-stage measurement workflow

This workflow collects real host/action boundaries and directly correlated GPU
activity, keeps profiling separate from modeling, and evaluates frozen diagnostic
models on fresh runs. It does not certify a continuous trace of universally
available CPU/GPU capacity. Unknown quantities and failed qualification gates
remain explicit.

All commands below run from the repository root. Paths in capitals are explicit
local inputs, not automatically downloaded dependencies. Collection is exclusive:
do not run another benchmark, build, or test suite concurrently. Existing desktop
activity is part of the observed environment; a background actor being enabled
does not establish the actual resource service available to the foreground.

## 1. Build an independent native capture executable

Read the [native build and handshake contract](../profiling/native/physical_stages/README.md).
The wrapper reuses the byte-pinned GGML `Executor.action`; it does not replace the
execution algorithm with an approximate timing loop. Build into a new directory
when changing source after a campaign has been prepared. Retain old binaries,
build receipts, source snapshots and failed raw outputs.

```powershell
cmake -S profiling/native/physical_stages -B profiling/native/physical_stages/build-release-new -G "Visual Studio 17 2022" -A x64 -DCMAKE_TRY_COMPILE_CONFIGURATION=Release -DLLAMA_ROOT="EXISTING_LLAMA_ROOT" -DGGML_BUILD_DIR="EXISTING_GGML_BUILD" -DRUNTIME_DIR="MATCHING_F32_RUNTIME" -DCUDA_ROOT="EXISTING_CUDA_ROOT" -DCUPTI_ROOT="EXISTING_CUPTI_ROOT" -DPython3_EXECUTABLE="EXPLICIT_PYTHON"
cmake --build profiling/native/physical_stages/build-release-new --config Release
```

The successful build produces `Release/physical-stages-probe.exe` and
`Release/build-manifest.json`. The receipt binds the executable and supplied
sources, compiler, SDK assets and libraries. It does not prove every transitive
DLL selected by the dynamic loader. The direct capture requires a compatible
installed CUDA/CUPTI stack and the matching strict F32 GGML runtime.

Use a process-local runtime search path; do not change the machine's global PATH:

```powershell
$env:PATH = "MATCHING_F32_RUNTIME;EXISTING_CUDA_ROOT/bin;EXISTING_CUPTI_ROOT/lib64;" + $env:PATH
$env:PYTHONPATH = @((Get-Location).Path, "$(Get-Location)/src", "$(Get-Location)/modeling", "$(Get-Location)/mapping", "$(Get-Location)/profiling") -join [IO.Path]::PathSeparator
```

The orchestrator also accepts the three runtime directories explicitly. Use the
same directories recorded during preparation. `bare` still loads linked CUPTI
libraries, but disables activity capture.

## 2. Prepare, then preserve the manifest

The [campaign collector](../profiling/available_service/campaign.py) exports exact
programs, independent numerical reference ports, input seeds, hardware identity,
owned actor settings and a hashed source inventory before timed execution.

```powershell
python -m profiling.available_service.campaign prepare --out "NEW_CAMPAIGN" --binary "NEW_PHYSICAL_EXE" --background-binary "EXISTING_STATE_RESPONSE_EXE" --runtime-dir "MATCHING_F32_RUNTIME" --runtime-dir "EXISTING_CUDA_ROOT/bin" --runtime-dir "EXISTING_CUPTI_ROOT/lib64" --seed 202610231
```

Preparation creates a new directory and starts no timed native workload. The
current frontend deliberately selects one representative single-layer F32 case
(`h128-q16-history0-attention-ffn-cut128`, one CPU thread). It prepares three
profile inputs and three fresh heldout inputs with different seeds. This is a
measurement frontend for that declared case, not an automatic benchmark of all
model sizes, quantizations, CUDA graphs, or architectures. The reference and
background target action indices are explicit in the manifest; do not reuse
their integer values for a different program without inspecting its bindings.

The protocol declares idle, CPU-background, GPU-background and combined
conditions, 32 foreground invocations per process, 20 warmups, 32 GPU-reference
repetitions and 4096 CPU-reference repetitions. It also declares the observer
threshold, error thresholds, independent-process count and reference hold limit.
These are protocol settings, not successful experimental results.

Do not edit source/build inputs bound by a prepared campaign. Create a new
campaign for changes to the executor, collection protocol or workload. Auxiliary
analysis modules have their own provenance; they do not rewrite collected raw
evidence.

## 3. Smoke test and observer control

```powershell
python -m profiling.available_service.campaign collect --out "NEW_CAMPAIGN" --split smoke
python -m profiling.available_service.campaign collect --out "NEW_CAMPAIGN" --split observer
```

The smoke cohort executes two full calls and checks the end-to-end lifecycle,
serialization, numerical ports and retained capture. A smoke pass does not
establish latency accuracy.

Observer cohorts compare fresh processes with CUPTI disabled, CUPTI enabled, and
CUPTI plus disjoint reference actors. Both modes retain native action clocks and
batch CPU counters: `bare` is not the original clock-free executor. Reference
processes retain contexts/buffers while waiting; a nonoverlapping sensor window
does not prove the sensors are neutral. Keep observer overhead in the report.
Do not subtract its mean from a traced layer duration to manufacture an
uninstrumented result. A failed observer gate permits diagnosis but prevents a
claim that the traced experiment validates untraced hardware accuracy.

## 4. Collect profile data

```powershell
python -m profiling.available_service.campaign collect --out "NEW_CAMPAIGN" --split profile
```

For each job the owned actors validate and warm up before READY. The controller
checks process/thread identities, applies declared affinity and supplies GO in
raw Windows QPC ticks. Before references run sequentially (GPU, then CPU), then
the full foreground batch, then after references (GPU, then CPU). Background
actors, when requested, run across these windows.

Physical action clocks and the CUPTI timestamp callback use the same exact
floor-rounded QPC-to-nanosecond conversion:
`windows_qpc_floor_ns` / `absolute_qpc_zero`. There is no timestamp-offset fitting.
After its requested invocations, each physical actor emits a sampling-complete
receipt and waits for STOP. CUPTI flushing, final ports, destruction and output
serialization occur after release. A full invocation started before the GO
deadline may finish after it; recorded timestamps determine its actual end.

Every attempt retains an execution receipt, native output hashes, numerical
receipts, actor identities, normal-exit/cleanup evidence and raw activity. Failed
infrastructure attempts are retained. The
[job index](../profiling/physical_job_index.py) permits an explicitly recorded,
same-workload replacement for a verified readiness timeout; it does not select
attempts by measured latency or prediction error. Do not silently rerun slow
samples or discard unfavorable calls.

## 5. Freeze parameters and record a barrier before heldout execution

Use only completed profile/observer data to construct the diagnostic profile:

```powershell
$FrozenAt = (Get-Date).ToUniversalTime().ToString("o")
python -m profiling.physical_stage_analysis freeze --campaign "NEW_CAMPAIGN" --output "NEW_FROZEN_PROFILE.json" --frozen-at-utc $FrozenAt
```

The [analysis adapter](../profiling/physical_stage_analysis.py) exports measured
component baselines and predeclared reference-response candidates. Predictions
are evaluated through
[modeling's reference component evaluator](../modeling/llm_infer_model/performance/reference_components.py).
Profiling owns observed parameters; modeling owns the explicit calculation.
Disjoint wall partitions describe accounting, not identified active CPU service.

Before launching heldout, save a new barrier JSON with `frozen_at_utc`,
`manifest_sha256`, the absolute `frozen_file`, and `frozen_file_sha256`. The hashes
are SHA-256 of the saved file bytes. For example, with explicit paths in Python:

```python
from pathlib import Path
from profiling.state_calibration import run

root = Path("NEW_CAMPAIGN").resolve()
frozen_path = Path("NEW_FROZEN_PROFILE.json").resolve()
profile = run.read(frozen_path)
run.save(root / "frozen-barrier.json", dict(
    frozen_at_utc=profile["frozen_at_utc"],
    manifest_sha256=run.sha(root / "manifest.json"),
    frozen_file=str(frozen_path),
    frozen_file_sha256=run.sha(frozen_path),
))
```

Keep the frozen profile and barrier unchanged through both heldout cohorts. Do
not use heldout errors to change constants, choose a favorable candidate, or
relabel failed data. A correction informed by heldout becomes a new model
version requiring another fresh heldout campaign.

## 6. Independent heldout and fresh bare transfer check

```powershell
python -m profiling.available_service.campaign collect --out "NEW_CAMPAIGN" --split heldout --barrier "NEW_CAMPAIGN/frozen-barrier.json"
```

The collector checks the barrier and frozen bytes before launch and again during
collection. Evaluation also requires heldout start times to follow the freeze.
Keep all three independent process runs and all 32 calls per condition; reporting
only an aggregate mean can hide large individual errors and instability.

Declare the optional untraced-target transfer cohort before collecting it:

```powershell
python -m profiling.physical_bare_validation prepare --out "NEW_CAMPAIGN"
python -m profiling.physical_bare_validation collect --out "NEW_CAMPAIGN" --barrier "NEW_CAMPAIGN/frozen-barrier.json"
```

The [bare transfer driver](../profiling/physical_bare_validation.py) changes the
foreground to `bare`, while before/after reference actors deliberately remain
`cupti`. Their saved target actions still require exact device activity
attribution. The reference adapter checks instrumentation by actor role for
this reason. This cohort tests transfer from traced-profile predictions to
fresh untraced targets; it is not an overhead-subtraction calibration set.

### Operational amendment: stop backgrounds after the target

The current campaign encountered retained AFTER-GPU readiness failures while
background work was still active. The
[post-target collector](../profiling/physical_posttarget_collection.py) resumes
collection without editing the original manifest, collector, frozen profile or
barrier:

```powershell
python -m profiling.physical_posttarget_collection --out "NEW_CAMPAIGN" --split heldout --barrier "NEW_CAMPAIGN/frozen-barrier.json"
python -m profiling.physical_posttarget_collection --out "NEW_CAMPAIGN" --split heldout_bare --barrier "NEW_CAMPAIGN/frozen-barrier.json"
```

Prepare the bare cohort before declaring this amendment so its full job list is
included. Before any new launch, `posttarget-job-index.json` binds all effective
jobs, the frozen profile, source receipts, preserved completed heldout IDs and
the IDs receiving changed post-target cleanup. One exact same-workload `-retry1`
replaces the retained heldout infrastructure failure; its failed receipt and
hash remain in the index. The older profile failure remains in its separate
unchanged replacement index. Attempts are not selected by latency or error.

Before references and foreground conditions remain as declared. Immediately
after foreground sampling completes, the wrapper stops its owned backgrounds
and verifies normal process exit **before** constructing any AFTER reference.
Each changed execution binds a `posttarget-cleanup.json` receipt with QPC
completion/STOP/exit evidence and scope `unloaded_drift_only`. Those AFTER
references describe the subsequent unloaded condition; they are not same-load
drift measurements, recovery identification, or forward prediction inputs.
Already completed jobs retain their original lifecycle, so the report explicitly
records the mixed post-target protocols.

## 7. Offline audit, attribution, reference adaptation and evaluation

Perform batch parsing after hardware collection has ended:

```powershell
python -m profiling.physical_campaign_audit "NEW_CAMPAIGN" --output "NEW_AUDIT.json"
python -m profiling.physical_stage_analysis evaluate --campaign "NEW_CAMPAIGN" --frozen "NEW_FROZEN_PROFILE.json" --output "NEW_HELDOUT_EVALUATION.json"
python -m profiling.physical_stage_analysis evaluate --campaign "NEW_CAMPAIGN" --frozen "NEW_FROZEN_PROFILE.json" --split heldout_bare --output "NEW_BARE_EVALUATION.json"
```

For a campaign with `posttarget-job-index.json`, use the
[post-target comparator](../profiling/physical_posttarget_analysis.py) instead
of the two original evaluation commands:

```powershell
python -m profiling.physical_posttarget_analysis --campaign "NEW_CAMPAIGN" --frozen "NEW_FROZEN_PROFILE.json" --split heldout --output "NEW_HELDOUT_POSTTARGET_EVALUATION.json"
python -m profiling.physical_posttarget_analysis --campaign "NEW_CAMPAIGN" --frozen "NEW_FROZEN_PROFILE.json" --split heldout_bare --output "NEW_BARE_POSTTARGET_EVALUATION.json"
```

This adapter validates the amendment and reuses the same frozen predictor and
parameters without refitting. The reference adapter also recognizes the bound
post-target index and records AFTER reference state scope. Use the campaign audit
to check actual background coverage during before/main windows and verified
background exit before AFTER construction.

The [campaign audit](../profiling/physical_campaign_audit.py) checks saved actor
lifecycle, observer, numerical and barrier evidence without fitting a model.
Missing jobs remain pending; missing evidence does not pass vacuously. Capture
integrity, observer neutrality, numerical validity, empirical stability and
prediction error are separate gates.

Reusable Python APIs:

```python
from profiling.available_service.physical_records import load_physical_records, matmul_work
from profiling.physical_reference_adapter import build_job_reference_supply
from profiling.available_service.reference_supply import validate_reference_supply, materialize_reference_window

supply, references = build_job_reference_supply("CAMPAIGN/jobs/JOB/execution.json")
if supply is not None:
    validate_reference_supply(supply)
```

[Physical record parsing](../profiling/available_service/physical_records.py)
uses CUPTI external correlations plus runtime/driver correlation IDs. It never
assigns an orphan kernel by nearby timestamps. Dropped records, invalid times,
foreign identities, unknown records and attribution conflicts remain visible.
Kernel/memcpy unions avoid double counting overlap. Runtime API intervals,
driver API intervals and device activity overlap and cannot be added directly.

The [reference adapter](../profiling/physical_reference_adapter.py) verifies the
prepared manifest/effective job, saved file hashes, program/action identity,
READY/output/completion clocks, counts, process/thread identity and independent
numerical receipts. It binds the saved oracle but does not rerun it. Jobs without
references return `None` rather than an invented zero-rate trajectory.

[Reference supply](../profiling/available_service/reference_supply.py) estimates
known reference GEMM FLOPs divided by the exact correlated kernel union. It is a
scoped reference-workload diagnostic, not universal device FLOPs or active SM
cycles. Only completed **before** references update a forward sample-and-hold
rate. **After** references audit drift. A rate is unknown before observation,
during sensor windows, after the predeclared hold expires, or when evidence
fails. Foreground duration and GPU utilization are not used to derive the rate.
The finite horizon includes all actor sampling ends; it is not extrapolated
beyond the recorded experiment.

CPU batch cumulative kernel+user counters include query/reset/runtime/capture
work and have coarse quantization. Small deltas can even exceed the enclosing
wall interval. They are retained without clipping. The adapter sets
`physical_interval_verified=False`: diagnostic candidates are visible, while
forward pure CPU-service rates remain unknown. A separate completed-reference
wall-response comparison does not resolve that missing physical service.

`materialize_reference_window` refuses unknown/stale coverage. Returned data is
`observed_diagnostic` and requires the modeling evaluator's explicit diagnostic
permission; it must not be relabeled as a neutral measured availability trace.

## Current receipt and remaining limits

The retained [physical-v2b-h128-t1 manifest](../profiling/available_service/results/physical-v2b-h128-t1/manifest.json)
records the concrete campaign for which the current collection path was exercised.
Its completed two-call smoke and one completed idle reference job were used for
offline adapter checks. Those checks established saved-evidence consistency and
the intended unknown/known gates, not a hardware accuracy result. Consult its
final audit/evaluation artifacts for cohort completion and metrics; this guide
does not predeclare their outcome.

Static background conditions across fresh processes do not identify an
instantaneous within-call resource-transition trace or a recovery law. Reference
probes perturb the system and become stale. GPU kernel envelopes can include
preemption/stalls; host API duration is not CPU-active time. CPU scheduling
Running/Ready/Blocked states are not separated by these counters. After-reference
delay also includes process creation and warmup, so it cannot isolate recovery.

These limitations define the next measurements needed for a stronger unified
resource model. Preserve them alongside any improved error metric. A lower
average error alone does not prove physical mechanisms, untraced accuracy,
stability, generalization to other hardware, or valid dynamic-resource recovery.
