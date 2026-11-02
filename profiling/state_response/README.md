# Scoped state-response calibration

This optional source-checkout package estimates **completed host-action**
responses from independent measured processes. Modeling owns the response table,
remaining-progress integration, dependencies and execution-time predictions.
There is no Ridge/DNN predictor, whole-layer fitted offset or second simulator.

The latest thread/runtime/recovery diagnosis, `docs/THREAD_RUNTIME_RECOVERY_DIAGNOSIS_20261022.md` (kept with the paper),
includes bounded kernel evidence, sixteen controls and an owned main-thread
placement intervention. A separate three-profile/three-fresh-process check
covers 686 invocations: primary median error 10.50%, P90 39.80%, with GPU-load
and crossing subsets still inaccurate. Complete dynamic-state accuracy is not
qualified; diagnostic interruptions are never supplied as heldout predictions.

The four numerical inputs are owned CPU replay request `0/1`, owned GPU replay
request `0/1`, owned DRAM hold bytes and owned VRAM hold bytes. A replay request
is not a utilization percentage. Held bytes are not physical residency or
memory-bandwidth pressure. Total CPU/GPU utilization includes the foreground
and is diagnostic only. A table applies only to its exact operation sequence,
backend, shapes, precision, threadpool, hardware and synchronization boundary.

## Protocol and ownership

1. Declare the numerical grid, graph/context, controller schedule family,
   independent seeds/process inventory, sources and gates before collection.
2. Profile full stamped replays under every supported state. All actions,
   disjoint before-action gaps and the tail are retained. A process survives
   state changes; initialization/teardown holes remain unknown.
3. For each interval/state cell, take each process's exact median and the exact
   median of those medians, giving processes equal weight. At least three
   independent processes are required. No trimming or whole-total fitting.
4. Freeze all profile records, raw-source hashes, parameters, gates and protocol
   before any heldout launch. Unstable numerical estimates remain diagnostic
   estimates with `qualified=false`; missing estimates remain unknown.
5. Collect fresh **bare** replays with disjoint numeric seeds and independently
   scheduled temporal orders/durations. No heldout internal action timestamps
   are needed or supplied to the predictor.
6. Pass the external request trace to `modeling.conditioned_response`. Evaluate
   steady invocations, known state crossings and unknown transitions separately.
   Every observation and every unpredictable invocation remains in the report.

Four candidates are fixed before heldout collection and reuse the same frozen
action cells: instantaneous dynamic integration; CPU/GPU requests fixed at `0/0`
with the same memory holds; the entry state held constant; and dynamic integration
with four independently profiled worker-response delays. Every candidate calls
the same modeling evaluator. None refits heldout data. The new protocol declares
the lag candidate as primary before collection; it does not select a winner from
measured errors. This is an experimental hypothesis, not an accuracy claim.

The original within-process CV 15%, between-process median CV 20%, three-process
and timing-error 20% criteria remain recorded. Compute/copy source qualification
and accuracy are independent fields. Tiny gap/control CV is reported separately;
it is not silently advertised as an original source criterion. A lower CV does
not demonstrate a more accurate model. Pooled raw CV across deliberately
different states is diagnostic, not a stationary-measurement acceptance gate.

The newer protocol declares `minimum_cell_samples_per_process: 10` before
collection and uses longer profile holds. Sparse cells retain their numerical
diagnostic estimate while failing the sample-count qualification gate. Eligibility
comes from a process's declared **calibration** states: other memory-hold cohorts
are not missing replicates. A declared state with no complete sample still fails.
Statistics use one transaction-local sample index; this changes no samples,
weights, medians, CV thresholds or byte-verification rules.

The effective response includes host submission, backend work, waiting and
synchronization inside the inclusive action boundary. It is **not** pure GPU
throughput. Never add correlated API/kernel times to an already inclusive action
or subtract overlapping GPU time to fabricate an exclusive host cost. CPU-active
versus host-wall diagnostics can locate unresolved waiting, but do not identify
an additive fixed overhead by themselves.

## API

`make_run_record(program, native, events, *, context, run_id, split, seed,
process_id, started_at_utc, ended_at_utc, source_files,
numerical_verification, raw_bindings=None)` normalizes existing continuous
`blocks[].iterations/action_intervals` or `iterations[].action_intervals`.
`context.state_semantics` must be `offered_load_controlled`.

`events` accepts historical `steady_begin_tick/steady_end_tick/requested_resources`
or explicit `{begin_tick,end_tick,state}` segments. A dictionary may provide
`known_segments` for dynamic prediction and separate `events` for profile-only
settled calibration windows. Extra phase metadata remains in the raw evidence.
All action samples, including those outside calibration windows, are retained.
State has exactly
`cpu_replay,gpu_replay,host_memory_bytes,device_memory_bytes`. Missing intervals
or `state:null` remain unknown. The new collector's common GO schedule is an
external *request* trace; delayed worker reaction and GPU queue drain remain
model assumptions to test rather than claims of instantaneous physical occupancy.

`source_files` maps absolute paths to SHA-256. `raw_bindings` maps exactly
`program,native,events,numerical` to paths in that inventory. Events JSON may
contain a list or an `events`/`known_segments` field. Load/evaluation reconstruct
all extraction and statistics from these immutable files. Numerical checks are
the bound collector receipts; offline analysis does not pretend to rerun NumPy.
Omitting raw bindings allows diagnostic statistics but prevents acceptance.

Optional `worker_cpu` and `worker_gpu` raw bindings support a separately
predeclared **request-response lag** candidate. With
`context.time_origin="planned_go_start_v1"`, all requests and invocations keep
the external GO origin instead of moving the first request to the first measured
foreground call. Profile-only worker invocation records estimate four constants:
CPU/GPU start and stop response delays. On response uses the first newly
authorized worker invocation after a quiescent on edge; off response uses the
completion of the final invocation authorized by the old request. A zero off
delay requires a completed old invocation and a sampling clock that covers the
off boundary. Missing, cancelled, nonquiescent or censored edges remain explicit
unknown evidence, never invented zeros. Each process's exact edge median has
equal weight; only the final integer-ns model boundary uses the declared ceiling.
The four model inputs are `cpu_start_ns`, `cpu_stop_ns`, `gpu_start_ns` and
`gpu_stop_ns`; each remains `None` when unidentified. Variability failures retain
diagnostic estimates with `qualified=false`, rather than disguising an unstable
parameter as an established one.

These delays describe an **owned worker call busy-interval proxy**. They are not
pure GPU activity, physical utilization, or additive inference-time offsets.
Four constants may miss state/history-dependent response. Their variability,
unknown edges and qualification remain visible even when diagnostic estimates
are usable. The model owns delayed request transformation and cancellation.
At external GO time zero, the effective CPU/GPU controls are explicitly idle
`0/0`; the owned memory allocations are already held. These controls describe
the absence of owned replay work, not a claim that the computer is otherwise
idle. CPU and GPU pending responses evolve independently. A reversed request
cancels its pending response, including a reversal at that response's due time;
an unchanged request does not restart its delay. Missing necessary delays reject
the candidate rather than becoming zero. This control model may still miss
queueing, frequency changes and other state/history effects.

Heldout predictions use only the original external request schedule and frozen
profile delays. Heldout worker timings are never parsed into predictor inputs.
Unknown request history before an invocation makes the lag candidate unresolved:
its effective state cannot be reconstructed from a known entry state alone.
The original instantaneous dynamic, no-owned-load and entry-state controls remain
available. `lagged_dynamic_state_integration` is a fourth fixed candidate, and
`protocol.primary_prediction_model` declares the primary before heldout collection.
Reports retain the original `overall` meaning and expose the declared primary
under `predeclared_primary_model`, `predeclared_primary_summary` and
`predeclared_primary_buckets`. Unknown primary names reject the request; results
do not choose their own best model.

`freeze_profile(program, records, *, context, required_states, frozen_at_utc,
protocol, gates=None)` returns a source-bound profile-only artifact.
`load_frozen` reconstructs its exact statistics and qualification decisions.
`evaluate_heldout(program, frozen, records, *, barrier=None)` reports all errors,
coverage and stability. The collector barrier has schema
`state-response-heldout-barrier-1`, `frozen_artifact_sha256`, UTC `created_at_utc`
and sorted `profile_record_sha256s`. The collector separately owns actual Popen
PID binding, source-inventory checks and chronology protection.

Predictions are **whole-call durations conditional on observed invocation start
times** on the external clock. Measured internal stage times are not inputs.
This does not establish a free-running rollout that predicts invocation launch
times, arbitrary unseen load values, other graphs, asynchronous execution or
memory changes within a PID. The current hardware protocol holds DRAM/VRAM
allocations constant per PID; CPU/GPU requests vary over time. Cross-boundary
integration additionally assumes memoryless effective progress; separate heldout
crossing/recovery results test that assumption.

The stamped profile versus bare heldout measurement control has not yet been
independently established for the new native/state scope. Therefore `accepted`
and `full_accuracy_qualified` remain false. `steady_accepted`,
`known_crossing_accepted` and `statistical_protocol_accepted` describe only their
explicit new statistical protocol, separately from raw numeric error and CV.
They do not replace the previous 0/14 complete single-layer qualification result.
When `protocol.profile_settle_ms` is declared, recovery errors before that request
age and settled errors after it are reported without excluding either cohort.

## Offline commands

All commands require fresh output filenames and write LF UTF-8 JSON. They do not
launch hardware. With the library installed, or `modeling` on `PYTHONPATH`:

```text
python -m profiling.state_response import-legacy --campaign OLD --split profile --out imported.json
python -m profiling.state_response freeze --program program.json --records profile-records.json --context context.json --states grid.json --protocol protocol.json --out frozen.json
python -m profiling.state_response evaluate --program program.json --frozen frozen.json --records heldout-records.json --barrier barrier.json --out evaluation.json
```

The legacy importer recovers full action evidence from mixed-mode historical
resource campaigns and never executes or imports the old Ridge model. An old
one-process smoke remains insufficient for three-process qualification. Historic
hashes identify that version; no current-source equivalence is claimed.

## Source-bound hardware collector

`prepare` generates fresh inputs, independent process/seed schedules, source and
runtime hashes, hardware/context metadata and a predeclared protocol. It performs
no timed hardware measurement. `collect` launches the bounded owned processes
under the exclusive measurement lock. Supply a successfully built compatible
probe and explicit runtime libraries; the collector does not guess their paths.
An example for the currently defined Windows scope is:

```powershell
$campaign = "profiling/state_response/results/my-new-m0"
python -m profiling.state_response.collect prepare `
  --out $campaign `
  --binary "profiling/native/state_response/build-release-v3/Release/state-response-probe.exe" `
  --runtime-dir "profiling/native/highlevel_accuracy/build-runtime-f32-v2/bin" `
  --runtime-dir "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin" `
  --cuda-runtime "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/cudart64_12.dll" `
  --seed 20261029 --memory-index 0 --repeats 3 `
  --profile-hold-ms 2000 --settle-ms 150 --heldout-cycles 4 `
  --main-affinity-mask 5 --cpu-worker-affinity-mask 5 --gpu-worker-affinity-mask 48

python -m profiling.state_response.collect collect --out $campaign --split profile
```

Next use the offline `freeze` command or `freeze_profile` API with that campaign's
profile records, exact manifest context/protocol and declared state grid. Select
one matching profile program; numeric initial values/seeds may differ while its
execution contract stays exact. Save the frozen artifact to a **new** file. Only
after that artifact exists may the independent heldout collector run:

```powershell
python -m profiling.state_response.collect collect `
  --out $campaign --split heldout --frozen "$campaign/frozen.json"
```

Then use the offline `evaluate` command with `heldout-records.json` and
`heldout-barrier.json`. It reconstructs frozen statistics and validates the
declared chronology before comparing model-owned predictions. Keep successful
and failed attempts, manifest, barrier, frozen artifact and raw evidence together.
The alternative wrapper `python -m profiling.state_response collect ...`
delegates its remaining arguments to this same collector; the direct module
commands above avoid confusing its module name with its `collect` subcommand.

### Memory cohorts and execution scope

`--memory-index` declares one allocation cohort before collection:

| Index | Owned DRAM hold | Owned VRAM hold |
|---|---:|---:|
| 0 | 0 | 0 |
| 1 | 0 | `--memory-bytes` |
| 2 | `--memory-bytes` | 0 |
| 3 | `--memory-bytes` | `--memory-bytes` |

The default nonzero hold is 64 MiB. Each selected cohort has three independent
profile and three independent heldout foreground processes. All four CPU/GPU
request combinations occur within each profile trajectory. Its DRAM/VRAM holds
remain constant. Separate cohort campaigns and parameter freezes reduce peak
offline RAM usage: full stamped action records, their normalized forms and
reconstruction can otherwise coexist in memory. This organization changes no
sample weights and is not an accuracy-based selection of memory conditions.
Each frozen table covers only its declared allocation coordinates; results from
one cohort do not establish the other three or memory changes during a call.

CPU affinity is also part of the source-bound experimental scope. The earlier
main/CPU-worker mask `3` selects logical processors `{0,1}`; the new mask `5`
selects `{0,2}`. The GPU worker's host thread changes from mask `12` (`{2,3}`)
to mask `48` (`{4,5}`). Main and CPU-worker processes intentionally share their
selected CPUs, while the GPU-worker host process uses its separately declared
set. A bit mask specifies logical processors, not a portable physical-core or
SMT topology. Record the actual hardware and masks, and supply masks valid on
the target computer. A result under `5/5/48` cannot be silently presented as an
accuracy improvement under the old `3/3/12` configuration.

`--main-affinity-mask` is a **process** mask. The optional prepare argument
`--main-thread-affinity-mask` declares a separate, single-bit binding for the
owned native main thread, applied after warmup and before GO. It must be a
subset of the process mask and agree across manifest protocol, job and context.
The collector verifies process/thread birth identity and the actual thread mask
before and after sampling. Other pool threads keep their process permissions;
this option does not pin the entire pool, change interrupt routing or infer
physical-core topology. Omitting it performs no per-thread pin. For example,
mask 4 selects logical CPU2 inside process mask 5 on the measured host; it is not
a portable recommendation to always select CPU2. Formal cohort
`development-v5-core2-m0` freezes that explicit thread binding as part of context.

The current primary native scope retains `legacy_double_sync`, serial completed
actions, F32 precision and disabled CUDA graphs. GPU phase-decomposition probes
are a different proposed boundary and do not retroactively replace these action
prices. Earlier shorter holds, different origins, masks, source versions and
failed qualification remain reproducible from their archived sources and raw
artifacts. No earlier campaign or measured outcome is rewritten by this protocol.

### Startup failures and measurement isolation

At most **one** pre-GO retry is declared. It is permitted only when GO was never
issued, every created owned process has a verified exit, and cleanup has no
error. Every failed attempt retains its directory and receipt. Once GO has been
issued, a failure is a timed campaign failure: it cannot be relabeled as startup,
repeated until favorable, or dropped from the evidence. Changing this rule
requires a new protocol and campaign.

Never build, run tests, modify bound Python/native sources or launch a second
hardware collector during a measurement campaign. A changed source version
requires a fresh prepare/source binding. Old artifacts retain the source
snapshots used to produce them and are not rewritten to match current code.

## Recorded development results and limits

The completed **development-v3-m0** campaign has 3,398 profile invocations from
three foreground PIDs and 487 independent bare heldout invocations from three
fresh PIDs. All successful foreground/background actors pass independent
numerical checks. All 487 heldout calls have known request support. The primary
was declared before heldout collection, not selected from these errors.

| Candidate | Process-balanced median ARE | Pooled P90 ARE | Worst call ARE |
|---|---:|---:|---:|
| Primary: dynamic integration with profile-only response delays | 13.243% | 68.125% | 122.575% |
| Instantaneous dynamic integration | 14.390% | 68.269% | 122.575% |
| No owned CPU/GPU work | 23.066% | 75.126% | 98.338% |
| Entry state held constant | 14.390% | 68.395% | 122.575% |

The primary's 444 within-state calls have median/P90 ARE 12.679%/67.070%; its
43 state-crossing calls have 26.885%/88.618%. The median is the median of the
three per-process medians of invocation ARE; P90/worst pool all retained calls.
A lower typical error has not resolved the tail: source qualification is false,
180 original compute/copy cells fail their source criteria, and a matching
stamped/bare compatibility control is not independently established.
`accepted=false` and `full_accuracy_qualified=false` remain the correct result.

The subsequent **development-v4-m0 profile-only** observer campaign shows why
request bits are insufficient as actual load intensity. With the same CPU/GPU
request `1/1`, mean CPU-worker core equivalents are **0.962, 0.016, 0.981** across
the three profile PIDs, while main-process values are **0.035, 0.987, 0.027**.
CPU-off counter windows wholly after the final old-request call also retain
positive cumulative CPU increments. Call completion therefore does not certify
process silence. The diagnostic does not identify spinning, infer a quiescence
threshold, or fit a recovery/throughput formula.

Nominal 10 ms counter sampling has actual median intervals around **15.6 ms**;
the smallest observed positive CPU increment is **15.625 ms**, despite a 100 ns
QPC tick. Every observed actor uses the same actual
`C:\Windows\System32\vcomp140.dll` path and on-disk version **14.51.36247.0**;
path/hash/version receipts agree. The exact OpenMP launch-variable mapping is
empty, meaning unset variables, not known runtime defaults. These observations
do not verify mapped-image memory bytes or identify a unique cause of the
runtime variation. These profile diagnostics are separate from the heldout
validation below.

Independent v4 heldout validation has subsequently completed: 383 complete calls
from three fresh foreground PIDs, after freezing 2,226 profile calls. The
predeclared lagged primary has process-balanced median ARE **64.027%**, pooled
P90 **69.412%** and worst-call ARE **309.376%**. All calls have known request
support and numerical predictions; this is coverage, not successful accuracy.
The three per-process median errors are 68.004%, 64.027% and 8.593%. In
particular, the first two PIDs remain slow inside unchanged request intervals.
The source and full accuracy gates fail. This repeat does not support a claim
that v3's 13.243% median generalizes across processes.

New evaluation output adds `primary_validation`, which checks the model chosen
in the frozen protocol, its lag qualification, complete prediction coverage,
state-specific error/stability and three distinct crossing PIDs. Older top-level
`overall`, `buckets` and acceptance fields remain the instantaneous legacy
baseline and are explicitly labeled by `legacy_validation_scope`; they must not
be read as lagged-primary acceptance. Historical evaluation files are immutable.

Only **m0** has been measured in independent hardware validation. Prepared
m1/m2/m3 campaigns have not been measured; they need fresh prepare after source
changes. The four-coordinate design is not an assertion that all 16 numerical
states passed hardware acceptance. Installed library use does not depend on
these local experiment artifacts or on generated profiles.

See the dated calibration report, `docs/STATE_RESPONSE_CALIBRATION_20261021.md` (kept with the paper),
for numerical results, source-snapshot preservation and known limits. The
read-only `activity_analysis.py` verifies consumed raw SHA receipts and compares
owned-process CPU activity by declared state and after call completion. It
performs no parameter fit, no foreground-total correction and no heldout-counter
prediction. The next model needs identifiable, causally available service-state
evidence; neither a guessed counter-to-available-rate conversion nor a PID
offset supplies that evidence.
