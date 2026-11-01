# Independent transfer response and observer diagnostics

This follow-up preserves the earlier frozen Dense campaign. It has two separate
questions: whether a transparent size-dependent COPY response transfers to new
sizes, and whether an observer contrast remains consistent within one process.
Neither question is answered by refitting complete target-strategy times.

`modeling/llm_infer_model/performance/transfer_response.py` owns the response
formula and exact-anchor counterfactual arithmetic. `transfer_fit.py` only fits
nonnegative terms using independent calibration sizes. `task1.py` owns programs,
raw acquisition, freeze barriers, numerical checks and heldout reporting.

## Transfer contract

The measured path is the original GGML default CPU-buffer/CUDA-buffer COPY
operation plus its declared completion synchronization. All allocation, input
generation and byte verification are outside timing. Host page-lock status is
not inferred. H2D and D2H are independent coordinates.

The fitting candidates are a nonnegative affine response and continuous
two-regime responses with a hinge at 16 KiB, 256 KiB or 1 MiB. Relative least
squares gives small and large copies explicit error weight; training-only
leave-one-size-out error selects the candidate. The heldout split is never read
for parameter fitting or candidate choice. An ordinary absolute-error affine
fit and a relative-error affine fit remain visible heldout controls.

The model terms are fixed inclusive response plus effective byte-service costs.
A size-dependent slope is not proof of a particular physical mechanism, nor a
measurement of PCIe-only bandwidth. `scenario_ns` preserves an exact measured
anchor, scales only its byte term, rejects a negative residual, and requires an
explicit effective-byte-service counterfactual assumption unless hardware
scaling has separately been qualified.

The fixed protocol uses 8 training sizes and 8 disjoint heldout sizes, 2 routes
and 3 fresh processes per size. Each process performs at least 500 ms and 20
active warmup iterations, then 8 trials of 8 identical COPY responses. Every
payload byte is checked by an independent procedural SHA256 oracle. No outlier
or failed process is removed.

## Observer contract

`native/paired_probe.cpp` builds a new diagnostic binary around byte-pinned
historical generated Executor helpers. It does not modify the original source
or replace the original binary. Each initialized process alternates bare and
per-action-QPC-stamped execution in an ABBA or BAAB sequence. Every block has the
same untimed reset and one plain preconditioning execution. All frame storage
is allocated before timing; serialization and port checking follow all timers.

The 24 complete Dense programs each receive 4 fresh processes and 8 adjacent
four-block contrasts per process. Four representative difficult programs also
receive 4 bare/bare sham processes. Balanced positions cancel a linear temporal
trend; nonlinear state variation can remain. Sham controls expose that remaining
variation without assigning it to an operating-system mechanism.

The new binary and preconditioning are a changed execution context. Its results
are diagnostic evidence, not a retroactive qualification of predictions for the
old binary. A new complete accuracy campaign would require independent prices,
freeze and targets under the same new binary context.

## Reproduction

Set `PYTHONPATH` to the repository plus `src`, `mapping`, `modeling` and `profiling`.
Use explicit existing GGML source/import-library/CUDA toolkit paths to build
`profiling/native/steady_costs` first. It generates the pinned helper includes;
no previously measured profile is required. Build the additive paired probe:

```powershell
cmake -S profiling/dac_followup/native -B build/paired-probe `
  -G "Visual Studio 17 2022" -A x64 -T host=x64 `
  -DCMAKE_TRY_COMPILE_CONFIGURATION=Release `
  -DLLAMA_ROOT=<llama-source> -DGGML_BUILD_DIR=<llama-build> `
  -DSTEADY_GENERATED_DIR=<steady-probe-build> -DCUDA_ROOT=<cuda-toolkit>
cmake --build build/paired-probe --config Release
python -m profiling.dac_followup.task1 prepare --out build/task1 `
  --binary <steady-costs-probe.exe> --observer-binary <paired-probe.exe> `
  --runtime-dir <strict-f32-runtime-directory>
python -m profiling.dac_followup.task1 calibration --out build/task1
python -m profiling.dac_followup.task1 freeze --out build/task1
python -m profiling.dac_followup.task1 transfer-heldout --out build/task1
python -m profiling.dac_followup.task1 observer --out build/task1
python -m profiling.dac_followup.task1 evaluate --out build/task1
```

An optional `--reuse-program-manifest` reuses only compiler-exported programs
and independent numerical reference artifacts. It does not read old performance
parameters. Omit it to create those inputs afresh. Use one collector at a time;
the shared exclusive measurement lock rejects concurrent native campaigns.

## Separate process-affinity intervention

The original launcher constrains the entire owned native process to logical-CPU
mask 4, in addition to fixing the main thread to mask 4 and GGML computation to
one thread. This also restricts any runtime/helper threads and must be part of
the profiling context. One compute thread does not imply that every driver or
runtime thread should have the same restriction.

`affinity.py` defines a separately frozen 32-process diagnostic: H2D and D2H at
1 KiB and 1 MiB, process masks 4 and 20, four balanced fresh-process rounds. The
main thread remains at mask 4, GGML remains one-threaded, and each process uses
the same 256-replay measurement window. Both parent and native process-affinity
readbacks must match the declared mask. It reports only the overall effect of
permitting another logical CPU; neither physical-core placement nor a particular
helper-thread mechanism is assumed.

The collector applies an explicit source-bound request-to-affinity launch
adapter inside its own Python process. It never edits the historical launch
module, changes another process or changes system settings. A mask-20 price is
a different resource context and cannot silently share an old mask-4 cost key.

```powershell
python -m profiling.dac_followup.task1_v2 --out build/task1-long-window --previous build/task1
python -m profiling.dac_followup.task1 calibration --out build/task1-long-window
python -m profiling.dac_followup.task1 freeze --out build/task1-long-window
python -m profiling.dac_followup.task1 transfer-heldout --out build/task1-long-window
python -m profiling.dac_followup.task1 evaluate --out build/task1-long-window
python -m profiling.dac_followup.affinity prepare --out build/task1-affinity --previous build/task1
python -m profiling.dac_followup.affinity collect --out build/task1-affinity
python -m profiling.dac_followup.affinity evaluate --out build/task1-affinity
```

The `--previous` argument reuses source-bound probe programs, numerical oracles
and runtime paths. The long-window fitter reads only its newly acquired training
observations; the affinity diagnostic fits no performance model at all.

## Profile identity and promotion

New `prepare` calls record fresh static CPU/GPU/driver/topology identity, process
affinity, main-thread affinity, compute-thread count and the sampling window.
Binary and runtime-DLL hashes are separate required coordinates. Identity is
not a measurement of changing available compute, clock frequency or background
load. Equal static identity never implies equal runtime state.

The first October 25 follow-up manifests did not contain a fresh pre-campaign
hardware query. Their separate post-campaign identity receipt matches the prior
same-machine E2 identity; it cannot reconstruct the earlier resource trajectory.
The future-prepare metadata patch was made only after every native process had
finished. Executed source snapshots and original manifests remain unchanged.

`export_transfer_profile.py` produces an explicit parameter document containing
route-specific qualification/failure, independent evidence and full known
scope. It does not install a default cost table. A passed size-response test is
not physical-link scaling qualification, mixed-strategy accuracy qualification,
or permission to mix different devices, binaries or sampling windows.
