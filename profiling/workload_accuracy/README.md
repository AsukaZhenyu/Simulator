# Traced multi-layer workload and allocator validation

This directory adds an independent, bounded F32 GGML CPU/CUDA evaluator. It does
not change historical native probes, model contracts, builds, raw measurements,
or accepted profiles. The evaluator executes **actual supplied hidden activation
traces**, not generated tokens. Every round processes fresh supplied inputs;
outputs feed the next layer, and every layer retains its actual K/V tensors
across rounds. Repeating samples resets the complete workload, while history
evolves within each sample. The example has two layers and histories
`0 -> 2 -> 3 -> 4 -> 5`, not four copies of an immutable single-layer batch.

The physical graph implements input RMSNorm and scaling, separate Q/K/V,
optional Q/K RMSNorm, adjacent-pair normal RoPE at absolute positions, causal GQA,
optional sigmoid attention gating, O projection and residual, post-attention
RMSNorm and scaling, Dense SwiGLU or selected expert SwiGLU and weighted combine,
and the final residual. F32 matmuls use `GGML_PREC_F32`; CUDA additionally requires
and records `NVIDIA_TF32_OVERRIDE=0` and `GGML_CUDA_DISABLE_GRAPHS=1`. Unsupported
backends/ops fail; there is no CPU fallback for CUDA.

MoE IDs and normalized selected scores are evaluator inputs. The evaluator
does not run a learned router. Only the current layer/round routes drive native
expert residency after an explicit `route_known` barrier following completed
attention; future route IDs cannot trigger loads. `all` retains every
expert; `active` unloads experts absent now, reuses those still present, and
reloads returning experts with new allocation IDs. Immutable model weights may
exist in the host source independently of what is resident on the backend.
Version 2 records a shared event sequence for attention completion, route
revelation, residency actions, actual allocations/releases and expert execution.
The contract rejects early expert load/eviction, missing barriers or revealed
future routes. Static `all` residency is route independent and loads at model
setup; route-dependent `active` loads require the current barrier.

## Numerical and public compiler checks

A separate pass computes an independent scalar reference with double
accumulation and validates Q/K/V, optional gate projection, full persistent K/V,
attention output and final layer output. It checks finite values, exact shapes,
and every element with `abs_error <= 2e-4 + 2e-3 * abs(reference)`. Sampled runs
skip reference computation and compare complete round outputs after timing.

The same case JSON exports actual public `Model`, `Workload`, and `ModelPlan`
inputs. `public_audit.py` compiles and runs the real simulator and compares each
layer/round's positions, token count, existing and appended KV blocks/bytes,
current MoE trace and hidden-output boundary. Dense numerical operation
semantics/configuration match the declared public F32 decoder variant.

The audit deliberately rejects unsupported facade variants and records
non-equivalences. The native probe uses per-head GGML graphs and explicit host
materialization, so historical Dense macro timings cannot be reused as its
physical prices. Public MoE currently exposes an opaque synthetic projection,
router, grouped FFN and combine; its physical topology is not this trace-selected
decoder graph. Neither analytical simulator timing nor existing opaque MoE
costs are used to fit/score this evaluator. Native fixed-capacity KV buffers also
differ from Modeling's logical used KV blocks. An exact old-profile bind and
physical capacity equivalence are explicitly false.

Mixed Dense/MoE cases use the existing heterogeneous `global_m2` `ModelPlan`,
composed from ordinary public-Model-generated Dense and MoE layer documents.
The homogeneous convenience `Model` is not extended. Existing opaque MoE and
physical Dense hidden layouts currently compose only for single-token rounds;
the mixed audit rejects larger rounds. Homogeneous Dense/MoE cases separately
exercise a multi-token prefill followed by growing single-token histories.

## Timing protocol

`run.py` collects fresh sequential independent processes, freezes costs using
profile processes alone, then starts heldout processes with disjoint seeds and
different weights/actual input values. Cases, topology and route sequence are
declared in advance. Stage prices are medians keyed by exact operation semantics,
shape, current history, expert row-count distribution, backend, device and
thread count. A separate profile-only effective host gap accounts for graph
construction/allocation, materialized transfers, actual expert lifecycle and
release; its key additionally binds complete topology, KV capacity and routes.
Prediction is sum of exact covered stage prices plus that frozen effective gap.
Unknown stage or host-gap keys are rejected rather than interpolated.

The whole-workload wall timer covers real decoder execution and the declared
graph setup/allocation/materialization/lifecycle/release. Scalar reference,
comparisons, full activation serialization, memory event JSON, source weight
generation and backend initialization are outside it. No measured validation
time is subtracted. Minimal stage timers/counters remain instrumentation; this
profile is an effective price for this evaluator and is not a claim about a
production runtime or arbitrary mixed-placement `CompiledPlan` actions.

Acceptance preserves the existing thresholds: within-process CV <= 0.15,
between-process CV <= 0.20 and heldout absolute relative error <= 0.20, with at
least three complete profile and heldout processes. Numerical/shape coverage,
source/runtime stability, exact price coverage and pre-heldout freezing must
also pass. Failure is retained and reported; no threshold is weakened.

The runner acquires `profiling/.measurement.lock`, verifies source, executable,
GGML headers, link libraries and runtime DLL hashes before/after each process,
and checks the frozen file before/after every heldout process. Raw command,
stdout, stderr, cases, public model audit and hashes are retained even on native
failure or timeout. `verify` replays all contracts and regenerates profile-only
coefficients before accepting a saved report. Hash binding is reproducibility
evidence, not an attestation service.

## Memory claims

The native ledger records allocation/release identities, requested and reserved
payload bytes and exact live/peak sums for backend resident weight buffers,
expert buffers, K/V capacity, graph allocator arenas, GGML metadata capacity and
explicit CPU plan workspace. Resident logical bytes are measured with
`ggml_nbytes`; reserved resident bytes and actual graph arena sizes come from
GGML allocation APIs. Allocator size simulation is checked before the real
allocation, and the resulting buffer size must agree. Lifetimes prove weight
reuse, expert unload/reload, fixed KV capacity, and complete tracked release.

`capacity.py` first measures the real tracked peak, then executes the same trace
with peak minus one byte (must reject before crossing allocation), exact peak,
and peak plus one byte (must complete and release). These small bounded tests do
not try to exhaust machine memory. Native declared tensor payload is capped at
64 MiB and the caller's ledger budget is bounded at 512 MiB.

**A tracked allocator budget passing is only acceptance for that ledger.** It is
not a physical-fit proof or full process/device peak. Standard allocator
overhead, host weight/reference/control storage, backend handles/pools and CUDA
private workspace are not fully tracked. `process_peak_bytes` and
`cuda_private_workspace_bytes` stay null, the private bound is unknown, and
`physical_capacity_verified` stays false. Synchronized `cudaMemGetInfo` snapshots
are device-wide diagnostics only and cannot replace these unknowns with zero.

## Run

From the Simulator root (PowerShell), build against the existing tree; no GGML
source/build is edited:

```powershell
cmake -S profiling/native/workload_accuracy -B profiling/native/workload_accuracy/build-release `
  -G 'Visual Studio 17 2022' -A x64 -DCMAKE_TRY_COMPILE_CONFIGURATION=Release `
  -DCMAKE_CONFIGURATION_TYPES=Release -DCMAKE_CXX_FLAGS_RELEASE='/O2 /MD /DNDEBUG' `
  -DGGML_SOURCE_DIR='<llama.cpp>/ggml' -DGGML_BUILD_DIR='<llama.cpp>/build' `
  -DNLOHMANN_INCLUDE_DIR='<llama.cpp>/vendor' `
  -DCUDAToolkit_ROOT='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6'
cmake --build profiling/native/workload_accuracy/build-release --config Release -j 2
python -m unittest discover -s profiling/workload_accuracy/tests -v
python -m profiling.workload_accuracy.examples --backend CPU --output profiling/workload_accuracy/cpu-suite.json
python -m profiling.workload_accuracy.run run --allow-hardware `
  --binary profiling/native/workload_accuracy/build-release/Release/ggml-workload-accuracy-probe.exe `
  --suite profiling/workload_accuracy/cpu-suite.json --output profiling/workload_accuracy/results-cpu
python -m profiling.workload_accuracy.run verify profiling/workload_accuracy/results-cpu
python -m profiling.workload_accuracy.capacity --allow-hardware `
  --binary profiling/native/workload_accuracy/build-release/Release/ggml-workload-accuracy-probe.exe `
  --case profiling/workload_accuracy/results-cpu/profile-1-0.case.json `
  --output profiling/workload_accuracy/capacity-cpu
```

Generate a CUDA suite with `--backend CUDA`; the runner sets/verifies the required
CUDA flags and matches hashed runtime DLLs. Collect only when the shared machine
is exclusively available. Software tests never launch native hardware work.
`debug-smoke*/` records are troubleshooting evidence and are never formal
performance acceptance, even when numerics pass.

The earlier `formal-{cpu,cuda}-20261018` directories are preserved diagnostic
protocols: version 1 loaded route-selected experts before the attention/router
barrier. Their original raw files, frozen models, reports and binary remain
unchanged; a separate `PROTOCOL_DIAGNOSTIC_ONLY.txt` records this rejection.
`archive-protocol-v1` preserves exact original Python/native source bytes and
the native build binding. The corrected native build is `build-release-v2`;
version 2 uses new suite files, seeds and output directories.

See [RESULTS.md](RESULTS.md) for version 2 numerical/memory results and rejected
timing stability gates. Final verification also requires the independent
`tests/test_v2_evidence.py` auditor: it checks both expert metadata and payload
births/releases after the current trace barrier and injects early-allocation
corruptions for both. The frozen collector's original validation covers payload
births; the independent auditor supplies the stricter metadata check without
changing collected source/model/native bytes.
