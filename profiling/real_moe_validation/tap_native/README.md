# Callback-free CPU expert route observation (v4)

This is an additive diagnostic for the frozen OLMoE CPU and CPU-experts/GPU-other
placements. Previous collection/runtime revisions and measurements are retained.
It is not an end-to-end latency predictor.

The private CPU DLL observes `MUL_MAT_ID` at the generic CPU dispatch entry,
before ordinary/repacked kernel selection. Only thread zero invokes the observer.
The source IDs are already materialized for the existing CPU expert consumer;
the observer reads the same strided `src[2]` values that the kernel consumes.
No scheduler evaluation callback, tensor flag, allocator change, graph node,
additional GPU copy, synchronization or barrier is introduced. Neither llama nor
the CUDA DLL is rebuilt. Registration and removal happen outside all computation.

The callback copies IDs into bounded, preallocated POD storage. It does not
allocate, serialize JSON, throw or mutate tensors. After the request, observations
are grouped by forward/layer, and the actually executed up/gate/down stage lists
and IDs are compared. Three stages are reported only if three really execute.
An expert assigned to another backend is unsupported by this CPU-only observer;
there is no legacy callback fallback.

The build also prepares a pinned internal-context header with two read-only
inline accessors and no changed class fields. The diagnostic probe reads the
actual completed-forward graph metadata and scheduler assignment. This relies on
the matching internal ABI and bound source/runtime version, and is not a public
portable llama API. Fingerprints cover graph operations, types, flags, geometry,
strides, named inputs, operation parameters, assigned backends and split/copy
counts. They do not directly expose private CUDA fusion decisions.

Normal, replay and capture use the same model/context/backend settings. Per-step
greedy IDs are recorded independently from forced input/output IDs. Optional
logit observation copies the complete F32 logits into preallocated storage; hashes,
top logits and margins are produced after the request interval. Graph metadata
observation and its hashing are diagnostic host work and are timed separately.
Any request with route, logit or graph observation has
`timing_usable_as_bare_reference=false`. Normal latency truth must be collected
without these observers. Observation is not claimed to be free of time/cache
effects, and matching logits alone cannot prove hidden bare routes identical.

`tap_campaign.py prepare` fixes the old calibration suite's first short/long
requests to eight output tokens, keeps the reviewed resident capacity guards,
and prepares original-CPU-DLL versus private-CPU-DLL normal runs. Actual normal
outputs are then used for fixed-token replay and capture. Preparation never runs
inference. `collect` is serialized by the global measurement lock, retains the
200 ms external capacity observer, and never retries or overwrites a job.

Acceptance separates acquisition, binding, structure, numerical observations and
partition. All required rows, absolute positions, 16 layers and top-8 unique
expert IDs must be present; final-layer prefill pruning remains explicit. The
actual expert-node backend must be CPU. Normal/replay/capture greedy IDs and full
logits hashes, plus graph/partition fingerprints, must match at every step.
Unsupported stages/costs remain unknown. The smoke check establishes neither
whole-model performance accuracy nor broad strategy equivalence.

## Small-graph mechanism checkpoint (2026-10-27)

`consumer_contract.cpp` is the low-memory native contract for the exact v4b
consumer callback. It uses small F32 weights, real GGML `MUL_MAT_ID` nodes and
one CPU scheduler. Sixteen artificial layers execute up/gate/down, with a real
strided IDs view (`nb1=256` bytes although eight selected IDs occupy 32 bytes).
The four cases are threads 1/16 × query rows 1/3. Hook-off → hook-on → hook-off
must preserve output bits and graph/CPU partition fingerprints; all 48 actual
stage observations must match the independent initializer. Invalid-ID/layer and
POD-capacity checks call the observer directly and never execute invalid IDs in
a compute kernel. This is not a quantized LLM accuracy experiment.

The single declared execution passed 4/4 cases and 16/16 rejection checks, with
23,494,656 bytes peak working set. Build command, bound source/runtime receipts,
native results and the external safety observer are retained in
`build/real-moe-validation-20261027/consumer-contract-v1/`. The native interface is
`consumer-contract NEW_OUTPUT_JSON`; its recorded `COMPILE_LINK.json` gives the
explicit compiler/import-library command. It must use the private v4b CPU DLL,
and an existing output is refused. Run it only in a serialized authorized window.

The actual 4.2 GB OLMoE smoke was rejected by the unchanged parent capacity
precondition before a model process started. Its refusal remains in
`build/real-moe-validation-20261027/tap-smoke-v4b/`. By user choice, real-model
revalidation is deferred until memory is available. Therefore actual-model
sixteen-layer consumer coverage, original/private-runtime logit equivalence,
hybrid capture equivalence and complete quantized end-to-end cost coverage are
still unverified. The small-graph pass does not change those statuses.
