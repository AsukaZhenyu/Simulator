# Real MoE workload collection

This additive tool collects actual GGUF model requests, expert selections and
normal runtime response times. It is not an end-to-end latency predictor and does
not certify the simulator merely because a model run succeeds.

Execution revision v2 uses OLMoE-1B-7B-0924-Instruct Q4_K_M from the public
bartowski conversion: 16 layers, 64 experts, top-8, 4,213,513,024 bytes. The official
AllenAI config and prompt template, conversion revision, publisher LFS SHA256,
downloaded content hash and exact byte count are retained. `acquire_model.py`
only promotes a download after its publisher hash matches. Qwen collection v1
is preserved as tokenizer-only evidence: its 20.50 GiB GGUF exceeds this host's
15.22 GiB physical RAM, and it is not described as a completed warm-resident run.
Mixed-quantized kernels, Attention, routing/aggregation, Embedding and LM head
still require explicit model/profile coverage before prediction accuracy can be
claimed. F32 toy profiles are not substitutes.

## Native execution boundaries

`native/real_moe_probe.cpp` is a separate executable linked against explicitly
matching llama/GGML libraries. It does not edit the upstream source tree or call
the old debug callback. Its modes are:

- `tokenize-only`: load vocabulary metadata only, normalize public candidates to
  actual model token IDs. No model weights or inference context are allocated.
- `generate`: normal greedy generation with a maximum output count and EOS stop.
- `replay`: fixed prompt/output token sequence; normal forward/output masks.
- `trace`: the same forward inventory with a routing-only callback. Timing from
  this mode is diagnostic, never the bare performance reference.

Normal prefill marks only the final prompt token for logits. Earlier pruned
last-layer rows are not invented. No final sampled token is decoded merely to
complete a routing trace. Each route stores the real forward index, phase,
absolute positions, layer and expert IDs using actual tensor strides. The
callback rejects all other tensors at `ask`, before any readback.

Bare timing contains native loop and sampler work. Core decode timing separately
contains only each completed forward. Prompt tokenization, model load, state
reset and optional full-request warmup are outside these target intervals.
Prefill chunk host-return stamps are not mislabeled as device completion;
complete prefill timing ends at the final required synchronization. JSON output
and forward-record construction are performed after the target ends.

Both CPU and CPU-experts/GPU-other placement are explicit. A missing CUDA0 or
failed allocation rejects, without CPU fallback. The tool records every model
tensor's actual buffer and dtype, not just requested CLI placement. It bypasses
the common automatic-fit helper. The parent binds the full model content hash;
the executable also checks byte count. Host/GPU free-memory preconditions remain
explicit. CPU process affinity is enforced before model initialization.

## Public request corpus

`dataset.py snapshot` downloads a bounded snapshot of GSM8K, MBPP and UltraChat.
GSM8K/MBPP raw files use pinned repository commits. UltraChat uses a bounded rows
API snapshot and records the requested revision, URL, headers and exact raw hash;
the API cache revision is explicitly not independently proven. Source data are
kept under their original licenses; the repository code license does not replace
dataset attribution.

Code/math long prompts use distinct public few-shot examples rather than repeated
padding. Examples and target questions are disjoint. Dialogue uses genuine
conversation prefixes. The vocab-only tokenizer determines length selection:
three domains × two length windows × three different source targets = eighteen
requests. One per cell calibrates, two are held out. Missing length coverage
raises an error; text is not silently padded. All content and source IDs are
frozen before any performance-based selection.

Revision v2 rebuilds the same public source candidates with the pinned official
checkpoint Jinja template, BOS/EOS and generation prompt. It uses the same
trim/lstrip-block settings as Hugging Face chat templating. Qwen ChatML prompts
and token IDs are not reused. The actual GGUF template and tokenizer metadata
are also recorded for comparison. Model architecture, layer count, expert count
and top-k are a generic explicit contract verified against loaded GGUF metadata.

Revision v3 changes the execution boundary to an explicitly owned persistent
CPU threadpool shared by Prefill and Decode, attached before any warmup. This
matches the persistent-pool convention used by the conventional CLI and the
subgraph probes. The previous v2 artifacts remain tokenizer-only evidence. The
`reuse_suite.py` gate permits token reuse only when the exact model, official
template/config, runtime receipts, native tokenizer function and all eighteen
selected token sequences still match. Threadpool setup is excluded from target
timing; the policy appears in requests, native results and provenance.

## Collection

Build the independent CMake target with explicit source/build paths. Then:

```text
python -m profiling.real_moe_validation.dataset snapshot --out NEW_CORPUS
python -m profiling.real_moe_validation.acquire_model --out NEW_MODEL_DIR
python -m profiling.real_moe_validation.dataset rebuild --snapshot CORPUS --tokenizer-config NEW_MODEL_DIR/original-tokenizer_config.json --out NEW_CANDIDATES.json
python -m profiling.real_moe_validation.run prepare --out NEW_RUN --binary PROBE --model MODEL.gguf --runtime-dir MATCHING_DLL_DIR --acquisition-proof NEW_MODEL_DIR/acquisition-proof.json
python -m profiling.real_moe_validation.run make-request --out NEW_RUN --inputs CANDIDATES_OR_SUITE.json --request-output NEW_REQUEST.json --mode tokenize-only --placement cpu --warmup-replays 0
python -m profiling.real_moe_validation.run collect --out NEW_RUN --job-id UNIQUE --request REQUEST.json
python -m profiling.real_moe_validation.dataset select --tokenized TOKENIZER_RESULT.json --out NEW_SUITE.json
```

`make-request --inputs GENERATE_RESULT.json --mode trace|replay` preserves the
exact actually generated token stream. `--ids` selects a fixed comma-separated
subset for an initial pilot. The `request(manifest, ...)` helper is also available.
Root orchestration
chooses acquisition windows; the collector shares the global measurement lock.
Every model hash is rechecked before a child starts. All native failures,
stderr, timeouts and exit evidence are retained without retries. Jobs never
overwrite old results. A route/normal-output mismatch is a reported validation
failure, not a reason to force a convenient token sequence without disclosure.

Normal per-placement generate/trace/replay outputs must be reconciled before
using the data for strategy ranking. Two placements can generate different
expert sequences because of numerical differences. A full route file belongs to
the modeling workload; Mapping still sees routes only at causal disclosure
boundaries. Unknown future experts cannot drive a cache or prefetch policy.

## Real quantized subgraph calibration

`inventory.py` reads the GGUF directory without materializing weight payloads.
The current model has 195 tensors: 97 Q4_K, 17 Q6_K and 81 F32. Its Q4_K_M label
does not mean every matrix has the same dtype: expert gate/up are Q4_K, while
expert down and V projection vary between Q4_K and Q6_K by layer. A layer's three
expert matrices occupy 216–249 MiB. The directory and coverage audit retain all
individual tensor dtypes, shapes, byte counts and offsets.

`quantized_native/quantized_probe.cpp` is an independent GGML-only target. It
reads actual packed matrix bytes from the pinned GGUF, without dequantizing or
substituting F32 weights. It currently measures:

- A regular resident Q4_K/Q6_K/F32 `MUL_MAT` on explicit CPU or CUDA0.
- A homogeneous CPU MoE FFN from the post-FFN-norm input to the MoE output:
  actual F32 router, softmax/top-k, three vectorized `MUL_MAT_ID` operations,
  fused SwiGLU, selected probabilities, expert views and seven ordered adds.

The FFN graph follows the checked OLMoE source topology and deliberately avoids
the old F32 per-expert grouped implementation. A metadata-only self-test uses
the same graph factory for q=1 and q=256, checks the 27-node graph and output
shape, and allocates no packed weights or activations. It never computes a graph.

CPU threadpool lifetime and CUDA graph policy are explicit. Profile warmup is
at least five calls and 500 ms, outside the target samples. The two cache contexts
are `isolated_warm` and an explicit 512 MiB CPU memory-touch predecessor. The
latter is a declared conditioning operation, not proof of physical cache
residency. Its time is excluded from the target interval. Cache context is part
of the price coordinate; repeated hot prices cannot silently become cold
multilayer prices. An external F32 input file is content-bound but does not by
itself prove that its values came from a real model activation.

`quantized.py` serializes acquisition through the global measurement lock,
rechecks the full model hash, preserves failed attempts and validates actual
quantized types, routing and graph topology. Profiling estimates the scoped
sample mean and checks within-process CV. Export creates a `SteadyActionCost`;
the existing model-owned `compile_steady_actions` performs explainable serial
composition and resource-timeline evaluation. The exporter accepts no whole
model label as a component price and adds no fitted whole-model multiplier.

These prices currently do **not** establish complete OLMoE prediction coverage.
Attention/F16 padded KV, embedding, normalization and runtime transitions still
need matching boundaries. In particular, a homogeneous CPU FFN cannot price a
GPU router / CPU expert / GPU aggregation partition without observing its
actual cuts and copies. Original-model activation/port equivalence and
cross-process stability are also separate gates. Unknown CUDA private workspace
remains unknown. Successful collection of the real-model corpus and successful
simulator accuracy validation are reported as separate outcomes.

```text
python -m profiling.real_moe_validation.inventory --manifest COLLECTION/manifest.json --gguf-python-dir MATCHING_GGUF_PY --out NEW_AUDIT
python -m profiling.real_moe_validation.quantized prepare --out NEW_PROFILES --collection COLLECTION --directory NEW_AUDIT/directory.json --binary QUANTIZED_PROBE
python -m profiling.real_moe_validation.quantized collect --out NEW_PROFILES --job-id UNIQUE --request PREPARED_REQUEST.json
python -m profiling.real_moe_validation.quantized export --out NEW_PRICE.json --result RESULT.json --execution EXECUTION.json
```

Collection order is smoke calibration requests, six calibration requests and
independent subgraph profiles, model/profile freeze, then twelve heldout requests
with three fresh bare processes per placement. Trace results remain diagnostic.
Hardware collection must wait for the orchestration window; preparation and a
passing metadata self-test are not measurement results.

## Postcollection routes and causal input

`postcollection.py` independently checks effective batch/microbatch/sequence
values from the actual runtime log, every forward index and every required
layer/token routing row. For a 1029-token prompt, the normal outer calls are
256+256+256+256+5. In OLMoE, final-layer FFN rows are pruned after Attention:
the first four chunks have no final-layer FFN rows, and the final chunk has one
such row. Earlier layers retain all submitted rows. Decode must cover all sixteen
layers at each actual forwarded token; the final sampled output is not invented
as another forward. Missing settings yield unknown, mismatched observed rows
fail, and raw files remain unchanged. q256, q-tail and q1 are distinct shapes.

After an actual normal generate job completes, `prepare-pair` writes replay and
trace requests using its exact prompt/output token IDs and records executable
argument lists in `PAIR_PLAN.json`. It requires the completed acquisition receipt
and source/model binding. These hardware jobs still require the serial window.

```text
python -m profiling.real_moe_validation.postcollection prepare-pair --collection COLLECTION --bare GENERATE_RESULT.json --out NEW_PAIR
python -m profiling.real_moe_validation.postcollection audit --trace TRACE_RESULT.json --bare GENERATE_RESULT.json --out NEW_AUDIT.json
python -m profiling.real_moe_validation.route_analysis --trace TRACE_RESULT.json --bare GENERATE_RESULT.json --out NEW_ANALYSIS
```

`route_analysis.py` accepts completed actual native traces and passes them through
the independent audit before writing empirical tables. It reports per-layer
expert frequencies, grouped request domains/lengths/splits, and Decode adjacent
top-8 reuse. Frequency denominators are actual observed token-layer rows; the
pruned final layer is never assigned all prompt tokens. Reuse is
`|top8(t) intersect top8(t-1)| / 8`, within the same request and layer at adjacent
positions. With no adjacent pair it remains unknown, rather than becoming zero.

Complete observed Decode events are exported under `modeling-private/` as one
Decode-only container and sixteen existing `m2_moe_route_fragment_v1` entries per
forward. No scores, initial state, costs or quantized physical layer are invented.
The old `moe_inference_routes_v1` bundle requires complete Prefill at every layer,
so normally pruned Prefill cannot be forced into it. The route fragments pass the
existing M2 route parser; `m2_invocation_patch` returns the same route/request
fields as the established invocation-slice adapter. Matching real history,
layer identity and qualified costs remain the modeling driver's responsibility.

The full container and expert-frequency summaries are privileged researcher/model
inputs. Mapping must receive only `CausalM2Session.observe()` or
`GlobalM2Session.observe()`. Existing model-side visibility reveals a layer's
routes after its router completes. The exporter requires `causal_replay` and
does not provide the full future trace or privileged loader to a policy. Route
import compatibility does not certify quantized performance simulation.

## Hybrid numerical diagnosis and the private MMID ordering fix

`numerical_v6.py` / `numerical_collect.py` export the complete logits already
copied to host memory. They add no GPU read or synchronization for that export.
Compare vectors only when actual token, position, sequence and output-mask
prefixes match. A changed generated token stream is not a timing error and does
not justify comparing later rows with different inputs. Diagnostic observer
timings are not bare performance references.

The pinned CUDA MMID helper has a shared-memory ordering hazard between a lane's
store and another lane's read. `mmid_primitive.py` extracts the unchanged helper
and compares it with a variant containing one full-warp `__syncwarp`. Integer
oracle checks and Compute Sanitizer racecheck are separate checks: correct
outputs alone do not establish race freedom. A sanitizer attachment failure is
unsupported tool evidence, never a clean racecheck result. The bounded
`mmid_attach_check.py` records a separate attachment diagnostic without changing
registry or driver settings or replacing earlier failed attempts.

`mmid_runtime_fix.py` prepares and builds a **new private runtime**. It replaces
only the MMID object, preserves the pinned math/graph flags, and verifies all
original source/object/DLL hashes. It does not edit the original runtime or
silently promote the candidate. `mmid_numerical.py` prepares an independent
original-versus-fixed, generate/replay/capture comparison with full vectors,
actual-input checks, graph fingerprints and consumer route inventories. A
separate root authorization and the existing capacity guard remain required.

Eliminating this helper's reported race does not by itself prove the cause of
the earlier whole-model logit mismatch, certify graph-replay paths not exercised
by the check, or qualify whole-model performance accuracy.

## Actual-runtime performance windows

`runtime_windows.py` builds a new private base/CPU observation runtime and an
independent host executable. It reuses the already verified private MMID-fence
CUDA DLL. The original source tree, original binaries and earlier observations
remain unchanged. The build verifies that the new backend object is actually
linked and that both observer ABI exports exist before declaring completion.

The observer writes QPC stamps and actual node/shape/type metadata into bounded
preallocated arrays. CPU worker zero marks each ordinary node dispatch; the next
dispatch follows the existing worker barrier, and the synchronous CPU graph
return closes the last node. The observer adds no CPU barrier or GPU
synchronization. GPU graph submissions alone are not completion events. Existing
dependency synchronizations and final request joins close runtime bridges, whose
coverage explicitly lists the submitted GPU nodes. Existing set/get API byte
counts retain actual selected-expert transfer requests and padding. Those counts
are not presented as independent PCIe bus measurements.

The resulting windows partition the complete request host clock. Node prices,
runtime bridges, host preparation and sampling are mutually exclusive; device
intervals and copy diagnostics are never added a second time. Original
multilayer predecessors, real quantized weights, persistent CPU pools and the
default CUDA lifecycle are retained. An entire CPU forward cannot masquerade
as one CPU-node price. Metadata-only software fixtures test conservation,
coverage and explicit domain checks without executing inference.

`runtime_window_campaign.py` prepares disjoint calibration/held-out public
requests and uses the unchanged host/GPU capacity guards and global measurement
lock. Window prices come only from calibration requests. The held-out structural
observer supplies an operator/copy workload inventory; a separate export strips
all elapsed values before forecasting. Mapping is not given this private future
route/workload inventory. Prediction prices and inventories are frozen before
fresh unrecorded timing references execute. The held-out normal token trajectory
and final full-logit hash must match the structural reference.

`runtime_window_predict.py` estimates nonnegative affine response parameters in
profiling. `modeling.performance.moe_execution` owns their evaluation and serial
composition using the existing Graph/evaluator: intercept, query-token work,
query-by-padded-KV work, and requested H2D bytes. Unsupported coordinates remain
unsupported; explicit extrapolation is labeled. No complete held-out time is
used for parameter fitting or an overall correction multiplier. The principal
metrics are complete request, completed Prefill, and the post-Prefill Decode
interval including greedy sampling. Component windows are implementation
response coordinates, not portable isolated-kernel or physical-bandwidth laws.

Preparing or building this path is not an accuracy result. The root orchestrator
must grant an exclusive hardware window, and independently measured errors are
required before any performance qualification claim.
# Mechanism-scoped price revision (v9)

The v8 four-column affine catalog is retained as historical evidence. New
calibration declares `WindowWorkSpec` from actual recorded operators and copy
calls before fitting. The same completed-window model is used for every scope;
each work declaration permits only physically supported response terms.

| Actual recorded work | Permitted response terms |
| --- | --- |
| Ordinary or expert matrix operation | Fixed invocation and actual local query rows |
| Identified RMS norm or expert SwiGLU | Fixed invocation and actual local rows from its output shape |
| Attention QK, PV or softmax | Fixed invocation, query rows and actual query-by-KV work |
| GPU runtime bridge | Terms justified by its actual covered operators; byte term only with observed copy calls |
| Host window with observed H2D calls | Fixed invocation and observed requested bytes; other host work remains unidentified |
| Unclassified host, unknown compute or sampler tail | Fixed observed response, explicit unknown mechanisms and conservative dimension-domain guards |

KV length is not a proxy for expert-cache state, CPU availability, ordinary
matrix work or an unobserved OS effect. Local row coordinates preserve the
normal final-layer output pruning: a prompt forward may have Attention q=125
while its final expert FFN, SwiGLU and output head have q=1. A window containing
an identifiable attention-mask preparation stage could declare q-by-KV work,
but the current native boundaries do not identify that stage separately.

The active-set estimator reads only the original calibration-profile raw
windows. Prohibited coefficients are exactly zero, not merely regularized.
Prices retain sample counts, permitted design rank, active terms, unknown
mechanisms and a common 1e-9 ns coefficient grid. A constant response for
unknown work does not certify independence from all input shapes.

For the new independent check, preparation selects the lexicographically
second heldout ID in each fixed domain/length stratum (`--heldout-index 1`).
It rejects overlap in IDs and complete prompt-token sequences with the
calibration set or the previously diagnosed first heldout stratum. Selection
does not inspect any latency. `--adopt-calibration-only` retains the four old
calibration-profile/control jobs and requires new structural-inventory jobs.
The previous six heldout requests, their timings and numerical failures remain
unchanged as historical diagnostics.

The sequence is: prepare/freeze sources, fit and freeze calibration prices,
collect two new structural inventories, freeze forecasts with explicit domain
or extrapolation flags, then collect four fresh bare processes and evaluate.
The unchanged endpoint gate compares prompt/output tokens and the final full
logit digest within each placement. It neither proves all intermediate routes
identical nor relaxes an existing numerical failure. No native, H2D, compiler
or CPU-state mechanism changes are part of this price revision.

`runtime_window_predict posthoc --out NEW_COHORT --baseline OLD_V8_COHORT`
compares newly frozen prices on the previous twelve, already diagnosed
conditions. It strips old observer durations before prediction and retains the
original numerical exclusions. `OLD_CASE_POSTHOC.json` is explicitly a
known-case diagnostic; its results do not select coefficients or replace the
disjoint-index independent evaluation.

