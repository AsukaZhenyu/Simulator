# Physical expert-policy validation

This additive executor evaluates the **routed F32 expert-FFN boundary**, with
actual CPU/GPU arithmetic and actual copies into a physically bounded GPU
weight-slot pool. It is separate from the real OLMoE end-to-end trace collector.
It is not a reproduction of a whole published system, a router benchmark, or
an end-to-end language-model benchmark. The declared grouped expert operation
has host inputs and host outputs; its GPU cost therefore includes activation
upload/download. The compiler must not charge these transfers a second time.

## Physical contract

The program schema is `moe-expert-policy-native-1`. Explicit `config` contains
`hidden`, `intermediate`, `experts`, `gpu_slots`, `threads`, `cuda_device`.
`resource_guards` contains `host_reserve_bytes` and `gpu_reserve_bytes`.
`seed` selects reproducible F32 weights; input arrays are supplied per step.
Each step contains `input[q][hidden]`, `routes[q][top_k]`, and matching `scores`
that sum to one per row. An expert receives exactly the token rows selecting it.

The native process consumes already compiled actions in their exact order:

* `LOAD {expert, slot}` copies gate/up/down weights from DRAM into one GPU slot.
* `RELEASE {expert, slot}` releases ownership of that slot for future reuse.
* `EXPERT {step, expert, device, slot?}` gathers current token rows, executes
  gate/up/SwiGLU/down, and materializes host output. GPU activation transfers and
  completion synchronization are inside this operation's physical boundary.
* `COMBINE {step}` sums score-weighted expert results and adds input residual once.

All experts have CPU backing. GPU weight backing exists for **K slots only**,
not for every expert. The pool is allocated before timing; releasing a slot does
not call `cudaFree`. Native checks reject overwriting a live slot, computing an
unresident GPU expert, duplicate expert work, or combining incomplete outputs.
Graph workspaces and ports are reported separately. CUDA-private scratch remains
unknown, so the reported weight budget is not a complete process-VRAM claim.

## Measurement and oracle

Invoke `moe-expert-policy-probe PROGRAM.json REQUEST.json OUTPUT.json` with
`mode`, `warmup`, `samples`, and `repeats` in the request. `bare` times complete
serial action sequences; logical cache reset is outside each repeat timer, but
all cold-cache LOAD copies are inside. `stamped` adds per-action stamps and is a
separate diagnostic. `validate` executes without collecting timing samples.
`profile` also requires `target_action_index`: its exact prefix executes outside
the timer, followed by hot repeated target actions. Such profiles test hot-action
composition, not all memory histories. Measured target latency must never be
used to calibrate its own prediction. No event overlap is claimed.

Construction, DLL startup, weight generation, graph creation, final output reads,
reference checking, and JSON I/O are outside sequence timing. Bare repeats keep
the same supplied inputs; there is no autoregressive generation claim. CPU uses
an owned persistent thread pool; CUDA graphs and TF32 are explicitly disabled.

`reference.py` independently evaluates SwiGLU and weighted residual with NumPy
float64 arithmetic on declared F32 weights/inputs. It checks all output elements,
absolute error, relative L2 error, exact compiled action equality, weight/activation
copy payload bytes, and bounded GPU slots. Tests do not require a GPU.

## Build

Provide existing matching GGML headers/import libraries, an explicit strict-F32
runtime, and installed CUDA runtime. No download or upstream modification occurs.

```text
python -m profiling.moe_policy_validation.build_native --out NEW_BUILD --llama-root LLAMA_SOURCE --ggml-build LLAMA_BUILD --runtime-dir STRICT_F32_RUNTIME
python -m unittest profiling.moe_policy_validation.test_reference
```

The reusable build helper freezes source, binary, runtime DLL and build-cache
hashes. Put the selected GGML and CUDA runtime directories on the child process
PATH. GPU absence is an error, never an implicit CPU substitution. Execute
physical acquisitions only while holding the repository measurement lock.

## Frozen strategy campaigns

The 2026-10-29 experiment fixes H=256, I=512, 8 F32 experts, top-2,
q=1/32, K=2/4 GPU slots and two auditable eight-call route sequences.
Five labels form a CPU reference plus GPU-only/current-cost crossed with
release-after-use/LRU. `compile_expert_program(..., retention="release_after_use")`
declares completed recoverable weights to the existing common release policy;
it does not treat `cache.rule="none"` as release-after-use.

The original `campaign.py` uses repeated hot isolated-action profiles. The
follow-up `lifecycle_campaign.py` measures six complete calibration sequences
per q/K (CPU/GPU-release/GPU-LRU crossed with the two route patterns), two fresh
processes each and 20 stamped samples. It averages per-process category means;
the shared modeling evaluator composes LOAD/RELEASE/EXPERT/COMBINE prices.
Parameters and predictions freeze before independent bare validation: three
fresh processes, 10 complete samples each, per nominal condition. It adds no
correction fitted to validation and no implicit OS scheduling model.

```text
python -m profiling.moe_policy_validation.campaign --out NEW_HOT_CAMPAIGN --build BUILD_COMPLETE.json
python -m profiling.moe_policy_validation.lifecycle_campaign --out NEW_LIFECYCLE_CAMPAIGN --build BUILD_COMPLETE.json
python -m profiling.moe_policy_validation.analyze NEW_LIFECYCLE_CAMPAIGN
python -m unittest profiling.moe_policy_validation.test_bridge profiling.moe_policy_validation.test_reference profiling.moe_policy_validation.test_response_model mapping.tests.test_composition_policy
```

`FINAL_ANALYSIS.json` is the reporting table: exact native-program aliases are
pooled before performance ranking. Forty nominal labels reduce to 24 distinct
executable conditions in this campaign. Their 120 fresh native processes yield
3/6/9 independent process means per unique condition. A zero observed choice
regret is a point-estimate statement in this finite candidate set; confidence
intervals and remaining prediction errors must still be shown. In particular,
q=1 current-cost chooses CPU, while q=32 chooses GPU; identical P2/P4 programs
cannot establish a performance benefit of one label over the other. No causal
prefetch or asynchronous overlap performance is measured by these five labels.

All raw calibration, freeze, validation, source snapshots and independent
numerical reference artifacts remain in the selected new output directory.
`analyze.py` also checks declared LOAD counts, future-route prefix invariance
and explicit negative cases. These are evidence about separate mechanisms,
not a publication quality score.
