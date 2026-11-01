# Finite single-CPU/GPU serving semantics acceptance

This extends the original [38-case IR baseline](../ir_acceptance/README.md)
without changing its manifest, case meanings or denominators. The new fixed
manifest contains 16 requirements: 6 mechanism cases, 4 combinations,
4 explicit negative cases and 2 causal-prefix contrasts.

```bash
python experiments/serving_acceptance/run.py --out build/serving-acceptance
python -m unittest discover -s tests -p test_serving_acceptance.py -v
```

Use a new or empty output directory. The research runner verifies all source
bytes and the retained local accuracy-evidence freeze before/after the run.
It preserves every pass/failure/unsupported result and imported-source identity.
Any source drift invalidates a campaign even when its individual cases pass.

The portable unittest executes the same freshly generated semantic cases and
source-immutability checks without opening local native profiling results.
It can run in a clean source checkout. It does not claim to verify the separate
accuracy freeze. The core library and standalone examples also need no profile.

Each adapter writes independent expectations before compilation and uses the
existing public API, IR, compiler and shared modeling runtime. Oracles check:

- Public arrivals/retirement, no pre-arrival input allocation, actual retired-KV
  EVICT bytes while another request continues, and asynchronous per-request
  layer/round dependencies.
- Positive CPU-Attention/GPU-FFN overlap, complete physical capacity/cache
  bookkeeping and exact shared weight identity-to-byte inventories.
- Every layer's Prefill KV and final hidden output as real first-Decode gate
  operands, charged transfer bytes and rejection with exactly one KV missing.
- Router-revealed expert execution/order, actual cache reuse and charged
  cross-layer prefetch under one ledger.
- Identical public observation/decision prefixes with changed private future
  arrival/prompt/Decode lengths, or changed future expert routes; both compared
  workloads must finish.

The batch-axis case separately verifies an explicit invocation-batch-2,
kernel-batch-1 Dense contract. It never divides a measured whole-batch duration.
Request streams use independently priced B1 kernels, not fused dynamic batching.

Prices are explicit synthetic inputs, not measurements. This is not a blind
held-out study, hardware validation, universal expressiveness claim, distributed
P/D server, or multi-GPU implementation. The
[design and public example](../../docs/SINGLE_CPU_GPU_REQUEST_FLOW_20261023.md)
describe supported choices and remaining mechanisms.
