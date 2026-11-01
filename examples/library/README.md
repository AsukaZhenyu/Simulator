# Standalone library examples

Install from the repository root with `python -m pip install .`, then run:

```bash
python examples/library/quickstart.py --model dense --out example-dense
python examples/library/quickstart.py --model moe --out example-moe
python examples/library/phase_and_state.py --out example-phase-state
python examples/library/strategy_ir.py --out example-strategy-ir
python examples/library/request_pipeline.py --kind dense --out example-requests-dense
python examples/library/request_pipeline.py --kind moe --out example-requests-moe
```

The scripts can be copied outside the repository: they import the installed
library and never read fixtures or generated profile directories. Inputs are
fictional hardware, declared geometry, analytical costs, and explicitly labeled
synthetic MoE routes. These demonstrate semantics, not measured speedups.

Each output contains a persistent `project/` and detailed `result/`. Use new
directories. To replace the example model, hardware or workload with your own
declarations, start from the snippet in the
[top-level README](../../README.md#declare-your-own-experiment) and the docstrings
in `llm_simulator`.

`strategy_ir.py` adds readable module placement, layer-scoped weight preparation
and CE/SM transport, declarations-only inspection and the compiled IR. It writes
`declaration.json`, `inspection.json`, `explanation.txt` and detailed execution
evidence. The service rates are explicit analytical inputs; CE and SM share one
link and all modules share the runtime capacity ledger. Scope restrictions and
the checks that inspection intentionally leaves unproved are stated in
`tensor_mapping.python_frontend`.

`request_pipeline.py` adds independent arrivals, unequal Decode lengths,
request priorities and a charged all-layer P/D handoff. The MoE variant also
selects revealed experts by Token count and current CPU/GPU costs. It retains
one shared runtime and explicitly prices independent B1 kernels, with no fused
batching claim.

`shared_memory.py` declares one physical memory pool with
`Hardware.shared_memory(cpu=..., gpu=..., capacity_bytes=...)`. DRAM and VRAM
remain logical access views; shared references count one allocation. The result
reports `peak_shared_bytes` and zero-payload `SHARE` events. CPU/GPU compute
prices and memory-access costs are unchanged. No H2D/D2H bandwidth is required
for shared visibility; an explicitly requested real copy still requires a
transfer cost and a separate allocation. The zero-time visibility action is a
declared abstraction, not measured synchronization, vendor support, or an
automatic shared-bandwidth contention model.
