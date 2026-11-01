# Cross-scenario strategy expression evidence

This analytical study compares scoped CPU/GPU execution choices and demand or
spare static-weight preparation through the public Python frontend, shared IR,
compiler and modeling runtime. It does not collect native timings or fit costs.

The five fixed structures are a Dense macro chain, refined Attention, two-request
pipeline, host P/D handoff, and opaque trace-routed MoE. All use H=8, I=16, one
layer, explicit F32 ports, fixed persistent weight cache and 1 MiB per domain.
Kernel costs are declared witness values, including CPU/GPU FFN120/60 µs.
Shared-PCIe payload costs are explicitly1 or50 ns/B (1GB/s or20MB/s), not measured
hardware. Policies prepare static weights only; an expert demand cannot read
future routes. Compiler output is checked against a separately authored operand
DAG, independent physical ledger and logical numerical interpretation.

## Run

From the Simulator directory, with repository dependencies available:

```powershell
python -m unittest experiments.ir_value.test_reference
python -m experiments.ir_value.campaign --out build/NEW_EMPTY_CAMPAIGN
python -m experiments.ir_value.exact_oracle --directory build/NEW_EMPTY_CAMPAIGN --certify
python -m experiments.ir_value.report --directory build/NEW_EMPTY_CAMPAIGN
```

The exact bound certificate requires NumPy and SciPy. The broad reference family
uses16 within-machine tie-rule pairs and is not an exhaustive schedule oracle.
For tiny Dense, valid precedence/resource lower bounds and a feasible independent
schedule coincide for every candidate's global comparison, certifying140100ns
and143236ns across the declared placements/storage choices and all nonpreemptive
orders of the independent fixed inventory. This is not all-IR completeness.

## Retained campaign

The authoritative final evidence is `build/ir-value-20261029-final-evidence/`. Its machine data
is `paper-data.json`, `paper-table.csv`, and the readable `REPORT_zh.html`.

The syntactic target grid has928 points.800 admitted points pass independently
checked operands, placement, transfer inventory/cost, dependencies, capacity,
lifetime, causal expert visibility and declared numerical semantics.128 P/D
points request GPU Decode Attention while fixing the handoff reader endpoint at
DRAM, and are rejected by the existing typed contract. They remain in the grid
and report; physically possible host handoff followed by reupload is a broader
strategy outside this particular contract.

The original base campaign is `build/ir-value-20261029-final-v2/`. Its first MoE
reference accidentally priced request-input preparation at32ns instead of the
RequestFlow contract's36ns (hidden32B plus one I32 position). All128 MoE points
hit that explicit assertion. Original source and failure artifacts remain.
`build/ir-value-20261029-moe-price-contract-v3/` freezes and reruns only those128
points with the corrected independent contract; all pass. Production code and
profile parameters were unchanged. The merged file records both protocol/report
hashes and all1056 raw attempts, instead of hiding or deleting the correction.

Independent audit also corrected shared-weight demand readiness: any ready
consumer may trigger the common COPY, rather than the dictionary-first request.
The corrected references are regenerated from declared workload algebra and
choices, with an OR of alternative consumer readiness sets. All800 actual traces
satisfy the corrected reference obligations. Actual executions are not rerun or
changed. The prior references remain in the earlier campaigns.

The final evidence can be regenerated from retained campaigns with:

```powershell
python -m experiments.ir_value.merge_evidence --base build/ir-value-20261029-final-v2 --correction build/ir-value-20261029-moe-price-contract-v3 --out build/NEW_EMPTY_MERGE
python -m experiments.ir_value.reference_correction --source build/NEW_EMPTY_MERGE --out build/NEW_EMPTY_FINAL
python -m experiments.ir_value.report --directory build/NEW_EMPTY_FINAL
```

The main quantity is latency reduction from jointly optimizing execution and
preparation compared with reoptimizing one while fixing the other. A28.3949%
latency reduction is a1.39655× speedup; those two percentages are not interchangeable.
The broad reference-family selection regret is distinct from the signed default
lowering gap. Individual gap extrema are not uniformly small, so this evidence
does not justify claiming that hidden implementation details never matter.
