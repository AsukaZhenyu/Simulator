# Version 2 results, 2026-10-18

The corrected route-barrier protocol completed 48 independent processes:
CPU/CUDA × four cases × three profile and three heldout processes. Each process
had a separate validation pass, two warmups, and 31 complete traced-workload
samples. Source, executable, runtime and pre-heldout model barriers passed.

All numerical, shape, persistent-KV, layer-output lineage, route-barrier,
allocator-lifetime and real public ModelPlan audits passed. A separate offline
auditor additionally checked **every expert metadata and payload allocation**
against the barrier. It rejected 24 deliberate early-allocation corruptions per
backend, covering metadata and payload separately. The frozen collector/native
and prediction algorithms were not changed for this stronger independent audit.

Timing acceptance is **false for both backends** under the original gates. No
threshold or fitted formula was changed after heldout inspection. CPU had 23/24
profile stage keys above the within-process CV limit; CUDA had 24/24, and 12/24
CUDA stage keys exceeded the between-process CV limit. All CPU heldout errors
were <= 9.356%; CUDA had three heldout errors above 20% (22.103%, 20.397%,
20.075%). These data are not an accepted stable production cost profile.

| Case | CPU heldout absolute errors | CUDA heldout absolute errors | CPU tracked peak B | CUDA tracked peak B |
|---|---|---|---:|---:|
| Dense, two layers | 3.974%, 2.225%, 2.956% | 7.139%, 6.344%, 6.518% | 3,011,920 | 3,011,920 |
| MoE, active residency, changing/revisited routes | 3.274%, 3.050%, 4.913% | 4.386%, 3.097%, 8.438% | 3,846,608 | 3,846,992 |
| MoE, all residency, same route trace | 9.356%, 2.658%, 2.130% | 22.103%, 20.397%, 16.298% | 5,513,680 | 5,514,064 |
| Mixed Dense/MoE, active, single-token rounds | 1.858%, 0.367%, 0.712% | 18.866%, 16.350%, 20.075% | 3,425,232 | 3,425,616 |

Shapes are H=128, I=256, four query heads, two KV heads and F32. Homogeneous
cases have histories `0 -> 2 -> 3 -> 4 -> 5`; the mixed case has
`0 -> 1 -> 2 -> 3 -> 4`, respecting the existing public mixed hidden-layout
restriction. These are supplied activation traces with layer outputs feeding
the next layer, not language/token generation.

Direct active-MoE capacity trials passed on both backends. A budget one byte
below the measured tracked peak rejected before the crossing allocation; exact
peak and peak plus one byte completed, released every tracked byte, and observed
returning experts reload. These are ledger capacity checks. Full process peak,
CUDA private workspace and physical-fit proof remain unknown/false.

Artifacts relative to this directory:

- `formal-v2-cpu-20261018/` and `formal-v2-cuda-20261018/`: cases, every original
  stdout/stderr/raw record, source bindings, frozen profile-only models,
  generated public inputs and heldout reports.
- Each contains `independent-expert-lifetime-audit.json`, with
  `all_records_verified`, `metadata_and_payload_causality_pass` and
  `corruption_tests_pass` true; `timing_acceptance` false.
- `capacity-v2-cpu-20261018/` and `capacity-v2-cuda-20261018/`: baseline and
  below/exact/above-budget raw evidence and capacity reports.
- `archive-protocol-v2/`: exact collected Python/native source bytes and native
  build binding. The preserved executable/build is
  `../native/workload_accuracy/build-release-v2/`.
- `formal-{cpu,cuda}-20261018/`, `archive-protocol-v1/` and the original
  `build-release/` retain the rejected earlier route-before-attention protocol.
  Original raw/model/report bytes remain untouched; separate diagnostic-only
  sidecars state the causal rejection.

Re-run offline verification, including metadata and payload corruption gates:

```powershell
python -m profiling.workload_accuracy.tests.test_v2_evidence profiling/workload_accuracy/formal-v2-cpu-20261018
python -m profiling.workload_accuracy.tests.test_v2_evidence profiling/workload_accuracy/formal-v2-cuda-20261018
```

The independent auditor is outside the frozen collector source set and binds its
own SHA256 into its audit reports. It adds strict validation; it does not refit
or adjust the saved predictions.
