# Same-process single-layer timing diagnosis

This diagnostic completed **24/24 declared native processes** and four blocks
per process. Every block passed the independent full-port numerical checks at
collection. The records retain all 96 blocks and every invocation; no observation
was removed. These results are diagnostic controls, not parameter profiles,
heldout accuracy acceptance, or a validated strategy ranking.

The two F32 Dense programs are H128/q16/history0/FFN cut128 and
H256/q32/history0/FFN cut252. Each control has three fresh processes per geometry
and threadpool setting (`default` or `poll=0`). Within a process, one Executor,
program, numerical seed and CPU pool execute predeclared ABBA or BAAB blocks.
Every block repeats 30 warmup invocations and retains 101 timed observations.
CUDA graphs remain disabled; completed-work synchronization is unchanged.
User applications, affinity, clocks and power settings were not modified.

## Instrumentation does not provide a universal correction

Here A is complete serial replay with only invocation clocks; B adds per-action
POD clocks. JSON conversion and allocation occur after each timed block. Each
process ratio is the B label median divided by the A label median. Each label
median gives the two blocks equal weight. The final column is the median of the
three process ratios, not a fitted execution-time parameter.

| Geometry / pool | Three same-process B/A ratios | Median ratio |
|---|---|---:|
| H128 / default | 0.9744, 1.0398, 0.9571 | 0.9744 |
| H128 / poll=0 | 0.9579, 0.94994, 1.00184 | 0.9579 |
| H256 / default | 0.6078, 1.0475, 1.0410 | 1.0410 |
| H256 / poll=0 | 0.8893, 1.0131, 0.9707 | 0.9707 |

All four group medians are within 5% of one, while **9/12 individual process
ratios** are within that interval. H128/poll=0's 0.94994 ratio is outside the
interval; it is not rounded into a pass. This reference comparison does not
turn the diagnostic into a formal accepted calibration.

The earlier profile campaign's H128 context/bare cohort difference of 26--28%
does not reproduce as a constant same-process timing-instrumentation penalty.
Several ratios are below one. An intrinsic clock-reading cost cannot be
estimated by treating such a reduction as a negative overhead: drift and
execution-state changes remain in the comparison.

The H256/default process with ratio 0.6078 illustrates this limit. Its two
identically labelled bare blocks have medians **14.4979 and 6.1682 ms**; the two
instrumented blocks have medians 5.9663 and 6.5944 ms. Pool, program and PID stay
the same. Averaging the bare blocks does not establish a stable bare reference.
No observation or first block is discarded.

## A fixed context multiplier is also insufficient

Here A repeats only the exact GPU Attention PV graph, whereas B measures that
same target inside the complete real CPU/GPU/copy action sequence. B's metric is
the selected action duration, not the whole sequence. The PV graph includes its
declared layout/materialization and KV-output work; it is not a bare GEMM.

| Geometry / pool | Three same-process context/isolated ratios | Median ratio |
|---|---|---:|
| H128 / default | 0.5213, 1.0320, 1.0137 | 1.0137 |
| H128 / poll=0 | 1.0179, 0.5406, 1.0010 | 1.0010 |
| H256 / default | 1.1731, 1.8622, 1.0358 | 1.1731 |
| H256 / poll=0 | 1.0898, 1.0629, 1.0943 | 1.0898 |

Three observations constrain the explanation:

- In one H128/default ABBA process, the first and last isolated A blocks measure
  **4.8392 and 1.4932 ms**, with within-block CVs **7.00% and 5.26%**. Each block
  looks internally consistent, yet the same action changes by over three times
  within the same PID.
- An H128/poll=0 ABBA process repeats the pattern: isolated A changes from
  **4.7863 to 1.4394 ms**, with CVs **4.73% and 5.89%**. Setting polling to zero
  therefore does not eliminate the observed state changes.
- In one H256/default ABBA process, the two isolated A blocks measure
  2.7757/2.8687 ms, while the two identically labelled context B blocks measure
  **2.9878/7.5231 ms**. The B-block CVs are 54.60%/53.25%. A static context label
  does not identify one stable service time.

The hardware result is stronger than merely observing different processes:
**changing PID is not required for the large latency regime changes.** However,
the controls do not identify clocks, power, driver scheduling, cache state,
threadpool polling, kernel selection or another mechanism as the cause.
ABBA/BAAB counterbalancing does not restore the device/OS state between blocks.
With three repetitions, each group has two runs of one order and one of the
other; the data do not support a separate robust order-effect estimate.

## Variation remains visible

The table counts blocks with invocation CV above 15%, retained as a descriptive
reference to the original stability threshold. Every row has six A and six B
blocks; the comparisons use different metrics across the two controls.

| Geometry / pool | Control | A blocks above 15% | B blocks above 15% |
|---|---|---:|---:|
| H128 / default | Context versus isolated PV | 1/6 | 5/6 |
| H128 / poll=0 | Context versus isolated PV | 0/6 | 4/6 |
| H256 / default | Context versus isolated PV | 1/6 | 6/6 |
| H256 / poll=0 | Context versus isolated PV | 2/6 | 6/6 |
| H128 / default | Instrumented versus bare whole replay | 6/6 | 4/6 |
| H128 / poll=0 | Instrumented versus bare whole replay | 4/6 | 4/6 |
| H256 / default | Instrumented versus bare whole replay | 5/6 | 4/6 |
| H256 / poll=0 | Instrumented versus bare whole replay | 5/6 | 3/6 |

Overall **60/96 blocks** exceed that reference CV. The three H256/poll=0 PV
ratios cluster at 1.063--1.094, but all six corresponding context blocks have
CVs 58.73--61.82%. A repeatable ratio of block medians does not establish stable
per-invocation parameters or accepted prediction accuracy.

Windows process/main-thread CPU counters are retained as sampling-envelope
observations. Non-main CPU work includes all native auxiliary threads; it does
not identify polling or GPU kernel activity. Both polling configurations still
show substantial auxiliary CPU activity in mixed execution.

## Synchronized state observations constrain simple explanations

The [contained-query evidence](data/single-layer-paired-contained-telemetry-20261018.json)
joins the saved QPC sampling envelopes to the same process's telemetry stream.
The **entire combined GPU/CPU query** must lie within one envelope. CPU busy time
additionally requires its complete preceding OS counter interval inside that
same envelope. Queries crossing an edge are excluded, not interpolated. The
GPU query interval, accepted sample indices, CPU intervals, exact source hashes
and all retained field values are included in the JSON.

The table describes the two H128 isolated blocks above, the matching poll=0
counterexample, and the two H256 context blocks. Counts are contained GPU / CPU
observations. Frequencies and temperatures are queried observations, not
per-action effective service rates.

| Program / pool / block | PV median (ms) | GPU / CPU samples | SM clock min / median / max (MHz) | Observed P-states | Median memory clock (MHz) |
|---|---:|---:|---|---|---:|
| H128 / default / A0 | 4.8392 | 10 / 9 | 210 / 1605 / 1605 | P8: 1, P3: 9 | 7001 |
| H128 / default / A3 | 1.4932 | 3 / 2 | 1605 / 1605 / 1605 | P4: 3 | 6001 |
| H128 / poll=0 / A0 | 4.7863 | 10 / 9 | 210 / 1605 / 1605 | P8: 2, P3: 8 | 7001 |
| H128 / poll=0 / A3 | 1.4394 | 3 / 2 | 1605 / 1605 / 1605 | P4: 3 | 6001 |
| H256 / default / B1 | 2.9878 | 15 / 14 | 1605 / 1605 / 1605 | P0: 8, P3: 7 | 8001 |
| H256 / default / B2 | 7.5231 | 24 / 23 | 1605 / 1605 / 1605 | P3: 6, P4: 18 | 6001 |

| Program / pool / block | Median temperature (C) | Median power query (W) | Median machine CPU busy (%) | Main-thread CPU / envelope wall |
|---|---:|---:|---:|---:|
| H128 / default / A0 | 68 | 14.737 | 15.91 | 0.3845 |
| H128 / default / A3 | 68 | 12.547 | 16.69 | 0.9374 |
| H128 / poll=0 / A0 | 68 | 13.165 | 11.11 | 0.3539 |
| H128 / poll=0 / A3 | 68 | 12.133 | 8.72 | 0.9637 |
| H256 / default / B1 | 68 | 14.486 | 19.81 | 0.8913 |
| H256 / default / B2 | 68 | 12.851 | 14.74 | 0.5020 |

The H256 PV median rises **2.52 times** while every accepted SM-clock query
remains 1605 MHz and every temperature query remains 68 C. An equation based
only on these sampled SM-clock or temperature values cannot explain this pair.
Memory-clock and P-state observations change, but they are not sufficient causal
evidence. For example, a model assuming throughput strictly proportional to the
median memory clock would predict only 8001/6001 = 1.33 times, under its fixed-work
assumption. In H128, the slower block instead has the *higher* median memory
clock. These counterexamples reject such simple sampled-clock multipliers;
they do not exclude unsampled frequency changes, effective bandwidth changes,
cache effects or driver scheduling.

The H128 counter evidence gives another useful distinction. In the default
pair, envelope wall time changes from **487.6504 to 150.0084 ms**, but main-thread
CPU time changes only from **187.5 to 140.625 ms**. In poll=0, wall time changes
from **485.6747 to 145.9257 ms**, while main-thread CPU time changes from
**171.875 to 140.625 ms**. Most additional wall time therefore does not accumulate
as additional main-thread CPU time. This supports investigating GPU/driver
waiting and host descheduling rather than assigning the entire difference to
extra CPU arithmetic. Cumulative OS counters are coarse and do not distinguish
those waiting causes; process auxiliary CPU activity is also not a kernel trace.

Sampling was declared at 50 ms; the accepted observations in these six envelopes
have maximum inter-query gaps of about 49.94--51.10 ms. The fast H128 envelopes
contain only three GPU queries and two fully contained CPU intervals. The
145.9257 ms poll=0 envelope fails the existing three-period sustained-association
check, explicitly retained in the JSON. No state is assigned to an individual
1--8 ms invocation. NVML utilization uses an unexposed rolling window and
includes the foreground workload. On this RTX 4070 Laptop GPU, power queries
also average a preceding one-second interval, whose boundaries are not exposed.
Containing the API query does not contain those hidden averaging windows.
Neither utilization nor power is an external-contention or causal predictor.

The observed state changes motivate lower-level host/GPU tracing, but they do
not yet identify the dominant physical mechanism or justify a calibrated
state-to-throughput equation.

## Consequences for calibration

The original single-layer profile already explicitly measured all action
intervals, host dispatch gaps and the tail. Its measured gaps were only about
0.15--0.25% of the complete context duration. Adding a large constant host gap
would therefore contradict that decomposition. These paired controls add two
further constraints: neither one timestamp-overhead constant nor one universal
isolated-to-context multiplier explains all retained observations.

Parameter estimation should retain process and per-invocation distributions
instead of hiding them behind a single median. A distribution-based composition
model can be tested using profile-only action samples and a newly frozen model,
but its independence assumptions require explicit testing. Shared state shifts
can correlate action times, and a sum of action medians is not the median of a
complete execution. None of the paired observations may enter a fresh heldout
fit, and no improved accuracy follows until independent validation succeeds.

If lower-level causal attribution is needed, an independently labelled profiler
diagnostic can inspect host APIs and GPU activity. Profiler-instrumented times
must remain separate from primary timing parameters and uninstrumented heldout
acceptance.

## Evidence binding

This note reads the saved paired phase records and contained telemetry queries.
The reusable runner checked all four blocks numerically at collection; its
separately executed summary command also completed raw-byte and full-port
reverification for all 24 processes/eight groups. No hardware replay,
coefficient fit or accuracy calculation was performed for this note.

Local source directory:
`profiling/single_layer_context_pair/results/diagnostic-v1`.

| File bytes | SHA-256 |
|---|---|
| `manifest.json` | `8b1c202a2821a5fd1696664f3680386f6de6042c6d6aabbaba21686919690bd5` |
| `phase.json` | `b574f4c4be78d3c122b1f7551adf2adb8072fa7d3679464a8b2360d4b614555e` |
| `summary.json` | `32058b9c3511750ddf71858002d18d416b51f249cd37c790d4b8e3525444a47f` |
| Contained telemetry JSON in `docs/data` | `190ba23520110bac027fe285b5a0bf6eb722cad54fd0615d0abab3a8497ebdeb` |

The phase lists 24 distinct native PIDs, 24 successful jobs and zero failed jobs.
Supplied source/build/program/library bytes are bound by the campaign. Resolved
OS loader paths, kernel identity and physical memory capacity are not claimed.
