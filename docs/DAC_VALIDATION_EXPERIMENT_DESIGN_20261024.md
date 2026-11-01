# DAC validation experiment design

Date: 2026-10-24. Status: **proposed, not executed**. This protocol adds new
evidence; it does not revise the outcomes or thresholds of the October 23
campaign. Numeric targets below are project goals, not DAC acceptance criteria.

The candidate contribution is a strategy abstraction that separates computation
refinement/execution from tensor placement and movement while giving them common
dependency, physical-storage, resource and causal semantics. The compiler handles
coordination; the model evaluates the resulting execution. The studies must test
both what this boundary preserves and what useful decisions it enables.

The [Chinese design report](../../paper-write/第七教学周/experiment-reports/dac-validation-plan-20261024/EXPERIMENT_DESIGN_zh.md)
contains the explanation, complete scenario table and conditional paper claims.
This file is the implementation-facing protocol. If either is revised, reconcile
their counts and gates before freezing an execution manifest.

## Scope and evidence discipline

- One CPU and one GPU. Multi-GPU, a real NPU and universal strategy coverage are
  outside the main claim. Dense/MoE semantic coverage is distinct from native
  timing coverage.
- Within a comparison group, hold logical inputs, weights, numerical semantics,
  precision, threads, arrivals/routes, initial residency, final goals and objective
  fixed. A strategy may change only declared dimensions.
- Model-owned future routes are revealed causally. All candidate policies receive
  the same available information. Never supply target-inclusive device telemetry
  or target execution outcomes as exogenous prediction inputs.
- Freeze manifests, source/binary hashes, costs and their provenance, candidate
  sets, seeds, metrics and gates before heldout execution. Retain unsupported,
  infeasible, failed, timed-out and unstable cases in their original denominators.
- Changes after inspecting heldout results require a new version and independent
  validation data. A proposed gate must not relabel historical results as passed.
- “Best” always means best within the declared finite candidate set and objective.
  Enumeration is sufficient; no new search algorithm is claimed.

## E1: expression sufficiency and compiler fidelity

### Controls

1. A correct independently authored low-level action DAG, including operands,
   allocations, transfers, dependencies, resources and terminal goals. Helpers and
   parameterization are allowed. Do not manufacture an incorrect baseline by
   dropping dependencies, copies, capacity checks or causal restrictions.
2. An independent enumerator for a **declared finite set** of legal residual
   lowerings of tiny graphs. Freeze allowed event orders/decision epochs. It is
   an oracle only for that finite model, not all continuous-time schedules.
3. Pinned versions of LLMCompass and LLMServingSim, attempted on appropriate common
   requirements. Distinguish native configuration, adapter work, core changes,
   out-of-scope, untested and failed setup. An inability to install a tool is not
   evidence that its abstraction cannot express a policy. Do not compare latency
   unless the workload and cost contracts can be aligned.

External tools already have software/hardware or runtime strategy modeling;
novelty must be tied to a demonstrated difference in composition or semantics,
not to the mere presence of an IR. Source changes and complete required helper
code must be disclosed. AST/source counts do not establish human productivity.

### Fixed finite suite

Twelve patterns each have four geometry variants: H in {8,16}, q in {1,3}, I=2H,
with valid declared head/KV geometry. P/D patterns use q as prefill length and
then two decode steps. Freeze all histories, routes and byte budgets before run.

| Pattern | Required behavior |
|---|---|
| A01 | Whole-macro CPU/GPU placement |
| A02 | Attention and FFN refinement |
| A03 | Materialized FFN partition and merge |
| A04 | Residency, demand loading and prefetch |
| A05 | Two-request execution pipeline |
| A06 | Causal MoE expert scheduling |
| A07 | Refinement with caching |
| A08 | Refinement with prefetch |
| A09 | Request pipeline with cross-layer prefetch |
| A10 | P/D handoff with retained KV |
| A11 | MoE cache and migration |
| A12 | Shared physical storage with parent views/partitions |

This is **48 positive cases**. Existing cases are development evidence, not a
blind holdout. Prepare six negative cases: bad partition coverage, missing merge
dependency, wrong tensor version, physical-capacity overflow, release with live
readers, and use of unrevealed experts. Static errors must be rejected at compile
time; dynamic errors must be rejected before the invalid action takes effect.
Preserve partial traces but never publish their duration as successful execution.
Two causal pairs separately check expert placement and prefetch: identical public
prefixes with different hidden future routes must produce identical decisions
before revelation.

Independent oracles specify tensor identities, shapes, coverage, dataflow,
physical aliases/copies, legal termination and finite numerical reference results.
The compiler provides observations, not expected answers. Fully ordered cases
must agree on duration to one simulation tick. For partially ordered policies,
check legality and final semantics rather than demanding one arbitrary trace.

F32 numerical cases retain atol=rtol=5e-5. Report numerical coverage separately:
structural cases without a numerical executor cannot count as numerical passes.
The full numerical target is all 48; failure to implement that coverage narrows
the claim. Do not change tolerances after seeing errors.

### Residual lowering audit

Use four tiny graph types (partition/merge, cross-layer prefetch, two requests,
causal expert cache), three regimes (compute, transfer, capacity constrained),
and four high-level candidates per condition: **48 candidate instances in 12
choice groups**. The independent oracle enumerates only decisions left open by
the same strategy, preserving cache rules, placement, prefetch windows, causal
visibility and initial/final states.

Every L(p) must contain its default lowering; an illegal default is a compiler
failure. Break argmin ties by frozen candidate ID. Retain infeasible candidates
without ranking partial durations. All 12 groups need at least one feasible
candidate and a completed enumeration; all-infeasible or timed-out groups do not
pass. Report 12 pattern categories separately from the 48 geometry instances.

For the causal expert micrograph, enumerate policy trees jointly over at least
two routes sharing a public prefix. Equal visible histories must induce the same
action. Freeze the objective to the equally weighted mean duration of the two
routes; T below denotes that mean in this group. Independently optimizing each
known full route would give the oracle clairvoyance and is not permitted.

For finite legal L(p):

```text
C(p) = T(p, default_lowering) / min[l in L(p)] T(p,l) - 1
p_hat = argmin[p] T(p, default_lowering)
A = min[l in L(p_hat)] T(p_hat,l) / min[p,l in L(p)] T(p,l) - 1
```

C is compiler loss; A is selection loss caused by residual lowering freedom.
Large C need not invalidate the abstraction. Large A means the hidden decisions
materially affect selection and should be exposed or the claim restricted.
Report all C values; target **max A <=5% across all 12 completed groups**. Preserve
enumeration counts, pruning justification and completion receipts. Timeouts are
unresolved, not optimality certificates.

Kernel fusion, layout, CUDA Graph mode and thread setup remain fixed backend
contracts. No statement about their universal lack of optimization opportunity
follows from this audit.

### Gates and claims

- 48/48 positive semantic cases, 6/6 intended rejections, 2/2 causal pairs.
- No manually edited lowered artifacts or case-specific runtime patches in any
  positive case. Report numeric coverage and all unsupported cases separately.
- All declared external comparison attempts have inspectable configurations or
  a justified status. Equality with a capable external baseline is an outcome,
  not a reason to weaken that baseline.
- At most 5% residual selection loss in the finite tiny model; report compiler
  loss independently.

Conditional claim: “Across [xx]/48 declared cases, the abstraction composes
execution and storage policies under common dependency, physical-capacity and
causality rules. Exhaustive evaluation of the declared tiny lowering spaces
bounds observed selection loss by [xx]%.”

If a hidden dimension changes the decision, name that dimension and narrow the
claim. Passing tests alone does not prove novelty or universal expressiveness.

## E2: independent real-device strategy selection

### E2A: current native scope

Six workloads are the cross product of:

- H256/I512/4 query heads/2 KV heads, or H512/I1024/8 query heads/2 KV heads;
- P16/history0, P64/history0, or D1/history128.

Head dimension 64, F32, batch 1, CPU threads 1, CUDA0, CUDA Graph disabled. Use the
existing declared complete Dense block. This is not a production-model benchmark.

Four candidates per workload:

- S0: all CPU.
- S1: all GPU, including necessary transfers.
- S2: GPU projection/FFN with CPU attention.
- S3: GPU projection/attention with half-width materialized CPU/GPU FFN parts,
  GPU normalization/merge and **serial** execution of the parts.

Existing native kernels cover these operations, but the new 24-program campaign
does not exist yet. Validate export limits and numerical equivalence first. Do
not silently substitute a smaller workload after timing results are known.
The current executor rejects overlapping launch; S3 is not concurrent TP.

All strategies share parent weights and data, not seeds indexed by strategy.
Verify leaf slices reconstruct the same parents and that common final hidden/KV
match independent numerical references, with atol=rtol=5e-5. Initial weights,
inputs, positions and old KV are in DRAM. Required final hidden/KV are in DRAM.
Physical backing allocated in advance does not establish logical residency.
Offline partition preparation is excluded and does not validate online remapping.

### Reusable calibration is a prerequisite for the strong claim

Current steady action keys bind a full program signature. Per-program profile
followed by fresh-process heldout validates repeatability of already measured
programs, not prediction of unmeasured compositions. Retain it only as a clearly
labeled diagnostic control. Removing program identity from existing prices is
not a valid conversion to reusable costs.

For the main study, measure independent kernels/macros, copies and mechanism
probes; freeze parameters before any target complete-strategy timing. It is
permitted to profile required shapes independently: the claim is composition
generalization, not unseen-shape extrapolation. Preserve operation semantics,
shape, dtype, layout, backend/binary, threads, hardware and execution boundary in
the reusable contract. Context corrections must come from independent transition
or concurrency probes. Unknown costs remain unknown.

Put mechanisms in modeling and measured parameters/provenance in profiling.
Target full-stamped diagnostics may be collected **after** the prediction freeze
for observer checks and explanation, but must not feed the primary predictor.
Freeze an accounting table for costs already included inside actions versus
additional inter-action increments. Native actions may include submission and
synchronization; an independently measured synchronization duration must not be
added again unless the probe identifies an extra, non-overlapping mechanism.

### Sampling and metrics

Eight complete independent rounds per workload; each round runs all four
candidates in fresh processes. Freeze two balanced four-position sequences.
There are **6*4*8=192 heldout processes**, excluding calibration/observer controls.
Each process retains 12 trials of 8 fixed-input replays after warmup. Include
declared computation, copies, switching and synchronization; exclude process
startup, graph building, allocation, reset, verification and serialization.
Report replay-average latency, not token tail latency or autoregressive serving.

Primary T_real is the mean of independent process means. Median summaries are
sensitivity analysis. Preserve all samples. At the paired-round level, bootstrap
10,000 resamples, retain within-round candidate pairing and recompute the empirical
best each time; keep the prediction-selected candidate fixed. Freeze bootstrap
seed=20261024. Use approximate simultaneous 95% upper bounds across the six
workloads with one-sided quantile 1-0.05/6=0.9916667, and show per-group intervals. Eight rounds
can leave substantial uncertainty; do not claim precision from inner repeats.

```text
s_hat = argmin[s] T_pred(s)
R = T_real(s_hat) / min[s] T_real(s) - 1
G = min[s in restricted set] T_real(s) / min[s in full set] T_real(s) - 1
```

Retain original gates: each process mean APE<=10%, trial-average P90 APE<=20%,
per-process trial CV<=10%, observer relative mean difference<=5%. Report between-
process variability separately. Calculate trial P90 from the 8*12=96 heldout
trial averages per candidate; mean-error and CV checks remain per process.
Each of the 24 programs has four paired fresh-process stamped/bare observer
controls, two AB and two BA pairs: another 192 processes. Within each pair hold
inputs, warmup and 8 trials*8 replays fixed. Collect only after prediction freeze
and do not use these target diagnostics for calibration.
Only a group whose **every candidate** passes
the quality contract counts as fully validated. Failed/unstable groups remain
in the six-group denominator and retain their error/regret summaries.

Primary project target: all six groups complete and qualified; each point
estimate R<=5%, each simultaneous upper bound<=10%. Stronger “within 5%” language
requires those upper bounds<=5%. Report maximum and distribution, not only a
median. Distinguish separated winners, near ties and unresolved rankings;
exact Top-1 and rank correlation are secondary.

Compare with every fixed recipe and the CPU/GPU-only restricted set. A proposed
strong limitation witness is G>=10% with a paired lower confidence bound>0 in
at least two distinct declared workloads, with familywise comparisons accounted
for. This is a hypothesis, not a required favorable outcome. If a fixed strategy
always wins, report reliable selection without claiming adaptation benefit.
These are mechanism controls, not full NEO/KTransformers reproductions.

Report profiling cost, compile/simulate/search wall time, evaluation count and
real benchmarking cost separately. Do not assume the simulator is faster for
these small workloads. Actual end-to-end runtime speedup and simulation speedup
are different measurements.

### E2B: conditional extension for execution/storage performance claims

Real cache, prefetch and overlap benefits require an additional bounded native
trace executor; they cannot be inferred from E2A. Plan three two-layer H512
workloads, P16 followed by D8: Dense, hot MoE and rotating MoE (4 experts/top2).
Inputs/routes are supplied traces, revealed causally; hidden and KV really flow
through the full window. This is not single-layer time multiplied by repetitions.

Two GPU arena budgets per workload: a loose budget fitting weights plus working
state, and a tight feasible budget forcing reuse/release. Derive exact bytes from
tensor/workspace inventories before measuring time. For each of six groups,
cross two execution rules with two storage rules:

- all GPU versus a frozen hybrid rule (CPU attention/GPU FFN for Dense; causal
  expert placement from revealed routes and current known costs for MoE);
- demand+LRU versus the same LRU plus bounded prefetch (known next-layer GPU
  consumers for Dense; history-only expert predictions for MoE).

Four candidates, eight rounds: another **192 processes**. Each trial resets to
the declared initial state, executes one whole P/D window, and preserves caches
inside that window. Apply E2A's selection/quality targets as a separate six-group
family. Do not pool success denominators with E2A.

Required additions: real asynchronous dependencies/events/copies, bounded
physical arena reuse and peak accounting, causal MoE/multiple invocations, and
independent numeric checks. Current logical EVICT and globally preallocated
backing cannot validate capacity/eviction behavior. If not completed, keep
execution/storage composition as semantic/analytical evidence and restrict real
selection claims to E2A.

Before formal timing, validate arena lifetime on tiny graphs, then verify all
four candidates can complete legally at both budgets, with live-buffer protection,
correct asynchronous dependencies and equivalent outputs. Include measured
workspace, prefetch reservations and copy-source lifetimes in admission; weight
inventory alone does not establish a tight feasible budget.

Conditional E2 claim: “Using independent component/mechanism profiles frozen
before complete-strategy measurement, selected candidates have median/max real
regret of [xx]%/[xx]% over [xx]/6 declared workload groups.” Only completed E2B
permits extending that sentence to real prefetch/cache combinations.

## E3: hardware sensitivity and architecture analysis

### Anchored scope and controls

Start with E2A's H512 P64/history0 and D1/history128 and the same four serial
candidates. Independent-memory reference costs use the frozen measured contracts.
Shared storage is a controlled architectural counterfactual, not calibrated Mac,
phone or NPU performance. Large/MoE analyses remain separate until supported by
additional profiles and E2B.

Compare (a) the reference winner's fixed recipe at each hardware point with
(b) selecting again from the **same** candidate set at that point. Also compare
independent/shared storage under the same compute contracts and total physical
capacity. Fix the independent DRAM:VRAM capacity ratio. Fixed recipe means fixed
rules; a rule permitted to observe current costs may adapt its actions.

Shared capacity counts physical allocations once. Zero explicit COPY does not
remove memory traffic, synchronization or coherence. The existing shared-pool
model does not automatically model contention, migration or cache coherence.
Use an explicitly idealized shared-visibility contract until those effects are
represented. Native E2A does not establish real allocator/capacity accuracy.
Unknown workspace remains unknown or an explicit scenario assumption.

### Fixed scan

Five levels per axis: {0.5,0.8,1,1.2,1.5} times the baseline.

| Architecture | Axes | Unique one-axis points |
|---|---|---|
| Independent storage | CPU compute-action duration, GPU compute-action duration, CPU/GPU payload bandwidth, total physical capacity at fixed pool ratio | 1+4*4=17 |
| Shared storage | CPU compute-action duration, GPU compute-action duration, shared capacity | 1+3*4=13 |

Add two 3x3 interactions at {0.8,1,1.2}: independent GPU duration x link bandwidth;
shared GPU duration x capacity. Each adds four off-axis corners, so total points
are 17+13+4+4=38. **38 points * 2 workloads * 4 candidates = 304 cases.**

Action-duration multipliers are effective cost scenarios. Do not rename them as
frequency/FLOP/s/memory-bandwidth upgrades without an identified component model.
Modify only the declared CPU/GPU compute action costs; keep separately priced
copies, switching and other-device work fixed. Payload bandwidth does not remove
fixed transfer latency. Capacity affects feasibility/scheduling, not a global
time multiplier. Shared GPU slowdown cases test whether copying savings survive
a worse effective execution cost; they are not measurements of shared-bandwidth
contention.

At corresponding capacity factors, shared capacity equals independent DRAM+VRAM.
Choose baseline bytes from the frozen workload inventory before timing; small
normalized capacities are not descriptions of commercial devices. Equal capacity
does not mean equal price, energy or chip area.

### Metrics and qualification

```text
fixed-policy gain = 1 - T(s_base,h_new)/T(s_base,h_base)
reselected total gain = 1 - min[s]T(s,h_new)/min[s]T(s,h_base)
regret without reselection = T(s_base,h_new)/min[s]T(s,h_new) - 1
```

Report feasibility, critical-path compute/copy/wait, transferred bytes, physical
peak, and simulator wall time. Total independent-memory peak is
max_t(DRAM(t)+VRAM(t)), not the sum of asynchronous peaks. Do not double-count
overlap durations or shared aliases. Infeasible baseline policies have no finite
speedup ratio; report feasibility transitions instead.

Targets:

- All 304 cases have audited outcomes, including infeasibility and unsupported
  contracts. All successful traces pass dependency/physical-storage/copy audits.
- Real anchor comparisons retain E2 gates where the contracts actually overlap.
  A single real anchor does not validate unmeasured shapes or architecture points.
- Every main sensitivity claim has paired trace evidence for the changed cost or
  resource term. An unmodeled axis is reported as such.
- Four prespecified focus conditions (two workloads x two architectures) test
  baseline and **all** nominated improvement directions, not only the nominal
  top two. Improvements are CPU/GPU duration*0.8 and bandwidth/capacity*1.2
  (no link axis for shared storage). Rank reselected gains; preserve ties.
  These are equal fractional scenarios, not equal economic investments.
  Independent storage varies CPU action, GPU action, capacity and payload
  bandwidth uncertainty; shared storage varies CPU action, GPU action, capacity
  and separately accounted shared handoff cost. Two bounds per dimension yield
  16 joint corners. Reevaluate all four policies at the baseline and every
  improvement: at most (2*5+2*4)*16*4 = 1152 additional cases.
  Use independent measured bounds where available; otherwise declare +/-20%
  scenarios for nonzero multiplicative quantities. A zero handoff baseline needs
  a separately declared absolute upper bound; multiplying zero is not a test.
  Handoff costs cannot duplicate included action overhead. Missing parameters or
  undeclared bounds make that robustness conclusion unresolved.
- A recommendation is called robust only if the same leading direction remains
  beneficial at all 16 declared corners; nominal gain>=5% is the practical target
  for a main recommendation. Finite-corner checks are neither a probability nor
  proof over the interior. If no direction improves performance, report that.

Unknown UMA coherence effects require separate optimistic/conservative scenario
bounds or an explicit idealized limitation. E2's 10% error gate is not a certified
error bound over new hardware. No cost/energy recommendation without such data.

Retain the existing 616-case synthetic study separately, including the 51/51 groups
where stagewise and joint selection were equal. A superset's minimum being no
worse than a subset's is not an algorithmic contribution. Do not erase null
results by replacing the matrix.

Conditional claim: “Under the declared parameter and uncertainty scenarios,
[change] improves [workload] by[xx]%, while retaining the reference policy loses
[xx]%. The direction is preserved at[xx]/16 tested corners.” Without UMA data,
this is architectural scenario analysis, not a real-product ranking.

## Order, deliverables and stopping rules

1. Freeze E1/E2A requirements, independent oracles and cost-reuse contracts.
2. In parallel, implement E1 controls and collect independent E2A probes. Freeze
   parameters before the 192 heldout processes. Do not reopen unrestricted device
   fluctuation calibration under the guise of this study.
3. Diagnose failed cases by mechanism; changes require new validation versions.
   No hidden relaxation of numeric, accuracy or stability gates.
4. Build E2B only for the stronger real execution/storage claim. In parallel,
   prepare E3's 304-case scenario scan with E2A anchors. Add robustness separately.
5. Preserve manifests, full IR/action/state traces, independent checks, raw
   measurements, cost provenance, all outcomes and artifact hashes.

Paper outputs: an abstraction/comparator table, one high-level-to-actions example,
semantic and residual-choice results, real strategy-regret plot, execution/storage
mechanism example only when supported, and hardware/reselection phase maps with
infeasible and uncertain regions visible. Report negative conclusions explicitly.

The current 8-case expression evidence, program-scoped accuracy measurements and
616 synthetic design cases are foundations, not evidence that this new protocol
has passed. Strong publication claims require measurable added value relative to
capable prior approaches; passing engineering gates does not guarantee acceptance.

## Sources and implementation references

- [DAC 2027 research submissions](https://dac.com/2027/program/research-manuscript-submissions)
- [LLMCompass official implementation](https://github.com/PrincetonUniversity/LLMCompass)
- [LLMServingSim official implementation](https://github.com/casys-kaist/LLMServingSim)
- [Previous three-study protocol](DAC_THREE_STUDIES_PROTOCOL.md)
- [Shared-storage contract and limitations](SHARED_MEMORY.md)
- [Native coverage audit](../build/dac-evidence-20261023/root-independent-review/NEXT_MEASUREMENT_COVERAGE.md)
- [Dense case builder](../profiling/highlevel_accuracy/cases.py)
- [Native export and numerical checking](../profiling/highlevel_accuracy/bridge.py)
- [Current scoped calibration](../profiling/steady_costs/calibration.py)
- [Current measurement gates](../profiling/steady_costs/campaign.py)
