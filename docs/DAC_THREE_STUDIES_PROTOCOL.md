# Three evidence studies for strategy modeling

Date: 2026-10-23. This is the consolidated protocol and evidence map for the
authorized experiment campaign. Each runner saves its own declarations and
source hashes before execution. This document was consolidated while the
studies were running; it is not an independently preregistered research study.

## Question and scope

The candidate contribution is a common description of supported computation
refinements and causal tensor-copy management, compiled under one set of
dependency, physical-capacity and resource rules. The experiments test whether
that description preserves meaning, supports credible decisions, and exposes
execution/storage interactions. Merely having an IR or a working simulator does
not establish novelty or prediction accuracy.

The platform scope is one CPU plus one GPU. Shared physical storage is included
in the semantic references. Multi-GPU, a real NPU, a universal strategy language,
and a novel search algorithm are outside this campaign.

## Study 1: expression and compilation fidelity

Runner: `experiments/paper_expression_study.py`.
Protocol: `docs/PAPER_EXPRESSION_STUDY.md`.

Four existing reference mechanisms and four new finite compositions form a
fixed eight-case denominator. The new cases combine attention refinement,
unequal FFN partitions, initialization, request pipelines, prefetch, model-owned
MoE routing, P/D handoff, resource interruptions and shared-parent slices.
They are newly authored combinations, not blind-heldout algorithms.

Pass requires the declared operands, dimensions, transfers, dependencies,
capacity accounting and causal-prefix behavior. Independent geometry and
event-ledger checks are used where applicable. Negative mutations must be
rejected. Synthetic costs make this a semantic experiment, not a timing
accuracy experiment. No core/runtime fixes are allowed during a frozen run.

Authoring statistics count selected recipe/helper extracts, including cost
declarations; they do not include every transitive code dependency. They do not
measure human effort or prove an advantage over another language. A matched external
tool and low-level authoring comparison remains separate work.

## Study 2: credibility of strategy selection

Runner: `experiments/paper_selection_study.py`.
Protocol: `docs/PAPER_SELECTION_STUDY.md`.

Use existing frozen real measurements without re-fitting or new timing runs.
Keep October 18 historical predictors distinct from the October 22 calibrated
model. Recompute raw heldout values and original quality gates.

The fixed inventories contain 143 historical chain paths grouped into 23
comparable resident choices, 72 overlap paths grouped into 36 matched choices,
and 11 current steady-cost cases with three heldout processes each. Preload
paths remain in latency results but are not compared as resident-only choices.
Current whole-layer cases are checked for matching alternatives; absence of
alternatives is reported as not evaluable, never as a zero-error selection.

For candidate set S, choose s_hat = argmin_s predicted_time(s), then report

    regret = measured_time(s_hat) / min_s measured_time(s) - 1.

Compare identical data, precision, workload, initial state and final goals.
Keep all candidate errors and all failed stability gates. Report median, P90,
maximum error/regret, winner agreement and whether the observed winner is
separated. Three-process empirical resampling is coarse uncertainty evidence;
it does not prove population stability. Do not pool historical/current
predictors or treat batched repetition averages as individual latency tails.

## Study 3: controlled design-space evaluation

Runner: `experiments/paper_design_study.py`.
Protocol: `docs/PAPER_DESIGN_STUDY.md`.

The fixed matrix is 616 cases:

* 504 cases: VRAM {2,4,12} GiB x link {2,8,24} GB/s x P8D1 x two initial
  placements x Dense (8 candidates), hot MoE (10), rotating MoE (10).
* 112 cases: P8D8 and P8D32 at the fixed 12 GiB / 8 GB/s reference, with the
  same initial placements, workloads and candidates.

All cases use two layers, H6144/I16384, 24 query heads, six KV heads, F32, and
declared analytical costs. MoE has four experts/top-2. This is not calibrated
hardware performance. The candidate set contains macro placement, causal
expert device selection, and optional next-layer prefetch. It does not silently
assign prices to partitioned kernels whose contracts are absent.

Cold means weights initially in DRAM. GPU-resident means all weights initially
in VRAM only, charged before any action; CPU consumers must obtain DRAM copies.
It is an initial-placement experiment, not steady-state throughput. Impossible
initial placements remain in the denominator. Every within-group candidate
has identical initial state and mandatory final hidden/KV locations.

The workload's CPU trace-next-token gate consumes the preceding round's final
hidden state. Intermediate layer hidden values need not return to DRAM, but
round boundaries require the final-layer hidden there. All retained KV blocks
require DRAM views at the global end; GPU-only terminal KV strategies are not
in this matrix. Token inputs are supplied by the trace, not generated numerically.

Compare execution-only with fixed storage, storage-only with fixed execution,
execution-then-storage selection, and full finite enumeration. Report evaluation
counts. A superset's optimum cannot be worse than its subset; this fact alone
is not an algorithmic contribution. Also freeze the reference-point winner and
measure its regret or infeasibility at other hardware points.

## Evidence discipline and stopping conditions

Saved declarations, compiled IR, events, cost provenance, source hashes and
audits support each result. Harness errors are retained and explained in a new
output version; no silent overwrite or threshold changes. Independent reviews
check fairness and statistical boundaries. The runtime and frozen calibration
artifacts remain unchanged. Existing 352-case regression is not relabeled as a
new experiment or needlessly rerun when its execution sources are unchanged.

An experiment is complete when every declared case has an audited outcome,
including infeasible, rejected or negative results. Research qualification is
separate: a completed study may show a missing ability or insufficient evidence.
Do not report all three scientific hypotheses as supported merely because all
scripts finished.

## External comparison still required

The official [LLMCompass repository](https://github.com/PrincetonUniversity/LLMCompass)
provides hardware/software models and design-space exploration. The official
[LLMServingSim repository](https://github.com/casys-kaist/LLMServingSim) describes
a runtime-driven simulator with heterogeneous hardware and serving techniques.
They must not be dismissed as having no strategy or design-space support.

This campaign does not execute those tools, establish that a particular
composition is impossible in them, or claim productivity/performance superiority.
A subsequent matched comparison should hold hardware costs and workload
boundaries fixed, attempt the same small expressions, and record configuration,
adapter and core modifications separately. Unsupported and untested are
different outcomes. Tool versions must be pinned before that comparison.
