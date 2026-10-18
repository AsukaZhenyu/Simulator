# One execution-time functional over resource traces

The primary modeling target is one function of time-varying resource inputs:

\[
T=\mathcal F(G,P,H,K;\ c(t),g(t),m_D(t),m_V(t)).
\]

`G` is the computation and data-dependency graph, `P` its execution and storage
strategy, `H` the hardware description, and `K` the pinned numerical/backend
contract. These are fixed within a matched experiment. CPU and GPU external
compute pressure, external DRAM allocation, and external VRAM allocation are
time-dependent inputs. Further state channels can be necessary: this formulation
does not establish that these four channels explain every physical latency.

The old constant and named-condition tables are retained as historical
diagnostic controls. They are not the desired simulator interface. A background
condition name is not an independent performance model.

## Progress, completion, and capacity

An operation has fixed reference work `W`. One calibrated response surface maps
its exact operation contract and current resource inputs to effective service
rate. Completion is the first time at which accumulated service reaches `W`:

\[
\int_{t_0}^{t_0+T}r_\theta(o,c(t),g(t),m_D(t),m_V(t))\,dt\ \geq W_o.
\]

The implementation integrates piecewise-constant traces while retaining exact
fractional progress. A state change affects remaining work, including work that
started before the change. Different kernels can have different calibrated
coefficients within this one evaluation rule. A CPU matmul, GPU softmax, and a
copy need not respond identically to the same resource pressure.

Memory capacity is a separate constraint. External allocations plus the
strategy's live tensors, reservations, and workspace must fit the declared
capacity. Allocating bytes does not, by itself, establish a bandwidth slowdown.
An infeasible strategy requires a different explicit storage schedule; a timing
model must not silently add a spill or an eviction.

A completed GGML action includes host submission and synchronization. Fitting
its effective service rate does not establish that every part of this interval
physically scales with GPU throughput. The first implementation preserves that
timing boundary rather than claiming to have separated fixed launch costs.

## A single fitted response

The initial response is a bounded, regularized regression over numeric resource
inputs, separately parameterized for each exact operation contract. A frozen
feature basis includes CPU pressure, GPU pressure, normalized memory allocations,
and CPU/GPU interaction. There is one model artifact and one prediction path.
Training does not branch on `baseline`, `cpu_load`, or `gpu_load` labels.

Regularization and the feature map are declared before evaluation. Unobserved
operation contracts, input semantics mismatches, missing states, unsupported
states, and nonpositive predictions remain explicit failures. Marginal feature
bounds do not prove that all joint combinations inside a bounding box have been
validated. Heldout combinations and traces must test that assumption.

## Input semantics and measurement

External compute occupancy is not directly supplied by the existing NVML total
GPU-utilization counter: that counter includes the foreground task. Using it
as if it were independent background occupancy can leak the measured execution
into its own predictor. System CPU utilization has a similar distinction.

The first executable collection protocol therefore uses a more limited,
explicit input contract, `offered_load_controlled`:

- CPU and GPU inputs are numeric on/off requests for owned replay workloads.
- Memory inputs are the bytes held by owned DRAM and CUDA allocation workers.
- Total device utilization, clocks, and memory observations are diagnostic data.
  They are not silently substituted for the controlled input functions.

This is a controlled load-response experiment. Its inputs are available before
the foreground completion is measured. It does not establish a calibrated
mapping from arbitrary real applications' occupancy to available compute, nor
does it prove total physical memory feasibility. The generic external-state
contract and this offered-load contract are explicitly distinguished.

The foreground native probe remains in the same process across a sequence of
phases. A full binary grid exercises CPU replay, GPU replay, host allocation,
and device allocation both separately and in combination. Training and heldout
runs use independently shuffled phase orders, disjoint numerical seeds, and
fresh foreground processes. Requested deadlines, resource readiness, actual
steady intervals, and transition intervals are recorded separately.

All individual execution samples are retained. Transitions are not relabeled as
instantaneous physical load changes. Samples that cannot be assigned a supported
input trace remain unknown and are counted in coverage. Phase summaries cannot
establish the absence of within-process or between-process variation.

## What establishes success

Three distinct checks are required:

1. Hand-computable trace tests establish correct progress integration and memory
   constraints, including a resource change during an operation.
2. Hardware collection establishes that the intended numerical graph executes
   correctly and that controlled input transitions and observations are recorded.
3. A model frozen after training and before heldout collection must predict new
   executions without fitting their durations, filtering failures, or treating
   repeated invocations as independent foreground processes.

Passing software checks or a small end-to-end smoke run establishes neither
physical accuracy nor a universal occupancy-to-time law. Existing rejected
calibration data and acceptance thresholds remain unchanged.

Source-checkout entry point: [resource calibration workflow](../profiling/resource_calibration/README.md).

Implemented behavior, first hardware evidence and unresolved accuracy are
recorded in [the implementation report](UNIFIED_RESOURCE_RESULTS_20261018.md).
