# Unified resource functional: implementation and first hardware check

**One prediction interface is implemented and connected to the common runtime.
The first hardware smoke experiment completed. Hardware accuracy has not passed.**

The simulator's modeling target is now
`T = F(workload, strategy, hardware, backend; CPU(t), GPU(t), DRAM(t), VRAM(t))`.
The older constant and named-condition tables remain historical diagnostic
controls. They are not the desired primary simulator model interface.

## Implemented behavior

- One frozen response model consumes numeric resource coordinates, with no
  `baseline`/`CPU-load`/`GPU-load` model-selection branches.
- CPU, GPU and copy operation responses share the same fitting and integration
  API. The first hardware fit covers one exact GPU PV contract only.
- Remaining reference work is integrated across piecewise state changes using
  exact fractions. Memory feasibility is distinct from performance response.
- A restricted CPU/GPU timing adapter drives the actual common
  `SimulationSession`. It cannot apply one PV price to a full LLM or certify
  physical memory. Its four-byte input/output carrier is explicitly synthetic.
- One foreground native process continues across all 16 combinations of owned
  CPU replay, GPU replay, DRAM allocation, and VRAM allocation.
- Actual transitions, resource readiness, steady holds, background invocation
  coverage, allocation lifetimes, device observations, owned CPU activity,
  numerical checks, raw samples, and cleanup evidence are retained.

The mathematical contract and limitations are in
[the functional design](UNIFIED_RESOURCE_FUNCTIONAL.md); commands and input
semantics are in [the workflow guide](../profiling/resource_calibration/README.md).

## Hardware check

The physical graph is H128/q16, single-layer Dense prefill, batch one, F32,
history zero. Attention stages and FFN partitions execute serially on CPU/GPU
through the same public compilation path used by the previous experiment.
The modeled target is GPU `attention_pv`, including its materialization scope;
CUDA graphs remain disabled and the legacy completed-work boundary is fixed.

This smoke used one training and one independent heldout foreground process.
Each process traversed all 16 binary control combinations in independently
shuffled order. Both allocation controls used 64 MiB when enabled. Each declared
steady hold lasted 500 ms after readiness; startup/teardown took additional,
separately recorded time. Numerical seeds were disjoint, and the model was
frozen and verified before the heldout process launched.

Inputs have `offered_load_controlled` semantics. An enabled native replay is a
work request, not a measured utilization percentage. Allocation bytes are the
owned payload, not total external occupancy. Native replay workers themselves
initialize both backend contexts and buffers. Total NVML/system utilization is
diagnostic data and is not fed back into its own timing predictor.

All 33 owned processes exited normally with verified cleanup: two foregrounds,
17 replay workers and 14 allocation workers. No forced termination occurred.
Independent foreground and replay numerical checks passed. The exclusive
measurement lock was released.

## Prediction and coverage

The predeclared first response is an affine surface with CPU/GPU interaction,
ridge penalty `1/1000`, and physical-capacity-normalized memory coordinates.
It uses one equal-weight exact median per training run/steady phase, not one
regression vote per invocation. No heldout duration changed the coefficients.

| Evidence | Training | Heldout |
|---|---:|---:|
| Foreground processes | 1 | 1 |
| Declared steady phases | 16 | 16 |
| All retained PV samples | 3,285 | 1,828 |
| Complete PV samples inside steady phases | 488 | 476 |
| Transition/outside PV samples without an assigned steady price | 2,797 | 1,352 |
| Phases passing raw within-phase CV <= 15% | 11/16 | 12/16 |

All samples remain in raw artifacts. Unknown transition coverage is reported,
not discarded as if every sample had been predicted. The model predicts all
16 heldout **steady-phase medians**:

| Metric | Result |
|---|---:|
| Training median relative error | 36.10% |
| Training worst relative error | 343.80% |
| Heldout median relative error | 28.22% |
| Heldout worst relative error | 108.87% |
| Heldout phases meeting the error-only 20% threshold | 3/16 |
| Formal hardware acceptance | **Not established** |

The worst heldout phase requests no CPU/GPU replay, no owned DRAM payload, and
64 MiB of owned device payload. This is not an otherwise idle-machine claim.
The [public evidence](data/unified-resource-smoke-20261018.json) retains every
phase error, the frozen coefficients, exact training rows, coverage, and raw
artifact hashes. Raw native files remain locally in
`profiling/resource_calibration/results/smoke-v1/`.

The poor training fit is itself evidence that this first response is inadequate;
the error cannot all be assigned to heldout randomness or external applications.
Possible deficiencies include the restricted feature basis, strongly shrunk
small memory coordinates, missing state/history, and timing variability. This
experiment does not determine their relative contributions. Its error numbers
are not directly comparable with the older three-condition campaign, which has
different trajectories and sampling units.

## Software and runtime verification

All **241 relevant software tests passed**: 111 new functional, collector,
counter, frozen-model and runtime tests, plus 130 existing state-calibration
regressions. No test was skipped in these two suites. Unchanged core suites were
not rerun merely to increase the count.

The actual frozen hardware-fit artifact also passed **32/32** common-runtime
reconstructions: 16 fixed-state timings and 16 constructed mid-operation state
changes. Each matched the functional's exact remaining-work calculation and
integer completion. These reconstructions launch no native process and are not
hardware accuracy checks.

The public receipt includes the software-log and runtime-reconstruction hashes.
The frozen artifact file SHA-256 is
`892123a4cd0bfd1e9646ed453d687599abc9ae389068f561992716f49fa73aed`;
the evaluation file SHA-256 is
`6a0b812e365fd2945bf3a9b69c356e1b117449e1ed58999d1496c79d44eda7d5`.
The model's semantic digest differs from its enclosing artifact file digest;
both meanings remain explicit in the JSON.

## Remaining work

1. Improve the response using training evidence: the current fit is inadequate
   even on its own training states. Any revised feature map or regularization
   needs a new frozen protocol and fresh heldout evidence.
2. Collect at least three independent foreground processes per split, with
   longer stable holds and all original stability/observation gates retained.
3. Supply better measured external compute-pressure channels. Requested load,
   process CPU activity, target-inclusive utilization, capacity occupancy and
   bandwidth contention must retain their distinct meanings.
4. Validate the actual hardware response across transitions. Unknown startup and
   teardown intervals currently prevent that claim; software integration tests
   alone do not resolve it.
5. Calibrate more operation contracts before using the functional for complete
   CPU/GPU/transfer LLM timelines. Full-graph strategy selection accuracy remains
   a separate, required evaluation.
