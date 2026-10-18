# Explicit device-state service rates

The causal runtime accepts piecewise constant CPU/GPU compute service rates,
including changes while an operation is running. This is an execution model for
explicit supplied rates. It does not identify a physical rate from GPU
utilization, clocks, temperature, power, or other applications.

## Rate contract

An operation's existing `duration_ns` is its **reference service requirement**.
A rate of `1/2` serves half a reference nanosecond per wall-clock nanosecond;
that operation takes twice its reference duration if this rate remains constant.
A rate of `2/1` serves twice as fast. A zero rate pauses the compute resource.
Omitting a rate change preserves its current value; the initial reference rate
is exactly `1/1`. This neutral reference convention does not imply that an
unmeasured device state is calibrated.

```python
from llm_simulator import EnvironmentEvent

events = (
    EnvironmentEvent(
        t_ns=1_000,
        gpu_service_rate_numerator=1,
        gpu_service_rate_denominator=2,
    ),
    EnvironmentEvent(
        t_ns=4_000,
        gpu_service_rate_numerator=1,
        gpu_service_rate_denominator=1,
    ),
)
# Pass events to Simulator.compile/run(environment_events=events), or construct
# the corresponding explicit events in a ModelPlan's environment JSON file.
```

Both integer parts must appear in the same event. Numerators must be nonnegative
integers and denominators positive integers; booleans, floats, null declarations,
missing halves, and duplicate same-time changes to one device are rejected.
The fields are optional extensions of `environment-trace-0.1` and of the
`events` array inside `environment-trace-0.2`:

```json
{
  "format_version": "environment-trace-0.1",
  "events": [
    {
      "t_ns": 1000,
      "gpu_service_rate_numerator": 1,
      "gpu_service_rate_denominator": 2
    }
  ]
}
```

## Exact progress and resource semantics

The runtime retains remaining reference work as an exact rational number.
For example, a task needing 10 reference ns serves 3 ns by wall time 3 at rate
one. Changing its rate to one half then requires 14 more wall ns, finishing at
17 rather than restarting the whole task at the slower rate.

Completion is rounded up to an integer wall-clock ns. Only actual remaining
reference work is charged in the last tick; rounding never manufactures extra
work. Unrelated memory, controller, other compute, and copy boundaries do not
round or discard fractional remaining work.

`cpu_available=False` forces effective CPU rate zero while retaining its
declared rate. A later rate update during the blackout is retained and becomes
effective when CPU availability returns. Setting a CPU/GPU rate to zero also
pauses running compute and rejects new compute on that resource. Built-in
policies wait or select an available device using current public information;
an unsupported custom policy that submits unavailable compute receives an
explicit rejection.

Pausing retains input read locks, output reservations and workspace allocations.
At a boundary, simultaneous completions occur before the atomic environment
group. If the group's external memory allocation exceeds capacity, the entire
group is rejected: its rate changes do not partially commit.

Compute rates do not change PCIe/SSD transfer prices or CE/SM shared-link byte
service. A zero GPU **compute** rate therefore does not declare the whole GPU,
its copy engines or the host-to-device link unavailable. Transfer interference
must be supplied through its own explicit service model. This distinction avoids
silently equating compute throttling with bandwidth throttling.

## Public observations and saved evidence

Policy observations contain only the currently effective normalized CPU/GPU
rate pairs. Future states, event times and fractional private work are withheld.
`DevicePrice.duration_ns` remains the immutable reference cost.
`DevicePrice.current_duration_ns` is a distinct whole-operation estimate at the
currently observed rate, assuming no future changes; it is `None` at zero
effective rate in model-owned observations. Existing strategy ranking rules
continue to use reference prices unless they explicitly adopt the current-state
estimate. The runtime itself always applies the declared service-rate timeline.

Private results include:

- accepted/rejected environment records with the declared rate pairs;
- service intervals with exact reference work served and their effective rate;
- `compute_progress`, with rational remaining work and current effective rate.

`state.running[*].remaining_ns` is an integer wall-time projection at the current
rate; at zero rate it is only the ceiling of remaining reference work and is not
a predicted completion. This field alone cannot restore a running task.
Persistent inventory documents remain explicitly **not resumable checkpoints**.
Supported checkpoints replay the same source-bound accepted commands at the same
absolute clock and bind exact fractional compute work in their runtime signature.

## Calibration boundary

A single device-wide rate is an intentionally limited model. Contention can
affect matrix multiplication, softmax, small launches and transfers differently.
An empirically fitted rate for one operator, shape, precision, execution context
and observed state must not be advertised as a universal GPU slowdown factor.
State-conditioned cost tables and measured state labels can provide the rate
for the evaluated operation; unobserved states remain uncalibrated. These runtime
arithmetic tests do not establish a causal physical relationship between any
telemetry counter and latency.

The supplied reference duration is treated as divisible service. If a measured
duration bundles fixed host launch or synchronization overhead with device work,
scaling its entire value is an approximation. A physical throughput model should
separate fixed overhead or validate an effective rate in that exact timing scope;
this implementation does not claim that host overhead follows device throughput.

## Connecting frozen target measurements

`profiling.state_calibration.runtime_adapter` connects a frozen campaign price
to the common runtime through an intentionally restricted one-task timing
carrier:

```python
from profiling.state_calibration.runtime_adapter import load_measured_target

target = load_measured_target(
    "path/to/campaign", case_id="h256-q32", mode="mixed"
)
runtime = target.build_session(
    changes=((40_000, "gpu_replay_load"), (120_000, "baseline"))
)
result = runtime.run()
print(result.evaluation.makespan_ns)
print(runtime.provenance())
```

The loader binds the frozen model, exact context, profile rows, native target
graph and source program hashes. It accepts the measured GPU `attention_pv`
target only; whole-iteration or CPU timing cannot become a GPU service rate.
Default behavior rejects failed source stability gates. Explicit
`allow_unaccepted_diagnostic=True` retains those failures and enables a labeled
timing diagnostic; numerical failures and unknown regimes never receive a
fallback. The complete campaign verifier remains required for the other timing
and alignment contracts.

The current price context requires the manifest's explicit
`primary_execution_boundary="legacy_double_sync"` and matching native boundary
evidence. Alternative asynchronous boundary diagnostics and missing historical
boundary declarations are rejected, because execution boundary is not yet an
independent axis of the fitted price table.

The carrier's 4 B input, 4 B output, 8 B fixture capacity and zero synthetic
workspace are declared test-fixture values. They do not replace native tensor
sizes, real target workspace or private CUDA scratch, and establish no physical
capacity accuracy. This API cannot attach its measured rate to an existing LLM
pipeline: that would incorrectly scale unrelated operators. It checks the time
and exact-progress connection for one measured target and does not itself
establish heldout prediction accuracy.
