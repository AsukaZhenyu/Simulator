# Read-only single-layer integration audit

Scope: `modeling/llm_infer_model/performance/serial_actions.py`,
`serial_distributions.py`, and `profiling/single_layer_calibration/{run,analysis,distribution_profiles}.py`.
This review inspected code only. It did not modify bound Python sources, run
tests, collect hardware data, inspect heldout outcomes, or certify calibration.

No blocking defect was identified in the reviewed prediction and acceptance
path:

- Completed actions, preceding gaps and the tail partition the serial timeline.
  Inclusive action parameters cannot also price gaps. The deterministic path
  delegates to the common Graph evaluator; the stochastic specialization checks
  the same ordered graph and sums integer-nanosecond sampled intervals.
- Profiling retains source observations, statistics, histograms and provenance.
  Modeling performs duration prediction and distribution propagation. Matched
  iteration arrays are used only for descriptive covariance/accounting checks;
  prediction draws each interval independently within one selected process.
- The original profile anchor remains signed and unchanged. The distribution
  artifact and combined wrapper have their own hashes; the global barrier binds
  every predeclared case/poll wrapper before heldout collection.
- Collection checks the barrier before every heldout job and at phase end.
  Evaluation checks wrapper bytes, reconstructs cached predictions, rehashes
  source evidence, and requires fresh heldout observations after both model and
  global freezes. Heldout values do not update parameters.
- The original median-error, compute/copy-source CV, heldout CV and
  instrumentation gates remain primary. Expected-mean error and interval
  coverage are supplements. The distribution candidate does not replace the
  primary result or relax its thresholds.

One nonblocking evidence-validation gap deserves a later guard:
`verified_phase()` validates its declared run inventory, while `records_for()`
independently obtains records by a directory glob. An extra matching
`record.json` outside the phase inventory can therefore enter freezing or
evaluation rather than being rejected as an undeclared observation. Fresh
exclusive campaign directories prevent the ordinary reuse case, but the
stronger invariant is exact equality between the records consumed for each
case/poll/split and the phase-bound subset. This review did not demonstrate
extra records in the current campaign. Address this after the current frozen
experiment rather than silently changing its sources.

Remaining declared limits matter: serial host-completed prices include backend
submission and synchronization; they are not exclusive device throughput.
Conditional independence does not preserve within-invocation correlations.
Monte Carlo rank bounds cover integration of frozen empirical distributions,
not hardware uncertainty. A successful software path cannot establish physical
capacity, unseen-strategy generalization, or acceptable hardware latency error.
