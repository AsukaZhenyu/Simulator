"""Conditional-independent propagation of explicit serial duration distributions.

Profiling supplies finite empirical supports and frequencies with provenance.
This module neither reads raw measurements, constructs their histograms, fits
coefficients, nor accepts complete-iteration durations as model parameters.

For each simulated invocation, choose one supplied profile process uniformly.
Within that process, draw each serial interval independently from its supplied
marginal distribution and sum it. This retains a shared process regime, while
deliberately discarding same-invocation cross-interval temporal correlations.
It is not complete trace resampling: each interval has its own independent draw.

The optimization below is valid only for SerialTimingProgram: the graph is a
single ordered chain of inclusive completed actions and disjoint host gaps.
Its sample sum is equivalent to the common Graph evaluator for integer-ns input
prices. No parallel-DAG, device-kernel, physical-capacity or accuracy claim is
made. Independent uninstrumented runs remain necessary for calibration.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
import random
import struct
import sys
from types import MappingProxyType
from typing import Mapping

from .parameters import (InvalidModel, Parameter, ParameterBundle, UnknownParameter,
                         encode_fraction, exact_fields, integer, nonempty)
from .serial_actions import (SerialTimingEvaluation, SerialTimingProgram,
                             build_serial_action_graph, evaluate_serial_actions)

SCHEMA_VERSION = "serial-conditional-duration-distributions-1"
QUANTILE_PROBABILITIES = (Fraction(1, 10), Fraction(1, 2), Fraction(9, 10))


@dataclass(frozen=True)
class DiscreteDuration:
    """A caller-supplied finite ns support and its positive integer frequencies.

    Frequencies are measurement counts, not arbitrary learned weights. Each
    value must be a known nonnegative integer nanosecond observation. In
    particular ``None`` cannot be represented by an observed zero. Repeated
    equal observations belong to one support value with a corresponding count;
    input supports must already be unique and increasing.
    """

    values_ns: tuple[int, ...]
    counts: tuple[int, ...]

    def __post_init__(self):
        if type(self.values_ns) is not tuple or not self.values_ns:
            raise InvalidModel("duration support must be a nonempty tuple")
        if type(self.counts) is not tuple or len(self.counts) != len(self.values_ns):
            raise InvalidModel("duration support and frequency tuples must have equal lengths")
        for value in self.values_ns:
            if value is None:
                raise UnknownParameter("unknown duration support cannot become zero")
            integer(value, "duration support in ns")
        for count in self.counts:
            integer(count, "duration observation count", 1)
        if any(a >= b for a, b in zip(self.values_ns, self.values_ns[1:])):
            raise InvalidModel("duration support must be strictly increasing without duplicates")

    @property
    def total_count(self):
        return sum(self.counts)

    @property
    def expectation_ns(self):
        return Fraction(sum(value * count for value, count in zip(self.values_ns, self.counts)),
                        self.total_count)

    def to_dict(self):
        return dict(values_ns=list(self.values_ns), counts=list(self.counts))

    @classmethod
    def from_dict(cls, value):
        obj = exact_fields(value, {"values_ns", "counts"}, "discrete duration")
        if not isinstance(obj["values_ns"], list) or not isinstance(obj["counts"], list):
            raise InvalidModel("duration supports and frequencies must be JSON lists")
        return cls(tuple(obj["values_ns"]), tuple(obj["counts"]))


@dataclass(frozen=True)
class ProcessDurationDistribution:
    """Known duration marginals associated with one independent profile process.

    Exact parameter names bind these marginals to SerialTimingProgram. Caller
    evidence must establish numerical correctness, profile-only provenance,
    chronology, timing boundaries and truly independent process identities.
    This model does not infer those facts from a process name or frequencies.
    """

    process_id: str
    durations: Mapping[str, DiscreteDuration]

    def __post_init__(self):
        nonempty(self.process_id, "profile process identity")
        if not isinstance(self.durations, Mapping) or not self.durations:
            raise InvalidModel("process durations must be a nonempty mapping")
        copied = {}
        for name, distribution in self.durations.items():
            nonempty(name, "duration parameter name")
            if distribution is None:
                raise UnknownParameter(f"unknown duration distribution: {name}")
            if not isinstance(distribution, DiscreteDuration):
                raise InvalidModel("process durations must contain DiscreteDuration values")
            copied[name] = distribution
        object.__setattr__(self, "durations", MappingProxyType(copied))

    def to_dict(self):
        return dict(process_id=self.process_id,
                    durations={name: distribution.to_dict()
                               for name, distribution in self.durations.items()})

    @classmethod
    def from_dict(cls, value):
        obj = exact_fields(value, {"process_id", "durations"}, "process durations")
        if not isinstance(obj["durations"], dict):
            raise InvalidModel("process durations must be a JSON mapping")
        return cls(obj["process_id"], {name: DiscreteDuration.from_dict(distribution)
                   for name, distribution in obj["durations"].items()})


def distributions_to_dict(processes: tuple[ProcessDurationDistribution, ...]):
    _validate_process_tuple(processes)
    return dict(schema_version=SCHEMA_VERSION,
                process_weight="equal_independent_profile_process_weight",
                processes=[process.to_dict() for process in processes])


def distributions_from_dict(value):
    obj = exact_fields(value, {"schema_version", "process_weight", "processes"},
                       "serial duration distributions")
    if obj["schema_version"] != SCHEMA_VERSION:
        raise InvalidModel(f"requires {SCHEMA_VERSION}")
    if obj["process_weight"] != "equal_independent_profile_process_weight":
        raise InvalidModel("serial duration propagation requires equal process weights")
    if not isinstance(obj["processes"], list):
        raise InvalidModel("duration processes must be a JSON list")
    result = tuple(ProcessDurationDistribution.from_dict(item) for item in obj["processes"])
    _validate_process_tuple(result)
    return result


def _validate_process_tuple(processes):
    if type(processes) is not tuple or not processes or any(
            not isinstance(process, ProcessDurationDistribution) for process in processes):
        raise InvalidModel("processes must be a nonempty tuple of ProcessDurationDistribution")
    if len({process.process_id for process in processes}) != len(processes):
        raise InvalidModel("profile process identities must be unique")


def _serial_intervals(program):
    graph = build_serial_action_graph(program)
    intervals = []
    for action in program.actions:
        intervals.append((f"action:{action.action_index}:gap_before", action.gap_before_parameter,
                          action.action_index, "gap_before"))
        intervals.append((f"action:{action.action_index}:completed", action.duration_parameter,
                          action.action_index, "completed_action"))
    intervals.append(("program:tail", program.tail_parameter, None, "tail"))
    # No generic DAG argument is accepted. Check the specialized sum corresponds
    # exactly to the Graph emitted by the deterministic modeling adapter.
    if len(graph.tasks) != len(intervals):
        raise InvalidModel("serial graph and interval inventory disagree")
    for index, (task, interval) in enumerate(zip(graph.tasks, intervals)):
        if (task.id != interval[0] or task.fixed_parameter != interval[1]
                or task.resource != "serial_host_completed" or task.extra_resources
                or task.demands or task.memory
                or task.depends_on != (() if index == 0 else (graph.tasks[index - 1].id,))):
            raise InvalidModel("distribution fast path requires an unmodified strictly serial fixed-duration graph")
    return tuple(intervals)


def _quantile(sorted_values, probability):
    # Generalized inverse empirical CDF: Q(q) = inf{x: F_n(x) >= q}.
    rank = math.ceil(probability * len(sorted_values))
    return sorted_values[max(0, min(len(sorted_values) - 1, rank - 1))]


@dataclass(frozen=True)
class ExpectedIntervalContribution:
    interval_id: str
    parameter: str
    action_index: int | None
    part: str
    expectation_ns: Fraction

    def to_dict(self):
        return dict(interval_id=self.interval_id, parameter=self.parameter,
                    action_index=self.action_index, part=self.part,
                    expectation_ns=encode_fraction(self.expectation_ns))


@dataclass(frozen=True)
class SerialDistributionEvaluation:
    expectation_ns: Fraction
    simulated_totals_ns: tuple[int, ...]
    quantiles_ns: tuple[tuple[Fraction, int], ...]
    quantile_value_intervals_ns: tuple[tuple[Fraction, int, int], ...]
    expected_intervals: tuple[ExpectedIntervalContribution, ...]
    process_draw_counts: tuple[tuple[str, int], ...]
    deterministic_reference: SerialTimingEvaluation
    minimum_possible_ns: int
    maximum_possible_ns: int
    seed: int
    cdf_failure_probability: Fraction
    cdf_rank_error_bound: float

    @property
    def trials(self):
        return len(self.simulated_totals_ns)

    def to_dict(self):
        quantiles = {str(probability): value for probability, value in self.quantiles_ns}
        # This digest binds the generated sequence, without mistaking simulated
        # totals for measured source observations or requiring large JSON output.
        digest = hashlib.sha256()
        for value in self.simulated_totals_ns:
            encoded = str(value).encode("ascii")
            digest.update(struct.pack(">Q", len(encoded)))
            digest.update(encoded)
        return dict(schema_version="serial-conditional-distribution-prediction-1",
            expected_duration_ns=encode_fraction(self.expectation_ns),
            monte_carlo_mean_ns=encode_fraction(Fraction(sum(self.simulated_totals_ns), self.trials)),
            p10_ns=quantiles["1/10"], p50_ns=quantiles["1/2"], p90_ns=quantiles["9/10"],
            interval_expected_contributions=[item.to_dict() for item in self.expected_intervals],
            possible_duration_bounds_ns=dict(minimum=self.minimum_possible_ns, maximum=self.maximum_possible_ns),
            monte_carlo=dict(seed=self.seed, trials=self.trials,
                random_engine="python.random.Random/MT19937; integer-randrange; serial-propagation-1",
                python_version=sys.version.split()[0],
                process_draw_counts=[dict(process_id=name, count=count) for name, count in self.process_draw_counts],
                simulated_sequence_sha256=digest.hexdigest(),
                quantile_definition="generalized inverse empirical CDF (nearest rank; no interpolation)",
                cdf_failure_probability=encode_fraction(self.cdf_failure_probability),
                cdf_rank_error_bound=self.cdf_rank_error_bound,
                quantile_value_intervals_ns=[dict(probability=encode_fraction(q), lower=lower, upper=upper)
                    for q, lower, upper in self.quantile_value_intervals_ns],
                rank_error_scope="DKW uniform CDF integration bound conditional on frozen supplied marginals; not hardware confidence intervals",
                source_estimation_or_hardware_uncertainty_included=False),
            deterministic_mean_parameter_reference=self.deterministic_reference.to_dict(),
            deterministic_reference_scope="common Graph evaluation at exact marginal mean parameters; integer stage rounding disclosed; not the predicted median",
            prediction_provider="llm_infer_model.performance.serial_distributions.evaluate_serial_distributions",
            process_mixture_equal_weight=True,
            conditional_cross_interval_independence_assumed=True,
            process_common_regime_retained=True,
            same_invocation_cross_interval_correlation_modeled=False,
            whole_trace_resampling=False,
            whole_iteration_totals_used_as_parameters=False,
            serial_specialization_only=True,
            physical_kernel_timeline_claimed=False, physical_capacity_claimed=False,
            instrumentation_overhead_subtracted=False,
            independently_validated_accuracy_claimed=False,
            raw_measurements_read_here=False, profile_histograms_estimated_here=False)


def evaluate_serial_distributions(program: SerialTimingProgram,
        processes: tuple[ProcessDurationDistribution, ...], *, trials: int, seed: int,
        cdf_failure_probability: Fraction = Fraction(1, 100)) -> SerialDistributionEvaluation:
    """Propagate explicit marginal distributions through a strictly serial model.

    The exact expectation uses linearity and does not require independence.
    Median/tail predictions do require the stated conditional independence
    assumption. Monte Carlo uncertainty describes numerical integration of the
    fixed empirical input model; it does not cover measurement error, process
    representativeness, instrumentation, correlation or physical model bias.
    """
    intervals = _serial_intervals(program)
    _validate_process_tuple(processes)
    integer(trials, "Monte Carlo trial count", 1)
    integer(seed, "Monte Carlo seed")
    if type(cdf_failure_probability) is not Fraction or not 0 < cdf_failure_probability < 1:
        raise InvalidModel("CDF failure probability must be an exact Fraction strictly between zero and one")
    probability_float = float(cdf_failure_probability)
    if probability_float == 0 or not math.isfinite(probability_float):
        raise InvalidModel("CDF failure probability is outside finite numerical integration precision")
    names = {interval[1] for interval in intervals}
    for process in processes:
        missing = names - process.durations.keys()
        if missing:
            raise UnknownParameter(f"process {process.process_id} misses explicit intervals: {sorted(missing)}")
        if set(process.durations) != names:
            raise InvalidModel("duration inputs may contain only referenced serial interval parameters; no totals or extra prices")
    means = {name: sum((process.durations[name].expectation_ns for process in processes), Fraction())
             / len(processes) for name in names}
    contributions = tuple(ExpectedIntervalContribution(interval_id, name, action_index, part, means[name])
                          for interval_id, name, action_index, part in intervals)
    expectation = sum((item.expectation_ns for item in contributions), Fraction())
    reference_bundle = ParameterBundle(tuple(Parameter(name, means[name], "ns", "declared", (),
        "derived exact marginal expectation for deterministic serial Graph reference only") for name in sorted(names)))
    reference = evaluate_serial_actions(program, reference_bundle)
    if reference.unrounded_interval_sum_ns != expectation:
        raise InvalidModel("deterministic Graph reference and exact serial expectation disagree")

    compiled = []
    minima, maxima = [], []
    for process in processes:
        samplers = []
        for _, name, _, _ in intervals:
            distribution = process.durations[name]
            cumulative, count = [], 0
            for frequency in distribution.counts:
                count += frequency
                cumulative.append(count)
            samplers.append((distribution.values_ns, tuple(cumulative), count))
        compiled.append(tuple(samplers))
        minima.append(sum(process.durations[name].values_ns[0] for _, name, _, _ in intervals))
        maxima.append(sum(process.durations[name].values_ns[-1] for _, name, _, _ in intervals))
    minimum, maximum = min(minima), max(maxima)
    generator = random.Random(seed)
    totals, draws = [], [0] * len(processes)
    for _ in range(trials):
        process_index = generator.randrange(len(processes))
        draws[process_index] += 1
        total = 0
        # Every interval draws independently, even if it explicitly references
        # the same parameter as another interval. No iteration index is shared.
        for values, cumulative, count in compiled[process_index]:
            if len(values) == 1:
                total += values[0]
            else:
                total += values[bisect_right(cumulative, generator.randrange(count))]
        totals.append(total)
    ordered = sorted(totals)
    quantiles = tuple((q, _quantile(ordered, q)) for q in QUANTILE_PROBABILITIES)
    epsilon = math.sqrt((math.log(2) - math.log(probability_float)) / (2 * trials))
    uncertainty = []
    for q in QUANTILE_PROBABILITIES:
        lower_q, upper_q = float(q) - epsilon, float(q) + epsilon
        lower = minimum if lower_q <= 0 else _quantile(ordered, lower_q)
        upper = maximum if upper_q >= 1 else _quantile(ordered, upper_q)
        uncertainty.append((q, lower, upper))
    return SerialDistributionEvaluation(expectation, tuple(totals), quantiles, tuple(uncertainty),
        contributions, tuple((process.process_id, count) for process, count in zip(processes, draws)),
        reference, minimum, maximum, seed, cdf_failure_probability, epsilon)
