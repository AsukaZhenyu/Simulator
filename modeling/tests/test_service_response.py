"""Analytic declared scenarios for one resource-agnostic stateful service law."""
from dataclasses import replace
from fractions import Fraction
import json
import math
import unittest

from llm_infer_model.performance import (
    Graph, INSTANTANEOUS_RATE, InvalidModel, Parameter, ParameterBundle,
    ResourcePoint, ResourceTimeline, Task, UnknownParameter, WorkDemand,
)
from llm_infer_model.performance.service_response import (
    BackgroundReservoir, ServiceInterference, ServiceRequestPoint,
    ServiceRequestTrajectory, ServiceResponseModel, ServiceResponseParameter,
    ServiceResponseResult, build_service_response,
)

FULL = 1_000_000_000
MS = 1_000_000


def value(name, amount, unit="dimensionless"):
    return ServiceResponseParameter(name, amount, unit, "declared", ("hand-derived declaration:" + name,),
                                    "normalized residual-interference proxy; no hardware claim")


def model(*, names=("load",), targets=("supply",), capacity=10, arrival=1, drain=1,
          activity=1, backlog=0, coefficients=None, domains=None):
    parameters, competitors, links = [], [], []
    for name in names:
        parameters.extend((value(name + ":cap", capacity, "normalized_work"),
            value(name + ":arrival", arrival, "normalized_work/ns"),
            value(name + ":drain", drain, "normalized_work/ns"),
            value(name + ":activity", activity), value(name + ":backlog", backlog)))
        competitors.append(BackgroundReservoir(name, name + ":cap", name + ":arrival", name + ":drain",
                                               name + ":activity", name + ":backlog"))
        for target in targets:
            name_k = target + ":" + name + ":k"
            coefficient = coefficients.get((target, name), 1) if coefficients is not None else 1
            parameters.append(value(name_k, coefficient))
            domain = domains.get((target, name), "nonnegative_contention") if domains else "nonnegative_contention"
            links.append(ServiceInterference(target, name, name_k, domain))
    return ServiceResponseModel("a" * 64, tuple(parameters), tuple(competitors), tuple(links),
                                ("explicit finite normalized reservoir law; not an identified hardware queue",))


def bundle(*parameters):
    if not parameters:
        parameters = (("supply", FULL, "thread_cpu_ns/s"),)
    return ParameterBundle(tuple(Parameter(name, amount, unit, "declared", ("hand scenario",),
        "base service excludes declared competitor interference") for name, amount, unit in parameters))


def trajectory(*, initial=None, work=None, points=(), origin="predeclared_scenario"):
    initial = {"load": 1} if initial is None else initial
    work = {name: 0 for name in initial} if work is None else work
    return ServiceRequestTrajectory(initial, work, points, origin, ("declared request/initial state",),
                                    "scenario_ns", "scenario-start")


def build(spec=None, base=None, requests=None, *, end=100, step=1, max_points=1000,
          baseline=None, replay=False):
    return build_service_response(spec if spec is not None else model(), base if base is not None else bundle(),
        requests if requests is not None else trajectory(), end_ns=end, max_step_ns=step,
        max_points=max_points, baseline_timeline=baseline, allow_observed_diagnostic=replay)


def work(amount=10, parameter="supply", unit="thread_cpu_ns", task_id="work"):
    return Graph((Task(task_id, task_id, (WorkDemand(parameter, amount, unit),)),))


class ServiceResponseTests(unittest.TestCase):
    def test_empty_queue_can_still_have_active_background_service(self):
        result = build(model(capacity=10, arrival=1, drain=1, activity=1, backlog=1))
        self.assertTrue(all(s.backlog["load"] == 0 for s in result.samples))
        self.assertEqual(result.samples[0].pressure["load"], 1)
        self.assertEqual(result.samples[0].rates["supply"], FULL // 2)
        self.assertEqual(result.evaluate(work()).duration_ns, 20)

    def test_unsaturated_empty_queue_uses_explicit_fractional_activity(self):
        result = build(model(arrival=1, drain=4, activity=1, backlog=1))
        self.assertEqual(result.samples[0].pressure["load"], Fraction(1, 4))
        self.assertEqual(result.evaluate(work()).duration_ns, 13)

    def test_same_off_request_different_initial_history_changes_completion(self):
        requests = trajectory(initial={"load": 0}, work={"load": 5})
        busy = build(requests=requests)
        quiet = build(requests=trajectory(initial={"load": 0}, work={"load": 0}))
        self.assertEqual([(s.at_ns, s.backlog["load"]) for s in busy.samples],
                         [(0, Fraction(5)), (5, Fraction(0)), (100, Fraction(0))])
        # First 5 ns supply only 2.5 ns work; 7.5 ns work remains after drain.
        self.assertEqual(busy.evaluate(work()).duration_ns, 13)
        self.assertEqual(quiet.evaluate(work()).duration_ns, 10)

    def test_off_request_does_not_discard_pending_residual_proxy(self):
        requests = trajectory(points=(ServiceRequestPoint(5, {"load": 0}),))
        result = build(model(capacity=10, arrival=3, drain=1, activity=1, backlog=0), requests=requests)
        samples = {s.at_ns: s for s in result.samples}
        self.assertEqual(samples[5].backlog["load"], 10)
        self.assertEqual(samples[5].pressure["load"], 1)
        self.assertEqual(samples[15].backlog["load"], 0)
        self.assertEqual(samples[15].rates["supply"], FULL)
        self.assertEqual(result.evaluate(work()).duration_ns, 18)

    def test_queue_saturation_and_drain_are_exact_and_bounded(self):
        requests = trajectory(points=(ServiceRequestPoint(15, {"load": 0}),))
        result = build(model(capacity=10, arrival=3, drain=1, activity=0, backlog=1),
                       requests=requests, step=3)
        samples = {s.at_ns: s for s in result.samples}
        self.assertEqual(samples[5].backlog["load"], 10)
        self.assertEqual(samples[15].backlog["load"], 10)
        self.assertEqual(samples[25].backlog["load"], 0)
        self.assertTrue(all(0 <= s.backlog["load"] <= 10 for s in result.samples))

    def test_combined_pressure_uses_same_law_without_combined_fit(self):
        spec = model(names=("cpu_load", "gpu_load"),
                     coefficients={("supply", "cpu_load"): 1, ("supply", "gpu_load"): 2})
        result = build(spec, requests=trajectory(initial={"cpu_load": 1, "gpu_load": 1}))
        self.assertEqual(result.samples[0].rates["supply"], FULL // 4)
        self.assertEqual(result.evaluate(work()).duration_ns, 40)
        self.assertFalse(result.accuracy_qualified)

    def test_cpu_gpu_and_copy_use_identical_generic_equation_and_explicit_cross_terms(self):
        spec = model(names=("cpu_load", "gpu_load"), targets=("cpu", "gpu", "copy"), coefficients={
            ("cpu", "cpu_load"): 1, ("cpu", "gpu_load"): 2,
            ("gpu", "cpu_load"): 1, ("gpu", "gpu_load"): 3,
            ("copy", "cpu_load"): 0, ("copy", "gpu_load"): 1})
        base = bundle(("cpu", FULL, "thread_cpu_ns/s"), ("gpu", 2 * FULL, "flop/s"),
                      ("copy", FULL, "byte/s"))
        result = build(spec, base, trajectory(initial={"cpu_load": 1, "gpu_load": 0}))
        graph = Graph((Task("cpu", "cpu", (WorkDemand("cpu", 10, "thread_cpu_ns"),)),
                       Task("gpu", "gpu", (WorkDemand("gpu", 10, "flop"),)),
                       Task("copy", "copy", (WorkDemand("copy", 10, "byte"),))))
        self.assertEqual({t.task_id: t.end_ns for t in result.evaluate(graph).tasks},
                         {"cpu": 20, "gpu": 10, "copy": 10})

    def test_slowed_copy_shifts_dependent_compute_against_same_absolute_state_trace(self):
        spec = model(targets=("copy", "gpu"), coefficients={("copy", "load"): 0, ("gpu", "load"): 1})
        base = bundle(("copy", FULL, "byte/s"), ("gpu", FULL, "flop/s"))
        requests = trajectory(initial={"load": 0}, points=(ServiceRequestPoint(12, {"load": 1}),))
        baseline = ResourceTimeline((ResourcePoint(0, {"copy": FULL // 2}),))
        result = build(spec, base, requests, baseline=baseline)
        graph = Graph((Task("copy", "copy", (WorkDemand("copy", 10, "byte"),)),
                       Task("gpu", "gpu", (WorkDemand("gpu", 10, "flop"),), depends_on=("copy",))))
        times = {t.task_id: (t.start_ns, t.end_ns) for t in result.evaluate(graph).tasks}
        self.assertEqual(times, {"copy": (0, 20), "gpu": (20, 40)})

    def test_precomputed_window_preserves_history_and_rebases_only_time(self):
        spec = model(capacity=10, arrival=2, drain=1, activity=0, backlog=1)
        requests = trajectory(points=(ServiceRequestPoint(10, {"load": 0}),))
        result = build(spec, requests=requests)
        window = result.window(15, 25)
        self.assertEqual(window.points[0].rates["supply"], Fraction(2 * FULL, 3))
        self.assertEqual(window.points[0].at_ns, 0)
        self.assertEqual(window.points[5].at_ns, 5)
        self.assertEqual(window.points[5].rates["supply"], FULL)
        self.assertEqual(result.samples[0].backlog["load"], 0)

    def test_flat_empty_full_and_zero_coefficient_periods_do_not_emit_fixed_cadence(self):
        self.assertEqual(len(build(end=1_000_000, step=1, max_points=4).samples), 2)
        full = build(model(capacity=10, arrival=3, drain=1, activity=1, backlog=0),
                     end=1_000_000, step=1, max_points=4)
        self.assertEqual([s.at_ns for s in full.samples], [0, 5, 1_000_000])
        zero = build(model(capacity=10, arrival=3, drain=1, activity=1, backlog=1,
                           coefficients={("supply", "load"): 0}), end=1_000_000, step=1, max_points=4)
        self.assertEqual(len(zero.samples), 3)

    def test_half_step_converges_to_analytic_nonconstant_service(self):
        spec = model(capacity=1, arrival=Fraction(2, 10 * MS), drain=Fraction(1, 10 * MS), activity=0, backlog=1)
        coarse = build(spec, end=20 * MS, step=250_000)
        fine = build(spec, end=20 * MS, step=125_000)
        exact_end = 10 * MS * math.expm1(0.5)  # integral dt/(1+t/10ms) = 5ms.
        coarse_end = coarse.evaluate(work(5 * MS)).duration_ns
        fine_end = fine.evaluate(work(5 * MS)).duration_ns
        self.assertLess(abs(fine_end - exact_end), abs(coarse_end - exact_end))
        self.assertLess(abs(fine_end - exact_end), 0.6 * abs(coarse_end - exact_end))
        self.assertLess(fine.integrated_service_error_bound_work["supply"],
                        coarse.integrated_service_error_bound_work["supply"])
        for result in (coarse, fine):
            supplied = sum(s.rates["supply"] * (s.interval_end_ns - s.at_ns) / FULL
                           for s in result.samples if s.at_ns < 10 * MS)
            exact_supply = 10 * MS * math.log(2)
            self.assertLessEqual(abs(float(supplied) - exact_supply),
                                 float(result.integrated_service_error_bound_work["supply"]) + 1e-6)

    def test_fractional_empty_event_is_ceiled_and_error_is_reported(self):
        result = build(model(capacity=1, arrival=0, drain=Fraction(2, 3), activity=1, backlog=0),
                       requests=trajectory(initial={"load": 0}, work={"load": 1}), step=100)
        self.assertEqual([s.at_ns for s in result.samples], [0, 2, 100])
        self.assertEqual(result.samples[1].backlog["load"], 0)
        self.assertGreater(result.integrated_service_error_bound_work["supply"], 0)

    def test_baseline_updates_and_unrelated_memory_changes_are_preserved(self):
        base = bundle(("supply", FULL, "thread_cpu_ns/s"), ("unbound_gpu", FULL, "flop/s"))
        baseline = ResourceTimeline((ResourcePoint(3, {"supply": FULL // 2, "unbound_gpu": FULL // 4},
                                                    {"dram": 20}, {"dram": 4}),))
        result = build(base=base, baseline=baseline)
        point = next(p for p in result.timeline.points if p.at_ns == 3)
        self.assertEqual(point.rates["supply"], FULL // 4)
        self.assertEqual(point.rates["unbound_gpu"], FULL // 4)
        entry = result.window(5, 10).points[0]
        self.assertEqual(entry.rates["unbound_gpu"], FULL // 4)
        self.assertEqual(dict(entry.capacities), {"dram": 20})
        self.assertEqual(dict(entry.external_memory), {"dram": 4})

    def test_serial_complete_responses_and_explicit_zero_controls_use_common_engine(self):
        base = bundle(("supply", FULL // 10, "invocation/s"),
                      ("control", INSTANTANEOUS_RATE, "invocation/s"), ("gap", 3, "ns"))
        result = build(base=base)
        evaluation = result.evaluate_serial_response(("gap", "control", "supply"), start_ns=7)
        self.assertEqual(evaluation.duration_ns, 23)
        self.assertEqual([t.start_ns for t in evaluation.tasks], [0, 3, 3])

    def test_serial_wrapper_rejects_inferred_physical_work_and_horizon_extrapolation(self):
        result = build()
        with self.assertRaisesRegex(InvalidModel, "not physical service"):
            result.evaluate_serial_response(("supply",))
        with self.assertRaisesRegex(UnknownParameter, "horizon"):
            result.evaluate(work(), start_ns=99)
        with self.assertRaisesRegex(UnknownParameter, "horizon"): result.window(0, 101)

    def test_model_requests_and_derived_result_round_trip_with_baseline(self):
        baseline = ResourceTimeline((ResourcePoint(8, {"supply": FULL // 2}),))
        result = build(baseline=baseline)
        encoded = json.loads(json.dumps(result.to_dict()))
        restored = ServiceResponseResult.from_dict(encoded)
        self.assertEqual(restored.to_dict(), encoded)
        self.assertEqual(restored.evaluate(work()), result.evaluate(work()))
        self.assertFalse(encoded["accuracy_qualified"])
        self.assertFalse(encoded["model"]["source_qualified"])
        encoded["samples"][0]["backlog"]["load"]["numerator"] += 1
        with self.assertRaisesRegex(InvalidModel, "differs from model reconstruction"):
            ServiceResponseResult.from_dict(encoded)

    def test_observed_requests_and_initial_state_require_explicit_unqualified_replay(self):
        requests = trajectory(origin="observed_diagnostic")
        with self.assertRaisesRegex(InvalidModel, "explicit replay"): build(requests=requests)
        result = build(requests=requests, replay=True)
        self.assertTrue(result.diagnostic_replay)
        self.assertFalse(result.accuracy_qualified)
        self.assertTrue(ServiceResponseResult.from_dict(result.to_dict()).diagnostic_replay)

    def test_missing_or_unknown_coefficients_never_become_zero(self):
        valid = model(names=("cpu", "gpu"))
        with self.assertRaisesRegex(InvalidModel, "every target/competitor"):
            replace(valid, interference=valid.interference[:-1])
        unknown = replace(valid, parameters=tuple(replace(p, value=None) if p.name.endswith(":k") else p
                                                  for p in valid.parameters))
        with self.assertRaises(UnknownParameter):
            build(unknown, requests=trajectory(initial={"cpu": 1, "gpu": 1}))

    def test_unknown_workload_scales_or_base_supply_reject(self):
        for suffix in (":cap", ":arrival", ":drain", ":activity", ":backlog"):
            valid = model()
            unknown = replace(valid, parameters=tuple(replace(p, value=None) if p.name.endswith(suffix) else p
                                                      for p in valid.parameters))
            with self.subTest(parameter=suffix), self.assertRaises(UnknownParameter): build(unknown)
        with self.assertRaises(UnknownParameter): build(base=bundle(("supply", None, "thread_cpu_ns/s")))
        with self.assertRaises(UnknownParameter): build(base=ParameterBundle(()))

    def test_no_guess_for_initial_history_unknown_competitor_or_utilization(self):
        for requests in (trajectory(initial={}), trajectory(work={}),
                         trajectory(points=(ServiceRequestPoint(1, {"unknown": 1}),)),
                         trajectory(work={"load": 11})):
            with self.subTest(requests=requests), self.assertRaises(InvalidModel): build(requests=requests)
        for request in (True, 0.5, 2, -1):
            with self.subTest(request=request), self.assertRaises(InvalidModel):
                ServiceRequestPoint(0, {"load": request})

    def test_nonpositive_capacity_or_drain_and_negative_weights_are_rejected(self):
        for changes in (dict(capacity=0), dict(drain=0), dict(activity=-1), dict(backlog=-1)):
            with self.subTest(changes=changes), self.assertRaises(InvalidModel): model(**changes)
        for bad in (0.5, True):
            with self.subTest(value=bad), self.assertRaises(InvalidModel): value("bad", bad)
        with self.assertRaises(InvalidModel): value("bad", -1, "normalized_work/ns")

    def test_signed_inclusive_response_is_explicit_and_not_physical_negative_contention(self):
        with self.assertRaisesRegex(InvalidModel, "explicit signed"):
            model(coefficients={("supply", "load"): Fraction(-1, 2)})
        spec = model(coefficients={("supply", "load"): Fraction(-1, 2)},
                     domains={("supply", "load"): "signed_inclusive_response"})
        with self.assertRaisesRegex(InvalidModel, "require invocation/s"): build(spec)
        result = build(spec, bundle(("supply", FULL // 10, "invocation/s")))
        self.assertEqual(result.evaluate_serial_response(("supply",)).duration_ns, 5)
        encoded = json.loads(json.dumps(result.to_dict()))
        self.assertEqual(encoded["model"]["interference"][0]["coefficient_domain"], "signed_inclusive_response")
        self.assertEqual(ServiceResponseResult.from_dict(encoded).to_dict(), encoded)
        self.assertFalse(result.accuracy_qualified)

    def test_signed_denominator_guard_checks_full_domain_not_only_current_state(self):
        spec = model(activity=1, backlog=1, coefficients={("supply", "load"): Fraction(-1, 2)},
                     domains={("supply", "load"): "signed_inclusive_response"})
        with self.assertRaisesRegex(InvalidModel, "full allowed pressure"):
            build(spec, bundle(("supply", FULL, "invocation/s")), trajectory(initial={"load": 0}))

    def test_negative_cross_terms_are_guarded_together_without_clipping(self):
        spec = model(names=("cpu", "gpu"), coefficients={
            ("supply", "cpu"): Fraction(-3, 5), ("supply", "gpu"): Fraction(-3, 5)},
            domains={("supply", "cpu"): "signed_inclusive_response", ("supply", "gpu"): "signed_inclusive_response"})
        with self.assertRaisesRegex(InvalidModel, "denominator"):
            build(spec, bundle(("supply", FULL, "invocation/s")), trajectory(initial={"cpu": 0, "gpu": 0}))

    def test_instantaneous_target_and_wrong_units_are_rejected(self):
        with self.assertRaisesRegex(InvalidModel, "instantaneous response"):
            build(base=bundle(("supply", INSTANTANEOUS_RATE, "invocation/s")))
        with self.assertRaisesRegex(InvalidModel, "service-rate parameters"):
            build(base=bundle(("supply", 10, "ns")))
        with self.assertRaisesRegex(InvalidModel, "instantaneous baseline"):
            build(base=bundle(("supply", FULL, "invocation/s")),
                  baseline=ResourceTimeline((ResourcePoint(3, {"supply": INSTANTANEOUS_RATE}),)))

    def test_explicit_point_budget_bounds_nonconstant_approximation(self):
        with self.assertRaisesRegex(InvalidModel, "point budget"):
            build(model(capacity=1000, arrival=2, drain=1, activity=0, backlog=1),
                  end=100, step=1, max_points=10)
        with self.assertRaisesRegex(InvalidModel, "point budget"): build(max_points=200_001)
        for options in (dict(step=0), dict(end=0), dict(max_points=1)):
            with self.subTest(options=options), self.assertRaises(InvalidModel): build(**options)

    def test_provenance_schema_and_structure_are_validated(self):
        with self.assertRaises(InvalidModel): replace(model(), assumptions=())
        with self.assertRaises(InvalidModel): replace(model(), context_sha256="unknown")
        with self.assertRaises(InvalidModel): replace(value("a", 1), evidence=())
        with self.assertRaises(InvalidModel): replace(trajectory(), origin_id="")
        with self.assertRaises(InvalidModel): replace(trajectory(), input_origin="measured_utilization")
        encoded = model().to_dict(); encoded["schema_version"] = "unknown"
        with self.assertRaises(InvalidModel): ServiceResponseModel.from_dict(encoded)
        encoded = trajectory().to_dict(); encoded["schema_version"] = "unknown"
        with self.assertRaises(InvalidModel): ServiceRequestTrajectory.from_dict(encoded)


if __name__ == "__main__": unittest.main()
