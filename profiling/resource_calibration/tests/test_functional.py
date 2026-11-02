"""Hand-calculable resource functional tests; never launch hardware probes."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import itertools
import json
import unittest

from profiling.resource_calibration.functional import (
    CalibrationRow, FitConfig, Hardware, Infeasible, Operation, ResourceFunctional,
    ResourceState, ResourceTrace, TraceSegment, UnknownCapacity, UnknownSupport,
    evaluate_heldout, fit_model, predict, predicted_duration_ns, predict_sequence,
)


HW = Hardware("synthetic-hardware", 1000, 1000)
SOURCE = "a" * 64
FROZEN = 1_000_000


def operation(resource="cpu", **changes):
    values = dict(key="synthetic-matrix", resource=resource, reference_ns=100,
                  constants=(("shape", "8x8"), ("dtype", "f32"), ("execution_context", "serial"),
                             ("timing_boundary", "submit-and-sync"), ("backend", resource)),
                  dram_bytes=10, vram_bytes=10, memory_footprint_complete=True)
    values.update(changes)
    return Operation(**values)


def state(cpu=0, gpu=0, dram=0, vram=0):
    return ResourceState(cpu, gpu, dram, vram)


def training_rows(op=None, *, coefficients=(100, 100, 0, 0, 0, 0), semantics="external"):
    op = operation() if op is None else op
    result = []
    for index, point in enumerate(itertools.product((0, 1), (0, 1), (0, 500), (0, 500))):
        cpu, gpu, dram, vram = point
        features = (1, cpu, gpu, Fraction(dram, 1000), Fraction(vram, 1000), cpu * gpu)
        duration = sum(Fraction(c) * x for c, x in zip(coefficients, features))
        assert duration.denominator == 1 and duration > 0
        for repeat in range(3):
            number = index * 3 + repeat
            result.append(CalibrationRow(f"train-{op.resource}-{number}", "train", f"run-{repeat}", 1,
                                         number * 1000, int(duration), number * 1000 + int(duration) + 100,
                                         op, ResourceState(*point), SOURCE, "serial-v1", "qpc-session", semantics, True))
    return result


def model_for(op=None, *, coefficients=(100, 100, 0, 0, 0, 0), semantics="external", regularization=0):
    return fit_model(training_rows(op, coefficients=coefficients, semantics=semantics), HW, FROZEN,
                     FitConfig(regularization=regularization))


def heldout(op=None, resource_state=None, elapsed=150, **changes):
    values = dict(row_id="heldout-0", split="heldout", run_id="new-run", seed=2, started_ns=FROZEN,
                  elapsed_ns=elapsed, measured_until_ns=FROZEN + elapsed + 100,
                  operation=operation() if op is None else op,
                  state=state(Fraction(1, 2)) if resource_state is None else resource_state,
                  source_sha256=SOURCE, protocol_id="serial-v1", clock_id="qpc-session",
                  state_semantics="external", numerical_passed=True)
    values.update(changes)
    return CalibrationRow(**values)


class SchemaTests(unittest.TestCase):
    def test_exact_rationals_and_round_trip(self):
        original = ResourceState("0.25", "1/3", 20, 30)
        self.assertEqual(original.cpu_busy, Fraction(1, 4))
        self.assertEqual(ResourceState.from_dict(json.loads(json.dumps(original.to_dict()))), original)
        with self.assertRaises(FrozenInstanceError):
            original.dram_used_bytes = 99

    def test_invalid_or_unknown_state_is_not_filled(self):
        for bad in (None, True, .5, "nan", "inf", "1/0", -1, 2):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ResourceState(bad, 0, 0, 0)
        for bad in (None, True, -1, .5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ResourceState(0, 0, bad, 0)
        with self.assertRaises(ValueError):
            ResourceState.from_dict({"cpu_busy": "0", "gpu_busy": "0", "dram_used_bytes": 0})
        with self.assertRaises(ValueError):
            ResourceState.from_dict(dict(state().to_dict(), condition="baseline"))

    def test_unknown_memory_is_distinct_from_explicit_zero(self):
        unknown = operation(dram_bytes=None, vram_bytes=None, memory_footprint_complete=False)
        zero = operation(dram_bytes=0, vram_bytes=0, memory_footprint_complete=True)
        self.assertNotEqual(unknown.signature, zero.signature)
        with self.assertRaises(ValueError):
            operation(dram_bytes=None)
        with self.assertRaises(ValueError):
            operation(dram_bytes=True)

    def test_constants_are_normalized_and_immutable(self):
        constants = [["shape", "8x8"], ["dtype", "f32"]]
        op = operation(constants=constants)
        constants[0][1] = "changed"
        self.assertEqual(op.constants, (("dtype", "f32"), ("shape", "8x8")))
        self.assertEqual(Operation.from_dict(op.to_dict()), op)
        for bad in ((('a', 'b'), ('a', 'c')), ("ab",), (("a", 4),), (("a", ""),)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                operation(constants=bad)

    def test_hardware_and_feature_configuration_guards(self):
        for values in (("", 1, 1), ("h", 0, 1), ("h", True, 1), ("h", 1, -1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Hardware(*values)
        for options in ({"regularization": -1}, {"regularization": .001},
                        {"min_training_runs": 0}, {"include_cpu_gpu_interaction": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                FitConfig(**options)

    def test_trace_requires_complete_contiguous_intervals(self):
        for segments in ((), (TraceSegment(0, 10, state()), TraceSegment(11, 20, state())),
                         (TraceSegment(0, 10, state()), TraceSegment(9, 20, state())),
                         (TraceSegment(10, 20, state()), TraceSegment(0, 10, state()))):
            with self.subTest(segments=segments), self.assertRaises(ValueError):
                ResourceTrace(segments, "external")
        for begin, end in ((0, 0), (2, 1), (-1, 2), (True, 3)):
            with self.subTest(begin=begin, end=end), self.assertRaises(ValueError):
                TraceSegment(begin, end, state())
        with self.assertRaises(ValueError):
            ResourceTrace((TraceSegment(0, 1, state()),), "total_observed")

    def test_trace_round_trip_defensively_copies_segments(self):
        segments = [TraceSegment(0, 10, state()), TraceSegment(10, 20, state(1))]
        trace = ResourceTrace(segments, "external")
        segments.clear()
        self.assertEqual(len(trace.segments), 2)
        self.assertEqual(ResourceTrace.from_dict(trace.to_dict()), trace)

    def test_row_full_window_and_metadata_are_validated(self):
        item = heldout()
        self.assertEqual(CalibrationRow.from_dict(item.to_dict()), item)
        for options in ({"elapsed_ns": 0}, {"numerical_passed": 1}, {"source_sha256": "bad"},
                        {"clock_id": ""}, {"state_semantics": "total_observed"},
                        {"measured_until_ns": FROZEN + 149}, {"split": "unspecified"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                replace(item, **options)


class FitTests(unittest.TestCase):
    def test_exact_linear_surface_recovers_known_function(self):
        coefficients = (100, 100, 200, 40, 60, 80)
        fitted = model_for(coefficients=coefficients)
        self.assertEqual(fitted.surfaces[0].coefficients_ns, coefficients)
        middle = state(Fraction(1, 4), Fraction(3, 4), 250, 125)
        expected = 100 + 25 + 150 + 10 + Fraction(15, 2) + 15
        self.assertEqual(predicted_duration_ns(fitted, operation(), middle), expected)
        self.assertTrue(all(value == 0 for value in fitted.surfaces[0].training_relative_errors))
        self.assertFalse(fitted.to_dict()["heldout_used_for_fit"])
        self.assertFalse(fitted.to_dict()["hardware_accuracy_accepted"])

    def test_ridge_handles_constant_inputs_without_hidden_fallback(self):
        entries = [replace(training_rows()[i], state=state(), elapsed_ns=100) for i in range(3)]
        fitted = fit_model(entries, HW, FROZEN)
        self.assertEqual(predicted_duration_ns(fitted, operation(), state()), 100)
        self.assertEqual(fitted.surfaces[0].coefficients_ns[1:], (0, 0, 0, 0, 0))
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            fit_model(entries, HW, FROZEN, FitConfig(regularization=0))
        with self.assertRaises(UnknownSupport):
            predicted_duration_ns(fitted, operation(), state(Fraction(1, 10)))

    def test_default_ridge_is_explicit_and_frozen(self):
        fitted = fit_model(training_rows(), HW, FROZEN)
        self.assertEqual(fitted.config.regularization, Fraction(1, 1000))
        self.assertNotEqual(fitted.surfaces[0].coefficients_ns, model_for().surfaces[0].coefficients_ns)

    def test_half_nanosecond_medians_are_not_rounded_during_fit(self):
        rows = [replace(item, elapsed_ns=Fraction(201, 2)) for item in training_rows()]
        fitted = fit_model(rows, HW, FROZEN)
        self.assertEqual(predicted_duration_ns(fitted, operation(), state()), Fraction(201, 2))
        self.assertEqual(rows[0].to_dict()["elapsed_ns"], "201/2")
        result = predict(fitted, operation(), ResourceTrace((TraceSegment(0, 1000, state()),), "external"))
        self.assertEqual(result.exact_elapsed_ns, Fraction(201, 2))
        self.assertEqual(result.elapsed_ns, 101)

    def test_training_input_order_does_not_change_frozen_model(self):
        rows = training_rows()
        forward = fit_model(rows, HW, FROZEN)
        backward = fit_model(reversed(rows), HW, FROZEN)
        self.assertEqual(forward.model_sha256, backward.model_sha256)

    def test_no_condition_label_enters_row_or_model(self):
        item = training_rows()[0].to_dict()
        item["condition"] = "baseline"
        with self.assertRaises(ValueError):
            CalibrationRow.from_dict(item)
        self.assertNotIn("condition", json.dumps(model_for().to_dict()))

    def test_fit_rejects_heldout_duplicate_bad_numeric_and_mismatched_bindings(self):
        original = training_rows()
        changes = ({"split": "heldout"}, {"numerical_passed": False}, {"source_sha256": "b" * 64},
                   {"protocol_id": "other"}, {"clock_id": "other"}, {"state_semantics": "offered_load_controlled"})
        for options in changes:
            entries = list(original)
            entries[0] = replace(entries[0], **options)
            with self.subTest(options=options), self.assertRaises(ValueError):
                fit_model(entries, HW, FROZEN)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            fit_model(original + [original[0]], HW, FROZEN)

    def test_freeze_uses_entire_measurement_window_not_median(self):
        entries = training_rows()
        entries[0] = replace(entries[0], measured_until_ns=FROZEN + 1)
        self.assertLess(entries[0].started_ns + entries[0].elapsed_ns, FROZEN)
        with self.assertRaisesRegex(ValueError, "freeze"):
            fit_model(entries, HW, FROZEN)

    def test_independent_runs_required_per_operation(self):
        entries = [replace(row, run_id="one-run") for row in training_rows()]
        with self.assertRaisesRegex(ValueError, "independent"):
            fit_model(entries, HW, FROZEN)

    def test_exact_constants_not_shape_extrapolation(self):
        fitted = model_for()
        changed = replace(operation(), constants=operation().constants + (("history", "128"),))
        for op in (changed, replace(operation(), resource="gpu"), replace(operation(), reference_ns=200)):
            with self.subTest(op=op), self.assertRaises(UnknownSupport):
                predicted_duration_ns(fitted, op, state())

    def test_feature_range_guard_includes_memory_and_interaction(self):
        fitted = model_for()
        with self.assertRaisesRegex(UnknownSupport, "dram_fraction"):
            predicted_duration_ns(fitted, operation(), state(dram=501))
        rows = [row for row in training_rows() if not (row.state.cpu_busy and row.state.gpu_busy)]
        fitted = fit_model(rows, HW, FROZEN)
        with self.assertRaisesRegex(UnknownSupport, "interaction"):
            predicted_duration_ns(fitted, operation(), state(1, 1))

    def test_model_round_trip_and_tampering_guards(self):
        fitted = model_for()
        encoded = json.loads(json.dumps(fitted.to_dict()))
        restored = ResourceFunctional.from_dict(encoded)
        self.assertEqual(restored, fitted)
        self.assertEqual(restored.model_sha256, fitted.model_sha256)
        for key, value in (("heldout_used_for_fit", True), ("hardware_accuracy_accepted", True),
                           ("compute_semantics", "unknown"), ("feature_names", []), ("schema_version", True)):
            changed = dict(encoded, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                ResourceFunctional.from_dict(changed)
        encoded["surfaces"][0]["coefficients_ns"].clear()
        self.assertEqual(len(fitted.surfaces[0].coefficients_ns), 6)

    def test_cpu_only_hardware_does_not_require_gpu_capacity(self):
        hardware = Hardware("cpu-only", 1000, 0)
        op = operation(vram_bytes=0)
        entries = [replace(row, state=state(row.state.cpu_busy, 0, row.state.dram_used_bytes, 0))
                   for row in training_rows(op)]
        fitted = fit_model(entries, hardware, FROZEN)
        self.assertGreater(predicted_duration_ns(fitted, op, state()), 0)
        with self.assertRaises(Infeasible):
            predicted_duration_ns(fitted, op, state(vram=1))


class FunctionalTests(unittest.TestCase):
    def test_mid_operation_slowdown_integrates_reference_work(self):
        fitted = model_for()
        trace = ResourceTrace((TraceSegment(0, 50, state()), TraceSegment(50, 1000, state(1))), "external")
        result = predict(fitted, operation(), trace, require_physical_capacity=True)
        self.assertEqual(result.exact_finish_ns, 150)
        self.assertEqual([item.reference_service_ns for item in result.slices], [50, 50])
        self.assertEqual([item.service_rate for item in result.slices], [1, Fraction(1, 2)])
        self.assertTrue(result.physical_capacity_verified)

    def test_slow_then_fast_and_future_state_after_completion_is_unused(self):
        trace = ResourceTrace((TraceSegment(0, 50, state(1)), TraceSegment(50, 200, state()),
                               TraceSegment(200, 1000, state(dram=1000))), "external")
        result = predict(model_for(), operation(), trace)
        self.assertEqual(result.finish_ns, 125)
        self.assertEqual(len(result.slices), 2)

    def test_only_final_completion_rounds_up(self):
        # Duration at cpu=1 is 103 ns; service50 consumes 50/103 reference units.
        fitted = model_for(coefficients=(100, 3, 0, 0, 0, 0))
        trace = ResourceTrace((TraceSegment(0, 50, state(1)), TraceSegment(50, 1000, state())), "external")
        result = predict(fitted, operation(), trace)
        self.assertEqual(result.exact_finish_ns, Fraction(10450, 103))
        self.assertEqual(result.finish_ns, 102)
        self.assertEqual(sum(item.reference_service_ns for item in result.slices), 100)

    def test_all_resources_share_same_functional_path(self):
        ops = (operation("cpu"), operation("gpu"), operation("copy"))
        entries = []
        for op, base in zip(ops, (100, 200, 300)):
            entries.extend(training_rows(op, coefficients=(base, 0, 0, 0, 0, 0)))
        fitted = fit_model(entries, HW, FROZEN, FitConfig(regularization=0))
        trace = ResourceTrace((TraceSegment(0, 2000, state()),), "external")
        results = predict_sequence(fitted, ops, trace)
        self.assertEqual([item.elapsed_ns for item in results], [100, 200, 300])
        self.assertEqual(results[-1].finish_ns, 600)
        self.assertEqual(len(fitted.surfaces), 3)

    def test_capacity_checked_at_mid_operation_state_change(self):
        op = operation(dram_bytes=600)
        # Train only known-feasible states. Capacity failure must precede the
        # support guard when an external allocation appears during execution.
        rows = [row for row in training_rows(op) if row.state.dram_used_bytes == 0]
        fitted = fit_model(rows, HW, FROZEN)
        trace = ResourceTrace((TraceSegment(0, 50, state()), TraceSegment(50, 1000, state(dram=500))), "external")
        with self.assertRaisesRegex(Infeasible, "DRAM"):
            predict(fitted, op, trace)

    def test_unknown_workspace_cannot_prove_capacity(self):
        op = operation(dram_bytes=None, vram_bytes=None, memory_footprint_complete=False)
        fitted = model_for(op)
        trace = ResourceTrace((TraceSegment(0, 1000, state()),), "external")
        self.assertFalse(predict(fitted, op, trace).physical_capacity_verified)
        with self.assertRaises(UnknownCapacity):
            predict(fitted, op, trace, require_physical_capacity=True)

    def test_controlled_load_uses_same_path_but_no_physical_capacity_claim(self):
        fitted = model_for(semantics="offered_load_controlled")
        trace = ResourceTrace((TraceSegment(0, 50, state()), TraceSegment(50, 1000, state(1))), "offered_load_controlled")
        result = predict(fitted, operation(), trace)
        self.assertEqual(result.finish_ns, 150)
        self.assertFalse(result.physical_capacity_verified)
        self.assertEqual(fitted.compute_semantics, "offered_load")
        self.assertEqual(fitted.memory_semantics, "owned_external_allocations")
        with self.assertRaises(UnknownCapacity):
            predict(fitted, operation(), trace, require_physical_capacity=True)
        with self.assertRaisesRegex(UnknownSupport, "semantics"):
            predict(fitted, operation(), replace(trace, state_semantics="external"))

    def test_no_one_minus_busy_formula(self):
        # An empirically constant response is allowed even at offeredbusy=1.
        fitted = model_for(coefficients=(100, 0, 0, 0, 0, 0))
        trace = ResourceTrace((TraceSegment(0, 1000, state(1, 1)),), "external")
        self.assertEqual(predict(fitted, operation(), trace).finish_ns, 100)

    def test_memory_response_uses_calibrated_coefficients_not_capacity_rule(self):
        trace = ResourceTrace((TraceSegment(0, 50, state()), TraceSegment(50, 1000, state(dram=500))), "external")
        memory_sensitive = model_for(coefficients=(100, 0, 0, 1000, 0, 0))
        result = predict(memory_sensitive, operation(), trace)
        self.assertEqual(result.finish_ns, 350)
        self.assertTrue(result.physical_capacity_verified)
        capacity_only = model_for(coefficients=(100, 0, 0, 0, 0, 0))
        self.assertEqual(predict(capacity_only, operation(), trace).finish_ns, 100)

    def test_capacity_equality_allowed_but_one_byte_over_rejected(self):
        op = operation(dram_bytes=500)
        fitted = model_for(op)
        exact = ResourceTrace((TraceSegment(0, 1000, state(dram=500)),), "external")
        self.assertTrue(predict(fitted, op, exact, require_physical_capacity=True).physical_capacity_verified)
        exceeded = ResourceTrace((TraceSegment(0, 1000, state(dram=501)),), "external")
        with self.assertRaises(Infeasible):
            predict(fitted, op, exceeded)

    def test_nonpositive_prediction_rejected_without_clipping(self):
        # A valid affine fit on observed states can be negative at an unseen
        # joint combination inside its marginal feature box.
        # Construct manually because the absent joint corner has negative time.
        rows = []
        for point in ((0, 0), (1, 0), (0, 1)):
            for repeat in range(3):
                rows.append(replace(training_rows()[repeat], row_id=f"{point}-{repeat}",
                                    state=state(*point), elapsed_ns=100 - 70 * sum(point)))
        fitted = fit_model(rows, HW, FROZEN, FitConfig(include_cpu_gpu_interaction=False))
        with self.assertRaisesRegex(UnknownSupport, "nonpositive"):
            predicted_duration_ns(fitted, operation(), state(1, 1))

    def test_start_and_end_support_are_explicit(self):
        fitted = model_for()
        trace = ResourceTrace((TraceSegment(50, 100, state()),), "external")
        for start in (0, 50, 100, 101):
            with self.subTest(start=start), self.assertRaises(UnknownSupport):
                predict(fitted, operation(), trace, start_ns=start)
        long = ResourceTrace((TraceSegment(50, 1000, state()),), "external")
        self.assertEqual(predict(fitted, operation(), long, start_ns=70).finish_ns, 170)


class HeldoutTests(unittest.TestCase):
    def test_heldout_evaluation_is_read_only_with_exact_errors(self):
        fitted = model_for()
        before = fitted.model_sha256
        report = evaluate_heldout(fitted, [heldout(elapsed=100)])
        self.assertEqual(report["median_relative_error"], "1/2")
        self.assertEqual(report["predicted_count"], 1)
        self.assertEqual(fitted.model_sha256, before)
        self.assertFalse(report["heldout_used_for_fit"])

    def test_heldout_leakage_and_binding_guards(self):
        fitted = model_for()
        changes = ({"split": "train"}, {"run_id": "run-0"}, {"seed": 1},
                   {"row_id": fitted.training_row_ids[0]}, {"started_ns": FROZEN - 1},
                   {"clock_id": "other"}, {"protocol_id": "other"}, {"source_sha256": "b" * 64},
                   {"state_semantics": "offered_load_controlled"}, {"numerical_passed": False})
        for options in changes:
            with self.subTest(options=options), self.assertRaises(ValueError):
                evaluate_heldout(fitted, [heldout(**options)])
        with self.assertRaises(ValueError):
            evaluate_heldout(fitted, [heldout(), heldout()])

    def test_unknown_heldout_rows_are_retained_not_excluded(self):
        fitted = model_for()
        rows = [heldout(), heldout(resource_state=state(dram=501), row_id="unsupported")]
        report = evaluate_heldout(fitted, rows)
        self.assertEqual(report["row_count"], 2)
        self.assertEqual(report["predicted_count"], 1)
        self.assertEqual(report["unknown_or_infeasible_count"], 1)
        self.assertEqual(report["rows"][1]["status"], "unknown")

    def test_heldout_shape_without_calibration_stays_unknown(self):
        op = operation(constants=(("shape", "larger"),))
        report = evaluate_heldout(model_for(), [heldout(op)])
        self.assertEqual(report["predicted_count"], 0)
        self.assertEqual(report["rows"][0]["status"], "unknown")
        self.assertIsNone(report["median_relative_error"])


if __name__ == "__main__":
    unittest.main()
