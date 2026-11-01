"""Hand-counted interval oracles, independent of model cost calculations."""
import unittest

from experiments.strategy_comparison_metrics import timeline_metrics


def compute(device, start, end):
    return dict(kind="COMPUTE", device=device, start_ns=start, end_ns=end)


def copy(start, end, kind="COPY"):
    return dict(kind=kind, start_ns=start, end_ns=end)


class TimelineMetricsTests(unittest.TestCase):
    def test_nested_touching_disjoint_spans_have_handwritten_coverage(self):
        # CPU [0,12), [20,22); GPU [5,15), [24,26).
        # COPY [8,18), [21,25). Any engine [0,18), [20,26).
        events = [compute("cpu", 0, 10), compute("cpu", 2, 5),
                  compute("cpu", 10, 12), compute("cpu", 20, 22),
                  compute("gpu", 5, 15), compute("gpu", 24, 26),
                  copy(8, 18), copy(9, 10), copy(21, 25)]
        expected = dict(cpu_event_span_ns=14, gpu_event_span_ns=12,
            copy_event_span_ns=14, compute_event_span_ns=19,
            cpu_gpu_event_overlap_ns=7, copy_compute_event_overlap_ns=9,
            any_engine_event_span_ns=24, all_engine_event_uncovered_ns=6)
        self.assertEqual(timeline_metrics(events, 30), expected)
        self.assertEqual(timeline_metrics(list(reversed(events)), 30), expected)
        self.assertEqual(events[0], compute("cpu", 0, 10))

    def test_cross_engine_overlap_is_not_summed_as_elapsed_time(self):
        metrics = timeline_metrics([compute("cpu", 0, 10), compute("gpu", 0, 10), copy(0, 10)], 10)
        self.assertEqual(metrics["compute_event_span_ns"], 10)
        self.assertEqual(metrics["cpu_gpu_event_overlap_ns"], 10)
        self.assertEqual(metrics["copy_compute_event_overlap_ns"], 10)
        self.assertEqual(metrics["any_engine_event_span_ns"], 10)
        self.assertEqual(metrics["all_engine_event_uncovered_ns"], 0)

    def test_touching_boundaries_have_no_positive_overlap(self):
        metrics = timeline_metrics([compute("cpu", 0, 5), compute("gpu", 5, 10), copy(10, 15)], 20)
        self.assertEqual(metrics["cpu_gpu_event_overlap_ns"], 0)
        self.assertEqual(metrics["copy_compute_event_overlap_ns"], 0)
        self.assertEqual(metrics["any_engine_event_span_ns"], 15)
        self.assertEqual(metrics["all_engine_event_uncovered_ns"], 5)

    def test_zero_length_and_empty_work_do_not_occupy_time(self):
        self.assertEqual(timeline_metrics([], 0), {name: 0 for name in (
            "cpu_event_span_ns", "gpu_event_span_ns", "copy_event_span_ns",
            "compute_event_span_ns", "cpu_gpu_event_overlap_ns",
            "copy_compute_event_overlap_ns", "any_engine_event_span_ns",
            "all_engine_event_uncovered_ns")})
        metrics = timeline_metrics([compute("cpu", 0, 0), copy(7, 7)], 7)
        self.assertEqual(metrics["any_engine_event_span_ns"], 0)
        self.assertEqual(metrics["all_engine_event_uncovered_ns"], 7)

    def test_real_administrative_shapes_and_untimed_records_are_ignored(self):
        metrics = timeline_metrics([
            dict(kind="ADVANCE", t_ns=10, start_ns=None, end_ns=10),
            dict(kind="EVICT", t_ns=10, start_ns=10, end_ns=10),
            dict(kind="ADVANCE"), dict(kind="EVICT"), compute("gpu", 1, 3)], 10)
        self.assertEqual(metrics["gpu_event_span_ns"], 2)
        self.assertEqual(metrics["all_engine_event_uncovered_ns"], 8)

    def test_engine_dataclass_contract_and_legacy_resource(self):
        from llm_infer_model.tensor.engine import Event
        metrics = timeline_metrics([
            Event(action_index=0, kind="COMPUTE", target_id="a", resource="cpu_compute",
                  t_ns=0, start_ns=0, end_ns=4, device="cpu"),
            Event(action_index=1, kind="COMPUTE", target_id="b", resource="gpu_compute",
                  t_ns=2, start_ns=2, end_ns=6),
            Event(action_index=2, kind="ADVANCE", target_id=None, resource=None,
                  t_ns=6, end_ns=6)], 6)
        self.assertEqual(metrics["cpu_event_span_ns"], 4)
        self.assertEqual(metrics["gpu_event_span_ns"], 4)
        self.assertEqual(metrics["cpu_gpu_event_overlap_ns"], 2)

    def test_all_transfer_kinds_contribute_to_copy_union(self):
        metrics = timeline_metrics([copy(0, 3, "COPY_H2D"), copy(2, 5, "COPY"),
                                   copy(4, 7, "READ_SSD"), copy(9, 10, "WRITE_SSD")], 12)
        self.assertEqual(metrics["copy_event_span_ns"], 8)
        self.assertEqual(metrics["all_engine_event_uncovered_ns"], 4)

    def test_invalid_makespan_fails_instead_of_coercing(self):
        for value in (None, -1, True, 2.0, "2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                timeline_metrics([], value)

    def test_missing_or_incomplete_work_intervals_fail(self):
        for kind in ("COMPUTE", "COPY", "COPY_H2D", "READ_SSD", "WRITE_SSD"):
            for times in ({}, dict(start_ns=1), dict(end_ns=2), dict(start_ns=1, end_ns=None)):
                with self.subTest(kind=kind, times=times), self.assertRaises(ValueError):
                    timeline_metrics([dict(kind=kind, device="cpu", **times)], 10)

    def test_negative_noninteger_reversed_and_out_of_window_times_fail(self):
        for start, end in ((-1, 2), (1, -1), (3, 2), (0, 11), (11, 11),
                           (True, 3), (0, False), (0.0, 3), (0, "3")):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                timeline_metrics([copy(start, end)], 10)
        with self.assertRaises(ValueError):
            timeline_metrics([dict(kind="ADVANCE", t_ns=-1)], 10)

    def test_invalid_administrative_spans_are_not_silently_dropped(self):
        for event in (dict(kind="EVICT", start_ns=1, end_ns=2),
                      dict(kind="EVICT", end_ns=2),
                      dict(kind="ADVANCE", start_ns=1, end_ns=2),
                      dict(kind="ADVANCE", start_ns=1),
                      dict(kind="ADVANCE", end_ns=11)):
            with self.subTest(event=event), self.assertRaises(ValueError):
                timeline_metrics([event], 10)

    def test_unknown_event_and_bad_compute_identity_fail(self):
        for event in (dict(kind="WAIT", start_ns=0, end_ns=2), {}, "COPY", dict(kind=[]),
                      dict(kind="COMPUTE", start_ns=0, end_ns=2),
                      compute("gpu0", 0, 2),
                      dict(kind="COMPUTE", device="cpu", resource="gpu_compute", start_ns=0, end_ns=2)):
            with self.subTest(event=event), self.assertRaises(ValueError):
                timeline_metrics([event], 10)


if __name__ == "__main__":
    unittest.main()
