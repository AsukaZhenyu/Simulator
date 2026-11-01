"""Small mathematical checks for metric definitions and shared-reference scope."""
from copy import deepcopy
import unittest
from . import lowering_metrics as metric


class LoweringMetricTests(unittest.TestCase):
    def test_joint_gap_is_distinct_from_choice_and_selected_lowering(self):
        rows=[dict(id="a",compiler_duration_ns=100,reference_family_best_ns=120,candidate_signed_gap_percent=-100/6),
              dict(id="b",compiler_duration_ns=110,reference_family_best_ns=90,candidate_signed_gap_percent=100*2/9)]
        result=metric.metrics(rows,"a")
        self.assertAlmostEqual(result["lower_signed_relative_gap_percent"],100/9)
        self.assertAlmostEqual(result["choice_regret_percent"],100/3)
        self.assertAlmostEqual(result["selected_candidate_lowering_signed_gap_percent"],-100/6)
        self.assertEqual(result["candidate_gap_distribution"]["negative_count"],1)
        with self.assertRaisesRegex(ValueError,"not a compiler"):
            metric.metrics(rows,"b")

    def test_all_candidate_extrema_are_retained(self):
        result=metric.stats([-20,0,1,75])
        self.assertEqual(result["minimum_percent"],-20)
        self.assertEqual(result["maximum_percent"],75)
        self.assertEqual(result["negative_count"],1)
        self.assertEqual(result["positive_count"],2)

    def test_reference_accepts_any_ready_consumer_but_not_overlapping_engine(self):
        nodes={
            "a":dict(machine="cpu",duration=100,release=0,deps=[]),
            "b":dict(machine="gpu",duration=20,release=0,deps=[]),
            "copy":dict(machine="pcie",duration=10,release=0,deps=[],dependency_alternatives=[["a"],["b"]])}
        events=[dict(id="a",machine="cpu",start_ns=0,end_ns=100),
                dict(id="b",machine="gpu",start_ns=0,end_ns=20),
                dict(id="copy",machine="pcie",start_ns=20,end_ns=30)]
        self.assertEqual(metric.verify_schedule(nodes,events),100)
        bad=deepcopy(events);bad[-1].update(start_ns=10,end_ns=20)
        with self.assertRaisesRegex(AssertionError,"OR"):
            metric.verify_schedule(nodes,bad)
        overlap=deepcopy(nodes);overlap["b"]["machine"]="cpu"
        bad=deepcopy(events);bad[1]["machine"]="cpu"
        with self.assertRaisesRegex(AssertionError,"resource"):
            metric.verify_schedule(overlap,bad)

    def test_table_ii_baselines_reoptimize_the_other_dimension(self):
        rows=[]
        for p,choices in (("cpu",{"x":"cpu","y":"cpu"}),
                          ("gpu",{"x":"gpu","y":"gpu"}),
                          ("hybrid",{"x":"cpu","y":"gpu"})):
            for storage in ("demand","spare"):
                cost={"cpu":200,"gpu":180,"hybrid":150}[p]-(40 if storage=="spare" else 0)
                rows.append(dict(id=p+storage,choices=choices,storage=storage,compiler_duration_ns=cost))
        result=metric.baseline_sets(rows)
        self.assertEqual(result["execution_only_candidates"],3)
        self.assertEqual(result["storage_only_candidates"],4)
        self.assertEqual(result["isolated_union_candidates"],5)
        self.assertEqual(result["execution_only_best_ns"],150)
        self.assertEqual(result["storage_only_best_ns"],140)


if __name__=="__main__":unittest.main()
