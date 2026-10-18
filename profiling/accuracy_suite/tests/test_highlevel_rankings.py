"""Synthetic common-boundary comparisons; no hardware/native process runs."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT/"profiling") not in sys.path:sys.path.insert(0,str(ROOT/"profiling"))
from accuracy_suite import highlevel_rankings as ranking
import test_verify_highlevel as fixtures


class HighLevelRankingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.OfflineHighLevelVerifierTests.setUpClass()
        cls.fixture=fixtures.OfflineHighLevelVerifierTests
        cls.root=cls.fixture.experiment
        cls.campaign=ranking.read(cls.root/"campaign.json")
        cls.report=ranking.read(cls.root/"report.json")

    @classmethod
    def tearDownClass(cls):cls.fixture.tearDownClass()

    @contextmanager
    def changed(self,path,mutator):
        original=path.read_bytes()
        try:
            document=json.loads(original);mutator(document);path.write_text(json.dumps(document));yield
        finally:path.write_bytes(original)

    def rank(self):return ranking.rank_highlevel(self.root,self.campaign,self.report,bootstrap_repetitions=200)

    def test_parent_and_partition_contents_compare_at_one_boundary(self):
        result=self.rank()
        self.assertEqual(result["declared_paths"],3)
        self.assertEqual(result["known_paths"],3,result["paths"])
        self.assertEqual(result["comparable_groups"],1,result["rankings"])
        row=result["rankings"][0]
        self.assertTrue(row["known"])
        self.assertEqual(len(row["all_declared_pairs"]),3)
        self.assertEqual(len({p["boundary_sha256"] for p in result["paths"]}),1)
        self.assertEqual(len({p["profile_data_sha256"] for p in result["paths"]}),1)
        self.assertFalse(result["physical_capacity_claimed"])
        self.assertFalse(result["protocol_verifier_execution_claimed"])

    def test_fixed_process_median_bootstrap_is_reproducible(self):
        self.assertEqual(ranking.interval([100,101,102],1907,200),ranking.interval([100,101,102],1907,200))
        result=self.rank();self.assertEqual(result["bootstrap_seed"],1907)
        self.assertIn("independent_process_medians",result["interval_unit"])

    def test_regret_uses_complete_path_observed_latency(self):
        result=self.rank();row=result["rankings"][0]
        paths={p["strategy_id"]:p for p in result["paths"]}
        expected=paths[row["predicted_best"]]["observed_ns"]-paths[row["observed_best"]]["observed_ns"]
        self.assertEqual(row["selection_regret_ns"],expected)

    def test_missing_raw_process_prevents_survivor_subset_selection(self):
        case=self.root/self.campaign["cases"][0]["id"]
        file=case/"heldout-process-1/execution.json";original=file.read_bytes();file.unlink()
        try:
            result=self.rank()
            self.assertEqual(result["known_paths"],2)
            self.assertFalse(result["rankings"][0]["known"])
            self.assertNotIn("predicted_best",result["rankings"][0])
        finally:file.write_bytes(original)

    def test_changed_numerical_weights_are_incomparable_even_with_same_seed(self):
        case=self.root/self.campaign["cases"][0]["id"]
        program=ranking.read(case/"program.json")
        role=next(iter(program["initial_values"]))
        original=ranking.external_boundary(program)
        # Keep the seed fixed but alter one real F32 value and reseal identities.
        program["initial_values"][role][0]+=0.01
        program["identity"]["initial_values"]=ranking.digest(program["initial_values"])
        program.pop("program_sha256");program["program_sha256"]=ranking.digest(program)
        changed=ranking.external_boundary(program)
        self.assertEqual(original["seed"],changed["seed"])
        self.assertEqual(original["static_sha256"],changed["static_sha256"])
        self.assertNotEqual(original["data_sha256"],changed["data_sha256"])

    def test_different_output_location_changes_boundary(self):
        case=self.root/self.campaign["cases"][0]["id"]
        program=ranking.read(case/"program.json");original=ranking.external_boundary(program)
        goals=program["graph"]["scenario"]["goal"]["output_locations"]
        hidden=next(b["outputs"]["result"] for b in program["bindings"] if b["kind"]=="dense_ffn")
        goals[hidden]="vram";program["identity"]["graph"]=ranking.digest(program["graph"])
        program.pop("program_sha256");program["program_sha256"]=ranking.digest(program)
        self.assertNotEqual(original["static_sha256"],ranking.external_boundary(program)["static_sha256"])

    def test_ambiguous_intervals_remain_explicit(self):
        # Marginal process-median intervals overlap, even though point medians
        # suggest an order. The interval function preserves that ambiguity.
        a=ranking.interval([100,150,200],1907,200)
        b=ranking.interval([110,160,210],1907,200)
        self.assertFalse(a[1]<b[0] or b[1]<a[0])

    def test_unstable_overlapping_paths_keep_descriptive_order_without_acceptance(self):
        common=dict(known=True,latency_quality_accepted=False,source_process_indices=[0,1,2],profile_seed=1,
            profile_data_sha256="same",heldout_cohorts=[dict(process_index=i,seed=i+10,data_sha256=str(i),samples=3) for i in range(3)],
            boundary_sha256="same-boundary",external_boundary={})
        paths={"A":dict(common,case_id="A",strategy_id="A",predicted_ns=100,observed_ns=1000,heldout_process_medians_ns=[500,1000,1500]),
               "B":dict(common,case_id="B",strategy_id="B",predicted_ns=200,observed_ns=1100,heldout_process_medians_ns=[900,1100,1300])}
        campaign=dict(cases=[dict(id=name,kwargs={}) for name in ("A","B")])
        with patch.object(ranking,"path_evidence",side_effect=lambda root,item,report:deepcopy(paths[item["id"]])):
            result=ranking.rank_highlevel(self.root,campaign,{},bootstrap_repetitions=200)
        row=result["rankings"][0]
        self.assertTrue(row["known"])
        self.assertEqual(row["predicted_order"],["A","B"])
        self.assertTrue(row["ambiguous_overlapping_intervals"])
        self.assertFalse(row["selection_evidence_accepted"])
        self.assertEqual(row["inference"],"inconclusive_overlapping_process_intervals")


if __name__=="__main__":unittest.main()
