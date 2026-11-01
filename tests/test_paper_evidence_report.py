"""Fail-closed report input checks; no simulator implementation is mirrored."""
import json
import tempfile
import unittest
from pathlib import Path

from experiments.paper_evidence_report import canonical_sha, dump, read_design, paired_prefetch


class EvidenceReportInputs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.rows = [{"status": "valid", "success": True,
                      "calibrated": False, "hardware_measurement": False,
                      "workload":"dense","decode_steps":1,"initial_state":"cold",
                      "vram_gib":12,"link_gbs":8,"candidate":candidate,
                      "shared_input_sha256":"fixture-same-input", "h2d_bytes":cost*2,
                      "completed_makespan_ns":cost}
                     for candidate,cost in [('e0-s0',100),('e0-s1',120),('e1-s0',200),('e1-s1',220)]]
        dump(self.path / "protocol.json", {"cases": 4})
        dump(self.path / "records.json", self.rows)
        self.analysis={"groups":[{"group":"dense-d1-cold-v12-b8","feasible":True,
            "workload":"dense","decode_steps":1,"initial_state":"cold","vram_gib":12,"link_gbs":8,
            "choices":{name:{"candidate":"e0-s0","latency_ns":100,"regret_vs_finite_best":0.,"evaluations":count}
                       for name,count in [('joint',4),('execution_only_s0',2),('storage_only_e0',2),('stagewise_execution_then_storage',3)]},
            "tied_best":["e0-s0"],"center_frozen_candidate":"e0-s0","center_frozen_regret":0.,"center_frozen_status":"success"}]}
        dump(self.path / "analysis.json", self.analysis)
        self.receipt = {"complete": True, "sources_unchanged": True,
                        "accuracy_evidence_unchanged": True,
                        "audit": {"pass_": True}, "cases": 4, "success": 4,
                        "statuses": {"valid": 4},
                        "records_sha256": canonical_sha(self.rows),
                        "protocol_sha256": canonical_sha({"cases":4})}
        self.write_receipt()

    def write_receipt(self):
        dump(self.path / "completion.json", self.receipt)

    def test_completed_evidence_accepted(self):
        self.assertEqual(len(read_design(self.path, 4)[1]), 4)

    def test_partial_campaign_rejected(self):
        self.receipt["complete"] = False
        self.write_receipt()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            read_design(self.path, 4)

    def test_changed_source_or_accuracy_cannot_be_reported(self):
        for key in ["sources_unchanged", "accuracy_evidence_unchanged"]:
            with self.subTest(key=key):
                self.receipt[key] = False
                self.write_receipt()
                with self.assertRaises(ValueError):
                    read_design(self.path, 4)
                self.receipt[key] = True

    def test_tampered_records_rejected(self):
        dump(self.path / "records.json", self.rows + self.rows)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            read_design(self.path, 4)

    def test_failed_independent_audit_rejected(self):
        self.receipt["audit"]["pass_"] = False
        self.write_receipt()
        with self.assertRaisesRegex(ValueError, "audit"):
            read_design(self.path, 4)

    def test_denominator_cannot_shrink(self):
        with self.assertRaisesRegex(ValueError, "denominator"):
            read_design(self.path, 5)

    def test_infeasible_cases_cannot_be_hidden(self):
        self.receipt["statuses"] = {"valid": 3, "initial_capacity_infeasible": 1}
        self.write_receipt()
        with self.assertRaisesRegex(ValueError, "statuses"):
            read_design(self.path, 4)

    def test_analytical_campaign_cannot_be_relabelled_measured(self):
        self.rows[0]["hardware_measurement"] = True
        dump(self.path / "records.json", self.rows)
        self.receipt["records_sha256"] = canonical_sha(self.rows)
        self.write_receipt()
        with self.assertRaisesRegex(ValueError, "measured"):
            read_design(self.path, 4)

    def test_unsealed_analysis_cannot_change_winner_or_cost(self):
        for field,value in [('candidate','e1-s0'),('latency_ns',1),('regret_vs_finite_best',.5),('evaluations',1)]:
            with self.subTest(field=field):
                choice=self.analysis['groups'][0]['choices']['joint']
                previous=choice[field]
                choice[field]=value
                dump(self.path/'analysis.json',self.analysis)
                with self.assertRaises(ValueError):
                    read_design(self.path,4)
                choice[field]=previous

    def test_unsealed_analysis_cannot_change_center_regret(self):
        self.analysis['groups'][0]['center_frozen_regret']=.5
        dump(self.path/'analysis.json',self.analysis)
        with self.assertRaisesRegex(ValueError,'center regret'):
            read_design(self.path,4)

    def test_prefetch_pairs_keep_denominator_and_actual_relative_delta(self):
        result=paired_prefetch(self.rows)
        self.assertEqual(len(result),2)
        self.assertTrue(all(r['status']=='slower' for r in result))
        self.assertAlmostEqual(result[0]['relative_delta'],.2)
        self.assertEqual(result[0]['h2d_delta_bytes'],40)
        for row in self.rows:
            row.update(success=False,status='initial_capacity_infeasible')
        result=paired_prefetch(self.rows)
        self.assertEqual(len(result),2)
        self.assertTrue(all(r['status']=='initial_capacity_infeasible' and 'relative_delta' not in r for r in result))

    def test_prefetch_requires_matching_inputs(self):
        self.rows[1]['shared_input_sha256']='different-input'
        with self.assertRaisesRegex(ValueError,'mismatch'):
            paired_prefetch(self.rows)

    def test_prefetch_requires_both_candidates(self):
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            paired_prefetch(self.rows[:-1])


if __name__ == "__main__":
    unittest.main()
