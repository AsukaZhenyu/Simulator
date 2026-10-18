"""Small synthetic report tests; no native process or hardware is executed."""
from pathlib import Path
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("accuracy_report",HERE/"report.py")
r=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(r)


class Reader:
    def __init__(self,documents):self.documents=documents
    def read(self,path,**kwargs):
        return self.documents.get(str(Path(path)).replace("\\","/"),kwargs.get("fallback",{}))


def chain_fixture():
    case=dict(id="ffn-case",family="ffn",hidden=128,intermediate=256,tokens=1,cpu_intermediate=32)
    plan=[];rows=[]
    for split in ("profile","heldout"):
        for process in range(3):
            row=dict(id=f"{split}-{process}",case=case,split=split,status="measured",seed=process+(0 if split=="profile" else 10))
            plan.append(row);rows.append(row)
    gates=dict(numeric_and_process_coverage=True,model_frozen_before_heldout=True,
               within_process_cv=False,between_process_cv=True,heldout_relative_error=True)
    base=dict(case_id=case["id"],strategy="gpu/resident",predicted_ns=101,observed_ns=100,
              absolute_relative_error=.01,calibrated=False,gates=gates,source_evidence=[dict(row_id="profile-0",cv=.5)],
              heldout_evidence=[dict(row_id="heldout-0",cv=.01)],between_process_cv={"heldout_strategy":.01})
    trace=dict(case_id=case["id"],strategy="gpu/resident",predicted_ns=101,observed_uninstrumented_ns=100,
               absolute_relative_error=.01,calibrated=False,gates=gates,profile_cost_cv={"tiny_gap":dict(max_within_process_cv=.8,between_process_medians_cv=.1)})
    shape=dict(case_id=case["id"],strategy="gpu/resident",observed_ns=100,
               full=dict(predicted_ns=101,absolute_relative_error=.01),
               leave_one_case_out=dict(predicted_ns=101,absolute_relative_error=.01))
    docs={"chain/suite.json":dict(cases=[case],quality=r.QUALITY),"chain/plan.json":plan,"chain/measurements.json":rows,
          "chain/analysis/report.json":dict(results=[base],all_declared_strategies_calibrated=False),
          "chain/trace-report.json":dict(results=[trace],all_declared_strategies_calibrated=False),
          "chain/shape-report.json":dict(results=[shape],frozen_before_heldout=True,
             profile_target_sources=[dict(case_id="other",within_process_cv=.9)],
             profile_target_between_process=[dict(case_id="other",between_process_cv=.01,processes=3)]),
          "chain/diagnostic-summary.json":dict(timestamp_overhead_subtracted=False),"chain/manifest.json":dict(assets_unchanged=True)}
    return docs


class ReportTests(unittest.TestCase):
    def test_unknown_errors_are_not_zero(self):
        summary=r.error_summary([{},dict(absolute_relative_error=.1)])
        self.assertEqual(1,summary["unknown"])
        self.assertEqual(.1,summary["max"])

    def test_missing_paths_remain_declared_unknown(self):
        out=r.chain_summary("chain",Reader(chain_fixture()))
        self.assertEqual(7,len(out["paths"]))
        self.assertEqual(6,out["model_summaries"]["independent"]["errors"]["unknown"])
        self.assertEqual(0,out["model_summaries"]["independent"]["accepted"])

    def test_low_error_never_overrides_source_cv_failure(self):
        out=r.chain_summary("chain",Reader(chain_fixture()))
        row=next(x for x in out["paths"] if x["strategy"]=="gpu/resident")
        for name in ("independent","inchain","shared","leave_one_case_out"):
            self.assertEqual(.01,row[name]["absolute_relative_error"])
            self.assertFalse(row[name]["accepted"])
        self.assertFalse(row["shared"]["gates"]["profile_joint_target_within_process_cv"])

    def test_process_counts_do_not_count_each_path_as_new_process(self):
        out=r.chain_summary("chain",Reader(chain_fixture()))
        self.assertEqual(6,out["retained_processes"])
        self.assertEqual(6,out["families"]["ffn"]["measured_processes"])
        self.assertEqual(3,out["cases"][0]["processes"]["profile"]["measured"])

    def test_failed_and_missing_processes_cannot_disappear(self):
        docs=chain_fixture();docs["chain/measurements.json"][0]=dict(docs["chain/measurements.json"][0],status="failed",error="synthetic failure")
        docs["chain/measurements.json"].pop()
        out=r.chain_summary("chain",Reader(docs))
        self.assertEqual(1,len(out["failures"]))
        self.assertEqual(1,len(out["missing_process_ids"]))
        self.assertFalse(out["process_coverage_complete"])

    def test_strict_json_rejects_duplicate_and_nonfinite_values(self):
        for value in ('{"x":1,"x":2}','{"x":NaN}'):
            with self.assertRaises(ValueError):r.strict_json(value)

    def test_missing_campaigns_generate_explicit_incomplete_report(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-report-",dir=HERE/"tests") as temp:
            root=Path(temp).resolve()
            self.assertTrue(root.is_relative_to((HERE/"tests").resolve()))
            out=r.assemble(chain=root/"chain",highlevel=root/"high",workload_cpu=root/"cpu",workload_cuda=root/"cuda",output=root/"report")
            self.assertFalse(out["evidence_assembly_complete"])
            self.assertFalse(out["all_reported_latency_scopes_accepted"])
            self.assertFalse(out["all_accuracy_gaps_closed"])
            self.assertTrue((root/"report/REPORT.md").is_file())
            with self.assertRaises(ValueError):
                r.assemble(chain=root/"chain",highlevel=root/"high",workload_cpu=root/"cpu",workload_cuda=root/"cuda",output=root/"report")

    def test_source_mutation_marks_assembly_invalid(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-report-",dir=HERE/"tests") as temp:
            root=Path(temp).resolve()
            self.assertTrue(root.is_relative_to((HERE/"tests").resolve()))
            before=dict(files={"synthetic":"a"},missing_paths=[])
            after=dict(files={"synthetic":"b"},missing_paths=[])
            with patch.object(r,"snapshot",side_effect=[before,after]):
                out=r.assemble(chain=root/"chain",highlevel=root/"high",workload_cpu=root/"cpu",workload_cuda=root/"cuda",output=root/"report")
            self.assertFalse(out["input_assets_unchanged"])
            self.assertFalse(out["evidence_assembly_complete"])

    def test_overlap_preserves_all_paths_and_failed_stability_gate(self):
        case=dict(id="prefetch",hidden=256,intermediate=512,tokens=1,payload_bytes=1048576)
        plan=[dict(id=f"{split}-{n}",case=case,split=split)for split in("profile","heldout")for n in range(3)]
        rows=[dict(job,status="measured",metrics={},diagnostic_host_and_copy_events={})for job in plan]
        docs={"overlap/suite.json":dict(cases=[case]),"overlap/plan.json":plan,"overlap/measurements.json":rows,
            "overlap/manifest.json":dict(assets_unchanged=True),
            "overlap/report.json":dict(results=[dict(case_id="prefetch",strategy="overlap/one_copy_compute",absolute_relative_error=.01,
                 accepted=False,gates=dict(within_process_cv=False),source_evidence=[dict(cv=.9)],heldout_within_process_cv=[.01]*3)])}
        result=r.overlap_summary("overlap",Reader(docs))
        self.assertEqual(4,result["declared_paths"])
        self.assertEqual(6,result["retained_processes"])
        self.assertFalse(result["all_paths_accepted"])
        self.assertEqual(3,result["errors"]["unknown"])
        self.assertFalse(result["kernel_overlap_proved"])

    def test_overlap_missing_process_is_not_silently_dropped(self):
        case=dict(id="prefetch",hidden=256,intermediate=512,tokens=1,payload_bytes=1048576)
        plan=[dict(id="profile-0",case=case,split="profile")]
        docs={"overlap/suite.json":dict(cases=[case]),"overlap/plan.json":plan,"overlap/measurements.json":[],
              "overlap/manifest.json":{},"overlap/report.json":{}}
        result=r.overlap_summary("overlap",Reader(docs))
        self.assertEqual(["profile-0"],result["missing_process_ids"])
        self.assertFalse(result["process_coverage_complete"])


if __name__=="__main__":unittest.main()
