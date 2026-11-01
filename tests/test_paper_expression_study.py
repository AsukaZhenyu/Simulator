"""Mutation tests for the paper study's independent event checker."""
import copy
import unittest

from experiments.paper_expression_study import CASES, RUNNERS, blackout_events, check_flat_record, source_metrics


def valid_document():
    return dict(operations=[dict(id="a",inputs=["x"],outputs=["y"]),
                            dict(id="b",inputs=["y"],outputs=["z"])],
        tensors=[dict(id=name,storage_bytes=32) for name in ("x","y","z")],
        events=[dict(kind="COPY",target_id="x",bytes_transferred=32,start_ns=0,end_ns=3),
                dict(kind="COMPUTE",target_id="a",device="cpu",start_ns=10,end_ns=20),
                dict(kind="COMPUTE",target_id="b",device="gpu",start_ns=21,end_ns=30)],
        memory=[[dict(used_bytes=96,capacity_bytes=128)]],
        initial_blackout=dict(device="cpu",until_ns=10))


class PaperExpressionCheckerTests(unittest.TestCase):
    def test_fixed_eight_cases(self):
        self.assertEqual(len(CASES),8)
        self.assertEqual({c["id"] for c in CASES},set(RUNNERS))
        self.assertEqual(sum(c["group"]=="new_composition" for c in CASES),4)

    def test_valid_reference(self):
        self.assertEqual(check_flat_record(valid_document())["dependency_edges"],1)

    def test_gpu_blackout_uses_existing_service_rate_contract(self):
        events=blackout_events(dict(device="gpu",until_ns=10000))
        self.assertEqual([(e.t_ns,e.gpu_service_rate_numerator,e.gpu_service_rate_denominator) for e in events],[(0,0,1),(10000,1,1)])

    def assert_bad(self,mutate,message):
        document=valid_document()
        mutate(document)
        with self.assertRaisesRegex(AssertionError,message):
            check_flat_record(document)

    def test_wrong_copy_payload(self):
        self.assert_bad(lambda d:d["events"][0].update(bytes_transferred=16),"COPY physical payload")

    def test_free_copy(self):
        self.assert_bad(lambda d:d["events"][0].update(end_ns=0),"uncharged COPY")

    def test_early_consumer(self):
        self.assert_bad(lambda d:d["events"][2].update(start_ns=19),"consumer before producer")

    def test_duplicate_charge(self):
        self.assert_bad(lambda d:d["events"].append(copy.deepcopy(d["events"][1])),"duplicate compute charge")

    def test_missing_operation(self):
        self.assert_bad(lambda d:d["events"].pop(),"incomplete operation")

    def test_capacity_overflow(self):
        self.assert_bad(lambda d:d["memory"][0][0].update(used_bytes=129),"physical capacity exceeded")

    def test_live_input_release(self):
        self.assert_bad(lambda d:d["events"].append(dict(kind="EVICT",target_id="x",location="dram",start_ns=15,end_ns=15)),"live input evicted")

    def test_blackout_ignored(self):
        self.assert_bad(lambda d:d["events"][1].update(start_ns=9),"compute during initial blackout")

    def test_helper_statements_are_counted(self):
        self.assertEqual(source_metrics("def recipe():\n    x = 1\n    return x\n")["ast_statements"],3)


if __name__=="__main__":
    unittest.main()
