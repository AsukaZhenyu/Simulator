"""Meaningful coordinate and unknown-context checks, with no native timing."""
import copy
from fractions import Fraction
import unittest

from llm_infer_model.performance.context_response import ContextResponseTable, context_coordinate
from llm_infer_model.performance.reusable_action_costs import digest


def program():
    tensor=dict(id="x", dtype="F32", ne=[4,1,1,1], nb=[4,16,16,16], storage_bytes=16)
    return dict(graph=dict(workload=dict(tensors=[tensor])),
                actions=[dict(kind="COPY",target_id="x",src="dram",dst="vram"),
                         dict(kind="ADVANCE"),
                         dict(kind="COPY",target_id="x",src="vram",dst="dram")])


class ContextResponseTests(unittest.TestCase):
    def test_cyclic_predecessor_and_ignored_logical_control(self):
        p=program()
        self.assertEqual(context_coordinate(p,0,{})["previous_service"]["src"],"vram")
        self.assertEqual(context_coordinate(p,2,{})["previous_service"]["src"],"dram")
        self.assertIsNone(context_coordinate(p,1,{})["previous_service"])

    def test_renaming_does_not_change_price_and_unknown_history_rejects(self):
        p=program();context={"binary":"frozen"};entries=[]
        for i in range(3):
            c=context_coordinate(p,i,context)
            entries.append(dict(key=digest(c),coordinate=c,mean_ns="10",sample_count=3,evidence=["independent-test-fixture"]))
        table=ContextResponseTable(tuple(entries))
        self.assertEqual(table.evaluate(p,context)["duration_ns"],Fraction(30))
        renamed=copy.deepcopy(p);renamed["graph"]["workload"]["tensors"][0]["id"]="other"
        for a in renamed["actions"]:
            if "target_id" in a:a["target_id"]="other"
        self.assertEqual(table.evaluate(renamed,context)["duration_ns"],Fraction(30))
        changed=copy.deepcopy(p);changed["actions"][2]["src"]="dram"
        with self.assertRaises(KeyError):table.evaluate(changed,context)


if __name__=="__main__":unittest.main()
