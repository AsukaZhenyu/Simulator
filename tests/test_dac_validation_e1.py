"""Mutation tests for the E1 evidence checkers, not implementation-mirroring tests."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/"src",ROOT/"modeling",ROOT/"mapping"):
    sys.path.insert(0,str(path))

from experiments.dac_validation_e1 import spec,build
from experiments.dac_validation_e1_oracle import declared_dag,check_graph,replay_physical
from experiments.dac_validation_e1_numeric import check_numerical
from experiments.dac_validation_e1_lowering import action_dag,reschedule,verify_runtime


class EvidenceMutations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base=ROOT/"build/dac-validation-20261025/test-tmp";base.mkdir(parents=True,exist_ok=True)
        cls.temp=tempfile.TemporaryDirectory(dir=base)
        cls.scenario=spec("A03",8,1);cls.expected=declared_dag(cls.scenario)
        cls.compiled=build(cls.scenario,Path(cls.temp.name)/"baseline")
        cls.result=cls.compiled.run();assert cls.result.success
        cls.graph=cls.result.raw[1].execution_document();cls.events=[asdict(e) for e in cls.result.events]

    @classmethod
    def tearDownClass(cls):
        cls.compiled.close();cls.temp.cleanup()

    def test_independent_contract_and_numerical_paths_agree(self):
        self.assertTrue(check_graph(self.scenario,self.expected,self.graph,self.events)["exact_operations"])
        self.assertEqual(check_numerical(self.scenario,self.expected,self.graph,self.events)["status"],"pass")

    def test_wrong_operand_is_detected(self):
        bad=deepcopy(self.graph)
        op=next(o for o in bad["workload"]["operations"] if o["semantic_op"]=="ffn_sum")
        op["inputs"][0]=op["inputs"][1]
        with self.assertRaisesRegex(AssertionError,"inputs differs"):
            check_graph(self.scenario,self.expected,bad,self.events)

    def test_wrong_physical_shape_is_detected(self):
        bad=deepcopy(self.graph);bad["workload"]["tensors"][0]["ne"][0]+=1
        with self.assertRaisesRegex(AssertionError,"ne differs"):
            check_graph(self.scenario,self.expected,bad,self.events)

    def test_missing_copy_cannot_pass_structural_replay(self):
        removed=False;bad=[]
        for e in self.events:
            if e["kind"]=="COPY" and not removed:removed=True;continue
            bad.append(e)
        with self.assertRaises(AssertionError):
            replay_physical(self.scenario,self.expected,self.graph,bad,self.result.makespan_ns)

    def test_wrong_copy_duration_cannot_pass(self):
        bad=deepcopy(self.events);next(e for e in bad if e["kind"]=="COPY")["end_ns"]+=1
        with self.assertRaises(AssertionError):
            replay_physical(self.scenario,self.expected,self.graph,bad,self.result.makespan_ns)

    def test_swiglu_operand_swap_is_detected_numerically(self):
        bad=deepcopy(self.graph)
        for o in bad["workload"]["operations"]:
            if o["semantic_op"]=="ffn_swiglu":o["inputs"]=list(reversed(o["inputs"]))
        with self.assertRaisesRegex(AssertionError,"numeric mismatch"):
            check_numerical(self.scenario,self.expected,bad,self.events)

    def test_capacity_checker_rejects_impossible_initial_inventory(self):
        bad=dict(self.scenario,physical_capacity_bytes=1)
        with self.assertRaisesRegex(AssertionError,"capacity overflow"):
            replay_physical(bad,self.expected,self.graph,self.events,self.result.makespan_ns)

    def test_independent_lowering_is_accepted_by_real_runtime(self):
        dag=action_dag(self.scenario,self.expected,self.graph,self.events)
        trial=reschedule(self.scenario,self.expected,dag,"copy_cpu_gpu_evict")
        self.assertEqual(trial["status"],"pass")
        physical=replay_physical(self.scenario,self.expected,self.graph,trial["events"],trial["duration_ns"])
        receipt=verify_runtime(self.compiled,trial["events"])
        self.assertEqual(receipt["duration_ns"],physical["makespan_ns"])


if __name__=="__main__":unittest.main()
