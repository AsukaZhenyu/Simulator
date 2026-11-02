"""Meaningful cost-contract, finite arithmetic and balanced-order regressions."""
from copy import deepcopy
from fractions import Fraction
import unittest

from llm_infer_model.performance.reusable_action_costs import ReusableResponseTable, coordinate, digest
from profiling.dac_validation.e2a import SEQUENCES, matched, request, verify_output


def tiny():
    tensor=lambda tid,n:dict(id=tid,dtype="GGML_TYPE_F32",ne=[n,1,1,1],nb=[4,4*n,4*n,4*n],storage_bytes=4*n)
    return dict(actions=[dict(kind="COPY",target_id="x",src="dram",dst="vram"),dict(kind="COMPUTE",target_id="o",device="gpu")],
        graph=dict(workload=dict(tensors=[tensor("x",4),tensor("y",4)])),
        bindings=[dict(operation_id="o",kind="projection",shape=dict(tokens=1),op_params={},operands=dict(hidden="x"),outputs=dict(q="y"),history=[],new_kv=[])],
        physical=dict(config=dict(hidden=4),threads=1),boundary_contract="test_exact_f32")


class ReusableContractTests(unittest.TestCase):
    def test_coordinate_ignores_identity_and_numeric_values(self):
        p=tiny();q=deepcopy(p);q.update(seed=7,program_sha256="new",strategy="irrelevant",initial_values={"x":[7]*4})
        self.assertEqual(coordinate(p,1,{}),coordinate(q,1,{}))
        # Order is not the coordinate: exact compute moved to another ordinal.
        q["actions"].reverse()
        self.assertEqual(coordinate(p,1,{}),coordinate(q,0,{}))

    def test_exact_shape_device_dtype_and_runtime_rejected(self):
        p=tiny();coord=coordinate(p,1,{"binary":"A"})
        row=dict(key=digest(coord),coordinate=coord,mean_ns="17/2",sample_count=1,evidence=["independent"])
        table=ReusableResponseTable((row,))
        self.assertEqual(table.lookup(coord),Fraction(17,2))
        for mutation in ("shape","device","dtype","runtime"):
            q=deepcopy(p);context={"binary":"A"}
            if mutation=="shape":q["bindings"][0]["shape"]["tokens"]=2
            if mutation=="device":q["actions"][1]["device"]="cpu"
            if mutation=="dtype":q["graph"]["workload"]["tensors"][0]["dtype"]="GGML_TYPE_F16"
            if mutation=="runtime":context["binary"]="B"
            with self.assertRaises(KeyError):table.lookup(coordinate(q,1,context))

    def test_serial_sum_and_no_extra_inclusive_overhead(self):
        p=tiny();coords=[coordinate(p,i,{}) for i in range(2)]
        entries=tuple(dict(key=digest(c),coordinate=c,mean_ns=str(n),sample_count=1,evidence=["independent"]) for c,n in zip(coords,(5,11)))
        result=ReusableResponseTable(entries).evaluate(p,{})
        self.assertEqual(result["duration_ns"],16);self.assertEqual(result["extra_transition_ns"],0)
        self.assertFalse(result["target_outcome_used"])

    def test_duplicate_or_changed_parameter_rejected(self):
        coord=coordinate(tiny(),0,{})
        row=dict(key=digest(coord),coordinate=coord,mean_ns="1",sample_count=1,evidence=["independent"])
        with self.assertRaises(ValueError):ReusableResponseTable((row,row))
        with self.assertRaises(ValueError):ReusableResponseTable((dict(row,key="wrong"),))
        with self.assertRaises(ValueError):ReusableResponseTable((dict(row,mean_ns=None),))

    def test_eight_rounds_balance_each_candidate_position(self):
        self.assertEqual(len(SEQUENCES),8)
        for order in SEQUENCES:self.assertEqual(sorted(order),[0,1,2,3])
        for strategy in range(4):
            self.assertEqual([sum(order[position]==strategy for order in SEQUENCES) for position in range(4)],[2]*4)

    def test_primitive_and_target_minimum_warmup_and_batch_boundary(self):
        primitive=request("action",target=0);target=request("full",trials=12)
        self.assertEqual(primitive["target_action_index"],0)
        self.assertEqual(primitive["warmup_min_ms"],target["warmup_min_ms"])
        self.assertEqual(primitive["warmup_min_ms"],500)
        self.assertEqual(primitive["repeats"],target["repeats"])

    def test_retained_initial_kv_goal_is_not_a_produced_port(self):
        p=dict(initial_values={"old_kv":[.25,.5]},graph=dict(workload=dict(tensors=[]),
            boundary=dict(ffn_execution=dict(weight_partitions=[])),scenario=dict(goal=dict(output_locations={"old_kv":"dram","new_hidden":"dram"}))))
        programs=[deepcopy(p) for _ in range(4)];refs=[{"new_hidden":[1.,2.]} for _ in range(4)]
        self.assertTrue(matched(programs,refs)["passed"])
        programs[3]["initial_values"]["old_kv"][0]=.75
        with self.assertRaises(ValueError):matched(programs,refs)


if __name__=="__main__":unittest.main()
