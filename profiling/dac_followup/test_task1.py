from dataclasses import replace
import unittest

from llm_infer_model.performance.transfer_response import TransferResponse
from .transfer_fit import fit_from_training, predict
from .task1 import byte_reference
from profiling.steady_costs.contracts import component_reference, digest


class TransferResponseTests(unittest.TestCase):
    def response(self):
        return TransferResponse("dram->vram",10,2,1,100,4,1000,"explicit host buffer", "copy+sync",("sha",),True)

    def test_anchor_preserving_payload_scenario(self):
        response=self.response()
        self.assertEqual(response.predict_ns(200),310)
        self.assertEqual(response.payload_ns(200),300)
        self.assertEqual(response.scenario_ns(400,200,1),400)
        self.assertEqual(response.scenario_ns(400,200,2,assume_effective_payload_scaling=True),250)
        with self.assertRaises(ValueError):response.scenario_ns(299,200,1)
        with self.assertRaises(ValueError):response.scenario_ns(400,200,2)
        with self.assertRaises(ValueError):response.predict_ns(200,payload_bandwidth_factor=2)

    def test_unknown_scope_and_bad_parameters_rejected(self):
        with self.assertRaises(ValueError):self.response().predict_ns(2000)
        with self.assertRaises(ValueError):replace(self.response(),qualified=False).predict_ns(100)
        with self.assertRaises(ValueError):replace(self.response(),fixed_ns=-1)
        with self.assertRaises(ValueError):replace(self.response(),route="unknown")
        with self.assertRaises(ValueError):replace(self.response(),evidence=())
        with self.assertRaises(ValueError):self.response().scenario_ns(400,200,0)

    def test_training_only_and_known_hinge_reconstruction(self):
        sizes=(4,256,4096,65536,1048576,4194304,16777216,67108864)
        rows=[dict(split="calibration",bytes=size,mean_ns=10000+.02*min(size,262144)+.1*max(0,size-262144)) for size in sizes]
        result=fit_from_training(rows)
        self.assertEqual(result["selected"]["breakpoint_bytes"],262144)
        self.assertAlmostEqual(predict(result["selected"],16384),10327.68,places=6)
        with self.assertRaises(ValueError):fit_from_training([dict(rows[0],split="heldout"),*rows[1:]])

    def test_chunked_complete_byte_oracle_matches_existing_independent_generator(self):
        for size in (4,1024,262149):
            program=dict(format_version="steady-component-program-1",kind="transfer",device="gpu",threads=1,cuda_device=0,
                         seed=2026102511,shape=dict(bytes=size,src="dram",dst="vram"),allocation_budget_bytes=1<<30)
            program["program_sha256"]=digest(program)
            self.assertEqual(byte_reference(program["seed"],size)["sha256"],component_reference(program)["sha256"])

if __name__=="__main__":unittest.main()
