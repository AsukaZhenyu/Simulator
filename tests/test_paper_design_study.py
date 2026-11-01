"""Controls for experiment design and finite-candidate selection analysis."""
import copy
import unittest
from experiments.paper_design_study import Case, GIB, analyze, inputs, jobs, weight_bytes


class DesignStudyTests(unittest.TestCase):
    def test_frozen_factorial(self):
        main, long = jobs("main"), jobs("long")
        self.assertEqual(len(main), 504)
        self.assertEqual(len(long), 112)
        self.assertEqual(len({c.key for c in main+long}), 616)
        self.assertEqual({(c.vram_gib,c.link_gbs) for c in main},
                         {(v,b) for v in (2,4,12) for b in (2,8,24)})

    def test_strategy_does_not_change_hardware_or_workload(self):
        for name in ("dense","moe_hot","moe_rotating"):
            cases=[c for c in jobs("main") if c.workload==name and c.initial_state=="cold" and c.vram_gib==4 and c.link_gbs==8]
            baseline=inputs(cases[0])[:3]
            self.assertTrue(all(inputs(case)[:3]==baseline for case in cases))
            for case in cases:
                self.assertEqual(inputs(case)[3].cache.budget_bytes, 4*GIB)

    def test_initial_weights_capacity_is_graph_derived(self):
        dense=inputs(Case("dense",1,"cold",4,8,"e0-s0"))[0]
        moe=inputs(Case("moe_hot",1,"cold",4,8,"e0-s0"))[0]
        self.assertTrue(2*GIB < weight_bytes(dense) < 4*GIB)
        self.assertTrue(4*GIB < weight_bytes(moe) < 12*GIB)

    def test_selection_methods_and_costs_not_tautological_claim(self):
        rows=[]
        # Joint wins at e1-s1, but fixing s0 chooses e0 before stagewise s1.
        for candidate,latency in (("e0-s0",100),("e0-s1",90),("e1-s0",110),("e1-s1",60)):
            row=dict(workload="dense",decode_steps=1,initial_state="cold",vram_gib=12,link_gbs=8,
                     candidate=candidate,success=True,completed_makespan_ns=latency)
            rows.append(row)
        group=analyze(rows)["groups"][0]
        self.assertEqual(group["choices"]["joint"]["candidate"],"e1-s1")
        self.assertEqual(group["choices"]["stagewise_execution_then_storage"]["candidate"],"e0-s1")
        self.assertEqual(group["choices"]["stagewise_execution_then_storage"]["regret_vs_finite_best"],.5)
        self.assertEqual(group["choices"]["stagewise_execution_then_storage"]["evaluations"],3)
        failed=copy.deepcopy(rows)
        for row in failed:
            row.update(success=False,completed_makespan_ns=None,status="initial_capacity_infeasible")
        self.assertFalse(analyze(failed)["groups"][0]["feasible"])


if __name__ == "__main__":
    unittest.main()
