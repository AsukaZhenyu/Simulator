"""Guard scenario accounting, finite denominators and invalid-run ranking."""
from fractions import Fraction
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.dac_validation_e3 import (
    CANDIDATES, WORKLOADS, ROOT, Point, copy_scenario_cost, inventory,
    lower_serial_actions, main_points, response_costs, robustness_points,
    run_one, select, summarize, pack,
)


class E3ProtocolTests(unittest.TestCase):
    def test_complete_prespecified_main_and_corner_matrices(self):
        self.assertEqual(len(main_points()), 38)
        self.assertEqual(len(robustness_points()), 144)
        self.assertEqual(len(main_points())*len(WORKLOADS)*len(CANDIDATES), 304)
        self.assertEqual(len(robustness_points())*len(WORKLOADS)*len(CANDIDATES), 1152)
        self.assertEqual(len(inventory()["main_jobs"]), 304)
        for architecture, per_corner in (("separate", 5), ("shared", 4)):
            for corner in range(16):
                selected=[p for p in robustness_points() if p.architecture==architecture and p.corner==corner]
                self.assertEqual(len(selected), per_corner)
                self.assertTrue(any(p.focus_direction=="baseline" for p in selected))

    def test_bandwidth_changes_payload_only_and_keeps_exact_anchor(self):
        self.assertEqual(copy_scenario_cost(100, 20, 1, None), 100)
        self.assertEqual(copy_scenario_cost(100, 20, 2, 2), 80)
        self.assertEqual(copy_scenario_cost(100, 20, 0.5, 2), 140)
        with self.assertRaises(KeyError):
            copy_scenario_cost(100, 20, 2, None)
        with self.assertRaises(ValueError):
            copy_scenario_cost(10, 20, 2, 2)

    def test_only_selected_compute_device_changes(self):
        program=dict(graph=dict(workload=dict(tensors=[dict(id="x",storage_bytes=20)])),
            actions=[dict(kind="COMPUTE",device="cpu"),dict(kind="COMPUTE",device="gpu"),
                     dict(kind="COPY",target_id="x",src="dram",dst="vram"),dict(kind="EVICT")])
        prices=response_costs(program,[100,200,50,7],Point("separate","p",cpu=0.5))
        self.assertEqual([r["scenario_ns"] for r in prices],[50,200,50,7])
        shared=response_costs(program,[100,200,50,7],Point("shared","s",handoff_ns=3))
        self.assertEqual([r["scenario_ns"] for r in shared],[100,200,3,7])
        self.assertEqual(shared[2]["modeled_kind"],"SHARE")

    def test_independent_price_audit_rejects_scaling_entire_copy(self):
        from experiments.dac_validation_e3_audit import price_override_checks
        program=dict(graph=dict(workload=dict(tensors=[dict(id="x",storage_bytes=20)])),
                     actions=[dict(kind="COPY",target_id="x",src="dram",dst="vram"),dict(kind="ADVANCE")])
        point=Point("separate","p",link=1.2)
        prices=response_costs(program,[100,5],point,{"dram_to_vram":dict(qualified=True,slope_ns_per_byte=2)})
        evidence=dict(response_ledger=pack(prices))
        models={"dram->vram":dict(qualified=True,payload_ns_per_byte=2)}
        self.assertEqual(price_override_checks(evidence,dict(point=asdict(point)),program,models),2)
        evidence["response_ledger"][0]["scenario_ns"]=pack(Fraction(100)/Fraction(6,5))
        with self.assertRaises(ValueError):
            price_override_checks(evidence,dict(point=asdict(point)),program,models)

    def test_shared_wait_is_removed_but_compute_wait_is_preserved(self):
        program=dict(actions=[dict(kind="COPY",target_id="x",src="dram",dst="vram"),
                              dict(kind="ADVANCE"),dict(kind="COMPUTE",target_id="op",device="gpu"),
                              dict(kind="ADVANCE")])
        actions,indices=lower_serial_actions(program,True)
        self.assertEqual([a.kind for a in actions],["SHARE","COMPUTE","ADVANCE"])
        self.assertEqual(indices,(0,2,3))

    def test_failed_partial_time_never_enters_argmin(self):
        rows=[dict(candidate="S0",success=False,completed_makespan_ns=None,diagnostic_partial_ns=1),
              dict(candidate="S1",success=True,completed_makespan_ns=100)]
        self.assertEqual(select(rows)["candidate"],"S1")
        self.assertIsNone(select(rows[:1]))

    def test_unknown_candidate_prevents_partial_best_claim(self):
        rows=[dict(workload=WORKLOADS[0],candidate=c,point=asdict(Point("separate","separate-base")),
                   success=c!="S3",completed_makespan_ns=100 if c!="S3" else None,
                   status="valid" if c!="S3" else "unsupported_contract") for c in CANDIDATES]
        summary=summarize(rows)
        self.assertIsNone(summary["choices"][0]["selected"])
        self.assertFalse(summary["choices"][0]["all_outcomes_available"])

    def test_tensor_engine_and_independent_audit_agree_for_shared(self):
        from profiling.highlevel_accuracy.cases import build_case
        from profiling.highlevel_accuracy.bridge import export_plan
        with TemporaryDirectory(dir=ROOT/"build") as directory:
            plan=build_case(Path(directory),hidden=8,intermediate=16,heads=2,kv_heads=1,
                head_dim=4,tokens=1,history=3,threads=1,devices=("gpu","cpu","gpu"))
            program=export_plan(plan.compile(),seed=20261025)
            responses=[Fraction(100 if a["kind"] in ("COPY","COMPUTE") else 2)
                       for a in program["actions"]]
            separate=run_one(program,responses,Point("separate","base"),1<<20)
            shared=run_one(program,responses,Point("shared","base"),1<<20)
            self.assertTrue(separate["summary"]["success"],separate["summary"])
            self.assertTrue(shared["summary"]["success"],shared["summary"])
            self.assertGreater(separate["summary"]["h2d_bytes"],0)
            self.assertEqual(shared["summary"]["h2d_bytes"],0)
            self.assertEqual(shared["summary"]["d2h_bytes"],0)
            self.assertGreater(shared["summary"]["audit"]["share_events"],0)
            self.assertLess(shared["summary"]["completed_makespan_ns"],
                            separate["summary"]["completed_makespan_ns"])
            self.assertEqual(separate["summary"]["completed_makespan_ns"],sum(responses))
            from experiments.dac_validation_e3_audit import capacity_rejection
            initial=sum(t["storage_bytes"] for t in program["graph"]["workload"]["tensors"]
                        if "dram" in t.get("initial_locations",[]))
            failed=run_one(program,responses,Point("separate","tight"),initial)
            self.assertFalse(failed["summary"]["success"])
            self.assertEqual(failed["summary"]["status"],"capacity_infeasible")
            self.assertTrue(capacity_rejection(failed,failed["summary"])["initial"])

    def test_all_four_serial_candidate_topologies(self):
        from profiling.highlevel_accuracy.cases import build_case
        from profiling.highlevel_accuracy.bridge import export_plan
        specifications=[dict(devices=("cpu","cpu","cpu")),dict(devices=("gpu","gpu","gpu")),
                        dict(devices=("gpu","cpu","gpu")),
                        dict(devices=("gpu","gpu","gpu"),ffn_cut=8,ffn_devices=("cpu","gpu"))]
        for specification in specifications:
            with self.subTest(specification=specification),TemporaryDirectory(dir=ROOT/"build") as directory:
                program=export_plan(build_case(Path(directory),hidden=8,intermediate=16,heads=2,kv_heads=1,
                    head_dim=4,tokens=1,history=3,threads=1,**specification).compile(),seed=20261025)
                prices=[Fraction(100 if a["kind"] in ("COMPUTE","COPY") else 2) for a in program["actions"]]
                for architecture in ("shared","separate"):
                    outcome=run_one(program,prices,Point(architecture,"base"),1<<20)
                    self.assertTrue(outcome["summary"]["success"],outcome["summary"])
                    self.assertGreater(outcome["summary"]["audit"]["compute_events"],0)


if __name__ == "__main__":
    unittest.main()
