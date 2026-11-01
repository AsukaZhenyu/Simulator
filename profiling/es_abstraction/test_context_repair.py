"""Lightweight contract tests; no native process or GPU is started."""
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from profiling.es_abstraction import context_repair as repair


class ContextRepairTests(unittest.TestCase):
    def test_eight_calls_preserve_heavy_light_rows_and_physical_lifecycle(self):
        specs=repair.specifications()
        self.assertEqual(len(specs),6)
        for spec in specs:
            program=repair.make_program(spec,repair.CALIBRATION_SEED,dimensions=(8,16))
            self.assertEqual(len(program["steps"]),8)
            self.assertTrue(all(len(s["input"])==32+spec["light_q"] for s in program["steps"]))
            prices=Counter(repair.model.action_price_keys(program))
            device="cpu" if spec["lifecycle"]=="cpu" else "gpu"
            self.assertEqual(prices[f"EXPERT:{device}:q=32"],16)
            self.assertEqual(prices[f"EXPERT:{device}:q={spec['light_q']}"],16)
            self.assertEqual(prices[f"COMBINE:q={32+spec['light_q']}:topk=2"],8)
            loads={"cpu":0,"lru":4,"release":32}[spec["lifecycle"]]
            self.assertEqual(prices["LOAD"],loads)
            self.assertEqual(prices["RELEASE"],32 if spec["lifecycle"]=="release" else 0)

    def test_pure_cpu_k_changes_only_unused_capacity_and_validation_is_independent(self):
        specs=repair.specifications()
        a,b=[repair.make_program(next(s for s in specs if s["id"]==f"real-k{k}-q8-cpu"),
                                repair.CALIBRATION_SEED,dimensions=(8,16)) for k in (2,4)]
        self.assertEqual(a["steps"],b["steps"])
        self.assertEqual(a["actions"],b["actions"])
        a["config"]["gpu_slots"]=4
        self.assertEqual(a,b)
        independent=repair.make_program(specs[0],repair.VALIDATION_SEED,dimensions=(8,16))
        calibration=repair.make_program(specs[0],repair.CALIBRATION_SEED,dimensions=(8,16))
        self.assertEqual(independent["actions"],calibration["actions"])
        self.assertNotEqual(independent["steps"][0]["input"],calibration["steps"][0]["input"])
        self.assertNotEqual(independent["seed"],calibration["seed"])

    def test_random_blocks_have_exact_pairs_and_balanced_mode_order(self):
        specs=repair.specifications(); order=repair.blocked_order(specs,4,"calibration")
        self.assertEqual(len(order),48)
        self.assertEqual(order,repair.blocked_order(specs,4,"calibration"))
        seen=Counter((r["block"],r["case_id"],r["mode"]) for r in order)
        self.assertEqual(set(seen.values()),{1})
        first=defaultdict(Counter)
        for index in range(0,len(order),2):
            a,b=order[index:index+2]
            self.assertEqual((a["case_id"],a["block"]),(b["case_id"],b["block"]))
            self.assertEqual({a["mode"],b["mode"]},{"bare","stamped"})
            first[a["case_id"]][a["mode"]]+=1
        self.assertTrue(all(x==Counter(bare=2,stamped=2) for x in first.values()))
        validation=repair.blocked_order(specs,4,"validation")
        self.assertEqual(len(validation),24)
        self.assertEqual({x["mode"] for x in validation},{"bare"})
        with self.assertRaisesRegex(ValueError,"even"):
            repair.blocked_order(specs,3,"calibration")

    def test_category_measurements_reject_missing_or_incompatible_stamps(self):
        p=repair.make_program(repair.specifications()[0],32030,dimensions=(8,16))
        with self.assertRaisesRegex(ValueError,"requires"):
            repair.category_means(p,{})
        with self.assertRaisesRegex(ValueError,"count"):
            repair.category_means(p,{"action_samples_ns":[[1,2]]})
        samples=[[float(i+1) for i in range(len(p["actions"]))]]
        means=repair.category_means(p,{"action_samples_ns":samples})
        self.assertEqual(set(means),set(repair.model.action_price_keys(p)))
        samples[0][0]=float("nan")
        with self.assertRaisesRegex(ValueError,"invalid"):
            repair.category_means(p,{"action_samples_ns":samples})

    def test_freeze_keeps_contexts_separate_and_does_not_read_validation(self):
        # Synthetic measurement records test the data contract, not accuracy.
        parent=repair.base.ROOT/"build";parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="context-contract-test-",dir=parent) as folder:
            out=Path(folder).resolve()
            self.assertTrue(out.is_relative_to(parent.resolve()))
            specs=repair.specifications(); entries=[]
            for phase,seed in (("calibration",32030),("validation",42030)):
                for spec in specs:
                    p=repair.make_program(spec,seed,dimensions=(8,16))
                    rel=f"programs/{phase}/{spec['id']}.json"
                    repair.base.save(out/rel,p)
                    entries.append(dict(phase=phase,case_id=spec["id"],path=rel,
                                        file_sha256=repair.base.digest(out/rel)))
                    if phase!="calibration":continue
                    keys=set(repair.model.action_price_keys(p))
                    for block in range(4):
                        for mode in repair.MODES:
                            value=100*spec["light_q"]+spec["k"]+block
                            row=dict(numerically_equal=True,compiled_action_sequence_equal=True,
                                     bounded_weight_slots=True,process_mean_ns=10000+block,
                                     mean_sequence_minus_action_stamps_ns=0,
                                     component_means_ns={key:float(value) for key in keys} if mode=="stamped" else None)
                            repair.base.save(out/"calibration"/spec["id"]/mode/f"process-{block}.verification.json",row)
            protocol=dict(specs=specs,programs=entries,blocks=4,sources={},
                          price_partition=["light_q","gpu_slots","lifecycle"],
                          unknown_states=["OS ready/block timing"],
                          order={"calibration":repair.blocked_order(specs,4,"calibration")})
            repair.base.save(out/"PROTOCOL.json",protocol)
            real_read=repair.base.read; read_paths=[]
            def observed_read(path):
                read_paths.append(Path(path))
                return real_read(path)
            with patch.object(repair,"check_sources",return_value=None), \
                 patch.object(repair.base,"invoke",side_effect=AssertionError("native forbidden")), \
                 patch.object(repair.base,"read",side_effect=observed_read):
                frozen=repair.freeze(out)
            a=real_read(out/"profiles/small-k4-q1-lru.json")
            b=real_read(out/"profiles/small-k4-q8-lru.json")
            self.assertNotEqual(a["prices_ns"]["EXPERT:gpu:q=32"],b["prices_ns"]["EXPERT:gpu:q=32"])
            self.assertEqual(len(frozen["cases"]),6)
            self.assertFalse(frozen["old_validation_data_used"])
            self.assertTrue(all(p.is_relative_to(out) for p in read_paths))
            self.assertFalse(any("validation" in p.relative_to(out).parts[:1] for p in read_paths))
            with patch.object(repair,"check_sources",return_value=None):
                with self.assertRaisesRegex(ValueError,"already frozen"):
                    repair.freeze(out)

    def test_paired_units_not_individual_invocations(self):
        result=repair.paired_contrast([10,20,30,40],[11,22,33,44])
        self.assertEqual(result["independent_blocks"],4)
        self.assertAlmostEqual(result["second_vs_first_percent"],10)
        self.assertEqual(result["paired_block_differences_ns"],[1,2,3,4])
        with self.assertRaisesRegex(ValueError,"equal"):
            repair.paired_contrast([1,2],[2])

    def test_baseline_reader_only_opens_allowlisted_old_profile_files(self):
        parent=repair.base.ROOT/"build";parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="context-baseline-test-",dir=parent) as folder:
            root=Path(folder).resolve();specs=repair.specifications()
            self.assertTrue(root.is_relative_to(parent.resolve()))
            expected=set()
            for geometry,k in {(s["geometry"],s["k"]) for s in specs}:
                path=root/"calibration"/geometry/f"k{k}"/"calibration.json"
                repair.base.save(path,dict(schema="moe-expert-policy-calibration-1",
                                          boundary=repair.model.TIMING_BOUNDARY,prices_ns={"LOAD":1}))
                expected.add(path)
            # Deliberately invalid old result: loading it would fail the test.
            (root/"validation").mkdir()
            (root/"validation/ANALYSIS_FINAL.json").write_text("DO NOT READ",encoding="utf-8")
            read_paths=[];real_read=repair.base.read
            def observed(path):
                read_paths.append(Path(path));return real_read(path)
            with patch.object(repair.base,"read",side_effect=observed):
                tables=repair.baseline_tables(root,specs)
            self.assertEqual(set(read_paths),expected)
            self.assertEqual(len(tables),3)
            self.assertTrue(all(len(x["origin_sha256"])==64 for x in tables.values()))


if __name__=="__main__":
    unittest.main()
