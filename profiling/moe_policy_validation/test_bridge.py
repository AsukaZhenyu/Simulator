"""Causal policy compilation, bounded physical slots, and explicit macro costs."""
from copy import deepcopy
import unittest

from llm_simulator.expert_recipes import EXPERT_RECIPE_NAMES
from llm_infer_model.performance.expert_policy_response import evaluate_expert_policy
from moe_policy_validation.bridge import compile_expert_program


def inputs():
    config = dict(hidden=8, intermediate=16, experts=4, gpu_slots=2, threads=1, cuda_device=0)
    steps = [dict(input=[[0.1] * 8], routes=[[a, b]], scores=[[0.4, 0.6]])
             for a, b in ((0, 1), (1, 2), (2, 3), (0, 1))]
    prices = dict(schema="moe-expert-policy-calibration-1", config=config,
        boundary="serial_host_expert_response_v1", prices_ns={
            "LOAD": 20, "RELEASE": 1, "EXPERT:cpu:q=1": 100,
            "EXPERT:gpu:q=1": 20, "COMBINE:q=1:topk=2": 5})
    return dict(config=config, steps=steps, seed=42, prices=prices,
                resource_guards=dict(host_reserve_bytes=1048576, gpu_reserve_bytes=1048576))


class ExpertBridgeTests(unittest.TestCase):
    def test_release_after_use_and_full_cache_gpu_path_have_declared_load_counts(self):
        for slots in (2, 4):
            for locality in ("hot", "capacity-sensitive"):
                pairs = [(0,1)] * 8 if locality == "hot" else [(0,1),(2,3),(0,1),(2,3),(4,5),(6,7),(4,5),(6,7)]
                source = inputs()
                source["config"].update(experts=8, gpu_slots=slots)
                source["steps"] = [dict(input=[[0.1]*8],routes=[list(pair)],scores=[[0.6,0.4]]) for pair in pairs]
                for recipe in ("gpu_demand_lru", "current_cost_lru"):
                    for retention in ("lru", "release_after_use"):
                        result = compile_expert_program(**source, recipe=recipe, retention=retention)
                        actions = result["program"]["actions"]
                        expected = 16 if retention == "release_after_use" else (2 if locality == "hot" else (16 if slots==2 else 8))
                        self.assertEqual(sum(a["kind"] == "LOAD" for a in actions), expected,
                                         (slots, locality, recipe, retention))
                        self.assertEqual({a["device"] for a in actions if a["kind"] == "EXPERT"}, {"gpu"})
                        evaluate_expert_policy(result["program"], source["prices"])
                        if retention == "release_after_use":
                            for index, action in enumerate(actions):
                                if action["kind"] == "EXPERT":
                                    self.assertEqual(actions[index+1]["kind"], "RELEASE")

    def test_release_policy_prefix_is_invariant_to_unknown_future_routes(self):
        first=inputs(); alternative=deepcopy(first)
        alternative["steps"][2]["routes"]=[[0,2]]
        for recipe in ("gpu_demand_lru", "current_cost_lru"):
            a=compile_expert_program(**first,recipe=recipe,retention="release_after_use")
            b=compile_expert_program(**alternative,recipe=recipe,retention="release_after_use")
            self.assertEqual([r for r in a["compilation_evidence"]["records"] if r["step"]<2],
                             [r for r in b["compilation_evidence"]["records"] if r["step"]<2])

    def test_all_recipes_compile_and_pass_independent_model_state_validation(self):
        for name in EXPERT_RECIPE_NAMES:
            with self.subTest(recipe=name):
                source = inputs()
                result = compile_expert_program(**source, recipe=name)
                evaluation = evaluate_expert_policy(result["program"], source["prices"])
                self.assertGreater(evaluation["duration_ns"], 0)
                self.assertFalse(result["compilation_evidence"]["full_tensor_runtime_replay"])
                self.assertEqual(sum(a["kind"] == "COMBINE" for a in result["program"]["actions"]), 4)
                self.assertEqual(sum(a["kind"] == "EXPERT" for a in result["program"]["actions"]), 8)

    def test_real_evictions_and_fixed_placement_are_not_just_labels(self):
        gpu = compile_expert_program(**inputs(), recipe="gpu_demand_lru")["program"]
        self.assertEqual(sum(a["kind"] == "LOAD" for a in gpu["actions"]), 6)
        self.assertEqual(sum(a["kind"] == "RELEASE" for a in gpu["actions"]), 4)
        self.assertEqual({a["device"] for a in gpu["actions"] if a["kind"] == "EXPERT"}, {"gpu"})
        cpu = compile_expert_program(**inputs(), recipe="cpu_experts_lru")["program"]
        self.assertFalse(any(a["kind"] == "LOAD" for a in cpu["actions"]))
        self.assertEqual({a["device"] for a in cpu["actions"] if a["kind"] == "EXPERT"}, {"cpu"})

    def test_unknown_future_cannot_change_compiled_prefix_or_observation_hash(self):
        first = inputs()
        alternative = deepcopy(first)
        alternative["steps"][2]["routes"] = [[0, 2]]
        alternative["steps"][3]["routes"] = [[2, 3]]
        for name in EXPERT_RECIPE_NAMES:
            a = compile_expert_program(**first, recipe=name)
            b = compile_expert_program(**alternative, recipe=name)
            self.assertEqual([row for row in a["compilation_evidence"]["records"] if row["step"] < 2],
                             [row for row in b["compilation_evidence"]["records"] if row["step"] < 2], name)

    def test_missing_price_mismatched_boundary_overlap_and_routes_fail(self):
        bad = inputs()
        del bad["prices"]["prices_ns"]["EXPERT:gpu:q=1"]
        with self.assertRaisesRegex(ValueError, "missing.*calibration"):
            compile_expert_program(**bad, recipe="gpu_demand_lru")
        bad = inputs()
        bad["prices"]["boundary"] = "pure_gpu_kernel_only"
        with self.assertRaisesRegex(ValueError, "cost boundary"):
            compile_expert_program(**bad, recipe="gpu_demand_lru")
        bad = inputs()
        bad["config"]["allow_cpu_gpu_overlap"] = True
        with self.assertRaisesRegex(ValueError, "serial"):
            compile_expert_program(**bad, recipe="gpu_demand_lru")
        bad = inputs()
        bad["steps"][0]["routes"] = [[1, 1]]
        with self.assertRaisesRegex(ValueError, "distinct"):
            compile_expert_program(**bad, recipe="gpu_demand_lru")


if __name__ == "__main__":
    unittest.main()
