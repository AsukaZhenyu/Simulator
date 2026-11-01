"""Recipe claims, compiled placement, real cache pressure and causal prefixes."""
from dataclasses import replace
import unittest

from llm_simulator import Simulator, Workload
from llm_simulator.demo import make_demo
from llm_simulator.expert_recipes import (
    EXPERT_RECIPE_NAMES, expert_recipe, expert_recipe_scope,
)


class ExpertRecipeTests(unittest.TestCase):
    def test_unknown_names_and_implicit_budget_are_rejected(self):
        with self.assertRaises(ValueError):
            expert_recipe("Fiddler", gpu_weight_budget_bytes=100)
        for budget in (True, None, 1.5, -1):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                expert_recipe("gpu_demand_lru", gpu_weight_budget_bytes=budget)
        with self.assertRaises(TypeError):
            expert_recipe("gpu_demand_lru")

    def test_recipes_share_budget_and_expose_honest_scope(self):
        for name in EXPERT_RECIPE_NAMES:
            declaration = expert_recipe(name, gpu_weight_budget_bytes=24576)
            self.assertEqual(declaration.cache.budget_bytes, 24576)
            self.assertEqual(declaration.cache.rule, "lru")
            self.assertEqual(declaration.cache.unit, "expert_group")
            self.assertEqual(declaration.prefetch.next_layers, 0)
            self.assertFalse(expert_recipe_scope(name)["full_paper_reproduction"])
            self.assertFalse(expert_recipe_scope(name)["future_route_visibility"])

    def test_all_recipes_use_common_compiler_and_honor_placement(self):
        for name in EXPERT_RECIPE_NAMES:
            with self.subTest(recipe=name):
                simulator, _ = make_demo("moe")
                with simulator.compile(scheduling=expert_recipe(name, gpu_weight_budget_bytes=24576)) as compiled:
                    result = compiled.run()
                    self.assertTrue(result.success, result.summary())
                    kinds = {op.operation_id: op.kind for op in compiled.plan._compiled.model.aggregate.tasks}
                    experts = [event for event in result.events
                               if event.kind == "COMPUTE" and kinds.get(event.target_id) == "expert_ffn"]
                    self.assertTrue(experts)
                    if name in ("gpu_demand_lru", "cpu_experts_lru"):
                        expected = "gpu" if name == "gpu_demand_lru" else "cpu"
                        self.assertEqual({event.device for event in experts}, {expected})
                    if name == "gpu_demand_lru":
                        # The budget is smaller than all expert weights; there
                        # must be actual evictions, not merely a cache label.
                        self.assertTrue(any(event.kind == "EVICT" for event in result.events))
                    self.assertFalse(result.summary()["hardware_measurement"])

    def test_future_route_change_does_not_change_pre_router_decisions(self):
        simulator, _ = make_demo("moe")
        routes = tuple(replace(route, expert_ids=(2, 3)) if route.absolute_position == 4 else route
                       for route in simulator.workload.routes)
        altered = Simulator(model=simulator.model, hardware=simulator.hardware,
            workload=replace(simulator.workload, routes=routes), cost_model=simulator.cost_model)
        for name in EXPERT_RECIPE_NAMES:
            sequences = []
            for source in (simulator, altered):
                with source.compile(scheduling=expert_recipe(name, gpu_weight_budget_bytes=24576)) as compiled:
                    result = compiled.run()
                    self.assertTrue(result.success, result.summary())
                    # Stop immediately before the first router; all future
                    # routes, including changed Decode rows, remain private.
                    kinds = {op.operation_id: op.kind for op in compiled.plan._compiled.model.aggregate.tasks}
                    prefix = []
                    for record in result.raw[0].decisions:
                        if record.decision is None:
                            break
                        if kinds.get(record.decision.target_id) == "router":
                            break
                        prefix.append(record.decision)
                    self.assertTrue(prefix)
                    sequences.append(prefix)
            self.assertEqual(sequences[0], sequences[1], name)


if __name__ == "__main__":
    unittest.main()
