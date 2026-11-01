"""Counterexamples for inclusive runtime composition and causal phase progress."""
import unittest
from dataclasses import replace
from fractions import Fraction

from llm_infer_model.performance.runtime_response import (
    HostPlacement, RuntimeScope, RuntimeStage, RuntimeCell, RuntimeProfile,
    RuntimeProgram, InvalidModel, UnknownParameter)


class RuntimeResponseTests(unittest.TestCase):
    def setUp(self):
        self.evidence = ("hand-calculated unit fixture, not hardware evidence",)
        self.placement = HostPlacement(4, 4, (1,), (1,), "a" * 64)
        self.scope = RuntimeScope("b" * 64, self.placement)
        self.stages = (RuntimeStage("before", "c" * 64, "gap"),
                       RuntimeStage("call", "d" * 64, "api"),
                       RuntimeStage("after", "e" * 64, "gap"))
        self.program = RuntimeProgram(self.scope, self.stages, self.evidence)

    def profile(self, api=5_000_000, gap=0, on_api=None):
        cells = []
        for stage in self.stages:
            for state in ((0, 0), (1, 0), (0, 1), (1, 1)):
                value = gap if stage.role == "gap" else on_api if state[0] and on_api is not None else api
                cells.append(RuntimeCell(stage.coordinate_sha256, stage.role, state, value, self.evidence))
        return RuntimeProfile(self.scope, tuple(cells), self.evidence, qualified=False)

    def predict(self, profile, **changes):
        args = dict(cpu_intervals_ns=(), gpu_intervals_ns=(), provider="memoryless",
                    origin_ns=0, horizon_ns=100_000_000, start_ns=0,
                    replays=1, block_replays=1, require_qualified=False)
        args.update(changes)
        return self.program.predict(profile, **args)

    def test_inclusive_api_does_not_add_internal_dma(self):
        result = self.predict(self.profile())
        # The 5ms response can include 4ms DMA; it is still 5ms, never 9ms.
        self.assertEqual(result["total_ns"], 5_000_000)
        self.assertFalse(result["internal_GPU_or_DMA_time_added"])
        self.assertFalse(result["active_CPU_time_inferred"])
        with self.assertRaises(InvalidModel):
            RuntimeStage("internal", "f" * 64, "physical_gpu")

    def test_external_transition_integrates_own_stage_progress(self):
        # At half speed the first 2ms complete 1/4 call. Remaining 3/4 take
        # 3ms after the actor stops. Entry/exit are predictions, not observations.
        result = self.predict(self.profile(api=4_000_000, on_api=8_000_000),
                              cpu_intervals_ns=((0, 2_000_000),))
        self.assertEqual(result["total_ns"], 5_000_000)
        self.assertFalse(result["future_target_state_used"])
        self.assertFalse(result["measured_target_boundaries_used"])

    def test_intercall_gaps_are_not_dropped(self):
        result = self.predict(self.profile(api=5_000_000, gap=1_000_000))
        self.assertEqual(result["total_ns"], 7_000_000)
        with self.assertRaises(InvalidModel):
            RuntimeProgram(self.scope, (self.stages[1],), self.evidence)

    def test_unknown_and_absent_cells_do_not_fall_back_to_idle(self):
        p = self.profile()
        with self.assertRaises(UnknownParameter):
            self.predict(replace(p, cells=p.cells[:-1]))
        cells = list(p.cells)
        cells[0] = replace(cells[0], mean_ns=None)
        with self.assertRaises(UnknownParameter):
            self.predict(replace(p, cells=tuple(cells)))

    def test_affinity_and_topology_are_not_global_utilization(self):
        disjoint = HostPlacement(4, 16, (1,), (2,), "a" * 64)
        self.assertTrue(self.placement.physical_core_overlap)
        self.assertFalse(disjoint.physical_core_overlap)
        with self.assertRaises(UnknownParameter):
            self.predict(replace(self.profile(), scope=RuntimeScope("b" * 64, disjoint)))
        # Different logical CPUs can be siblings of one physical core.
        self.assertTrue(HostPlacement(4, 8, (1,), (1,), "a" * 64).physical_core_overlap)

    def test_unqualified_profiles_need_explicit_diagnostic_permission(self):
        with self.assertRaises(UnknownParameter):
            self.predict(self.profile(), require_qualified=True)
        with self.assertRaises(UnknownParameter):
            self.predict(replace(self.profile(), scope=replace(self.scope, context_sha256="0" * 64)))

    def test_zero_is_clock_approximation_not_unknown_or_physical_rate(self):
        result = self.predict(self.profile(api=0))
        self.assertEqual(result["total_ns"], 0)
        with self.assertRaises(UnknownParameter):
            self.predict(self.profile(api=0), provider="stateful", transition_ns=1_000_000)

    def test_response_contract_rejects_wrong_units_and_duplicates(self):
        with self.assertRaises(InvalidModel):
            RuntimeScope("b" * 64, self.placement, "thread_cpu_ns")
        for bad in (True, 1.5, -1):
            with self.assertRaises(InvalidModel):
                RuntimeCell("d" * 64, "api", (0, 0), bad, self.evidence)
        with self.assertRaises(InvalidModel):
            p = self.profile()
            RuntimeProfile(self.scope, p.cells + (p.cells[0],), self.evidence)

    def test_exact_rational_cell_is_supported(self):
        self.assertEqual(self.predict(self.profile(api=Fraction(9, 2)))["total_ns"], 5)


if __name__ == "__main__":
    unittest.main()
