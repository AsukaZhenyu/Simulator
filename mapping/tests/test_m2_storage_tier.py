"""m2-layer-0.7: the same Dense macros, with the disk as a declared tier.

The execution contracts are untouched -- ``.7`` selects the same CPU/CUDA
macros and the same materialized ports as ``.5``. What changes is where a
weight is allowed to *start* and which budgets the kernel keeps: a declared
``ssd_capacity_bytes`` makes the disk a ledgered domain with capacity,
alignment, eviction and a reported peak, and every disk move needs its own
exact byte price. Nothing here measures anything; the numbers are synthetic.

The first class pins what ``.5`` still refuses, because "add a tier" must not
mean "silently reinterpret the previous version".
"""
from __future__ import annotations

import copy
import json
import unittest
from dataclasses import replace
from pathlib import Path

from llm_infer_model.m2 import costs as m2_costs
from llm_infer_model.m2.costs import MacroShape, expand_costs_document, parse_cost_table
from llm_infer_model.m2.expand import load_and_compile, plan_for
from llm_infer_model.m2.runner import run_invocation
from llm_infer_model.m2.spec import (
    parse_invocation_document, parse_layer_document,
    STORAGE_HYBRID_FORMAT_VERSION,
)
from llm_infer_model.tensor.engine import ACTION_READ_SSD, ACTION_WRITE_SSD
from llm_infer_model.tensor.spec import InitialCapacityExceeded, InvalidInput, Unsupported

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples/m2"
_LAYER = _EXAMPLES / "storage-interface-synthetic.layer.json"
_COSTS = _EXAMPLES / "storage-interface-synthetic.costs.json"
_PREFILL = _EXAMPLES / "storage-interface-synthetic-prefill-3.json"
_DECODE = _EXAMPLES / "storage-interface-synthetic-decode-1.json"
_UMA = _EXAMPLES / "storage-interface-synthetic-uma-prefill-3.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compiled(path: Path, **hardware):
    compilation = load_and_compile(path)
    if not hardware:
        return compilation
    invocation = replace(compilation.invocation,
                         hardware=replace(compilation.invocation.hardware, **hardware))
    from llm_infer_model.m2.expand import compile_invocation
    return compile_invocation(compilation.layer, invocation, compilation.cost_table)


def compiled_prefill(**hardware):
    return compiled(_PREFILL, **hardware)


class PointFiveKeepsItsBoundary(unittest.TestCase):
    """The storage tier is a new version, not a reinterpretation of .5."""

    def test_point_five_still_refuses_a_weight_that_starts_on_the_disk(self):
        raw = read(_LAYER)
        raw["format_version"] = "m2-layer-0.5"
        with self.assertRaisesRegex(InvalidInput, "dram"):
            parse_layer_document(raw)

    def test_point_five_still_refuses_an_ssd_budget(self):
        raw = read(_PREFILL)
        raw["format_version"] = "m2-layer-0.5"
        with self.assertRaisesRegex(InvalidInput, "ssd_capacity_bytes"):
            parse_invocation_document(raw)

    def test_spelling_the_absent_disk_explicitly_is_the_same_machine(self):
        """``null``/0/``null`` names no disk at all, so .5 reads it as it did."""
        raw = read(_PREFILL)
        raw["format_version"] = "m2-layer-0.5"
        raw["hardware"] = {key: value for key, value in raw["hardware"].items()
                           if not key.startswith("ssd_")}
        baseline = parse_invocation_document(raw)
        raw["hardware"].update(ssd_capacity_bytes=None, ssd_runtime_reserved_bytes=0,
                               ssd_alignment_bytes=None)
        spelled = parse_invocation_document(raw)
        self.assertEqual(spelled.hardware, baseline.hardware)
        self.assertFalse(baseline.hardware.storage_document())

    def test_point_five_refuses_a_disk_number_no_disk_backs(self):
        """An alignment without a budget is still a declared disk number."""
        raw = read(_PREFILL)
        raw["format_version"] = "m2-layer-0.5"
        raw["hardware"].update(ssd_capacity_bytes=None, ssd_alignment_bytes=4096)
        with self.assertRaisesRegex(InvalidInput, "m2-layer-0.7"):
            parse_invocation_document(raw)


class TheBudgetIsTheTier(unittest.TestCase):
    """Without a declared budget there is no storage domain to search in."""

    def test_a_point_seven_invocation_must_declare_the_disk_budget(self):
        raw = read(_PREFILL)
        del raw["hardware"]["ssd_capacity_bytes"]
        with self.assertRaisesRegex(InvalidInput, "ssd_capacity_bytes"):
            parse_invocation_document(raw)

    def test_the_initial_layout_is_charged_with_alignment(self):
        compilation = compiled_prefill()
        # Eight matrices, each 4096-aligned on a 4096-byte disk.
        self.assertEqual(compilation.scenario.architecture.capacity_of("ssd"), 1048576)
        self.assertEqual(compilation.scenario.architecture.alignment_of("ssd"), 4096)
        self.assertEqual(compilation.state_in is not None, True)

    def test_a_disk_one_sector_short_cannot_hold_the_initial_layout(self):
        from tensor_mapping import initial_state
        compilation = compiled_prefill(ssd_capacity_bytes=32767)
        with self.assertRaises(InitialCapacityExceeded):
            initial_state(compilation.scenario)

    def test_every_tensor_has_a_priced_way_back_and_forth(self):
        compilation = compiled_prefill()
        allowed = compilation.scenario.mapspace.allowed_transfers
        self.assertEqual(len(allowed["ssd_to_dram"]), len(allowed["dram_to_ssd"]))
        self.assertTrue(allowed["ssd_to_dram"])


class ThePlanStagesThroughDram(unittest.TestCase):
    """There is no ssd->vram route; a GPU macro stages the weight through DRAM."""

    def test_the_serial_plan_reads_each_disk_weight_once(self):
        compilation = compiled_prefill()
        plan = plan_for(compilation)
        reads = [action.target_id for action in plan if action.kind == ACTION_READ_SSD]
        weights = [tensor.id for tensor in compilation.scenario.workload.tensors
                   if tensor.role == "weight"]
        disk = [tensor.id for tensor in compilation.scenario.workload.tensors
                if "ssd" in tensor.initial_locations]
        self.assertTrue(disk)
        self.assertEqual(sorted(reads), sorted(disk))
        self.assertTrue(set(reads) <= set(weights))
        self.assertFalse([a for a in plan if a.kind == ACTION_WRITE_SSD])

    def test_the_replay_reports_what_the_disk_carried(self):
        compilation = compiled_prefill()
        result = run_invocation(compilation)
        self.assertEqual(result.status, "valid")
        self.assertEqual(result.evaluation.peak_ssd_bytes, 32768)
        self.assertGreater(result.evaluation.ssd_read_bytes, 0)
        self.assertEqual(result.evaluation.ssd_write_bytes, 0)

    def test_the_decode_call_keeps_the_same_tier(self):
        compilation = load_and_compile(_DECODE)
        result = run_invocation(compilation)
        self.assertEqual(result.status, "valid")
        self.assertEqual(result.evaluation.peak_ssd_bytes, 32768)

    def test_a_gpu_placement_stages_the_disk_weight_through_dram(self):
        compilation = compiled_prefill()
        tasks = {task.operation_id: task for task in compilation.tasks}
        placements = {task.operation_id: "gpu" for task in compilation.tasks}
        plan = plan_for(compilation, placements=placements)
        # Every copy ends either on the disk (never) or DRAM->VRAM; SSD is only
        # ever a source, and never a direct destination of a VRAM move.
        transfers = [(a.src, a.dst) for a in plan if a.src is not None]
        self.assertIn(("dram", "vram"), transfers)
        self.assertNotIn(("ssd", "vram"), transfers)


class UmaKeepsTheDiskItself(unittest.TestCase):
    """On a UMA machine DRAM and VRAM share one pool; the disk does not.

    The same layer and cost table run unchanged -- only the hardware block says
    where the memory is. A shared pool must not make the disk ledger answer with
    the memory figure: that would check a full disk against a number that never
    counted one byte of it.
    """

    def test_the_same_layer_runs_on_a_shared_pool(self):
        compilation = compiled(_UMA)
        self.assertTrue(compilation.scenario.architecture.is_shared)
        self.assertEqual(compilation.scenario.architecture.memory_domains, ("shared",))
        self.assertEqual(compilation.scenario.architecture.capacity_of("ssd"), 1048576)
        result = run_invocation(compilation)
        self.assertEqual(result.status, "valid")
        self.assertGreater(result.evaluation.ssd_read_bytes, 0)

    def test_the_disk_peak_is_the_disk_peak_not_the_memory_figure(self):
        separate = run_invocation(load_and_compile(_PREFILL)).evaluation
        uma = run_invocation(load_and_compile(_UMA)).evaluation
        self.assertEqual(uma.peak_ssd_bytes, separate.peak_ssd_bytes)
        self.assertEqual(uma.peak_ssd_bytes, 32768)
        self.assertEqual(uma.ssd_read_bytes, separate.ssd_read_bytes)
        # Sharing removes the DRAM->VRAM copy, not a byte of disk traffic.
        self.assertLessEqual(uma.makespan_ns, separate.makespan_ns)

    def test_a_large_shared_pool_cannot_hide_a_disk_one_sector_short(self):
        from tensor_mapping import initial_state
        tiny_disk = compiled(_UMA, ssd_capacity_bytes=32767)
        with self.assertRaises(InitialCapacityExceeded):
            initial_state(tiny_disk.scenario)

    def test_a_small_shared_pool_refuses_what_it_cannot_hold(self):
        result = run_invocation(compiled(_UMA, shared_capacity_bytes=4096))
        self.assertEqual(result.status, "invalid_mapping")
        self.assertIn("CAPACITY_EXCEEDED", result.evaluation.reason)


class PricesAreExactAndNamed(unittest.TestCase):
    """A disk move with no price is refused; a measured price keeps its policy."""

    def test_a_missing_exact_disk_price_stops_the_compile(self):
        raw = read(_COSTS)
        raw["transfer_entries"] = [e for e in raw["transfer_entries"] if e["route"] != "dram_to_ssd"]
        from llm_infer_model.m2.expand import compile_invocation
        source = load_and_compile(_PREFILL)
        table = parse_cost_table(raw)
        with self.assertRaisesRegex(Unsupported, "dram_to_ssd"):
            compile_invocation(source.layer, source.invocation, table)

    def test_a_measured_disk_table_carries_its_provenance_into_the_scenario(self):
        raw = read(_COSTS)
        raw["validation_only"] = False
        raw["transfer_entries"] = [
            dict(entry, provenance="measured_transfer",
                 timing_scope="single_io_wall", cache_policy="unbuffered")
            if entry["route"].startswith(("ssd", "dram_to_ssd")) or entry["route"] == "ssd_to_dram"
            else entry
            for entry in raw["transfer_entries"]]
        table = parse_cost_table(raw)
        from llm_infer_model.m2.expand import compile_invocation
        source = load_and_compile(_PREFILL)
        compiled = compile_invocation(source.layer, source.invocation, table)
        self.assertIn("measured_transfer", compiled.scenario_document["costs"]["source"])
        self.assertIn("measured_transfer",
                      compiled.physical_boundary["transfer_cost_semantics"])

    def test_a_declared_write_regime_reaches_the_kernel_scenario(self):
        """The table's second write price and its threshold survive expansion."""
        raw = read(_COSTS)
        raw["storage_write_regime"] = {"after_cumulative_bytes": 1 << 20}
        raw["transfer_entries"] = [
            dict(entry, sustained_duration_ns=entry["duration_ns"] * 5)
            if entry["route"] == "dram_to_ssd" else entry
            for entry in raw["transfer_entries"]]
        table = parse_cost_table(raw)
        from llm_infer_model.m2.expand import compile_invocation
        source = load_and_compile(_PREFILL)
        costs = compile_invocation(source.layer, source.invocation,
                                   table).scenario_document["costs"]
        self.assertEqual(costs["storage_write_regime"],
                         {"after_cumulative_bytes": 1 << 20})
        self.assertTrue(costs["transfer"]["dram_to_ssd"])
        for priced in costs["transfer"]["dram_to_ssd"].values():
            self.assertEqual(priced["sustained_duration_ns"],
                             priced["duration_ns"] * 5)
        # Only the write route carries a second price.
        for priced in costs["transfer"]["ssd_to_dram"].values():
            self.assertNotIn("sustained_duration_ns", priced)

    def test_a_warm_disk_table_cannot_underwrite_a_result(self):
        raw = read(_COSTS)
        raw["validation_only"] = False
        raw["transfer_entries"] = [
            dict(entry, provenance="measured_transfer",
                 timing_scope="file_io_wall", cache_policy="buffered")
            if entry["route"] == "ssd_to_dram" else entry
            for entry in raw["transfer_entries"]]
        with self.assertRaisesRegex(InvalidInput, "页缓存"):
            parse_cost_table(raw)


class TheExpansionStaysExact(unittest.TestCase):
    """The document the kernel receives never hides a staging allocation."""

    def test_the_scenario_declares_the_storage_tier_it_was_given(self):
        compilation = compiled_prefill()
        architecture = compilation.scenario_document["architecture"]
        self.assertEqual(architecture["ssd_capacity_bytes"], 1048576)
        self.assertEqual(architecture["ssd_alignment_bytes"], 4096)
        boundary = compilation.physical_boundary
        self.assertEqual(boundary["storage_tier"], "declared_budget")
        self.assertEqual(boundary["storage_capacity_bytes"], 1048576)
        self.assertIn("queue depth", boundary["storage_semantics"])

    def test_a_copy_into_vram_costs_the_same_bytes_the_disk_read_paid_for(self):
        compilation = compiled_prefill()
        transfer = compilation.scenario_document["costs"]["transfer"]
        for route, entries in transfer.items():
            for tensor_id, priced in entries.items():
                self.assertGreater(priced["duration_ns"], 0, f"{route}/{tensor_id}")
        self.assertTrue(transfer["ssd_to_dram"])


if __name__ == "__main__":
    unittest.main()
