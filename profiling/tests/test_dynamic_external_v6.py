import copy
import unittest

from profiling.dynamic_validation.external_v6 import callback_ledger, physical_core_contract


class CallbackLedgerTests(unittest.TestCase):
    def fixture(self):
        phases = {1: dict(begin_tick=0, end_tick=20), 2: dict(begin_tick=20, end_tick=40)}
        names = ["cudaLaunchKernel_v7000", "cudaMemcpyAsync_v3020", "cudaStreamSynchronize_v3020"]
        rows = []
        for phase, offset in ((1, 0), (2, 20)):
            for metadata_index, (a, b) in enumerate(((1, 6), (4, 8), (10, 11))):
                rows.append(dict(ordinal=len(rows), phase_id=phase, metadata_index=metadata_index,
                                 enter_tick=offset+a, exit_tick=offset+b, exit_received=True, return_status=0))
        return dict(enabled=True, all_selected_calls_retained=True, activity_driver_kernel_collection_enabled=False,
                    metadata=[dict(name=name) for name in names], callback_records=rows, record_count=6), phases, {name: 1 for name in names}

    def test_overlapping_callback_spans_use_union_and_close_host_wall(self):
        callbacks, phases, expected = self.fixture()
        result = callback_ledger(callbacks, phases, 10_000_000, expected_counts=expected)
        self.assertEqual(result["records"], 6)
        self.assertEqual(result["phase_partitions"][0]["selected_API_union_ns"], 800)
        self.assertEqual(result["phase_partitions"][0]["outside_selected_API_union_ns"], 1200)
        self.assertFalse(result["GPU_service_or_enqueue_acceptance_identified"])

    def test_missing_or_incomplete_pairs_are_rejected(self):
        for update in (dict(exit_received=False), dict(return_status=None), dict(exit_tick=100)):
            callbacks, phases, expected = self.fixture()
            callbacks["callback_records"][0].update(update)
            with self.assertRaises(ValueError):
                callback_ledger(callbacks, phases, 10_000_000, expected_counts=expected)

    def test_missing_phase_or_API_does_not_become_zero_price(self):
        callbacks, phases, expected = self.fixture()
        callbacks["callback_records"].pop()
        callbacks["record_count"] -= 1
        with self.assertRaises(ValueError):
            callback_ledger(callbacks, phases, 10_000_000, expected_counts=expected)

    def test_control_cannot_contain_observer_rows(self):
        callbacks, phases, expected = self.fixture()
        callbacks["enabled"] = False
        with self.assertRaises(ValueError):
            callback_ledger(callbacks, phases, 10_000_000, expected_counts=expected)
        callbacks.update(metadata=[], callback_records=[], record_count=0)
        self.assertEqual(callback_ledger(callbacks, phases, 10_000_000, expected_counts=expected)["records"], 0)

    def test_overflow_or_record_count_mismatch_rejected(self):
        for changes in (dict(all_selected_calls_retained=False), dict(record_count=8)):
            callbacks, phases, expected = self.fixture()
            callbacks.update(changes)
            with self.assertRaises(ValueError):
                callback_ledger(callbacks, phases, 10_000_000, expected_counts=expected)


class PhysicalTopologyTests(unittest.TestCase):
    def fixture(self):
        return dict(known=True, provider="software_fixture", physical_core_record_count=3,
                    all_core_records=[dict(core_record=i, efficiency_class=0,
                                           group_masks=[dict(group=0, logical_processors=logical)])
                                      for i, logical in enumerate(([2, 3], [4, 5], [16]))])

    def test_distinct_physical_core_required_not_just_distinct_logical(self):
        result = physical_core_contract(self.fixture())
        self.assertTrue(result["CPU2_CPU4_physical_cores_disjoint"])
        self.assertEqual(result["selected"]["4"]["mask"], 16)
        topology = self.fixture()
        topology["all_core_records"][0]["group_masks"][0]["logical_processors"] = [2, 4]
        topology["all_core_records"][1]["group_masks"][0]["logical_processors"] = [5]
        with self.assertRaises(ValueError):
            physical_core_contract(topology)

    def test_unknown_topology_refused(self):
        with self.assertRaises(ValueError):
            physical_core_contract(dict(known=False, reason="unavailable"))


if __name__ == "__main__":
    unittest.main()
