"""Hand-built traces exercise the independent audit and its rejection paths."""
from copy import deepcopy
import unittest

from experiments.hardware_system_audit import AuditError, audit_trace, ceil_scaled


def shared_fixture():
    def event(kind, target=None, **extra):
        return dict(kind=kind, target_id=target, resource=None, t_ns=0,
            start_ns=0, end_ns=0, bytes_transferred=0, workspace_bytes=0,
            bytes_released=0, device=None, src=None, dst=None, location=None, **extra)
    events = [event("SHARE", "w"), event("COMPUTE", "op"), event("ADVANCE"),
              event("SHARE", "out"), event("EVICT", "w"), event("EVICT", "out")]
    events[0].update(src="dram", dst="vram")
    events[1].update(device="gpu", resource="gpu_compute", end_ns=10)
    events[2].update(t_ns=10, start_ns=None, end_ns=10)
    events[3].update(src="vram", dst="dram", t_ns=10, start_ns=10, end_ns=10)
    for index in (4, 5):
        events[index].update(location="vram", t_ns=10, start_ns=10, end_ns=10)
    for index, row in enumerate(events):
        row["action_index"] = index
    tensors = [dict(id="w", allocation_bytes=[["dram", 10], ["vram", 10]],
                    storage_bytes=10, initial_locations=["dram"], goal_location=None,
                    copies=[["dram", "READY"], ["vram", "ABSENT"], ["ssd", "ABSENT"]],
                    allocation_refs=[["dram", "a0"], ["vram", None]], transfers=[]),
               dict(id="out", allocation_bytes=[["dram", 4], ["vram", 4]],
                    storage_bytes=4, initial_locations=[], goal_location="dram",
                    copies=[["dram", "READY"], ["vram", "ABSENT"], ["ssd", "ABSENT"]],
                    allocation_refs=[["dram", "a1"], ["vram", None]], transfers=[])]
    operations = [dict(id="op", inputs=["w"], outputs=["out"], status="DONE",
                       prices=[dict(device="gpu", duration_ns=10, allocated_workspace_bytes=0)])]
    hardware = dict(memory_mode="shared", shared_capacity_bytes=100,
                    dram_capacity_bytes=100, vram_capacity_bytes=100,
                    shared_runtime_reserved_bytes=0)
    summary = dict(success=True, completed_makespan_ns=10, peak_shared_bytes=14,
        peak_dram_bytes=14, peak_vram_bytes=14, physical_memory_peaks_bytes={"shared": 14},
        h2d_bytes=0, d2h_bytes=0, ssd_read_bytes=0, ssd_write_bytes=0,
        event_counts=dict(SHARE=2, COMPUTE=1, ADVANCE=1, EVICT=2),
        cpu_event_span_ns=0, gpu_event_span_ns=10, copy_event_span_ns=0,
        compute_event_span_ns=10, cpu_gpu_event_overlap_ns=0,
        copy_compute_event_overlap_ns=0, any_engine_event_span_ns=10,
        all_engine_event_uncovered_ns=0)
    return events, operations, tensors, summary, hardware


class HardwareSystemAuditTests(unittest.TestCase):
    def test_shared_aliases_consume_one_allocation_and_drop_zero_bytes(self):
        result = audit_trace(*shared_fixture())
        self.assertEqual(result["replayed_physical_peaks_bytes"], {"shared": 14})
        self.assertEqual(result["share_events"], 2)
        self.assertEqual(result["zero_release_events"], 2)

    def test_rejects_double_counted_shared_peak(self):
        values = shared_fixture()
        values[3].update(peak_shared_bytes=28, peak_dram_bytes=28, peak_vram_bytes=28)
        with self.assertRaisesRegex(AuditError, "shared peak"):
            audit_trace(*values)

    def test_rejects_phantom_share_cost_or_copy(self):
        for field in ("bytes_transferred", "workspace_bytes", "end_ns"):
            values = shared_fixture()
            values[0][0][field] = 1
            with self.subTest(field=field), self.assertRaisesRegex(AuditError, "SHARE has physical"):
                audit_trace(*values)

    def test_rejects_compute_before_input_visibility(self):
        values = shared_fixture()
        values[0][0], values[0][1] = values[0][1], values[0][0]
        for index, event in enumerate(values[0]):
            event["action_index"] = index
        with self.assertRaisesRegex(AuditError, "unready input"):
            audit_trace(*values)

    def test_rejects_alias_eviction_while_reader_running(self):
        values = shared_fixture()
        eviction = values[0].pop(4)
        eviction.update(t_ns=0, start_ns=0, end_ns=0)
        values[0].insert(2, eviction)
        for index, event in enumerate(values[0]):
            event["action_index"] = index
        with self.assertRaisesRegex(AuditError, "physical reader"):
            audit_trace(*values)

    def test_rejects_false_release_and_excess_workspace(self):
        values = shared_fixture()
        values[0][4]["bytes_released"] = 10
        with self.assertRaisesRegex(AuditError, "double-counts"):
            audit_trace(*values)
        values = shared_fixture()
        values[4]["shared_capacity_bytes"] = 12
        values[4]["dram_capacity_bytes"] = values[4]["vram_capacity_bytes"] = 12
        with self.assertRaisesRegex(AuditError, "capacity exceeded"):
            audit_trace(*values)

    def test_exact_integer_proxy_ceiling_is_independent(self):
        self.assertEqual(ceil_scaled(2**60 + 1, 0.5), 2**59 + 1)
        self.assertEqual(ceil_scaled(101, 2), 202)
        with self.assertRaises(AuditError):
            ceil_scaled(0, 1)


if __name__ == "__main__":
    unittest.main()
