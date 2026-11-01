"""Independent synthetic/mutated evidence, without importing simulation code."""
from copy import deepcopy
import unittest

from experiments.paper_design_independent_audit import (
    AuditError, FIELDS, audit_trace, check_final_goals, check_geometry_and_initial_state,
    check_execution_devices, check_graph_dependencies, check_macro_catalogue, check_matrix, check_selection_analysis, expected_jobs, expected_weights,
)


def finite_matrix(mode="main"):
    rows = [dict(zip(FIELDS, key)) for key in sorted(expected_jobs(mode))]
    return rows, dict(mode=mode, declared_cases=len(rows), jobs=deepcopy(rows))


def weight_catalogue(workload="dense", state="cold"):
    location = "dram" if state == "cold" else "vram"
    tensors = [dict(id=tid, role="weight", storage_bytes=size,
                    allocation_bytes=[["dram", size], ["vram", size]],
                    initial_locations=[location],
                    goal_location=None if workload == "dense" else location)
               for tid, size in expected_weights(workload).items()]
    row = dict(workload=workload, initial_state=state,
               initial_weight_bytes=sum(t["storage_bytes"] for t in tensors))
    return row, tensors


def final_catalogue():
    row = dict(workload="dense", decode_steps=1)
    ids = [f"kv:layer{layer}:request0:{position}" for layer in range(2) for position in range(9)]
    ids.append("g:layer1-decode0:t:layer1:request0:ffn")
    return row, [dict(id=tid, role="output", goal_location="dram", copies=[["dram", "READY"]],
                     storage_bytes=12288 if tid.startswith("kv:") else 24576) for tid in ids]


def physical_trace():
    def event(index, kind, now, end, target=None, **extra):
        result = dict(action_index=index, kind=kind, target_id=target, t_ns=now,
                      start_ns=now, end_ns=end, bytes_transferred=0, bytes_released=0,
                      workspace_bytes=0, resource=None, device=None, src=None, dst=None, location=None)
        result.update(extra)
        return result
    events = [event(0,"COPY",0,2,"w",src="dram",dst="vram",resource="pcie_copy",bytes_transferred=10),
              event(1,"ADVANCE",2,2),
              event(2,"COMPUTE",2,12,"op",device="gpu",resource="gpu_compute",workspace_bytes=3),
              event(3,"ADVANCE",12,12),
              event(4,"COPY",12,14,"out",src="vram",dst="dram",resource="pcie_copy",bytes_transferred=4),
              event(5,"ADVANCE",14,14)]
    tensors = []
    for tid, size, initial, goal in (("w",10,["dram"],None),("out",4,[],"dram")):
        tensors.append(dict(id=tid, storage_bytes=size, allocation_bytes=[["dram",size],["vram",size]],
            initial_locations=initial, goal_location=goal,
            copies=[["dram","READY"],["vram","READY"],["ssd","ABSENT"]],
            transfers=[dict(src="dram",dst="vram",duration_ns=2),dict(src="vram",dst="dram",duration_ns=2)]))
    ops = [dict(id="op",inputs=["w"],outputs=["out"],prices=[dict(device="gpu",duration_ns=10,allocated_workspace_bytes=3)])]
    summary = dict(success=True,completed_makespan_ns=14,peak_dram_bytes=14,peak_vram_bytes=17,
        h2d_bytes=10,d2h_bytes=4,ssd_read_bytes=0,ssd_write_bytes=0,event_counts=dict(COPY=2,ADVANCE=3,COMPUTE=1),
        cpu_event_span_ns=0,gpu_event_span_ns=10,copy_event_span_ns=4,compute_event_span_ns=10,
        cpu_gpu_event_overlap_ns=0,copy_compute_event_overlap_ns=0,any_engine_event_span_ns=14,all_engine_event_uncovered_ns=0)
    hw = dict(memory_mode="separate",dram_capacity_bytes=100,vram_capacity_bytes=100)
    return events, ops, tensors, summary, hw


class IndependentDesignAuditTests(unittest.TestCase):
    def test_exact_denominators_and_geometry(self):
        for mode, count in (("pilot",6),("main",504),("long",112)):
            rows, protocol = finite_matrix(mode)
            self.assertEqual(check_matrix(rows, protocol), count)
        self.assertEqual(sum(expected_weights("dense").values()),3170992128)
        self.assertEqual(sum(expected_weights("moe_hot").values()),10418847744)

    def test_rejects_missing_and_duplicate_candidates(self):
        rows, protocol = finite_matrix()
        with self.assertRaisesRegex(AuditError,"missing or undeclared"):
            check_matrix(rows[:-1], protocol)
        with self.assertRaisesRegex(AuditError,"duplicate"):
            check_matrix(rows+[rows[0]], protocol)

    def test_initial_state_has_no_free_host_copy(self):
        row, tensors = weight_catalogue("moe_hot","gpu_resident")
        self.assertEqual(check_geometry_and_initial_state(row,tensors),12)
        tensors[0]["initial_locations"].append("dram")
        with self.assertRaisesRegex(AuditError,"free second copy"):
            check_geometry_and_initial_state(row,tensors)

    def test_rejects_weight_undercharge(self):
        row, tensors = weight_catalogue()
        tensors[0]["storage_bytes"] //= 2
        with self.assertRaisesRegex(AuditError,"F32 geometry"):
            check_geometry_and_initial_state(row,tensors)

    def test_goal_oracle_rejects_missing_kv_and_false_dram_readiness(self):
        row, tensors = final_catalogue()
        self.assertEqual(check_final_goals(row,tensors),19)
        with self.assertRaisesRegex(AuditError,"KV block coverage"):
            check_final_goals(row,tensors[1:])
        tensors[0]["copies"] = [["dram","ABSENT"]]
        with self.assertRaisesRegex(AuditError,"not ready"):
            check_final_goals(row,tensors)

    def test_dependency_check_rejects_consumer_before_producer(self):
        ops = [dict(id="a",inputs=["root"],outputs=["mid"]),dict(id="b",inputs=["mid"],outputs=["out"])]
        tensors = [dict(id="root",initial_locations=["dram"]),dict(id="mid",initial_locations=[]),dict(id="out",initial_locations=[])]
        events = [dict(kind="COMPUTE",target_id="a",start_ns=0,end_ns=10),dict(kind="COMPUTE",target_id="b",start_ns=10,end_ns=11)]
        self.assertEqual(check_graph_dependencies(ops,tensors,events),1)
        events[1]["start_ns"] = 9
        with self.assertRaisesRegex(AuditError,"before producer"):
            check_graph_dependencies(ops,tensors,events)

    def test_macro_oracle_rejects_dropped_stage_and_wrong_token_count(self):
        row = dict(workload="dense",decode_steps=1)
        ops = [dict(id="g:decode0:trace_next:request0",kind="state_update",token_count=1)]
        for round_id,tokens in (("prefill",8),("decode0",1)):
            for layer in range(2):
                for stage,kind in (("proj","projection"),("attn","attention"),("ffn","dense_ffn")):
                    ops.append(dict(id=f"g:layer{layer}-{round_id}:op:layer{layer}:request0:{stage}",kind=kind,token_count=tokens))
        self.assertEqual(check_macro_catalogue(row,ops),13)
        with self.assertRaisesRegex(AuditError,"coverage differs"):
            check_macro_catalogue(row,ops[:-1])
        ops[-1]["token_count"] = 8
        with self.assertRaisesRegex(AuditError,"coverage differs"):
            check_macro_catalogue(row,ops)

    def test_replay_accounts_workspace_and_physical_copy(self):
        result = audit_trace(*physical_trace())
        self.assertEqual(result["replayed_physical_peaks_bytes"],dict(dram=14,vram=17))
        self.assertEqual(result["copy_events"],2)

    def test_execution_labels_match_actual_devices(self):
        ops = [dict(id="attention",kind="attention"),dict(id="ffn",kind="dense_ffn")]
        events = [dict(kind="COMPUTE",target_id="attention",device="gpu"),dict(kind="COMPUTE",target_id="ffn",device="cpu")]
        self.assertEqual(check_execution_devices(dict(candidate="e2-s1"),ops,events),2)
        with self.assertRaisesRegex(AuditError,"candidate placement"):
            check_execution_devices(dict(candidate="e0-s1"),ops,events)

    def test_analysis_rejects_underreported_joint_evaluations(self):
        rows = [dict(workload="dense",decode_steps=1,initial_state="cold",vram_gib=12,link_gbs=8,
                     candidate=candidate,success=True,completed_makespan_ns=latency)
                for candidate,latency in (("e0-s0",10),("e0-s1",8),("e1-s0",9),("e1-s1",7))]
        choices = {}
        for method,candidate,latency,evaluations in (("joint","e1-s1",7,4),("execution_only_s0","e1-s0",9,2),
              ("storage_only_e0","e0-s1",8,2),("stagewise_execution_then_storage","e1-s1",7,3)):
            choices[method] = dict(candidate=candidate,latency_ns=latency,regret_vs_finite_best=latency/7-1,evaluations=evaluations)
        analysis = dict(groups=[dict(group="dense-d1-cold-v12-b8",feasible=True,choices=choices,tied_best=["e1-s1"])])
        self.assertEqual(check_selection_analysis(rows,analysis),4)
        choices["joint"]["evaluations"] = 1
        with self.assertRaisesRegex(AuditError,"evaluation budget"):
            check_selection_analysis(rows,analysis)

    def test_replay_rejects_missing_copy_source(self):
        values = physical_trace()
        values[2][0]["initial_locations"] = []
        with self.assertRaisesRegex(AuditError,"COPY readiness"):
            audit_trace(*values)

    def test_replay_rejects_compute_before_copy_completes(self):
        values = physical_trace()
        values[0][1].update(t_ns=1,start_ns=1,end_ns=1)
        values[0][2].update(t_ns=1,start_ns=1,end_ns=11)
        with self.assertRaisesRegex(AuditError,"unready input"):
            audit_trace(*values)

    def test_replay_rejects_understated_peak_and_payload(self):
        values = physical_trace()
        values[3]["peak_vram_bytes"] = 14
        with self.assertRaisesRegex(AuditError,"memory peak"):
            audit_trace(*values)
        values = physical_trace()
        values[0][0]["bytes_transferred"] = 1
        with self.assertRaisesRegex(AuditError,"COPY payload"):
            audit_trace(*values)


if __name__ == "__main__":
    unittest.main()
