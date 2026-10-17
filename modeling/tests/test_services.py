"""Hand-calculated slots, CE/SM service and common-ledger constraints."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import unittest

from llm_infer_model.tensor.engine import Action, CODE_RESOURCE_BUSY, CODE_TENSOR_IN_USE
from llm_infer_model.tensor.environment import (EnvironmentEvent, CODE_EXTERNAL_MEMORY_CONFLICT,
    CODE_INFORMATION_NOT_KNOWN)
from llm_infer_model.tensor.services import (CODE_SLOT_OCCUPIED, ServiceSession,
    parse_services, services_document, allocated_scenario)
from llm_infer_model.tensor.spec import InvalidInput, Unsupported, COPY_READY, OP_DONE, build_scenario


def scene(payloads=(64, 16), *, capacity=256, initial=None, overlap=True, gpu_duration=10):
    tensors, operations, compute = [], [], {}
    for i, size in enumerate(payloads):
        ne = [size // 4, 1, 1, 1]; nb = [4, size, size, size]
        tensors.extend([dict(id=f"x{i}", name=f"x{i}", role="input", dtype="GGML_TYPE_F32",
            ne=ne, nb=nb, storage_bytes=size,
            initial_locations=(initial[i] if initial is not None else ["dram"])),
            dict(id=f"y{i}", name=f"y{i}", role="output", dtype="GGML_TYPE_F32", ne=ne, nb=nb,
                 storage_bytes=size, initial_locations=[])])
        operations.append(dict(id=f"op{i}", semantic_op="RELU", ggml_op="GGML_OP_UNARY",
            unary_op="GGML_UNARY_OP_RELU", op_params=[], inputs=[f"x{i}"], output=f"y{i}"))
        compute[f"op{i}"] = {"cpu": dict(duration_ns=gpu_duration, workspace_bytes=4),
                              "gpu": dict(duration_ns=gpu_duration, workspace_bytes=4)}
    ids = [t["id"] for t in tensors]
    routes = ("dram_to_vram", "vram_to_dram", "ssd_to_dram", "dram_to_ssd")
    workload = dict(schema_version="0.2", id="services-workload",
        origin=dict(kind="synthetic_fixture", ggml_version=None, sample=None),
        tensors=tensors, operations=operations, outputs=[f"y{i}" for i in range(len(payloads))])
    document = dict(schema_version="0.2", id="services-scene",
        architecture=dict(dram_capacity_bytes=capacity, vram_capacity_bytes=capacity,
            dram_runtime_reserved_bytes=0, vram_runtime_reserved_bytes=0,
            dram_alignment_bytes=1, vram_alignment_bytes=1, cpu_compute_slots=1,
            gpu_compute_slots=1, pcie_copy_slots=1, ssd_io_slots=1),
        costs=dict(source="synthetic_services", compute=compute,
            transfer={route: {tid: dict(duration_ns=12345) for tid in ids} for route in routes}),
        mapspace=dict(allowed_devices={f"op{i}": ["cpu", "gpu"] for i in range(len(payloads))},
            allowed_transfers={route: ids for route in routes}, allow_eviction=True,
            allow_copy_compute_overlap=overlap, allow_cpu_gpu_overlap=True, allow_recomputation=False),
        goal=dict(output_locations={f"y{i}": "vram" for i in range(len(payloads))}),
        mapper=dict(algorithm="uniform_cost", max_expanded_states=0, wall_time_limit_s=0))
    return build_scenario(workload, document)


def document(*, slots=None, sm_cap=10**9):
    return dict(format_version="tensor-services-0.1", pcie=dict(bandwidth_bytes_per_second=10**9,
        ce_bytes_per_second=10**9, sm_bytes_per_second=sm_cap, arbitration="max_min_fair",
        timing_scope="payload_only_no_launch_latency"), slots=[] if slots is None else slots)


def runtime(*, slots=None, payloads=(64,16), **kwargs):
    return ServiceSession(scene(payloads, **kwargs), parse_services(document(slots=slots)))


def copy(i, src="dram", dst="vram"):
    return Action("COPY", f"x{i}", src=src, dst=dst)


def frac(document):
    return Fraction(document["numerator"], document["denominator"])


class ServicesContractTests(unittest.TestCase):
    def test_roundtrip_and_strict_positive_explicit_ack(self):
        value=document(); self.assertEqual(services_document(parse_services(value)),value)
        for field, invalid in (("ce_bytes_per_second",0),("sm_bytes_per_second",True),
                               ("bandwidth_bytes_per_second",1.2),("arbitration","independent"),
                               ("timing_scope","inferred_launch_overhead")):
            changed=deepcopy(value); changed["pcie"][field]=invalid
            with self.subTest(field=field),self.assertRaises((InvalidInput,Unsupported)):parse_services(changed)
        changed=deepcopy(value); changed["unknown"]=1
        with self.assertRaises(InvalidInput):parse_services(changed)

    def test_slots_reject_unknown_members_duplicates_size_and_initial_multiple_owners(self):
        base=dict(id="slot",domain="vram",capacity_bytes=64,members=["x0","x1"])
        for change in (dict(domain="ssd"),dict(members=["x0","x0"]),dict(capacity_bytes=True)):
            obj=deepcopy(base); obj.update(change)
            with self.assertRaises((InvalidInput,Unsupported)):parse_services(document(slots=[obj]))
        for change in (dict(members=["missing"]),dict(capacity_bytes=16)):
            obj=deepcopy(base); obj.update(change)
            with self.assertRaises(InvalidInput):allocated_scenario(scene(),parse_services(document(slots=[obj])))
        with self.assertRaises(InvalidInput):allocated_scenario(scene(initial=[["dram","vram"],["dram","vram"]]),parse_services(document(slots=[base])))
        with self.assertRaises(InvalidInput):parse_services(document(slots=[base,{**base,"id":"other"}]))

    def test_direct_typed_invalid_contract_does_not_bypass_parser(self):
        typed=replace(parse_services(document()),ce_bytes_per_second=True)
        with self.assertRaises(InvalidInput):ServiceSession(scene(),typed)


class FixedSlotTests(unittest.TestCase):
    SLOT = dict(id="promotions",domain="vram",capacity_bytes=64,members=["x0","x1"])

    def test_slot_empty_still_reserves_capacity_payload_unchanged(self):
        session=runtime(slots=[self.SLOT])
        memory=next(m for m in session.observe().memory if m.domain=="vram")
        self.assertEqual((memory.used_bytes,memory.runtime_reserved_bytes,memory.tensor_bytes),(64,64,0))
        self.assertEqual(session.scenario.size_alloc("x0","vram"),0)
        self.assertEqual(session.scenario.size_alloc("x0","dram"),64)
        self.assertEqual(session.scenario.workload.tensor_by_id["x0"].storage_bytes,64)
        self.assertTrue(session.submit_transfer(copy(0),engine="ce").ok)
        self.assertEqual(session.result().evaluation.events[0].bytes_transferred,64)
        session.advance()
        self.assertEqual(session.service_document()["slots"][0]["owner"],"x0")
        session.submit(Action("EVICT","x0",location="vram"))
        self.assertEqual(session.service_document()["slots"][0]["owner"],None)
        self.assertEqual(next(m.used_bytes for m in session.observe().memory if m.domain=="vram"),64)
        self.assertEqual(session.result().evaluation.events[-1].bytes_released,0)

    def test_reserved_and_ready_owner_forbid_overwrite(self):
        for complete in (False,True):
            session=runtime(slots=[self.SLOT]); session.submit_transfer(copy(0),engine="ce")
            if complete:session.advance()
            failed=session.submit_transfer(copy(1),engine="sm")
            self.assertFalse(failed.ok); self.assertEqual(failed.error_code,CODE_SLOT_OCCUPIED)
            self.assertEqual(session.state.copy_at("x1","vram"),"ABSENT")

    def test_last_reader_done_and_explicit_evict_before_slot_reuse(self):
        session=runtime(slots=[self.SLOT]); session.submit_transfer(copy(0),engine="ce");session.advance()
        session.submit(Action("COMPUTE","op0",device="gpu"));session.advance()
        self.assertEqual(session.state.op("op0"),OP_DONE)
        self.assertTrue(session.submit(Action("EVICT","x0",location="vram")).ok)
        self.assertTrue(session.submit_transfer(copy(1),engine="ce").ok)
        session.advance();session.submit(Action("COMPUTE","op1",device="gpu"));session.advance()
        self.assertTrue(session.finished)
        self.assertEqual(session.result().evaluation.peak_vram_bytes,148)

    def test_launch_does_not_end_read_lock(self):
        session=runtime(slots=[self.SLOT]);session.submit_transfer(copy(0),engine="ce");session.advance()
        session.submit(Action("COMPUTE","op0",device="gpu"))
        failed=session.submit(Action("EVICT","x0",location="vram"))
        self.assertEqual(failed.error_code,CODE_TENSOR_IN_USE)

    def test_external_memory_conflict_counts_empty_slot(self):
        session=ServiceSession(scene(capacity=128),parse_services(document(slots=[self.SLOT])),
                               environment_events=[EnvironmentEvent(0,vram_external_bytes=65)])
        self.assertEqual(session.error_code,CODE_EXTERNAL_MEMORY_CONFLICT)
        self.assertEqual(session.result().environment_events[0].memory[1].runtime_reserved_bytes,64)

    def test_initial_single_owner_is_charged_once_not_as_payload_plus_slot(self):
        session=runtime(slots=[self.SLOT],capacity=80,initial=[["dram","vram"],["dram"]])
        self.assertEqual(next(m.used_bytes for m in session.observe().memory if m.domain=="vram"),64)
        self.assertEqual(session.service_document()["slots"][0]["owner"],"x0")

    def test_occupied_slot_rejection_preserves_copy_ledger_and_transfer_counts(self):
        session=runtime(slots=[self.SLOT]);session.submit_transfer(copy(0),engine="ce");session.advance()
        before=session.state
        session.submit_transfer(copy(1),engine="sm",source_index=3)
        self.assertEqual(session.state,before)
        result=session.result()
        self.assertEqual(result.evaluation.h2d_bytes,64)
        self.assertEqual(result.action_sources,(None,None,3))
        self.assertEqual(len(session.service_document()["transfer_engines"]),1)

    def test_compute_output_cannot_overwrite_an_initial_slot_owner(self):
        slot={**self.SLOT,"members":["x0","y1"]}
        session=runtime(slots=[slot],initial=[["dram","vram"],["dram","vram"]])
        before=session.state
        step=session.submit(Action("COMPUTE","op1",device="gpu"))
        self.assertEqual(step.error_code,CODE_SLOT_OCCUPIED)
        self.assertEqual(session.state,before)
        self.assertEqual(session.state.op("op1"),"NOT_STARTED")

    def test_visibility_gate_precedes_private_slot_or_stream_checks(self):
        class Invisible:
            def allow_action(self,action,state):return False
            def visible_operation_ids(self,state):return ()
            def visible_tensor_ids(self,state):return ()
        slot={**self.SLOT,"members":["x0","y1"]}
        for action, engine in ((Action("COMPUTE","op1",device="gpu"),None),
                               (copy(1),"ce"),(copy(1),None)):
            session=ServiceSession(scene(initial=[["dram","vram"],["dram","vram"]]),
                parse_services(document(slots=[slot])),visibility=Invisible())
            before=session.state
            result=session.submit(action) if engine is None else session.submit_transfer(action,engine=engine)
            self.assertEqual(result.error_code,CODE_INFORMATION_NOT_KNOWN)
            self.assertNotIn("slot",result.error_message)
            self.assertEqual(session.state,before)


class SharedLinkTests(unittest.TestCase):
    def test_long_ce_short_sm_share_link_short_completes_first_and_long_reaccelerates(self):
        session=runtime()
        self.assertTrue(session.submit_transfer(copy(0),engine="ce").ok)
        self.assertTrue(session.submit_transfer(copy(1),engine="sm").ok)
        self.assertEqual(session.advance().t_ns,32)
        self.assertEqual(session.state.copy_at("x1","vram"),COPY_READY)
        self.assertEqual(session.state.task_on("pcie_ce").remaining_ns,48)
        self.assertEqual(session.advance().t_ns,80)
        result=session.result()
        self.assertEqual([e.end_ns for e in result.evaluation.events[:2]],[80,32])
        self.assertEqual(result.nominal_ends[:2],(64,16))
        self.assertEqual(result.evaluation.h2d_bytes,80)
        intervals=session.service_document()["link_service_intervals"]
        self.assertEqual([[(f["resource"],frac(f["served_bytes"])) for f in i["flows"]] for i in intervals],
                         [[("pcie_ce",16),("pcie_sm",16)],[("pcie_ce",48)]])
        for interval in intervals:self.assertLessEqual(sum(frac(f["served_bytes"]) for f in interval["flows"]),frac(interval["link_budget_bytes"]))

    def test_engine_cap_redistributes_unused_fair_share(self):
        session=ServiceSession(scene(),parse_services(document(sm_cap=250_000_000)))
        session.submit_transfer(copy(0),engine="ce");session.submit_transfer(copy(1),engine="sm")
        self.assertEqual(session.advance().t_ns,64);self.assertEqual(session.advance().t_ns,80)
        flows=session.service_document()["link_service_intervals"][0]["flows"]
        self.assertEqual([frac(f["rate_bytes_per_ns"]) for f in flows],[Fraction(3,4),Fraction(1,4)])

    def test_fractional_payload_progress_is_conservative_at_integer_ns_boundaries(self):
        obj=document();obj["pcie"].update(bandwidth_bytes_per_second=3*10**9,ce_bytes_per_second=3*10**9,sm_bytes_per_second=3*10**9)
        session=ServiceSession(scene((8,4)),parse_services(obj))
        session.submit_transfer(copy(0),engine="ce");session.submit_transfer(copy(1),engine="sm")
        self.assertEqual(session.advance().t_ns,3);self.assertEqual(session.advance().t_ns,5)
        intervals=session.service_document()["link_service_intervals"]
        self.assertEqual(sum(frac(f["served_bytes"]) for i in intervals for f in i["flows"]),12)
        self.assertGreater(sum(frac(i["unused_budget_bytes"]) for i in intervals),0)

    def test_second_transfer_launch_changes_existing_remaining_service(self):
        session=ServiceSession(scene(),parse_services(document()),environment_events=[EnvironmentEvent(8,cpu_available=False)])
        session.submit_transfer(copy(0),engine="ce");self.assertEqual(session.advance().t_ns,8)
        session.submit_transfer(copy(1),engine="sm")
        self.assertEqual(session.advance().t_ns,40);self.assertEqual(session.advance().t_ns,80)

    def test_same_engine_busy_and_missing_engine_fail_loudly(self):
        session=runtime();session.submit_transfer(copy(0),engine="ce")
        self.assertEqual(session.submit_transfer(copy(1),engine="ce").error_code,CODE_RESOURCE_BUSY)
        session=runtime();self.assertFalse(session.submit(copy(0)).ok)
        for engine in (None,[],"dma"):
            with self.assertRaises(Unsupported):runtime().submit_transfer(copy(0),engine=engine)

    def test_sm_mutex_with_gpu_compute_in_both_launch_orders_ce_can_overlap(self):
        session=runtime(payloads=(64,16,16),initial=[["dram"],["dram"],["dram","vram"]])
        self.assertTrue(session.submit(Action("COMPUTE","op2",device="gpu")).ok)
        self.assertTrue(session.submit_transfer(copy(0),engine="ce").ok)
        self.assertEqual(session.submit_transfer(copy(1),engine="sm").error_code,CODE_RESOURCE_BUSY)
        session=runtime(payloads=(64,16,16),initial=[["dram"],["dram"],["dram","vram"]])
        session.submit_transfer(copy(1),engine="sm")
        self.assertEqual(session.submit(Action("COMPUTE","op2",device="gpu")).error_code,CODE_RESOURCE_BUSY)

    def test_common_ledger_preserves_source_read_lock(self):
        session=runtime();session.submit_transfer(copy(0),engine="ce")
        failed=session.submit(Action("EVICT","x0",location="dram"))
        self.assertEqual(failed.error_code,CODE_TENSOR_IN_USE)

    def test_copy_compute_overlap_flag_retained_with_new_engines(self):
        session=runtime(payloads=(64,16,16),initial=[["dram"],["dram"],["dram","vram"]],overlap=False)
        session.submit_transfer(copy(0),engine="ce")
        self.assertEqual(session.submit(Action("COMPUTE","op2",device="gpu")).error_code,CODE_RESOURCE_BUSY)

    def test_copy_compute_overlap_flag_retained_in_reverse_launch_order(self):
        session=runtime(payloads=(64,16,16),initial=[["dram"],["dram"],["dram","vram"]],overlap=False)
        session.submit(Action("COMPUTE","op2",device="cpu"))
        self.assertEqual(session.submit_transfer(copy(0),engine="ce").error_code,CODE_RESOURCE_BUSY)

    def test_unknown_source_and_reserved_target_use_original_tensor_checks(self):
        session=runtime()
        self.assertFalse(session.submit_transfer(copy(0,"vram","dram"),engine="ce").ok)
        session=runtime();session.submit_transfer(copy(0),engine="ce")
        before=session.state
        self.assertFalse(session.submit_transfer(copy(0),engine="sm").ok)
        self.assertEqual(session.state,before)

    def test_cpu_blackout_changes_cpu_only_shared_link_keeps_progress(self):
        session=ServiceSession(scene((64,16,16),initial=[["dram"],["dram"],["dram","vram"]]),parse_services(document()),
            environment_events=[EnvironmentEvent(5,cpu_available=False),EnvironmentEvent(10,cpu_available=True)])
        session.submit(Action("COMPUTE","op2",device="cpu"))
        session.submit_transfer(copy(0),engine="ce");session.submit_transfer(copy(1),engine="sm")
        self.assertEqual([session.advance().t_ns for _ in range(5)],[5,10,15,32,80])
        self.assertEqual(session.result().evaluation.events[0].end_ns,15)
        cpu_intervals=[(s.start_ns,s.end_ns) for s in session.result().service_intervals if s.resource=="cpu_compute"]
        self.assertEqual(cpu_intervals,[(0,5),(10,15)])

    def test_same_time_completion_precedes_environment_failure(self):
        session=ServiceSession(scene(capacity=128),parse_services(document()),
            environment_events=[EnvironmentEvent(16,vram_external_bytes=113)])
        session.submit_transfer(copy(1),engine="sm")
        step=session.advance();self.assertFalse(step.ok)
        self.assertEqual(step.error_code,CODE_EXTERNAL_MEMORY_CONFLICT)
        self.assertEqual(session.state.copy_at("x1","vram"),COPY_READY)
        self.assertEqual(session.result().evaluation.events[0].end_ns,16)

    def test_ssd_uses_original_cost_and_no_pcie_service(self):
        session=runtime(payloads=(64,16),initial=[["ssd"],["dram"]])
        self.assertTrue(session.submit(Action("READ_SSD","x0")).ok)
        self.assertEqual(session.advance().t_ns,12345)
        self.assertEqual(session.service_document()["link_service_intervals"],[])
        self.assertEqual(session.result().evaluation.ssd_read_bytes,64)

    def test_reverse_direction_uses_same_shared_pcie(self):
        session=runtime(initial=[["vram"],["vram"]])
        session.submit_transfer(copy(0,"vram","dram"),engine="ce")
        session.submit_transfer(copy(1,"vram","dram"),engine="sm")
        self.assertEqual(session.advance().t_ns,32);self.assertEqual(session.advance().t_ns,80)
        self.assertEqual(session.result().evaluation.d2h_bytes,80)
        self.assertEqual(session.state.copy_at("x0","dram"),COPY_READY)

    def test_document_is_detached_and_hash_identifies_contract(self):
        session=runtime();session.submit_transfer(copy(0),engine="ce");session.advance()
        snapshot=session.service_document(); digest=snapshot["contract_sha256"]
        snapshot["transfer_engines"].clear();snapshot["link_service_intervals"][0]["flows"].clear()
        self.assertEqual(len(session.service_document()["transfer_engines"]),1)
        self.assertEqual(len(session.service_document()["link_service_intervals"][0]["flows"]),1)
        altered=document();altered["pcie"]["sm_bytes_per_second"]=10**8
        self.assertNotEqual(digest,ServiceSession(scene(),parse_services(altered)).service_document()["contract_sha256"])


if __name__=="__main__":unittest.main()
