"""Physical-pool acceptance: aliases, true copies, locks and workspace."""
from copy import deepcopy
from dataclasses import replace
import unittest

from llm_infer_model.tensor.engine import (
    Action, MappingError, initial_state, transition, used_bytes, used_physical_bytes,
    allocation_identity, can_access, shared_source, incremental_allocation_bytes,
    released_physical_bytes, has_independent_backup, evaluate_mapping, _event_for,
)
from llm_infer_model.tensor.spec import (
    Architecture, InitialCapacityExceeded, InvalidInput, Unsupported, build_scenario,
    scenario_fingerprint,
)


def documents(*, sizes=(32,), capacity=256, initial=None, mode="shared", workspace=0,
              shared_alignment=1, reserve=0, priced=True, ssd=None, ssd_alignment=None,
              ssd_reserve=0):
    tensors, ops, compute = [], [], {}
    for i, size in enumerate(sizes):
        for prefix, role, locations in (("x", "input", ["dram"] if initial is None else initial[i]),
                                        ("y", "output", [])):
            tensors.append(dict(id=f"{prefix}{i}", name=f"{prefix}{i}", role=role,
                dtype="GGML_TYPE_F32", ne=[size // 4, 1, 1, 1], nb=[4, size, size, size],
                storage_bytes=size, initial_locations=locations))
        ops.append(dict(id=f"op{i}", semantic_op="RELU", ggml_op="GGML_OP_UNARY",
            unary_op="GGML_UNARY_OP_RELU", op_params=[], inputs=[f"x{i}"], output=f"y{i}"))
        compute[f"op{i}"] = {device: dict(duration_ns=10, workspace_bytes=workspace)
                                 for device in ("cpu", "gpu")}
    ids = [t["id"] for t in tensors]
    routes = ("dram_to_vram", "vram_to_dram", "ssd_to_dram", "dram_to_ssd")
    architecture = dict(dram_capacity_bytes=capacity, vram_capacity_bytes=capacity)
    if mode == "shared":
        architecture.update(memory_mode=mode, shared_capacity_bytes=capacity,
                            shared_alignment_bytes=shared_alignment,
                            shared_runtime_reserved_bytes=reserve)
    if ssd is not None:
        architecture.update(ssd_capacity_bytes=ssd, ssd_alignment_bytes=ssd_alignment,
                            ssd_runtime_reserved_bytes=ssd_reserve)
    workload = dict(schema_version="0.2", id="shared-core-workload",
        origin=dict(kind="synthetic_fixture"), tensors=tensors, operations=ops,
        outputs=[f"y{i}" for i in range(len(sizes))])
    scene = dict(schema_version="0.2", id="shared-core-scenario", architecture=architecture,
        costs=dict(source="declared_hand_calculated", compute=compute,
            transfer={route: {tid: dict(duration_ns=7) for tid in ids}
                      for route in routes if priced or route in ("ssd_to_dram", "dram_to_ssd")}),
        mapspace=dict(allowed_devices={op["id"]: ["cpu", "gpu"] for op in ops},
            allowed_transfers={route: ids for route in routes}, allow_eviction=True,
            allow_copy_compute_overlap=True, allow_cpu_gpu_overlap=True, allow_recomputation=False),
        goal=dict(output_locations={f"y{i}": "vram" for i in range(len(sizes))}),
        mapper=dict(algorithm="uniform_cost", max_expanded_states=0, wall_time_limit_s=0))
    return workload, scene


def scene(**kwargs):
    return build_scenario(*documents(**kwargs))


def share(tensor="x0", src="dram", dst="vram"):
    return Action("SHARE", tensor, src=src, dst=dst)


def compute(i=0, device="gpu"):
    return Action("COMPUTE", f"op{i}", device=device)


class SharedStorageCoreTests(unittest.TestCase):
    def test_initial_dual_views_count_once_only_in_shared_mode(self):
        shared = scene(initial=[["dram", "vram"]])
        state = initial_state(shared)
        self.assertEqual(used_physical_bytes(shared, state), {"shared": 32})
        self.assertEqual(state.allocation_ref("x0", "dram"), state.allocation_ref("x0", "vram"))
        separate = scene(initial=[["dram", "vram"]], mode="separate")
        self.assertEqual(used_physical_bytes(separate, initial_state(separate)), {"dram": 32, "vram": 32})

    def test_one_ten_gib_shared_buffer_fits_sixteen_gib(self):
        value = scene(sizes=(10 * 2**30,), capacity=16 * 2**30, initial=[["dram", "vram"]])
        self.assertEqual(used_bytes(value, initial_state(value), "shared"), 10 * 2**30)

    def test_two_distinct_ten_gib_buffers_do_not_fit_sixteen_gib(self):
        value = scene(sizes=(10 * 2**30, 10 * 2**30), capacity=16 * 2**30,
                      initial=[["dram"], ["vram"]])
        with self.assertRaises(InitialCapacityExceeded):
            initial_state(value)

    def test_share_is_instantaneous_zero_payload_without_copy_quote(self):
        value = scene(priced=False)
        before = initial_state(value)
        after = transition(value, before, share())
        event = _event_for(value, after, 0, share(), before=before)
        self.assertEqual((after.t, event.start_ns, event.end_ns, event.bytes_transferred), (0, 0, 0, 0))
        self.assertEqual(used_bytes(value, after, "shared"), 32)
        self.assertEqual(after.running, ())

    def test_actual_copy_is_positive_duration_and_independent_allocation(self):
        value = scene()
        state = transition(value, initial_state(value), Action("COPY", "x0", src="dram", dst="vram"))
        self.assertEqual(used_bytes(value, state, "shared"), 64)
        self.assertEqual(state.running[0].remaining_ns, 7)
        self.assertNotEqual(state.allocation_ref("x0", "dram"), state.allocation_ref("x0", "vram"))
        state = transition(value, state, Action("ADVANCE"))
        self.assertTrue(has_independent_backup(value, state, "x0", "vram"))
        self.assertEqual(released_physical_bytes(value, state, "x0", "vram"), 32)

    def test_missing_real_copy_cost_is_rejected_not_coerced_to_zero(self):
        value = scene(priced=False)
        with self.assertRaisesRegex(MappingError, "explicit positive copy price"):
            transition(value, initial_state(value), Action("COPY", "x0", src="dram", dst="vram"))

    def test_share_rejected_on_separate_memory(self):
        value = scene(mode="separate")
        with self.assertRaises(MappingError):
            transition(value, initial_state(value), share())

    def test_compute_does_not_implicitly_share_an_absent_view(self):
        value = scene()
        before = initial_state(value)
        self.assertFalse(can_access(value, before, "x0", "vram"))
        self.assertEqual(shared_source(value, before, "x0", "vram"), "dram")
        with self.assertRaises(MappingError) as caught:
            transition(value, before, compute())
        self.assertEqual(caught.exception.code, "INPUT_NOT_READY")

    def test_producer_must_complete_before_share_or_consume(self):
        value = scene()
        state = transition(value, initial_state(value), compute(device="cpu"))
        with self.assertRaises(MappingError) as caught:
            transition(value, state, share("y0"))
        self.assertEqual(caught.exception.code, "INPUT_NOT_READY")
        state = transition(value, state, Action("ADVANCE"))
        state = transition(value, state, share("y0"))
        self.assertEqual(state.copy_at("y0", "vram"), "READY")

    def test_cross_alias_read_lock_prevents_eviction(self):
        value = scene()
        state = transition(value, initial_state(value), share())
        state = transition(value, state, compute())
        with self.assertRaises(MappingError) as caught:
            transition(value, state, Action("EVICT", "x0", location="dram"))
        self.assertEqual(caught.exception.code, "TENSOR_IN_USE")

    def test_drop_reference_releases_zero_then_last_reference_releases_buffer(self):
        value = scene()
        state = transition(value, initial_state(value), share())
        self.assertFalse(has_independent_backup(value, state, "x0", "vram"))
        self.assertEqual(released_physical_bytes(value, state, "x0", "dram"), 0)
        before = state
        state = transition(value, state, Action("EVICT", "x0", location="dram"))
        event = _event_for(value, state, 0, Action("EVICT", "x0", location="dram"), before=before)
        self.assertEqual(event.bytes_released, 0)
        state = transition(value, state, compute())
        state = transition(value, state, Action("ADVANCE"))
        self.assertEqual(released_physical_bytes(value, state, "x0", "vram"), 32)
        state = transition(value, state, Action("EVICT", "x0", location="vram"))
        self.assertEqual(used_bytes(value, state, "shared"), 32)

    def test_live_last_value_cannot_be_destroyed(self):
        value = scene()
        with self.assertRaises(MappingError) as caught:
            transition(value, initial_state(value), Action("EVICT", "x0", location="dram"))
        self.assertEqual(caught.exception.code, "LIVE_VALUE_LOSS")

    def test_cpu_and_gpu_workspaces_are_additive_in_one_pool(self):
        value = scene(sizes=(32, 32), capacity=159, workspace=16)
        state = transition(value, initial_state(value), share("x1"))
        state = transition(value, state, compute(0, "cpu"))
        self.assertEqual(used_bytes(value, state, "shared"), 112)
        with self.assertRaises(MappingError) as caught:
            transition(value, state, compute(1, "gpu"))
        self.assertEqual(caught.exception.code, "CAPACITY_EXCEEDED")
        roomy = replace(value, architecture=replace(value.architecture, shared_capacity_bytes=160))
        state = transition(roomy, state, compute(1, "gpu"))
        self.assertEqual(used_bytes(roomy, state, "shared"), 160)
        state = transition(roomy, state, Action("ADVANCE"))
        self.assertEqual(used_bytes(roomy, state, "shared"), 128)

    def test_shared_alignment_and_reservation_charged_once(self):
        value = scene(initial=[["dram", "vram"]], shared_alignment=64, reserve=12)
        self.assertEqual(used_bytes(value, initial_state(value), "shared"), 76)

    def test_shared_reservation_uses_shared_not_legacy_gpu_capacity(self):
        value = scene(reserve=128)
        value = replace(value, architecture=replace(value.architecture, vram_capacity_bytes=16))
        self.assertEqual(used_bytes(value, initial_state(value), "shared"), 160)

    def test_complete_mapping_goal_share_has_zero_d2h_and_correct_snapshot(self):
        value = scene()
        actions = (compute(device="cpu"), Action("ADVANCE"), share("y0"))
        result = evaluate_mapping(value, actions)
        self.assertEqual((result.status, result.makespan_ns, result.h2d_bytes, result.d2h_bytes), ("valid", 10, 0, 0))
        self.assertEqual(result.peak_shared_bytes, 64)
        self.assertEqual(result.states[-1].used_shared_bytes, 64)
        self.assertEqual(len(result.states[-1].storage_allocations), 3)

    def test_fingerprint_preserves_default_separate_and_distinguishes_shared(self):
        workload, raw = documents(mode="separate")
        old = build_scenario(workload, raw)
        explicit = deepcopy(raw)
        explicit["architecture"]["memory_mode"] = "separate"
        self.assertEqual(scenario_fingerprint(old), scenario_fingerprint(build_scenario(workload, explicit)))
        self.assertNotEqual(scenario_fingerprint(old), scenario_fingerprint(scene()))

    def test_helpers_distinguish_access_share_and_copy(self):
        value = scene()
        state = initial_state(value)
        self.assertEqual(incremental_allocation_bytes(value, state, "x0", "vram"), 32)
        self.assertEqual(incremental_allocation_bytes(value, state, "x0", "vram", share_from="dram"), 0)
        self.assertIsNone(allocation_identity(value, state, "x0", "vram"))

    def test_alias_classes_canonicalize_allocation_names_in_search_key(self):
        value = scene()
        state = transition(value, initial_state(value), share())
        renamed = replace(state, storage_allocations=tuple((t, l, "renamed") for t, l, _ in state.storage_allocations), next_allocation_id=999)
        self.assertEqual(state.key(), renamed.key())

    def test_shared_configuration_is_explicit_and_strict(self):
        for change in (dict(memory_mode="unknown"), dict(memory_mode="shared"),
                       dict(shared_capacity_bytes=64), dict(memory_mode="shared", shared_capacity_bytes=True)):
            with self.subTest(change=change), self.assertRaises(InvalidInput):
                Architecture(vram_capacity_bytes=256, **change)


class SharedPoolKeepsTheDiskItself(unittest.TestCase):
    """The UMA pool is a *memory* identity; the disk is still its own device.

    On a shared machine DRAM and VRAM answer with one pool figure. The SSD must
    not: it has its own declared budget and its own copies, and answering the
    disk question with the memory figure would check a full disk against a
    number that never counted a byte of it.
    """

    def test_the_disk_ledger_is_not_the_pool_ledger(self):
        value = scene(sizes=(32, 64), initial=[["ssd"], ["dram"]], ssd=128)
        state = initial_state(value)
        self.assertEqual(used_bytes(value, state, "shared"), 64)
        self.assertEqual(used_bytes(value, state, "ssd"), 32)

    def test_a_large_shared_pool_does_not_enlarge_the_disk(self):
        with self.assertRaises(InitialCapacityExceeded):
            initial_state(scene(initial=[["ssd"]], capacity=1 << 20, ssd=31))

    def test_reading_a_disk_copy_charges_the_pool_and_keeps_the_disk_copy(self):
        value = scene(sizes=(32, 64), initial=[["ssd"], ["dram"]], ssd=128)
        state = transition(value, initial_state(value), Action("READ_SSD", "x0"))
        self.assertEqual(used_bytes(value, state, "shared"), 96)
        self.assertEqual(used_bytes(value, state, "ssd"), 32)


class MemoryModeTests(unittest.TestCase):
    """The one word that decides whether the accelerator shares the host's memory.

    It is not decoration: `shared` adds SHARE to the action space and makes the
    pool ledger one number instead of two. A value the parser merely accepted as
    a string would be read as `separate`, so a typo would run a whole campaign on
    a UMA machine with discrete-memory semantics and report nothing.
    """

    def test_a_misspelt_memory_mode_is_refused_and_named(self):
        workload, document = documents()
        for wrong in ("Shared", "uma", "unified", "shared "):
            document["architecture"]["memory_mode"] = wrong
            with self.assertRaises(InvalidInput) as raised:
                build_scenario(deepcopy(workload), deepcopy(document))
            self.assertIn("memory_mode", str(raised.exception))
            self.assertIn("separate", str(raised.exception))

    def test_the_two_modes_are_accepted(self):
        for mode in ("shared", "separate"):
            workload, document = documents(mode=mode)
            document["architecture"]["memory_mode"] = mode
            build_scenario(workload, document)


if __name__ == "__main__":
    unittest.main()
