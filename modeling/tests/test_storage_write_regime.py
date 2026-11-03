"""The disk's write price is a state function, not a constant.

A consumer SSD absorbs writes at cache speed until its cache and the free-block
pool behind it are exhausted, then settles at the media's rate. Measured on the
laptop drive that is roughly 2.5 GB/s against 0.46 GB/s -- the same bytes for
five times the time, and which one you get depends on how much this scenario
has already written. So the cost table carries a second price on ``dram_to_ssd``
and a threshold, and the kernel keeps the running total in its state.

What this file pins:

* the two declarations must be supplied together -- a threshold with no second
  price changes nothing, and a second price with no threshold can never be
  selected;
* the second price exists only on the write route, because a full write cache
  is a property of the disk, not of the copy;
* the switch happens at the write that *starts* at or past the threshold, and
  never switches back inside one scenario (a serving loop has no idle window);
* the read path is untouched, even after the threshold is crossed;
* a scenario that declares nothing keeps exactly the prices and the state keys
  it always had.
"""
from __future__ import annotations

import unittest

from llm_infer_model.tensor.engine import (
    ACTION_ADVANCE, ACTION_READ_SSD, ACTION_WRITE_SSD, Action, State, initial_state,
    transition,
)
from llm_infer_model.tensor.spec import (
    InvalidInput, Unsupported, build_scenario, scenario_fingerprint,
)

BURST_NS = 100
SUSTAINED_NS = 500
READ_NS = 30
THRESHOLD = 64


def documents(*, sizes=(40, 40, 40), regime=THRESHOLD, sustained=True,
              sustained_route="dram_to_ssd"):
    """Three weights in DRAM, one weight already on disk, exact byte prices."""
    tensors, ops, compute, ids = [], [], {}, []
    for i, size in enumerate(sizes):
        for prefix, role, locations in (("x", "weight", ["dram"]),
                                        ("y", "output", [])):
            tensors.append(dict(
                id=f"{prefix}{i}", name=f"{prefix}{i}", role=role,
                dtype="GGML_TYPE_F32", ne=[size // 4, 1, 1, 1],
                nb=[4, size, size, size], storage_bytes=size,
                initial_locations=locations))
            ids.append(f"{prefix}{i}")
        ops.append(dict(id=f"op{i}", semantic_op="RELU", ggml_op="GGML_OP_UNARY",
                        unary_op="GGML_UNARY_OP_RELU", op_params=[],
                        inputs=[f"x{i}"], output=f"y{i}"))
        compute[f"op{i}"] = {device: dict(duration_ns=10, workspace_bytes=0)
                             for device in ("cpu", "gpu")}
    # One weight starts on the disk so a read can be priced after the threshold.
    tensors.extend([
        dict(id="disk", name="disk", role="weight", dtype="GGML_TYPE_F32",
             ne=[10, 1, 1, 1], nb=[4, 40, 40, 40], storage_bytes=40,
             initial_locations=["ssd"]),
        dict(id="ydisk", name="ydisk", role="output", dtype="GGML_TYPE_F32",
             ne=[10, 1, 1, 1], nb=[4, 40, 40, 40], storage_bytes=40,
             initial_locations=[]),
    ])
    ops.append(dict(id="opdisk", semantic_op="RELU", ggml_op="GGML_OP_UNARY",
                    unary_op="GGML_UNARY_OP_RELU", op_params=[],
                    inputs=["disk"], output="ydisk"))
    compute["opdisk"] = {device: dict(duration_ns=10, workspace_bytes=0)
                         for device in ("cpu", "gpu")}
    outputs = [f"y{i}" for i in range(len(sizes))] + ["ydisk"]
    ids.extend(["disk", "ydisk"])
    transfer = {}
    for route in ("dram_to_vram", "vram_to_dram", "ssd_to_dram", "dram_to_ssd"):
        transfer[route] = {}
        for tensor_id in ids:
            price = {"duration_ns": BURST_NS if route == "dram_to_ssd"
                     else READ_NS if route == "ssd_to_dram" else 7}
            if sustained and route == sustained_route:
                price["sustained_duration_ns"] = SUSTAINED_NS
            transfer[route][tensor_id] = price
    costs = dict(source="declared_hand_calculated", compute=compute, transfer=transfer)
    if regime is not None:
        costs["storage_write_regime"] = {"after_cumulative_bytes": regime}
    workload = dict(schema_version="0.2", id="write-regime-workload",
                    origin=dict(kind="synthetic_fixture"), tensors=tensors,
                    operations=ops, outputs=outputs)
    scene = dict(schema_version="0.2", id="write-regime-scenario",
                 architecture=dict(dram_capacity_bytes=1024, vram_capacity_bytes=1024,
                                   ssd_capacity_bytes=1024, ssd_alignment_bytes=1),
                 costs=costs,
                 mapspace=dict(allowed_devices={op["id"]: ["cpu", "gpu"] for op in ops},
                               allowed_transfers={route: ids for route in
                                                  ("dram_to_vram", "vram_to_dram",
                                                   "ssd_to_dram", "dram_to_ssd")},
                               allow_eviction=True, allow_copy_compute_overlap=True,
                               allow_cpu_gpu_overlap=True, allow_recomputation=False),
                 goal=dict(output_locations={tensor_id: "vram" for tensor_id in outputs}),
                 mapper=dict(algorithm="uniform_cost", max_expanded_states=0,
                             wall_time_limit_s=0))
    return workload, scene


def scene(**kwargs):
    return build_scenario(*documents(**kwargs))


def written(scenario, state, tensor_id):
    """Start one write and return the state once it has landed."""
    return transition(scenario, transition(
        scenario, state, Action(ACTION_WRITE_SSD, tensor_id)), Action(ACTION_ADVANCE))


class TheThresholdSelectsThePrice(unittest.TestCase):
    def test_the_write_that_starts_past_the_threshold_pays_the_sustained_price(self):
        value = scene()
        state = initial_state(value)
        self.assertEqual(state.storage_write_debt_bytes, 0)

        first = transition(value, state, Action(ACTION_WRITE_SSD, "x0"))
        self.assertEqual(first.running[0].remaining_ns, BURST_NS)
        self.assertEqual(first.storage_write_debt_bytes, 40)

        state = transition(value, first, Action(ACTION_ADVANCE))
        second = transition(value, state, Action(ACTION_WRITE_SSD, "x1"))
        # 40 bytes written so far, so this one still starts below the threshold...
        self.assertEqual(second.running[0].remaining_ns, BURST_NS)
        state = transition(value, second, Action(ACTION_ADVANCE))
        # ...but it pushes the total to 80, which the ledger records as 64.
        self.assertEqual(state.storage_write_debt_bytes, THRESHOLD)

        third = transition(value, state, Action(ACTION_WRITE_SSD, "x2"))
        self.assertEqual(third.running[0].remaining_ns, SUSTAINED_NS)
        self.assertEqual(third.storage_write_debt_bytes, THRESHOLD)

    def test_the_debt_never_exceeds_the_threshold(self):
        value = scene()
        state = initial_state(value)
        for index in range(3):
            state = written(value, state, f"x{index}")
            self.assertLessEqual(state.storage_write_debt_bytes, THRESHOLD)
        self.assertEqual(state.storage_write_debt_bytes, THRESHOLD)

    def test_reads_are_not_repriced_after_the_threshold(self):
        value = scene()
        state = initial_state(value)
        for index in range(3):
            state = written(value, state, f"x{index}")
        read = transition(value, state, Action(ACTION_READ_SSD, "disk"))
        self.assertEqual(read.running[0].remaining_ns, READ_NS)
        self.assertEqual(read.storage_write_debt_bytes, THRESHOLD)

    def test_two_states_that_wrote_different_amounts_are_different_states(self):
        value = scene()
        state = initial_state(value)
        after_one = written(value, state, "x0")
        after_two = written(value, after_one, "x1")
        self.assertNotEqual(after_one.key(), after_two.key())

    def test_the_debt_alone_separates_two_states(self):
        """Same ledgers, different write history: not the same future."""
        bare = dict(t=0, op_status=(), copy_status=(), running=())
        dry = State(**bare)
        wet = State(**bare, storage_write_debt_bytes=40)
        self.assertEqual(dry.storage_write_debt_bytes, 0)
        self.assertNotEqual(dry.key(), wet.key())
        # And a state with no debt has exactly the key it had before the field
        # existed: nothing is appended when the ledger never moves.
        self.assertEqual(len(dry.key()), len(State(**bare, storage_write_debt_bytes=0).key()))

    def test_without_a_regime_every_write_keeps_the_burst_price(self):
        value = scene(regime=None, sustained=False)
        state = initial_state(value)
        for index in range(3):
            started = transition(value, state, Action(ACTION_WRITE_SSD, f"x{index}"))
            self.assertEqual(started.running[0].remaining_ns, BURST_NS)
            self.assertEqual(started.storage_write_debt_bytes, 0)
            state = transition(value, started, Action(ACTION_ADVANCE))


class TheDeclarationMustBeComplete(unittest.TestCase):
    def test_a_sustained_price_without_a_threshold_is_refused(self):
        with self.assertRaisesRegex(InvalidInput, "storage_write_regime"):
            scene(regime=None, sustained=True)

    def test_a_threshold_without_a_sustained_price_is_refused(self):
        with self.assertRaisesRegex(InvalidInput, "changes no price"):
            scene(regime=THRESHOLD, sustained=False)

    def test_only_the_write_route_can_carry_a_second_price(self):
        with self.assertRaisesRegex(Unsupported, "dram_to_ssd"):
            scene(sustained_route="ssd_to_dram", regime=THRESHOLD, sustained=True)

    def test_a_sustained_price_faster_than_the_burst_is_refused(self):
        workload, raw = documents()
        raw["costs"]["transfer"]["dram_to_ssd"]["x0"]["sustained_duration_ns"] = BURST_NS - 1
        with self.assertRaisesRegex(InvalidInput, "swapped"):
            build_scenario(workload, raw)

    def test_both_numbers_reach_the_cost_model(self):
        value = scene()
        self.assertEqual(value.costs.transfer["dram_to_ssd"]["x0"].sustained_duration_ns,
                         SUSTAINED_NS)
        self.assertEqual(value.costs.storage_write_regime.after_cumulative_bytes,
                         THRESHOLD)

    def test_the_threshold_is_part_of_the_fingerprint(self):
        plain = scene(regime=None, sustained=False)
        regimed = scene(regime=THRESHOLD)
        self.assertNotEqual(scenario_fingerprint(plain), scenario_fingerprint(regimed))
        self.assertEqual(scenario_fingerprint(regimed),
                         scenario_fingerprint(scene(regime=THRESHOLD)))


if __name__ == "__main__":
    unittest.main()
