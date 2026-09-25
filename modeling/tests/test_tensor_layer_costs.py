"""The cost bridge, and the kernel's independence from ``mapping``.

``llm_infer_model.tensor.layer_costs`` exists so that the window policy and the
search see the same durations the calibrated layer model produces, without the
formula being written a second time. These tests pin exactly that:

* the measured overrides win, and the fallback is the *existing* closed form
  (spelled out longhand below, so a silent change to it fails here);
* the conversion to integer nanoseconds refuses to swallow a duration;
* the ID/wiring checks refuse a mismatched correspondence instead of guessing;
* the convenience entry names every service it cannot yet carry over.

Plus one test that the kernel is usable with ``mapping`` **absent**: the public
API has to import, step and replay on its own, or the dependency direction in
``ALIGNMENT_IMPLEMENTATION.md`` §3 is only a comment.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

from llm_infer_model.model import HardwareSpec, LayerSpec, ModelConfig, PolicySpec, StateSpec
from llm_infer_model.tensor import (
    Architecture,
    Costs,
    DTYPE_F32,
    InvalidInput,
    LOC_DRAM,
    LOC_VRAM,
    MapperConfig,
    Mapspace,
    Operation,
    Origin,
    ROLE_INPUT,
    ROLE_INTERMEDIATE,
    ROLE_OUTPUT,
    ROLE_WEIGHT,
    SEMANTIC_ADD,
    SEMANTIC_MUL_MAT,
    Scenario,
    Tensor,
    Unsupported,
    Workload,
)
from llm_infer_model.tensor.layer_costs import (
    SOURCE_LAYER_COSTS,
    assert_weights_copyable,
    costs_for_chain,
    costs_from_layers,
    costs_from_model_config,
    seconds_to_ns,
)

NS = 1_000_000_000
MODELING_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_HARDWARE = HardwareSpec(
    gpu_effective_flops=16_000.0,
    h2d_bandwidth_bytes_per_s=32_000.0,
    h2d_latency_s=1e-3,
    vram_capacity_bytes=1024,
)

# 4x4 F32 weights (64 B) and 4x1 F32 activations (16 B): the shapes the
# checked-in fixtures use, so a cost table built here can be compared with one
# loaded from ``mapping/examples`` without a shape mismatch in the way.
_WEIGHT_SHAPE = {"ne": (4, 4, 1, 1), "nb": (4, 16, 64, 64), "storage_bytes": 64}
_ACTIVATION_SHAPE = {"ne": (4, 1, 1, 1), "nb": (4, 16, 16, 16), "storage_bytes": 16}


def _weight(tensor_id: str) -> Tensor:
    return Tensor(
        id=tensor_id,
        name=f"blk.{tensor_id}.weight",
        role=ROLE_WEIGHT,
        dtype=DTYPE_F32,
        initial_locations=(LOC_DRAM,),
        **_WEIGHT_SHAPE,
    )


def _activation(tensor_id: str, role: str, initial: tuple[str, ...] = ()) -> Tensor:
    return Tensor(
        id=tensor_id,
        name=tensor_id,
        role=role,
        dtype=DTYPE_F32,
        initial_locations=initial,
        **_ACTIVATION_SHAPE,
    )


def _mul_mat(operation_id: str, weight_id: str, activation_in: str, output: str) -> Operation:
    return Operation(
        id=operation_id,
        semantic_op=SEMANTIC_MUL_MAT,
        ggml_op="GGML_OP_MUL_MAT",
        unary_op=None,
        op_params=(),
        inputs=(weight_id, activation_in),
        output=output,
    )


def chain_workload() -> Workload:
    """``h1 = W1*x; h2 = W2*h1; y = W3*h2`` -- the chain of §6-B."""
    return Workload(
        id="three-layer-chain",
        origin=Origin(kind="synthetic_fixture", sample="h1=W1*x; h2=W2*h1; y=W3*h2"),
        tensors=(
            _weight("W1"),
            _weight("W2"),
            _weight("W3"),
            _activation("x", ROLE_INPUT, (LOC_VRAM,)),
            _activation("h1", ROLE_INTERMEDIATE),
            _activation("h2", ROLE_INTERMEDIATE),
            _activation("y", ROLE_OUTPUT),
        ),
        operations=(
            _mul_mat("c1", "W1", "x", "h1"),
            _mul_mat("c2", "W2", "h1", "h2"),
            _mul_mat("c3", "W3", "h2", "y"),
        ),
        outputs=("y",),
    )


def add_tail_workload() -> Workload:
    """``h = W1*x; y = h + x`` -- a node with no weight of its own."""
    return Workload(
        id="chain-with-add",
        origin=Origin(kind="synthetic_fixture"),
        tensors=(
            _weight("W1"),
            _activation("x", ROLE_INPUT, (LOC_VRAM,)),
            _activation("h", ROLE_INTERMEDIATE),
            _activation("y", ROLE_OUTPUT),
        ),
        operations=(
            _mul_mat("c1", "W1", "x", "h"),
            Operation(
                id="add",
                semantic_op=SEMANTIC_ADD,
                ggml_op="GGML_OP_ADD",
                unary_op=None,
                op_params=(),
                inputs=("h", "x"),
                output="y",
            ),
        ),
        outputs=("y",),
    )


def shared_weight_workload() -> Workload:
    """``h = W1*x; y = W1*h`` -- one weight read by two operations."""
    return Workload(
        id="shared-weight",
        origin=Origin(kind="synthetic_fixture"),
        tensors=(
            _weight("W1"),
            _activation("x", ROLE_INPUT, (LOC_VRAM,)),
            _activation("h", ROLE_INTERMEDIATE),
            _activation("y", ROLE_OUTPUT),
        ),
        operations=(
            _mul_mat("c1", "W1", "x", "h"),
            _mul_mat("c2", "W1", "h", "y"),
        ),
        outputs=("y",),
    )


def measured_layer(compute_ms: float, transfer_ms: float, *, name: str = "layer") -> LayerSpec:
    """A layer whose costs come from measurements, not from the closed form."""
    return LayerSpec(
        name=name,
        weight_bytes=64,
        flops=64.0,
        measured_compute_seconds=compute_ms / 1e3,
        measured_transfer_seconds=transfer_ms / 1e3,
    )


def measured_layers_by_operation() -> dict[str, LayerSpec]:
    return {f"c{index}": measured_layer(2.0, 3.0, name=f"layer{index}") for index in (1, 2, 3)}


def chain_costs() -> Costs:
    return costs_for_chain(chain_workload(), _HARDWARE, measured_layers_by_operation())


def scenario_for(
    workload: Workload, costs: Costs, *, capacity: int = 1024, id_: str = "unit"
) -> Scenario:
    return Scenario(
        schema_version="0.1",
        id=id_,
        workload=workload,
        architecture=Architecture(vram_capacity_bytes=capacity),
        costs=costs,
        mapspace=Mapspace(
            compute_device="gpu",
            copy_tensor_ids=tuple(t.id for t in workload.tensors if t.role == ROLE_WEIGHT),
            allow_copy_compute_overlap=True,
            allow_eviction=True,
        ),
        mapper=MapperConfig(algorithm="uniform_cost"),
    )


# ---------------------------------------------------------------------------
# seconds -> integer nanoseconds
# ---------------------------------------------------------------------------


class SecondsToNanosecondsTests(unittest.TestCase):
    def test_converts_milliseconds_exactly(self) -> None:
        self.assertEqual(seconds_to_ns(2e-3, "x"), 2_000_000)
        self.assertEqual(seconds_to_ns(1.4838e-3, "x"), 1_483_800)

    def test_rejects_a_duration_that_rounds_away(self) -> None:
        """The silent failure this guards: 4e-10 s would become 0 ns, not an error."""
        with self.assertRaises(InvalidInput) as caught:
            seconds_to_ns(4e-10, "c1 的计算时长")
        message = str(caught.exception)
        self.assertIn("整数纳秒", message)
        self.assertIn("c1", message)

    def test_rejects_zero_negative_and_non_finite(self) -> None:
        for bad in (0.0, -1e-3, float("inf"), float("-inf"), float("nan"), None, "2ms"):
            with self.subTest(value=bad):
                with self.assertRaises(InvalidInput):
                    seconds_to_ns(bad, "x")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# costs_from_layers: the two rules, spelled out longhand
# ---------------------------------------------------------------------------


class CostsFromLayersTests(unittest.TestCase):
    def test_measured_overrides_win(self) -> None:
        layer = measured_layer(2.0, 3.0)
        costs = costs_from_layers(
            _HARDWARE, compute_layers={"c1": layer}, weight_layers={"W1": layer}
        )
        self.assertEqual(costs.source, SOURCE_LAYER_COSTS)
        self.assertEqual(costs.compute["c1"].duration_ns, 2_000_000)
        self.assertEqual(costs.h2d["W1"].duration_ns, 3_000_000)

    def test_formula_path_reuses_the_existing_layer_rules(self) -> None:
        """With no measured value, the *old* closed form must be what runs.

        The expectations are written out rather than taken from
        ``LayerSpec.compute_seconds``: if that method changes, this test has to be
        changed deliberately instead of following along.
        """
        hardware = HardwareSpec(
            gpu_effective_flops=16_000.0,
            h2d_bandwidth_bytes_per_s=32_000.0,
            h2d_latency_s=1e-3,
            vram_capacity_bytes=1024,
        )
        layer = LayerSpec(name="layer1", weight_bytes=64, flops=32.0)

        costs = costs_from_layers(
            hardware, compute_layers={"c1": layer}, weight_layers={"W1": layer}
        )
        self.assertEqual(costs.compute["c1"].duration_ns, round(32.0 / 16_000.0 * NS))
        self.assertEqual(costs.h2d["W1"].duration_ns, round((1e-3 + 64 / 32_000.0) * NS))

    def test_the_fixed_h2d_latency_is_charged_per_weight(self) -> None:
        """Not once per layer: a streamed layer's weight pays it on every load.

        Pinned because this term decides whether a small weight is bandwidth-bound
        or latency-bound, and it is what a "divide the layer cost by the number of
        ops" shortcut would get wrong.
        """
        hardware = HardwareSpec(
            gpu_effective_flops=1.0,
            h2d_bandwidth_bytes_per_s=1.0,
            h2d_latency_s=1.4838e-3,
            vram_capacity_bytes=1024,
        )
        layer = LayerSpec(name="tiny", weight_bytes=1, flops=1.0)
        costs = costs_from_layers(
            hardware, compute_layers={"c1": layer}, weight_layers={"W1": layer}
        )
        self.assertEqual(costs.h2d["W1"].duration_ns, round((1.4838e-3 + 1.0) * NS))

    def test_workspace_lands_on_every_compute(self) -> None:
        costs = costs_from_layers(
            _HARDWARE,
            compute_layers=measured_layers_by_operation(),
            weight_layers={
                f"W{index}": layer
                for index, layer in zip((1, 2, 3), measured_layers_by_operation().values())
            },
            workspace_bytes=128,
        )
        self.assertEqual({cost.workspace_bytes for cost in costs.compute.values()}, {128})

    def test_empty_maps_are_rejected(self) -> None:
        layer = measured_layer(2.0, 3.0)
        with self.assertRaises(InvalidInput):
            costs_from_layers(_HARDWARE, compute_layers={}, weight_layers={"W1": layer})
        with self.assertRaises(InvalidInput):
            costs_from_layers(_HARDWARE, compute_layers={"c1": layer}, weight_layers={})


# ---------------------------------------------------------------------------
# costs_for_chain: the correspondence has to be right, or it fails
# ---------------------------------------------------------------------------


class CostsForChainTests(unittest.TestCase):
    def test_derives_the_weight_side_from_the_operation_inputs(self) -> None:
        costs = chain_costs()
        self.assertEqual(sorted(costs.compute), ["c1", "c2", "c3"])
        self.assertEqual(sorted(costs.h2d), ["W1", "W2", "W3"])
        for operation_id, weight_id in (("c1", "W1"), ("c2", "W2"), ("c3", "W3")):
            self.assertEqual(costs.compute[operation_id].duration_ns, 2_000_000)
            self.assertEqual(costs.h2d[weight_id].duration_ns, 3_000_000)

    def test_weight_bytes_must_match_the_tensor(self) -> None:
        """The check that catches 'the layer list is off by one'."""
        layers = measured_layers_by_operation()
        layers["c2"] = LayerSpec(
            name="layer-wrong",
            weight_bytes=128,
            flops=64.0,
            measured_compute_seconds=2e-3,
            measured_transfer_seconds=3e-3,
        )
        with self.assertRaises(InvalidInput) as caught:
            costs_for_chain(chain_workload(), _HARDWARE, layers)
        message = str(caught.exception)
        self.assertIn("weight_bytes=128", message)
        self.assertIn("storage_bytes=64", message)

    def test_operation_coverage_must_match_exactly(self) -> None:
        with self.subTest("missing"):
            partial = measured_layers_by_operation()
            del partial["c3"]
            with self.assertRaises(InvalidInput) as caught:
                costs_for_chain(chain_workload(), _HARDWARE, partial)
            self.assertIn("c3", str(caught.exception))
        with self.subTest("extra"):
            extra = measured_layers_by_operation()
            extra["c9"] = measured_layer(2.0, 3.0, name="layer9")
            with self.assertRaises(InvalidInput) as caught:
                costs_for_chain(chain_workload(), _HARDWARE, extra)
            self.assertIn("c9", str(caught.exception))

    def test_an_operation_without_a_weight_is_rejected(self) -> None:
        layers = {"c1": measured_layer(2.0, 3.0), "add": measured_layer(1.0, 1.0)}
        with self.assertRaises(Unsupported) as caught:
            costs_for_chain(add_tail_workload(), _HARDWARE, layers)
        message = str(caught.exception)
        self.assertIn("add", message)
        self.assertIn("恰好一份权重", message)

    def test_a_shared_weight_is_rejected(self) -> None:
        layers = {"c1": measured_layer(2.0, 3.0), "c2": measured_layer(2.0, 3.0)}
        with self.assertRaises(Unsupported) as caught:
            costs_for_chain(shared_weight_workload(), _HARDWARE, layers)
        self.assertIn("W1", str(caught.exception))


class WeightsCopyableTests(unittest.TestCase):
    def test_a_cost_without_a_copyable_weight_is_rejected(self) -> None:
        """``Costs.h2d`` naming a weight the mapspace cannot copy is a scenario bug
        that would otherwise surface much later, as an operation that never runs."""
        workload = chain_workload()
        costs = chain_costs()
        good = scenario_for(workload, costs)
        assert_weights_copyable(good)  # the good case stays silent

        hobbled = Scenario(
            schema_version=good.schema_version,
            id=good.id,
            workload=workload,
            architecture=good.architecture,
            costs=costs,
            mapspace=Mapspace(
                compute_device="gpu",
                copy_tensor_ids=("W1", "W2"),  # W3 is missing
                allow_copy_compute_overlap=True,
                allow_eviction=True,
            ),
            mapper=good.mapper,
        )
        with self.assertRaises(InvalidInput) as caught:
            assert_weights_copyable(hobbled)
        self.assertIn("W3", str(caught.exception))


# ---------------------------------------------------------------------------
# costs_from_model_config: takes the hardware, refuses the rest
# ---------------------------------------------------------------------------


def _config(**overrides: object) -> ModelConfig:
    hardware = overrides.pop("hardware", _HARDWARE)
    policy = overrides.pop("policy", PolicySpec(window_size=2))
    state = overrides.pop("state", None)
    return ModelConfig(
        name="three-layer",
        layers=tuple(measured_layer(2.0, 3.0, name=f"layer{index}") for index in (1, 2, 3)),
        hardware=hardware,  # type: ignore[arg-type]
        policy=policy,  # type: ignore[arg-type]
        state=state,  # type: ignore[arg-type]
    )


class CostsFromModelConfigTests(unittest.TestCase):
    def test_a_config_with_no_extra_services_is_accepted(self) -> None:
        costs = costs_from_model_config(
            _config(), chain_workload(), measured_layers_by_operation()
        )
        self.assertEqual(costs.source, SOURCE_LAYER_COSTS)
        self.assertEqual(costs.compute["c1"].duration_ns, 2_000_000)

    def test_window_size_is_not_read_here(self) -> None:
        """K belongs to the policy call, not to the cost table.

        ``window_mapping`` takes it as an argument, so the demo passes it at the
        call site and "which K ran" stays visible there. If the bridge ever began
        reading it, these two cost tables would stop being equal.
        """
        workload = chain_workload()
        layers = measured_layers_by_operation()
        small = costs_from_model_config(_config(policy=PolicySpec(window_size=1)), workload, layers)
        large = costs_from_model_config(_config(policy=PolicySpec(window_size=8)), workload, layers)
        self.assertEqual(small, large)

    def test_every_extra_service_is_named_and_rejected(self) -> None:
        workload = chain_workload()
        layers = measured_layers_by_operation()
        cases = {
            "cpu_effective_flops": {"hardware": dataclasses.replace(_HARDWARE, cpu_effective_flops=1.0)},
            "global_bytes": {"hardware": dataclasses.replace(_HARDWARE, global_bytes=32)},
            "workspace_bytes": {"hardware": dataclasses.replace(_HARDWARE, workspace_bytes=8)},
            "kv_bytes": {"hardware": dataclasses.replace(_HARDWARE, kv_bytes=4096)},
            "activation_transfer_bytes": {
                "hardware": dataclasses.replace(_HARDWARE, activation_transfer_bytes=16)
            },
            "host_staging_bandwidth_bytes_per_s": {
                "hardware": dataclasses.replace(
                    _HARDWARE, host_staging_bandwidth_bytes_per_s=1.0
                )
            },
            "state_transfer_bandwidth_bytes_per_s": {
                "hardware": dataclasses.replace(
                    _HARDWARE, state_transfer_bandwidth_bytes_per_s=1.0
                )
            },
            "kv_storage_cache_bytes": {
                "hardware": dataclasses.replace(_HARDWARE, kv_storage_cache_bytes=1024)
            },
            "static_gpu_layers": {"policy": PolicySpec(window_size=2, static_gpu_layers=1)},
            "token_pipeline_overhead_s": {
                "policy": PolicySpec(window_size=2, token_pipeline_overhead_s=1e-3)
            },
            "layer_scheduler_overhead_s": {
                "policy": PolicySpec(window_size=2, layer_scheduler_overhead_s=1e-4)
            },
            "state": {
                "state": StateSpec(
                    context_tokens=8,
                    attention_kv_head_count=2,
                    attention_key_length=4,
                    attention_value_length=4,
                )
            },
        }
        for field_name, kwargs in cases.items():
            with self.subTest(service=field_name):
                with self.assertRaises(Unsupported) as caught:
                    costs_from_model_config(_config(**kwargs), workload, layers)
                message = str(caught.exception)
                self.assertIn("未迁移", message)
                # The offending field is named, so the fix is obvious.
                self.assertIn(field_name, message)

    def test_the_rejection_lists_every_gap_at_once(self) -> None:
        """One pass should show the whole remaining to-do, not one item at a time."""
        hardware = dataclasses.replace(_HARDWARE, kv_bytes=1, global_bytes=1)
        policy = PolicySpec(window_size=2, static_gpu_layers=1)
        with self.assertRaises(Unsupported) as caught:
            costs_from_model_config(
                _config(hardware=hardware, policy=policy),
                chain_workload(),
                measured_layers_by_operation(),
            )
        message = str(caught.exception)
        for field_name in ("kv_bytes", "global_bytes", "static_gpu_layers"):
            self.assertIn(field_name, message)


# ---------------------------------------------------------------------------
# The kernel stands on its own
# ---------------------------------------------------------------------------

_ISOLATED_KERNEL = textwrap.dedent(
    """
    import sys

    BLOCKED = "tensor_mapping"


    class Blocker:
        \"\"\"Fail loudly if anything reaches for the mapping package.\"\"\"

        def find_spec(self, name, path=None, target=None):
            if name == BLOCKED or name.startswith(BLOCKED + "."):
                raise AssertionError("llm_infer_model.tensor imported " + name)
            return None


    sys.meta_path.insert(0, Blocker())

    # Everything below is built from the kernel's own vocabulary: the point is
    # that the kernel needs nothing outside itself to be driven.
    from llm_infer_model.tensor import (
        ACTION_ADVANCE, ACTION_COMPUTE, ACTION_COPY_H2D, ACTION_EVICT,
        REASON_GOAL_REACHED, STATUS_VALID, Action, Architecture, ComputeCost,
        Costs, DTYPE_F32, H2DCost, LOC_DRAM, LOC_VRAM, MapperConfig, Mapspace,
        Operation, Origin, ROLE_INPUT, ROLE_INTERMEDIATE, ROLE_OUTPUT,
        ROLE_WEIGHT, SEMANTIC_MUL_MAT, Scenario, Tensor, Workload,
        evaluate_mapping, initial_state, is_goal, legal_actions, transition,
    )


    def weight(index):
        return Tensor(
            id="W%d" % index, name="blk.%d.weight" % index, role=ROLE_WEIGHT,
            dtype=DTYPE_F32, ne=(4, 4, 1, 1), nb=(4, 16, 64, 64), storage_bytes=64,
            initial_locations=(LOC_DRAM,),
        )


    def activation(tensor_id, role, initial=()):
        return Tensor(
            id=tensor_id, name=tensor_id, role=role, dtype=DTYPE_F32,
            ne=(4, 1, 1, 1), nb=(4, 16, 16, 16), storage_bytes=16,
            initial_locations=initial,
        )


    def mul_mat(operation_id, weight_id, activation_in, output):
        return Operation(
            id=operation_id, semantic_op=SEMANTIC_MUL_MAT, ggml_op="GGML_OP_MUL_MAT",
            unary_op=None, op_params=(), inputs=(weight_id, activation_in), output=output,
        )


    workload = Workload(
        id="three-layer-chain",
        origin=Origin(kind="synthetic_fixture"),
        tensors=(
            weight(1), weight(2), weight(3),
            activation("x", ROLE_INPUT, (LOC_VRAM,)),
            activation("h1", ROLE_INTERMEDIATE),
            activation("h2", ROLE_INTERMEDIATE),
            activation("y", ROLE_OUTPUT),
        ),
        operations=(
            mul_mat("c1", "W1", "x", "h1"),
            mul_mat("c2", "W2", "h1", "h2"),
            mul_mat("c3", "W3", "h2", "y"),
        ),
        outputs=("y",),
    )

    scenario = Scenario(
        schema_version="0.1",
        id="three-layer-K1",
        workload=workload,
        architecture=Architecture(vram_capacity_bytes=160),
        costs=Costs(
            source="hand_written",
            compute={("c%d" % i): ComputeCost(2_000_000, 0) for i in (1, 2, 3)},
            h2d={("W%d" % i): H2DCost(3_000_000) for i in (1, 2, 3)},
        ),
        mapspace=Mapspace(
            compute_device="gpu", copy_tensor_ids=("W1", "W2", "W3"),
            allow_copy_compute_overlap=True, allow_eviction=True,
        ),
        mapper=MapperConfig(algorithm="uniform_cost"),
    )

    # 1. The state stepper works with no mapping package present.
    state = initial_state(scenario)
    assert not is_goal(scenario, state)
    first = legal_actions(scenario, state)
    assert first, "no legal action in the initial state"
    transition(scenario, state, first[0])

    # 2. A hand-written K=1 plan replays: load/compute/release each layer in turn.
    #    The intermediates are kept rather than released, so the peak is 128 B,
    #    above the window policy's 112 B for the same graph -- the release choices
    #    are exactly what the two plans differ in.
    plan = (
        Action(ACTION_COPY_H2D, "W1"), Action(ACTION_ADVANCE),
        Action(ACTION_COMPUTE, "c1"), Action(ACTION_ADVANCE),
        Action(ACTION_EVICT, "W1"),
        Action(ACTION_COPY_H2D, "W2"), Action(ACTION_ADVANCE),
        Action(ACTION_COMPUTE, "c2"), Action(ACTION_ADVANCE),
        Action(ACTION_EVICT, "W2"),
        Action(ACTION_COPY_H2D, "W3"), Action(ACTION_ADVANCE),
        Action(ACTION_COMPUTE, "c3"), Action(ACTION_ADVANCE),
    )
    result = evaluate_mapping(scenario, plan)
    assert result.status == STATUS_VALID, (result.status, result.reason)
    assert result.reason == REASON_GOAL_REACHED, result.reason
    assert result.makespan_ns == 15_000_000, result.makespan_ns
    assert result.h2d_bytes == 192, result.h2d_bytes
    assert result.peak_vram_bytes == 128, result.peak_vram_bytes
    assert len(result.events) == len(plan), len(result.events)

    assert BLOCKED not in sys.modules, [
        name for name in sys.modules if name.startswith(BLOCKED)
    ]
    print("ok", result.makespan_ns, result.h2d_bytes, result.peak_vram_bytes, len(result.events))
    """
)


class KernelIndependenceTests(unittest.TestCase):
    """§6-C: the public API imports, steps and replays with ``mapping`` absent.

    The blocker sits on ``sys.meta_path`` rather than relying on the package being
    uninstalled, so the test pins the *import graph* and keeps holding in a
    checkout where both packages are installed side by side.
    """

    def test_the_public_api_replays_a_plan_without_the_mapping_package(self) -> None:
        environment = dict(os.environ)
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(MODELING_ROOT) if not existing else str(MODELING_ROOT) + os.pathsep + existing
        )
        completed = subprocess.run(
            [sys.executable, "-c", _ISOLATED_KERNEL],
            cwd=MODELING_ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"stdout={completed.stdout!r}\nstderr={completed.stderr!r}",
        )
        self.assertEqual(completed.stdout.strip(), "ok 15000000 192 128 14")


if __name__ == "__main__":
    unittest.main()
