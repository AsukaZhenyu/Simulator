"""The window policy, measured against the old model and against the search.

``ALIGNMENT_IMPLEMENTATION.md`` §6-B asks for a three-row table proving the old
window rule and the new kernel agree, and §6-C asks that a plan from the search,
replayed by the independent evaluator, produces the search's own numbers. Both
are here, with the expectations written as literals so the tables stay readable
and a change has to be deliberate.

The module is deliberately blunt about *which* failure a case is. Three things
can go wrong and they mean three different things:

* ``Unsupported`` -- the graph's shape is outside what the policy reads. The
  generic search is unaffected, so this is not a verdict about the scenario.
* ``WindowPolicyFailed`` -- the shape and K are fine, the policy just cannot get
  there from here. **Not** "infeasible": the tests below show the search solving
  the very scenario the policy declines.
* ``InvalidInput`` -- the argument itself is not a window size.

Nothing here re-implements a rule: every legality and timing number comes out of
the kernel's own evaluator.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

from llm_infer_model.model import HardwareSpec, LayerSpec, ModelConfig, PolicySpec
from llm_infer_model.simulator import simulate_decode
from llm_infer_model.tensor import (
    COPY_ABSENT,
    LOC_VRAM,
    InvalidInput,
    SpecError,
    Unsupported,
    evaluate_mapping,
    load_and_validate,
    load_mapping,
)
from llm_infer_model.tensor.layer_costs import SOURCE_LAYER_COSTS, costs_from_model_config

from tensor_mapping.mapper import (
    STATUS_INFEASIBLE,
    STATUS_OPTIMAL,
    TERMINATION_EXHAUSTED,
    TERMINATION_GOAL_POPPED,
    search,
)
from tensor_mapping.policies import (
    WindowPolicyFailed,
    chain_weight_ids,
    derive_chain,
    window_mapping,
)

from .support import (
    EXAMPLES,
    load_example,
    ms,
    with_capacity,
    with_resident_weights,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CHAIN = "three_layer_chain-cap1024"

# The three layers of the chain example, in the units the old model takes.
FLOPS_PER_LAYER = 32.0  # 4x4 weight on a 4-vector = 16 MACs
WEIGHT_BYTES = 64
COMPUTE_MS = 2.0
TRANSFER_MS = 3.0

# 32 FLOP / 16000 FLOP-per-s = 2 ms; 1 ms latency + 64 B / 32 kB-per-s = 3 ms.
_OLD_HARDWARE = HardwareSpec(
    gpu_effective_flops=16_000.0,
    h2d_bandwidth_bytes_per_s=32_000.0,
    h2d_latency_s=1e-3,
    vram_capacity_bytes=1024,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _layers(scenario) -> dict[str, LayerSpec]:
    """One LayerSpec per operation, using the closed form (no measured overrides)."""
    return {
        operation_id: LayerSpec(
            name=f"blk.{index}.attn_q", weight_bytes=WEIGHT_BYTES, flops=FLOPS_PER_LAYER
        )
        for index, operation_id in enumerate(sorted(scenario.workload.operation_by_id))
    }


def _old_config(scenario, window_size: int) -> ModelConfig:
    """The v0.8 config equivalent to ``scenario`` at window ``window_size``.

    The layer list follows the *chain* order, not the file order, so that
    ``TraceEvent.layer`` indexes straight into :func:`derive_chain`'s links and
    the two event timelines can be compared item by item.
    """
    layers = _layers(scenario)
    chain = derive_chain(scenario.workload)
    return ModelConfig(
        name=scenario.id,
        layers=tuple(layers[link.operation_id] for link in chain),
        hardware=dataclasses.replace(
            _OLD_HARDWARE, vram_capacity_bytes=scenario.architecture.vram_capacity_bytes
        ),
        policy=PolicySpec(window_size=window_size),
    )


def _old_timeline(old, chain) -> list[tuple[str, str, int, int]]:
    """The old model's loads and computes, renamed to the kernel's vocabulary."""
    timeline = []
    for event in old.events:
        if event.task not in ("LOAD_WEIGHT", "COMPUTE"):
            continue
        link = chain[event.layer]
        timeline.append(
            (
                "COPY_H2D" if event.task == "LOAD_WEIGHT" else "COMPUTE",
                link.weight_id if event.task == "LOAD_WEIGHT" else link.operation_id,
                round(event.start_s * 1e9),
                round(event.end_s * 1e9),
            )
        )
    return timeline


def _new_timeline(result) -> list[tuple[str, str, int, int]]:
    """The kernel's loads and computes. EVICT and ADVANCE are bookkeeping the old
    model does not emit, so they are dropped here -- which is exactly the licence
    §6-B gives the new side."""
    return [
        (event.kind, event.target_id, event.start_ns, event.end_ns)
        for event in result.events
        if event.kind in ("COPY_H2D", "COMPUTE")
    ]


def _weight_peak_bytes(scenario, evaluation) -> int:
    """Peak bytes held by the chain's weights alone, over the whole replay.

    The kernel reports total occupancy (weights + activations + workspace); the
    old model's ``peak_streamed_weight_bytes`` counts only weights, so the two
    are compared through this instead of being forced equal.
    """
    weights = set(chain_weight_ids(scenario))
    peak = 0
    for snapshot in evaluation.states:
        held = sum(
            scenario.size_alloc(tensor_id)
            for tensor_id, status in snapshot.copy_status
            if tensor_id in weights and status != COPY_ABSENT
        )
        peak = max(peak, held)
    return peak


def _max_concurrent_weights(scenario, evaluation) -> int:
    weights = set(chain_weight_ids(scenario))
    return max(
        sum(
            1
            for tensor_id, status in snapshot.copy_status
            if tensor_id in weights and status != COPY_ABSENT
        )
        for snapshot in evaluation.states
    )


def _splice(scenario, *, outputs: tuple[str, ...] | None = None, **changes: str):
    """Swap inputs/outputs on a copy of the workload (no new fixture file).

    ``DESIGN.md`` §9 says not to add a near-identical example file for a
    mechanical variation, and these are exactly that: one edge moved. ``Workload``
    re-validates on construction, so a splice that orphans an operation is
    rejected here rather than producing a graph the policy would misread.
    """
    operations = tuple(
        dataclasses.replace(operation, **changes[operation.id])
        if operation.id in changes
        else operation
        for operation in scenario.workload.operations
    )
    workload = dataclasses.replace(
        scenario.workload,
        operations=operations,
        outputs=scenario.workload.outputs if outputs is None else outputs,
    )
    return dataclasses.replace(scenario, workload=workload)


# ---------------------------------------------------------------------------
# §6-B: the old rule and the new kernel agree
# ---------------------------------------------------------------------------


class OldModelAgreementTests(unittest.TestCase):
    """The three rows of §6-B, both columns, plus the K bound behind them.

    ``simulate_decode`` is the v0.8 rule the policy is meant to reproduce on this
    graph: same hardware, same closed-form layer costs, only the window differs.
    The resident row pairs with the old model's no-transfer branch
    (``window_size >= layer_count``) -- see :func:`support.with_resident_weights`.
    """

    # label, window K, resident?, old/new makespan ms, h2d bytes, weight peak B, total peak B
    ROWS = (
        ("streamed K=1", 1, False, 15.0, 192, 64, 112),
        ("streamed K=2", 2, False, 11.0, 192, 128, 176),
        ("resident K=3", 3, True, 6.0, 0, 192, 224),
    )

    def _scenario(self, resident: bool):
        scenario = load_example(CHAIN)
        return with_resident_weights(scenario) if resident else scenario

    def test_the_two_columns_agree_row_by_row(self) -> None:
        for label, window_size, resident, makespan_ms, h2d_bytes, weight_peak, _ in self.ROWS:
            scenario = self._scenario(resident)
            with self.subTest(row=label):
                old = simulate_decode(_old_config(scenario, window_size))
                new = evaluate_mapping(scenario, window_mapping(scenario, window_size))

                self.assertEqual(round(old.makespan_seconds * 1e9), ms(makespan_ms))
                self.assertEqual(new.makespan_ns, ms(makespan_ms), "new side makespan")
                self.assertEqual(old.bytes_transferred, h2d_bytes)
                self.assertEqual(new.h2d_bytes, h2d_bytes, "new side transfer")
                self.assertEqual(old.peak_streamed_weight_bytes, weight_peak)
                self.assertEqual(_weight_peak_bytes(scenario, new), weight_peak, "new side weight peak")

    def test_the_load_and_compute_intervals_line_up(self) -> None:
        """§6-B: compare the two timelines, event by event, in nanoseconds.

        Makespans agreeing could still hide a different schedule, so the old
        model's ``LOAD_WEIGHT``/``COMPUTE`` events are renamed to the kernel's
        vocabulary and compared as sequences. The new side's EVICT and ADVANCE
        entries are dropped, which is the licence §6-B gives it.
        """
        for label, window_size, resident, *_ in self.ROWS:
            scenario = self._scenario(resident)
            chain = derive_chain(scenario.workload)
            with self.subTest(row=label):
                old = simulate_decode(_old_config(scenario, window_size))
                new = evaluate_mapping(scenario, window_mapping(scenario, window_size))
                self.assertEqual(_old_timeline(old, chain), _new_timeline(new))

    def test_the_total_peak_is_not_the_weight_peak(self) -> None:
        """§6-B says the two must not be forced equal: activations live too."""
        for label, window_size, resident, _, _, weight_peak, total_peak in self.ROWS:
            scenario = self._scenario(resident)
            with self.subTest(row=label):
                result = evaluate_mapping(scenario, window_mapping(scenario, window_size))
                self.assertEqual(result.peak_vram_bytes, total_peak)
                self.assertGreater(
                    result.peak_vram_bytes, weight_peak, "activations must be counted"
                )
                self.assertLessEqual(
                    result.peak_vram_bytes, scenario.architecture.vram_capacity_bytes
                )

    def test_the_window_bound_holds_in_every_state(self) -> None:
        """K is a bound on *concurrently occupied* weights, in flight included."""
        for label, window_size, resident, _, _, _, _ in self.ROWS:
            scenario = self._scenario(resident)
            with self.subTest(row=label):
                result = evaluate_mapping(scenario, window_mapping(scenario, window_size))
                self.assertLessEqual(_max_concurrent_weights(scenario, result), window_size)

    def test_a_resident_weight_is_never_copied_in(self) -> None:
        scenario = with_resident_weights(load_example(CHAIN))
        actions = window_mapping(scenario, 3)
        self.assertFalse([action for action in actions if action.kind == "COPY_H2D"])
        self.assertEqual(evaluate_mapping(scenario, actions).h2d_bytes, 0)
        # And the activations that must already be there still are: nothing evicts
        # the workload's own input or output.
        kinds = {action.target_id for action in actions if action.kind == "EVICT"}
        self.assertEqual(kinds, {"W1", "W2", "h1"})


class PlanShapeTests(unittest.TestCase):
    """What the policy actually emits at K=1, spelled out.

    Pinned as a sequence because the priority order (release -> compute -> load ->
    wait) *is* the policy: the module docstring describes it, and this is the same
    statement in a form that fails when it changes.

    ``h1`` is released only once ``c2`` is done -- releasing it earlier is refused
    by the kernel (``CODE_TENSOR_IN_USE`` is not it; the tensor has a pending
    reader), which is why the plan has a second EVICT in the middle.
    """

    EXPECTED_K1 = (
        ("COPY_H2D", "W1"),
        ("ADVANCE", None),
        ("COMPUTE", "c1"),
        ("ADVANCE", None),
        ("EVICT", "W1"),
        ("COPY_H2D", "W2"),
        ("ADVANCE", None),
        ("COMPUTE", "c2"),
        ("ADVANCE", None),
        ("EVICT", "W2"),
        ("EVICT", "h1"),
        ("COPY_H2D", "W3"),
        ("ADVANCE", None),
        ("COMPUTE", "c3"),
        ("ADVANCE", None),
    )

    def test_k1_is_load_compute_release_per_layer(self) -> None:
        scenario = load_example(CHAIN)
        actions = window_mapping(scenario, 1)
        self.assertEqual(tuple((a.kind, a.target_id) for a in actions), self.EXPECTED_K1)
        self.assertEqual(evaluate_mapping(scenario, actions).makespan_ns, ms(15.0))

    def test_the_hand_computed_occupancy_matches_the_states(self) -> None:
        """Walk the plan by hand and compare with the kernel's own snapshots.

        16 B of input, +64 per weight, +16 per activation, -64 when a weight goes.
        Written out rather than derived so a kernel change to *when* a tensor is
        charged shows up here.
        """
        scenario = load_example(CHAIN)
        actions = window_mapping(scenario, 1)
        result = evaluate_mapping(scenario, actions)
        expected = [
            16,   # initial: x
            80,   # + W1
            80,
            96,   # + h1
            96,
            32,   # - W1
            96,   # + W2
            96,
            112,  # + h2
            112,
            48,   # - W2
            32,   # - h1
            96,   # + W3
            96,
            112,  # + y
            112,
        ]
        self.assertEqual([state.used_vram_bytes for state in result.states], expected)

    def test_k2_overlaps_the_next_load_with_the_current_compute(self) -> None:
        """At K=2 the second weight is fetched *while* c1 runs, which is where the
        11 ms (against K=1's 15 ms) comes from: 2 ms of the 3 ms copy hides behind
        the compute, and the remaining 1 ms is exposed."""
        scenario = load_example(CHAIN)
        actions = window_mapping(scenario, 2)
        self.assertEqual(len(actions), 15)
        result = evaluate_mapping(scenario, actions)
        self.assertEqual(result.makespan_ns, ms(11.0))
        self.assertEqual(_max_concurrent_weights(scenario, result), 2)

        events = {(event.kind, event.target_id): event for event in result.events}
        compute_c1 = events[("COMPUTE", "c1")]
        copy_w2 = events[("COPY_H2D", "W2")]
        self.assertEqual(copy_w2.start_ns, compute_c1.start_ns, "copy starts with the compute")
        self.assertGreater(copy_w2.end_ns, compute_c1.end_ns, "1 ms of the copy is exposed")
        self.assertEqual(copy_w2.end_ns - compute_c1.end_ns, ms(1.0))
        self.assertEqual(compute_c1.start_ns, ms(3.0))


# ---------------------------------------------------------------------------
# §6-B: the window policy's refusals are not verdicts
# ---------------------------------------------------------------------------


class WindowPreconditionTests(unittest.TestCase):
    def test_initial_occupancy_above_k_is_a_policy_failure(self) -> None:
        """A resident window that does not fit K is the policy's own limit."""
        scenario = with_resident_weights(load_example(CHAIN))
        for window_size in (1, 2):
            with self.subTest(window_size=window_size):
                with self.assertRaises(WindowPolicyFailed) as caught:
                    window_mapping(scenario, window_size)
                # *Which* failure this is matters: a policy limit is not a spec
                # error, and the message points at the alternatives.
                self.assertNotIsInstance(caught.exception, SpecError)
                self.assertIn("search", str(caught.exception))

    def test_the_search_solves_the_scenario_the_policy_declines(self) -> None:
        """The point of the previous test: the failure is the policy's, not the graph's."""
        scenario = with_resident_weights(load_example(CHAIN))
        with self.assertRaises(WindowPolicyFailed):
            window_mapping(scenario, 1)
        result = search(scenario)
        self.assertEqual(result.status, STATUS_OPTIMAL)
        self.assertEqual(result.makespan_ns, ms(6.0))

    def test_window_size_must_be_a_positive_int(self) -> None:
        scenario = load_example(CHAIN)
        for bad in (0, -1, True, False, 2.5, "2", None):
            with self.subTest(window_size=bad):
                with self.assertRaises(InvalidInput):
                    window_mapping(scenario, bad)  # type: ignore[arg-type]


class UnsupportedGraphTests(unittest.TestCase):
    """Shapes the policy refuses, and the search that still handles them.

    These are rejections of the *graph shape*, so they must be ``Unsupported``
    even though the search finds plans for the same scenarios.
    """

    def test_an_activation_read_by_two_operations_is_rejected(self) -> None:
        """The fan-out branch. No checked-in fixture reaches it: ``fork`` and
        ``residual`` both hit the per-operation check first (their ADD has two
        activations), so this shape is built by moving one edge of the chain.

        ``h2`` stays a requested output, otherwise ``c2`` would be orphaned and
        ``Workload``'s own validation would reject the splice before the policy
        ever saw it."""
        scenario = _splice(
            load_example(CHAIN),
            outputs=("y", "h2"),
            c3={"inputs": ("W3", "h1"), "output": "y"},
        )
        with self.assertRaises(Unsupported) as caught:
            derive_chain(scenario.workload)
        self.assertIn("分叉", str(caught.exception))
        self.assertIn("h1", str(caught.exception))

    def test_one_weight_used_twice_is_rejected(self) -> None:
        scenario = _splice(load_example(CHAIN), c2={"inputs": ("W1", "h1")})
        with self.assertRaises(Unsupported) as caught:
            derive_chain(scenario.workload)
        self.assertIn("共享权重", str(caught.exception))

    def test_a_residual_add_is_rejected_but_still_searchable(self) -> None:
        for name in ("fork", "residual"):
            with self.subTest(example=name):
                scenario = load_example(f"{name}-cap160")
                with self.assertRaises(Unsupported) as caught:
                    window_mapping(scenario, 2)
                self.assertIn("add", str(caught.exception))
                # The same scenario is solvable: the refusal is about the shape of
                # the window rule, not about feasibility.
                self.assertIsNotNone(search(scenario).evaluation)

    def test_a_single_matvec_is_a_one_link_chain(self) -> None:
        """The policy is not limited to multi-layer graphs."""
        scenario = load_example("matvec-cap160")
        chain = derive_chain(scenario.workload)
        self.assertEqual([link.weight_id for link in chain], ["W"])
        result = evaluate_mapping(scenario, window_mapping(scenario, 1))
        self.assertEqual(result.status, "valid")


# ---------------------------------------------------------------------------
# §6-B: the cost bridge feeds the policy from the old layer model
# ---------------------------------------------------------------------------


class CostBridgeTests(unittest.TestCase):
    def test_the_example_costs_are_what_the_layer_model_produces(self) -> None:
        """The fixture's hand-written numbers are the bridge's numbers.

        This is what makes the agreement table above meaningful: both columns eat
        the same cost table, one from the old simulator and one from the kernel.
        """
        scenario = load_example(CHAIN)
        layers = _layers(scenario)
        bridge = costs_from_model_config(
            _old_config(scenario, 1), scenario.workload, layers
        )
        self.assertEqual(bridge.source, SOURCE_LAYER_COSTS)
        for operation_id, cost in scenario.costs.compute.items():
            self.assertEqual(bridge.compute[operation_id].duration_ns, cost.duration_ns)
            self.assertEqual(bridge.compute[operation_id].workspace_bytes, cost.workspace_bytes)
        for tensor_id, cost in scenario.costs.h2d.items():
            self.assertEqual(bridge.h2d[tensor_id].duration_ns, cost.duration_ns)


# ---------------------------------------------------------------------------
# §6-C: search and evaluation share one set of rules
# ---------------------------------------------------------------------------


class SearchReplayTests(unittest.TestCase):
    def test_the_search_answer_replays_field_for_field(self) -> None:
        scenario = load_example(CHAIN)
        result = search(scenario)
        self.assertEqual(result.status, STATUS_OPTIMAL)
        self.assertEqual(result.termination_reason, TERMINATION_GOAL_POPPED)
        self.assertTrue(result.optimality_proven)

        replay = evaluate_mapping(scenario, result.actions)
        self.assertEqual(replay.makespan_ns, result.makespan_ns)
        self.assertEqual(replay.peak_vram_bytes, result.peak_vram_bytes)
        self.assertEqual(replay.h2d_bytes, result.h2d_bytes)
        self.assertEqual(replay.status, result.evaluation.status)
        self.assertEqual(replay.events, result.evaluation.events)
        self.assertEqual(replay.states, result.evaluation.states)

    def test_the_window_policy_is_optimal_at_k2_and_not_at_k1(self) -> None:
        """A useful cross-check: the search proves 11 ms is optimal, and the K=2
        policy plan *is* 11 ms, so the policy is optimal for this graph -- while
        K=1's 15 ms is measurably worse."""
        scenario = load_example(CHAIN)
        optimum = search(scenario).makespan_ns
        self.assertEqual(optimum, ms(11.0))
        self.assertEqual(evaluate_mapping(scenario, window_mapping(scenario, 2)).makespan_ns, optimum)
        self.assertGreater(
            evaluate_mapping(scenario, window_mapping(scenario, 1)).makespan_ns, optimum
        )

    def test_a_capacity_shortfall_is_refused_and_the_verdict_left_to_the_search(self) -> None:
        """§6-C wants an explicit refusal on a capacity shortfall.

        The floor is 96 B -- the last operation alone needs ``h2 + W3 + y`` live --
        so 95 B is genuinely infeasible. The policy runs out of legal actions and
        says so *with the numbers*, since "nothing is legal" on its own reads like
        a graph problem. The authoritative verdict is the search's, and it is
        ``infeasible``, which the policy deliberately does not claim.
        """
        scenario = load_example(CHAIN)
        squeezed = with_capacity(scenario, 95)
        with self.assertRaises(WindowPolicyFailed) as caught:
            window_mapping(squeezed, 2)
        message = str(caught.exception)
        self.assertIn("95 B", message)
        self.assertIn("不是「场景无解」", message)

        verdict = search(squeezed)
        self.assertEqual(verdict.status, STATUS_INFEASIBLE)
        self.assertEqual(verdict.termination_reason, TERMINATION_EXHAUSTED)
        self.assertIsNone(verdict.evaluation)

    def test_a_tight_but_solvable_capacity_still_yields_a_plan(self) -> None:
        """The policy is not the tightest possible planner, and the difference is
        visible here: its K=1 plan keeps ``x`` and both intermediates alive and
        needs 112 B, while the search's optimum fits in 96 B by evicting ``x``.
        At 128 B the policy plan is accepted unchanged."""
        scenario = with_capacity(load_example(CHAIN), 128)
        result = evaluate_mapping(scenario, window_mapping(scenario, 1))
        self.assertEqual(result.status, "valid")
        self.assertEqual(result.peak_vram_bytes, 112)
        self.assertGreater(112, 96, "the search's floor is lower than the policy's")
        self.assertEqual(search(with_capacity(load_example(CHAIN), 96)).status, STATUS_OPTIMAL)


# ---------------------------------------------------------------------------
# §6-C: the public API loads and replays with the mapping package absent
# ---------------------------------------------------------------------------

_ISOLATED_LOADER = textwrap.dedent(
    """
    import sys

    BLOCKED = "tensor_mapping"


    class Blocker:
        def find_spec(self, name, path=None, target=None):
            if name == BLOCKED or name.startswith(BLOCKED + "."):
                raise AssertionError("the kernel imported " + name)
            return None


    sys.meta_path.insert(0, Blocker())

    from llm_infer_model.tensor import (
        STATUS_VALID, evaluate_mapping, load_and_validate, load_mapping,
    )

    scenario = load_and_validate({scenario_path!r})
    mapping = load_mapping({mapping_path!r}, scenario)
    result = evaluate_mapping(scenario, mapping.actions)

    assert result.status == STATUS_VALID, (result.status, result.reason)
    assert result.makespan_ns == 8_000_000, result.makespan_ns
    assert result.peak_vram_bytes == 160, result.peak_vram_bytes
    assert result.h2d_bytes == 128, result.h2d_bytes
    assert len(result.events) == len(mapping.actions) == 10, len(result.events)
    # The checked-in plan is hand-written and carries no fingerprints by design.
    assert not mapping.fingerprint_verified

    assert BLOCKED not in sys.modules, [
        name for name in sys.modules if name.startswith(BLOCKED)
    ]
    print("ok", result.makespan_ns, result.peak_vram_bytes, result.h2d_bytes)
    """
)


class IsolatedLoaderTests(unittest.TestCase):
    """§6-C: ``load_and_validate`` / ``load_mapping`` / ``evaluate_mapping`` on
    their own, with ``tensor_mapping`` blocked on ``sys.meta_path``."""

    def test_a_checked_in_plan_loads_and_replays_without_the_mapping_package(self) -> None:
        environment = dict(os.environ)
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(REPO_ROOT) if not existing else str(REPO_ROOT) + os.pathsep + existing
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                _ISOLATED_LOADER.format(
                    scenario_path=str(EXAMPLES / "chain-cap160.json"),
                    mapping_path=str(EXAMPLES / "chain.mapping.json"),
                ),
            ],
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"stdout={completed.stdout!r}\nstderr={completed.stderr!r}",
        )
        self.assertEqual(completed.stdout.strip(), "ok 8000000 160 128")


if __name__ == "__main__":
    unittest.main()
