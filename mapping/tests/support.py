"""Shared helpers for the M0 test suite.

Two things live here that the tests would otherwise each re-invent:

* scenario variants -- most acceptance points differ from ``chain-cap160`` in
  exactly one number, and cloning the loaded scenario with
  :func:`dataclasses.replace` keeps that difference visible instead of burying
  it in a near-identical JSON file (``DESIGN.md`` §9 says not to pile up files
  for mechanical variations);
* :class:`Replay`, a hand-driven stepper, because most of the semantic table in
  ``ACCEPTANCE.md`` §5 reads "do these actions, then attempt this illegal one"
  and is far clearer as an explicit action list than as an offset into whatever
  mapping the search happens to return.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from tensor_mapping import (
    Action,
    Costs,
    Scenario,
    State,
    transition,
    used_vram_bytes,
)
from tensor_mapping.engine import (
    ACTION_ADVANCE,
    MappingError,
    initial_state,
)
from tensor_mapping.spec import (
    LOC_VRAM,
    ROLE_WEIGHT,
    ComputeCost,
    H2DCost,
    load_and_validate,
)

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

NS_PER_MS = 1_000_000


def ms(value: float) -> int:
    """Milliseconds to the integer nanoseconds the model speaks."""
    return int(round(value * NS_PER_MS))


def load_example(name: str) -> Scenario:
    """Load one of the checked-in example scenarios (e.g. ``"chain-cap160"``)."""
    return load_and_validate(EXAMPLES / f"{name}.json")


# ---------------------------------------------------------------------------
# Scenario variants
# ---------------------------------------------------------------------------


def with_capacity(scenario: Scenario, vram_capacity_bytes: int) -> Scenario:
    """The same scenario with a different VRAM budget."""
    return dataclasses.replace(
        scenario,
        id=f"{scenario.id.rsplit('-', 1)[0]}-cap{vram_capacity_bytes}",
        architecture=dataclasses.replace(
            scenario.architecture, vram_capacity_bytes=vram_capacity_bytes
        ),
    )


def with_alignment(scenario: Scenario, alignment_bytes: int) -> Scenario:
    return dataclasses.replace(
        scenario,
        architecture=dataclasses.replace(
            scenario.architecture, allocation_alignment_bytes=alignment_bytes
        ),
    )


def with_runtime_reserve(scenario: Scenario, reserve_bytes: int) -> Scenario:
    return dataclasses.replace(
        scenario,
        architecture=dataclasses.replace(
            scenario.architecture, runtime_reserved_bytes=reserve_bytes
        ),
    )


def with_overlap(scenario: Scenario, allowed: bool) -> Scenario:
    return dataclasses.replace(
        scenario,
        mapspace=dataclasses.replace(
            scenario.mapspace, allow_copy_compute_overlap=allowed
        ),
    )


def with_eviction(scenario: Scenario, allowed: bool) -> Scenario:
    return dataclasses.replace(
        scenario, mapspace=dataclasses.replace(scenario.mapspace, allow_eviction=allowed)
    )


def with_resident_weights(scenario: Scenario, tensor_ids: tuple[str, ...] | None = None) -> Scenario:
    """The same scenario with those weights already in VRAM.

    ``simulate_decode`` has a dedicated "everything resident" branch
    (``window_size >= layer_count``, ``simulator.py:90``) that charges no transfer
    at all. The tensor kernel has no such shortcut: residency is a property of the
    workload, so this is how the equivalent new-side scenario is written. Without
    an explicit ``tensor_ids`` every weight in the workload is made resident.
    """
    if tensor_ids is None:
        tensor_ids = tuple(
            tensor.id for tensor in scenario.workload.tensors if tensor.role == ROLE_WEIGHT
        )
    chosen = set(tensor_ids)
    tensors = tuple(
        dataclasses.replace(tensor, initial_locations=(LOC_VRAM,))
        if tensor.id in chosen
        else tensor
        for tensor in scenario.workload.tensors
    )
    return dataclasses.replace(
        scenario,
        id=f"{scenario.id}-resident",
        workload=dataclasses.replace(scenario.workload, tensors=tensors),
    )


def with_compute(
    scenario: Scenario, operation_id: str, duration_ms: float, workspace_bytes: int = 0
) -> Scenario:
    """Override one operation's cost."""
    compute = dict(scenario.costs.compute)
    compute[operation_id] = ComputeCost(
        duration_ns=ms(duration_ms), workspace_bytes=workspace_bytes
    )
    return dataclasses.replace(
        scenario,
        costs=Costs(source=scenario.costs.source, compute=compute, h2d=scenario.costs.h2d),
    )


def with_h2d(scenario: Scenario, tensor_id: str, duration_ms: float) -> Scenario:
    """Override one tensor's copy duration. Used to force simultaneous completions."""
    h2d = dict(scenario.costs.h2d)
    h2d[tensor_id] = H2DCost(duration_ns=ms(duration_ms))
    return dataclasses.replace(
        scenario,
        costs=Costs(source=scenario.costs.source, compute=scenario.costs.compute, h2d=h2d),
    )


# ---------------------------------------------------------------------------
# Hand-driven replay
# ---------------------------------------------------------------------------


def parse_action(spec: str | Action) -> Action:
    """``"COPY_H2D W1"`` -> Action. ``"ADVANCE"`` takes no target.

    An :class:`Action` passes through unchanged, so a helper that already builds
    actions (like a stored optimal mapping) can be spliced into a spec list.
    """
    if isinstance(spec, Action):
        return spec
    parts = spec.split()
    if not parts:
        raise ValueError("empty action spec")
    kind, rest = parts[0], parts[1:]
    if kind == ACTION_ADVANCE:
        if rest:
            raise ValueError(f"{ACTION_ADVANCE} takes no target: {spec!r}")
        return Action(kind, None)
    if len(rest) != 1:
        raise ValueError(f"expected exactly one target in {spec!r}")
    return Action(kind, rest[0])


def parse_actions(*specs: str | Action) -> tuple[Action, ...]:
    return tuple(parse_action(spec) for spec in specs)


class Replay:
    """Step a mapping by hand, one action at a time.

    ``do`` applies an action and lets a :class:`MappingError` propagate;
    ``refuse`` asserts that the action is rejected and returns the error, which
    is what the semantic table needs.
    """

    def __init__(self, scenario: Scenario, state: State | None = None) -> None:
        self.scenario = scenario
        self.state = initial_state(scenario) if state is None else state
        self.applied: list[Action] = []

    def do(self, *specs: str | Action) -> State:
        for action in parse_actions(*specs):
            self.state = transition(self.scenario, self.state, action)
            self.applied.append(action)
        return self.state

    def refuse(self, spec: str | Action) -> MappingError:
        """Apply nothing; assert the action is rejected and return the error."""
        action = parse_action(spec)
        try:
            transition(self.scenario, self.state, action)
        except MappingError as error:
            return error
        except Exception as exc:  # noqa: BLE001 - surface a wrong exception type loudly
            raise AssertionError(
                f"{spec} raised {type(exc).__name__} instead of MappingError: {exc}"
            ) from exc
        raise AssertionError(f"{spec} was accepted but should have been rejected")

    # Convenience accessors so tests read like the acceptance table.
    @property
    def t(self) -> int:
        return self.state.t

    def used_bytes(self) -> int:
        return used_vram_bytes(self.scenario, self.state)


# ---------------------------------------------------------------------------
# Raw document fixtures
# ---------------------------------------------------------------------------


def example_document(kind: str, name: str) -> dict[str, Any]:
    """A fresh copy of a checked-in document, ready to be mutated."""
    suffix = ".workload.json" if kind == "workload" else ".json"
    return json.loads((EXAMPLES / f"{name}{suffix}").read_text(encoding="utf-8"))


@contextmanager
def documents(
    workload: dict[str, Any] | str | None = None,
    scenario: dict[str, Any] | str | None = None,
    *,
    workload_name: str = "w",
    scenario_name: str = "s",
) -> Iterator[Path]:
    """Write a workload/scenario pair to a temp dir and yield the scenario path.

    Either argument may be a document dict or the name of an example to start
    from (``"chain"`` for the workload, ``"residual-cap160"`` for the
    scenario). Starting from a real example keeps the malformed documents in the
    validation tests recognisably close to the good ones, so the one broken
    field is the only thing that differs.
    """
    if isinstance(workload, str):
        workload = example_document("workload", workload)
    if isinstance(scenario, str):
        scenario = example_document("scenario", scenario)
    if workload is None:
        workload = example_document("workload", "chain")
    if scenario is None:
        scenario = example_document("scenario", "chain-cap160")
    workload = copy.deepcopy(workload)
    scenario = copy.deepcopy(scenario)
    scenario["workload_file"] = f"{workload_name}.json"

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as raw_dir:
        directory = Path(raw_dir)
        _write(directory / f"{workload_name}.json", workload)
        target = directory / f"{scenario_name}.json"
        _write(target, scenario)
        yield target


@contextmanager
def scenario_document(
    workload: dict[str, Any] | str | None = None,
    scenario: dict[str, Any] | str | None = None,
) -> Iterator[Scenario]:
    """A validated :class:`Scenario` built from documents that exist on disk.

    Some contracts can only be exercised through the loader -- the workload
    schema has no in-memory constructor for a raw document -- so these tests
    round-trip through real files rather than patching a loaded object.
    """
    with documents(workload, scenario) as path:
        yield load_and_validate(path)


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
