"""On-disk artefacts for the ``model`` and ``search`` entry points.

``DESIGN.md`` §8 names four files and says what each must carry::

    stats.json    mode, status, reason, fingerprints, cost source,
                  makespan_ns, peak_vram_bytes, h2d_bytes, action count,
                  limits and assumptions
    events.json   action index, action kind, related id, resource,
                  start/end times and byte counts
    states.json   the initial state and the state after every action
    mapping.json  search only: the replayable action list

Everything here is a *rendering* of decisions the core already made. This module
decides no legality, no timing and no optimality of its own: if a number is not
already on an :class:`~tensor_mapping.engine.EvaluationResult` or a
:class:`~tensor_mapping.mapper.SearchResult`, it does not belong in an artefact.
Three consequences are deliberate.

``mapping.json`` is built from ``SearchResult.actions`` -- the tuple the
evaluator actually replayed -- and never re-read from
``SearchResult.evaluation``. Deriving it the other way would turn "the search
reports what the evaluator produced" into a coincidence that a later edit could
break (``mapper.py``).

The ``comment`` on ``mapping.json`` is *generated* from the result rather than
written by hand. That file is the one artefact that gets copied elsewhere and
replayed by someone who will not have ``stats.json`` beside it, and its top
level cannot carry a status field (``load_mapping`` rejects unknown keys), so
the comment is the only place left that can say whether the action list is a
proven optimum or a plan that a budget cut short.

Each ``assumptions`` sentence is emitted only when the fact it asserts actually
holds -- the DRAM sentence only when DRAM really is unbounded, the cost sentence
only when the cost source really is synthetic. Flipping a scenario field
therefore changes the artefact, which is testable; a hardcoded paragraph of
caveats would silently go stale instead.

All four documents carry ``"schema_version": "0.1"`` (``DESIGN.md`` §4), and are
written with ``newline="\\n"`` -- see :func:`write_json`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .engine import (
    ACTION_ADVANCE,
    ACTION_COMPUTE,
    Action,
    EvaluationResult,
    Event,
    StateSnapshot,
)
from .mapper import SearchResult
from .spec import (
    ORIGIN_GGML,
    SCHEMA_VERSION,
    Architecture,
    Scenario,
    scenario_fingerprint,
    workload_fingerprint,
)

__all__ = [
    "MODE_MODEL",
    "MODE_SEARCH",
    "action_to_dict",
    "assumptions",
    "events_document",
    "limits_block",
    "mapping_document",
    "mapping_note",
    "search_block",
    "stats_document",
    "states_document",
    "write_json",
]

# `spec.py` has no constant for the cost source; it is a free string in the
# scenario document whose only checked-in value is this one.
COST_SOURCE_SYNTHETIC = "synthetic_fixed"

MODE_MODEL = "model"
MODE_SEARCH = "search"


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one artefact, deterministically and with LF line endings.

    ``newline="\\n"`` is load-bearing on Windows. In text mode Python translates
    every ``\\n`` into ``os.linesep``, so the artefacts would come out as CRLF --
    contradicting the repository's ``* text=auto eol=lf`` and making two runs of
    the same command differ for no real reason. The C++ exporter dodges the same
    trap with ``std::ios::binary``.

    Key order is the dict's insertion order, not ``sort_keys``: the documents are
    built from literal dicts so that the emitted files can be diffed against each
    other and against the hand-written examples, and so that a new key shows up
    as a readable one-line diff instead of a reshuffle.
    """
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


# ---------------------------------------------------------------------------
# Pieces shared by more than one document
# ---------------------------------------------------------------------------


def _identity(scenario: Scenario) -> dict[str, Any]:
    """The keys that let an artefact say what it is an artefact *of*.

    Every document gets these, not just ``stats.json``: an ``events.json``
    copied into a report on its own would otherwise carry no identity at all,
    which is the same argument the contract makes for putting fingerprints in
    the mapping (``DESIGN.md`` §4.3).
    """
    return {
        "scenario_id": scenario.id,
        "scenario_fingerprint": scenario_fingerprint(scenario),
        "workload_fingerprint": workload_fingerprint(scenario.workload),
    }


def _graph_origin(scenario: Scenario) -> dict[str, Any]:
    """Where the *graph* came from, kept separate from where the *costs* did.

    ``DESIGN.md`` §8 asks for the graph origin and the cost source to be
    disclosed separately, and ``ACCEPTANCE.md`` §7 reserves ``kind == "ggml"``
    for a real export. A ``costs.source`` field alone cannot tell a real export
    from a hand-written fixture, so this block is the only place that can.
    """
    origin = scenario.workload.origin
    return {
        "kind": origin.kind,
        "ggml_version": origin.ggml_version,
        "sample": origin.sample,
    }


def limits_block(
    scenario: Scenario,
    *,
    max_expanded_states: int,
    wall_time_limit_s: float,
) -> dict[str, Any]:
    """The configuration actually in force, as numbers.

    ``DESIGN.md`` §8's "限制与假设" is split in two: the *limits* are these
    numbers, derived from the scenario so that they cannot go stale, and the
    *assumptions* are prose (see :func:`assumptions`). A reader summing integers
    should never trip over a sentence, and a reader hunting for caveats should
    not have to know which integers are secretly caveats.

    The two budget values are passed in rather than read off ``scenario.mapper``
    because the search budget can be overridden on the command line. The caller
    passes the *effective* values -- for a search, the ones on its
    ``SearchResult`` -- so this block can never contradict the ``search`` block
    sitting next to it in ``stats.json``.
    """
    architecture: Architecture = scenario.architecture
    return {
        "vram_capacity_bytes": architecture.vram_capacity_bytes,
        "runtime_reserved_bytes": architecture.runtime_reserved_bytes,
        "allocation_alignment_bytes": architecture.allocation_alignment_bytes,
        "gpu_compute_slots": architecture.gpu_compute_slots,
        "h2d_copy_slots": architecture.h2d_copy_slots,
        # Always None in M0: a finite DRAM budget would need an eviction
        # decision the action space does not have (spec.Architecture).
        "dram_capacity_bytes": architecture.dram_capacity_bytes,
        "compute_device": scenario.mapspace.compute_device,
        "copy_tensor_ids": list(scenario.mapspace.copy_tensor_ids),
        "allow_eviction": scenario.mapspace.allow_eviction,
        "allow_copy_compute_overlap": scenario.mapspace.allow_copy_compute_overlap,
        "allow_recomputation": scenario.mapspace.allow_recomputation,
        "algorithm": scenario.mapper.algorithm,
        "max_expanded_states": max_expanded_states,
        "wall_time_limit_s": wall_time_limit_s,
    }


def assumptions(scenario: Scenario) -> list[str]:
    """M0's model caveats, each emitted only when it is true of this scenario.

    These are ``DESIGN.md`` §1's scope statements rather than anything derivable
    from a single field, so they live here and are checked against the scenario
    where they can be. M1 adds CPU math and bidirectional transfer; the sentences
    about the single copy resource and about the missing CPU path must change
    then, and the test that pins them will fail until they do.
    """
    architecture = scenario.architecture
    notes = [
        "costs are fixed per-operation durations; M0 has no bandwidth model and "
        "no calibration against hardware",
        "peak_vram_bytes and allocation_alignment_bytes are this model's own "
        "accounting, not a vendor allocator's",
    ]
    if architecture.dram_capacity_bytes is None:
        notes.append(
            "DRAM is modelled as an unbounded, read-only source store, which is "
            "what makes a weight reloadable after EVICT"
        )
    if architecture.gpu_compute_slots == 1 and architecture.h2d_copy_slots == 1:
        notes.append(
            "exactly one GPU compute resource and one H2D copy resource; no CPU "
            "math, no device-to-host copy and no SSD"
        )
    if scenario.costs.source == COST_SOURCE_SYNTHETIC:
        notes.append(
            "costs.source is synthetic_fixed: the durations were chosen for the "
            "acceptance tables, not measured on a GPU"
        )
    else:
        notes.append(
            f"costs.source is {scenario.costs.source!r}; M0 accepts only "
            "externally supplied fixed durations"
        )
    origin = scenario.workload.origin
    if origin.kind == ORIGIN_GGML:
        notes.append(
            f"the graph is a real GGML export (ggml {origin.ggml_version})"
        )
    else:
        notes.append(
            f"the graph is {origin.kind!r}, i.e. hand-written, not a real GGML "
            "export"
        )
    return notes


# ---------------------------------------------------------------------------
# events / states
# ---------------------------------------------------------------------------


def action_to_dict(action: Action) -> dict[str, Any]:
    """One action in the JSON spelling ``load_mapping`` reads back.

    ``Action`` stores a single ``target_id`` whatever the kind, but the document
    format splits it into ``tensor_id`` for COPY_H2D/EVICT and ``operation_id``
    for COMPUTE (``engine._parse_action``). Emitting ``target_id`` would produce
    a mapping this package cannot load.
    """
    if action.kind == ACTION_ADVANCE:
        return {"kind": ACTION_ADVANCE}
    if action.kind == ACTION_COMPUTE:
        return {"kind": ACTION_COMPUTE, "operation_id": action.target_id}
    return {"kind": action.kind, "tensor_id": action.target_id}


def _event_to_dict(event: Event) -> dict[str, Any]:
    """An event's fields, spelled out one by one.

    Not ``dataclasses.asdict``: the field names here are the contract's own
    vocabulary, and writing them out means a renamed dataclass field cannot
    silently rename an artefact key along with it.
    """
    return {
        "action_index": event.action_index,
        "kind": event.kind,
        "target_id": event.target_id,
        "resource": event.resource,
        "t_ns": event.t_ns,
        "start_ns": event.start_ns,
        "end_ns": event.end_ns,
        "bytes_transferred": event.bytes_transferred,
        "workspace_bytes": event.workspace_bytes,
        "bytes_released": event.bytes_released,
    }


def _state_to_dict(snapshot: StateSnapshot, action_index: int | None) -> dict[str, Any]:
    """One state summary, plus the action index that produced it.

    ``action_index`` is ``None`` for the initial state and ``i`` for the state
    after ``actions[i]``. ``StateSnapshot`` does not carry the index, and
    ``_snapshot`` receives ``index + 1`` rather than ``index`` (``engine.py``),
    so the writer computes it from the list position and never reads it off the
    snapshot. Trusting a field there would be off by one in a way that still
    looks plausible on every line.

    The status pairs are already sorted by id, so the dicts come out in the same
    order on every run.
    """
    return {
        "action_index": action_index,
        "t_ns": snapshot.t,
        "used_vram_bytes": snapshot.used_vram_bytes,
        "op_status": dict(snapshot.op_status),
        "copy_status": dict(snapshot.copy_status),
    }


def _log_document(
    *,
    mode: str,
    scenario: Scenario,
    key: str,
    entries: Sequence[Any],
    note: str | None,
) -> dict[str, Any]:
    document: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "mode": mode}
    document.update(_identity(scenario))
    if note is not None:
        document["note"] = note
    document[key] = list(entries)
    return document


def events_document(
    *,
    mode: str,
    scenario: Scenario,
    evaluation: EvaluationResult | None,
    note: str | None = None,
) -> dict[str, Any]:
    """``events.json``: one object per applied action, in action order.

    The list order *is* ``action_index`` order, which is what ``DESIGN.md`` §8
    means by "same-instant actions keep their indices".
    """
    events = () if evaluation is None else evaluation.events
    return _log_document(
        mode=mode,
        scenario=scenario,
        key="events",
        entries=[_event_to_dict(event) for event in events],
        note=note,
    )


def states_document(
    *,
    mode: str,
    scenario: Scenario,
    evaluation: EvaluationResult | None,
    note: str | None = None,
) -> dict[str, Any]:
    """``states.json``: the initial state, then one summary per action.

    ``states[0]`` is the initial state and ``states[i + 1]`` follows
    ``actions[i]``, so the list always has one more entry than there were
    actions applied (``engine.evaluate_mapping``).
    """
    snapshots = () if evaluation is None else evaluation.states
    entries = [
        _state_to_dict(snapshot, None if index == 0 else index - 1)
        for index, snapshot in enumerate(snapshots)
    ]
    return _log_document(
        mode=mode, scenario=scenario, key="states", entries=entries, note=note
    )


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def search_block(result: SearchResult) -> dict[str, Any]:
    """The budget accounting ``DESIGN.md`` §8 asks search to report.

    ``wall_time_s`` is a real measured duration and deliberately sits under this
    block rather than beside the simulated integers: ``DESIGN.md`` §7 warns that
    real wall time must not be confused with the model's ``t``, and the field
    name is the reminder.
    """
    return {
        "termination_reason": result.termination_reason,
        "optimality_proven": result.optimality_proven,
        "expanded_states": result.expanded_states,
        "visited_states": result.visited_states,
        "wall_time_s": result.wall_time_s,
        "max_expanded_states": result.max_expanded_states,
        "wall_time_limit_s": result.wall_time_limit_s,
    }


def stats_document(
    *,
    mode: str,
    scenario: Scenario,
    status: str,
    reason: str,
    makespan_ns: int | None,
    peak_vram_bytes: int | None,
    h2d_bytes: int | None,
    action_count: int | None,
    declared_action_count: int,
    limits: Mapping[str, Any],
    error: Mapping[str, Any] | None = None,
    search: Mapping[str, Any] | None = None,
    scenario_file: str | None = None,
    mapping_file: str | None = None,
) -> dict[str, Any]:
    """``stats.json``: the run's verdict, its numbers, and what was in force.

    ``action_count`` and ``declared_action_count`` are both reported because the
    core's ``EvaluationResult.action_count`` means different things on different
    paths: the number of actions *applied* when a mapping is rejected (it is set
    from the event log), and the number *declared* when the mapping is valid or
    merely ran out. On a rejected mapping those differ, and "the file lists ten
    actions and it died on the fourth" is exactly the diagnosis.

    ``action_count`` is ``None`` rather than ``0`` when nothing was applied at
    all, so that ``0`` never has to double as "absent".
    """
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "status": status,
        "reason": reason,
    }
    document.update(_identity(scenario))
    document["scenario_file"] = scenario_file
    if mapping_file is not None:
        document["mapping_file"] = mapping_file
    document["workload_id"] = scenario.workload.id
    document["graph_origin"] = _graph_origin(scenario)
    document["cost_source"] = scenario.costs.source
    document["makespan_ns"] = makespan_ns
    document["peak_vram_bytes"] = peak_vram_bytes
    document["h2d_bytes"] = h2d_bytes
    document["action_count"] = action_count
    document["declared_action_count"] = declared_action_count
    document["limits"] = dict(limits)
    document["assumptions"] = assumptions(scenario)
    if error is not None:
        document["error"] = dict(error)
    if search is not None:
        document["search"] = dict(search)
    return document


# ---------------------------------------------------------------------------
# mapping
# ---------------------------------------------------------------------------


def mapping_note(result: SearchResult, scenario_id: str) -> str:
    """The ``comment`` for a search-produced mapping, generated from the result.

    ``scenario_id`` is passed in rather than read off the result: ``SearchResult``
    carries the two fingerprints but not the id, so the caller supplies it from
    the scenario it just searched -- the same object the fingerprints came from.

    ``mapping.json`` cannot carry extra keys -- ``load_mapping`` rejects anything
    outside its allow-list -- so status, optimality and the budget live in
    ``stats.json``. But this file is the one that gets copied, committed and
    replayed by someone who has no ``stats.json``, and an unqualified action list
    reads as "the optimum" even when a budget cut the search short and left a
    merely ``feasible`` plan behind. So the comment states the strength of the
    claim, and the termination reason, in the one place that travels with it.
    """
    strength = (
        "a proven optimum (every mapping within the budget was no better)"
        if result.status == "optimal"
        else "NOT proven optimal: it is the best mapping found before the "
        "budget ran out"
    )
    return (
        f"Produced by `python -m tensor_mapping search` on scenario "
        f"{scenario_id!r}: {strength}. Status {result.status!r}, "
        f"termination_reason {result.termination_reason!r}. Search budget: "
        f"max_expanded_states={result.max_expanded_states} "
        f"(0 means unlimited), wall_time_limit_s={result.wall_time_limit_s} "
        f"(0 means unlimited)."
    )


def mapping_document(
    *, scenario: Scenario, result: SearchResult, comment: str
) -> dict[str, Any]:
    """``mapping.json``: the action list, its scenario, and both fingerprints.

    Both digests are written or neither is: ``load_mapping`` reports
    ``fingerprint_verified`` only when it saw both and they matched, and a
    partial claim is not a verification.

    No timestamp goes in here. The file has to be byte-identical between two
    runs of the same command, and it must never embed a filesystem path either --
    that is what makes two runs into different output directories comparable.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": scenario.id,
        "comment": comment,
        "workload_fingerprint": workload_fingerprint(scenario.workload),
        "scenario_fingerprint": scenario_fingerprint(scenario),
        "actions": [action_to_dict(action) for action in result.actions],
    }
