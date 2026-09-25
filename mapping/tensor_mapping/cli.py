"""Command-line entry points: ``python -m tensor_mapping model|search``.

``DESIGN.md`` §8 defines exactly two commands::

    python -m tensor_mapping model  --scenario S.json --mapping M.json --out DIR
    python -m tensor_mapping search --scenario S.json --out DIR

Both are thin orchestration over the core. ``model`` loads a scenario and a
mapping document and hands the actions to
:func:`~tensor_mapping.engine.evaluate_mapping`; ``search`` hands the scenario to
:func:`~tensor_mapping.mapper.search`. Neither calls ``legal_actions`` or
``transition`` itself and neither re-checks legality, because a second opinion
about what is legal is the one thing ``DESIGN.md`` §4.3 forbids -- it would
eventually disagree with the first, and the disagreement would show up as two
tools that accept different mappings.

**Exit codes.** ``DESIGN.md`` does not specify any, so this round fixes them and
``README.md`` records them:

======  =====================================================================
``0``   a usable result: ``model`` reached ``valid``; ``search`` returned
        ``optimal`` or ``feasible`` (a ``feasible`` result is still a
        replayable mapping)
``1``   the input was refused: unreadable file, malformed JSON, a shape the
        schema does not allow, ``unsupported`` semantics, or a mapping that
        belongs to a different scenario
``2``   the run happened and produced no usable mapping: ``invalid_mapping``,
        ``incomplete_mapping`` or ``initial_capacity_exceeded`` for ``model``;
        ``infeasible`` or ``unknown`` for ``search``
======  =====================================================================

1 and 2 are kept apart so a shell can tell "you gave me the wrong input" from
"the answer is that it cannot be done". That is the same distinction
``DESIGN.md`` §8:270 draws when it files ``initial_capacity_exceeded`` under the
verdict rather than beside the JSON and shape errors, and the one
``ACCEPTANCE.md`` §6:110 draws when it forbids reporting a budget cut-off as
infeasibility.

**Where artefacts go.** Exit 1 writes nothing at all. No run happened, and a
directory holding half a ``stats.json`` -- identity keys and no fingerprints --
is worse than an empty one, because the next reader cannot tell it apart from a
completed run. Exit 0 and exit 2 both write the same filenames, so a consumer
never has to distinguish "file missing" from "genuinely no actions"
(``DESIGN.md`` §8:266-268).

**Streams.** The one-line summary of what happened goes to stdout, for both a
good and a bad verdict -- an exit-2 run is a result, not a crash. Diagnostics
(error code, action index, simulated time, and why nothing could be read) go to
stderr, so ``python -m tensor_mapping ... > /dev/null`` still shows the failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, NoReturn, Sequence

from .artifacts import (
    MODE_MODEL,
    MODE_SEARCH,
    events_document,
    limits_block,
    mapping_document,
    mapping_note,
    search_block,
    stats_document,
    states_document,
    write_json,
)
from .engine import (
    MappingDocument,
    MappingIdentityError,
    REASON_INITIAL_CAPACITY_EXCEEDED,
    STATUS_VALID,
    evaluate_mapping,
    load_mapping,
)
from .mapper import STATUS_FEASIBLE, STATUS_OPTIMAL, SearchResult, search
from .spec import (
    InitialCapacityExceeded,
    Scenario,
    SpecError,
    check_initial_capacity,
    load_and_validate,
)

__all__ = [
    "EVENTS_FILE",
    "EXIT_INPUT_REFUSED",
    "EXIT_NO_USABLE_MAPPING",
    "EXIT_OK",
    "MAPPING_FILE",
    "STATES_FILE",
    "STATS_FILE",
    "main",
]

EXIT_OK = 0
EXIT_INPUT_REFUSED = 1
EXIT_NO_USABLE_MAPPING = 2

STATS_FILE = "stats.json"
EVENTS_FILE = "events.json"
STATES_FILE = "states.json"
MAPPING_FILE = "mapping.json"

# The status a refusal reports when the mapping is legal but belongs elsewhere.
# Taken from the exception class so the two cannot drift apart.
STATUS_IDENTITY_MISMATCH = MappingIdentityError.code


class _Refusal(Exception):
    """An input-level refusal: the run never started, so nothing is written.

    ``status`` uses the vocabulary of ``DESIGN.md`` §8:270, so what lands on
    stderr is the same word an artefact would have carried had the run been
    possible at all.
    """

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class _ArgumentParser(argparse.ArgumentParser):
    """An ``ArgumentParser`` that reports usage errors as a return code, not a signal.

    ``argparse`` exits with status **2** from ``error()``, which is exactly the
    code this CLI uses for "the run happened and there is no usable mapping". A
    typo in an argument would therefore be indistinguishable from
    ``infeasible`` to any script checking ``$?``. Raising instead lets ``main``
    map it to exit 1 with everything else about bad input.

    ``--help`` still exits 0: it raises ``SystemExit`` from ``exit()``, which is
    a different path and is meant to leave the process immediately.
    """

    def error(self, message: str) -> NoReturn:
        raise _UsageError(f"{self.prog}: {message}")


class _UsageError(Exception):
    """A command line this tool cannot even parse."""


def _non_negative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value < 0:
        raise argparse.ArgumentTypeError(
            f"{text!r} is negative; 0 means unlimited, so a budget is never negative"
        )
    return value


def _non_negative_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if value < 0:
        raise argparse.ArgumentTypeError(
            f"{text!r} is negative; 0 means unlimited, so a budget is never negative"
        )
    return value


def _build_parser() -> _ArgumentParser:
    parser = _ArgumentParser(
        prog="python -m tensor_mapping",
        description="Evaluate a fixed execution mapping, or search for the best one.",
    )
    commands = parser.add_subparsers(dest="mode", required=True)

    model = commands.add_parser(
        "model",
        help="replay a fixed mapping and report what happens",
        description=(
            "Replay a fixed mapping against a scenario and write the resulting "
            "statistics, events and states."
        ),
    )
    model.add_argument(
        "--scenario", required=True, metavar="S.json", help="scenario document"
    )
    model.add_argument(
        "--mapping", required=True, metavar="M.json", help="mapping document to replay"
    )
    model.add_argument(
        "--out", required=True, metavar="DIR", help="directory for the artefacts"
    )

    find = commands.add_parser(
        "search",
        help="search for a minimum-makespan mapping",
        description=(
            "Search for a minimum-makespan mapping and write the statistics, "
            "events, states and the mapping itself."
        ),
    )
    find.add_argument(
        "--scenario", required=True, metavar="S.json", help="scenario document"
    )
    find.add_argument(
        "--out", required=True, metavar="DIR", help="directory for the artefacts"
    )
    find.add_argument(
        "--max-expanded-states",
        type=_non_negative_int,
        default=None,
        metavar="N",
        help=(
            "override the scenario's expansion budget (0 means unlimited); a "
            "budget that runs out yields `unknown`, never `infeasible`"
        ),
    )
    find.add_argument(
        "--wall-time-limit-s",
        type=_non_negative_float,
        default=None,
        metavar="F",
        help="override the scenario's wall-clock budget (0 means unlimited)",
    )
    return parser


# ---------------------------------------------------------------------------
# Loading, and refusing to load
# ---------------------------------------------------------------------------


def _load_scenario(path_text: str) -> Scenario:
    """Load a scenario, turning the core's refusals into one-line diagnostics.

    ``InitialCapacityExceeded`` is a ``SpecError`` subclass but is *not* handled
    here: ``load_and_validate`` never calls ``check_initial_capacity``, because
    an unstartable scenario is a verdict about the scenario rather than a broken
    file (``DESIGN.md`` §8:270). It is raised at evaluation time and both entry
    points already convert it to ``infeasible``, so by the time this function
    returns, the question of capacity is the core's to answer. The CLI only
    re-runs the check later, for its message (:func:`_initial_capacity_message`).
    """
    try:
        return load_and_validate(path_text)
    except SpecError as exc:
        raise _Refusal(exc.code, f"{path_text}: {exc}") from exc


def _load_mapping_document(path_text: str, scenario: Scenario) -> MappingDocument:
    try:
        return load_mapping(path_text, scenario)
    except MappingIdentityError as exc:
        # Not an `invalid_mapping` verdict. The mapping may be perfectly
        # well-formed and legal; it is a mapping of a different workload, so what
        # has to change is the input pair, not the plan. Refusing the input is
        # therefore the actionable answer (engine.MappingIdentityError).
        raise _Refusal(STATUS_IDENTITY_MISMATCH, f"{path_text}: {exc}") from exc
    except SpecError as exc:
        raise _Refusal(exc.code, f"{path_text}: {exc}") from exc


def _prepare_out_dir(path_text: str) -> Path:
    """Create the output directory if needed, and check that it is one.

    Only the files this tool owns are ever written here; nothing existing is
    removed. ``DESIGN.md`` §8 does not authorise deleting a previous run's
    artefacts, and a mistyped ``--out`` should not be destructive.
    """
    path = Path(path_text)
    # Checked before `mkdir`: on Windows a directory that is really a file makes
    # `mkdir(parents=True, exist_ok=True)` raise `FileExistsError`, whose message
    # ("cannot create a file when that file already exists") says nothing about
    # what the user actually got wrong.
    if path.exists() and not path.is_dir():
        raise _Refusal(
            "invalid_input",
            f"cannot use --out {path_text}: it exists and is not a directory",
        )
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise _Refusal("invalid_input", f"cannot use --out {path_text}: {exc}") from exc
    return path


# ---------------------------------------------------------------------------
# Reading the evaluator's verdict
# ---------------------------------------------------------------------------


def _detail(reason: str, code: str | None) -> str:
    """The prose half of the evaluator's ``reason`` string.

    ``_failure`` (``engine.py``) builds ``reason`` as ``f"{token}: {message}"``,
    using ``error_code`` when there is one and the reason token otherwise. Both
    halves are the model's own words, so splitting on the *first* ``": "``
    restores ``message`` verbatim rather than composing new prose here. A reason
    with no separator is returned whole.
    """
    prefix, separator, rest = reason.partition(": ")
    if not separator:
        return reason
    if code is not None and prefix == code:
        return rest
    if code is None and " " not in prefix:
        return rest
    return reason


def _initial_capacity_message(scenario: Scenario) -> str | None:
    """The detail for an initial-capacity refusal, or ``None`` if that is not the verdict.

    The evaluator reports this case as a bare ``reason="initial_capacity_exceeded"``
    with no message (``engine.evaluate_mapping``), and it does not keep the byte
    counts the check computed. Re-running the check purely to harvest its
    message is safe -- the same pure function on the same scenario must reach
    the same verdict -- and it is used for prose only: the verdict still comes
    from the evaluator and is never recomputed here.
    """
    try:
        check_initial_capacity(scenario)
    except InitialCapacityExceeded as exc:
        return str(exc)
    return None


def _error_block(
    *, code: str | None, action_index: int | None, t_ns: int | None, message: str
) -> dict[str, Any]:
    """The structured diagnosis ``DESIGN.md`` §8:272 asks for.

    Emitted whenever the run did not produce a usable mapping, which is *not* the
    same as "``error_code`` is set". ``incomplete_mapping`` leaves
    ``error_code`` as ``None`` while ``error_action_index`` and ``error_t_ns``
    are exactly its diagnosis -- the mapping simply ran out, and where. Gating on
    the code would throw those two numbers away in the one case that has nothing
    else to say.
    """
    return {
        "code": code,
        "action_index": action_index,
        "t_ns": t_ns,
        "message": message,
    }


def _empty_note(*, what: str, status: str, reason: str) -> str:
    """Why a log is empty, for the reader who opens this file on its own.

    An empty list that carries no explanation is the most misleading thing an
    artefact can contain: it looks like a run that did nothing, when in fact
    there was no mapping to do anything with.
    """
    return (
        f"no action was applied in this run, so there is no {what} to record. "
        f"status={status!r}, reason={reason!r}; the verdict is in stats.json."
    )


# ---------------------------------------------------------------------------
# The two commands
# ---------------------------------------------------------------------------


def _run_model(args: argparse.Namespace) -> int:
    scenario = _load_scenario(args.scenario)
    document = _load_mapping_document(args.mapping, scenario)
    out_dir = _prepare_out_dir(args.out)

    evaluation = evaluate_mapping(scenario, document.actions)
    ok = evaluation.status == STATUS_VALID

    if ok:
        # No `error` block on a usable run. `EvaluationResult.error_code` is None
        # here anyway, but the block is gated on the verdict rather than on the
        # code so that the two can never disagree about whether something went
        # wrong.
        error = None
    else:
        message = _detail(evaluation.reason, evaluation.error_code)
        if evaluation.error_code == REASON_INITIAL_CAPACITY_EXCEEDED:
            message = _initial_capacity_message(scenario) or message
        error = _error_block(
            code=evaluation.error_code,
            action_index=evaluation.error_action_index,
            t_ns=evaluation.error_t_ns,
            message=message,
        )

    note = (
        _empty_note(what="event", status=evaluation.status, reason=evaluation.reason)
        if not evaluation.events
        else None
    )
    state_note = (
        _empty_note(what="state", status=evaluation.status, reason=evaluation.reason)
        if not evaluation.states
        else None
    )

    limits = limits_block(
        scenario,
        max_expanded_states=scenario.mapper.max_expanded_states,
        wall_time_limit_s=scenario.mapper.wall_time_limit_s,
    )
    stats = stats_document(
        mode=MODE_MODEL,
        scenario=scenario,
        status=evaluation.status,
        reason=evaluation.reason,
        makespan_ns=evaluation.makespan_ns,
        peak_vram_bytes=evaluation.peak_vram_bytes,
        h2d_bytes=evaluation.h2d_bytes,
        action_count=evaluation.action_count,
        # From the document, not from `evaluation.actions`: at an
        # initial-capacity refusal the evaluator returns before it has looked at
        # the mapping at all, so its action tuple is empty while the file the
        # user passed may have listed twenty.
        declared_action_count=len(document.actions),
        limits=limits,
        error=error,
        scenario_file=args.scenario,
        mapping_file=args.mapping,
    )

    write_json(out_dir / STATS_FILE, stats)
    write_json(
        out_dir / EVENTS_FILE,
        events_document(
            mode=MODE_MODEL, scenario=scenario, evaluation=evaluation, note=note
        ),
    )
    write_json(
        out_dir / STATES_FILE,
        states_document(
            mode=MODE_MODEL, scenario=scenario, evaluation=evaluation, note=state_note
        ),
    )

    summary = (
        f"model: {evaluation.status} ({evaluation.reason.split(':')[0]}) "
        f"makespan_ns={evaluation.makespan_ns} "
        f"peak_vram_bytes={evaluation.peak_vram_bytes} "
        f"h2d_bytes={evaluation.h2d_bytes} "
        f"action_count={evaluation.action_count} -> {out_dir}"
    )
    if ok:
        print(summary)
        return EXIT_OK

    _report_failure(
        summary,
        f"{evaluation.status}: code={evaluation.error_code} "
        f"action_index={evaluation.error_action_index} "
        f"t_ns={evaluation.error_t_ns}: {message}",
    )
    return EXIT_NO_USABLE_MAPPING


def _run_search(args: argparse.Namespace) -> int:
    scenario = _load_scenario(args.scenario)
    out_dir = _prepare_out_dir(args.out)

    result: SearchResult = search(
        scenario,
        max_expanded_states=args.max_expanded_states,
        wall_time_limit_s=args.wall_time_limit_s,
    )
    usable = result.status in (STATUS_OPTIMAL, STATUS_FEASIBLE)
    # The effective budget comes back on the result, not from the scenario: the
    # command line can override it, and `limits` must not contradict the `search`
    # block printed beside it.
    limits = limits_block(
        scenario,
        max_expanded_states=result.max_expanded_states,
        wall_time_limit_s=result.wall_time_limit_s,
    )
    evaluation = result.evaluation
    action_count = evaluation.action_count if evaluation is not None else None

    note = (
        _empty_note(
            what="event", status=result.status, reason=result.termination_reason
        )
        if evaluation is None or not evaluation.events
        else None
    )
    state_note = (
        _empty_note(
            what="state", status=result.status, reason=result.termination_reason
        )
        if evaluation is None or not evaluation.states
        else None
    )

    stats = stats_document(
        mode=MODE_SEARCH,
        scenario=scenario,
        status=result.status,
        reason=result.termination_reason,
        makespan_ns=result.makespan_ns,
        peak_vram_bytes=result.peak_vram_bytes,
        h2d_bytes=result.h2d_bytes,
        action_count=action_count,
        declared_action_count=len(result.actions),
        limits=limits,
        search=search_block(result),
        scenario_file=args.scenario,
    )
    write_json(out_dir / STATS_FILE, stats)
    write_json(
        out_dir / EVENTS_FILE,
        events_document(
            mode=MODE_SEARCH, scenario=scenario, evaluation=evaluation, note=note
        ),
    )
    write_json(
        out_dir / STATES_FILE,
        states_document(
            mode=MODE_SEARCH, scenario=scenario, evaluation=evaluation, note=state_note
        ),
    )

    # Only when there is a plan to replay. An empty action list is a structurally
    # legal mapping document, so writing one for `infeasible` would hand the
    # reader a file that looks like a plan and replays as `incomplete_mapping`.
    if evaluation is not None:
        write_json(
            out_dir / MAPPING_FILE,
            mapping_document(
                scenario=scenario,
                result=result,
                comment=mapping_note(result, scenario.id),
            ),
        )

    summary = (
        f"search: {result.status} ({result.termination_reason}) "
        f"makespan_ns={result.makespan_ns} "
        f"peak_vram_bytes={result.peak_vram_bytes} "
        f"h2d_bytes={result.h2d_bytes} "
        f"optimality_proven={result.optimality_proven} "
        f"expanded_states={result.expanded_states} "
        f"wall_time_s={result.wall_time_s:.6f} -> {out_dir}"
    )
    if usable:
        print(summary)
        return EXIT_OK

    _report_failure(
        summary,
        f"{result.status}: termination_reason={result.termination_reason} "
        f"optimality_proven={result.optimality_proven}; no mapping was produced",
    )
    return EXIT_NO_USABLE_MAPPING


def _report_failure(summary: str, diagnostic: str) -> None:
    print(summary)
    print(diagnostic, file=sys.stderr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and return its exit code. Never raises for bad input."""
    try:
        args = _build_parser().parse_args(argv)
    except _UsageError as exc:
        print(f"{exc}", file=sys.stderr)
        return EXIT_INPUT_REFUSED

    try:
        if args.mode == MODE_MODEL:
            return _run_model(args)
        return _run_search(args)
    except _Refusal as refusal:
        # Nothing has been written at this point: every refusal is raised while
        # loading input or preparing the output directory, before the first
        # write_json. Exit 1 therefore means "there are no artefacts".
        print(f"{refusal.status}: {refusal.message}", file=sys.stderr)
        return EXIT_INPUT_REFUSED
