"""Exact mapping search for small GGML compute graphs under limited VRAM.

M0 answers one deliberately narrow question: given a fixed GGML DAG, one GPU
compute resource, one host-to-device copy resource, and a finite VRAM budget,
how should loads, computes and releases be ordered so that the requested outputs
become ready as early as possible? These graphs are far too small to be worth
running on hardware, so M0 searches them exhaustively and provably rather than
sampling schedules (``mapping/DESIGN.md`` §1).

The package is split along the contract's own seams:

``spec``
    Immutable contract objects, input validation, and fingerprints. Refuses
    anything M0 does not model instead of approximating it.
``engine``
    The state, the four actions, and fixed-mapping replay. The single
    implementation of the transition rules -- the search imports it rather than
    restating it.
``mapper``
    Uniform-cost search over the action space.
``artifacts``
    Result objects rendered as the four on-disk documents of ``DESIGN.md`` §8.
    Decides nothing: every number it writes is already on an evaluation or a
    search result.
``cli``
    The ``model`` and ``search`` entry points -- argument handling, exit codes,
    and the mapping from a core refusal to a one-line diagnostic.

The core is pure standard library, so the test suite runs with no installation
step (``mapping/DESIGN.md`` §2). Times are integers in nanoseconds; this is
deliberately unlike ``modeling/``, which works in float seconds.

Status
------
The M0 Python core, the GGML exporter under ``mapping/ggml/``, and the
``model``/``search`` entry points with their four artefacts are all implemented
and covered by tests. Every example fixture is still marked ``origin.kind =
"synthetic_fixture"`` except the four under ``examples/ggml/``, which are real
exporter output (``ACCEPTANCE.md`` §7 reserves ``"ggml"`` for exactly those).

Passing the tests is **not** M0 acceptance on its own: ``DESIGN.md`` §9 requires
the GGML build environment for the integration half of the verification, and
``ACCEPTANCE.md`` §7 lists the items that must each be answered before M0 can be
called complete.
"""

from __future__ import annotations

from .engine import (
    ACTION_ADVANCE,
    ACTION_COMPUTE,
    ACTION_COPY_H2D,
    ACTION_EVICT,
    Action,
    EvaluationResult,
    Event,
    MappingDocument,
    MappingError,
    MappingIdentityError,
    State,
    StateSnapshot,
    evaluate_mapping,
    initial_state,
    is_goal,
    legal_actions,
    load_mapping,
    transition,
    used_vram_bytes,
)
from .mapper import STATUS_FEASIBLE, STATUS_INFEASIBLE, STATUS_OPTIMAL, STATUS_UNKNOWN
from .mapper import SearchResult, search
from .spec import (
    SCHEMA_VERSION,
    Architecture,
    ComputeCost,
    Costs,
    H2DCost,
    InitialCapacityExceeded,
    InvalidInput,
    MapperConfig,
    Mapspace,
    Operation,
    Origin,
    Scenario,
    SpecError,
    Tensor,
    Unsupported,
    Workload,
    aligned_size,
    load_and_validate,
    load_workload,
    scenario_fingerprint,
    workload_fingerprint,
)

__all__ = [
    "ACTION_ADVANCE",
    "ACTION_COMPUTE",
    "ACTION_COPY_H2D",
    "ACTION_EVICT",
    "SCHEMA_VERSION",
    "STATUS_FEASIBLE",
    "STATUS_INFEASIBLE",
    "STATUS_OPTIMAL",
    "STATUS_UNKNOWN",
    "Action",
    "Architecture",
    "ComputeCost",
    "Costs",
    "EvaluationResult",
    "Event",
    "H2DCost",
    "InitialCapacityExceeded",
    "InvalidInput",
    "MapperConfig",
    "MappingDocument",
    "MappingError",
    "MappingIdentityError",
    "Mapspace",
    "Operation",
    "Origin",
    "Scenario",
    "SearchResult",
    "SpecError",
    "State",
    "StateSnapshot",
    "Tensor",
    "Unsupported",
    "Workload",
    "aligned_size",
    "evaluate_mapping",
    "initial_state",
    "is_goal",
    "legal_actions",
    "load_and_validate",
    "load_mapping",
    "load_workload",
    "scenario_fingerprint",
    "search",
    "transition",
    "used_vram_bytes",
    "workload_fingerprint",
]

__version__ = "0.1.0"
