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

M1 (``schema_version = "0.2"``) is the same machine with three more resources and
a third location: CPU and GPU compute, one PCIe slot shared by both copy
directions, and one SSD slot shared by both SSD directions. Weights may start on
the SSD and an intermediate may be written back to it as a snapshot; both
domains have a finite capacity with their own reserve and alignment, while the
SSD has neither -- it is unbounded and never evicted. A goal names the location
each output must end up in, and an SSD target counts as delivered only once its
write has completed. It lives in the same ``spec``/``engine`` pair rather than a
parallel implementation, so the state, the clock, the ledger and the evaluator
cannot drift apart between versions -- but the *rules* are two rule sets,
because 0.1's codes and wording are a frozen contract. SSD traffic is modelled
by an average effective read bandwidth and a separate write bandwidth; the
device's internals are not simulated (``M1_DESIGN.md`` §1, §4).

The core is pure standard library, so the test suite runs with no installation
step (``mapping/DESIGN.md`` §2). Times are integers in nanoseconds; this is
deliberately unlike ``modeling/``, which works in float seconds.

Status
------
The M0 Python core, the GGML exporter under ``mapping/ggml/``, and the
``model``/``search`` entry points with their four artefacts are all implemented
and covered by tests. M1 adds the 0.2 schemas, the two-domain ledger, the CPU
and SSD resources, the A--F example pairs, and their acceptance tests. Every
example fixture is still marked ``origin.kind = "synthetic_fixture"`` except the
four under ``examples/ggml/``, which are real exporter output (``ACCEPTANCE.md``
§7 reserves ``"ggml"`` for exactly those); every M1 cost is synthetic and is
labelled as such.

Passing the tests is **not** M0 acceptance on its own: ``DESIGN.md`` §9 requires
the GGML build environment for the integration half of the verification, and
``ACCEPTANCE.md`` §7 lists the items that must each be answered before M0 can be
called complete.
"""

from __future__ import annotations

from llm_infer_model.tensor.engine import (
    ACTION_ADVANCE,
    ACTION_COMPUTE,
    ACTION_COPY,
    ACTION_COPY_H2D,
    ACTION_EVICT,
    ACTION_READ_SSD,
    ACTION_WRITE_SSD,
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
    used_bytes,
    used_vram_bytes,
)
from .mapper import STATUS_FEASIBLE, STATUS_INFEASIBLE, STATUS_OPTIMAL, STATUS_UNKNOWN
from .mapper import SearchRecorder, SearchResult, search
from llm_infer_model.tensor.spec import (
    SCHEMA_VERSION,
    SCHEMA_VERSION_M1,
    Architecture,
    ComputeCost,
    Costs,
    H2DCost,
    InitialCapacityExceeded,
    InvalidInput,
    M1Costs,
    M1Goal,
    M1Mapspace,
    MapperConfig,
    Mapspace,
    Operation,
    Origin,
    Scenario,
    SpecError,
    Tensor,
    TransferCost,
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
    "ACTION_COPY",
    "ACTION_COPY_H2D",
    "ACTION_EVICT",
    "ACTION_READ_SSD",
    "ACTION_WRITE_SSD",
    "SCHEMA_VERSION",
    "SCHEMA_VERSION_M1",
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
    "M1Costs",
    "M1Goal",
    "M1Mapspace",
    "MapperConfig",
    "MappingDocument",
    "MappingError",
    "MappingIdentityError",
    "Mapspace",
    "Operation",
    "Origin",
    "Scenario",
    "SearchRecorder",
    "SearchResult",
    "SpecError",
    "State",
    "StateSnapshot",
    "Tensor",
    "TransferCost",
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
    "used_bytes",
    "used_vram_bytes",
    "workload_fingerprint",
]

__version__ = "0.1.0"
