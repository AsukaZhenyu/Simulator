"""Strategy expression, compilation and bounded mapping search.

The current Python frontend lives in ``python_frontend`` and lowers through
``composition_ir`` into the unified causal runtime. ``causal`` compiles public
policy decisions into commands executed by ``llm_infer_model.tensor``; mapping
owns no second copy of the clock, dependency or capacity rules.

This package also retains fixed-action IR, paper-mechanism adapters and exact
search for small declared graphs. The top-level exports below are the stable
M0/M1 compatibility API; they do not enumerate every newer frontend type.
``spec`` and ``engine`` only re-export the modeling implementation.

Install modeling and mapping together as documented in ``mapping/README.md``.
Simulation examples may use synthetic costs; passing logic tests does not
establish real-hardware performance accuracy.
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
