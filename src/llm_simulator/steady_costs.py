"""Scoped steady costs and causal resource-input contracts.

Importing this module needs no profiler, CUDA installation or saved profiles.
Completed-action prices describe a measured serial execution boundary. Device
intervals remain diagnostics; they are not added as independent physical work.
External scenarios may be functions of time, while own occupancy stays in the
model. These contracts do not forecast the OS or convert utilization to speed.
"""
from llm_infer_model.performance.steady_action_costs import (
    SteadyActionKey, SteadyActionCost, SteadyActionTable,
    CompiledSteadyActions, compile_steady_actions,
)
from llm_infer_model.performance.causal_inputs import (
    ResourceInput, validate_resource_input, validate_budget_contract,
)

__all__ = [
    "SteadyActionKey", "SteadyActionCost", "SteadyActionTable",
    "CompiledSteadyActions", "compile_steady_actions", "ResourceInput",
    "validate_resource_input", "validate_budget_contract",
]
