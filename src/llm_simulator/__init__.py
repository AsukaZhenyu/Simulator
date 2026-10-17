"""Python strategies, a shared tensor runtime, and explicit performance models.

The convenience API does not load a previously generated profile. Choose
AnalyticalCosts explicitly, or use ModelPlan with your own exact cost inputs.
"""
from .config import Device, Hardware, Model, Workload
from .analytical import AnalyticalCosts
from .simulation import CompiledSimulation, SimulationResult, Simulator
from llm_infer_model.m2.moe_spec import RouteTraceEntry as Route
from llm_infer_model.tensor.environment import EnvironmentEvent
from llm_infer_model.tensor.spec import InvalidInput, SpecError, Unsupported
from tensor_mapping.causal import ControlDecision, Decision
from tensor_mapping.phase_scheduling import DataPreparation, PhasePlacement
from tensor_mapping.python_frontend import (
    DEFER, Cache, CompiledPlan, ModelPlan, PlanModule, Prefetch, RuntimePolicy,
    Scheduling, Select, Transport,
)
from tensor_mapping.reconfiguration import ReconfigurationSpec

__version__ = "0.1.0a2"

__all__ = [
    "__version__", "AnalyticalCosts", "Cache", "CompiledPlan",
    "CompiledSimulation", "ControlDecision", "DataPreparation", "DEFER",
    "Decision", "Device", "EnvironmentEvent", "Hardware", "InvalidInput",
    "Model", "ModelPlan", "PhasePlacement", "PlanModule", "Prefetch",
    "ReconfigurationSpec", "Route", "RuntimePolicy", "Scheduling", "Select",
    "SimulationResult", "Simulator", "SpecError", "Transport", "Unsupported",
    "Workload",
]
