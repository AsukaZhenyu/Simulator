"""Public access to modeling-owned, interpretable performance calculation.

This module needs neither profiling nor saved measurements. Parameters may be
explicit declarations or a validated bundle produced by the optional profiling
workflow. The model computes progress, waits and completion; it never fits data.
"""
from llm_infer_model.performance import (
    BlockedExecution, CapacityExceeded, Evaluation, Graph, InvalidModel,
    MemoryDomain, MemoryReservation, Parameter, ParameterBundle, ResourcePoint,
    ResourceTimeline, Task, TaskExecution, UnknownParameter, WaitInterval,
    WorkDemand, evaluate,
)

__all__ = [
    "BlockedExecution", "CapacityExceeded", "Evaluation", "Graph", "InvalidModel",
    "MemoryDomain", "MemoryReservation", "Parameter", "ParameterBundle",
    "ResourcePoint", "ResourceTimeline", "Task", "TaskExecution",
    "UnknownParameter", "WaitInterval", "WorkDemand", "evaluate",
]
