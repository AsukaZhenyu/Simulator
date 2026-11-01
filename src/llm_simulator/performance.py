"""Public access to modeling-owned, interpretable performance calculation.

This module needs neither profiling nor saved measurements. Parameters may be
explicit declarations or a validated bundle produced by the optional profiling
workflow. The model computes progress, waits and completion; it never fits data.
"""
from llm_infer_model.performance import (
    BlockedExecution, CapacityExceeded, Evaluation, Graph, InvalidModel,
    MemoryDomain, MemoryReservation, Parameter, ParameterBundle, ResourcePoint,
    ResourceTimeline, Task, TaskExecution, UnknownParameter, WaitInterval,
    WorkDemand, evaluate, INSTANTANEOUS_RATE, InstantaneousRate,
    ConditionedResponseEvaluation, ResponseCell, ResponsePoint, ResponseState,
    ResponseTable, ResponseTrajectory, evaluate_response,
    RequestResponseDelays, apply_request_response_delays,
    CpuInterruptionTimebase, ThreadCpuBinding, CpuInterruption,
    CpuInterruptionTrace, CpuInterruptionUnion, CpuInterruptionComposition,
    compose_cpu_interruptions,
    ServiceResponseParameter, BackgroundReservoir, ServiceInterference,
    ServiceResponseModel, ServiceRequestPoint, ServiceRequestTrajectory,
    ServiceResponseSample, ServiceResponseResult, build_service_response,
    AvailableResources, ServiceBinding, AvailableResourceEvaluation,
    evaluate_available_resources,
    TransferResponse, AffineResponseImpact, TransferRecoveryScope, SizeAwareRecoveryResponse,
    PCIeState, PCIeResponseCell, PCIeStatePoint,
    PCIeStatePlan, PCIeResponseTable,
    HostPlacement, RuntimeScope, RuntimeStage, RuntimeCell, RuntimeProfile, RuntimeProgram,
)

__all__ = [
    "TransferResponse", "AffineResponseImpact", "TransferRecoveryScope", "SizeAwareRecoveryResponse",
    "PCIeState", "PCIeResponseCell", "PCIeStatePoint",
    "PCIeStatePlan", "PCIeResponseTable",
    "HostPlacement", "RuntimeScope", "RuntimeStage", "RuntimeCell", "RuntimeProfile", "RuntimeProgram",
    "BlockedExecution", "CapacityExceeded", "Evaluation", "Graph", "InvalidModel",
    "MemoryDomain", "MemoryReservation", "Parameter", "ParameterBundle",
    "ResourcePoint", "ResourceTimeline", "Task", "TaskExecution",
    "UnknownParameter", "WaitInterval", "WorkDemand", "evaluate",
    "INSTANTANEOUS_RATE", "InstantaneousRate", "ResponseState", "ResponseCell",
    "ResponseTable", "ResponsePoint", "ResponseTrajectory",
    "ConditionedResponseEvaluation", "evaluate_response",
    "RequestResponseDelays", "apply_request_response_delays",
    "CpuInterruptionTimebase", "ThreadCpuBinding", "CpuInterruption",
    "CpuInterruptionTrace", "CpuInterruptionUnion", "CpuInterruptionComposition",
    "compose_cpu_interruptions",
    "ServiceResponseParameter", "BackgroundReservoir", "ServiceInterference",
    "ServiceResponseModel", "ServiceRequestPoint", "ServiceRequestTrajectory",
    "ServiceResponseSample", "ServiceResponseResult", "build_service_response",
    "AvailableResources", "ServiceBinding", "AvailableResourceEvaluation",
    "evaluate_available_resources",
]
