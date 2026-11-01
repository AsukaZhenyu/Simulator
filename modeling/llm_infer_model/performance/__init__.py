"""Interpretable performance calculation, separate from profiling calibration.

Physical parameters are declared or evidence-backed. Profiling may estimate
those parameters; only modeling computes execution progress and predictions.
"""
from .analytical import analytical_compute_duration_ns, analytical_transfer_duration_ns
from .parameters import (INSTANTANEOUS_RATE, InstantaneousRate, InvalidModel,
                         Parameter, ParameterBundle, UnknownParameter)
from .graph import (Graph, MemoryDomain, MemoryReservation, ResourcePoint,
                    ResourceTimeline, Task, WorkDemand)
from .evaluator import (BlockedExecution, CapacityExceeded, Evaluation, TaskExecution,
                        WaitInterval, evaluate)
from .available_resources import (AvailableResources, ServiceBinding,
    AvailableResourceEvaluation, evaluate_available_resources)
from .conditioned_response import (ConditionedResponseEvaluation, ResponseCell,
    ResponsePoint, ResponseState, ResponseTable, ResponseTrajectory, evaluate_response,
    RequestResponseDelays, apply_request_response_delays)
from .cpu_interruptions import (CpuInterruption, CpuInterruptionComposition,
    CpuInterruptionTimebase, CpuInterruptionTrace, CpuInterruptionUnion,
    ThreadCpuBinding, compose_cpu_interruptions)
from .service_response import (ServiceResponseParameter, BackgroundReservoir,
    ServiceInterference, ServiceResponseModel, ServiceRequestPoint,
    ServiceRequestTrajectory, ServiceResponseSample, ServiceResponseResult,
    build_service_response)
from .transfer_response import TransferResponse
from .transfer_recovery import (AffineResponseImpact, TransferRecoveryScope,
                               SizeAwareRecoveryResponse)
from .pcie_response import (PCIeState, PCIeResponseCell, PCIeStatePoint,
                            PCIeStatePlan, PCIeResponseTable)
from .runtime_response import (HostPlacement, RuntimeScope, RuntimeStage,
    RuntimeCell, RuntimeProfile, RuntimeProgram)

__all__ = ["TransferResponse", "AffineResponseImpact", "TransferRecoveryScope",
    "SizeAwareRecoveryResponse", "PCIeState", "PCIeResponseCell", "PCIeStatePoint",
    "HostPlacement", "RuntimeScope", "RuntimeStage", "RuntimeCell", "RuntimeProfile", "RuntimeProgram",
    "PCIeStatePlan", "PCIeResponseTable", "InvalidModel", "UnknownParameter", "Parameter", "ParameterBundle",
    "WorkDemand", "Task", "Graph", "MemoryDomain", "MemoryReservation", "ResourcePoint",
    "ResourceTimeline", "evaluate", "Evaluation", "TaskExecution", "WaitInterval",
    "CapacityExceeded", "BlockedExecution", "analytical_compute_duration_ns",
    "analytical_transfer_duration_ns", "INSTANTANEOUS_RATE", "InstantaneousRate",
    "ResponseState", "ResponseCell", "ResponseTable", "ResponsePoint",
    "ResponseTrajectory", "ConditionedResponseEvaluation", "evaluate_response",
    "RequestResponseDelays", "apply_request_response_delays",
    "CpuInterruption", "CpuInterruptionComposition", "CpuInterruptionTimebase",
    "CpuInterruptionTrace", "CpuInterruptionUnion", "ThreadCpuBinding",
    "compose_cpu_interruptions", "ServiceResponseParameter", "BackgroundReservoir",
    "ServiceInterference", "ServiceResponseModel", "ServiceRequestPoint",
    "ServiceRequestTrajectory", "ServiceResponseSample", "ServiceResponseResult",
    "build_service_response", "AvailableResources", "ServiceBinding",
    "AvailableResourceEvaluation", "evaluate_available_resources"]
