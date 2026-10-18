"""Restricted resource-functional to common-runtime timing connection.

This module executes one exact modeled CPU or GPU operation in a synthetic
four-byte input/output carrier.  It tests that the real SimulationSession
consumes the same exact remaining reference work as the functional.  It is not
a numerical kernel execution, a physical memory simulation, or an arbitrary
pipeline adapter.  Copy operations need their link-specific runtime treatment
and are explicitly unsupported here.

Resource traces belong to the trusted environment driver.  Policy code may
receive ``observe()`` only: it contains current rates, not a future trace or
environment queue.  ``provenance()`` and ``result()`` are diagnostic driver
interfaces.  No public API accepts extra actions, a policy or another runtime.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from typing import Any

from .functional import Operation, Prediction, ResourceFunctional, ResourceTrace, predict


_FACTORY_TOKEN = object()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _timing_carrier(operation: Operation):
    """An explicit synthetic ledger, independently built through public specs."""
    from llm_infer_model.tensor.spec import build_scenario

    location = "dram" if operation.resource == "cpu" else "vram"
    tensors = [dict(id=name, name=name, role=role, dtype="GGML_TYPE_F32", ne=[1, 1, 1, 1],
                    nb=[4, 4, 4, 4], storage_bytes=4, initial_locations=locations)
               for name, role, locations in (("timing-input", "input", [location]),
                                              ("timing-output", "output", []))]
    workload = dict(schema_version="0.2", id="resource-functional-one-operation-carrier",
                    origin=dict(kind="synthetic_fixture", ggml_version=None, sample=None), tensors=tensors,
                    operations=[dict(id=operation.key, semantic_op="RELU", ggml_op="GGML_OP_UNARY",
                                     unary_op="GGML_UNARY_OP_RELU", op_params=[], inputs=["timing-input"],
                                     output="timing-output")], outputs=["timing-output"])
    scenario = dict(schema_version="0.2", id="synthetic-timing-ledger-not-physical-operation-memory",
                    architecture=dict(dram_capacity_bytes=8, vram_capacity_bytes=8,
                                      dram_runtime_reserved_bytes=0, vram_runtime_reserved_bytes=0,
                                      dram_alignment_bytes=1, vram_alignment_bytes=1,
                                      cpu_compute_slots=1, gpu_compute_slots=1, pcie_copy_slots=1, ssd_io_slots=1),
                    costs=dict(source="fitted-resource-functional-timing-only; synthetic-fixture-workspace-zero",
                               compute={operation.key: {operation.resource: dict(duration_ns=operation.reference_ns,
                                                                               workspace_bytes=0)}}, transfer={}),
                    mapspace=dict(allowed_devices={operation.key: [operation.resource]}, allowed_transfers={},
                                  allow_eviction=False, allow_cpu_gpu_overlap=False, allow_copy_compute_overlap=False,
                                  allow_recomputation=False),
                    goal=dict(output_locations={"timing-output": location}),
                    mapper=dict(algorithm="uniform_cost", max_expanded_states=0, wall_time_limit_s=0))
    return build_scenario(workload, scenario)


def build_runtime(model: ResourceFunctional, operation: Operation, trace: ResourceTrace, *,
                  start_ns: int | None = None) -> OperationRuntime:
    """Preflight a complete trace, then create one sealed timing-only session.

    No hardware is launched and no calibration or heldout fitting occurs.  An
    unsupported state, missing trace tail, nonpositive fitted response or known
    capacity violation is rejected before construction of SimulationSession.
    The driver uses operation-specific response coefficients; they are never
    promoted to universal CPU/GPU multipliers for unrelated operations.
    """
    if not isinstance(operation, Operation):
        raise ValueError("operation must be Operation")
    if operation.resource == "copy":
        raise ValueError("copy operations require link-specific transfer rates; this adapter supports CPU/GPU compute only")
    expected = predict(model, operation, trace, start_ns=start_ns)
    from llm_infer_model.tensor.environment import EnvironmentEvent, SimulationSession

    events = []
    for service in expected.slices:
        relative_start = service.start_ns - expected.start_ns
        if relative_start.denominator != 1:
            raise ValueError("environment change times must be integer nanoseconds")
        fields = {f"{operation.resource}_service_rate_numerator": service.service_rate.numerator,
                  f"{operation.resource}_service_rate_denominator": service.service_rate.denominator}
        events.append(EnvironmentEvent(relative_start.numerator, **fields))
    runtime = SimulationSession(_timing_carrier(operation), environment_events=tuple(events))
    provenance = dict(schema_version=1, adapter="single-operation-resource-functional-timing-carrier",
                      model_sha256=model.model_sha256, source_sha256=model.source_sha256,
                      protocol_id=model.protocol_id, clock_id=model.clock_id,
                      training_data_sha256=model.training_data_sha256,
                      operation=operation.to_dict(), operation_signature=operation.signature,
                      trace_sha256=_digest(trace.to_dict()), state_semantics=model.state_semantics,
                      compute_semantics=model.compute_semantics, memory_semantics=model.memory_semantics,
                      origin_ns=expected.start_ns, functional_exact_elapsed_ns=str(expected.exact_elapsed_ns),
                      functional_elapsed_ns=expected.elapsed_ns, reference_work_ns=operation.reference_ns,
                      diagnostic_only=True, hardware_accuracy_accepted=False,
                      heldout_used_for_fit=False, native_hardware_executed=False,
                      numerical_operation_executed=False, arbitrary_pipeline_adapter=False,
                      universal_device_multiplier=False, utilization_to_rate_heuristic=False,
                      future_trace_exposed_to_policy=False,
                      synthetic_fixture_memory=dict(input_bytes=4, output_bytes=4, workspace_bytes=0,
                                                    dram_capacity_bytes=8, vram_capacity_bytes=8),
                      source_operation_footprint=dict(dram_bytes=operation.dram_bytes,
                                                      vram_bytes=operation.vram_bytes,
                                                      memory_footprint_complete=operation.memory_footprint_complete,
                                                      private_workspace_bytes=None),
                      source_memory_states_used_in_carrier_ledger=False,
                      physical_capacity_claimed=False)
    return OperationRuntime(runtime, operation, expected, _canonical(provenance), _factory_token=_FACTORY_TOKEN)


class OperationRuntime:
    """One sealed operation; expose ``observe`` alone to any mapping policy."""

    def __init__(self, runtime, operation: Operation, expected: Prediction, provenance_json: str, *,
                 _factory_token=None):
        if _factory_token is not _FACTORY_TOKEN:
            raise ValueError("use build_runtime(); arbitrary runtime injection is unsupported")
        self._runtime = runtime
        self._operation = operation
        self._expected = expected
        self._provenance_json = provenance_json
        self._started = False
        self._check_scope()

    def _check_scope(self) -> None:
        scenario = self._runtime.scenario
        expected_resource = f"{self._operation.resource}_compute"
        if (len(scenario.workload.operations) != 1
                or scenario.workload.operations[0].id != self._operation.key
                or scenario.mapspace.devices_for(self._operation.key) != (self._operation.resource,)
                or scenario.mapspace.allowed_transfers
                or any(task.resource != expected_resource or task.target_id != self._operation.key
                       for task in self._runtime.state.running)):
            raise ValueError("operation-specific rates cannot be injected into another operation or pipeline")

    def provenance(self) -> dict[str, Any]:
        """Immutable-copy driver evidence; deliberately contains no future queue."""
        self._check_scope()
        return json.loads(self._provenance_json)

    def observe(self):
        self._check_scope()
        return self._runtime.observe()

    def start(self):
        self._check_scope()
        if self._started:
            raise ValueError("the single modeled operation has already started")
        from llm_infer_model.tensor.engine import Action
        step = self._runtime.submit(Action("COMPUTE", self._operation.key, device=self._operation.resource))
        if step.ok:
            self._started = True
        return step

    def advance(self):
        self._check_scope()
        if not self._started:
            raise ValueError("start the modeled operation before advancing")
        return self._runtime.advance()

    def result(self):
        self._check_scope()
        return self._runtime.result()

    def run(self):
        if not self._started:
            step = self.start()
            if not step.ok:
                raise ValueError(step.error_message)
        while not self.observe().finished:
            step = self.advance()
            if not step.ok:
                raise ValueError(step.error_message)
        result = self.result()
        if result.evaluation.makespan_ns != self._expected.elapsed_ns:
            raise ValueError("common runtime completion differs from the resource functional")
        work = sum((Fraction(item.reference_service_ns_numerator, item.reference_service_ns_denominator)
                    for item in result.service_intervals), Fraction(0))
        if work != self._operation.reference_ns:
            raise ValueError("common runtime did not consume exactly the declared reference work")
        return result
