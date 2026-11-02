"""Source-bound measured PV prices in a sealed, one-task timing-only runtime.

No API here applies a target price to an existing pipeline. The scalar tensor
ledger is deliberately synthetic: it tests the common runtime's time/progress
arithmetic and does not model the real target's numerical work or memory use.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Iterable

from .model import UnknownCost, lookup

_FACTORY_TOKEN = object()


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _read(path, digest=None):
    raw = Path(path).read_bytes()
    if digest is not None and hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"bound target evidence bytes changed: {path}")
    return json.loads(raw)


def _context(case_id, mode, instrumentation, telemetry):
    return dict(case_id=case_id, mode=mode, instrumentation=instrumentation, telemetry=telemetry)


def _validate_target(native, program):
    target = native.get("target", {})
    action, binding = target.get("action", {}), target.get("binding", {})
    index = native.get("target_action_index")
    if (action.get("kind") != "COMPUTE" or action.get("device") != "gpu"
            or binding.get("kind") != "attention_pv"
            or binding.get("operation_id") != action.get("target_id")):
        raise ValueError("adapter supports the measured GPU attention_pv target only")
    if type(index) is not int or index < 0 or index >= len(program.get("actions", [])):
        raise ValueError("target action index is invalid")
    if native.get("actions") != program["actions"] or program["actions"][index] != action:
        raise ValueError("target action is not the source program's selected action")
    matches = [row for row in program.get("bindings", []) if row.get("operation_id") == action["target_id"]]
    if matches != [binding]:
        raise ValueError("target binding differs from the exact source program")
    nodes = target.get("graph_nodes")
    if not isinstance(nodes, list) or not nodes or not any(row.get("op") == "MUL_MAT" for row in nodes):
        raise ValueError("native target graph provenance is missing")
    if any(row.get("type") != "f32" or (row.get("op") == "MUL_MAT"
            and row.get("requested_precision") != "GGML_PREC_F32") for row in nodes):
        raise ValueError("adapter requires the recorded F32 target graph")
    return dict(action=action, binding=binding, graph_nodes=nodes)


@dataclass(frozen=True)
class MeasuredTarget:
    """Immutable exact target/context, reconstructed from frozen profile evidence."""
    _model_json: str
    _context_json: str
    _evidence_json: str

    def build_session(self, *, initial_condition="baseline", changes: Iterable[tuple[int, str]] = (),
                      allow_unaccepted_diagnostic=False):
        """Create one timing carrier; condition changes are model-owned inputs.

        Stability failures need explicit diagnostic opt-in. Numerically invalid
        or unobserved cells never receive a fallback, including in that mode.
        """
        if type(allow_unaccepted_diagnostic) is not bool:
            raise ValueError("allow_unaccepted_diagnostic must be an explicit boolean")
        changes = tuple(changes)
        previous = 0
        for change in changes:
            if not isinstance(change, tuple) or len(change) != 2:
                raise ValueError("changes require (positive_integer_time_ns, known_condition) tuples")
            t_ns, _ = change
            if type(t_ns) is not int or t_ns <= previous:
                raise ValueError("condition change times must be strictly increasing and greater than zero")
            previous = t_ns
        model, context, evidence = (json.loads(value) for value in
            (self._model_json, self._context_json, self._evidence_json))
        reference = lookup(model, **context, condition="baseline", metric="target", family="conditional")
        prices = [(0, lookup(model, **context, condition=initial_condition, metric="target", family="conditional"))]
        prices.extend((t_ns, lookup(model, **context, condition=condition, metric="target", family="conditional"))
                      for t_ns, condition in changes)
        all_prices = [reference, *(price for _, price in prices)]
        unaccepted = [price for price in all_prices if not price["profile_accepted"]]
        if unaccepted and not allow_unaccepted_diagnostic:
            raise UnknownCost("source profile gates failed; explicitly allow_unaccepted_diagnostic for timing diagnostics")
        reference_ns = reference["price_ns"]
        if type(reference_ns) is not int or reference_ns <= 0:
            raise UnknownCost("positive integer baseline reference price is unavailable")
        from llm_infer_model.tensor.environment import EnvironmentEvent, SimulationSession
        events, priced_events = [], []
        for t_ns, price in prices:
            # The runtime's baseline work is integer ns; retain the conditional
            # exact rational median when adapting its rate, including half ns.
            rate = Fraction(reference_ns) / Fraction(**price["price_ns_exact"])
            events.append(EnvironmentEvent(t_ns, gpu_service_rate_numerator=rate.numerator,
                                           gpu_service_rate_denominator=rate.denominator))
            priced_events.append(dict(t_ns=t_ns, price=price,
                adapter_service_rate=dict(numerator=rate.numerator, denominator=rate.denominator)))
        target_id = evidence["target"]["action"]["target_id"]
        runtime = SimulationSession(_timing_carrier(target_id, reference_ns), environment_events=tuple(events))
        provenance = dict(schema="measured-target-runtime-0.1", context=context, metric="target", device="gpu",
            target_kind="attention_pv", target_id=target_id, model_sha256=model["model_sha256"],
            target_evidence=evidence, baseline_reference=reference, regime_events=priced_events,
            reference_work_ns=reference_ns, allow_unaccepted_diagnostic=allow_unaccepted_diagnostic,
            diagnostic_only=bool(unaccepted), all_used_profile_prices_accepted=not unaccepted,
            heldout_used_for_fit=False, heldout_prediction_accuracy_validated_here=False,
            scope="One measured GPU target timing carrier only; not a full LLM or numerical execution",
            synthetic_fixture_memory=dict(input_bytes=4, output_bytes=4, workspace_bytes=0,
                                          vram_capacity_bytes=8, dram_capacity_bytes=8),
            native_target_memory_modeled=False, private_cuda_workspace_bytes=None,
            physical_capacity_claimed=False, universal_device_multiplier=False,
            telemetry_to_rate_inference=False)
        return MeasuredTargetRuntime(runtime, target_id, _canonical(provenance).decode(), _factory_token=_FACTORY_TOKEN)


def load_measured_target(campaign_directory, *, case_id, mode, instrumentation="stamped", telemetry=True,
                         metric="target", device="gpu") -> MeasuredTarget:
    """Bind the frozen model to matching native PV/program profile artifacts.

    This verifies the target-specific subset of saved evidence. Run the complete
    campaign verifier separately for timing alignment and all other protocol
    checks. It never executes hardware or modifies the campaign directory.
    """
    if metric != "target" or device != "gpu":
        raise ValueError("whole-iteration or CPU prices cannot become a GPU target service rate")
    root = Path(campaign_directory)
    model = _read(root / "frozen-model.json")
    context = _context(case_id, mode, instrumentation, telemetry)
    # Validates frozen model integrity and exact measured baseline support.
    lookup(model, **context, condition="baseline", metric="target", family="conditional")
    manifest_digest = model["source_bindings"].get("manifest_sha256")
    if manifest_digest is None:
        raise ValueError("frozen model does not bind its campaign manifest")
    manifest = _read(root / "manifest.json", manifest_digest)
    primary_boundary = manifest.get("protocol", {}).get("primary_execution_boundary")
    if primary_boundary != "legacy_double_sync":
        raise ValueError("adapter requires an explicit legacy_double_sync primary execution boundary")
    rows = _read(root / "profile-rows.json")
    if _sha(rows) != model["source_rows_sha256"]:
        raise ValueError("profile rows differ from the frozen model's source rows")
    selected = [row for row in rows if all(row["job"].get(key) == value for key, value in context.items())]
    if not selected:
        raise UnknownCost("no source profile processes for the exact target context")
    descriptor, artifacts, programs = None, [], {}
    for row in selected:
        job, summary = row["job"], row["summary"]
        if job.get("split") != "profile" or row.get("status") != "measured" or summary.get("native_numeric_passed") is not True:
            raise UnknownCost("target source has missing or failed numerical evidence")
        native_digest = summary.get("native_sha256")
        if not isinstance(native_digest, str) or row.get("artifacts_sha256", {}).get("native.json") != native_digest:
            raise ValueError("target source native hash binding is missing or inconsistent")
        native = _read(root / "jobs" / job["id"] / "native.json", native_digest)
        # Primary jobs may inherit the manifest's explicit boundary. Native
        # output must independently record it; older missing fields are not
        # guessed. Alternative boundary diagnostics cannot enter this price.
        if (job.get("execution_boundary", primary_boundary) != primary_boundary
                or native.get("execution_boundary") != primary_boundary):
            raise ValueError("target execution boundary differs from the exact primary price context")
        binding = manifest["programs"][case_id][job["program_key"]]
        if row.get("program_sha256") != binding["sha256"]:
            raise ValueError("process program hash differs from its campaign binding")
        key = (binding["path"], binding["sha256"])
        if key not in programs:
            programs[key] = _read(binding["path"], binding["sha256"])
        candidate = _validate_target(native, programs[key])
        if native.get("mode") != mode or str(native["target_action_index"]) != row["command"][3]:
            raise ValueError("native target mode/index differs from its source command")
        if descriptor is None:
            descriptor = candidate
        elif descriptor != candidate:
            raise ValueError("one measured context contains different target graphs")
        artifacts.append(dict(job_id=job["id"], condition=job["condition"], execution_boundary=primary_boundary,
                              native_sha256=native_digest,
                              program_sha256=binding["sha256"], source_row_sha256=_sha(row)))
    evidence = dict(target=descriptor, target_identity_sha256=_sha(descriptor),
        profile_rows_sha256=model["source_rows_sha256"], manifest_sha256=manifest_digest,
        source_processes=artifacts, primary_execution_boundary=primary_boundary,
        complete_campaign_verification_required=True)
    return MeasuredTarget(_canonical(model).decode(), _canonical(context).decode(), _canonical(evidence).decode())


def _timing_carrier(target_id, reference_ns):
    """Explicit synthetic ledger; its zero workspace is not a physical PV price."""
    from llm_infer_model.tensor.spec import build_scenario
    tensors = [dict(id=name, name=name, role=role, dtype="GGML_TYPE_F32", ne=[1, 1, 1, 1],
        nb=[4, 4, 4, 4], storage_bytes=4, initial_locations=locations)
        for name, role, locations in (("timing-input", "input", ["vram"]), ("timing-output", "output", []))]
    workload = dict(schema_version="0.2", id="measured-target-timing-carrier",
        origin=dict(kind="synthetic_fixture", ggml_version=None, sample=None), tensors=tensors,
        operations=[dict(id=target_id, semantic_op="RELU", ggml_op="GGML_OP_UNARY",
            unary_op="GGML_UNARY_OP_RELU", op_params=[], inputs=["timing-input"], output="timing-output")],
        outputs=["timing-output"])
    scenario = dict(schema_version="0.2", id="synthetic-timing-ledger-not-physical-memory",
        architecture=dict(dram_capacity_bytes=8, vram_capacity_bytes=8, dram_runtime_reserved_bytes=0,
            vram_runtime_reserved_bytes=0, dram_alignment_bytes=1, vram_alignment_bytes=1,
            cpu_compute_slots=1, gpu_compute_slots=1, pcie_copy_slots=1, ssd_io_slots=1),
        costs=dict(source="source_bound_measured_target_timing_only; synthetic_carrier_workspace_zero",
                   compute={target_id: {"gpu": dict(duration_ns=reference_ns, workspace_bytes=0)}}, transfer={}),
        mapspace=dict(allowed_devices={target_id: ["gpu"]}, allowed_transfers={}, allow_eviction=False,
            allow_cpu_gpu_overlap=False, allow_copy_compute_overlap=False, allow_recomputation=False),
        goal=dict(output_locations={"timing-output": "vram"}),
        mapper=dict(algorithm="uniform_cost", max_expanded_states=0, wall_time_limit_s=0))
    return build_scenario(workload, scenario)


class MeasuredTargetRuntime:
    """Restricted adapter: no arbitrary commands, operation additions or graph injection."""
    def __init__(self, runtime, target_id, provenance_json, *, _factory_token=None):
        if _factory_token is not _FACTORY_TOKEN:
            raise ValueError("use load_measured_target(...).build_session(); arbitrary runtime injection is unsupported")
        self._runtime, self._target_id, self._provenance_json = runtime, target_id, provenance_json
        self._started = False
        self._check_scope()

    def _check_scope(self):
        scenario = self._runtime.scenario
        if (len(scenario.workload.operations) != 1 or scenario.workload.operations[0].id != self._target_id
                or scenario.mapspace.devices_for(self._target_id) != ("gpu",)
                or scenario.mapspace.allowed_transfers
                or any(task.resource != "gpu_compute" or task.target_id != self._target_id
                       for task in self._runtime.state.running)):
            raise ValueError("measured target rate cannot be applied to an arbitrary pipeline")

    def provenance(self):
        return json.loads(self._provenance_json)

    def observe(self):
        self._check_scope()
        return self._runtime.observe()

    def result(self):
        self._check_scope()
        return self._runtime.result()

    def start(self):
        self._check_scope()
        if self._started:
            raise ValueError("the one measured target was already started")
        from llm_infer_model.tensor.engine import Action
        result = self._runtime.submit(Action("COMPUTE", self._target_id, device="gpu"))
        self._started = True
        return result

    def advance(self):
        self._check_scope()
        if not self._started:
            raise ValueError("start the measured target before advancing")
        return self._runtime.advance()

    def run(self):
        if not self._started:
            step = self.start()
            if not step.ok:
                raise ValueError(step.error_message)
        while not self.observe().finished:
            step = self.advance()
            if not step.ok:
                raise ValueError(step.error_message)
        return self.result()
