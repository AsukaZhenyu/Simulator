"""Measured-component anchored serial hardware scenarios for the DAC protocol.

No commercial-device ranking is made.  The tensor engine checks dependencies,
capacity and unique allocations; the existing reusable serial response model
prices the same native action boundaries.  Shared visibility removes COPY and
uses SHARE.  This is explicitly idealized, not measured coherence or UMA.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VERSION = "dac-validation-e3-1"
LEVELS = (0.5, 0.8, 1.0, 1.2, 1.5)
WORKLOADS = ("h512-p64-h0", "h512-d1-h128")
CANDIDATES = ("S0", "S1", "S2", "S3")
SHARED_HANDOFF_UPPER_NS = 10_000


class UnknownScenarioCost(ValueError):
    """A declared axis has no identified, nonnegative physical cost contract."""


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def rational(value):
    if isinstance(value, dict) and {"numerator", "denominator"} <= value.keys():
        return Fraction(value["numerator"], value["denominator"])
    return Fraction(str(value))


def pack(value):
    if isinstance(value, Fraction):
        return dict(numerator=value.numerator, denominator=value.denominator)
    if isinstance(value, dict):
        return {k: pack(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [pack(v) for v in value]
    return value


@dataclass(frozen=True)
class Point:
    architecture: str
    id: str
    cpu: float = 1.0
    gpu: float = 1.0
    link: float = 1.0
    capacity: float = 1.0
    handoff_ns: int = 0
    family: str = "main"
    focus_direction: str | None = None
    corner: int | None = None

    def __post_init__(self):
        if self.architecture not in ("separate","shared"):
            raise ValueError("declared separate/shared architecture required")
        if any(not math.isfinite(v) or v<=0 for v in (self.cpu,self.gpu,self.link,self.capacity)):
            raise ValueError("positive finite scenario multipliers required")
        if type(self.handoff_ns) is not int or self.handoff_ns<0:
            raise ValueError("nonnegative integer handoff scenario cost required")


def main_points():
    points = []
    for architecture, axes in (("separate", ("cpu", "gpu", "link", "capacity")),
                               ("shared", ("cpu", "gpu", "capacity"))):
        base = Point(architecture, f"{architecture}-base")
        points.append(base)
        for axis in axes:
            for factor in LEVELS:
                if factor != 1:
                    points.append(replace(base, id=f"{architecture}-{axis}-{factor:g}", **{axis: factor}))
        other = "link" if architecture == "separate" else "capacity"
        for gpu, factor in itertools.product((0.8, 1.2), repeat=2):
            points.append(replace(base, id=f"{architecture}-gpu-{gpu:g}-{other}-{factor:g}",
                                  gpu=gpu, **{other: factor}))
    assert len(points) == 38 and len({p.id for p in points}) == 38
    return tuple(points)


def directions(architecture):
    return (("baseline", None, 1.0), ("cpu-faster", "cpu", 0.8),
            ("gpu-faster", "gpu", 0.8), ("more-capacity", "capacity", 1.2)) + (
                (("more-link-bandwidth", "link", 1.2),) if architecture == "separate" else ())


def robustness_points():
    points = []
    for architecture in ("separate", "shared"):
        for corner, values in enumerate(itertools.product((0.8, 1.2), repeat=4)):
            cpu, gpu, capacity, fourth = values
            base = Point(architecture, "", cpu=cpu, gpu=gpu, capacity=capacity,
                link=fourth if architecture == "separate" else 1,
                handoff_ns=(0 if fourth == 0.8 else SHARED_HANDOFF_UPPER_NS)
                    if architecture == "shared" else 0,
                family="robustness", corner=corner)
            for name, axis, factor in directions(architecture):
                point = replace(base, id=f"robust-{architecture}-c{corner:02}-{name}", focus_direction=name)
                if axis:
                    point = replace(point, **{axis: getattr(point, axis) * factor})
                points.append(point)
    assert len(points) == 144
    return tuple(points)


def inventory():
    points = main_points()
    return dict(schema=VERSION, status="prepared_not_executed", declared_main_cases=304,
        declared_robustness_cases=1152,
        main_points=[asdict(p) for p in points], robustness_points=[asdict(p) for p in robustness_points()],
        main_jobs=[dict(workload=w, candidate=c, point=p.id)
                   for p in points for w in WORKLOADS for c in CANDIDATES],
        workloads=list(WORKLOADS), candidates=list(CANDIDATES),
        total_capacity_rule="two equal pools; each ceil(1.05 * max whole tensor-catalogue bytes across frozen candidates for this workload)); shared capacity equals their sum",
        capacity_rule_uses_timing=False, separate_dram_vram_ratio="1:1",
        unknown_native_workspace="remains unknown; capacity scenarios explicitly assume zero additional workspace beyond declared tensor inventory",
        shared_visibility="idealized SHARE of one allocation; compute memory work remains in measured inclusive action responses",
        shared_handoff_uncertainty=dict(lower_ns=0, upper_ns=SHARED_HANDOFF_UPPER_NS,
            measured=False, certified_bound=False, scope="additional per-SHARE scenario cost; no claim of actual UMA coherence coverage"),
        uncertainty="predeclared +/-20% scenario corners; not statistical confidence or measured hardware-error bounds",
        excluded_claims=["real UMA accuracy", "Mac/phone/NPU ranking", "FLOP/s or frequency sensitivity", "global optimality", "concurrent tensor parallel speedup"])


def copy_scenario_cost(anchor_mean, nbytes, bandwidth_factor, slope_ns_per_byte=None):
    """Scale only an independently identified payload term; retain exact anchor.

    The fixed term includes the measured per-coordinate residual.  Negative
    residuals cannot be reinterpreted as physical latency and are rejected.
    """
    mean = rational(anchor_mean)
    factor = rational(bandwidth_factor)
    if mean <= 0 or factor <= 0 or nbytes <= 0:
        raise ValueError("positive measured copy response/bytes/factor required")
    if factor == 1:
        return mean
    if slope_ns_per_byte is None:
        raise KeyError("payload/fixed transfer decomposition is unknown")
    payload = rational(slope_ns_per_byte) * nbytes
    if payload < 0 or mean - payload < 0:
        raise ValueError("identified payload exceeds exact copy response; negative fixed residual")
    return mean - payload + payload / factor


def response_costs(program, action_ns, point, transfer_models=None):
    tensors = {t["id"]: t for t in program["graph"]["workload"]["tensors"]}
    if len(action_ns) != len(program["actions"]):
        raise ValueError("one independently measured response per native action required")
    rows = []
    retained_indices = set(lower_serial_actions(program, True)[1]) if point.architecture == "shared" else None
    for index, (action, original) in enumerate(zip(program["actions"], action_ns)):
        duration = rational(original)
        if duration < 0:
            raise ValueError("negative measured response")
        kind = action["kind"]
        modeled_kind = kind
        if retained_indices is not None and index not in retained_indices:
            duration = Fraction()
            modeled_kind = "OMITTED_COPY_WAIT"
        elif kind == "COMPUTE":
            if action.get("device") not in ("cpu","gpu"):
                raise ValueError("only the declared CPU/GPU execution boundary is supported")
            duration *= rational(point.cpu if action["device"] == "cpu" else point.gpu)
            if duration <= 0:
                raise ValueError("compute requires a positive measured response")
        elif kind == "COPY":
            if point.architecture == "shared":
                duration = Fraction(point.handoff_ns)
                modeled_kind = "SHARE"
            else:
                route = action["src"] + "_to_" + action["dst"]
                model = (transfer_models or {}).get(route)
                slope = model.get("slope_ns_per_byte") if model and model.get("qualified") else None
                try:
                    duration = copy_scenario_cost(duration, tensors[action["target_id"]]["storage_bytes"], point.link, slope)
                except (KeyError,ValueError) as error:
                    raise UnknownScenarioCost(str(error)) from error
        elif kind not in ("ADVANCE", "EVICT"):
            raise ValueError("undeclared native action: " + kind)
        rows.append(dict(index=index, native_kind=kind, modeled_kind=modeled_kind,
                         original_ns=rational(original), scenario_ns=duration))
    return rows


def make_scenario(program, prices, point, total_capacity_bytes, workspace_assumption=0):
    """Build a selected-device-only scenario; unmeasured alternate prices vanish."""
    from llm_infer_model.tensor.spec import build_scenario
    workload = deepcopy(program["graph"]["workload"])
    document = deepcopy(program["graph"]["scenario"])
    half = max(1, math.floor(Fraction(total_capacity_bytes, 2) * rational(point.capacity)))
    total = 2 * half
    architecture = document["architecture"]
    architecture.update(dram_capacity_bytes=half, vram_capacity_bytes=total-half,
                        dram_runtime_reserved_bytes=0, vram_runtime_reserved_bytes=0,
                        dram_alignment_bytes=1, vram_alignment_bytes=1)
    if point.architecture == "shared":
        architecture.update(memory_mode="shared", shared_capacity_bytes=total,
                            shared_runtime_reserved_bytes=0, shared_alignment_bytes=1,
                            dram_capacity_bytes=total, vram_capacity_bytes=total)
    else:
        for key in ("memory_mode", "shared_capacity_bytes", "shared_runtime_reserved_bytes", "shared_alignment_bytes"):
            architecture.pop(key, None)
    compute, transfers, allowed = {}, defaultdict(dict), {}
    by_index = {row["index"]: row for row in prices}
    for index, action in enumerate(program["actions"]):
        row = by_index[index]
        if action["kind"] == "COMPUTE":
            oid, device = action["target_id"], action["device"]
            allowed[oid] = [device]
            compute[oid] = {device: dict(duration_ns=max(1, math.ceil(row["scenario_ns"])),
                                        workspace_bytes=workspace_assumption)}
        elif action["kind"] == "COPY" and point.architecture == "separate":
            route = action["src"] + "_to_" + action["dst"]
            transfers[route][action["target_id"]] = dict(duration_ns=max(1, math.ceil(row["scenario_ns"])))
    document["costs"] = dict(source="independent semantic action probes plus explicit E3 scenario multipliers", compute=compute,
                              transfer=dict(transfers))
    document["mapspace"]["allowed_devices"] = allowed
    # Restrict transfers to those demanded by the frozen serial strategy. Shared
    # access uses these same route permissions and requires no fabricated COPY.
    permitted = defaultdict(set)
    for action in program["actions"]:
        if action["kind"] == "COPY":
            permitted[action["src"] + "_to_" + action["dst"]].add(action["target_id"])
    document["mapspace"]["allowed_transfers"] = {route: sorted(ids) for route, ids in permitted.items()}
    return build_scenario(workload, document), workload, document


def lower_serial_actions(program, shared=False):
    """Demand-only, last-use recipe: SHARE replaces copy, and its wait vanishes."""
    from llm_infer_model.tensor.engine import Action
    actions, source_indices = [], []
    skip_copy_wait = False
    for index, native in enumerate(program["actions"]):
        raw = dict(native)
        if shared and raw["kind"] == "ADVANCE" and skip_copy_wait:
            skip_copy_wait = False
            continue
        if shared and raw["kind"] == "COPY":
            raw["kind"] = "SHARE"
            skip_copy_wait = True
        elif raw["kind"] not in ("EVICT",):
            skip_copy_wait = False
        actions.append(Action(**raw))
        source_indices.append(index)
    return tuple(actions), tuple(source_indices)


def catalogues(scenario, evaluation):
    """Expose sufficient records for an independent stdlib physical audit."""
    from llm_infer_model.tensor.spec import ROUTE_ENDS
    state = evaluation.states[-1]
    tensors = []
    refs = {(tid, loc): identity for tid, loc, identity in state.storage_allocations}
    for tensor in scenario.workload.tensors:
        transfers = []
        for route, (src, dst) in ROUTE_ENDS.items():
            price = scenario.costs.transfer_cost(route, tensor.id)
            if price is not None:
                transfers.append(dict(src=src, dst=dst, duration_ns=price.duration_ns))
        tensors.append(dict(id=tensor.id, storage_bytes=tensor.storage_bytes, initial_locations=list(tensor.initial_locations),
            role=tensor.role, allocation_bytes=[(loc, scenario.size_alloc(tensor.id, loc)) for loc in ("dram", "vram")],
            copies=[],
            goal_location=scenario.goal.output_locations.get(tensor.id), transfers=transfers,
            allocation_refs=[(loc, refs.get((tensor.id, loc))) for loc in ("dram", "vram")]))
        tensors[-1]["copies"] = [(loc, dict(state.statuses(loc))[tensor.id]) for loc in ("dram", "vram", "ssd")]
    operations = []
    for operation in scenario.workload.operations:
        prices = []
        for device in scenario.mapspace.devices_for(operation.id):
            price = scenario.costs.compute_cost(operation.id, device)
            if price is not None:
                prices.append(dict(device=device, duration_ns=price.duration_ns,
                    allocated_workspace_bytes=scenario.workspace_alloc(operation.id, device)))
        operations.append(dict(id=operation.id, kind=operation.semantic_op, inputs=list(operation.inputs),
                               outputs=list(operation.outputs), prices=prices))
    return tensors, operations


def run_one(program, action_ns, point, total_capacity_bytes, transfer_models=None):
    from llm_infer_model.tensor.engine import evaluate_mapping
    from experiments.strategy_comparison_metrics import timeline_metrics
    from experiments.hardware_system_audit import audit_trace
    prices = response_costs(program, action_ns, point, transfer_models)
    scenario, workload, document = make_scenario(program, prices, point, total_capacity_bytes)
    actions, indices = lower_serial_actions(program, point.architecture == "shared")
    result = evaluate_mapping(scenario, actions)
    summary = dict(success=result.status == "valid", status=result.status, reason=result.reason,
        completed_makespan_ns=None, semantic_engine_makespan_ns=result.makespan_ns if result.status == "valid" else None,
        calibrated_hardware=False, measured_component_anchor=True, real_product=False,
        additional_workspace_assumption_bytes=0, native_workspace_measured=False,
        h2d_bytes=result.h2d_bytes, d2h_bytes=result.d2h_bytes, ssd_read_bytes=result.ssd_read_bytes,
        ssd_write_bytes=result.ssd_write_bytes, peak_dram_bytes=result.peak_dram_bytes,
        peak_vram_bytes=result.peak_vram_bytes, peak_shared_bytes=result.peak_shared_bytes,
        physical_memory_peaks_bytes=({"shared": result.peak_shared_bytes} if point.architecture == "shared"
                                    else {"dram": result.peak_dram_bytes, "vram": result.peak_vram_bytes}))
    summary["engine_error_code"] = result.error_code
    if result.error_code in ("CAPACITY_EXCEEDED", "initial_capacity_exceeded"):
        summary["status"] = "capacity_infeasible"
    events = [asdict(e) for e in result.events]
    summary["event_counts"] = dict(Counter(e["kind"] for e in events))
    if summary["success"]:
        # The serial response model includes host control actions. The tensor
        # engine's integer semantic clock is audited separately; neither clock
        # may change physical admission in this disturbance-free serial study.
        exact = sum((r["scenario_ns"] for r in prices), Fraction())
        summary.update(completed_makespan_ns=float(exact), exact_response_ns=pack(exact),
            semantic_rounding_ns=float(Fraction(result.makespan_ns) - sum(
                (r["scenario_ns"] for r in prices if r["modeled_kind"] in ("COMPUTE", "COPY")), Fraction())),
            control_and_visibility_ns=float(sum((r["scenario_ns"] for r in prices
                if r["modeled_kind"] not in ("COMPUTE", "COPY")), Fraction())),
            cpu_compute_ns=float(sum((r["scenario_ns"] for r, a in zip(prices, program["actions"])
                if a["kind"] == "COMPUTE" and a["device"] == "cpu"), Fraction())),
            gpu_compute_ns=float(sum((r["scenario_ns"] for r, a in zip(prices, program["actions"])
                if a["kind"] == "COMPUTE" and a["device"] == "gpu"), Fraction())),
            copy_response_ns=float(sum((r["scenario_ns"] for r in prices if r["modeled_kind"] == "COPY"), Fraction())))
        summary.update(timeline_metrics(result.events, result.makespan_ns))
        # Audit the model semantic clock, which excludes explicitly separate host
        # bookkeeping. This avoids claiming such increments are GPU work.
        tensors, operations = catalogues(scenario, result)
        semantic_summary = dict(summary, completed_makespan_ns=result.makespan_ns)
        hardware = dict(document["architecture"], memory_mode=point.architecture)
        receipt = audit_trace(events, operations, tensors, semantic_summary, hardware)
        summary["audit"] = receipt
        physical = [s.used_dram_bytes + s.used_vram_bytes for s in result.states]
        summary["peak_total_physical_bytes"] = result.peak_shared_bytes if point.architecture == "shared" else max(physical)
    else:
        tensors, operations = catalogues(scenario, result) if result.states else ([], [])
        summary["diagnostic_partial_ns"] = result.makespan_ns
        summary["rejected_action"] = asdict(actions[len(events)]) if len(events) < len(actions) else None
    return dict(summary=summary, events=events, tensor_catalogue=tensors, operation_catalogue=operations,
                scenario=document, workload=workload, response_ledger=pack(prices), action_source_indices=indices)


def select(rows):
    valid = [r for r in rows if r.get("success") and r.get("completed_makespan_ns") is not None]
    return min(valid, key=lambda r: (r["completed_makespan_ns"], r["candidate"])) if valid else None


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["workload"], row["point"]["id"]].append(row)
    baselines = {}
    for (workload, point), members in grouped.items():
        if point in ("separate-base", "shared-base"):
            complete = len(members) == 4 and all(r.get("success") or r.get("status") == "capacity_infeasible" for r in members)
            baselines[workload, members[0]["point"]["architecture"]] = select(members) if complete else None
    choices = []
    for (workload, point), members in sorted(grouped.items()):
        qualified = len(members) == 4 and all(r.get("success") or r.get("status") == "capacity_infeasible" for r in members)
        best = select(members) if qualified else None
        architecture = members[0]["point"]["architecture"]
        base = baselines.get((workload, architecture))
        fixed = next((r for r in members if base and r["candidate"] == base["candidate"]), None)
        choice = dict(workload=workload, point=members[0]["point"], candidate_count=len(members),
            successful_candidates=sum(bool(r.get("success")) for r in members),
            selected=None if best is None else best["candidate"],
            selected_ns=None if best is None else best["completed_makespan_ns"],
            frozen_reference_candidate=None if base is None else base["candidate"],
            fixed_reference_status=None if fixed is None else fixed.get("status"),
            fixed_reference_ns=fixed["completed_makespan_ns"] if fixed and fixed.get("success") else None,
            all_outcomes_available=qualified)
        choice["regret_without_reselection"] = (fixed["completed_makespan_ns"] / best["completed_makespan_ns"] - 1
            if fixed and fixed.get("success") and best else None)
        choice["fixed_policy_gain"] = (1 - fixed["completed_makespan_ns"] / base["completed_makespan_ns"]
            if fixed and fixed.get("success") and base else None)
        choice["reselected_total_gain"] = (1 - best["completed_makespan_ns"] / base["completed_makespan_ns"]
            if best and base else None)
        choices.append(choice)
    matched_architectures=[]
    joint_by_contract=defaultdict(dict)
    for row in choices:
        point=row["point"]
        if point["link"]==1:
            key=(row["workload"],point["cpu"],point["gpu"],point["capacity"])
            joint_by_contract[key][point["architecture"]]=row
    for key,pair in sorted(joint_by_contract.items()):
        if set(pair)!={"shared","separate"}:
            continue
        separate,shared=pair["separate"],pair["shared"]
        matched_architectures.append(dict(workload=key[0],cpu=key[1],gpu=key[2],capacity=key[3],
            separate_selected=separate["selected"],shared_selected=shared["selected"],
            separate_ns=separate["selected_ns"],shared_ns=shared["selected_ns"],
            ideal_shared_best_gain=(1-shared["selected_ns"]/separate["selected_ns"]
                if separate["selected_ns"] and shared["selected_ns"] else None),
            shared_only_feasible=shared["selected"] is not None and separate["selected"] is None
                if shared["all_outcomes_available"] and separate["all_outcomes_available"] else None,
            equal_total_physical_capacity=True,compute_cost_contract_unchanged=True,
            idealized_visibility_not_real_UMA=True))
    return dict(cases=len(rows), status_counts=dict(Counter(r.get("status") for r in rows)),
        successes=sum(bool(r.get("success")) for r in rows), successful_trace_audits=sum(bool(r.get("audit")) for r in rows),
        groups=len(grouped), choices=choices,architecture_comparisons=matched_architectures,
        fully_known_all_infeasible_groups=sum(r["all_outcomes_available"] and r["selected"] is None for r in choices),
        unresolved_groups=sum(not r["all_outcomes_available"] for r in choices),
        fixed_reference_infeasible_but_another_candidate_feasible_groups=sum(
            r["all_outcomes_available"] and r["selected"] is not None and r["fixed_reference_status"]=="capacity_infeasible" for r in choices),
        max_reselection_regret=max((r["regret_without_reselection"] for r in choices
                                   if r["regret_without_reselection"] is not None), default=None))


def load_anchors(directory):
    """Read frozen profile programs, never target heldout observations."""
    from llm_infer_model.performance.reusable_action_costs import predict_program
    directory = Path(directory).resolve()
    frozen_path = directory/"profile-frozen.json"
    frozen = read(frozen_path)
    anchors = {}
    manifest_path=directory/"manifest.json"
    manifest=read(manifest_path) if manifest_path.exists() else {}
    paths=[]
    for key,entry in manifest.get("programs",{}).items():
        if "profile" not in key or "heldout" in key:
            continue
        receipt=entry.get("program") if isinstance(entry,dict) else None
        if isinstance(receipt,dict) and receipt.get("path"):
            path=Path(receipt["path"])
            if not path.is_absolute():path=directory/path
            if receipt.get("sha256") and file_sha(path)!=receipt["sha256"]:
                raise ValueError("program source differs from E2A preparation receipt")
            paths.append(path)
    if not paths:
        paths=[path for path in sorted((directory/"programs").glob("*.json"))
               if "profile" in path.stem and "heldout" not in path.stem]
    for path in paths:
        program = read(path)
        if not isinstance(program, dict) or not program.get("bindings"):
            continue
        config = program.get("physical", {}).get("config", {})
        if config.get("hidden") != 512:
            continue
        binding = program["bindings"][0]
        shape = binding["shape"]
        workload = ({(64, 0): WORKLOADS[0], (1, 128): WORKLOADS[1]}).get(
            (shape["tokens"], shape["visible_history"]))
        if workload is None:
            continue
        compute = [a for a in program["actions"] if a["kind"] == "COMPUTE"]
        kinds = {b["kind"] for b in program["bindings"]}
        if "ffn_norm" in kinds:
            candidate = "S3"
        elif {a["device"] for a in compute} == {"cpu"}:
            candidate = "S0"
        elif {a["device"] for a in compute} == {"gpu"}:
            candidate = "S1"
        else:
            placements = {next(b["kind"] for b in program["bindings"]
                if b["operation_id"] == a["target_id"]): a["device"] for a in compute}
            if placements != {"projection": "gpu", "attention": "cpu", "dense_ffn": "gpu"}:
                raise ValueError("unrecognized mixed native candidate")
            candidate = "S2"
        key = (workload, candidate)
        if key in anchors:
            raise ValueError("duplicate frozen profile candidate: " + str(key))
        ledger = predict_program(program, frozen)
        if ledger.get("extra_overhead_rows"):
            raise ValueError("new nonzero extra mechanism needs an explicit E3 adapter")
        anchors[key] = dict(program=program, prices=[r["duration_ns"] for r in ledger["action_rows"]],
            prediction_ns=ledger["total_ns"], source=str(path), source_sha256=file_sha(path), ledger=ledger)
    if set(anchors) != set(itertools.product(WORKLOADS, CANDIDATES)):
        raise ValueError("missing exact H512 E2A anchor programs: " + str(set(itertools.product(WORKLOADS, CANDIDATES))-set(anchors)))
    prediction_path=directory/"predictions.json"
    predictions=read(prediction_path)
    prediction_index={(r["workload"],r["strategy"]):r for r in predictions}
    cross_checks=[]
    for (workload,candidate),anchor in sorted(anchors.items()):
        external_workload={WORKLOADS[0]:"h512-p64",WORKLOADS[1]:"h512-d1h128"}[workload]
        prediction=prediction_index[external_workload,int(candidate[1:])]
        expected=rational(prediction["prediction"]["duration_ns"])
        action_prices=[rational(value) for value in prediction["prediction"]["action_ns"]]
        keys=[r["key"] for r in anchor["ledger"]["action_rows"]]
        if expected!=anchor["prediction_ns"] or action_prices!=anchor["prices"] or keys!=prediction["keys"]:
            raise ValueError("profile-seed anchor differs from frozen heldout-program prediction: "+str((workload,candidate)))
        cross_checks.append(dict(workload=workload,candidate=candidate,heldout_program_key=prediction["program_key"],
            profile_program_path=anchor["source"],profile_seed=anchor["program"]["seed"],
            heldout_seed=manifest["programs"][prediction["program_key"]]["seed"],
            exact_profile_response_ns=pack(anchor["prediction_ns"]),exact_heldout_prediction_ns=pack(expected),
            semantic_keys_equal=True,all_action_prices_equal=True,exact_sums_equal=True))
    capacities = {}
    for workload in WORKLOADS:
        maximum_catalogue = max(sum(t["storage_bytes"] for t in anchor["program"]["graph"]["workload"]["tensors"])
                                for (w, _), anchor in anchors.items() if w == workload)
        half = (105 * maximum_catalogue + 99) // 100
        capacities[workload] = 2 * half
    return anchors, capacities, dict(path=str(frozen_path), sha256=file_sha(frozen_path),
        program_manifest=None if not manifest_path.exists() else dict(path=str(manifest_path),sha256=file_sha(manifest_path)),
        predictions=dict(path=str(prediction_path),sha256=file_sha(prediction_path)),
        cross_prediction_check=dict(passed=True,checked=8,required=8,entries=cross_checks))


def source_manifest():
    result = {}
    for folder in ("modeling/llm_infer_model", "mapping/tensor_mapping", "src/llm_simulator"):
        for path in sorted((ROOT/folder).rglob("*.py")):
            result[path.relative_to(ROOT).as_posix()] = file_sha(path)
    for relative in ("experiments/dac_validation_e3.py", "experiments/dac_validation_e3_audit.py", "experiments/hardware_system_audit.py",
                     "experiments/strategy_comparison_metrics.py", "tests/test_dac_validation_e3.py"):
        result[relative] = file_sha(ROOT/relative)
    return result


def robustness_analysis(rows, main_summary):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["workload"], row["point"]["architecture"], row["point"]["corner"],
                row["point"]["focus_direction"]].append(row)
    nominal = {(r["workload"], r["point"]["id"]): r for r in main_summary["choices"]}
    direction_id = {"cpu-faster": "cpu-0.8", "gpu-faster": "gpu-0.8",
                    "more-capacity": "capacity-1.2", "more-link-bandwidth": "link-1.2"}
    conditions = []
    for workload in WORKLOADS:
        for architecture in ("separate", "shared"):
            corners = []
            for corner in range(16):
                selected = {}
                for name, _, _ in directions(architecture):
                    cases = grouped[workload, architecture, corner, name]
                    complete = len(cases) == 4 and all(r.get("success") or r.get("status") == "capacity_infeasible" for r in cases)
                    selected[name] = select(cases) if complete else None
                base = selected["baseline"]
                gains = {name: (1-best["completed_makespan_ns"]/base["completed_makespan_ns"]
                               if base and best else None)
                         for name, best in selected.items() if name != "baseline"}
                complete = all(value is not None for value in gains.values())
                maximum = max(gains.values()) if complete else None
                leaders = [name for name, value in gains.items()
                           if maximum is not None and abs(value-maximum) <= 1e-12]
                corners.append(dict(corner=corner, gains=gains, leaders=leaders,
                    complete=complete, all_nonnegative=complete and min(gains.values()) >= 0))
            qualified = all(c["complete"] for c in corners)
            robust = []
            nominal_gains = {}
            for name, _, _ in directions(architecture)[1:]:
                condition = nominal.get((workload, architecture+"-"+direction_id[name]))
                gain = condition.get("reselected_total_gain") if condition else None
                nominal_gains[name] = gain
                if qualified and all(name in c["leaders"] and c["gains"][name] > 0 for c in corners):
                    robust.append(name)
            conditions.append(dict(workload=workload, architecture=architecture, completed_corners=sum(c["complete"] for c in corners),
                nominal_gains=nominal_gains, robust_leading_directions=robust,
                main_recommendations=[name for name in robust if nominal_gains[name] is not None and nominal_gains[name] >= .05],
                finite_scenario_only=True, real_UMA_robustness="unresolved" if architecture == "shared" else None,
                corners=corners))
    return dict(additional_cases=len(rows), expected_additional_cases=1152, conditions=conditions,
                scope="predeclared finite corners; no probability, interior guarantee, equal-cost investment or real UMA validation")


def run_campaign(out, anchors_directory, transfer_models_path=None, include_robustness=True):
    from experiments.hardware_system_audit import AuditError
    out = Path(out).resolve()
    frozen_inventory = read(out/"inventory.json")
    if frozen_inventory != inventory():
        raise ValueError("prepared matrix/assumptions changed")
    if (out/"execution-manifest.json").exists():
        raise ValueError("write-once campaign; use a new output directory for another version")
    anchors, capacities, frozen_receipt = load_anchors(anchors_directory)
    transfer_models = read(transfer_models_path) if transfer_models_path else read(frozen_receipt["path"]).get("transfer_decomposition")
    if transfer_models and "models" in transfer_models:
        transfer_models = transfer_models["models"]
    if transfer_models:
        transfer_models = {route.replace("->","_to_"): dict(model,
            slope_ns_per_byte=model.get("slope_ns_per_byte",model.get("payload_ns_per_byte")))
            for route,model in transfer_models.items()}
    before = source_manifest()
    manifest = dict(schema=VERSION, inventory_sha256=file_sha(out/"inventory.json"),
        frozen_profile=frozen_receipt, anchor_programs={f"{w}/{c}": dict(path=a["source"],sha256=a["source_sha256"])
            for (w,c),a in anchors.items()}, source_sha256=before, total_capacity_bytes=capacities,
        transfer_models=(dict(source="frozen_profile.transfer_decomposition", models=transfer_models)
            if transfer_models_path is None else dict(path=str(Path(transfer_models_path).resolve()),sha256=file_sha(transfer_models_path))),
        main_cases=304, robustness_cases=1152 if include_robustness else 0,
        results_claim="measured-component anchored scenarios; complete-strategy accuracy qualification remains E2A's separate result")
    write(out/"execution-manifest.json", manifest)
    write(out/"anchor-cross-check.json",frozen_receipt["cross_prediction_check"])
    started = time.perf_counter()
    families = (("main",main_points()), ("robustness",robustness_points())) if include_robustness else (("main",main_points()),)
    family_rows = {}
    for family, points in families:
        rows = []
        for point in points:
            for workload, candidate in itertools.product(WORKLOADS, CANDIDATES):
                anchor = anchors[workload,candidate]
                case_directory = out/family/workload/point.id/candidate
                case_started = time.perf_counter()
                row = dict(workload=workload,candidate=candidate,point=asdict(point),success=False,
                           completed_makespan_ns=None,artifact_directory=case_directory.relative_to(out).as_posix())
                try:
                    evidence = run_one(anchor["program"],anchor["prices"],point,capacities[workload],transfer_models)
                    row.update(evidence.pop("summary"))
                    write(case_directory/"evidence.json",evidence)
                    row["evidence_sha256"] = file_sha(case_directory/"evidence.json")
                    if point.id == "separate-base" and row["success"]:
                        if abs(row["completed_makespan_ns"]-float(anchor["prediction_ns"])) > 1e-6:
                            raise ValueError("reference response ledger differs from E2A frozen prediction")
                        row["reference_cost_anchor_exact"] = True
                except UnknownScenarioCost as error:
                    row.update(success=False,completed_makespan_ns=None,status="unsupported_contract",reason=str(error),
                               error_type=type(error).__name__)
                except AuditError as error:
                    row.update(success=False,completed_makespan_ns=None,status="audit_failure",reason=str(error),
                               error_type=type(error).__name__)
                except (KeyError,ValueError) as error:
                    row.update(success=False,completed_makespan_ns=None,status="execution_contract_failure",reason=str(error),
                               error_type=type(error).__name__)
                row["harness_wall_seconds"] = time.perf_counter()-case_started
                write(case_directory/"summary.json",row)
                rows.append(row)
            print(json.dumps(dict(family=family,point=point.id,completed_cases=len(rows))),flush=True)
        write(out/f"{family}-records.json", rows)
        family_rows[family] = rows
    main_summary = summarize(family_rows["main"])
    robustness = robustness_analysis(family_rows.get("robustness",[]),main_summary)
    after = source_manifest()
    if before != after:
        raise ValueError("executed sources changed during campaign")
    for receipt in [frozen_receipt, *manifest["anchor_programs"].values()]:
        if file_sha(receipt["path"]) != receipt["sha256"]:
            raise ValueError("frozen E2A source changed during campaign")
    if frozen_receipt.get("program_manifest"):
        receipt=frozen_receipt["program_manifest"]
        if file_sha(receipt["path"]) != receipt["sha256"]:
            raise ValueError("frozen E2A manifest changed during campaign")
    receipt=frozen_receipt["predictions"]
    if file_sha(receipt["path"])!=receipt["sha256"]:
        raise ValueError("frozen E2A predictions changed during campaign")
    summary = dict(schema=VERSION, main=main_summary,robustness=robustness,
        calibrated_hardware=False, native_workspace_measured=False,shared_idealized=True,
        main_scan_complete=len(family_rows["main"])==304,
        all_main_successful_traces_audited=main_summary["successes"]==main_summary["successful_trace_audits"],
        finite_scenario_contract_satisfied=main_summary["cases"]==304 and all(r["all_outcomes_available"] for r in main_summary["choices"]),
        real_anchor_accuracy_status="not assessed by this hardware runner; retain E2A's independent quality gates",
        full_E3_protocol_status="not certified here: requires E2A anchor qualification and declared architectural limitations",
        harness_wall_seconds=time.perf_counter()-started, executed_sources_unchanged=True,
        comments=["Full-strategy independent accuracy is separately evaluated by E2A, not implied by component anchoring.",
            "Shared-storage results use idealized visibility and explicit workspace/handoff assumptions.",
            "Faster effective action duration does not identify a FLOP/s, clock or memory-bandwidth upgrade.",
            "Finite candidate enumeration is not a new search algorithm; failed partial executions are never ranked."])
    write(out/"summary.json",summary)
    write(out/"completion.json",dict(complete=True,main_cases=len(family_rows["main"]),robustness_cases=len(family_rows.get("robustness",[])),
        summary_sha256=file_sha(out/"summary.json"),main_records_sha256=file_sha(out/"main-records.json"),
        source_manifest_sha256=digest(before),inventory_sha256=file_sha(out/"inventory.json")))
    return summary


def export_report(out):
    """Standalone plotting consumes retained results; never changes predictions."""
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    out=Path(out)
    summary=read(out/"summary.json")
    rows=read(out/"main-records.json")
    def save_figure(fig,path,dpi):
        if "LAYOUT FIXTURE" in summary.get("scope","") and not getattr(fig,"_layout_fixture_marked",False):
            fig.text(.5,.5,"LAYOUT FIXTURE / NO RESULTS",ha="center",va="center",rotation=30,
                     fontsize=22,color="#9a2222",alpha=.35)
            fig._layout_fixture_marked=True
        fig.savefig(path,dpi=dpi)
    fields=("workload","candidate","point","architecture","cpu","gpu","link","capacity",
            "status","success","completed_makespan_ns","cpu_compute_ns","gpu_compute_ns",
            "copy_response_ns","control_and_visibility_ns","h2d_bytes","d2h_bytes",
            "peak_total_physical_bytes","harness_wall_seconds")
    with (out/"main-cases.csv").open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator="\n")
        writer.writeheader()
        for row in rows:
            flat={key:row.get(key) for key in fields}
            flat.update(point=row["point"]["id"],**{k:row["point"][k] for k in ("architecture","cpu","gpu","link","capacity")})
            writer.writerow(flat)
    plots=out/"figures"
    plots.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":12,"axes.spines.top":False,"axes.spines.right":False})
    directions_ids=(("CPU -20%","cpu-0.8"),("GPU -20%","gpu-0.8"),
                    ("Cap. +20%","capacity-1.2"),("Link BW +20%","link-1.2"))
    index={(r["workload"],r["point"]["id"]):r for r in summary["main"]["choices"]}
    titles={WORKLOADS[0]:"H512 / P64",WORKLOADS[1]:"H512 / D1, history 128"}
    def sensitivity_panel(ax,workload,architecture):
        pairs=directions_ids if architecture=="separate" else directions_ids[:3]
        for k,(label,suffix) in enumerate(pairs):
            choice=index[workload,architecture+"-"+suffix]
            gain=choice.get("reselected_total_gain")
            if gain is None:
                state="infeasible" if choice["all_outcomes_available"] else "unresolved"
                ax.text(k,0,state,rotation=90,ha="center",va="bottom",color="#666",fontsize=10)
                ax.scatter([k],[0],marker="x",color="#666",zorder=4)
            else:
                ax.bar(k,100*gain,color="#2c6ca2" if gain>=0 else "#b54b44",width=.65)
                ax.annotate(f"{gain*100:.1f}%",(k,gain*100),xytext=(0,3 if gain>=0 else -14),textcoords="offset points",ha="center",fontsize=10)
        ax.axhline(0,color="#aaa",lw=.8)
        ax.set_xticks(range(len(pairs)),[p[0] for p in pairs],rotation=18,ha="right",fontsize=10)
        ax.set_ylabel("Latency reduction (%)",fontsize=11)
        ax.set_title(titles[workload]+"\n"+("Idealized shared" if architecture=="shared" else "Separate pools"),fontsize=12)
        if ax.get_ylim()[1]<1:ax.set_ylim(top=1)
        ax.margins(y=.28)
    fig,axes=plt.subplots(2,2,figsize=(7.5,6.9),layout="constrained")
    for i,workload in enumerate(WORKLOADS):
        for j,architecture in enumerate(("separate","shared")):
            sensitivity_panel(axes[i,j],workload,architecture)
    fig.suptitle("Effective-action scenarios; idealized shared visibility",fontsize=13)
    for extension in ("png","svg"):
        save_figure(fig,plots/f"e3-nominal-sensitivity.{extension}",180)
    plt.close(fig)
    for workload,architecture in itertools.product(WORKLOADS,("separate","shared")):
        fig,ax=plt.subplots(figsize=(4.3,3.7),layout="constrained")
        sensitivity_panel(ax,workload,architecture)
        for extension in ("png","svg"):
            save_figure(fig,plots/f"e3-sensitivity-{workload}-{architecture}.{extension}",210)
        plt.close(fig)
    def architecture_panel(ax,workload):
        for offset,architecture,color in ((-.18,"separate","#2c6ca2"),(.18,"shared","#55a487")):
            values=[]
            for candidate in CANDIDATES:
                row=next(r for r in rows if r["workload"]==workload and r["point"]["id"]==architecture+"-base" and r["candidate"]==candidate)
                values.append(row["completed_makespan_ns"]/1e6 if row["success"] else np.nan)
                if not row["success"]:
                    label="infeasible" if row["status"]=="capacity_infeasible" else "unresolved"
                    ax.text(len(values)-1+offset,0,label,rotation=90,fontsize=10,color="#666",ha="center",va="bottom")
                    ax.scatter([len(values)-1+offset],[0],marker="x",color="#666",zorder=4)
            ax.bar(np.arange(4)+offset,values,width=.34,color=color,label="Idealized shared" if architecture=="shared" else "Separate")
        ax.set_xticks(range(4),CANDIDATES)
        ax.set_title(titles[workload],fontsize=12)
        ax.set_ylabel("Predicted replay mean (ms)",fontsize=11)
        ax.legend(frameon=False,fontsize=10)
    fig,axes=plt.subplots(1,2,figsize=(7.5,3.9),layout="constrained")
    for ax,workload in zip(axes,WORKLOADS):
        architecture_panel(ax,workload)
    fig.suptitle("Equal capacity and compute responses; idealized shared architecture",fontsize=12)
    for extension in ("png","svg"):
        save_figure(fig,plots/f"e3-architecture-comparison.{extension}",180)
    plt.close(fig)
    for workload in WORKLOADS:
        fig,ax=plt.subplots(figsize=(4.3,3.7),layout="constrained")
        architecture_panel(ax,workload)
        for extension in ("png","svg"):
            save_figure(fig,plots/f"e3-architecture-{workload}.{extension}",210)
        plt.close(fig)
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    palette=["#d6e6f5","#d2ecdf","#ffe4bd","#e6dcf3","#d4d4d4","#fff7df"]
    state_index={"S0":0,"S1":1,"S2":2,"S3":3}
    phase_data={}
    def phase_panel(ax,workload,architecture):
        axes_names=("cpu","gpu","link","capacity") if architecture=="separate" else ("cpu","gpu","capacity")
        values=[]
        labels=[]
        for axis in axes_names:
            value_row=[]
            label_row=[]
            for factor in LEVELS:
                pid=architecture+"-base" if factor==1 else f"{architecture}-{axis}-{factor:g}"
                choice=index[workload,pid]
                selected=choice["selected"]
                label=selected if selected else ("X" if choice["all_outcomes_available"] else "?")
                value_row.append(state_index[selected] if selected else (4 if label=="X" else 5))
                label_row.append(label)
                phase_data[workload,architecture,axis,factor]=dict(workload=workload,architecture=architecture,axis=axis,factor=factor,
                    selected=selected,label=label,qualified=choice["all_outcomes_available"])
            values.append(value_row);labels.append(label_row)
        ax.imshow(values,cmap=ListedColormap(palette),vmin=-.5,vmax=5.5,aspect="auto")
        for y,row_labels in enumerate(labels):
            for x,label in enumerate(row_labels):ax.text(x,y,label,ha="center",va="center",fontsize=12)
        ax.set_xticks(range(5),[f"{v:g}" for v in LEVELS],fontsize=10)
        ax.set_yticks(range(len(axes_names)),[{"cpu":"CPU duration","gpu":"GPU duration","link":"Payload BW","capacity":"Capacity"}[a] for a in axes_names],fontsize=10)
        ax.set_xlabel("Factor relative to anchor",fontsize=11)
        ax.set_title(titles[workload]+"\n"+("Idealized shared" if architecture=="shared" else "Separate pools"),fontsize=12)
        ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False)
    fig,axes=plt.subplots(2,2,figsize=(7.5,6.5),layout="constrained")
    for i,workload in enumerate(WORKLOADS):
        for j,architecture in enumerate(("separate","shared")):
            phase_panel(axes[i,j],workload,architecture)
    fig.suptitle("Finite candidate selection: X = infeasible; ? = unresolved",fontsize=13)
    fig.legend(handles=[Patch(facecolor=palette[i],label=c) for i,c in enumerate(CANDIDATES)],
               loc="outside lower center",ncol=4,frameon=False,fontsize=11)
    for extension in ("png","svg"):
        save_figure(fig,plots/f"e3-policy-phase-map.{extension}",180)
    plt.close(fig)
    for workload,architecture in itertools.product(WORKLOADS,("separate","shared")):
        fig,ax=plt.subplots(figsize=(4.3,3.7),layout="constrained")
        phase_panel(ax,workload,architecture)
        ax.set_title(ax.get_title()+"\nX: infeasible; ?: unresolved",fontsize=11)
        for extension in ("png","svg"):
            save_figure(fig,plots/f"e3-phase-{workload}-{architecture}.{extension}",210)
        plt.close(fig)
    write(out/"phase-map-data.json",[value for _,value in sorted(phase_data.items())])
    text=["# E3 hardware scenario study", "", "This study is anchored to independently measured semantic action responses. It is not calibrated hardware or commercial-device ranking.", "",
          f"Main cases: {summary['main']['cases']}/304. Successful: {summary['main']['successes']}; independently audited successful traces: {summary['main']['successful_trace_audits']}.",
          f"Main status counts: {summary['main']['status_counts']}.",
          f"Robustness cases: {summary['robustness']['additional_cases']}/1152.",
          f"Finite scenario contract satisfied: {summary['finite_scenario_contract_satisfied']}. Sources unchanged during execution: {summary['executed_sources_unchanged']}.",
          f"Real anchor accuracy: {summary['real_anchor_accuracy_status']}.", "",
          "## Fixed assumptions", "", "One CPU/GPU, F32, serial launches, one complete declared Dense block, P64/history0 or D1/history128. Initial weights/input/history in DRAM; terminal hidden/KV in DRAM. S3's FFN parts run serially.", "",
          "Capacity is normalized from untimed tensor inventories with equal separate DRAM/VRAM pools. Shared capacity equals their sum and allocations are counted once. Additional backend-private workspace is unknown and is explicitly assumed zero for these capacity scenarios.", "",
          "CPU/GPU factors scale effective completed compute-action responses, not FLOP/s or clocks. Link factors scale only the identified payload term and preserve fixed/per-coordinate response residuals. Unqualified decomposition yields unsupported cases, never fabricated prices.", "",
          "Compute-action and copy-response totals include their native dispatch/completion boundaries. They are host-completed action spans, not pure GPU busy time or a measurement of physical utilization. Separately measured logical controller responses remain separate, so synchronization is not added twice.", "",
          "Shared COPY becomes zero-payload SHARE; compute memory work stays in the measured action responses. Coherence, migration and shared-bandwidth contention are not measured. The 0/10 us additional SHARE interval is a prespecified hypothetical sensitivity bound, not an empirically validated UMA bound.", "",
          "## Finite-corner findings", ""]
    for condition in summary["robustness"]["conditions"]:
        text.append(f"- {condition['workload']} / {condition['architecture']}: completed {condition['completed_corners']}/16 corners; robust leading directions {condition['robust_leading_directions']}; nominal>=5% recommendations {condition['main_recommendations']}.")
    text.extend(["", "A direction is called robust only within the declared finite corners when it remains a beneficial leader at all 16. No probability or claim over the interior follows. Real UMA robustness remains unresolved.", "",
                 "## Reproduction", "", "Run preparation once; run measurements separately; then invoke the run command with the frozen E2A directory and qualified transfer-model file. Every output case retains its source action ledger, scenario and event/catalogue evidence. Failed partial execution times never enter strategy ranking."])
    (out/"REPORT.md").write_text("\n".join(text)+"\n",encoding="utf-8",newline="\n")
    write(out/"plot-data.json",dict(choices=summary["main"]["choices"],robustness=summary["robustness"]["conditions"]))
    return dict(report=str(out/"REPORT.md"),figures=str(plots))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare","run","report"))
    parser.add_argument("--out", type=Path, default=ROOT/"build/dac-validation-20261025/e3-v1")
    parser.add_argument("--anchors",type=Path)
    parser.add_argument("--transfer-models",type=Path)
    parser.add_argument("--no-robustness",action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        args.out.mkdir(parents=True, exist_ok=False)
        write(args.out/"inventory.json", inventory())
        write(args.out/"source-receipt.json", {str(Path(__file__).relative_to(ROOT)): file_sha(__file__)})
        print(json.dumps(dict(status="prepared_not_executed", main_cases=304, robustness_cases=1152, output=str(args.out))))
    elif args.command == "run":
        if args.anchors is None:
            parser.error("--anchors is required")
        summary=run_campaign(args.out,args.anchors,args.transfer_models,not args.no_robustness)
        print(json.dumps({key:value for key,value in summary.items() if key not in ("main","robustness")}))
    elif args.command == "report":
        print(json.dumps(export_report(args.out)))


if __name__ == "__main__":
    main()
