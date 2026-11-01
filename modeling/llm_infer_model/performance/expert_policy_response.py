"""Serial completed-response model for the bounded physical MoE policy probe.

The workload supplies expert inputs/routes; the mapping compiler supplies the
action list. Profiling supplies independent exact response cells. This module
validates execution obligations and computes their serial composition. It does
not measure, fit, infer operating-system behavior, or add a startup correction.

EXPERT prices include gather, activation copies, complete SwiGLU computation,
output return, and host output storage. LOAD prices include all three expert
weight copies. COMBINE includes the weighted residual. Those costs must not be
added a second time. Allocated GPU weight slots and occupied slots are distinct;
unknown native/CUDA workspace prevents a claim about total device capacity.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from math import isfinite

from .parameters import InvalidModel, UnknownParameter, integer


PROGRAM_SCHEMA = "moe-expert-policy-native-1"
CALIBRATION_SCHEMA = "moe-expert-policy-calibration-1"
RESULT_SCHEMA = "moe-expert-policy-prediction-1"
TIMING_BOUNDARY = "serial_host_expert_response_v1"


def _number(value, name, *, minimum=None):
    if type(value) not in (int, float) or not isfinite(value):
        raise InvalidModel(f"{name} must be a finite number")
    if minimum is not None and value < minimum:
        raise InvalidModel(f"{name} must be >= {minimum}")
    return value


def _object(value, name):
    if not isinstance(value, dict):
        raise InvalidModel(f"{name} must be an object")
    return value


def _required(value, key, name):
    if key not in value:
        raise InvalidModel(f"{name} requires {key}")
    return value[key]


def _validate(program):
    """Return exact action coordinates and a reference-checkable byte ledger."""
    _object(program, "program")
    if program.get("schema") != PROGRAM_SCHEMA:
        raise InvalidModel(f"program schema must be {PROGRAM_SCHEMA}")
    config = _object(_required(program, "config", "program"), "config")
    bounds = {"hidden": (1, 4096), "intermediate": (1, 16384),
              "experts": (1, 256), "gpu_slots": (0, 256),
              "threads": (1, 64), "cuda_device": (0, None)}
    for field, (minimum, maximum) in bounds.items():
        value = integer(_required(config, field, "config"), field, minimum)
        if maximum is not None and value > maximum:
            raise InvalidModel(f"{field} exceeds native supported bound {maximum}")
    if config["gpu_slots"] > config["experts"]:
        raise InvalidModel("gpu_slots exceeds the number of experts")
    seed = integer(_required(program, "seed", "program"), "seed")
    if seed > 2**32 - 1:
        raise InvalidModel("seed must fit an unsigned 32-bit value")
    guards = _object(_required(program, "resource_guards", "program"), "resource_guards")
    for field in ("host_reserve_bytes", "gpu_reserve_bytes"):
        integer(_required(guards, field, "resource_guards"), field)

    h, width, experts, slots = (config[key] for key in
                               ("hidden", "intermediate", "experts", "gpu_slots"))
    expert_bytes = 3 * h * width * 4
    if expert_bytes * experts > 2 * 1024**3:
        raise InvalidModel("host expert inventory exceeds native 2 GiB weight envelope")
    steps = _required(program, "steps", "program")
    if not isinstance(steps, list) or not steps:
        raise InvalidModel("steps must be a nonempty list")
    coordinates = []
    for index, step in enumerate(steps):
        _object(step, f"step {index}")
        inputs = _required(step, "input", f"step {index}")
        routes = _required(step, "routes", f"step {index}")
        scores = _required(step, "scores", f"step {index}")
        if not isinstance(inputs, list) or not 1 <= len(inputs) <= 1024:
            raise InvalidModel("step input must contain 1..1024 token rows")
        if not isinstance(routes, list) or not isinstance(scores, list):
            raise InvalidModel("routes and scores must be lists")
        if len(routes) != len(inputs) or len(scores) != len(inputs):
            raise InvalidModel("input, route and score token counts differ")
        expert_rows = Counter()
        top_k = None
        for row, (values, selected, weights) in enumerate(zip(inputs, routes, scores)):
            if not isinstance(values, list) or len(values) != h:
                raise InvalidModel("input hidden dimension differs from configuration")
            for value in values:
                _number(value, "input element")
            if not isinstance(selected, list) or not selected:
                raise InvalidModel("each route must select at least one expert")
            if not isinstance(weights, list) or len(weights) != len(selected):
                raise InvalidModel("route and score dimensions differ")
            if top_k is None:
                top_k = len(selected)
            if len(selected) != top_k:
                raise InvalidModel("exact COMBINE coordinate requires constant top-k within a step")
            seen = set()
            for expert in selected:
                integer(expert, "routed expert")
                if expert >= experts or expert in seen:
                    raise InvalidModel("route contains an out-of-range or duplicate expert")
                seen.add(expert)
                expert_rows[expert] += 1
            for weight in weights:
                _number(weight, "route score", minimum=0)
            if abs(sum(weights) - 1.0) >= 1e-5:
                raise InvalidModel("route scores must sum to one")
        coordinates.append({"q": len(inputs), "topk": top_k,
                            "expert_rows": expert_rows, "done": set(), "combined": False})

    actions = _required(program, "actions", "program")
    if not isinstance(actions, list) or not actions:
        raise InvalidModel("actions must be a nonempty list")
    owners = [None] * slots
    rows = []
    peak_occupied = 0
    counts = Counter()
    weight_h2d_bytes = activation_h2d_bytes = output_d2h_bytes = 0
    for index, action in enumerate(actions):
        _object(action, f"action {index}")
        kind = action.get("kind")
        if kind not in ("LOAD", "RELEASE", "EXPERT", "COMBINE"):
            raise InvalidModel(f"unsupported action kind: {kind!r}")
        row = {"index": index, "kind": kind}
        counts[kind] += 1
        if kind in ("LOAD", "RELEASE", "EXPERT"):
            expert = integer(_required(action, "expert", f"action {index}"), "expert")
            if expert >= experts:
                raise InvalidModel("action expert is outside the declared inventory")
            row["expert"] = expert
        if kind in ("LOAD", "RELEASE"):
            slot = integer(_required(action, "slot", f"action {index}"), "slot")
            if slot >= slots:
                raise InvalidModel("action slot exceeds physical GPU slot capacity")
            if kind == "LOAD":
                if owners[slot] is not None:
                    raise InvalidModel("LOAD overwrites an occupied slot without RELEASE")
                owners[slot] = expert
                weight_h2d_bytes += expert_bytes
            else:
                if owners[slot] != expert:
                    raise InvalidModel("RELEASE has absent or mismatched slot ownership")
                owners[slot] = None
            row.update(slot=slot, price_key=kind)
        else:
            step_index = integer(_required(action, "step", f"action {index}"), "step")
            if step_index >= len(coordinates):
                raise InvalidModel("action step is outside the workload")
            step = coordinates[step_index]
            row["step"] = step_index
            if step["combined"]:
                raise InvalidModel("duplicate COMBINE or EXPERT after combined output")
            if kind == "EXPERT":
                if expert not in step["expert_rows"]:
                    raise InvalidModel("EXPERT is not selected by the declared route")
                if expert in step["done"]:
                    raise InvalidModel("selected expert executes more than once in a step")
                device = action.get("device")
                if device not in ("cpu", "gpu"):
                    raise InvalidModel("EXPERT device must be cpu or gpu")
                q = step["expert_rows"][expert]
                if device == "gpu":
                    slot = integer(_required(action, "slot", f"action {index}"), "slot")
                    if slot >= slots or owners[slot] != expert:
                        raise InvalidModel("GPU EXPERT lacks its weights in the selected slot")
                    row["slot"] = slot
                    activation_h2d_bytes += q * h * 4
                    output_d2h_bytes += q * h * 4
                step["done"].add(expert)
                row.update(device=device, q=q, price_key=f"EXPERT:{device}:q={q}")
            else:
                if step["done"] != set(step["expert_rows"]):
                    raise InvalidModel("COMBINE precedes required expert outputs")
                step["combined"] = True
                row.update(q=step["q"], topk=step["topk"],
                           price_key=f"COMBINE:q={step['q']}:topk={step['topk']}")
        peak_occupied = max(peak_occupied, sum(owner is not None for owner in owners))
        rows.append(row)
    if any(not step["combined"] for step in coordinates):
        raise InvalidModel("program omits one or more step COMBINE outputs")
    uses_gpu = any(row["kind"] == "LOAD" or row.get("device") == "gpu" for row in rows)
    ledger = {
        "action_counts": dict(counts),
        "expert_bytes": expert_bytes,
        "weight_h2d_bytes": weight_h2d_bytes,
        "activation_h2d_bytes": activation_h2d_bytes,
        "total_h2d_bytes": weight_h2d_bytes + activation_h2d_bytes,
        "total_d2h_bytes": output_d2h_bytes,
        "peak_resident_experts": peak_occupied,
        "peak_resident_weight_bytes": peak_occupied * expert_bytes,
        "host_weight_logical_bytes": experts * expert_bytes,
        "gpu_weight_logical_bytes": slots * expert_bytes if uses_gpu else 0,
        "workspace_bytes": None,
        "total_device_capacity_certified": False,
        "causal_policy_certified": False,
        "boundary": TIMING_BOUNDARY,
    }
    return rows, ledger


def action_price_keys(program):
    """Validate obligations, then return one exact calibration key per action."""
    rows, _ = _validate(program)
    return tuple(row["price_key"] for row in rows)


def evaluate_expert_policy(program, calibration):
    """Predict a validated serial action program from frozen exact cell prices.

    ``calibration`` has schema ``moe-expert-policy-calibration-1``, an exact
    copy of the program's ``config``, and ``prices_ns``. Keys are LOAD, RELEASE,
    EXPERT:{cpu|gpu}:q=N and COMBINE:q=N:topk=K. Here expert q counts only token
    rows routed to that expert. Prices may be nonnegative finite numbers; a
    missing/None required cell raises UnknownParameter. No interpolation or
    substitution is permitted. Policy causality requires a separate prefix
    invariance test; it cannot be established from a completed action list.
    """
    rows, ledger = _validate(program)
    _object(calibration, "calibration")
    if calibration.get("schema") != CALIBRATION_SCHEMA:
        raise InvalidModel(f"calibration schema must be {CALIBRATION_SCHEMA}")
    if calibration.get("config") != program["config"]:
        raise UnknownParameter("calibration configuration differs from the requested implementation")
    prices = _object(_required(calibration, "prices_ns", "calibration"), "prices_ns")
    current = 0.0
    breakdown = defaultdict(float)
    priced = []
    for row in rows:
        key = row["price_key"]
        if key not in prices or prices[key] is None:
            raise UnknownParameter(f"missing exact inclusive expert response: {key}")
        duration = float(_number(prices[key], f"price {key}", minimum=0))
        category = f"EXPERT:{row['device']}" if row["kind"] == "EXPERT" else row["kind"]
        breakdown[category] += duration
        priced.append(dict(row, start_ns=current, duration_ns=duration, end_ns=current + duration))
        current += duration
    return dict(schema=RESULT_SCHEMA, duration_ns=current, actions=priced,
                breakdown_ns=dict(breakdown), config=dict(program["config"]), **ledger)
