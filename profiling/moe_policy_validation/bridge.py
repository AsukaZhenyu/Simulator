"""Compile Python expert recipes to a bounded serial native replay program.

This adapter reuses CompositionPolicy and compile_command. Its explicit opaque
host-response macro differs from the full M2 layer: gather and activation I/O
are inside EXPERT; only expert weight LOAD/RELEASE are separate storage actions.
The modeling expert-policy evaluator validates the emitted program and computes
latency. The bridge is not a replacement full-layer TensorRuntime backend.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from copy import deepcopy

from llm_simulator.expert_recipes import expert_recipe, expert_recipe_scope
from llm_infer_model.m2.causal import DevicePrice, PublicExpert, PublicOperation, PublicTensor, RevealedRoute, TransferPrice
from llm_infer_model.tensor.environment import MemoryUsage
from tensor_mapping.causal import compile_command, observation_hash
from tensor_mapping.composition_observation import CompositionObservation
from tensor_mapping.composition_policy import CompositionPolicy, parse_composition_policy
from tensor_mapping.python_frontend import Scheduling


BOUNDARY = "serial_host_expert_response_v1"


def compile_expert_program(*, config, steps, seed, recipe=None, prices, resource_guards,
                           retention="lru", scheduling=None, decision_cost_view="full_path"):
    """Return {program, compilation_evidence} without touching the hardware.

    prices is a frozen moe-expert-policy-calibration-1 dictionary with exact
    config, explicit boundary, and prices_ns (LOAD, RELEASE, EXPERT:cpu:q=N,
    EXPERT:gpu:q=N, COMBINE:q=N:topk=K). Missing prices raise an error.
    Serial execution is the only supported envelope; no overlap is invented.
    """
    config, steps, prices = deepcopy(config), deepcopy(steps), deepcopy(prices)
    if retention not in ("lru", "policy", "release_after_use"):
        raise ValueError("retention must be policy, lru or release_after_use")
    if decision_cost_view not in ("full_path", "compute_only"):
        raise ValueError("decision_cost_view must be full_path or compute_only")
    if (recipe is None) == (scheduling is None):
        raise ValueError("supply exactly one ordinary Scheduling or named recipe")
    if scheduling is not None and not isinstance(scheduling, Scheduling):
        raise ValueError("scheduling must be an ordinary Scheduling declaration")
    if prices.get("schema") != "moe-expert-policy-calibration-1" or prices.get("config") != config:
        raise ValueError("calibration schema/config must match the native program exactly")
    if prices.get("boundary") != BOUNDARY:
        raise ValueError("cost boundary must be serial_host_expert_response_v1; activation I/O cannot be double-counted")
    if any(config.get(key, False) for key in ("overlap", "allow_cpu_gpu_overlap", "allow_copy_compute_overlap")):
        raise ValueError("native expert replay supports serial actions only")
    for key in ("hidden", "intermediate", "experts", "gpu_slots", "threads"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"config.{key} must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an explicit uint32")
    if not isinstance(steps, list) or not steps:
        raise ValueError("steps must be a nonempty list")
    if set(resource_guards) != {"host_reserve_bytes", "gpu_reserve_bytes"} or any(
            type(value) is not int or value < 0 for value in resource_guards.values()):
        raise ValueError("explicit nonnegative host/GPU reserves are required")

    def price(key):
        value = prices["prices_ns"].get(key)
        if type(value) is not int or value < 0:
            raise ValueError(f"missing nonnegative integer calibration price: {key}")
        return value

    count = config["experts"]
    weight_bytes = 12 * config["hidden"] * config["intermediate"]
    budget = config["gpu_slots"] * weight_bytes
    scheduling = (expert_recipe(recipe, gpu_weight_budget_bytes=budget)
                  if scheduling is None else deepcopy(scheduling))
    sections = scheduling.sections()
    if set(sections) != {"policy", "cache", "prefetch", "transport"}:
        raise ValueError("serial native bridge accepts scheduling/storage only, not resource reconfiguration")
    if sections["prefetch"]["enabled"]:
        raise ValueError("serial native bridge does not implement prefetch or overlap")
    if sections["cache"]["budget_bytes"] != budget or sections["cache"]["unit"] != "expert_group":
        raise ValueError("native Scheduling requires the exact physical K-slot expert-group budget")
    if any(rule["engine"] is not None for rule in sections["transport"]["rules"]):
        raise ValueError("native bridge has one fixed weight-copy implementation, not selectable engines")
    document = {**sections["policy"], **{key: sections[key] for key in ("cache", "prefetch", "transport")}}
    document["placements"] = {"expert_ffn": scheduling.placements.get("expert_ffn", "gpu")}
    document["transport"]["boundary_release"] = ("declared_dead_intermediates"
        if retention == "release_after_use" else "disabled")
    policy = CompositionPolicy(parse_composition_policy(document))
    slots, actions, records, history = {}, [], [], []
    expert_catalogue = tuple(PublicExpert(expert, f"E{expert}", "expert-stage") for expert in range(count))

    for step_index, step in enumerate(steps):
        routes = step.get("routes")
        if not isinstance(routes, list) or len(routes) != len(step.get("input", ())):
            raise ValueError("step routes must match the input row count")
        if not routes or not routes[0]:
            raise ValueError("each step requires positive row count and top-k")
        top_k = len(routes[0])
        for row in routes:
            if len(row) != top_k or len(set(row)) != top_k or any(type(e) is not int or not 0 <= e < count for e in row):
                raise ValueError("route rows require consistent distinct valid expert IDs")
        if len(step.get("scores", ())) != len(routes) or any(len(row) != top_k for row in step["scores"]):
            raise ValueError("each route requires its supplied mixture score")
        request = f"step{step_index}"
        history.extend(RevealedRoute(request, "expert-stage", row, tuple(experts), tuple(step["scores"][row]))
                       for row, experts in enumerate(routes))
        active = sorted(set(e for row in routes for e in row))
        done = set()
        for _ in range(16 * (count + len(active) + 1)):
            if len(done) == len(active) and (retention != "release_after_use" or not slots):
                break
            tensors = tuple(PublicTensor(f"E{expert}", "weight", weight_bytes,
                (("dram", "READY"), ("vram", "READY" if expert in slots else "ABSENT")),
                (TransferPrice("dram_to_vram", "dram", "vram",
                    0 if decision_cost_view == "compute_only" else price("LOAD")),),
                allocation_bytes=(("dram", weight_bytes), ("vram", weight_bytes)),
                initial_locations=("dram",)) for expert in range(count))
            operations = []
            for expert in active:
                rows = tuple((request, row) for row, experts in enumerate(routes) if expert in experts)
                operation_prices = tuple(DevicePrice(device, price(f"EXPERT:{device}:q={len(rows)}"), 0, 0)
                                         for device in ("cpu", "gpu"))
                operations.append(PublicOperation(f"{request}:E{expert}", "expert_ffn",
                    "DONE" if expert in done else "NOT_STARTED", (f"E{expert}",), (), operation_prices,
                    expert_id=expert, work_items=rows, token_count=len(rows), phase="prefill",
                    request_id=request, region_id=f"E{expert}", execution_stage="expert_ffn",
                    execution_variant="host_expert_response_v1", layer_id="expert-stage", invocation_id=request))
            used = len(slots) * weight_bytes
            # No latency is predicted here. This serial compilation exposes
            # completed actions and residency; modeling evaluates durations.
            observation = CompositionObservation(0, tuple(operations), tensors, (),
                (MemoryUsage("dram", count * weight_bytes, 0, 0, count * weight_bytes, 0, count * weight_bytes),
                 MemoryUsage("vram", budget, 0, 0, used, 0, used)), True, tuple(history), (), False, None,
                experts=expert_catalogue, allow_eviction=True, task_graph_complete=True,
                layer_order=("expert-stage",), active_invocations=((request, "expert-stage"),),
                weight_catalogue=tuple((f"E{e}", "expert-stage", "expert_ffn", e) for e in range(count)))
            if retention == "release_after_use":
                from dataclasses import replace
                observation = replace(observation, recoverable_release_candidates=tuple(
                    (f"E{expert}", "vram") for expert in sorted(done & slots.keys())))
            decision = policy(observation)
            if decision is None:
                raise ValueError(f"recipe cannot progress at step {step_index}: {sorted(done)}")
            command = compile_command(decision, observation)
            action = command.action
            if action.kind == "COPY":
                if action.src != "dram" or action.dst != "vram":
                    raise ValueError("native bridge only accepts explicit weight H2D")
                expert = int(action.target_id[1:])
                if expert in slots or len(slots) >= config["gpu_slots"]:
                    raise ValueError("compiled COPY violates the bounded physical expert slot pool")
                slot = min(set(range(config["gpu_slots"])) - set(slots.values()))
                native = {"kind": "LOAD", "expert": expert, "slot": slot}
                slots[expert] = slot
            elif action.kind == "EVICT":
                expert = int(action.target_id[1:])
                if action.location != "vram" or expert not in slots:
                    raise ValueError("native bridge accepts only resident GPU weight eviction")
                native = {"kind": "RELEASE", "expert": expert, "slot": slots.pop(expert)}
                price("RELEASE")
            elif action.kind == "COMPUTE":
                selected = next(op for op in operations if op.id == action.target_id)
                expert = selected.expert_id
                if expert in done or action.device == "gpu" and expert not in slots:
                    raise ValueError("compiled expert duplicated or missing physical GPU weights")
                native = {"kind": "EXPERT", "step": step_index, "expert": expert, "device": action.device}
                if action.device == "gpu":
                    native["slot"] = slots[expert]
                done.add(expert)
            else:
                raise ValueError(f"unsupported scoped native action: {action.kind}")
            records.append({"step": step_index, "observation_sha256": observation_hash(observation),
                            "decision": asdict(decision), "command": asdict(command), "native_action": native})
            actions.append(native)
        if len(done) != len(active):
            raise ValueError("compiler exceeded the bounded progress limit")
        actions.append({"kind": "COMBINE", "step": step_index})
        price(f"COMBINE:q={len(routes)}:topk={top_k}")

    program = {"schema": "moe-expert-policy-native-1", "config": config, "seed": seed,
               "resource_guards": deepcopy(resource_guards), "steps": steps, "actions": actions}
    evidence = {"schema": "moe-expert-policy-compilation-1", "recipe": (
        expert_recipe_scope(recipe) if recipe is not None else
        {"id": "ordinary-Scheduling", "full_paper_reproduction": False,
         "future_route_visibility": False, "prefetch_enabled": False}),
        "retention": retention,
        "decision_cost_view": decision_cost_view,
        "decision_weight_transfer_price_ns": (0 if decision_cost_view == "compute_only" else price("LOAD")),
        "physical_weight_transfer_price_ns": price("LOAD"),
        "decision_ablation_scope": "compute_only hides weight LOAD prices only while ranking devices; actual LOAD actions and physical costs remain unchanged",
        "normalized_policy": document, "records": records, "boundary": BOUNDARY,
        "compiler": "existing CompositionPolicy and compile_command; scoped serial native projection",
        "full_layer_compilation": False, "full_tensor_runtime_replay": False,
        "input_visibility": "current step routes revealed only after preceding COMBINE; no future rows in observations",
        "memory_scope": "K persistent expert slots; native macro owns activation workspace and host response",
        "time_scope": "compilation has no time prediction; all completed public snapshots use t=0; modeling computes durations",
        "combine_semantics": "required weighted sum and one residual after every selected expert completes",
        "activation_transfer_accounting": "included once inside EXPERT; no additional activation COPY action",
        "action_counts": dict(Counter(action["kind"] for action in actions))}
    return {"program": program, "compilation_evidence": evidence}
