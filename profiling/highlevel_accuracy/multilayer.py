"""Prepare genuine connected Dense layers through the public ModelPlan compiler.

Compile prices only specify fixed execution choices. They are never measured
predictions. Each layer has its own immutable weights and KV identities; the
global compiler connects the preceding layer's actual output to the next input.
"""
from __future__ import annotations

import json
from pathlib import Path

from tensor_mapping.python_frontend import ModelPlan, Scheduling
from .bridge import SerialExecutionPolicy, _require
from .cases import build_case


def _write(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def build_multilayer_case(directory, *, layers=2, **kwargs):
    _require(type(layers) is int and 1 <= layers <= 32, "explicit one to thirty-two layer native campaign required")
    _require(not kwargs.get("attention_stages") and kwargs.get("ffn_cut") is None,
             "this campaign builder admits complete macro placement recipes only")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    _require(not any(directory.iterdir()), "multilayer case directory must be empty")
    entries = []
    for index in range(layers):
        folder = directory / f"layer{index}"
        build_case(folder, **kwargs)
        layer_file, invocation_file = folder / "layer.json", folder / "invocation.json"
        layer = json.loads(layer_file.read_text(encoding="utf-8"))
        invocation = json.loads(invocation_file.read_text(encoding="utf-8"))
        old_id, new_id = layer["id"], f"highlevel-dense-L{index}"
        layer["id"] = new_id
        invocation["layer"] = new_id
        for state in invocation["state_in"]["requests"]:
            for block in state["blocks"]:
                block["id"] = block["id"].replace(f"kv:{old_id}:", f"kv:{new_id}:", 1)
        _write(layer_file, layer)
        _write(invocation_file, invocation)
        entries.append(dict(id=f"dense-layer{index}", invocation_file=f"layer{index}/invocation.json", execution_file=None))
    document = dict(format_version="global-execution-0.1", id="native-connected-dense-layers",
        rounds=[dict(id="one-round", input_mode="initial_supplied_hidden", gate_cost=None, invocations=entries)])
    _write(directory / "global.json", document)
    _write(directory / "environment.json", dict(format_version="environment-trace-0.1", events=[]))
    devices = kwargs.get("devices", ("cpu", "gpu", "cpu"))
    plan = ModelPlan(id="native-connected-dense-plan", model_kind="global_m2", model_file=directory / "global.json",
        environment_file=directory / "environment.json", max_decisions=200000,
        scheduling=Scheduling(placements=dict(zip(("projection", "attention", "dense_ffn"), devices)), release="last_use"))
    plan.runtime_policy(SerialExecutionPolicy)
    return plan
