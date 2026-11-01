"""Record the actual native admission boundary for the conditional E2B study.

This is a capability audit, not a replacement timing experiment. It never
executes a target on CPU/GPU or assigns synthetic prices to measured work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "src", ROOT / "mapping", ROOT / "modeling"):
    sys.path.insert(0, str(folder))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(out):
    from llm_simulator import AnalyticalCosts, Device, Hardware, Model, Scheduling, Simulator, Workload
    from profiling.highlevel_accuracy.bridge import BridgeError, export_plan

    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    bridge = ROOT / "profiling/highlevel_accuracy/bridge.py"
    native = ROOT / "profiling/native/highlevel_accuracy/highlevel_probe.cpp"
    bridge_text, native_text = bridge.read_text(encoding="utf-8"), native.read_text(encoding="utf-8")
    anchors = [
        ("multi_round_token_gate", bridge, bridge_text, "multi-round trace token gates have no numeric native implementation"),
        ("real_moe", bridge, bridge_text, "a complete Dense physical boundary is required"),
        ("overlap", bridge, bridge_text, "overlapping launch is unsupported; author a serial RuntimePolicy explicitly"),
        ("physical_arena", native, native_text, "logical_liveness_on_preallocated_independent_backing"),
        ("complete_cuda_workspace", native, native_text, "CUDA_private_backend_scratch_unknown; not_complete_native_memory"),
    ]
    requirements = []
    for key, path, content, anchor in anchors:
        if anchor not in content:
            raise RuntimeError("Audit anchor changed; inspect rather than reuse old capability result: " + key)
        line = content[:content.index(anchor)].count("\n") + 1
        requirements.append(dict(requirement=key, supported=False,
                                 source=str(path.relative_to(ROOT)), line=line,
                                 exact_anchor=anchor, sha256=sha(path)))

    # Identical geometry and invocation window to E2B. These explicit fictitious
    # rates only make a legal compiler input; no timing prediction is reported.
    hardware = Hardware(cpu=Device(1e9, 1e9), gpu=Device(1e9, 1e9),
                        dram_capacity_bytes=1 << 30, vram_capacity_bytes=1 << 30,
                        transfer_bandwidth_bytes_per_second=1e9)
    simulation = Simulator(model=Model.dense(layers=2, hidden_size=512, intermediate_size=1024,
                                             heads=8, kv_heads=2),
                           hardware=hardware, workload=Workload(prefill_tokens=16, decode_steps=8),
                           cost_model=AnalyticalCosts(workspace_bytes=0))
    with simulation.compile(scheduling=Scheduling(default_device="gpu", release="last_use"),
                            directory=out / "compiler-input") as compiled:
        write_new(out / "compiled-ir.json", compiled.ir_document())
        scopes = len(compiled.plan._compiled.model.invocation_scopes)
        try:
            export_plan(compiled.plan, seed=20261025)
        except BridgeError as error:
            rejection = str(error)
        else:
            raise RuntimeError("Bridge unexpectedly admitted E2B; audit the new capability and execute its numerical checks")
    expected = anchors[0][3]
    if rejection != expected:
        raise RuntimeError("Unexpected admission failure: " + rejection)
    summary = dict(schema="dac-e2b-admission-1", status="not_admitted",
                   target=dict(layers=2, hidden=512, intermediate=1024, prefill=16, decode_steps=8),
                   actual_invocation_scopes=scopes, actual_export_attempt=True,
                   actual_export_rejection=rejection, requirements=requirements,
                   target_native_measurements=0, substitute_measurements=False,
                   synthetic_costs_used_only_for_compiler_input=True,
                   timing_accuracy_claimed=False, real_cache_prefetch_benefit_claimed=False,
                   homogeneous_single_round_layers_supported=True,
                   conclusion="The native bridge now admits homogeneous Dense layers in one round, but this multi-round trace-token target remains unsupported. Preallocated backing also cannot validate real cache/prefetch/eviction performance. E2A and semantic/analytical evidence retain their own scope.")
    write_new(out / "summary.json", summary)
    write_new(out / "artifact-hashes.json", {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    result = audit(parser.parse_args().out)
    print(json.dumps({key: result[key] for key in ("status", "actual_invocation_scopes", "actual_export_rejection", "target_native_measurements")}, indent=2))
