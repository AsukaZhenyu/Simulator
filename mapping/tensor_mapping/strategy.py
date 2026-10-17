"""Compile and replay a declared strategy, without invoking any search.

    python -m tensor_mapping.strategy strategy.json --out new-directory
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

from llm_infer_model.tensor.engine import STATUS_VALID
from llm_infer_model.tensor.spec import SpecError, scenario_fingerprint

from .artifacts import action_to_dict, events_document, states_document, write_json
from .strategy_compiler import compile_strategy_file, compiled_document, run_strategy
from .strategy_ir import strategy_to_document


def write_run(output: Path, compiled, result) -> None:
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"output must be a new or empty directory: {output}")
    for source in compiled.sources:
        if hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError(f"input changed during compilation/replay: {source['path']}")
    output.mkdir(parents=True, exist_ok=True)
    evaluation = result.evaluation
    identity = {"strategy_id": compiled.ir.id,
                "base_scenario_fingerprint": scenario_fingerprint(compiled.scenario)}
    write_json(output / "strategy.json", strategy_to_document(compiled.ir))
    write_json(output / "compiled.json", compiled_document(compiled))
    profiles = {scenario_fingerprint(s): s for s in (compiled.scenario, *compiled.device_states.values())}
    environments = {
        "format_version": "strategy-environments-0.1", **identity,
        "profiles": {key: {"scenario_id": profile.id,
                            "cost_source": profile.costs.source,
                            "architecture": asdict(profile.architecture)}
                     for key, profile in profiles.items()},
        "events": [asdict(event) for event in result.environment_events],
    }
    write_json(output / "environments.json", environments)
    write_json(output / "execution.json", {
        "format_version": "strategy-execution-0.1", **identity,
        "information_mode": "offline_fixed", "online_policy_claimed": False,
        "status": evaluation.status,
        "error_ir_instruction_index": result.error_source_index,
        "actions": [{"index": i, "action": action_to_dict(action, m1=True),
                     "ir_instruction_index": result.action_sources[i],
                     "scenario_fingerprint": result.action_profiles[i]}
                    for i, action in enumerate(evaluation.actions)],
        "note": "Actions spanning device states must be replayed with their declared profile events, not one static scenario.",
    })
    events = events_document(mode="strategy", scenario=compiled.scenario, evaluation=evaluation,
        note="The scenario fingerprint identifies the base; each action's active profile is in execution.json.")
    events["action_profiles"] = list(result.action_profiles)
    write_json(output / "events.json", events)
    states = states_document(mode="strategy", scenario=compiled.scenario, evaluation=evaluation,
        note="Action snapshots share one modeling ledger; boundary device updates are recorded in environments.json.")
    states["action_profiles"] = list(result.action_profiles)
    write_json(output / "states.json", states)
    write_json(output / "summary.json", {
        "format_version": "strategy-result-0.1", **identity,
        "status": evaluation.status, "reason": evaluation.reason,
        "makespan_ns": evaluation.makespan_ns,
        "peak_dram_bytes": evaluation.peak_dram_bytes, "peak_vram_bytes": evaluation.peak_vram_bytes,
        "h2d_bytes": evaluation.h2d_bytes, "d2h_bytes": evaluation.d2h_bytes,
        "ssd_read_bytes": evaluation.ssd_read_bytes, "ssd_write_bytes": evaluation.ssd_write_bytes,
        "action_count": evaluation.action_count,
        "error_code": evaluation.error_code,
        "error_message": evaluation.reason if evaluation.status != STATUS_VALID else None,
        "error_action_index": evaluation.error_action_index,
        "error_t_ns": evaluation.error_t_ns,
        "error_ir_instruction_index": result.error_source_index,
        "device_state_change_count": len(result.environment_events),
        "information_mode": "offline_fixed", "online_policy_claimed": False,
        "search_performed": False, "physical_calibration_claimed": False,
        "mid_task_device_changes_supported": False,
        "sources": list(compiled.sources),
    })
    if compiled.m2_binding is not None and evaluation.status == STATUS_VALID:
        from llm_infer_model.m2.runner import state_out_from_result
        from llm_infer_model.m2.spec import to_state_document
        state = state_out_from_result(compiled.m2_binding, evaluation)
        write_json(output / "state_out.json", to_state_document(state))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("strategy", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
            raise ValueError(f"output must be a new or empty directory: {args.out}")
        compiled = compile_strategy_file(args.strategy)
        result = run_strategy(compiled)
        write_run(args.out, compiled, result)
    except (SpecError, ValueError, OSError, json.JSONDecodeError) as error:
        print(f"strategy: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result.evaluation.status,
                      "makespan_ns": result.evaluation.makespan_ns,
                      "output": str(args.out.resolve())}, ensure_ascii=False))
    return 0 if result.evaluation.status == STATUS_VALID else 3


if __name__ == "__main__":
    raise SystemExit(main())
