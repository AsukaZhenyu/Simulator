"""Shared command-line plumbing; strategy choices stay in the two examples."""
import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SIMULATOR = HERE.parents[2]
FIXTURES = SIMULATOR / "modeling/tests/fixtures/global_execution"
SERVICES = HERE.parent / "unified/global.services.json"
ENVIRONMENT = HERE.parent / "unified/global.environment.json"
MODES = ("data_dependencies", "scope_barriers")


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--out", type=Path, required=True,
                        help="New or empty result directory. Existing files are never replaced.")
    result.add_argument("--mode", choices=MODES, default="data_dependencies",
                        help="Only changes visibility gates, not the model's mathematical dependencies.")
    return result


def run_plan(plan, *, output, example_source, example_name):
    output = output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("--out must name a new or empty directory")
    compiled = plan.compile(source_files=(Path(example_source).resolve(), Path(__file__).resolve()))
    result = compiled.run()
    compiled.write_result(output, result)
    run, _, policy, _ = result
    audit = policy.audit_document()
    print(json.dumps(dict(example=example_name, status=run.evaluation.status,
                          termination_reason=run.termination_reason,
                          compilation_error=run.compilation_error,
                          makespan_ns=run.evaluation.makespan_ns,
                          module_count=len(compiled.ir_document()["modules"]),
                          cost_source="synthetic_fixture_not_hardware_measurements",
                          physical_calibration_claimed=False,
                          python_state=audit.get("python_state"),
                          output=str(output)), ensure_ascii=False))
    return 0 if run.evaluation.status == "valid" and run.termination_reason == "goal_reached" else 3
