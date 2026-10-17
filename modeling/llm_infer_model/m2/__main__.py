"""Replay a declared M2 invocation and save its scope, bindings and state.

This performs simulation only. It neither measures hardware nor searches for
an optimal mapping; the fixed-device or hybrid serial baseline is deterministic.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

from ..tensor.spec import InvalidInput, Unsupported
from .expand import load_and_compile
from .runner import run_invocation
from .spec import to_state_document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--invocation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise InvalidInput(f"output directory must be empty: {output}")
    invocation_path = args.invocation.resolve()
    compiled = load_and_compile(invocation_path)
    result = run_invocation(compiled)
    evidence = {"invocation": str(invocation_path),
                "invocation_sha256": hashlib.sha256(invocation_path.read_bytes()).hexdigest()}
    for key, relative in (("layer", compiled.invocation.layer_file), ("costs", compiled.invocation.costs_file)):
        if relative is not None:
            path = (invocation_path.parent / relative).resolve()
            evidence.update({key: str(path), key + "_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    summary = {"status": result.status, "reason": result.evaluation.reason,
               "makespan_ns": result.makespan_ns,
               "peak_dram_bytes": result.evaluation.peak_dram_bytes,
               "peak_vram_bytes": result.evaluation.peak_vram_bytes,
               "cost_provenance": result.cost_provenance,
               "cost_table_source": compiled.cost_table.source,
               "validation_only_costs": compiled.cost_table.validation_only,
               "hardware_measurement_performed": False, "search_performed": False,
               "physical_boundary": compiled.physical_boundary,
               "source_evidence": evidence}
    documents = {"workload.json": compiled.workload_document,
                 "scenario.json": compiled.scenario_document,
                 "replay.json": asdict(result.evaluation), "summary.json": summary}
    if result.state_out is not None:
        # The generic byte-explicit state schema has not changed from 0.1.
        documents["state_out.json"] = to_state_document(result.state_out)
    output.mkdir(parents=True, exist_ok=True)
    for name, document in documents.items():
        with (output / name).open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(document, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    print(f"M2 replay {result.status}: makespan_ns={result.makespan_ns}, peak_dram_bytes={result.evaluation.peak_dram_bytes}; simulation_only=true")
    return 0 if result.state_out is not None else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, InvalidInput, Unsupported) as error:
        print(f"m2 replay: {error}", file=sys.stderr)
        raise SystemExit(1)
