"""Readable modular strategies and IR inspection, with no saved profiles.

All numbers are declared analytical examples. This checks expression and
compilation, not hardware speedups, numerical LLM execution or timing accuracy.
Run: python examples/library/strategy_ir.py --out build/strategy-ir-example
The output directory must be new or empty.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from llm_simulator import Cache, ModelPlan, Scheduling, Select, Simulator, Transport
from llm_simulator.demo import make_demo


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def configure(plan: ModelPlan):
    """Construction-time Python: record choices; the compiler builds actions."""
    for layer_id, engine in (("layer0", "ce"), ("layer1", "sm")):
        with plan.module(layer_id, layer_id=layer_id):
            plan.place("dense_ffn", "gpu")
            # Physical weights belong to a layer, even when reused next round.
            plan.transfer(engine=engine)
            plan.prepare(destination="dram", trigger="spare")
    with plan.module("decode_ffn", phase="decode"):
        plan.place("dense_ffn", "cpu", select=Select(layer_id="layer0"), priority=1)
    return plan


def run_example(output):
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("--out must name a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    # Explicit payload-only service law, matching the example's declared link.
    # CE and SM share this one link; these are not two full-bandwidth links.
    services = dict(format_version="tensor-services-0.1", pcie=dict(
        bandwidth_bytes_per_second=1_000_000_000,
        ce_bytes_per_second=1_000_000_000, sm_bytes_per_second=500_000_000,
        arbitration="max_min_fair", timing_scope="payload_only_no_launch_latency"), slots=[])
    service_file = output / "services.json"
    write(service_file, services)
    original, _ = make_demo("dense")
    simulator = Simulator(model=original.model,
        hardware=replace(original.hardware, transfer_bandwidth_bytes_per_second=1e9,
                         transfer_latency_ns=0),
        workload=original.workload, cost_model=original.cost_model)
    scheduling = Scheduling(default_device="gpu", release="last_use",
                            cache=Cache(rule="lru", budget_bytes=1024 * 1024),
                            transport=Transport(engine="ce"))

    def build(plan):
        configure(plan)
        write(output / "declaration.json", plan.declaration_document())
        write(output / "inspection.json", plan.inspect().document())
        (output / "explanation.txt").write_text(plan.explain() + "\n", encoding="utf-8", newline="\n")

    with simulator.compile(scheduling=scheduling, configure=build,
                           directory=output / "inputs", services_file=service_file) as compiled:
        result = compiled.run()
        result.save(output / "execution")
        write(output / "summary.json", result.summary())
        print((output / "explanation.txt").read_text(encoding="utf-8"))
        print(json.dumps(result.summary(), ensure_ascii=False))
        if not result.success:
            raise RuntimeError(f"strategy did not finish: {result.summary()}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run_example(args.out)


if __name__ == "__main__":
    main()
