"""Composable Dense/MoE execution transforms and charged weight preparation.

Python expresses the choices. Separate exact synthetic cost fixtures provide
the stage prices; this script does not invent, interpolate or divide timings.
"""
from dataclasses import replace
import argparse
import json
from pathlib import Path
import sys

from llm_infer_model.m2.ffn_execution import FFNShard, parse_ffn_execution
from llm_infer_model.tensor.spec import SpecError
from tensor_mapping.phase_scheduling import DataPreparation
from tensor_mapping.python_frontend import Cache, ModelPlan, Scheduling, Transport


HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures/execution-transforms"


def _ffn(cost_file, *, cpu_end, intermediate):
    """Changing these cuts also requires exact fixture costs for the new shapes."""
    spec = parse_ffn_execution(json.loads(cost_file.read_text(encoding="utf-8")))
    return replace(spec, intermediate_size=intermediate,
        shards=(FFNShard("host", 0, cpu_end, "cpu"),
                FFNShard("accelerator", cpu_end, intermediate, "gpu")),
        # Preserve the cost file's merge_input_count; never silently rewrite
        # a one-/three-input merge contract into this two-shard plan.
        preprocess_device="gpu", merge_device="gpu")


def build_dense_plan(mode="data_dependencies"):
    plan = ModelPlan(id="python-dense-attention-ffn-preparation-synthetic",
        model_kind="global_m2", model_file=FIXTURES / "dense.global.json",
        environment_file=FIXTURES / "environment.json", visibility_mode=mode,
        max_decisions=5000, scheduling=Scheduling(
            placements={"projection": "gpu", "attention": "gpu", "dense_ffn": "gpu"},
            release="last_use", cache=Cache(rule="lru", unit="tensor", budget_bytes=32768),
            transport=Transport()))

    # First expose the QK / softmax / PV / output dependencies. Then refine the
    # independent FFN module. Both become real tasks in the same runtime.
    plan.transform_file("dense-decode", FIXTURES / "dense.attention-stages.json")
    plan.transform("dense-decode", _ffn(FIXTURES / "dense.ffn-costs.json", cpu_end=8, intermediate=32))
    with plan.module("attention"):
        plan.place("attention_qk", "cpu")
        plan.place("attention_softmax", "cpu")
        plan.place("attention_pv", "gpu")
        plan.place("attention_output", "gpu")

    # CPU owns [0,8), GPU owns [8,32) of the FFN intermediate axis. Gate/up
    # output columns and down input rows use the same cuts. The GPU receives
    # the partial sum, after one Dense norm and before one residual addition.
    gpu_ffn_weights = tuple(f"w:hybrid-interface-synthetic:W_{stage}:ffn:accelerator"
                            for stage in ("gate", "up", "down"))
    plan.prepare(DataPreparation(tensor_ids=gpu_ffn_weights, destination="vram", trigger="startup"))
    return plan


def build_moe_plan(mode="data_dependencies"):
    plan = ModelPlan(id="python-moe-selected-rows-ffn-preparation-synthetic",
        model_kind="global_m2", model_file=FIXTURES / "moe.global.json",
        environment_file=FIXTURES / "environment.json", visibility_mode=mode,
        max_decisions=5000, scheduling=Scheduling(
            placements={"projection": "gpu", "attention": "gpu", "router": "gpu",
                        "expert_ffn": "gpu", "shared_expert_ffn": "gpu", "combine": "gpu"},
            release="last_use", cache=Cache(rule="lru", unit="expert_group", budget_bytes=4096),
            transport=Transport()))

    # Routing/gather/scatter/combine stay model-owned. Selected expert rows use
    # the same FFN transform as Dense, with H=4, I=8 and CPU/GPU cuts [0,2)/[2,8).
    plan.transform_file("moe-prefill", FIXTURES / "moe.selected-rows.json")
    # This separate synthetic workload explicitly supplies a 192B F32 Q/K/V/O
    # pack. The smaller old opaque projection is not reinterpreted as QKV.
    plan.transform_file("moe-prefill", FIXTURES / "moe.attention-stages.json")
    plan.transform("moe-prefill", _ffn(FIXTURES / "moe.ffn-costs.json", cpu_end=2, intermediate=8))
    with plan.module("attention"):
        plan.place("attention_projection", "gpu")
        plan.place("attention_qk", "cpu")
        plan.place("attention_softmax", "cpu")
        plan.place("attention_pv", "gpu")
        plan.place("attention_output", "gpu")

    # Expert 0 is a fixed architectural choice, not a prediction from private
    # route traces. Even an unused expert can be prepared, with charged cost.
    gpu_expert_weights = tuple(f"w:moe-small:E0:v1:{stage}:ffn:accelerator"
                               for stage in ("gate", "up", "down"))
    plan.prepare(DataPreparation(tensor_ids=gpu_expert_weights, destination="vram", trigger="startup"))
    # This second choice demonstrates optional preparation without evicting
    # active inputs. Expert 3 comes from the static architecture catalogue.
    plan.prepare(DataPreparation(expert_id=3, destination="vram", trigger="spare", max_count=2))
    return plan


def source_files(case):
    """Bind the files read by _ffn in addition to transform_file's own snapshots."""
    return (Path(__file__).resolve(), FIXTURES / f"{case}.ffn-costs.json")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("dense", "moe", "both"), default="both")
    parser.add_argument("--mode", choices=("data_dependencies", "scope_barriers"), default="data_dependencies")
    parser.add_argument("--out", type=Path, required=True, help="New or empty parent output directory")
    args=parser.parse_args(argv)
    try:
        output=args.out.resolve()
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError("--out must name a new or empty directory")
        names=("dense", "moe") if args.case=="both" else (args.case,)
        summaries=[]
        for name in names:
            plan=(build_dense_plan if name=="dense" else build_moe_plan)(args.mode)
            compiled=plan.compile(source_files=source_files(name))
            result=compiled.run()
            compiled.write_result(output/name,result)
            run=result[0]
            summaries.append(dict(case=name,status=run.evaluation.status,
                termination_reason=run.termination_reason,compilation_error=run.compilation_error,
                makespan_ns=run.evaluation.makespan_ns,
                cost_source="synthetic_fixture_not_hardware_measurements", output=str(output/name)))
        print(json.dumps(summaries,ensure_ascii=False))
        return 0 if all(row["status"]=="valid" and row["termination_reason"]=="goal_reached" for row in summaries) else 3
    except (SpecError,OSError,ValueError) as error:
        print(f"execution_transforms_demo: {error}",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
