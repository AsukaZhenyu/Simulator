"""Two-layer Prefill(3) -> Decode(1) combining all new execution mechanisms.

Each invocation has AttentionStages + FFN partitions. PhasePlacement moves
projection/Attention from GPU Prefill to CPU Decode; FFN shards keep their
explicit CPU/GPU bindings. Startup/spare preparation uses the same persistent
KV/weight inventory. All stage prices are independent synthetic declarations.
"""
from pathlib import Path
import sys

from llm_infer_model.tensor.spec import SpecError
from tensor_mapping.phase_scheduling import DataPreparation, PhasePlacement
from tensor_mapping.python_frontend import Cache, ModelPlan, Scheduling, Transport

from _common import parser, run_plan


HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures/execution-phase-transforms"


def build_plan(mode="data_dependencies"):
    plan = ModelPlan(id="python-combined-multiround-dense-synthetic", model_kind="global_m2",
        model_file=FIXTURES / "dense.global.json", environment_file=FIXTURES / "environment.json",
        visibility_mode=mode, max_decisions=5000,
        scheduling=Scheduling(placements={"projection": "gpu", "attention": "gpu", "dense_ffn": "gpu"},
            release="last_use", cache=Cache(rule="lru", budget_bytes=65536), transport=Transport()))
    for phase in ("prefill", "decode"):
        for layer in ("denseL0", "denseL1"):
            invocation = f"{layer}-{phase}"
            plan.transform_file(invocation, FIXTURES / f"attention.{phase}.json")
            plan.transform_file(invocation, FIXTURES / f"ffn.{phase}.json")
    attention = ("projection", "attention_qk", "attention_softmax", "attention_pv", "attention_output")
    PhasePlacement(prefill={kind: "gpu" for kind in attention},
                   decode={kind: "cpu" for kind in attention}).apply(plan)
    # The static model owns these exact materialized leaf identities.
    # These are not predictions from future routes or free runtime slices.
    for layer, trigger in (("denseL0", "startup"), ("denseL1", "spare")):
        plan.prepare(DataPreparation(
            tensor_ids=tuple(f"w:{layer}:W_{stage}:ffn:accelerator" for stage in ("gate", "up", "down")),
            destination="vram", trigger=trigger))
    return plan


def main(argv=None):
    args = parser(__doc__).parse_args(argv)
    try:
        return run_plan(build_plan(args.mode), output=args.out,
                        example_source=__file__, example_name="execution_combined_demo")
    except (SpecError, OSError, ValueError) as error:
        print(f"execution_combined_demo: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
