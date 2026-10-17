"""P/D placement plus charged preparation on the same CPU/GPU resources.

Prefill runs on GPU and Decode on CPU. The model retains eight KV block IDs
in one runtime; GPU-produced KV is copied to DRAM with real service/time and
capacity charges. This is phase placement, not independent P/D servers or
queues. All example prices are synthetic and establish no hardware speedup.
"""
import sys

from llm_infer_model.tensor.spec import SpecError
from tensor_mapping.phase_scheduling import DataPreparation, PhasePlacement
from tensor_mapping.python_frontend import Cache, ModelPlan, Scheduling, Transport

from _common import ENVIRONMENT, FIXTURES, SERVICES, parser, run_plan


def build_plan(mode="data_dependencies"):
    kinds = ("projection", "attention", "dense_ffn")
    plan = ModelPlan(id="python-pd-phase-and-preparation-synthetic", model_kind="global_m2",
        model_file=FIXTURES / "dense.global.json", environment_file=ENVIRONMENT,
        services_file=SERVICES, visibility_mode=mode, max_decisions=3000,
        scheduling=Scheduling(placements={kind: "gpu" for kind in kinds},
            cache=Cache(rule="lru", budget_bytes=32768), release="last_use",
            transport=Transport(engine="ce")))
    PhasePlacement(prefill={kind: "gpu" for kind in kinds},
                   decode={kind: "cpu" for kind in kinds}).apply(plan)
    # Both are static architectural choices. No future expert activation or
    # future device event is supplied to the policy.
    plan.prepare(DataPreparation(layer_id="denseL0", destination="vram", trigger="startup"))
    plan.prepare(DataPreparation(layer_id="denseL1", destination="vram", trigger="spare", max_count=2))
    return plan


def main(argv=None):
    args = parser(__doc__).parse_args(argv)
    try:
        return run_plan(build_plan(args.mode), output=args.out,
                        example_source=__file__, example_name="execution_phase_demo")
    except (SpecError, OSError, ValueError) as error:
        print(f"execution_phase_demo: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
