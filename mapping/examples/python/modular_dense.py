"""Two Dense layers: GPU Attention, a CPU FFN and a GPU FFN.

The model owns all mathematical operators, dimensions, states and synthetic
prices. Python only supplies module scopes and scheduling choices. This is
whole-module placement, not tensor parallelism or a measured speedup.
"""
import sys

from llm_infer_model.tensor.spec import SpecError
from tensor_mapping.python_frontend import (
    Cache, ModelPlan, PlanModule, Prefetch, Scheduling, Select, Transport,
)

from _common import ENVIRONMENT, FIXTURES, SERVICES, parser, run_plan


class DecoderPlacement(PlanModule):
    def __init__(self, ffn_device):
        self.ffn_device = ffn_device

    def build(self, plan):
        with plan.module("attention"):
            plan.place("projection", "gpu")
            plan.place("attention", "gpu")
        with plan.module("ffn"):
            plan.place("dense_ffn", self.ffn_device)


def build_plan(mode="data_dependencies"):
    plan = ModelPlan(
        id="python-modular-dense-synthetic",
        model_kind="global_m2",
        model_file=FIXTURES / "dense.global.json",
        environment_file=ENVIRONMENT,
        services_file=SERVICES,
        visibility_mode=mode,
        max_decisions=3000,
        scheduling=Scheduling(
            # Every task kind has an explicit fallback, even when scoped below.
            placements={"projection": "gpu", "attention": "gpu", "dense_ffn": "gpu"},
            expert_rule="fixed",
            release="last_use",
            cache=Cache(rule="lru", unit="tensor", budget_bytes=32768),
            prefetch=Prefetch(next_layers=1, max_count=2,
                              ranking="history_frequency", include_dense_weights=True),
            transport=Transport(engine="ce", slot_reuse="release_unlocked_backed_copy",
                                boundary_release="declared_dead_intermediates"),
        ),
    )
    # These scopes cover both Prefill and Decode of their selected model layer.
    for layer_id, device in (("denseL0", "cpu"), ("denseL1", "gpu")):
        DecoderPlacement(device)(plan, name=layer_id, select=Select(layer_id=layer_id))
    return plan


def main(argv=None):
    args = parser(__doc__).parse_args(argv)
    try:
        return run_plan(build_plan(args.mode), output=args.out,
                        example_source=__file__, example_name="modular_dense")
    except (SpecError, OSError, ValueError) as error:
        print(f"modular_dense: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
