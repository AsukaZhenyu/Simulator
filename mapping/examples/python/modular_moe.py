"""Two MoE layers, opaque or staged experts, with causal Python history.

Routes, operators and exact synthetic prices belong to modeling. The callback
only counts already revealed routes and returns DEFER to the explicit base
scheduler. It never opens route files, predicts future routes or changes devices.
"""
from collections import Counter
import sys

from llm_infer_model.tensor.spec import SpecError
from tensor_mapping.python_frontend import (
    Cache, DEFER, ModelPlan, PlanModule, Prefetch, RuntimePolicy,
    Scheduling, Select, Transport,
)

from _common import ENVIRONMENT, FIXTURES, SERVICES, parser, run_plan


class MoELayerPlacement(PlanModule):
    def __init__(self, *, expert_device, staged):
        self.expert_device = expert_device
        self.staged = staged

    def build(self, plan):
        with plan.module("attention_and_router"):
            for kind in ("projection", "attention", "router"):
                plan.place(kind, "gpu")
        with plan.module("experts"):
            kinds = (("expert_gate", "expert_up", "expert_swiglu", "expert_down")
                     if self.staged else ("expert_ffn",))
            for kind in kinds:
                plan.place(kind, self.expert_device)
        with plan.module("combine"):
            kinds = ("gather_rows", "scatter_rows", "combine_rows") if self.staged else ("combine",)
            for kind in kinds:
                plan.place(kind, "gpu")


class RevealedExpertHistory(RuntimePolicy):
    """A minimal online policy extension, not an expert-placement algorithm."""
    def __init__(self):
        self.seen = set()
        self.counts = Counter()
        self.observation_count = 0

    def decide(self, observation):
        self.observation_count += 1
        for route in observation.route_history:
            identity = (route.layer_id, route.request_id, route.absolute_position)
            if identity in self.seen:
                continue
            self.seen.add(identity)
            for expert_id in route.expert_ids:
                self.counts[(route.layer_id, str(expert_id))] += 1
        # DEFER continues the declared scheduler. None would stop the run.
        return DEFER

    def state_document(self):
        return dict(observation_count=self.observation_count,
                    revealed_route_count=len(self.seen),
                    expert_counts=[dict(layer_id=layer, expert_id=expert, activations=count)
                                   for (layer, expert), count in sorted(self.counts.items())],
                    future_route_access=False,
                    placement_decisions_made_by_this_callback=False)


def build_plan(mode="data_dependencies", variant="opaque"):
    if variant not in {"opaque", "swiglu_staged"}:
        raise ValueError("variant must be opaque or swiglu_staged")
    staged = variant == "swiglu_staged"
    layer_prefix = "fragmentL" if staged else "moeL"
    kinds = ["projection", "attention", "router"]
    kinds += (["gather_rows", "expert_gate", "expert_up", "expert_swiglu",
               "expert_down", "scatter_rows", "combine_rows"] if staged
              else ["expert_ffn", "combine"])
    plan = ModelPlan(
        id=f"python-modular-moe-{variant}-synthetic",
        model_kind="global_m2",
        model_file=FIXTURES / ("moe-fragments.global.json" if staged else "moe.global.json"),
        # The staged global model already binds each scope's exact execution
        # file. No legacy strategy JSON or proportional macro cost is used.
        environment_file=ENVIRONMENT,
        services_file=SERVICES,
        visibility_mode=mode,
        max_decisions=3000,
        scheduling=Scheduling(
            placements={kind: "gpu" for kind in kinds},
            expert_rule="fixed",
            release="last_use",
            cache=Cache(rule="lru", unit="expert_group", budget_bytes=2048),
            prefetch=Prefetch(next_layers=1, max_count=2,
                              ranking="history_frequency", include_dense_weights=False),
            transport=Transport(engine="ce", slot_reuse="release_unlocked_backed_copy",
                                boundary_release="declared_dead_intermediates"),
        ),
    )
    for index, device in ((0, "cpu"), (1, "gpu")):
        layer_id = f"{layer_prefix}{index}"
        MoELayerPlacement(expert_device=device, staged=staged)(
            plan, name=layer_id, select=Select(layer_id=layer_id))
    plan.runtime_policy(RevealedExpertHistory, parameters={},
                        allow_device_override=False, allow_transport_override=False,
                        allow_control_override=False)
    return plan


def main(argv=None):
    cli = parser(__doc__)
    cli.add_argument("--variant", choices=("opaque", "swiglu_staged"), default="opaque")
    args = cli.parse_args(argv)
    try:
        return run_plan(build_plan(args.mode, args.variant), output=args.out,
                        example_source=__file__, example_name=f"modular_moe_{args.variant}")
    except (SpecError, OSError, ValueError) as error:
        print(f"modular_moe: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
