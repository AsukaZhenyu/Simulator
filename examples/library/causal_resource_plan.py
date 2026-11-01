"""External resource f(t), internal LLM state, and causal policy visibility.

Run after installing the library. This is a declared teaching example, NOT a
hardware calibration. It requires no GPU, native measurement tool or profile.
All efficiencies are explicitly eta=1, not measured hardware efficiencies.
"""
from dataclasses import replace
import json

from llm_simulator.performance import (
    AvailableResources, Graph, InvalidModel, MemoryDomain, MemoryReservation,
    Parameter, ParameterBundle, ResourcePoint, ResourceTimeline, ServiceBinding,
    Task, WorkDemand,
)
from llm_simulator.steady_costs import (
    ResourceInput, validate_budget_contract, validate_resource_input,
)


def run_example():
    second=1_000_000_000
    GiB=1 << 30
    scope="declared teaching scenario; eta=1; not measured hardware"
    evidence=(scope,)

    # The pool is independent of this inference policy. Other applications
    # RESERVE 50 FLOP/s and 2 GiB from 1 s until 3 s; these are reservations,
    # not oversubscribed demands requiring a sharing/arbitration rule.
    gpu_pool=100
    vram_pool=8*GiB
    other_application_plan=((0,0,0),(second,50,2*GiB),(3*second,0,0))

    # Convert exactly once:
    # GPU_available(t) = GPU_pool(t) - other_GPU_reservations(t)
    # VRAM_budget(t)   = VRAM_pool(t) - other_VRAM_reservations(t)
    # AvailableResources receives these NET budgets. LLM-owned memory is
    # still charged by the Graph; never subtract the external load again.
    validate_budget_contract(semantics="net_remaining_budget",
        other_application_kind="already_net",external_reservations_charged_separately=False)
    timeline=ResourceTimeline(tuple(
        ResourcePoint(at,rates={"gpu_net":gpu_pool-other_gpu},
            capacities={"vram":vram_pool-other_memory})
        for at,other_gpu,other_memory in other_application_plan))

    # Metadata establishes when a policy can know the plan. Numerical f(t)
    # lives in timeline above; the metadata validator does not compute rates.
    announced=ResourceInput("announced-external-budget","available_service",
        "declared_external",0,10*second,0,"scenario_ns","zero",evidence,False,True)

    def validate_plan(plan,consumer):
        return validate_resource_input(plan,role="exogenous_input",consumer=consumer,
            cutoff_ns=0,clock_id="scenario_ns",clock_origin="zero",evaluation_policy="example")

    assert validate_plan(announced,"online_policy")["visible_at_cutoff"]
    late_announcement=replace(announced,available_at_ns=2*second)
    # The MODEL may replay the whole authored scenario. An ONLINE POLICY
    # deciding at t=0 cannot read a plan that is only announced at t=2 s.
    validate_plan(late_announcement,"model_scenario")
    future_announcement_rejected=False
    try:
        validate_plan(late_announcement,"online_policy")
    except InvalidModel:
        future_announcement_rejected=True
    assert future_announcement_rejected

    def parameter(name,value,unit):
        return Parameter(name,value,unit,"declared",evidence,scope)

    supply=ParameterBundle((
        parameter("cpu_net",100,"flop/s"),
        parameter("gpu_net",gpu_pool,"flop/s"),
        parameter("link_net",100,"byte/s"),
        parameter("vram_net",vram_pool,"byte"),
    ))
    available=AvailableResources(supply,timeline,10*second,"scenario_ns","zero",
        "declared_scenario",evidence,scope)

    # Own state is internal: 4 GiB resident tensors plus 1 GiB workspace for
    # copy/GPU execution. Peak 5 GiB fits the minimum net budget of 6 GiB.
    # external_bytes=0 because other applications were already deducted.
    workspace=(MemoryReservation("vram",GiB),)
    graph=Graph((
        Task("cpu","cpu",(WorkDemand("cpu_work",100,"flop"),)),
        Task("copy","link",(WorkDemand("copy_work",50,"byte"),),
            ("cpu",),memory=workspace),
        Task("gpu","gpu",(WorkDemand("gpu_work",200,"flop"),),
            ("copy",),memory=workspace),
    ),(MemoryDomain("vram","vram_net",4*GiB,0),))
    bindings=tuple(ServiceBinding(target,source,1,"declared",evidence,scope)
        for target,source in (("cpu_work","cpu_net"),("copy_work","link_net"),("gpu_work","gpu_net")))
    result=available.evaluate(graph,ParameterBundle(()),bindings)

    # CPU: [0,1] s. Copy: [1,1.5] s. GPU then makes 75 FLOPs before
    # t=3 s at 50 FLOP/s, and its remaining 125 FLOPs take 1.25 s at 100.
    assert result.duration_ns==4_250_000_000
    assert [(task.task_id,task.start_ns,task.end_ns) for task in result.tasks]==[
        ("cpu",0,second),("copy",second,1_500_000_000),("gpu",1_500_000_000,4_250_000_000)]
    return dict(scope=scope,duration_ns=result.duration_ns,
        tasks=[task.to_dict() for task in result.tasks],
        net_budget_points=[dict(at_ns=at,gpu_flop_per_s=gpu_pool-gpu_other,
            vram_bytes=vram_pool-memory_other) for at,gpu_other,memory_other in other_application_plan],
        own_resident_bytes=4*GiB,own_workspace_bytes=GiB,
        external_memory_charged_again=False,announced_plan_visible_to_policy=True,
        future_announcement_policy_rejected=future_announcement_rejected,
        full_external_plan_visible_to_model=True,OS_scheduler_simulated=False,
        hardware_or_profile_required=False,hardware_accuracy_claimed=False)


if __name__=="__main__":
    print(json.dumps(run_example(),indent=2))
