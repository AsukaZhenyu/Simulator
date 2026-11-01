"""Single CPU/GPU arrivals, request pipelines, expert choices and P/D handoff.

No saved profiles, model downloads or host hardware are used. All prices are
explicit uncalibrated analytical inputs; requests use independent B1 kernels.
"""
import argparse
import json
from pathlib import Path

from llm_simulator import (
    AnalyticalCosts, Cache, Device, Hardware, Model, PhaseHandoff, PhasePlacement,
    Prefetch, Request, RequestFlow, RequestPipeline, Route, Scheduling, Simulator,
)


def make_simulator(kind="dense"):
    model=(Model.dense(layers=2,hidden_size=8,intermediate_size=16,heads=2,kv_heads=1)
           if kind=="dense" else
           Model.moe(experts=3,top_k=1,layers=2,hidden_size=8,intermediate_size=16,heads=2,kv_heads=1))
    requests=(Request("interactive",prefill_tokens=2,decode_steps=2,arrival_ns=0),
              Request("background",prefill_tokens=3,decode_steps=1,arrival_ns=10000))
    routes=tuple(Route(r.request_id,f"layer{layer}",position,(position%3,))
                 for r in requests for layer in range(model.layers)
                 for position in range(r.prefill_tokens+r.decode_steps)) if kind=="moe" else ()
    workload=RequestFlow(requests= requests,routes=routes,
                         route_source="explicit illustrative token/expert trace" if routes else None)
    hardware=Hardware(cpu=Device(1e9,1e9),gpu=Device(4e9,4e9),
        dram_capacity_bytes=1024*1024,vram_capacity_bytes=1024*1024,
        transfer_bandwidth_bytes_per_second=1e9,
        ssd_read_bandwidth_bytes_per_second=1e9,ssd_write_bandwidth_bytes_per_second=1e9)
    return Simulator(model=model,hardware=hardware,workload=workload,cost_model=AnalyticalCosts())


def configure(plan,kind):
    # Same CPU/GPU and memory ledger for every request and both phases.
    kinds=("projection","attention","dense_ffn") if kind=="dense" else ("projection","attention","router","combine")
    PhasePlacement(prefill={k:"gpu" for k in kinds},decode={k:"cpu" for k in kinds}).apply(plan)
    plan.pipeline(RequestPipeline(ready_rule="decode_first"))
    # First Decode joins the final Prefill hidden output plus KV from all layers.
    # Actual COPY actions charge time and destination capacity; identity persists.
    plan.handoff(PhaseHandoff(destination="dram"))


def run_example(output,kind="dense"):
    output=Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("--out must name a new or empty directory")
    output.mkdir(parents=True,exist_ok=True)
    simulator=make_simulator(kind)
    scheduling=Scheduling(default_device="gpu",release="last_use",
        expert_rule="current_cost" if kind=="moe" else "fixed",
        expert_order="token_count_desc" if kind=="moe" else "graph_order",
        cache=Cache(rule="lru",budget_bytes=65536),
        prefetch=Prefetch(next_layers=1,max_count=2,include_dense_weights=True))
    def build(plan):
        configure(plan,kind)
        (output/"declaration.json").write_text(json.dumps(plan.declaration_document(),indent=2)+"\n",encoding="utf-8",newline="\n")
        (output/"inspection.json").write_text(json.dumps(plan.inspect().document(),indent=2)+"\n",encoding="utf-8",newline="\n")
    with simulator.compile(scheduling=scheduling,configure=build,directory=output/"inputs") as compiled:
        result=compiled.run()
        result.save(output/"execution")
        (output/"summary.json").write_text(json.dumps(result.summary(),indent=2)+"\n",encoding="utf-8",newline="\n")
        if not result.success:
            raise RuntimeError(f"request pipeline did not finish: {result.summary()}")
        print(json.dumps(result.summary(),ensure_ascii=False))
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--kind",choices=("dense","moe"),default="dense")
    args=parser.parse_args()
    run_example(args.out,args.kind)
