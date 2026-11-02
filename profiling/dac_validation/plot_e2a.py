"""Paper plot of every real candidate, retaining qualification failures."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def receipt(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--stem",default="real_strategy_selection",help="New figure basename; existing artifacts are never overwritten")
    args=parser.parse_args();out=args.out.resolve()
    summary=json.loads((out/"summary.json").read_text(encoding="utf-8"))
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none"})
    fig,axes=plt.subplots(2,3,figsize=(13,8.7))
    for ax,group in zip(axes.ravel(),summary["workloads"]):
        rows=[r for r in summary["programs"] if r["workload"]==group["workload"]]
        x=np.arange(4);actual=[float("nan") if r["mean_real_ns"] is None else r["mean_real_ns"]/1e6 for r in rows]
        pred=[r["prediction_ns"]/1e6 for r in rows]
        ax.bar(x-.18,actual,width=.34,color="#3b76ac",alpha=.9,label="Real: mean of 8 process means")
        ax.bar(x+.18,pred,width=.34,facecolor="none",edgecolor="#d77729",linewidth=1.5,hatch="//",label="Predicted: frozen independent profiles")
        for i,row in enumerate(rows):
            ys=[p["mean_ns"]/1e6 for p in row["processes"]]
            ax.scatter(np.full(len(ys),i-.18)+np.linspace(-.06,.06,len(ys)),ys,s=12,color="#172b3a",alpha=.65,zorder=4,label="Each independent process mean" if i==0 else None)
            ymax=max([pred[i],*ys]) if ys else pred[i]
            ax.text(i,ymax*1.08,"PASS" if row["fully_qualified"] else "UNQUAL.",ha="center",va="bottom",fontsize=7,color="#22764a" if row["fully_qualified"] else "#ac3030")
        total=sum(r["fully_qualified"] for r in rows)
        ax.set_title(group["workload"].upper()+f"   ({total}/4 programs qualified)",fontweight="bold",pad=12)
        ax.set_xticks(x,["S0\nCPU","S1\nGPU","S2\nCPU\nattention","S3\nserial\nFFN split"])
        ax.set_ylabel("Replay-average latency (ms)");ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
        visible=[v for v in [*actual,*pred,*[p["mean_ns"]/1e6 for row in rows for p in row["processes"]]] if np.isfinite(v)]
        ax.set_ylim(0,max(visible)*1.30)
        if group.get("regret") is not None:
            ax.text(.02,.95,f"Selected S{group['predicted_choice']}; real regret {100*group['regret']:.2f}%",transform=ax.transAxes,va="top",fontsize=8)
    handles,labels=axes.ravel()[0].get_legend_handles_labels()
    unique={label:handle for handle,label in zip(handles,labels)}
    fig.legend(unique.values(),unique.keys(),loc="upper center",bbox_to_anchor=(.5,.948),ncol=3,frameon=False)
    fig.suptitle("Independent component profiles versus matched complete Dense strategies",fontsize=15,fontweight="bold",y=.99)
    fig.text(.05,.015,"F32, one CPU thread and one CUDA GPU; serial native execution; fixed-input replay averages.\n"
        "UNQUAL. retains any accuracy, variability or observer failure. Dots show processes, not per-token tail latency.\n"
        "Startup, allocation, reset and numerical checking are outside timing; declared transfers remain inside.",fontsize=8,color="#364452")
    fig.tight_layout(rect=(.01,.09,.99,.91),h_pad=2.6,w_pad=1.8)
    dest=out/"figures";dest.mkdir(exist_ok=True)
    for extension in ("png","svg"):
        path=dest/(args.stem+"."+extension)
        if path.exists():raise ValueError("figure exists; retain original version")
        fig.savefig(path,dpi=180,bbox_inches="tight")
    manifest=dict(schema="dac-e2a-figure-1",source=receipt(__file__),summary=receipt(out/"summary.json"),
        figures=[receipt(dest/(args.stem+".png")),receipt(dest/(args.stem+".svg"))],
        matplotlib=matplotlib.__version__,all_declared_candidates_included=True,native_timing_active=False)
    with (dest/(args.stem+"-manifest.json")).open("x",encoding="utf-8",newline="\n") as stream:json.dump(manifest,stream,indent=2);stream.write("\n")
    print(json.dumps(manifest,indent=2))


if __name__=="__main__":main()
