"""Readable additive report and figures from retained follow-up observations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics

from . import task1


def finish(out):
    out=Path(out).resolve();summary=task1.read(out/"summary.json");manifest=task1.read(out/"manifest.json")
    expected={j["id"] for j in manifest["jobs"]};executions=[]
    for split in ("calibration","transfer-heldout","observer"):
        executions.extend(task1.read(out/(split+"-executions.json")))
    if len(executions)!=len(expected) or {e["job"]["id"] for e in executions}!=expected:
        raise ValueError("native process denominator differs from manifest")
    processes=[(e["process_id"],e["started_at_utc"]) for e in executions if "process_id" in e]
    if len(set(processes))!=len(processes):raise ValueError("duplicate owned process identity")
    for source in manifest["sources"]:
        if task1.sha(source["path"])!=source["sha256"]:raise ValueError("source changed after collection")
    rows=[]
    text=["# Transfer response and observer follow-up", "",
          "This experiment preserves the earlier frozen results. It tests a new independent size-response model and a separately identified observer diagnostic; it does not retroactively qualify the earlier 0/6 complete strategy groups.","",
          f"Native jobs: {len(executions)}/{len(expected)} retained; {sum(e['status']=='completed' for e in executions)} completed and {sum(e['status']!='completed' for e in executions)} failed. Numerical/copy and owned-process checks accompany each successful receipt.","",
          "## Independent transfer model", "",
          "| Route | Selected response | Maximum heldout size-mean APE | Heldout CV passes | Qualified | Absolute-OLS heldout maximum |",
          "|---|---|---|---|---|---|"]
    for route,result in summary["transfer"].items():
        model=result["model"]
        family="relative affine" if model["breakpoint_bytes"] is None else "continuous hinge at "+str(model["breakpoint_bytes"])+" B"
        text.append(f"| {route} | {family} | {result['maximum_heldout_size_mean_ape']:.2%} | {result['process_cv_passes']}/{result['processes']} | {result['qualified']} | {result['baseline_metrics']['ordinary_absolute_ols']['maximum_heldout_size_mean_ape']:.2%} |")
        for split,field in (("calibration","calibration_process_scatter"),("heldout","heldout_process_scatter")):
            for row in result[field]:rows.append(dict(route=route,split=split,bytes=row["bytes"],process_id=row["process_id"],mean_ns=row["mean_ns"],trial_cv=row["cv"],native_sha256=row["evidence"]["sha256"]))
    text += ["", "The fixed and size-dependent terms are inclusive effective backend responses. They are not independently identified driver, DMA, CPU-copy or PCIe-only costs. Both all-process scatter and the same-data affine controls remain visible. Heldout means and trials are never used to choose or refit the frozen model.","",
             "Qualification requires the maximum heldout size-mean APE to be at most 10% and every heldout process trial CV to be at most 10%. All small and large sizes remain in the denominator. Independently predicting sizes does not by itself qualify a physical-link bandwidth counterfactual.","",
             "The default CPU allocation path is source-bound GGML aligned heap allocation. Host page-lock status was not queried by the retained compact probe; pinned-buffer behavior is not claimed. H2D and D2H invoke the retained tensor_set/tensor_get CUDA asynchronous copy and stream-completion path, followed by the explicit backend synchronization in Component.action.",""]
    observer=summary["observer"]
    if observer["processes"]:
        real=[r for r in observer["rows"] if not r.get("sham",False)]
        sham=[r for r in observer["rows"] if r.get("sham",False)]
        text += ["## Matched observer controls","",
                 f"Bare/stamped contrasts pass in {observer['passed_processes']}/{observer['processes']} fresh processes and {observer['qualified_programs']}/24 complete program groups (all four processes must pass). Bare/bare sham contrasts pass in {observer['sham_passes']}/{observer['sham_processes']} processes.","",
                 "| Control | Mean signed contrast | Maximum absolute contrast | Frozen limit |",
                 "|---|---|---|---|"]
        for name,data in (("bare/stamped",real),("bare/bare sham",sham)):
            vals=[r["relative_difference"] for r in data if r["status"]=="completed"]
            text.append(f"| {name} | {statistics.mean(vals):.2%} | {max(map(abs,vals)):.2%} | 5% |")
        text += ["", "Within each initialized process, ABBA/BAAB positions balance a linear time trend. Each block uses the same untimed reset and one plain preconditioning execution. This separates a mode contrast from fresh-process differences more directly, but it does not eliminate nonlinear temporal variation. Bare/bare sham differences reveal remaining run variation without proving an OS or cache cause.","",
                 "The observer harness is a new binary and has additional preconditioning. Its prices must not be compared with old frozen prices as a strict accuracy qualification. These diagnostics do not alter the earlier 13/24 programs or 0/6 groups, and they do not validate dynamic-background response.",""]
    else:
        text += ["## Longer-window scope", "",
                 "This version has no new observer jobs. It is the separately frozen 256-replay transfer campaign, proposed after observing short-window calibration dispersion. The same sizes, model families and gates are retained. Longer bursts can shift runtime state and mean response; a reduced CV would not make these costs valid replacements for the historical 8-replay or mixed-strategy costs.",""]
    text += ["## Mechanism and reusable interface","",
             "The response mechanism and exact-anchor scenario arithmetic live in modeling/llm_infer_model/performance/transfer_response.py. Profiling fits nonnegative terms with training-only leave-one-size-out selection, retains all raw samples and validates new sizes. The modeling interface rejects missing scope/evidence, unsupported routes, out-of-range sizes and negative fixed residuals.","",
             "A non-unit byte-service scenario must either have separate scaling qualification or explicitly declare an effective-byte-service counterfactual assumption. An interpolation fit is never silently advertised as measured PCIe bandwidth. No DNN or complete-target correction is used.","",
             "Raw programs, requests, stdout/stderr, native outputs and execution receipts are in jobs/. The model and freeze barrier precede all heldout starts. native-build-receipt.json (where present) binds the diagnostic build, and the manifest binds old/new binary and runtime DLL hashes.",""]
    (out/"REPORT.md").write_text("\n".join(text),encoding="utf-8",newline="\n")
    with (out/"transfer_process_scatter.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    if os.environ.get("DAC_FOLLOWUP_TEXT_ONLY")=="1":
        return dict(report=str(out/"REPORT.md"),native_expected=len(expected),figures=0,
                    figures_and_artifact_hashing_deferred=True)
    # Plot only after native sampling has ended; rendering must not compete.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from .transfer_fit import predict
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
    figures=out/"figures";figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
    for ax,(route,result) in zip(axes,summary["transfer"].items()):
        for split,marker,color,label in (("calibration","o","#2c6ca2","Calibration processes"),("heldout","x","#d76624","Independent heldout processes")):
            rr=[r for r in rows if r["route"]==route and r["split"]==split]
            ax.scatter([r["bytes"] for r in rr],[r["mean_ns"]/1000 for r in rr],marker=marker,color=color,s=23,alpha=.65,label=label)
        xx=np.geomspace(4,67108864,300)
        ax.plot(xx,[predict(result["model"],x)/1000 for x in xx],color="black",lw=1.2,label="Frozen selected response")
        baseline=result["baseline_metrics"]["ordinary_absolute_ols"]["model"]
        yy=[predict(baseline,x)/1000 for x in xx]
        ax.plot(xx,[v if v>0 else float('nan') for v in yy],color="#7d7d7d",ls="--",lw=1,label="Absolute OLS affine control")
        ax.set_xscale("log",base=2);ax.set_yscale("log");ax.set_xlabel("Payload bytes");ax.set_ylabel("Inclusive COPY response (µs)")
        ax.set_title(f"{route}: heldout max APE {result['maximum_heldout_size_mean_ape']:.1%}\nQualified: {result['qualified']}")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8,loc="upper left")
    fig.savefig(figures/"transfer-heldout.png",dpi=220);fig.savefig(figures/"transfer-heldout.svg");plt.close(fig)
    if observer["processes"]:
        keys=sorted({r.get("program_key") for r in observer["rows"] if r.get("program_key") and not r.get("sham")})
        data=np.full((len(keys),4),np.nan)
        for row in observer["rows"]:
            if row.get("sham") or row.get("program_key") not in keys:continue
            data[keys.index(row["program_key"]),row["round"]]=row["relative_difference"]*100
        fig,ax=plt.subplots(figsize=(8,8.5),constrained_layout=True)
        bound=max(5,float(np.nanmax(np.abs(data))))
        im=ax.imshow(data,cmap="RdBu_r",vmin=-bound,vmax=bound,aspect="auto")
        ax.set_yticks(range(len(keys)),[k.removesuffix("-heldout") for k in keys],fontsize=9)
        ax.set_xticks(range(4),["ABBA r0","BAAB r1","ABBA r2","BAAB r3"])
        for (r,c),val in np.ndenumerate(data):
            ax.text(c,r,f"{val:+.1f}"+(" ×" if abs(val)>5 else ""),ha="center",va="center",fontsize=8,color="black" if abs(val)<bound*.65 else "white")
        ax.set_title("Same-process stamped versus bare response (%)\n× retains failure of the frozen 5% contrast gate")
        fig.colorbar(im,ax=ax,label="Relative difference (%)",shrink=.65)
        fig.savefig(figures/"observer-paired.png",dpi=220);fig.savefig(figures/"observer-paired.svg");plt.close(fig)
    artifact_paths=[p for p in out.rglob("*") if p.is_file() and p.name!="artifact-manifest.json"]
    receipt=dict(schema="dac-followup-task1-artifacts-1",native_expected=len(expected),native_completed=sum(e["status"]=="completed" for e in executions),native_failed=sum(e["status"]!="completed" for e in executions),
                 old_results_unchanged=True,files=[task1.receipt(p) for p in sorted(artifact_paths)])
    task1.write(out/"artifact-manifest.json",receipt)
    return dict(report=str(out/"REPORT.md"),native_expected=len(expected),figures=len(list(figures.glob("*.png"))))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True);args=parser.parse_args();print(finish(args.out))
