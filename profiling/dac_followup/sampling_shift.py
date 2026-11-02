"""Post-freeze mean/dispersion comparison across two declared sampling windows.

The campaigns occur at different times. This is a descriptive domain-shift
check, not a randomized attribution of every difference to repeat count.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics
from . import task1


def compare(short,long,out):
    short=Path(short).resolve();long=Path(long).resolve();out=Path(out).resolve()
    a=task1.read(short/"summary.json");b=task1.read(long/"summary.json");rows=[]
    for route in a["transfer"]:
        for split,field in (("calibration","calibration_process_scatter"),("heldout","heldout_process_scatter")):
            before=a["transfer"][route][field];after=b["transfer"][route][field]
            for size in sorted({r["bytes"] for r in before}):
                xs=[r for r in before if r["bytes"]==size];ys=[r for r in after if r["bytes"]==size]
                if len(xs)!=3 or len(ys)!=3:raise ValueError("size/fresh-process coverage changed")
                am=statistics.mean(r["mean_ns"] for r in xs);bm=statistics.mean(r["mean_ns"] for r in ys)
                rows.append(dict(route=route,split=split,bytes=size,short_mean_ns=am,long_mean_ns=bm,
                    relative_mean_shift=bm/am-1,short_maximum_trial_cv=max(r["cv"] for r in xs),long_maximum_trial_cv=max(r["cv"] for r in ys),
                    short_between_process_cv=statistics.stdev(r["mean_ns"] for r in xs)/am,
                    long_between_process_cv=statistics.stdev(r["mean_ns"] for r in ys)/bm))
    out.mkdir(parents=True,exist_ok=False)
    with (out/"mean-dispersion-shift.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    receipt=dict(schema="dac-followup-sampling-window-shift-1",short=task1.receipt(short/"summary.json"),long=task1.receipt(long/"summary.json"),
                 source=task1.receipt(__file__),rows=rows,maximum_absolute_mean_shift=max(abs(r["relative_mean_shift"]) for r in rows),
                 same_binary=task1.read(short/"manifest.json")["binaries"]["transfer"]==task1.read(long/"manifest.json")["binaries"]["transfer"],
                 means_equivalent_not_assumed=True,repeat_count_is_only_known_deterministic_protocol_change=True,
                 temporal_state_confounded=True,old_prices_replaced=False)
    task1.write(out/"summary.json",receipt)
    return dict(rows=len(rows),maximum_absolute_mean_shift=receipt["maximum_absolute_mean_shift"])

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--short",type=Path,required=True);parser.add_argument("--long",type=Path,required=True);parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();print(compare(args.short,args.long,args.out))
