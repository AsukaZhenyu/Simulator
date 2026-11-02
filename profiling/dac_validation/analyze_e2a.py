"""Post-freeze explanation only: never refit a price from target diagnostics."""
from __future__ import annotations
import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def receipt(path):
    p=Path(path).resolve();return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",type=Path,required=True);args=parser.parse_args()
    out=args.out.resolve();manifest=read(out/"manifest.json");summary=read(out/"summary.json");predictions=read(out/"predictions.json")
    observer=read(out/"observer-executions.json");rows=[];programs=[];kv_rows=[]
    for program_row in summary["programs"]:
        wid,strategy=program_row["workload"],program_row["strategy"]
        key=f"{wid}-s{strategy}-heldout";program=read(manifest["programs"][key]["program"]["path"])
        prediction=next(p for p in predictions if p["program_key"]==key)
        if wid.endswith("d1h128"):
            old={b["id"] for binding in program["bindings"] for b in binding["history"]}
            new={b["id"] for binding in program["bindings"] for b in binding["new_kv"]}
            tensors={t["id"]:t for t in program["graph"]["workload"]["tensors"]}
            counts={};byte_counts={};duration=Fraction();details=[]
            for index,action in enumerate(program["actions"]):
                if action["kind"]!="COPY" or action["target_id"] not in old|new:continue
                part="old" if action["target_id"] in old else "new"
                category=part+":"+action["src"]+"->"+action["dst"]
                elapsed=Fraction(prediction["prediction"]["action_ns"][index]);size=tensors[action["target_id"]]["storage_bytes"]
                counts[category]=counts.get(category,0)+1;byte_counts[category]=byte_counts.get(category,0)+size;duration+=elapsed
                details.append(dict(index=index,tensor_id=action["target_id"],kind=part,src=action["src"],dst=action["dst"],bytes=size,predicted_ns=str(elapsed)))
            kv_rows.append(dict(workload=wid,strategy=strategy,kv_copy_count=sum(counts.values()),kv_copy_bytes=sum(byte_counts.values()),
                kv_predicted_ns=str(duration),total_predicted_ns=prediction["prediction"]["duration_ns"],
                kv_predicted_fraction=float(duration/Fraction(prediction["prediction"]["duration_ns"])),
                counts_by_kind_route=counts,bytes_by_kind_route=byte_counts,actions=details,
                copy_coalescing_measured=False,backend_contract="token materialized KV blocks; one COPY per declared block",future_decode_generation_claimed=False))
        relevant=[e for e in observer if e["job"]["program_key"]==key and e["job"]["request"]["mode"]=="full-stamped" and e["status"]=="completed"]
        categories=[]
        bindings={b["operation_id"]:b for b in program["bindings"]}
        for action in program["actions"]:
            categories.append("compute:"+action["device"]+":"+bindings[action["target_id"]]["kind"] if action["kind"]=="COMPUTE" else
                "copy:"+action["src"]+"->"+action["dst"] if action["kind"]=="COPY" else action["kind"].lower())
        measurements={category:[] for category in set(categories)};unattributed=[];whole=[]
        for execution in relevant:
            native=read(execution["native"]["path"])
            for trial in native["trials"]:
                own={category:0. for category in measurements}
                for index,stamp in enumerate(trial["actions"]):own[categories[index]]+=stamp["wall_ns"]/trial["repeats"]
                for category,value in own.items():measurements[category].append(value)
                whole.append(trial["wall_ns"]/trial["repeats"])
                unattributed.append(trial["wall_ns"]/trial["repeats"]-sum(own.values()))
        qualified=all(pair["passed"] for pair in program_row["observer_pairs"])
        for category,values in sorted(measurements.items()):
            standalone=float(Fraction(prediction["prediction_breakdown_ns"][category]));mixed=statistics.mean(values) if values else None
            rows.append(dict(workload=wid,strategy=strategy,category=category,action_count=prediction["action_counts"][category],
                copy_bytes=prediction["copy_bytes"].get(category,0),isolated_prediction_ns=standalone,
                mixed_stamped_mean_ns=mixed,signed_gap_ns=None if mixed is None else mixed-standalone,
                mixed_over_isolated=None if mixed is None or standalone<=0 else mixed/standalone,
                stamped_trials=len(values),observer_qualified=qualified,used_to_refit=False))
        real=program_row["mean_real_ns"]
        programs.append(dict(workload=wid,strategy=strategy,mean_real_bare_ns=real,predicted_ns=program_row["prediction_ns"],
            aggregate_program_mean_ape=None if real is None else abs(program_row["prediction_ns"]-real)/real,
            completed_stamped_processes=len(relevant),stamped_whole_mean_ns=None if not whole else statistics.mean(whole),
            stamped_unattributed_loop_clock_mean_ns=None if not unattributed else statistics.mean(unattributed),
            observer_qualified=qualified,observer_relative_differences=[p.get("relative_difference") for p in program_row["observer_pairs"]],
            attribution="inclusive action boundary difference only; not identification of OS scheduling, driver wait, clock or cache mechanism",
            unbiased_target_action_truth_claimed=False,used_to_refit=False))
    finite=[p for p in programs if p["aggregate_program_mean_ape"] is not None]
    doc=dict(schema="dac-e2a-post-freeze-diagnostics-1",sources=[receipt(out/"summary.json"),receipt(out/"profile-frozen.json"),receipt(out/"predictions.json"),receipt(__file__)],
        price_refitted=False,prediction_freeze_unchanged=True,programs=programs,worst_programs=sorted(finite,key=lambda p:p["aggregate_program_mean_ape"],reverse=True)[:5],
        caution="Observer diagnostics are potentially perturbed, especially when any paired stamped/bare difference exceeds 5%; they explain boundary gaps but are not unbiased physical stage truth.")
    with (out/"postfreeze-diagnostics.json").open("x",encoding="utf-8",newline="\n") as stream:json.dump(doc,stream,indent=2,allow_nan=False);stream.write("\n")
    with (out/"component_diagnostics.csv").open("x",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    with (out/"kv-copy-ledger.json").open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(dict(schema="dac-e2a-frozen-kv-copy-ledger-1",sources=[receipt(out/"manifest.json"),receipt(out/"predictions.json")],rows=kv_rows,
            claim="Frozen-program action accounting only; no copy coalescing or hypothetical timing measured"),stream,indent=2);stream.write("\n")
    with (out/"kv_copy_metrics.csv").open("x",encoding="utf-8",newline="") as stream:
        fields=["workload","strategy","kv_copy_count","kv_copy_bytes","kv_predicted_ns","total_predicted_ns","kv_predicted_fraction"]
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore",lineterminator="\n");writer.writeheader();writer.writerows(kv_rows)
    with (out/"diagnostic-manifest.json").open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(dict(schema="dac-e2a-diagnostic-artifacts-1",files=[receipt(out/"postfreeze-diagnostics.json"),receipt(out/"component_diagnostics.csv"),receipt(out/"kv-copy-ledger.json"),receipt(out/"kv_copy_metrics.csv")],sources=doc["sources"]),stream,indent=2);stream.write("\n")
    print(json.dumps(dict(programs=len(programs),component_rows=len(rows),worst=doc["worst_programs"]),indent=2))


if __name__=="__main__":main()
