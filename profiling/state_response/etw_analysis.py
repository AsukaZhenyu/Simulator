"""Offline, owned-PID-only scheduler reconstruction. Never controls ETW sessions.
Running denotes scheduler occupancy, not useful compute. Waiting and Ready are
only assigned between observed scheduler transitions; unresolved spans stay unknown.
"""
from __future__ import annotations
import argparse, bisect, collections, csv, hashlib, json, re
from pathlib import Path

OWNED_FIELDS={"CSwitch":("New Process Name ( PID)","Old Process Name ( PID)"),"ReadyThread":("Rdy Process Name ( PID)",)}
TYPES={"CSwitch","ReadyThread","T-Start","T-End","T-DCStart","T-DCEnd","P-Start","P-End","P-DCStart","P-DCEnd","Mark"}
RAW=re.compile(r"#\s*(\d+)\s*#\s*(0x[0-9a-fA-F]+)\s*#\s*(\d+)\s*#")

def dump_filtered(source:Path,pid:int,instance_name:str):
    headers={}; owned=[]; raw=None; counts=collections.Counter(); marker=[]; origins=set()
    def is_owned(value):return re.search(r"\(\s*"+str(pid)+r"\s*\)",value) is not None
    with source.open(encoding="utf-8-sig",newline="") as stream:
        for number,line in enumerate(stream,1):
            if line.startswith("#"):
                m=RAW.match(line);raw={"qpc":int(m[2],16),"cpu":int(m[1]),"relative_ns":int(m[3])} if m else None
                continue
            row=[value.strip() for value in next(csv.reader([line],skipinitialspace=True))]
            if not row:continue
            kind=row[0]
            if len(row)>1 and row[1]=="TimeStamp":headers[kind]=row;continue
            if kind not in TYPES:continue
            counts[kind]+=1
            if kind not in headers:raise ValueError(f"missing actual xperf header: {kind}")

            d=dict(zip(headers[kind],row));d.update(_line=number,_raw=raw,_type=kind)
            if len(row)>len(headers[kind]):d["_unmapped_trailing_fields"]=row[len(headers[kind]):]
            if len(row)<len(headers[kind]):d["_missing_trailing_fields"]=headers[kind][len(row):]
            if kind=="Mark" and d.get("Mark") in ("LLMSimBegin_"+instance_name,"LLMSimEnd_"+instance_name):marker.append(d)
            if not any(is_owned(d.get(field,"")) for field in OWNED_FIELDS.get(kind,("Process Name ( PID)",))):continue
            needed={"CSwitch":("New TID","Old TID","OldState","Wait Reason","CPU"),"ReadyThread":("Rdy TID",),"T-Start":("ThreadID",),"T-End":("ThreadID",)}.get(kind,())
            if any(key not in d for key in needed):raise ValueError(f"missing required scheduler field for {kind} at line {number}")
            if raw is None:raise ValueError("owned event has no raw-QPC row")
            if kind=="CSwitch" and int(d["CPU"])!=raw["cpu"]:raise ValueError("CSwitch CPU differs from raw prefix CPU")
            origins.add(raw["qpc"]*100-raw["relative_ns"])
            for key in list(d):
                if "Process Name" in key and not is_owned(d[key]):d[key]="other"
            if kind=="CSwitch":
                if not is_owned(d["New Process Name ( PID)"]):d["New TID"]="other"
                if not is_owned(d["Old Process Name ( PID)"]):d["Old TID"]="other"
            if kind=="ReadyThread" and not is_owned(d["Process Name ( PID)"]):d["ThreadID"]="other"
            for key in ("ParentPID","UserSid","Command Line"):d.pop(key,None)
            owned.append(d)
    return owned,{"headers":{k:v for k,v in headers.items() if k in counts},"counts_all":dict(counts),"owned_rows":len(owned),"marker_rows":marker,"raw_qpc_relative_ns_origins":sorted(origins)}

def reconstruct(rows,tid,begin,end):
    events=[]; starts=[]; end_events=[]; sequence=0
    for r in rows:
        k=r["_type"];q=r["_raw"]["qpc"]
        if k=="T-Start" and r["ThreadID"]==str(tid):starts.append(r)
        if k=="T-End" and r["ThreadID"]==str(tid):end_events.append(r)
        if k=="CSwitch":
            if r["Old TID"]==str(tid):events.append((q,0,r["_line"],"out",r))
            if r["New TID"]==str(tid):events.append((q,2,r["_line"],"in",r))
        elif k=="ReadyThread" and r["Rdy TID"]==str(tid):events.append((q,1,r["_line"],"ready",r))
    if len(starts)!=1:raise ValueError(f"TID {tid} has {len(starts)} actual start events; lifetime ambiguous")
    life_start=starts[0]["_raw"]["qpc"]
    life_end=end_events[0]["_raw"]["qpc"] if len(end_events)==1 else None
    if len(end_events)>1:raise ValueError("multiple thread end events")
    if begin<life_start or (life_end is not None and end>life_end):raise ValueError("query outside exact thread lifetime")
    state="unknown";at=life_start;reason=None;origin=None;cpu=None;segments=[];anomalies=[]
    def close(q,finish,override=None):
        nonlocal at
        if q<at:raise ValueError("nonmonotonic event")
        lo=max(at,begin);hi=min(q,end)
        if hi>lo:segments.append({"begin_qpc":lo,"end_qpc":hi,"state":override or state,"wait_reason":reason if (override or state)=="waiting" else None,"cpu":cpu if (override or state)=="running" else None,"begin_event_line":origin,"end_event_line":finish})
        at=q
    for q,_,_,kind,r in sorted(events):
        if q<life_start:raise ValueError("scheduler event before thread creation")
        if life_end is not None and q>life_end:raise ValueError("scheduler event after thread exit")
        line=r["_line"]
        if kind=="out":
            valid=state=="running" and cpu==int(r["CPU"])
            if not valid:anomalies.append({"line":line,"issue":"switch_out_without_matching_switch_in_or_cpu","prior_state":state})
            close(q,line,None if valid else "unknown")
            old=r["OldState"]
            state="waiting" if old=="Waiting" else "ready" if old in ("Ready","Standby","DeferredReady") else "unknown"
            reason=r["Wait Reason"] if state=="waiting" else None;origin=line;cpu=None
        elif kind=="ready":
            if state=="ready":continue # duplicate observed ready does not reset queue start
            if state=="running":
                anomalies.append({"line":line,"issue":"ready_while_running"});continue
            close(q,line);state="ready";reason=None;origin=line;cpu=None
        else:
            override="unknown_wait_or_ready" if state=="waiting" else "unknown" if state=="running" else None
            if state in ("waiting","running"):anomalies.append({"line":line,"issue":"switch_in_without_observed_ready" if state=="waiting" else "repeated_switch_in"})
            close(q,line,override);state="running";reason=None;origin=line;cpu=int(r["CPU"])
        if at>=end:break
    if at<end:close(min(end,life_end) if life_end is not None else end,None,"unknown_censored")
    if sum(x["end_qpc"]-x["begin_qpc"] for x in segments)!=end-begin:raise ValueError("timeline does not partition requested interval")
    return {"thread_id":tid,"start_qpc":life_start,"end_qpc":life_end,"entry_image_function":starts[0].get("Image!Function"),"segments":segments,"anomalies":anomalies}

def intersect(segments,begin,end):
    result=collections.Counter()
    for s in segments:
        size=min(end,s["end_qpc"])-max(begin,s["begin_qpc"])
        if size>0:result[s["state"]]+=size
    if sum(result.values())!=end-begin:raise ValueError("intersection coverage mismatch")
    return dict(result)

def summary_ns(ticks,freq):return {state:round(value*1e9/freq) for state,value in ticks.items()}

def validate_native_identity(identity,result,process):
    actor=identity["actors"][0]; ready=process["ready"]
    if (process["pid"]!=actor["process_id"] or ready["process_id"]!=actor["process_id"]
            or ready["main_thread_id"]!=actor["main_thread_id"]
            or process["launch_tick"]!=actor["launch_tick"] or ready["ready_tick"]!=actor["ready_tick"]):
        raise ValueError("owned process/READY identity semantics differ from declared identity")
    clock=ready["clock"]
    if clock["clock_id"]!="windows_qpc" or clock["frequency_hz"]!=actor["ready_qpc_frequency"]:
        raise ValueError("owned READY clock differs from declared identity")
    if (result["process_id"]!=actor["process_id"] or result["main_thread_id"]!=actor["main_thread_id"]
            or result["clock"]["clock_id"]!=clock["clock_id"] or result["clock"]["frequency_hz"]!=clock["frequency_hz"]
            or result["ready_tick"]!=ready["ready_tick"] or result["sampling_end_tick"]!=actor["sampling_end_tick"]
            or result["program_sha256"]!=ready["program_sha256"]):
        raise ValueError("native result identity/clock/program differs from owned READY")
    complete=process.get("sampling_complete")
    if complete and (complete["process_id"]!=actor["process_id"] or complete["main_thread_id"]!=actor["main_thread_id"]
            or complete["sampling_end_tick"]!=result["sampling_end_tick"] or complete["clock"]["frequency_hz"]!=clock["frequency_hz"]
            or complete["clock"]["clock_id"]!=clock["clock_id"]):
        raise ValueError("sampling-complete identity/clock differs from native result")

def analyze(rows,meta,raw,identity,result):
    actor=identity["actors"][0];pid=actor["process_id"];tid=actor["main_thread_id"];freq=actor["ready_qpc_frequency"]
    if result["process_id"]!=pid or result["main_thread_id"]!=tid or result["clock"]["clock_id"]!="windows_qpc" or result["clock"]["frequency_hz"]!=freq:
        raise ValueError("native result identity/clock differs from owned evidence")
    if raw["identity"]!=identity:raise ValueError("raw header export identity differs")
    for header in [raw["header"]]+[p["after_ProcessTrace"] for p in raw["pass_receipts"]]:
        if any(header[k] for k in ("header_events_lost","header_buffers_lost","logfile_events_lost")):raise ValueError("nonzero actual loss counter")
    if freq!=10_000_000 or raw["header"]["perf_frequency"]!=freq:raise ValueError("raw xperf timestamp mapping expects verified 10MHz QPC")
    if not raw["clock_is_qpc"] or not raw["header_loss_counters_zero"]:raise ValueError("loss/clock qualification failed")
    if len(meta["raw_qpc_relative_ns_origins"])!=1:raise ValueError("xperf raw timestamps are not one QPC-relative mapping")
    starts=[r for r in rows if r["_type"]=="P-Start"]
    if len(starts)!=1 or not actor["launch_tick"]<=starts[0]["_raw"]["qpc"]<=actor["ready_tick"]:raise ValueError("owned process start outside launch/READY bracket")
    if any(r["_type"] in ("P-End","P-DCStart") for r in rows):raise ValueError("unexpected process end/rundown start; lifetime requires review")
    begin=result["sampling_begin_tick"];end=result["sampling_end_tick"]
    if not actor["ready_tick"]<begin<end:raise ValueError("invalid native sampling interval")
    tids=sorted({int(r["ThreadID"]) for r in rows if r["_type"]=="T-Start" and r["_raw"]["qpc"]<=end})
    timelines=[]
    for thread in tids:
        exits=[r["_raw"]["qpc"] for r in rows if r["_type"]=="T-End" and r["ThreadID"]==str(thread)]
        if exits and exits[0]<begin:continue
        timelines.append(reconstruct(rows,thread,begin,end))
    main=next(t for t in timelines if t["thread_id"]==tid)
    categories=collections.defaultdict(collections.Counter);by_action=collections.defaultdict(collections.Counter);records=[];aux=collections.defaultdict(collections.Counter)
    action_descriptors=result["actions"]
    for block in result["blocks"]:
        for a in block["action_intervals"]:
            desc=action_descriptors[a["action_index"]]
            category=("cpu_compute" if desc["device"]=="cpu" else "gpu_compute") if desc["kind"]=="COMPUTE" else desc["kind"].lower()
            values=intersect(main["segments"],a["begin_tick"],a["end_tick"])
            categories[category].update(values);by_action[str(a["action_index"])].update(values)
            record={**a,"category":category,"main_state_ns":summary_ns(values,freq),"descriptor":desc};records.append(record)
            for t in timelines:
                if t["thread_id"]!=tid:aux[str(t["thread_id"])+":"+category].update(intersect(t["segments"],a["begin_tick"],a["end_tick"]))
    wait_reasons=collections.Counter()
    for s in main["segments"]:
        if s["state"]=="waiting":wait_reasons[s["wait_reason"]]+=s["end_qpc"]-s["begin_qpc"]
    summary={"schema":"owned-etw-scheduler-analysis-1","pid":pid,"main_thread_id":tid,"qpc_frequency":freq,"sampling_begin_tick":begin,"sampling_end_tick":end,"sampling_wall_ns":round((end-begin)*1e9/freq),"iteration_count":sum(b["iteration_count"] for b in result["blocks"]),"action_count":len(records),"clock_is_qpc":True,"all_read_loss_counters_zero":True,"raw_qpc_mapping_single_origin":True,"process_launch_lifetime_verified":True,"raw_marker_clock_verified":False,"marker_count_in_current_export":len(meta["marker_rows"]),"marker_limitation":"No actual Mark row found in provider-filtered dumper; API receipts are not treated as emitted-marker proof.","main_state_ns":summary_ns(intersect(main["segments"],begin,end),freq),"by_category_state_ns":{k:summary_ns(v,freq) for k,v in categories.items()},"by_action_state_ns":{k:{"descriptor":action_descriptors[int(k)],"state_ns":summary_ns(v,freq)} for k,v in by_action.items()},"main_wait_reason_ns":summary_ns(wait_reasons,freq),"auxiliary_threads":[{"thread_id":t["thread_id"],"entry_image_function":t["entry_image_function"],"state_ns":summary_ns(intersect(t["segments"],begin,end),freq),"anomalies":len(t["anomalies"])} for t in timelines if t["thread_id"]!=tid],"auxiliary_by_category_state_ns":{k:summary_ns(v,freq) for k,v in aux.items()},"main_anomalies":main["anomalies"],"dpc_isr_time_subtracted":False,"limitations":["Running is scheduled occupancy, not useful computation; DPC/ISR may overlap.","Thread entry module identifies creation entry only; it does not prove all subsequent samples execute in that module.","Auxiliary CPU occupancy overlaps main wall time and must not be added to it.","Kernel wait reasons are retained as labels and do not establish application cause.","No CUDA kernel timing is measured by these events.","Partial boundaries and unobserved ready transitions remain unknown.","Diagnostic ETW run is not a cost table or independent accuracy validation."]}
    return summary,timelines,records

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--dump",type=Path);ap.add_argument("--filtered",type=Path);ap.add_argument("--meta",type=Path);ap.add_argument("--raw",type=Path,required=True);ap.add_argument("--identity",type=Path,required=True);ap.add_argument("--result",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);args=ap.parse_args()
    args.out.mkdir(exist_ok=False)
    if args.dump:rows,meta=dump_filtered(args.dump,json.loads(args.identity.read_text())["actors"][0]["process_id"],json.loads(args.identity.read_text())["instance_name"])
    else:rows=[json.loads(x) for x in args.filtered.read_text().splitlines()];meta=json.loads(args.meta.read_text())
    actor=json.loads(args.identity.read_text())["actors"][0]
    if hashlib.sha256(args.result.read_bytes()).hexdigest()!=actor["result_evidence_sha256"]:
        raise ValueError("native result digest mismatch")
    process_path=Path(actor["process_evidence"])
    if hashlib.sha256(process_path.read_bytes()).hexdigest()!=actor["process_evidence_sha256"]:
        raise ValueError("owned process evidence digest mismatch")
    # Legacy first filtered pass did not retain origins: compute using its preserved raw rows.
    meta["raw_qpc_relative_ns_origins"]=sorted({r["_raw"]["qpc"]*100-r["_raw"]["relative_ns"] for r in rows})
    for row in rows:
        for key in ("Old TID","New TID"):
            name="Old Process Name ( PID)" if key.startswith("Old") else "New Process Name ( PID)"
            if row.get(name)=="other" and key in row:row[key]="other"
        if row.get("Process Name ( PID)")=="other" and row["_type"]=="ReadyThread":row["ThreadID"]="other"
    validate_native_identity(json.loads(args.identity.read_text()),json.loads(args.result.read_text()),json.loads(process_path.read_text()))
    summary,timelines,actions=analyze(rows,meta,json.loads(args.raw.read_text()),json.loads(args.identity.read_text()),json.loads(args.result.read_text()))
    for name,value in (("analysis.json",summary),("timelines.json",timelines),("schema-and-counts.json",meta)):(args.out/name).write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8",newline="\n")
    for name,value in (("owned-schedule.jsonl",rows),("action-scheduler-states.jsonl",actions)):(args.out/name).write_text("".join(json.dumps(x)+"\n" for x in value),encoding="utf-8",newline="\n")
    evidence={"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"inputs":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.dump,args.filtered,args.meta,args.raw,args.identity,args.result) if p},"claims_scope":"offline derived diagnostic only; no performance parameter fitting"}
    (args.out/"evidence.json").write_text(json.dumps(evidence,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({k:summary[k] for k in ("iteration_count","action_count","sampling_wall_ns","main_state_ns","by_category_state_ns")}))

if __name__=="__main__":main()
