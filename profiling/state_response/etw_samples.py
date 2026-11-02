"""Offline owned-thread IP module histogram and marker checks. No symbols guessed."""
from __future__ import annotations
import argparse,bisect,collections,csv,hashlib,json,re
from pathlib import Path
from .etw_analysis import RAW

def parse(path):
    headers={};raw=None
    with path.open(encoding="utf-8-sig") as stream:
        for number,line in enumerate(stream,1):
            if line.startswith("#"):
                m=RAW.match(line);raw={"qpc":int(m[2],16),"cpu":int(m[1]),"relative_ns":int(m[3])} if m else None;continue
            row=[x.strip() for x in next(csv.reader([line],skipinitialspace=True))]
            if not row:continue
            if len(row)>1 and row[1]=="TimeStamp":headers[row[0]]=row;continue
            if row[0] not in headers:continue
            relevant=row[0] in ("I-Start","I-End","I-DCStart","I-DCEnd","Mark","SampledProfile","SampledProfileNmi","Interrupt","DPC","DPCTmr")

            decoded=dict(zip(headers[row[0]],row),_type=row[0],_raw=raw,_line=number)
            if relevant and len(row)>len(headers[row[0]]):decoded["_unmapped_trailing_fields"]=row[len(headers[row[0]]):]
            if relevant and len(row)<len(headers[row[0]]):decoded["_missing_trailing_fields"]=headers[row[0]][len(row):]
            required={"I-Start":("BaseAddr","EndAddr","FileName"),"I-End":("BaseAddr",),"I-DCStart":("BaseAddr","EndAddr","FileName"),"I-DCEnd":("BaseAddr","EndAddr","FileName"),"SampledProfile":("ThreadID","PrgrmCtr","CPU"),"SampledProfileNmi":("ThreadID","PrgrmCtr","CPU"),"Interrupt":("ElapsedTime","CPU","ServiceAddr"),"DPC":("ElapsedTime","CPU","ServiceAddr"),"DPCTmr":("ElapsedTime","CPU","ServiceAddr"),"Mark":("Mark",)}.get(row[0],())
            if any(k not in decoded for k in required):raise ValueError(f"missing required xperf field at line {number}")
            yield decoded

def image_ranges(rows,begin,end):
    loads=[]
    for r in rows:
        kind=r["_type"]
        if kind not in ("I-Start","I-DCStart"):continue
        base=int(r["BaseAddr"],16);stop=int(r["EndAddr"],16);q=r["_raw"]["qpc"];name=r["FileName"]
        exits=[x["_raw"]["qpc"] for x in rows if x["_type"]=="I-End" and int(x["BaseAddr"],16)==base and x["_raw"]["qpc"]>=q]
        until=min(exits) if exits else None
        if kind=="I-DCStart":
            # Initial kernel rundown is an observation boundary, not a fabricated load time.
            ends=[x["_raw"]["qpc"] for x in rows if x["_type"]=="I-DCEnd" and x["BaseAddr"]==r["BaseAddr"] and x["EndAddr"]==r["EndAddr"] and x["FileName"].casefold()==name.casefold()]
            if not ends or not q<begin<end<max(ends):continue
        loads.append({"begin_qpc":q,"end_qpc":until,"base":base,"end_address":stop,"module":Path(name).name,"file":name,"begin_kind":kind,"line":r["_line"],"kernel":base>=1<<63})
    return loads

def match_module(loads,pc,q):
    hits=[x for x in loads if x["begin_qpc"]<=q and (x["end_qpc"] is None or q<x["end_qpc"]) and x["base"]<=pc<x["end_address"]]
    return hits[0]["module"] if len(hits)==1 else "unknown" if not hits else "ambiguous_module",[x["line"] for x in hits]

def verify_markers(scan,markers,identity):
    if scan["header_clock_type"]!=1 or scan["qpc_frequency"]!=10_000_000 or any(scan[k] for k in ("events_lost","buffers_lost","logfile_events_lost","process_trace_status","close_trace_status")):raise ValueError("marker raw scan clock/loss failure")
    verified=[]
    for phase,key in (("Begin","begin_marker_receipt"),("End","end_marker_receipt")):
        needle="LLMSim"+phase+"_"+identity["instance_name"]
        hits=[x for x in scan["matches"] if x["needle"]==needle and x["provider"]=="ce1dbfb4-137e-4da6-87b0-3f59aa102cbc" and x["opcode"]==34 and x["version"]==0 and bytes.fromhex(x["raw_hex"]).rstrip(b"\x00")==needle.encode("ascii")]
        if len(hits)!=1:raise ValueError("expected exactly one dedicated PerfInfo Mark event")
        hit=hits[0];csv_hits=[x for x in markers if x["Mark"]==needle and x["_raw"]["qpc"]==hit["qpc_raw"]]
        if len(csv_hits)!=1:raise ValueError("raw marker does not match xperf actual Mark schema")
        receipt=json.loads(Path(identity[key]).read_text());q=hit["qpc_raw"]
        if receipt["exit_code"]!=0 or receipt["qpc_frequency"]!=10_000_000 or not receipt["qpc_before"]<=q<=receipt["qpc_after"]:raise ValueError("marker outside independently recorded QPC command bracket")
        verified.append({"phase":phase,"qpc_raw":q,"provider":hit["provider"],"opcode":34,"version":0,"dedicated_payload_ascii_verified":True,"xperf_Mark_schema_verified":True,"qpc_before":receipt["qpc_before"],"qpc_after":receipt["qpc_after"]})
    return verified

def merge_ranges(ranges):
    merged=[]
    for a,b in sorted(ranges):
        if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
        else:merged.append((a,b))
    return merged


INTERRUPT_LAYOUTS = {50: ("Interrupt", 24), 67: ("Interrupt", 20), 68: ("DPC", 16), 69: ("DPCTmr", 16)}

def raw_interrupt_index(path, receipt):
    for key in ("events_lost", "buffers_lost", "logfile_events_lost", "process_trace_status", "close_trace_status"):
        if receipt[key] != 0: raise ValueError("raw interrupt reader loss/status failure")
    if receipt["clock_type"] != 1 or receipt["qpc_frequency"] != 10_000_000:
        raise ValueError("raw interrupt QPC clock mismatch")
    rows = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        layout = INTERRUPT_LAYOUTS.get(r["opcode"])
        if (r["provider"] != "ce1dbfb4-137e-4da6-87b0-3f59aa102cbc" or r["version"] != 2
                or layout is None or r["payload_length"] != layout[1] or not r["header_flags"] & 0x40):
            raise ValueError("unsupported actual interrupt descriptor/pointer width; no guessed layout")
        data = bytes.fromhex(r["raw_payload_hex"])
        if len(data) != r["payload_length"]: raise ValueError("raw interrupt payload length mismatch")
        initial = int.from_bytes(data[:8], "little", signed=True)
        routine = int.from_bytes(data[8:16], "little")
        if initial != r["initial_qpc_candidate"] or routine != r["routine_candidate"] or initial > r["end_qpc"]:
            raise ValueError("invalid initial/end QPC or routine")
        key = (r["end_qpc"], r["cpu"], layout[0])
        if key in rows: raise ValueError("duplicate interrupt key")
        rows[key] = r
    if len(rows) != receipt["selected_events"]: raise ValueError("raw interrupt record count mismatch")
    return rows

def qualify_interrupt(raw, decoded):
    initial = raw["initial_qpc_candidate"]; end = raw["end_qpc"]
    if abs((end-initial)/10 - int(decoded["ElapsedTime"])) > 0.500001:
        raise ValueError("raw InitialTimeQPC does not agree with independent xperf elapsed time")
    if raw["routine_candidate"] != int(decoded["ServiceAddr"], 16):
        raise ValueError("raw routine does not agree with xperf schema")
    return initial, end

def contains(unions, qpc):
    i = bisect.bisect_right(unions, (qpc, 1 << 63)) - 1
    return i >= 0 and unions[i][0] <= qpc < unions[i][1]

def overlap_ranges(segments, unions):
    result = []
    for s in segments:
        if s["state"] != "running": continue
        ranges = unions.get(s["cpu"], [])
        for a,b in ranges:
            if a >= s["end_qpc"]: break
            lo,hi = max(a,s["begin_qpc"]),min(b,s["end_qpc"])
            if lo < hi: result.append((lo,hi))
    return merge_ranges(result)

def overlap_amount(ranges, begin, end):
    total = 0
    i = max(0, bisect.bisect_right(ranges, (begin, 1 << 63)) - 1)
    while i < len(ranges):
        a,b = ranges[i]
        if a >= end: break
        if b > begin: total += max(0,min(end,b)-max(begin,a))
        i += 1
    return total

def run(args):
    analysis=json.loads(args.analysis.read_text());identity=json.loads(args.identity.read_text());start=analysis["sampling_begin_tick"];end=analysis["sampling_end_tick"];pid=analysis["pid"]
    own=re.compile(r"\(\s*"+str(pid)+r"\s*\)");kernel=re.compile(r"\(\s*0\s*\)")
    images=[r for r in parse(args.images) if r["_type"] in ("I-Start","I-End","I-DCStart","I-DCEnd") and (own.search(r.get("Process Name ( PID)","")) or kernel.search(r.get("Process Name ( PID)","")))]
    loads=image_ranges(images,start,end)
    raw_index=raw_interrupt_index(args.raw_interrupts,json.loads(args.raw_interrupt_receipt.read_text())); matched_raw=set()
    acts=[json.loads(x) for x in args.actions.read_text().splitlines()];begins=[a["begin_tick"] for a in acts];samples=[];hist=collections.Counter();groups=collections.Counter();peraction=collections.Counter();marks=[];interrupts=[];disagreements=[]
    for r in parse(args.perf):
        kind=r["_type"];q=r["_raw"]["qpc"] if r["_raw"] else None
        if kind=="Mark" and r.get("Mark","") in ("LLMSimBegin_"+identity["instance_name"],"LLMSimEnd_"+identity["instance_name"]):marks.append(r)
        if kind in ("Interrupt","DPC","DPCTmr") and q is not None:
            key=(q,int(r["CPU"]),kind)
            if key not in raw_index:
                if start<=q<end: raise ValueError("xperf interrupt missing in raw export")
                continue
            z=raw_index[key]; initial,finish=qualify_interrupt(z,r); matched_raw.add(key)
            module,lines=match_module(loads,z["routine_candidate"],q)
            interrupts.append({"kind":kind,"begin_qpc":initial,"end_qpc":finish,"elapsed_us_rounded":int(r["ElapsedTime"]),"cpu":int(r["CPU"]),"routine":hex(z["routine_candidate"]),"routine_module":module,"image_load_lines":lines});continue
        if q is None or not start<=q<end:continue
        if kind not in ("SampledProfile","SampledProfileNmi") or not own.search(r.get("Process Name ( PID)","")):continue
        tid=int(r["ThreadID"]);pc=int(r["PrgrmCtr"],16);module,lines=match_module(loads,pc,q);xperf_module=(r.get("Image!Function") or "").split("!")[0]
        if module not in ("unknown","ambiguous_module") and module.casefold()!=xperf_module.casefold():disagreements.append({"qpc":q,"tid":tid,"range_module":module,"xperf_module":xperf_module})
        i=bisect.bisect_right(begins,q)-1;a=acts[i] if i>=0 and acts[i]["begin_tick"]<=q<acts[i]["end_tick"] else None
        category=a["category"] if a else "between_actions";idx=a["action_index"] if a else None;weight=int(r.get("Count","1"))
        hist[(tid,module)]+=weight;groups[(tid,category,module)]+=weight;peraction[(tid,idx,module)]+=weight
        samples.append({"qpc":q,"tid":tid,"pc":hex(pc),"cpu":int(r["CPU"]),"module_from_load_lifetime":module,"xperf_image_function":r.get("Image!Function"),"weight":weight,"category":category,"action_index":idx,"image_load_lines":lines})
    if matched_raw != set(raw_index): raise ValueError("raw interrupt records not matched by xperf decoded events")
    verified=verify_markers(json.loads(args.marker_scan.read_text()),marks,identity)
    timelines=json.loads(args.timelines.read_text());main=next(t for t in timelines if t["thread_id"]==analysis["main_thread_id"])
    unions={cpu:merge_ranges([(max(start,r["begin_qpc"]),min(end,r["end_qpc"])) for r in interrupts if r["cpu"]==cpu]) for cpu in {r["cpu"] for r in interrupts}}
    main_interrupts=overlap_ranges(main["segments"],unions)
    overlap=sum(b-a for a,b in main_interrupts)
    sample_split=collections.Counter()
    for s in samples:
        inside=contains(unions.get(s["cpu"],[]),s["qpc"])
        s["inside_same_cpu_interrupt_interval"]=inside
        sample_split[(s["tid"],"inside_interrupt" if inside else "outside_interrupt",s["module_from_load_lifetime"])]+=s["weight"]
    action_overlap=collections.defaultdict(lambda: {"wall_ns":0,"assigned_running_ns":0,"interrupt_overlap_ns":0,"assigned_running_outside_interrupt_ns":0})
    for a in acts:
        n=overlap_amount(main_interrupts,a["begin_tick"],a["end_tick"])*100
        g=action_overlap[str(a["action_index"])]; g["wall_ns"]+=a["elapsed_ns"];g["assigned_running_ns"]+=a["main_state_ns"].get("running",0);g["interrupt_overlap_ns"]+=n;g["assigned_running_outside_interrupt_ns"]+=a["main_state_ns"].get("running",0)-n
    auxiliary_overlap=[]
    for t in timelines:
        if t["thread_id"]==analysis["main_thread_id"]:continue
        spans=overlap_ranges(t["segments"],unions);value=sum(b-a for a,b in spans)*100
        assigned=sum(s["end_qpc"]-s["begin_qpc"] for s in t["segments"] if s["state"]=="running")*100
        auxiliary_overlap.append({"tid":t["thread_id"],"assigned_running_ns":assigned,"same_cpu_interrupt_overlap_ns":value,"assigned_running_outside_interrupt_ns":assigned-value})
    report={"schema":"owned-etw-module-samples-2","sample_count":len(samples),"weighted_sample_count":sum(s["weight"] for s in samples),"image_events":len(images),"qualified_image_ranges":len(loads),"raw_marker_clock_verified":True,"markers":verified,"histogram":[{"tid":t,"module":m,"samples":n} for (t,m),n in hist.most_common()],"by_category":[{"tid":t,"category":c,"module":m,"samples":n} for (t,c,m),n in groups.most_common()],"by_action":[{"tid":t,"action_index":i,"module":m,"samples":n} for (t,i,m),n in peraction.most_common()],"module_range_xperf_disagreements":disagreements,"DPC_ISR":{"event_count_all_cpus_in_window":len(interrupts),"main_running_overlap_ns":overlap*100,"main_running_overlap_fraction":overlap*100/analysis["main_state_ns"]["running"],"duration_source":"raw InitialTimeQPC and raw event end QPC; every descriptor, routine and duration qualified against xperf; union per CPU; nested ISR/DPC not summed twice","subtracted_from_running":False,"precision_limitation":"100ns QPC ticks; this is diagnostic observed interruption overlap, not a cost table or a fitted CPU-service price."},"limitations":["Running is scheduler occupancy, not useful compute.","Instruction-pointer samples identify sampled module location; no loop/function or useful-work cause is inferred.","Sample counts are statistical counts, not exact time or additive overhead estimates.","Kernel module initial/end rundowns bound observed residency; unloaded/reloaded ambiguity remains unknown.","Owned user modules require a unique address/time image load match; unresolved PCs remain unknown.","Auxiliary CPU activity overlaps main wall time and must not be added to it.","No performance parameters are fitted and no CUDA kernel durations are inferred."]}
    report["by_action_interrupt_overlap"]=dict(action_overlap)
    report["auxiliary_interrupt_overlap"]=auxiliary_overlap
    report["IP_samples_split_by_interrupt"]=[{"tid":t,"scope":scope,"module":m,"samples":n} for (t,scope,m),n in sample_split.most_common()]
    report["raw_interrupt_xperf_matched_count"]=len(matched_raw)
    args.out.mkdir(exist_ok=False)
    for name,value in (("module-analysis.json",report),("qualified-image-ranges.json",loads),("image-events.json",images),("interrupt-events.json",interrupts)):(args.out/name).write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8",newline="\n")
    (args.out/"owned-pc-samples.jsonl").write_text("".join(json.dumps(x)+"\n" for x in samples),encoding="utf-8",newline="\n")
    evidence={"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"inputs":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.analysis,args.identity,args.images,args.perf,args.actions,args.marker_scan,args.timelines,args.raw_interrupts,args.raw_interrupt_receipt)}}
    (args.out/"evidence.json").write_text(json.dumps(evidence,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({k:report[k] for k in ("raw_marker_clock_verified","sample_count","histogram","DPC_ISR")}))

def main():
    ap=argparse.ArgumentParser()
    for name in ("analysis","identity","images","perf","actions","marker-scan","timelines","raw-interrupts","raw-interrupt-receipt","out"):ap.add_argument("--"+name,type=Path,required=True)
    run(ap.parse_args())
if __name__=="__main__":main()
