"""Synthetic scheduler/lifetime/interruption guards; never records or invokes native code."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from profiling.state_response import etw_analysis as A
from profiling.state_response import etw_samples as S

def event(kind,q,line,**values):
    return dict(_type=kind,_raw=dict(qpc=q,cpu=0,relative_ns=q*100),_line=line,**values)

def start(tid=42):return event("T-Start",0,1,ThreadID=str(tid),**{"Image!Function":"test.dll!0x1"})
def switch(q,kind,state="Ready",cpu=0,line=None):
    return event("CSwitch",q,line or q,**{"Old TID":"42" if kind=="out" else "other","New TID":"42" if kind=="in" else "other","OldState":state,"Wait Reason":"UserRequest","CPU":str(cpu)})
def ready(q):return event("ReadyThread",q,q,**{"Rdy TID":"42"})

class IdentityTests(unittest.TestCase):
    def test_process_ready_result_identity_is_semantically_bound(self):
        clock=dict(clock_id="windows_qpc",frequency_hz=10000000)
        actor=dict(process_id=7,main_thread_id=42,launch_tick=1,ready_tick=5,ready_qpc_frequency=10000000,sampling_end_tick=20)
        identity=dict(actors=[actor]);ready=dict(process_id=7,main_thread_id=42,ready_tick=5,clock=clock,program_sha256="a")
        process=dict(pid=7,launch_tick=1,ready=ready)
        result=dict(process_id=7,main_thread_id=42,ready_tick=5,clock=clock,program_sha256="a",sampling_end_tick=20)
        A.validate_native_identity(identity,result,process)
        for key,value in (("pid",8),("launch_tick",2)):
            bad=deepcopy(process);bad[key]=value
            with self.assertRaises(ValueError):A.validate_native_identity(identity,result,bad)
        for key,value in (("process_id",8),("main_thread_id",43),("ready_tick",6),("program_sha256","b")):
            bad=deepcopy(result);bad[key]=value
            with self.assertRaises(ValueError):A.validate_native_identity(identity,bad,process)
    def test_csv_cpu_raw_cpu_disagreement_rejected(self):
        text="CSwitch, TimeStamp, New Process Name ( PID), New TID, Old Process Name ( PID), Old TID, OldState, Wait Reason, CPU\n# 0 # 0x1 # 100 # 0x0\nCSwitch, 1, owned.exe (7), 42, other.exe (8), 43, Ready, Executive, 1\n"
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"dump.csv";path.write_text(text)
            with self.assertRaises(ValueError):A.dump_filtered(path,7,"own")
    def test_only_exact_instance_markers_retained(self):
        text="Mark, TimeStamp, Mark\n# 0 # 0x1 # 100 # 0x0\nMark, 1, unrelated_marker\n# 0 # 0x2 # 200 # 0x0\nMark, 2, LLMSimBegin_own\n"
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"dump.csv";path.write_text(text)
            rows,meta=A.dump_filtered(path,7,"own")
            self.assertEqual(rows,[]);self.assertEqual([x["Mark"] for x in meta["marker_rows"]],["LLMSimBegin_own"])

class SchedulerTests(unittest.TestCase):
    def test_running_wait_ready_partition(self):
        rows=[start(),ready(1),switch(2,"in"),switch(5,"out","Waiting"),ready(9),switch(12,"in"),switch(20,"out")]
        r=A.reconstruct(rows,42,2,18)
        self.assertEqual(A.intersect(r["segments"],2,18),dict(running=9,waiting=4,ready=3))
    def test_duplicate_ready_does_not_reset_queue(self):
        rows=[start(),ready(1),ready(4),switch(9,"in"),switch(12,"out")]
        r=A.reconstruct(rows,42,1,10)
        self.assertEqual(A.intersect(r["segments"],1,10),dict(ready=8,running=1))
    def test_wait_without_ready_stays_unknown(self):
        rows=[start(),switch(1,"in"),switch(3,"out","Waiting"),switch(8,"in"),switch(12,"out")]
        r=A.reconstruct(rows,42,1,10)
        self.assertEqual(A.intersect(r["segments"],1,10),dict(running=4,unknown_wait_or_ready=5))
    def test_cpu_mismatch_does_not_claim_execution(self):
        r=A.reconstruct([start(),switch(1,"in",cpu=0),switch(8,"out",cpu=2)],42,1,7)
        self.assertEqual(A.intersect(r["segments"],1,7),dict(unknown=6))
    def test_right_boundary_without_transition_is_censored(self):
        r=A.reconstruct([start(),switch(1,"in")],42,1,10)
        self.assertEqual(A.intersect(r["segments"],1,10),dict(unknown_censored=9))
    def test_observed_transition_beyond_window_bounds_running(self):
        r=A.reconstruct([start(),switch(1,"in"),switch(20,"out")],42,2,10)
        self.assertEqual(A.intersect(r["segments"],2,10),dict(running=8))
    def test_reused_tid_is_rejected(self):
        with self.assertRaises(ValueError):A.reconstruct([start(),start()],42,2,10)
    def test_query_before_creation_rejected(self):
        s=start();s["_raw"]["qpc"]=5
        with self.assertRaises(ValueError):A.reconstruct([s],42,2,10)
    def test_missing_coverage_rejected(self):
        with self.assertRaises(ValueError):A.intersect([],1,4)
    def test_half_open_intersection(self):
        self.assertEqual(A.intersect([dict(begin_qpc=1,end_qpc=3,state="ready"),dict(begin_qpc=3,end_qpc=5,state="running")],3,5),dict(running=2))

class InterruptTests(unittest.TestCase):
    def test_nested_union_not_double_counted(self):
        self.assertEqual(S.merge_ranges([(1,8),(2,3),(6,10),(12,15)]),[(1,10),(12,15)])
    def test_interruption_requires_same_cpu(self):
        seg=[dict(begin_qpc=0,end_qpc=10,state="running",cpu=0)]
        self.assertEqual(S.overlap_ranges(seg,{1:[(1,9)]}),[])
        self.assertEqual(S.overlap_ranges(seg,{0:[(1,3),(8,12)]}),[(1,3),(8,10)])
    def test_sample_membership_half_open(self):
        self.assertTrue(S.contains([(1,4)],1));self.assertFalse(S.contains([(1,4)],4))
    def test_exact_initial_qpc_qualifies_rounded_elapsed(self):
        self.assertEqual(S.qualify_interrupt(dict(initial_qpc_candidate=100,end_qpc=115,routine_candidate=12),dict(ElapsedTime="2",ServiceAddr="0xc")),(100,115))
    def test_bad_elapsed_or_routine_refused(self):
        r=dict(initial_qpc_candidate=100,end_qpc=120,routine_candidate=12)
        for d in (dict(ElapsedTime="4",ServiceAddr="0xc"),dict(ElapsedTime="2",ServiceAddr="0xf")):
            with self.assertRaises(ValueError):S.qualify_interrupt(r,d)
    def test_raw_descriptor_clock_and_duplicates(self):
        row=dict(provider="ce1dbfb4-137e-4da6-87b0-3f59aa102cbc",opcode=68,version=2,payload_length=16,header_flags=848,end_qpc=120,initial_qpc_candidate=100,routine_candidate=12,cpu=0,raw_payload_hex=(100).to_bytes(8,"little").hex()+(12).to_bytes(8,"little").hex())
        receipt=dict(events_lost=0,buffers_lost=0,logfile_events_lost=0,process_trace_status=0,close_trace_status=0,clock_type=1,qpc_frequency=10000000,selected_events=1)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"raw.jsonl";path.write_text(json.dumps(row)+"\n")
            self.assertEqual(len(S.raw_interrupt_index(path,receipt)),1)
            bad=deepcopy(row);bad["version"]=9;path.write_text(json.dumps(bad)+"\n")
            with self.assertRaises(ValueError):S.raw_interrupt_index(path,receipt)
            path.write_text(json.dumps(row)+"\n"+json.dumps(row)+"\n")
            with self.assertRaises(ValueError):S.raw_interrupt_index(path,receipt)
            bad=deepcopy(receipt);bad["clock_type"]=2
            with self.assertRaises(ValueError):S.raw_interrupt_index(path,bad)
    def test_module_lifetime_and_ambiguity(self):
        loads=[dict(begin_qpc=1,end_qpc=8,base=100,end_address=200,module="a.dll",line=1)]
        self.assertEqual(S.match_module(loads,120,3),("a.dll",[1]))
        self.assertEqual(S.match_module(loads,120,8),("unknown",[]))
        self.assertEqual(S.match_module(loads*2,120,3)[0],"ambiguous_module")
    def test_marker_commandline_is_not_emitted_marker(self):
        identity=dict(instance_name="owned");scan=dict(header_clock_type=1,qpc_frequency=10000000,events_lost=0,buffers_lost=0,logfile_events_lost=0,process_trace_status=0,close_trace_status=0,matches=[]);markers=[]
        with tempfile.TemporaryDirectory() as d:
            for phase,q,key in (("Begin",100,"begin_marker_receipt"),("End",200,"end_marker_receipt")):
                needle="LLMSim"+phase+"_owned";path=Path(d)/(phase+".json");path.write_text(json.dumps(dict(exit_code=0,qpc_frequency=10000000,qpc_before=q-1,qpc_after=q+1)));identity[key]=str(path)
                scan["matches"].append(dict(needle=needle,provider="ce1dbfb4-137e-4da6-87b0-3f59aa102cbc",opcode=34,version=0,raw_hex=(needle.encode()+b"\x00").hex(),qpc_raw=q))
                markers.append(dict(Mark=needle,_raw=dict(qpc=q)))
            self.assertEqual(len(S.verify_markers(scan,markers,identity)),2)
            bad=deepcopy(scan);bad["matches"][0]["provider"]="3d6fa8d0-fe05-11d0-9dda-00c04fd7ba7c"
            with self.assertRaises(ValueError):S.verify_markers(bad,markers,identity)
            bad=deepcopy(scan);bad["matches"][0]["qpc_raw"]=1010
            with self.assertRaises(ValueError):S.verify_markers(bad,markers,identity)
    def test_kernel_rundown_requires_both_observation_bounds(self):
        a=event("I-DCStart",1,1,BaseAddr="0xffff000000000000",EndAddr="0xffff000000001000",FileName="k.sys")
        self.assertEqual(S.image_ranges([a],2,8),[])
        b=deepcopy(a);b["_type"]="I-DCEnd";b["_raw"]["qpc"]=10
        self.assertEqual(len(S.image_ranges([a,b],2,8)),1)

if __name__=="__main__":unittest.main()
