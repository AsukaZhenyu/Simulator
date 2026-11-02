// Optional owned-process thread snapshots. Never a timing predictor or scheduler.
// .NET Framework compiler: csc /optimize+ /platform:x64 /r:System.Web.Extensions.dll.
using System;
using System.Collections.Generic;
using System.Collections;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Web.Script.Serialization;

internal static class ThreadSampler {
    const uint QueryLimited = 0x0800;
    const uint ProcessQueryLimited = 0x1000;
    const uint TimerModify = 0x0002, Synchronize = 0x00100000;
    const uint WaitObject0 = 0, WaitTimeout = 258;
    const int MaxSamples = 20000, MaxThreads = 128;
    static readonly JavaScriptSerializer Json = new JavaScriptSerializer { MaxJsonLength = Int32.MaxValue };
    [StructLayout(LayoutKind.Sequential)] struct FileTime { public uint Low, High; public long Value { get { return ((long)High << 32) | Low; } } }
    [DllImport("kernel32.dll", SetLastError=true)] static extern IntPtr OpenProcess(uint rights, bool inherit, uint pid);
    [DllImport("kernel32.dll", SetLastError=true)] static extern uint GetProcessId(IntPtr handle);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool GetProcessTimes(IntPtr handle, out FileTime created, out FileTime exited, out FileTime kernel, out FileTime user);
    [DllImport("kernel32.dll", SetLastError=true)] static extern IntPtr OpenThread(uint rights, bool inherit, uint tid);
    [DllImport("kernel32.dll", SetLastError=true)] static extern uint GetProcessIdOfThread(IntPtr handle);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool GetThreadTimes(IntPtr handle, out FileTime created, out FileTime exited, out FileTime kernel, out FileTime user);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool QueryThreadCycleTime(IntPtr handle, out ulong cycles);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool CloseHandle(IntPtr handle);
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)] static extern IntPtr CreateWaitableTimerEx(IntPtr attrs, string name, uint flags, uint access);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool SetWaitableTimer(IntPtr timer, ref long due, int period, IntPtr completion, IntPtr arg, bool resume);
    [DllImport("kernel32.dll", SetLastError=true)] static extern uint WaitForSingleObject(IntPtr handle, uint ms);
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)] static extern IntPtr GetModuleHandle(string name);
    [DllImport("kernel32.dll", CharSet=CharSet.Ansi, SetLastError=true)] static extern IntPtr GetProcAddress(IntPtr module,string name);
    [UnmanagedFunctionPointer(CallingConvention.Winapi)] delegate int NtQuery(uint kind,IntPtr buffer,uint capacity,out uint length);
    [StructLayout(LayoutKind.Sequential)] struct UnicodeString { public ushort Length,MaximumLength;public IntPtr Buffer; }
    // Public Windows SDK winternl.h declarations; reserved members remain opaque.
    [StructLayout(LayoutKind.Sequential)] struct ProcessInfo {
        public uint NextEntryOffset,NumberOfThreads;
        [MarshalAs(UnmanagedType.ByValArray,SizeConst=48)] public byte[] Reserved1;
        public UnicodeString ImageName; public int BasePriority; public IntPtr UniqueProcessId,Reserved2;
        public uint HandleCount,SessionId;public IntPtr Reserved3;
        public UIntPtr PeakVirtualSize,VirtualSize;public uint Reserved4;
        public UIntPtr PeakWorkingSetSize,WorkingSetSize;public IntPtr Reserved5;
        public UIntPtr QuotaPagedPoolUsage;public IntPtr Reserved6;
        public UIntPtr QuotaNonPagedPoolUsage,PagefileUsage,PeakPagefileUsage,PrivatePageCount;
        [MarshalAs(UnmanagedType.ByValArray,SizeConst=6)] public long[] Reserved7;
    }
    [StructLayout(LayoutKind.Sequential)] struct ClientId { public IntPtr Process,Thread; }
    [StructLayout(LayoutKind.Sequential)] struct ThreadInfo {
        [MarshalAs(UnmanagedType.ByValArray,SizeConst=3)] public long[] Reserved1;
        public uint Reserved2;public IntPtr StartAddress;public ClientId ClientId;
        public int Priority,BasePriority;public uint Reserved3,ThreadState,WaitReason;
    }

    static long Tick() { return Stopwatch.GetTimestamp(); }
    static Dictionary<string,object> D(params object[] pairs) {
        var result = new Dictionary<string,object>();
        for (int i=0; i<pairs.Length; i+=2) result.Add((string)pairs[i], pairs[i+1]);
        return result;
    }
    static void Require(bool okay, string message) { if (!okay) throw new InvalidOperationException(message); }
    static string Failure(string api) { return api + " failed, win32=" + Marshal.GetLastWin32Error(); }
    static long Number(Dictionary<string,object> data, string key, long low, long high) {
        Require(data.ContainsKey(key) && !(data[key] is bool), "missing integer " + key);
        long value = Convert.ToInt64(data[key]); Require(low <= value && value <= high, key + " out of range"); return value;
    }
    static void SaveNew(string path, object value) {
        byte[] bytes = new UTF8Encoding(false).GetBytes(Json.Serialize(value) + "\n");
        using(var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.Read)) stream.Write(bytes,0,bytes.Length);
    }
    static object Clock() { return D("clock_id","windows_qpc","frequency_hz",Stopwatch.Frequency); }
    static void LayoutGuard() {
        Require(IntPtr.Size==8 && Marshal.SizeOf(typeof(ProcessInfo))==256 && Marshal.SizeOf(typeof(ThreadInfo))==80,
            "unsupported documented NT snapshot layout size; x64 SDK layout required");
        Require(Marshal.OffsetOf(typeof(ProcessInfo),"UniqueProcessId").ToInt32()==80 &&
            Marshal.OffsetOf(typeof(ProcessInfo),"NumberOfThreads").ToInt32()==4 &&
            Marshal.OffsetOf(typeof(ThreadInfo),"ClientId").ToInt32()==40 &&
            Marshal.OffsetOf(typeof(ThreadInfo),"ThreadState").ToInt32()==68 &&
            Marshal.OffsetOf(typeof(ThreadInfo),"WaitReason").ToInt32()==72,
            "unsupported documented NT snapshot field offsets");
    }
    static string StateName(uint state) {
        string[] names={"Initialized","Ready","Running","Standby","Terminated","Wait","Transition"};
        return state<names.Length?names[state]:"OtherOrUnknown";
    }
    sealed class NtSnapshot : IDisposable {
        public long Begin,End; public readonly List<object> Attempts=new List<object>();
        public Dictionary<int,List<Dictionary<string,object>>> Rows;
        IntPtr buffer;int capacity=4*1024*1024;NtQuery query;
        public NtSnapshot() {
            LayoutGuard();IntPtr module=GetModuleHandle("ntdll.dll");Require(module!=IntPtr.Zero,Failure("GetModuleHandle ntdll"));
            IntPtr function=GetProcAddress(module,"NtQuerySystemInformation");Require(function!=IntPtr.Zero,"NtQuerySystemInformation unavailable");
            query=(NtQuery)Marshal.GetDelegateForFunctionPointer(function,typeof(NtQuery));
            buffer=Marshal.AllocHGlobal(capacity);
        }
        public void Capture(HashSet<int> owned) {
            Attempts.Clear();uint length=0;bool okay=false;
            for(int attempt=0;attempt<4;++attempt) {
                Begin=Tick();int status=query(5,buffer,(uint)capacity,out length);End=Tick();
                Attempts.Add(D("query_begin_tick",Begin,"query_end_tick",End,"ntstatus",status,
                    "capacity_bytes",capacity,"returned_length_bytes",length));
                if(status==0){okay=true;break;}
                Require(status==unchecked((int)0xC0000004),"NtQuerySystemInformation failure, NTSTATUS="+status);
                long desired=Math.Max((long)capacity*2,(long)length+65536);
                Require(desired<=32*1024*1024,"NT snapshot buffer exceeds 32 MiB bound");
                Marshal.FreeHGlobal(buffer);buffer=IntPtr.Zero;capacity=(int)desired;buffer=Marshal.AllocHGlobal(capacity);
            }
            Require(okay,"NT snapshot resize retry bound reached");
            Require(length>=256 && length<=capacity,"NT snapshot returned length invalid");
            Rows=Parse(buffer,(int)length,owned);
        }
        public static Dictionary<int,List<Dictionary<string,object>>> Parse(IntPtr source,int length,HashSet<int> owned) {
            LayoutGuard();Require(length>=256,"NT snapshot shorter than process header");
            var result=new Dictionary<int,List<Dictionary<string,object>>>();int offset=0,entries=0;
            while(true) {
                Require(++entries<=100000,"NT process inventory bound exceeded");
                Require(offset>=0 && offset<=length-256,"NT process header outside returned buffer");
                IntPtr entry=IntPtr.Add(source,offset);uint next=(uint)Marshal.ReadInt32(entry,0);
                uint count=(uint)Marshal.ReadInt32(entry,4);long rawPid=Marshal.ReadInt64(entry,80);
                int span;
                if(next==0)span=length-offset;
                else {
                    Require(next>=256 && next%8==0 && next<=length-offset,"NT NextEntryOffset invalid");span=(int)next;
                }
                Require(count<=100000 && 256L+count*80L<=span,"NT thread inventory exceeds its process span");
                if(rawPid>0 && rawPid<=Int32.MaxValue && owned.Contains((int)rawPid)) {
                    int pid=(int)rawPid;Require(!result.ContainsKey(pid),"duplicate owned PID in NT snapshot");
                    Require(count<=MaxThreads,"owned thread snapshot exceeds 128-thread bound");
                    var threads=new List<Dictionary<string,object>>();var tids=new HashSet<long>();
                    for(int i=0;i<count;++i) {
                        IntPtr thread=IntPtr.Add(entry,256+i*80);long owner=Marshal.ReadInt64(thread,40),tid=Marshal.ReadInt64(thread,48);
                        Require(owner==pid && tid>0 && tid<=Int32.MaxValue && tids.Add(tid),"NT owned thread identity invalid");
                        uint state=(uint)Marshal.ReadInt32(thread,68),wait=(uint)Marshal.ReadInt32(thread,72);
                        threads.Add(D("thread_id",(int)tid,"state_code",state,"state_name",StateName(state),
                            "native_wait_reason_code",state==5?(object)wait:null,"wait_reason_code",null,"wait_reason_name",null,
                            "wait_reason_scope","raw native wait code; never cast to managed ThreadWaitReason",
                            "thread_state_provider","NtQuerySystemInformation:SystemProcessInformation"));
                    }
                    result.Add(pid,threads);
                }
                if(next==0)break;
                offset=checked(offset+(int)next);
            }
            return result;
        }
        public void Dispose(){if(buffer!=IntPtr.Zero){Marshal.FreeHGlobal(buffer);buffer=IntPtr.Zero;}}
    }
    sealed class Waiter : IDisposable {
        IntPtr timer;
        public readonly string Provider;
        public Waiter() {
            timer=CreateWaitableTimerEx(IntPtr.Zero,null,2,TimerModify|Synchronize);
            Provider=timer!=IntPtr.Zero ? "Windows_private_high_resolution_waitable_timer" : "Thread.Sleep";
        }
        public void Until(long deadline) {
            long remaining=deadline-Tick(); if(remaining<=0)return;
            if(timer==IntPtr.Zero) { Thread.Sleep(1); return; }
            long due=-Math.Max(1,(long)Math.Ceiling(remaining*10000000.0/Stopwatch.Frequency));
            Require(SetWaitableTimer(timer,ref due,0,IntPtr.Zero,IntPtr.Zero,false), Failure("SetWaitableTimer"));
            uint status=WaitForSingleObject(timer,100);
            Require(status==WaitObject0 || status==WaitTimeout,Failure("WaitForSingleObject"));
        }
        public void Dispose(){ if(timer!=IntPtr.Zero){CloseHandle(timer);timer=IntPtr.Zero;} }
    }
    sealed class Actor : IDisposable {
        public string Name; public int Pid; public long Created; public Process Process; public IntPtr Handle;
        public Actor(Dictionary<string,object> value) {
            Name=(string)value["actor_id"]; Require(!String.IsNullOrWhiteSpace(Name),"actor id required");
            Pid=(int)Number(value,"process_id",1,Int32.MaxValue);
            Created=Number(value,"creation_time_100ns",1,Int64.MaxValue);
            Handle=OpenProcess(ProcessQueryLimited,false,(uint)Pid); Require(Handle!=IntPtr.Zero,Failure("OpenProcess owned PID"));
            try { Verify(); Process=Process.GetProcessById(Pid); }
            catch { Dispose(); throw; }
        }
        public void Verify() {
            FileTime c,e,k,u; Require(GetProcessId(Handle)==Pid,"owned PID handle identity changed");
            Require(GetProcessTimes(Handle,out c,out e,out k,out u),Failure("GetProcessTimes"));
            Require(c.Value==Created,"owned process creation time differs");
        }
        public object Snapshot() {
            long begin=Tick(); var threads=new List<object>(); var unknown=new List<string>();
            long stateBegin=Tick(), stateEnd=stateBegin;
            try {
                Verify();
                stateBegin=Tick(); Process.Refresh(); ProcessThreadCollection snapshot=Process.Threads; stateEnd=Tick();
                if(snapshot.Count>MaxThreads) throw new InvalidOperationException("owned thread snapshot exceeds 128-thread bound");
                foreach(ProcessThread thread in snapshot) {
                    long entryBegin=Tick(); var item=D("thread_id",thread.Id,"query_begin_tick",entryBegin,
                        "state_code",null,"state_name",null,"wait_reason_code",null,"wait_reason_name",null,
                        "thread_creation_time_100ns",null,"kernel_time_100ns",null,"user_time_100ns",null,
                        "cycle_total",null,"cycle_query_begin_tick",null,"cycle_query_end_tick",null,
                        "handle_owner_pid",null,"state_snapshot_begin_tick",stateBegin,"state_snapshot_end_tick",stateEnd,
                        "state_scope","refreshed OS snapshot; not a continuous interval or exact property-read instant",
                        "state_and_cycle_queries_are_atomic",false);
                    var missing=new List<string>();
                    try {
                        var state=thread.ThreadState; item["state_code"]=(int)state; item["state_name"]=state.ToString();
                        if(state==System.Diagnostics.ThreadState.Wait) {
                            try { var reason=thread.WaitReason; item["wait_reason_code"]=(int)reason;item["wait_reason_name"]=reason.ToString(); }
                            catch(Exception error) { missing.Add("wait reason: "+error.GetType().Name+": "+error.Message); }
                        }
                    } catch(Exception error) { missing.Add("state: "+error.GetType().Name+": "+error.Message); }
                    IntPtr handle=OpenThread(QueryLimited,false,(uint)thread.Id);
                    if(handle==IntPtr.Zero) missing.Add(Failure("OpenThread owned thread"));
                    else try {
                        uint owner=GetProcessIdOfThread(handle); item["handle_owner_pid"]=owner;
                        if(owner!=Pid) throw new InvalidOperationException("thread handle no longer belongs to owned PID");
                        FileTime c,e,k,u;
                        if(GetThreadTimes(handle,out c,out e,out k,out u)) {
                            item["thread_creation_time_100ns"]=c.Value;item["kernel_time_100ns"]=k.Value;item["user_time_100ns"]=u.Value;
                        } else missing.Add(Failure("GetThreadTimes"));
                        item["cycle_query_begin_tick"]=Tick(); ulong cycles;
                        if(QueryThreadCycleTime(handle,out cycles)) item["cycle_total"]=cycles;
                        else missing.Add(Failure("QueryThreadCycleTime"));
                        item["cycle_query_end_tick"]=Tick();
                    } catch(Exception error) { missing.Add(error.GetType().Name+": "+error.Message); }
                    finally { CloseHandle(handle); }
                    item["query_end_tick"]=Tick();item["unknown"]=missing;threads.Add(item);
                    thread.Dispose();
                }
                Verify();
            } catch(Exception error) { if(error.Message.Contains("128-thread bound"))throw;unknown.Add(error.GetType().Name+": "+error.Message); }
            return D("actor_id",Name,"process_id",Pid,"creation_time_100ns",Created,
                "query_begin_tick",begin,"query_end_tick",Tick(),"state_snapshot_begin_tick",stateBegin,
                "state_snapshot_end_tick",stateEnd,"threads",threads,"unknown",unknown,
                "status",unknown.Count==0?"observed":"partial_or_unavailable",
                "primary_thread_inferred_from_collection_order",false);
        }
        public object SnapshotNt(NtSnapshot snapshot) {
            var rows=new List<object>();var unknown=new List<string>();long counterBegin=Tick();
            try {
                Verify();List<Dictionary<string,object>> states;
                Require(snapshot.Rows.TryGetValue(Pid,out states),"owned PID absent from NT snapshot");
                foreach(var state in states) {
                    var item=new Dictionary<string,object>(state);int tid=(int)item["thread_id"];
                    item["query_begin_tick"]=Tick();item["state_snapshot_begin_tick"]=snapshot.Begin;item["state_snapshot_end_tick"]=snapshot.End;
                    item["state_scope"]="one OS snapshot during shared NT query bracket; not a continuous interval";
                    item["state_and_cycle_queries_are_atomic"]=false;
                    item["cycle_total"]=null;item["thread_creation_time_100ns"]=null;
                    item["kernel_time_100ns"]=null;item["user_time_100ns"]=null;
                    item["cycle_query_begin_tick"]=null;item["cycle_query_end_tick"]=null;item["handle_owner_pid"]=null;
                    var missing=new List<string>();IntPtr handle=OpenThread(QueryLimited,false,(uint)tid);
                    if(handle==IntPtr.Zero)missing.Add(Failure("OpenThread owned thread"));
                    else try {
                        uint owner=GetProcessIdOfThread(handle);item["handle_owner_pid"]=owner;
                        Require(owner==Pid,"thread handle no longer belongs to owned PID");
                        FileTime c,e,k,u;
                        if(GetThreadTimes(handle,out c,out e,out k,out u)) {
                            item["thread_creation_time_100ns"]=c.Value;item["kernel_time_100ns"]=k.Value;item["user_time_100ns"]=u.Value;
                        } else missing.Add(Failure("GetThreadTimes"));
                        item["cycle_query_begin_tick"]=Tick();ulong cycles;
                        if(QueryThreadCycleTime(handle,out cycles))item["cycle_total"]=cycles;
                        else missing.Add(Failure("QueryThreadCycleTime"));
                        item["cycle_query_end_tick"]=Tick();
                    } catch(Exception failure){missing.Add(failure.GetType().Name+": "+failure.Message);}
                    finally{CloseHandle(handle);}
                    item["query_end_tick"]=Tick();item["unknown"]=missing;rows.Add(item);
                }
                Verify();
            } catch(Exception failure){unknown.Add(failure.GetType().Name+": "+failure.Message);}
            return D("actor_id",Name,"process_id",Pid,"creation_time_100ns",Created,
                "query_begin_tick",snapshot.Begin,"query_end_tick",Tick(),
                "owned_counter_query_begin_tick",counterBegin,"state_snapshot_begin_tick",snapshot.Begin,
                "state_snapshot_end_tick",snapshot.End,"threads",rows,"unknown",unknown,
                "status",unknown.Count==0?"observed":"partial_or_unavailable",
                "primary_thread_inferred_from_collection_order",false);
        }
        public void Dispose(){ if(Process!=null){Process.Dispose();Process=null;} if(Handle!=IntPtr.Zero){CloseHandle(Handle);Handle=IntPtr.Zero;} }
    }
    static int Main(string[] args) {
        if(args.Length==3 && args[0]=="--parse-fixture") {
            try {
                var input=Json.Deserialize<Dictionary<string,object>>(File.ReadAllText(args[1],Encoding.UTF8));
                byte[] bytes=Convert.FromBase64String((string)input["buffer_base64"]);var pids=new HashSet<int>();
                foreach(object pid in (IEnumerable)input["owned_pids"])pids.Add(Convert.ToInt32(pid));
                IntPtr buffer=Marshal.AllocHGlobal(bytes.Length);
                try {
                    Marshal.Copy(bytes,0,buffer,bytes.Length);var parsed=NtSnapshot.Parse(buffer,bytes.Length,pids);var copied=new List<object>();
                    foreach(var pair in parsed)copied.Add(D("process_id",pair.Key,"threads",pair.Value));
                    SaveNew(args[2],D("parsed",copied,"native_queries_performed",false));
                }
                finally{Marshal.FreeHGlobal(buffer);}return 0;
            }catch(Exception failure){Console.Error.WriteLine(failure.GetType().Name+": "+failure.Message);return 2;}
        }
        var actors=new List<Actor>(); var records=new List<object>(); object output=null; int result=0;
        long begin=0,end=0,period=0,missed=0; string reason="not_started",error=null,provider=null;
        long requestedMask=0,observedMask=0; int interval=0,maximum=0;string stateProvider=null;
        try {
            Require(args.Length==4,"usage: thread-sampler CONFIG READY STOP OUTPUT");
            Require(Stopwatch.IsHighResolution,"QPC stopwatch provider required");
            foreach(string path in new[]{args[1],args[2],args[3]}) Require(!File.Exists(path),"handshake/output already exists");
            var config=Json.Deserialize<Dictionary<string,object>>(File.ReadAllText(args[0],Encoding.UTF8));
            Require((string)config["schema"]=="owned-thread-snapshot-config-1","unknown config schema");
            interval=(int)Number(config,"interval_ms",1,1000); maximum=(int)Number(config,"max_duration_ms",1,15000);
            stateProvider=(string)config["state_provider"];
            Require(stateProvider=="nt_system_process_information_x64" || stateProvider=="dotnet_process_threads", "unknown explicit thread state provider");
            requestedMask=Number(config,"affinity_mask",1,Int64.MaxValue);
            using(var self=Process.GetCurrentProcess()) { self.ProcessorAffinity=new IntPtr(requestedMask); observedMask=self.ProcessorAffinity.ToInt64(); }
            Require(observedMask==requestedMask,"sampler affinity observation differs");
            var seenNames=new HashSet<string>();var seenPids=new HashSet<int>();
            var values=(IEnumerable)config["actors"];
            foreach(var raw in values) { Require(actors.Count<8,"one to eight owned actors required");var actor=new Actor((Dictionary<string,object>)raw); actors.Add(actor);Require(seenNames.Add(actor.Name)&&seenPids.Add(actor.Pid),"duplicate owned actor"); }
            Require(actors.Count>0,"one to eight owned actors required");
            period=Math.Max(1,(long)Math.Ceiling(interval*Stopwatch.Frequency/1000.0));
            using(var nt=stateProvider=="nt_system_process_information_x64"?new NtSnapshot():null)
            using(var waiter=new Waiter()) {
                provider=waiter.Provider; begin=Tick();
                SaveNew(args[1],D("schema","owned-thread-sampler-ready-1","process_id",Process.GetCurrentProcess().Id,
                    "clock",Clock(),"ready_tick",begin,"requested_affinity_mask",requestedMask,
                    "observed_affinity_mask",observedMask,"wait_provider",provider,"state_provider",stateProvider));
                long deadline=begin+(long)Math.Ceiling(maximum*Stopwatch.Frequency/1000.0),scheduled=begin;
                while(true) {
                    if(File.Exists(args[2])) {reason="owned_stop";break;}
                    if(Tick()>=deadline) {reason="maximum_duration";break;}
                    if(records.Count>=MaxSamples) throw new InvalidOperationException("20000-snapshot storage bound reached; no silent drop");
                    long queryBegin=Tick();var rows=new List<object>();
                    if(nt!=null) {foreach(var actor in actors)actor.Verify();nt.Capture(seenPids);foreach(var actor in actors)rows.Add(actor.SnapshotNt(nt));}
                    else foreach(var actor in actors) rows.Add(actor.Snapshot());
                    records.Add(D("sample_index",records.Count,"scheduled_tick",scheduled,
                        "query_begin_tick",queryBegin,"query_end_tick",Tick(),"actors",rows,
                        "shared_native_snapshot_query_attempts",nt==null?null:new List<object>(nt.Attempts)));
                    scheduled+=period;long now=Tick();
                    if(scheduled<=now) {long skip=(now-scheduled)/period+1;missed+=skip;scheduled+=skip*period;}
                    while(Tick()<scheduled && Tick()<deadline && !File.Exists(args[2])) waiter.Until(Math.Min(scheduled,deadline));
                }
            }
        } catch(Exception failure) {result=2;error=failure.GetType().Name+": "+failure.Message;reason="failed";}
        finally {
            end=Tick(); foreach(var actor in actors)actor.Dispose();
            output=D("schema","owned-thread-snapshots-1","process_id",Process.GetCurrentProcess().Id,
                "clock",Clock(),"sampling_begin_tick",begin,"sampling_end_tick",end,
                "requested_interval_ms",interval,"maximum_duration_ms",maximum,"missed_periods",missed,
                "requested_affinity_mask",requestedMask,"observed_affinity_mask",observedMask,
                "wait_provider",provider,"global_timer_settings_changed",false,
                "state_provider",stateProvider,
                "sampling_end_reason",reason,"error",error,"records",records,
                "sample_count",records.Count,"max_samples",MaxSamples,"max_threads_per_actor",MaxThreads,
                "exact_thread_state_durations_claimed",false,"cycles_converted_to_time",false,
                "useful_work_distinguished_from_runtime_spin",false,"used_as_predictor",false,
                "runtime",D("clr_version",Environment.Version.ToString(),"process_assembly",typeof(Process).Assembly.Location,
                    "serializer_assembly",typeof(JavaScriptSerializer).Assembly.Location,"core_assembly",typeof(object).Assembly.Location));
            if(args.Length==4) try { SaveNew(args[3],output); } catch(Exception failure) {Console.Error.WriteLine(failure);result=2;}
        }
        if(error!=null)Console.Error.WriteLine(error);
        return result;
    }
}
