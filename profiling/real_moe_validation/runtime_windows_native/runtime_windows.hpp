// Read-only CPU-node and natural-runtime boundary observer. No added GPU sync.
#pragma once
#include <array>
#include <cstring>
#include <vector>
struct PerfNode {
 char name[GGML_MAX_NAME]{};int op=0,type=0,weight_type=-1;int64_t ne[4]{},weight_ne[4]{};
};
struct PerfEvent {int kind=0,forward=-1,q=0,position=0,ordinal=0;bool decode=false,cpu=false;int first_node=0,node_count=0;uint64_t bytes=0;int64_t stamp=0;
 // Cumulative thread CPU time at this boundary. Wall time alone cannot tell
 // "waiting" from "working": a window that blocks on the device and a window
 // that computes for the same wall duration are indistinguishable. The
 // difference of two of these is the CPU time actually consumed inside the
 // window, so wall - CPU is the blocked part.
 //
 // GetThreadTimes is NOT usable at this scale: its accounting is quantised to
 // the 15.625 ms clock tick, which is wider than almost every window here.
 // QueryThreadCycleTime is per-thread and counts at the cycle counter, so the
 // stamp is emitted in cycles together with the thread it belongs to; only a
 // window whose two ends ran on the same thread has a meaningful difference.
 uint64_t cycles=0;unsigned thread_id=0;};
static uint64_t thread_cycles(unsigned &id){
 id=unsigned(GetCurrentThreadId());ULONGLONG c=0;
 if(!QueryThreadCycleTime(GetCurrentThread(),&c))return 0;
 return uint64_t(c);
}
struct RuntimeWindows {
 using Hook=void(*)(int,ggml_backend_t,const ggml_cgraph*,const ggml_tensor*,uint64_t,void*);
 using Setter=void(*)(Hook,void*);
 std::vector<PerfNode> nodes;std::vector<PerfEvent> events;
 size_t nodes_used=0,events_used=0;bool enabled=false,recorded=false;int error=0,forward=-1,q=0,position=0,ordinal=0;bool decode=false;
 Setter base_set=nullptr,cpu_set=nullptr;
 // Cycles per second for the calling thread, measured once by spinning on the
 // same thread that stamps every boundary. Without it a cycle difference cannot
 // be turned into time; measuring it here keeps the conversion inside the probe
 // rather than inferring it from the data being measured.
 double cycles_per_second=0;
 void calibrate(){
  if(cycles_per_second>0)return;
  unsigned id=0;const uint64_t start=thread_cycles(id);if(!start)return;
  const int64_t t0=tick();uint64_t span=0;
  for(;;){const uint64_t now=thread_cycles(id);if(now<start)return;span=now-start;if(span>=4u<<20)break;if(tick()-t0>int64_t(frequency())/4)break;}
  const int64_t elapsed=tick()-t0;
  if(elapsed>0&&span>0)cycles_per_second=double(span)*double(frequency())/double(elapsed);
 }
 RuntimeWindows():nodes(100000),events(120000){calibrate();
  auto base=GetModuleHandleA("ggml-base.dll"),cpu=GetModuleHandleA("ggml-cpu.dll");
  require(base&&cpu,"profile runtime modules unavailable");
  base_set=reinterpret_cast<Setter>(GetProcAddress(base,"simulator_runtime_window_set"));
  cpu_set=reinterpret_cast<Setter>(GetProcAddress(cpu,"simulator_runtime_cpu_window_set"));
  require(base_set&&cpu_set,"explicit runtime window observer ABI unavailable");
  base_set(nullptr,nullptr);cpu_set(nullptr,nullptr);
 }
 ~RuntimeWindows(){enabled=false;if(base_set)base_set(nullptr,nullptr);if(cpu_set)cpu_set(nullptr,nullptr);}
 void begin(bool active){recorded=enabled=active;nodes_used=events_used=0;error=0;forward=-1;q=position=ordinal=0;decode=false;base_set(active?callback:nullptr,active?this:nullptr);cpu_set(active?callback:nullptr,active?this:nullptr);}
 void finish(){enabled=false;base_set(nullptr,nullptr);cpu_set(nullptr,nullptr);}
 void context(int f,int query,int pos,bool d,int ord){forward=f;q=query;position=pos;decode=d;ordinal=ord;}
 void mark(int kind,int64_t stamp=0){if(!enabled||error)return;PerfEvent e{};e.kind=kind;e.forward=forward;e.q=q;e.position=position;e.decode=decode;e.ordinal=ordinal;e.stamp=stamp?stamp:tick();e.cycles=thread_cycles(e.thread_id);push(e);}
 void push(const PerfEvent&e){if(events_used==events.size()){error=1;return;}events[events_used++]=e;}
 static void node(PerfNode &out,const ggml_tensor*t) noexcept {
  std::memcpy(out.name,t->name,sizeof(out.name));out.name[sizeof(out.name)-1]=0;out.op=int(t->op);out.type=int(t->type);
  for(int k=0;k<4;++k){out.ne[k]=t->ne[k];out.weight_ne[k]=t->src[0]?t->src[0]->ne[k]:0;}
  out.weight_type=t->src[0]?int(t->src[0]->type):-1;
 }
 static void callback(int kind,ggml_backend_t backend,const ggml_cgraph*g,const ggml_tensor*t,uint64_t bytes,void*user) noexcept {
  auto&s=*static_cast<RuntimeWindows*>(user);if(!s.enabled||s.error)return;
  PerfEvent e{};e.kind=kind;e.forward=s.forward;e.q=s.q;e.position=s.position;e.ordinal=s.ordinal;e.decode=s.decode;e.stamp=tick();
  e.cycles=thread_cycles(e.thread_id);
  e.bytes=bytes;e.cpu=kind==10||(backend&&ggml_backend_dev_type(ggml_backend_get_device(backend))==GGML_BACKEND_DEVICE_TYPE_CPU);
  if(!backend&&t){const auto*owner=t->view_src?t->view_src:t;if(owner->buffer)e.cpu=ggml_backend_buffer_is_host(owner->buffer);}
  e.first_node=int(s.nodes_used);
  if(kind==1&&g){const int count=ggml_graph_n_nodes(const_cast<ggml_cgraph*>(g));
   if(s.nodes_used+size_t(count)>s.nodes.size()){s.error=2;return;}
   for(int i=0;i<count;++i)node(s.nodes[s.nodes_used++],ggml_graph_node(const_cast<ggml_cgraph*>(g),i));e.node_count=count;
  }else if((kind==10||(kind>=5&&kind<=8))&&t){if(s.nodes_used==s.nodes.size()){s.error=2;return;}node(s.nodes[s.nodes_used++],t);e.node_count=1;}
  s.push(e);
 }
 json output()const{
  require(error==0,"runtime window POD capacity exceeded; no partial export");json es=json::array(),ns=json::array();
  for(size_t i=0;i<nodes_used;++i){const auto&n=nodes[i];ns.push_back(json{{"name",n.name},{"op",n.op},{"type",n.type},{"ne",{n.ne[0],n.ne[1],n.ne[2],n.ne[3]}},{"weight_type",n.weight_type},{"weight_ne",{n.weight_ne[0],n.weight_ne[1],n.weight_ne[2],n.weight_ne[3]}}});}
  for(size_t i=0;i<events_used;++i){const auto&e=events[i];es.push_back(json{{"kind",e.kind},{"tick",e.stamp},{"forward",e.forward},{"query_tokens",e.q},{"position",e.position},{"phase",e.decode?"decode":"prefill"},{"ordinal",e.ordinal},{"cpu",e.cpu},{"first_node",e.first_node},{"node_count",e.node_count},{"requested_bytes",e.bytes},{"thread_cycles",e.cycles},{"thread_id",e.thread_id}});}
  return json{{"schema","real-moe-runtime-windows-1"},{"events",es},{"nodes",ns},{"added_GPU_synchronizations",0},{"scheduler_eval_callback",false},{"CPU_observer_thread",0},{"nodes_and_events_are_pods","thread_cycles is the calling thread's QueryThreadCycleTime; a window is only CPU-divided when both ends carry the same thread_id"},{"thread_cycles_per_second",cycles_per_second},{"CPU_node_boundary","next-node start after ordinary worker barrier; final node closes at synchronous CPU graph return"},{"GPU_window_boundary","async submissions and natural waits remain inclusive host runtime bridges, never independent device work"},{"observer_in_timing",recorded},{"POD_capacity_bytes",nodes.size()*sizeof(PerfNode)+events.size()*sizeof(PerfEvent)}};
 }
};
static RuntimeWindows *runtime_windows=nullptr;
