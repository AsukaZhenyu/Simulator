// Additional same-stream async D2H observation. Never a bare timing source.
#pragma once
#include "gpu_tap_api.h"
struct GPUEvent {int forward=-1,nodes=0,experts=0,compiled=0,use=0,update=0;};
struct GPUCopyMeta {int forward=-1,layer=-1,stage=-1,rows=0,slots=0;size_t source_nb0=0,source_nb1=0,destination_offset=0;uintptr_t stream=0;};
struct GPUCapture {
 using Alloc=int(*)(size_t,void**);using Free=int(*)(void*);
 Capture data;bool enabled=false;int error=0,forward=-1;bool decode=false;int position=0,submitted=0;
 void *pinned=nullptr;size_t pinned_bytes=0;Alloc alloc=nullptr;Free free=nullptr;
 std::vector<GPUEvent> events;std::vector<GPUCopyMeta> copies;size_t event_count=0;int64_t enqueue_ticks=0;size_t copy_count=0;
 ~GPUCapture(){if(pinned&&free)free(pinned);}
 void prepare(int forwards,int rows,int layers,int experts,int k,bool record){
  enabled=record;error=0;forward=-1;event_count=0;enqueue_ticks=0;copy_count=0;
  events.assign(16384,{});data.prepare(forwards,rows,layers,experts,k);copies.assign(data.records.size(),{});
  // A dense model declares no routed expert, so there is no ID channel to
  // reserve. The absence is spelled out here rather than by reserving a
  // zero-byte buffer, which some allocators refuse.
  if(record&&k>0){const auto bytes=data.ids.size()*sizeof(int32_t);require(bytes<=80000000,"pinned observer envelope exceeded");
   if(bytes>pinned_bytes){if(pinned){require(free(pinned)==0,"pinned observer free failed");pinned=nullptr;pinned_bytes=0;}
    require(alloc&&alloc(bytes,&pinned)==0&&pinned,"pinned observer allocation failed");pinned_bytes=bytes;}}
 }
 static void graph(const ggml_cgraph *cg,int compiled,int use,int update,void *user) noexcept {
  auto &g=*static_cast<GPUCapture*>(user);if(g.error)return;
  if(g.event_count>=g.events.size()){g.error=11;return;}
  // Pinned GGML read-only getters omit const in their public signature.
  auto *read_only_graph=const_cast<ggml_cgraph*>(cg);
  int experts=0;const int n=ggml_graph_n_nodes(read_only_graph);for(int i=0;i<n;++i)if(ggml_graph_node(read_only_graph,i)->op==GGML_OP_MUL_MAT_ID)++experts;
  g.events[g.event_count++]={g.forward,n,experts,compiled,use,update};
  // Host consumer entry is bypassed by graph replay. Never emit stale IDs.
  if(g.enabled&&use&&experts)g.error=10;
 }
 static void *reserve(const ggml_tensor *dst,const ggml_tensor *ids,void *stream,size_t row_bytes,void *user) noexcept {
  auto &g=*static_cast<GPUCapture*>(user);if(!g.enabled||g.error||g.data.top_k==0)return nullptr;
  LARGE_INTEGER first{},last{};QueryPerformanceCounter(&first);int layer=-1;const int stage=Capture::stage_layer(dst->name,layer);
  if(stage<0){g.error=12;return nullptr;}
  if(!stream||!ids||ids->type!=GGML_TYPE_I32||ids->nb[0]!=sizeof(int32_t)||ids->ne[0]!=g.data.top_k||ids->ne[1]<1||ids->ne[1]>4096||ids->ne[2]!=1||ids->ne[3]!=1||!ids->data||layer<0||layer>=g.data.layers||row_bytes!=size_t(g.data.top_k)*4||ids->nb[1]<row_bytes){g.error=13;return nullptr;}
  const int rows=static_cast<int>(ids->ne[1]);const size_t count=size_t(rows)*g.data.top_k;
  if((rows!=g.submitted&&rows!=1)||g.data.used>=g.data.records.size()||g.data.ids_used+count>g.data.ids.size()||(g.data.ids_used+count)*4>g.pinned_bytes){g.error=14;return nullptr;}
  g.data.records[g.data.used++]={g.forward,layer,stage,rows,g.data.top_k,g.position,g.submitted,g.decode,g.data.ids_used};
  g.copies[g.copy_count++]={g.forward,layer,stage,rows,g.data.top_k,ids->nb[0],ids->nb[1],g.data.ids_used*4,reinterpret_cast<uintptr_t>(stream)};
  auto *slot=static_cast<char*>(g.pinned)+g.data.ids_used*4;g.data.ids_used+=count;
  QueryPerformanceCounter(&last);g.enqueue_ticks+=last.QuadPart-first.QuadPart;return slot;
 }
 static void status(int status,void *user) noexcept {auto &g=*static_cast<GPUCapture*>(user);if(status&&!g.error)g.error=1000+status;}
 json graph_output()const {json out=json::array();for(size_t i=0;i<event_count;++i){const auto &e=events[i];out.push_back(json{{"forward_index",e.forward},{"partition_nodes",e.nodes},{"expert_nodes",e.experts},{"USE_CUDA_GRAPH_compiled",bool(e.compiled)},{"use_cuda_graph",bool(e.use)},{"cuda_graph_update_required",bool(e.update)},{"execution",!e.use?"direct":e.update?"capture_update_launch":"replay_launch"},{"device_completion_observed_here",false}});}return out;}
 json copy_output()const {json out=json::array();for(size_t i=0;i<copy_count;++i){const auto &c=copies[i];out.push_back(json{{"forward_index",c.forward},{"layer",c.layer},{"stage",c.stage},{"rows",c.rows},{"slots",c.slots},{"source_nb0",c.source_nb0},{"source_nb1",c.source_nb1},{"pinned_destination_offset_bytes",c.destination_offset},{"bytes",size_t(c.rows)*c.slots*4},{"actual_consumer_stream_opaque_id",c.stream},{"same_stream_before_consumer",true}});}return out;}
 json output(){
  if(error)return json::array();
  // Existing llama_synchronize completed all preceding same-stream copies.
  if(data.ids_used)std::memcpy(data.ids.data(),pinned,data.ids_used*sizeof(int32_t));
  for(size_t i=0;i<data.ids_used;++i)if(data.ids[i]<0||data.ids[i]>=data.experts){error=15;return json::array();}
  auto out=data.output();for(auto &row:out){row["consumer_backend"]="CUDA0";row["observation_transport"]="same_stream_async_D2H_before_consumer";}return out;
 }
};
struct GPUBinding {
 using Setter=void(*)(sim_gpu_reserve,sim_gpu_copy_status,sim_gpu_graph_event,void*);Setter setter=nullptr;
 GPUBinding(GPUCapture &g,bool capture){auto m=GetModuleHandleW(L"ggml-cuda.dll");require(m,"actual CUDA DLL missing");
  setter=reinterpret_cast<Setter>(GetProcAddress(m,"simulator_gpu_tap_set_v5"));
  g.alloc=reinterpret_cast<GPUCapture::Alloc>(GetProcAddress(m,"simulator_gpu_tap_host_alloc_v5"));g.free=reinterpret_cast<GPUCapture::Free>(GetProcAddress(m,"simulator_gpu_tap_host_free_v5"));
  require(setter&&g.alloc&&g.free,"private GPU observer v5 ABI missing");setter(capture?GPUCapture::reserve:nullptr,GPUCapture::status,GPUCapture::graph,&g);}
 ~GPUBinding(){if(setter)setter(nullptr,nullptr,nullptr,nullptr);}
};
// Pure POD/shape rejection checks: no CUDA allocation, backend or DMA executed.
static int gpu_tap_pod_contract(){
 std::vector<int32_t> owned(128,0);GPUCapture g;g.data.prepare(1,3,1,4,2);g.copies.assign(3,{});
 g.enabled=true;g.pinned=owned.data();g.pinned_bytes=owned.size()*4;g.forward=0;g.submitted=3;
 ggml_tensor dst{},ids{};ids.type=GGML_TYPE_I32;ids.ne[0]=2;ids.ne[1]=3;ids.ne[2]=ids.ne[3]=1;ids.nb[0]=4;ids.nb[1]=256;ids.data=owned.data();
 const int32_t values[]={0,1,1,2,2,3};const char *names[]={"ffn_moe_up-0","ffn_moe_gate-0","ffn_moe_down-0"};
 for(const char *name:names){std::strcpy(dst.name,name);void *slot=GPUCapture::reserve(&dst,&ids,reinterpret_cast<void*>(uintptr_t(1)),8,&g);require(slot,"POD valid reservation rejected");std::memcpy(slot,values,sizeof(values));}
 auto result=g.output();require(!g.error&&result.size()==1&&result[0]["all_three_stages_observed"]==true&&g.copy_count==3,"POD stage/stride grouping failed");
 int checks=1;g.error=0;ids.nb[0]=8;GPUCapture::reserve(&dst,&ids,reinterpret_cast<void*>(uintptr_t(1)),8,&g);require(g.error==13,"POD noncontiguous slot accepted");++checks;ids.nb[0]=4;
 g.error=0;ids.nb[1]=4;GPUCapture::reserve(&dst,&ids,reinterpret_cast<void*>(uintptr_t(1)),8,&g);require(g.error==13,"POD short row pitch accepted");++checks;ids.nb[1]=256;
 g.error=0;GPUCapture::reserve(&dst,&ids,reinterpret_cast<void*>(uintptr_t(1)),8,&g);require(g.error==14,"POD full record storage accepted");++checks;
 g.error=0;std::strcpy(dst.name,"unknown_consumer");GPUCapture::reserve(&dst,&ids,reinterpret_cast<void*>(uintptr_t(1)),8,&g);require(g.error==12,"POD unknown stage accepted");++checks;
 g.error=0;GPUCapture::status(1,&g);require(g.error==1001,"POD async failure discarded");++checks;
 g.pinned=nullptr;return checks;
}
