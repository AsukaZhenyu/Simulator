// Small actual-GGML consumer-hook contract. Not an LLM accuracy experiment.
#define NOMINMAX
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "nlohmann/json.hpp"
#include <windows.h>
#include <psapi.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using json=nlohmann::json;
static void require(bool value,const std::string &message){if(!value)throw std::runtime_error(message);}
static int64_t elapsed(int64_t a,int64_t b){LARGE_INTEGER hz{};require(QueryPerformanceFrequency(&hz)!=0,"QPC unavailable");return (b-a)/hz.QuadPart*1000000000LL+(b-a)%hz.QuadPart*1000000000LL/hz.QuadPart;}
#include "route_tap_pod.hpp"

struct Owned {
 ggml_backend_t cpu=nullptr;ggml_threadpool_t pool=nullptr;ggml_backend_sched_t scheduler=nullptr;
 ggml_context *resident=nullptr,*compute=nullptr;ggml_backend_buffer_t backing=nullptr;
 ggml_cgraph *graph=nullptr;ggml_tensor *x=nullptr,*y=nullptr;
 std::vector<ggml_tensor*> weights,owners,selected,expert_nodes;
 ~Owned(){if(scheduler)ggml_backend_sched_free(scheduler);if(backing)ggml_backend_buffer_free(backing);if(compute)ggml_free(compute);if(resident)ggml_free(resident);if(cpu)ggml_backend_free(cpu);if(pool)ggml_threadpool_free(pool);}
};
static int expert_id(int layer,int row,int slot){return (layer*3+row*11+slot*7)%64;}
static json graph_evidence(const Owned &s){
 json nodes=json::array();for(int i=0;i<ggml_graph_n_nodes(s.graph);++i){const auto *n=ggml_graph_node(s.graph,i);auto *b=ggml_backend_sched_get_tensor_backend(s.scheduler,const_cast<ggml_tensor*>(n));
  json shape=json::array(),stride=json::array(),sources=json::array();for(int k=0;k<4;++k){shape.push_back(n->ne[k]);stride.push_back(n->nb[k]);}for(auto *p:n->src)sources.push_back(p?json(p->name):json(nullptr));
  nodes.push_back(json{{"name",n->name},{"op",int(n->op)},{"type",int(n->type)},{"flags",n->flags},{"ne",shape},{"nb",stride},{"src",sources},{"backend",b?ggml_backend_name(b):"unassigned_view"}});
 }
 const auto text=nodes.dump();return json{{"node_count",nodes.size()},{"splits",ggml_backend_sched_get_n_splits(s.scheduler)},{"copies",ggml_backend_sched_get_n_copies(s.scheduler)},
  {"graph_and_partition_sha256",tap_sha256(text.data(),text.size())},{"scope","actual CPU-only scheduler graph; not a real-model heterogeneous partition"}};
}
static void initialize(Owned &s,int threads,int tokens){
 constexpr int hidden=32,layers=16,experts=64,topk=8;
 s.cpu=ggml_backend_cpu_init();require(s.cpu,"CPU backend failed");ggml_backend_cpu_set_n_threads(s.cpu,threads);
 auto params=ggml_threadpool_params_default(threads);params.poll=0;s.pool=ggml_threadpool_new(&params);require(s.pool,"owned persistent pool failed");ggml_backend_cpu_set_threadpool(s.cpu,s.pool);
 s.resident=ggml_init({1024*1024,nullptr,true});s.compute=ggml_init({4*1024*1024,nullptr,true});require(s.resident&&s.compute,"metadata allocation failed");
 s.graph=ggml_new_graph_custom(s.compute,2048,false);s.x=ggml_new_tensor_2d(s.resident,GGML_TYPE_F32,hidden,tokens);ggml_set_input(s.x);ggml_set_name(s.x,"input");
 for(int layer=0;layer<layers;++layer){auto *ids=ggml_new_tensor_2d(s.resident,GGML_TYPE_I32,experts,tokens);ggml_set_input(ids);ggml_format_name(ids,"strided_expert_backing-%d",layer);s.owners.push_back(ids);
  for(int stage=0;stage<3;++stage){auto *w=ggml_new_tensor_3d(s.resident,GGML_TYPE_F32,hidden,hidden,experts);ggml_set_input(w);ggml_format_name(w,"actual_small_weight-%d-%d",layer,stage);s.weights.push_back(w);}
 }
 s.backing=ggml_backend_alloc_ctx_tensors(s.resident,s.cpu);require(s.backing,"resident backing allocation failed");require(ggml_backend_buffer_get_size(s.backing)<32*1024*1024,"small resident budget exceeded");
 std::vector<float> x(size_t(hidden)*tokens);for(size_t i=0;i<x.size();++i)x[i]=float(int(i%13)-6)/32.f;ggml_backend_tensor_set(s.x,x.data(),0,x.size()*sizeof(float));
 for(size_t index=0;index<s.weights.size();++index){auto *w=s.weights[index];std::vector<float> values(static_cast<size_t>(ggml_nelements(w)));
  for(size_t i=0;i<values.size();++i)values[i]=float(int((i*17+index*7)%23)-11)/1024.f;ggml_backend_tensor_set(w,values.data(),0,values.size()*sizeof(float));}
 for(int layer=0;layer<layers;++layer){std::vector<int32_t> ids(size_t(experts)*tokens,-99);for(int row=0;row<tokens;++row)for(int k=0;k<topk;++k)ids[size_t(row)*experts+k]=expert_id(layer,row,k);
  ggml_backend_tensor_set(s.owners[layer],ids.data(),0,ids.size()*sizeof(int32_t));}
 auto *current=s.x;
 for(int layer=0;layer<layers;++layer){auto *ids=ggml_view_2d(s.compute,s.owners[layer],topk,tokens,experts*sizeof(int32_t),0);ggml_format_name(ids,"actual_strided_ids-%d",layer);s.selected.push_back(ids);
  auto *input=ggml_reshape_3d(s.compute,current,hidden,1,tokens);
  auto *up=ggml_mul_mat_id(s.compute,s.weights[layer*3],input,ids);ggml_format_name(up,"ffn_moe_up-%d",layer);
  auto *gate=ggml_mul_mat_id(s.compute,s.weights[layer*3+1],input,ids);ggml_format_name(gate,"ffn_moe_gate-%d",layer);
  auto *activation=ggml_swiglu_split(s.compute,gate,up);
  auto *down=ggml_mul_mat_id(s.compute,s.weights[layer*3+2],activation,ids);ggml_format_name(down,"ffn_moe_down-%d",layer);
  s.expert_nodes.insert(s.expert_nodes.end(),{up,gate,down});auto *sum=ggml_view_2d(s.compute,down,hidden,tokens,down->nb[2],0);
  for(int k=1;k<topk;++k){auto *part=ggml_view_2d(s.compute,down,hidden,tokens,down->nb[2],k*down->nb[1]);sum=ggml_add(s.compute,sum,part);}
  current=ggml_add(s.compute,current,sum);ggml_format_name(current,"layer_out-%d",layer);
 }
 s.y=current;ggml_set_output(s.y);ggml_build_forward_expand(s.graph,s.y);
 ggml_backend_t backends[]={s.cpu};ggml_backend_buffer_type_t bufts[]={ggml_backend_get_default_buffer_type(s.cpu)};
 s.scheduler=ggml_backend_sched_new(backends,bufts,1,2048,false,false);require(s.scheduler,"CPU-only scheduler failed");
 require(ggml_backend_sched_alloc_graph(s.scheduler,s.graph),"CPU-only graph allocation failed");
 require(ggml_backend_sched_get_buffer_size(s.scheduler,s.cpu)+ggml_backend_buffer_get_size(s.backing)<64*1024*1024,"known tensor/workspace budget exceeded");
}
static std::vector<float> compute(Owned &s){require(ggml_backend_sched_graph_compute(s.scheduler,s.graph)==GGML_STATUS_SUCCESS,"actual graph compute failed");std::vector<float> result(size_t(ggml_nelements(s.y)));ggml_backend_tensor_get(s.y,result.data(),0,result.size()*sizeof(float));for(float x:result)require(std::isfinite(x),"small graph produced nonfinite result");return result;}
static json rejection_contract(Owned &s,int tokens){
 auto *dst=s.expert_nodes.front();auto *source=s.selected.front();json cases=json::array();
 auto run=[&](const char *name,int expected,int tweak){Capture c;c.prepare(1,tokens,16,64,8);c.enabled=true;c.forward=0;c.position=100;c.submitted=tokens;
  ggml_tensor changed=*dst;int32_t first=-1;ggml_backend_tensor_get(s.owners.front(),&first,0,sizeof(first));
  if(tweak==0){int32_t bad=64;ggml_backend_tensor_set(s.owners.front(),&bad,0,sizeof(bad));}
  if(tweak==1)c.ids.clear();if(tweak==2)std::strcpy(changed.name,"ffn_moe_up-16");if(tweak==3)c.records.clear();
  Capture::callback(&changed,source,&c);
  ggml_backend_tensor_set(s.owners.front(),&first,0,sizeof(first));require(c.error==expected,"invalid consumer observation not explicitly rejected");
  bool rejected=false;try{c.output();}catch(const std::runtime_error &){rejected=true;}require(rejected,"invalid capture silently exported");
  cases.push_back(json{{"case",name},{"error_code",c.error},{"explicitly_rejected",true},{"invalid_ids_sent_to_compute_kernel",false}});
 };
 run("expert_ID_equal_to_expert_count",4,0);run("ID_POD_capacity_exhausted",3,1);run("layer_out_of_range",1,2);run("record_POD_capacity_exhausted",3,3);return cases;
}
static json one_case(int threads,int tokens){
 Owned s;initialize(s,threads,tokens);Capture capture;capture.prepare(1,tokens,16,64,8);capture.forward=0;capture.position=100;capture.submitted=tokens;capture.decode=tokens==1;
 TapBinding binding(capture,false);const auto before=compute(s);const auto before_graph=graph_evidence(s);require(binding.setter,"private CPU tap ABI missing");
 capture.enabled=true;binding.setter(Capture::callback,&capture);const auto observed=compute(s);capture.enabled=false;const auto after_graph=graph_evidence(s);binding.setter(nullptr,nullptr);const auto after=compute(s);
 require(before.size()==observed.size()&&std::memcmp(before.data(),observed.data(),before.size()*sizeof(float))==0&&std::memcmp(before.data(),after.data(),before.size()*sizeof(float))==0,"hook changed actual output bits");
 require(before_graph==after_graph,"hook changed graph topology/CPU scheduler partition");
 const auto routes=capture.output();require(routes.size()==16&&capture.used==48,"actual sixteen-layer/three-stage consumer coverage incomplete");
 for(const auto &route:routes){const int layer=route.at("layer");require(route.at("stage_count")==3&&route.at("all_three_stages_observed")==true,"actual stage coverage incomplete");
  for(int row=0;row<tokens;++row){require(route.at("absolute_positions").at(row)==100+row,"actual strided row mapped to wrong position");for(int k=0;k<8;++k)require(route.at("expert_ids").at(row).at(k)==expert_id(layer,row,k),"actual consumed strided expert ID differs");}}
 const auto rejected=rejection_contract(s,tokens);
 return json{{"threads",threads},{"query_tokens",tokens},{"actual_layers",16},{"actual_MUL_MAT_ID_nodes",48},{"top_k",8},{"experts",64},
  {"source_ids_nb0",s.selected.front()->nb[0]},{"source_ids_nb1",s.selected.front()->nb[1]},{"source_ids_row_contiguous_bytes",32},{"actual_strided_ids",s.selected.front()->nb[1]>32},
  {"normal_hook_normal_outputs_bitwise_equal",true},{"graph_and_CPU_partition_unchanged",true},{"graph",before_graph},{"actual_route_records",routes.size()},
  {"actual_stage_observations",capture.used},{"actual_consumed_IDs_match_independent_initializer",true},{"gate_up_down_actual_IDs_equal",true},{"rejections",rejected},
  {"output_sha256",tap_sha256(before.data(),before.size()*sizeof(float))},{"known_backing_bytes",ggml_backend_buffer_get_size(s.backing)},
  {"known_scheduler_workspace_bytes",ggml_backend_sched_get_buffer_size(s.scheduler,s.cpu)},{"hook_completed_host_observation_ns",elapsed(0,capture.hook_ticks)},
  {"hook_time_is_performance_calibration",false},{"passed",true}};
}
int main(int argc,char **argv){try{
 require(argc==2,"usage: consumer-contract NEW_OUTPUT_JSON");const auto out=std::filesystem::u8path(argv[1]);require(!std::filesystem::exists(out),"contract output already exists");
 json cases=json::array();for(int threads:{1,16})for(int tokens:{1,3})cases.push_back(one_case(threads,tokens));
 PROCESS_MEMORY_COUNTERS_EX mem{};require(GetProcessMemoryInfo(GetCurrentProcess(),reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&mem),sizeof(mem))!=0,"process resident counter unavailable");
 require(mem.PeakWorkingSetSize<=192ULL*1024*1024,"low-memory native contract resident budget exceeded");
 json result{{"schema","real-moe-consumer-small-graph-contract-1"},{"source_sha256",CONSUMER_CONTRACT_SOURCE_SHA256},{"process_id",GetCurrentProcessId()},
  {"passed",true},{"cases",cases},{"native_cases_completed",cases.size()},{"peak_working_set_bytes",mem.PeakWorkingSetSize},{"private_commit_bytes",mem.PrivateUsage},
  {"resident_budget_bytes",192ULL*1024*1024},{"model_loaded",false},{"CUDA_backend_used",false},{"weight_dtype","F32 small contract only"},
  {"scope","actual CPU GGML MUL_MAT_ID hook mechanism; not real-model MoE numerical/performance accuracy or actual-model sixteen-layer coverage"},
  {"real_model_accuracy_certified",false},{"actual_LLM_layer_coverage_certified",false},{"performance_parameters_fitted",false}};
 std::ofstream f(out,std::ios::binary);require(f.good(),"contract output unavailable");f<<result.dump(2)<<'\n';f.close();std::cout<<json{{"passed",true},{"native_cases_completed",cases.size()},{"peak_working_set_bytes",mem.PeakWorkingSetSize}}.dump()<<'\n';return 0;
}catch(const std::exception &e){std::cerr<<"consumer-contract: "<<e.what()<<'\n';return 2;}}
