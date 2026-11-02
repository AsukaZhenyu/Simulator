// Bounded physical expert-cache replay. Policies and latency prediction live in Python.
// Construction/reset/readback/serialization are outside bare sequence timers.
#define NOMINMAX
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "nlohmann/json.hpp"
#include <windows.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
using json=nlohmann::json;
static void require(bool b,const std::string&s){if(!b)throw std::runtime_error(s);}
static int64_t tick(){LARGE_INTEGER x;QueryPerformanceCounter(&x);return x.QuadPart;}
static int64_t frequency(){LARGE_INTEGER x;QueryPerformanceFrequency(&x);return x.QuadPart;}
static double nanoseconds(int64_t a,int64_t b){return double(b-a)*1e9/double(frequency());}
static uint32_t mix(uint32_t seed,uint32_t index,uint32_t ordinal){uint32_t z=seed+(index+1)*0x9e3779b9u+(ordinal+1)*0x85ebca6bu;z^=z>>16;z*=0x7feb352du;z^=z>>15;z*=0x846ca68bu;return z^(z>>16);}
struct Weight {std::array<ggml_tensor*,3> matrix{};};
struct Graph {
 ggml_context *ctx=nullptr,*ports=nullptr;ggml_cgraph *graph=nullptr;ggml_gallocr_t arena=nullptr;ggml_backend_buffer_t buffer=nullptr;
 ggml_tensor *x=nullptr,*y=nullptr;ggml_backend_t backend=nullptr;std::vector<float> input,output;size_t workspace=0;
 ~Graph(){if(arena)ggml_gallocr_free(arena);if(buffer)ggml_backend_buffer_free(buffer);if(ctx)ggml_free(ctx);if(ports)ggml_free(ports);}
};
struct Step {int tokens=0;std::vector<float> input,result;std::vector<std::vector<int>> routes;std::vector<std::vector<float>> scores;std::map<int,std::vector<int>> rows;std::map<int,std::vector<float>> expert_outputs;std::set<int> done;bool combined=false;};
struct Executor {
 json program;int h,i,e,k,threads;uint32_t seed;bool use_gpu=false;size_t expert_bytes=0,host_weight_bytes=0,device_weight_bytes=0;
 ggml_backend_t cpu=nullptr,gpu=nullptr;ggml_threadpool_t pool=nullptr;ggml_context *host_ctx=nullptr,*device_ctx=nullptr;ggml_backend_buffer_t host_buffer=nullptr,device_buffer=nullptr;
 std::vector<Weight> weights,slots;std::vector<int> owners;std::vector<Step> steps;std::map<std::string,std::unique_ptr<Graph>> graphs;
 std::vector<json> observations;size_t h2d_bytes=0,d2h_bytes=0,peak_resident_experts=0;bool record=false;
 explicit Executor(json p):program(std::move(p)){
  require(program.at("schema")=="moe-expert-policy-native-1","unsupported program schema");const auto &c=program.at("config");
  h=c.at("hidden");i=c.at("intermediate");e=c.at("experts");k=c.at("gpu_slots");threads=c.at("threads");seed=program.at("seed");
  require(h>0&&h<=4096&&i>0&&i<=16384&&e>0&&e<=256&&k>=0&&k<=e&&threads>0&&threads<=64,"invalid explicit geometry");
  expert_bytes=size_t(h)*i*3*sizeof(float);host_weight_bytes=expert_bytes*e;device_weight_bytes=expert_bytes*k;
  require(host_weight_bytes<=2ULL*1024*1024*1024,"declared weight envelope exceeds2GiB");
  MEMORYSTATUSEX memory{};memory.dwLength=sizeof(memory);require(GlobalMemoryStatusEx(&memory)!=0,"host memory query failed");
  const auto host_reserve=program.at("resource_guards").at("host_reserve_bytes").get<uint64_t>();require(memory.ullAvailPhys>=host_weight_bytes+host_reserve,"insufficient available host memory; reserve unchanged");
  for(const auto&a:program.at("actions"))if(a.at("kind")=="LOAD"||(a.at("kind")=="EXPERT"&&a.at("device")=="gpu"))use_gpu=true;
  require(!use_gpu||k>0,"GPU actions require positive physical slots");
  cpu=ggml_backend_cpu_init();require(cpu,"CPU unavailable");ggml_backend_cpu_set_n_threads(cpu,threads);auto pp=ggml_threadpool_params_default(threads);pool=ggml_threadpool_new(&pp);require(pool,"CPU threadpool allocation failed");ggml_backend_cpu_set_threadpool(cpu,pool);
  if(use_gpu){gpu=ggml_backend_cuda_init(c.at("cuda_device"));require(gpu,"CUDA unavailable; no fallback");size_t free=0,total=0;ggml_backend_dev_memory(ggml_backend_get_device(gpu),&free,&total);require(free>=device_weight_bytes+program.at("resource_guards").at("gpu_reserve_bytes").get<uint64_t>(),"insufficient GPU memory reserve");}
  host_ctx=ggml_init({ggml_tensor_overhead()*(3*e+8),nullptr,true});require(host_ctx,"host metadata failed");weights.resize(e);
  for(auto&w:weights)for(int n=0;n<3;++n){w.matrix[n]=ggml_new_tensor_2d(host_ctx,GGML_TYPE_F32,n==2?i:h,n==2?h:i);ggml_set_input(w.matrix[n]);}
  host_buffer=ggml_backend_alloc_ctx_tensors(host_ctx,cpu);require(host_buffer,"host weights failed");
  for(int ex=0;ex<e;++ex)for(int n=0;n<3;++n){auto*t=weights[ex].matrix[n];std::vector<float> data(ggml_nelements(t));const double scale=1.0/std::sqrt(double(n==2?i:h));for(size_t idx=0;idx<data.size();++idx)data[idx]=float((double(mix(seed,uint32_t(idx),uint32_t(3*ex+n))>>8)/16777216.0-0.5)*scale);ggml_backend_tensor_set(t,data.data(),0,data.size()*sizeof(float));}
  if(use_gpu){device_ctx=ggml_init({ggml_tensor_overhead()*(3*k+8),nullptr,true});require(device_ctx,"GPU metadata failed");slots.resize(k);for(auto&w:slots)for(int n=0;n<3;++n){w.matrix[n]=ggml_new_tensor_2d(device_ctx,GGML_TYPE_F32,n==2?i:h,n==2?h:i);ggml_set_input(w.matrix[n]);}device_buffer=ggml_backend_alloc_ctx_tensors(device_ctx,gpu);require(device_buffer,"bounded GPU slots allocation failed");}
  owners.assign(k,-1);
  for(const auto&s:program.at("steps")){Step t;t.tokens=int(s.at("input").size());require(t.tokens>0&&t.tokens<=1024,"invalid token rows");require(s.at("routes").size()==size_t(t.tokens)&&s.at("scores").size()==size_t(t.tokens),"route row count differs");t.routes=s.at("routes").get<std::vector<std::vector<int>>>();t.scores=s.at("scores").get<std::vector<std::vector<float>>>();
   for(int q=0;q<t.tokens;++q){require(s.at("input").at(q).size()==size_t(h),"input hidden width differs");for(const auto&v:s.at("input").at(q)){float x=v;require(std::isfinite(x),"nonfinite input");t.input.push_back(x);}require(!t.routes[q].empty()&&t.routes[q].size()==t.scores[q].size(),"route score extent differs");std::set<int> selected;double sum=0;for(size_t n=0;n<t.routes[q].size();++n){int ex=t.routes[q][n];float score=t.scores[q][n];require(ex>=0&&ex<e&&selected.insert(ex).second&&std::isfinite(score)&&score>=0,"invalid route/score");sum+=score;t.rows[ex].push_back(q);}require(std::abs(sum-1.0)<1e-5,"scores must sum to one");}
   t.result.resize(t.input.size());for(const auto&rows:t.rows)t.expert_outputs[rows.first].resize(size_t(rows.second.size())*h);steps.push_back(std::move(t));}
  require(!steps.empty(),"no steps");
  for(const auto&a:program.at("actions"))if(a.at("kind")=="EXPERT"){const int step=a.at("step"),ex=a.at("expert");require(step>=0&&size_t(step)<steps.size()&&steps[step].rows.count(ex),"EXPERT is not active in declared step");const bool dev=a.at("device")=="gpu";require(a.at("device")=="cpu"||dev,"unknown device");int index=dev?a.at("slot").get<int>():ex;require(index>=0&&index<(dev?k:e),"slot/expert out of range");const int q=int(steps[step].rows.at(ex).size());const auto key=graph_key(dev,index,q);if(!graphs.count(key))graphs[key]=build(dev,index,q);}
 }
 ~Executor(){graphs.clear();if(device_buffer)ggml_backend_buffer_free(device_buffer);if(host_buffer)ggml_backend_buffer_free(host_buffer);if(device_ctx)ggml_free(device_ctx);if(host_ctx)ggml_free(host_ctx);if(gpu)ggml_backend_free(gpu);if(cpu)ggml_backend_free(cpu);if(pool)ggml_threadpool_free(pool);}
 static std::string graph_key(bool gpu,int index,int q){return std::string(gpu?"gpu":"cpu")+":"+std::to_string(index)+":"+std::to_string(q);}
 std::unique_ptr<Graph> build(bool dev,int index,int q){auto g=std::make_unique<Graph>();g->backend=dev?gpu:cpu;g->ports=ggml_init({ggml_tensor_overhead()*8,nullptr,true});g->x=ggml_new_tensor_2d(g->ports,GGML_TYPE_F32,h,q);g->y=ggml_new_tensor_2d(g->ports,GGML_TYPE_F32,h,q);ggml_set_input(g->x);ggml_set_output(g->y);g->buffer=ggml_backend_alloc_ctx_tensors(g->ports,g->backend);require(g->buffer,"graph ports allocation failed");g->ctx=ggml_init({ggml_tensor_overhead()*64+ggml_graph_overhead_custom(64,false),nullptr,true});g->graph=ggml_new_graph_custom(g->ctx,64,false);auto&w=(dev?slots:weights).at(index);auto*gate=ggml_mul_mat(g->ctx,w.matrix[0],g->x);auto*up=ggml_mul_mat(g->ctx,w.matrix[1],g->x);auto*out=ggml_cpy(g->ctx,ggml_mul_mat(g->ctx,w.matrix[2],ggml_swiglu_split(g->ctx,gate,up)),g->y);ggml_build_forward_expand(g->graph,out);
  for(int n=0;n<ggml_graph_n_nodes(g->graph);++n){auto*t=ggml_graph_node(g->graph,n);if(t->op==GGML_OP_MUL_MAT)ggml_mul_mat_set_prec(t,GGML_PREC_F32);require(ggml_backend_supports_op(g->backend,t),"backend unsupported exact operation");}g->arena=ggml_gallocr_new(ggml_backend_get_default_buffer_type(g->backend));require(g->arena&&ggml_gallocr_reserve(g->arena,g->graph)&&ggml_gallocr_alloc_graph(g->arena,g->graph),"graph arena allocation failed");g->workspace=ggml_gallocr_get_buffer_size(g->arena,0);if(!dev)g->workspace+=ggml_graph_plan(g->graph,threads,pool).work_size;g->input.resize(size_t(h)*q);g->output.resize(size_t(h)*q);return g;}
 void reset(){std::fill(owners.begin(),owners.end(),-1);for(auto&s:steps){s.done.clear();s.combined=false;std::fill(s.result.begin(),s.result.end(),0);}h2d_bytes=d2h_bytes=peak_resident_experts=0;observations.clear();}
 void action(const json&a,bool strict=true){const std::string kind=a.at("kind");
  if(kind=="LOAD"){const int ex=a.at("expert"),slot=a.at("slot");require(ex>=0&&ex<e&&slot>=0&&slot<k,"LOAD extent invalid");require(!strict||owners[slot]<0,"LOAD overwrites live expert without RELEASE");for(int n=0;n<3;++n)ggml_backend_tensor_copy(weights[ex].matrix[n],slots[slot].matrix[n]);ggml_backend_synchronize(gpu);owners[slot]=ex;h2d_bytes+=expert_bytes;peak_resident_experts=std::max(peak_resident_experts,size_t(std::count_if(owners.begin(),owners.end(),[](int x){return x>=0;})));}
  else if(kind=="RELEASE"){const int ex=a.at("expert"),slot=a.at("slot");require(slot>=0&&slot<k&&(!strict||owners[slot]==ex),"RELEASE of absent/mismatched expert");owners[slot]=-1;}
  else if(kind=="EXPERT"){const int st=a.at("step"),ex=a.at("expert");auto&s=steps.at(st);require(!strict||(!s.combined&&!s.done.count(ex)),"duplicate/late expert execution");const bool dev=a.at("device")=="gpu";const int index=dev?a.at("slot").get<int>():ex;if(dev)require(owners.at(index)==ex,"GPU expert unavailable in physical slot");const auto&rows=s.rows.at(ex);auto&g=*graphs.at(graph_key(dev,index,int(rows.size())));for(size_t row=0;row<rows.size();++row)std::copy_n(s.input.data()+size_t(rows[row])*h,h,g.input.data()+row*h);ggml_backend_tensor_set(g.x,g.input.data(),0,g.input.size()*sizeof(float));require(ggml_backend_graph_compute(g.backend,g.graph)==GGML_STATUS_SUCCESS,"expert compute failed");ggml_backend_synchronize(g.backend);ggml_backend_tensor_get(g.y,g.output.data(),0,g.output.size()*sizeof(float));std::copy(g.output.begin(),g.output.end(),s.expert_outputs.at(ex).begin());s.done.insert(ex);if(dev){h2d_bytes+=g.input.size()*sizeof(float);d2h_bytes+=g.output.size()*sizeof(float);}}
  else if(kind=="COMBINE"){auto&s=steps.at(a.at("step").get<int>());require(!strict||(!s.combined&&s.done.size()==s.rows.size()),"incomplete/duplicate COMBINE");std::copy(s.input.begin(),s.input.end(),s.result.begin());for(const auto&entry:s.rows){const int ex=entry.first;const auto&ys=s.expert_outputs.at(ex);for(size_t row=0;row<entry.second.size();++row){const int q=entry.second[row];const auto it=std::find(s.routes[q].begin(),s.routes[q].end(),ex);const float score=s.scores[q][size_t(it-s.routes[q].begin())];for(int dim=0;dim<h;++dim)s.result[size_t(q)*h+dim]+=score*ys[row*h+dim];}}s.combined=true;}
  else throw std::runtime_error("unsupported action: "+kind);
 }
 void full(bool strict=true){for(const auto&a:program.at("actions"))action(a,strict);for(const auto&s:steps)require(s.combined,"not all step outputs completed");}
 json run(const json&r){const std::string mode=r.at("mode");const int warmup=r.at("warmup"),samples=r.at("samples"),repeats=r.at("repeats");require(warmup>=0&&warmup<=100&&samples>0&&samples<=1000&&repeats>0&&repeats<=1000,"invalid measurement bounds");require(mode=="bare"||mode=="profile"||mode=="stamped"||mode=="validate","unknown measurement mode");reset();full();json times=json::array(),stamps=json::array();
  if(mode=="profile"){const int target=r.at("target_action_index");require(target>=0&&size_t(target)<program.at("actions").size(),"invalid action index");for(int n=-warmup;n<samples;++n){reset();for(int idx=0;idx<target;++idx)action(program.at("actions").at(idx));const auto begin=tick();for(int rep=0;rep<repeats;++rep)action(program.at("actions").at(target),false);const auto end=tick();if(n>=0)times.push_back(nanoseconds(begin,end)/repeats);}}
  else if(mode!="validate"){for(int n=-warmup;n<samples;++n){double total=0;json row=json::array();for(int rep=0;rep<repeats;++rep){reset();if(mode=="stamped"){const auto begin=tick();for(const auto&a:program.at("actions")){const auto a0=tick();action(a);const auto a1=tick();if(n>=0&&rep==0)row.push_back(nanoseconds(a0,a1));}total+=nanoseconds(begin,tick());}else{const auto begin=tick();full();total+=nanoseconds(begin,tick());}}if(n>=0){times.push_back(total/repeats);if(mode=="stamped")stamps.push_back(row);}}}
  reset();full();json outputs=json::array();for(const auto&s:steps){json rows=json::array();for(int q=0;q<s.tokens;++q)rows.push_back(std::vector<float>(s.result.begin()+size_t(q)*h,s.result.begin()+size_t(q+1)*h));outputs.push_back(rows);}size_t arena=0,ports=0;for(const auto&g:graphs){arena+=g.second->workspace;ports+=ggml_backend_buffer_get_size(g.second->buffer);}json counts=json::object();for(const auto&a:program.at("actions")){const std::string kind=a.at("kind");counts[kind]=counts.value(kind,0)+1;}
  return json{{"schema","moe-expert-policy-result-1"},{"source_sha256",PROBE_SOURCE_SHA256},{"request",r},{"config",program.at("config")},{"seed",seed},{"outputs",outputs},{"samples_ns",times},{"action_samples_ns",stamps},{"actions",program.at("actions")},{"action_counts",counts},{"weight_h2d_bytes",counts.value("LOAD",0)*expert_bytes},{"total_h2d_bytes",h2d_bytes},{"total_d2h_bytes",d2h_bytes},{"peak_resident_experts",peak_resident_experts},{"host_weight_logical_bytes",host_weight_bytes},{"gpu_weight_logical_bytes",use_gpu?device_weight_bytes:0},{"host_weight_allocated_bytes",ggml_backend_buffer_get_size(host_buffer)},{"gpu_weight_allocated_bytes",device_buffer?ggml_backend_buffer_get_size(device_buffer):0},{"workspace_bytes",arena},{"graph_port_allocated_bytes",ports},{"cuda_private_workspace_bytes",nullptr},{"timing_scope","serial prebuilt expert graphs, synchronized weight and activation copies, gather and weighted residual combine; initialization and cache reset excluded"},{"cache_contract","fixed K physical GPU weight slots; RELEASE frees slot ownership, not CUDA allocation; no backing for unresident GPU experts"},{"input_contract","externally supplied routed FFN inputs; not a full language model or router benchmark"},{"weight_generator","mix32-index-v1; ordinal=3*expert+matrix(gate0,up1,down2); F32((u-0.5)/sqrt(input_width))"}};
 }
};
int main(int argc,char**argv){try{if(argc==2&&std::string(argv[1])=="--self-test"){require(mix(1,0,0)==mix(1,0,0),"mix failed");std::cout<<json{{"passed",true},{"source_sha256",PROBE_SOURCE_SHA256},{"hardware_executed",false}}.dump()<<'\n';return 0;}require(argc==4,"usage: probe PROGRAM REQUEST OUTPUT");_putenv_s("NVIDIA_TF32_OVERRIDE","0");_putenv_s("GGML_CUDA_DISABLE_GRAPHS","1");std::ifstream pf(argv[1]),rf(argv[2]);require(pf.good()&&rf.good(),"inputs unreadable");json p,r;pf>>p;rf>>r;Executor ex(std::move(p));auto result=ex.run(r);std::ifstream exists(argv[3]);require(!exists.good(),"output exists; never overwrite evidence");std::ofstream out(argv[3],std::ios::binary);require(out.good(),"output not writable");out<<result.dump()<<'\n';std::cout<<json{{"status","ok"},{"mode",r.at("mode")},{"samples",result.at("samples_ns").size()}}.dump()<<'\n';return 0;}catch(const std::exception&e){std::cerr<<"moe-expert-policy-probe: "<<e.what()<<'\n';return 2;}}
