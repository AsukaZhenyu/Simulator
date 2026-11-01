// Actual packed GGUF weights and vectorized GGML MoE graph; no F32 weight substitute.
#define NOMINMAX
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "nlohmann/json.hpp"
#include <windows.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
using json=nlohmann::json;
static void require(bool b,const std::string &s){if(!b)throw std::runtime_error(s);}
static int64_t tick(){LARGE_INTEGER x{};require(QueryPerformanceCounter(&x)!=0,"QPC unavailable");return x.QuadPart;}
static int64_t hz(){LARGE_INTEGER x{};require(QueryPerformanceFrequency(&x)!=0,"QPC frequency unavailable");return x.QuadPart;}
static int64_t ns(int64_t a,int64_t b){return (b-a)/hz()*1000000000LL+(b-a)%hz()*1000000000LL/hz();}
static ggml_type type(const std::string &s){if(s=="Q4_K")return GGML_TYPE_Q4_K;if(s=="Q6_K")return GGML_TYPE_Q6_K;if(s=="F32")return GGML_TYPE_F32;throw std::runtime_error("unsupported actual packed dtype: "+s);}
static void write_new(const std::filesystem::path &p,const json &j){const auto data=j.dump()+"\n";HANDLE f=CreateFileW(p.c_str(),GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);require(f!=INVALID_HANDLE_VALUE,"exclusive output creation failed");size_t done=0;bool ok=true;
 while(done<data.size()){DWORD n=0;const DWORD count=static_cast<DWORD>(std::min<size_t>(data.size()-done,1<<20));if(!WriteFile(f,data.data()+done,count,&n,nullptr)||n!=count){ok=false;break;}done+=n;}CloseHandle(f);require(ok,"result write failed");}
struct Session{
 ggml_context *resident=nullptr,*ctx=nullptr;ggml_backend_t backend=nullptr;ggml_backend_buffer_t buffer=nullptr;ggml_gallocr_t allocator=nullptr;ggml_threadpool_t pool=nullptr;ggml_cgraph *graph=nullptr;
 ~Session(){if(backend)ggml_backend_synchronize(backend);if(allocator)ggml_gallocr_free(allocator);if(buffer)ggml_backend_buffer_free(buffer);if(backend)ggml_backend_free(backend);if(pool)ggml_threadpool_free(pool);if(ctx)ggml_free(ctx);if(resident)ggml_free(resident);}
};
static json geometry(const ggml_tensor *t){return json{{"dtype",ggml_type_name(t->type)},{"ne",{t->ne[0],t->ne[1],t->ne[2],t->ne[3]}},{"nb",{t->nb[0],t->nb[1],t->nb[2],t->nb[3]}},{"bytes",ggml_nbytes(t)}};}
static json graph_inventory(const Session &s){json nodes=json::array();std::map<const ggml_tensor*,int> index;for(int i=0;i<ggml_graph_n_nodes(s.graph);++i)index[ggml_graph_node(s.graph,i)]=i;
 for(int i=0;i<ggml_graph_n_nodes(s.graph);++i){auto *t=ggml_graph_node(s.graph,i);json inputs=json::array();for(int k=0;k<GGML_MAX_SRC;++k)if(t->src[k])inputs.push_back(json{{"name",t->src[k]->name},{"node",index.count(t->src[k])?json(index.at(t->src[k])):json(nullptr)},{"geometry",geometry(t->src[k])}});
  std::vector<int32_t> params(std::begin(t->op_params),std::end(t->op_params));nodes.push_back(json{{"index",i},{"name",t->name},{"op",ggml_op_name(t->op)},{"geometry",geometry(t)},{"op_params",params},{"inputs",inputs}});}return nodes;}
static void load_weight(ggml_tensor *t,const json &d,const std::filesystem::path &model,uint64_t model_bytes){
 const uint64_t offset=d.at("data_offset"),bytes=d.at("bytes");require(bytes==ggml_nbytes(t)&&offset<=model_bytes&&bytes<=model_bytes-offset,"packed tensor extent differs");
 std::ifstream file(model,std::ios::binary);require(file.good(),"packed model unreadable");file.seekg(static_cast<std::streamoff>(offset));std::vector<uint8_t> packed(static_cast<size_t>(bytes));file.read(reinterpret_cast<char*>(packed.data()),static_cast<std::streamsize>(packed.size()));require(file.gcount()==static_cast<std::streamsize>(packed.size()),"packed tensor truncated");ggml_backend_tensor_set(t,packed.data(),0,packed.size());
}
static void completed_compute(Session &s){require(ggml_backend_graph_compute(s.backend,s.graph)==GGML_STATUS_SUCCESS,"subgraph compute failed");ggml_backend_synchronize(s.backend);}
static std::pair<ggml_tensor*,ggml_tensor*> build_ffn(Session &s,const std::map<std::string,ggml_tensor*> &weights,ggml_tensor *x,int top_k,int experts){
 auto name=[](ggml_tensor *t,const char *n){ggml_set_name(t,n);return t;};auto *router=weights.at("router"),*gate=weights.at("gate"),*up=weights.at("up"),*down=weights.at("down");const int64_t hidden=x->ne[0],tokens=x->ne[1];
 require(weights.size()==4&&router->type==GGML_TYPE_F32&&router->ne[1]==experts&&experts==64&&top_k==8&&hidden==2048,"OLMoE router geometry differs");
 require(gate->ne[0]==hidden&&up->ne[0]==hidden&&gate->ne[1]==1024&&up->ne[1]==1024&&down->ne[0]==1024&&down->ne[1]==hidden&&gate->ne[2]==experts&&up->ne[2]==experts&&down->ne[2]==experts,"OLMoE expert geometry differs");
 require((gate->type==GGML_TYPE_Q4_K||gate->type==GGML_TYPE_Q6_K)&&(up->type==GGML_TYPE_Q4_K||up->type==GGML_TYPE_Q6_K)&&(down->type==GGML_TYPE_Q4_K||down->type==GGML_TYPE_Q6_K),"expert weight is not the declared actual quantized format");
 auto *logits=name(ggml_mul_mat(s.ctx,router,x),"ffn_moe_logits");auto *probs=name(ggml_soft_max(s.ctx,logits),"ffn_moe_probs");auto *selected=name(ggml_argsort_top_k(s.ctx,probs,top_k),"ffn_moe_topk");name(selected->src[0],"ffn_moe_argsort");auto *scores=name(ggml_get_rows(s.ctx,ggml_reshape_3d(s.ctx,probs,1,experts,tokens),selected),"ffn_moe_weights");
 ggml_build_forward_expand(s.graph,scores);auto *shaped=ggml_reshape_3d(s.ctx,x,hidden,1,tokens);auto *u=name(ggml_mul_mat_id(s.ctx,up,shaped,selected),"ffn_moe_up");auto *g=name(ggml_mul_mat_id(s.ctx,gate,shaped,selected),"ffn_moe_gate");auto *activation=name(ggml_swiglu_split(s.ctx,g,u),"ffn_moe_swiglu");auto *d=name(ggml_mul_mat_id(s.ctx,down,activation,selected),"ffn_moe_down");auto *weighted=name(ggml_mul(s.ctx,d,scores),"ffn_moe_weighted");
 std::vector<ggml_tensor*> views;for(int i=0;i<top_k;++i){auto *v=ggml_view_2d(s.ctx,weighted,hidden,tokens,weighted->nb[2],i*weighted->nb[1]);views.push_back(v);ggml_build_forward_expand(s.graph,v);}auto *y=views[0];for(int i=1;i<top_k;++i)y=ggml_add(s.ctx,y,views[i]);name(y,"ffn_moe_out");return {y,selected};
}
static json metadata_graph_test(){json results=json::array();for(const int tokens:{1,256}){Session s;s.ctx=ggml_init({ggml_tensor_overhead()*512+ggml_graph_overhead_custom(512,false),nullptr,true});require(s.ctx,"metadata context failed");s.graph=ggml_new_graph_custom(s.ctx,512,false);
 std::map<std::string,ggml_tensor*> w{{"router",ggml_new_tensor_2d(s.ctx,GGML_TYPE_F32,2048,64)},{"up",ggml_new_tensor_3d(s.ctx,GGML_TYPE_Q4_K,2048,1024,64)},{"gate",ggml_new_tensor_3d(s.ctx,GGML_TYPE_Q4_K,2048,1024,64)},{"down",ggml_new_tensor_3d(s.ctx,GGML_TYPE_Q6_K,1024,2048,64)}};
 auto *x=ggml_new_tensor_2d(s.ctx,GGML_TYPE_F32,2048,tokens);const auto pair=build_ffn(s,w,x,8,64);ggml_build_forward_expand(s.graph,pair.first);std::map<ggml_op,int> counts;for(int i=0;i<ggml_graph_n_nodes(s.graph);++i)++counts[ggml_graph_node(s.graph,i)->op];
 require(counts[GGML_OP_MUL_MAT_ID]==3&&counts[GGML_OP_MUL_MAT]==1&&counts[GGML_OP_GLU]==1&&counts[GGML_OP_ADD]==7&&pair.first->ne[0]==2048&&pair.first->ne[1]==tokens,"shared FFN graph topology/geometry failed");results.push_back(json{{"tokens",tokens},{"nodes",ggml_graph_n_nodes(s.graph)},{"output_shape",{pair.first->ne[0],pair.first->ne[1]}},{"weight_payload_allocated",false}});}return results;}
int main(int argc,char **argv){try{
 if(argc==2&&std::string(argv[1])=="--self-test"){require(type("Q4_K")==GGML_TYPE_Q4_K&&type("Q6_K")==GGML_TYPE_Q6_K,"packed dtype parser failed");bool rejected=false;try{type("Q8_0");}catch(...){rejected=true;}require(rejected,"unsupported dtype admitted");std::cout<<json{{"passed",true},{"model_loaded",false},{"hardware_timing_executed",false},{"shared_measurement_graphs",metadata_graph_test()},{"source_sha256",QUANTIZED_SOURCE_SHA256}}.dump()<<'\n';return 0;}
 require(argc==3,"usage: real-moe-quantized-probe REQUEST OUTPUT");const auto output=std::filesystem::u8path(argv[2]);require(!std::filesystem::exists(output),"output exists");std::ifstream input(std::filesystem::u8path(argv[1]));json r;input>>r;
 require(r.at("schema")=="real-moe-quantized-subgraph-request-1"&&r.at("source_sha256")==QUANTIZED_SOURCE_SHA256,"request/source binding differs");
 const std::string operation=r.at("operation"),backend=r.at("backend");require(operation=="olmoe_cpu_ffn"||operation=="regular_mul_mat","unsupported subgraph");require(backend=="CPU"||backend=="CUDA0","explicit backend required");require(operation!="olmoe_cpu_ffn"||backend=="CPU","heterogeneous FFN split is unsupported, not approximated by CPU price");
 const int tokens=r.at("tokens"),threads=r.at("threads"),warmup=r.at("warmup"),samples=r.at("samples");require(tokens>0&&tokens<=256&&threads>0&&threads<=32&&warmup>=1&&warmup<=100&&samples>=3&&samples<=1000,"sampling/geometry envelope invalid");
 const int warmup_ms=r.at("warmup_min_ms");require(warmup_ms>=0&&warmup_ms<=10000,"explicit warmup interval invalid");
 const std::string cache_context=r.at("cache_context");const size_t foreign_bytes=r.at("foreign_working_set_bytes");
 require(r.at("cpu_threadpool_policy")==json(backend=="CPU"?"explicit_persistent_owned":"not_applicable"),"explicit CPU threadpool policy differs");
 require((cache_context=="isolated_warm"&&foreign_bytes==0)||(cache_context=="foreign_cpu_touch_512MiB"&&foreign_bytes==512ULL*1024*1024&&backend=="CPU"),"unsupported explicit cache context");
 require(r.at("cuda_graph_policy")=="runtime_default_enabled"&&std::getenv("GGML_CUDA_DISABLE_GRAPHS")==nullptr,"runtime graph policy differs");const char *tf32=std::getenv("NVIDIA_TF32_OVERRIDE");require(tf32&&std::string(tf32)=="0","explicit TF32 policy differs");
 const auto requested_mask=r.at("expected_process_mask").get<uint64_t>();DWORD_PTR actual_mask=0,system_mask=0;require(SetProcessAffinityMask(GetCurrentProcess(),static_cast<DWORD_PTR>(requested_mask))&&GetProcessAffinityMask(GetCurrentProcess(),&actual_mask,&system_mask)&&actual_mask==requested_mask,"process affinity enforcement failed");
 MEMORYSTATUSEX memory{};memory.dwLength=sizeof(memory);require(GlobalMemoryStatusEx(&memory)!=0&&memory.ullAvailPhys>=r.at("minimum_free_host_bytes").get<uint64_t>(),"profile host free-memory precondition failed");
 const auto model=std::filesystem::u8path(r.at("model_path").get<std::string>());const uint64_t model_bytes=r.at("model_bytes");require(std::filesystem::file_size(model)==model_bytes,"pinned model byte count differs");
 ggml_backend_load_all();auto device=ggml_backend_dev_by_name(backend.c_str());require(device,"explicit backend missing; no fallback");size_t free=0,total=0;ggml_backend_dev_memory(device,&free,&total);require(backend=="CPU"||free>=r.at("minimum_free_gpu_bytes").get<uint64_t>(),"profile GPU free-memory precondition failed");
 Session s;s.backend=ggml_backend_dev_init(device,nullptr);require(s.backend,"backend initialization failed");if(backend=="CPU"){auto p=ggml_threadpool_params_default(threads);s.pool=ggml_threadpool_new(&p);require(s.pool,"CPU threadpool initialization failed");ggml_backend_cpu_set_n_threads(s.backend,threads);ggml_backend_cpu_set_threadpool(s.backend,s.pool);}
 s.resident=ggml_init({ggml_tensor_overhead()*32,nullptr,true});s.ctx=ggml_init({ggml_tensor_overhead()*512+ggml_graph_overhead_custom(512,false),nullptr,true});require(s.resident&&s.ctx,"metadata allocation failed");s.graph=ggml_new_graph_custom(s.ctx,512,false);
 std::map<std::string,ggml_tensor*> weights;
 for(auto it=r.at("weights").begin();it!=r.at("weights").end();++it){const auto &d=it.value();const auto shape=d.at("shape").get<std::vector<int64_t>>();require(!shape.empty()&&shape.size()<=3,"weight rank unsupported");int64_t ne[4]={1,1,1,1};std::copy(shape.begin(),shape.end(),ne);auto *t=ggml_new_tensor(s.resident,type(d.at("dtype")),static_cast<int>(shape.size()),ne);ggml_set_name(t,d.at("name").get<std::string>().c_str());ggml_set_input(t);weights[it.key()]=t;}
 const int64_t hidden=operation=="regular_mul_mat"?weights.at("weight")->ne[0]:weights.at("router")->ne[0];auto *x=ggml_new_tensor_2d(s.resident,GGML_TYPE_F32,hidden,tokens);ggml_set_name(x,"post_ffn_norm_input");ggml_set_input(x);
 auto name=[&](ggml_tensor *t,const char *n){ggml_set_name(t,n);return t;};ggml_tensor *y=nullptr,*selected=nullptr;
 if(operation=="regular_mul_mat"){require(weights.size()==1&&weights.at("weight")->ne[2]==1,"regular matmul requires one actual matrix");y=name(ggml_mul_mat(s.ctx,weights.at("weight"),x),"regular_mul_mat_output");}
 else{const auto pair=build_ffn(s,weights,x,r.at("top_k"),r.at("experts"));y=pair.first;selected=pair.second;}
 ggml_set_output(y);ggml_build_forward_expand(s.graph,y);for(int i=0;i<ggml_graph_n_nodes(s.graph);++i)require(ggml_backend_supports_op(s.backend,ggml_graph_node(s.graph,i)),"backend does not support exact graph op; no fallback");
 s.buffer=ggml_backend_alloc_ctx_tensors(s.resident,s.backend);require(s.buffer,"packed resident allocation failed");for(auto it=r.at("weights").begin();it!=r.at("weights").end();++it)load_weight(weights.at(it.key()),it.value(),model,model_bytes);
 std::vector<float> hidden_data(static_cast<size_t>(hidden)*tokens);
 if(r.at("input_source")=="external_f32_binary"){std::ifstream data(std::filesystem::u8path(r.at("input_f32_file").get<std::string>()),std::ios::binary);require(data.good(),"external input missing");data.read(reinterpret_cast<char*>(hidden_data.data()),static_cast<std::streamsize>(hidden_data.size()*sizeof(float)));require(data.gcount()==static_cast<std::streamsize>(hidden_data.size()*sizeof(float))&&data.peek()==std::char_traits<char>::eof(),"external input exact byte count differs");}
 else{require(r.at("input_source")=="declared_synthetic_f32","unknown input source");std::mt19937 rng(r.at("seed").get<uint32_t>());for(auto &v:hidden_data)v=(static_cast<int32_t>(rng()>>8)-8388608)/8388608.0f;}
 require(std::all_of(hidden_data.begin(),hidden_data.end(),[](float f){return std::isfinite(f);}),"nonfinite input");ggml_backend_tensor_set(x,hidden_data.data(),0,ggml_nbytes(x));s.allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(s.backend));require(s.allocator&&ggml_gallocr_alloc_graph(s.allocator,s.graph),"exact graph allocation failed");
 std::vector<uint8_t> foreign(foreign_bytes,0);uint64_t touch_checksum=0;
 auto condition=[&](){volatile uint8_t *p=foreign.data();for(size_t i=0;i<foreign.size();i+=64){p[i]=static_cast<uint8_t>(p[i]+1);touch_checksum+=p[i];}};
 const auto warmup_begin=tick();int actual_warmup=0;do{condition();completed_compute(s);++actual_warmup;require(actual_warmup<1000000,"warmup iteration envelope exceeded");}while(actual_warmup<warmup||ns(warmup_begin,tick())<static_cast<int64_t>(warmup_ms)*1000000LL);
 std::vector<int64_t> samples_ns;for(int i=0;i<samples;++i){condition();const auto a=tick();completed_compute(s);samples_ns.push_back(ns(a,tick()));}
 std::vector<float> out(static_cast<size_t>(ggml_nelements(y)));ggml_backend_tensor_get(y,out.data(),0,ggml_nbytes(y));require(std::all_of(out.begin(),out.end(),[](float v){return std::isfinite(v);}),"nonfinite subgraph output");
 json routes=nullptr;if(selected){std::vector<int32_t> ids(static_cast<size_t>(ggml_nelements(selected)));const auto *src=selected->view_src?selected->view_src:selected;std::vector<uint8_t> raw(ggml_nbytes(selected));ggml_backend_tensor_get(selected,raw.data(),0,raw.size());for(int j=0;j<tokens;++j)for(int k=0;k<8;++k)std::memcpy(&ids[j*8+k],raw.data()+j*selected->nb[1]+k*selected->nb[0],sizeof(int32_t));routes=ids;(void)src;}
 const size_t cpu_workspace=backend=="CPU"?ggml_graph_plan(s.graph,threads,s.pool).work_size:0;
 json result{{"schema","real-moe-quantized-subgraph-native-1"},{"request",r},{"source_sha256",QUANTIZED_SOURCE_SHA256},{"process_id",GetCurrentProcessId()},{"observed_process_mask",static_cast<uint64_t>(actual_mask)},
  {"samples_ns",samples_ns},{"graph_nodes",graph_inventory(s)},{"actual_weight_types",json::object()},{"actual_backend",ggml_backend_name(s.backend)},{"resident_allocated_bytes",ggml_backend_buffer_get_size(s.buffer)},{"graph_allocated_bytes",ggml_gallocr_get_buffer_size(s.allocator,0)},
  {"workspace_bytes",backend=="CPU"?json(cpu_workspace):json(nullptr)},{"workspace_source",backend=="CPU"?"GGML CPU graph-plan work_size":"CUDA private workspace not exported"},{"observed_routes_flat",routes},{"output_finite",true},
  {"scope",operation=="olmoe_cpu_ffn"?"homogeneous CPU post-FFN-norm input to MoE output; excludes layer norm, residual, inter-backend copies and llama scheduler":"one completed regular packed-weight MUL_MAT"},
  {"graph_policy","runtime_default_enabled"},{"actual_graph_capture_use",nullptr},{"clock",json{{"clock_id","windows_qpc"},{"frequency_hz",hz()}}},{"packed_weights_read_from_model",true},{"weights_dequantized_or_substituted",false},{"external_input_supplied",r.at("input_source")=="external_f32_binary"},{"original_model_activation_origin_verified",false},
  {"warmup_actual_iterations",actual_warmup},{"warmup_min_ms",warmup_ms},{"warmup_in_target_samples",false},
  {"cache_context",cache_context},{"foreign_working_set_bytes",foreign_bytes},{"foreign_touch_checksum",touch_checksum},{"cache_conditioning_in_target_interval",false},{"hardware_cache_residency_directly_verified",false},
  {"cpu_threadpool_policy",r.at("cpu_threadpool_policy")},
  {"graph_source_homomorphism_reviewed",false},{"original_model_output_equivalence_verified",false},{"whole_model_accuracy_certified",false},{"actual_backend_partition_equals_whole_runtime",false}};
 for(const auto &w:weights)result["actual_weight_types"][w.first]=ggml_type_name(w.second->type);write_new(output,result);return 0;
 }catch(const std::exception &e){std::cerr<<"real-moe-quantized-probe: "<<e.what()<<'\n';return 2;}}
