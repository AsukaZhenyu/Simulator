// Revision v4: observe CPU-consumed expert IDs; no llama scheduler callback.
// Derived from frozen v3; new private executable/runtime only.
#define NOMINMAX
#include "llama.h"
#include "llama-model.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "nlohmann/json.hpp"
#include <windows.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
using json=nlohmann::json;
static void require(bool ok,const std::string &message){if(!ok)throw std::runtime_error(message);}
static int64_t tick(){LARGE_INTEGER n{};require(QueryPerformanceCounter(&n)!=0,"QPC failed");return n.QuadPart;}
static int64_t frequency(){LARGE_INTEGER n{};require(QueryPerformanceFrequency(&n)!=0,"QPC frequency failed");return n.QuadPart;}
static int64_t elapsed(int64_t a,int64_t b){const auto hz=frequency();return (b-a)/hz*1000000000LL+(b-a)%hz*1000000000LL/hz;}
static bool hash_format(const std::string &v){return v.size()==64&&v.find_first_not_of("0123456789abcdef")==std::string::npos;}
static void write_new(const std::filesystem::path &path,const json &value){
 const std::string data=value.dump()+"\n";
 HANDLE file=CreateFileW(path.c_str(),GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
 require(file!=INVALID_HANDLE_VALUE,"exclusive output creation failed");
 size_t done=0;bool ok=true;
 while(done<data.size()){DWORD n=0;const DWORD count=static_cast<DWORD>(std::min<size_t>(data.size()-done,1<<20));
  if(!WriteFile(file,data.data()+done,count,&n,nullptr)||n!=count){ok=false;break;}done+=n;}
 CloseHandle(file);require(ok,"native output write failed");
}

#include "route_tap_pod.hpp"
#include "route_context_inspect.hpp"

struct ModelOwner{llama_model *value=nullptr;~ModelOwner(){if(value)llama_model_free(value);}};
struct ContextOwner{llama_context *value=nullptr;~ContextOwner(){if(value)llama_free(value);}};
struct ThreadpoolOwner{ggml_threadpool_t value=nullptr;~ThreadpoolOwner(){if(value)ggml_threadpool_free(value);}};
struct SamplerOwner{llama_sampler *value=nullptr;~SamplerOwner(){if(value)llama_sampler_free(value);}};
struct BatchOwner{llama_batch value;explicit BatchOwner(int count):value(llama_batch_init(count,0,1)){}~BatchOwner(){llama_batch_free(value);}};
struct ForwardStamp{int index=0,position=0,tokens=0,output_rows=0;bool decode=false;int64_t begin=0,end=0;};

static std::vector<llama_token> tokenize(const llama_vocab *v,const std::string &prompt){
 // The pinned official chat template already renders BOS/EOS. Adding another
 // model-default BOS here would silently change the actual request.
 const int n=llama_tokenize(v,prompt.data(),static_cast<int32_t>(prompt.size()),nullptr,0,false,true);
 require(n<0,"prompt must tokenize to a nonempty sequence");std::vector<llama_token> result(-n);
 require(llama_tokenize(v,prompt.data(),static_cast<int32_t>(prompt.size()),result.data(),static_cast<int32_t>(result.size()),false,true)==-n,"tokenizer length changed");return result;
}
static void fill_batch(llama_batch &b,const std::vector<llama_token> &tokens,size_t first,int count,int position,bool last_output){
 b.n_tokens=count;for(int i=0;i<count;++i){b.token[i]=tokens[first+i];b.pos[i]=position+i;b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=last_output&&i==count-1;}
}
static json top_logits(llama_context *ctx,const llama_vocab *v){
 const auto *logits=llama_get_logits_ith(ctx,-1);require(logits,"final logits missing");
 const int n=llama_vocab_n_tokens(v);std::vector<std::pair<float,int>> top;
 for(int i=0;i<n;++i){require(std::isfinite(logits[i]),"nonfinite logits");top.emplace_back(logits[i],i);}
 const size_t k=std::min<size_t>(8,top.size());
 std::partial_sort(top.begin(),top.begin()+k,top.end(),[](const auto&a,const auto&b){return a.first>b.first||(a.first==b.first&&a.second<b.second);});
 json result=json::array();for(size_t i=0;i<k;++i)result.push_back(json{{"token",top[i].second},{"logit",top[i].first}});return result;
}
static json placement_inventory(llama_model *model){
 json totals=json::object(),items=json::array();
 for(const auto &item:model->tensors_by_name){const auto *t=item.second;const std::string buffer=t->buffer?ggml_backend_buffer_name(t->buffer):"unknown";
  const auto bytes=ggml_nbytes(t);totals[buffer]=totals.value(buffer,uint64_t(0))+bytes;
  items.push_back(json{{"name",item.first},{"dtype",ggml_type_name(t->type)},{"ne",{t->ne[0],t->ne[1],t->ne[2],t->ne[3]}},{"bytes",bytes},{"actual_buffer",buffer}});}
 return json{{"tensor_bytes_by_actual_buffer",totals},{"tensors",items},{"scope","model tensors only; not process RSS or activation peak"}};
}
static json model_metadata(llama_model *model){json meta=json::object();for(int i=0;i<llama_model_meta_count(model);++i){char key[1024]{},value[8192]{};
 const int nk=llama_model_meta_key_by_index(model,i,key,sizeof(key)),nv=llama_model_meta_val_str_by_index(model,i,value,sizeof(value));
 if(nk>0&&nk<static_cast<int>(sizeof(key))&&nv>=0&&nv<static_cast<int>(sizeof(value)))meta[key]=value;
 else meta["unrepresented_metadata_index_"+std::to_string(i)]=json{{"key_length",nk},{"value_length",nv}};}
 return meta;}
static json verified_contract(const json &meta,const json &expected){
 const auto arch=expected.at("architecture").get<std::string>();
 require(meta.at("general.architecture")==arch,"GGUF architecture differs from declared model contract");
 json observed{{"architecture",arch},{"layers",std::stoi(meta.at(arch+".block_count").get<std::string>())},
  {"experts",std::stoi(meta.at(arch+".expert_count").get<std::string>())},
  {"top_k",std::stoi(meta.at(arch+".expert_used_count").get<std::string>())}};
 require(observed==expected,"GGUF model geometry differs from declared model contract");return observed;
}

static json execute(llama_context *ctx,const llama_vocab *v,const json &request,Capture &capture,bool record){
 const auto tokens=request.at("prompt_token_ids").get<std::vector<llama_token>>();
 require(!tokens.empty(),"prompt tokens empty");const int chunk=request.at("batch_size").get<int>();
 const int max_new=request.at("max_output_tokens").get<int>();require(chunk>0&&chunk<=4096&&max_new>=1&&max_new<=4096,"request exceeds explicit envelope");
 const bool fixed=request.contains("output_token_ids");std::vector<llama_token> wanted=fixed?request.at("output_token_ids").get<std::vector<llama_token>>():std::vector<llama_token>{};
 require(!fixed||(!wanted.empty()&&wanted.size()<=static_cast<size_t>(max_new)),"invalid frozen output token list");
 const int outputs=fixed?static_cast<int>(wanted.size()):max_new;
 require(tokens.size()+outputs<=llama_n_ctx(ctx),"request exceeds declared context capacity");
 llama_memory_clear(llama_get_memory(ctx),true);llama_perf_context_reset(ctx);
 SamplerOwner sampler;sampler.value=llama_sampler_init_greedy();require(sampler.value,"greedy sampler unavailable");
 BatchOwner batch(chunk);std::vector<ForwardStamp> forward_stamps;forward_stamps.reserve((tokens.size()+chunk-1)/chunk+outputs);std::vector<llama_token> generated;generated.reserve(outputs);
 std::vector<int64_t> decode_ns;decode_ns.reserve(outputs);std::vector<int64_t> emission_ticks;emission_ticks.reserve(outputs);
 std::vector<bool> token_matches;token_matches.reserve(outputs);int forward=0;int64_t prefill_end=0;bool eog=false;
 const bool observe_logits=request.value("observe_logits",false),observe_graph=request.value("observe_graph",false);
 const int vocab_size=llama_vocab_n_tokens(v);require(!observe_logits||uint64_t(outputs)*vocab_size<=20000000,"bounded logit observation required");
 std::vector<float> logit_snapshots(observe_logits?size_t(outputs)*vocab_size:0);std::vector<int> sampled_tokens;sampled_tokens.reserve(outputs);
 json graph_observations=json::array();int64_t logit_copy_ns=0,graph_observer_ns=0;
 if(record)capture.prepare(static_cast<int>((tokens.size()+chunk-1)/chunk)+outputs,static_cast<int>(tokens.size())+outputs,capture.layers,capture.experts,capture.top_k);
 capture.enabled=record;const auto begin=tick();
 for(size_t first=0;first<tokens.size();first+=chunk){const int n=static_cast<int>(std::min<size_t>(chunk,tokens.size()-first));
  fill_batch(batch.value,tokens,first,n,static_cast<int>(first),first+n==tokens.size());
  if(record){capture.forward=forward;capture.decode=false;capture.position=static_cast<int>(first);capture.submitted=n;}
  const auto a=tick();require(llama_decode(ctx,batch.value)==0,"prefill llama_decode failed");const auto b=tick();
  forward_stamps.push_back({forward++,static_cast<int>(first),n,first+n==tokens.size()?1:0,false,a,b});
  if(observe_graph){const auto x=tick();auto graph=tap_graph_summary(ctx);graph["forward_index"]=forward-1;graph_observations.push_back(std::move(graph));graph_observer_ns+=elapsed(x,tick());}
 }
 llama_synchronize(ctx);prefill_end=tick();
 for(int i=0;i<outputs;++i){const auto sampled=llama_sampler_sample(sampler.value,ctx,-1);sampled_tokens.push_back(sampled);
  if(observe_logits){const auto x=tick();const auto *values=llama_get_logits_ith(ctx,-1);require(values,"step logits missing");std::memcpy(logit_snapshots.data()+size_t(i)*vocab_size,values,size_t(vocab_size)*sizeof(float));logit_copy_ns+=elapsed(x,tick());}
  const auto id=fixed?wanted[i]:sampled;token_matches.push_back(!fixed||sampled==id);generated.push_back(id);llama_sampler_accept(sampler.value,id);emission_ticks.push_back(tick());
  eog=llama_vocab_is_eog(v,id);
  if(eog||i+1==outputs)break; // No forward is invented for the final sampled token.
  batch.value.n_tokens=1;batch.value.token[0]=id;batch.value.pos[0]=static_cast<int>(tokens.size())+i;batch.value.n_seq_id[0]=1;batch.value.seq_id[0][0]=0;batch.value.logits[0]=true;
  if(record){capture.forward=forward;capture.decode=true;capture.position=static_cast<int>(tokens.size())+i;capture.submitted=1;}
  const auto a=tick();require(llama_decode(ctx,batch.value)==0,"decode llama_decode failed");llama_synchronize(ctx);const auto b=tick();
  decode_ns.push_back(elapsed(a,b));forward_stamps.push_back({forward++,static_cast<int>(tokens.size())+i,1,1,true,a,b});
  if(observe_graph){const auto x=tick();auto graph=tap_graph_summary(ctx);graph["forward_index"]=forward-1;graph_observations.push_back(std::move(graph));graph_observer_ns+=elapsed(x,tick());}
 }
 const auto end=tick();capture.enabled=false;const auto perf=llama_perf_context(ctx);
 json logit_observations=json::array();if(observe_logits)for(size_t i=0;i<generated.size();++i){auto l=tap_logit_summary(logit_snapshots.data()+i*vocab_size,vocab_size,sampled_tokens[i]);l["output_index"]=i;logit_observations.push_back(std::move(l));}
 json forwards=json::array();for(const auto &s:forward_stamps){json row{{"index",s.index},{"phase",s.decode?"decode":"prefill"},{"position",s.position},{"tokens",s.tokens},{"output_rows",s.output_rows},{"begin_tick",s.begin},{"end_tick",s.end},{"completed_wall_ns",s.decode?json(elapsed(s.begin,s.end)):json(nullptr)},{"host_call_is_device_completion",s.decode}};forwards.push_back(std::move(row));}
 json emission=json::array();for(size_t i=0;i<emission_ticks.size();++i)emission.push_back(json{{"index",i},{"tick",emission_ticks[i]},{"since_begin_ns",elapsed(begin,emission_ticks[i])},{"interval_ns",i?json(elapsed(emission_ticks[i-1],emission_ticks[i])):json(nullptr)}});
 return json{{"id",request.at("id")},{"prompt_token_ids",tokens},{"output_token_ids",generated},{"teacher_forced",fixed},
  {"greedy_matches_frozen_token",token_matches},{"all_greedy_tokens_match",std::all_of(token_matches.begin(),token_matches.end(),[](bool b){return b;})},
  {"stopped_on_eog",eog},{"begin_tick",begin},{"prefill_end_tick",prefill_end},{"end_tick",end},
  {"prefill_completed_ns",elapsed(begin,prefill_end)},{"core_decode_forward_ns",decode_ns},{"forward_inventory",forwards},
  {"forward_count",forward},{"decode_forward_count",decode_ns.size()},{"output_count",generated.size()},
  {"token_ready_ttft_ns",emission_ticks.empty()?json(nullptr):json(elapsed(begin,emission_ticks.front()))},
  {"whole_request_wall_ns",elapsed(begin,end)},{"emissions",emission},{"final_top_logits",top_logits(ctx,v)},
  {"sampled_greedy_token_ids",sampled_tokens},{"logit_observations",logit_observations},{"graph_observations",graph_observations},
  {"observer",json{{"cpu_consumer_tap",record},{"step_logit_copy",observe_logits},{"graph_metadata",observe_graph},{"tap_callback_ns",record?elapsed(0,capture.hook_ticks):0},{"logit_copy_ns",logit_copy_ns},{"graph_metadata_ns",graph_observer_ns},{"overhead_is_inclusive_host_observation_not_GPU_active_time",true}}},
  {"llama_perf",json{{"prompt_tokens",perf.n_p_eval},{"decode_tokens",perf.n_eval},{"prompt_ms",perf.t_p_eval_ms},{"decode_ms",perf.t_eval_ms}}},
  {"scope","token-ready local request; excludes model load/tokenization/reset/warmup; includes native loop and greedy sampling; core_decode_forward_ns excludes sampling"},
  {"timing_usable_as_bare_reference",!record&&!observe_logits&&!observe_graph},{"trace_timing_is_performance_truth",false}};
}

int main(int argc,char **argv){try{
 if(argc==2&&std::string(argv[1])=="--self-test"){
  int layer=-1;require(Capture::stage_layer("ffn_moe_up-15",layer)==0&&layer==15,"consumer stage parser failed");require(Capture::stage_layer("result_norm",layer)==-1,"nonexpert node admitted");
  require(hash_format(std::string(64,'a'))&&!hash_format("unknown"),"SHA parser failed");
  std::cout<<json{{"passed",true},{"model_loaded",false},{"hardware_timing_executed",false},{"source_sha256",REAL_MOE_SOURCE_SHA256}}.dump()<<'\n';return 0;}
 require(argc==3,"usage: real-moe-probe REQUEST OUTPUT");const auto output=std::filesystem::u8path(argv[2]);require(!std::filesystem::exists(output),"output exists");
 std::ifstream stream(std::filesystem::u8path(argv[1]));require(stream.good(),"request unreadable");json request;stream>>request;
 require(request.at("schema")=="real-moe-tap-request-4","request schema differs");require(request.at("source_sha256")==REAL_MOE_SOURCE_SHA256,"native source SHA differs");
 require(request.at("tokenization_add_special")==false,"official rendered prompts require add_special=false");
 const auto mode=request.at("mode").get<std::string>();require(mode=="tokenize-only"||mode=="generate"||mode=="replay"||mode=="capture","unknown mode");
 require(request.at("threadpool_policy")==json(mode=="tokenize-only"?"metadata_only_no_compute":"explicit_persistent_owned"),"explicit CPU threadpool lifetime differs");
 require(hash_format(request.at("model_sha256").get<std::string>()),"explicit pinned model SHA required");
 require(request.at("cuda_graph_policy")=="runtime_default_enabled"&&std::getenv("GGML_CUDA_DISABLE_GRAPHS")==nullptr,"CUDA graph environment differs from explicit request");
 const char *tf32=std::getenv("NVIDIA_TF32_OVERRIDE");require(request.at("tf32_override")=="0"&&tf32&&std::string(tf32)=="0","explicit TF32 flag differs");
 const auto model_path=std::filesystem::u8path(request.at("model_path").get<std::string>());require(std::filesystem::is_regular_file(model_path),"model missing");
 require(std::filesystem::file_size(model_path)==request.at("model_bytes").get<uint64_t>(),"pinned model byte count differs");
 const auto placement=request.at("placement").get<std::string>();require(placement=="cpu"||placement=="cpu-experts-gpu-other"||placement=="cpu-attn-gpu-other","unknown placement");const bool attn_on_cpu=placement=="cpu-attn-gpu-other";
 const int threads=request.at("threads").get<int>(),ctx_size=request.at("context_size").get<int>(),batch_size=request.at("batch_size").get<int>();
 require(threads>=1&&threads<=32&&ctx_size>=128&&ctx_size<=16384&&batch_size>=1&&batch_size<=4096,"explicit context envelope exceeded");
 const auto requested_mask=request.at("expected_process_mask").get<uint64_t>();DWORD_PTR before_mask=0,system_mask=0,actual_mask=0,after_system=0;
 require(GetProcessAffinityMask(GetCurrentProcess(),&before_mask,&system_mask)!=0&&requested_mask>0&&(requested_mask&system_mask)==requested_mask,"explicit process affinity invalid");
 require(SetProcessAffinityMask(GetCurrentProcess(),static_cast<DWORD_PTR>(requested_mask))!=0&&GetProcessAffinityMask(GetCurrentProcess(),&actual_mask,&after_system)!=0&&actual_mask==requested_mask,"owned process affinity enforcement failed");
 MEMORYSTATUSEX host_memory{};host_memory.dwLength=sizeof(host_memory);require(GlobalMemoryStatusEx(&host_memory)!=0,"host capacity observation failed");
 require(mode=="tokenize-only"||host_memory.ullAvailPhys>=request.at("minimum_free_host_bytes").get<uint64_t>(),"host free-memory precondition failed");
 ggml_backend_load_all();llama_backend_init();ModelOwner model;auto mp=llama_model_default_params();
 mp.n_gpu_layers=0;mp.use_mmap=true;mp.use_mlock=false;mp.vocab_only=mode=="tokenize-only";
 std::vector<ggml_backend_dev_t> devices;std::vector<llama_model_tensor_buft_override> overrides;
 if(placement!="cpu"&&mode!="tokenize-only"){
  auto gpu=ggml_backend_dev_by_name("CUDA0");require(gpu&&ggml_backend_dev_type(gpu)==GGML_BACKEND_DEVICE_TYPE_GPU,"explicit CUDA0 unavailable; no fallback");
  size_t free=0,total=0;ggml_backend_dev_memory(gpu,&free,&total);require(free>=request.at("minimum_free_gpu_bytes").get<uint64_t>(),"GPU free-memory precondition failed");
  devices={gpu,nullptr};mp.devices=devices.data();mp.n_gpu_layers=std::numeric_limits<int32_t>::max();mp.split_mode=LLAMA_SPLIT_MODE_NONE;
  overrides=attn_on_cpu?std::vector<llama_model_tensor_buft_override>{{"blk\\.\\d+\\.attn_.*",ggml_backend_cpu_buffer_type()},{nullptr,nullptr}}:std::vector<llama_model_tensor_buft_override>{{"\\.ffn_(up|down|gate|gate_up)_(ch|)exps",ggml_backend_cpu_buffer_type()},{nullptr,nullptr}};mp.tensor_buft_overrides=overrides.data();
 }
 const auto load_begin=tick();model.value=llama_model_load_from_file(model_path.string().c_str(),mp);const auto load_end=tick();require(model.value,"explicit real model load failed");
 const auto metadata=model_metadata(model.value),contract=verified_contract(metadata,request.at("model_contract"));
 require(mode=="tokenize-only"||llama_model_n_layer(model.value)==contract.at("layers").get<int>(),"loaded layer geometry differs");
 const auto *vocab=llama_model_get_vocab(model.value);json normalized=json::array();
 for(auto row:request.at("requests")){
  if(row.contains("prompt_token_ids")){require(row.at("prompt_token_ids").is_array()&&!row.at("prompt_token_ids").empty(),"explicit prompt tokens invalid");}
  else row["prompt_token_ids"]=tokenize(vocab,row.at("prompt").get<std::string>());
  row["batch_size"]=batch_size;normalized.push_back(std::move(row));
 }
 json result{{"schema","real-moe-tap-native-4"},{"request",request},{"mode",mode},{"process_id",GetCurrentProcessId()},
  {"source_sha256",REAL_MOE_SOURCE_SHA256},{"clock",json{{"clock_id","windows_qpc"},{"frequency_hz",frequency()}}},
  {"process_affinity",json{{"requested_mask",requested_mask},{"initial_mask",static_cast<uint64_t>(before_mask)},{"observed_mask",static_cast<uint64_t>(actual_mask)},{"enforced_before_model_load",true}}},
  {"host_capacity_before_model_load",json{{"total_bytes",host_memory.ullTotalPhys},{"available_bytes",host_memory.ullAvailPhys}}},
  {"model_load_wall_ns",elapsed(load_begin,load_end)},{"model_sha256_verified_by_parent",request.at("model_sha256")},
  {"native_model_hash_verification","byte-count only; parent must bind and verify full content hash before collection"},
  {"model_metadata",metadata},{"model_layers",llama_model_n_layer(model.value)},
  {"model_contract_verified",true},{"observed_model_contract",contract},
  {"tokenization_add_special",false},{"model_default_add_bos",llama_vocab_get_add_bos(vocab)},
  {"threadpool_policy",request.at("threadpool_policy")},
  {"automatic_fit_used",false},{"actual_placement",mode=="tokenize-only"?json(nullptr):placement_inventory(model.value)},
  {"observed_runtime_flags",json{{"GGML_CUDA_DISABLE_GRAPHS",nullptr},{"NVIDIA_TF32_OVERRIDE",tf32},{"actual_graph_capture_use",nullptr},{"actual_graph_capture_use_status","not exported by backend"}}},
  {"all_prompt_tokens_marked_output",false},{"extra_final_decode",false},{"parameters_fitted",false},{"hardware_accuracy_qualified",false}};
 if(mode=="tokenize-only"){result["tokenized_requests"]=normalized;result["model_weights_loaded"]=false;write_new(output,result);return 0;}
 Capture capture;capture.layers=contract.at("layers");capture.experts=contract.at("experts");capture.top_k=contract.at("top_k");ThreadpoolOwner pool;ContextOwner context;auto cp=llama_context_default_params();cp.n_ctx=ctx_size;cp.n_batch=batch_size;cp.n_ubatch=batch_size;cp.n_seq_max=1;cp.n_threads=threads;cp.n_threads_batch=threads;
 cp.offload_kqv=placement!="cpu"&&!attn_on_cpu;cp.op_offload=placement!="cpu"&&!attn_on_cpu;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.no_perf=false;
 require(cp.cb_eval==nullptr,"v4 must preserve scheduler callback-free execution");
 context.value=llama_init_from_model(model.value,cp);require(context.value,"explicit context initialization failed");
 auto pool_params=ggml_threadpool_params_default(threads);pool.value=ggml_threadpool_new(&pool_params);require(pool.value,"owned persistent CPU threadpool creation failed");
 llama_attach_threadpool(context.value,pool.value,pool.value);
 result["cpu_threadpool"]=json{{"policy","explicit_persistent_owned"},{"threads",threads},{"shared_prefill_decode_pool",true},{"attached_before_warmup",true},{"creation_in_target_interval",false}};
 TapBinding tap_binding(capture,mode=="capture");
 const int warmups=request.at("warmup_replays").get<int>();require(warmups>=0&&warmups<=3,"warmup envelope exceeded");
 json results=json::array();
 for(auto row:normalized){
  if(mode=="replay")require(row.contains("output_token_ids"),"replay requires frozen output token IDs");
  row["observe_logits"]=request.value("observe_logits",false);row["observe_graph"]=request.value("observe_graph",false);
  for(int w=0;w<warmups;++w)execute(context.value,vocab,row,capture,false);
  auto measured=execute(context.value,vocab,row,capture,mode=="capture");
  measured["routes"]=mode=="capture"?capture.output():json::array();measured["normal_output_pruning_preserved"]=true;measured["warmup_replays"]=warmups;
  results.push_back(std::move(measured));
 }
 result["results"]=results;result["model_weights_loaded"]=true;result["trace_callback_enabled"]=false;result["cpu_consumer_tap_enabled"]=mode=="capture";result["scheduler_callback_enabled"]=false;
 result["tap_hook_available"]=tap_binding.setter!=nullptr;result["graph_or_allocator_modified"]=false;
 result["trace_callback_asks_all_nodes"]=false;result["timing_scope"]="normal output mask; callback-free CPU-consumer tap; capture or logit/graph observer timings are diagnostic only";
 write_new(output,result);return 0;
 }catch(const std::exception &error){std::cerr<<"real-moe-probe: "<<error.what()<<'\n';return 2;}}
