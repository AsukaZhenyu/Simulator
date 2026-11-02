// First response and sustained COPY windows, without a device-time observer.
#include "../v3/common.hpp"
#include <tuple>

namespace lifecycle_v10 {
using namespace diagnostic_v3;
constexpr size_t arena_bytes=8388608;
struct CopyArena {
 ggml_context *host_ctx=nullptr,*device_ctx=nullptr;
 ggml_backend_buffer_t host_buffer=nullptr,device_buffer=nullptr;
 ggml_tensor *source=nullptr,*target=nullptr;
 CopyArena(Executor &e,ggml_tensor *seed){
  host_ctx=ggml_init({ggml_tensor_overhead()*4,nullptr,true});device_ctx=ggml_init({ggml_tensor_overhead()*4,nullptr,true});require(host_ctx&&device_ctx,"copy arena metadata failed");
  source=ggml_new_tensor_1d(host_ctx,GGML_TYPE_F32,arena_bytes/4);target=ggml_new_tensor_1d(device_ctx,GGML_TYPE_F32,arena_bytes/4);
  host_buffer=ggml_backend_alloc_ctx_tensors(host_ctx,e.cpu);device_buffer=ggml_backend_alloc_ctx_tensors(device_ctx,e.gpu);require(host_buffer&&device_buffer,"owned copy arena allocation failed");
  std::vector<uint8_t> pattern(ggml_nbytes(seed)),value(arena_bytes);ggml_backend_tensor_get(seed,pattern.data(),0,pattern.size());
  for(size_t i=0;i<value.size();++i)value[i]=pattern[i%pattern.size()];ggml_backend_tensor_set(source,value.data(),0,value.size());
 }
 ~CopyArena(){if(host_buffer)ggml_backend_buffer_free(host_buffer);if(device_buffer)ggml_backend_buffer_free(device_buffer);if(host_ctx)ggml_free(host_ctx);if(device_ctx)ggml_free(device_ctx);}
};
class HostRegistration {
 void *base_;size_t bytes_;bool active_=false,closed_=false;json proof_;
public:
 HostRegistration(CopyArena &a,const std::string &mode){
  base_=ggml_backend_buffer_get_base(a.host_buffer);bytes_=ggml_backend_buffer_get_size(a.host_buffer);require(base_&&bytes_>=arena_bytes,"owned host arena unavailable");
  const auto before=pointer_kind(base_);require(before.at("unregistered_pageable")==true,"initial copy arena not pageable");
  if(mode=="registered"){require(cudaHostRegister(base_,bytes_,cudaHostRegisterDefault)==cudaSuccess,"owned registration failed; no fallback");active_=true;}
  const auto during=pointer_kind(base_);require(during.at(mode=="registered"?"registered_host":"unregistered_pageable")==true,"observed memory mode differs");
  proof_=json{{"mode",mode},{"before",before},{"during",during},{"arena_bytes",bytes_},{"registration_timed",false},{"scope","separate owned eight-MiB COPY arena; original preparation arena unchanged"}};
 }
 ~HostRegistration(){if(active_)cudaHostUnregister(base_);}
 json finish(){require(!closed_,"registration already closed");if(active_){require(cudaHostUnregister(base_)==cudaSuccess,"unregister failed");active_=false;}
  proof_["after"]=pointer_kind(base_);require(proof_.at("after").at("unregistered_pageable")==true,"copy arena not restored");proof_["restored"]=true;closed_=true;return proof_;}
};
struct Alias {size_t bytes;std::string id;ggml_tensor *source,*target;json action;};
class Views {
 Executor &e_;ggml_context *ctx_;
public:
 std::vector<Alias> values;
 Views(Executor &e,CopyArena &arena,const json &action,const json &sizes):e_(e){
  ctx_=ggml_init({ggml_tensor_overhead()*32,nullptr,true});require(ctx_,"view metadata allocation failed");
  for(const auto &size:sizes){size_t bytes=size.get<size_t>();require(bytes>0&&bytes<=arena_bytes&&bytes%4==0,"invalid copy extent");
   Alias a{bytes,str(action,"target_id")+":lifecycle_v10:"+std::to_string(bytes),ggml_view_1d(ctx_,arena.source,bytes/4,0),ggml_view_1d(ctx_,arena.target,bytes/4,0),action};
   require(ggml_backend_view_init(a.source)==GGML_STATUS_SUCCESS&&ggml_backend_view_init(a.target)==GGML_STATUS_SUCCESS,"view initialization failed");
   require(a.source->data==arena.source->data&&a.target->data==arena.target->data&&ggml_nbytes(a.source)==bytes&&ggml_nbytes(a.target)==bytes,"view range differs");
   a.action["target_id"]=a.id;Tensor t;t.dram=a.source;t.vram=a.target;t.descriptor=json{{"id",a.id},{"initial_locations",json::array()},{"storage_bytes",bytes}};
   require(e_.tensors.emplace(a.id,std::move(t)).second,"copy alias identity collision");values.push_back(std::move(a));
  }
 }
 ~Views(){for(const auto &a:values)e_.tensors.erase(a.id);if(ctx_)ggml_free(ctx_);}
};
static json qualify(Executor &e,const Alias &a,CopyArena &arena){
 std::vector<uint8_t> expected(arena_bytes),actual(arena_bytes);ggml_backend_tensor_get(arena.source,expected.data(),0,arena_bytes);
 ggml_backend_tensor_memset(arena.target,0xa5,0,arena_bytes);ggml_backend_synchronize(e.gpu);ggml_backend_tensor_get(arena.target,actual.data(),0,arena_bytes);
 require(!std::equal(expected.begin(),expected.begin()+a.bytes,actual.begin()),"vacuous copy oracle");e.action(a.action);ggml_backend_tensor_get(arena.target,actual.data(),0,arena_bytes);
 require(std::equal(expected.begin(),expected.begin()+a.bytes,actual.begin()),"copy range differs");require(std::all_of(actual.begin()+a.bytes,actual.end(),[](uint8_t x){return x==0xa5;}),"copy overwrote outside view");
 std::vector<uint8_t> unchanged(arena_bytes);ggml_backend_tensor_get(arena.source,unchanged.data(),0,arena_bytes);require(unchanged==expected,"source modified");
 return json{{"bytes",a.bytes},{"selected_range_initially_different",true},{"selected_bytes_equal_after_copy",true},{"outside_view_unchanged",true},{"timed",false}};
}
static void contract(const json &r) {
 const auto split=str(r,"split");require(split=="profile"||split=="heldout","explicit split required");
 require(r.at("sizes_bytes")== (split=="profile"?json::array({1024,131072,1048576,8388608}):json::array({4096,65536,524288,2097152,4194304})),"size split differs");
 require(r.at("idle_ms")== (split=="profile"?json::array({0,1,5,20}):json::array({2,10})),"idle split differs");
 require(r.at("windows")== (split=="profile"?json::array({1,16,128}):json::array({8,64})),"window split differs");
 require(str(r,"memory_mode")=="registered"||str(r,"memory_mode")=="pageable","explicit memory required");
 require(r.at("performance_boundary")=="completed_original_Executor.action_COPY"&&r.at("telemetry_between_bursts")==false,"bare response boundary differs");
 const auto &conditions=r.at("conditions"),&orders=r.at("condition_orders");
 require(conditions.size()==r.at("sizes_bytes").size()*r.at("idle_ms").size()*r.at("windows").size(),"condition inventory differs");
 std::set<std::tuple<int,int,int>> seen;
 for(const auto &c:conditions) {
  int s=c.at("size_index"),i=c.at("idle_index"),n=c.at("window_index");
  require(s>=0&&size_t(s)<r.at("sizes_bytes").size()&&i>=0&&size_t(i)<r.at("idle_ms").size()&&n>=0&&size_t(n)<r.at("windows").size(),"invalid condition");
  require(seen.emplace(s,i,n).second,"duplicate condition");
 }
 require(orders.is_array()&&orders.size()==12,"twelve complete trials required");
 for(const auto &order:orders){std::set<int> indices;require(order.size()==conditions.size(),"order incomplete");
  for(const auto &v:order){require(v.is_number_integer(),"order integer required");int i=v;require(i>=0&&size_t(i)<conditions.size()&&indices.insert(i).second,"order not permutation");}}
}
struct Burst {
 int trial,slot,condition,size_index,idle,copies_count;
 int64_t prime_begin,prime_end,idle_begin,idle_end,begin,end;
 std::vector<Copy> copies;
};
static json run(Executor &e,const steady::Request &sampling,const json &r) {
 contract(r);require(e.gpu&&e.program.at("physical").at("threads")==1&&sampling.mode=="action"&&sampling.trials==12&&sampling.poll=="0"&&sampling.instrumentation=="bare"&&sampling.main_mask==4,"one-thread bare COPY required");
 const auto &actions=e.program.at("actions"),&target=actions.at(size_t(sampling.target));
 require(target.at("kind")=="COPY"&&target.at("src")=="dram"&&target.at("dst")=="vram","H2D target required");
 auto *seed=e.tensor(str(target,"target_id"),"dram");
 const auto pool=single_layer::configure_pool(e,"0");e.validate();const auto initial=single_layer::capture_verification(e);
 CopyArena arena(e,seed);auto *src=arena.source,*dst=arena.target;HostRegistration registration(arena,str(r,"memory_mode"));Views views(e,arena,target,r.at("sizes_bytes"));
 json oracles=json::array();for(const auto &a:views.values)oracles.push_back(qualify(e,a,arena));
 auto full=[&](){for(const auto &a:actions)e.action(a);};e.reset();full();const auto warm=steady::warm(sampling,full);const auto pointer=pointer_kind(src->data);
 dynamic_control::OwnedHighResolutionTimer timer;std::vector<Burst> bursts;
 bursts.reserve(12*r.at("conditions").size());
 for(int trial=0;trial<12;++trial)for(size_t slot=0;slot<r.at("conditions").size();++slot){
  Burst b;b.trial=trial;b.slot=int(slot);b.condition=r.at("condition_orders").at(trial).at(slot);const auto &c=r.at("conditions").at(size_t(b.condition));
  b.size_index=c.at("size_index");b.idle=r.at("idle_ms").at(c.at("idle_index").get<size_t>());b.copies_count=r.at("windows").at(c.at("window_index").get<size_t>());
  b.copies.resize(size_t(b.copies_count));e.reset();b.prime_begin=steady::tick();full();b.prime_end=steady::tick();b.idle_begin=steady::tick();
  if(b.idle)timer.wait_ns(int64_t(b.idle)*1000000);b.idle_end=steady::tick();
  const auto &a=views.values.at(size_t(b.size_index));b.begin=steady::tick();
  for(auto &copy:b.copies){copy.begin=steady::tick();e.action(a.action);copy.end=steady::tick();}b.end=steady::tick();bursts.push_back(std::move(b));
 }
 const size_t checked_bytes=r.at("sizes_bytes").back().get<size_t>();std::vector<uint8_t> expected(checked_bytes),actual(checked_bytes);
 ggml_backend_tensor_get(src,expected.data(),0,checked_bytes);ggml_backend_tensor_get(dst,actual.data(),0,checked_bytes);require(expected==actual,"timed payload range differs");
 const json exact={{"bytes",checked_bytes},{"arena_bytes",arena_bytes},{"full_byte_equality",true},{"scope","entire largest declared timed view, not unused arena suffix"},
  {"source_sha256",steady::sha256_bytes(expected.data(),expected.size())},{"target_sha256",steady::sha256_bytes(actual.data(),actual.size())}};
 e.reset();full();const auto final=single_layer::capture_verification(e);json rows=json::array();
 for(const auto &b:bursts){json copies=json::array();int ordinal=0;for(const auto &c:b.copies)copies.push_back(json{{"ordinal",ordinal++},{"begin_tick",c.begin},{"end_tick",c.end},{"wall_ns",steady::ns(c.begin,c.end)}});
  rows.push_back(json{{"trial",b.trial},{"slot",b.slot},{"condition",b.condition},{"size_index",b.size_index},{"bytes",views.values.at(size_t(b.size_index)).bytes},{"requested_idle_ms",b.idle},{"copies_count",b.copies_count},
   {"prime_begin_tick",b.prime_begin},{"prime_end_tick",b.prime_end},{"idle_begin_tick",b.idle_begin},{"idle_end_tick",b.idle_end},{"actual_idle_ns",steady::ns(b.idle_begin,b.idle_end)},
   {"effective_prime_to_first_copy_ns",steady::ns(b.prime_end,b.copies[0].begin)},{"burst_begin_tick",b.begin},{"burst_end_tick",b.end},{"burst_wall_ns",steady::ns(b.begin,b.end)},{"copies",copies}});
 }
 return json{{"schema","dynamic-h2d-lifecycle-native-10"},{"request",r},{"source_sha256",H2D_LIFECYCLE_SOURCE_SHA256},{"program_sha256",e.program.at("program_sha256")},{"process_id",single_layer::pid()},
  {"main_thread_id",GetCurrentThreadId()},{"clock",state_response::clock_json()},{"target_action",target},{"target_action_index",sampling.target},{"source_pointer_kind",pointer},
  {"source_buffer_host",ggml_backend_buffer_is_host(src->buffer)},{"target_buffer_host",ggml_backend_buffer_is_host(dst->buffer)},{"threadpool",pool},{"warmup",warm},{"arena_registration",registration.finish()},
  {"slice_copy_oracles",oracles},{"full_payload",exact},{"timed_payload_checked_before_final_reset",true},{"bursts",rows},{"verification_native",final},
  {"native_repeat_consistency",state_response::compare_ports(initial.at("ports"),final.at("ports"))},{"CUPTI_initialized",false},{"CUDA_events_created",false},{"NVML_initialized",false},
  {"telemetry_between_bursts",false},{"performance_parameters_fitted",false}};
}
}
int main(int argc,char **argv){try {
 if(argc==2&&std::string(argv[1])=="--self-test"){std::cout<<json{{"passed",true},{"hardware_executed",false},{"source_sha256",H2D_LIFECYCLE_SOURCE_SHA256}}.dump()<<'\n';return 0;}
 require(argc==4,"usage: dynamic-h2d-lifecycle-v10 PROGRAM REQUEST NEW_OUTPUT");require(!state_response::exists(argv[3]),"new exclusive output required");
 _putenv_s("NVIDIA_TF32_OVERRIDE","0");_putenv_s("GGML_CUDA_DISABLE_GRAPHS","1");
 std::ifstream p(std::filesystem::u8path(argv[1])),q(std::filesystem::u8path(argv[2]));json program,r;require(p.good()&&q.good(),"input unavailable");p>>program;q>>r;
 require(r.at("schema")=="dynamic-h2d-lifecycle-request-10"&&str(r,"source_sha256")==H2D_LIFECYCLE_SOURCE_SHA256,"request/source differs");lifecycle_v10::contract(r);
 steady::Request sampling(r.at("sampling"));const auto process=diagnostic_v3::apply_affinity(r);steady::OwnThreadAffinity thread(4);json result;
 {Executor e(std::move(program));result=lifecycle_v10::run(e,sampling,r);}result["process_affinity"]=diagnostic_v3::finish_affinity(process);result["main_thread_affinity"]=thread.finish();state_response::write_new(argv[3],result);return 0;
}catch(const std::exception &e){std::cerr<<"dynamic-h2d-lifecycle-v10: "<<e.what()<<'\n';return 2;}}
