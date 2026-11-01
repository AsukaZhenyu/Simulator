// Same retained Executor and COPY completion API; only a contiguous alias view
// controls copied bytes. No events, CUPTI or runtime/source/kernel alteration.
#define main historical_steady_entry_not_called
#include "../../native/steady_costs/probe.cpp"
#undef main
#include "pcie_nvml.hpp"
#include <array>
#include <filesystem>

namespace link_copy {
constexpr size_t arena_bytes=2097152;
static void validate_sizes(const json &r){
 const std::string split=str(r,"split");require(split=="profile"||split=="heldout","unknown size split");
 const std::set<size_t> expected=split=="profile"?std::set<size_t>{131072,524288,2097152}:std::set<size_t>{262144,1048576};
 const auto sizes=r.at("sizes_bytes").get<std::vector<size_t>>();require(std::set<size_t>(sizes.begin(),sizes.end())==expected&&sizes.size()==expected.size(),"exact calibration/unseen-size set required");
 const auto &orders=r.at("orders");require(orders.is_array()&&orders.size()==12,"twelve predeclared balanced trials required");
 for(const auto &order:orders){require(order.is_array()&&order.size()==sizes.size(),"size order length differs");std::set<int> seen;for(const auto &x:order){int i=x.get<int>();require(i>=0&&i<int(sizes.size())&&seen.insert(i).second,"size order not a permutation");}}
 for(size_t slot=0;slot<sizes.size();++slot)for(size_t i=0;i<sizes.size();++i){int n=0;for(const auto &row:orders)n+=row.at(slot)==i;require(n==12/int(sizes.size()),"size positions not balanced");}
 const auto &modes=r.at("query_orders");require(modes.size()==12*sizes.size(),"bare/query order inventory missing");int first_queries=0;
 for(const auto &m:modes){require(m==json::array({"bare","state_queries"})||m==json::array({"state_queries","bare"}),"both bare and query controls required");first_queries+=m.at(0)=="state_queries";}
 require(first_queries*2==int(modes.size()),"bare/query first position is not balanced");
 require(r.at("memory_mode")=="registered"&&r.at("idle_ms")==0,"first base calibration is registered/zero-idle only");
}
static json pointer_kind(void *pointer){cudaPointerAttributes a{};const auto status=cudaPointerGetAttributes(&a,pointer);
 if(status==cudaErrorInvalidValue){cudaGetLastError();return json{{"registered_host",false},{"unregistered_pageable",true}};}
 require(status==cudaSuccess,"owned pointer query failed");return json{{"registered_host",a.type==cudaMemoryTypeHost},{"unregistered_pageable",a.type==cudaMemoryTypeUnregistered}};}
class Registration{
 void *base=nullptr;size_t bytes=0;bool active=false;json proof;
public:
 explicit Registration(Executor &e){base=ggml_backend_buffer_get_base(e.dram_buffer);bytes=ggml_backend_buffer_get_size(e.dram_buffer);require(base&&bytes>0,"original owned host arena missing");
  auto before=pointer_kind(base);require(before.at("unregistered_pageable")==true,"initial arena must be observed pageable");int device=-1,supported=0;
  require(cudaGetDevice(&device)==cudaSuccess&&cudaDeviceGetAttribute(&supported,cudaDevAttrHostRegisterSupported,device)==cudaSuccess&&supported,"host registration unsupported");
  require(cudaHostRegister(base,bytes,cudaHostRegisterDefault)==cudaSuccess,"explicit original arena registration failed");active=true;auto during=pointer_kind(base);require(during.at("registered_host"),"actual registration differs");
  proof=json{{"mode","registered"},{"before",before},{"during",during},{"arena_bytes",bytes},{"registration_timed",false},{"scope","original entire Executor DRAM arena"}};
 }
 ~Registration(){if(active)cudaHostUnregister(base);}
 json finish(){require(active,"registration already closed");require(cudaHostUnregister(base)==cudaSuccess,"owned unregister failed");active=false;proof["after"]=pointer_kind(base);proof["restored"]=proof["after"]["unregistered_pageable"];require(proof["restored"]==true,"arena category not restored");return proof;}
};
static json affinity(const json &r){const auto wanted=static_cast<DWORD_PTR>(r.at("expected_process_mask").get<uint64_t>());require(wanted==4294967295ULL,"all-32 process affinity required");DWORD_PTR initial=0,system=0,actual=0;
 require(GetProcessAffinityMask(GetCurrentProcess(),&initial,&system)!=0&&(wanted&system)==wanted&&SetProcessAffinityMask(GetCurrentProcess(),wanted)!=0&&GetProcessAffinityMask(GetCurrentProcess(),&actual,&system)!=0&&actual==wanted,"owned process affinity enforcement failed");
 return json{{"initial_mask",uint64_t(initial)},{"requested_mask",uint64_t(wanted)},{"observed_mask",uint64_t(actual)},{"before_backend_initialization",true}};}
struct Alias {size_t bytes=0;std::string id;json action;ggml_tensor *source=nullptr,*target=nullptr;};
class Views{
 Executor &e;ggml_context *ctx=nullptr;
public:
 std::vector<Alias> aliases;
 Views(Executor &executor,ggml_tensor *source,ggml_tensor *target,const json &original,const json &sizes):e(executor){
  require(source!=target&&source->data!=target->data&&source->buffer!=target->buffer,"distinct original host/device backing required");
  require(source->type==GGML_TYPE_F32&&target->type==GGML_TYPE_F32&&ggml_is_contiguous(source)&&ggml_is_contiguous(target)&&ggml_nbytes(source)==arena_bytes&&ggml_nbytes(target)==arena_bytes,"original contiguous F32 two-MiB tensors required");
  ctx=ggml_init({ggml_tensor_overhead()*32,nullptr,true});require(ctx,"view metadata allocation failed");
  for(const auto &v:sizes){size_t bytes=v.get<size_t>();require(bytes>0&&bytes<=arena_bytes&&bytes%sizeof(float)==0,"invalid continuous COPY slice");
   Alias a;a.bytes=bytes;a.id=str(original,"target_id")+":link_slice:"+std::to_string(bytes);a.action=original;a.action["target_id"]=a.id;
   a.source=ggml_view_1d(ctx,source,bytes/sizeof(float),0);a.target=ggml_view_1d(ctx,target,bytes/sizeof(float),0);
   require(ggml_backend_view_init(a.source)==GGML_STATUS_SUCCESS&&ggml_backend_view_init(a.target)==GGML_STATUS_SUCCESS,"existing-buffer slice view init failed");
   require(a.source->data==source->data&&a.target->data==target->data&&ggml_nbytes(a.source)==bytes&&ggml_nbytes(a.target)==bytes,"slice addresses or exact byte extent differ");
   Tensor t;t.dram=a.source;t.vram=a.target;t.descriptor=json{{"id",a.id},{"initial_locations",json::array()},{"storage_bytes",bytes}};
   require(e.tensors.emplace(a.id,std::move(t)).second,"slice alias collided with original tensor identity");aliases.push_back(std::move(a));
  }
 }
 ~Views(){for(const auto &a:aliases)e.tensors.erase(a.id);if(ctx)ggml_free(ctx);}
};
static json qualify_copy(Executor &e,const Alias &a,ggml_tensor *source,ggml_tensor *target){
 std::vector<uint8_t> expected(arena_bytes),observed(arena_bytes);ggml_backend_tensor_get(source,expected.data(),0,arena_bytes);
 ggml_backend_tensor_memset(target,0xa5,0,arena_bytes);ggml_backend_synchronize(e.gpu);ggml_backend_tensor_get(target,observed.data(),0,arena_bytes);
 require(std::all_of(observed.begin(),observed.end(),[](uint8_t x){return x==0xa5;}),"full target poison failed");
 require(!std::equal(expected.begin(),expected.begin()+a.bytes,observed.begin()),"COPY oracle vacuous: selected range already equal");
 e.action(a.action);ggml_backend_tensor_get(target,observed.data(),0,arena_bytes);
 std::vector<uint8_t> unchanged(arena_bytes);ggml_backend_tensor_get(source,unchanged.data(),0,arena_bytes);require(unchanged==expected,"COPY changed original source bytes");
 require(std::equal(expected.begin(),expected.begin()+a.bytes,observed.begin()),"selected COPY bytes differ");
 require(std::all_of(observed.begin()+a.bytes,observed.end(),[](uint8_t x){return x==0xa5;}),"COPY overwrote bytes beyond selected view");
 return json{{"bytes",a.bytes},{"full_arena_bytes_checked",arena_bytes},{"selected_range_initially_different",true},{"selected_range_equal_after_copy",true},{"outside_range_still_poison",true},
  {"source_sha256",steady::sha256_bytes(expected.data(),expected.size())},{"original_source_unchanged",true},{"actual_target_sha256",steady::sha256_bytes(observed.data(),observed.size())},{"timed",false},{"when","qualification before full warmup; no poison between formal prime and burst"}};
}
struct Stamp {int64_t begin=0,end=0;};
struct Burst{int trial=0,slot=0,size_index=0,control_slot=0;bool queries=false;int64_t prime_begin=0,prime_end=0,begin=0,end=0;json before,after;std::array<Stamp,16> copies{};};
static json run(Executor &e,const steady::Request &sampling,const json &r){
 validate_sizes(r);require(e.gpu&&e.program.at("physical").at("threads")==1,"original GPU and one CPU thread required");
 require(sampling.mode=="action"&&sampling.target>=0&&sampling.poll=="0"&&sampling.warmup==20&&sampling.min_ms==500&&sampling.repeats==16&&sampling.instrumentation=="bare","retained COPY/warmup contract differs");
 const auto &actions=e.program.at("actions");const auto &original=actions.at(size_t(sampling.target));require(original.at("kind")=="COPY"&&original.at("src")=="dram"&&original.at("dst")=="vram","original H2D COPY required");
 auto *source=e.tensor(str(original,"target_id"),"dram"),*target=e.tensor(str(original,"target_id"),"vram");
 const auto pool=single_layer::configure_pool(e,"0");e.validate();const auto initial=single_layer::capture_verification(e);Registration registration(e);Views views(e,source,target,original,r.at("sizes_bytes"));
 json oracles=json::array();for(const auto &a:views.aliases)oracles.push_back(qualify_copy(e,a,source,target));
 auto full=[&](){for(const auto &a:actions)e.action(a);};e.reset();full();const auto warm=steady::warm(sampling,full);
 char bus[64]{};require(cudaDeviceGetPCIBusId(bus,sizeof(bus),0)==cudaSuccess,"actual CUDA0 PCI identity missing");PCIeNVML nvml(str(r,"nvml_dll_path"),bus);
 std::vector<Burst> bursts(12*views.aliases.size()*2);size_t cursor=0;
 for(int trial=0;trial<12;++trial)for(size_t slot=0;slot<views.aliases.size();++slot)for(int control=0;control<2;++control){auto &b=bursts[cursor++];b.trial=trial;b.slot=int(slot);b.control_slot=control;b.size_index=r.at("orders").at(trial).at(slot).get<int>();b.queries=r.at("query_orders").at(trial*views.aliases.size()+slot).at(control)=="state_queries";
  e.reset();b.prime_begin=steady::tick();full();b.prime_end=steady::tick();if(b.queries)b.before=nvml.sample();
  const auto &a=views.aliases.at(size_t(b.size_index));b.begin=steady::tick();for(int ordinal=0;ordinal<16;++ordinal){auto &s=b.copies[ordinal];s.begin=steady::tick();e.action(a.action);s.end=steady::tick();}b.end=steady::tick();if(b.queries)b.after=nvml.sample();
 }
 e.reset();full();const auto final=single_layer::capture_verification(e);json rows=json::array();
 for(const auto &b:bursts){json copies=json::array();int64_t suffix=0;for(int ordinal=0;ordinal<16;++ordinal){const auto &s=b.copies[ordinal];const auto ns=steady::ns(s.begin,s.end);if(ordinal)suffix+=ns;copies.push_back(json{{"ordinal",ordinal},{"begin_tick",s.begin},{"end_tick",s.end},{"wall_ns",ns}});}
  rows.push_back(json{{"trial",b.trial},{"size_slot",b.slot},{"size_index",b.size_index},{"bytes",views.aliases.at(size_t(b.size_index)).bytes},{"control_slot",b.control_slot},{"control",b.queries?"state_queries":"bare"},
   {"prime_begin_tick",b.prime_begin},{"prime_end_tick",b.prime_end},{"requested_idle_ms",0},{"burst_begin_tick",b.begin},{"burst_end_tick",b.end},{"burst_wall_ns",steady::ns(b.begin,b.end)},
   {"pcie_before",b.queries?b.before:json(nullptr)},{"pcie_after",b.queries?b.after:json(nullptr)},{"copies",copies},{"first_copy_ns",steady::ns(b.copies[0].begin,b.copies[0].end)},{"suffix_15_sum_ns",suffix},
   {"state_input_scope","pre-burst PCIe queries only; post-burst values qualification/diagnosis only; bare controls have unknown state"}});
 }
 return json{{"schema","link-copy-size-native-1"},{"request",r},{"source_sha256",LINK_COPY_SOURCE_SHA256},{"process_id",single_layer::pid()},{"program_sha256",e.program.at("program_sha256")},
  {"original_target_action",original},{"original_target_action_index",sampling.target},{"original_arena_bytes",arena_bytes},{"slice_copy_oracles",oracles},{"source_pointer_kind",pointer_kind(source->data)},
  {"warmup",warm},{"threadpool",pool},{"arena_registration",registration.finish()},{"actual_CUDA0_bus_id",bus},{"nvml",nvml.describe()},{"bursts",rows},
  {"verification_native",final},{"native_repeat_consistency",state_response::compare_ports(initial.at("ports"),final.at("ports"))},
  {"boundary","original Executor.action on contiguous alias views: GGML tensor_copy + GPU sync + CPU sync"},{"all_sizes_share_original_buffers_context_and_registration",true},
  {"full_original_untimed_prime_before_every_burst",true},{"slice_poison_only_before_formal_warmup",true},{"clock",json{{"clock_id","windows_qpc"},{"frequency_hz",steady::frequency()}}},
  {"cuda_events_used",false},{"cupti_used",false},{"queries_inside_copy_timer",false},{"logging_inside_copy_timer",false},{"post_state_used_as_prediction_input",false},{"theoretical_link_rate_used",false},{"performance_parameters_fitted",false},{"LLM_accuracy_certified",false}};
}
}
int main(int argc,char **argv){try{
 if(argc==2&&std::string(argv[1])=="--self-test"){
  for(int n:{2,3}){json orders=json::array(),query=json::array();for(int t=0;t<12;++t){json row=json::array();for(int i=0;i<n;++i)row.push_back((t+i)%n);if((t/n)%2)std::reverse(row.begin(),row.end());orders.push_back(row);for(int slot=0;slot<n;++slot)query.push_back((t+slot)%2?json::array({"state_queries","bare"}):json::array({"bare","state_queries"}));}
   json r{{"split",n==3?"profile":"heldout"},{"sizes_bytes",n==3?json::array({131072,524288,2097152}):json::array({262144,1048576})},{"orders",orders},{"query_orders",query},{"memory_mode","registered"},{"idle_ms",0}};
   link_copy::validate_sizes(r);r["orders"][0][1]=r["orders"][0][0];bool rejected=false;try{link_copy::validate_sizes(r);}catch(const std::exception&){rejected=true;}require(rejected,"duplicate size order accepted");}
  std::cout<<json{{"passed",true},{"native_hardware_executed",false},{"backend_constructed",false},{"NVML_constructed",false},{"source_sha256",LINK_COPY_SOURCE_SHA256}}.dump()<<'\n';return 0;}
 require(argc==4,"usage: link-copy-probe PROGRAM REQUEST NEW_OUTPUT");require(!state_response::exists(argv[3]),"new exclusive output required");_putenv_s("NVIDIA_TF32_OVERRIDE","0");_putenv_s("GGML_CUDA_DISABLE_GRAPHS","1");
 std::ifstream p(std::filesystem::u8path(argv[1])),q(std::filesystem::u8path(argv[2]));require(p.good()&&q.good(),"explicit inputs unavailable");json program,r;p>>program;q>>r;
 require(r.at("schema")=="link-copy-size-request-1"&&r.at("source_sha256")==LINK_COPY_SOURCE_SHA256,"request/source differs");steady::Request sampling(r.at("sampling"));require(sampling.main_mask==4,"original CPU2 target affinity required");
 const auto process=link_copy::affinity(r);steady::OwnThreadAffinity thread(4);json result;{Executor e(std::move(program));result=link_copy::run(e,sampling,r);}DWORD_PTR actual=0,system=0;require(GetProcessAffinityMask(GetCurrentProcess(),&actual,&system)!=0&&actual==4294967295ULL,"owned process affinity changed");result["process_affinity"]=process;result["process_affinity"]["final_mask"]=uint64_t(actual);result["main_thread_affinity"]=thread.finish();state_response::write_new(argv[3],result);return 0;
}catch(const std::exception &e){std::cerr<<"link-copy-probe: "<<e.what()<<'\n';return 2;}}
