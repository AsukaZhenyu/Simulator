// New explicit CUDA primitive boundary; never relabeled as Executor.action.
#include "../v3/common.hpp"
#include "../../../link_calibration/native/pcie_nvml.hpp"
#include <array>

namespace h2d_primitive_v7 {
using namespace diagnostic_v3;
constexpr size_t arena_bytes=2097152;
constexpr int trials=12,repeats=16;
static const std::array<std::string,3> modes{{"ggml_bare","primitive_stamped","primitive_events"}};
static void permutation(const json &rows,int width){
 require(rows.is_array()&&rows.size()==trials,"twelve complete trial orders required");
 for(const auto &row:rows){require(row.is_array()&&row.size()==size_t(width),"order width differs");std::set<int> seen;
  for(const auto &x:row){require(x.is_number_integer(),"integer order required");int v=x.get<int>();require(v>=0&&v<width&&seen.insert(v).second,"order is not a permutation");}}
 for(int slot=0;slot<width;++slot)for(int value=0;value<width;++value){int count=0;for(const auto &row:rows)count+=row.at(slot)==value;require(count==trials/width,"order positions must be balanced");}
}
static void contract(const json &r){
 const auto split=str(r,"split");require(split=="profile"||split=="heldout","explicit split required");
 const json sizes=split=="profile"?json::array({131072,524288,2097152}):json::array({262144,1048576});
 const json idles=split=="profile"?json::array({0,20}):json::array({2,10});
 require(r.at("sizes_bytes")==sizes&&r.at("idle_ms")==idles,"predeclared unseen size/idle split differs");
 require(r.at("modes")==json::array({modes[0],modes[1],modes[2]}),"explicit three observation boundaries required");
 require(str(r,"memory_mode")=="registered"||str(r,"memory_mode")=="pageable","explicit host memory required");
 permutation(r.at("size_orders"),int(sizes.size()));permutation(r.at("condition_orders"),6);
 require(r.at("prior_query_placement")=="before_full_prime"&&r.at("post_queries_are_predictor_input")==false,"query causality contract differs");
}
struct Alias {size_t bytes=0;std::string id;ggml_tensor *source=nullptr,*target=nullptr;json action;};
class Views {
 Executor &e_;ggml_context *ctx_=nullptr;
public:
 std::vector<Alias> values;
 Views(Executor &e,ggml_tensor *source,ggml_tensor *target,const json &action,const json &sizes):e_(e){
  require(source&&target&&source->buffer&&target->buffer&&source->data!=target->data,"distinct existing buffers required");
  require(ggml_backend_buffer_is_host(source->buffer)&&!ggml_backend_buffer_is_host(target->buffer),"actual host-to-device buffers required");
  require(source->type==GGML_TYPE_F32&&target->type==GGML_TYPE_F32&&ggml_is_contiguous(source)&&ggml_is_contiguous(target)&&ggml_nbytes(source)==arena_bytes&&ggml_nbytes(target)==arena_bytes,"retained contiguous two-MiB backing required");
  ctx_=ggml_init({ggml_tensor_overhead()*32,nullptr,true});require(ctx_,"view metadata allocation failed");
  for(const auto &v:sizes){Alias a;a.bytes=v.get<size_t>();require(a.bytes>0&&a.bytes<=arena_bytes&&a.bytes%4==0,"view extent invalid");
   a.id=str(action,"target_id")+":primitive_v7:"+std::to_string(a.bytes);a.action=action;a.action["target_id"]=a.id;
   a.source=ggml_view_1d(ctx_,source,a.bytes/4,0);a.target=ggml_view_1d(ctx_,target,a.bytes/4,0);
   require(ggml_backend_view_init(a.source)==GGML_STATUS_SUCCESS&&ggml_backend_view_init(a.target)==GGML_STATUS_SUCCESS,"existing buffer view initialization failed");
   require(a.source->data==source->data&&a.target->data==target->data&&ggml_nbytes(a.source)==a.bytes&&ggml_nbytes(a.target)==a.bytes,"view addresses/extent differ");
   Tensor t;t.dram=a.source;t.vram=a.target;t.descriptor=json{{"id",a.id},{"initial_locations",json::array()},{"storage_bytes",a.bytes}};
   require(e_.tensors.emplace(a.id,std::move(t)).second,"view tensor identity collided");values.push_back(std::move(a));
  }
 }
 ~Views(){for(const auto &a:values)e_.tensors.erase(a.id);if(ctx_)ggml_free(ctx_);}
};
class Events {
 std::array<cudaEvent_t,repeats> first_{},last_{};
public:
 Events(){try{for(int i=0;i<repeats;++i){require(cudaEventCreate(&first_[i])==cudaSuccess,"event start allocation failed");require(cudaEventCreate(&last_[i])==cudaSuccess,"event end allocation failed");}}
  catch(...){for(auto v:first_)if(v)cudaEventDestroy(v);for(auto v:last_)if(v)cudaEventDestroy(v);throw;}}
 ~Events(){for(auto v:first_)if(v)cudaEventDestroy(v);for(auto v:last_)if(v)cudaEventDestroy(v);}
 void first(int i){require(cudaEventRecord(first_[i],cudaStreamPerThread)==cudaSuccess,"same PTDS start record failed");}
 void last(int i){require(cudaEventRecord(last_[i],cudaStreamPerThread)==cudaSuccess,"same PTDS end record failed");}
 float span(int i){float ms=-1;require(cudaEventElapsedTime(&ms,first_[i],last_[i])==cudaSuccess&&std::isfinite(ms)&&ms>=0,"completed event span unavailable");return ms;}
};
struct Stamp {int64_t begin=0,end=0,event_a_begin=0,event_a_end=0,copy_begin=0,copy_end=0,event_b_begin=0,event_b_end=0,sync_begin=0,sync_end=0,gpu_begin=0,gpu_end=0,cpu_begin=0,cpu_end=0;float stream_ms=-1;};
static void primitive(Executor &e,const Alias &a,Stamp &s,Events *events,int ordinal){
 s.begin=steady::tick();
 if(events){s.event_a_begin=steady::tick();events->first(ordinal);s.event_a_end=steady::tick();}
 s.copy_begin=steady::tick();require(cudaMemcpyAsync(a.target->data,a.source->data,a.bytes,cudaMemcpyHostToDevice,cudaStreamPerThread)==cudaSuccess,"explicit PTDS H2D failed");s.copy_end=steady::tick();
 if(events){s.event_b_begin=steady::tick();events->last(ordinal);s.event_b_end=steady::tick();}
 s.sync_begin=steady::tick();require(cudaStreamSynchronize(cudaStreamPerThread)==cudaSuccess,"explicit PTDS inner synchronization failed");s.sync_end=steady::tick();
 // Preserve the two extra completion calls present in retained Executor.action.
 s.gpu_begin=steady::tick();ggml_backend_synchronize(e.gpu);s.gpu_end=steady::tick();
 s.cpu_begin=steady::tick();ggml_backend_synchronize(e.cpu);s.cpu_end=steady::tick();s.end=steady::tick();
}
static json qualify(Executor &e,const Alias &a,ggml_tensor *source,ggml_tensor *target,Events &events){
 std::vector<uint8_t> expected(arena_bytes),actual(arena_bytes);ggml_backend_tensor_get(source,expected.data(),0,arena_bytes);
 json result=json::array();for(const std::string mode:{"ggml_bare","primitive_stamped","primitive_events"}){
  ggml_backend_tensor_memset(target,0xa5,0,arena_bytes);ggml_backend_synchronize(e.gpu);ggml_backend_tensor_get(target,actual.data(),0,arena_bytes);
  require(std::all_of(actual.begin(),actual.end(),[](uint8_t x){return x==0xa5;}),"qualification poison failed");
  require(!std::equal(expected.begin(),expected.begin()+a.bytes,actual.begin()),"vacuous copy oracle");
  if(mode=="ggml_bare")e.action(a.action);else{Stamp s;primitive(e,a,s,mode=="primitive_events"?&events:nullptr,0);if(mode=="primitive_events")require(events.span(0)>=0,"preflight events did not complete");}
  ggml_backend_tensor_get(target,actual.data(),0,arena_bytes);require(std::equal(expected.begin(),expected.begin()+a.bytes,actual.begin()),"copied bytes differ");
  require(std::all_of(actual.begin()+a.bytes,actual.end(),[](uint8_t x){return x==0xa5;}),"copy overwrote range outside view");
  result.push_back(json{{"mode",mode},{"bytes",a.bytes},{"selected_range_initially_different",true},{"selected_bytes_equal_after_copy",true},{"outside_view_unchanged",true},{"timed",false}});
 }
 std::vector<uint8_t> unchanged(arena_bytes);ggml_backend_tensor_get(source,unchanged.data(),0,arena_bytes);require(unchanged==expected,"source changed");return result;
}
struct Burst {int trial=0,size_slot=0,size_index=0,condition_slot=0,condition=0,mode=0,idle=0;int64_t prime_begin=0,prime_end=0,idle_begin=0,idle_end=0,begin=0,end=0,event_read_begin=0,event_read_end=0;json prior,after;std::array<Stamp,repeats> copies{};};
static json frame(const Stamp &s,int ordinal,bool parts){
 json r{{"ordinal",ordinal},{"begin_tick",s.begin},{"end_tick",s.end},{"wall_ns",steady::ns(s.begin,s.end)},{"inner_response_parts_available",parts}};
 if(parts){json segments=json::array();auto part=[&](const char *name,int64_t begin,int64_t end){segments.push_back(json{{"name",name},{"begin_tick",begin},{"end_tick",end},{"wall_ns",steady::ns(begin,end)}});};
  if(s.event_a_begin)part("event_start_record_API",s.event_a_begin,s.event_a_end);
  part("cudaMemcpyAsync_API_response",s.copy_begin,s.copy_end);
  if(s.event_b_begin)part("event_stop_record_API",s.event_b_begin,s.event_b_end);
  part("inner_PTDS_sync_API_response",s.sync_begin,s.sync_end);part("extra_GGML_GPU_sync_response",s.gpu_begin,s.gpu_end);part("extra_GGML_CPU_sync_response",s.cpu_begin,s.cpu_end);
  int64_t covered=0;for(const auto &segment:segments)covered+=segment.at("end_tick").get<int64_t>()-segment.at("begin_tick").get<int64_t>();
  r["host_response_segments"]=segments;r["outside_segments_ticks"]=s.end-s.begin-covered;r["device_event_span_ns"]=s.stream_ms>=0?json(double(s.stream_ms)*1000000.0):json(nullptr);
 }
 return r;
}
static json run(Executor &e,const steady::Request &sampling,const json &r){
 contract(r);require(e.gpu&&e.program.at("physical").at("threads")==1&&sampling.mode=="action"&&sampling.repeats==repeats&&sampling.trials==trials&&sampling.poll=="0"&&sampling.instrumentation=="bare"&&sampling.main_mask==4,"retained one-thread COPY protocol required");
 const auto &actions=e.program.at("actions"),&target=actions.at(size_t(sampling.target));require(target.at("kind")=="COPY"&&target.at("src")=="dram"&&target.at("dst")=="vram","retained H2D target required");
 auto *source=e.tensor(str(target,"target_id"),"dram"),*destination=e.tensor(str(target,"target_id"),"vram");
 const auto pool=single_layer::configure_pool(e,"0");e.validate();const auto initial=single_layer::capture_verification(e);Registration registration(e,str(r,"memory_mode"));
 Views views(e,source,destination,target,r.at("sizes_bytes"));Events events;json oracles=json::array();for(const auto &a:views.values)oracles.push_back(qualify(e,a,source,destination,events));
 // All modes share this allocation/lifecycle; the CUDA runtime itself is unchanged.
 auto full=[&](){for(const auto &a:actions)e.action(a);};e.reset();full();const auto warm=steady::warm(sampling,full);
 const auto pointer=pointer_kind(source->data);int device=-1;require(cudaGetDevice(&device)==cudaSuccess&&device==0,"current actual CUDA device must be CUDA0");char bus[64]{};require(cudaDeviceGetPCIBusId(bus,sizeof(bus),device)==cudaSuccess,"CUDA0 bus identity unavailable");PCIeNVML nvml(str(r,"nvml_dll_path"),bus);
 dynamic_control::OwnedHighResolutionTimer timer;std::vector<Burst> bursts(size_t(trials)*views.values.size()*6);size_t cursor=0;
 for(int trial=0;trial<trials;++trial)for(size_t slot=0;slot<views.values.size();++slot)for(int cs=0;cs<6;++cs){auto &b=bursts[cursor++];b.trial=trial;b.size_slot=int(slot);b.size_index=r.at("size_orders").at(trial).at(slot).get<int>();b.condition_slot=cs;b.condition=r.at("condition_orders").at(trial).at(cs).get<int>();b.mode=b.condition/2;b.idle=r.at("idle_ms").at(b.condition%2).get<int>();
  b.prior=nvml.sample();e.reset();b.prime_begin=steady::tick();full();b.prime_end=steady::tick();b.idle_begin=steady::tick();if(b.idle)timer.wait_ns(int64_t(b.idle)*1000000);b.idle_end=steady::tick();
  const auto &a=views.values.at(size_t(b.size_index));b.begin=steady::tick();for(int ordinal=0;ordinal<repeats;++ordinal){auto &s=b.copies[ordinal];if(b.mode==0){s.begin=steady::tick();e.action(a.action);s.end=steady::tick();}else primitive(e,a,s,b.mode==2?&events:nullptr,ordinal);}b.end=steady::tick();
  b.event_read_begin=steady::tick();if(b.mode==2)for(int ordinal=0;ordinal<repeats;++ordinal)b.copies[ordinal].stream_ms=events.span(ordinal);b.event_read_end=steady::tick();b.after=nvml.sample();
 }
 // Preserve the actual completed timed target before reset/full overwrites it.
 const auto exact=payload(source,destination);e.reset();full();const auto final=single_layer::capture_verification(e);json rows=json::array();
 for(const auto &b:bursts){json copies=json::array();for(int i=0;i<repeats;++i)copies.push_back(frame(b.copies[i],i,b.mode!=0));
  rows.push_back(json{{"trial",b.trial},{"size_slot",b.size_slot},{"size_index",b.size_index},{"bytes",views.values.at(size_t(b.size_index)).bytes},{"condition_slot",b.condition_slot},{"condition_index",b.condition},{"mode",modes[b.mode]},{"requested_idle_ms",b.idle},{"prior_pcie_before_prime",b.prior},{"post_pcie_diagnostic_only",b.after},
   {"prime_begin_tick",b.prime_begin},{"prime_end_tick",b.prime_end},{"idle_begin_tick",b.idle_begin},{"idle_end_tick",b.idle_end},{"actual_idle_ns",steady::ns(b.idle_begin,b.idle_end)},{"effective_prime_to_first_copy_ns",steady::ns(b.prime_end,b.copies[0].begin)},
   {"burst_begin_tick",b.begin},{"burst_end_tick",b.end},{"burst_wall_ns",steady::ns(b.begin,b.end)},{"event_read_begin_tick",b.event_read_begin},{"event_read_end_tick",b.event_read_end},{"copies",copies}});
 }
 return json{{"schema","dynamic-h2d-primitive-native-7"},{"request",r},{"source_sha256",H2D_PRIMITIVE_SOURCE_SHA256},{"original_CUDA_source_sha256",H2D_PRIMITIVE_ORIGINAL_CUDA_SHA256},{"original_backend_source_sha256",H2D_PRIMITIVE_ORIGINAL_BACKEND_SHA256},{"program_sha256",e.program.at("program_sha256")},{"process_id",single_layer::pid()},{"main_thread_id",GetCurrentThreadId()},{"clock",state_response::clock_json()},
  {"target_action",target},{"target_action_index",sampling.target},{"source_pointer_kind",pointer},{"source_buffer_host",ggml_backend_buffer_is_host(source->buffer)},{"target_buffer_host",ggml_backend_buffer_is_host(destination->buffer)},{"source_buffer_name",ggml_backend_buffer_name(source->buffer)},{"target_buffer_name",ggml_backend_buffer_name(destination->buffer)},{"CUDA_device",device},{"CUDA0_pci_bus_id",bus},{"primitive_stream",json{{"kind","cudaStreamPerThread"},{"opaque_runtime_handle",uint64_t(reinterpret_cast<uintptr_t>(cudaStreamPerThread))},{"GGML_bare_internal_stream_observed_dynamically",false},{"choice_evidence","bound original host-buffer tensor_copy -> CUDA buffer_set_tensor uses explicit cudaStreamPerThread"}}},
  {"warmup",warm},{"threadpool",pool},{"arena_registration",registration.finish()},{"slice_copy_oracles",oracles},{"full_payload",exact},{"timed_payload_checked_before_final_reset",true},{"every_timed_copy_independently_poisoned",false},{"bursts",rows},{"nvml",nvml.describe()},{"idle_controller",dynamic_control::OwnedHighResolutionTimer::describe()},
  {"verification_native",final},{"native_repeat_consistency",state_response::compare_ports(initial.at("ports"),final.at("ports"))},{"same_original_buffers_context_registration",true},{"CUPTI_initialized",false},{"foreground_queries_between_prime_and_first_copy",false},
  {"primitive_is_GGML_action",false},{"GPU_event_is_pure_DMA",false},{"GPU_event_QPC_aligned",false},{"GPU_event_and_host_wait_additive",false},{"host_API_response_is_CPU_active",false},{"post_target_state_used_for_prediction",false},{"hardware_accuracy_qualified",false},{"performance_parameters_fitted",false}};
}
}
int main(int argc,char **argv){try{
 if(argc==2&&std::string(argv[1])=="--self-test"){json orders=json::array();for(int t=0;t<12;++t){json row=json::array();for(int s=0;s<6;++s)row.push_back((t+s)%6);orders.push_back(row);}h2d_primitive_v7::permutation(orders,6);bool rejected=false;orders[0][0]=orders[0][1];try{h2d_primitive_v7::permutation(orders,6);}catch(...){rejected=true;}require(rejected,"bad permutation accepted");std::cout<<json{{"passed",true},{"backend_created",false},{"CUDA_event_created",false},{"NVML_created",false},{"hardware_executed",false},{"source_sha256",H2D_PRIMITIVE_SOURCE_SHA256}}.dump()<<'\n';return 0;}
 require(argc==4,"usage: dynamic-h2d-primitive-v7 PROGRAM REQUEST NEW_OUTPUT");require(!state_response::exists(argv[3]),"new exclusive output required");_putenv_s("NVIDIA_TF32_OVERRIDE","0");_putenv_s("GGML_CUDA_DISABLE_GRAPHS","1");
 std::ifstream p(std::filesystem::u8path(argv[1])),q(std::filesystem::u8path(argv[2]));json program,r;require(p.good()&&q.good(),"explicit input unavailable");p>>program;q>>r;
 require(r.at("schema")=="dynamic-h2d-primitive-request-7"&&str(r,"source_sha256")==H2D_PRIMITIVE_SOURCE_SHA256&&str(r,"original_CUDA_source_sha256")==H2D_PRIMITIVE_ORIGINAL_CUDA_SHA256&&str(r,"original_backend_source_sha256")==H2D_PRIMITIVE_ORIGINAL_BACKEND_SHA256,"source/request binding differs");
 h2d_primitive_v7::contract(r);steady::Request sampling(r.at("sampling"));const auto process=diagnostic_v3::apply_affinity(r);steady::OwnThreadAffinity thread(4);json result;{Executor e(std::move(program));result=h2d_primitive_v7::run(e,sampling,r);}result["process_affinity"]=diagnostic_v3::finish_affinity(process);result["main_thread_affinity"]=thread.finish();state_response::write_new(argv[3],result);return 0;
}catch(const std::exception &e){std::cerr<<"dynamic-h2d-primitive-v7: "<<e.what()<<'\n';return 2;}}
