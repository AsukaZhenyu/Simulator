// Completed original GGML COPY only. No CUPTI/event/NVML observer is armed.
#include "../v3/common.hpp"
#include <array>

namespace h2d_bare_v9 {
using namespace diagnostic_v3;
constexpr int trials=12,repeats=16;
constexpr size_t arena_bytes=2097152;
static void permutation(const json &rows,int width){
 require(width>0&&trials%width==0&&rows.is_array()&&rows.size()==trials,"balanced twelve-trial order required");
 for(const auto &row:rows){require(row.is_array()&&row.size()==size_t(width),"order width differs");std::set<int> seen;
  for(const auto &v:row){require(v.is_number_integer(),"integer order required");int n=v.get<int>();require(n>=0&&n<width&&seen.insert(n).second,"invalid permutation");}}
 for(int slot=0;slot<width;++slot)for(int value=0;value<width;++value){int count=0;for(const auto &row:rows)count+=row.at(slot)==value;require(count==trials/width,"unbalanced order position");}
}
static void contract(const json &r){
 auto split=str(r,"split");require(split=="profile"||split=="heldout","explicit split required");
 require(r.at("sizes_bytes")== (split=="profile"?json::array({131072,524288,2097152}):json::array({262144,1048576})),"unseen size split differs");
 require(r.at("idle_ms")== (split=="profile"?json::array({0,1,5,20}):json::array({2,10})),"unseen idle split differs");
 require(str(r,"memory_mode")=="registered"||str(r,"memory_mode")=="pageable","explicit host category required");
 require(r.at("performance_boundary")=="completed_original_Executor.action_COPY"&&r.at("telemetry_between_bursts")==false,"bare response boundary differs");
 permutation(r.at("size_orders"),int(r.at("sizes_bytes").size()));permutation(r.at("condition_orders"),int(r.at("idle_ms").size()));
}
struct Alias {size_t bytes;std::string id;ggml_tensor *source,*target;json action;};
class Views {
 Executor &e_;ggml_context *ctx_;
public:
 std::vector<Alias> values;
 Views(Executor &e,ggml_tensor *src,ggml_tensor *dst,const json &action,const json &sizes):e_(e){
  require(src&&dst&&src->data!=dst->data&&ggml_backend_buffer_is_host(src->buffer)&&!ggml_backend_buffer_is_host(dst->buffer),"distinct host/device buffers required");
  require(src->type==GGML_TYPE_F32&&dst->type==GGML_TYPE_F32&&ggml_is_contiguous(src)&&ggml_is_contiguous(dst)&&ggml_nbytes(src)==arena_bytes&&ggml_nbytes(dst)==arena_bytes,"retained contiguous backing required");
  ctx_=ggml_init({ggml_tensor_overhead()*32,nullptr,true});require(ctx_,"view metadata allocation failed");
  for(const auto &size:sizes){size_t bytes=size.get<size_t>();require(bytes>0&&bytes<=arena_bytes&&bytes%4==0,"invalid view extent");
   Alias a{bytes,str(action,"target_id")+":bare_v9:"+std::to_string(bytes),ggml_view_1d(ctx_,src,bytes/4,0),ggml_view_1d(ctx_,dst,bytes/4,0),action};
   require(ggml_backend_view_init(a.source)==GGML_STATUS_SUCCESS&&ggml_backend_view_init(a.target)==GGML_STATUS_SUCCESS,"buffer view initialization failed");
   require(a.source->data==src->data&&a.target->data==dst->data&&ggml_nbytes(a.source)==bytes&&ggml_nbytes(a.target)==bytes,"view pointer/range differs");
   a.action["target_id"]=a.id;Tensor t;t.dram=a.source;t.vram=a.target;t.descriptor=json{{"id",a.id},{"initial_locations",json::array()},{"storage_bytes",bytes}};
   require(e_.tensors.emplace(a.id,std::move(t)).second,"view identity collided");values.push_back(std::move(a));
  }
 }
 ~Views(){for(const auto &a:values)e_.tensors.erase(a.id);if(ctx_)ggml_free(ctx_);}
};
static json qualify(Executor &e,const Alias &a,ggml_tensor *src,ggml_tensor *dst){
 std::vector<uint8_t> expected(arena_bytes),actual(arena_bytes);ggml_backend_tensor_get(src,expected.data(),0,arena_bytes);
 ggml_backend_tensor_memset(dst,0xa5,0,arena_bytes);ggml_backend_synchronize(e.gpu);ggml_backend_tensor_get(dst,actual.data(),0,arena_bytes);
 require(std::all_of(actual.begin(),actual.end(),[](uint8_t x){return x==0xa5;}),"qualification poison failed");
 require(!std::equal(expected.begin(),expected.begin()+a.bytes,actual.begin()),"vacuous range oracle");e.action(a.action);ggml_backend_tensor_get(dst,actual.data(),0,arena_bytes);
 require(std::equal(expected.begin(),expected.begin()+a.bytes,actual.begin()),"copied range differs");require(std::all_of(actual.begin()+a.bytes,actual.end(),[](uint8_t x){return x==0xa5;}),"outside view overwritten");
 std::vector<uint8_t> unchanged(arena_bytes);ggml_backend_tensor_get(src,unchanged.data(),0,arena_bytes);require(unchanged==expected,"source modified");
 return json{{"bytes",a.bytes},{"selected_range_initially_different",true},{"selected_bytes_equal_after_copy",true},{"outside_view_unchanged",true},{"timed",false}};
}
struct Burst {int trial,size_slot,size_index,condition_slot,condition,idle;int64_t prime_begin,prime_end,idle_begin,idle_end,begin,end;std::array<Copy,repeats> copies;};
static json run(Executor &e,const steady::Request &sampling,const json &r){
 contract(r);require(e.gpu&&e.program.at("physical").at("threads")==1&&sampling.mode=="action"&&sampling.repeats==repeats&&sampling.trials==trials&&sampling.poll=="0"&&sampling.instrumentation=="bare"&&sampling.main_mask==4,"one-thread original COPY protocol required");
 const auto &actions=e.program.at("actions"),&target=actions.at(size_t(sampling.target));require(target.at("kind")=="COPY"&&target.at("src")=="dram"&&target.at("dst")=="vram","H2D target required");
 auto *src=e.tensor(str(target,"target_id"),"dram"),*dst=e.tensor(str(target,"target_id"),"vram");
 const auto pool=single_layer::configure_pool(e,"0");e.validate();const auto initial=single_layer::capture_verification(e);Registration registration(e,str(r,"memory_mode"));
 Views views(e,src,dst,target,r.at("sizes_bytes"));json oracles=json::array();for(const auto &a:views.values)oracles.push_back(qualify(e,a,src,dst));
 auto full=[&](){for(const auto &a:actions)e.action(a);};e.reset();full();const auto warm=steady::warm(sampling,full);const auto pointer=pointer_kind(src->data);
 dynamic_control::OwnedHighResolutionTimer timer;const int conditions=int(r.at("idle_ms").size());std::vector<Burst> bursts(size_t(trials)*views.values.size()*conditions);size_t cursor=0;
 for(int trial=0;trial<trials;++trial)for(size_t slot=0;slot<views.values.size();++slot)for(int cs=0;cs<conditions;++cs){auto &b=bursts[cursor++];b.trial=trial;b.size_slot=int(slot);b.size_index=r.at("size_orders").at(trial).at(slot).get<int>();b.condition_slot=cs;b.condition=r.at("condition_orders").at(trial).at(cs).get<int>();b.idle=r.at("idle_ms").at(b.condition).get<int>();
  e.reset();b.prime_begin=steady::tick();full();b.prime_end=steady::tick();b.idle_begin=steady::tick();if(b.idle)timer.wait_ns(int64_t(b.idle)*1000000);b.idle_end=steady::tick();
  const auto &a=views.values.at(size_t(b.size_index));b.begin=steady::tick();for(auto &copy:b.copies){copy.begin=steady::tick();e.action(a.action);copy.end=steady::tick();}b.end=steady::tick();
 }
 const auto exact=payload(src,dst);e.reset();full();const auto final=single_layer::capture_verification(e);json rows=json::array();
 for(const auto &b:bursts){json copies=json::array();for(int ordinal=0;ordinal<repeats;++ordinal){const auto &c=b.copies[ordinal];copies.push_back(json{{"ordinal",ordinal},{"begin_tick",c.begin},{"end_tick",c.end},{"wall_ns",steady::ns(c.begin,c.end)}});}
  rows.push_back(json{{"trial",b.trial},{"size_slot",b.size_slot},{"size_index",b.size_index},{"condition_slot",b.condition_slot},{"condition_index",b.condition},{"bytes",views.values.at(size_t(b.size_index)).bytes},{"requested_idle_ms",b.idle},{"prime_begin_tick",b.prime_begin},{"prime_end_tick",b.prime_end},{"idle_begin_tick",b.idle_begin},{"idle_end_tick",b.idle_end},{"actual_idle_ns",steady::ns(b.idle_begin,b.idle_end)},{"effective_prime_to_first_copy_ns",steady::ns(b.prime_end,b.copies[0].begin)},{"burst_begin_tick",b.begin},{"burst_end_tick",b.end},{"burst_wall_ns",steady::ns(b.begin,b.end)},{"copies",copies}});
 }
 return json{{"schema","dynamic-h2d-bare-native-9"},{"request",r},{"source_sha256",H2D_BARE_SOURCE_SHA256},{"program_sha256",e.program.at("program_sha256")},{"process_id",single_layer::pid()},{"main_thread_id",GetCurrentThreadId()},{"clock",state_response::clock_json()},{"target_action",target},{"target_action_index",sampling.target},{"source_pointer_kind",pointer},{"source_buffer_host",ggml_backend_buffer_is_host(src->buffer)},{"target_buffer_host",ggml_backend_buffer_is_host(dst->buffer)},{"threadpool",pool},{"warmup",warm},{"arena_registration",registration.finish()},{"slice_copy_oracles",oracles},{"full_payload",exact},{"timed_payload_checked_before_final_reset",true},{"bursts",rows},{"verification_native",final},{"native_repeat_consistency",state_response::compare_ports(initial.at("ports"),final.at("ports"))},{"CUPTI_initialized",false},{"CUDA_events_created",false},{"NVML_initialized",false},{"telemetry_between_bursts",false},{"host_API_response_is_CPU_active",false},{"performance_parameters_fitted",false}};
}
}
int main(int argc,char **argv){try{
 if(argc==2&&std::string(argv[1])=="--self-test"){for(int width:{2,3,4}){json rows=json::array();for(int t=0;t<12;++t){json row=json::array();for(int s=0;s<width;++s)row.push_back((t+s)%width);rows.push_back(row);}h2d_bare_v9::permutation(rows,width);rows[0][0]=rows[0][1];bool rejected=false;try{h2d_bare_v9::permutation(rows,width);}catch(...){rejected=true;}require(rejected,"bad order accepted");}std::cout<<json{{"passed",true},{"hardware_executed",false},{"source_sha256",H2D_BARE_SOURCE_SHA256}}.dump()<<'\n';return 0;}
 require(argc==4,"usage: dynamic-h2d-bare-v9 PROGRAM REQUEST NEW_OUTPUT");require(!state_response::exists(argv[3]),"exclusive new output required");_putenv_s("NVIDIA_TF32_OVERRIDE","0");_putenv_s("GGML_CUDA_DISABLE_GRAPHS","1");
 std::ifstream p(std::filesystem::u8path(argv[1])),q(std::filesystem::u8path(argv[2]));json program,r;require(p.good()&&q.good(),"input unavailable");p>>program;q>>r;require(r.at("schema")=="dynamic-h2d-bare-request-9"&&str(r,"source_sha256")==H2D_BARE_SOURCE_SHA256,"request/source differs");h2d_bare_v9::contract(r);
 steady::Request sampling(r.at("sampling"));const auto process=diagnostic_v3::apply_affinity(r);steady::OwnThreadAffinity thread(4);json result;{Executor e(std::move(program));result=h2d_bare_v9::run(e,sampling,r);}result["process_affinity"]=diagnostic_v3::finish_affinity(process);result["main_thread_affinity"]=thread.finish();state_response::write_new(argv[3],result);return 0;
}catch(const std::exception &e){std::cerr<<"dynamic-h2d-bare-v9: "<<e.what()<<'\n';return 2;}}
