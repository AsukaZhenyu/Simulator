// Original Executor/owned actors; continuous component calibration, no observer.
#include "../v3/common.hpp"
#include "../v3/thread_inventory.hpp"
#include <atomic>
#include <exception>
#include <mutex>
#include "../v8/actors.inc"

namespace continuous_v10 {
using external_stage_v8::Control;using external_stage_v8::Actor;
struct Stamp {int64_t begin=0,end=0;};
struct Block {int64_t begin=0,end=0;};
constexpr int64_t minimum_profile_ns=1000000000;
constexpr int maximum_profile_calls=512;
static void contract(const json &r){
 auto split=str(r,"split");require(split=="profile"||split=="heldout","explicit split required");
 require(str(r,"instrumentation")=="stamped"&&r.at("attention_action_index")==22&&r.at("actor_epoch_delay_ms")==1000&&r.at("target_entry_max_lateness_ns")==10000000,"fixed original Attention/controller required");
 require(r.at("foreground_lifecycle")=="continuous_invocation"&&r.at("init_policy")=="pinned_init","explicit lifecycle/initialization required");
 require(r.at("cpu_actor").at("main_thread_mask")==4&&r.at("gpu_actor").at("main_thread_mask")==65536,"retained CPU2/CPU16 placement required");
 for(const auto &device:{"cpu_actor","gpu_actor"})require(r.at(device).at("init_policy")=="pinned_init","actor initialization differs");
 if(split=="profile"){
  require(r.at("profile_min_duration_ns")==minimum_profile_ns&&r.at("profile_max_calls")==maximum_profile_calls&&r.at("under_state_warmup_calls")==1,"predeclared continuous lifecycle differs");
  auto state=r.at("offered_state");require(state.is_array()&&state.size()==2,"exact state required");for(int i=0;i<2;++i){require(state.at(i)==0||state.at(i)==1,"binary state required");
   const auto expected=state.at(i)==1?json::array({json::array({0,30000})}):json::array();require(r.at(i==0?"cpu_actor":"gpu_actor").at("intervals_ms")==expected,"fixed30second ON window required");}
  require(r.at("target_entry_offset_ms")==((state.at(0)==0&&state.at(1)==0)?0:1000),"steady entry differs");
 }else{
  require(r.at("replays")==256&&r.at("block_replays")==16&&r.at("target_entry_offset_ms")==0,"fixed independent256/16 workload required");
  for(const auto &device:{"cpu_actor","gpu_actor"})for(const auto &p:r.at(device).at("intervals_ms"))require(p.at(0)>=100&&p.at(1)<=2000,"bounded initial-OFF new pulse plan required");
 }
}
static json action_row(const Stamp &s,int index){return json{{"ordinal",index},{"global_replay",index},{"action_index",22},{"begin_tick",s.begin},{"end_tick",s.end},{"wall_ns",steady::ns(s.begin,s.end)}};}
static json run(Executor &e,const steady::Request &sampling,const json &r,steady::OwnThreadAffinity &initialization){
 contract(r);const bool profile=str(r,"split")=="profile";require(e.gpu&&e.program.at("physical").at("threads")==1&&sampling.poll=="0"&&sampling.warmup==20&&sampling.min_ms==500,"retained thread/backend/warmup required");
 const auto &actions=e.program.at("actions"),&target=actions.at(22);require(target.at("kind")=="COMPUTE"&&target.at("device")=="gpu"&&str(target,"target_id").find(":attn")!=std::string::npos,"actual GPU Attention required");
 require(sampling.mode=="action"&&sampling.target==22&&sampling.instrumentation=="bare"&&sampling.main_mask==4,"sampling scope differs");
 require(profile?(sampling.trials==1&&sampling.repeats==1):(sampling.trials==16&&sampling.repeats==16),"predeclared loop control differs");
 const auto pool=single_layer::configure_pool(e,"0");e.validate();const auto initial=single_layer::capture_verification(e);Control control;Actor cpu(control,r.at("cpu_actor")),gpu(control,r.at("gpu_actor"));
 try{
  dynamic_control::OwnedHighResolutionTimer timer;auto waiting=steady::tick();while(!cpu.ready()||!gpu.ready()){require(steady::ns(waiting,steady::tick())<30000000000LL,"actor readiness timed out");timer.wait_ns(1000000);}cpu.check_error();gpu.check_error();
  const auto inventory=diagnostic_v3::own_thread_inventory("all_backends_and_actors_ready_before_future_epoch");const auto init_affinity=initialization.finish();steady::OwnThreadAffinity affinity(4);
  auto full=[&](){for(const auto &a:actions)e.action(a);};const auto warm=steady::warm(sampling,full);e.reset();full();
  // Fixed POD capacity is allocated before the actor epoch; no JSON/telemetry in the timed loop.
  std::vector<Stamp> stamps(profile?maximum_profile_calls:256);std::vector<Block> blocks(profile?1:16);Stamp under_state_warm{};
  const int64_t hz=steady::frequency(),epoch=steady::tick()+hz,entry=epoch+r.at("target_entry_offset_ms").get<int64_t>()*hz/1000;
  control.announce(epoch);require(timer.wait_until_qpc(entry,hz,control.cancelled),"entry cancelled");cpu.check_error();gpu.check_error();
  const auto calibration_begin=steady::tick();if(profile){under_state_warm.begin=steady::tick();e.action(target);under_state_warm.end=steady::tick();}
  const auto begin=steady::tick();int count=0;
  if(profile){blocks[0].begin=steady::tick();do{auto &s=stamps[size_t(count)];s.begin=steady::tick();e.action(target);s.end=steady::tick();++count;
    }while(count<maximum_profile_calls&&steady::ns(begin,steady::tick())<minimum_profile_ns);blocks[0].end=steady::tick();
  }else for(int b=0;b<16;++b){blocks[size_t(b)].begin=steady::tick();for(int j=0;j<16;++j){auto &s=stamps[size_t(count)];s.begin=steady::tick();e.action(target);s.end=steady::tick();++count;}blocks[size_t(b)].end=steady::tick();cpu.check_error();gpu.check_error();}
  const auto end=steady::tick();control.finish();const auto cpu_result=cpu.finish(),gpu_result=gpu.finish();const auto final=single_layer::capture_verification(e);
  json rows=json::array(),all=json::array(),gaps=json::array();int cursor=0;int64_t previous=begin;for(size_t b=0;b<blocks.size();++b){json list=json::array();const int width=profile?count:16;
   for(int j=0;j<width;++j){auto row=action_row(stamps[size_t(cursor)],cursor);list.push_back(row);all.push_back(row);gaps.push_back(json{{"begin_tick",previous},{"end_tick",stamps[size_t(cursor)].begin},{"wall_ns",steady::ns(previous,stamps[size_t(cursor)].begin)}});previous=stamps[size_t(cursor)].end;++cursor;}
   rows.push_back(json{{"block",b},{"replays",width},{"begin_tick",blocks[b].begin},{"end_tick",blocks[b].end},{"wall_ns",steady::ns(blocks[b].begin,blocks[b].end)},{"actions",list}});
  }gaps.push_back(json{{"begin_tick",previous},{"end_tick",end},{"wall_ns",steady::ns(previous,end)}});
  const auto observed_entry=profile?calibration_begin:begin;const auto late=steady::ns(entry,observed_entry);const bool contained=!profile||(calibration_begin>=epoch&&end<=epoch+30*hz);const bool elapsed=steady::ns(begin,end)>=minimum_profile_ns;
  json result{{"program_sha256",e.program.at("program_sha256")},{"actions",actions},{"target_action",target},{"attention_action_index",22},{"instrumentation","stamped"},{"blocks",rows},{"continuous_replays",count},
   {"target_expected_origin_tick",epoch},{"planned_target_entry_tick",entry},{"entry_observed_tick",observed_entry},{"calibration_begin_tick",calibration_begin},{"target_begin_tick",begin},{"target_end_tick",end},{"target_wall_ns",steady::ns(begin,end)},
   {"initial_start_lateness_ns",late},{"target_entry_controller_passed",observed_entry>=entry&&late<=10000000},{"steady_window_begin_tick",epoch},{"steady_window_end_tick",epoch+30*hz},{"profile_window_contained",contained},
   {"warmup",warm},{"threadpool",pool},{"initialization_affinity",init_affinity},{"target_thread_affinity",affinity.finish()},{"threads_after_initialization",inventory},{"actors",json{{"cpu",cpu_result},{"gpu",gpu_result}}},
   {"verification_native",final},{"native_repeat_consistency",state_response::compare_ports(initial.at("ports"),final.at("ports"))},{"timed_result_checked_before_reset",true},{"complete_attention_graph_only",true},{"complete_LLM_strategy_replayed",false},
   {"foreground_lifecycle","continuous_invocation"},{"reset_prime_between_timed_blocks",false},{"callback_stage_prices_used",false},{"profiling_CUPTI_initialized",false},{"profiling_NVML_initialized",false},{"profiling_extra_CUDA_events_created",false},
   {"actor_timeline_triggered_by_target_latency",false},{"future_target_state_used",false},{"API_response_is_CPU_active",false},{"internal_GPU_work_added",false},{"hardware_accuracy_qualified",false},{"performance_parameters_fitted",false}};
  if(profile){result["under_state_warmup"]=json{{"begin_tick",under_state_warm.begin},{"end_tick",under_state_warm.end},{"wall_ns",steady::ns(under_state_warm.begin,under_state_warm.end)},{"action_index",22},{"included_in_price",false}};
   result["continuous_profile"]=json{{"begin_tick",begin},{"end_tick",end},{"actions",all},{"gaps",gaps},{"min_duration_ns",minimum_profile_ns},{"max_calls",maximum_profile_calls},{"minimum_duration_reached",elapsed},{"stop_reason",elapsed?"minimum_duration":"max_calls"},{"accuracy_dependent_stop",false}};
  }return result;
 }catch(...){control.abort();throw;}
}
}
int main(int argc,char **argv){try{
 if(argc==2&&std::string(argv[1])=="--self-test"){auto result=steady::self_test();result["hardware_executed"]=false;result["actor_or_backend_created"]=false;result["source_sha256"]=CONTINUOUS_V10_SOURCE_SHA256;std::cout<<result.dump()<<'\n';return 0;}
 require(argc==4,"usage: dynamic-continuous-v10 PROGRAM REQUEST NEW_OUTPUT");require(!state_response::exists(argv[3]),"exclusive output required");_putenv_s("NVIDIA_TF32_OVERRIDE","0");_putenv_s("GGML_CUDA_DISABLE_GRAPHS","1");
 std::ifstream p(std::filesystem::u8path(argv[1])),q(std::filesystem::u8path(argv[2]));json program,r;require(p.good()&&q.good(),"inputs unavailable");p>>program;q>>r;
 require(r.at("schema")=="dynamic-continuous-request-10"&&str(r,"source_sha256")==CONTINUOUS_V10_SOURCE_SHA256&&str(r,"actor_source_sha256")==CONTINUOUS_V10_ACTOR_SHA256,"binary/source/request differs");continuous_v10::contract(r);
 steady::Request sampling(r.at("sampling"));auto process=external_stage_v8::process_affinity(r);steady::OwnThreadAffinity initialization(4);json result;{Executor e(std::move(program));result=continuous_v10::run(e,sampling,r,initialization);}
 DWORD_PTR actual=0,system=0;require(GetProcessAffinityMask(GetCurrentProcess(),&actual,&system)!=0&&actual==4294967295ULL,"process affinity changed");process["final_mask"]=uint64_t(actual);
 result.update(json{{"schema","dynamic-continuous-native-10"},{"request",r},{"source_sha256",CONTINUOUS_V10_SOURCE_SHA256},{"actor_source_sha256",CONTINUOUS_V10_ACTOR_SHA256},{"process_id",single_layer::pid()},{"main_thread_id",GetCurrentThreadId()},{"process_affinity",process},{"clock",json{{"clock_id","windows_qpc"},{"frequency_hz",steady::frequency()}}}});state_response::write_new(argv[3],result);return 0;
}catch(const std::exception &e){std::cerr<<"dynamic-continuous-v10: "<<e.what()<<'\n';return 2;}}
