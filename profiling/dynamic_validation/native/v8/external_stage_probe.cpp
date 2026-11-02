// Complete Attention graph only. Existing owned actors and Executor are reused.
#include "../v3/common.hpp"
#include "../v3/thread_inventory.hpp"
#include "../v6/runtime_callbacks.hpp"
#include <atomic>
#include <exception>
#include <mutex>
#include "actors.inc"

namespace external_stage_v8 {
struct Stamp {int64_t begin=0,end=0;};
struct Block {int64_t begin=0,end=0;};
static void contract(const json &r){
 const auto split=str(r,"split"),mode=str(r,"instrumentation");
 require(split=="profile"||split=="heldout","predeclared split required");
 require(mode=="stamped"||mode=="runtime_callbacks","unknown observer mode");
 require(r.at("attention_action_index")==22&&r.at("actor_epoch_delay_ms")==1000&&r.at("target_entry_max_lateness_ns")==10000000,"fixed action/controller contract differs");
 require(r.at("cpu_actor").at("main_thread_mask")==4&&r.at("gpu_actor").at("main_thread_mask")==65536,"shared CPU2 target and CPU16 GPU submitter required");
 require(r.at("cpu_actor").at("init_policy")=="pinned_init"&&r.at("gpu_actor").at("init_policy")=="pinned_init"&&r.at("init_policy")=="pinned_init","original pinned initialization required");
 if(split=="profile"){
  require(r.at("replays")==1&&r.at("block_replays")==1&&r.at("max_API_records")==1024,"one complete336-call Attention observation required");
  const auto state=r.at("offered_state");require(state.is_array()&&state.size()==2,"exact offered state required");
  for(int i=0;i<2;++i){require(state.at(i)==0||state.at(i)==1,"binary offered state required");
   const auto expected=state.at(i)==1?json::array({json::array({0,30000})}):json::array();
   require(r.at(i==0?"cpu_actor":"gpu_actor").at("intervals_ms")==expected,"fixed30-second independent ON window required");}
  require(r.at("target_entry_offset_ms")==((state.at(0)==0&&state.at(1)==0)?0:1000),"steady ON entry/off control contract differs");
 }else{
  require(mode=="stamped"&&r.at("replays")==256&&r.at("block_replays")==16&&r.at("max_API_records")==131072&&r.at("target_entry_offset_ms")==0,"heldout is256 complete original Attention graphs,16 perblock, no callback");
  for(const auto &device:{"cpu_actor","gpu_actor"})for(const auto &p:r.at(device).at("intervals_ms"))require(p.at(0)>=100&&p.at(1)<=2000,"heldout needs initial OFF and fixed bounded pulses");
 }
}
static json run(Executor &e,const steady::Request &sampling,const json &r,steady::OwnThreadAffinity &initialization){
 contract(r);require(e.gpu&&e.program.at("physical").at("threads")==1&&sampling.poll=="0"&&sampling.warmup==20&&sampling.min_ms==500,"retained fixed backend/thread/warmup required");
 const auto &actions=e.program.at("actions");const int attention=r.at("attention_action_index").get<int>();
 require(actions.at(attention).at("kind")=="COMPUTE"&&actions.at(attention).at("device")=="gpu"&&str(actions.at(attention),"target_id").find(":attn")!=std::string::npos,"actual prepared GPU Attention required");
 const auto pool=single_layer::configure_pool(e,"0");e.validate();const auto initial=single_layer::capture_verification(e);
 Control control;Actor cpu(control,r.at("cpu_actor")),gpu(control,r.at("gpu_actor"));
 try{
  dynamic_control::OwnedHighResolutionTimer timer;const auto waiting=steady::tick();
  while(!cpu.ready()||!gpu.ready()){require(steady::ns(waiting,steady::tick())<30000000000LL,"actor preparation timed out");timer.wait_ns(1000000);}cpu.check_error();gpu.check_error();
  const auto init_threads=diagnostic_v3::own_thread_inventory("all_backends_and_actors_ready_before_future_epoch");
  const auto init_affinity=initialization.finish();steady::OwnThreadAffinity target_affinity(4);
  auto full=[&](){for(const auto &a:actions)e.action(a);};const auto warm=steady::warm(sampling,full);e.reset();full();
  const int replays=r.at("replays").get<int>(),per_block=r.at("block_replays").get<int>();
  require(replays>0&&replays<=256&&per_block>0&&replays%per_block==0,"bounded whole-block replay count required");
  require(sampling.mode=="action"&&sampling.target==attention&&sampling.trials==replays/per_block&&sampling.repeats==per_block&&sampling.instrumentation=="bare"&&sampling.main_mask==4,"sampling/executed stage inventory differs");
  std::vector<Stamp> stamps(static_cast<size_t>(replays));std::vector<Block> blocks(static_cast<size_t>(replays/per_block));
  runtime_callbacks_v6::Observer callbacks(str(r,"instrumentation")=="runtime_callbacks",GetCurrentThreadId(),r.at("max_API_records").get<size_t>());
  // All initialization, allocation and observer subscription precede epoch selection.
  const int64_t hz=steady::frequency(),epoch=steady::tick()+hz,entry=epoch+r.at("target_entry_offset_ms").get<int64_t>()*hz/1000;
  control.announce(epoch);require(timer.wait_until_qpc(entry,hz,control.cancelled),"target wait cancelled");cpu.check_error();gpu.check_error();
  const auto begin=steady::tick();
  for(int b=0;b<replays/per_block;++b){blocks[size_t(b)].begin=steady::tick();
   for(int j=0;j<per_block;++j){const int index=b*per_block+j;auto &s=stamps[size_t(index)];callbacks.arm(uint64_t(index+1));
    s.begin=steady::tick();e.action(actions.at(attention));s.end=steady::tick();callbacks.disarm();}
   blocks[size_t(b)].end=steady::tick();cpu.check_error();gpu.check_error();
  }
  const auto end=steady::tick();control.finish();const auto cpu_result=cpu.finish(),gpu_result=gpu.finish();
  // Capture the completed timed graph's ports before any reset/replay overwrites it.
  const auto final=single_layer::capture_verification(e);const auto finalize_begin=steady::tick();auto records=callbacks.finish();const auto finalize_end=steady::tick();
  json rows=json::array();for(int b=0;b<replays/per_block;++b){json actual=json::array();for(int j=0;j<per_block;++j){int index=b*per_block+j;const auto &s=stamps[size_t(index)];actual.push_back(json{{"global_replay",index},{"action_index",attention},{"callback_phase_id",index+1},{"begin_tick",s.begin},{"end_tick",s.end},{"wall_ns",steady::ns(s.begin,s.end)}});}
   rows.push_back(json{{"block",b},{"replays",per_block},{"begin_tick",blocks[size_t(b)].begin},{"end_tick",blocks[size_t(b)].end},{"wall_ns",steady::ns(blocks[size_t(b)].begin,blocks[size_t(b)].end)},{"actions",actual}});}
  const auto late=steady::ns(entry,begin);const bool entry_ok=begin>=entry&&late<=10000000;
  const bool contained=str(r,"split")!="profile"||(begin>=epoch&&end<=epoch+hz*30);
  return json{{"program_sha256",e.program.at("program_sha256")},{"actions",actions},{"target_action",actions.at(attention)},{"attention_action_index",attention},{"instrumentation",r.at("instrumentation")},{"blocks",rows},{"runtime_callbacks",records},
   {"target_expected_origin_tick",epoch},{"planned_target_entry_tick",entry},{"target_begin_tick",begin},{"target_end_tick",end},{"target_wall_ns",steady::ns(begin,end)},{"initial_start_lateness_ns",late},{"target_entry_controller_passed",entry_ok},{"profile_window_contained",contained},
   {"profile_state_support",entry_ok&&contained?"candidate_individual_stage_labels_only":"unsupported_entry_or_state_window"},{"steady_window_begin_tick",epoch},{"steady_window_end_tick",epoch+hz*30},
   {"callback_finalize_begin_tick",finalize_begin},{"callback_finalize_end_tick",finalize_end},{"callback_finalize_ns_outside_target",steady::ns(finalize_begin,finalize_end)},{"observer_flush_gap_in_target_wall_ns",0},{"activity_capture_instantiated",false},
   {"warmup",warm},{"threadpool",pool},{"init_policy",r.at("init_policy")},{"initialization_affinity",init_affinity},{"target_thread_affinity",target_affinity.finish()},{"threads_after_initialization",init_threads},{"actors",json{{"cpu",cpu_result},{"gpu",gpu_result}}},
   {"verification_native",final},{"native_repeat_consistency",state_response::compare_ports(initial.at("ports"),final.at("ports"))},{"timed_result_checked_before_reset",true},
   {"continuous_replays",replays},{"reset_prime_between_timed_blocks",false},{"complete_attention_graph_only",true},{"complete_LLM_strategy_replayed",false},
   {"actor_timeline_triggered_by_target_latency",false},{"future_target_state_used",false},{"CPU_running_ready_blocked_identified",false},{"CUPTI_activity_GPU_time_collected",false},
   {"API_response_is_CPU_active",false},{"internal_GPU_work_added",false},{"hardware_accuracy_qualified",false},{"performance_parameters_fitted",false},
   {"scope","one fixed-shape complete Attention graph, original Executor.action; controlled ownedGEMM actors; inclusive host response stages; no claim of wholeLLM dynamic accuracy"}};
 }catch(...){control.abort();throw;}
}
}
int main(int argc,char **argv){try{
 if(argc==2&&std::string(argv[1])=="--self-test"){auto result=steady::self_test();result["hardware_executed"]=false;result["actor_or_backend_created"]=false;result["source_sha256"]=EXTERNAL_STAGE_V8_SOURCE_SHA256;std::cout<<result.dump()<<'\n';return 0;}
 require(argc==4,"usage: dynamic-external-stage-v8 PROGRAM REQUEST NEW_OUTPUT");require(!state_response::exists(argv[3]),"new exclusive output required");_putenv_s("NVIDIA_TF32_OVERRIDE","0");_putenv_s("GGML_CUDA_DISABLE_GRAPHS","1");
 std::ifstream p(std::filesystem::u8path(argv[1])),q(std::filesystem::u8path(argv[2]));json program,r;require(p.good()&&q.good(),"explicit inputs unavailable");p>>program;q>>r;
 require(r.at("schema")=="dynamic-external-stage-request-8"&&str(r,"source_sha256")==EXTERNAL_STAGE_V8_SOURCE_SHA256&&str(r,"callback_source_sha256")==EXTERNAL_STAGE_V8_CALLBACK_SHA256&&str(r,"actor_source_sha256")==EXTERNAL_STAGE_V8_ACTOR_SHA256,"request/binary/source differs");
 external_stage_v8::contract(r);steady::Request sampling(r.at("sampling"));const auto process=external_stage_v8::process_affinity(r);steady::OwnThreadAffinity initialization(4);json result;
 {Executor e(std::move(program));result=external_stage_v8::run(e,sampling,r,initialization);}DWORD_PTR actual=0,system=0;require(GetProcessAffinityMask(GetCurrentProcess(),&actual,&system)!=0&&actual==4294967295ULL,"owned process affinity changed");
 auto proof=process;proof["final_mask"]=uint64_t(actual);result.update(json{{"schema","dynamic-external-stage-native-8"},{"request",r},{"source_sha256",EXTERNAL_STAGE_V8_SOURCE_SHA256},{"callback_source_sha256",EXTERNAL_STAGE_V8_CALLBACK_SHA256},{"actor_source_sha256",EXTERNAL_STAGE_V8_ACTOR_SHA256},{"process_id",single_layer::pid()},{"main_thread_id",GetCurrentThreadId()},{"process_affinity",proof},{"clock",json{{"clock_id","windows_qpc"},{"frequency_hz",steady::frequency()}}}});
 state_response::write_new(argv[3],result);return 0;
}catch(const std::exception &e){std::cerr<<"dynamic-external-stage-v8: "<<e.what()<<'\n';return 2;}}
