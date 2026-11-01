#include "common.hpp"
#include "thread_inventory.hpp"

namespace diagnostic_v3 {
static json state_snapshot(const char *window,bool prior){
    const auto begin=steady::tick();FILETIME created{},exited{},kernel{},user{};ULONG64 cycles=0;MEMORYSTATUSEX memory{};memory.dwLength=sizeof(memory);
    const bool cpu=GetThreadTimes(GetCurrentThread(),&created,&exited,&kernel,&user)!=0,cycle=QueryThreadCycleTime(GetCurrentThread(),&cycles)!=0,ram=GlobalMemoryStatusEx(&memory)!=0;
    auto ft=[](const FILETIME &v){return (uint64_t(v.dwHighDateTime)<<32)|v.dwLowDateTime;};
    return json{{"query_begin_tick",begin},{"query_end_tick",steady::tick()},
        {"thread_user_100ns",cpu?json(ft(user)):json(nullptr)},{"thread_kernel_100ns",cpu?json(ft(kernel)):json(nullptr)},
        {"thread_cycles",cycle?json(cycles):json(nullptr)},{"host_available_bytes",ram?json(memory.ullAvailPhys):json(nullptr)},
        {"window",window},{"known_before_target_burst",prior},{"future_target_state_used_for_forecast",false},
        {"running_ready_blocked_separated",false},{"CPU_cycles_converted_to_time",false}};
}
struct HostStage {int64_t begin=0,api_enter=0,api_return=0,gpu_enter=0,gpu_return=0,cpu_enter=0,cpu_return=0,end=0;};
static void copy_stages(Executor &e,const json &a,HostStage &s){
    s.begin=steady::tick();
    auto *source=e.tensor(str(a,"target_id"),str(a,"src")),*dest=e.tensor(str(a,"target_id"),str(a,"dst"));
    s.api_enter=steady::tick();ggml_backend_tensor_copy(source,dest);s.api_return=steady::tick();
    s.gpu_enter=steady::tick();if(e.gpu)ggml_backend_synchronize(e.gpu);s.gpu_return=steady::tick();
    s.cpu_enter=steady::tick();ggml_backend_synchronize(e.cpu);s.cpu_return=steady::tick();s.end=steady::tick();
}
static json stage_json(const HostStage &s){
    return json{{"begin_tick",s.begin},{"copy_API_enter_tick",s.api_enter},{"copy_API_return_tick",s.api_return},
        {"extra_GPU_sync_enter_tick",s.gpu_enter},{"extra_GPU_sync_return_tick",s.gpu_return},
        {"CPU_sync_enter_tick",s.cpu_enter},{"CPU_sync_return_tick",s.cpu_return},{"end_tick",s.end},
        {"host_wall_ns",steady::ns(s.begin,s.end)},{"copy_API_inclusive_ns",steady::ns(s.api_enter,s.api_return)},
        {"extra_GPU_sync_ns",steady::ns(s.gpu_enter,s.gpu_return)},{"CPU_sync_ns",steady::ns(s.cpu_enter,s.cpu_return)},
        {"QPC_wrapper_gaps_ns",steady::ns(s.begin,s.api_enter)+steady::ns(s.api_return,s.gpu_enter)+steady::ns(s.gpu_return,s.cpu_enter)+steady::ns(s.cpu_return,s.end)},
        {"internal_submit_and_sync_identified_by","CUPTI correlated runtime/driver API records, not this inclusive GGML API span"},
        {"device_span_added_to_host_wall",false}};
}
struct Condition {int trial=0,slot=0,index=0;int64_t reset_begin=0,reset_end=0,prime_begin=0,prime_end=0,idle_begin=0,idle_end=0,burst_begin=0,burst_end=0;json prior,before_burst,after_burst;};
static json run_h2d(Executor &e,const steady::Request &sampling,const json &request,const json &pool,const json &initial){
    require(sampling.mode=="action"&&sampling.target>=0&&sampling.trials==16&&sampling.repeats==16&&sampling.instrumentation=="bare","explicit retained16x4x16 H2D protocol required");
    validate_orders(request.at("idle_ms"),request.at("orders"),16);
    const auto mode=str(request,"instrumentation");require(mode=="bare"||mode=="stages","unknown v3 instrumentation");
    require(request.at("focus_idle_ms")==json::array({0,20}),"predeclared focused idle conditions required");
    const auto &actions=e.program.at("actions");const auto &target=actions.at(sampling.target);
    require(target.at("kind")=="COPY"&&target.at("src")=="dram"&&target.at("dst")=="vram","actual retained H2D target required");
    auto *source=e.tensor(str(target,"target_id"),"dram"),*dest=e.tensor(str(target,"target_id"),"vram");
    require(e.gpu&&ggml_nbytes(source)==2097152&&ggml_nbytes(dest)==2097152,"actual GPU and two-MiB retained payload required");
    Registration registration(e,str(request,"memory_mode"));auto full=[&](){for(const auto &a:actions)e.action(a);};
    const auto warm=steady::warm(sampling,full);const auto before=payload(source,dest),category=pointer_kind(source->data);
    const auto warmed_threads=own_thread_inventory("after_warmup_before_condition_timers");
    std::vector<Condition> conditions(64);std::vector<Copy> copies(1024);std::vector<HostStage> stages(1024);std::vector<uint64_t> ids(1024,0);
    dynamic_control::OwnedHighResolutionTimer timer;
    std::unique_ptr<physical_capture::Capture> capture;if(mode=="stages")capture=std::make_unique<physical_capture::Capture>();
    const auto observer_threads=own_thread_inventory("after_observer_setup_before_condition_timers");
    size_t si=0,ci=0;const auto experiment_begin=steady::tick();
    for(int trial=0;trial<16;++trial)for(int slot=0;slot<4;++slot){
        auto &s=conditions[si++];s.trial=trial;s.slot=slot;s.index=request.at("orders").at(trial).at(slot).get<int>();
        const int idle=request.at("idle_ms").at(s.index).get<int>();const bool focused=idle==0||idle==20;s.prior=state_snapshot("before_reset_prime",true);
        s.reset_begin=steady::tick();e.reset();s.reset_end=steady::tick();s.prime_begin=steady::tick();full();s.prime_end=steady::tick();
        s.idle_begin=steady::tick();if(idle)timer.wait_ns(int64_t(idle)*1000000);s.idle_end=steady::tick();s.before_burst=state_snapshot("before_target_burst",true);s.burst_begin=steady::tick();
        for(int ordinal=0;ordinal<16;++ordinal){auto &c=copies[ci];
            if(capture&&focused){ids[ci]=ci+1;capture->push(ids[ci]);copy_stages(e,target,stages[ci]);capture->pop(ids[ci]);c.begin=stages[ci].begin;c.end=stages[ci].end;}
            else {c.begin=steady::tick();e.action(target);c.end=steady::tick();}
            ++ci;
        }
        s.burst_end=steady::tick();s.after_burst=state_snapshot("after_target_burst_diagnostic_only",false);
    }
    const auto experiment_end=steady::tick();json physical=capture?capture->finish():json{{"schema","physical-stage-cupti-1"},{"enabled",false},{"records",json::array()}};
    const auto after=payload(source,dest);require(before.at("source_sha256")==after.at("source_sha256"),"owned source changed");
    const auto final=single_layer::capture_verification(e);json rows=json::array();ci=0;
    for(const auto &s:conditions){json values=json::array();int64_t last=s.idle_end;
        for(int ordinal=0;ordinal<16;++ordinal){const auto &c=copies[ci];values.push_back(json{{"ordinal",ordinal},{"begin_tick",c.begin},{"end_tick",c.end},
            {"wall_ns",steady::ns(c.begin,c.end)},{"gap_since_previous_completion_ns",steady::ns(last,c.begin)},
            {"external_id",ids[ci]?json(ids[ci]):json(nullptr)},{"host_stages",ids[ci]?stage_json(stages[ci]):json(nullptr)}});last=c.end;++ci;}
        rows.push_back(json{{"trial",s.trial},{"condition_slot",s.slot},{"condition_index",s.index},{"requested_idle_ms",request.at("idle_ms").at(s.index)},
            {"prior_state",s.prior},{"before_burst_state",s.before_burst},{"after_burst_state",s.after_burst},
            {"CPU_counter_bracket_scope","includes counter query gaps; not exact per-action running/ready/blocked identification"},
            {"reset_begin_tick",s.reset_begin},{"reset_end_tick",s.reset_end},{"reset_ns",steady::ns(s.reset_begin,s.reset_end)},
            {"prime_begin_tick",s.prime_begin},{"prime_end_tick",s.prime_end},{"prime_ns",steady::ns(s.prime_begin,s.prime_end)},
            {"idle_begin_tick",s.idle_begin},{"idle_end_tick",s.idle_end},{"actual_idle_ns",steady::ns(s.idle_begin,s.idle_end)},
            {"effective_prime_end_to_first_copy_ns",steady::ns(s.prime_end,copies[(s.trial*4+s.slot)*16].begin)},
            {"burst_begin_tick",s.burst_begin},{"burst_end_tick",s.burst_end},{"burst_wall_ns",steady::ns(s.burst_begin,s.burst_end)},{"copies",values}});
    }
    return json{{"program_sha256",e.program.at("program_sha256")},{"target_action_index",sampling.target},{"target_action",target},{"target_bytes",2097152},
        {"source_pointer_during_measurement",category},{"payload_before",before},{"payload_after",after},{"arena_registration",registration.finish()},
        {"warmup",warm},{"threadpool",pool},{"threads_after_warmup",warmed_threads},{"threads_after_observer_setup",observer_threads},
        {"sequences",rows},{"experiment_begin_tick",experiment_begin},{"experiment_end_tick",experiment_end},
        {"cupti",physical},{"verification_native",final},{"native_repeat_consistency",state_response::compare_ports(initial.at("ports"),final.at("ports"))},
        {"idle_controller",dynamic_control::OwnedHighResolutionTimer::describe()},
        {"focused_diagnostic_conditions_ms",request.at("focus_idle_ms")},{"conditioning_idle_schedule_preserved",true},
        {"bare_uses_unchanged_Executor_action",true},{"observer_copy_operation_sequence_matches_Executor",true},
        {"host_wall_partition_is_additive",true},{"host_and_device_intervals_are_additive",false},
        {"device_stream_inferred_from_PTDS_or_default",false},{"physical_device_stream_source","CUPTI activity records only"},
        {"CPU_running_ready_blocked_identified",false},{"performance_parameters_fitted",false},{"future_target_state_used_for_prediction",false},
        {"scope","diagnosis only; matched bare/observer evidence, no new performance prices or accuracy claim"}};
}
}

int main(int argc,char **argv){try{
    if(argc==2&&std::string(argv[1])=="--self-test"){
        diagnostic_v3::validate_orders(json::array({0,1,5,20}),json::array({json::array({0,1,2,3})}),1);
        auto result=steady::self_test();result["source_sha256"]=H2D_V3_SOURCE_SHA256;
        result["stage_schema_parsed_without_capture_or_backend"]=true;std::cout<<result.dump()<<'\n';return 0;
    }
    require(argc==4,"usage: dynamic-h2d-stages-v3 PROGRAM REQUEST OUTPUT");require(!state_response::exists(argv[3]),"v3 output exists");
    _putenv_s("NVIDIA_TF32_OVERRIDE","0");_putenv_s("GGML_CUDA_DISABLE_GRAPHS","1");
    std::ifstream pf(std::filesystem::u8path(argv[1])),rf(std::filesystem::u8path(argv[2]));json program,request;require(pf.good()&&rf.good(),"explicit inputs unavailable");pf>>program;rf>>request;
    require(request.at("schema")=="dynamic-h2d-stage-request-3"&&str(request,"source_sha256")==H2D_V3_SOURCE_SHA256,"v3 request/source differs");
    steady::Request sampling(request.at("sampling"));const auto affinity=diagnostic_v3::apply_affinity(request);json result;
    const auto policy=str(request,"init_policy");require(policy=="pinned_init"||policy=="wide_init","explicit init policy required");
    const auto before_threads=diagnostic_v3::own_thread_inventory("before_Executor_initialization");
    std::unique_ptr<Executor> executor;json pool,initial,init_affinity,after_threads;
    {steady::OwnThreadAffinity init(policy=="pinned_init"?sampling.main_mask:static_cast<DWORD_PTR>(4294967295ULL));
        executor=std::make_unique<Executor>(std::move(program));pool=single_layer::configure_pool(*executor,sampling.poll);
        executor->validate();initial=single_layer::capture_verification(*executor);
        after_threads=diagnostic_v3::own_thread_inventory("after_backend_and_first_validation_before_target_pin");init_affinity=init.finish();}
    steady::OwnThreadAffinity thread(sampling.main_mask);
    result=diagnostic_v3::run_h2d(*executor,sampling,request,pool,initial);executor.reset();
    result["init_policy"]=policy;result["initialization_affinity"]=init_affinity;result["threads_before_initialization"]=before_threads;result["threads_after_initialization"]=after_threads;
    result.update(json{{"schema","dynamic-h2d-stage-native-3"},{"request",request},{"source_sha256",H2D_V3_SOURCE_SHA256},
        {"process_id",single_layer::pid()},{"main_thread_id",GetCurrentThreadId()},{"clock",state_response::clock_json()},{"process_affinity",diagnostic_v3::finish_affinity(affinity)},
        {"main_thread_affinity",thread.finish()}});state_response::write_new(argv[3],result);return 0;
}catch(const std::exception &e){std::cerr<<"dynamic-h2d-stages-v3: "<<e.what()<<'\n';return 2;}}
