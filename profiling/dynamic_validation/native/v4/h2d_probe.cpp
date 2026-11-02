#include "../v3/common.hpp"
#include "../v3/thread_inventory.hpp"
#include "memcpy_capture.hpp"
#include <psapi.h>

namespace diagnostic_v4 {
using namespace diagnostic_v3;
static json state(const char *window,bool prior){
    const auto begin=steady::tick();FILETIME created{},exited{},kernel{},user{};ULONG64 cycles=0;MEMORYSTATUSEX memory{};memory.dwLength=sizeof(memory);
    const bool cpu=GetThreadTimes(GetCurrentThread(),&created,&exited,&kernel,&user)!=0,cycle=QueryThreadCycleTime(GetCurrentThread(),&cycles)!=0,ram=GlobalMemoryStatusEx(&memory)!=0;
    auto ft=[](const FILETIME &v){return(uint64_t(v.dwHighDateTime)<<32)|v.dwLowDateTime;};
    return json{{"query_begin_tick",begin},{"query_end_tick",steady::tick()},{"window",window},{"known_before_target_burst",prior},
        {"thread_user_100ns",cpu?json(ft(user)):json(nullptr)},{"thread_kernel_100ns",cpu?json(ft(kernel)):json(nullptr)},
        {"thread_cycles",cycle?json(cycles):json(nullptr)},{"host_available_bytes",ram?json(memory.ullAvailPhys):json(nullptr)},
        {"running_ready_blocked_separated",false},{"CPU_cycles_converted_to_time",false},{"future_target_state_used_for_forecast",false}};
}
static json pages(void *source,size_t bytes,const char *window){
    const auto begin=steady::tick();MEMORY_BASIC_INFORMATION mbi{};const auto queried=VirtualQuery(source,&mbi,sizeof(mbi));
    SYSTEM_INFO info{};GetSystemInfo(&info);require(info.dwPageSize>0,"OS page size unavailable");
    const auto first=reinterpret_cast<uintptr_t>(source)/info.dwPageSize*info.dwPageSize;
    const auto last=(reinterpret_cast<uintptr_t>(source)+bytes-1)/info.dwPageSize*info.dwPageSize;
    std::vector<PSAPI_WORKING_SET_EX_INFORMATION> observations((last-first)/info.dwPageSize+1);
    for(size_t i=0;i<observations.size();++i)observations[i].VirtualAddress=reinterpret_cast<void*>(first+i*info.dwPageSize);
    const bool ok=QueryWorkingSetEx(GetCurrentProcess(),observations.data(),static_cast<DWORD>(observations.size()*sizeof(observations[0])))!=0;
    const auto error=ok?0:GetLastError();size_t valid=0,locked=0,shared=0,large=0;
    if(ok)for(const auto &x:observations){valid+=x.VirtualAttributes.Valid!=0;if(x.VirtualAttributes.Valid){locked+=x.VirtualAttributes.Locked!=0;shared+=x.VirtualAttributes.Shared!=0;large+=x.VirtualAttributes.LargePage!=0;}}
    return json{{"window",window},{"query_begin_tick",begin},{"query_end_tick",steady::tick()},{"target_bytes",bytes},
        {"VirtualQuery_available",queried==sizeof(mbi)},{"region_bytes",queried?json(mbi.RegionSize):json(nullptr)},
        {"region_state",queried?json(mbi.State):json(nullptr)},{"region_protection",queried?json(mbi.Protect):json(nullptr)},
        {"region_type",queried?json(mbi.Type):json(nullptr)},{"page_bytes",info.dwPageSize},{"source_pages",observations.size()},
        {"QueryWorkingSetEx_available",ok},{"QueryWorkingSetEx_error",error},{"valid_pages",ok?json(valid):json(nullptr)},
        {"OS_working_set_locked_bit_pages",ok?json(locked):json(nullptr)},{"shared_pages",ok?json(shared):json(nullptr)},
        {"large_pages",ok?json(large):json(nullptr)},{"OS_locked_bit_is_CUDA_registration_proof",false},{"snapshot_outside_condition_timers",true}};
}
struct Condition {int trial=0,slot=0,index=0;int64_t reset_begin=0,reset_end=0,prime_begin=0,prime_end=0,idle_begin=0,idle_end=0,burst_begin=0,burst_end=0;json prior,before,after;};
static json block(Executor &e,const steady::Request &sampling,const json &request,const std::string &mode,int index){
    require(mode=="bare"||mode=="memcpy_only","explicit low-intrusion observer mode required");
    const auto &actions=e.program.at("actions"),&target=actions.at(sampling.target);
    auto full=[&](){for(const auto &a:actions)e.action(a);};
    std::vector<Condition> conditions(64);std::vector<Copy> copies(1024);std::vector<uint64_t> ids(1024,0);
    std::unique_ptr<memcpy_capture_v4::Capture> capture;if(mode=="memcpy_only")capture=std::make_unique<memcpy_capture_v4::Capture>();
    const auto threads=own_thread_inventory("after_block_observer_setup_before_conditions");dynamic_control::OwnedHighResolutionTimer timer;
    size_t si=0,ci=0;const auto begin=steady::tick();
    for(int trial=0;trial<16;++trial)for(int slot=0;slot<4;++slot){
        auto &s=conditions[si++];s.trial=trial;s.slot=slot;s.index=request.at("orders").at(trial).at(slot).get<int>();
        const int idle=request.at("idle_ms").at(s.index).get<int>();const bool focused=idle==0||idle==20;
        s.prior=state("before_reset_prime",true);s.reset_begin=steady::tick();e.reset();s.reset_end=steady::tick();
        s.prime_begin=steady::tick();full();s.prime_end=steady::tick();s.idle_begin=steady::tick();if(idle)timer.wait_ns(int64_t(idle)*1000000);s.idle_end=steady::tick();
        s.before=state("before_target_burst",true);s.burst_begin=steady::tick();
        for(int ordinal=0;ordinal<16;++ordinal){auto &c=copies[ci];if(capture&&focused){ids[ci]=uint64_t(index)*1024+ci+1;capture->push(ids[ci]);}
            c.begin=steady::tick();e.action(target);c.end=steady::tick();if(ids[ci])capture->pop(ids[ci]);++ci;}
        s.burst_end=steady::tick();s.after=state("after_target_burst_diagnostic_only",false);
    }
    const auto end=steady::tick();json physical=capture?capture->finish():json{{"schema","memcpy-only-cupti-4"},{"enabled",false},{"records",json::array()}};capture.reset();
    json rows=json::array();ci=0;
    for(const auto &s:conditions){json values=json::array();int64_t last=s.idle_end;
        for(int ordinal=0;ordinal<16;++ordinal){const auto &c=copies[ci];values.push_back(json{{"ordinal",ordinal},{"begin_tick",c.begin},{"end_tick",c.end},
            {"wall_ns",steady::ns(c.begin,c.end)},{"gap_since_previous_completion_ns",steady::ns(last,c.begin)},{"external_id",ids[ci]?json(ids[ci]):json(nullptr)}});last=c.end;++ci;}
        rows.push_back(json{{"trial",s.trial},{"condition_slot",s.slot},{"condition_index",s.index},{"requested_idle_ms",request.at("idle_ms").at(s.index)},
            {"prior_state",s.prior},{"before_burst_state",s.before},{"after_burst_state",s.after},
            {"reset_begin_tick",s.reset_begin},{"reset_end_tick",s.reset_end},{"reset_ns",steady::ns(s.reset_begin,s.reset_end)},
            {"prime_begin_tick",s.prime_begin},{"prime_end_tick",s.prime_end},{"prime_ns",steady::ns(s.prime_begin,s.prime_end)},
            {"idle_begin_tick",s.idle_begin},{"idle_end_tick",s.idle_end},{"actual_idle_ns",steady::ns(s.idle_begin,s.idle_end)},
            {"effective_prime_end_to_first_copy_ns",steady::ns(s.prime_end,copies[(s.trial*4+s.slot)*16].begin)},
            {"burst_begin_tick",s.burst_begin},{"burst_end_tick",s.burst_end},{"burst_wall_ns",steady::ns(s.burst_begin,s.burst_end)},{"copies",values}});
    }
    return json{{"block",index},{"instrumentation",mode},{"experiment_begin_tick",begin},{"experiment_end_tick",end},{"sequences",rows},{"cupti",physical},
        {"threads_after_observer_setup",threads},{"per_copy_Executor_action_unchanged",true},{"push_pop_outside_per_copy_timer",true},
        {"registered_CUPTI_lifecycle_may_persist_into_later_bare_blocks",index>0},{"activity_span_added_to_host_wall",false}};
}
static json run(Executor &e,const steady::Request &sampling,const json &request,const json &pool,const json &initial){
    require(sampling.mode=="action"&&sampling.trials==16&&sampling.repeats==16&&sampling.instrumentation=="bare","retained16x4x16 protocol required");
    validate_orders(request.at("idle_ms"),request.at("orders"),16);require(request.at("focus_idle_ms")==json::array({0,20}),"focused0/20 conditions required");
    const auto &target=e.program.at("actions").at(sampling.target);require(e.gpu&&target.at("kind")=="COPY"&&target.at("src")=="dram"&&target.at("dst")=="vram","retained actual H2D target required");
    auto *source=e.tensor(str(target,"target_id"),"dram"),*destination=e.tensor(str(target,"target_id"),"vram");
    require(ggml_nbytes(source)==2097152&&ggml_nbytes(destination)==2097152,"actual retained2MiB required");
    const auto order=request.at("block_modes");require(order==json::array({"bare","memcpy_only","bare"})||order==json::array({"memcpy_only","bare","memcpy_only"}),"predeclared ABA/BAB only");
    Registration registration(e,str(request,"memory_mode"));auto full=[&](){for(const auto &a:e.program.at("actions"))e.action(a);};
    const auto warm=steady::warm(sampling,full),before=payload(source,destination),pages_before=pages(source->data,ggml_nbytes(source),"after_common_warmup_before_all_blocks");
    json blocks=json::array();for(int i=0;i<3;++i)blocks.push_back(block(e,sampling,request,order.at(i).get<std::string>(),i));
    const auto pages_after=pages(source->data,ggml_nbytes(source),"after_all_blocks"),after=payload(source,destination),final=single_layer::capture_verification(e);
    require(before.at("source_sha256")==after.at("source_sha256"),"owned source changed");
    return json{{"program_sha256",e.program.at("program_sha256")},{"target_action_index",sampling.target},{"target_action",target},{"target_bytes",2097152},
        {"payload_before",before},{"payload_after",after},{"source_pointer_during_measurement",pointer_kind(source->data)},{"source_pages_before",pages_before},{"source_pages_after",pages_after},
        {"arena_registration",registration.finish()},{"warmup",warm},{"threadpool",pool},{"blocks",blocks},{"verification_native",final},
        {"native_repeat_consistency",state_response::compare_ports(initial.at("ports"),final.at("ports"))},{"idle_controller",dynamic_control::OwnedHighResolutionTimer::describe()},
        {"fresh_context_is_equivalent_to_v2",false},{"all_blocks_share_backend_and_registered_arena",true},{"performance_parameters_fitted",false},
        {"scope","low-intrusion observer diagnosis; block order and CUPTI lifecycle are not a pure overhead subtraction or a new response model"}};
}
}
int main(int argc,char **argv){try{
    if(argc==2&&std::string(argv[1])=="--self-test"){auto result=steady::self_test();result["source_sha256"]=H2D_V4_SOURCE_SHA256;result["backend_or_capture_constructed"]=false;std::cout<<result.dump()<<'\n';return 0;}
    require(argc==4,"usage: dynamic-h2d-memcpy-v4 PROGRAM REQUEST OUTPUT");require(!state_response::exists(argv[3]),"v4 output exists");
    _putenv_s("NVIDIA_TF32_OVERRIDE","0");_putenv_s("GGML_CUDA_DISABLE_GRAPHS","1");
    std::ifstream pf(std::filesystem::u8path(argv[1])),rf(std::filesystem::u8path(argv[2]));json program,request;require(pf.good()&&rf.good(),"explicit inputs unavailable");pf>>program;rf>>request;
    require(request.at("schema")=="dynamic-h2d-memcpy-request-4"&&str(request,"source_sha256")==H2D_V4_SOURCE_SHA256,"v4 request/source differs");
    steady::Request sampling(request.at("sampling"));require(sampling.main_mask==4,"target CPU2 required");const auto affinity=diagnostic_v3::apply_affinity(request);
    const auto before=diagnostic_v3::own_thread_inventory("before_backend_initialization");steady::OwnThreadAffinity thread(4);
    Executor executor(std::move(program));const auto pool=single_layer::configure_pool(executor,sampling.poll);executor.validate();const auto initial=single_layer::capture_verification(executor);
    const auto initialized=diagnostic_v3::own_thread_inventory("after_pinned_backend_first_validation");
    auto result=diagnostic_v4::run(executor,sampling,request,pool,initial);
    result.update(json{{"schema","dynamic-h2d-memcpy-native-4"},{"request",request},{"source_sha256",H2D_V4_SOURCE_SHA256},
        {"process_id",single_layer::pid()},{"main_thread_id",GetCurrentThreadId()},{"clock",state_response::clock_json()},
        {"process_affinity",diagnostic_v3::finish_affinity(affinity)},{"main_thread_affinity",thread.finish()},
        {"threads_before_initialization",before},{"threads_after_initialization",initialized}});
    state_response::write_new(argv[3],result);return 0;
}catch(const std::exception &e){std::cerr<<"dynamic-h2d-memcpy-v4: "<<e.what()<<'\n';return 2;}}
