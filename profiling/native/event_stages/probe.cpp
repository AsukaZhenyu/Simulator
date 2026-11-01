// Event-parked physical-stage metrology. Historical Executor.action is unchanged.
#include "state_response_generated.inc"
#include "cupti_capture.hpp"

namespace physical_stages {
using physical_capture::qpc_now_ns;
using state_response::exists;
using state_response::write_new;

struct CyclePoint {uint64_t begin=0,end=0,value=0;bool known=false;DWORD error=0;};
static CyclePoint cycle_point() {
    CyclePoint result;result.begin=qpc_now_ns();ULONG64 value=0;
    result.known=QueryThreadCycleTime(GetCurrentThread(),&value)!=0;
    result.error=result.known?0:GetLastError();result.value=value;result.end=qpc_now_ns();return result;
}
struct ActionInterval { uint64_t begin=0,end=0,external=0; size_t index=0;CyclePoint cycles_before,cycles_after; };
struct Invocation { uint64_t begin=0,end=0,reset_begin=0,reset_end=0; };

class EventParking {
    HANDLE go_=nullptr,stop_=nullptr,timer_=nullptr;
    std::wstring go_name_,stop_name_;
    bool high_resolution_=false;
    DWORD timer_fallback_error_=0;
    static std::wstring env(const wchar_t *name) {
        const DWORD size=GetEnvironmentVariableW(name,nullptr,0);
        require(size>1&&size<=512,"explicit parent-owned Local event name required");
        std::wstring value(size,L'\0');
        const DWORD copied=GetEnvironmentVariableW(name,value.data(),size);
        require(copied>0&&copied<size,"event environment changed while reading");value.resize(copied);
        require(value.rfind(L"Local\\",0)==0&&value.size()>6,"event must be in Local namespace");return value;
    }
    static std::string utf8(const std::wstring &value) {
        const int size=WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,value.data(),static_cast<int>(value.size()),nullptr,0,nullptr,nullptr);
        require(size>0,"event name UTF8 conversion failed");std::string result(size,'\0');
        require(WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,value.data(),static_cast<int>(value.size()),result.data(),size,nullptr,nullptr)==size,"event name UTF8 conversion changed");return result;
    }
public:
    EventParking() {
        try {
            go_name_=env(L"SIMULATOR_PHYSICAL_GO_EVENT");stop_name_=env(L"SIMULATOR_PHYSICAL_STOP_EVENT");
            require(go_name_!=stop_name_,"GO and STOP events must differ");
            go_=OpenEventW(SYNCHRONIZE,FALSE,go_name_.c_str());require(go_!=nullptr,"parent-owned GO event unavailable");
            stop_=OpenEventW(SYNCHRONIZE,FALSE,stop_name_.c_str());require(stop_!=nullptr,"parent-owned STOP event unavailable");
            timer_=CreateWaitableTimerExW(nullptr,nullptr,0x00000002,TIMER_MODIFY_STATE|SYNCHRONIZE);
            high_resolution_=timer_!=nullptr;
            if(!timer_){timer_fallback_error_=GetLastError();timer_=CreateWaitableTimerExW(nullptr,nullptr,0,TIMER_MODIFY_STATE|SYNCHRONIZE);}
            require(timer_!=nullptr,"private start timer unavailable");
        } catch (...) {close();throw;}
    }
    ~EventParking(){close();}
    EventParking(const EventParking &)=delete;EventParking &operator=(const EventParking &)=delete;
    void close(){if(timer_)CloseHandle(timer_);if(stop_)CloseHandle(stop_);if(go_)CloseHandle(go_);timer_=stop_=go_=nullptr;}
    bool stop_requested()const {
        const DWORD status=WaitForSingleObject(stop_,0);require(status==WAIT_OBJECT_0||status==WAIT_TIMEOUT,"STOP event query failed");return status==WAIT_OBJECT_0;
    }
    void wait_go()const {
        const HANDLE handles[]={stop_,go_};const DWORD status=WaitForMultipleObjects(2,handles,FALSE,30000);
        require(status!=WAIT_OBJECT_0,"STOP before GO");require(status==WAIT_OBJECT_0+1,"GO event timeout or wait failure");
    }
    void until(int64_t deadline)const {
        const HANDLE handles[]={stop_,timer_};
        while(single_layer::tick()<deadline) {
            const auto remaining=deadline-single_layer::tick();
            if(remaining<=0)break;
            LARGE_INTEGER due;due.QuadPart=-std::max<int64_t>(1,(state_response::ns(0,remaining)+99)/100);
            require(SetWaitableTimer(timer_,&due,0,nullptr,nullptr,FALSE)!=0,"common start timer failed");
            const DWORD status=WaitForMultipleObjects(2,handles,FALSE,120000);
            require(status!=WAIT_OBJECT_0,"STOP before common start");require(status==WAIT_OBJECT_0+1,"common start wait failed");
        }
    }
    void wait_stop()const {require(WaitForSingleObject(stop_,120000)==WAIT_OBJECT_0,"STOP release timeout or wait failure");}
    json describe()const {return json{{"schema","physical-event-parking-1"},{"mode","parent_owned_Windows_events"},
        {"go_event",utf8(go_name_)},{"stop_event",utf8(stop_name_)},{"parent_manual_reset_contract",true},
        {"native_event_set_or_reset_performed",false},{"post_READY_periodic_file_poll",false},
        {"post_sampling_periodic_wakeup",false},{"stop_check_during_samples","nonblocking event query per invocation"},
        {"common_start_timer",high_resolution_?"private_high_resolution_waitable_timer":"private_waitable_timer"},
        {"timer_fallback_win32_error",timer_fallback_error_},{"global_timer_settings_changed",false}};}
};

static bool cycles_enabled() {
    const char *value=std::getenv("SIMULATOR_PHYSICAL_CYCLES");
    require(!value||std::string(value)=="0"||std::string(value)=="1","SIMULATOR_PHYSICAL_CYCLES must be absent, 0 or 1");
    return value&&std::string(value)=="1";
}
static json cycle_row(const ActionInterval &interval) {
    const auto &a=interval.cycles_before,&b=interval.cycles_after;
    const bool known=a.known&&b.known&&b.value>=a.value;
    return json{{"enabled",true},{"before_query_begin_ns",a.begin},{"before_query_end_ns",a.end},
        {"after_query_begin_ns",b.begin},{"after_query_end_ns",b.end},
        {"before_cycles",a.known?json(a.value):json(nullptr)},{"after_cycles",b.known?json(b.value):json(nullptr)},
        {"before_win32_error",a.error},{"after_win32_error",b.error},{"monotonic",a.known&&b.known?json(b.value>=a.value):json(nullptr)},
        {"delta_cycles",known?json(b.value-a.value):json(nullptr)},{"unit","raw_QueryThreadCycleTime_cycles"},
        {"action_timer_excludes_counter_queries",true},{"invocation_timer_includes_counter_queries",true},
        {"cycles_converted_to_time",false},{"running_ready_blocked_separated",false},{"usable_as_heldout_resource_input",false}};
}

static json action_row(const json &action,const ActionInterval &interval,bool cycles=false) {
    json result{{"action_index",interval.index},{"kind",action.at("kind")},
        {"target_id",action.value("target_id",json(nullptr))},{"external_id",interval.external},
        {"begin_ns",interval.begin},{"end_ns",interval.end},
        {"device",action.value("device",json(nullptr))},
        {"timing_scope","inclusive unchanged Executor.action host completion"}};
    if(cycles)result["thread_cycles"]=cycle_row(interval);
    return result;
}

static json serialization_self_test() {
    const ActionInterval interval{10,20,1,0};
    const auto advance=action_row(json{{"kind","ADVANCE"},{"target_id",nullptr},{"device",nullptr}},interval);
    const auto copy=action_row(json{{"kind","COPY"},{"target_id","tensor"},{"device",nullptr}},interval);
    const auto missing=action_row(json{{"kind","ADVANCE"}},interval);
    require(advance.at("target_id").is_null()&&advance.at("device").is_null(),"null control serialization failed");
    require(copy.at("target_id")=="tensor"&&copy.at("device").is_null(),"copy serialization failed");
    require(missing.at("target_id").is_null()&&missing.at("device").is_null(),"absent optional serialization failed");
    auto measured=interval;measured.cycles_before.known=true;measured.cycles_before.value=100;
    measured.cycles_after.known=true;measured.cycles_after.value=137;
    const auto cycles=action_row(json{{"kind","ADVANCE"}},measured,true);
    require(cycles.at("thread_cycles").at("delta_cycles")==37,"raw cycle delta serialization failed");
    measured.cycles_after.value=99;
    require(action_row(json{{"kind","ADVANCE"}},measured,true).at("thread_cycles").at("delta_cycles").is_null(),"nonmonotonic cycles silently repaired");
    return json{{"schema","physical-stage-serialization-self-test-1"},{"passed",true},
        {"hardware_executed",false},{"raw_cycle_cases_passed",true},{"cases",json::array({advance,copy,missing,cycles})}};
}

static json clock_json() {
    return json{{"clock_id","windows_qpc_floor_ns"},{"clock_origin","absolute_qpc_zero"},
        {"frequency_hz",single_layer::frequency()},
        {"conversion","floor(QPC ticks / frequency * 1e9), quotient/remainder without overflow"},
        {"cross_process_monotonic_clock",true}};
}

static json cpu_delta(const single_layer::CpuSnapshot &first,const single_layer::CpuSnapshot &last,
                      uint64_t begin,uint64_t end) {
    const bool known=first.available&&last.available;
    require(!known||(last.process>=first.process&&last.main>=first.main),"CPU counters moved backwards");
    return json{{"begin_ns",begin},{"end_ns",end},
        {"process_active_cpu_ns",known?json(last.process-first.process):json(nullptr)},
        {"main_thread_active_cpu_ns",known?json(last.main-first.main):json(nullptr)},
        {"scope","batch process/main cumulative kernel+user counters; includes query/capture/runtime/reset work"},
        {"per_action_active_cpu_time_measured",false},{"running_ready_blocked_separated",false},
        {"counter_quantization_retained",true},{"available_cpu_service_inferred",false}};
}

static void wait_for_go(const std::string &go,json &source,const EventParking &events) {
    events.wait_go();std::ifstream input(go);require(bool(input),"GO event signaled without complete receipt");input>>source;
    require(source.at("schema")=="physical-stages-go-1","GO schema differs");
    require(source.at("clock_id")=="windows_qpc"&&
            state_response::integer(source.at("frequency_hz"),"GO frequency",1)==single_layer::frequency(),
            "GO QPC clock differs");
    const auto begin=state_response::integer(source.at("start_tick"),"GO start",1);
    const auto end=state_response::integer(source.at("end_tick"),"GO end",begin+1);
    require(end-begin<=120*single_layer::frequency(),"GO window exceeds 120 seconds");
    require(end>single_layer::tick(),"GO window already ended");
    events.until(begin);
}

static json run(Executor &executor,const std::string &mode,const std::string &instrument,
                int target,int warmup,int samples,const std::string &poll,
                const std::string &ready,const std::string &go,const std::string &stop,
                const state_response::CudaScheduleRequest &cuda_request,const EventParking &events,bool cycles) {
    const auto &actions=executor.program.at("actions");
    require(!actions.empty(),"empty action program");
    require(mode=="full"||mode=="target","unknown execution mode");
    require(instrument=="cupti"||instrument=="bare","unknown instrumentation mode");
    require((mode=="full"&&target<0)||(mode=="target"&&target>=0&&static_cast<size_t>(target)<actions.size()),
            "full requires target none; target requires a valid original action index");
    if (mode=="target") require(actions.at(target).at("kind")=="COMPUTE","isolated target must be COMPUTE");
    const size_t actions_per_iteration=mode=="full"?actions.size():1;
    require(static_cast<uint64_t>(samples)*actions_per_iteration<=2000000,"declared action output envelope exceeds 2M");
    const auto pool=single_layer::configure_pool(executor,poll);
    executor.validate();
    const auto initial=single_layer::capture_verification(executor);
    const auto inputs=mode=="target"?state_response::inputs(executor,actions.at(target)):std::vector<state_response::InputSnapshot>{};
    for (int i=0;i<warmup;++i) {
        require(!exists(stop),"stopped during warmup");
        if (mode=="full") {executor.reset();for(const auto &action:actions)executor.action(action);}
        else executor.action(actions.at(target));
    }
    const auto warmed=single_layer::capture_verification(executor);
    const auto warm_consistency=state_response::compare_ports(initial.at("ports"),warmed.at("ports"));
    std::vector<Invocation> invocations(samples);
    std::vector<ActionInterval> intervals(static_cast<size_t>(samples)*actions_per_iteration);
    std::unique_ptr<physical_capture::Capture> capture;
    if (instrument=="cupti") capture=std::make_unique<physical_capture::Capture>();
    const auto before_schedule=state_response::observe_cuda_schedule(cuda_request,"after_warmup_before_READY");
    const auto ready_tick=single_layer::tick();
    write_new(ready,json{{"schema","physical-stages-ready-1"},{"process_id",single_layer::pid()},
        {"main_thread_id",state_response::main_thread_id()},{"qpc_clock_id","windows_qpc"},
        {"frequency_hz",single_layer::frequency()},{"ready_tick",ready_tick},
        {"clock",state_response::clock_json()},{"physical_clock",clock_json()},
        {"mode",mode},{"instrumentation",instrument},
        {"samples_requested",samples},{"warmup_completed",warmup},
        {"program_sha256",executor.program.at("program_sha256")},
        {"target_action_index",target<0?json(nullptr):json(target)},
        {"warmup_repeat_consistency",warm_consistency},{"cuda_schedule",before_schedule},
        {"event_parking",events.describe()},{"thread_cycles_enabled",cycles}});
    json go_source;
    wait_for_go(go,go_source,events);
    const int64_t end_tick=go_source.at("end_tick");
    const auto cpu_first=single_layer::cpu_snapshot();
    const uint64_t batch_begin=qpc_now_ns();
    size_t completed=0;
    for (int iteration=0;iteration<samples;++iteration) {
        require(!events.stop_requested(),"STOP arrived before requested samples completed");
        require(single_layer::tick()<end_tick,"GO window ended before requested samples completed");
        auto &item=invocations[iteration];
        if (mode=="full") {item.reset_begin=qpc_now_ns();executor.reset();item.reset_end=qpc_now_ns();}
        item.begin=qpc_now_ns();
        for(size_t j=0;j<actions_per_iteration;++j) {
            const size_t index=mode=="full"?j:static_cast<size_t>(target);
            auto &interval=intervals[static_cast<size_t>(iteration)*actions_per_iteration+j];
            interval.index=index;interval.external=1+static_cast<uint64_t>(iteration)*actions_per_iteration+j;
            if(capture)capture->push(interval.external);
            if(cycles)interval.cycles_before=cycle_point();
            interval.begin=qpc_now_ns();
            executor.action(actions.at(index));
            interval.end=qpc_now_ns();
            if(cycles)interval.cycles_after=cycle_point();
            if(capture)capture->pop(interval.external);
        }
        item.end=qpc_now_ns();
        ++completed;
    }
    const uint64_t batch_end=qpc_now_ns();
    const auto cpu_last=single_layer::cpu_snapshot();
    const auto complete=std::filesystem::u8path(ready).parent_path()/"sampling-complete.json";
    write_new(complete.u8string(),json{{"schema","physical-stages-sampling-complete-1"},
        {"process_id",single_layer::pid()},{"sampling_begin_ns",batch_begin},{"sampling_end_ns",batch_end},
        {"completed_samples",completed},{"clock",clock_json()}});
    events.wait_stop();require(exists(stop),"STOP event signaled without receipt");
    json activities=capture?capture->finish():json{{"schema","physical-stage-cupti-1"},
        {"enabled",false},{"records",json::array()},{"available_resource_rate_inferred",false},
        {"hardware_accuracy_qualified",false}};
    capture.reset();
    const auto final=single_layer::capture_verification(executor);
    const auto repeat_consistency=state_response::compare_ports(initial.at("ports"),final.at("ports"));
    const auto input_checks=state_response::verify_inputs(executor,inputs);
    executor.validate();
    const auto after=single_layer::capture_verification(executor);
    const auto post_consistency=state_response::compare_ports(initial.at("ports"),after.at("ports"));
    const auto after_schedule=state_response::observe_cuda_schedule(cuda_request,"after_STOP_before_destructor");
    json rows=json::array();
    for(size_t i=0;i<completed;++i) {
        const auto &item=invocations[i];json stages=json::array();
        for(size_t j=0;j<actions_per_iteration;++j) {
            const auto &interval=intervals[i*actions_per_iteration+j];
            const auto &action=actions.at(interval.index);
            stages.push_back(action_row(action,interval,cycles));
        }
        rows.push_back(json{{"iteration",i},{"begin_ns",item.begin},{"end_ns",item.end},
            {"reset_begin_ns",mode=="full"?json(item.reset_begin):json(nullptr)},
            {"reset_end_ns",mode=="full"?json(item.reset_end):json(nullptr)},
            {"actions",stages}});
    }
    return json{{"schema","physical-stage-native-1"},{"process_id",single_layer::pid()},
        {"main_thread_id",state_response::main_thread_id()},{"clock",clock_json()},
        {"program_sha256",executor.program.at("program_sha256")},{"boundary_contract",BOUNDARY},
        {"mode",mode},{"instrumentation",instrument},{"target_action_index",target<0?json(nullptr):json(target)},
        {"samples_requested",samples},{"samples_completed",completed},{"warmup",warmup},
        {"pool",pool},{"actions",actions},{"go",go_source},
        {"sampling_begin_ns",batch_begin},{"sampling_end_ns",batch_end},{"iterations",rows},
        {"batch_cpu",cpu_delta(cpu_first,cpu_last,batch_begin,batch_end)},
        {"event_parking",events.describe()},{"thread_cycles_enabled",cycles},
        {"thread_cycle_scope","optional diagnostic counters; never elapsed time or a heldout resource input"},
        {"cupti",activities},{"initial_verification_native",initial},{"verification_native",final},
        {"warmup_repeat_consistency",warm_consistency},{"native_repeat_consistency",repeat_consistency},
        {"post_replay_consistency",post_consistency},{"target_input_checks",input_checks},
        {"cuda_schedule",json{{"before_ready",before_schedule},{"after_STOP",after_schedule}}},
        {"historical_Executor_action_unchanged",true},
        {"enqueue_sync_distinction","correlated CUPTI runtime APIs; not uncorrelated additive host durations"},
        {"bare_control_scope","same native QPC/counter instrumentation with CUPTI disabled; not clock-free execution"},
        {"post_sampling_work_deferred_until_STOP",true},{"independent_numeric_oracle_applied",false},
        {"available_resource_trajectory_measured",false},{"physical_capacity_claimed",false},
        {"hardware_accuracy_qualified",false}};
}
} // namespace physical_stages

int main(int argc,char **argv) {try {
    if(argc==2&&std::string(argv[1])=="--serialization-self-test") {
        std::cout<<physical_stages::serialization_self_test().dump()<<'\n';return 0;
    }
    require(argc==12,"usage: event-stages-probe PROGRAM.json full|target cupti|bare TARGET_INDEX|none WARMUP SAMPLES default|0 READY.json GO.json STOP.json OUTPUT.json");
    const std::string mode=argv[2],instrument=argv[3];
    const int target=std::string(argv[4])=="none"?-1:single_layer::integer(argv[4],0,10000,"target");
    const int warmup=single_layer::integer(argv[5],0,10000,"warmup");
    const int samples=single_layer::integer(argv[6],1,100000,"samples");
    const std::string poll=argv[7];
    std::set<std::filesystem::path> paths;
    for(int i=8;i<=11;++i) {
        const auto path=std::filesystem::absolute(std::filesystem::u8path(argv[i])).lexically_normal();
        require(paths.insert(path).second,"READY/GO/STOP/output paths must differ");
        require(!state_response::exists(argv[i]),"READY/GO/STOP/output path already exists");
    }
    _putenv_s("NVIDIA_TF32_OVERRIDE","0");_putenv_s("GGML_CUDA_DISABLE_GRAPHS","1");
    std::ifstream input(argv[1]);require(bool(input),"cannot read program");
    json source;input>>source;
    physical_stages::EventParking events;const bool cycles=physical_stages::cycles_enabled();
    const auto cuda_request=state_response::configure_cuda_schedule_from_environment();
    json result;
    {Executor executor(std::move(source));result=physical_stages::run(executor,mode,instrument,target,warmup,samples,poll,
        argv[8],argv[9],argv[10],cuda_request,events,cycles);}
    result["owned_Executor_destructor_returned"]=true;
    state_response::write_new(argv[11],result);
    std::cout<<json{{"status","ok"},{"process_id",result.at("process_id")},
        {"samples_completed",result.at("samples_completed")},{"instrumentation",instrument}}.dump()<<'\n';
    return 0;
}catch(const std::exception &error) {
    std::cerr<<"event-stages-probe: "<<error.what()<<'\n';return 2;
}}
