// Persistent, scheduled resource-response execution. No fitting or profiler.
#include "single_layer_generated.inc"
#include "ggml-backend-impl.h"
#include <cuda_runtime_api.h>
#include <filesystem>
#include <thread>
#include <limits>
#include <cstdlib>
#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif
namespace state_response {
using single_layer::tick;using single_layer::frequency;using single_layer::pid;
static int64_t ns(int64_t first,int64_t last){return single_layer::ns(first,last,frequency());}
#include "timing_events.inc"
#include "numeric_helpers.inc"
static json main_thread_id(){
#if defined(_WIN32)
    return static_cast<uint64_t>(GetCurrentThreadId());
#else
    return nullptr;
#endif
}

static int64_t integer(const json &value,const char *name,int64_t minimum=0) {
    require(value.is_number_integer()&&!value.is_boolean(),std::string(name)+" must be an integer");
    const auto number=value.get<int64_t>();require(number>=minimum,std::string(name)+" below minimum");return number;
}
static json clock_json(){return json{{"clock_id",
#if defined(_WIN32)
    "windows_qpc"
#else
    "posix_clock_monotonic"
#endif
    },{"frequency_hz",frequency()},{"cross_process_monotonic_clock",true}};}
struct CudaScheduleRequest {json requested=nullptr;unsigned flags=0;bool overridden=false;json setter=nullptr;};
static unsigned requested_cuda_schedule_flags(const std::string &value) {
    if(value=="auto")return cudaDeviceScheduleAuto;
    if(value=="spin")return cudaDeviceScheduleSpin;
    if(value=="yield")return cudaDeviceScheduleYield;
    if(value=="blocking")return cudaDeviceScheduleBlockingSync;
    throw std::runtime_error("SIMULATOR_CUDA_SCHEDULE must be exactly auto, spin, yield or blocking; unset means no override");
}
static const char *observed_cuda_schedule_name(unsigned flags) {
    switch(flags&cudaDeviceScheduleMask) {
        case cudaDeviceScheduleAuto:return "auto";
        case cudaDeviceScheduleSpin:return "spin";
        case cudaDeviceScheduleYield:return "yield";
        case cudaDeviceScheduleBlockingSync:return "blocking";
        default:return "unrecognized_schedule_mask";
    }
}
static json cuda_query_status(cudaError_t status) {
    return json{{"code",static_cast<int>(status)},{"name",cudaGetErrorName(status)},
                {"description",cudaGetErrorString(status)}};
}
static CudaScheduleRequest configure_cuda_schedule_from_environment() {
    CudaScheduleRequest result;const char *value=std::getenv("SIMULATOR_CUDA_SCHEDULE");
    if(value==nullptr)return result; // Unset does not invoke a setter or assume a default.
    result.requested=std::string(value);result.flags=requested_cuda_schedule_flags(value);
    const auto status=cudaSetDeviceFlags(result.flags);result.setter=cuda_query_status(status);
    require(status==cudaSuccess,"explicit CUDA schedule override failed: "+result.setter.dump());
    result.overridden=true;return result;
}
static json observe_cuda_schedule(const CudaScheduleRequest &request,const char *window) {
    const auto begin=tick();int device=-1;unsigned flags=0;
    const auto device_status=cudaGetDevice(&device);const auto flags_status=cudaGetDeviceFlags(&flags);
    const auto end=tick();const bool known=device_status==cudaSuccess&&flags_status==cudaSuccess;
    const bool preserved=known&&((flags&cudaDeviceScheduleMask)==request.flags);
    json result={{"schema","cuda-device-schedule-observation-1"},{"requested",request.requested},
        {"override_applied",request.overridden},{"requested_raw_schedule_flags",request.overridden?json(request.flags):json(nullptr)},
        {"setter_status",request.setter},{"observation_window",window},{"query_begin_tick",begin},{"query_end_tick",end},
        {"status",known?"observed":"unknown"},{"cuda_logical_device",device_status==cudaSuccess?json(device):json(nullptr)},
        {"raw_device_flags",flags_status==cudaSuccess?json(flags):json(nullptr)},
        {"schedule_mask",flags_status==cudaSuccess?json(flags&cudaDeviceScheduleMask):json(nullptr)},
        {"schedule_name",flags_status==cudaSuccess?json(observed_cuda_schedule_name(flags)):json(nullptr)},
        {"device_query_status",cuda_query_status(device_status)},{"flags_query_status",cuda_query_status(flags_status)},
        {"requested_schedule_preserved",request.overridden?json(preserved):json(nullptr)},
        {"auto_wait_behavior_resolved",false},{"observed_schedule_proves_thread_activity",false},
        {"observation_scope","current logical device flags at this endpoint; no continuous mode/activity observation"}};
    require(!request.overridden||preserved,"requested CUDA schedule was not preserved or cannot be verified: "+result.dump());
    return result;
}
static bool exists(const std::string &path){std::error_code error;const bool result=std::filesystem::exists(std::filesystem::u8path(path),error);require(!error,"handshake path query failed");return result;}
static void write_new(const std::string &path,const json &value) {
    const std::string bytes=value.dump(2)+"\n";
#if defined(_WIN32)
    HANDLE file=CreateFileW(std::filesystem::u8path(path).c_str(),GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
    require(file!=INVALID_HANDLE_VALUE,"cannot exclusively create output");size_t offset=0;bool okay=true;
    while(offset<bytes.size()){DWORD written=0;const auto count=static_cast<DWORD>(std::min<size_t>(bytes.size()-offset,1<<20));if(!WriteFile(file,bytes.data()+offset,count,&written,nullptr)||written==0){okay=false;break;}offset+=written;}
    const bool closed=CloseHandle(file)!=0;require(okay&&closed&&offset==bytes.size(),"output write failed");
#else
    const int file=open(path.c_str(),O_CREAT|O_EXCL|O_WRONLY,0600);require(file>=0,"cannot exclusively create output");size_t offset=0;bool okay=true;
    while(offset<bytes.size()){const auto written=write(file,bytes.data()+offset,bytes.size()-offset);if(written<=0){okay=false;break;}offset+=static_cast<size_t>(written);}const int closed=close(file);require(okay&&closed==0,"output write failed");
#endif
}
class PrivateWait {
#if defined(_WIN32)
    HANDLE timer=nullptr;
#endif
    bool high_resolution=false;
public:
    PrivateWait(){
#if defined(_WIN32)
        timer=CreateWaitableTimerExW(nullptr,nullptr,0x00000002,TIMER_MODIFY_STATE|SYNCHRONIZE);high_resolution=timer!=nullptr;
#endif
    }
    ~PrivateWait(){
#if defined(_WIN32)
        if(timer)CloseHandle(timer);
#endif
    }
    void until(int64_t deadline) {
        const auto now=tick();if(now>=deadline)return;const auto delay=std::min<int64_t>(ns(now,deadline),2000000);
#if defined(_WIN32)
        if(timer){LARGE_INTEGER due;due.QuadPart=-std::max<int64_t>(1,(delay+99)/100);require(SetWaitableTimer(timer,&due,0,nullptr,nullptr,FALSE)!=0,"private timer set failed");const auto status=WaitForSingleObject(timer,20);require(status==WAIT_OBJECT_0||status==WAIT_TIMEOUT,"private timer wait failed");return;}
#endif
        std::this_thread::sleep_for(std::chrono::nanoseconds(delay));
    }
    json describe()const{return json{{"provider",high_resolution?"Windows_private_high_resolution_waitable_timer":"std::this_thread::sleep_for"},{"maximum_requested_sleep_ns",2000000},{"cadence_guaranteed",false},{"global_timer_settings_changed",false}};}
};
struct Phase {int64_t begin=0,end=0;int cpu=0,gpu=0;};
struct Schedule {json source;int64_t start=0,end=0;std::vector<Phase> phases;};
static Schedule schedule_from(const json &source) {
    require(source.at("schema")=="state-response-go-v1","GO schema differs");require(source.at("clock").at("clock_id")==clock_json().at("clock_id")&&integer(source.at("clock").at("frequency_hz"),"GO frequency",1)==frequency(),"GO clock differs");
    Schedule result;result.source=source;result.start=integer(source.at("start_tick"),"start tick",1);result.end=integer(source.at("end_tick"),"end tick",result.start+1);
    require(ns(result.start,result.end)<=120000000000LL,"GO trajectory exceeds 120 seconds");
    const auto &phases=source.at("phases");require(phases.is_array()&&!phases.empty()&&phases.size()<=256,"GO phases envelope invalid");int64_t cursor=result.start;size_t index=0;
    for(const auto &p:phases){require(integer(p.at("phase_index"),"phase index")==static_cast<int64_t>(index++),"phase indices differ");Phase phase;phase.begin=integer(p.at("begin_tick"),"phase begin",1);phase.end=integer(p.at("end_tick"),"phase end",phase.begin+1);phase.cpu=static_cast<int>(integer(p.at("cpu_replay"),"CPU request"));phase.gpu=static_cast<int>(integer(p.at("gpu_replay"),"GPU request"));require(phase.begin==cursor&&phase.end<=result.end&&phase.cpu<=1&&phase.gpu<=1,"GO state/coverage differs");cursor=phase.end;result.phases.push_back(phase);}
    require(cursor==result.end,"GO phases do not cover end");return result;
}
static size_t phase_at(const Schedule &schedule,int64_t now){size_t index=0;while(index+1<schedule.phases.size()&&now>=schedule.phases[index].end)++index;return index;}
struct CpuPoint {single_layer::CpuSnapshot times;uint64_t cycles=0;bool cycles_known=false;int64_t begin=0,end=0;};
static CpuPoint cpu_point(){CpuPoint result;result.begin=tick();result.times=single_layer::cpu_snapshot();
#if defined(_WIN32)
    ULONG64 cycles=0;result.cycles_known=QueryThreadCycleTime(GetCurrentThread(),&cycles)!=0;result.cycles=cycles;
#endif
    result.end=tick();return result;}
static json cpu_delta(const CpuPoint &a,const CpuPoint &b){const bool known=a.times.available&&b.times.available;require(!known||(b.times.process>=a.times.process&&b.times.main>=a.times.main),"CPU counter moved backwards");const bool cycles=a.cycles_known&&b.cycles_known&&b.cycles>=a.cycles;return json{{"cpu_before_query_begin_tick",a.begin},{"cpu_before_query_end_tick",a.end},{"cpu_after_query_begin_tick",b.begin},{"cpu_after_query_end_tick",b.end},{"process_cpu_ns",known?json(b.times.process-a.times.process):json(nullptr)},{"main_thread_cpu_ns",known?json(b.times.main-a.times.main):json(nullptr)},{"main_thread_cycles",cycles?json(b.cycles-a.cycles):json(nullptr)},{"cpu_time_scope","counter query brackets encompass work and counter-read overhead; CPU accounting granularity can produce zero deltas"},{"cycles_scope","raw QueryThreadCycleTime; no conversion to elapsed nanoseconds"}};}
struct ActionStamp {int64_t begin=0,end=0;};
struct Iteration {int64_t begin=0,end=0,decision=0;};
struct ThreadDiagnostic {CpuPoint before,after;};
static void perform_thread_diagnostic(Executor &executor,const json &action,ActionStamp &stamp,ThreadDiagnostic &record) {
    record.before=cpu_point();stamp.begin=tick();
    executor.action(action);
    stamp.end=tick();record.after=cpu_point();
}
static json thread_diagnostic_json(const ThreadDiagnostic &d,const ActionStamp &stamp,size_t iteration,size_t action_index) {
    return json{{"iteration",iteration},{"action_index",action_index},{"main_thread_id",main_thread_id()},
        {"action_begin_tick",stamp.begin},{"action_end_tick",stamp.end},{"action_elapsed_ns",ns(stamp.begin,stamp.end)},
        {"cpu_counter_bracket",cpu_delta(d.before,d.after)},
        {"before_query_ns",ns(d.before.begin,d.before.end)},{"after_query_ns",ns(d.after.begin,d.after.end)},
        {"outer_begin_tick",d.before.begin},{"outer_end_tick",d.after.end},{"outer_elapsed_ns",ns(d.before.begin,d.after.end)},
        {"diagnostic_only",true},{"usable_for_parameter_fit",false},{"normal_Executor_action_unchanged",true},
        {"action_timer_excludes_counter_queries",true},{"invocation_timer_includes_counter_queries",true},
        {"cpu_counter_bracket_is_exact_action_cpu_time",false},{"cycles_converted_to_nanoseconds",false},
        {"running_ready_blocked_separated_by_these_counters",false},{"runtime_wait_or_spin_identified",false}};
}
struct Diagnostic {int64_t begin=0,end=0,enq_begin=0,enq_end=0,sync1_begin=0,sync1_end=0,sync2_begin=0,sync2_end=0;CpuPoint points[4];float stream_ms=0;bool events=false;};
static void perform(Executor &executor,const json &action,Diagnostic *record,TimingEvents *events) {
    if(!record){executor.action(action);return;}
    auto &g=*executor.graphs.at(str(action,"target_id"));require(g.backend==executor.gpu,"diagnostic target is not GPU");record->events=events!=nullptr;record->begin=tick();record->points[0]=cpu_point();
    if(events)events->begin(g.backend);record->enq_begin=tick();const auto status=ggml_backend_graph_compute_async(g.backend,g.graph);record->enq_end=tick();require(status==GGML_STATUS_SUCCESS,"GGML async compute failed");if(events)events->end(g.backend);record->points[1]=cpu_point();
    record->sync1_begin=tick();ggml_backend_synchronize(g.backend);record->sync1_end=tick();record->points[2]=cpu_point();
    record->sync2_begin=tick();ggml_backend_synchronize(g.backend);record->sync2_end=tick();record->points[3]=cpu_point();
    if(events)record->stream_ms=events->elapsed_ms();record->end=tick();
}
static json diagnostic_json(const Diagnostic &d,size_t iteration,int target){return json{{"iteration",iteration},{"action_index",target},{"begin_tick",d.begin},{"end_tick",d.end},{"host_elapsed_ns",ns(d.begin,d.end)},{"host_enqueue_begin_tick",d.enq_begin},{"host_enqueue_end_tick",d.enq_end},{"host_enqueue_ns",ns(d.enq_begin,d.enq_end)},{"host_sync1_begin_tick",d.sync1_begin},{"host_sync1_end_tick",d.sync1_end},{"host_sync1_ns",ns(d.sync1_begin,d.sync1_end)},{"host_sync2_begin_tick",d.sync2_begin},{"host_sync2_end_tick",d.sync2_end},{"host_sync2_ns",ns(d.sync2_begin,d.sync2_end)},{"stream_span_ms",d.events?json(d.stream_ms):json(nullptr)},{"stream_span_ns",d.events?json(static_cast<int64_t>(std::llround(double(d.stream_ms)*1000000.0))):json(nullptr)},{"cpu_total",cpu_delta(d.points[0],d.points[3])},{"cpu_enqueue_bracket",cpu_delta(d.points[0],d.points[1])},{"cpu_sync1_bracket",cpu_delta(d.points[1],d.points[2])},{"cpu_sync2_bracket",cpu_delta(d.points[2],d.points[3])},{"diagnostic_only",true},{"stream_span_is_active_kernel_time",false},{"host_CPU_and_GPU_intervals_are_additive",false}};}

static json run(Executor &executor,const std::string &role,const std::string &instrumentation,int target,int warmup,const std::string &ready,const std::string &go,const std::string &stop,const CudaScheduleRequest &cuda_request) {
    const bool main=role=="main",diagnostic=instrumentation=="diagnostic"||instrumentation=="diagnostic-control",stamped=instrumentation!="bare";
    const bool thread_diagnostic=instrumentation=="thread-diagnostic";
    require(!thread_diagnostic||(main&&target<0),"thread-diagnostic requires whole main replay and target none");
    const auto &actions=executor.program.at("actions");require(!actions.empty(),"empty action trace");
    if(!main||diagnostic){require(target>=0&&static_cast<size_t>(target)<actions.size(),"explicit target index required");const auto &a=actions.at(target);require(a.at("kind")=="COMPUTE","target must be COMPUTE");require(a.at("device")==(!main&&role=="cpu-worker"?"cpu":"gpu"),"target device differs from role");}
    if(diagnostic)require(main||role=="gpu-worker","CUDA diagnostics require GPU target");
    executor.validate();const auto initial=single_layer::capture_verification(executor);
    const auto snapshots=!main?inputs(executor,actions.at(target)):std::vector<InputSnapshot>{};
    std::unique_ptr<TimingEvents> events;if(instrumentation=="diagnostic")events=std::make_unique<TimingEvents>(executor.gpu,executor.program.at("physical").value("cuda_device",0));
    for(int i=0;i<warmup;++i){require(!exists(stop),"stopped during warmup");if(main){executor.reset();for(const auto &a:actions)executor.action(a);}else executor.action(actions.at(target));}
    auto warmed=single_layer::capture_verification(executor);auto warm_consistency=compare_ports(initial.at("ports"),warmed.at("ports"));
    const size_t actions_per_iteration=main?actions.size():1;
    const size_t maximum_iterations=std::min<size_t>(main?20000:200000,2000000/actions_per_iteration);
    require(maximum_iterations*actions_per_iteration<=2000000,"POD action envelope exceeded");
    std::vector<Iteration> iterations(maximum_iterations);std::vector<ActionStamp> stamps(maximum_iterations*actions_per_iteration);
    std::vector<ThreadDiagnostic> thread_diagnostics(thread_diagnostic?maximum_iterations*actions_per_iteration:0);
    std::vector<Diagnostic> diagnostics(diagnostic?maximum_iterations:0);size_t count=0;
    const auto schedule_before=observe_cuda_schedule(cuda_request,"after_warmup_before_READY");
    PrivateWait waiter;const auto ready_tick=tick();
    write_new(ready,json{{"schema","state-response-ready-v1"},{"format_version","state-response-ready-v1"},{"process_id",pid()},{"main_thread_id",main_thread_id()},{"role",role},{"clock",clock_json()},{"ready_tick",ready_tick},{"program_sha256",executor.program.at("program_sha256")},{"instrumentation",instrumentation},{"target_action_index",target<0?json(nullptr):json(target)},{"warmup_completed",warmup},{"cuda_schedule",schedule_before},{"independent_numeric_oracle_applied",false},{"warmup_repeat_consistency",warm_consistency},{"pod_iteration_capacity",maximum_iterations},{"pod_action_capacity",stamps.size()}});
    json go_json;const auto go_deadline=ready_tick+30*frequency();
    while(true){require(tick()<go_deadline,"GO wait deadline exceeded");require(!exists(stop),"stopped before GO");if(exists(go)){try{std::ifstream input(go);input>>go_json;break;}catch(const json::parse_error &){}}waiter.until(std::min(go_deadline,tick()+frequency()/500));}
    const auto schedule=schedule_from(go_json);require(schedule.end>tick(),"GO schedule already ended");
    while(tick()<schedule.start){require(!exists(stop),"stopped before common start");waiter.until(schedule.start);}
    const auto cpu_first=cpu_point();const auto begin=tick();bool stopped=false;uint64_t reset_count=0,disabled_waits=0;
    while(true){const auto now=tick();if(now>=schedule.end)break;if(exists(stop)){stopped=true;break;}const auto phase=phase_at(schedule,now);const auto &request=schedule.phases[phase];const bool enabled=main||(role=="cpu-worker"?request.cpu:request.gpu)!=0;
        if(!enabled){++disabled_waits;waiter.until(std::min(schedule.end,request.end));continue;}
        require(count<maximum_iterations,"POD iteration capacity reached; no observations dropped");auto &iteration=iterations[count];iteration.decision=now;
        if(main){executor.reset();++reset_count;}
        iteration.begin=tick();
        if(main){size_t index=0;for(const auto &a:actions){auto &stamp=stamps[count*actions_per_iteration+index];if(thread_diagnostic)perform_thread_diagnostic(executor,a,stamp,thread_diagnostics[count*actions_per_iteration+index]);else{if(stamped)stamp.begin=tick();perform(executor,a,diagnostic&&static_cast<int>(index)==target?&diagnostics[count]:nullptr,events.get());if(stamped)stamp.end=tick();}++index;}}
        else{auto &stamp=stamps[count];if(stamped)stamp.begin=tick();perform(executor,actions.at(target),diagnostic?&diagnostics[count]:nullptr,events.get());if(stamped)stamp.end=tick();}
        iteration.end=tick();++count;
    }
    const auto end=tick();const auto cpu_last=cpu_point();require(!main||count>0,"foreground completed no iterations");
    const auto complete=(std::filesystem::u8path(ready).parent_path()/"sampling-complete.json").u8string();
    write_new(complete,json{{"schema","state-response-sampling-complete-v1"},{"role",role},{"process_id",pid()},{"main_thread_id",main_thread_id()},{"clock",clock_json()},{"program_sha256",executor.program.at("program_sha256")},{"sampling_end_tick",end},{"iteration_count",count}});
    const auto release_deadline=tick()+120*frequency();
    while(!exists(stop)){require(tick()<release_deadline,"sampling release deadline exceeded");waiter.until(release_deadline);}
    const auto release_tick=tick();
    const auto schedule_after=observe_cuda_schedule(cuda_request,"after_STOP_before_validation");
    auto final=single_layer::capture_verification(executor);auto consistency=compare_ports(initial.at("ports"),final.at("ports"));auto input_checks=verify_inputs(executor,snapshots);
    executor.validate();const auto replay=single_layer::capture_verification(executor);const auto post_consistency=compare_ports(initial.at("ports"),replay.at("ports"));
    json rows=json::array(),action_rows=json::array(),target_rows=json::array(),thread_rows=json::array();int64_t completed=0;
    for(size_t i=0;i<count;++i){const auto &iteration=iterations[i];const auto elapsed=ns(iteration.begin,iteration.end);require(elapsed>0,"nonpositive completed work");completed+=elapsed;rows.push_back(json{{"iteration",i},{"begin_tick",iteration.begin},{"end_tick",iteration.end},{"elapsed_ns",elapsed},{"enabled_check_tick",iteration.decision}});
        if(stamped)for(size_t j=0;j<actions_per_iteration;++j){const auto &stamp=stamps[i*actions_per_iteration+j];const auto first=ns(iteration.begin,stamp.begin),last=ns(iteration.begin,stamp.end);const size_t index=main?j:static_cast<size_t>(target);action_rows.push_back(json{{"iteration",i},{"action_index",index},{"kind",actions.at(index).at("kind")},{"begin_tick",stamp.begin},{"end_tick",stamp.end},{"begin_offset_ns",first},{"end_offset_ns",last},{"elapsed_ns",last-first}});}
        if(thread_diagnostic)for(size_t j=0;j<actions_per_iteration;++j)thread_rows.push_back(thread_diagnostic_json(thread_diagnostics[i*actions_per_iteration+j],stamps[i*actions_per_iteration+j],i,j));
        if(diagnostic)target_rows.push_back(diagnostic_json(diagnostics[i],i,target));
    }
    json block={{"block_index",0},{"begin_tick",begin},{"end_tick",end},{"wall_elapsed_ns",ns(begin,end)},{"completed_work_ns",completed},{"iteration_count",count},{"iterations",rows},{"action_intervals",action_rows},{"target_diagnostics",target_rows},{"thread_action_diagnostics",thread_rows}};
    auto cpu=cpu_delta(cpu_first,cpu_last);cpu["sampling_wall_ns"]=ns(begin,end);
    return json{{"format_version","state-response-native-v1"},{"process_id",pid()},{"main_thread_id",main_thread_id()},{"role",role},{"program_sha256",executor.program.at("program_sha256")},{"boundary_contract",BOUNDARY},{"clock",clock_json()},{"native_clock",clock_json()},{"instrumentation",instrumentation},{"execution_boundary",diagnostic?"explicit_async_double_sync_diagnostic":"legacy_double_sync"},{"diagnostic_only",diagnostic||thread_diagnostic},{"usable_for_parameter_fit",main&&!diagnostic&&!thread_diagnostic},{"thread_diagnostic_scope",thread_diagnostic?"per-action CPU counter query brackets around unchanged Executor.action; overhead included in whole invocation; diagnostic only":"not_requested"},{"go_schedule",go_json},{"ready_tick",ready_tick},{"sampling_begin_tick",begin},{"sampling_end_tick",end},{"sampling_release_tick",release_tick},{"sampling_release_scope","all verification, port reads, destructor and result serialization wait for owned STOP release; no graph or tensor work after sampling-complete and before STOP"},{"scheduled_start_tick",schedule.start},{"scheduled_end_tick",schedule.end},{"start_lateness_ns",begin>schedule.start?ns(schedule.start,begin):0},{"end_reason",stopped?"cooperative_stop":"scheduled_end_after_complete_invocation"},{"graceful_stop_requested",stopped},{"blocks",json::array({block})},{"actions",actions},{"target_action_index",target<0?json(nullptr):json(target)},{"worker_work_scope",main?"whole_strategy_serial_replay":"single_target_compute_not_full_layer_or_pure_resource_occupancy"},{"cpu_diagnostics",cpu},{"disabled_wait_count",disabled_waits},{"private_wait",waiter.describe()},{"untimed_reset_count",reset_count},{"reset_scope",main?"before_each_whole_invocation_outside_work_timer_but_inside_sampling_envelope":"initial_validation_only; target_input_bytes_checked"},{"pod_storage",json{{"iteration_capacity",maximum_iterations},{"action_capacity",stamps.size()},{"iteration_bytes",iterations.size()*sizeof(Iteration)},{"action_bytes",stamps.size()*sizeof(ActionStamp)},{"diagnostic_bytes",diagnostics.size()*sizeof(Diagnostic)},{"thread_diagnostic_bytes",thread_diagnostics.size()*sizeof(ThreadDiagnostic)},{"bare_and_stamped_action_buffers_same_size",true}}},{"initial_ports",initial.at("ports")},{"ports",final.at("ports")},{"verification_native",final},{"warmup_repeat_consistency",warm_consistency},{"native_repeat_consistency",consistency},{"post_replay_consistency",post_consistency},{"target_input_checks",input_checks},{"copies",executor.copies},{"copy_values_equal",executor.copy_equal},{"cuda_schedule",json{{"before_ready",schedule_before},{"after_release",schedule_after}}},{"observed_runtime_flags",final.at("observed_runtime_flags")},{"offered_state_scope","known schedule requests; not measured utilization, immediate worker response or zero residual work"},{"device_state_transition_response_verified",false},{"native_memory_capacity_claimed",false},{"stream_span_limit","actual GGML stream event span includes host launch gaps/stream delays; not summed active kernel time"},{"CPU_GPU_times_additive",false},{"performance_parameters_fitted",false}};
}
}
int main(int argc,char **argv){try {
    require(argc==10,"usage: state-response-probe PROGRAM.json main|cpu-worker|gpu-worker bare|stamped|diagnostic|diagnostic-control|thread-diagnostic TARGET_INDEX|none WARMUP READY.json GO.json STOP.json OUTPUT.json");
    const std::string role=argv[2],instrumentation=argv[3];require(role=="main"||role=="cpu-worker"||role=="gpu-worker","unknown role");require(instrumentation=="bare"||instrumentation=="stamped"||instrumentation=="diagnostic"||instrumentation=="diagnostic-control"||instrumentation=="thread-diagnostic","unknown instrumentation");
    const int target=std::string(argv[4])=="none"?-1:single_layer::integer(argv[4],0,10000,"target"),warmup=single_layer::integer(argv[5],0,10000,"warmup");
    std::set<std::filesystem::path> paths;for(int i=6;i<=9;++i){const auto p=std::filesystem::absolute(std::filesystem::u8path(argv[i])).lexically_normal();require(paths.insert(p).second,"handshake/output paths must differ");require(!state_response::exists(argv[i]),"handshake/output path already exists");}
#if defined(_WIN32)
    _putenv_s("NVIDIA_TF32_OVERRIDE","0");_putenv_s("GGML_CUDA_DISABLE_GRAPHS","1");
#else
    setenv("NVIDIA_TF32_OVERRIDE","0",1);setenv("GGML_CUDA_DISABLE_GRAPHS","1",1);
#endif
    std::ifstream input(argv[1]);require(bool(input),"cannot read program");json source;input>>source;json result;
    const auto cuda_request=state_response::configure_cuda_schedule_from_environment();
    {Executor executor(std::move(source));result=state_response::run(executor,role,instrumentation,target,warmup,argv[6],argv[7],argv[8],cuda_request);}
    result["owned_Executor_destructor_returned"]=true;state_response::write_new(argv[9],result);
    std::cout<<json{{"status","ok"},{"role",role},{"process_id",result.at("process_id")},{"iteration_count",result.at("blocks").at(0).at("iteration_count")}}.dump()<<'\n';return 0;
}catch(const std::exception &error){std::cerr<<"state-response-probe: "<<error.what()<<'\n';return 2;}}
