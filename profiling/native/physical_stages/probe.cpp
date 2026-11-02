// New physical-stage metrology. Historical Executor.action is included unchanged.
#include "state_response_generated.inc"
#include "cupti_capture.hpp"

namespace physical_stages {
using physical_capture::qpc_now_ns;
using state_response::exists;
using state_response::write_new;

struct ActionInterval { uint64_t begin=0,end=0,external=0; size_t index=0; };
struct Invocation { uint64_t begin=0,end=0,reset_begin=0,reset_end=0; };

static json action_row(const json &action,const ActionInterval &interval) {
    return json{{"action_index",interval.index},{"kind",action.at("kind")},
        {"target_id",action.value("target_id",json(nullptr))},{"external_id",interval.external},
        {"begin_ns",interval.begin},{"end_ns",interval.end},
        {"device",action.value("device",json(nullptr))},
        {"timing_scope","inclusive unchanged Executor.action host completion"}};
}

static json serialization_self_test() {
    const ActionInterval interval{10,20,1,0};
    const auto advance=action_row(json{{"kind","ADVANCE"},{"target_id",nullptr},{"device",nullptr}},interval);
    const auto copy=action_row(json{{"kind","COPY"},{"target_id","tensor"},{"device",nullptr}},interval);
    const auto missing=action_row(json{{"kind","ADVANCE"}},interval);
    require(advance.at("target_id").is_null()&&advance.at("device").is_null(),"null control serialization failed");
    require(copy.at("target_id")=="tensor"&&copy.at("device").is_null(),"copy serialization failed");
    require(missing.at("target_id").is_null()&&missing.at("device").is_null(),"absent optional serialization failed");
    return json{{"schema","physical-stage-serialization-self-test-1"},{"passed",true},
        {"hardware_executed",false},{"cases",json::array({advance,copy,missing})}};
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

static void wait_for_go(const std::string &go,const std::string &stop,json &source) {
    state_response::PrivateWait waiter;
    const auto deadline=single_layer::tick()+30*single_layer::frequency();
    for (;;) {
        require(single_layer::tick()<deadline,"GO wait timeout");
        require(!exists(stop),"stopped before GO");
        if (exists(go)) {
            try { std::ifstream input(go);input>>source;break; }
            catch (const json::parse_error &) {}
        }
        waiter.until(std::min(deadline,single_layer::tick()+single_layer::frequency()/500));
    }
    require(source.at("schema")=="physical-stages-go-1","GO schema differs");
    require(source.at("clock_id")=="windows_qpc"&&
            state_response::integer(source.at("frequency_hz"),"GO frequency",1)==single_layer::frequency(),
            "GO QPC clock differs");
    const auto begin=state_response::integer(source.at("start_tick"),"GO start",1);
    const auto end=state_response::integer(source.at("end_tick"),"GO end",begin+1);
    require(end-begin<=120*single_layer::frequency(),"GO window exceeds 120 seconds");
    require(end>single_layer::tick(),"GO window already ended");
    while (single_layer::tick()<begin) {
        require(!exists(stop),"stopped before common start");
        waiter.until(begin);
    }
}

static json run(Executor &executor,const std::string &mode,const std::string &instrument,
                int target,int warmup,int samples,const std::string &poll,
                const std::string &ready,const std::string &go,const std::string &stop,
                const state_response::CudaScheduleRequest &cuda_request) {
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
        {"warmup_repeat_consistency",warm_consistency},{"cuda_schedule",before_schedule}});
    json go_source;
    wait_for_go(go,stop,go_source);
    const int64_t end_tick=go_source.at("end_tick");
    const auto cpu_first=single_layer::cpu_snapshot();
    const uint64_t batch_begin=qpc_now_ns();
    size_t completed=0;
    for (int iteration=0;iteration<samples;++iteration) {
        require(!exists(stop),"STOP arrived before requested samples completed");
        require(single_layer::tick()<end_tick,"GO window ended before requested samples completed");
        auto &item=invocations[iteration];
        if (mode=="full") {item.reset_begin=qpc_now_ns();executor.reset();item.reset_end=qpc_now_ns();}
        item.begin=qpc_now_ns();
        for(size_t j=0;j<actions_per_iteration;++j) {
            const size_t index=mode=="full"?j:static_cast<size_t>(target);
            auto &interval=intervals[static_cast<size_t>(iteration)*actions_per_iteration+j];
            interval.index=index;interval.external=1+static_cast<uint64_t>(iteration)*actions_per_iteration+j;
            if(capture)capture->push(interval.external);
            interval.begin=qpc_now_ns();
            executor.action(actions.at(index));
            interval.end=qpc_now_ns();
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
    state_response::PrivateWait waiter;
    const auto release_deadline=single_layer::tick()+120*single_layer::frequency();
    while(!exists(stop)) {
        require(single_layer::tick()<release_deadline,"STOP release timeout");
        waiter.until(release_deadline);
    }
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
            stages.push_back(action_row(action,interval));
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
    require(argc==12,"usage: physical-stages-probe PROGRAM.json full|target cupti|bare TARGET_INDEX|none WARMUP SAMPLES default|0 READY.json GO.json STOP.json OUTPUT.json");
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
    const auto cuda_request=state_response::configure_cuda_schedule_from_environment();
    json result;
    {Executor executor(std::move(source));result=physical_stages::run(executor,mode,instrument,target,warmup,samples,poll,
        argv[8],argv[9],argv[10],cuda_request);}
    result["owned_Executor_destructor_returned"]=true;
    state_response::write_new(argv[11],result);
    std::cout<<json{{"status","ok"},{"process_id",result.at("process_id")},
        {"samples_completed",result.at("samples_completed")},{"instrumentation",instrument}}.dump()<<'\n';
    return 0;
}catch(const std::exception &error) {
    std::cerr<<"physical-stages-probe: "<<error.what()<<'\n';return 2;
}}
