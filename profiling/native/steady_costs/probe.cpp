// Batched steady-state prices: no CPU cycle counters and no residual inference.
#include "state_response_generated.inc"
#include "components.hpp"

namespace steady {
using single_layer::tick;
using single_layer::frequency;
static int64_t ns(int64_t first,int64_t last){return single_layer::ns(first,last,frequency());}

struct Request {
    json source;std::string mode,instrumentation,poll;int warmup=0,trials=0,repeats=0,min_ms=0;int target=-1;DWORD_PTR main_mask=0;
    explicit Request(json value):source(std::move(value)) {
        require(source.at("schema")=="steady-cost-request-1","steady request schema differs");
        mode=str(source,"mode");instrumentation=str(source,"instrumentation");poll=str(source,"poll");
        require(mode=="full"||mode=="full-stamped"||mode=="action"||mode=="all-actions"||mode=="async-graph"||mode=="sync-idle","unknown steady mode");
        require(instrumentation=="bare"||instrumentation=="events","unknown steady instrumentation");
        require(poll=="0","steady CPU pool must use explicit poll=0");
        const auto w=state_response::integer(source.at("warmup"),"warmup");require(w<=10000,"warmup exceeds envelope");warmup=static_cast<int>(w);
        trials=static_cast<int>(positive(source.at("trials"),"trials",10000));
        repeats=static_cast<int>(positive(source.at("repeats"),"repeats",100000));
        const auto ms=state_response::integer(source.at("warmup_min_ms"),"warmup min ms");require(ms<=60000,"warmup duration exceeds envelope");min_ms=static_cast<int>(ms);
        require(warmup<=10000&&min_ms<=60000&&uint64_t(trials)*repeats<=2000000,"steady request sampling envelope exceeded");
        if(!source.at("target_action_index").is_null()) {const auto index=state_response::integer(source.at("target_action_index"),"target index");require(index<=10000,"target index exceeds envelope");target=static_cast<int>(index);}
        main_mask=static_cast<DWORD_PTR>(positive(source.at("main_thread_mask"),"main thread affinity mask",INT64_MAX));
        require(instrumentation=="bare"||mode=="async-graph","events are admitted only for explicit async-graph mode");
    }
};

class OwnThreadAffinity {
    DWORD_PTR previous_=0;bool closed_=false;GROUP_AFFINITY before_{},after_{};json receipt_;
public:
    explicit OwnThreadAffinity(DWORD_PTR requested) {
        require(GetThreadGroupAffinity(GetCurrentThread(),&before_)!=0,"cannot observe own initial thread group affinity");
        DWORD_PTR process_mask=0,system_mask=0;require(GetProcessAffinityMask(GetCurrentProcess(),&process_mask,&system_mask)!=0,"cannot observe own process affinity");
        require((requested&process_mask)==requested,"requested main thread mask is outside current process affinity");
        previous_=SetThreadAffinityMask(GetCurrentThread(),requested);require(previous_!=0,"own main thread affinity set failed");
        if(!GetThreadGroupAffinity(GetCurrentThread(),&after_)||after_.Group!=before_.Group||after_.Mask!=requested){SetThreadAffinityMask(GetCurrentThread(),previous_);closed_=true;throw std::runtime_error("own main thread affinity verification failed");}
        receipt_=json{{"schema","steady-owned-main-thread-affinity-1"},{"requested_mask",static_cast<uint64_t>(requested)},
            {"observed_mask",static_cast<uint64_t>(after_.Mask)},{"observed_group",after_.Group},
            {"previous_mask",static_cast<uint64_t>(previous_)},{"process_mask",static_cast<uint64_t>(process_mask)},
            {"system_mask",static_cast<uint64_t>(system_mask)},{"set_succeeded",true},{"applied_before_backend_allocation_and_warmup",true},
            {"scope","current owned main thread only; no process/system settings changed"},{"continuous_CPU_residency_measured",false}};
    }
    ~OwnThreadAffinity(){if(!closed_&&previous_)SetThreadAffinityMask(GetCurrentThread(),previous_);}
    json finish() {
        GROUP_AFFINITY final{};require(GetThreadGroupAffinity(GetCurrentThread(),&final)!=0&&final.Group==after_.Group&&final.Mask==after_.Mask,"own thread affinity changed during measurement");
        receipt_["final_mask"]=static_cast<uint64_t>(final.Mask);receipt_["final_group"]=final.Group;
        require(SetThreadAffinityMask(GetCurrentThread(),previous_)!=0,"own original thread affinity restore failed");closed_=true;
        GROUP_AFFINITY restored{};require(GetThreadGroupAffinity(GetCurrentThread(),&restored)!=0&&restored.Group==before_.Group&&restored.Mask==before_.Mask,"own original affinity restore verification failed");
        receipt_["restored"]=true;return receipt_;
    }
};

struct Frame {
    int64_t begin=0,end=0,enqueue_begin=0,enqueue_end=0,sync_begin=0,sync_end=0;
    double stream_ms=-1;
    std::vector<uint64_t> action_ticks;
};

template<class Function>
static json warm(const Request &request,Function execute) {
    const auto begin=tick();uint64_t iterations=0;
    while(iterations<static_cast<uint64_t>(request.warmup)||ns(begin,tick())<int64_t(request.min_ms)*1000000) {
        require(ns(begin,tick())<120000000000LL,"warmup exceeded fixed 120 second safety envelope");
        execute();++iterations;
    }
    const auto end=tick();return json{{"requested_min_iterations",request.warmup},{"requested_min_ms",request.min_ms},
        {"actual_iterations",iterations},{"begin_tick",begin},{"end_tick",end},{"wall_ns",ns(begin,end)},
        {"completed_outside_all_trial_timers",true}};
}

static json frame_json(const Frame &frame,int trial,const Request &request) {
    json actions=json::array();for(size_t i=0;i<frame.action_ticks.size();++i)actions.push_back(json{{"action_index",i},
        {"wall_ns",single_layer::ns(0,frame.action_ticks[i],frequency())}});
    return json{{"trial",trial},{"begin_tick",frame.begin},{"end_tick",frame.end},{"wall_ns",ns(frame.begin,frame.end)},
        {"repeats",request.repeats},{"actions",actions},
        {"enqueue_ns",request.mode=="async-graph"?json(ns(frame.enqueue_begin,frame.enqueue_end)):json(nullptr)},
        {"sync_ns",request.mode=="async-graph"?json(ns(frame.sync_begin,frame.sync_end)):json(nullptr)},
        {"stream_elapsed_ms",frame.stream_ms>=0?json(frame.stream_ms):json(nullptr)},
        {"stream_elapsed_ns",frame.stream_ms>=0?json(static_cast<int64_t>(std::llround(frame.stream_ms*1000000.0))):json(nullptr)}};
}

template<class Enqueue,class Synchronize>
static Frame async_frame(const Request &request,Enqueue enqueue,Synchronize synchronize,state_response::TimingEvents *events,ggml_backend_t backend) {
    Frame result;result.begin=tick();if(events)events->begin(backend);
    result.enqueue_begin=tick();for(int repeat=0;repeat<request.repeats;++repeat)enqueue();result.enqueue_end=tick();
    if(events)events->end(backend);result.sync_begin=tick();synchronize();result.sync_end=tick();result.end=tick();
    if(events)result.stream_ms=events->elapsed_ms();return result;
}

static json run_full(Executor &executor,const Request &request) {
    const auto &actions=executor.program.at("actions");
    require(!actions.empty()&&uint64_t(actions.size())*request.repeats*request.trials<=2000000,"full steady action output envelope exceeded");
    const bool full=request.mode=="full"||request.mode=="full-stamped";
    require((full||request.mode=="all-actions")?request.target<0:request.target>=0&&static_cast<size_t>(request.target)<actions.size(),"mode/target inventory differs");
    const auto pool=single_layer::configure_pool(executor,request.poll);
    executor.validate();const auto initial=single_layer::capture_verification(executor);
    std::vector<state_response::InputSnapshot> snapshots;
    if(request.target>=0&&actions.at(request.target).at("kind")=="COMPUTE")snapshots=state_response::inputs(executor,actions.at(request.target));
    ggml_backend_t async_backend=nullptr;Graph *async_graph=nullptr;
    if(request.mode=="async-graph") {
        require(actions.at(request.target).at("kind")=="COMPUTE","async-graph requires a compute action");
        async_graph=executor.graphs.at(str(actions.at(request.target),"target_id")).get();async_backend=async_graph->backend;
    }
    if(request.instrumentation=="events")require(async_backend==executor.gpu&&executor.gpu,"events require an actual GPU graph backend");
    auto execute_plain=[&](){
        if(full||request.mode=="all-actions")for(const auto &action:actions)executor.action(action);
        else if(request.mode=="async-graph") {require(ggml_backend_graph_compute_async(async_backend,async_graph->graph)==GGML_STATUS_SUCCESS,"warm async compute failed");ggml_backend_synchronize(async_backend);}
        else if(request.mode=="sync-idle") {ggml_backend_synchronize(executor.cpu);if(executor.gpu)ggml_backend_synchronize(executor.gpu);}
        else executor.action(actions.at(request.target));
    };
    const auto warmed=warm(request,execute_plain);
    std::unique_ptr<state_response::TimingEvents> events;
    if(request.instrumentation=="events")events=std::make_unique<state_response::TimingEvents>(executor.gpu,executor.program.at("physical").value("cuda_device",0));
    std::vector<Frame> frames(request.trials);
    for(auto &frame:frames)if(request.mode=="full-stamped")frame.action_ticks.resize(actions.size());
    json groups=json::array();
    if(request.mode=="all-actions") {
        for(size_t selected=0;selected<actions.size();++selected) {
            const auto single_warm=warm(request,[&](){executor.action(actions.at(selected));});
            std::vector<Frame> local(request.trials);
            for(auto &frame:local){frame.begin=tick();for(int repeat=0;repeat<request.repeats;++repeat)executor.action(actions.at(selected));frame.end=tick();}
            json trials=json::array();for(int i=0;i<request.trials;++i)trials.push_back(frame_json(local[i],i,request));
            groups.push_back(json{{"action_index",selected},{"action",actions.at(selected)},{"warmup",single_warm},{"trials",trials}});
        }
    } else for(int trial=0;trial<request.trials;++trial) {
        if(full)executor.reset(); // Outside the entire R-repeat price interval.
        auto &frame=frames[trial];
        if(request.mode=="async-graph") {
            frame=async_frame(request,[&](){require(ggml_backend_graph_compute_async(async_backend,async_graph->graph)==GGML_STATUS_SUCCESS,"timed async compute failed");},
                [&](){ggml_backend_synchronize(async_backend);},events.get(),async_backend);
        } else {
            frame.begin=tick();
            for(int repeat=0;repeat<request.repeats;++repeat) {
                if(request.mode=="full-stamped")for(size_t index=0;index<actions.size();++index) {
                    const auto first=tick();executor.action(actions.at(index));const auto last=tick();frame.action_ticks[index]+=last-first;
                } else execute_plain();
            }
            frame.end=tick();
        }
    }
    const auto final=single_layer::capture_verification(executor);
    const auto consistency=state_response::compare_ports(initial.at("ports"),final.at("ports"));
    const auto input_checks=state_response::verify_inputs(executor,snapshots);
    json trials=json::array();if(request.mode!="all-actions")for(int i=0;i<request.trials;++i)trials.push_back(frame_json(frames[i],i,request));
    return json{{"program_sha256",executor.program.at("program_sha256")},{"program_format",executor.program.at("format_version")},
        {"actions",actions},{"trials",trials},{"groups",groups},{"warmup",warmed},{"verification_native",final},
        {"native_repeat_consistency",consistency},{"target_input_checks",input_checks},{"threadpool",pool},
        {"reset_scope",full?"one reset before each trial, outside the entire R-repeat interval; repeated fixed-input full graphs do not advance generation state":"validation only; target input bytes checked where compute"},
        {"timing_scope",request.mode=="all-actions"?"isolated hot same-prepared-input action batches; not mixed-context prices":
            request.mode=="async-graph"?"explicit async graph batch; one final backend synchronization, changed boundary":
            request.mode=="sync-idle"?"empty backend CPU/GPU synchronization calls only; not busy wait calibration":
            "unchanged Executor.action fixed-input repeated full/action completion; initialization, reset, warmup, verification and serialization excluded"},
        {"component_output",nullptr},{"component",nullptr},{"unchanged_Executor_action_used",request.mode!="async-graph"&&request.mode!="sync-idle"}};
}

static json run_component(Component &component,const Request &request) {
    require(request.target<0,"compact component has no original action index");
    require(request.mode=="action"||request.mode=="async-graph"||request.mode=="sync-idle","compact component admits action, async-graph or sync-idle only");
    require(request.mode!="async-graph"||component.kind!="transfer","compact transfer async mode not identified; use actual synchronous copy path");
    require(request.instrumentation!="events"||component.device=="gpu"&&component.graph,"event measurement requires the actual compute GPU backend");
    component.action();const auto initial=component.validate_output();
    auto plain=[&](){if(request.mode=="sync-idle")component.synchronize();else component.action();};
    const auto warmed=warm(request,plain);
    std::unique_ptr<state_response::TimingEvents> events;
    if(request.instrumentation=="events")events=std::make_unique<state_response::TimingEvents>(component.gpu,component.spec.at("cuda_device").get<int>());
    std::vector<Frame> frames(request.trials);
    for(auto &frame:frames) {
        if(request.mode=="async-graph")frame=async_frame(request,[&](){component.enqueue();},[&](){component.synchronize();},events.get(),component.compute_backend);
        else {frame.begin=tick();for(int repeat=0;repeat<request.repeats;++repeat)plain();frame.end=tick();}
    }
    auto final=component.validate_output();
    require(initial.at("output_sha256")==final.at("output_sha256"),"component output changed after fixed-input batches");
    json trials=json::array();for(int i=0;i<request.trials;++i)trials.push_back(frame_json(frames[i],i,request));
    if(final.contains("output")){final["values"]=final.at("output");final.erase("output");}
    final["sha256"]=final.at("output_sha256");final.erase("output_sha256");
    return json{{"program_sha256",component.spec.at("program_sha256")},{"program_format",component.spec.at("format_version")},
        {"program_spec",component.spec},{"actions",json::array()},{"trials",trials},{"groups",json::array()},{"warmup",warmed},
        {"verification_native",nullptr},{"native_repeat_consistency",json{{"passed",true},{"exact_output_bytes_unchanged",true}}},
        {"component_output",final},{"component",component.describe()},{"unchanged_Executor_action_used",false},
        {"timing_scope",request.mode=="async-graph"?"new standalone GGML graph asynchronous batch and one final sync; stream span can include host gaps":
            request.mode=="sync-idle"?"empty selected backend synchronize calls only; not queued device work":
            "new standalone GGML component completion batch; actual compute+materialization or copy path; procedural inputs and graph construction excluded"}};
}

static json self_test() {
    const auto value=mix32(123,0,0);require(value==mix32(123,0,0)&&value!=mix32(123,1,0),"mix32 deterministic/index contract failed");
    const std::string text="abc";require(sha256_bytes(text.data(),text.size())=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad","SHA256 known vector failed");
    Request request(json{{"schema","steady-cost-request-1"},{"mode","action"},{"target_action_index",nullptr},
        {"warmup",0},{"warmup_min_ms",0},{"trials",1},{"repeats",1},{"poll","0"},{"instrumentation","bare"},{"main_thread_mask",4}});
    auto zero_json=request.source;zero_json["target_action_index"]=0;Request zero(zero_json);require(zero.target==0,"action zero parser failed");
    bool unknown_mode_rejected=false;try {auto invalid=request.source;invalid["mode"]="implicit";Request rejected(invalid);}catch(const std::exception &){unknown_mode_rejected=true;}
    require(unknown_mode_rejected,"unknown request mode accepted");
    return json{{"passed",true},{"backend_or_Executor_constructed",false},{"native_hardware_timing_executed",false},
        {"mix32_seed123_index0_ordinal0",value},{"SHA256_known_vector_passed",true},{"target_zero_accepted",true},{"unknown_request_mode_rejected",true}};
}
}

int main(int argc,char **argv){try{
    if(argc==2&&std::string(argv[1])=="--self-test"){std::cout<<steady::self_test().dump()<<'\n';return 0;}
    require(argc==4,"usage: steady-costs-probe PROGRAM.json REQUEST.json OUTPUT.json");
    require(!state_response::exists(argv[3]),"steady output already exists");
    _putenv_s("NVIDIA_TF32_OVERRIDE","0");_putenv_s("GGML_CUDA_DISABLE_GRAPHS","1");
    std::ifstream program_file(std::filesystem::u8path(argv[1])),request_file(std::filesystem::u8path(argv[2]));
    require(program_file.good()&&request_file.good(),"cannot read explicit program/request");json program,request_json;program_file>>program;request_file>>request_json;
    steady::Request request(request_json);steady::OwnThreadAffinity affinity(request.main_mask);json result;
    if(program.at("format_version")=="steady-component-program-1") {
        steady::Component component(program,request.poll);result=steady::run_component(component,request);
    } else {Executor executor(std::move(program));result=steady::run_full(executor,request);}
    result.update(json{{"schema","steady-cost-native-1"},{"request",request_json},{"process_id",single_layer::pid()},
        {"main_thread_affinity",affinity.finish()},
        {"main_thread_id",GetCurrentThreadId()},{"clock",json{{"clock_id","windows_qpc"},{"frequency_hz",single_layer::frequency()}}},
        {"collected_cpu_cycles",false},{"per_action_CPU_counter_queries",false},{"performance_parameters_fitted",false},
        {"CPU_GPU_times_additive",false},{"stream_elapsed_is_kernel_active_time",false},
        {"stream_elapsed_scope","actual pinned GGML CUDA backend stream event envelope; includes possible host submission gaps and stream delays"},
        {"time_units","wall/enqueue/sync/action/stream ns are totals over repeats; divide by request.repeats for per-operation response"},
        {"observed_runtime_flags",json{{"GGML_CUDA_DISABLE_GRAPHS",std::getenv("GGML_CUDA_DISABLE_GRAPHS")},{"NVIDIA_TF32_OVERRIDE",std::getenv("NVIDIA_TF32_OVERRIDE")},{"matmul_precision","GGML_PREC_F32"}}},
        {"hardware_accuracy_qualified",false}});
    state_response::write_new(argv[3],result);std::cout<<json{{"status","ok"},{"schema","steady-cost-native-1"},{"process_id",single_layer::pid()}}.dump()<<'\n';return 0;
}catch(const std::exception &error){std::cerr<<"steady-costs-probe: "<<error.what()<<'\n';return 2;}}
