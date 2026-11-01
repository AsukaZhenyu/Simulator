// Scoped controlled competition: actors execute real work on declared resources.
#define main historical_steady_entry_not_called
#include "../../native/steady_costs/probe.cpp"
#undef main
#include <atomic>
#include <exception>
#include <mutex>
#include "owned_timer.hpp"

namespace dynamic_external {
struct Interval {int64_t begin_ms=0,end_ms=0;uint64_t calls=0;int64_t first=0,last=0,response_ticks=0;};
static std::vector<Interval> intervals(const json &values){
    require(values.is_array()&&values.size()<=16,"bounded actor interval array required");std::vector<Interval> result;int64_t previous=-5000;
    for(const auto &v:values){require(v.is_array()&&v.size()==2&&v.at(0).is_number_integer()&&v.at(1).is_number_integer(),"integer interval bounds required");
        Interval i;i.begin_ms=v.at(0).get<int64_t>();i.end_ms=v.at(1).get<int64_t>();
        require(i.begin_ms>=-2000&&i.begin_ms>=previous&&i.end_ms>i.begin_ms&&i.end_ms<=120000,"actor interval order/envelope differs");
        previous=i.end_ms;result.push_back(i);}
    return result;
}
static int active_index(const std::vector<Interval> &v,int64_t elapsed_ticks,int64_t hz){
    for(size_t i=0;i<v.size();++i){const auto begin=v[i].begin_ms*hz/1000,end=v[i].end_ms*hz/1000;
        if(elapsed_ticks>=begin&&elapsed_ticks<end)return static_cast<int>(i);}
    return -1;
}
static json process_affinity(const json &request){
    auto mask=static_cast<DWORD_PTR>(steady::positive(request.at("expected_process_mask"),"process mask",INT64_MAX));
    require(static_cast<uint64_t>(mask)==4294967295ULL,"all-32 owned process mask required");DWORD_PTR first=0,system=0,actual=0;
    require(GetProcessAffinityMask(GetCurrentProcess(),&first,&system)!=0&&(mask&system)==mask,"process affinity unavailable");
    require(SetProcessAffinityMask(GetCurrentProcess(),mask)!=0&&GetProcessAffinityMask(GetCurrentProcess(),&actual,&system)!=0&&actual==mask,"process affinity readback differs");
    return json{{"requested_mask",static_cast<uint64_t>(mask)},{"observed_mask",static_cast<uint64_t>(actual)},
        {"initial_mask",static_cast<uint64_t>(first)},{"applied_before_backend_initialization",true}};
}
struct Control {
    std::atomic<int64_t> epoch{0};std::atomic<bool> target_done{false},cancel{false};
    HANDLE epoch_ready=nullptr,target_finished=nullptr,cancelled=nullptr;
    Control(){
        epoch_ready=CreateEventW(nullptr,TRUE,FALSE,nullptr);target_finished=CreateEventW(nullptr,TRUE,FALSE,nullptr);cancelled=CreateEventW(nullptr,TRUE,FALSE,nullptr);
        if(!epoch_ready||!target_finished||!cancelled){if(epoch_ready)CloseHandle(epoch_ready);if(target_finished)CloseHandle(target_finished);if(cancelled)CloseHandle(cancelled);throw std::runtime_error("owned actor events unavailable");}
    }
    ~Control(){CloseHandle(epoch_ready);CloseHandle(target_finished);CloseHandle(cancelled);}
    void announce(int64_t tick){epoch.store(tick,std::memory_order_release);require(SetEvent(epoch_ready)!=0,"owned epoch event failed");}
    void finish(){target_done.store(true,std::memory_order_release);require(SetEvent(target_finished)!=0,"owned target-finished event failed");}
    void abort(){cancel.store(true);target_done.store(true);SetEvent(cancelled);SetEvent(epoch_ready);SetEvent(target_finished);}
};
class Actor {
    Control &control_;json specification_;std::vector<Interval> intervals_;std::thread thread_;
    std::atomic<bool> ready_{false},finished_{false};std::exception_ptr error_;std::mutex mutex_;json result_;
    void worker(){try{
        const std::string device=str(specification_,"device");
        const auto mask=static_cast<DWORD_PTR>(steady::positive(specification_.at("main_thread_mask"),"actor mask",INT64_MAX));
        require(mask==(device=="cpu"?4:65536),"actor must use explicit target-CPU or separate GPU-submission CPU");
        steady::OwnThreadAffinity affinity(mask);
        dynamic_control::OwnedHighResolutionTimer timer;
        steady::Component work(specification_.at("component"),"0");
        for(int i=0;i<20;++i)work.action();
        const auto initial=work.validate_output();
        const auto ready_tick=steady::tick();ready_.store(true,std::memory_order_release);
        require(WaitForSingleObject(control_.epoch_ready,INFINITE)==WAIT_OBJECT_0,"owned epoch event wait failed");
        const int64_t hz=steady::frequency(),epoch=control_.epoch.load();
        const int64_t last_end=intervals_.empty()?0:intervals_.back().end_ms*hz/1000;
        while(!control_.cancel.load()){
            const auto now=steady::tick(),elapsed=now-epoch;
            if(control_.target_done.load()&&elapsed>=last_end)break;
            const int index=active_index(intervals_,elapsed,hz);
            if(index<0){
                int64_t next=INT64_MAX;for(const auto &v:intervals_)if(v.begin_ms*hz/1000>elapsed){next=epoch+v.begin_ms*hz/1000;break;}
                if(next!=INT64_MAX){if(!timer.wait_until_qpc(next,hz,control_.cancelled))break;}
                else {HANDLE events[2]={control_.target_finished,control_.cancelled};require(WaitForMultipleObjects(2,events,FALSE,INFINITE)<WAIT_OBJECT_0+2,"owned actor completion wait failed");}
                continue;
            }
            auto &v=intervals_.at(static_cast<size_t>(index));const auto begin=steady::tick();work.action();const auto end=steady::tick();
            if(v.calls==0)v.first=begin;v.last=end;++v.calls;v.response_ticks+=end-begin;
        }
        const auto stop=steady::tick();const auto final=work.validate_output();
        require(initial.at("output_sha256")==final.at("output_sha256"),"owned actor computation changed its output");
        json rows=json::array();for(const auto &v:intervals_)rows.push_back(json{{"planned_begin_ms",v.begin_ms},{"planned_end_ms",v.end_ms},
            {"calls",v.calls},{"actual_first_tick",v.calls?json(v.first):json(nullptr)},
            {"actual_last_completion_tick",v.calls?json(v.last):json(nullptr)},
            {"sum_inclusive_work_response_ns",single_layer::ns(0,v.response_ticks,hz)},
            {"start_lateness_ns",v.calls?json(single_layer::ns(epoch+v.begin_ms*hz/1000,v.first,hz)):json(nullptr)},
            {"last_completion_overrun_ns",v.calls&&v.last>epoch+v.end_ms*hz/1000?json(single_layer::ns(epoch+v.end_ms*hz/1000,v.last,hz)):json(0)}});
        result_=json{{"device",device},{"thread_affinity",affinity.finish()},{"ready_tick",ready_tick},{"stopped_tick",stop},
            {"intervals",rows},{"component",work.describe()},{"component_output_sha256",final.at("output_sha256")},
            {"repeated_output_equal",true},{"offered_work_is_available_service",false},
            {"kernel_may_finish_after_planned_stop",true},{"cancelled",control_.cancel.load()},
            {"controller",dynamic_control::OwnedHighResolutionTimer::describe()},{"idle_actor_busy_polling",false}};
    }catch(...){std::lock_guard<std::mutex> guard(mutex_);error_=std::current_exception();ready_.store(true);}
        finished_.store(true,std::memory_order_release);
    }
public:
    Actor(Control &control,json specification):control_(control),specification_(std::move(specification)),
        intervals_(intervals(specification_.at("intervals_ms"))){thread_=std::thread([this](){worker();});}
    ~Actor(){if(thread_.joinable()){control_.abort();thread_.join();}}
    bool ready(){return ready_.load(std::memory_order_acquire);}
    void check_error(){std::lock_guard<std::mutex> guard(mutex_);if(error_)std::rethrow_exception(error_);}
    json finish(){if(thread_.joinable())thread_.join();check_error();require(finished_.load(),"actor did not terminate");return result_;}
};
struct Stamp {int64_t begin=0,end=0;};
struct Block {int64_t begin=0,end=0;};
static json run(Executor &e,const steady::Request &sampling,const json &request){
    require(e.gpu&&e.program.at("physical").at("threads")==1,"GPU and one target CPU thread required");
    require(sampling.mode=="full"&&sampling.target<0&&sampling.instrumentation=="bare"&&sampling.trials==128&&sampling.repeats==16,
        "continuous contract requires128 blocks of16 full replays");
    require(sampling.warmup==20&&sampling.min_ms==500,"explicit20/500ms target warmup required");
    const auto instrument=str(request,"instrumentation");require(instrument=="bare"||instrument=="stamped","unknown primary/profile instrumentation");
    const auto &actions=e.program.at("actions");require(actions.size()*2048<=2000000,"action trace envelope exceeded");
    const auto pool=single_layer::configure_pool(e,sampling.poll);e.validate();const auto initial=single_layer::capture_verification(e);
    Control control;Actor cpu(control,request.at("cpu_actor")),gpu(control,request.at("gpu_actor"));
    try{
        dynamic_control::OwnedHighResolutionTimer timer;
        const auto waiting=steady::tick();while(!cpu.ready()||!gpu.ready()){
            require(steady::ns(waiting,steady::tick())<30000000000LL,"actor preparation timed out");timer.wait_ns(1000000);}
        cpu.check_error();gpu.check_error();
        auto full=[&](){for(const auto &a:actions)e.action(a);};const auto warm=steady::warm(sampling,full);
        e.reset();full(); // One initial prime. No reset/prime is inserted between timed blocks.
        std::vector<Block> blocks(128);std::vector<Stamp> stamps(instrument=="stamped"?actions.size()*2048:0);
        const auto hz=steady::frequency();const int64_t epoch=steady::tick()+hz*2200/1000;
        control.announce(epoch);require(timer.wait_until_qpc(epoch,hz,control.cancelled),"owned target wait cancelled");cpu.check_error();gpu.check_error();
        const auto begin=steady::tick();size_t cursor=0;
        for(int block=0;block<128;++block){auto &b=blocks.at(block);b.begin=steady::tick();
            for(int repeat=0;repeat<16;++repeat){
                if(instrument=="bare")full();
                else for(const auto &a:actions){auto &s=stamps[cursor++];s.begin=steady::tick();e.action(a);s.end=steady::tick();}
            }
            b.end=steady::tick();cpu.check_error();gpu.check_error();
        }
        const auto end=steady::tick();control.finish();
        const auto cpu_result=cpu.finish(),gpu_result=gpu.finish();
        const auto final=single_layer::capture_verification(e);json rows=json::array();cursor=0;
        for(int block=0;block<128;++block){const auto &b=blocks.at(block);json measured=json::array();
            if(instrument=="stamped")for(int repeat=0;repeat<16;++repeat)for(size_t index=0;index<actions.size();++index){
                const auto &s=stamps[cursor++];measured.push_back(json{{"global_replay",block*16+repeat},{"action_index",index},
                    {"begin_tick",s.begin},{"end_tick",s.end},{"wall_ns",steady::ns(s.begin,s.end)}});}
            rows.push_back(json{{"block",block},{"replays",16},{"begin_tick",b.begin},{"end_tick",b.end},
                {"wall_ns",steady::ns(b.begin,b.end)},{"actions",measured}});
        }
        return json{{"program_sha256",e.program.at("program_sha256")},{"actions",actions},{"instrumentation",instrument},
            {"blocks",rows},{"target_expected_origin_tick",epoch},{"target_begin_tick",begin},{"target_end_tick",end},
            {"target_wall_ns",steady::ns(begin,end)},{"initial_start_lateness_ns",steady::ns(epoch,begin)},
            {"actors",json{{"cpu",cpu_result},{"gpu",gpu_result}}},{"warmup",warm},{"threadpool",pool},
            {"controller",dynamic_control::OwnedHighResolutionTimer::describe()},
            {"verification_native",final},{"native_repeat_consistency",state_response::compare_ports(initial.at("ports"),final.at("ports"))},
            {"continuous_replays",2048},{"reset_prime_between_timed_blocks",false},
            {"actor_timeline_triggered_by_target_latency",false},{"future_target_state_used",false},
            {"external_scope","owned competing threads, CPU-targetcore and CUDA0 independent backend; not a universal other-application model"},
            {"timing_scope","inclusive completed unchanged Executor.action; only outer block QPC in bare target; initial prime/warmup and actor join excluded"},
            {"hardware_accuracy_qualified",false},{"performance_parameters_fitted",false}};
    }catch(...){control.abort();throw;}
}
}

int main(int argc,char **argv){try{
    if(argc==2&&std::string(argv[1])=="--self-test"){
        auto v=dynamic_external::intervals(json::array({json::array({-2000,1000}),json::array({2000,3000})}));
        require(dynamic_external::active_index(v,-1000,1000)==0&&dynamic_external::active_index(v,1500,1000)==-1,"actor interval semantics differ");
        auto result=steady::self_test();result["source_sha256"]=DYNAMIC_EXTERNAL_SOURCE_SHA256;result["actor_parser_passed"]=true;
        result["hardware_executed"]=false;std::cout<<result.dump()<<'\n';return 0;
    }
    require(argc==4,"usage: dynamic-external-probe PROGRAM REQUEST OUTPUT");require(!state_response::exists(argv[3]),"external output already exists");
    _putenv_s("NVIDIA_TF32_OVERRIDE","0");_putenv_s("GGML_CUDA_DISABLE_GRAPHS","1");
    std::ifstream pf(std::filesystem::u8path(argv[1])),rf(std::filesystem::u8path(argv[2]));require(pf.good()&&rf.good(),"explicit external inputs unavailable");json program,request;pf>>program;rf>>request;
    require(request.at("schema")=="dynamic-external-request-1"&&str(request,"source_sha256")==DYNAMIC_EXTERNAL_SOURCE_SHA256,"external request/source changed");
    steady::Request sampling(request.at("sampling"));require(sampling.main_mask==4,"explicit target CPU2 mask required");
    const auto process=dynamic_external::process_affinity(request);steady::OwnThreadAffinity thread(4);json result;
    {Executor e(std::move(program));result=dynamic_external::run(e,sampling,request);}
    DWORD_PTR actual=0,system=0;require(GetProcessAffinityMask(GetCurrentProcess(),&actual,&system)!=0&&static_cast<uint64_t>(actual)==4294967295ULL,"owned process mask changed");
    json final_process=process;final_process["final_mask"]=static_cast<uint64_t>(actual);
    result.update(json{{"schema","dynamic-external-native-1"},{"request",request},{"source_sha256",DYNAMIC_EXTERNAL_SOURCE_SHA256},
        {"process_id",single_layer::pid()},{"main_thread_affinity",thread.finish()},{"process_affinity",final_process},
        {"clock",json{{"clock_id","windows_qpc"},{"frequency_hz",steady::frequency()}}}});
    state_response::write_new(argv[3],result);std::cout<<json{{"status","ok"},{"process_id",single_layer::pid()}}.dump()<<'\n';return 0;
}catch(const std::exception &error){std::cerr<<"dynamic-external-probe: "<<error.what()<<'\n';return 2;}}
