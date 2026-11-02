// Standalone idle/recovery observation; no historical probe is modified.
#define main historical_steady_entry_not_called
#include "../../native/steady_costs/probe.cpp"
#undef main
#include "owned_timer.hpp"

namespace dynamic_idle {
static json pointer_kind(void *pointer) {
    cudaPointerAttributes a{};auto status=cudaPointerGetAttributes(&a,pointer);
    if(status==cudaErrorInvalidValue){cudaGetLastError();return json{{"status","observed"},{"registered_host",false},{"unregistered_pageable",true}};}
    require(status==cudaSuccess,"host pointer query failed");
    return json{{"status","observed"},{"cuda_memory_type",static_cast<int>(a.type)},
        {"registered_host",a.type==cudaMemoryTypeHost},{"unregistered_pageable",a.type==cudaMemoryTypeUnregistered}};
}
class Registration {
    void *base_;size_t bytes_;bool active_=false,closed_=false;json proof_;
public:
    Registration(Executor &e,const std::string &mode) {
        require(mode=="pageable"||mode=="registered","unknown requested host memory");
        base_=ggml_backend_buffer_get_base(e.dram_buffer);bytes_=ggml_backend_buffer_get_size(e.dram_buffer);
        require(base_&&bytes_>0,"owned arena unavailable");auto before=pointer_kind(base_);
        require(before.at("unregistered_pageable")==true,"initial arena is not observed pageable");
        if(mode=="registered"){
            int device=-1,supported=0;require(cudaGetDevice(&device)==cudaSuccess&&
                cudaDeviceGetAttribute(&supported,cudaDevAttrHostRegisterSupported,device)==cudaSuccess&&supported,
                "explicit host registration unsupported");
            require(cudaHostRegister(base_,bytes_,cudaHostRegisterDefault)==cudaSuccess,"owned registration failed; no fallback");active_=true;
        }
        auto during=pointer_kind(base_);require(during.at(mode=="registered"?"registered_host":"unregistered_pageable")==true,"actual memory differs");
        proof_=json{{"mode",mode},{"before",before},{"during",during},{"arena_bytes",bytes_},
            {"registration_timed",false},{"scope","owned complete DRAM arena"}};
    }
    ~Registration(){if(active_)cudaHostUnregister(base_);}
    json finish(){require(!closed_,"registration already finished");if(active_){require(cudaHostUnregister(base_)==cudaSuccess,"owned unregister failed");active_=false;}
        proof_["after"]=pointer_kind(base_);require(proof_.at("after").at("unregistered_pageable")==true,"arena category not restored");
        proof_["restored"]=true;closed_=true;return proof_;}
};
static json apply_affinity(const json &request){
    const auto wanted=static_cast<DWORD_PTR>(steady::positive(request.at("expected_process_mask"),"process mask",INT64_MAX));
    require(static_cast<uint64_t>(wanted)==4294967295ULL,"explicit all-32 process mask required");
    DWORD_PTR initial=0,system=0,actual=0;
    require(GetProcessAffinityMask(GetCurrentProcess(),&initial,&system)!=0&&(wanted&system)==wanted,"process mask unavailable");
    require(SetProcessAffinityMask(GetCurrentProcess(),wanted)!=0&&GetProcessAffinityMask(GetCurrentProcess(),&actual,&system)!=0&&actual==wanted,"owned affinity readback differs");
    return json{{"requested_mask",static_cast<uint64_t>(wanted)},{"initial_mask",static_cast<uint64_t>(initial)},
        {"observed_mask",static_cast<uint64_t>(actual)},{"applied_before_backend_initialization",true}};
}
static json finish_affinity(json proof){DWORD_PTR actual=0,system=0;require(GetProcessAffinityMask(GetCurrentProcess(),&actual,&system)!=0&&
        static_cast<uint64_t>(actual)==proof.at("requested_mask").get<uint64_t>(),"owned process affinity changed");proof["final_mask"]=static_cast<uint64_t>(actual);return proof;}
static void validate_orders(const json &idle,const json &orders,int trials){
    require(idle.is_array()&&idle.size()==4&&orders.is_array()&&orders.size()==static_cast<size_t>(trials),"four conditions and complete explicit orders required");
    std::set<int> values;
    for(const auto &value:idle){int v=static_cast<int>(state_response::integer(value,"idle milliseconds"));require(v<=1000&&values.insert(v).second,"invalid/duplicate idle condition");}
    require(values.count(0)==1,"zero-idle control required");
    for(const auto &order:orders){require(order.is_array()&&order.size()==4,"condition order must be a four-element permutation");std::set<int> seen;
        for(const auto &value:order){int i=static_cast<int>(state_response::integer(value,"condition index"));require(i<4&&seen.insert(i).second,"condition order is not a permutation");}}
}
struct Copy {int64_t begin=0,end=0;};
struct Sequence {int64_t idle_begin=0,idle_end=0,burst_begin=0,burst_end=0;int trial=0,slot=0,condition=0;};
static json payload(ggml_tensor *source,ggml_tensor *target){
    const size_t bytes=ggml_nbytes(source);std::vector<uint8_t> a(bytes),b(bytes);
    ggml_backend_tensor_get(source,a.data(),0,bytes);ggml_backend_tensor_get(target,b.data(),0,bytes);
    require(a==b,"target H2D payload differs");
    return json{{"bytes",bytes},{"full_byte_equality",true},{"source_sha256",steady::sha256_bytes(a.data(),a.size())},{"target_sha256",steady::sha256_bytes(b.data(),b.size())}};
}
static json run(Executor &e,const steady::Request &sampling,const json &request){
    require(request.at("idle_controller")=="owned-high-resolution-waitable-timer-v1","explicit v2 controller required");
    dynamic_control::OwnedHighResolutionTimer timer;
    require(e.gpu&&e.program.at("physical").at("threads")==1,"explicit GPU and one GGML CPU thread required");
    require(sampling.mode=="action"&&sampling.target>=0&&sampling.instrumentation=="bare"&&sampling.repeats==16&&sampling.trials==16,
        "idle contract requires action, bare, sixteen trials and sixteen-copy bursts");
    require(sampling.warmup==20&&sampling.min_ms==500,"explicit 20/500ms warmup required");
    validate_orders(request.at("idle_ms"),request.at("orders"),sampling.trials);
    const auto &actions=e.program.at("actions");const size_t index=static_cast<size_t>(sampling.target);require(index<actions.size(),"target outside original action stream");
    const auto &target=actions.at(index);require(target.at("kind")=="COPY"&&target.at("src")=="dram"&&target.at("dst")=="vram","actual H2D action required");
    auto *source=e.tensor(str(target,"target_id"),"dram"),*dest=e.tensor(str(target,"target_id"),"vram");
    require(ggml_nbytes(source)==2097152&&ggml_nbytes(dest)==2097152,"actual two-MiB payload required");
    const auto pool=single_layer::configure_pool(e,sampling.poll);e.validate();const auto initial=single_layer::capture_verification(e);
    Registration registration(e,str(request,"memory_mode"));const auto measured_source_category=pointer_kind(source->data);
    auto full=[&](){for(const auto &a:actions)e.action(a);};
    const auto warm=steady::warm(sampling,full);const auto before=payload(source,dest);
    std::vector<Sequence> sequences(static_cast<size_t>(sampling.trials)*4);
    std::vector<Copy> copies(sequences.size()*sampling.repeats);
    size_t sequence_index=0,copy_index=0;
    for(int trial=0;trial<sampling.trials;++trial)for(int slot=0;slot<4;++slot){
        auto &s=sequences[sequence_index++];s.trial=trial;s.slot=slot;s.condition=request.at("orders").at(trial).at(slot).get<int>();
        const DWORD idle=request.at("idle_ms").at(s.condition).get<DWORD>();
        e.reset();full(); // Same exact preparation immediately before every idle condition.
        s.idle_begin=steady::tick();if(idle>0)timer.wait_ns(static_cast<int64_t>(idle)*1000000);s.idle_end=steady::tick();
        s.burst_begin=steady::tick();
        for(int ordinal=0;ordinal<sampling.repeats;++ordinal){auto &c=copies[copy_index++];c.begin=steady::tick();e.action(target);c.end=steady::tick();}
        s.burst_end=steady::tick();
    }
    const auto after=payload(source,dest);require(before.at("source_sha256")==after.at("source_sha256"),"owned source bytes changed");
    const auto final=single_layer::capture_verification(e);const auto registration_proof=registration.finish();
    json rows=json::array();copy_index=0;
    for(const auto &s:sequences){json samples=json::array();int64_t last=s.idle_end;
        for(int ordinal=0;ordinal<sampling.repeats;++ordinal){const auto &c=copies[copy_index++];
            samples.push_back(json{{"ordinal",ordinal},{"begin_tick",c.begin},{"end_tick",c.end},{"wall_ns",steady::ns(c.begin,c.end)},
                {"gap_since_previous_completion_ns",steady::ns(last,c.begin)}});last=c.end;}
        rows.push_back(json{{"trial",s.trial},{"condition_slot",s.slot},{"condition_index",s.condition},
            {"requested_idle_ms",request.at("idle_ms").at(s.condition)},{"idle_begin_tick",s.idle_begin},{"idle_end_tick",s.idle_end},
            {"actual_idle_ns",steady::ns(s.idle_begin,s.idle_end)},
            {"burst_begin_tick",s.burst_begin},{"burst_end_tick",s.burst_end},{"burst_wall_ns",steady::ns(s.burst_begin,s.burst_end)},
            {"copies",samples}});
    }
    return json{{"program_sha256",e.program.at("program_sha256")},{"target_action_index",index},{"target_action",target},
        {"target_bytes",2097152},{"source_pointer_during_measurement",measured_source_category},{"payload_before",before},{"payload_after",after},
        {"sequences",rows},{"idle_controller",dynamic_control::OwnedHighResolutionTimer::describe()},{"warmup",warm},{"threadpool",pool},{"arena_registration",registration_proof},
        {"verification_native",final},{"native_repeat_consistency",state_response::compare_ports(initial.at("ports"),final.at("ports"))},
        {"preparation_contract",json{{"reset_before_condition",true},{"full_untimed_prime_passes",1},{"burst_copies",16},
            {"condition_order_explicit",true},{"idle_excluded_from_copy_timer",true}}},
        {"instrumentation",json{{"cupti",false},{"per_copy_qpc",true},{"logging_inside_timer",false},
            {"copy_boundary","unchanged inclusive completed Executor.action"},{"physical_payload_only",false}}},
        {"future_target_state_used",false},{"performance_parameters_fitted",false},{"hardware_accuracy_qualified",false}};
}
}

int main(int argc,char **argv){try{
    if(argc==2&&std::string(argv[1])=="--self-test"){
        dynamic_idle::validate_orders(json::array({0,1,5,20}),json::array({json::array({0,1,2,3})}),1);
        bool rejected=false;try{dynamic_idle::validate_orders(json::array({0,1,5,20}),json::array({json::array({0,1,1,3})}),1);}catch(const std::exception &){rejected=true;}
        require(rejected,"invalid order accepted");auto result=steady::self_test();result["balanced_condition_validation_passed"]=true;
        result["high_resolution_due_parser_passed"]=dynamic_control::OwnedHighResolutionTimer::relative_due_100ns(1)==-1&&dynamic_control::OwnedHighResolutionTimer::relative_due_100ns(1000000)==-10000;
        result["source_sha256"]=DYNAMIC_IDLE_SOURCE_SHA256;result["native_hardware_executed"]=false;std::cout<<result.dump()<<'\n';return 0;
    }
    require(argc==4,"usage: dynamic-idle-probe-v2 PROGRAM REQUEST OUTPUT");require(!state_response::exists(argv[3]),"idle output already exists");
    _putenv_s("NVIDIA_TF32_OVERRIDE","0");_putenv_s("GGML_CUDA_DISABLE_GRAPHS","1");
    std::ifstream pf(std::filesystem::u8path(argv[1])),rf(std::filesystem::u8path(argv[2]));require(pf.good()&&rf.good(),"cannot open explicit program/request");json program,request;pf>>program;rf>>request;
    require(request.at("schema")=="dynamic-idle-request-1"&&str(request,"source_sha256")==DYNAMIC_IDLE_SOURCE_SHA256,"request/source identity differs");
    steady::Request sampling(request.at("sampling"));require(sampling.main_mask==4,"explicit main thread mask4 required");
    const auto process=dynamic_idle::apply_affinity(request);steady::OwnThreadAffinity thread(sampling.main_mask);json result;
    {Executor executor(std::move(program));result=dynamic_idle::run(executor,sampling,request);}
    result.update(json{{"schema","dynamic-idle-native-1"},{"request",request},{"process_id",single_layer::pid()},
        {"source_sha256",DYNAMIC_IDLE_SOURCE_SHA256},{"caller_source_sha256_match",true},
        {"clock",json{{"clock_id","windows_qpc"},{"frequency_hz",steady::frequency()}}},
        {"process_affinity",dynamic_idle::finish_affinity(process)},{"main_thread_affinity",thread.finish()}});
    state_response::write_new(argv[3],result);std::cout<<json{{"status","ok"},{"process_id",single_layer::pid()}}.dump()<<'\n';return 0;
}catch(const std::exception &error){std::cerr<<"dynamic-idle-probe-v2: "<<error.what()<<'\n';return 2;}}
