// Additive harness: the old Executor, mathematics, sources and binaries are unchanged.
#define main historical_steady_entry_not_called
#include "../../native/steady_costs/probe.cpp"
#undef main

namespace repair {
static DWORD_PTR expected_mask(const json &request) {
    return static_cast<DWORD_PTR>(steady::positive(request.at("expected_process_mask"),"expected process affinity mask",INT64_MAX));
}

static json apply_process_mask(const json &request) {
    if(!request.contains("expected_process_mask")||request.at("expected_process_mask").is_null())
        return json{{"status","unknown"},{"requested_mask",nullptr},{"enforced",false},
                    {"scope","not requested; no process mask inferred"}};
    const auto requested=expected_mask(request);DWORD_PTR before=0,system=0,observed=0,observed_system=0;
    require(GetProcessAffinityMask(GetCurrentProcess(),&before,&system)!=0,"cannot observe own initial process affinity");
    require((requested&system)==requested,"expected process mask outside observed system affinity");
    require(SetProcessAffinityMask(GetCurrentProcess(),requested)!=0,"cannot set explicit owned process affinity before backend initialization");
    require(GetProcessAffinityMask(GetCurrentProcess(),&observed,&observed_system)!=0&&observed==requested,
            "explicit owned process affinity readback differs");
    return json{{"status","observed"},{"requested_mask",static_cast<uint64_t>(requested)},
        {"initial_mask",static_cast<uint64_t>(before)},{"observed_mask",static_cast<uint64_t>(observed)},
        {"system_mask",static_cast<uint64_t>(observed_system)},{"enforced",true},
        {"applied_before_backend_initialization",true},
        {"scope","current owned process only; explicit request, no global settings changed"}};
}

static json finish_process_mask(json receipt) {
    if(receipt.at("status")=="unknown")return receipt;
    DWORD_PTR final_mask=0,system=0;
    require(GetProcessAffinityMask(GetCurrentProcess(),&final_mask,&system)!=0&&
            static_cast<uint64_t>(final_mask)==receipt.at("requested_mask").get<uint64_t>(),
            "owned process affinity changed during measurement");
    receipt["final_mask"]=static_cast<uint64_t>(final_mask);return receipt;
}

static json pointer_kind(void *pointer) {
    cudaPointerAttributes attrs{};const auto status=cudaPointerGetAttributes(&attrs,pointer);
    if(status==cudaErrorInvalidValue) {
        cudaGetLastError();
        return json{{"status","observed"},{"query","cudaPointerGetAttributes"},
                    {"registered_host",false},{"unregistered_pageable",true},
                    {"runtime_error_code",static_cast<int>(status)}};
    }
    require(status==cudaSuccess,"explicit host pointer-attribute query failed");
    return json{{"status","observed"},{"query","cudaPointerGetAttributes"},
                {"cuda_memory_type",static_cast<int>(attrs.type)},
                {"registered_host",attrs.type==cudaMemoryTypeHost},
                {"unregistered_pageable",attrs.type==cudaMemoryTypeUnregistered}};
}

static uint64_t filetime(FILETIME value) {
    return (uint64_t(value.dwHighDateTime)<<32)|value.dwLowDateTime;
}

static json preblock_state(bool gpu) {
    // A precondition observation, never inferred from the following target duration.
    const auto at=steady::tick();MEMORYSTATUSEX memory{};memory.dwLength=sizeof(memory);
    FILETIME idle{},kernel{},user{};
    const bool memory_ok=GlobalMemoryStatusEx(&memory)!=0;
    const bool times_ok=GetSystemTimes(&idle,&kernel,&user)!=0;
    json result{{"sample_tick",at},{"outside_timed_interval",true},
        {"sampled_before_untimed_prime",true},{"derived_from_target_duration",false},
        {"cpu_utilization",nullptr},{"gpu_utilization",nullptr},{"gpu_clock_mhz",nullptr},
        {"utilization_status","unknown; no independent utilization interval sampled"},
        {"cpu_cumulative_times_100ns",times_ok?json{{"idle",filetime(idle)},{"kernel_including_idle",filetime(kernel)},{"user",filetime(user)}}:json(nullptr)},
        {"system_memory",memory_ok?json{{"total_bytes",memory.ullTotalPhys},{"available_bytes",memory.ullAvailPhys},{"load_percent",memory.dwMemoryLoad}}:json(nullptr)}};
    if(gpu) {
        size_t free_bytes=0,total_bytes=0;const auto status=cudaMemGetInfo(&free_bytes,&total_bytes);
        result["cuda_memory"]=status==cudaSuccess?json{{"status","observed"},{"free_bytes",free_bytes},{"total_bytes",total_bytes}}:
            json{{"status","unknown"},{"runtime_error_code",static_cast<int>(status)}};
        if(status!=cudaSuccess)cudaGetLastError();
    } else result["cuda_memory"]=nullptr;
    return result;
}

static json dense_contract(const steady::Request &r) {
    return json{{"reset_before_batch",true},{"untimed_complete_prime_passes",1},
        {"timed_replays_per_batch",r.repeats},{"fixed_input_replay",true},
        {"warmup_iterations_min",r.warmup},{"warmup_ms_min",r.min_ms}};
}

static void check_order(const std::string &order) {
    require(order=="ABBA"||order=="BAAB","explicit ABBA or BAAB order required");
}

static json dense(Executor &executor,const steady::Request &r,bool observer,const std::string &order) {
    require(r.mode=="full"&&r.target<0&&r.repeats==8,"Dense repair uses explicit full mode and eight timed replays");
    check_order(order);const auto &actions=executor.program.at("actions");
    require(!actions.empty()&&uint64_t(actions.size())*r.repeats*r.trials*(observer?4:1)<=2000000,
            "Dense repair action envelope exceeded");
    const auto pool=single_layer::configure_pool(executor,r.poll);
    executor.validate();const auto initial=single_layer::capture_verification(executor);
    auto full=[&](){for(const auto &a:actions)executor.action(a);};
    const auto warm=steady::warm(r,full);
    const int blocks=observer?4:1;std::vector<steady::Frame> frames(r.trials*blocks);
    std::vector<json> states(r.trials*blocks);
    for(int t=0;t<r.trials;++t)for(int b=0;b<blocks;++b)
        if(observer&&order[b]=='B')frames[t*blocks+b].action_ticks.resize(actions.size());
    for(int t=0;t<r.trials;++t)for(int b=0;b<blocks;++b) {
        auto &frame=frames[t*blocks+b];states[t*blocks+b]=preblock_state(executor.gpu!=nullptr);
        executor.reset();full(); // Identical preparation for primary and every observer A/B block.
        frame.begin=steady::tick();
        for(int repeat=0;repeat<r.repeats;++repeat) {
            if(observer&&order[b]=='B')for(size_t i=0;i<actions.size();++i) {
                const auto first=steady::tick();executor.action(actions.at(i));frame.action_ticks[i]+=steady::tick()-first;
            } else full();
        }
        frame.end=steady::tick();
    }
    json rows=json::array();for(int t=0;t<r.trials;++t)for(int b=0;b<blocks;++b) {
        auto row=steady::frame_json(frames[t*blocks+b],t,r);
        row.update(json{{"block",b},{"actual_stamped",observer&&order[b]=='B'},
            {"preblock_state",states[t*blocks+b]},{"preparation_contract",dense_contract(r)}});rows.push_back(row);
    }
    const auto final=single_layer::capture_verification(executor);
    return json{{"program_sha256",executor.program.at("program_sha256")},{"program_format",executor.program.at("format_version")},
        {"trials",rows},{"groups",json::array()},{"actions",actions},{"warmup",warm},{"threadpool",pool},
        {"preparation_contract",dense_contract(r)},{"observer_order",observer?json(order):json(nullptr)},
        {"verification_native",final},{"native_repeat_consistency",state_response::compare_ports(initial.at("ports"),final.at("ports"))},
        {"unchanged_Executor_action_used",true},{"whole_program_time_used_as_parameter",false},
        {"timing_scope","unchanged fixed-input Executor.action completion; reset, one full prime, state queries and verification excluded"}};
}

static json hot_actions(Executor &executor,const steady::Request &r,const json &indices) {
    require(indices.is_array()&&!indices.empty(),"explicit action-price indices required");
    const auto &actions=executor.program.at("actions");const auto pool=single_layer::configure_pool(executor,r.poll);
    executor.validate();const auto initial=single_layer::capture_verification(executor);
    auto full=[&](){for(const auto &a:actions)executor.action(a);};const auto warm=steady::warm(r,full);
    json groups=json::array();for(const auto &value:indices) {
        const auto index=static_cast<size_t>(state_response::integer(value,"action-price index"));
        require(index<actions.size(),"action-price index outside program");const auto &action=actions.at(index);
        const auto action_warm=steady::warm(r,[&](){executor.action(action);});json rows=json::array();
        for(int trial=0;trial<r.trials;++trial) {
            const auto state=preblock_state(executor.gpu!=nullptr);
            const auto begin=steady::tick();for(int repeat=0;repeat<r.repeats;++repeat)executor.action(action);const auto end=steady::tick();
            rows.push_back(json{{"trial",trial},{"begin_tick",begin},{"end_tick",end},{"wall_ns",steady::ns(begin,end)},
                {"repeats",r.repeats},{"preblock_state",state}});
        }
        groups.push_back(json{{"action_index",index},{"action",action},{"trials",rows},{"warmup",action_warm}});
    }
    const auto final=single_layer::capture_verification(executor);
    return json{{"program_sha256",executor.program.at("program_sha256")},{"groups",groups},{"trials",json::array()},
        {"warmup",warm},{"threadpool",pool},{"verification_native",final},
        {"native_repeat_consistency",state_response::compare_ports(initial.at("ports"),final.at("ports"))},
        {"preparation_contract",json{{"scope","isolated hot same-prepared-input action batches; not Dense reset+prime"},
            {"reset_before_batch",false},{"untimed_complete_prime_passes",0},{"timed_replays_per_batch",r.repeats},
            {"warmup_iterations_min",r.warmup},{"warmup_ms_min",r.min_ms}}},
        {"whole_program_time_used_as_parameter",false},{"unchanged_Executor_action_used",true}};
}

struct Registration {
    void *pointer;size_t bytes;bool active=false;json transitions=json::array();
    Registration(void *p,size_t n):pointer(p),bytes(n) {
        require(pointer_kind(pointer).at("unregistered_pageable")==true,"owned destination must initially be observed pageable memory");
    }
    ~Registration(){if(active)cudaHostUnregister(pointer);}
    void set(const std::string &mode) {
        require(mode=="pageable"||mode=="registered","explicit pageable or registered host memory mode required");
        const auto before=pointer_kind(pointer);std::string operation="none";
        if(mode=="registered"&&!active) {
            require(cudaHostRegister(pointer,bytes,cudaHostRegisterDefault)==cudaSuccess,"explicit cudaHostRegister failed; no fallback");
            active=true;operation="cudaHostRegister";
        } else if(mode=="pageable"&&active) {
            require(cudaHostUnregister(pointer)==cudaSuccess,"owned cudaHostUnregister failed");active=false;operation="cudaHostUnregister";
        }
        const auto after=pointer_kind(pointer);
        require(after.at(mode=="registered"?"registered_host":"unregistered_pageable")==true,"actual host registration differs from requested mode");
        transitions.push_back(json{{"requested_mode",mode},{"operation",operation},{"before",before},{"after",after},{"outside_trial_timers",true}});
    }
    void finish(){set("pageable");require(!active,"owned host registration not released");}
};

static json transfer_visit(steady::Component &c,const steady::Request &r,const std::string &mode,
                           int visit,const std::string &order,Registration &registration) {
    check_order(order);if(c.gpu)ggml_backend_synchronize(c.gpu);ggml_backend_synchronize(c.cpu);
    registration.set(mode);const auto observed=pointer_kind(c.output->data);
    c.action();const auto initial=c.validate_output();
    const auto warm=steady::warm(r,[&](){c.action();});json rows=json::array();
    for(int trial=0;trial<r.trials;++trial)for(int block=0;block<4;++block) {
        const auto state=preblock_state(true);c.action(); // Equal one-copy prime outside every timer.
        int64_t copy=0,gpu=0,cpu=0;const auto begin=steady::tick();
        for(int repeat=0;repeat<r.repeats;++repeat) {
            if(order[block]=='A')c.action();else {
                const auto first=steady::tick();ggml_backend_tensor_copy(c.left,c.output);const auto submitted=steady::tick();
                ggml_backend_synchronize(c.gpu);const auto gpu_done=steady::tick();
                ggml_backend_synchronize(c.cpu);const auto complete=steady::tick();
                copy+=submitted-first;gpu+=gpu_done-submitted;cpu+=complete-gpu_done;
            }
        }
        const auto end=steady::tick();const bool stamped=order[block]=='B';
        rows.push_back(json{{"trial",trial},{"visit",visit},{"block",block},{"memory_mode",mode},{"actual_stamped",stamped},
            {"begin_tick",begin},{"end_tick",end},{"wall_ns",steady::ns(begin,end)},{"repeats",r.repeats},
            {"copy_call_ns",stamped?json(single_layer::ns(0,copy,steady::frequency())):json(nullptr)},
            {"submit_copy_api_ns",stamped?json(single_layer::ns(0,copy,steady::frequency())):json(nullptr)},
            {"gpu_sync_ns",stamped?json(single_layer::ns(0,gpu,steady::frequency())):json(nullptr)},
            {"cpu_sync_ns",stamped?json(single_layer::ns(0,cpu,steady::frequency())):json(nullptr)},
            {"host_pointer",observed},{"preblock_state",state}});
    }
    const auto final=c.validate_output();require(initial.at("output_sha256")==final.at("output_sha256"),"transfer output changed in memory-mode visit");
    return json{{"visit",visit},{"memory_mode",mode},{"host_pointer",observed},{"warmup",warm},{"trials",rows},
        {"full_byte_verification",final},{"observer_order",order}};
}

static json transfer(steady::Component &c,const steady::Request &r,const json &request,bool diagnostic) {
    require(c.kind=="transfer"&&c.src=="vram"&&c.dst=="dram"&&c.gpu,"repair transfer supports explicit D2H only");
    require(r.target<0&&r.mode=="action","transfer repair requires compact action mode");
    const auto order=str(request,"order");check_order(order);
    int device=-1;require(cudaGetDevice(&device)==cudaSuccess,"cannot observe selected CUDA device");
    require(device==c.spec.at("cuda_device").get<int>(),"observed CUDA device differs from explicit transfer device");
    int supported=0;require(cudaDeviceGetAttribute(&supported,cudaDevAttrHostRegisterSupported,device)==cudaSuccess,"host-register support query failed");
    Registration registration(c.output->data,ggml_nbytes(c.output));json visits=json::array(),rows=json::array();
    const auto memory_order=diagnostic?str(request,"memory_order"):std::string();
    if(diagnostic)check_order(memory_order);
    const int count=diagnostic?4:1;
    for(int visit=0;visit<count;++visit) {
        const auto mode=diagnostic?(memory_order[visit]=='A'?"pageable":"registered"):str(request,"memory_mode");
        require(mode!="registered"||supported!=0,"device does not support explicit host registration; no fallback");
        auto local=transfer_visit(c,r,mode,visit,order,registration);
        for(const auto &row:local.at("trials"))rows.push_back(row);visits.push_back(std::move(local));
    }
    registration.finish();const auto final=c.validate_output();
    return json{{"program_sha256",c.spec.at("program_sha256")},{"visits",visits},{"trials",rows},
        {"component",c.describe()},{"component_output",final},{"host_registration_transitions",registration.transitions},
        {"host_registration_restored",true},{"same_owned_destination_buffer_all_modes",true},
        {"cuda_device",device},{"cuda_host_register_supported",supported!=0},
        {"memory_order",diagnostic?json(memory_order):json(nullptr)},{"observer_order",order},
        {"preparation_contract",json{{"reset_before_batch",false},{"untimed_transfer_prime_passes",1},
            {"timed_replays_per_batch",r.repeats},{"fixed_input_replay",true},
            {"warmup_iterations_min",r.warmup},{"warmup_ms_min",r.min_ms},{"equal_warmup_per_memory_mode_visit",true}}},
        {"measurement_api","ggml_tensor_copy_then_gpu_cpu_synchronize"},
        {"completion_semantic","host_response_including_copy_api_and_gpu_cpu_completion"},
        {"parameter_samples","bare blocks only; stamped phase intervals are diagnostic"},
        {"copy_call_scope","unchanged inclusive GGML copy API response: submission, any staging/internal completion cannot be separated by these wall stamps; submit_copy_api_ns aliases copy_call_ns, not an additive phase"},
        {"physical_PCIe_active_time_identified",false},{"target_duration_used_as_state_input",false}};
}
}

int main(int argc,char **argv) {try {
    if(argc==2&&std::string(argv[1])=="--self-test") {
        auto result=steady::self_test();
        repair::check_order("ABBA");repair::check_order("BAAB");bool rejected=false;
        try {repair::check_order("ABAB");}catch(const std::exception &){rejected=true;}
        require(rejected,"unbalanced order accepted");result["balanced_order_parser_passed"]=true;
        require(repair::expected_mask(json{{"expected_process_mask",4}})==4,"explicit process-mask parser failed");
        require(repair::apply_process_mask(json::object()).at("status")=="unknown","missing process-mask request was inferred");
        result["explicit_and_unknown_process_mask_parser_passed"]=true;
        result["source_sha256"]=REPAIR_SOURCE_SHA256;std::cout<<result.dump()<<'\n';return 0;
    }
    require(argc==4,"usage: accuracy-repair-probe PROGRAM REQUEST OUTPUT");
    require(!state_response::exists(argv[3]),"accuracy repair output exists");
    _putenv_s("NVIDIA_TF32_OVERRIDE","0");_putenv_s("GGML_CUDA_DISABLE_GRAPHS","1");
    std::ifstream pf(std::filesystem::u8path(argv[1])),rf(std::filesystem::u8path(argv[2]));
    require(pf.good()&&rf.good(),"cannot read explicit repair program/request");json program,request;pf>>program;rf>>request;
    require(request.at("schema")=="dac-accuracy-repair-request-1","repair request schema differs");
    require(str(request,"source_sha256")==REPAIR_SOURCE_SHA256,"caller source hash differs from compiled repair source");
    steady::Request sampling(request.at("sampling"));const auto process_affinity=repair::apply_process_mask(request);
    steady::OwnThreadAffinity affinity(sampling.main_mask);
    require(sampling.main_mask==4,"repair manifest fixes main thread mask=4");const auto mode=str(request,"mode");json result;
    if(mode=="transfer-diagnostic"||mode=="transfer-mode") {
        steady::Component component(program,sampling.poll);result=repair::transfer(component,sampling,request,mode=="transfer-diagnostic");
    } else {
        Executor executor(std::move(program));
        if(mode=="hot-actions")result=repair::hot_actions(executor,sampling,request.at("indices"));
        else {require(mode=="dense-primary"||mode=="dense-observer","unknown accuracy repair mode");
            result=repair::dense(executor,sampling,mode=="dense-observer",str(request,"order"));}
    }
    result.update(json{{"schema","dac-accuracy-repair-native-1"},{"request",request},{"process_id",single_layer::pid()},
        {"source_sha256",REPAIR_SOURCE_SHA256},{"caller_source_sha256_match",true},{"main_thread_affinity",affinity.finish()},
        {"process_affinity",repair::finish_process_mask(process_affinity)},
        {"clock",json{{"clock_id","windows_qpc"},{"frequency_hz",steady::frequency()}}},
        {"observed_runtime_flags",json{{"GGML_CUDA_DISABLE_GRAPHS","1"},{"NVIDIA_TF32_OVERRIDE","0"},{"matmul_precision","GGML_PREC_F32"}}},
        {"collected_cpu_cycles",false},{"performance_parameters_fitted",false},{"hardware_accuracy_qualified",false}});
    state_response::write_new(argv[3],result);std::cout<<json{{"status","ok"},{"process_id",single_layer::pid()}}.dump()<<'\n';return 0;
} catch(const std::exception &error){std::cerr<<"accuracy-repair-probe: "<<error.what()<<'\n';return 2;}}
