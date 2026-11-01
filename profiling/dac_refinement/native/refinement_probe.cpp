// Additive executable. The original Executor, action helpers and binaries stay unchanged.
#define main historical_steady_entry_not_called
#include "../../native/steady_costs/probe.cpp"
#undef main

static json pointer_kind(void *pointer) {
    cudaPointerAttributes attrs{};
    const auto status=cudaPointerGetAttributes(&attrs,pointer);
    if(status==cudaErrorInvalidValue) {
        cudaGetLastError();
        return json{{"query","cudaPointerGetAttributes"},{"status","unregistered_pageable"},
                    {"runtime_error_code",static_cast<int>(status)}};
    }
    require(status==cudaSuccess,"host memory CUDA pointer-attribute query failed");
    return json{{"query","cudaPointerGetAttributes"},{"status","observed"},
                {"cuda_memory_type",static_cast<int>(attrs.type)},
                {"registered_host",attrs.type==cudaMemoryTypeHost},
                {"unregistered_pageable",attrs.type==cudaMemoryTypeUnregistered}};
}

static json matched_actions(Executor &executor,const steady::Request &request,
                            const json &indices) {
    const auto &actions=executor.program.at("actions");
    require(indices.is_array()&&!indices.empty(),"explicit calibration action indices required");
    const auto pool=single_layer::configure_pool(executor,request.poll);
    executor.validate();const auto initial=single_layer::capture_verification(executor);
    auto full=[&](){for(const auto &a:actions)executor.action(a);};
    const auto warmed=steady::warm(request,full);
    json groups=json::array();
    for(const auto &value:indices) {
        const auto index=static_cast<size_t>(state_response::integer(value,"calibration index"));
        require(index<actions.size(),"calibration action index outside program");
        const auto &action=actions.at(index);
        const bool physical=action.at("kind")=="COMPUTE"||action.at("kind")=="COPY";
        auto condition=[&](){
            // Match a cyclic fixed-input working set without reading a total target time.
            executor.reset();full();
            for(size_t j=0;j<index;++j)executor.action(actions.at(j));
        };
        const auto hot_warm=steady::warm(request,[&](){executor.action(action);});
        json hot_rows=json::array();
        for(int trial=0;trial<request.trials;++trial) {
            const auto first=steady::tick();
            for(int repeat=0;repeat<request.repeats;++repeat)executor.action(action);
            const auto last=steady::tick();
            hot_rows.push_back(json{{"trial",trial},{"begin_tick",first},{"end_tick",last},
                {"wall_ns",steady::ns(first,last)},{"repeats",request.repeats}});
        }
        const auto warm=steady::warm(request,[&](){if(physical)condition();executor.action(action);});
        json rows=json::array();
        for(int trial=0;trial<request.trials;++trial) {
            json segments=json::array();int64_t ticks=0;
            if(physical) {
                for(int repeat=0;repeat<request.repeats;++repeat) {
                    condition();
                    const auto first=steady::tick();executor.action(action);const auto last=steady::tick();
                    ticks+=last-first;segments.push_back(json{{"begin_tick",first},{"end_tick",last}});
                }
            } else {
                const auto first=steady::tick();
                for(int repeat=0;repeat<request.repeats;++repeat)executor.action(action);
                const auto last=steady::tick();ticks=last-first;
                segments.push_back(json{{"begin_tick",first},{"end_tick",last}});
            }
            rows.push_back(json{{"trial",trial},{"wall_ns",single_layer::ns(0,ticks,steady::frequency())},
                    {"repeats",request.repeats},{"segments",segments}});
        }
        groups.push_back(json{{"action_index",index},{"action",action},{"trials",rows},{"warmup",warm},
            {"hot_trials",hot_rows},{"hot_warmup",hot_warm},
            {"physical_preconditioning",physical},
            {"scope",physical?"target-only QPC intervals after an untimed full working-set sweep and exact prefix":"hot logical action batch; no physical service"}});
    }
    const auto final=single_layer::capture_verification(executor);
    return json{{"program_sha256",executor.program.at("program_sha256")},
        {"groups",groups},{"trials",json::array()},{"warmup",warmed},{"threadpool",pool},
        {"verification_native",final},
        {"native_repeat_consistency",state_response::compare_ports(initial.at("ports"),final.at("ports"))},
        {"all_full_sweeps_and_prefixes_untimed",true},{"whole_program_time_used_as_parameter",false},
        {"unchanged_Executor_action_used",true}};
}

static json paired_full(Executor &executor,const steady::Request &request,const std::string &order) {
    require(order=="ABBA"||order=="BAAB","balanced observer order required");
    const auto &actions=executor.program.at("actions");
    const auto pool=single_layer::configure_pool(executor,request.poll);
    executor.validate();const auto initial=single_layer::capture_verification(executor);
    auto full=[&](){for(const auto &a:actions)executor.action(a);};
    const auto warm=steady::warm(request,full);
    std::vector<steady::Frame> frames(request.trials*4);
    for(int t=0;t<request.trials;++t)for(int b=0;b<4;++b)
        if(order[b]=='B')frames[t*4+b].action_ticks.resize(actions.size());
    for(int t=0;t<request.trials;++t)for(int b=0;b<4;++b) {
        executor.reset();full();auto &f=frames[t*4+b];f.begin=steady::tick();
        for(int repeat=0;repeat<request.repeats;++repeat) {
            if(order[b]=='B')for(size_t i=0;i<actions.size();++i) {
                const auto first=steady::tick();executor.action(actions.at(i));f.action_ticks[i]+=steady::tick()-first;
            } else full();
        }
        f.end=steady::tick();
    }
    json rows=json::array();
    for(int t=0;t<request.trials;++t)for(int b=0;b<4;++b) {
        auto row=steady::frame_json(frames[t*4+b],t,request);
        row.update(json{{"block",b},{"actual_stamped",order[b]=='B'}});rows.push_back(row);
    }
    const auto final=single_layer::capture_verification(executor);
    return json{{"program_sha256",executor.program.at("program_sha256")},{"trials",rows},
        {"groups",json::array()},{"warmup",warm},{"threadpool",pool},{"verification_native",final},
        {"native_repeat_consistency",state_response::compare_ports(initial.at("ports"),final.at("ports"))},
        {"observer_order",order},{"whole_program_time_used_as_parameter",false}};
}

static json transfer_phases(steady::Component &component,const steady::Request &request,const std::string &order) {
    require(component.kind=="transfer","transfer phase probe requires actual transfer component");
    require(order=="ABBA"||order=="BAAB","balanced transfer phase order required");
    auto memory=pointer_kind(component.src=="dram"?component.left->data:component.output->data);
    component.action();const auto initial=component.validate_output();
    const auto warm=steady::warm(request,[&](){component.action();});
    json rows=json::array();
    for(int trial=0;trial<request.trials;++trial)for(int block=0;block<4;++block) {
        component.action(); // matched immediate preconditioning outside all timers
        int64_t copy=0,gpu_sync=0,cpu_sync=0;
        const auto begin=steady::tick();
        for(int repeat=0;repeat<request.repeats;++repeat) {
            if(order[block]=='A')component.action();
            else {
                auto first=steady::tick();ggml_backend_tensor_copy(component.left,component.output);auto second=steady::tick();
                if(component.gpu)ggml_backend_synchronize(component.gpu);auto third=steady::tick();
                ggml_backend_synchronize(component.cpu);auto last=steady::tick();
                copy+=second-first;gpu_sync+=third-second;cpu_sync+=last-third;
            }
        }
        const auto end=steady::tick();
        rows.push_back(json{{"trial",trial},{"block",block},{"actual_stamped",order[block]=='B'},
            {"begin_tick",begin},{"end_tick",end},{"wall_ns",steady::ns(begin,end)},{"repeats",request.repeats},
            {"copy_call_ns",order[block]=='B'?json(single_layer::ns(0,copy,steady::frequency())):json(nullptr)},
            {"gpu_sync_ns",order[block]=='B'?json(single_layer::ns(0,gpu_sync,steady::frequency())):json(nullptr)},
            {"cpu_sync_ns",order[block]=='B'?json(single_layer::ns(0,cpu_sync,steady::frequency())):json(nullptr)}});
    }
    auto final=component.validate_output();
    require(initial.at("output_sha256")==final.at("output_sha256"),"transfer output changed");
    return json{{"program_sha256",component.spec.at("program_sha256")},{"warmup",warm},{"trials",rows},
        {"component_output",final},{"host_pointer",memory},{"component",component.describe()},
        {"parameter_samples","bare blocks only; stamped physical phase observations are diagnostic"},
        {"copy_call_scope","GGML API response includes any internal staging/synchronization; not isolated PCIe active time"}};
}

int main(int argc,char **argv) {try {
    require(argc==4,"usage: refinement-probe PROGRAM REQUEST OUTPUT");
    require(!state_response::exists(argv[3]),"refinement output exists");
    _putenv_s("NVIDIA_TF32_OVERRIDE","0");_putenv_s("GGML_CUDA_DISABLE_GRAPHS","1");
    std::ifstream pf(std::filesystem::u8path(argv[1])),rf(std::filesystem::u8path(argv[2]));
    require(pf.good()&&rf.good(),"cannot read explicit refinement inputs");json program,request;pf>>program;rf>>request;
    require(request.at("schema")=="dac-refinement-request-1","refinement request schema differs");
    steady::Request sampling(request.at("sampling"));steady::OwnThreadAffinity affinity(sampling.main_mask);
    json result;const auto mode=str(request,"mode");
    if(mode=="transfer-phases") {
        steady::Component component(program,sampling.poll);result=transfer_phases(component,sampling,str(request,"order"));
    } else {
        Executor executor(std::move(program));
        if(mode=="matched-actions")result=matched_actions(executor,sampling,request.at("indices"));
        else if(mode=="observer")result=paired_full(executor,sampling,str(request,"order"));
        else {require(mode=="steady","unknown refinement mode");result=steady::run_full(executor,sampling);}
    }
    result.update(json{{"schema","dac-refinement-native-1"},{"request",request},{"process_id",single_layer::pid()},
        {"main_thread_affinity",affinity.finish()},
        {"clock",json{{"clock_id","windows_qpc"},{"frequency_hz",steady::frequency()}}},
        {"observed_runtime_flags",json{{"GGML_CUDA_DISABLE_GRAPHS","1"},{"NVIDIA_TF32_OVERRIDE","0"},{"matmul_precision","GGML_PREC_F32"}}},
        {"collected_cpu_cycles",false},{"per_action_CPU_counter_queries",false},{"performance_parameters_fitted",false},
        {"hardware_accuracy_qualified",false}});
    state_response::write_new(argv[3],result);std::cout<<json{{"status","ok"},{"process_id",single_layer::pid()}}.dump()<<'\n';return 0;
}catch(const std::exception &e){std::cerr<<"refinement-probe: "<<e.what()<<'\n';return 2;}}
