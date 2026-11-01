// Additive diagnostic only. Historical generated Executor and native action
// helpers remain byte-pinned, and the old binary is never rebuilt or replaced.
#define main steady_historical_entry_not_called
#include "../../native/steady_costs/probe.cpp"
#undef main

static json paired_observer(Executor &executor, const steady::Request &request,
                            const std::string &order, bool sham) {
    require(order=="ABBA"||order=="BAAB", "explicit balanced pair order required");
    const auto &actions=executor.program.at("actions");
    require(!actions.empty()&&actions.size()*request.repeats*request.trials*4<=2000000,
            "paired action envelope exceeded");
    const auto pool=single_layer::configure_pool(executor, request.poll);
    executor.validate();
    const auto initial=single_layer::capture_verification(executor);
    auto execute=[&](){for(const auto &action:actions)executor.action(action);};
    const auto warmed=steady::warm(request, execute);
    // All allocations precede measured execution. Each unit is an adjacent,
    // balanced four-block contrast within one initialized native process.
    std::vector<steady::Frame> frames(request.trials*4);
    for(int trial=0;trial<request.trials;++trial)for(int block=0;block<4;++block)
        if(order[block]=='B'&&!sham)frames[trial*4+block].action_ticks.resize(actions.size());
    json rows=json::array();
    for(int trial=0;trial<request.trials;++trial) {
        for(int block=0;block<4;++block) {
            executor.reset();
            // Identical immediate preconditioning for both modes, untimed.
            execute();
            auto &frame=frames[trial*4+block];
            const bool stamped=order[block]=='B'&&!sham;
            frame.begin=steady::tick();
            for(int repeat=0;repeat<request.repeats;++repeat) {
                if(stamped)for(size_t index=0;index<actions.size();++index) {
                    const auto first=steady::tick();executor.action(actions.at(index));
                    frame.action_ticks[index]+=steady::tick()-first;
                } else execute();
            }
            frame.end=steady::tick();
        }
    }
    // Serialize and verify only after every timed block has ended.
    for(int trial=0;trial<request.trials;++trial)for(int block=0;block<4;++block) {
        auto row=steady::frame_json(frames[trial*4+block],trial,request);
        row.update(json{{"block",block},{"assigned_label",std::string(1,order[block])},
                        {"actual_stamped",order[block]=='B'&&!sham}});
        rows.push_back(row);
    }
    const auto final=single_layer::capture_verification(executor);
    return json{{"program_sha256",executor.program.at("program_sha256")},
                {"schema","dac-followup-paired-native-1"}, {"program_format",executor.program.at("format_version")},
                {"trials",rows},{"warmup",warmed},{"threadpool",pool},
                {"verification_native",final},
                {"native_repeat_consistency",state_response::compare_ports(initial.at("ports"),final.at("ports"))},
                {"order",order},{"sham_bare_bare",sham},
                {"timing_scope","adjacent within-process ABBA/BAAB; one untimed reset+plain preconditioning per block; unchanged inclusive Executor.action"},
                {"changed_context_not_used_for_old_prediction_qualification",true},
                {"same_process_contrasts_remove_fresh_process_confound",true},
                {"drift_fully_removed",false}};
}

int main(int argc,char **argv) {try {
    require(argc==4,"usage: paired-probe PROGRAM REQUEST OUTPUT");
    require(!state_response::exists(argv[3]),"paired output already exists");
    _putenv_s("NVIDIA_TF32_OVERRIDE","0");_putenv_s("GGML_CUDA_DISABLE_GRAPHS","1");
    std::ifstream program_file(std::filesystem::u8path(argv[1])),request_file(std::filesystem::u8path(argv[2]));
    require(program_file.good()&&request_file.good(),"cannot read explicit paired program/request");
    json program,request_json;program_file>>program;request_file>>request_json;
    require(request_json.at("schema")=="dac-followup-paired-request-1","paired request schema differs");
    steady::Request request(request_json.at("sampling"));
    require(request.mode=="full"&&request.instrumentation=="bare","paired request requires explicit full bare base");
    steady::OwnThreadAffinity affinity(request.main_mask);
    Executor executor(std::move(program));
    auto result=paired_observer(executor,request,str(request_json,"order"),request_json.at("sham_bare_bare").get<bool>());
    result.update(json{{"request",request_json},{"process_id",single_layer::pid()},
                {"main_thread_affinity",affinity.finish()},
                {"clock",json{{"clock_id","windows_qpc"},{"frequency_hz",steady::frequency()}}},
                {"collected_cpu_cycles",false},{"per_action_CPU_counter_queries",false},
                {"performance_parameters_fitted",false},{"hardware_accuracy_qualified",false}});
    state_response::write_new(argv[3],result);std::cout<<json{{"status","ok"},{"process_id",single_layer::pid()}}.dump()<<'\n';return 0;
}catch(const std::exception &error){std::cerr<<"paired-probe: "<<error.what()<<'\n';return 2;}}
