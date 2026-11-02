// Diagnostic only: the same Executor/PID alternates target-isolated and context.
#include "single_layer_generated.inc"

namespace context_pair {
static void clear_validation_records(Executor &executor) {
    // Executor.validate accumulates these convenience records. A reused
    // executor needs fresh validation records for each block, never timer edits.
    executor.copies=json::array();executor.ports=json::object();executor.copy_equal=true;
}
static json isolated_target(Executor &executor,size_t target,int warmup,int samples,const json &pool) {
    using namespace single_layer;
    const auto &actions=executor.program.at("actions");const auto &selected=actions.at(target);const auto hz=frequency();
    executor.validate();std::vector<Stamp> observations(samples);
    for(int i=0;i<warmup;++i)executor.action(selected);
    Envelope envelope;envelope.first=cpu_snapshot();envelope.begin=tick();
    for(auto &stamp:observations){stamp.begin=tick();executor.action(selected);stamp.end=tick();}
    envelope.end=tick();envelope.last=cpu_snapshot();
    json durations=json::array(),intervals=json::array();
    for(size_t i=0;i<observations.size();++i){const auto &stamp=observations[i];const auto duration=ns(stamp.begin,stamp.end,hz);require(duration>0,"nonpositive isolated PV duration");durations.push_back(duration);intervals.push_back(json{{"sample",i},{"begin_tick",stamp.begin},{"end_tick",stamp.end},{"elapsed_ns",duration}});}
    auto verification=capture_verification(executor);
    return json{{"format_version","single-layer-calibration-native-v1"},{"process_id",pid()},
        {"program_sha256",executor.program.at("program_sha256")},{"boundary_contract",BOUNDARY},
        {"execution_boundary","legacy_double_sync"},{"mode","isolated-components"},{"warmup",warmup},{"samples",samples},
        {"actions",actions},{"target_action_index",target},{"component_subset",true},
        {"component_subset_scope","exactly_one_selected_GPU_attention_pv; not_a_complete_component_profile"},
        {"components",json::array({json{{"action_index",target},{"action",selected},{"samples_ns",durations},{"intervals",intervals},
            {"workspace_bytes",executor.graphs.at(str(selected,"target_id"))->workspace}}})},
        {"iterations",json::array()},{"total_samples_ns",json::array()},
        {"verification_native",verification},{"threadpool",pool},{"cpu_diagnostics",cpu_envelopes({envelope},hz)},
        {"native_clock",json{{"clock_id",
#if defined(_WIN32)
        "windows_qpc"
#else
        "posix_clock_monotonic"
#endif
        },{"frequency_hz",hz},{"cross_process_monotonic_clock",true}}},
        {"execution_contract","prebuilt_synchronized_GGML_actions;reset_excluded;preallocated_trace_buffer_v1"},
        {"timing_scope","same_Executor.action_unchanged; target_repeated_in_isolation; POD_clocks_inside; JSON_outside"},
        {"sampling_envelope_scope","selected_target_sampling_only; validation_and_warmup_excluded"},
        {"stream_active_time_measured",false},{"host_and_device_times_may_be_added",false},
        {"diagnostic_only",true},{"usable_for_parameter_fit",false},{"usable_for_heldout_acceptance",false}};
}
static json run(Executor &executor,const std::string &pair_kind,size_t target,int warmup,int samples,const std::string &order,const json &pool) {
    require(pair_kind=="context_vs_isolated"||pair_kind=="instrumentation","pair kind must be explicit context_vs_isolated or instrumentation");
    require(order=="ABBA"||order=="BAAB","paired order must be explicit ABBA or BAAB");
    const auto &actions=executor.program.at("actions");require(target<actions.size(),"target outside action trace");
    const auto &selected=actions.at(target);require(selected.at("kind")=="COMPUTE"&&selected.at("device")=="gpu","paired target must be GPU COMPUTE");
    require(executor.bindings.at(str(selected,"target_id")).at("kind")=="attention_pv","paired diagnostic supports exact attention_pv target only");
    json blocks=json::array();const auto begin=single_layer::tick();
    for(size_t index=0;index<order.size();++index) {
        const char label=order.at(index);clear_validation_records(executor);const auto block_begin=single_layer::tick();
        json raw;
        if(label=='A'&&pair_kind=="context_vs_isolated")raw=isolated_target(executor,target,warmup,samples,pool);
        else if(label=='A')raw=single_layer::run(executor,"bare-replay",warmup,samples,pool);
        else raw=single_layer::run(executor,"context-trace",warmup,samples,pool);
        const auto block_end=single_layer::tick();raw["diagnostic_only"]=true;raw["usable_for_parameter_fit"]=false;raw["usable_for_heldout_acceptance"]=false;
        blocks.push_back(json{{"block_index",index},{"label",std::string(1,label)},
            {"context",label=='A'&&pair_kind=="context_vs_isolated"?"target_PV_repeated_in_isolation":"whole_real_action_sequence"},
            {"block_begin_tick",block_begin},{"block_end_tick",block_end},
            {"block_wall_ns",single_layer::ns(block_begin,block_end,single_layer::frequency())},{"raw",std::move(raw)}});
    }
    return json{{"format_version","single-layer-context-pair-native-v2"},{"process_id",single_layer::pid()},
        {"program_sha256",executor.program.at("program_sha256")},{"seed",executor.program.at("seed")},
        {"pair_kind",pair_kind},{"A_label",pair_kind=="instrumentation"?"bare-replay":"isolated-components"},{"B_label","context-trace"},
        {"target_action_index",target},{"target_action",selected},{"block_order",order},{"warmup_per_block",warmup},{"samples_per_block",samples},
        {"threadpool",pool},{"blocks",blocks},{"begin_tick",begin},{"end_tick",single_layer::tick()},
        {"same_PID_Executor_program_and_pool_for_all_blocks",true},
        {"native_clock",json{{"clock_id",
#if defined(_WIN32)
        "windows_qpc"
#else
        "posix_clock_monotonic"
#endif
        },{"frequency_hz",single_layer::frequency()},{"cross_process_monotonic_clock",true}}},
        {"diagnostic_only",true},{"usable_for_parameter_fit",false},{"usable_for_heldout_acceptance",false},
        {"counterbalance_scope","caller predeclares ABBA or BAAB; repeat in independent PIDs with rotated order; no data-dependent block ordering"},
        {"block_boundary_scope","validation, warmup, port capture and JSON assembly outside action clocks; each block repeats warmup; exact GPU/OS state restoration not claimed"},
        {"causal_limit","same-PID counterbalancing removes between-PID identity as a required explanation, but uncontrolled time-varying device state and instrumentation remain possible"}};
}
}

int main(int argc,char **argv){try {
    require(argc==9,"usage: context-pair-probe PROGRAM.json context_vs_isolated|instrumentation TARGET_ACTION_INDEX WARMUP SAMPLES default|0 ABBA|BAAB OUTPUT.json");
    const std::string pair_kind=argv[2];require(pair_kind=="context_vs_isolated"||pair_kind=="instrumentation","pair kind must be explicit context_vs_isolated or instrumentation");
    const int target=single_layer::integer(argv[3],0,10000,"target"),warmup=single_layer::integer(argv[4],0,10000,"warmup"),samples=single_layer::integer(argv[5],3,10000,"samples");
    const std::string poll=argv[6],order=argv[7];require(poll=="default"||poll=="0","pool poll must be explicit default or 0");
    require(order=="ABBA"||order=="BAAB","paired order must be explicit ABBA or BAAB");
    std::ifstream existing(argv[8],std::ios::binary);require(!existing.good(),"output already exists");
#if defined(_WIN32)
    _putenv_s("NVIDIA_TF32_OVERRIDE","0");_putenv_s("GGML_CUDA_DISABLE_GRAPHS","1");
#else
    setenv("NVIDIA_TF32_OVERRIDE","0",1);setenv("GGML_CUDA_DISABLE_GRAPHS","1",1);
#endif
    std::ifstream input(argv[1]);require(bool(input),"cannot read program");json program;input>>program;Executor executor(std::move(program));
    const auto pool=single_layer::configure_pool(executor,poll);const auto result=context_pair::run(executor,pair_kind,static_cast<size_t>(target),warmup,samples,order,pool);
    std::ofstream output(argv[8],std::ios::binary);require(bool(output),"cannot create output");output<<result.dump(2)<<'\n';output.close();require(bool(output),"output write failed");
    std::cout<<json{{"status","ok"},{"process_id",result.at("process_id")},{"pair_kind",pair_kind},{"order",order},{"blocks",4},{"diagnostic_only",true}}.dump()<<'\n';return 0;
}catch(const std::exception &error){std::cerr<<"context-pair-probe: "<<error.what()<<'\n';return 2;}}
