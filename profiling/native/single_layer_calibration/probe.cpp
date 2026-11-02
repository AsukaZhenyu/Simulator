// Exact historical execution, with POD-only clocks and deferred serialization.
#include "highlevel_generated.inc"
#include <cstdint>
#include <limits>
#if defined(_WIN32)
#include <windows.h>
#else
#include <time.h>
#include <unistd.h>
#endif

namespace single_layer {
static int64_t tick() {
#if defined(_WIN32)
    LARGE_INTEGER value; require(QueryPerformanceCounter(&value)!=0,"QPC failed");return value.QuadPart;
#else
    timespec value;require(clock_gettime(CLOCK_MONOTONIC,&value)==0,"clock_gettime failed");
    return int64_t(value.tv_sec)*1000000000LL+value.tv_nsec;
#endif
}
static int64_t frequency() {
#if defined(_WIN32)
    LARGE_INTEGER value;require(QueryPerformanceFrequency(&value)!=0&&value.QuadPart>0,"QPC frequency failed");return value.QuadPart;
#else
    return 1000000000LL;
#endif
}
static uint64_t pid() {
#if defined(_WIN32)
    return GetCurrentProcessId();
#else
    return getpid();
#endif
}
static int64_t ns(int64_t begin,int64_t end,int64_t hz) {
    require(end>=begin,"monotonic clock moved backwards");
    return static_cast<int64_t>(std::llround(static_cast<long double>(end-begin)*1000000000.0L/hz));
}
static int integer(const char *raw,int minimum,int maximum,const char *name) {
    const std::string value(raw);require(!value.empty()&&value.size()<=8&&std::all_of(value.begin(),value.end(),[](char c){return c>='0'&&c<='9';}),std::string(name)+" must be an explicit decimal integer");
    const int parsed=std::stoi(value);require(parsed>=minimum&&parsed<=maximum,std::string(name)+" outside declared envelope");return parsed;
}
static json pool_description(const ggml_threadpool_params &params) {
    json mask=json::array();for(int i=0;i<GGML_MAX_N_THREADS;++i)if(params.cpumask[i])mask.push_back(i);
    return json{{"n_threads",params.n_threads},{"poll",params.poll},{"priority",static_cast<int>(params.prio)},
        {"strict_cpu",params.strict_cpu},{"paused",params.paused},{"cpu_mask",mask}};
}
static json configure_pool(Executor &executor,const std::string &requested) {
    require(requested=="default"||requested=="0","pool poll must be explicit default or 0");
    const int threads=executor.program.at("physical").at("threads");auto params=ggml_threadpool_params_default(threads);
    const auto original=pool_description(params);
    if(requested=="0") {
        ggml_backend_synchronize(executor.cpu);if(executor.gpu)ggml_backend_synchronize(executor.gpu);
        params.poll=0;auto *replacement=ggml_threadpool_new(&params);require(replacement,"replacement persistent threadpool failed");
        auto *previous=executor.pool;ggml_backend_cpu_set_threadpool(executor.cpu,replacement);executor.pool=replacement;
        ggml_threadpool_free(previous);
    }
    return json{{"requested_poll",requested},{"original_default_params",original},{"effective_params",pool_description(params)},
        {"configured_n_threads",threads},{"observed_n_threads",nullptr},
        {"observed_n_threads_unknown_reason","supplied GGML header declares getter but pinned source/runtime provides no implementation or export"},
        {"parameter_observation_scope","public default parameter constructor plus explicitly applied poll override; no private pool-state inspection"},
        {"replacement_before_validation_and_timing",requested=="0"}};
}
struct Stamp {int64_t begin=0,end=0;};
struct Iteration {Stamp whole;std::vector<Stamp> actions;};
struct Component {size_t index=0;std::vector<Stamp> samples;};
struct CpuSnapshot {bool available=false;uint64_t process=0,main=0;};
static CpuSnapshot cpu_snapshot() {
    CpuSnapshot result;
#if defined(_WIN32)
    FILETIME created,exited,kernel,user;
    auto value=[](const FILETIME &time){ULARGE_INTEGER result;result.LowPart=time.dwLowDateTime;result.HighPart=time.dwHighDateTime;return result.QuadPart;};
    require(GetProcessTimes(GetCurrentProcess(),&created,&exited,&kernel,&user)!=0,"GetProcessTimes failed");result.process=(value(kernel)+value(user))*100;
    require(GetThreadTimes(GetCurrentThread(),&created,&exited,&kernel,&user)!=0,"GetThreadTimes failed");result.main=(value(kernel)+value(user))*100;result.available=true;
#endif
    return result;
}
struct Envelope {int64_t begin=0,end=0;CpuSnapshot first,last;};
static json cpu_envelopes(const std::vector<Envelope> &envelopes,int64_t hz) {
    json rows=json::array();int64_t wall=0;uint64_t process=0,main=0;bool available=true;
    for(const auto &envelope:envelopes) {
        const auto duration=ns(envelope.begin,envelope.end,hz);wall+=duration;const bool known=envelope.first.available&&envelope.last.available;available=available&&known;
        uint64_t pc=0,mc=0;if(known){require(envelope.last.process>=envelope.first.process&&envelope.last.main>=envelope.first.main,"CPU times moved backwards");pc=envelope.last.process-envelope.first.process;mc=envelope.last.main-envelope.first.main;process+=pc;main+=mc;}
        rows.push_back(json{{"begin_tick",envelope.begin},{"end_tick",envelope.end},{"wall_ns",duration},
            {"process_cpu_ns",known?json(pc):json(nullptr)},{"main_thread_cpu_ns",known?json(mc):json(nullptr)},
            {"non_main_cpu_ns",known?json(static_cast<int64_t>(pc)-static_cast<int64_t>(mc)):json(nullptr)}});
    }
    return json{{"available",available},{"sampling_wall_ns",wall},{"process_cpu_ns",available?json(process):json(nullptr)},
        {"main_thread_cpu_ns",available?json(main):json(nullptr)},
        {"non_main_cpu_ns",available?json(static_cast<int64_t>(process)-static_cast<int64_t>(main)):json(nullptr)},
        {"sampling_envelopes",rows},{"prediction_parameter",false},
        {"non_main_scope","process CPU minus current main-thread CPU; includes all native auxiliary workers; not proof of GGML polling"},
        {"counter_scope","Windows cumulative process/current-thread kernel+user counters, queried outside every completed-work timer; non-Windows unknown"}};
}

static json capture_verification(Executor &executor) {
    json ports=json::object();
    for(const auto &action:executor.program.at("actions"))if(action.at("kind")=="COMPUTE") {
        const auto &op=executor.operations.at(str(action,"target_id"));const std::string location=action.at("device")=="cpu"?"dram":"vram";
        ports[str(op,"output")]=executor.read(str(op,"output"),location);
        for(const auto &id:op.value("extra_outputs",json::array()))ports[id.get<std::string>()]=executor.read(id,location);
    }
    return json{{"format_version","highlevel-accuracy-native-v1"},{"program_sha256",executor.program.at("program_sha256")},{"boundary_contract",BOUNDARY},
        {"actions",executor.program.at("actions")},{"ports",ports},{"copies",executor.copies},{"copy_values_equal",executor.copy_equal},
        {"observed_runtime_flags",json{{"GGML_CUDA_DISABLE_GRAPHS",std::getenv("GGML_CUDA_DISABLE_GRAPHS")?std::getenv("GGML_CUDA_DISABLE_GRAPHS"):""},
            {"NVIDIA_TF32_OVERRIDE",std::getenv("NVIDIA_TF32_OVERRIDE")?std::getenv("NVIDIA_TF32_OVERRIDE"):""},{"matmul_precision","GGML_PREC_F32"}}},
        {"final_execution_ports_captured_outside_timing",true}};
}

static json run(Executor &executor,const std::string &mode,int warmup,int samples,const json &pool) {
    require(mode=="isolated-components"||mode=="context-trace"||mode=="bare-replay","unsupported mode; no fallback");
    const auto &actions=executor.program.at("actions");const size_t count=actions.size();
    require(count>0&&count<=10000&&uint64_t(count)*samples<=2000000,"action/sample output envelope exceeded");
    const auto hz=frequency();executor.validate();
    std::vector<Iteration> iterations;std::vector<Component> components;std::vector<Envelope> envelopes;
    if(mode=="isolated-components") {
        // As in the historical executor, complete untimed replay has prepared
        // exact upstream inputs. Repeating one action deliberately changes context.
        for(size_t index=0;index<count;++index)if(actions.at(index).at("kind")=="COMPUTE"||actions.at(index).at("kind")=="COPY") {
            components.push_back(Component{index,std::vector<Stamp>(samples)});
        }
        for(auto &component:components) {
            const auto &action=actions.at(component.index);for(int i=0;i<warmup;++i)executor.action(action);
            Envelope envelope;envelope.first=cpu_snapshot();envelope.begin=tick();
            for(auto &stamp:component.samples){stamp.begin=tick();executor.action(action);stamp.end=tick();}
            envelope.end=tick();envelope.last=cpu_snapshot();envelopes.push_back(envelope);
        }
    } else {
        for(int i=0;i<warmup;++i){executor.reset();for(const auto &action:actions)executor.action(action);}
        iterations.resize(samples);if(mode=="context-trace")for(auto &iteration:iterations)iteration.actions.resize(count);
        Envelope envelope;envelope.first=cpu_snapshot();envelope.begin=tick();
        for(auto &iteration:iterations) {
            executor.reset(); // Input restoration is outside this invocation timer.
            iteration.whole.begin=tick();
            if(mode=="context-trace") {
                size_t index=0;for(const auto &action:actions){auto &stamp=iteration.actions[index++];stamp.begin=tick();executor.action(action);stamp.end=tick();}
            } else {for(const auto &action:actions)executor.action(action);}
            iteration.whole.end=tick();
        }
        envelope.end=tick();envelope.last=cpu_snapshot();envelopes.push_back(envelope);
    }
    // Every heap allocation and JSON conversion below is outside all clocks.
    json iteration_json=json::array(),component_json=json::array(),totals=json::array();
    for(size_t i=0;i<iterations.size();++i) {
        const auto &iteration=iterations[i];const auto total=ns(iteration.whole.begin,iteration.whole.end,hz);require(total>0,"nonpositive completed invocation");totals.push_back(total);
        json action_json=json::array();
        for(size_t index=0;index<iteration.actions.size();++index) {
            const auto &stamp=iteration.actions[index];const auto first=ns(iteration.whole.begin,stamp.begin,hz),last=ns(iteration.whole.begin,stamp.end,hz);
            require(first<=last&&last<=total,"action outside completed invocation");
            action_json.push_back(json{{"action_index",index},{"kind",actions.at(index).at("kind")},{"begin_tick",stamp.begin},{"end_tick",stamp.end},
                {"begin_offset_ns",first},{"end_offset_ns",last},{"elapsed_ns",last-first}});
        }
        iteration_json.push_back(json{{"iteration",i},{"begin_tick",iteration.whole.begin},{"end_tick",iteration.whole.end},{"elapsed_ns",total},{"action_intervals",action_json}});
    }
    for(const auto &component:components) {
        json observations=json::array(),intervals=json::array();
        for(size_t i=0;i<component.samples.size();++i){const auto &stamp=component.samples[i];const auto duration=ns(stamp.begin,stamp.end,hz);require(duration>0,"nonpositive isolated action");observations.push_back(duration);intervals.push_back(json{{"sample",i},{"begin_tick",stamp.begin},{"end_tick",stamp.end},{"elapsed_ns",duration}});}
        const auto &action=actions.at(component.index);size_t workspace=0;if(action.at("kind")=="COMPUTE")workspace=executor.graphs.at(str(action,"target_id"))->workspace;
        component_json.push_back(json{{"action_index",component.index},{"action",action},{"samples_ns",observations},{"intervals",intervals},{"workspace_bytes",workspace}});
    }
    auto verification=capture_verification(executor);
    return json{{"format_version","single-layer-calibration-native-v1"},{"process_id",pid()},{"program_sha256",executor.program.at("program_sha256")},
        {"boundary_contract",BOUNDARY},{"execution_boundary","legacy_double_sync"},{"mode",mode},{"warmup",warmup},{"samples",samples},
        {"actions",actions},{"iterations",iteration_json},{"components",component_json},{"total_samples_ns",totals},{"verification_native",verification},{"threadpool",pool},
        {"cpu_diagnostics",cpu_envelopes(envelopes,hz)},
        {"sampling_envelope_scope",mode=="isolated-components"?"sum_of_action_sampling_envelopes; per_component_warmup_excluded":"whole_sampling_loop; resets_included; invocation_timers_exclude_resets; warmup_excluded"},
        {"native_clock",json{{"clock_id",
#if defined(_WIN32)
        "windows_qpc"
#else
        "posix_clock_monotonic"
#endif
        },{"frequency_hz",hz},{"cross_process_monotonic_clock",true}}},
        {"timing_scope","prebuilt_GGML_Executor.action_unchanged; reset_excluded; POD_clocks_inside; JSON_allocation_and_serialization_outside"},
        {"execution_contract","prebuilt_synchronized_GGML_actions;reset_excluded;preallocated_trace_buffer_v1"},
        {"instrumentation_scope",mode=="context-trace"?"per_invocation_and_every_action_POD_clocks":mode=="bare-replay"?"per_invocation_POD_clocks_only":"isolated_same_action_POD_clocks"},
        {"stream_active_time_measured",false},{"host_and_device_times_may_be_added",false},
        {"workspace_scope","known_gallocr_capacity_plus_CPU_cplan; CUDA_private_backend_scratch_unknown"},
        {"physical_capacity_claimed",false},{"runtime_loader_resolution_proven",false}};
}
}
int main(int argc,char **argv){try {
    require(argc==7,"usage: single-layer-probe PROGRAM.json isolated-components|context-trace|bare-replay WARMUP SAMPLES default|0 OUTPUT.json");
    const int warmup=single_layer::integer(argv[3],0,10000,"warmup"),samples=single_layer::integer(argv[4],3,10000,"samples");
    const std::string mode=argv[2],poll=argv[5];require(mode=="isolated-components"||mode=="context-trace"||mode=="bare-replay","unsupported mode; no fallback");
    require(poll=="default"||poll=="0","pool poll must be explicit default or 0");
    std::ifstream existing(argv[6],std::ios::binary);require(!existing.good(),"output already exists");
#if defined(_WIN32)
    _putenv_s("NVIDIA_TF32_OVERRIDE","0");_putenv_s("GGML_CUDA_DISABLE_GRAPHS","1");
#else
    setenv("NVIDIA_TF32_OVERRIDE","0",1);setenv("GGML_CUDA_DISABLE_GRAPHS","1",1);
#endif
    std::ifstream input(argv[1]);require(bool(input),"cannot read program");json program;input>>program;Executor executor(std::move(program));
    const auto pool=single_layer::configure_pool(executor,poll);auto result=single_layer::run(executor,mode,warmup,samples,pool);
    std::ofstream output(argv[6],std::ios::binary);require(bool(output),"cannot create output");output<<result.dump(2)<<'\n';output.close();require(bool(output),"output write failed");
    std::cout<<json{{"status","ok"},{"process_id",result.at("process_id")},{"mode",mode},{"samples",samples},{"poll",poll}}.dump()<<'\n';return 0;
}catch(const std::exception &error){std::cerr<<"single-layer-probe: "<<error.what()<<'\n';return 2;}}
