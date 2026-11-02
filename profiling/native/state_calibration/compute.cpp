// State-aligned diagnostic wrapper. The generated helper is an exact, pinned
// historical executor with only its main function renamed by build.py.
#include "highlevel_generated.inc"
#include "execution_boundary.h"
#include "ggml-backend-impl.h"
#include <cuda_runtime_api.h>
#include <filesystem>
#include <limits>
#if defined(_WIN32)
#include <windows.h>
#else
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#endif

namespace state_probe {
static int64_t tick() {
#if defined(_WIN32)
    LARGE_INTEGER value; require(QueryPerformanceCounter(&value)!=0,"QPC failed"); return value.QuadPart;
#else
    timespec value; require(clock_gettime(CLOCK_MONOTONIC,&value)==0,"clock_gettime failed");
    return int64_t(value.tv_sec)*1000000000LL+value.tv_nsec;
#endif
}
static int64_t frequency() {
#if defined(_WIN32)
    LARGE_INTEGER value; require(QueryPerformanceFrequency(&value)!=0&&value.QuadPart>0,"QPC frequency failed"); return value.QuadPart;
#else
    return 1000000000LL;
#endif
}
static json clock_descriptor() {
    return json{{"clock_id",
#if defined(_WIN32)
        "windows_qpc"
#else
        "posix_clock_monotonic"
#endif
    },{"frequency_hz",frequency()},{"cross_process_monotonic_clock",true}};
}
static uint64_t pid() {
#if defined(_WIN32)
    return GetCurrentProcessId();
#else
    return getpid();
#endif
}
static std::string optional_environment(const char *name) {
    const char *value=std::getenv(name);return value?std::string(value):std::string();
}
static int fixed_iterations() {
    const char *raw=std::getenv("STATE_PROBE_FIXED_ITERATIONS");if(!raw)return 0;
    const std::string value(raw);require(!value.empty()&&value.size()<=6&&std::all_of(value.begin(),value.end(),[](char c){return c>='0'&&c<='9';}),"fixed iterations must be a positive decimal integer");
    const int count=std::stoi(value);require(count>=1&&count<=200000,"fixed iterations outside declared envelope");return count;
}
static bool exists(const std::string &path) {
    if(path.empty())return false;std::error_code error;const bool present=std::filesystem::exists(std::filesystem::u8path(path),error);
    require(!error,"cannot inspect explicit handshake path");return present;
}
static void exclusive_ready(const std::string &path,const json &value) {
    if(path.empty())return;const std::string bytes=value.dump(2)+"\n";
#if defined(_WIN32)
    const auto native_path=std::filesystem::u8path(path);HANDLE file=CreateFileW(native_path.c_str(),GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
    require(file!=INVALID_HANDLE_VALUE,"cannot exclusively create explicit ready file");DWORD written=0;
    const bool okay=WriteFile(file,bytes.data(),static_cast<DWORD>(bytes.size()),&written,nullptr)!=0&&written==bytes.size();
    const bool closed=CloseHandle(file)!=0;require(okay&&closed,"failed writing explicit ready file");
#else
    const int file=open(path.c_str(),O_CREAT|O_EXCL|O_WRONLY,0600);require(file>=0,"cannot exclusively create explicit ready file");
    const auto written=write(file,bytes.data(),bytes.size());const int closed=close(file);
    require(written==static_cast<ssize_t>(bytes.size())&&closed==0,"failed writing explicit ready file");
#endif
}
static int64_t ns(int64_t begin,int64_t end) {
    require(end>=begin,"monotonic clock moved backwards");
    return static_cast<int64_t>(std::llround(static_cast<long double>(end-begin)*1000000000.0L/frequency()));
}
static void cuda_ok(cudaError_t status,const char *operation) {
    require(status==cudaSuccess,std::string(operation)+": "+cudaGetErrorString(status));
}
struct TimingEvents {
    cudaEvent_t first=nullptr,last=nullptr;ggml_backend_event first_wrapper{},last_wrapper{};
    explicit TimingEvents(ggml_backend_t backend,int device) {
        cuda_ok(cudaSetDevice(device),"event cudaSetDevice");
        cuda_ok(cudaEventCreateWithFlags(&first,cudaEventDefault),"first timing event create");
        const auto status=cudaEventCreateWithFlags(&last,cudaEventDefault);
        if(status!=cudaSuccess){cudaEventDestroy(first);first=nullptr;cuda_ok(status,"last timing event create");}
        first_wrapper.device=last_wrapper.device=ggml_backend_get_device(backend);
        first_wrapper.context=first;last_wrapper.context=last;
    }
    ~TimingEvents(){if(first)cudaEventDestroy(first);if(last)cudaEventDestroy(last);}
    void begin(ggml_backend_t backend){ggml_backend_event_record(&first_wrapper,backend);}
    void end(ggml_backend_t backend){ggml_backend_event_record(&last_wrapper,backend);}
    float elapsed_ms(){float value=0;cuda_ok(cudaEventElapsedTime(&value,first,last),"timing event elapsed");require(std::isfinite(value)&&value>=0,"invalid CUDA event duration");return value;}
};
struct TargetTiming {
    bool valid=false;int64_t begin=0,enqueue_begin=0,enqueue_end=0,sync_begin=0,sync_end=0,end=0;float stream_ms=0;bool events=false;
    json describe(int iteration,size_t target,StateExecutionBoundary boundary) const {
        return json{{"iteration",iteration},{"action_index",target},{"execution_boundary",state_boundary_name(boundary)},
            {"begin_tick",begin},{"end_tick",end},{"host_elapsed_ns",ns(begin,end)},
            {"host_enqueue_begin_tick",enqueue_begin},{"host_enqueue_end_tick",enqueue_end},{"host_enqueue_ns",ns(enqueue_begin,enqueue_end)},
            {"host_sync_begin_tick",sync_begin},{"host_sync_end_tick",sync_end},{"host_sync_ns",ns(sync_begin,sync_end)},
            {"stream_span_ms",events?json(stream_ms):json(nullptr)},
            {"stream_span_ns",events?json(static_cast<int64_t>(std::llround(static_cast<double>(stream_ms)*1000000.0))):json(nullptr)}};
    }
};
static json capture_ports(Executor &executor) {
    json output=json::object();
    for(const auto &action:executor.program.at("actions"))if(action.at("kind")=="COMPUTE") {
        const auto &op=executor.operations.at(str(action,"target_id"));const std::string location=action.at("device")=="cpu"?"dram":"vram";
        output[str(op,"output")]=executor.read(str(op,"output"),location);
        for(const auto &id:op.value("extra_outputs",json::array()))output[id.get<std::string>()]=executor.read(id,location);
    }
    return output;
}
static json compare_ports(const json &expected,const json &actual) {
    require(expected.size()==actual.size(),"port identity changed");double maximum_abs=0,maximum_relative=0;size_t values=0;
    for(const auto &item:expected.items()) {
        require(actual.contains(item.key())&&actual.at(item.key()).size()==item.value().size(),"port layout changed");
        for(size_t i=0;i<item.value().size();++i) {
            const double a=item.value().at(i),b=actual.at(item.key()).at(i);require(std::isfinite(a)&&std::isfinite(b),"non-finite output");
            const double difference=std::abs(a-b);maximum_abs=std::max(maximum_abs,difference);maximum_relative=std::max(maximum_relative,difference/std::max(std::abs(a),1e-30));
            require(difference<=5e-5+5e-5*std::abs(a),"timed result differs from identical untimed replay");++values;
        }
    }
    return json{{"passed",true},{"checked_values",values},{"maximum_absolute_error",maximum_abs},{"maximum_relative_error",maximum_relative},{"atol",5e-5},{"rtol",5e-5},
        {"reference_scope","same_native_executor_untimed_full_replay; independent_scalar_or_NumPy_oracle_required_externally"}};
}
struct InputSnapshot {std::string id,location;std::vector<uint8_t> bytes;};
static std::vector<InputSnapshot> inputs(Executor &executor,const json &action) {
    const auto &op=executor.operations.at(str(action,"target_id"));const std::string location=action.at("device")=="cpu"?"dram":"vram";
    std::set<std::string> outputs={str(op,"output")};for(const auto &id:op.value("extra_outputs",json::array()))outputs.insert(id);
    std::vector<InputSnapshot> result;
    for(const auto &id:op.at("inputs")) {
        require(!outputs.count(id.get<std::string>()),"isolated action aliases an input/output");
        auto *tensor=executor.tensor(id,location);InputSnapshot value{id,location,std::vector<uint8_t>(ggml_nbytes(tensor))};
        ggml_backend_tensor_get(tensor,value.bytes.data(),0,value.bytes.size());result.push_back(std::move(value));
    }
    return result;
}
static json verify_inputs(Executor &executor,const std::vector<InputSnapshot> &snapshots) {
    json result=json::array();for(const auto &value:snapshots) {
        std::vector<uint8_t> observed(value.bytes.size());ggml_backend_tensor_get(executor.tensor(value.id,value.location),observed.data(),0,observed.size());
        require(observed==value.bytes,"target input mutated during timed diagnostic");
        result.push_back(json{{"tensor_id",value.id},{"location",value.location},{"storage_bytes",value.bytes.size()},{"exact_bytes_unchanged",true}});
    }return result;
}
static json graph_description(Executor &executor,const json &action) {
    const auto id=str(action,"target_id");const auto &binding=executor.bindings.at(id);auto &graph=*executor.graphs.at(id);json nodes=json::array();
    for(int index=0;index<ggml_graph_n_nodes(graph.graph);++index) {
        auto *node=ggml_graph_node(graph.graph,index);json shape=json::array();for(int i=0;i<4;++i)shape.push_back(node->ne[i]);
        json source_shapes=json::array();for(int source=0;source<GGML_MAX_SRC;++source)if(node->src[source]) {
            json ne=json::array();for(int i=0;i<4;++i)ne.push_back(node->src[source]->ne[i]);source_shapes.push_back(ne);
        }
        nodes.push_back(json{{"index",index},{"op",ggml_op_name(node->op)},{"type",ggml_type_name(node->type)},{"ne",shape},{"sources_ne",source_shapes},
            {"requested_precision",node->op==GGML_OP_MUL_MAT?json("GGML_PREC_F32"):json(nullptr)}});
    }
    return json{{"action",action},{"binding",binding},{"graph_nodes",nodes},{"workspace_bytes",graph.workspace},
        {"workspace_scope","known_gallocr_capacity_plus_CPU_cplan; CUDA_private_backend_scratch_unknown"}};
}
static json run(Executor &executor,const std::string &mode,size_t target,int warmup,int blocks,int minimum_ms,bool stamped) {
    require(mode=="isolated"||mode=="mixed","mode must be isolated or mixed");
    require(target<executor.program.at("actions").size(),"target action out of range");const auto &selected=executor.program.at("actions").at(target);
    require(selected.at("kind")=="COMPUTE","target must be a COMPUTE action");
    require(warmup>=0&&warmup<=10000&&blocks>=1&&blocks<=1000&&minimum_ms>=200&&minimum_ms<=60000,"invalid sample envelope");
    const std::string ready_path=optional_environment("STATE_PROBE_READY_FILE"),stop_path=optional_environment("STATE_PROBE_STOP_FILE");
    const int fixed_count=fixed_iterations();
    const auto boundary=state_parse_boundary(optional_environment("STATE_PROBE_EXECUTION_BOUNDARY"));
    if(boundary!=StateExecutionBoundary::legacy_double_sync)require(selected.at("device")=="gpu","alternate execution boundaries require a selected GPU COMPUTE action");
    require(ready_path.empty()||!exists(ready_path),"explicit ready file already exists");
    require(stop_path.empty()||!exists(stop_path),"explicit stop file already exists at diagnostic start");
    require(ready_path.empty()||stop_path.empty()||std::filesystem::absolute(std::filesystem::u8path(ready_path)).lexically_normal()!=std::filesystem::absolute(std::filesystem::u8path(stop_path)).lexically_normal(),"ready and stop paths must differ");
    executor.validate();const auto initial_ports=capture_ports(executor);const auto snapshots=inputs(executor,selected);
    std::unique_ptr<TimingEvents> timing_events;
    if(boundary==StateExecutionBoundary::async_events)timing_events=std::make_unique<TimingEvents>(executor.gpu,executor.program.at("physical").value("cuda_device",0));
    auto perform=[&](const json &action,size_t index,TargetTiming &diagnostic) {
        if(index!=target||boundary==StateExecutionBoundary::legacy_double_sync){executor.action(action);return;}
        auto &graph=*executor.graphs.at(str(action,"target_id"));require(graph.backend==executor.gpu,"alternate boundary target backend mismatch");
        diagnostic.valid=true;diagnostic.events=bool(timing_events);diagnostic.begin=tick();
        if(timing_events)timing_events->begin(graph.backend);
        diagnostic.enqueue_begin=tick();const auto status=ggml_backend_graph_compute_async(graph.backend,graph.graph);diagnostic.enqueue_end=tick();
        require(status==GGML_STATUS_SUCCESS,"GGML asynchronous compute failed");
        if(timing_events)timing_events->end(graph.backend);
        diagnostic.sync_begin=tick();ggml_backend_synchronize(graph.backend);diagnostic.sync_end=tick();
        if(timing_events)diagnostic.stream_ms=timing_events->elapsed_ms();
        diagnostic.end=tick();
    };
    auto replay=[&](json *intervals,json *diagnostics,int iteration) {
        if(mode=="isolated") {
            TargetTiming diagnostic;
            if(intervals){const auto begin=tick();perform(selected,target,diagnostic);const auto end=tick();intervals->push_back(json{{"iteration",iteration},{"action_index",target},{"begin_tick",begin},{"end_tick",end},{"elapsed_ns",ns(begin,end)}});}
            else perform(selected,target,diagnostic);
            if(diagnostic.valid&&diagnostics)diagnostics->push_back(diagnostic.describe(iteration,target,boundary));
        } else {
            size_t index=0;for(const auto &action:executor.program.at("actions")) {
                TargetTiming diagnostic;
                if(intervals){const auto begin=tick();perform(action,index,diagnostic);const auto end=tick();intervals->push_back(json{{"iteration",iteration},{"action_index",index},{"begin_tick",begin},{"end_tick",end},{"elapsed_ns",ns(begin,end)}});}
                else perform(action,index,diagnostic);
                if(diagnostic.valid&&diagnostics)diagnostics->push_back(diagnostic.describe(iteration,target,boundary));++index;
            }
        }
    };
    const auto warmup_begin=tick();for(int iteration=0;iteration<warmup;++iteration){if(mode=="mixed")executor.reset();replay(nullptr,nullptr,iteration);}const auto warmup_end=tick();
    const auto ready_tick=tick();exclusive_ready(ready_path,json{{"format_version","state-calibration-ready-v1"},{"clock",clock_descriptor()},{"ready_tick",ready_tick},
        {"process_id",pid()},{"target_action_index",target},{"mode",mode},{"execution_boundary",state_boundary_name(boundary)},{"phase","validation_and_warmup_completed_before_first_timed_block"},
        {"executing_work_proven_by_ready_file",false}});
    json result_blocks=json::array();size_t total_intervals=0;bool stop_requested=false;int64_t stop_observed_tick=0;
    for(int block=0;block<blocks;++block) {
        json iterations=json::array(),intervals=json::array(),diagnostics=json::array();const auto block_begin=tick();auto block_end=block_begin;int count=0;int64_t timed_ns=0;
        do {
            if(mode=="mixed")executor.reset(); // Identical input restoration excluded from each iteration interval.
            const auto begin=tick();replay(stamped?&intervals:nullptr,&diagnostics,count);const auto end=tick();const auto duration=ns(begin,end);
            iterations.push_back(json{{"iteration",count},{"begin_tick",begin},{"end_tick",end},{"elapsed_ns",duration}});timed_ns+=duration;
            block_end=tick();require(++count<=200000,"block iteration limit exceeded");
            require(total_intervals+intervals.size()<=2000000,"action interval output envelope exceeded");
            // Explicit caller-owned stop signal is checked outside completed-work
            // timers and only after a whole invocation has completed.
            if(!stop_path.empty()&&exists(stop_path)){stop_requested=true;stop_observed_tick=tick();break;}
        }while(fixed_count>0?count<fixed_count:ns(block_begin,block_end)<int64_t(minimum_ms)*1000000LL);
        total_intervals+=intervals.size();
        result_blocks.push_back(json{{"block_index",block},{"begin_tick",block_begin},{"end_tick",block_end},{"wall_elapsed_ns",ns(block_begin,block_end)},
            {"completed_work_ns",timed_ns},{"iterations",iterations},{"action_intervals",intervals},{"target_diagnostics",diagnostics},{"iteration_count",count},
            {"minimum_elapsed_satisfied",ns(block_begin,block_end)>=int64_t(minimum_ms)*1000000LL},
            {"fixed_iteration_count_satisfied",fixed_count>0?json(count==fixed_count):json(nullptr)}});
        if(stop_requested)break;
    }
    const auto final_ports=capture_ports(executor);auto consistency=compare_ports(initial_ports,final_ports);auto input_checks=verify_inputs(executor,snapshots);
    executor.validate();auto post_consistency=compare_ports(initial_ports,capture_ports(executor));
    return json{{"format_version","state-calibration-native-v1"},{"program_sha256",executor.program.at("program_sha256")},{"boundary_contract",BOUNDARY},
        {"process_id",pid()},{"clock",clock_descriptor()},{"mode",mode},{"target_action_index",target},{"target",graph_description(executor,selected)},
        {"execution_boundary",state_boundary_name(boundary)},{"execution_boundary_scope","only_selected_GPU_COMPUTE_action; all_other_actions_unchanged; validation_uses_legacy_executor"},
        {"target_diagnostics_scope",json{{"legacy_double_sync","Original Executor.action unchanged; public graph_compute already synchronizes, followed by its explicit second synchronize. Outer action intervals only; enqueue/sync/stream split unknown."},
            {"async_one_sync","Public graph_compute_async then one backend synchronize; no CUDA events."},
            {"async_events","Timing-enabled CUDA events recorded by actual GGML backend callback before/after graph_compute_async, then one backend synchronize and elapsed-time query; no additional event synchronize."},
            {"stream_span_limit","GPU stream elapsed interval includes queued work and host launch gaps between events; excludes waiting before the first event executes; not summed kernel-active time or proof of kernel identity."},
            {"instrumentation_limit","Optional target diagnostic timestamp/query and JSON recording cost is retained; compare async_one_sync against async_events. These boundaries are diagnostic-only and must not share legacy cost-model cells."}}},
        {"warmup_iterations",warmup},{"warmup_begin_tick",warmup_begin},{"warmup_end_tick",warmup_end},{"minimum_block_ms",minimum_ms},{"blocks",result_blocks},
        {"ready_tick",ready_tick},{"requested_blocks",blocks},{"graceful_stop_requested",stop_requested},{"stop_observed_tick",stop_requested?json(stop_observed_tick):json(nullptr)},
        {"iteration_policy",fixed_count>0?"fixed_count":"minimum_wall_time"},{"requested_iterations_per_block",fixed_count>0?json(fixed_count):json(nullptr)},
        {"full_fixed_count",fixed_count>0?json(!stop_requested&&result_blocks.size()==static_cast<size_t>(blocks)):json(nullptr)},
        {"handshake",json{{"ready_file_requested",!ready_path.empty()},{"stop_file_requested",!stop_path.empty()},
            {"stop_poll_scope","between_complete_iterations_outside_work_timer"},{"partial_final_block_allowed_only_after_stop",true}}},
        {"instrumentation",stamped?"per_iteration_and_per_action_host_completed_work":"per_iteration_host_completed_work_only"},
        {"initial_ports",initial_ports},{"ports",final_ports},{"native_repeat_consistency",consistency},{"post_replay_consistency",post_consistency},
        {"target_input_checks",input_checks},{"copies",executor.copies},{"copy_values_equal",executor.copy_equal},{"actions",executor.program.at("actions")},
        {"timing_scope","prebuilt_synchronized_GGML_actions; mixed_reset_excluded_from_iteration_but_included_in_block_wall; JSON_interval_recording_in_iteration"},
        {"observed_runtime_flags",json{{"GGML_CUDA_DISABLE_GRAPHS",std::getenv("GGML_CUDA_DISABLE_GRAPHS")?std::getenv("GGML_CUDA_DISABLE_GRAPHS"):""},
            {"NVIDIA_TF32_OVERRIDE",std::getenv("NVIDIA_TF32_OVERRIDE")?std::getenv("NVIDIA_TF32_OVERRIDE"):""},{"matmul_precision","GGML_PREC_F32"}}},
        {"kernel_execution_path_proven",false},{"physical_device_capacity_claimed",false},{"cuda_private_workspace_bytes",nullptr}};
}
}
int main(int argc,char **argv) {try {
#if defined(_WIN32)
    _putenv_s("NVIDIA_TF32_OVERRIDE","0");_putenv_s("GGML_CUDA_DISABLE_GRAPHS","1");
#else
    setenv("NVIDIA_TF32_OVERRIDE","0",1);setenv("GGML_CUDA_DISABLE_GRAPHS","1",1);
#endif
    require(argc==9,"usage: state-calibration-probe PROGRAM.json isolated|mixed TARGET_ACTION_INDEX WARMUP BLOCKS MIN_BLOCK_MS stamped|bare OUTPUT.json");
    const int target=std::stoi(argv[3]);require(target>=0,"negative target action");const std::string instrumentation=argv[7];
    require(instrumentation=="stamped"||instrumentation=="bare","instrumentation must be stamped or bare");
    std::ifstream input(argv[1]);require(bool(input),"cannot read program");json source;input>>source;
    std::ifstream existing(argv[8],std::ios::binary);require(!existing.good(),"output already exists");
    Executor executor(std::move(source));const auto process_begin=state_probe::tick();
    auto result=state_probe::run(executor,argv[2],static_cast<size_t>(target),std::stoi(argv[4]),std::stoi(argv[5]),std::stoi(argv[6]),instrumentation=="stamped");
    result["diagnostic_begin_tick"]=process_begin;result["diagnostic_end_tick"]=state_probe::tick();
    std::ofstream output(argv[8],std::ios::binary);require(bool(output),"cannot write output");output<<result.dump(2)<<'\n';output.close();require(bool(output),"failed writing output");
    std::cout<<json{{"status","ok"},{"program_sha256",result.at("program_sha256")},{"mode",argv[2]},{"process_id",result.at("process_id")},{"blocks",result.at("blocks").size()}}.dump()<<'\n';return 0;
}catch(const std::exception &error){std::cerr<<"state-calibration-probe: "<<error.what()<<'\n';return 2;}}
