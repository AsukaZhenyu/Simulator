// Additive history metrology; the historical Executor and its mathematics remain unchanged.
#define main historical_steady_entry_not_called
#include "../../native/steady_costs/probe.cpp"
#undef main
// CUPTI is the third instrumentation -- an observer that timestamps the engine's
// own stream -- and it is CUDA's. A Metal build refuses `instrument=cupti` (see
// the request check below) and does not carry the header at all, so nothing here
// depends on a toolkit it does not have.
#if !defined(PROBE_BACKEND_METAL)
#include "../../native/physical_stages/cupti_capture.hpp"
#endif
// <psapi.h> is the Windows half of what platform.hpp already brings in; the
// probes' affinity, accounting and system-counter calls are spelled the same on
// both hosts. See ../../native/portable/platform.hpp.
#include "../../native/portable/platform.hpp"

namespace history_probe {
// QPC ticks to floor-converted ns. CUPTI's header carries this on a CUDA build;
// the conversion is two lines and has nothing to do with CUPTI, so a Metal build
// states it here rather than pulling in an observer it will not use.
inline uint64_t ticks_to_ns(uint64_t value, uint64_t hz) {
    return (value / hz) * 1000000000ULL + (value % hz) * 1000000000ULL / hz;
}
static uint64_t filetime(FILETIME value) {
    return (uint64_t(value.dwHighDateTime)<<32)|value.dwLowDateTime;
}
static uint64_t absolute_ns(int64_t tick) {
    require(tick>=0,"negative QPC timestamp");
    return ticks_to_ns(static_cast<uint64_t>(tick),static_cast<uint64_t>(steady::frequency()));
}
static json pointer_kind(void *pointer) {
#if defined(PROBE_BACKEND_METAL)
    // One memory. "Is this pointer registered with the device?" has no second
    // answer to give, and inventing one -- either `host` (pinned) or
    // `unregistered` (pageable) -- would put a category in the receipt that was
    // never observed. The receipt says what is true instead.
    (void)pointer;
    return json{{"status","not_applicable"},{"query","none"},
        {"reason","unified memory: the accelerator has no separate address space, so "
                  "there is no pageable/registered distinction to observe"},
        {"device_memory","shared"}};
#else
    cudaPointerAttributes attrs{};const auto status=cudaPointerGetAttributes(&attrs,pointer);
    if(status==cudaErrorInvalidValue) {
        cudaGetLastError();
        return json{{"status","observed"},{"query","cudaPointerGetAttributes"},
            {"registered_host",false},{"unregistered_pageable",true},{"runtime_error_code",static_cast<int>(status)}};
    }
    require(status==cudaSuccess,"host pointer category query failed");
    return json{{"status","observed"},{"query","cudaPointerGetAttributes"},
        {"cuda_memory_type",static_cast<int>(attrs.type)},
        {"registered_host",attrs.type==cudaMemoryTypeHost},
        {"unregistered_pageable",attrs.type==cudaMemoryTypeUnregistered}};
#endif
}
static unsigned online_cpus(DWORD_PTR mask) {
    unsigned count=0;for(int cpu=0;cpu<64;++cpu)if((mask>>cpu)&1)++count;return count;
}
static json apply_process_mask(const json &request) {
    require(PORTABLE_AFFINITY!=0,"this platform exposes no enforced CPU affinity; the history protocol "
        "pins every CPU and one main thread, and cannot be honoured here");
    const auto requested=static_cast<DWORD_PTR>(steady::positive(request.at("expected_process_mask"),"process mask",INT64_MAX));
    DWORD_PTR initial=0,system=0,observed=0,after_system=0;
    require(GetProcessAffinityMask(GetCurrentProcess(),&initial,&system)!=0,"cannot observe process affinity");
    // The protocol is "every CPU this host offers, and prove it" -- not "32 CPUs".
    // Written as 0xFFFFFFFF it could only ever run on a 32-CPU Windows machine,
    // so a device with 20 cores had no mask to send and the refusal read like a
    // measurement failure. The requested mask must be exactly the online set,
    // and the receipt says how many CPUs that turned out to be.
    require(requested==system,"requested process mask must name every online CPU; this host offers "
        +std::to_string(online_cpus(static_cast<DWORD_PTR>(system)))+" (0x"
        +std::to_string(static_cast<unsigned long long>(system))+")");
    require((requested&system)==requested,"requested process mask outside system affinity");
    require(SetProcessAffinityMask(GetCurrentProcess(),requested)!=0,"cannot set owned process mask");
    require(GetProcessAffinityMask(GetCurrentProcess(),&observed,&after_system)!=0&&observed==requested,"owned process mask readback differs");
    return json{{"status","observed"},{"initial_mask",static_cast<uint64_t>(initial)},
        {"requested_mask",static_cast<uint64_t>(requested)},{"observed_mask",static_cast<uint64_t>(observed)},
        {"system_mask",static_cast<uint64_t>(after_system)},{"online_cpus",online_cpus(requested)},
        {"mask_semantics","every online CPU of this host, not a fixed width"},{"enforced",true},
        {"applied_before_backend_initialization",true},{"scope",PORTABLE_AFFINITY_SCOPE}};
}
static json finish_process_mask(json receipt) {
    DWORD_PTR observed=0,system=0;
    require(GetProcessAffinityMask(GetCurrentProcess(),&observed,&system)!=0&&
        static_cast<uint64_t>(observed)==receipt.at("requested_mask").get<uint64_t>(),"owned process mask changed during run");
    receipt["final_mask"]=static_cast<uint64_t>(observed);return receipt;
}

// These are endpoint diagnostics. CPU accounting is not a per-action Running/Ready/Blocked trace.
struct CpuPoint {
    bool thread_ok=false,process_ok=false,memory_ok=false,cycles_ok=false;
    uint64_t thread_ns=0,process_ns=0,cycles=0;
    DWORD page_faults=0;
};
static CpuPoint cpu_point() {
    CpuPoint point;FILETIME created{},exited{},kernel{},user{};
    point.thread_ok=GetThreadTimes(GetCurrentThread(),&created,&exited,&kernel,&user)!=0;
    if(point.thread_ok)point.thread_ns=100ULL*(filetime(kernel)+filetime(user));
    point.process_ok=GetProcessTimes(GetCurrentProcess(),&created,&exited,&kernel,&user)!=0;
    if(point.process_ok)point.process_ns=100ULL*(filetime(kernel)+filetime(user));
    ULONG64 cycles=0;point.cycles_ok=QueryThreadCycleTime(GetCurrentThread(),&cycles)!=0;point.cycles=cycles;
    PROCESS_MEMORY_COUNTERS memory{};memory.cb=sizeof(memory);
    point.memory_ok=GetProcessMemoryInfo(GetCurrentProcess(),&memory,sizeof(memory))!=0;
    if(point.memory_ok)point.page_faults=memory.PageFaultCount;
    return point;
}
static json cpu_delta(const CpuPoint &first,const CpuPoint &last) {
    require(!first.thread_ok||!last.thread_ok||last.thread_ns>=first.thread_ns,"thread accounting moved backwards");
    require(!first.process_ok||!last.process_ok||last.process_ns>=first.process_ns,"process accounting moved backwards");
    require(!first.cycles_ok||!last.cycles_ok||last.cycles>=first.cycles,"thread cycle counter moved backwards");
    return json{{"thread_accounted_cpu_ns",first.thread_ok&&last.thread_ok?json(last.thread_ns-first.thread_ns):json(nullptr)},
        {"process_accounted_cpu_ns",first.process_ok&&last.process_ok?json(last.process_ns-first.process_ns):json(nullptr)},
        {"thread_cycles",first.cycles_ok&&last.cycles_ok?json(last.cycles-first.cycles):json(nullptr)},
        {"process_page_fault_delta",first.memory_ok&&last.memory_ok&&last.page_faults>=first.page_faults?json(last.page_faults-first.page_faults):json(nullptr)},
        {"sampled_outside_timed_batch",true},{"accounting_unit_ns",100},
        {"scope","endpoint accounting brackets include counter-query work; cycle counts are not useful FLOPs"},
        {"running_ready_blocked_separated",false},{"external_contention_identified",false}};
}
static json preblock_state(bool gpu) {
    const auto at=steady::tick();MEMORYSTATUSEX memory{};memory.dwLength=sizeof(memory);
    FILETIME idle{},kernel{},user{};
    const bool memory_ok=GlobalMemoryStatusEx(&memory)!=0;
    const bool times_ok=GetSystemTimes(&idle,&kernel,&user)!=0;
#if defined(PROBE_BACKEND_METAL)
    // `ggml_backend_dev_memory` is the backend-neutral way to ask the same
    // question `cudaMemGetInfo` answers, and on a unified machine the answer is
    // the host's memory: there is no second pool to report.
    (void)gpu;
    return json{{"sample_tick",at},{"sample_ns",absolute_ns(at)},
        {"outside_timed_interval",true},{"sampled_before_untimed_prime",true},
        {"derived_from_target_duration",false},{"cpu_utilization",nullptr},
        {"gpu_utilization",nullptr},{"gpu_clock_mhz",nullptr},{"gpu_temperature_c",nullptr},
        {"utilization_status","unknown; no independent utilization interval sampled"},
        {"cpu_cumulative_times_100ns",times_ok?json{{"idle",filetime(idle)},{"kernel_including_idle",filetime(kernel)},{"user",filetime(user)}}:json(nullptr)},
        {"system_memory",memory_ok?json{{"total_bytes",memory.ullTotalPhys},{"available_bytes",memory.ullAvailPhys},{"load_percent",memory.dwMemoryLoad}}:json(nullptr)},
        {"cuda_memory",nullptr},
        {"device_memory",json{{"status","not_queried"},{"reason","unified memory: DRAM and the accelerator's memory are one pool, so a free/total pair would be the host's"}}},
        {"memory_capacity_is_bandwidth",false},{"external_only_state",false}};
#else
    size_t free_bytes=0,total_bytes=0;cudaError_t status=cudaSuccess;
    if(gpu){status=cudaMemGetInfo(&free_bytes,&total_bytes);if(status!=cudaSuccess)cudaGetLastError();}
    return json{{"sample_tick",at},{"sample_ns",absolute_ns(at)},
        {"outside_timed_interval",true},{"sampled_before_untimed_prime",true},
        {"derived_from_target_duration",false},{"cpu_utilization",nullptr},
        {"gpu_utilization",nullptr},{"gpu_clock_mhz",nullptr},{"gpu_temperature_c",nullptr},
        {"utilization_status","unknown; no independent utilization interval sampled"},
        {"cpu_cumulative_times_100ns",times_ok?json{{"idle",filetime(idle)},{"kernel_including_idle",filetime(kernel)},{"user",filetime(user)}}:json(nullptr)},
        {"system_memory",memory_ok?json{{"total_bytes",memory.ullTotalPhys},{"available_bytes",memory.ullAvailPhys},{"load_percent",memory.dwMemoryLoad}}:json(nullptr)},
        {"cuda_memory",!gpu?json(nullptr):status==cudaSuccess?json{{"status","observed"},{"free_bytes",free_bytes},{"total_bytes",total_bytes}}:json{{"status","unknown"},{"runtime_error_code",static_cast<int>(status)}}},
        {"memory_capacity_is_bandwidth",false},{"external_only_state",false}};
#endif
}

class ArenaRegistration {
    void *base_=nullptr;size_t size_=0;bool registered_=false,closed_=false,no_cuda_=false;json receipt_;
public:
    ArenaRegistration(Executor &executor,const std::string &mode) {
        require(mode=="pageable"||mode=="registered","unknown host-memory mode");
        require(executor.dram_buffer!=nullptr,"missing owned DRAM arena");
        base_=ggml_backend_buffer_get_base(executor.dram_buffer);size_=ggml_backend_buffer_get_size(executor.dram_buffer);
        require(base_!=nullptr&&size_>0,"empty DRAM arena");
#if defined(PROBE_BACKEND_METAL)
        // One memory: the arena the CPU backend owns is already in the address
        // space the accelerator reads, so there is no pageable/registered pair
        // to observe or move between. `pageable` is the only mode that names
        // something real here; `registered` would be a request to change a
        // category that does not exist, so it is refused rather than reported
        // as an already-satisfied one.
        require(mode=="pageable","registered arena requires a pageable/registered distinction; unified memory has none");
        no_cuda_=true;
        const json category=pointer_kind(base_);
        receipt_=json{{"mode",mode},{"arena_bytes",size_},{"before",category},{"during",category},
            {"scope","owned CPU backend arena; unified memory needs no registration"},{"registration_timed",false},
            {"registration_is_free_initialization",false},{"cuda_memory_category_verified",false},
            {"device_memory","shared"},{"registration_status","not_applicable"}};
        return;
#else
        no_cuda_=executor.gpu==nullptr;
        if(no_cuda_) {
            require(mode=="pageable","registered arena requires GPU context");
            const json category{{"status","not_queried"},{"reason","CPU-only control; no CUDA pointer query or context creation"},
                {"host_cpu_backend",true},{"registered_host",nullptr},{"unregistered_pageable",nullptr}};
            receipt_=json{{"mode",mode},{"arena_bytes",size_},{"before",category},{"during",category},
                {"scope","owned CPU backend arena; no registration attempted"},{"registration_timed",false},
                {"registration_is_free_initialization",false},{"cuda_memory_category_verified",false}};
            return;
        }
        const auto before=pointer_kind(base_);
        require(before.at("unregistered_pageable")==true,"baseline DRAM arena is not observed pageable");
        receipt_=json{{"mode",mode},{"arena_bytes",size_},{"before",before},
            {"scope","entire owned DRAM backend arena only"},{"registration_timed",false},
            {"registration_is_free_initialization",false}};
        if(mode=="registered") {
            int device=-1,supported=0;
            require(cudaGetDevice(&device)==cudaSuccess&&cudaDeviceGetAttribute(&supported,cudaDevAttrHostRegisterSupported,device)==cudaSuccess&&supported!=0,
                "explicit host registration unsupported");
            require(cudaHostRegister(base_,size_,cudaHostRegisterDefault)==cudaSuccess,"explicit owned arena registration failed; no fallback");
            registered_=true;
        }
        receipt_["during"]=pointer_kind(base_);
        require(receipt_.at("during").at(mode=="registered"?"registered_host":"unregistered_pageable")==true,"actual arena category differs from request");
#endif
    }
    ~ArenaRegistration(){if(registered_)cudaHostUnregister(base_);}
    json finish() {
        require(!closed_,"registration receipt already finished");
        if(no_cuda_){receipt_["after"]=receipt_.at("during");receipt_["restored"]=true;closed_=true;return receipt_;}
        if(registered_) {require(cudaHostUnregister(base_)==cudaSuccess,"owned arena unregister failed");registered_=false;}
        receipt_["after"]=pointer_kind(base_);require(receipt_.at("after").at("unregistered_pageable")==true,"arena pointer category was not restored");
        receipt_["restored"]=true;closed_=true;return receipt_;
    }
};

struct ActionStamp {
    int64_t begin=0,copy_return=0,gpu_return=0,end=0;
    uint64_t external_id=0;size_t action_index=0;int invocation=0;
};
struct Block {
    int64_t begin=0,end=0;json state;CpuPoint before,after;
};
struct Intervention {
    int64_t begin=0,end=0;uint64_t checksum=0,bytes_read=0;
    int trial=0,invocation=0;size_t action_index=0;
};
static Intervention intervene(Executor &executor,const json &action,const std::string &kind,
                             int trial,int invocation,size_t index) {
    Intervention record;record.trial=trial;record.invocation=invocation;record.action_index=index;
    auto *source=executor.tensor(str(action,"target_id"),"dram");
    const size_t bytes=ggml_nbytes(source);record.begin=steady::tick();
    if(kind=="idle")Sleep(5);
    else {
        require(kind=="source-read","unknown intervention");
        const volatile uint8_t *data=static_cast<const volatile uint8_t *>(source->data);
        uint64_t sum=0;for(size_t at=0;at<bytes;at+=64){sum+=data[at];++record.bytes_read;}
        record.checksum=sum;
    }
    record.end=steady::tick();return record;
}
static json semantic_action(Executor &executor,const json &action,size_t index) {
    json row{{"action_index",index},{"kind",action.at("kind")},
        {"target_id",action.value("target_id",json(nullptr))},
        {"device",action.value("device",json(nullptr))}};
    if(action.at("kind")=="COPY") {
        const auto id=str(action,"target_id");const auto &descriptor=executor.tensors.at(id).descriptor;
        row.update(json{{"src",action.at("src")},{"dst",action.at("dst")},
            {"bytes",descriptor.at("storage_bytes")},{"dtype",descriptor.at("dtype")},
            {"ne",descriptor.at("ne")},{"nb",descriptor.at("nb")}});
        if(action.at("src")=="dram")row["source_pointer_category"]=pointer_kind(executor.tensor(id,"dram")->data);
        if(action.at("dst")=="dram")row["destination_pointer_category"]=pointer_kind(executor.tensor(id,"dram")->data);
    }
    return row;
}
static void action(Executor &executor,const json &value,ActionStamp *stamp) {
    if(!stamp) {executor.action(value);return;}
    stamp->begin=steady::tick();
    if(value.at("kind")=="COPY") {
        auto *source=executor.tensor(str(value,"target_id"),str(value,"src"));
        auto *target=executor.tensor(str(value,"target_id"),str(value,"dst"));
        // Same operations and order as Executor.action; only QPC boundaries are added.
        ggml_backend_tensor_copy(source,target);stamp->copy_return=steady::tick();
        if(executor.gpu)ggml_backend_synchronize(executor.gpu);stamp->gpu_return=steady::tick();
        ggml_backend_synchronize(executor.cpu);
    } else executor.action(value);
    stamp->end=steady::tick();
}
static json action_row(const ActionStamp &stamp,int trial,const json &value) {
    json result{{"trial",trial},{"invocation",stamp.invocation},{"action_index",stamp.action_index},
        {"external_id",stamp.external_id},{"kind",value.at("kind")},
        {"begin_tick",stamp.begin},{"end_tick",stamp.end},
        {"begin_ns",absolute_ns(stamp.begin)},{"end_ns",absolute_ns(stamp.end)},
        {"wall_ns",steady::ns(stamp.begin,stamp.end)},
        {"copy_call_ns",nullptr},{"submit_copy_api_ns",nullptr},
        {"gpu_sync_ns",nullptr},{"cpu_sync_ns",nullptr}};
    if(value.at("kind")=="COPY") {
        result["copy_return_tick"]=stamp.copy_return;result["gpu_return_tick"]=stamp.gpu_return;
        result["copy_call_ns"]=steady::ns(stamp.begin,stamp.copy_return);
        result["submit_copy_api_ns"]=result["copy_call_ns"];
        result["gpu_sync_ns"]=steady::ns(stamp.copy_return,stamp.gpu_return);
        result["cpu_sync_ns"]=steady::ns(stamp.gpu_return,stamp.end);
    }
    return result;
}

static json run(Executor &executor,const steady::Request &sampling,const json &request) {
    const auto mode=str(request,"mode"),instrument=str(request,"instrumentation"),memory_mode=str(request,"memory_mode");
    const std::string intervention=request.value("history_intervention",std::string("none"));
    require(intervention=="none"||intervention=="source-read"||intervention=="idle","unknown history intervention");
    require(mode=="full"||mode=="target-hot","unknown history execution mode");
    require(instrument=="bare"||instrument=="stamped"||instrument=="cupti","unknown history instrumentation");
#if defined(PROBE_BACKEND_METAL)
    // CUPTI observes CUDA's streams and this build carries neither. Refusing is
    // the whole reason the request is checked here instead of at a call site:
    // a receipt must never say "cupti" and mean "no records kept".
    require(instrument!="cupti","this build is Metal; the cupti observer is CUDA's. Use bare or stamped");
#endif
    require(sampling.instrumentation=="bare","outer history instrumentation must be bare; use explicit history instrumentation field");
    require(sampling.mode==(mode=="full"?"full":"action"),"sampling mode differs from history execution mode");
    require(mode!="full"||sampling.repeats==8,"full history context requires exactly eight replays per block");
    require(sampling.warmup==20&&sampling.min_ms==500,"history context requires explicit twenty/500ms warmup minima");
    require(executor.program.at("physical").at("threads")==1,"history context requires one GGML CPU thread");
    require(executor.gpu!=nullptr||(mode=="full"&&intervention=="none"&&instrument!="cupti"&&memory_mode=="pageable"),
        "CPU-only control requires full unmodified pageable mode with bare or stamped instrumentation");
    const auto &actions=executor.program.at("actions");require(!actions.empty(),"empty history program");
    size_t target=0;
    if(mode=="target-hot"||intervention!="none") {
        target=static_cast<size_t>(state_response::integer(request.at("target_action_index"),"target index"));
        require(target<actions.size(),"target outside action list");const auto &value=actions.at(target);
        require(value.at("kind")=="COPY"&&value.at("src")=="dram"&&value.at("dst")=="vram","target-hot requires actual original H2D COPY");
        if(mode=="target-hot")require(sampling.target==static_cast<int>(target),"sampling and history target indices differ");
        else require(sampling.target<0,"full sampling target must be null");
    } else require(request.at("target_action_index").is_null()&&sampling.target<0,"unintervened full history target must be null");
    const size_t count=mode=="full"?actions.size():1;
    require(static_cast<uint64_t>(sampling.trials)*sampling.repeats*count<=2000000,"history action output envelope exceeds 2M");
    const auto pool=single_layer::configure_pool(executor,sampling.poll);
    executor.validate();const auto initial=single_layer::capture_verification(executor);
    ArenaRegistration registration(executor,memory_mode);
    json action_semantics=json::array();for(size_t index=0;index<actions.size();++index)action_semantics.push_back(semantic_action(executor,actions.at(index),index));
    auto full=[&](){for(const auto &value:actions)executor.action(value);};
    const auto warm=steady::warm(sampling,full);
    std::vector<Block> blocks(sampling.trials);
    std::vector<ActionStamp> stamps(instrument=="bare"?0:static_cast<size_t>(sampling.trials)*sampling.repeats*count);
    std::vector<Intervention> interventions(intervention=="none"?0:static_cast<size_t>(sampling.trials)*sampling.repeats);
    // The CUPTI observer is compiled in only on a CUDA build. On Metal the
    // instrumentation check above has already refused `cupti`, so there is no
    // observer to hold and nothing to bracket the actions with.
#if !defined(PROBE_BACKEND_METAL)
    std::unique_ptr<physical_capture::Capture> capture;
    if(instrument=="cupti")capture=std::make_unique<physical_capture::Capture>();
#endif
    uint64_t serial=0;
    for(int trial=0;trial<sampling.trials;++trial) {
        auto &block=blocks[trial];block.state=preblock_state(executor.gpu!=nullptr);
        executor.reset();full();
        block.before=cpu_point();block.begin=steady::tick();
        for(int repeat=0;repeat<sampling.repeats;++repeat)for(size_t local=0;local<count;++local) {
            const size_t index=mode=="full"?local:target;
            if(intervention!="none"&&index==target)
                interventions[static_cast<size_t>(trial)*sampling.repeats+repeat]=intervene(executor,actions.at(index),intervention,trial,repeat,index);
            ActionStamp *stamp=instrument=="bare"?nullptr:&stamps[serial];
            if(stamp){stamp->action_index=index;stamp->invocation=repeat;stamp->external_id=serial+1;}
#if !defined(PROBE_BACKEND_METAL)
            if(capture)capture->push(serial+1);
#endif
            action(executor,actions.at(index),stamp);
#if !defined(PROBE_BACKEND_METAL)
            if(capture)capture->pop(serial+1);
#endif
            ++serial;
        }
        block.end=steady::tick();block.after=cpu_point();
    }
#if defined(PROBE_BACKEND_METAL)
    const json activities=json{{"schema","physical-stage-cupti-1"},{"enabled",false},{"records",json::array()},
        {"hardware_accuracy_qualified",false},
        {"reason","this build is Metal; the cupti observer is CUDA's and was refused at the request check"}};
    bool capture_integrity=instrument!="cupti";
#else
    json activities=capture?capture->finish():json{{"schema","physical-stage-cupti-1"},{"enabled",false},{"records",json::array()},
        {"hardware_accuracy_qualified",false}};
    capture.reset();
    bool capture_integrity=instrument!="cupti"||activities.at("all_requested_records_retained").get<bool>();
    if(instrument=="cupti")for(const auto &record:activities.at("records"))
        if(record.at("kind")!="external_correlation"&&!record.at("timestamp_valid").get<bool>())capture_integrity=false;
#endif
    const auto final=single_layer::capture_verification(executor);
    const auto registration_receipt=registration.finish();
    json rows=json::array();size_t cursor=0;
    for(int trial=0;trial<sampling.trials;++trial) {
        const auto &block=blocks[trial];json recorded=json::array();
        if(instrument!="bare")for(int repeat=0;repeat<sampling.repeats;++repeat)for(size_t local=0;local<count;++local) {
            const auto &stamp=stamps[cursor++];recorded.push_back(action_row(stamp,trial,actions.at(stamp.action_index)));
        }
        rows.push_back(json{{"trial",trial},{"block",0},{"begin_tick",block.begin},{"end_tick",block.end},
            {"begin_ns",absolute_ns(block.begin)},{"end_ns",absolute_ns(block.end)},
            {"wall_ns",steady::ns(block.begin,block.end)},{"repeats",sampling.repeats},
            {"actual_stamped",instrument!="bare"},{"preblock_state",block.state},
            {"cpu_accounting",cpu_delta(block.before,block.after)},{"action_intervals",recorded}});
    }
    const size_t dram_bytes=ggml_backend_buffer_get_size(executor.dram_buffer);
    const size_t vram_bytes=executor.vram_buffer?ggml_backend_buffer_get_size(executor.vram_buffer):0;
    size_t graph_bytes=0;for(const auto &entry:executor.graphs)graph_bytes+=entry.second->workspace;
    json intervention_rows=json::array();for(const auto &item:interventions)
        intervention_rows.push_back(json{{"trial",item.trial},{"invocation",item.invocation},{"action_index",item.action_index},
            {"begin_tick",item.begin},{"end_tick",item.end},{"begin_ns",absolute_ns(item.begin)},{"end_ns",absolute_ns(item.end)},
            {"wall_ns",steady::ns(item.begin,item.end)},{"sampled_source_bytes",item.bytes_read},{"read_checksum",item.checksum}});
    return json{{"program_sha256",executor.program.at("program_sha256")},{"program_format",executor.program.at("format_version")},
        {"mode",mode},{"instrumentation",instrument},{"memory_mode",memory_mode},
        {"target_action_index",mode=="target-hot"||intervention!="none"?json(target):json(nullptr)},
        {"history_intervention",intervention},{"intervention_intervals",intervention_rows},
        {"intervention_scope","immediately before chosen target; outside target span, inside timed block; source-read reads one byte per 64 bytes and never writes source; idle requests Sleep(5)"},
        {"intervened_whole_program_usable_for_strategy_accuracy",false},
        {"unintervened_whole_program_usable_for_accuracy",mode=="full"&&intervention=="none"&&instrument=="bare"},
        {"preparation_contract",json{{"reset_before_batch",true},{"untimed_complete_prime_passes",1},
            {"timed_replays_per_batch",sampling.repeats},{"fixed_input_replay",true},
            {"blocks_per_trial",1},{"warmup_iterations_min",sampling.warmup},{"warmup_ms_min",sampling.min_ms},
            {"target_hot_scope",mode=="target-hot"?json("same actual full-program arena and prime; repeated target history differs from full execution"):json(nullptr)}}},
        {"trials",rows},{"actions",actions},{"action_semantics",action_semantics},
        {"warmup",warm},{"threadpool",pool},{"arena_registration",registration_receipt},
        {"known_allocation_bytes",json{{"dram_arena",dram_bytes},{"vram_arena",vram_bytes},{"graph_workspaces",graph_bytes},
            {"total",dram_bytes+vram_bytes+graph_bytes},{"complete_physical_peak",false},{"cuda_private_workspace",nullptr}}},
        {"verification_native",final},{"native_repeat_consistency",state_response::compare_ports(initial.at("ports"),final.at("ports"))},
        {"cupti",activities},{"capture_integrity_pass",capture_integrity},
        {"physical_activity_usable_for_diagnosis",instrument=="cupti"&&capture_integrity},
        {"cupti_scope","capture includes untimed reset/prime activity; parameter diagnostics require explicit timed-action external-ID correlation"},
        {"unchanged_Executor_action_used",instrument=="bare"},
        {"stamped_copy_operation_sequence_matches_Executor",true},
        {"timing_scope","fixed-input host-completed actions; reset, full prime, state/counter queries, verification and serialization excluded"},
        {"copy_call_scope","inclusive GGML tensor-copy API; includes any internal staging and synchronization; submit_copy_api_ns is an alias, not a separate additive phase"},
        {"physical_interval_scope","device memcpy and inclusive runtime/driver/host API intervals overlap and must not be added"},
        {"diagnostic_state_used_as_future_input",false},{"whole_program_time_used_as_parameter",false},
        {"causal_mechanism_identified",false},{"initial_binary_buffers_verified",depth_assets::buffers_verified},{"initial_binary_bytes_verified",depth_assets::total_bytes_verified},{"performance_parameters_fitted",false},{"hardware_accuracy_qualified",false}};
}
}

int main(int argc,char **argv) {try {
    if(argc==2&&std::string(argv[1])=="--self-test") {
        auto result=steady::self_test();result["source_sha256"]=HISTORY_SOURCE_SHA256;
        result["history_probe_hardware_executed"]=false;std::cout<<result.dump()<<'\n';return 0;
    }
    require(argc==4,"usage: h2d-history-probe PROGRAM REQUEST OUTPUT");
    require(!state_response::exists(argv[3]),"history output already exists");
    _putenv_s("NVIDIA_TF32_OVERRIDE","0");_putenv_s("GGML_CUDA_DISABLE_GRAPHS","1");
    std::ifstream pf(std::filesystem::u8path(argv[1])),rf(std::filesystem::u8path(argv[2]));
    require(pf.good()&&rf.good(),"cannot read explicit history program/request");json program,request;pf>>program;rf>>request;
    require(request.at("schema")=="h2d-history-request-1","history request schema differs");
    require(str(request,"source_sha256")==HISTORY_SOURCE_SHA256,"caller source hash differs from compiled history source");
    steady::Request sampling(request.at("sampling"));
    require(sampling.main_mask==4,"history context requires explicit main thread mask4");
    const auto process_affinity=history_probe::apply_process_mask(request);
    steady::OwnThreadAffinity affinity(sampling.main_mask);json result;
    {Executor executor(std::move(program));result=history_probe::run(executor,sampling,request);}
    result.update(json{{"schema","h2d-history-native-1"},{"request",request},{"process_id",single_layer::pid()},
        {"main_thread_id",state_response::main_thread_id()},
        {"source_sha256",HISTORY_SOURCE_SHA256},{"caller_source_sha256_match",true},
        {"main_thread_affinity",affinity.finish()},{"process_affinity",history_probe::finish_process_mask(process_affinity)},
        {"clock",json{{"clock_id",PORTABLE_CLOCK_ID},{"frequency_hz",steady::frequency()}}},
        {"physical_clock",json{{"clock_id",PORTABLE_PHYSICAL_CLOCK_ID},{"clock_origin",CLOCK_ORIGIN},
            {"frequency_hz",steady::frequency()},{"conversion","floor(QPC ticks / frequency * 1e9)"}}},
        {"observed_runtime_flags",json{{"GGML_CUDA_DISABLE_GRAPHS","1"},{"NVIDIA_TF32_OVERRIDE","0"},{"matmul_precision","GGML_PREC_F32"}}}});
    state_response::write_new(argv[3],result);std::cout<<json{{"status","ok"},{"process_id",single_layer::pid()}}.dump()<<'\n';return 0;
}catch(const std::exception &error){std::cerr<<"h2d-history-probe: "<<error.what()<<'\n';return 2;}}
