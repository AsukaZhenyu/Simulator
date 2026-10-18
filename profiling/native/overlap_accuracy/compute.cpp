// New CE prefetch/compute experiment. All historical helpers remain immutable.
#include "context_generated.inc"

struct Prefetch {
    cudaStream_t stream=nullptr;cudaEvent_t begin=nullptr,end=nullptr;
    void*host=nullptr,*device=nullptr;size_t bytes;
    Prefetch(size_t count,uint32_t seed):bytes(count){
        cuda_ok(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"prefetch nonblocking stream");
        cuda_ok(cudaEventCreate(&begin),"copy begin event");cuda_ok(cudaEventCreate(&end),"copy end event");
        cuda_ok(cudaMallocHost(&host,bytes),"page-locked payload");cuda_ok(cudaMalloc(&device,bytes),"distinct next-payload allocation");
        auto*p=static_cast<uint32_t*>(host);for(size_t n=0;n<bytes/4;++n)p[n]=0x3e000000U|uint32_t((n*2654435761ULL+seed)&0x007fffffU);
    }
    ~Prefetch(){if(stream)cudaStreamSynchronize(stream);if(device)cudaFree(device);if(host)cudaFreeHost(host);if(begin)cudaEventDestroy(begin);if(end)cudaEventDestroy(end);if(stream)cudaStreamDestroy(stream);}
    void enqueue(bool diagnostic){if(diagnostic)cuda_ok(cudaEventRecord(begin,stream),"copy begin");cuda_ok(cudaMemcpyAsync(device,host,bytes,cudaMemcpyHostToDevice,stream),"H2D prefetch");if(diagnostic)cuda_ok(cudaEventRecord(end,stream),"copy end");}
    void wait(){cuda_ok(cudaStreamSynchronize(stream),"completed prefetch");}
    int64_t event_ns(){float ms=0;cuda_ok(cudaEventElapsedTime(&ms,begin,end),"copy event elapsed");return int64_t(std::llround(double(ms)*1000000));}
    bool check(){std::vector<uint8_t>v(bytes);cuda_ok(cudaMemcpy(v.data(),device,bytes,cudaMemcpyDeviceToHost),"untimed payload validation");return std::memcmp(v.data(),host,bytes)==0;}
};
struct OverlapTrace {int64_t total=0,copy_a_submit_begin=0,copy_a_submit_end=0,copy_b_submit_begin=-1,copy_b_submit_end=-1,compute_begin=0,compute_end=0,join_begin=0,join_end=0,event_a=0,event_b=-1;};
static int64_t stamp(){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}
static bool disjoint(const void*a,size_t an,const void*b,size_t bn){auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);return x+an<=y||y+bn<=x;}
struct OverlapOptions {Options common;size_t payload=0;bool validate_only=false;};
static OverlapOptions overlap_options(int argc,char**argv){
    OverlapOptions result;std::vector<std::string>args;bool seen=false,mode_seen=false;
    for(int n=0;n<argc;++n){if(n&&std::string(argv[n])=="--mode"){if(mode_seen||n+1==argc)throw std::runtime_error("explicit unique mode required");std::string mode=argv[++n];if(mode!="measure"&&mode!="validate")throw std::runtime_error("mode must be measure or validate");result.validate_only=mode=="validate";mode_seen=true;}
        else if(n&&std::string(argv[n])=="--payload-bytes"){if(seen||n+1==argc)throw std::runtime_error("explicit unique payload bytes required");auto value=std::string(argv[++n]);size_t used=0;auto bytes=std::stoull(value,&used);if(used!=value.size()||bytes<1024*1024||bytes>64*1024*1024||bytes%4)throw std::runtime_error("payload budget 1..64 MiB, F32 aligned");result.payload=size_t(bytes);seen=true;}else args.emplace_back(argv[n]);}
    if(!seen||!mode_seen)throw std::runtime_error("explicit --payload-bytes and --mode required");std::vector<char*>raw;for(auto&s:args)raw.push_back(s.data());result.common=parse(int(raw.size()),raw.data());
    if(result.common.family!="ffn"||result.common.backend!="gpu"||result.common.threads!=1)throw std::runtime_error("only explicit resident GPU F32 FFN/one host CPU thread supported");return result;
}
static int overlap_run(const OverlapOptions&options){
    const Options&o=options.common;const Data data=make_data(o);Session current(o,true,data);force_f32_matmul(current);
    Prefetch a(options.payload,o.seed+17),b(options.payload,o.seed+31);
    bool storage_disjoint=disjoint(a.device,a.bytes,b.device,b.bytes);
    for(auto&p:current.ports)storage_disjoint=storage_disjoint&&disjoint(a.device,a.bytes,p.second->data,ggml_nbytes(p.second))&&disjoint(b.device,b.bytes,p.second->data,ggml_nbytes(p.second));
    if(!storage_disjoint)throw std::runtime_error("prefetch/current tensor storage aliases");
    Times times,overhead;Checks checks;std::map<std::string,std::vector<OverlapTrace>>traces;
    if(!options.validate_only){times["copy/a"]=measure(o,[&](){a.enqueue(false);a.wait();});times["copy/b"]=measure(o,[&](){b.enqueue(false);b.wait();});
    times["compute/current_ffn"]=measure(o,[&](){current.compute("joint");});}
    auto execute=[&](bool parallel,bool two,OverlapTrace*trace){
        const int64_t origin=trace?stamp():0;
        if(trace)trace->copy_a_submit_begin=stamp()-origin;a.enqueue(trace!=nullptr);if(trace)trace->copy_a_submit_end=stamp()-origin;
        if(!parallel)a.wait();
        if(two){if(trace)trace->copy_b_submit_begin=stamp()-origin;b.enqueue(trace!=nullptr);if(trace)trace->copy_b_submit_end=stamp()-origin;if(!parallel)b.wait();}
        if(trace)trace->compute_begin=stamp()-origin;current.compute("joint");if(trace)trace->compute_end=stamp()-origin;
        if(trace)trace->join_begin=stamp()-origin;a.wait();if(two)b.wait();if(trace){trace->join_end=stamp()-origin;trace->total=stamp()-origin;trace->event_a=a.event_ns();if(two)trace->event_b=b.event_ns();}
    };
    for(bool two:{false,true})for(bool parallel:{false,true}){
        const std::string name=std::string(parallel?"overlap/":"serial/")+(two?"two_copy_compute":"one_copy_compute");
        if(options.validate_only)execute(parallel,two,nullptr);else times[name]=measure(o,[&](){execute(parallel,two,nullptr);});
        checks[name+"/output"]=compare(current.read("output"),data.at("output"));
        if(!a.check()||(two&&!b.check()))throw std::runtime_error("untimed completed prefetch payload check failed: "+name);
        if(!options.validate_only){auto&entries=traces[name];for(int n=0;n<o.samples;++n){OverlapTrace trace;execute(parallel,two,&trace);entries.push_back(trace);}}
        checks[name+"/diagnostic_output"]=compare(current.read("output"),data.at("output"));
        if(!a.check()||(two&&!b.check()))throw std::runtime_error("untimed diagnostic payload check failed: "+name);
    }
    for(auto name:{"gate_weight","up_weight","down_weight","input"})checks[std::string("current_inputs_unchanged/")+name]=compare(current.read(name),data.at(name));
    if(!options.validate_only){overhead["clock_pair"]=measure(o,[](){volatile auto x=stamp();volatile auto y=stamp();(void)x;(void)y;});
    overhead["empty_cuda_event_pair_completed"]=measure(o,[&](){cuda_ok(cudaEventRecord(a.begin,a.stream),"empty event begin");cuda_ok(cudaEventRecord(a.end,a.stream),"empty event end");a.wait();});}
    bool numeric=true;for(auto&p:checks)numeric=numeric&&p.second.passed;
    auto emit_times=[](const Times&entries){bool first=true;std::cout<<'{';for(auto&p:entries){std::cout<<(first?"":",")<<'"'<<p.first<<"\":";emit_array(p.second);first=false;}std::cout<<'}';};
    std::cout<<std::setprecision(12)<<"{\"schema\":\"overlap-accuracy-native-0.1\",\"hidden\":"<<o.h<<",\"intermediate\":"<<o.i<<",\"tokens\":"<<o.q<<",\"payload_bytes\":"<<options.payload<<",\"seed\":"<<o.seed<<",\"samples\":"<<o.samples<<",\"warmup\":"<<o.warmup<<",\"min_warmup_ms\":"<<o.min_warmup_ms<<",\"cpu_id\":"<<o.cpu_id<<",\"device\":"<<o.device<<",\"threads\":"<<o.threads<<",\"source_binding\":{\"probe_sha256\":\""<<OVERLAP_PROBE_SHA<<"\",\"context_sha256\":\""<<OVERLAP_CONTEXT_SHA<<"\",\"matched_sha256\":\""<<OVERLAP_MATCHED_SHA<<"\"},\"wall_ns\":";
    emit_times(times);std::cout<<",\"mode\":\""<<(options.validate_only?"validate":"measure")<<"\",\"instrumentation_overhead_ns\":";emit_times(overhead);std::cout<<",\"diagnostic_host_and_copy_events\":{";bool first=true;
    for(auto&p:traces){std::cout<<(first?"":",")<<'"'<<p.first<<"\":[";first=false;bool initial=true;for(auto&t:p.second){std::cout<<(initial?"":",")<<"{\"total_ns\":"<<t.total<<",\"copy_a_submit_begin_ns\":"<<t.copy_a_submit_begin<<",\"copy_a_submit_end_ns\":"<<t.copy_a_submit_end<<",\"copy_b_submit_begin_ns\":"<<t.copy_b_submit_begin<<",\"copy_b_submit_end_ns\":"<<t.copy_b_submit_end<<",\"compute_begin_ns\":"<<t.compute_begin<<",\"compute_end_ns\":"<<t.compute_end<<",\"join_begin_ns\":"<<t.join_begin<<",\"join_end_ns\":"<<t.join_end<<",\"copy_a_event_elapsed_ns\":"<<t.event_a<<",\"copy_b_event_elapsed_ns\":"<<t.event_b<<'}';initial=false;}std::cout<<']';}
    std::cout<<"},\"checks\":{";first=true;for(auto&p:checks){auto&c=p.second;std::cout<<(first?"":",")<<'"'<<p.first<<"\":{\"passed\":"<<(c.passed?"true":"false")<<",\"max_abs_error\":"<<c.maxabs<<",\"max_relative_error\":"<<c.maxrel<<",\"elements\":"<<c.elements<<'}';first=false;}
    std::cout<<"},\"prefetch_payload_checks_passed\":true,\"prefetch_destinations_disjoint_from_each_other_and_all_current_ports\":"<<(storage_disjoint?"true":"false")<<",\"current_output_location\":\"vram\",\"prefetch_destination_location\":\"vram\",\"prefetch_source\":\"preinitialized_page_locked_host_memory\",\"completed_work_timer\":\"steady_clock\",\"matmul_precision\":\"GGML_PREC_F32\",\"nvidia_tf32_override\":\""<<(std::getenv("NVIDIA_TF32_OVERRIDE")?std::getenv("NVIDIA_TF32_OVERRIDE"):"unset")<<"\",\"cuda_graphs\":\"disabled\",\"known_gallocr_workspace_bytes\":"<<current.graphs.at("joint")->internal<<",\"complete_cuda_workspace_bytes\":null,\"physical_capacity_verified\":false,\"kernel_overlap_proved\":false,\"copy_events_prove_kernel_overlap\":false,\"instrumentation_overhead_subtracted\":false,\"all_checks_passed\":"<<(numeric?"true":"false")<<"}\n";return numeric?0:2;
}
int main(int argc,char**argv){try{return overlap_run(overlap_options(argc,argv));}catch(const std::exception&e){std::cerr<<"overlap-accuracy-probe: "<<e.what()<<'\n';return 1;}}
