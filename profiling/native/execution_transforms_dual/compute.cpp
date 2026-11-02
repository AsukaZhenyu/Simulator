// Dual-logical-CPU variant copied from execution_transforms/compute.cpp SHA256:
// de455ca3f5963bc9427496cf560c6cc95044eb920bfecc1f575fa221123b1b92
// Mathematical graph/reference code is unchanged; only affinity and its metadata differ.
// Independent transform-boundary calibration. Shared helpers are source-bound;
// the old binary, source, and diagnostic data are never modified.
#define main matched_legacy_main
#include "../matched_mechanisms/compute.cpp"
#undef main
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

// F32 storage is not a complete arithmetic contract. In the source-bound GGML
// b8705 backend, ggml-cuda/common.cuh:1407 creates a TF32-enabled cuBLAS handle;
// ggml-cuda/ggml-cuda.cu:1395-1428 uses cublasSgemm for F32 tensors. Explicit
// GGML_PREC_F32 plus the externally recorded NVIDIA_TF32_OVERRIDE=0 are required
// for this calibration's intended precision. Precision belongs in profile keys.
static void force_f32_matmul(ggml_cgraph*graph) {
    for(int n=0;n<ggml_graph_n_nodes(graph);++n){auto*node=ggml_graph_node(graph,n);if(node->op==GGML_OP_MUL_MAT)ggml_mul_mat_set_prec(node,GGML_PREC_F32);}
}
static void force_f32_matmul(Session&session) {for(auto&item:session.graphs)force_f32_matmul(item.second->graph);}

struct TransformOptions { Options common; int cpu_intermediate = 0; int dispatch_cpu_id = -1; uint64_t observed_process_mask=0,observed_main_mask=0,observed_main_group=0; };
static TransformOptions transform_options(int argc, char **argv) {
    TransformOptions result; std::vector<std::string> args; bool seen=false;
    std::string family, requested_backend;
    for(int n=1;n+1<argc;n+=2) {
        if(std::string(argv[n])=="--family") family=argv[n+1];
        if(std::string(argv[n])=="--backend") requested_backend=argv[n+1];
    }
    for(int n=0;n<argc;++n) {
        if(n && std::string(argv[n])=="--cpu-intermediate") {
            if(seen || n+1==argc) throw std::runtime_error("missing/duplicate --cpu-intermediate");
            result.cpu_intermediate=integer(argv[++n]);seen=true;
        } else if(n && std::string(argv[n])=="--backend" && family=="attention" && requested_backend=="hybrid") {
            args.push_back(argv[n++]);args.push_back("gpu");
        } else args.push_back(argv[n]);
    }
    if(!seen)throw std::runtime_error("explicit --cpu-intermediate required (0 for attention)");
    std::vector<char*> raw;for(auto &arg:args)raw.push_back(arg.data());
    result.common=parse(static_cast<int>(raw.size()),raw.data());
    result.common.backend=requested_backend;
#if defined(_WIN32)
    const char*dispatch_env=std::getenv("GGML_CALIBRATION_GPU_DISPATCH_CPU_ID");
    if(!dispatch_env)throw std::runtime_error("explicit GGML_CALIBRATION_GPU_DISPATCH_CPU_ID required; no default");
    result.dispatch_cpu_id=integer(dispatch_env);
    if(result.dispatch_cpu_id>=64 || result.dispatch_cpu_id==result.common.cpu_id)throw std::runtime_error("dispatch logical CPU must differ from main CPU and be below 64");
    const DWORD_PTR main_mask=DWORD_PTR(1)<<result.common.cpu_id;
    const DWORD_PTR dispatch_mask=DWORD_PTR(1)<<result.dispatch_cpu_id;
    if(!SetProcessAffinityMask(GetCurrentProcess(),main_mask|dispatch_mask))throw std::runtime_error("cannot apply explicit dual-CPU process affinity");
    if(!SetThreadAffinityMask(GetCurrentThread(),main_mask))throw std::runtime_error("cannot apply explicit main-thread affinity");
    DWORD_PTR observed_process=0,system_mask=0;GROUP_AFFINITY observed_main{};
    if(!GetProcessAffinityMask(GetCurrentProcess(),&observed_process,&system_mask) || observed_process!=(main_mask|dispatch_mask))throw std::runtime_error("dual-CPU process affinity readback differs");
    if(!GetThreadGroupAffinity(GetCurrentThread(),&observed_main) || observed_main.Group!=0 || observed_main.Mask!=main_mask)throw std::runtime_error("main-thread affinity readback differs; only Windows processor group 0 supported");
    result.observed_process_mask=observed_process;result.observed_main_mask=observed_main.Mask;result.observed_main_group=observed_main.Group;
#else
    throw std::runtime_error("dual logical CPU affinity is implemented only on Windows; no silent fallback");
#endif
    if(result.common.backend!="hybrid")throw std::runtime_error("this probe requires explicit --backend hybrid; all strategies are observed together");
    if(family=="ffn" && (result.cpu_intermediate<1 || result.cpu_intermediate>=result.common.i))throw std::runtime_error("FFN split must leave nonempty CPU and GPU shards");
    if(family=="attention" && result.cpu_intermediate!=0)throw std::runtime_error("attention requires cpu-intermediate=0");
    return result;
}

// One persistent dispatch thread, created outside all timed sections. The empty
// handshake is separately observed; it is not silently subtracted from timings.
class Worker {
    std::thread thread;std::mutex mutex;std::condition_variable wake,done;
    bool pending=false,finished=false,stop=false;std::function<void()> task;
    std::exception_ptr error;
public:
    uint64_t observed_affinity_mask=0,observed_group=0;
    explicit Worker(int device,int dispatch_cpu_id) {
        thread=std::thread([this,device,dispatch_cpu_id](){
            try{
#if defined(_WIN32)
                if(dispatch_cpu_id<0 || dispatch_cpu_id>=64 || !SetThreadAffinityMask(GetCurrentThread(),DWORD_PTR(1)<<dispatch_cpu_id))throw std::runtime_error("cannot apply explicit GPU-dispatch-thread affinity");
                GROUP_AFFINITY observed{};
                if(!GetThreadGroupAffinity(GetCurrentThread(),&observed) || observed.Group!=0 || observed.Mask!=(DWORD_PTR(1)<<dispatch_cpu_id))throw std::runtime_error("GPU-dispatch-thread affinity readback differs; only processor group 0 supported");
                observed_affinity_mask=observed.Mask;observed_group=observed.Group;
#else
                throw std::runtime_error("GPU dispatch affinity unavailable on this OS");
#endif
                cuda_ok(cudaSetDevice(device),"worker CUDA device");
            }catch(...){error=std::current_exception();}
            std::unique_lock<std::mutex>lock(mutex);
            for(;;){wake.wait(lock,[this](){return pending||stop;});if(stop)break;
                auto run=task;lock.unlock();try{if(!error)run();}catch(...){error=std::current_exception();}
                lock.lock();pending=false;finished=true;done.notify_one();}
        });
    }
    ~Worker(){{std::lock_guard<std::mutex>lock(mutex);stop=true;}wake.notify_one();thread.join();}
    void start(std::function<void()> operation){{std::lock_guard<std::mutex>lock(mutex);task=std::move(operation);finished=false;pending=true;}wake.notify_one();}
    void wait(){std::unique_lock<std::mutex>lock(mutex);done.wait(lock,[this](){return finished;});if(error)std::rethrow_exception(error);}
};

struct HostTransfer {
    cudaStream_t stream=nullptr;void*pinned=nullptr;size_t bytes=0;
    explicit HostTransfer(size_t n):bytes(n){cuda_ok(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"transfer stream");cuda_ok(cudaMallocHost(&pinned,n),"pinned staging");}
    ~HostTransfer(){if(stream)cudaStreamSynchronize(stream);if(pinned)cudaFreeHost(pinned);if(stream)cudaStreamDestroy(stream);}
    void d2h(ggml_tensor*source,void*destination){if(ggml_nbytes(source)>bytes)throw std::runtime_error("transfer size");cuda_ok(cudaMemcpyAsync(pinned,source->data,ggml_nbytes(source),cudaMemcpyDeviceToHost,stream),"D2H");cuda_ok(cudaStreamSynchronize(stream),"D2H completion");std::memcpy(destination,pinned,ggml_nbytes(source));}
    void h2d(const void*source,ggml_tensor*destination){if(ggml_nbytes(destination)>bytes)throw std::runtime_error("transfer size");std::memcpy(pinned,source,ggml_nbytes(destination));cuda_ok(cudaMemcpyAsync(destination->data,pinned,ggml_nbytes(destination),cudaMemcpyHostToDevice,stream),"H2D");cuda_ok(cudaStreamSynchronize(stream),"H2D completion");}
};

struct ExtraGraphs {
    Session&session;ggml_context*resident=nullptr;ggml_backend_buffer_t buffer=nullptr;
    std::map<std::string,ggml_tensor*> ports;std::map<std::string,std::unique_ptr<Graph>> graphs;
    explicit ExtraGraphs(Session&s):session(s){resident=ggml_init({ggml_tensor_overhead()*16,nullptr,true});if(!resident)throw std::runtime_error("extra resident metadata");}
    ~ExtraGraphs(){ggml_backend_synchronize(session.backend);graphs.clear();if(buffer)ggml_backend_buffer_free(buffer);if(resident)ggml_free(resident);}
    ggml_tensor*add(const std::string&name,int a,int b=1,int c=1){elements(a,b,c);auto*t=ggml_new_tensor_3d(resident,GGML_TYPE_F32,a,b,c);ggml_set_name(t,name.c_str());ggml_set_input(t);ggml_set_output(t);ports[name]=t;return t;}
    void allocate(){buffer=ggml_backend_alloc_ctx_tensors(resident,session.backend);if(!buffer)throw std::runtime_error("extra allocation");ggml_backend_buffer_clear(buffer,0);}
    ggml_context*begin(std::unique_ptr<Graph>&g){g=std::make_unique<Graph>();g->ctx=ggml_init({ggml_tensor_overhead()*160+ggml_graph_overhead_custom(256,false),nullptr,true});if(!g->ctx)throw std::runtime_error("extra graph metadata");g->graph=ggml_new_graph_custom(g->ctx,256,false);return g->ctx;}
    void finish(const std::string&name,std::unique_ptr<Graph>g,ggml_tensor*out){ggml_set_output(out);ggml_build_forward_expand(g->graph,out);
        force_f32_matmul(g->graph);
        for(int n=0;n<ggml_graph_n_nodes(g->graph);++n)if(!ggml_backend_supports_op(session.backend,ggml_graph_node(g->graph,n)))throw std::runtime_error("extra op unsupported; no fallback");
        g->allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(session.backend));if(!g->allocator||!ggml_gallocr_alloc_graph(g->allocator,g->graph))throw std::runtime_error("extra graph allocation");g->internal=ggml_gallocr_get_buffer_size(g->allocator,0);
        if(!session.gpu){g->plan=ggml_graph_plan(g->graph,session.o.threads,session.pool);g->work.resize(g->plan.work_size);g->plan.work_data=g->work.data();}graphs[name]=std::move(g);}
    void compute(const std::string&name){auto&g=*graphs.at(name);if(session.gpu){if(ggml_backend_graph_compute_async(session.backend,g.graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("extra CUDA graph failed");ggml_backend_synchronize(session.backend);}else if(ggml_graph_compute(g.graph,&g.plan)!=GGML_STATUS_SUCCESS)throw std::runtime_error("extra CPU graph failed");}
    std::vector<float>read(const std::string&name){auto*t=ports.at(name);std::vector<float>v(ggml_nelements(t));ggml_backend_tensor_get(t,v.data(),0,ggml_nbytes(t));return v;}
};

static Data shard_data(const Options&o,const Data&full,int start,int width){
    Data result;result["input"]=full.at("input");
    for(auto name:{"gate_weight","up_weight"}){auto&v=result[name];v.resize(elements(o.h,width));std::copy_n(full.at(name).begin()+size_t(start)*o.h,v.size(),v.begin());}
    auto&down=result["down_weight"];down.resize(elements(width,o.h));
    for(int row=0;row<o.h;++row)std::copy_n(full.at("down_weight").begin()+size_t(row)*o.i+start,width,down.begin()+size_t(row)*width);
    result["gate"]=matmul(result["gate_weight"],result["input"],o.h,width,o.q);result["up"]=matmul(result["up_weight"],result["input"],o.h,width,o.q);
    auto&active=result["activated"];active.resize(elements(width,o.q));for(size_t n=0;n<active.size();++n)active[n]=float(result["gate"][n]/(1+std::exp(-double(result["gate"][n])))*result["up"][n]);
    result["output"]=matmul(down,active,width,o.h,o.q);return result;
}

using Times=std::map<std::string,std::vector<int64_t>>;
using Checks=std::map<std::string,Check>;
struct Output {
    Times times;Checks checks;std::map<std::string,std::pair<size_t,bool>> workspace;
    std::map<std::string,std::pair<size_t,std::string>> ports;
    std::map<std::string,int64_t> diagnostic;
    uint64_t observed_worker_mask=0,observed_worker_group=0;
    void inventory(const std::string&prefix,const Session&s,const Data&data){
        for(auto&p:s.graphs)workspace[prefix+p.first]={p.second->internal+p.second->work.size(),s.gpu};
        for(auto&p:s.ports)ports[prefix+p.first]={ggml_nbytes(p.second),digest(data.at(p.first))};
    }
    void extras(const std::string&prefix,const ExtraGraphs&s){for(auto&p:s.graphs)workspace[prefix+p.first]={p.second->internal+p.second->work.size(),s.session.gpu};}
};

static Output ffn_transform(const TransformOptions&t){
    const Options&o=t.common;Output result;auto&times=result.times;auto&checks=result.checks;
    const Data full=make_data(o);Options c=o,g=o;c.i=t.cpu_intermediate;g.i=o.i-c.i;
    const Data cd=shard_data(o,full,0,c.i),gd=shard_data(o,full,c.i,g.i);
    Session cpu(c,false,cd),gpu(g,true,gd),all(o,true,full);
    force_f32_matmul(cpu);force_f32_matmul(gpu);force_f32_matmul(all);
    ExtraGraphs merge(cpu);auto*partial=merge.add("gpu_partial",o.h,o.q);auto*output=merge.add("sum_output",o.h,o.q);merge.allocate();
    std::unique_ptr<Graph> graph;auto*ctx=merge.begin(graph);merge.finish("sum",std::move(graph),ggml_cpy(ctx,ggml_add(ctx,cpu.ports.at("output"),partial),output));
    size_t maxbytes=ggml_nbytes(partial);for(auto name:{"gate_weight","up_weight","down_weight"})maxbytes=std::max(maxbytes,ggml_nbytes(gpu.ports.at(name)));
    HostTransfer transfer(maxbytes);Worker worker(o.device,t.dispatch_cpu_id);
    for(auto name:{"gate","up","activation","down"}){
        times["cpu/shard_"+std::string(name)]=measure(o,[&](){cpu.compute(name);});
        times["gpu/shard_"+std::string(name)]=measure(o,[&](){gpu.compute(name);});
        times["gpu/full_"+std::string(name)]=measure(o,[&](){all.compute(name);});
    }
    auto copy_shard=[&](){transfer.d2h(gpu.ports.at("output"),partial->data);};
    times["hybrid/output_d2h"]=measure(o,copy_shard);
    times["gpu/output_d2h"]=measure(o,[&](){transfer.d2h(all.ports.at("output"),partial->data);});
    times["cpu/sum"]=measure(o,[&](){merge.compute("sum");});
    auto preload=[&](){for(auto name:{"gate_weight","up_weight","down_weight"})transfer.h2d(gd.at(name).data(),gpu.ports.at(name));};
    times["hybrid/preload_weights"]=measure(o,preload);
    times["hybrid/dispatch_barrier"]=measure(o,[&](){worker.start([](){});worker.wait();});
    auto parallel=[&](){worker.start([&](){gpu.serial();});cpu.serial();worker.wait();};
    times["hybrid/parallel_branches"]=measure(o,parallel);
    auto mixed=[&](){parallel();copy_shard();merge.compute("sum");};
    times["gpu/resident"]=measure(o,[&](){all.serial();transfer.d2h(all.ports.at("output"),partial->data);});
    checks["gpu/resident/output"]=compare(merge.read("gpu_partial"),full.at("output"));
    times["hybrid/resident"]=measure(o,mixed);checks["hybrid/resident/output"]=compare(merge.read("sum_output"),full.at("output"));
    times["hybrid/preload"]=measure(o,[&](){preload();mixed();});checks["hybrid/preload/output"]=compare(merge.read("sum_output"),full.at("output"));
    for(auto name:{"gate","up","activated","output"}){checks["cpu/shard/"+std::string(name)]=compare(cpu.read(name),cd.at(name));checks["gpu/shard/"+std::string(name)]=compare(gpu.read(name),gd.at(name));checks["gpu/full/"+std::string(name)]=compare(all.read(name),full.at(name));}
    // Untimed evidence: intervals bracket completed work on the two host
    // threads, including dispatch/wait. They are not CUDA kernel intervals.
    int64_t cb=0,ce=0,gb=0,ge=0;const auto origin=Clock::now();
    auto stamp=[&](){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-origin).count();};
    worker.start([&](){gb=stamp();gpu.serial();ge=stamp();});cb=stamp();cpu.serial();ce=stamp();worker.wait();
    result.diagnostic={{"cpu_begin_ns",cb},{"cpu_end_ns",ce},{"gpu_host_begin_ns",gb},{"gpu_host_end_ns",ge},{"completed_work_bracket_overlap_ns",std::max<int64_t>(0,std::min(ce,ge)-std::max(cb,gb))}};
    result.observed_worker_mask=worker.observed_affinity_mask;result.observed_worker_group=worker.observed_group;
    result.inventory("cpu/shard/",cpu,cd);result.inventory("gpu/shard/",gpu,gd);result.inventory("gpu/full/",all,full);result.extras("cpu/",merge);return result;
}

static void attention_graphs(ExtraGraphs&extra){
    Session&s=extra.session;const Options&o=s.o;int len=o.hist+o.q;
    auto*scores=extra.add("scores",len,o.q,o.heads),*probs=extra.add("probabilities",len,o.q,o.heads);extra.allocate();
    auto get=[&](const char*n){return s.ports.at(n);};std::unique_ptr<Graph>g;
    auto*ctx=extra.begin(g);auto*k=ggml_cpy(ctx,get("key_cache"),get("key_appended"));
    k=ggml_set_inplace(ctx,k,get("new_key"),k->nb[1],k->nb[2],k->nb[3],size_t(o.hist)*o.d*sizeof(float));
    auto*scaled=ggml_scale(ctx,ggml_mul_mat(ctx,k,get("query")),1/std::sqrt(float(o.d)));
    auto*out=ggml_cpy(ctx,scaled,scores);extra.finish("qk",std::move(g),out);
    ctx=extra.begin(g);out=ggml_cpy(ctx,ggml_soft_max(ctx,ggml_diag_mask_inf(ctx,scores,o.hist)),probs);extra.finish("softmax",std::move(g),out);
    ctx=extra.begin(g);auto*v=ggml_cpy(ctx,get("value_cache"),get("value_appended"));
    v=ggml_set_inplace(ctx,v,get("new_value"),v->nb[1],v->nb[2],v->nb[3],size_t(o.hist)*o.d*sizeof(float));
    // PV owns externally observable append completion. QK's key_appended is a
    // private materialization scratch; old immutable cache inputs are unchanged.
    auto*ka=ggml_cpy(ctx,get("key_cache"),get("key_appended"));ka=ggml_set_inplace(ctx,ka,get("new_key"),ka->nb[1],ka->nb[2],ka->nb[3],size_t(o.hist)*o.d*sizeof(float));
    ggml_build_forward_expand(g->graph,ka);
    auto*attended=ggml_mul_mat(ctx,ggml_cont(ctx,ggml_transpose(ctx,v)),probs);
    out=ggml_cpy(ctx,ggml_reshape_2d(ctx,ggml_cont(ctx,ggml_permute(ctx,attended,0,2,1,3)),o.h,o.q),get("context"));extra.finish("pv",std::move(g),out);
}

static Output attention_transform(const TransformOptions&t){
    const Options&o=t.common;Output result;auto&times=result.times;auto&checks=result.checks;
    Data data=make_data(o);Session cpu(o,false,data),gpu(o,true,data);force_f32_matmul(cpu);force_f32_matmul(gpu);ExtraGraphs c(cpu),g(gpu);attention_graphs(c);attention_graphs(g);
    const size_t output_bytes=ggml_nbytes(gpu.ports.at("output"));
    HostTransfer transfer(std::max(output_bytes,ggml_nbytes(g.ports.at("probabilities"))));std::vector<float>delivered(size_t(o.h)*o.q);
    for(auto name:{"qk","softmax","pv"}){times["cpu/"+std::string(name)]=measure(o,[&](){c.compute(name);});times["gpu/"+std::string(name)]=measure(o,[&](){g.compute(name);});}
    times["gpu/output"]=measure(o,[&](){gpu.compute("attention_output");});
    times["hybrid/probabilities_h2d"]=measure(o,[&](){transfer.h2d(c.ports.at("probabilities")->data,g.ports.at("probabilities"));});
    times["gpu/output_d2h"]=measure(o,[&](){transfer.d2h(gpu.ports.at("output"),delivered.data());});
    std::vector<float>scores(size_t(o.hist+o.q)*o.q*o.heads),probabilities(scores.size());
    const int len=o.hist+o.q;for(int head=0;head<o.heads;++head)for(int token=0;token<o.q;++token){
        const int kh=head/(o.heads/o.kvh);double largest=-1e100,total=0;
        for(int key=0;key<len;++key){double dot=0;for(int j=0;j<o.d;++j)dot+=double(data.at("query")[size_t(head*o.q+token)*o.d+j])*data.at("key_appended")[size_t(kh*len+key)*o.d+j];
            const size_t index=size_t(head*o.q+token)*len+key;scores[index]=float(dot/std::sqrt(double(o.d)));if(key<=o.hist+token)largest=std::max(largest,double(scores[index]));}
        for(int key=0;key<=o.hist+token;++key)total+=std::exp(double(scores[size_t(head*o.q+token)*len+key])-largest);
        for(int key=0;key<len;++key)probabilities[size_t(head*o.q+token)*len+key]=key<=o.hist+token?float(std::exp(double(scores[size_t(head*o.q+token)*len+key])-largest)/total):0;
    }
    c.compute("qk");c.compute("softmax");g.compute("qk");g.compute("softmax");
    checks["cpu/qk/scores"]=compare(c.read("scores"),scores);checks["gpu/qk/scores"]=compare(g.read("scores"),scores);
    checks["cpu/softmax/probabilities"]=compare(c.read("probabilities"),probabilities);checks["gpu/softmax/probabilities"]=compare(g.read("probabilities"),probabilities);
    auto all=[&](){g.compute("qk");g.compute("softmax");g.compute("pv");gpu.compute("attention_output");transfer.d2h(gpu.ports.at("output"),delivered.data());};
    auto mixed=[&](){c.compute("qk");c.compute("softmax");transfer.h2d(c.ports.at("probabilities")->data,g.ports.at("probabilities"));g.compute("pv");gpu.compute("attention_output");transfer.d2h(gpu.ports.at("output"),delivered.data());};
    times["gpu/resident"]=measure(o,all);checks["gpu/resident/output"]=compare(delivered,data.at("output"));
    times["hybrid/resident"]=measure(o,mixed);checks["hybrid/resident/output"]=compare(delivered,data.at("output"));
    checks["gpu/pv/context"]=compare(gpu.read("context"),data.at("context"));
    checks["gpu/pv/key_appended"]=compare(gpu.read("key_appended"),data.at("key_appended"));checks["gpu/pv/value_appended"]=compare(gpu.read("value_appended"),data.at("value_appended"));
    checks["cpu/cache_unchanged"]=compare(cpu.read("key_cache"),data.at("key_cache"));checks["gpu/cache_unchanged"]=compare(gpu.read("key_cache"),data.at("key_cache"));
    checks["cpu/value_cache_unchanged"]=compare(cpu.read("value_cache"),data.at("value_cache"));checks["gpu/value_cache_unchanged"]=compare(gpu.read("value_cache"),data.at("value_cache"));
    c.compute("pv");checks["cpu/pv/context"]=compare(cpu.read("context"),data.at("context"));
    result.inventory("cpu/",cpu,data);result.inventory("gpu/",gpu,data);result.extras("cpu/",c);result.extras("gpu/",g);return result;
}

static int transform_run(const TransformOptions&t){
    const Options&o=t.common;Output r=o.family=="ffn"?ffn_transform(t):attention_transform(t);
    bool passed=true;for(auto&p:r.checks)passed=passed&&p.second.passed;
    std::cout<<std::setprecision(12)<<"{\"schema\":\"execution-transform-native-0.1\",\"family\":\""<<o.family<<"\",\"backend\":\"hybrid\",\"dtype\":\"f32\",\"hidden\":"<<o.h<<",\"intermediate\":"<<o.i<<",\"cpu_intermediate\":"<<t.cpu_intermediate<<",\"tokens\":"<<o.q<<",\"history\":"<<o.hist<<",\"heads\":"<<o.heads<<",\"kv_heads\":"<<o.kvh<<",\"head_dim\":"<<o.d<<",\"threads\":"<<o.threads<<",\"cpu_id\":"<<o.cpu_id<<",\"device\":"<<o.device<<",\"seed\":"<<o.seed<<",\"samples\":"<<o.samples<<",\"warmup\":"<<o.warmup<<",\"min_warmup_ms\":"<<o.min_warmup_ms<<",\"cuda_graphs\":\"disabled\",\"timer\":\"steady_clock_completed_work\",\"boundary\":{\"input_location\":\"all_required_immutable_inputs_already_resident_in_both_cpu_and_gpu\",\"output_location\":\"dram\",\"merge_device\":\"cpu\",\"weights\":\"generated_f32_not_model_weights\",\"timing_includes\":\"dispatch_sync_pinned_staging_memcpy_and_explicit_delivery\",\"timing_excludes\":\"allocation_initialization_numerical_validation\",\"ffn_contract\":\"compact_selected_rows_gate_up_swiglu_down_no_norm_or_residual\",\"ffn_partition\":\"gate_up_output_columns_and_down_input_rows_materialized_before_timing\",\"attention_contract\":\"post_RoPE_QKV_to_causal_GQA_QK_softmax_PV_sigmoid_gate_O_residual\",\"kv_materialization\":\"QK_private_K_scratch_then_PV_KV_append_outputs; old_cache_inputs_immutable\",\"preload_contract\":\"every_iteration_copy_three_GPU_shard_weights_from_DRAM_via_pinned_staging_before_execution; no_claim_of_cold_cache\",\"cpu_affinity\":\"Windows_process_two_explicit_logical_CPUs_main_thread_cpu_id_GPU_worker_dispatch_cpu_id;_other_OS_rejected\",\"sample_order\":\"grouped_components_then_strategies_no_outlier_removal\"},\"wall_ns\":{";
    bool first=true;for(auto&p:r.times){std::cout<<(first?"":",")<<'"'<<p.first<<"\":";emit_array(p.second);first=false;}
    std::cout<<"},\"ports\":{";first=true;for(auto&p:r.ports){std::cout<<(first?"":",")<<'"'<<p.first<<"\":{\"bytes\":"<<p.second.first<<",\"input_fixture_fnv1a64\":\""<<p.second.second<<"\"}";first=false;}
    std::cout<<"},\"workspaces\":{";first=true;for(auto&p:r.workspace){std::cout<<(first?"":",")<<'"'<<p.first<<"\":{\"known_gallocr_plus_cplan_bytes\":"<<p.second.first<<",\"workspace_bytes\":";if(p.second.second)std::cout<<"null,\"source\":\"unknown_private_cuda_scratch\"}";else std::cout<<p.second.first<<",\"source\":\"gallocr_plus_explicit_cplan\"}";first=false;}
    std::cout<<"},\"checks\":{";first=true;for(auto&p:r.checks){auto&c=p.second;std::cout<<(first?"":",")<<'"'<<p.first<<"\":{\"passed\":"<<(c.passed?"true":"false")<<",\"max_abs_error\":"<<c.maxabs<<",\"max_relative_error\":"<<c.maxrel<<",\"elements\":"<<c.elements<<",\"atol\":0.00001,\"rtol\":0.0001}";first=false;}
    std::cout<<"},\"diagnostic_completed_work_brackets\":{\"scope\":\"untimed_single_execution_host_brackets_include_dispatch_and_wait_not_kernel_intervals\",\"kernel_overlap_proved\":false,\"times_ns\":{";first=true;for(auto&p:r.diagnostic){std::cout<<(first?"":",")<<'"'<<p.first<<"\":"<<p.second;first=false;}
    const char*tf32=std::getenv("NVIDIA_TF32_OVERRIDE");
    std::cout<<"}},\"dispatch_cpu_id\":"<<t.dispatch_cpu_id<<",\"affinity_variant\":\"dual_logical_cpu_explicit\"";
    std::cout<<",\"observed_affinity\":{\"process_mask\":"<<t.observed_process_mask<<",\"main_thread_mask\":"<<t.observed_main_mask<<",\"main_thread_group\":"<<t.observed_main_group<<",\"dispatch_worker_created\":"<<(o.family=="ffn"?"true":"false")<<",\"dispatch_thread_mask\":";
    if(o.family=="ffn")std::cout<<r.observed_worker_mask;else std::cout<<"null";
    std::cout<<",\"dispatch_thread_group\":";if(o.family=="ffn")std::cout<<r.observed_worker_group;else std::cout<<"null";std::cout<<'}';
    std::cout<<",\"matmul_precision\":\"GGML_PREC_F32_explicit\",\"nvidia_tf32_override\":";
    if(!tf32)std::cout<<"null";else if(std::string(tf32)=="0"||std::string(tf32)=="1")std::cout<<'"'<<tf32<<'"';else std::cout<<"\"unrecognized\"";
    std::cout<<",\"all_checks_passed\":"<<(passed?"true":"false")<<",\"gpu_capacity_proved\":false,\"control_overhead_separately_identified\":false}\n";return passed?0:2;
}
int main(int argc,char**argv){try{return transform_run(transform_options(argc,argv));}catch(const std::exception&e){std::cerr<<"execution-transform-probe: "<<e.what()<<'\n';return 1;}}
