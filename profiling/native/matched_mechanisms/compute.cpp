// Independent F32 resident stage / chain / explicit CE offload measurements.
// No model loading, hidden defaults, GPU fallback, or inferred workspace.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#if defined(MATCHED_HAVE_CUDA)
#include "ggml-cuda.h"
#include <cuda_runtime_api.h>
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif

using Clock = std::chrono::steady_clock;
struct Options {
    std::string family, backend;
    int h=0, i=0, q=0, hist=0, heads=0, kvh=0, d=0, threads=0, cpu_id=0;
    int warmup=0, samples=0, device=0, min_warmup_ms=0;
    uint32_t seed=0;
};
static int integer(const std::string &v) {
    size_t n=0; auto x=std::stoll(v,&n);
    if(n!=v.size() || x<0 || x>1000000) throw std::runtime_error("invalid integer");
    return static_cast<int>(x);
}
static Options parse(int argc, char **argv) {
    Options o; std::set<std::string> seen;
    const std::set<std::string> keys={"--family","--backend","--hidden","--intermediate","--tokens",
        "--history","--heads","--kv-heads","--head-dim","--threads","--cpu-id","--warmup",
        "--samples","--seed","--device","--min-warmup-ms","--dtype","--cuda-graphs"};
    for(int a=1;a<argc;a+=2) {
        std::string k=argv[a];
        if(a+1==argc || !keys.count(k) || !seen.insert(k).second) throw std::runtime_error("missing, duplicate, or unknown option: "+k);
        std::string v=argv[a+1];
        if(k=="--family") o.family=v;
        else if(k=="--backend") o.backend=v;
        else if(k=="--hidden") o.h=integer(v);
        else if(k=="--intermediate") o.i=integer(v);
        else if(k=="--tokens") o.q=integer(v);
        else if(k=="--history") o.hist=integer(v);
        else if(k=="--heads") o.heads=integer(v);
        else if(k=="--kv-heads") o.kvh=integer(v);
        else if(k=="--head-dim") o.d=integer(v);
        else if(k=="--threads") o.threads=integer(v);
        else if(k=="--cpu-id") o.cpu_id=integer(v);
        else if(k=="--warmup") o.warmup=integer(v);
        else if(k=="--samples") o.samples=integer(v);
        else if(k=="--seed") o.seed=static_cast<uint32_t>(integer(v));
        else if(k=="--device") o.device=integer(v);
        else if(k=="--min-warmup-ms") o.min_warmup_ms=integer(v);
        else if(k=="--dtype" && v!="f32") throw std::runtime_error("only explicit f32 supported");
        else if(k=="--cuda-graphs" && v!="disabled") throw std::runtime_error("only explicit disabled CUDA graphs supported");
    }
    if(seen!=keys) throw std::runtime_error("every documented option is required");
    if(o.family!="ffn" && o.family!="attention") throw std::runtime_error("unknown family");
    if(o.backend!="cpu" && o.backend!="gpu" && o.backend!="hybrid") throw std::runtime_error("unknown backend");
    if(o.h<1 || o.h>1024 || o.q<1 || o.q>128 || o.samples<3 || o.samples>10000 || o.threads<1 || o.threads>GGML_MAX_N_THREADS || o.hist>4096) throw std::runtime_error("unsupported explicit probe budget");
    if(o.family=="ffn" && (o.i<1 || o.i>4096 || o.heads || o.kvh || o.d || o.hist)) throw std::runtime_error("FFN geometry invalid");
    if(o.family=="attention" && (o.i || o.heads<1 || o.kvh<1 || o.d<1 || o.heads%o.kvh || o.heads*o.d!=o.h || o.backend=="hybrid")) throw std::runtime_error("attention geometry / hybrid unsupported");
#if !defined(MATCHED_HAVE_CUDA)
    if(o.backend!="cpu") throw std::runtime_error("CUDA was not built; no CPU fallback");
#endif
#if defined(_WIN32)
    if(o.cpu_id>=64 || !SetProcessAffinityMask(GetCurrentProcess(), DWORD_PTR(1)<<o.cpu_id)) throw std::runtime_error("explicit CPU affinity cannot be applied");
    if(o.threads!=1) throw std::runtime_error("single CPU-id contract requires threads=1");
    if(_putenv_s("GGML_CUDA_DISABLE_GRAPHS","1")) throw std::runtime_error("cannot disable CUDA graphs");
#else
    // Portable CPU execution remains available, but affinity must be observed externally.
    if(setenv("GGML_CUDA_DISABLE_GRAPHS","1",1)) throw std::runtime_error("cannot disable CUDA graphs");
#endif
    return o;
}
static size_t elements(int64_t a,int64_t b=1,int64_t c=1) {
    if(a<1 || b<1 || c<1 || a*b*c>100000000) throw std::runtime_error("unsupported allocation budget");
    return size_t(a*b*c);
}
static void emit_array(const std::vector<int64_t>&v) { std::cout<<'[';for(size_t n=0;n<v.size();++n)std::cout<<(n?",":"")<<v[n];std::cout<<']'; }
static std::string digest(const std::vector<float>&v) {
    uint64_t hash=14695981039346656037ULL;
    for(size_t n=0;n<v.size()*sizeof(float);++n) {hash^=reinterpret_cast<const uint8_t*>(v.data())[n];hash*=1099511628211ULL;}
    std::ostringstream s;s<<std::hex<<std::setfill('0')<<std::setw(16)<<hash;return s.str();
}
using Data=std::map<std::string,std::vector<float>>;
static std::vector<float> matmul(const std::vector<float>&w,const std::vector<float>&x,int cols,int rows,int tokens) {
    std::vector<float> out(elements(rows,tokens));
    for(int t=0;t<tokens;++t)for(int r=0;r<rows;++r) {
        double sum=0;for(int c=0;c<cols;++c)sum+=double(w[size_t(r)*cols+c])*x[size_t(t)*cols+c];out[size_t(t)*rows+r]=float(sum);
    }return out;
}
static Data make_data(const Options&o) {
    Data d;std::mt19937 rng(o.seed);
    auto add=[&](const std::string&name,size_t n,double scale) {auto&v=d[name];v.resize(n);for(auto&x:v)x=float((double(rng())/4294967295.0-.5)*.83*scale/1.37);};
    if(o.family=="ffn") {
        add("input",elements(o.h,o.q),1);add("gate_weight",elements(o.h,o.i),1/std::sqrt(double(o.h)));
        add("up_weight",elements(o.h,o.i),1/std::sqrt(double(o.h)));add("down_weight",elements(o.i,o.h),1/std::sqrt(double(o.i)));
        d["gate"]=matmul(d["gate_weight"],d["input"],o.h,o.i,o.q);d["up"]=matmul(d["up_weight"],d["input"],o.h,o.i,o.q);
        auto&v=d["activated"];v.resize(elements(o.i,o.q));for(size_t n=0;n<v.size();++n)v[n]=float(d["gate"][n]/(1+std::exp(-double(d["gate"][n])))*d["up"][n]);
        d["output"]=matmul(d["down_weight"],v,o.i,o.h,o.q);
    } else {
        add("query",elements(o.d,o.q,o.heads),1);add("new_key",elements(o.d,o.q,o.kvh),1);add("new_value",elements(o.d,o.q,o.kvh),1);
        add("key_cache",elements(o.d,o.hist+o.q,o.kvh),1);add("value_cache",elements(o.d,o.hist+o.q,o.kvh),1);
        add("hidden",elements(o.h,o.q),1);add("attention_gate",elements(o.h,o.q),1);add("output_weight",elements(o.h,o.h),1/std::sqrt(double(o.h)));
        const int len=o.hist+o.q;
        d["key_appended"]=d["key_cache"];d["value_appended"]=d["value_cache"];
        for(int k=0;k<o.kvh;++k)for(int t=0;t<o.q;++t)for(int j=0;j<o.d;++j) {
            const size_t dst=size_t(k*len+o.hist+t)*o.d+j,src=size_t(k*o.q+t)*o.d+j;
            d["key_appended"][dst]=d["new_key"][src];d["value_appended"][dst]=d["new_value"][src];
        }
        auto&out=d["context"];out.resize(elements(o.h,o.q));
        for(int t=0;t<o.q;++t)for(int head=0;head<o.heads;++head) {
            int kh=head/(o.heads/o.kvh);std::vector<double> probs(size_t(o.hist+t+1));double largest=-1e100,sum=0;
            for(size_t k=0;k<probs.size();++k) {double s=0;for(int j=0;j<o.d;++j)s+=double(d["query"][size_t(head*o.q+t)*o.d+j])*d["key_appended"][size_t(kh*len+int(k))*o.d+j];probs[k]=s/std::sqrt(double(o.d));largest=std::max(largest,probs[k]);}
            for(auto&v:probs) {v=std::exp(v-largest);sum+=v;}
            for(int j=0;j<o.d;++j) {double v=0;for(size_t k=0;k<probs.size();++k)v+=probs[k]/sum*d["value_appended"][size_t(kh*len+int(k))*o.d+j];out[size_t(t)*o.h+head*o.d+j]=float(v);}
        }
        auto gated=out;for(size_t n=0;n<gated.size();++n)gated[n]=float(double(out[n])/(1+std::exp(-double(d["attention_gate"][n]))));
        d["output"]=matmul(d["output_weight"],gated,o.h,o.h,o.q);for(size_t n=0;n<d["output"].size();++n)d["output"][n]+=d["hidden"][n];
    }return d;
}
struct Graph {
    ggml_context*ctx=nullptr;ggml_cgraph*graph=nullptr;ggml_gallocr_t allocator=nullptr;
    ggml_cplan plan{};std::vector<uint8_t> work;size_t internal=0;
    ~Graph(){if(allocator)ggml_gallocr_free(allocator);if(ctx)ggml_free(ctx);}
};
struct Session {
    const Options&o;bool gpu;ggml_context*resident=nullptr;ggml_backend_t backend=nullptr;
    ggml_backend_buffer_t buffer=nullptr;ggml_threadpool_t pool=nullptr;
    std::map<std::string,ggml_tensor*> ports;std::map<std::string,std::unique_ptr<Graph>>graphs;
    Session(const Options&options,bool use_gpu,const Data&d):o(options),gpu(use_gpu) {
#if defined(MATCHED_HAVE_CUDA)
        if(gpu) {if(cudaSetDevice(o.device)!=cudaSuccess)throw std::runtime_error("CUDA device unavailable");backend=ggml_backend_cuda_init(o.device);}
#endif
        if(!gpu) {backend=ggml_backend_cpu_init();auto tp=ggml_threadpool_params_default(o.threads);pool=ggml_threadpool_new(&tp);if(!pool)throw std::runtime_error("threadpool failed");ggml_backend_cpu_set_n_threads(backend,o.threads);ggml_backend_cpu_set_threadpool(backend,pool);}
        if(!backend)throw std::runtime_error("requested backend unavailable; no fallback");
        resident=ggml_init({ggml_tensor_overhead()*40,nullptr,true});if(!resident)throw std::runtime_error("resident metadata failed");
        auto add=[&](const std::string&n,int a,int b=1,int c=1){auto*t=ggml_new_tensor_3d(resident,GGML_TYPE_F32,a,b,c);ggml_set_name(t,n.c_str());ggml_set_input(t);ports[n]=t;};
        if(o.family=="ffn") {add("input",o.h,o.q);add("gate_weight",o.h,o.i);add("up_weight",o.h,o.i);add("down_weight",o.i,o.h);for(auto n:{"gate","up","activated"})add(n,o.i,o.q);add("output",o.h,o.q);}
        else {add("query",o.d,o.q,o.heads);for(auto n:{"new_key","new_value"})add(n,o.d,o.q,o.kvh);for(auto n:{"key_cache","value_cache","key_appended","value_appended"})add(n,o.d,o.hist+o.q,o.kvh);for(auto n:{"context","hidden","attention_gate","output"})add(n,o.h,o.q);add("output_weight",o.h,o.h);}
        for(auto n:{"gate","up","activated","context","output","key_appended","value_appended"})if(ports.count(n))ggml_set_output(ports.at(n));
        buffer=ggml_backend_alloc_ctx_tensors(resident,backend);if(!buffer)throw std::runtime_error("resident allocation failed");
        for(auto&p:ports) {auto it=d.find(p.first);if(it==d.end())throw std::runtime_error("missing initial port");ggml_backend_tensor_set(p.second,it->second.data(),0,ggml_nbytes(p.second));}
        const std::vector<std::string> names=o.family=="ffn"?std::vector<std::string>{"gate","up","activation","down","joint"}:std::vector<std::string>{"attention_core","attention_output","joint"};
        for(auto&n:names)build(n);
    }
    ~Session(){if(backend)ggml_backend_synchronize(backend);graphs.clear();if(buffer)ggml_backend_buffer_free(buffer);if(backend)ggml_backend_free(backend);if(pool)ggml_threadpool_free(pool);if(resident)ggml_free(resident);}
    void build(const std::string&kind) {
        auto g=std::make_unique<Graph>();g->ctx=ggml_init({ggml_tensor_overhead()*160+ggml_graph_overhead_custom(256,false),nullptr,true});if(!g->ctx)throw std::runtime_error("graph metadata failed");g->graph=ggml_new_graph_custom(g->ctx,256,false);
        auto*tctx=g->ctx;auto get=[&](const char*n){return ports.at(n);};
        auto store=[&](const char*n,ggml_tensor*t){auto*x=ggml_cpy(tctx,t,get(n));ggml_set_output(x);ggml_build_forward_expand(g->graph,x);return x;};
        if(o.family=="ffn") {
            ggml_tensor *gate=get("gate"),*up=get("up"),*active=get("activated");
            if(kind=="gate"||kind=="joint")gate=store("gate",ggml_mul_mat(tctx,get("gate_weight"),get("input")));
            if(kind=="up"||kind=="joint")up=store("up",ggml_mul_mat(tctx,get("up_weight"),get("input")));
            if(kind=="activation"||kind=="joint")active=store("activated",ggml_swiglu_split(tctx,gate,up));
            if(kind=="down"||kind=="joint")store("output",ggml_mul_mat(tctx,get("down_weight"),active));
        }else {
            ggml_tensor*context=get("context");
            if(kind=="attention_core"||kind=="joint") {
                auto*k=ggml_cpy(tctx,get("key_cache"),get("key_appended"));auto*v=ggml_cpy(tctx,get("value_cache"),get("value_appended"));
                const size_t offset=size_t(o.hist)*o.d*sizeof(float);
                k=ggml_set_inplace(tctx,k,get("new_key"),k->nb[1],k->nb[2],k->nb[3],offset);
                v=ggml_set_inplace(tctx,v,get("new_value"),v->nb[1],v->nb[2],v->nb[3],offset);
                auto*scores=ggml_diag_mask_inf(tctx,ggml_scale(tctx,ggml_mul_mat(tctx,k,get("query")),1/std::sqrt(float(o.d))),o.hist);
                auto*probs=ggml_soft_max(tctx,scores);auto*vt=ggml_cont(tctx,ggml_transpose(tctx,v));auto*attended=ggml_mul_mat(tctx,vt,probs);
                context=store("context",ggml_reshape_2d(tctx,ggml_cont(tctx,ggml_permute(tctx,attended,0,2,1,3)),o.h,o.q));
            }
            if(kind=="attention_output"||kind=="joint") {auto*gated=ggml_mul(tctx,context,ggml_sigmoid(tctx,get("attention_gate")));store("output",ggml_add(tctx,ggml_mul_mat(tctx,get("output_weight"),gated),get("hidden")));}
        }
        for(int n=0;n<ggml_graph_n_nodes(g->graph);++n)if(!ggml_backend_supports_op(backend,ggml_graph_node(g->graph,n)))throw std::runtime_error("requested backend does not support a graph operator");
        g->allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));if(!g->allocator||!ggml_gallocr_alloc_graph(g->allocator,g->graph))throw std::runtime_error("graph allocation failed");
        g->internal=ggml_gallocr_get_buffer_size(g->allocator,0);
        if(!gpu) {g->plan=ggml_graph_plan(g->graph,o.threads,pool);g->work.resize(g->plan.work_size);g->plan.work_data=g->work.data();}
        graphs[kind]=std::move(g);
    }
    void compute(const std::string&name) {auto&g=*graphs.at(name);if(gpu){if(ggml_backend_graph_compute_async(backend,g.graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("GPU compute failed");ggml_backend_synchronize(backend);}else if(ggml_graph_compute(g.graph,&g.plan)!=GGML_STATUS_SUCCESS)throw std::runtime_error("CPU compute failed");}
    void serial() {if(o.family=="ffn")for(auto n:{"gate","up","activation","down"})compute(n);else for(auto n:{"attention_core","attention_output"})compute(n);}
    std::vector<float>read(const std::string&name){auto*t=ports.at(name);std::vector<float>v(ggml_nelements(t));ggml_backend_tensor_get(t,v.data(),0,ggml_nbytes(t));return v;}
};
struct Check {double maxabs=0,maxrel=0;size_t elements=0;bool passed=true;};
static Check compare(const std::vector<float>&a,const std::vector<float>&b) {
    if(a.size()!=b.size())throw std::runtime_error("port size mismatch");Check c;c.elements=a.size();
    for(size_t n=0;n<a.size();++n){if(!std::isfinite(a[n])||!std::isfinite(b[n]))throw std::runtime_error("nonfinite numerical port");double e=std::abs(double(a[n])-b[n]);c.maxabs=std::max(c.maxabs,e);c.maxrel=std::max(c.maxrel,e/std::max(1e-12,std::abs(double(b[n]))));if(e>1e-5+1e-4*std::abs(b[n]))c.passed=false;}return c;
}
template<class F>static std::vector<int64_t> measure(const Options&o,F&&f) {
    int n=0;auto start=Clock::now();do{f();++n;}while(n<std::max(1,o.warmup)||std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now()-start).count()<o.min_warmup_ms);
    std::vector<int64_t>v;for(int k=0;k<o.samples;++k){auto a=Clock::now();f();auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-a).count();if(ns<1)throw std::runtime_error("nonpositive time");v.push_back(ns);}return v;
}
#if defined(MATCHED_HAVE_CUDA)
static void cuda_ok(cudaError_t x,const char*n){if(x!=cudaSuccess)throw std::runtime_error(std::string(n)+": "+cudaGetErrorString(x));}
struct Transfers {
    Session&cpu;Session&gpu;cudaStream_t stream=nullptr;std::map<std::string,void*>pinned;
    Transfers(Session&c,Session&g):cpu(c),gpu(g){cuda_ok(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"stream");for(auto&p:cpu.ports){void*v=nullptr;cuda_ok(cudaMallocHost(&v,ggml_nbytes(p.second)),"pinned");pinned[p.first]=v;}}
    ~Transfers(){if(stream)cudaStreamSynchronize(stream);for(auto&p:pinned)cudaFreeHost(p.second);if(stream)cudaStreamDestroy(stream);}
    void h2d(const std::string&name){auto*c=cpu.ports.at(name),*g=gpu.ports.at(name);std::memcpy(pinned.at(name),c->data,ggml_nbytes(c));cuda_ok(cudaMemcpyAsync(g->data,pinned.at(name),ggml_nbytes(g),cudaMemcpyHostToDevice,stream),"H2D");cuda_ok(cudaStreamSynchronize(stream),"H2D wait");}
    void d2h(const std::string&name){auto*c=cpu.ports.at(name),*g=gpu.ports.at(name);cuda_ok(cudaMemcpyAsync(pinned.at(name),g->data,ggml_nbytes(g),cudaMemcpyDeviceToHost,stream),"D2H");cuda_ok(cudaStreamSynchronize(stream),"D2H wait");std::memcpy(c->data,pinned.at(name),ggml_nbytes(c));}
};
#endif
static int run(const Options&o) {
    Data d=make_data(o);std::unique_ptr<Session> cpu,gpu;
    if(o.backend!="gpu")cpu=std::make_unique<Session>(o,false,d);
    if(o.backend!="cpu")gpu=std::make_unique<Session>(o,true,d);
    std::map<std::string,std::vector<int64_t>>times;std::map<std::string,Check>checks;
    auto test=[&](Session&s,const std::string&prefix){s.serial();for(auto&n:(o.family=="ffn"?std::vector<std::string>{"gate","up","activated","output"}:std::vector<std::string>{"context","key_appended","value_appended","output"}))checks[prefix+"serial/"+n]=compare(s.read(n),d.at(n));s.compute("joint");checks[prefix+"joint/output"]=compare(s.read("output"),d.at("output"));for(auto&p:s.graphs)times[prefix+p.first]=measure(o,[&](){s.compute(p.first);});times[prefix+"serial"]=measure(o,[&](){s.serial();});};
    if(cpu)test(*cpu,"cpu/");if(gpu)test(*gpu,"gpu/");
#if defined(MATCHED_HAVE_CUDA)
    if(o.backend=="hybrid") {
        Transfers tr(*cpu,*gpu);
        times["hybrid/gpu_weights"]=measure(o,[&](){for(auto n:{"gate_weight","up_weight","down_weight"})tr.h2d(n);gpu->serial();});
        checks["hybrid/gpu_weights/output"]=compare(gpu->read("output"),d.at("output"));
        times["hybrid/cpu_rows"]=measure(o,[&](){tr.d2h("input");cpu->serial();tr.h2d("output");});
        checks["hybrid/cpu_rows/output"]=compare(gpu->read("output"),d.at("output"));
        times["hybrid/split_cpu_to_gpu"]=measure(o,[&](){cpu->compute("gate");cpu->compute("up");cpu->compute("activation");tr.h2d("activated");gpu->compute("down");});
        checks["hybrid/split/output"]=compare(gpu->read("output"),d.at("output"));
        for(auto n:{"input","activated","output","gate_weight","up_weight","down_weight"})times["hybrid/h2d/"+std::string(n)]=measure(o,[&](){tr.h2d(n);});
        times["hybrid/d2h/input"]=measure(o,[&](){tr.d2h("input");});
    }
#endif
    bool passed=true;for(auto&c:checks)passed=passed&&c.second.passed;
    std::cout<<std::setprecision(12)<<"{\"schema\":\"matched-compute-native-0.1\",\"family\":\""<<o.family<<"\",\"backend\":\""<<o.backend<<"\",\"dtype\":\"f32\",\"hidden\":"<<o.h<<",\"intermediate\":"<<o.i<<",\"tokens\":"<<o.q<<",\"history\":"<<o.hist<<",\"heads\":"<<o.heads<<",\"kv_heads\":"<<o.kvh<<",\"head_dim\":"<<o.d<<",\"threads\":"<<o.threads<<",\"cpu_id\":"<<o.cpu_id<<",\"seed\":"<<o.seed<<",\"samples\":"<<o.samples<<",\"cuda_graphs\":\"disabled\",\"timer\":\"steady_clock_completed_work\",\"weights\":\"generated_f32_not_model_weights\",\"activation_layout\":\"GGML ne0 contiguous; token-major flattened except Q/K/V head-major\",\"sample_order\":\"grouped_by_stage_then_serial_then_paths\",\"timing_includes\":\"graph dispatch plus completion wait; hybrid includes pinned staging memcpy\",\"timing_excludes\":\"allocation initialization numerical validation\",\"wall_ns\":{";
    bool first=true;for(auto&p:times){std::cout<<(first?"":",")<<'"'<<p.first<<"\":";emit_array(p.second);first=false;}
    std::cout<<"},\"ports\":{";first=true;const auto&s=cpu?*cpu:*gpu;for(auto&p:s.ports){std::cout<<(first?"":",")<<'"'<<p.first<<"\":{\"bytes\":"<<ggml_nbytes(p.second)<<",\"ne\":["<<p.second->ne[0]<<','<<p.second->ne[1]<<','<<p.second->ne[2]<<"],\"input_fixture_fnv1a64\":\""<<digest(d.at(p.first))<<"\"}";first=false;}
    std::cout<<"},\"workspaces\":{";first=true;for(auto*session:{cpu.get(),gpu.get()})if(session)for(auto&p:session->graphs){std::cout<<(first?"":",")<<'"'<<(session->gpu?"gpu/":"cpu/")<<p.first<<"\":{\"known_gallocr_bytes\":"<<p.second->internal<<",\"workspace_bytes\":";if(session->gpu)std::cout<<"null,\"source\":\"unknown_private_cuda_scratch\"}";else std::cout<<p.second->internal+p.second->work.size()<<",\"source\":\"gallocr_plus_explicit_cplan\"}";first=false;}
    std::cout<<"},\"checks\":{";first=true;for(auto&p:checks){auto&c=p.second;std::cout<<(first?"":",")<<'"'<<p.first<<"\":{\"passed\":"<<(c.passed?"true":"false")<<",\"max_abs_error\":"<<c.maxabs<<",\"max_relative_error\":"<<c.maxrel<<",\"elements\":"<<c.elements<<",\"atol\":0.00001,\"rtol\":0.0001}";first=false;}
    std::cout<<"},\"all_checks_passed\":"<<(passed?"true":"false")<<",\"control_overhead_separately_identified\":false,\"gpu_capacity_proved\":false,\"source_values\":{";
    first=true;for(auto&p:d){std::cout<<(first?"":",")<<'"'<<p.first<<"\":[";for(size_t n=0;n<p.second.size();++n)std::cout<<(n?",":"")<<p.second[n];std::cout<<']';first=false;}std::cout<<"}}\n";
    return passed?0:2;
}
int main(int argc,char**argv){try{return run(parse(argc,argv));}catch(const std::exception&e){std::cerr<<"matched-compute-probe: "<<e.what()<<'\n';return 1;}}
