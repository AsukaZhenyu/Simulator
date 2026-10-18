// Execute the actual compiled ModelPlan graph/actions at independent GGML ports.
// F32 Dense .3/.5 macros plus public AttentionStages and FFNPartition only.
#define NOMINMAX
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef HIGHLEVEL_CUDA
#include "ggml-cuda.h"
#endif
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
static constexpr const char *BOUNDARY = "independent_contiguous_f32_ports_full_dense_norm_rope_residual_v1";
static void require(bool good, const std::string &message) { if (!good) throw std::runtime_error(message); }
static int64_t elapsed(Clock::time_point start) { return std::max<int64_t>(1, std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count()); }
static std::string str(const json &j, const char *key) { return j.at(key).get<std::string>(); }
struct Tensor {
    json descriptor;
    ggml_tensor *dram=nullptr, *vram=nullptr;
    std::vector<uint8_t> initial;
};
struct Graph {
    ggml_context *ctx=nullptr;
    ggml_cgraph *graph=nullptr;
    ggml_gallocr_t alloc=nullptr;
    ggml_backend_t backend=nullptr;
    size_t workspace=0;
    ~Graph() { if(alloc)ggml_gallocr_free(alloc); if(ctx)ggml_free(ctx); }
};
struct Executor {
    json program, config;
    std::map<std::string,Tensor> tensors;
    std::map<std::string,std::unique_ptr<Graph>> graphs;
    std::map<std::string,json> bindings, operations;
    ggml_backend_t cpu=nullptr, gpu=nullptr;
    ggml_threadpool_t pool=nullptr;
    ggml_context *dram_ctx=nullptr, *vram_ctx=nullptr;
    ggml_backend_buffer_t dram_buffer=nullptr, vram_buffer=nullptr;
    int64_t hidden=0, intermediate=0, heads=0, kvheads=0, dim=0;
    json ports=json::object(), copies=json::array();
    bool copy_equal=true;
    explicit Executor(json source):program(std::move(source)) {
        require(str(program,"format_version")=="highlevel-accuracy-program-v1", "unsupported program schema");
        require(str(program,"boundary_contract")==BOUNDARY,"unsupported boundary contract");
        config=program.at("physical").at("config");
        hidden=config.at("hidden"); intermediate=config.at("intermediate"); heads=config.at("heads"); kvheads=config.at("kv_heads"); dim=config.at("head_dim");
        require(hidden>0&&intermediate>0&&heads>0&&kvheads>0&&dim>0&&heads%kvheads==0,"invalid full Dense geometry");
        require(str(config,"rope_mode")=="normal","normal RoPE required");
        for(auto &dtype:config.at("weight_dtypes").items()) require(dtype.value()=="f32","only F32 weight execution supported");
        bool use_gpu=false;
        for(const auto &t:program.at("graph").at("workload").at("tensors"))
            for(const auto &location:t.value("initial_locations",json::array()))if(location=="vram")use_gpu=true;
        for(const auto &a:program.at("actions"))if(a.value("device",json())=="gpu"||a.value("src",json())=="vram"||a.value("dst",json())=="vram")use_gpu=true;
        cpu=ggml_backend_cpu_init();require(cpu,"CPU initialization failed");
        const int threads=program.at("physical").at("threads");require(threads>0,"positive threads required");
        ggml_backend_cpu_set_n_threads(cpu,threads);
        auto pool_params=ggml_threadpool_params_default(threads);pool=ggml_threadpool_new(&pool_params);
        require(pool,"persistent CPU threadpool initialization failed");ggml_backend_cpu_set_threadpool(cpu,pool);
        if(use_gpu){
#ifdef HIGHLEVEL_CUDA
#if defined(_WIN32)
            _putenv_s("GGML_CUDA_DISABLE_GRAPHS","1");
#else
            setenv("GGML_CUDA_DISABLE_GRAPHS","1",1);
#endif
            gpu=ggml_backend_cuda_init(program.at("physical").value("cuda_device",0));
            require(gpu,"requested CUDA backend unavailable");
#else
            throw std::runtime_error("program requires CUDA but executor was built CPU-only");
#endif
        }
        const size_t count=program.at("graph").at("workload").at("tensors").size();
        dram_ctx=ggml_init({ggml_tensor_overhead()*(count+8),nullptr,true});require(dram_ctx,"DRAM metadata allocation failed");
        if(use_gpu){vram_ctx=ggml_init({ggml_tensor_overhead()*(count+8),nullptr,true});require(vram_ctx,"VRAM metadata allocation failed");}
        for(const auto &description:program.at("graph").at("workload").at("tensors")){
            const auto id=str(description,"id"); require(!tensors.count(id),"duplicate tensor identity");
            Tensor t;t.descriptor=description;
            const auto dtype=str(description,"dtype");require(dtype=="GGML_TYPE_F32"||dtype=="GGML_TYPE_I32","unsupported tensor dtype");
            const auto type=dtype=="GGML_TYPE_F32"?GGML_TYPE_F32:GGML_TYPE_I32;
            int64_t ne[4];size_t bytes=4;
            require(description.at("ne").size()==4&&description.at("nb").size()==4,"exact four-dimensional layout required");
            for(int i=0;i<4;++i){ne[i]=description.at("ne").at(i);require(ne[i]>0&&static_cast<uint64_t>(ne[i])<=16000000/bytes,"tensor shape overflow/envelope");require(description.at("nb").at(i)==bytes,"non-contiguous tensor rejected");bytes*=ne[i];}
            require(description.at("storage_bytes")==bytes,"storage boundary mismatch");
            t.dram=ggml_new_tensor(dram_ctx,type,4,ne);ggml_set_input(t.dram);ggml_set_output(t.dram);
            if(use_gpu){t.vram=ggml_new_tensor(vram_ctx,type,4,ne);ggml_set_input(t.vram);ggml_set_output(t.vram);}
            if(program.at("initial_values").contains(id)){
                const auto &v=program.at("initial_values").at(id);require(v.size()==bytes/4,"initial value shape mismatch");
                t.initial.resize(bytes);
                for(size_t i=0;i<v.size();++i){if(type==GGML_TYPE_I32){int32_t z=v.at(i);std::memcpy(t.initial.data()+4*i,&z,4);}else{float z=v.at(i);require(std::isfinite(z),"non-finite initial data");std::memcpy(t.initial.data()+4*i,&z,4);}}
            }
            tensors.emplace(id,std::move(t));
        }
        dram_buffer=ggml_backend_alloc_ctx_tensors(dram_ctx,cpu);require(dram_buffer,"DRAM tensor allocation failed");
        if(use_gpu){vram_buffer=ggml_backend_alloc_ctx_tensors(vram_ctx,gpu);require(vram_buffer,"VRAM tensor allocation failed");}
        for(const auto &op:program.at("graph").at("workload").at("operations"))require(operations.emplace(str(op,"id"),op).second,"duplicate operation");
        for(const auto &b:program.at("bindings")){
            const auto id=str(b,"operation_id");require(bindings.emplace(id,b).second,"duplicate native binding");
            const auto &op=operations.at(id);require(op.at("semantic_op")==b.at("kind")&&op.at("op_params")==b.at("op_params"),"operation identity mismatch");
            std::set<std::string> ins,outs,expected_in,expected_out;
            for(const auto &item:b.at("operands").items())ins.insert(item.value());
            for(const auto &item:b.at("history"))ins.insert(str(item,"id"));
            for(const auto &item:b.at("outputs").items())outs.insert(item.value());
            for(const auto &item:b.at("new_kv"))outs.insert(str(item,"id"));
            for(const auto &item:op.at("inputs"))expected_in.insert(item);
            expected_out.insert(str(op,"output"));for(const auto &item:op.value("extra_outputs",json::array()))expected_out.insert(item);
            require(ins==expected_in&&outs==expected_out,"native operand/port graph mismatch");
        }
        require(operations.size()==bindings.size(),"incomplete native graph binding");
        reset();
        for(const auto &a:program.at("actions"))if(a.at("kind")=="COMPUTE"){
            const auto id=str(a,"target_id");require(!graphs.count(id),"duplicate compute action");
            graphs[id]=build(bindings.at(id),str(a,"device"));
        }
        require(graphs.size()==bindings.size(),"action trace does not execute complete graph");
    }
    ~Executor(){graphs.clear();if(dram_buffer)ggml_backend_buffer_free(dram_buffer);if(vram_buffer)ggml_backend_buffer_free(vram_buffer);if(dram_ctx)ggml_free(dram_ctx);if(vram_ctx)ggml_free(vram_ctx);if(cpu)ggml_backend_free(cpu);if(gpu)ggml_backend_free(gpu);if(pool)ggml_threadpool_free(pool);}
    ggml_tensor *tensor(const std::string &id,const std::string &location){auto &t=tensors.at(id);require(location=="dram"||location=="vram","unsupported tensor location");auto *p=location=="dram"?t.dram:t.vram;require(p,"unallocated requested backend");return p;}
    void reset(){for(auto &item:tensors){auto &t=item.second;for(const auto &loc:t.descriptor.value("initial_locations",json::array())){require(!t.initial.empty(),"missing initial physical data");auto *p=tensor(item.first,loc);ggml_backend_tensor_set(p,t.initial.data(),0,t.initial.size());}}}
    std::vector<float> read(const std::string &id,const std::string &loc){auto *p=tensor(id,loc);require(p->type==GGML_TYPE_F32,"non-F32 output port");std::vector<float> v(ggml_nelements(p));ggml_backend_tensor_get(p,v.data(),0,ggml_nbytes(p));return v;}
    std::unique_ptr<Graph> build(const json &b,const std::string &device){
        auto g=std::make_unique<Graph>();const std::string location=device=="cpu"?"dram":"vram";require(device=="cpu"||device=="gpu","unknown compute backend");
        g->backend=device=="cpu"?cpu:gpu;require(g->backend,"backend unavailable");
        g->ctx=ggml_init({ggml_tensor_overhead()*4096+ggml_graph_overhead_custom(4096,false),nullptr,true});require(g->ctx,"graph metadata allocation failed");
        auto *ctx=g->ctx;g->graph=ggml_new_graph_custom(ctx,4096,false);
        const int64_t n=b.at("shape").at("tokens"), past=b.at("shape").at("visible_history");
        const auto kind=str(b,"kind");
        auto x=[&](const std::string &role){return tensor(b.at("operands").at(role),location);};
        auto output=[&](const std::string &id,ggml_tensor *v){auto *destination=tensor(id,location);require(ggml_nelements(v)==ggml_nelements(destination),"computed output port shape mismatch");auto *node=ggml_cpy(ctx,v,destination);ggml_set_output(node);ggml_build_forward_expand(g->graph,node);};
        auto result=[&](ggml_tensor *v){output(b.at("outputs").at("result"),v);};
        auto norm=[&](ggml_tensor *v,ggml_tensor *gamma){return ggml_mul(ctx,ggml_rms_norm(ctx,v,config.at("norm_epsilon").get<float>()),gamma);};
        auto cache=[&](const std::string &role){
            ggml_tensor *assembled=nullptr;
            int64_t count=0;
            for(const auto &block:b.at("history")){
                require(block.at("first_position")==count,"history must be position-contiguous");
                const int64_t tokens=block.at("tokens");auto *flat=tensor(str(block,"id"),location);
                require(ggml_nelements(flat)==tokens*2*kvheads*dim,"historical KV layout mismatch");
                auto *view=ggml_view_3d(ctx,flat,dim,tokens,kvheads,2*kvheads*dim*4,dim*4,role=="k"?0:kvheads*dim*4);
                auto *value=ggml_cont(ctx,view);assembled=assembled?ggml_concat(ctx,assembled,value,1):value;count+=tokens;
            }
            require(count==past,"historical KV count mismatch");auto *fresh=x(role);return assembled?ggml_concat(ctx,assembled,fresh,1):fresh;
        };
        auto append=[&](){for(const auto &block:b.at("new_kv")){
            const int64_t start=block.at("first_position").get<int64_t>()-past,tokens=block.at("tokens");require(start>=0&&start+tokens<=n,"new KV position mismatch");
            auto part=[&](const std::string &role){auto *v=x(role);auto *view=ggml_view_3d(ctx,v,dim,tokens,kvheads,v->nb[1],v->nb[2],start*dim*4);return ggml_reshape_2d(ctx,ggml_cont(ctx,ggml_permute(ctx,view,0,2,1,3)),dim*kvheads,tokens);};
            output(str(block,"id"),ggml_cont(ctx,ggml_concat(ctx,part("k"),part("v"),0)));
        }};
        if(kind=="projection"){
            auto *normalized=norm(x("hidden"),x("input_norm"));
            for(const auto &role:{std::string("q"),std::string("k"),std::string("v")}){
                auto *v=ggml_reshape_3d(ctx,ggml_mul_mat(ctx,x(role),normalized),dim,role=="q"?heads:kvheads,n);
                if(role!="v"){
                    if(config.at("qk_norm")=="rms")v=norm(v,x(role+"_norm"));
                    v=ggml_rope_ext(ctx,v,x("positions"),nullptr,config.at("rope_dims"),GGML_ROPE_TYPE_NORMAL,0,config.at("rope_base"),1.0f,0.0f,1.0f,0.0f,0.0f);
                }
                output(b.at("outputs").at(role),ggml_cont(ctx,ggml_permute(ctx,v,0,2,1,3)));
            }
            if(b.at("operands").contains("gate_proj"))output(b.at("outputs").at("gate"),ggml_mul_mat(ctx,x("gate_proj"),normalized));
        }else if(kind=="attention"||kind=="attention_qk"||kind=="attention_pv"){
            ggml_tensor *prob=nullptr;
            if(kind!="attention_pv"){
                auto *scores=ggml_scale(ctx,ggml_mul_mat(ctx,cache("k"),x("q")),1.0f/std::sqrt(float(dim)));
                if(kind=="attention_qk")result(scores);
                else prob=ggml_soft_max(ctx,ggml_diag_mask_inf(ctx,scores,past));
            }else prob=x("probabilities");
            if(kind!="attention_qk"){
                auto *attended=ggml_mul_mat(ctx,ggml_cont(ctx,ggml_transpose(ctx,cache("v"))),prob);
                auto *merged=ggml_reshape_2d(ctx,ggml_cont(ctx,ggml_permute(ctx,attended,0,2,1,3)),dim*heads,n);
                if(kind=="attention"){
                    if(b.at("operands").contains("gate"))merged=ggml_mul(ctx,merged,ggml_sigmoid(ctx,x("gate")));
                    result(ggml_add(ctx,ggml_mul_mat(ctx,x("output"),merged),x("hidden")));
                }else result(merged);
                append();
            }
        }else if(kind=="attention_softmax")result(ggml_soft_max(ctx,ggml_diag_mask_inf(ctx,x("scores"),past)));
        else if(kind=="attention_output"){
            auto *v=x("context");if(b.at("operands").contains("gate"))v=ggml_mul(ctx,v,ggml_sigmoid(ctx,x("gate")));
            result(ggml_add(ctx,ggml_mul_mat(ctx,x("output"),v),x("hidden")));
        }else if(kind=="dense_ffn"){
            auto *nrm=norm(x("hidden"),x("post_attention_norm"));auto *gate=ggml_mul_mat(ctx,x("gate"),nrm),*up=ggml_mul_mat(ctx,x("up"),nrm);
            result(ggml_add(ctx,ggml_mul_mat(ctx,x("down"),ggml_swiglu_split(ctx,gate,up)),x("hidden")));
        }else if(kind=="ffn_norm")result(norm(x("x1"),x("x0")));
        else if(kind=="ffn_gate"||kind=="ffn_up"||kind=="ffn_down")result(ggml_mul_mat(ctx,x("x0"),x("x1")));
        else if(kind=="ffn_swiglu")result(ggml_swiglu_split(ctx,x("x0"),x("x1")));
        else if(kind=="ffn_sum"||kind=="ffn_residual"){
            auto *v=x("x0");for(size_t i=1;i<b.at("operands").size();++i)v=ggml_add(ctx,v,x("x"+std::to_string(i)));result(v);
        }else throw std::runtime_error("unsupported semantic operation: "+kind);
        for(int i=0;i<ggml_graph_n_nodes(g->graph);++i){auto *node=ggml_graph_node(g->graph,i);if(node->op==GGML_OP_MUL_MAT)ggml_mul_mat_set_prec(node,GGML_PREC_F32);require(ggml_backend_supports_op(g->backend,node),"requested backend does not support compiled GGML operation");}
        g->alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(g->backend));require(g->alloc&&ggml_gallocr_reserve(g->alloc,g->graph)&&ggml_gallocr_alloc_graph(g->alloc,g->graph),"GGML graph allocation failed");
        g->workspace=ggml_gallocr_get_buffer_size(g->alloc,0);
        if(device=="cpu")g->workspace+=ggml_graph_plan(g->graph,program.at("physical").at("threads"),pool).work_size;
        return g;
    }
    void action(const json &a){
        const auto kind=str(a,"kind");
        if(kind=="COMPUTE"){
            auto &g=*graphs.at(str(a,"target_id"));require(ggml_backend_graph_compute(g.backend,g.graph)==GGML_STATUS_SUCCESS,"GGML compute failed");ggml_backend_synchronize(g.backend);
        }else if(kind=="COPY"){
            auto *source=tensor(str(a,"target_id"),str(a,"src")),*target=tensor(str(a,"target_id"),str(a,"dst"));
            ggml_backend_tensor_copy(source,target);if(gpu)ggml_backend_synchronize(gpu);ggml_backend_synchronize(cpu);
        }else require(kind=="ADVANCE"||kind=="EVICT","unsupported physical action");
    }
    void validate(){
        reset();std::set<std::pair<std::string,std::string>> ready;
        for(const auto &item:tensors)for(const auto &loc:item.second.descriptor.value("initial_locations",json::array()))ready.insert({item.first,loc});
        size_t index=0;
        for(const auto &a:program.at("actions")){
            const auto kind=str(a,"kind");
            if(kind=="COMPUTE"){
                const std::string id=str(a,"target_id"),loc=a.at("device")=="cpu"?"dram":"vram";const auto &op=operations.at(id);
                for(const auto &tid:op.at("inputs"))require(ready.count({tid,loc}),"compute input copy not ready: "+tid.get<std::string>());
                action(a);
                std::vector<std::string> outputs={str(op,"output")};for(const auto &tid:op.value("extra_outputs",json::array()))outputs.push_back(tid);
                for(const auto &tid:outputs){require(!ready.count({tid,loc}),"duplicate output materialization");ready.insert({tid,loc});ports[tid]=read(tid,loc);}
            }else if(kind=="COPY"){
                const auto tid=str(a,"target_id"),src=str(a,"src"),dst=str(a,"dst");require(src!=dst&&ready.count({tid,src})&&!ready.count({tid,dst}),"copy liveness/route mismatch");
                action(a);ready.insert({tid,dst});const auto bytes=tensors.at(tid).descriptor.at("storage_bytes").get<size_t>();
                std::vector<uint8_t> left(bytes),right(bytes);ggml_backend_tensor_get(tensor(tid,src),left.data(),0,bytes);ggml_backend_tensor_get(tensor(tid,dst),right.data(),0,bytes);copy_equal=copy_equal&&(left==right);
                copies.push_back(json{{"index",index},{"tensor_id",tid},{"src",src},{"dst",dst},{"storage_bytes",bytes}});
            }else if(kind=="EVICT")require(ready.erase({str(a,"target_id"),str(a,"location")})==1,"eviction of absent copy");
            else require(kind=="ADVANCE","unsupported action");
            ++index;
        }
        for(const auto &goal:program.at("graph").at("scenario").at("goal").at("output_locations").items())require(ready.count({goal.key(),goal.value()}),"native output delivery incomplete");
        require(copy_equal,"native copy changed tensor bytes");
    }
    json measure(const std::string &mode,int warmup,int samples){
        validate();json components=json::array(),totals=json::array();
        if(mode=="components"){
            // A complete untimed replay prepares the EXACT upstream inputs.
            // Each independent graph/copy is then sampled at its own boundary.
            for(size_t i=0;i<program.at("actions").size();++i){const auto &a=program.at("actions").at(i);if(a.at("kind")!="COMPUTE"&&a.at("kind")!="COPY")continue;
                for(int j=0;j<warmup;++j)action(a);json ns=json::array();
                for(int j=0;j<samples;++j){auto start=Clock::now();action(a);ns.push_back(elapsed(start));}
                size_t workspace=0;if(a.at("kind")=="COMPUTE")workspace=graphs.at(str(a,"target_id"))->workspace;
                components.push_back(json{{"action_index",i},{"action",a},{"samples_ns",ns},{"workspace_bytes",workspace}});
            }
        }else if(mode!="validate"){
            require(mode=="replay","mode must be components or replay");
            for(int j=-warmup;j<samples;++j){reset();auto start=Clock::now();for(const auto &a:program.at("actions"))action(a);auto ns=elapsed(start);if(j>=0)totals.push_back(ns);}
        }
        // Read the outputs of the FINAL timed execution outside all timers.
        // The verifier compares every port independently against NumPy.
        for(const auto &a:program.at("actions"))if(a.at("kind")=="COMPUTE"){
            const auto &op=operations.at(str(a,"target_id"));const std::string loc=a.at("device")=="cpu"?"dram":"vram";
            ports[str(op,"output")]=read(str(op,"output"),loc);
            for(const auto &tid:op.value("extra_outputs",json::array()))ports[tid.get<std::string>()]=read(tid,loc);
        }
        return json{{"format_version","highlevel-accuracy-native-v1"},{"program_sha256",program.at("program_sha256")},{"boundary_contract",BOUNDARY},
            {"mode",mode},{"warmup",warmup},{"samples",samples},{"actions",program.at("actions")},{"ports",ports},{"copies",copies},{"copy_values_equal",copy_equal},
            {"components",components},{"total_samples_ns",totals},{"timing_scope","prebuilt_graph_compute_and_materialized_ports_plus_synchronized_exact_tensor_copy"},
            {"observed_runtime_flags",json{{"GGML_CUDA_DISABLE_GRAPHS",std::getenv("GGML_CUDA_DISABLE_GRAPHS")?std::getenv("GGML_CUDA_DISABLE_GRAPHS"):""},{"NVIDIA_TF32_OVERRIDE",std::getenv("NVIDIA_TF32_OVERRIDE")?std::getenv("NVIDIA_TF32_OVERRIDE"):""},{"matmul_precision","GGML_PREC_F32"}}},
            {"final_execution_ports_captured_outside_timing",true},
            {"workspace_scope","known_gallocr_capacity_plus_CPU_cplan; CUDA_private_backend_scratch_unknown; not_complete_native_memory"},
            {"native_memory_claimed",false},{"eviction_semantics","logical_liveness_on_preallocated_independent_backing"}};
    }
};
int main(int argc,char **argv){try{
#if defined(_WIN32)
    _putenv_s("NVIDIA_TF32_OVERRIDE","0");_putenv_s("GGML_CUDA_DISABLE_GRAPHS","1");
#else
    setenv("NVIDIA_TF32_OVERRIDE","0",1);setenv("GGML_CUDA_DISABLE_GRAPHS","1",1);
#endif
    require(argc==6,"usage: highlevel-probe PROGRAM.json validate|components|replay WARMUP SAMPLES OUTPUT.json");
    int warmup=std::stoi(argv[3]),samples=std::stoi(argv[4]);require(warmup>=0&&warmup<=10000&&samples>=1&&samples<=10000,"invalid sampling counts");
    std::ifstream input(argv[1]);require(bool(input),"cannot read program");json source;input>>source;
    Executor executor(std::move(source));auto result=executor.measure(argv[2],warmup,samples);
    std::ofstream output(argv[5],std::ios::binary);require(bool(output),"cannot write output");output<<result.dump(2)<<'\n';
    std::cout<<json{{"status","ok"},{"program_sha256",result.at("program_sha256")},{"mode",argv[2]},{"ports",result.at("ports").size()}}.dump()<<'\n';return 0;
}catch(const std::exception &error){std::cerr<<"highlevel-probe: "<<error.what()<<'\n';return 2;}}
