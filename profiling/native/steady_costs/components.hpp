// New compact GGML component boundaries. Historical Executor code is untouched.
#include "../portable/sha256.hpp"
#include <iomanip>
#include <sstream>

namespace steady {
static uint32_t mix32(uint32_t seed,uint32_t index,uint32_t ordinal) {
    uint32_t value=seed+(index+1u)*0x9e3779b9u+(ordinal+1u)*0x85ebca6bu;
    value^=value>>16;value*=0x7feb352du;value^=value>>15;value*=0x846ca68bu;value^=value>>16;
    return value;
}
static double uniform(uint32_t seed,uint32_t index,uint32_t ordinal) {
    return static_cast<double>(mix32(seed,index,ordinal)>>8)/16777216.0;
}
// One digest on every host: the same bytes must hash to the same 64 characters
// whether the component ran under Windows or under Linux, or a receipt could
// not be compared across the two. See ../portable/sha256.hpp.
static std::string sha256_bytes(const void *data,size_t size) {
    return portable_sha256::digest_hex(data,size);
}
static int64_t positive(const json &value,const char *name,int64_t maximum=1000000000) {
    const auto parsed=state_response::integer(value,name,1);require(parsed<=maximum,std::string(name)+" exceeds component envelope");return parsed;
}
// A CPU mask is a bit set, not a signed quantity. A host that offers this
// process all 64 CPUs sends 2^64-1, which JSON carries as an unsigned integer;
// read through `positive` -- which asks for int64_t -- it arrives as -1, and
// every minimum comparison then rejects a mask that is exactly what the
// protocol asked for. `get<uint64_t>` reads both encodings without narrowing.
static uint64_t mask(const json &value,const char *name) {
    require(value.is_number_integer()&&!value.is_boolean(),std::string(name)+" must be an integer");
    return value.get<uint64_t>();
}
static uint64_t product(std::initializer_list<uint64_t> values) {
    uint64_t result=1;for(auto value:values){require(value>0&&result<=1073741824ULL/value,"component shape product exceeds 1 GiB hard envelope");result*=value;}return result;
}

struct Component {
    json spec;std::string kind,device,src,dst;
    uint32_t seed=0;uint64_t budget=0,data_bytes=0;
    ggml_backend_t cpu=nullptr,gpu=nullptr,compute_backend=nullptr;
    ggml_threadpool_t pool=nullptr;
    ggml_context *primary=nullptr,*secondary=nullptr;
    ggml_backend_buffer_t primary_buffer=nullptr,secondary_buffer=nullptr;
    std::unique_ptr<Graph> graph;
    ggml_tensor *left=nullptr,*right=nullptr,*output=nullptr;
    std::vector<uint8_t> copy_source;
    json geometry;
    explicit Component(const json &source,const std::string &poll):spec(source) {
        require(source.at("format_version")=="steady-component-program-1","component program schema differs");
        require(poll=="0","compact component poll must be explicit 0");
        kind=str(spec,"kind");device=str(spec,"device");require(device=="cpu"||device=="gpu","component device must be cpu or gpu");
        require(kind=="matmul"||kind=="attention_pv"||kind=="transfer","unknown compact component kind");
        const auto numeric_seed=state_response::integer(spec.at("seed"),"component seed");require(numeric_seed<=UINT32_MAX,"seed must be uint32");seed=static_cast<uint32_t>(numeric_seed);
        budget=static_cast<uint64_t>(positive(spec.at("allocation_budget_bytes"),"allocation budget",1073741824));
        const int threads=static_cast<int>(positive(spec.at("threads"),"threads",256));
        cpu=ggml_backend_cpu_init();require(cpu,"component CPU backend init failed");
        ggml_backend_cpu_set_n_threads(cpu,threads);auto pool_params=ggml_threadpool_params_default(threads);pool_params.poll=0;
        pool=ggml_threadpool_new(&pool_params);require(pool,"component CPU threadpool init failed");ggml_backend_cpu_set_threadpool(cpu,pool);
        bool need_gpu=device=="gpu";
        if(kind=="transfer") {
            src=str(spec.at("shape"),"src");dst=str(spec.at("shape"),"dst");
            require((src=="dram"||src=="vram")&&(dst=="dram"||dst=="vram"),"unknown transfer memory domain");need_gpu=src=="vram"||dst=="vram";
            require(device==(need_gpu?"gpu":"cpu"),"transfer device must describe its actual participating GPU/CPU domain");
        }
        if(need_gpu){const int index=static_cast<int>(state_response::integer(spec.at("cuda_device"),"CUDA device"));gpu=ggml_backend_cuda_init(index);require(gpu,"component CUDA backend init failed");}
        compute_backend=device=="gpu"?gpu:cpu;
        const auto meta=ggml_tensor_overhead()*1024+ggml_graph_overhead_custom(1024,false);
        primary=ggml_init({meta,nullptr,true});require(primary,"component tensor metadata allocation failed");
        if(kind=="transfer") {
            const auto bytes=static_cast<uint64_t>(positive(spec.at("shape").at("bytes"),"transfer bytes",536870912));
            data_bytes=2*bytes;require(data_bytes<=budget,"transfer backing exceeds explicit allocation budget");
            secondary=ggml_init({ggml_tensor_overhead()*4,nullptr,true});require(secondary,"transfer target metadata allocation failed");
            left=ggml_new_tensor_1d(primary,GGML_TYPE_I8,bytes);output=ggml_new_tensor_1d(secondary,GGML_TYPE_I8,bytes);
            primary_buffer=ggml_backend_alloc_ctx_tensors(primary,src=="vram"?gpu:cpu);
            secondary_buffer=ggml_backend_alloc_ctx_tensors(secondary,dst=="vram"?gpu:cpu);
            require(primary_buffer&&secondary_buffer,"transfer backing allocation failed");
            copy_source.resize(static_cast<size_t>(bytes));for(size_t i=0;i<copy_source.size();++i)copy_source[i]=static_cast<uint8_t>(mix32(seed,static_cast<uint32_t>(i),0)&255u);
            ggml_backend_tensor_set(left,copy_source.data(),0,copy_source.size());
            geometry=json{{"bytes",bytes},{"src",src},{"dst",dst},{"distinct_backing_tensors",true},{"host_buffer_kind","GGML default CPU buffer; pinned status not inferred"}};
            return;
        }
        int64_t m=0,n=0,k=0,heads=0,kvheads=0,dim=0;
        const auto &shape=spec.at("shape");
        if(kind=="matmul") {
            k=positive(shape.at("k"),"GEMM k",1000000);m=positive(shape.at("m"),"GEMM m",1000000);n=positive(shape.at("n"),"GEMM n",1000000);
            data_bytes=4*(product({static_cast<uint64_t>(k),static_cast<uint64_t>(m)})+product({static_cast<uint64_t>(k),static_cast<uint64_t>(n)})+product({static_cast<uint64_t>(m),static_cast<uint64_t>(n)}));
            require(data_bytes<=budget,"GEMM backing exceeds explicit allocation budget");
            left=ggml_new_tensor_2d(primary,GGML_TYPE_F32,k,m);right=ggml_new_tensor_2d(primary,GGML_TYPE_F32,k,n);output=ggml_new_tensor_2d(primary,GGML_TYPE_F32,m,n);
            geometry=json{{"left_ne",{k,m,1,1}},{"right_ne",{k,n,1,1}},{"output_ne",{m,n,1,1}},{"gemm_flops",2ULL*k*m*n}};
        } else {
            heads=positive(shape.at("heads"),"PV heads",1024);kvheads=positive(shape.at("kv_heads"),"PV KV heads",1024);
            dim=positive(shape.at("head_dim"),"PV head dim",4096);n=positive(shape.at("query_tokens"),"PV queries",1000000);k=positive(shape.at("kv_tokens"),"PV history",1000000);
            require(heads%kvheads==0,"PV heads must be divisible by KV heads");
            const auto v=product({static_cast<uint64_t>(dim),static_cast<uint64_t>(k),static_cast<uint64_t>(kvheads)}),p=product({static_cast<uint64_t>(k),static_cast<uint64_t>(n),static_cast<uint64_t>(heads)}),y=product({static_cast<uint64_t>(dim),static_cast<uint64_t>(heads),static_cast<uint64_t>(n)});
            data_bytes=4*(v+p+y);require(data_bytes<=budget,"PV backing exceeds explicit allocation budget");
            left=ggml_new_tensor_3d(primary,GGML_TYPE_F32,dim,k,kvheads);right=ggml_new_tensor_3d(primary,GGML_TYPE_F32,k,n,heads);output=ggml_new_tensor_2d(primary,GGML_TYPE_F32,dim*heads,n);
            geometry=json{{"left_ne",{dim,k,kvheads,1}},{"right_ne",{k,n,heads,1}},{"output_ne",{dim*heads,n,1,1}},
                {"gemm_flops",2ULL*dim*k*n*heads},{"PV_boundary","V transpose+contiguous packing, GGML mul_mat, context merge and independent output-port materialization"}};
        }
        ggml_set_input(left);ggml_set_input(right);ggml_set_input(output);ggml_set_output(output);
        primary_buffer=ggml_backend_alloc_ctx_tensors(primary,compute_backend);require(primary_buffer,"component data backing allocation failed");
        std::vector<float> l(static_cast<size_t>(ggml_nelements(left))),r(static_cast<size_t>(ggml_nelements(right)));
        for(size_t i=0;i<l.size();++i){double value=uniform(seed,static_cast<uint32_t>(i),0)-.5;if(kind=="matmul")value/=std::sqrt(static_cast<double>(k));l[i]=static_cast<float>(value);}
        if(kind=="matmul")for(size_t i=0;i<r.size();++i)r[i]=static_cast<float>(uniform(seed,static_cast<uint32_t>(i),1)-.5);
        else for(size_t start=0;start<r.size();start+=static_cast<size_t>(k)) {
            double sum=0;for(int64_t i=0;i<k;++i)sum+=uniform(seed,static_cast<uint32_t>(start+i),1)+1.0/16777216.0;
            for(int64_t i=0;i<k;++i)r[start+i]=static_cast<float>((uniform(seed,static_cast<uint32_t>(start+i),1)+1.0/16777216.0)/sum);
        }
        ggml_backend_tensor_set(left,l.data(),0,ggml_nbytes(left));ggml_backend_tensor_set(right,r.data(),0,ggml_nbytes(right));
        graph=std::make_unique<Graph>();graph->backend=compute_backend;graph->ctx=ggml_init({meta,nullptr,true});require(graph->ctx,"component graph metadata allocation failed");
        graph->graph=ggml_new_graph_custom(graph->ctx,1024,false);
        ggml_tensor *value=nullptr;
        if(kind=="matmul")value=ggml_mul_mat(graph->ctx,left,right);
        else {
            auto *attended=ggml_mul_mat(graph->ctx,ggml_cont(graph->ctx,ggml_transpose(graph->ctx,left)),right);
            value=ggml_reshape_2d(graph->ctx,ggml_cont(graph->ctx,ggml_permute(graph->ctx,attended,0,2,1,3)),dim*heads,n);
        }
        auto *result=ggml_cpy(graph->ctx,value,output);ggml_set_output(result);ggml_build_forward_expand(graph->graph,result);
        for(int i=0;i<ggml_graph_n_nodes(graph->graph);++i) {
            auto *node=ggml_graph_node(graph->graph,i);if(node->op==GGML_OP_MUL_MAT)ggml_mul_mat_set_prec(node,GGML_PREC_F32);
            require(ggml_backend_supports_op(compute_backend,node),"component backend does not support exact GGML node");
        }
        graph->alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(compute_backend));
        require(graph->alloc&&ggml_gallocr_reserve(graph->alloc,graph->graph),"component graph reserve failed");
        graph->workspace=ggml_gallocr_get_buffer_size(graph->alloc,0);
        require(data_bytes+graph->workspace<=budget,"known component backing+gallocr exceeds explicit allocation budget");
        require(ggml_gallocr_alloc_graph(graph->alloc,graph->graph),"component graph allocation failed");
    }
    ~Component() {
        graph.reset();if(secondary_buffer)ggml_backend_buffer_free(secondary_buffer);if(primary_buffer)ggml_backend_buffer_free(primary_buffer);
        if(secondary)ggml_free(secondary);if(primary)ggml_free(primary);if(gpu)ggml_backend_free(gpu);if(cpu)ggml_backend_free(cpu);if(pool)ggml_threadpool_free(pool);
    }
    void synchronize(){ggml_backend_synchronize(compute_backend);}
    void enqueue(){require(graph!=nullptr,"async graph mode requires a compute component");require(ggml_backend_graph_compute_async(compute_backend,graph->graph)==GGML_STATUS_SUCCESS,"component async graph compute failed");}
    void action() {
        if(kind=="transfer") {ggml_backend_tensor_copy(left,output);if(gpu)ggml_backend_synchronize(gpu);ggml_backend_synchronize(cpu);}
        else {require(ggml_backend_graph_compute(compute_backend,graph->graph)==GGML_STATUS_SUCCESS,"component graph compute failed");synchronize();}
    }
    json validate_output() {
        if(kind=="transfer") {
            std::vector<uint8_t> observed(copy_source.size()),source(copy_source.size());
            ggml_backend_tensor_get(left,source.data(),0,source.size());ggml_backend_tensor_get(output,observed.data(),0,observed.size());
            require(source==copy_source&&observed==copy_source,"component transfer source or output bytes differ");
            return json{{"kind",kind},{"full_byte_equality",true},{"checked_bytes",copy_source.size()},
                {"source_sha256",sha256_bytes(source.data(),source.size())},{"output_sha256",sha256_bytes(observed.data(),observed.size())},
                {"independent_oracle_applied_in_native",false}};
        }
        std::vector<float> values(static_cast<size_t>(ggml_nelements(output)));ggml_backend_tensor_get(output,values.data(),0,ggml_nbytes(output));
        for(auto value:values)require(std::isfinite(value),"nonfinite compact component output");
        return json{{"kind",kind},{"output",values},{"output_ne",geometry.at("output_ne")},
            {"output_sha256",sha256_bytes(values.data(),values.size()*sizeof(float))},{"independent_oracle_applied_in_native",false}};
    }
    json describe()const {
        return json{{"kind",kind},{"device",device},{"shape",spec.at("shape")},{"geometry",geometry},{"seed",seed},
            {"input_generator","mix32-index-v1"},{"known_backing_bytes",data_bytes},{"gallocr_workspace_bytes",graph?json(graph->workspace):json(0)},
            {"allocation_budget_bytes",budget},{"CUDA_private_workspace_bytes",nullptr},{"complete_native_memory_claimed",false},
            {"matmul_precision",kind=="transfer"?json(nullptr):json("GGML_PREC_F32")},
            {"CPU_GPU_time_columns_additive",false},{"graph_construction_and_input_initialization_timed",false}};
    }
};
}
