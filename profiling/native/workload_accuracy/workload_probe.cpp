// Real traced decoder execution. Each layer owns persistent K/V and weights;
// current routes alone drive expert residency. No token-generation claim.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef WORKLOAD_HAVE_CUDA
#include "ggml-cuda.h"
#include <cuda_runtime_api.h>
#endif
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;
using Vec = std::vector<float>;
using Clock = std::chrono::steady_clock;
static int64_t ns(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count();
}
static void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}
static int number(const json &j, const char *key, int low, int high) {
    require(j.contains(key) && j.at(key).is_number_integer() && !j.at(key).is_boolean(), std::string("integer required: ")+key);
    int64_t v=j.at(key).get<int64_t>();
    require(v>=low && v<=high, std::string("integer outside bounded validation scope: ")+key);
    return int(v);
}
struct Config {
    int h,i,heads,kvheads,d,rd,cap,threads,device,seed;
    float eps,base;
    bool cuda,gate,qknorm;
    size_t budget;
    json input;
    explicit Config(json j):input(std::move(j)) {
        require(input.at("schema")=="workload_accuracy_case/1", "case schema");
        h=number(input,"hidden",2,256); i=number(input,"intermediate",1,512);
        heads=number(input,"heads",1,32); kvheads=number(input,"kv_heads",1,32);
        d=number(input,"head_dim",2,256); rd=number(input,"rope_dims",2,d);
        cap=number(input,"max_context",1,4096); threads=number(input,"threads",1,64);
        device=number(input,"cuda_device",0,15); seed=number(input,"seed",0,1000000);
        budget=size_t(number(input,"tracked_budget_bytes",1,512*1024*1024));
        eps=input.at("norm_eps").get<float>(); base=input.at("rope_base").get<float>();
        require(std::isfinite(eps)&&eps>0&&std::isfinite(base)&&base>1, "positive finite norm/RoPE parameters");
        require(h==heads*d && heads%kvheads==0 && rd%2==0, "invalid head geometry");
        auto backend=input.at("backend").get<std::string>();
        require(backend=="CPU"||backend=="CUDA", "unsupported backend; no fallback"); cuda=backend=="CUDA";
        auto ag=input.at("attention_gate").get<std::string>(), qk=input.at("qk_norm").get<std::string>();
        require(ag=="none"||ag=="sigmoid", "attention_gate"); gate=ag=="sigmoid";
        require(qk=="none"||qk=="rms", "qk_norm"); qknorm=qk=="rms";
        require(input.at("layers").is_array() && input.at("layers").size()>=2 && input.at("layers").size()<=8,"requires 2..8 layers");
        for (auto &layer:input.at("layers")) {
            auto kind=layer.at("kind").get<std::string>();
            require(kind=="dense"||kind=="moe","layer kind");
            if(kind=="moe") {
                int e=number(layer,"experts",1,32); number(layer,"top_k",1,e);
                require(layer.at("residency")=="all"||layer.at("residency")=="active","expert residency");
            }
        }
        const auto &rounds=input.at("rounds");
        require(rounds.is_array()&&rounds.size()>=2&&rounds.size()<=32,"requires 2..32 rounds");
        int history=0;
        for(auto &round:rounds) {
            auto &x=round.at("hidden_input"); require(x.is_array()&&x.size()>0&&x.size()<=32,"tokens per round 1..32");
            history+=int(x.size()); require(history<=cap,"trace exceeds persistent KV capacity");
            for(auto &row:x) {require(row.is_array()&&row.size()==size_t(h),"input hidden shape"); for(auto &v:row) require(v.is_number()&&std::isfinite(v.get<float>()),"finite hidden input");}
            require(round.at("routes").size()==input.at("layers").size()&&round.at("route_scores").size()==input.at("layers").size(),"one route entry per layer");
            for(size_t l=0;l<input.at("layers").size();++l) {
                auto &layer=input.at("layers").at(l); auto &routes=round.at("routes").at(l); auto &scores=round.at("route_scores").at(l);
                if(layer.at("kind")=="dense") {require(routes.is_null()&&scores.is_null(),"dense routes must be null");continue;}
                int k=layer.at("top_k"), e=layer.at("experts");
                require(routes.size()==x.size()&&scores.size()==x.size(),"routes must cover current tokens exactly");
                for(size_t t=0;t<x.size();++t) {
                    require(routes.at(t).size()==size_t(k)&&scores.at(t).size()==size_t(k),"top-k shape");
                    std::set<int> unique; double sum=0;
                    for(int s=0;s<k;++s) {auto &v=routes.at(t).at(s); require(v.is_number_integer()&&!v.is_boolean(),"expert integer");int id=v.get<int>(); require(id>=0&&id<e&&unique.insert(id).second,"invalid/duplicate expert");float score=scores.at(t).at(s).get<float>();require(std::isfinite(score)&&score>=0,"route score");sum+=score;}
                    require(std::abs(sum-1)<1e-5,"selected trace scores must sum to one");
                }
            }
        }
        require(size_t(h)*h*input.at("layers").size()*4 + size_t(h)*i*32*input.at("layers").size()*12 + size_t(cap)*kvheads*d*8*input.at("layers").size()<64*1024*1024,"bounded probe max declared tensor payload 64 MiB");
    }
};
struct Ledger {
    struct Item {std::string component; size_t requested,reserved;};
    size_t current=0, peak=0,budget; int round=-1,layer=-1; uint64_t serial=0,event_sequence=0;bool record;
    std::map<std::string,Item> live; json events=json::array();
    explicit Ledger(size_t limit,bool record_events):budget(limit),record(record_events){}
    std::string allocate(std::string component,size_t requested,size_t reserved) {
        require(reserved>=requested,"reserved storage smaller than logical requested storage");
        require(reserved<=budget && current<=budget-reserved,"tracked_budget_exceeded before allocation: "+component);
        std::string id="allocation"+std::to_string(serial++); current+=reserved; peak=std::max(peak,current);
        live.emplace(id,Item{component,requested,reserved});
        if(record)events.push_back({{"event","allocate"},{"sequence",event_sequence++},{"id",id},{"component",component},{"requested_bytes",requested},{"reserved_bytes",reserved},{"live_reserved_bytes",current},{"round",round},{"layer",layer}});
        return id;
    }
    void release(const std::string &id) {
        if(id.empty()) return;
        auto it=live.find(id);require(it!=live.end(),"double/untracked release");current-=it->second.reserved;
        if(record)events.push_back({{"event","release"},{"sequence",event_sequence++},{"id",id},{"component",it->second.component},{"requested_bytes",it->second.requested},{"reserved_bytes",it->second.reserved},{"live_reserved_bytes",current},{"round",round},{"layer",layer}});live.erase(it);
    }
};
struct Runtime {
    ggml_backend_t backend=nullptr; ggml_backend_buffer_type_t buft=nullptr;
    const Config &c; Ledger &ledger; json snapshots=json::array();bool validation;
    Runtime(const Config &config,Ledger &memory):c(config),ledger(memory),validation(memory.record) {
        if(c.cuda) {
#ifdef WORKLOAD_HAVE_CUDA
            require(cudaSetDevice(c.device)==cudaSuccess,"cudaSetDevice failed");backend=ggml_backend_cuda_init(c.device);
#else
            throw std::runtime_error("binary lacks CUDA; no CPU fallback");
#endif
        }else {backend=ggml_backend_cpu_init();if(backend)ggml_backend_cpu_set_n_threads(backend,c.threads);}
        require(backend!=nullptr,"backend initialization failed");buft=ggml_backend_get_default_buffer_type(backend);snapshot("after_backend_init");
    }
    ~Runtime(){if(backend)ggml_backend_free(backend);}
    void snapshot(const std::string &label) {
        if(!c.cuda||!validation) return;
#ifdef WORKLOAD_HAVE_CUDA
        ggml_backend_synchronize(backend);size_t free=0,total=0;require(cudaMemGetInfo(&free,&total)==cudaSuccess,"cudaMemGetInfo");
        snapshots.push_back({{"label",label},{"free_bytes",free},{"total_bytes",total},{"used_bytes",total-free},{"round",ledger.round},{"layer",ledger.layer}});
#endif
    }
};
struct Pack {
    Runtime &r; ggml_context *ctx=nullptr; ggml_backend_buffer_t buffer=nullptr;
    std::map<std::string,ggml_tensor*> tensors; std::map<std::string,Vec> values;
    std::string metadata_id,buffer_id,category; size_t logical=0;
    Pack(Runtime &runtime,std::string component):r(runtime),category(std::move(component)) {
        size_t bytes=ggml_tensor_overhead()*64;
        metadata_id=r.ledger.allocate(category+".metadata",bytes,bytes);ctx=ggml_init({bytes,nullptr,true});require(ctx,"resident metadata allocation");
    }
    ~Pack(){if(buffer)ggml_backend_buffer_free(buffer);r.ledger.release(buffer_id);if(ctx)ggml_free(ctx);r.ledger.release(metadata_id);}
    ggml_tensor *add(const std::string &name,int a,int b=1,Vec data={}) {
        auto *t=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,a,b);ggml_set_name(t,name.c_str());tensors[name]=t;logical+=ggml_nbytes(t);
        if(!data.empty()) values[name]=std::move(data);return t;
    }
    void allocate() {
        size_t reserved=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx,r.buft);
        buffer_id=r.ledger.allocate(category,logical,reserved);buffer=ggml_backend_alloc_ctx_tensors_from_buft(ctx,r.buft);require(buffer,"resident backend allocation");
        require(ggml_backend_buffer_get_size(buffer)==reserved,"resident reservation changed");ggml_backend_buffer_clear(buffer,0);
        for(auto &pair:values)ggml_backend_tensor_set(tensors.at(pair.first),pair.second.data(),0,pair.second.size()*4);
    }
    ggml_tensor *get(const std::string &name){return tensors.at(name);}
};
static Vec read(ggml_tensor *t){Vec v(ggml_nelements(t));ggml_backend_tensor_get(t,v.data(),0,ggml_nbytes(t));return v;}
struct Graph {
    Runtime &r; ggml_context *ctx=nullptr; ggml_cgraph *graph=nullptr;ggml_gallocr_t alloc=nullptr;
    ggml_cplan plan{};std::unique_ptr<uint8_t[]> work;std::string meta_id,graph_id,work_id;
    std::map<std::string,ggml_tensor*> ports;std::map<std::string,Vec> inputs;
    std::map<std::string,std::vector<int32_t>> indices;json ops=json::array();int64_t compute_ns=0;
    explicit Graph(Runtime &runtime):r(runtime) {
        size_t bytes=ggml_tensor_overhead()*4096+ggml_graph_overhead_custom(4096,false);
        meta_id=r.ledger.allocate("graph_metadata",bytes,bytes);ctx=ggml_init({bytes,nullptr,true});require(ctx,"graph metadata");graph=ggml_new_graph_custom(ctx,4096,false);
    }
    ~Graph(){if(alloc)ggml_gallocr_free(alloc);r.ledger.release(graph_id);work.reset();r.ledger.release(work_id);if(ctx)ggml_free(ctx);r.ledger.release(meta_id);}
    ggml_tensor *input(std::string name,int a,int b,const Vec &v){auto*t=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,a,b);ggml_set_input(t);ports[name]=t;inputs[name]=v;return t;}
    ggml_tensor *ids(std::string name,const std::vector<int32_t> &v){auto*t=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,int(v.size()));ggml_set_input(t);ports[name]=t;indices[name]=v;return t;}
    ggml_tensor *output(std::string name,ggml_tensor *t){ggml_set_output(t);ggml_set_name(t,name.c_str());ports[name]=t;ggml_build_forward_expand(graph,t);return t;}
    void run(){
        for(int n=0;n<ggml_graph_n_nodes(graph);++n){auto*t=ggml_graph_node(graph,n);if(t->op==GGML_OP_MUL_MAT)ggml_mul_mat_set_prec(t,GGML_PREC_F32);require(ggml_backend_supports_op(r.backend,t),std::string("backend unsupported op ")+ggml_op_name(t->op));if(r.validation)ops.push_back(ggml_op_name(t->op));}
        alloc=ggml_gallocr_new(r.buft);require(alloc,"graph allocator");size_t reserved=0;
        ggml_gallocr_reserve_n_size(alloc,graph,nullptr,nullptr,&reserved);
        graph_id=r.ledger.allocate("graph_buffers",reserved,reserved);
        // reserve_n_size measures and updates allocator topology but deliberately
        // leaves backing buffers null. Reserve actual buffers before alloc_graph.
        require(ggml_gallocr_reserve(alloc,graph),"graph reservation failed");
        require(ggml_gallocr_alloc_graph(alloc,graph),"graph allocation failed");require(ggml_gallocr_get_buffer_size(alloc,0)==reserved,"graph reservation changed");
        if(!r.c.cuda){plan=ggml_graph_plan(graph,r.c.threads,nullptr);work_id=r.ledger.allocate("cpu_workspace",plan.work_size,plan.work_size);work=std::make_unique<uint8_t[]>(plan.work_size);plan.work_data=work.get();}
        for(auto &pair:inputs)ggml_backend_tensor_set(ports.at(pair.first),pair.second.data(),0,pair.second.size()*4);
        for(auto &pair:indices)ggml_backend_tensor_set(ports.at(pair.first),pair.second.data(),0,pair.second.size()*4);
        auto start=Clock::now();auto status=r.c.cuda?ggml_backend_graph_compute(r.backend,graph):ggml_graph_compute(graph,&plan);ggml_backend_synchronize(r.backend);compute_ns=ns(start);require(status==GGML_STATUS_SUCCESS,"graph compute failed");
    }
};
static bool timed_phase=false;
static Vec weights(int a,int b,int layer,int role,int seed,bool norm=false){
    static std::map<std::string,Vec> host_weight_source;
    auto key=std::to_string(a)+":"+std::to_string(b)+":"+std::to_string(layer)+":"+std::to_string(role)+":"+std::to_string(seed)+":"+std::to_string(norm);
    auto found=host_weight_source.find(key);if(found!=host_weight_source.end())return found->second;
    require(!timed_phase,"weight source was not prepared outside inference timing");
    Vec v(size_t(a)*b);for(size_t j=0;j<v.size();++j)v[j]=(norm?1.f:0.f)+float(std::sin(double(j+1)*(role+11)*(layer+1)*.013+seed*.1))*(norm?.03f:.18f/std::sqrt(float(a)));
    host_weight_source[key]=v;return v;
}
static Vec mat(const Vec&w,const Vec&x,int in,int out,int n){Vec y(size_t(out)*n);for(int t=0;t<n;++t)for(int o=0;o<out;++o){double s=0;for(int j=0;j<in;++j)s+=double(w[size_t(o)*in+j])*x[size_t(t)*in+j];y[size_t(t)*out+o]=float(s);}return y;}
static Vec norm(const Vec&x,const Vec&gamma,int width,float eps){Vec y(x.size());for(size_t p=0;p<x.size();p+=width){double sum=0;for(int j=0;j<width;++j)sum+=double(x[p+j])*x[p+j];float scale=float(1/std::sqrt(sum/width+eps));for(int j=0;j<width;++j)y[p+j]=x[p+j]*scale*gamma[j];}return y;}
static Vec rope(Vec v,const Config&c,int heads,int n,int history){for(int t=0;t<n;++t)for(int h=0;h<heads;++h)for(int j=0;j<c.rd;j+=2){size_t p=size_t(t)*heads*c.d+h*c.d+j;double theta=(history+t)*std::pow(double(c.base),-double(j)/c.rd);float co=float(std::cos(theta)),si=float(std::sin(theta));float x=v[p],y=v[p+1];v[p]=x*co-y*si;v[p+1]=x*si+y*co;}return v;}
static ggml_tensor* rms(Graph&g,ggml_tensor*x,ggml_tensor*gamma,const Config&c){return ggml_mul(g.ctx,ggml_rms_norm(g.ctx,x,c.eps),gamma);}
static ggml_tensor* native_rope(Graph&g,ggml_tensor*x,ggml_tensor*pos,const Config&c,int heads,int n){return ggml_reshape_2d(g.ctx,ggml_rope_ext(g.ctx,ggml_reshape_3d(g.ctx,x,c.d,heads,n),pos,nullptr,c.rd,0,0,c.base,1.f,0.f,1.f,0.f,0.f),heads*c.d,n);}
static json check(const Vec&actual,const Vec&expected,const std::string&name){require(actual.size()==expected.size(),"port shape mismatch "+name);double max=0,err=0,scale=0;for(size_t j=0;j<actual.size();++j){require(std::isfinite(actual[j])&&std::isfinite(expected[j]),"nonfinite "+name);double d=std::abs(double(actual[j])-expected[j]);max=std::max(max,d);err+=d*d;scale+=double(expected[j])*expected[j];require(d<=2e-4+2e-3*std::abs(expected[j]),"numerical mismatch "+name);}return {{"port",name},{"elements",actual.size()},{"max_abs_error",max},{"relative_l2",std::sqrt(err/std::max(scale,1e-30))},{"passed",true}};}
struct Layer {
    Runtime&r;int index;json spec;std::unique_ptr<Pack> base,kv,dense;std::map<int,std::unique_ptr<Pack>> experts;
    Vec refk,refv;int history=0,route_known_round=-1;json residency=json::array();
    void mark(json event){if(r.validation&&spec.at("kind")=="moe"){event["sequence"]=r.ledger.event_sequence++;event["round"]=r.ledger.round;residency.push_back(std::move(event));}}
    Layer(Runtime&runtime,int id,json shape):r(runtime),index(id),spec(std::move(shape)){
        auto&c=r.c;int kd=c.kvheads*c.d;base=std::make_unique<Pack>(r,"resident_weights");
        for(auto name:{"input_norm","post_attention_norm"})base->add(name,c.h,1,weights(c.h,1,id,name==std::string("input_norm")?1:2,c.seed,true));
        base->add("q",c.h,c.h,weights(c.h,c.h,id,3,c.seed));base->add("k",c.h,kd,weights(c.h,kd,id,4,c.seed));base->add("v",c.h,kd,weights(c.h,kd,id,5,c.seed));base->add("output",c.h,c.h,weights(c.h,c.h,id,6,c.seed));
        if(c.gate)base->add("gate_proj",c.h,c.h,weights(c.h,c.h,id,7,c.seed));
        if(c.qknorm){base->add("q_norm",c.d,1,weights(c.d,1,id,8,c.seed,true));base->add("k_norm",c.d,1,weights(c.d,1,id,9,c.seed,true));}
        base->allocate();kv=std::make_unique<Pack>(r,"kv_capacity");kv->add("k",kd,c.cap);kv->add("v",kd,c.cap);kv->allocate();
        if(spec.at("kind")=="dense")dense=expert(-1);
        else if(spec.at("residency")=="all")for(int e=0;e<spec.at("experts").get<int>();++e)experts[e]=expert(e);
    }
    std::unique_ptr<Pack> expert(int e){auto&c=r.c;auto p=std::make_unique<Pack>(r,e<0?"resident_dense_ffn":"resident_expert."+std::to_string(e));for(auto name:{"gate","up"})p->add(name,c.h,c.i,weights(c.h,c.i,index,20+e*3+(name==std::string("gate")?0:1),c.seed));p->add("down",c.i,c.h,weights(c.i,c.h,index,22+e*3,c.seed));p->allocate();return p;}
    void current_routes(const json&routes){if(spec.at("kind")=="dense")return;require(route_known_round==r.ledger.round,"route-based residency before current router barrier");std::set<int>active;for(auto &row:routes)for(auto&id:row)active.insert(id.get<int>());
        if(spec.at("residency")=="active"){for(auto it=experts.begin();it!=experts.end();){if(!active.count(it->first)){mark({{"event","unload"},{"expert",it->first}});it=experts.erase(it);}else++it;}for(int e:active)if(!experts.count(e)){experts[e]=expert(e);mark({{"event","load"},{"expert",e}});}}
        if(r.validation){json ids=json::array(),allocs=json::object();for(auto&pair:experts){ids.push_back(pair.first);allocs[std::to_string(pair.first)]=pair.second->buffer_id;}mark({{"event","route_visible"},{"active_experts",active},{"resident_experts",ids},{"allocation_ids",allocs},{"information_scope","current_layer_round_only"}});}
    }
    Vec reference_ffn(const Vec&x,const json&routes,const json&scores,int n){auto&c=r.c;Vec y=x;auto z=norm(x,base->values.at("post_attention_norm"),c.h,c.eps);
        auto evaluate=[&](Pack&p,const Vec&input,int count){auto a=mat(p.values.at("gate"),input,c.h,c.i,count),u=mat(p.values.at("up"),input,c.h,c.i,count);for(size_t j=0;j<a.size();++j)a[j]=float(double(a[j])/(1+std::exp(-double(a[j]))))*u[j];return mat(p.values.at("down"),a,c.i,c.h,count);};
        if(dense){auto f=evaluate(*dense,z,n);for(size_t j=0;j<y.size();++j)y[j]+=f[j];}
        else for(int t=0;t<n;++t)for(int slot=0;slot<spec.at("top_k").get<int>();++slot){int e=routes.at(t).at(slot);Vec input(z.begin()+size_t(t)*c.h,z.begin()+size_t(t+1)*c.h);auto f=evaluate(*experts.at(e),input,1);float score=scores.at(t).at(slot);for(int j=0;j<c.h;++j)y[size_t(t)*c.h+j]+=f[j]*score;}
        return y;
    }
    Vec run(const Vec&x,const Vec&refx,int n,const json&routes,const json&scores,json&row){auto&c=r.c;int kd=c.kvheads*c.d,total=history+n;json checks=json::array(),stages=json::array();
        Vec rq,rk,rv,rg;
        if(r.validation){auto z=norm(refx,base->values.at("input_norm"),c.h,c.eps);rq=mat(base->values.at("q"),z,c.h,c.h,n);rk=mat(base->values.at("k"),z,c.h,kd,n);rv=mat(base->values.at("v"),z,c.h,kd,n);
        if(c.gate)rg=mat(base->values.at("gate_proj"),z,c.h,c.h,n);
        if(c.qknorm){rq=norm(rq,base->values.at("q_norm"),c.d,c.eps);rk=norm(rk,base->values.at("k_norm"),c.d,c.eps);}
        rq=rope(rq,c,c.heads,n,history);rk=rope(rk,c,c.kvheads,n,history);}
        Vec q,k,v,gate;
        {Graph g(r);auto*in=g.input("hidden_input",c.h,n,x);auto*normalized=rms(g,in,base->get("input_norm"),c);auto*qt=ggml_mul_mat(g.ctx,base->get("q"),normalized);auto*kt=ggml_mul_mat(g.ctx,base->get("k"),normalized);if(c.qknorm){qt=ggml_reshape_2d(g.ctx,rms(g,ggml_reshape_3d(g.ctx,qt,c.d,c.heads,n),base->get("q_norm"),c),c.h,n);kt=ggml_reshape_2d(g.ctx,rms(g,ggml_reshape_3d(g.ctx,kt,c.d,c.kvheads,n),base->get("k_norm"),c),kd,n);}
        std::vector<int32_t>positions(n);for(int t=0;t<n;++t)positions[t]=history+t;auto*pos=g.ids("positions",positions);g.output("q",native_rope(g,qt,pos,c,c.heads,n));g.output("k",native_rope(g,kt,pos,c,c.kvheads,n));g.output("v",ggml_mul_mat(g.ctx,base->get("v"),normalized));if(c.gate)g.output("gate_proj",ggml_mul_mat(g.ctx,base->get("gate_proj"),normalized));g.run();q=read(g.ports.at("q"));k=read(g.ports.at("k"));v=read(g.ports.at("v"));if(c.gate)gate=read(g.ports.at("gate_proj"));stages.push_back({{"kind","projection"},{"compute_ns",g.compute_ns},{"ggml_ops",g.ops}});}
        mark({{"event","projection_complete"}});
        if(r.validation){checks.push_back(check(q,rq,"q"));checks.push_back(check(k,rk,"k_new"));checks.push_back(check(v,rv,"v_new"));if(c.gate)checks.push_back(check(gate,rg,"gate_proj"));}
        ggml_backend_tensor_set(kv->get("k"),k.data(),size_t(history)*kd*4,k.size()*4);ggml_backend_tensor_set(kv->get("v"),v.data(),size_t(history)*kd*4,v.size()*4);
        Vec rattn;
        if(r.validation){refk.insert(refk.end(),rk.begin(),rk.end());refv.insert(refv.end(),rv.begin(),rv.end());
        Vec fullk(size_t(total)*kd),fullv(size_t(total)*kd);ggml_backend_tensor_get(kv->get("k"),fullk.data(),0,fullk.size()*4);ggml_backend_tensor_get(kv->get("v"),fullv.data(),0,fullv.size()*4);checks.push_back(check(fullk,refk,"persistent_k"));checks.push_back(check(fullv,refv,"persistent_v"));
        Vec context(size_t(c.h)*n,0);int group=c.heads/c.kvheads;for(int t=0;t<n;++t)for(int head=0;head<c.heads;++head){int kh=head/group;std::vector<double>p(history+t+1);double max=-1e100,sum=0;for(int s=0;s<=history+t;++s){double dot=0;for(int j=0;j<c.d;++j)dot+=double(refk[size_t(s)*kd+kh*c.d+j])*rq[size_t(t)*c.h+head*c.d+j];p[s]=dot/std::sqrt(double(c.d));max=std::max(max,p[s]);}for(auto&score:p){score=std::exp(score-max);sum+=score;}for(int j=0;j<c.d;++j){double a=0;for(int s=0;s<=history+t;++s)a+=p[s]/sum*refv[size_t(s)*kd+kh*c.d+j];context[size_t(t)*c.h+head*c.d+j]=float(a);}}
        if(c.gate)for(size_t j=0;j<context.size();++j)context[j]/=float(1+std::exp(-double(rg[j])));rattn=mat(base->values.at("output"),context,c.h,c.h,n);for(size_t j=0;j<rattn.size();++j)rattn[j]+=refx[j];}
        int group=c.heads/c.kvheads;Vec attention;
        {Graph g(r);auto*qin=g.input("q",c.h,n,q);ggml_tensor*combined=nullptr;for(int head=0;head<c.heads;++head){int kh=head/group;auto*qh=ggml_view_2d(g.ctx,qin,c.d,n,size_t(c.h)*4,size_t(head*c.d)*4);auto*kk=ggml_view_2d(g.ctx,kv->get("k"),c.d,total,size_t(kd)*4,size_t(kh*c.d)*4);auto*vv=ggml_view_2d(g.ctx,kv->get("v"),c.d,total,size_t(kd)*4,size_t(kh*c.d)*4);auto*logits=ggml_scale(g.ctx,ggml_mul_mat(g.ctx,kk,qh),1.f/std::sqrt(float(c.d)));auto*probs=ggml_soft_max(g.ctx,ggml_diag_mask_inf(g.ctx,logits,history));auto*out=ggml_mul_mat(g.ctx,ggml_cont(g.ctx,ggml_transpose(g.ctx,vv)),probs);combined=combined?ggml_concat(g.ctx,combined,out,0):out;}
        if(c.gate)combined=ggml_mul(g.ctx,combined,ggml_sigmoid(g.ctx,g.input("gate",c.h,n,gate)));auto*residual=g.input("residual",c.h,n,x);g.output("attention_output",ggml_add(g.ctx,ggml_mul_mat(g.ctx,base->get("output"),combined),residual));g.run();attention=read(g.ports.at("attention_output"));stages.push_back({{"kind","attention"},{"compute_ns",g.compute_ns},{"ggml_ops",g.ops}});}
        mark({{"event","attention_complete"}});
        // A trace-provided router barrier. No route-derived load, eviction or
        // expert work may occur before attention has completed and IDs/scores
        // have been explicitly revealed for this layer and round.
        auto barrier_start=Clock::now();
        if(!dense){route_known_round=r.ledger.round;mark({{"event","route_known"},{"routes",routes},{"selected_scores",scores},{"learned_router_executed",false}});current_routes(routes);mark({{"event","expert_ffn_begin"}});}
        int64_t route_barrier_residency_ns=ns(barrier_start);
        Vec expected;if(r.validation){checks.push_back(check(attention,rattn,"attention_output"));expected=reference_ffn(rattn,routes,scores,n);}Vec output;
        {Graph g(r);auto*in=g.input("attention_output",c.h,n,attention);auto*normalized=rms(g,in,base->get("post_attention_norm"),c);auto*sum=in;
        auto ffn=[&](Pack&p,ggml_tensor*input){return ggml_mul_mat(g.ctx,p.get("down"),ggml_swiglu_split(g.ctx,ggml_mul_mat(g.ctx,p.get("gate"),input),ggml_mul_mat(g.ctx,p.get("up"),input)));};
        if(dense)sum=ggml_add(g.ctx,sum,ffn(*dense,normalized));else for(auto&pair:experts){int e=pair.first;std::vector<int32_t>rows;Vec coeff;for(int t=0;t<n;++t)for(int slot=0;slot<spec.at("top_k").get<int>();++slot)if(routes.at(t).at(slot)==e){rows.push_back(t);coeff.push_back(scores.at(t).at(slot).get<float>());}if(rows.empty())continue;auto*ids=g.ids("rows"+std::to_string(e),rows);auto*grouped=ggml_get_rows(g.ctx,normalized,ids);auto*weighted=ggml_mul(g.ctx,ffn(*pair.second,grouped),g.input("scores"+std::to_string(e),1,int(rows.size()),coeff));sum=ggml_add(g.ctx,sum,ggml_get_rows_back(g.ctx,weighted,ids,in));}
        g.output("hidden_output",sum);g.run();output=read(g.ports.at("hidden_output"));stages.push_back({{"kind",dense?"dense_ffn":"moe_ffn_trace"},{"compute_ns",g.compute_ns},{"ggml_ops",g.ops}});}
        mark({{"event","expert_ffn_complete"}});
        if(r.validation)checks.push_back(check(output,expected,"hidden_output"));history=total;row={{"round",r.ledger.round},{"layer",index},{"kind",spec.at("kind")},{"tokens",n},{"history_before",total-n},{"history_after",total},{"kv_used_bytes",size_t(total)*kd*8},{"kv_capacity_bytes",size_t(c.cap)*kd*8},{"stages",stages},{"trace_barrier_and_residency_ns",route_barrier_residency_ns}};
        if(r.validation){row["checks"]=checks;row["hidden_input"]=x;row["hidden_output"]=output;row["reference_output"]=expected;row["input_boundary"]="hidden_activation";row["output_boundary"]="hidden_activation";}return output;
    }
};
static json run(const Config&c,bool validation){Ledger ledger(c.budget,validation);Runtime runtime(c,ledger);json rows=json::array(),residency=json::array(),rounds=json::array();std::vector<Vec> round_outputs;std::vector<int64_t>round_times;int64_t compute=0;auto start=Clock::now();{
    std::vector<std::unique_ptr<Layer>>layers;for(size_t l=0;l<c.input.at("layers").size();++l){ledger.layer=int(l);layers.push_back(std::make_unique<Layer>(runtime,int(l),c.input.at("layers").at(l)));}runtime.snapshot("after_resident_allocation");
    for(size_t ri=0;ri<c.input.at("rounds").size();++ri){ledger.round=int(ri);auto rs=Clock::now();auto&round=c.input.at("rounds").at(ri);Vec x;for(auto&row:round.at("hidden_input"))for(auto&v:row)x.push_back(v.get<float>());Vec refx=validation?x:Vec{};int n=int(round.at("hidden_input").size());for(size_t l=0;l<layers.size();++l){ledger.layer=int(l);json row;x=layers[l]->run(x,refx,n,round.at("routes").at(l),round.at("route_scores").at(l),row);if(validation)refx=row.at("reference_output").get<Vec>();for(auto&stage:row.at("stages"))compute+=stage.at("compute_ns").get<int64_t>();rows.push_back(row);}round_times.push_back(ns(rs));round_outputs.push_back(std::move(x));runtime.snapshot("after_round");}
    if(validation)for(auto&layer:layers)residency.push_back({{"layer",layer->index},{"events",layer->residency},{"final_history",layer->history}});
    ledger.round=int(c.input.at("rounds").size());ledger.layer=-1;
}int64_t wall=ns(start);runtime.snapshot("after_workload_release");require(ledger.current==0&&ledger.live.empty(),"allocation leak");
    // Port serialization and the scalar validator are outside inference timing.
    for(size_t ri=0;ri<round_outputs.size();++ri)rounds.push_back({{"round",ri},{"wall_ns",round_times[ri]},{"hidden_output",round_outputs[ri]}});
    return {{"schema","ggml_workload_accuracy/2"},{"causal_route_boundary","trace_known_after_attention_before_route_based_residency_and_ffn"},{"learned_router_executed",false},{"case_id",c.input.at("id")},{"backend",c.cuda?"CUDA":"CPU"},{"cuda_device",c.device},{"hardware_compute_performed",true},{"dtype","f32"},{"execution","actual_input_trace_multilayer_persistent_kv"},{"autoregressive_token_generation",false},{"numeric_pass",validation},{"rows",rows},{"rounds",rounds},{"residency",residency},{"timing",{{"full_workload_wall_ns",wall},{"stage_compute_sum_ns",compute},{"host_setup_materialization_gap_ns",wall-compute},{"scope",validation?"untimed_validation_diagnostics":"inference_graph_setup_alloc_materialization_weight_lifecycle_release"},{"scalar_reference_in_timed_region",validation},{"port_serialization_in_timed_region",false},{"backend_initialization_in_timed_region",false},{"memory_event_logging_in_timed_region",validation},{"source_weight_generation_in_timed_region",validation},{"suitable_for_uninstrumented_runtime_prediction",false}}},{"memory",{{"scope","tracked_backend_buffers_graph_metadata_explicit_cpu_work_only"},{"budget_bytes",c.budget},{"peak_tracked_reserved_bytes",ledger.peak},{"final_tracked_reserved_bytes",ledger.current},{"events",ledger.events},{"tracked_budget_pass",true},{"physical_capacity_verified",false},{"process_peak_bytes",nullptr},{"cuda_private_workspace_bytes",nullptr},{"cuda_private_workspace_bound_known",false},{"cuda_meminfo_scope","device_wide_diagnostic_snapshots_not_process_peak"},{"cuda_meminfo",runtime.snapshots},{"host_reference_control_storage_tracked",false}}}};
}
int main(int argc,char**argv){try{
    require(argc==7&&std::string(argv[1])=="--case"&&std::string(argv[3])=="--samples"&&std::string(argv[5])=="--warmup","usage: ggml-workload-accuracy-probe --case CASE.json --samples N --warmup N");
    int samples=std::stoi(argv[4]),warmup=std::stoi(argv[6]);require(samples>=3&&samples<=100&&warmup>=0&&warmup<=10,"samples 3..100 and warmup 0..10 required");
    std::ifstream file(argv[2]);require(bool(file),"cannot open case");json j;file>>j;Config c(j);
    if(c.cuda){const char*tf32=std::getenv("NVIDIA_TF32_OVERRIDE");const char*graphs=std::getenv("GGML_CUDA_DISABLE_GRAPHS");require(tf32&&std::string(tf32)=="0","CUDA F32 requires NVIDIA_TF32_OVERRIDE=0");require(graphs&&std::string(graphs)=="1","CUDA requires GGML_CUDA_DISABLE_GRAPHS=1");}
    auto validation_start=Clock::now();json result=run(c,true);result["validation_pass_wall_ns"]=ns(validation_start);
    timed_phase=true;for(int w=0;w<warmup;++w)run(c,false);
    json timings=json::array();for(int s=0;s<samples;++s){auto measured=run(c,false);json stages=json::array();for(auto&row:measured.at("rows"))for(auto&stage:row.at("stages"))stages.push_back({{"round",row.at("round")},{"layer",row.at("layer")},{"kind",stage.at("kind")},{"compute_ns",stage.at("compute_ns")}});
        json checks=json::array();for(size_t ri=0;ri<measured.at("rounds").size();++ri)checks.push_back(check(measured.at("rounds").at(ri).at("hidden_output").get<Vec>(),result.at("rounds").at(ri).at("hidden_output").get<Vec>(),"timed_round_output"));
        timings.push_back({{"sample",s},{"timing",measured.at("timing")},{"stages",stages},{"output_checks_after_timing",checks},{"peak_tracked_reserved_bytes",measured.at("memory").at("peak_tracked_reserved_bytes")}});}
    result["sample_timings"]=timings;result["samples"]=samples;result["warmup"]=warmup;result["each_sample_resets_workload_state"]=true;
    result["precision"]={{"matmul", "GGML_PREC_F32"},{"NVIDIA_TF32_OVERRIDE",c.cuda?json(std::getenv("NVIDIA_TF32_OVERRIDE")):json(nullptr)},{"GGML_CUDA_DISABLE_GRAPHS",c.cuda?json(std::getenv("GGML_CUDA_DISABLE_GRAPHS")):json(nullptr)}};
    std::cout<<result.dump()<<'\n';return 0;
}catch(const std::exception&e){std::cerr<<"ggml-workload-accuracy-probe: "<<e.what()<<'\n';return 1;}}
