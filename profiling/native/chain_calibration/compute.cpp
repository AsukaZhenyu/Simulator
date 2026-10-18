// Fresh protocol. Historical graph, data and numerical reference sources are
// included verbatim except for the generated terminal-main rename (see CMake).
#include "context_generated.inc"
#include <array>

struct Span {const char*name=nullptr;const char*thread="main";int64_t begin=0,end=0;};
struct Trace {
    std::array<Span,32> spans{};int64_t begin=0,end=0;
    int64_t cpu_begin=-1,cpu_end=-1,gpu_begin=-1,gpu_end=-1,join_begin=-1,join_end=-1;
};
static int64_t now_ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}
template<class F> static void part(Trace*t,int index,const char*name,const char*thread,F&&f){
    if(!t){f();return;}auto&s=t->spans.at(index);s.name=name;s.thread=thread;s.begin=now_ns();f();s.end=now_ns();
}
struct ChainOutput {
    Output base;std::map<std::string,std::vector<Trace>>traces;
    Times overhead;std::map<std::string,int>span_counts;
};
static TransformOptions chain_options(int argc,char**argv){
    // Preserve the historical parser and graph sources. The new protocol owns
    // its wider geometry budget, validated BEFORE the historical affinity/
    // option parser sees a bounded placeholder. Actual geometry is restored
    // before any data generation, graph construction, allocation or timing.
    std::map<std::string,std::string>values;std::vector<std::string>args;
    args.emplace_back(argv[0]);
    for(int n=1;n<argc;n+=2){if(n+1==argc||!values.emplace(argv[n],argv[n+1]).second)throw std::runtime_error("missing/duplicate chain option");args.emplace_back(argv[n]);args.emplace_back(argv[n+1]);}
    auto get=[&](const char*name){auto p=values.find(name);if(p==values.end())throw std::runtime_error(std::string("explicit chain geometry required: ")+name);return integer(p->second);};
    const int h=get("--hidden"),i=get("--intermediate"),cut=get("--cpu-intermediate"),heads=get("--heads"),kvh=get("--kv-heads"),d=get("--head-dim"),q=get("--tokens"),hist=get("--history");
    const bool ffn=values.at("--family")=="ffn";
    if(h<1||h>2048||i>8192||cut>8192||q<1||q>128||hist>4096||heads>1024||kvh>1024||d>2048)throw std::runtime_error("geometry exceeds explicit chain budget");
    if(ffn){if(!(0<cut&&cut<i)||hist||heads||kvh||d)throw std::runtime_error("invalid wider FFN geometry");}
    else if(values.at("--family")!="attention"||i||cut||heads<1||kvh<1||d<1||heads%kvh||heads*d!=h)throw std::runtime_error("invalid wider post-RoPE GQA geometry");
    for(size_t n=1;n<args.size();n+=2){
        if(args[n]=="--hidden")args[n+1]=std::to_string(ffn?std::min(h,1024):(h>1024?heads:h));
        if(ffn&&args[n]=="--intermediate")args[n+1]=std::to_string(std::min(i,4096));
        if(ffn&&args[n]=="--cpu-intermediate")args[n+1]="1";
        if(!ffn&&h>1024&&args[n]=="--head-dim")args[n+1]="1";
    }
    std::vector<char*>raw;for(auto&s:args)raw.push_back(s.data());auto t=transform_options(int(raw.size()),raw.data());
    t.common.h=h;t.common.i=i;t.cpu_intermediate=cut;t.common.d=d;return t;
}
template<class F>static void strategy(ChainOutput&r,const Options&o,const std::string&name,F&&f){
    // Completed-work totals have only the same outer timer as the historical
    // protocol. Diagnostics are separate invocations and never substituted.
    r.base.times[name]=measure(o,[&](){f(nullptr);});
    auto&traces=r.traces[name];traces.reserve(o.samples);
    for(int n=0;n<o.samples;++n){Trace t;t.begin=now_ns();f(&t);t.end=now_ns();traces.push_back(t);}
}
static void overhead(ChainOutput&r,const Options&o){
    r.overhead["clock_pair"]=measure(o,[](){volatile auto a=now_ns();volatile auto b=now_ns();(void)a;(void)b;});
    // The exact part() instrumentation, with no operation. Never subtracted.
    r.overhead["noop_stage_pair"]=measure(o,[](){Trace t;part(&t,0,"noop","main",[](){});volatile auto v=t.spans[0].end;(void)v;});
    for(auto&entry:r.traces){
        int count=0;for(auto&s:entry.second.front().spans)if(s.name)++count;
        r.span_counts[entry.first]=count;
        const bool branches=entry.second.front().cpu_begin>=0;
        r.overhead["noop_trace/"+entry.first]=measure(o,[count,branches](){Trace t;t.begin=now_ns();for(int n=0;n<count;++n)part(&t,n,"noop","main",[](){});if(branches){t.cpu_begin=now_ns();t.cpu_end=now_ns();t.gpu_begin=now_ns();t.gpu_end=now_ns();t.join_begin=now_ns();t.join_end=now_ns();}t.end=now_ns();volatile auto v=t.end;(void)v;});
    }
}
static void serial_stages(Session&s,Trace*t,int offset,const char*thread){
    int n=offset;for(auto name:{"gate","up","activation","down"})part(t,n++,name,thread,[&](){s.compute(name);});
}
static ChainOutput ffn_chain(const TransformOptions&t){
    const Options&o=t.common;ChainOutput r;auto&times=r.base.times;auto&checks=r.base.checks;
    const Data full=make_data(o);Options co=o,go=o;co.i=t.cpu_intermediate;go.i=o.i-co.i;
    const Data cd=shard_data(o,full,0,co.i),gd=shard_data(o,full,co.i,go.i);
    Session cpu(co,false,cd),gpu(go,true,gd),all_cpu(o,false,full),all_gpu(o,true,full);
    for(auto*s:{&cpu,&gpu,&all_cpu,&all_gpu})force_f32_matmul(*s);
    ExtraGraphs merge(cpu);auto*partial=merge.add("gpu_partial",o.h,o.q);auto*out=merge.add("sum_output",o.h,o.q);merge.allocate();
    std::unique_ptr<Graph> graph;auto*ctx=merge.begin(graph);merge.finish("sum",std::move(graph),ggml_cpy(ctx,ggml_add(ctx,cpu.ports.at("output"),partial),out));
    size_t maxbytes=ggml_nbytes(partial);for(auto name:{"gate_weight","up_weight","down_weight"})maxbytes=std::max(maxbytes,ggml_nbytes(gpu.ports.at(name)));
    HostTransfer transfer(maxbytes);Worker worker(o.device,t.dispatch_cpu_id);std::vector<float>delivered(size_t(o.h)*o.q);
    auto worker_measure=[&](const std::string&metric,auto&&f){worker.start([&](){
#if defined(_WIN32)
        GROUP_AFFINITY profile{};if(!GetThreadGroupAffinity(GetCurrentThread(),&profile)||profile.Group!=0||profile.Mask!=(DWORD_PTR(1)<<t.dispatch_cpu_id))throw std::runtime_error("GPU component profile worker affinity changed");
        r.base.observed_gpu_profile_mask=profile.Mask;r.base.observed_gpu_profile_group=profile.Group;
#endif
        times[metric]=measure(o,f);});worker.wait();};
    for(auto name:{"gate","up","activation","down","joint"}){
        times["cpu/shard_"+std::string(name)]=measure(o,[&](){cpu.compute(name);});
        worker_measure("gpu/shard_"+std::string(name),[&](){gpu.compute(name);});
        times["cpu/full_"+std::string(name)]=measure(o,[&](){all_cpu.compute(name);});
        times["gpu/full_"+std::string(name)]=measure(o,[&](){all_gpu.compute(name);});
    }
    auto copy=[&](){transfer.d2h(gpu.ports.at("output"),partial->data);};
    auto preload=[&](){for(auto name:{"gate_weight","up_weight","down_weight"})transfer.h2d(gd.at(name).data(),gpu.ports.at(name));};
    times["hybrid/output_d2h"]=measure(o,copy);times["cpu/sum"]=measure(o,[&](){merge.compute("sum");});
    times["gpu/output_d2h"]=measure(o,[&](){transfer.d2h(all_gpu.ports.at("output"),delivered.data());});
    times["hybrid/preload_weights"]=measure(o,preload);
    for(auto name:{"gate","up","down"})times["hybrid/preload_"+std::string(name)+"_weight_h2d"]=measure(o,[&](){std::string key=std::string(name)+"_weight";transfer.h2d(gd.at(key).data(),gpu.ports.at(key));});
    times["hybrid/dispatch_barrier"]=measure(o,[&](){worker.start([](){});worker.wait();});
    auto mixed=[&](Trace*tr,bool joint,bool reload){
        if(reload){int n=0;for(auto name:{"gate","up","down"})part(tr,n++,name==std::string("gate")?"preload_gate_weight_h2d":name==std::string("up")?"preload_up_weight_h2d":"preload_down_weight_h2d","main",[&](){std::string key=std::string(name)+"_weight";transfer.h2d(gd.at(key).data(),gpu.ports.at(key));});}
        part(tr,3,"dispatch","main",[&](){worker.start([&](){
            if(tr)tr->gpu_begin=now_ns();
            if(joint)part(tr,8,"gpu_joint","dispatch",[&](){gpu.compute("joint");});else serial_stages(gpu,tr,8,"dispatch");
            if(tr)tr->gpu_end=now_ns();
        });});
        if(tr)tr->cpu_begin=now_ns();
        if(joint)part(tr,4,"cpu_joint","main",[&](){cpu.compute("joint");});else serial_stages(cpu,tr,4,"main");
        if(tr){tr->cpu_end=now_ns();tr->join_begin=now_ns();}
        part(tr,12,"join","main",[&](){worker.wait();});if(tr)tr->join_end=now_ns();
        part(tr,13,"output_d2h","main",copy);part(tr,14,"sum","main",[&](){merge.compute("sum");});
    };
    for(bool joint:{false,true}){
        std::string suffix=joint?"joint_resident":"resident";
        strategy(r,o,"cpu/"+suffix,[&](Trace*tr){if(joint)part(tr,0,"cpu_full_joint","main",[&](){all_cpu.compute("joint");});else serial_stages(all_cpu,tr,0,"main");});
        checks["cpu/"+suffix+"/output"]=compare(all_cpu.read("output"),full.at("output"));
        strategy(r,o,"gpu/"+suffix,[&](Trace*tr){if(joint)part(tr,0,"gpu_full_joint","main",[&](){all_gpu.compute("joint");});else serial_stages(all_gpu,tr,0,"main");part(tr,5,"output_d2h","main",[&](){transfer.d2h(all_gpu.ports.at("output"),delivered.data());});});
        checks["gpu/"+suffix+"/output"]=compare(delivered,full.at("output"));
        strategy(r,o,"hybrid/"+suffix,[&](Trace*tr){mixed(tr,joint,false);});checks["hybrid/"+suffix+"/output"]=compare(merge.read("sum_output"),full.at("output"));
    }
    strategy(r,o,"hybrid/preload",[&](Trace*tr){mixed(tr,false,true);});checks["hybrid/preload/output"]=compare(merge.read("sum_output"),full.at("output"));
    for(auto name:{"gate","up","activated","output"}){checks["cpu/shard/"+std::string(name)]=compare(cpu.read(name),cd.at(name));checks["gpu/shard/"+std::string(name)]=compare(gpu.read(name),gd.at(name));checks["cpu/full/"+std::string(name)]=compare(all_cpu.read(name),full.at(name));checks["gpu/full/"+std::string(name)]=compare(all_gpu.read(name),full.at(name));}
    r.base.observed_worker_mask=worker.observed_affinity_mask;r.base.observed_worker_group=worker.observed_group;
    r.base.inventory("cpu/shard/",cpu,cd);r.base.inventory("gpu/shard/",gpu,gd);r.base.inventory("cpu/full/",all_cpu,full);r.base.inventory("gpu/full/",all_gpu,full);r.base.extras("cpu/",merge);
    overhead(r,o);return r;
}
static ChainOutput attention_chain(const TransformOptions&t){
    const Options&o=t.common;ChainOutput r;auto&times=r.base.times;auto&checks=r.base.checks;
    Data data=make_data(o);Session cpu(o,false,data),gpu(o,true,data);force_f32_matmul(cpu);force_f32_matmul(gpu);
    ExtraGraphs c(cpu),g(gpu);attention_graphs(c);attention_graphs(g);
    HostTransfer transfer(std::max(ggml_nbytes(gpu.ports.at("output")),ggml_nbytes(g.ports.at("probabilities"))));std::vector<float>delivered(size_t(o.h)*o.q);
    for(auto name:{"qk","softmax","pv"}){times["cpu/"+std::string(name)]=measure(o,[&](){c.compute(name);});times["gpu/"+std::string(name)]=measure(o,[&](){g.compute(name);});}
    for(auto name:{"attention_output","joint"}){std::string metric=std::string(name)=="attention_output"?"output":"joint";times["cpu/"+metric]=measure(o,[&](){cpu.compute(name);});times["gpu/"+metric]=measure(o,[&](){gpu.compute(name);});}
    auto copy=[&](){transfer.d2h(gpu.ports.at("output"),delivered.data());};
    auto h2d=[&](){transfer.h2d(c.ports.at("probabilities")->data,g.ports.at("probabilities"));};
    times["hybrid/probabilities_h2d"]=measure(o,h2d);times["gpu/output_d2h"]=measure(o,copy);
    for(bool on_gpu:{false,true}){
        auto&s=on_gpu?gpu:cpu;auto&e=on_gpu?g:c;std::string dev=on_gpu?"gpu":"cpu";
        strategy(r,o,dev+"/resident",[&](Trace*tr){int n=0;for(auto name:{"qk","softmax","pv"})part(tr,n++,name,"main",[&](){e.compute(name);});part(tr,3,"attention_output","main",[&](){s.compute("attention_output");});if(on_gpu)part(tr,4,"output_d2h","main",copy);});
        checks[dev+"/resident/output"]=compare(on_gpu?delivered:cpu.read("output"),data.at("output"));
        strategy(r,o,dev+"/joint_resident",[&](Trace*tr){part(tr,0,"joint","main",[&](){s.compute("joint");});if(on_gpu)part(tr,1,"output_d2h","main",copy);});
        checks[dev+"/joint_resident/output"]=compare(on_gpu?delivered:cpu.read("output"),data.at("output"));
    }
    strategy(r,o,"hybrid/resident",[&](Trace*tr){part(tr,0,"cpu_qk","main",[&](){c.compute("qk");});part(tr,1,"cpu_softmax","main",[&](){c.compute("softmax");});part(tr,2,"probabilities_h2d","main",h2d);part(tr,3,"gpu_pv","main",[&](){g.compute("pv");});part(tr,4,"gpu_attention_output","main",[&](){gpu.compute("attention_output");});part(tr,5,"output_d2h","main",copy);});
    checks["hybrid/resident/output"]=compare(delivered,data.at("output"));
    const int len=o.hist+o.q;std::vector<float>scores(size_t(len)*o.q*o.heads),probabilities(scores.size());
    for(int head=0;head<o.heads;++head)for(int token=0;token<o.q;++token){
        const int kh=head/(o.heads/o.kvh);double largest=-1e100,total=0;
        for(int key=0;key<len;++key){double dot=0;for(int j=0;j<o.d;++j)dot+=double(data.at("query")[size_t(head*o.q+token)*o.d+j])*data.at("key_appended")[size_t(kh*len+key)*o.d+j];
            const size_t index=size_t(head*o.q+token)*len+key;scores[index]=float(dot/std::sqrt(double(o.d)));if(key<=o.hist+token)largest=std::max(largest,double(scores[index]));}
        for(int key=0;key<=o.hist+token;++key)total+=std::exp(double(scores[size_t(head*o.q+token)*len+key])-largest);
        for(int key=0;key<len;++key)probabilities[size_t(head*o.q+token)*len+key]=key<=o.hist+token?float(std::exp(double(scores[size_t(head*o.q+token)*len+key])-largest)/total):0;
    }
    checks["cpu/qk/scores"]=compare(c.read("scores"),scores);checks["gpu/qk/scores"]=compare(g.read("scores"),scores);
    checks["cpu/softmax/probabilities"]=compare(c.read("probabilities"),probabilities);checks["gpu/softmax/probabilities"]=compare(g.read("probabilities"),probabilities);
    for(auto dev:{"cpu","gpu"}){auto&s=std::string(dev)=="cpu"?cpu:gpu;for(auto name:{"context","key_appended","value_appended"})checks[std::string(dev)+"/"+name]=compare(s.read(name),data.at(name));for(auto name:{"key_cache","value_cache"})checks[std::string(dev)+"/"+name+"_unchanged"]=compare(s.read(name),data.at(name));}
    r.base.inventory("cpu/",cpu,data);r.base.inventory("gpu/",gpu,data);r.base.extras("cpu/",c);r.base.extras("gpu/",g);overhead(r,o);return r;
}
static void emit_times(const Times&times){bool first=true;std::cout<<'{';for(auto&p:times){std::cout<<(first?"":",")<<'"'<<p.first<<"\":";emit_array(p.second);first=false;}std::cout<<'}';}
static void emit_trace(const Trace&tr){
    const auto origin=tr.begin;
    std::cout<<"{\"begin_ns\":0,\"end_ns\":"<<tr.end-origin<<",\"stages\":[";
    bool first=true;for(auto&s:tr.spans)if(s.name){std::cout<<(first?"":",")<<"{\"name\":\""<<s.name<<"\",\"thread\":\""<<s.thread<<"\",\"begin_ns\":"<<s.begin-origin<<",\"end_ns\":"<<s.end-origin<<'}';first=false;}
    std::cout<<"],\"branch_brackets\":";
    if(tr.cpu_begin<0){std::cout<<"null,\"branch_start_skew_ns\":null,\"join_wait_ns\":null,\"completed_work_bracket_overlap_ns\":null}";return;}
    auto bracket=[&](const char*name,int64_t begin,int64_t end){std::cout<<'"'<<name<<"\":{\"begin_ns\":"<<begin-origin<<",\"end_ns\":"<<end-origin<<'}';};
    std::cout<<'{';bracket("cpu",tr.cpu_begin,tr.cpu_end);std::cout<<',';bracket("gpu_host",tr.gpu_begin,tr.gpu_end);std::cout<<',';bracket("join",tr.join_begin,tr.join_end);
    std::cout<<"},\"branch_start_skew_ns\":"<<tr.gpu_begin-tr.cpu_begin<<",\"join_wait_ns\":"<<tr.join_end-tr.join_begin<<",\"completed_work_bracket_overlap_ns\":"<<std::max<int64_t>(0,std::min(tr.cpu_end,tr.gpu_end)-std::max(tr.cpu_begin,tr.gpu_begin))<<'}';
}
static void emit_traces(const std::map<std::string,std::vector<Trace>>&traces){
    std::cout<<'{';bool first=true;for(auto&p:traces){std::cout<<(first?"":",")<<'"'<<p.first<<"\":[";first=false;bool ft=true;for(auto&tr:p.second){std::cout<<(ft?"":",");emit_trace(tr);ft=false;}std::cout<<']';}std::cout<<'}';
}
static int chain_run(const TransformOptions&t){
    const auto&o=t.common;auto r=o.family=="ffn"?ffn_chain(t):attention_chain(t);bool passed=true;for(auto&p:r.base.checks)passed=passed&&p.second.passed;
    std::cout<<std::setprecision(12)<<"{\"schema\":\"chain-calibration-native-0.1\",\"family\":\""<<o.family<<"\",\"backend\":\"hybrid\",\"dtype\":\"f32\",\"hidden\":"<<o.h<<",\"intermediate\":"<<o.i<<",\"cpu_intermediate\":"<<t.cpu_intermediate<<",\"tokens\":"<<o.q<<",\"history\":"<<o.hist<<",\"heads\":"<<o.heads<<",\"kv_heads\":"<<o.kvh<<",\"head_dim\":"<<o.d<<",\"threads\":"<<o.threads<<",\"cpu_id\":"<<o.cpu_id<<",\"dispatch_cpu_id\":"<<t.dispatch_cpu_id<<",\"device\":"<<o.device<<",\"seed\":"<<o.seed<<",\"samples\":"<<o.samples<<",\"warmup\":"<<o.warmup<<",\"min_warmup_ms\":"<<o.min_warmup_ms<<",\"cuda_graphs\":\"disabled\",\"timer\":\"steady_clock_completed_work\",\"matmul_precision\":\"GGML_PREC_F32_explicit\",\"nvidia_tf32_override\":\""<<(std::getenv("NVIDIA_TF32_OVERRIDE")?std::getenv("NVIDIA_TF32_OVERRIDE"):"unset")<<"\",\"source_binding\":{\"context_sha256\":\""<<CHAIN_CONTEXT_SHA<<"\",\"matched_sha256\":\""<<CHAIN_MATCHED_SHA<<"\",\"probe_sha256\":\""<<CHAIN_PROBE_SHA<<"\"},\"wall_ns\":";
    emit_times(r.base.times);std::cout<<",\"timestamp_overhead_ns\":";emit_times(r.overhead);
    std::cout<<",\"chain_traces\":";emit_traces(r.traces);bool first=true;
    std::cout<<",\"checks\":{";first=true;for(auto&p:r.base.checks){auto&c=p.second;std::cout<<(first?"":",")<<'"'<<p.first<<"\":{\"passed\":"<<(c.passed?"true":"false")<<",\"max_abs_error\":"<<c.maxabs<<",\"max_relative_error\":"<<c.maxrel<<",\"elements\":"<<c.elements<<",\"atol\":0.00001,\"rtol\":0.0001}";first=false;}
    std::cout<<"},\"ports\":{";first=true;for(auto&p:r.base.ports){std::cout<<(first?"":",")<<'"'<<p.first<<"\":{\"bytes\":"<<p.second.first<<",\"input_fixture_fnv1a64\":\""<<p.second.second<<"\"}";first=false;}
    std::cout<<"},\"profile_gpu_shard_thread\":\""<<(o.family=="ffn"?"persistent_dispatch_worker_same_as_parallel_path":"not_applicable")<<"\",\"profile_gpu_shard_affinity_mask\":"<<r.base.observed_gpu_profile_mask<<",\"profile_gpu_shard_affinity_group\":"<<r.base.observed_gpu_profile_group;
    std::cout<<",\"workspaces\":{";first=true;for(auto&p:r.base.workspace){std::cout<<(first?"":",")<<'"'<<p.first<<"\":{\"known_gallocr_plus_cplan_bytes\":"<<p.second.first<<",\"workspace_bytes\":";if(p.second.second)std::cout<<"null,\"source\":\"unknown_private_cuda_scratch\"}";else std::cout<<p.second.first<<",\"source\":\"gallocr_plus_explicit_cplan\"}";first=false;}
    std::cout<<"},\"observed_affinity\":{\"process_mask\":"<<t.observed_process_mask<<",\"main_thread_mask\":"<<t.observed_main_mask<<",\"main_thread_group\":"<<t.observed_main_group<<",\"dispatch_thread_mask\":"<<r.base.observed_worker_mask<<",\"dispatch_thread_group\":"<<r.base.observed_worker_group<<"},\"boundary\":{\"input_location\":\"all_required_immutable_inputs_already_resident_in_both_cpu_and_gpu\",\"output_location\":\"dram\",\"ffn_contract\":\"compact_selected_rows_gate_up_swiglu_down_no_norm_or_residual\",\"attention_contract\":\"post_RoPE_QKV_to_causal_GQA_QK_softmax_PV_sigmoid_gate_O_residual\",\"weights\":\"generated_f32_not_model_weights\",\"preload\":\"three_gpu_shard_weights_each_invocation_no_cold_cache_claim\",\"trace_scope\":\"separate_instrumented_completed_work_host_intervals_not_CUDA_kernel_intervals\",\"overhead_subtracted\":false},\"kernel_overlap_proved\":false,\"gpu_capacity_proved\":false,\"context_switch_attribution_proved\":false,\"all_checks_passed\":"<<(passed?"true":"false")<<"}\n";return passed?0:2;
}
int main(int argc,char**argv){try{return chain_run(chain_options(argc,argv));}catch(const std::exception&e){std::cerr<<"chain-calibration-probe: "<<e.what()<<'\n';return 1;}}
