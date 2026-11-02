#pragma once
#define main historical_steady_entry_not_called
#include "../../../native/steady_costs/probe.cpp"
#undef main
#include "../owned_timer.hpp"
#include "../../../native/physical_stages/cupti_capture.hpp"

namespace diagnostic_v3 {
static json pointer_kind(void *pointer) {
    cudaPointerAttributes a{};auto status=cudaPointerGetAttributes(&a,pointer);
    if(status==cudaErrorInvalidValue){cudaGetLastError();return json{{"status","observed"},{"registered_host",false},{"unregistered_pageable",true}};}
    require(status==cudaSuccess,"host pointer query failed");
    return json{{"status","observed"},{"cuda_memory_type",static_cast<int>(a.type)},
        {"registered_host",a.type==cudaMemoryTypeHost},{"unregistered_pageable",a.type==cudaMemoryTypeUnregistered}};
}
class Registration {
    void *base_;size_t bytes_;bool active_=false,closed_=false;json proof_;
public:
    Registration(Executor &e,const std::string &mode) {
        require(mode=="pageable"||mode=="registered","unknown requested host memory");
        base_=ggml_backend_buffer_get_base(e.dram_buffer);bytes_=ggml_backend_buffer_get_size(e.dram_buffer);
        require(base_&&bytes_>0,"owned arena unavailable");auto before=pointer_kind(base_);
        require(before.at("unregistered_pageable")==true,"initial arena is not observed pageable");
        if(mode=="registered"){
            int device=-1,supported=0;require(cudaGetDevice(&device)==cudaSuccess&&
                cudaDeviceGetAttribute(&supported,cudaDevAttrHostRegisterSupported,device)==cudaSuccess&&supported,
                "explicit host registration unsupported");
            require(cudaHostRegister(base_,bytes_,cudaHostRegisterDefault)==cudaSuccess,"owned registration failed; no fallback");active_=true;
        }
        auto during=pointer_kind(base_);require(during.at(mode=="registered"?"registered_host":"unregistered_pageable")==true,"actual memory differs");
        proof_=json{{"mode",mode},{"before",before},{"during",during},{"arena_bytes",bytes_},
            {"registration_timed",false},{"scope","owned complete DRAM arena"}};
    }
    ~Registration(){if(active_)cudaHostUnregister(base_);}
    json finish(){require(!closed_,"registration already finished");if(active_){require(cudaHostUnregister(base_)==cudaSuccess,"owned unregister failed");active_=false;}
        proof_["after"]=pointer_kind(base_);require(proof_.at("after").at("unregistered_pageable")==true,"arena category not restored");
        proof_["restored"]=true;closed_=true;return proof_;}
};
static json apply_affinity(const json &request){
    const auto wanted=static_cast<DWORD_PTR>(steady::positive(request.at("expected_process_mask"),"process mask",INT64_MAX));
    require(static_cast<uint64_t>(wanted)==4294967295ULL,"explicit all-32 process mask required");
    DWORD_PTR initial=0,system=0,actual=0;
    require(GetProcessAffinityMask(GetCurrentProcess(),&initial,&system)!=0&&(wanted&system)==wanted,"process mask unavailable");
    require(SetProcessAffinityMask(GetCurrentProcess(),wanted)!=0&&GetProcessAffinityMask(GetCurrentProcess(),&actual,&system)!=0&&actual==wanted,"owned affinity readback differs");
    return json{{"requested_mask",static_cast<uint64_t>(wanted)},{"initial_mask",static_cast<uint64_t>(initial)},
        {"observed_mask",static_cast<uint64_t>(actual)},{"applied_before_backend_initialization",true}};
}
static json finish_affinity(json proof){DWORD_PTR actual=0,system=0;require(GetProcessAffinityMask(GetCurrentProcess(),&actual,&system)!=0&&
        static_cast<uint64_t>(actual)==proof.at("requested_mask").get<uint64_t>(),"owned process affinity changed");proof["final_mask"]=static_cast<uint64_t>(actual);return proof;}
static void validate_orders(const json &idle,const json &orders,int trials){
    require(idle.is_array()&&idle.size()==4&&orders.is_array()&&orders.size()==static_cast<size_t>(trials),"four conditions and complete explicit orders required");
    std::set<int> values;
    for(const auto &value:idle){int v=static_cast<int>(state_response::integer(value,"idle milliseconds"));require(v<=1000&&values.insert(v).second,"invalid/duplicate idle condition");}
    require(values.count(0)==1,"zero-idle control required");
    for(const auto &order:orders){require(order.is_array()&&order.size()==4,"condition order must be a four-element permutation");std::set<int> seen;
        for(const auto &value:order){int i=static_cast<int>(state_response::integer(value,"condition index"));require(i<4&&seen.insert(i).second,"condition order is not a permutation");}}
}
struct Copy {int64_t begin=0,end=0;};
struct Sequence {int64_t idle_begin=0,idle_end=0,burst_begin=0,burst_end=0;int trial=0,slot=0,condition=0;};
static json payload(ggml_tensor *source,ggml_tensor *target){
    const size_t bytes=ggml_nbytes(source);std::vector<uint8_t> a(bytes),b(bytes);
    ggml_backend_tensor_get(source,a.data(),0,bytes);ggml_backend_tensor_get(target,b.data(),0,bytes);
    require(a==b,"target H2D payload differs");
    return json{{"bytes",bytes},{"full_byte_equality",true},{"source_sha256",steady::sha256_bytes(a.data(),a.size())},{"target_sha256",steady::sha256_bytes(b.data(),b.size())}};
}

}
