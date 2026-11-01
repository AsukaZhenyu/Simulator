// Target-thread runtime callbacks only. No activity collection or GPU timing.
#pragma once
#include <cupti.h>
#include <cupti_callbacks.h>
#include <cupti_runtime_cbid.h>
// cupti.h already includes generated_cuda_runtime_api_meta.h (no include guard).
#include <windows.h>
#include <map>
#include <vector>

namespace runtime_callbacks_v6 {
using json=nlohmann::json;
struct Record {
    int64_t enter=0,exit=0;uint64_t phase=0,stream=0,bytes=0;uint32_t cbid=0,correlation=0;
    uint32_t grid[3]={0,0,0},block[3]={0,0,0};uint64_t shared=0;int kind=-1,status=-1;
    bool parameters=false,returned=false,symbol_available=false,symbol_truncated=false;
    char symbol[192]={0};
};
class Observer {
    DWORD thread_;std::vector<Record> records_;size_t used_=0;uint64_t phase_=0;
    CUpti_SubscriberHandle subscriber_=nullptr;bool enabled_=false;uint64_t overflow_=0,pair_errors_=0,clock_errors_=0;
    static int64_t tick()noexcept{LARGE_INTEGER t{};return QueryPerformanceCounter(&t)?t.QuadPart:0;}
    static void checked(CUptiResult result,const char *name){if(result==CUPTI_SUCCESS)return;const char *message=nullptr;
        cuptiGetResultString(result,&message);throw std::runtime_error(std::string(name)+": "+(message?message:"unknown CUPTI failure"));}
    static std::vector<CUpti_CallbackId> ids(){return{CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000,
        CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020,CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020};}
    static void CUPTIAPI callback(void *userdata,CUpti_CallbackDomain domain,CUpti_CallbackId cbid,const void *raw)noexcept{
        auto *self=static_cast<Observer*>(userdata);
        if(domain!=CUPTI_CB_DOMAIN_RUNTIME_API||GetCurrentThreadId()!=self->thread_)return;
        const auto *data=static_cast<const CUpti_CallbackData*>(raw);
        if(!data||!data->correlationData){++self->pair_errors_;return;}
        if(data->callbackSite==CUPTI_API_ENTER){
            *data->correlationData=0;if(!self->phase_)return;
            const auto now=tick();if(!now)++self->clock_errors_;
            if(self->used_>=self->records_.size()){++self->overflow_;return;}
            const auto index=self->used_++;auto &r=self->records_[index];r.enter=now;r.phase=self->phase_;r.cbid=cbid;r.correlation=data->correlationId;
            *data->correlationData=index+1;
            if(data->functionParams){r.parameters=true;
                if(cbid==CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000){const auto &p=*static_cast<const cudaLaunchKernel_v7000_params*>(data->functionParams);
                    r.grid[0]=p.gridDim.x;r.grid[1]=p.gridDim.y;r.grid[2]=p.gridDim.z;r.block[0]=p.blockDim.x;r.block[1]=p.blockDim.y;r.block[2]=p.blockDim.z;
                    r.shared=p.sharedMem;r.stream=reinterpret_cast<uintptr_t>(p.stream);
                    if(data->symbolName){r.symbol_available=true;size_t i=0;for(;i+1<sizeof(r.symbol)&&data->symbolName[i];++i)r.symbol[i]=data->symbolName[i];
                        r.symbol[i]=0;r.symbol_truncated=data->symbolName[i]!=0;}
                }else if(cbid==CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020){const auto &p=*static_cast<const cudaMemcpyAsync_v3020_params*>(data->functionParams);
                    r.bytes=p.count;r.kind=static_cast<int>(p.kind);r.stream=reinterpret_cast<uintptr_t>(p.stream);
                }else if(cbid==CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020){const auto &p=*static_cast<const cudaStreamSynchronize_v3020_params*>(data->functionParams);r.stream=reinterpret_cast<uintptr_t>(p.stream);}
            }
        }else if(data->callbackSite==CUPTI_API_EXIT){
            const auto saved=*data->correlationData;if(!saved)return;const auto now=tick();if(!now)++self->clock_errors_;
            if(saved>self->used_){++self->pair_errors_;return;}auto &r=self->records_[static_cast<size_t>(saved-1)];
            if(r.cbid!=cbid||r.correlation!=data->correlationId||r.returned){++self->pair_errors_;return;}
            r.exit=now;r.returned=true;if(data->functionReturnValue)r.status=static_cast<int>(*static_cast<const cudaError_t*>(data->functionReturnValue));
        }
    }
public:
    Observer(bool enabled,DWORD target,size_t capacity):thread_(target),records_(capacity<=1000000?capacity:0){
        if(capacity<1||capacity>1000000)throw std::runtime_error("bounded callback capacity required");
        if(enabled){checked(cuptiSubscribe(&subscriber_,callback,this),"cuptiSubscribe target callbacks");
            try{for(auto id:ids())checked(cuptiEnableCallback(1,subscriber_,CUPTI_CB_DOMAIN_RUNTIME_API,id),"enable selected runtime callback");enabled_=true;}
            catch(...){cuptiUnsubscribe(subscriber_);subscriber_=nullptr;throw;}}
    }
    ~Observer(){if(subscriber_)cuptiUnsubscribe(subscriber_);}
    void arm(uint64_t phase){if(GetCurrentThreadId()!=thread_||!phase||phase_)throw std::runtime_error("callback target phase arm differs");phase_=phase;}
    void disarm(){if(GetCurrentThreadId()!=thread_)throw std::runtime_error("callback disarm on foreign thread");phase_=0;}
    static json self_test(){
        Observer owner(false,GetCurrentThreadId(),4);uint64_t context=0;cudaError_t status=cudaSuccess;
        cudaLaunchKernel_v7000_params parameters{};parameters.gridDim=dim3(2,3,1);parameters.blockDim=dim3(32,1,1);
        CUpti_CallbackData data{};data.correlationData=&context;data.correlationId=91;data.functionParams=&parameters;data.functionReturnValue=&status;
        data.callbackSite=CUPTI_API_ENTER;callback(&owner,CUPTI_CB_DOMAIN_RUNTIME_API,CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000,&data);
        if(context!=0)throw std::runtime_error("unarmed callback was recorded");
        owner.arm(7);data.callbackSite=CUPTI_API_ENTER;callback(&owner,CUPTI_CB_DOMAIN_RUNTIME_API,CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000,&data);
        data.callbackSite=CUPTI_API_EXIT;callback(&owner,CUPTI_CB_DOMAIN_RUNTIME_API,CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000,&data);owner.disarm();
        auto result=owner.finish();if(result.at("record_count")!=1||!result.at("all_selected_calls_retained").get<bool>()||result.at("callback_records").at(0).at("phase_id")!=7||result.at("metadata").at(0).at("grid")!=json::array({2,3,1}))
            throw std::runtime_error("software callback pairing/metadata test failed");
        return json{{"passed",true},{"CUPTI_subscriber_created",false},{"backend_created",false},{"unarmed_callback_ignored",true},{"enter_exit_POD_pairing_passed",true},{"metadata_roundtrip_passed",true}};
    }
    json finish(){disarm();if(subscriber_){checked(cuptiUnsubscribe(subscriber_),"unsubscribe target callbacks");subscriber_=nullptr;}
        json metadata=json::array(),rows=json::array();std::map<std::string,size_t> keys;uint64_t unfinished=0,bad_time=0,errors=0;
        for(size_t index=0;index<used_;++index){const auto &r=records_[index];
            const char *name=r.cbid==CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000?"cudaLaunchKernel_v7000":
                r.cbid==CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020?"cudaMemcpyAsync_v3020":"cudaStreamSynchronize_v3020";
            json m={{"name",name},{"callback_id",r.cbid},{"parameters_available",r.parameters},
                {"symbol_name",r.symbol_available?json(r.symbol):json(nullptr)},{"symbol_truncated",r.symbol_truncated},
                {"grid",r.cbid==CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000&&r.parameters?json::array({r.grid[0],r.grid[1],r.grid[2]}):json(nullptr)},
                {"block",r.cbid==CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000&&r.parameters?json::array({r.block[0],r.block[1],r.block[2]}):json(nullptr)},
                {"shared_memory_bytes",r.cbid==CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000&&r.parameters?json(r.shared):json(nullptr)},
                {"bytes",r.cbid==CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020&&r.parameters?json(r.bytes):json(nullptr)},
                {"copy_kind_cudaMemcpyKind",r.cbid==CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020&&r.parameters?json(r.kind):json(nullptr)},
                {"stream_handle_opaque_process_local",r.parameters?json(r.stream):json(nullptr)}};
            const auto key=m.dump();auto found=keys.find(key);size_t coordinate;
            if(found==keys.end()){coordinate=metadata.size();keys[key]=coordinate;metadata.push_back(m);}else coordinate=found->second;
            if(!r.returned)++unfinished;if(!r.enter||r.exit<r.enter)++bad_time;if(r.status!=0)++errors;
            rows.push_back(json{{"ordinal",index},{"phase_id",r.phase},{"metadata_index",coordinate},{"correlation_id",r.correlation},
                {"enter_tick",r.enter},{"exit_tick",r.exit},{"exit_received",r.returned},{"return_status",r.returned?json(r.status):json(nullptr)}});
        }
        return json{{"schema","target-runtime-callbacks-6"},{"enabled",enabled_},{"target_thread_id",thread_},{"capacity",records_.size()},
            {"allocated_POD_bytes",records_.size()*sizeof(Record)},{"callback_records",rows},{"metadata",metadata},
            {"record_count",used_},{"overflow_records",overflow_},{"pair_errors",pair_errors_},{"clock_errors",clock_errors_},
            {"unfinished_calls",unfinished},{"invalid_time_calls",bad_time},{"nonzero_or_unknown_return_status_calls",errors},
            {"all_selected_calls_retained",overflow_==0&&pair_errors_==0&&clock_errors_==0&&unfinished==0&&bad_time==0&&errors==0},
            {"callback_target_only_POD_no_allocation_or_mutex",true},{"foreign_actor_threads_filtered_before_recording",true},
            {"activity_driver_kernel_collection_enabled",false},{"GPU_service_or_enqueue_acceptance_identified",false},
            {"timing_semantic","runtime callback enter-to-exit inclusive host response; callback overhead remains, not pure CPU work or GPU enqueue acceptance"},
            {"record_serialization_after_target_and_actor_join",true},{"performance_parameter_qualified",false}};
    }
};
}
