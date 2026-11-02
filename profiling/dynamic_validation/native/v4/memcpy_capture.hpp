// Scoped activity-only CUPTI observer. No runtime/driver/kernel trace.
#pragma once
#include <cupti.h>
#include <windows.h>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace memcpy_capture_v4 {
using json = nlohmann::json;
inline void checked(CUptiResult status,const char *what){
    if(status==CUPTI_SUCCESS)return;const char *message=nullptr;cuptiGetResultString(status,&message);
    throw std::runtime_error(std::string(what)+": "+(message?message:"unknown CUPTI status"));
}
class Capture {
    struct Record {bool external=false;uint64_t begin=0,end=0,bytes=0,id=0;uint32_t correlation=0,device=0,context=0,stream=0,copy=0,source=0,destination=0,external_kind=0;};
    static inline Capture *instance_=nullptr;
    static inline uint64_t hz_=0;
    static inline std::atomic<bool> clock_failed_{false};
    std::mutex mutex_;std::vector<Record> records_;std::vector<std::string> errors_;
    uint64_t dropped_=0,overflow_=0,begin_=0;bool active_=false;uint32_t version_=0;
    static uint64_t now(){LARGE_INTEGER q{};if(!QueryPerformanceCounter(&q)||q.QuadPart<0||!hz_){clock_failed_.store(true);return 0;}
        auto x=static_cast<uint64_t>(q.QuadPart);return(x/hz_)*1000000000ULL+(x%hz_)*1000000000ULL/hz_;}
    static uint64_t CUPTIAPI timestamp(){return now();}
    static void CUPTIAPI requested(uint8_t **buffer,size_t *size,size_t *maximum){
        *size=8*1024*1024;*maximum=0;*buffer=static_cast<uint8_t*>(std::malloc(*size));
        if(!*buffer){*size=0;if(instance_)instance_->error("activity buffer allocation failed");}}
    void error(const std::string &s){std::lock_guard<std::mutex> lock(mutex_);if(errors_.size()<100)errors_.push_back(s);}
    void add(CUpti_Activity *raw){Record r;
        if(raw->kind==CUPTI_ACTIVITY_KIND_MEMCPY){const auto &x=*reinterpret_cast<CUpti_ActivityMemcpy5*>(raw);
            r.begin=x.start;r.end=x.end;r.bytes=x.bytes;r.correlation=x.correlationId;r.device=x.deviceId;r.context=x.contextId;r.stream=x.streamId;
            r.copy=x.copyKind;r.source=x.srcKind;r.destination=x.dstKind;
        }else if(raw->kind==CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION){const auto &x=*reinterpret_cast<CUpti_ActivityExternalCorrelation*>(raw);
            r.external=true;r.correlation=x.correlationId;r.id=x.externalId;r.external_kind=static_cast<uint32_t>(x.externalKind);
        }else{error("unexpected activity kind "+std::to_string(raw->kind));return;}
        std::lock_guard<std::mutex> lock(mutex_);if(records_.size()<2000000)records_.push_back(r);else ++overflow_;
    }
    static void CUPTIAPI completed(CUcontext context,uint32_t stream,uint8_t *buffer,size_t,size_t valid){
        Capture *owner=instance_;if(!owner){std::free(buffer);return;}
        try{CUpti_Activity *record=nullptr;for(;;){auto s=cuptiActivityGetNextRecord(buffer,valid,&record);if(s==CUPTI_ERROR_MAX_LIMIT_REACHED)break;
            if(s!=CUPTI_SUCCESS){owner->error("get-next-record status "+std::to_string(s));break;}owner->add(record);}
            size_t dropped=0;auto s=cuptiActivityGetNumDroppedRecords(context,stream,&dropped);if(s!=CUPTI_SUCCESS)owner->error("dropped-record-query status "+std::to_string(s));
            else{std::lock_guard<std::mutex> lock(owner->mutex_);owner->dropped_+=dropped;}
        }catch(const std::exception &e){owner->error(e.what());}catch(...){owner->error("unknown callback error");}std::free(buffer);
    }
    static std::vector<CUpti_ActivityKind> kinds(){return{CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION,CUPTI_ACTIVITY_KIND_MEMCPY};}
public:
    Capture(){if(instance_)throw std::runtime_error("one scoped capture at a time required");LARGE_INTEGER hz{};
        if(!QueryPerformanceFrequency(&hz)||hz.QuadPart<=0)throw std::runtime_error("QPC frequency unavailable");hz_=static_cast<uint64_t>(hz.QuadPart);clock_failed_.store(false);
        records_.reserve(65536);checked(cuptiGetVersion(&version_),"cuptiGetVersion");instance_=this;
        try{checked(cuptiActivityRegisterTimestampCallback(timestamp),"register QPC timestamp");checked(cuptiActivityRegisterCallbacks(requested,completed),"register activity callbacks");
            for(auto kind:kinds())checked(cuptiActivityEnable(kind),"enable selected activity");active_=true;begin_=now();
        }catch(...){for(auto kind:kinds())cuptiActivityDisable(kind);instance_=nullptr;throw;}}
    ~Capture(){if(active_){for(auto kind:kinds())cuptiActivityDisable(kind);cuptiActivityFlushAll(0);}if(instance_==this)instance_=nullptr;}
    void push(uint64_t id){checked(cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0,id),"push external ID");}
    void pop(uint64_t expected){uint64_t actual=0;checked(cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0,&actual),"pop external ID");
        if(actual!=expected)throw std::runtime_error("external ID stack differs");}
    json finish(){if(!active_)throw std::runtime_error("capture already finished");const auto end=now();
        for(auto kind:kinds())checked(cuptiActivityDisable(kind),"disable selected activity");checked(cuptiActivityFlushAll(0),"flush activity");active_=false;
        json rows=json::array();std::lock_guard<std::mutex> lock(mutex_);
        for(const auto &r:records_){if(r.external)rows.push_back(json{{"kind","external_correlation"},{"correlation_id",r.correlation},{"external_id",r.id},{"external_kind",r.external_kind}});
            else rows.push_back(json{{"kind","memcpy"},{"begin_ns",r.begin},{"end_ns",r.end},{"timestamp_valid",r.begin>0&&r.end>r.begin},
                {"bytes",r.bytes},{"correlation_id",r.correlation},{"device_id",r.device},{"context_id",r.context},{"stream_id",r.stream},
                {"copy_kind",r.copy},{"source_memory_kind",r.source},{"destination_memory_kind",r.destination}});}
        return json{{"schema","memcpy-only-cupti-4"},{"enabled",true},{"cupti_version",version_},{"clock_id","windows_qpc_floor_ns"},
            {"qpc_frequency_hz",hz_},{"capture_begin_ns",begin_},{"capture_end_ns",end},
            {"requested_activities",json::array({"external_correlation","memcpy"})},{"runtime_driver_kernel_activity_enabled",false},
            {"timestamp_callback_failed",clock_failed_.load()},{"records",rows},{"cupti_dropped_records",dropped_},{"local_record_overflow",overflow_},
            {"callback_errors",errors_},{"all_requested_records_retained",dropped_==0&&overflow_==0&&errors_.empty()&&!clock_failed_.load()},
            {"activity_capture_changes_execution",true},{"hardware_accuracy_qualified",false}};
    }
};
}
