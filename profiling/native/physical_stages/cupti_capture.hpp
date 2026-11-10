// Direct CUPTI activity capture. No utilization or available-service inference.
#pragma once
#include <cupti.h>
#include <cuda_runtime_api.h>
#include "../portable/platform.hpp"
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace physical_capture {
using json = nlohmann::json;

inline uint64_t qpc_frequency() {
    LARGE_INTEGER value;
    if (!QueryPerformanceFrequency(&value) || value.QuadPart <= 0)
        throw std::runtime_error("QueryPerformanceFrequency failed");
    return static_cast<uint64_t>(value.QuadPart);
}

inline uint64_t ticks_to_ns(uint64_t value, uint64_t hz) {
    // Avoid overflowing value*1e9; the timestamp is floor-converted QPC ns.
    return (value / hz) * 1000000000ULL + (value % hz) * 1000000000ULL / hz;
}

inline uint64_t qpc_now_ns() {
    LARGE_INTEGER value;
    if (!QueryPerformanceCounter(&value) || value.QuadPart < 0)
        throw std::runtime_error("QueryPerformanceCounter failed");
    return ticks_to_ns(static_cast<uint64_t>(value.QuadPart), qpc_frequency());
}

inline void checked(CUptiResult status, const char *operation) {
    if (status == CUPTI_SUCCESS) return;
    const char *message = nullptr;
    cuptiGetResultString(status, &message);
    throw std::runtime_error(std::string(operation) + ": " + (message ? message : "unknown CUPTI failure"));
}

class Capture {
    struct Record {
        int kind = 0;
        uint64_t begin = 0, end = 0, bytes = 0, external_id = 0, graph_node_id = 0;
        uint32_t correlation = 0, device = 0, context = 0, stream = 0;
        uint32_t process = 0, thread = 0, cbid = 0, status = 0, graph = 0;
        uint32_t copy_kind = 0, src_kind = 0, dst_kind = 0, external_kind = 0;
        std::string name;
    };
    static inline Capture *instance_ = nullptr;
    static inline uint64_t frequency_ = 0;
    static inline std::atomic<bool> timestamp_failed_{false};
    static constexpr size_t max_records_ = 2000000;
    std::mutex mutex_;
    std::vector<Record> records_;
    std::vector<std::string> errors_;
    uint64_t dropped_ = 0, local_overflow_ = 0;
    bool enabled_ = false;
    uint32_t version_ = 0;
    uint64_t begin_ns_ = 0, end_ns_ = 0;

    static uint64_t CUPTIAPI timestamp() {
        LARGE_INTEGER value;
        if (!QueryPerformanceCounter(&value) || value.QuadPart < 0 || frequency_ == 0) {
            timestamp_failed_.store(true);
            return 0;
        }
        return ticks_to_ns(static_cast<uint64_t>(value.QuadPart), frequency_);
    }
    static void CUPTIAPI requested(uint8_t **buffer, size_t *size, size_t *max_records) {
        *size = 8 * 1024 * 1024;
        *max_records = 0;
        *buffer = static_cast<uint8_t *>(std::malloc(*size));
        if (!*buffer) {
            *size = 0;
            if (instance_) instance_->error("CUPTI activity buffer allocation failed");
        }
    }
    void error(const std::string &message) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (errors_.size() < 100) errors_.push_back(message);
    }
    void add(CUpti_Activity *raw) {
        Record r;
        r.kind = static_cast<int>(raw->kind);
        if (raw->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || raw->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
            const auto &v = *reinterpret_cast<CUpti_ActivityKernel9 *>(raw);
            r.begin=v.start; r.end=v.end; r.correlation=v.correlationId;
            r.device=v.deviceId; r.context=v.contextId; r.stream=v.streamId;
            r.graph=v.graphId; r.graph_node_id=v.graphNodeId;
            if (v.name) r.name=v.name;
        } else if (raw->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
            const auto &v = *reinterpret_cast<CUpti_ActivityMemcpy5 *>(raw);
            r.begin=v.start; r.end=v.end; r.bytes=v.bytes; r.correlation=v.correlationId;
            r.device=v.deviceId; r.context=v.contextId; r.stream=v.streamId;
            r.copy_kind=v.copyKind; r.src_kind=v.srcKind; r.dst_kind=v.dstKind;
            r.graph=v.graphId; r.graph_node_id=v.graphNodeId;
        } else if (raw->kind == CUPTI_ACTIVITY_KIND_RUNTIME || raw->kind == CUPTI_ACTIVITY_KIND_DRIVER) {
            const auto &v = *reinterpret_cast<CUpti_ActivityAPI *>(raw);
            r.begin=v.start; r.end=v.end; r.correlation=v.correlationId;
            r.process=v.processId; r.thread=v.threadId; r.cbid=v.cbid; r.status=v.returnValue;
        } else if (raw->kind == CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) {
            const auto &v = *reinterpret_cast<CUpti_ActivityExternalCorrelation *>(raw);
            r.external_id=v.externalId; r.external_kind=static_cast<uint32_t>(v.externalKind);
            r.correlation=v.correlationId;
        } else {
            error("unexpected CUPTI activity kind " + std::to_string(r.kind));
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (records_.size() < max_records_) records_.push_back(std::move(r));
        else ++local_overflow_;
    }
    static void CUPTIAPI completed(CUcontext context, uint32_t stream, uint8_t *buffer,
                                   size_t /*size*/, size_t valid) {
        Capture *owner=instance_;
        if (!owner) { std::free(buffer); return; }
        try {
            CUpti_Activity *record = nullptr;
            for (;;) {
                const auto status=cuptiActivityGetNextRecord(buffer,valid,&record);
                if (status==CUPTI_ERROR_MAX_LIMIT_REACHED) break;
                if (status!=CUPTI_SUCCESS) {
                    owner->error("cuptiActivityGetNextRecord status " + std::to_string(status));
                    break;
                }
                owner->add(record);
            }
            size_t dropped=0;
            const auto status=cuptiActivityGetNumDroppedRecords(context,stream,&dropped);
            if (status!=CUPTI_SUCCESS) owner->error("cuptiActivityGetNumDroppedRecords status " + std::to_string(status));
            else { std::lock_guard<std::mutex> lock(owner->mutex_); owner->dropped_+=dropped; }
        } catch (const std::exception &failure) {
            owner->error(std::string("activity callback: ")+failure.what());
        } catch (...) {
            owner->error("unknown activity callback failure");
        }
        std::free(buffer);
    }
    static std::vector<CUpti_ActivityKind> kinds() {
        return {CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION, CUPTI_ACTIVITY_KIND_RUNTIME, CUPTI_ACTIVITY_KIND_DRIVER,
                CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, CUPTI_ACTIVITY_KIND_MEMCPY};
    }
public:
    Capture() {
        if (instance_) throw std::runtime_error("one CUPTI Capture per process required");
        frequency_=qpc_frequency(); timestamp_failed_.store(false);
        records_.reserve(65536);
        checked(cuptiGetVersion(&version_),"cuptiGetVersion");
        instance_=this;
        try {
            checked(cuptiActivityRegisterTimestampCallback(timestamp),"cuptiActivityRegisterTimestampCallback");
            checked(cuptiActivityRegisterCallbacks(requested,completed),"cuptiActivityRegisterCallbacks");
            for (auto kind:kinds()) checked(cuptiActivityEnable(kind),"cuptiActivityEnable");
            enabled_=true; begin_ns_=qpc_now_ns();
        } catch (...) { instance_=nullptr; throw; }
    }
    ~Capture() {
        if (enabled_) {
            for (auto kind:kinds()) cuptiActivityDisable(kind);
            cuptiActivityFlushAll(0);
        }
        if (instance_==this) instance_=nullptr;
    }
    void push(uint64_t id) {
        checked(cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0,id),
                "cuptiActivityPushExternalCorrelationId");
    }
    void pop(uint64_t expected) {
        uint64_t observed=0;
        checked(cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0,&observed),
                "cuptiActivityPopExternalCorrelationId");
        if (observed!=expected) throw std::runtime_error("external correlation stack mismatch");
    }
    json finish() {
        if (!enabled_) throw std::runtime_error("CUPTI Capture already finished");
        end_ns_=qpc_now_ns();
        for (auto kind:kinds()) checked(cuptiActivityDisable(kind),"cuptiActivityDisable");
        checked(cuptiActivityFlushAll(0),"cuptiActivityFlushAll");
        enabled_=false;
        json rows=json::array();
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto &r:records_) {
            if (r.kind==CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) {
                rows.push_back(json{{"kind","external_correlation"},{"correlation_id",r.correlation},
                    {"external_id",r.external_id},{"external_kind",r.external_kind}});
                continue;
            }
            const char *kind=r.kind==CUPTI_ACTIVITY_KIND_RUNTIME?"runtime_api":
                r.kind==CUPTI_ACTIVITY_KIND_DRIVER?"driver_api":
                r.kind==CUPTI_ACTIVITY_KIND_MEMCPY?"memcpy":"kernel";
            json row={{"kind",kind},{"begin_ns",r.begin},{"end_ns",r.end},
                {"timestamp_valid",r.begin>0&&r.end>r.begin},{"correlation_id",r.correlation}};
            if (r.kind==CUPTI_ACTIVITY_KIND_RUNTIME || r.kind==CUPTI_ACTIVITY_KIND_DRIVER) {
                const char *name=nullptr;
                const auto domain=r.kind==CUPTI_ACTIVITY_KIND_RUNTIME?CUPTI_CB_DOMAIN_RUNTIME_API:CUPTI_CB_DOMAIN_DRIVER_API;
                const auto named=cuptiGetCallbackName(domain,r.cbid,&name);
                row["name"]=named==CUPTI_SUCCESS&&name?json(name):json(nullptr);
                row["callback_id"]=r.cbid; row["return_value"]=r.status;
                row["process_id"]=r.process; row["thread_id"]=r.thread;
            } else {
                row["device_id"]=r.device; row["context_id"]=r.context;
                row["stream_id"]=r.stream; row["graph_id"]=r.graph;
                row["graph_node_id"]=r.graph_node_id;
                if (r.kind==CUPTI_ACTIVITY_KIND_MEMCPY) {
                    row["bytes"]=r.bytes; row["copy_kind"]=r.copy_kind;
                    row["source_memory_kind"]=r.src_kind; row["destination_memory_kind"]=r.dst_kind;
                } else row["name"]=r.name;
            }
            rows.push_back(std::move(row));
        }
        return json{{"schema","physical-stage-cupti-1"},{"cupti_version",version_},
            // The clock this capture stamps against is the one `platform.hpp`
            // supplies, so the name comes from there rather than from a literal:
            // on Windows the macros resolve to `windows_qpc_floor_ns` and
            // `absolute_qpc_zero` (the published receipts are unchanged byte for
            // byte), and on Linux the same capture stamps against
            // CLOCK_MONOTONIC and says so. This header is shared -- the dense
            // column compiles it too -- so a literal here would have the dense
            // receipt claim a Windows clock on a Linux host.
            {"clock_id",PORTABLE_PHYSICAL_CLOCK_ID},{"clock_origin",CLOCK_ORIGIN},
            {"qpc_frequency_hz",frequency_},{"capture_begin_ns",begin_ns_},{"capture_end_ns",end_ns_},
            {"timestamp_callback_registered_before_activity_enable",true},
            {"external_correlation_kind",static_cast<uint32_t>(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0)},
            {"timestamp_callback_failed",timestamp_failed_.load()},
            {"records",rows},{"cupti_dropped_records",dropped_},{"local_record_overflow",local_overflow_},
            {"callback_errors",errors_},{"all_requested_records_retained",dropped_==0&&local_overflow_==0&&errors_.empty()&&!timestamp_failed_.load()},
            {"available_resource_rate_inferred",false},{"host_and_device_times_are_additive",false},
            {"activity_capture_changes_execution",true},{"hardware_accuracy_qualified",false}};
    }
};
} // namespace physical_capture
