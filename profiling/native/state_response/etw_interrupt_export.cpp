// Read-only offline ETL raw interrupt intervals, no logger-control API.
// Layout reference: Microsoft PerfView KernelTraceEventParser.cs, ISRTraceData,
// DPCTraceData: InitialTimeQPC at offset 0, Routine at offset 8. Python must
// qualify actual provider/opcode/version/length against xperf decoded rows.
#define NOMINMAX
#include <windows.h>
#include <evntrace.h>
#include <evntcons.h>
#include <nlohmann/json.hpp>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
using json=nlohmann::json;
static const GUID PERF={0xce1dbfb4,0x137e,0x4da6,{0x87,0xb0,0x3f,0x59,0xaa,0x10,0x2c,0xbc}};
static uint64_t callbacks=0,selected=0;static int64_t begin_qpc=0,end_qpc=0;static std::map<std::string,uint64_t>counts;static std::ofstream*events;static std::string error;
static std::string hex(const unsigned char*p,size_t n){const char*a="0123456789abcdef";std::string s;for(size_t i=0;i<n;i++){s.push_back(a[p[i]>>4]);s.push_back(a[p[i]&15]);}return s;}
static VOID WINAPI consume(EVENT_RECORD*e){try{++callbacks;const auto&h=e->EventHeader;const auto op=h.EventDescriptor.Opcode;if(!InlineIsEqualGUID(h.ProviderId,PERF)||(op!=50&&op!=66&&op!=67&&op!=68&&op!=69))return;
    int64_t candidate=0;uint64_t routine=0;if(e->UserDataLength>=8)std::memcpy(&candidate,e->UserData,8);if(e->UserDataLength>=16)std::memcpy(&routine,static_cast<unsigned char*>(e->UserData)+8,8);
    const auto q=h.TimeStamp.QuadPart;if(q<begin_qpc||candidate>=end_qpc)return;
    std::string key=std::to_string(op)+":"+std::to_string(h.EventDescriptor.Version)+":"+std::to_string(e->UserDataLength)+":"+std::to_string(h.Flags);++counts[key];
    *events<<json{{"provider","ce1dbfb4-137e-4da6-87b0-3f59aa102cbc"},{"opcode",op},{"version",h.EventDescriptor.Version},{"header_flags",h.Flags},{"payload_length",e->UserDataLength},{"raw_payload_hex",hex(static_cast<unsigned char*>(e->UserData),e->UserDataLength)},{"end_qpc",q},{"initial_qpc_candidate",candidate},{"routine_candidate",routine},{"cpu",e->BufferContext.ProcessorIndex}}.dump()<<'\n';++selected;
}catch(const std::exception&e){error=e.what();}}
int wmain(int argc,wchar_t**argv){try{if(argc!=4)throw std::runtime_error("usage: etw-interrupt-export TRACE.etl IDENTITY.json NEW_DIR");std::filesystem::path trace(argv[1]),cfgpath(argv[2]),out(argv[3]);if(std::filesystem::exists(out)||!std::filesystem::create_directory(out))throw std::runtime_error("new exclusive output required");json cfg;std::ifstream(cfgpath)>>cfg;begin_qpc=cfg.at("window_begin_qpc").get<int64_t>();end_qpc=cfg.at("window_end_qpc").get<int64_t>();if(begin_qpc>=end_qpc)throw std::runtime_error("positive window required");std::ofstream file(out/"raw-interrupts.jsonl",std::ios::binary);events=&file;EVENT_TRACE_LOGFILEW log{};auto path=trace.wstring();log.LogFileName=path.data();log.ProcessTraceMode=PROCESS_TRACE_MODE_RAW_TIMESTAMP|PROCESS_TRACE_MODE_EVENT_RECORD;log.EventRecordCallback=consume;auto handle=OpenTraceW(&log);if(handle==INVALID_PROCESSTRACE_HANDLE)throw std::runtime_error("OpenTrace failed");TRACEHANDLE handles[]={handle};auto status=ProcessTrace(handles,1,nullptr,nullptr);const auto &h=log.LogfileHeader;json summary={{"schema","raw-interrupt-initial-qpc-candidates-1"},{"offline_read_only",true},{"trace",trace.u8string()},{"identity",cfgpath.u8string()},{"clock_type",h.ReservedFlags},{"qpc_frequency",h.PerfFreq.QuadPart},{"events_lost",h.EventsLost},{"buffers_lost",h.BuffersLost},{"logfile_events_lost",log.EventsLost},{"process_trace_status",status},{"callbacks",callbacks},{"selected_events",selected},{"descriptor_counts",counts},{"window_begin_qpc",begin_qpc},{"window_end_qpc",end_qpc},{"error",error},{"qualification","InitialTimeQPC is a candidate until actual descriptors, length, routine and elapsed-time values agree with independent xperf dump. ThreadedDPC opcode66 is separate and must not be silently counted as interrupt preemption."},{"layout_reference","https://github.com/microsoft/perfview/blob/main/src/TraceEvent/Parsers/KernelTraceEventParser.cs#L8683-L8835"}};summary["close_trace_status"]=CloseTrace(handle);file.close();std::ofstream report(out/"raw-interrupt-export.json",std::ios::binary);report<<summary.dump(2)<<'\n';std::cout<<json{{"selected_events",selected},{"descriptor_counts",counts},{"status",status}}.dump()<<'\n';return status==0&&error.empty()?0:5;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
