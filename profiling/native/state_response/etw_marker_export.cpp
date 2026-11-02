// Offline, all-provider raw marker-pattern scan. Never controls recorders.
#define NOMINMAX
#include <windows.h>
#include <evntrace.h>
#include <evntcons.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
using json=nlohmann::json;
static json matches=json::array();static uint64_t count=0;static std::vector<std::string> patterns;static std::string error;
static std::string gid(const GUID &g){char s[40];snprintf(s,sizeof(s),"%08lx-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x",g.Data1,g.Data2,g.Data3,g.Data4[0],g.Data4[1],g.Data4[2],g.Data4[3],g.Data4[4],g.Data4[5],g.Data4[6],g.Data4[7]);return s;}
static std::string hex(const unsigned char*p,size_t n){const char*a="0123456789abcdef";std::string s;for(size_t i=0;i<n;i++){s.push_back(a[p[i]>>4]);s.push_back(a[p[i]&15]);}return s;}
static void WINAPI consume(EVENT_RECORD*e){try{++count;std::string raw(static_cast<char*>(e->UserData),e->UserDataLength);for(const auto&needle:patterns){std::string wide;for(char c:needle){wide.push_back(c);wide.push_back(0);}if(raw.find(needle)==std::string::npos&&raw.find(wide)==std::string::npos)continue;const auto &h=e->EventHeader;matches.push_back({{"needle",needle},{"qpc_raw",h.TimeStamp.QuadPart},{"provider",gid(h.ProviderId)},{"event_id",h.EventDescriptor.Id},{"opcode",h.EventDescriptor.Opcode},{"version",h.EventDescriptor.Version},{"flags",h.Flags},{"pid",h.ProcessId},{"tid",h.ThreadId},{"cpu",e->BufferContext.ProcessorIndex},{"payload_length",e->UserDataLength},{"raw_hex",hex(static_cast<unsigned char*>(e->UserData),e->UserDataLength)}});}}catch(const std::exception&e){error=e.what();}}
int wmain(int argc,wchar_t**argv){if(argc!=4)return 2;std::filesystem::path input(argv[1]),config(argv[2]),out(argv[3]);if(std::filesystem::exists(out))return 3;json cfg;std::ifstream(config)>>cfg;auto instance=cfg.at("instance_name").get<std::string>();patterns={"LLMSimBegin_"+instance,"LLMSimEnd_"+instance};EVENT_TRACE_LOGFILEW log{};auto path=input.wstring();log.LogFileName=path.data();log.ProcessTraceMode=PROCESS_TRACE_MODE_RAW_TIMESTAMP|PROCESS_TRACE_MODE_EVENT_RECORD;log.EventRecordCallback=consume;auto h=OpenTraceW(&log);if(h==INVALID_PROCESSTRACE_HANDLE)return 4;TRACEHANDLE handles[]={h};auto status=ProcessTrace(handles,1,nullptr,nullptr);const auto &header=log.LogfileHeader;json report={{"schema","all-provider-raw-marker-pattern-scan-1"},{"read_only_offline",true},{"event_count",count},{"process_trace_status",status},{"header_clock_type",header.ReservedFlags},{"qpc_frequency",header.PerfFreq.QuadPart},{"events_lost",header.EventsLost},{"buffers_lost",header.BuffersLost},{"logfile_events_lost",log.EventsLost},{"matches",matches},{"error",error},{"qualification","Pattern matches include process command lines. Only a dedicated marker event whose provider/schema is verified can qualify an emitted marker."}};auto close=CloseTrace(h);report["close_trace_status"]=close;std::ofstream file(out,std::ios::binary);file<<report.dump(2)<<'\n';std::cout<<json{{"event_count",count},{"matches",matches.size()},{"process_status",status}}.dump()<<'\n';return status==0&&close==0&&error.empty()?0:5;}
