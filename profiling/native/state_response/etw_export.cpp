// Offline ETL reader. No logger is started, stopped, updated or attached live.
#define NOMINMAX
#include <windows.h>
#include <evntrace.h>
#include <evntcons.h>
#include <tdh.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using json=nlohmann::json;
static const GUID THREAD_GUID={0x3d6fa8d1,0xfe05,0x11d0,{0x9d,0xda,0x00,0xc0,0x4f,0xd7,0xba,0x7c}};
static const GUID PROCESS_GUID={0x3d6fa8d0,0xfe05,0x11d0,{0x9d,0xda,0x00,0xc0,0x4f,0xd7,0xba,0x7c}};
static const GUID PERF_GUID={0xce1dbfb4,0x137e,0x4da6,{0x87,0xb0,0x3f,0x59,0xaa,0x10,0x2c,0xbc}};
static const GUID IMAGE_GUID={0x2cb15d1d,0x5fc1,0x11d2,{0xab,0xe1,0x00,0xa0,0xc9,0x11,0xf5,0x18}};
static const GUID STACK_GUID={0xdef2fe46,0x7bd6,0x4b80,{0xbd,0x94,0xf5,0x7f,0xe2,0x0d,0x0c,0xe3}};
static const GUID MARK_GUID={0x36b6f488,0xaad7,0x48c2,{0xaf,0xe3,0xd4,0xec,0x2c,0x8b,0x46,0xfa}};
static void need(bool ok,const std::string &s){if(!ok)throw std::runtime_error(s);}
static std::string utf8(const wchar_t *p){if(!p)return {};const int n=WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,p,-1,nullptr,0,nullptr,nullptr);if(!n)return {};std::string s(n,'\0');WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,p,-1,s.data(),n,nullptr,nullptr);s.pop_back();return s;}
static std::string guid(const GUID &g){char b[40];std::snprintf(b,sizeof(b),"%08lx-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x",g.Data1,g.Data2,g.Data3,g.Data4[0],g.Data4[1],g.Data4[2],g.Data4[3],g.Data4[4],g.Data4[5],g.Data4[6],g.Data4[7]);return b;}
static std::string lower(std::string s){for(char &c:s)c=static_cast<char>(std::tolower(static_cast<unsigned char>(c)));return s;}
static std::string hex(const unsigned char *p,size_t n){static const char *a="0123456789abcdef";std::string r;r.reserve(n*2);for(size_t i=0;i<n;++i){r.push_back(a[p[i]>>4]);r.push_back(a[p[i]&15]);}return r;}
static json scalar(USHORT type,const std::vector<unsigned char> &bytes){
    const auto n=bytes.size();if(!n)return nullptr;
    if(type==TDH_INTYPE_UNICODESTRING&&n>=2&&n%2==0){std::vector<wchar_t>s(n/2+1,0);std::memcpy(s.data(),bytes.data(),n);return utf8(s.data());}
    if(type==TDH_INTYPE_ANSISTRING){size_t size=0;while(size<n&&bytes[size]){if(bytes[size]>=128)return nullptr;++size;}return std::string(reinterpret_cast<const char*>(bytes.data()),size);}
    const bool u=(type==TDH_INTYPE_UINT8||type==TDH_INTYPE_UINT16||type==TDH_INTYPE_UINT32||type==TDH_INTYPE_UINT64||type==TDH_INTYPE_HEXINT32||type==TDH_INTYPE_HEXINT64||type==TDH_INTYPE_POINTER||type==TDH_INTYPE_SIZET);
    const bool s=(type==TDH_INTYPE_INT8||type==TDH_INTYPE_INT16||type==TDH_INTYPE_INT32||type==TDH_INTYPE_INT64);
    if((u||s)&&(n==1||n==2||n==4||n==8)){uint64_t v=0;std::memcpy(&v,bytes.data(),n);if(u)return v;if(n<8&&(v&(uint64_t(1)<<(n*8-1))))v|=(~uint64_t(0))<<(n*8);int64_t result;std::memcpy(&result,&v,8);return result;}
    return nullptr;
}
struct Decoded {json row;std::map<std::string,json> values;};
struct Life {uint32_t pid,tid;int64_t begin;int64_t end=std::numeric_limits<int64_t>::max();};
struct Context {
    json config,header,pass_receipts=json::array(),process_starts=json::array(),thread_lifetimes=json::array(),decode_errors=json::array();
    std::set<uint32_t> pids;std::vector<Life> lives;std::string instance,error;int pass=0;
    uint64_t callbacks=0,selected=0;int64_t first=std::numeric_limits<int64_t>::max(),last=0;
    std::ofstream *events=nullptr;std::map<std::string,std::vector<unsigned char>> schemas;
    bool owned(uint32_t tid,int64_t at) const {for(const auto &x:lives)if(x.tid==tid&&x.begin<=at&&at<=x.end)return true;return false;}
};
static uint64_t number(const Decoded &d,std::initializer_list<const char*>names){for(auto name:names){auto i=d.values.find(lower(name));if(i!=d.values.end()&&(i->second.is_number_unsigned()||i->second.is_number_integer())){if(i->second.is_number_integer()&&i->second.get<int64_t>()<0)continue;return i->second.get<uint64_t>();}}return 0;}
static Decoded decode(Context &c,EVENT_RECORD *e){
    Decoded d;const auto &h=e->EventHeader;const auto &ed=h.EventDescriptor;
    d.row={{"provider",guid(h.ProviderId)},{"event_id",ed.Id},{"version",ed.Version},{"opcode",ed.Opcode},{"task",ed.Task},{"header_flags",h.Flags},{"qpc_raw",h.TimeStamp.QuadPart},{"cpu_index",e->BufferContext.ProcessorIndex},{"cpu_number",e->BufferContext.ProcessorNumber},{"header_pid",h.ProcessId},{"header_tid",h.ThreadId},{"properties",json::object()}};
    std::string key=guid(h.ProviderId)+":"+std::to_string(ed.Id)+":"+std::to_string(ed.Version)+":"+std::to_string(ed.Opcode)+":"+std::to_string(ed.Task)+":"+std::to_string(h.Flags);
    auto found=c.schemas.find(key);
    if(found==c.schemas.end()){
        ULONG size=0;ULONG status=TdhGetEventInformation(e,0,nullptr,nullptr,&size);
        if(status!=ERROR_INSUFFICIENT_BUFFER||size<sizeof(TRACE_EVENT_INFO)){d.row["metadata_error"]=status;d.row["raw_payload_hex"]=hex(static_cast<unsigned char*>(e->UserData),e->UserDataLength);return d;}
        std::vector<unsigned char>buffer(size);status=TdhGetEventInformation(e,0,nullptr,reinterpret_cast<TRACE_EVENT_INFO*>(buffer.data()),&size);
        if(status!=ERROR_SUCCESS){d.row["metadata_error"]=status;return d;}
        found=c.schemas.emplace(key,std::move(buffer)).first;
    }
    auto *info=reinterpret_cast<TRACE_EVENT_INFO*>(found->second.data());const auto *base=found->second.data();
    auto text=[&](ULONG offset){return offset&&offset<found->second.size()?utf8(reinterpret_cast<const wchar_t*>(base+offset)):std::string();};
    d.row["provider_name"]=text(info->ProviderNameOffset);d.row["task_name"]=text(info->TaskNameOffset);d.row["opcode_name"]=text(info->OpcodeNameOffset);
    for(ULONG i=0;i<info->TopLevelPropertyCount;++i){
        auto &p=info->EventPropertyInfoArray[i];const auto name=text(p.NameOffset);if(name.empty())continue;
        json item={{"flags",p.Flags}};
        if(p.Flags&PropertyStruct){item["unsupported_struct"]=true;d.row["properties"][name]=item;continue;}
        const auto type=p.nonStructType.InType;item["in_type"]=type;
        PROPERTY_DATA_DESCRIPTOR descriptor{};descriptor.PropertyName=reinterpret_cast<ULONGLONG>(base+p.NameOffset);descriptor.ArrayIndex=ULONG_MAX;
        ULONG size=0;ULONG status=TdhGetPropertySize(e,0,nullptr,1,&descriptor,&size);
        if(status!=ERROR_SUCCESS||size>(1u<<20)){item["decode_error"]=status;item["requested_bytes"]=size;d.row["properties"][name]=item;continue;}
        std::vector<unsigned char>bytes(size);status=TdhGetProperty(e,0,nullptr,1,&descriptor,size,bytes.data());
        if(status!=ERROR_SUCCESS){item["decode_error"]=status;d.row["properties"][name]=item;continue;}
        item["raw_hex"]=hex(bytes.data(),bytes.size());item["value"]=scalar(type,bytes);d.values[lower(name)]=item["value"];d.row["properties"][name]=item;
    }
    return d;
}
static bool provider(const GUID&a,const GUID&b){return InlineIsEqualGUID(a,b)!=0;}
static VOID WINAPI consume(EVENT_RECORD *e){
    auto &c=*static_cast<Context*>(e->UserContext);if(!c.error.empty())return;
    try{
        ++c.callbacks;const auto at=e->EventHeader.TimeStamp.QuadPart;const auto &p=e->EventHeader.ProviderId;const auto op=e->EventHeader.EventDescriptor.Opcode;
        if(c.pass==1){
            if(!provider(p,THREAD_GUID)&&!provider(p,PROCESS_GUID))return;
            auto d=decode(c,e);auto pid=number(d,{"ProcessId","ProcessID","PID"});auto tid=number(d,{"TThreadId","ThreadId","ThreadID"});
            if(provider(p,PROCESS_GUID)&&c.pids.count(static_cast<uint32_t>(pid))&&(op==EVENT_TRACE_TYPE_START||op==EVENT_TRACE_TYPE_DC_START))c.process_starts.push_back(d.row);
            if(provider(p,THREAD_GUID)&&c.pids.count(static_cast<uint32_t>(pid))&&tid){
                if(op==EVENT_TRACE_TYPE_START||op==EVENT_TRACE_TYPE_DC_START)c.lives.push_back({static_cast<uint32_t>(pid),static_cast<uint32_t>(tid),at});
                else if(op==EVENT_TRACE_TYPE_END||op==EVENT_TRACE_TYPE_DC_END)for(auto i=c.lives.rbegin();i!=c.lives.rend();++i)if(i->tid==tid&&i->pid==pid&&i->end==std::numeric_limits<int64_t>::max()){i->end=at;break;}
            }
            return;
        }
        c.first=std::min<int64_t>(c.first,at);c.last=std::max<int64_t>(c.last,at);
        if(!provider(p,THREAD_GUID)&&!provider(p,PROCESS_GUID)&&!provider(p,PERF_GUID)&&!provider(p,IMAGE_GUID)&&!provider(p,STACK_GUID)&&!provider(p,MARK_GUID))return;
        auto d=decode(c,e);const auto pid=number(d,{"ProcessId","ProcessID","PID","StackProcess"});const auto tid=number(d,{"TThreadId","ThreadId","ThreadID","StackThread"});
        const auto old=number(d,{"OldThreadId"}),next=number(d,{"NewThreadId"});const auto task=lower(d.row.value("task_name",std::string()));const auto opcode=lower(d.row.value("opcode_name",std::string()));
        bool selected=c.pids.count(static_cast<uint32_t>(pid))||c.pids.count(e->EventHeader.ProcessId)||c.owned(static_cast<uint32_t>(tid),at)||c.owned(static_cast<uint32_t>(old),at)||c.owned(static_cast<uint32_t>(next),at)||c.owned(e->EventHeader.ThreadId,at);
        const bool interrupt=provider(p,PERF_GUID)&&(task.find("dpc")!=std::string::npos||task.find("isr")!=std::string::npos||task.find("interrupt")!=std::string::npos||opcode.find("dpc")!=std::string::npos||opcode.find("isr")!=std::string::npos);
        const bool marker=provider(p,MARK_GUID)&&d.row["properties"].dump().find(c.instance)!=std::string::npos;
        if(!selected&&!interrupt&&!marker)return;
        d.row["event_index"]=c.callbacks;d.row["is_instance_marker"]=marker;d.row["is_global_interrupt_candidate"]=interrupt;
        // Retain only owned lifecycle identities in the derived export. Other
        // IDs are explicit labels, never interpreted as owned thread activity.
        if(!c.pids.count(e->EventHeader.ProcessId))d.row["header_pid"]="other";
        if(!c.owned(e->EventHeader.ThreadId,at))d.row["header_tid"]="other";
        for(auto i=d.row["properties"].begin();i!=d.row["properties"].end();++i){const auto name=lower(i.key());auto &item=i.value();if(!item.contains("value")||!item["value"].is_number_integer())continue;const auto value=item["value"].get<uint64_t>();if(name.find("threadid")!=std::string::npos||name=="stackthread"){if(!c.owned(static_cast<uint32_t>(value),at)){item["value"]="other";item.erase("raw_hex");}}else if(name=="processid"||name=="pid"||name=="stackprocess"||name=="parentid"){if(!c.pids.count(static_cast<uint32_t>(value))){item["value"]="other";item.erase("raw_hex");}}}
        *c.events<<d.row.dump()<<'\n';need(bool(*c.events),"selected event output write failed");++c.selected;
    }catch(const std::exception &error){c.error=error.what();}catch(...){c.error="unknown callback failure";}
}
static json read_header(const EVENT_TRACE_LOGFILEW &log){const auto &h=log.LogfileHeader;return json{{"clock_type",h.ReservedFlags},{"perf_frequency",h.PerfFreq.QuadPart},{"start_time_raw",h.StartTime.QuadPart},{"end_time_raw",h.EndTime.QuadPart},{"boot_time_raw",h.BootTime.QuadPart},{"header_events_lost",h.EventsLost},{"header_buffers_lost",h.BuffersLost},{"logfile_events_lost",log.EventsLost},{"buffer_size_kb",h.BufferSize},{"buffers_written",h.BuffersWritten},{"processors",h.NumberOfProcessors},{"pointer_size",h.PointerSize},{"log_file_mode",h.LogFileMode},{"logger_name",utf8(h.LoggerName)},{"log_file_name",utf8(h.LogFileName)},{"sdk_TRACE_LOGFILE_HEADER_bytes",sizeof(TRACE_LOGFILE_HEADER)},{"sdk_EVENT_TRACE_LOGFILEW_bytes",sizeof(EVENT_TRACE_LOGFILEW)},{"process_trace_raw_timestamp_flag",true}};}
static void scan(Context &c,const std::filesystem::path &path){
    EVENT_TRACE_LOGFILEW log{};auto source=path.wstring();log.LogFileName=source.data();log.ProcessTraceMode=PROCESS_TRACE_MODE_EVENT_RECORD|PROCESS_TRACE_MODE_RAW_TIMESTAMP;log.EventRecordCallback=consume;log.Context=&c;
    const auto handle=OpenTraceW(&log);need(handle!=INVALID_PROCESSTRACE_HANDLE,"OpenTraceW failed: "+std::to_string(GetLastError()));
    if(c.pass==1)c.header=read_header(log);TRACEHANDLE handles[]={handle};const auto status=ProcessTrace(handles,1,nullptr,nullptr);c.pass_receipts.push_back({{"pass",c.pass},{"after_ProcessTrace",read_header(log)},{"process_trace_status",status}});const auto closed=CloseTrace(handle);
    need(status==ERROR_SUCCESS,"ProcessTrace failed: "+std::to_string(status));need(closed==ERROR_SUCCESS,"CloseTrace failed: "+std::to_string(closed));need(c.error.empty(),c.error);
}
int wmain(int argc,wchar_t **argv){try{
    need(argc==4,"usage: export-etw TRACE.etl OWNED-IDENTITIES.json NEW-OUTPUT-DIR");const std::filesystem::path input(argv[1]),owned(argv[2]),out(argv[3]);need(std::filesystem::is_regular_file(input)&&std::filesystem::is_regular_file(owned),"explicit ETL and identity files required");need(!std::filesystem::exists(out),"output already exists");need(std::filesystem::create_directory(out),"cannot exclusively create output directory");
    Context c;std::ifstream config(owned);config>>c.config;c.instance=c.config.at("instance_name").get<std::string>();need(!c.instance.empty(),"instance name required");for(const auto &actor:c.config.at("actors")){const auto pid=actor.at("process_id").get<uint32_t>();need(pid>0&&c.pids.insert(pid).second,"owned process IDs must be positive and unique");}
    c.pass=1;scan(c,input);const auto first_callbacks=c.callbacks;c.callbacks=0;
    std::ofstream events(out/"owned-events.jsonl",std::ios::binary);need(bool(events),"cannot create event output");c.events=&events;c.pass=2;scan(c,input);events.close();
    for(const auto &life:c.lives)c.thread_lifetimes.push_back({{"process_id",life.pid},{"thread_id",life.tid},{"begin_qpc",life.begin},{"end_qpc",life.end==std::numeric_limits<int64_t>::max()?json(nullptr):json(life.end)}});
    bool no_loss=c.header.at("header_events_lost")==0&&c.header.at("header_buffers_lost")==0&&c.header.at("logfile_events_lost")==0;for(const auto &receipt:c.pass_receipts){const auto &h=receipt.at("after_ProcessTrace");no_loss=no_loss&&h.at("header_events_lost")==0&&h.at("header_buffers_lost")==0&&h.at("logfile_events_lost")==0;}
    json report={{"schema","owned-etw-raw-export-1"},{"etl",input.u8string()},{"owned_identity_path",owned.u8string()},{"identity",c.config},{"header",c.header},{"pass_receipts",c.pass_receipts},{"process_starts",c.process_starts},{"thread_lifetimes",c.thread_lifetimes},{"first_pass_event_count",first_callbacks},{"second_pass_event_count",c.callbacks},{"selected_event_count",c.selected},{"first_event_qpc",c.first},{"last_event_qpc",c.last},{"header_loss_counters_zero",no_loss},{"clock_is_qpc",c.header.at("clock_type")==1},{"identity_lifetimes_fully_verified",false},{"raw_marker_clock_verified",false},{"exact_scheduler_intervals_validated",false},{"performance_prediction_or_fitting_performed",false},{"unrelated_process_names_exported",false},{"source_scope","offline SDK OpenTraceW/ProcessTrace/TDH decode; full ETL retained separately; classifier must validate actual schemas and lifetime/marker checks"}};
    std::ofstream summary(out/"raw-export.json",std::ios::binary);summary<<report.dump(2)<<'\n';need(bool(summary),"summary output failed");std::cout<<json{{"status","exported_unqualified"},{"selected_events",c.selected},{"header_loss_counters_zero",no_loss}}.dump()<<'\n';return 0;
}catch(const std::exception &error){std::cerr<<"export-etw: "<<error.what()<<'\n';return 2;}}
