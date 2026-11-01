#pragma once
#include <tlhelp32.h>

namespace diagnostic_v3 {
static json own_thread_inventory(const char *window){
    const auto begin=steady::tick();const DWORD owner=GetCurrentProcessId();json rows=json::array();
    HANDLE snapshot=CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD,0);
    if(snapshot==INVALID_HANDLE_VALUE)return json{{"window",window},{"query_begin_tick",begin},{"query_end_tick",steady::tick()},
        {"available",false},{"error",GetLastError()},{"threads",rows}};
    using Description=HRESULT(WINAPI*)(HANDLE,PWSTR*);const auto kernel=GetModuleHandleW(L"kernel32.dll");
    auto describe=reinterpret_cast<Description>(GetProcAddress(kernel,"GetThreadDescription"));
    THREADENTRY32 entry{};entry.dwSize=sizeof(entry);BOOL next=Thread32First(snapshot,&entry);
    while(next){
        if(entry.th32OwnerProcessID==owner){json row={{"thread_id",entry.th32ThreadID},{"is_calling_thread",entry.th32ThreadID==GetCurrentThreadId()},
            {"priority",nullptr},{"group",nullptr},{"affinity_mask",nullptr},{"description",nullptr},{"unknown",json::object()}};
            HANDLE thread=OpenThread(THREAD_QUERY_INFORMATION|THREAD_QUERY_LIMITED_INFORMATION,FALSE,entry.th32ThreadID);
            if(!thread)row["unknown"]["thread_handle"]=GetLastError();
            else {
                GROUP_AFFINITY affinity{};if(GetThreadGroupAffinity(thread,&affinity)){row["group"]=affinity.Group;row["affinity_mask"]=static_cast<uint64_t>(affinity.Mask);}
                else row["unknown"]["affinity"]=GetLastError();
                SetLastError(ERROR_SUCCESS);const auto priority=GetThreadPriority(thread);
                if(priority!=THREAD_PRIORITY_ERROR_RETURN)row["priority"]=priority;else row["unknown"]["priority"]=GetLastError();
                if(describe){PWSTR value=nullptr;const auto status=describe(thread,&value);
                    if(SUCCEEDED(status)&&value){const int bytes=WideCharToMultiByte(CP_UTF8,0,value,-1,nullptr,0,nullptr,nullptr);
                        if(bytes>0){std::vector<char> text(static_cast<size_t>(bytes));WideCharToMultiByte(CP_UTF8,0,value,-1,text.data(),bytes,nullptr,nullptr);row["description"]=std::string(text.data());}
                        else row["unknown"]["description_conversion"]=GetLastError();LocalFree(value);}
                    else row["unknown"]["description_hresult"]=static_cast<int64_t>(status);
                }else row["unknown"]["description_API"]="unavailable";
                CloseHandle(thread);
            }
            rows.push_back(std::move(row));
        }
        entry.dwSize=sizeof(entry);next=Thread32Next(snapshot,&entry);
    }
    CloseHandle(snapshot);return json{{"window",window},{"query_begin_tick",begin},{"query_end_tick",steady::tick()},
        {"available",true},{"owned_process_id",owner},{"threads",rows},
        {"scope","point observations of own-process threads; helper role/inheritance, running/ready/blocked and continuous CPU residency not inferred"}};
}
}
