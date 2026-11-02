#pragma once
// This object changes no process-wide or system-wide timer-resolution setting.
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif

namespace dynamic_control {
class OwnedHighResolutionTimer {
    HANDLE timer_=nullptr;
public:
    OwnedHighResolutionTimer(){
        timer_=CreateWaitableTimerExW(nullptr,nullptr,CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,TIMER_MODIFY_STATE|SYNCHRONIZE);
        require(timer_!=nullptr,"owned high-resolution waitable timer unavailable; no Sleep/spin fallback");
    }
    ~OwnedHighResolutionTimer(){if(timer_)CloseHandle(timer_);}
    OwnedHighResolutionTimer(const OwnedHighResolutionTimer&)=delete;
    OwnedHighResolutionTimer& operator=(const OwnedHighResolutionTimer&)=delete;
    static int64_t relative_due_100ns(int64_t requested_ns){
        require(requested_ns>0&&requested_ns<INT64_MAX-99,"positive bounded high-resolution wait required");
        return -(requested_ns+99)/100;
    }
    bool wait_ns(int64_t requested_ns,HANDLE cancellation=nullptr){
        LARGE_INTEGER due;due.QuadPart=relative_due_100ns(requested_ns);
        require(SetWaitableTimerEx(timer_,&due,0,nullptr,nullptr,nullptr,0)!=0,"owned high-resolution timer arm failed");
        HANDLE handles[2]={timer_,cancellation};const DWORD count=cancellation?2:1;
        const auto status=WaitForMultipleObjects(count,handles,FALSE,INFINITE);
        require(status==WAIT_OBJECT_0||(cancellation&&status==WAIT_OBJECT_0+1),"owned high-resolution timer wait failed");
        return status==WAIT_OBJECT_0;
    }
    bool wait_until_qpc(int64_t target,int64_t frequency,HANDLE cancellation=nullptr){
        require(frequency>0,"QPC frequency required for owned timer");
        while(true){
            const auto now=steady::tick();if(now>=target)return true;
            const auto remaining=target-now;
            // Round upward to the next 100ns timer tick. No busy-spin tail.
            const auto ns=static_cast<int64_t>(std::ceil(static_cast<long double>(remaining)*1000000000.0L/frequency));
            if(!wait_ns(ns,cancellation))return false;
        }
    }
    static json describe(){return json{{"kind","owned-high-resolution-waitable-timer-v1"},
        {"api","CreateWaitableTimerExW/SetWaitableTimerEx/WaitForMultipleObjects"},
        {"creation_flag","CREATE_WAITABLE_TIMER_HIGH_RESOLUTION"},{"timer_tolerable_delay_ms",0},
        {"busy_spin",false},{"global_timeBeginPeriod_called",false},{"Sleep_fallback",false},
        {"requested_time_is_actual_elapsed_time",false},{"handle_closed_by_RAII",true}};}
};
}
