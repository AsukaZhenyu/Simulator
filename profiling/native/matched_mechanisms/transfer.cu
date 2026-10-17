// Real copy APIs / mapped-host SM kernels. CE+SM is an observed pair, not a
// fitted general PCIe arbitration law. Empty-kernel timing is not subtracted.
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
static void ok(cudaError_t e,const char*n){if(e!=cudaSuccess)throw std::runtime_error(std::string(n)+": "+cudaGetErrorString(e));}
__global__ void sm_copy(unsigned char*dst,const unsigned char*src,size_t n){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(blockDim.x)*gridDim.x)dst[i]=src[i];}
__global__ void empty_kernel(){ }
static int integer(const std::string&v){size_t p=0;auto x=std::stoll(v,&p);if(p!=v.size()||x<0||x>100000000)throw std::runtime_error("invalid integer");return int(x);}
static void array(const std::vector<int64_t>&v){std::cout<<'[';for(size_t n=0;n<v.size();++n)std::cout<<(n?",":"")<<v[n];std::cout<<']';}
struct Buffers {
    void *ha=nullptr,*hb=nullptr,*ma=nullptr,*mb=nullptr,*da=nullptr,*db=nullptr;
    cudaStream_t ce=nullptr,sm=nullptr;cudaEvent_t ca=nullptr,cb=nullptr,sa=nullptr,sb=nullptr;
    ~Buffers(){if(ce)cudaStreamSynchronize(ce);if(sm)cudaStreamSynchronize(sm);for(auto x:{ca,cb,sa,sb})if(x)cudaEventDestroy(x);if(ce)cudaStreamDestroy(ce);if(sm)cudaStreamDestroy(sm);if(da)cudaFree(da);if(db)cudaFree(db);if(ha)cudaFreeHost(ha);if(hb)cudaFreeHost(hb);}
};
int main(int argc,char**argv){try {
    std::set<std::string>seen;std::map<std::string,std::string> args;
    const std::set<std::string>keys={"--mode","--bytes","--warmup","--samples","--device","--blocks","--threads-per-block"};
    for(int i=1;i<argc;i+=2){std::string k=argv[i];if(i+1==argc||!keys.count(k)||!seen.insert(k).second)throw std::runtime_error("unknown/missing/duplicate option");args[k]=argv[i+1];}
    if(seen!=keys)throw std::runtime_error("all options required");
    std::string mode=args.at("--mode");int bytes=integer(args.at("--bytes")),warmup=integer(args.at("--warmup")),samples=integer(args.at("--samples")),device=integer(args.at("--device")),blocks=integer(args.at("--blocks")),threads=integer(args.at("--threads-per-block"));
    const std::set<std::string>modes={"ce_h2d","ce_d2h","sm_h2d","sm_d2h","ce_sm_duplex","empty_kernel","empty_events"};
    if(!modes.count(mode)||bytes<1||bytes>67108864||warmup>10000||samples<3||samples>10000||blocks<1||blocks>1024||threads<32||threads>1024)throw std::runtime_error("unsupported execution budget");
    ok(cudaSetDevice(device),"device");ok(cudaSetDeviceFlags(cudaDeviceMapHost),"mapped host flag");cudaDeviceProp props{};ok(cudaGetDeviceProperties(&props,device),"properties");if(!props.canMapHostMemory)throw std::runtime_error("mapped-host SM copies unsupported");
    Buffers b;ok(cudaHostAlloc(&b.ha,bytes,cudaHostAllocMapped),"pinned A");ok(cudaHostAlloc(&b.hb,bytes,cudaHostAllocMapped),"pinned B");ok(cudaHostGetDevicePointer(&b.ma,b.ha,0),"mapped A");ok(cudaHostGetDevicePointer(&b.mb,b.hb,0),"mapped B");ok(cudaMalloc(&b.da,bytes),"device A");ok(cudaMalloc(&b.db,bytes),"device B");
    std::memset(b.ha,0x5a,bytes);std::memset(b.hb,0,bytes);ok(cudaMemset(b.da,0x5a,bytes),"init A");ok(cudaMemset(b.db,0x5a,bytes),"init B");
    ok(cudaStreamCreateWithFlags(&b.ce,cudaStreamNonBlocking),"CE stream");ok(cudaStreamCreateWithFlags(&b.sm,cudaStreamNonBlocking),"SM stream");for(auto*e:{&b.ca,&b.cb,&b.sa,&b.sb})ok(cudaEventCreate(e),"event");
    bool use_ce=mode=="ce_h2d"||mode=="ce_d2h"||mode=="ce_sm_duplex",use_sm=mode=="sm_h2d"||mode=="sm_d2h"||mode=="ce_sm_duplex"||mode=="empty_kernel"||mode=="empty_events";
    auto launch=[&](bool events){
        if(use_ce){if(events)ok(cudaEventRecord(b.ca,b.ce),"CE begin");if(mode=="ce_d2h")ok(cudaMemcpyAsync(b.hb,b.da,bytes,cudaMemcpyDeviceToHost,b.ce),"CE D2H");else ok(cudaMemcpyAsync(b.da,b.ha,bytes,cudaMemcpyHostToDevice,b.ce),"CE H2D");if(events)ok(cudaEventRecord(b.cb,b.ce),"CE end");}
        if(use_sm){if(events)ok(cudaEventRecord(b.sa,b.sm),"SM begin");if(mode=="sm_h2d")sm_copy<<<blocks,threads,0,b.sm>>>((unsigned char*)b.da,(unsigned char*)b.ma,bytes);else if(mode=="sm_d2h"||mode=="ce_sm_duplex")sm_copy<<<blocks,threads,0,b.sm>>>((unsigned char*)b.mb,(unsigned char*)b.db,bytes);else if(mode=="empty_kernel")empty_kernel<<<1,32,0,b.sm>>>();ok(cudaGetLastError(),"kernel enqueue");if(events)ok(cudaEventRecord(b.sb,b.sm),"SM end");}
        if(events){if(use_ce)ok(cudaEventSynchronize(b.cb),"CE completion");if(use_sm)ok(cudaEventSynchronize(b.sb),"SM completion");}else{if(use_ce)ok(cudaStreamSynchronize(b.ce),"CE warmup");if(use_sm)ok(cudaStreamSynchronize(b.sm),"SM warmup");}
    };
    for(int n=0;n<std::max(1,warmup);++n)launch(false);
    std::vector<int64_t>wall,ce,sm;for(int n=0;n<samples;++n){auto start=std::chrono::steady_clock::now();launch(true);wall.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count());float ms=0;if(use_ce){ok(cudaEventElapsedTime(&ms,b.ca,b.cb),"CE interval");ce.push_back(int64_t(std::llround(ms*1e6)));}if(use_sm){ok(cudaEventElapsedTime(&ms,b.sa,b.sb),"SM interval");sm.push_back(int64_t(std::llround(ms*1e6)));}}
    if(mode=="ce_h2d"||mode=="sm_h2d"||mode=="ce_sm_duplex"){std::vector<uint8_t>out(bytes);ok(cudaMemcpy(out.data(),b.da,bytes,cudaMemcpyDeviceToHost),"validate H2D");for(auto x:out)if(x!=0x5a)throw std::runtime_error("H2D mismatch");}
    if(mode=="ce_d2h"||mode=="sm_d2h"||mode=="ce_sm_duplex")for(int n=0;n<bytes;++n)if(((uint8_t*)b.hb)[n]!=0x5a)throw std::runtime_error("D2H mismatch");
    std::cout<<"{\"schema\":\"matched-transfer-native-0.1\",\"mode\":\""<<mode<<"\",\"bytes\":"<<bytes<<",\"samples\":"<<samples<<",\"device_index\":"<<device<<",\"device_name\":\""<<props.name<<"\",\"host_memory\":\"pinned_mapped\",\"kernel_blocks\":"<<blocks<<",\"kernel_threads\":"<<threads<<",\"ce_api\":\"cudaMemcpyAsync\",\"sm_api\":\"actual_cuda_kernel_copy_mapped_host_memory\",\"timing_scope\":\"enqueue_events_and_completed_work_wall\",\"general_shared_link_model_calibrated\":false,\"validation\":\"all_bytes_exact\",\"wall_ns\":";array(wall);std::cout<<",\"ce_interval_ns\":";array(ce);std::cout<<",\"sm_interval_ns\":";array(sm);std::cout<<",\"notes\":\"SM host mapping is a concrete UVA implementation; empty intervals include event scheduling and must not be subtracted as pure launch overhead\"}\n";return 0;
}catch(const std::exception&e){std::cerr<<"matched-transfer-probe: "<<e.what()<<'\n';return 1;}}
