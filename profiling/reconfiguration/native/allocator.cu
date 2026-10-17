// CUDA API microprobe, independent of GGML allocator and mapper control logic.
// No synthetic fallback, hidden defaults, or inferred private workspace.
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
using Clock=std::chrono::steady_clock;
static void ok(cudaError_t code,const char* label) {
    if(code!=cudaSuccess)throw std::runtime_error(std::string(label)+": "+cudaGetErrorString(code));
}
static int64_t integer(const std::string& text) {
    size_t used=0;auto value=std::stoll(text,&used);
    if(used!=text.size()||value<0||value>67108864)throw std::runtime_error("invalid explicit integer");
    return value;
}
struct Options {size_t bytes=0;int device=0,warmup=0,samples=0;uint32_t seed=0;};
static Options parse(int argc,char**argv) {
    const std::set<std::string> expected={"--bytes","--device","--warmup","--samples","--seed"};
    std::set<std::string> seen;Options o;
    for(int i=1;i<argc;i+=2){std::string flag=argv[i];if(i+1>=argc||!expected.count(flag)||!seen.insert(flag).second)throw std::runtime_error("missing, unknown, or duplicate option");
        auto value=integer(argv[i+1]);if(flag=="--bytes")o.bytes=size_t(value);else if(flag=="--device")o.device=int(value);else if(flag=="--warmup")o.warmup=int(value);else if(flag=="--samples")o.samples=int(value);else o.seed=uint32_t(value);
    }
    if(seen!=expected||o.bytes<1024||o.device>63||o.warmup>1000||o.samples<3||o.samples>1000||o.seed>1000000)throw std::runtime_error("all documented explicit fields required within small probe limits");
    return o;
}
template<class F>static int64_t timed(F&& f) {
    auto begin=Clock::now();f();auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-begin).count();
    if(ns<1)throw std::runtime_error("nonpositive measured time");return ns;
}
static void array(const std::vector<int64_t>& values) {std::cout<<'[';for(size_t i=0;i<values.size();++i)std::cout<<(i?",":"")<<values[i];std::cout<<']';}
struct Memory {size_t free=0,total=0;};
static Memory memory(){Memory m;ok(cudaMemGetInfo(&m.free,&m.total),"cudaMemGetInfo");return m;}
static int run(const Options&o) {
    ok(cudaSetDevice(o.device),"cudaSetDevice");ok(cudaFree(nullptr),"initialize context");
    cudaDeviceProp prop{};ok(cudaGetDeviceProperties(&prop,o.device),"cudaGetDeviceProperties");
    int driver=0,runtime=0;ok(cudaDriverGetVersion(&driver),"driver version");ok(cudaRuntimeGetVersion(&runtime),"runtime version");
    cudaStream_t stream=nullptr;ok(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"create stream");
    uint8_t* host=nullptr;ok(cudaMallocHost(reinterpret_cast<void**>(&host),o.bytes),"pinned host allocation");
    std::mt19937 rng(o.seed);for(size_t i=0;i<o.bytes;++i)host[i]=uint8_t(rng()>>24);
    std::vector<uint8_t> copy(o.bytes);std::map<std::string,std::vector<int64_t>> times;
    struct Checkpoint {Memory before,allocated,after;};std::vector<Checkpoint> checkpoints;
    for(int i=-o.warmup;i<o.samples;++i) {
        Checkpoint c;c.before=memory();uint8_t* device=nullptr;
        auto alloc=timed([&](){ok(cudaMalloc(reinterpret_cast<void**>(&device),o.bytes),"cudaMalloc");});c.allocated=memory();
        auto reload=timed([&](){ok(cudaMemcpyAsync(device,host,o.bytes,cudaMemcpyHostToDevice,stream),"H2D");ok(cudaStreamSynchronize(stream),"H2D wait");});
        // Check every byte; neither D2H validation nor comparison enters timing.
        ok(cudaMemcpy(copy.data(),device,o.bytes,cudaMemcpyDeviceToHost),"validation D2H");
        if(std::memcmp(copy.data(),host,o.bytes)!=0)throw std::runtime_error("reload byte validation failed");
        auto sync=timed([&](){ok(cudaStreamSynchronize(stream),"idle synchronize");});
        auto release=timed([&](){ok(cudaFree(device),"cudaFree");});c.after=memory();
        if(i>=0){times["cuda_malloc"].push_back(alloc);times["pinned_h2d_completed"].push_back(reload);times["idle_stream_synchronize"].push_back(sync);times["cuda_free"].push_back(release);checkpoints.push_back(c);}
    }
    ok(cudaFreeHost(host),"cudaFreeHost");ok(cudaStreamDestroy(stream),"destroy stream");
    std::cout<<"{\"schema\":\"reconfiguration-allocator-native-0.1\",\"bytes\":"<<o.bytes<<",\"device\":"<<o.device<<",\"warmup\":"<<o.warmup<<",\"samples\":"<<o.samples<<",\"seed\":"<<o.seed<<",\"device_name\":\""<<prop.name<<"\",\"cuda_driver_version\":"<<driver<<",\"cuda_runtime_version\":"<<runtime<<",\"host_memory\":\"cudaMallocHost_pinned\",\"device_allocator\":\"direct_cudaMalloc_cudaFree_not_GGML_or_async_pool\",\"timer\":\"steady_clock_completed_api_wall\",\"cuda_graphs\":\"not_used\",\"validation\":\"all_bytes_exact_each_iteration\",\"workspace_bytes\":null,\"logical_evict_cost_ns\":null,\"mapper_control_cost_ns\":null,\"allocator_release_guaranteed_by_observed_free_delta\":false,\"wall_ns\":{";
    bool first=true;for(const auto&t:times){std::cout<<(first?"":",")<<'"'<<t.first<<"\":";array(t.second);first=false;}
    std::cout<<"},\"memory_checkpoints\":[";for(size_t i=0;i<checkpoints.size();++i){const auto&c=checkpoints[i];std::cout<<(i?",":"")<<"{\"before_free_bytes\":"<<c.before.free<<",\"allocated_free_bytes\":"<<c.allocated.free<<",\"after_free_bytes\":"<<c.after.free<<",\"total_bytes\":"<<c.before.total<<'}';}
    std::cout<<"]}\n";return 0;
}
int main(int argc,char**argv){try{return run(parse(argc,argv));}catch(const std::exception&e){std::cerr<<"reconfiguration-allocator-probe: "<<e.what()<<'\n';return 1;}}
