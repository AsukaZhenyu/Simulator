// Standalone integer routing primitive. No model, GGML runtime or LLM timing.
#define NOMINMAX
#include <windows.h>
#include <cuda_runtime.h>
#include "mmid_extracted.cuh"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
using json=nlohmann::json;
static void require(bool value,const std::string &message){if(!value)throw std::runtime_error(message);}
static void check(cudaError_t error,const char *where){if(error!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(error));}
struct Device {
 int32_t *p=nullptr;size_t n=0;
 explicit Device(size_t count):n(count){require(count>0&&count<=262144,"bounded owned I32 allocation required");check(cudaMalloc(reinterpret_cast<void**>(&p),count*4),"cudaMalloc");}
 ~Device(){if(p)cudaFree(p);}
 Device(const Device&)=delete;Device&operator=(const Device&)=delete;
};
struct Stream {cudaStream_t value=nullptr;Stream(){check(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking),"stream create");}~Stream(){if(value)cudaStreamDestroy(value);}};
static std::vector<int32_t> integers(const json &v){
 require(v.is_array(),"integer array required");std::vector<int32_t> out;out.reserve(v.size());for(const auto &x:v){require(x.is_number_integer(),"I32 element required");auto n=x.get<int64_t>();require(n>=INT32_MIN&&n<=INT32_MAX,"I32 range exceeded");out.push_back(static_cast<int32_t>(n));}return out;
}
static json cpu_reference(const json &c){
 for(const char *key:{"n_tokens","n_experts","si1","sis1","nchannels_y"})require(c.at(key).is_number_integer(),"integer geometry required");
 const int n=c.at("n_tokens"),experts=c.at("n_experts"),si1=c.at("si1"),sis1=c.at("sis1"),channels=c.at("nchannels_y");
 require(n>=1&&n<=256&&experts>=8&&experts<=1024&&si1>=8&&si1<=4096&&sis1>=1&&sis1<=4096&&(channels==1||channels==8),"reviewed MMID8 shape/stride envelope exceeded");
 auto ids=integers(c.at("ids_storage"));require(ids.size()==size_t(n)*si1,"full strided input extent differs");
 for(int t=0;t<n;++t){std::vector<int32_t> seen;for(int k=0;k<8;++k){int e=ids[t*si1+k];require(e>=0&&e<experts&&std::find(seen.begin(),seen.end(),e)==seen.end(),"each token needs unique in-range top8");seen.push_back(e);}}
 std::vector<int32_t> src,dst,bounds;src.reserve(n*8);dst.reserve(n*8);bounds.reserve(experts+1);
 for(int expert=0;expert<experts;++expert){bounds.push_back(static_cast<int32_t>(src.size()));for(int t=0;t<n;++t)for(int k=0;k<8;++k)if(ids[t*si1+k]==expert){src.push_back(t*sis1+k%channels);dst.push_back(t*8+k);}}
 bounds.push_back(static_cast<int32_t>(src.size()));require(src.size()==size_t(n)*8,"reference selected extent differs");
 return json{{"ids_src1",src},{"ids_dst",dst},{"expert_bounds",bounds}};
}
static void save_new(const std::filesystem::path &path,const json &value){const auto data=value.dump()+"\n";HANDLE file=CreateFileW(path.c_str(),GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);require(file!=INVALID_HANDLE_VALUE,"exclusive output creation failed");size_t offset=0;bool ok=true;while(offset<data.size()){DWORD done=0;const DWORD count=static_cast<DWORD>(std::min<size_t>(data.size()-offset,1<<20));if(!WriteFile(file,data.data()+offset,count,&done,nullptr)||done!=count){ok=false;break;}offset+=done;}CloseHandle(file);require(ok,"output write failed");}
static json compare_output(const std::vector<int32_t>&actual,const std::vector<int32_t>&expected){
 require(actual.size()==expected.size()+32,"guarded output extent differs");bool guards=true;for(size_t i=0;i<16;++i)guards=guards&&actual[i]==int32_t(0x6b6b6b6b)&&actual[actual.size()-16+i]==int32_t(0x6b6b6b6b);
 size_t equal=0;json first=nullptr;for(size_t i=0;i<expected.size();++i){equal+=actual[16+i]==expected[i];if(actual[16+i]!=expected[i]&&first.is_null())first=json{{"index",i},{"actual",actual[16+i]},{"expected",expected[i]}};}
 return json{{"bytewise_equal",std::memcmp(actual.data()+16,expected.data(),expected.size()*4)==0},{"equal_I32_entries",equal},{"I32_entries",expected.size()},{"first_difference",first},{"redzones_unchanged",guards},{"actual",std::vector<int32_t>(actual.begin()+16,actual.end()-16)}};
}
int main(int argc,char **argv){try{
 require(argc==4,"usage: mmid-probe --validate-only|--run SUITE.json OUTPUT.json");const std::string mode=argv[1];require(mode=="--validate-only"||mode=="--run","unknown explicit mode");
 const auto output=std::filesystem::u8path(argv[3]);require(!std::filesystem::exists(output),"output exists; no overwrite");std::ifstream input(std::filesystem::u8path(argv[2]));require(input.good(),"suite unreadable");json suite;input>>suite;
 require(suite.at("schema")=="real-moe-mmid-primitive-suite-1"&&suite.at("top_k")==8,"reviewed primitive suite required");const int repeats=suite.at("launch_repeats");require(repeats>=1&&repeats<=3,"fixed repeat envelope exceeded");
 const auto variants=suite.at("variants").get<std::vector<std::string>>();require(!variants.empty()&&variants.size()<=2,"explicit variants required");for(const auto &v:variants)require(v=="original"||v=="syncwarp","variant outside original/fence-only scope");
 require(!suite.at("cases").empty()&&suite.at("cases").size()<=42,"fixed case envelope exceeded");
 for(const auto &c:suite.at("cases")){const auto ref=cpu_reference(c);for(const char *key:{"ids_src1","ids_dst","expert_bounds"})require(ref.at(key)==c.at("reference").at(key),"declared Python and native independent CPU references differ");}
 if(mode=="--validate-only"){save_new(output,json{{"schema","real-moe-mmid-suite-validation-1"},{"passed",true},{"cases",suite.at("cases").size()},{"GPU_executed",false}});return 0;}
 MEMORYSTATUSEX host{};host.dwLength=sizeof(host);require(GlobalMemoryStatusEx(&host)!=0&&host.ullAvailPhys>=512*1024*1024ULL,"fixed 512MiB host reserve not met");
 int devices=0;check(cudaGetDeviceCount(&devices),"device count");require(devices>0,"CUDA0 required; no fallback");check(cudaSetDevice(0),"select CUDA0");cudaDeviceProp properties{};check(cudaGetDeviceProperties(&properties,0),"device properties");require(properties.warpSize==32&&properties.major*100+properties.minor*10==890,"pinned NVIDIA arch89/warp32 target required");
 size_t free=0,total=0;check(cudaMemGetInfo(&free,&total),"device capacity");require(free>=64*1024*1024,"fixed 64MiB device reserve not met");
 json results=json::array();bool all=true;size_t maximum_owned=0;Stream stream;
 for(const auto &c:suite.at("cases")){
  const int n=c.at("n_tokens"),experts=c.at("n_experts"),si1=c.at("si1"),sis1=c.at("sis1"),channels=c.at("nchannels_y");auto ids=integers(c.at("ids_storage"));auto ref=cpu_reference(c);
  const size_t length=size_t(n)*8,blength=size_t(experts)+1;const size_t owned=(ids.size()+2*(2*(length+32)+(blength+32)))*4;require(owned<=1024*1024,"fixed one-MiB owned CUDA allocation budget exceeded");maximum_owned=std::max(maximum_owned,owned);
  Device device_ids(ids.size()),a_src(length+32),a_dst(length+32),a_bounds(blength+32),b_src(length+32),b_dst(length+32),b_bounds(blength+32);
  check(cudaMemcpyAsync(device_ids.p,ids.data(),ids.size()*4,cudaMemcpyHostToDevice,stream.value),"input H2D");
  std::vector<std::string> order=variants;if(c.at("case_index").get<int>()%2)std::reverse(order.begin(),order.end());
  for(int repeat=0;repeat<repeats;++repeat)for(const auto &variant:order){
   Device &s=variant=="original"?a_src:b_src,&d=variant=="original"?a_dst:b_dst,&bounds=variant=="original"?a_bounds:b_bounds;
   const int poison=variant=="original"?0xa5:0x5a;
   for(Device *p:{&s,&d,&bounds}){check(cudaMemsetAsync(p->p,0x6b,p->n*4,stream.value),"redzone initialize");check(cudaMemsetAsync(p->p+16,poison,(p->n-32)*4,stream.value),"distinct logical output poison");}
   if(variant=="original")mm_ids_helper_original<8><<<dim3(experts),dim3(32),size_t(n)*4,stream.value>>>(device_ids.p,s.p+16,d.p+16,bounds.p+16,n,8,channels,si1,sis1);
   else mm_ids_helper_syncwarp<8><<<dim3(experts),dim3(32),size_t(n)*4,stream.value>>>(device_ids.p,s.p+16,d.p+16,bounds.p+16,n,8,channels,si1,sis1);
   check(cudaGetLastError(),"primitive launch");check(cudaStreamSynchronize(stream.value),"primitive completion");
   std::vector<int32_t> hs(s.n),hd(d.n),hb(bounds.n);check(cudaMemcpy(hs.data(),s.p,s.n*4,cudaMemcpyDeviceToHost),"src1 D2H");check(cudaMemcpy(hd.data(),d.p,d.n*4,cudaMemcpyDeviceToHost),"dst D2H");check(cudaMemcpy(hb.data(),bounds.p,bounds.n*4,cudaMemcpyDeviceToHost),"bounds D2H");
   auto rs=compare_output(hs,integers(ref.at("ids_src1"))),rd=compare_output(hd,integers(ref.at("ids_dst"))),rb=compare_output(hb,integers(ref.at("expert_bounds")));
   const bool pass=rs.at("bytewise_equal").get<bool>()&&rd.at("bytewise_equal").get<bool>()&&rb.at("bytewise_equal").get<bool>()&&rs.at("redzones_unchanged").get<bool>()&&rd.at("redzones_unchanged").get<bool>()&&rb.at("redzones_unchanged").get<bool>();all=all&&pass;
   results.push_back(json{{"case_id",c.at("id")},{"variant",variant},{"repeat",repeat},{"passed",pass},{"ids_src1",rs},{"ids_dst",rd},{"expert_bounds",rb},{"launch_blocks",experts},{"launch_threads",32},{"dynamic_shared_bytes",n*4},{"direct_launch",true},{"same_actual_stream_opaque_id",reinterpret_cast<uintptr_t>(stream.value)}});
  }
 }
 save_new(output,json{{"schema","real-moe-mmid-primitive-result-1"},{"passed",all},{"results",results},{"launches",results.size()},{"case_count",suite.at("cases").size()},{"device",json{{"ordinal",0},{"name",properties.name},{"major",properties.major},{"minor",properties.minor},{"warp_size",properties.warpSize}}},{"host_available_before_CUDA_bytes",host.ullAvailPhys},{"GPU_free_before_bytes",free},{"maximum_owned_cuda_buffer_bytes",maximum_owned},{"model_loaded",false},{"LLM_accuracy_evaluated",false},{"performance_timing_evaluated",false},{"CUDA_graph_capture_replay_exercised",false},{"shared_memory_race_absence_proven",false}});return all?0:3;
 }catch(const std::exception &error){std::cerr<<"mmid primitive: "<<error.what()<<'\n';return 2;}}
