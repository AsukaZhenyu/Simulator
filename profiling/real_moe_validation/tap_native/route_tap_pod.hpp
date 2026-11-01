// CPU-consumer observation. No allocation, JSON, exception, tensor mutation,
// graph split or synchronization in the registered compute callback.
#pragma once
#include <array>
#include <map>
#include <bcrypt.h>
#include <iomanip>
#include <sstream>
#include "route_tap_api.h"

static std::string tap_sha256(const void * data, size_t bytes) {
 require(bytes<=ULONG_MAX,"bounded logit/metadata hash required");
 BCRYPT_ALG_HANDLE algorithm=nullptr; BCRYPT_HASH_HANDLE hash=nullptr;
 require(BCryptOpenAlgorithmProvider(&algorithm,BCRYPT_SHA256_ALGORITHM,nullptr,0)>=0,"SHA provider failed");
 ULONG size=0,done=0; require(BCryptGetProperty(algorithm,BCRYPT_OBJECT_LENGTH,reinterpret_cast<PUCHAR>(&size),sizeof(size),&done,0)>=0,"SHA object size failed");
 std::vector<uint8_t> object(size); std::array<uint8_t,32> digest{};
 require(BCryptCreateHash(algorithm,&hash,object.data(),size,nullptr,0,0)>=0,"SHA create failed");
 const auto a=BCryptHashData(hash,reinterpret_cast<PUCHAR>(const_cast<void*>(data)),static_cast<ULONG>(bytes),0);
 const auto b=BCryptFinishHash(hash,digest.data(),static_cast<ULONG>(digest.size()),0);
 BCryptDestroyHash(hash); BCryptCloseAlgorithmProvider(algorithm,0); require(a>=0&&b>=0,"SHA calculation failed");
 std::ostringstream out; out<<std::hex<<std::setfill('0'); for(auto v:digest)out<<std::setw(2)<<unsigned(v); return out.str();
}
struct TapRecord {int forward=-1,layer=-1,stage=-1,rows=0,slots=0,position=0,submitted=0; bool decode=false; size_t offset=0;};
struct Capture {
 bool enabled=false,decode=false; int forward=-1,position=0,submitted=0,error=0,layers=0,experts=0,top_k=0;
 size_t used=0,ids_used=0; int64_t hook_ticks=0;
 std::vector<TapRecord> records; std::vector<int32_t> ids;
 static int stage_layer(const char * name,int &layer) noexcept {
  const char * prefixes[]={"ffn_moe_up-","ffn_moe_gate-","ffn_moe_down-"};
  for(int s=0;s<3;++s){const auto n=std::strlen(prefixes[s]); if(std::strncmp(name,prefixes[s],n))continue;
   const char *p=name+n; if(!*p)return -1; layer=0;
   for(;*p;++p){if(*p<'0'||*p>'9'||layer>4096)return -1;layer=layer*10+(*p-'0');}return s;}
  return -1;
 }
 void prepare(int maximum_forwards,int total_rows,int layer_count,int expert_count,int k){
  require(maximum_forwards>0&&total_rows>0&&layer_count>0&&k>0,"bounded capture allocation required");
  const uint64_t n=uint64_t(maximum_forwards)*layer_count*3, z=uint64_t(total_rows)*layer_count*3*k;
  require(n<=1000000&&z<=20000000,"capture exceeds owned POD envelope");
  records.assign(static_cast<size_t>(n),{}); ids.assign(static_cast<size_t>(z),0);
  used=ids_used=0;hook_ticks=0;error=0;layers=layer_count;experts=expert_count;top_k=k;
 }
 static void callback(const ggml_tensor *dst,const ggml_tensor *source,void *user) noexcept {
  auto &c=*static_cast<Capture*>(user);if(!c.enabled||c.error)return;
  LARGE_INTEGER first{},last{};QueryPerformanceCounter(&first);
  int layer=-1;const int stage=stage_layer(dst->name,layer);
  if(stage<0)return;
  if(layer<0||layer>=c.layers||!source||source->type!=GGML_TYPE_I32||source->ne[0]!=c.top_k||source->ne[1]<1||source->ne[1]>4096||source->ne[2]!=1||source->ne[3]!=1||!source->data){c.error=1;return;}
  const auto *owner=source->view_src?source->view_src:source;
  if(!owner->buffer||!ggml_backend_buffer_is_host(owner->buffer)){c.error=2;return;}
  const int rows=static_cast<int>(source->ne[1]),slots=c.top_k;
  if((rows!=c.submitted&&rows!=1)||c.used>=c.records.size()||c.ids_used+size_t(rows)*slots>c.ids.size()){c.error=3;return;}
  auto &r=c.records[c.used++];r={c.forward,layer,stage,rows,slots,c.position,c.submitted,c.decode,c.ids_used};
  for(int row=0;row<rows;++row)for(int slot=0;slot<slots;++slot){int32_t v=-1;std::memcpy(&v,static_cast<const char*>(source->data)+row*source->nb[1]+slot*source->nb[0],sizeof(v));
   if(v<0||v>=c.experts){c.error=4;return;}c.ids[c.ids_used++]=v;}
  QueryPerformanceCounter(&last);c.hook_ticks+=last.QuadPart-first.QuadPart;
 }
 json output() const {
  require(error==0,"CPU consumer tap failed with code "+std::to_string(error));
  std::map<std::pair<int,int>,std::vector<const TapRecord*>> groups;
  for(size_t i=0;i<used;++i)groups[{records[i].forward,records[i].layer}].push_back(&records[i]);
  json result=json::array();const char * names[]={"up","gate","down"};
  for(const auto &item:groups){const auto &v=item.second;const auto &r=*v.front();json stages=json::array();std::array<bool,3> seen{};
   for(const auto *p:v){require(!seen[p->stage],"duplicate CPU expert stage in forward/layer");seen[p->stage]=true;stages.push_back(names[p->stage]);
    require(p->rows==r.rows&&p->slots==r.slots,"expert stage route shapes differ");
    require(std::equal(ids.begin()+r.offset,ids.begin()+r.offset+size_t(r.rows)*r.slots,ids.begin()+p->offset),"actual expert-stage IDs differ");}
   json positions=json::array(),selected=json::array();for(int row=0;row<r.rows;++row){positions.push_back(r.rows==r.submitted?r.position+row:r.position+r.submitted-1);json one=json::array();
    for(int k=0;k<r.slots;++k)one.push_back(ids[r.offset+size_t(row)*r.slots+k]);selected.push_back(std::move(one));}
   result.push_back(json{{"forward_index",r.forward},{"phase",r.decode?"decode":"prefill"},{"layer",r.layer},{"absolute_positions",positions},{"expert_ids",selected},
    {"row_mapping",r.rows==r.submitted?"all_submitted_rows":"last_output_row_after_normal_pruning"},{"consumer_backend","CPU"},{"actual_stages_observed",stages},
    {"all_three_stages_observed",seen[0]&&seen[1]&&seen[2]},{"observed_stage_routes_equal",true},{"stage_count",v.size()}});
  }return result;
 }
};
struct TapBinding {
 using Setter=void(*)(simulator_moe_tap_callback,void*);Setter setter=nullptr;
 explicit TapBinding(Capture &capture,bool enable){auto module=GetModuleHandleW(L"ggml-cpu.dll");require(module,"actual CPU DLL not loaded");
  setter=reinterpret_cast<Setter>(GetProcAddress(module,"simulator_moe_tap_set"));require(!enable||setter,"requested private CPU consumer tap ABI missing");if(setter)setter(enable?Capture::callback:nullptr,enable?&capture:nullptr);}
 ~TapBinding(){if(setter)setter(nullptr,nullptr);}
};
