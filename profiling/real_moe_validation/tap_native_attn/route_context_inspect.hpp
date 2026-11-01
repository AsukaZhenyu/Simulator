// Read-only pinned internal ABI. The build creates a header-identical context
// copy with two inline getters only: no class fields or runtime code change.
#pragma once
#include "private_llama_context.h"
static json tap_graph_summary(llama_context *ctx){
 auto *result=ctx->simulator_tap_graph_result();auto *scheduler=ctx->simulator_tap_scheduler();
 require(result&&scheduler,"actual graph metadata unavailable");auto *g=result->get_gf();require(g,"actual graph missing");
 json nodes=json::array(),experts=json::array();
 for(int i=0;i<ggml_graph_n_nodes(g);++i){auto *n=ggml_graph_node(g,i);auto *backend=ggml_backend_sched_get_tensor_backend(scheduler,n);
  json shape=json::array(),strides=json::array(),sources=json::array(),params=json::array();
  for(int k=0;k<4;++k){shape.push_back(n->ne[k]);strides.push_back(n->nb[k]);}
  for(auto *s:n->src)sources.push_back(s?json(s->name):json(nullptr));for(int32_t p:n->op_params)params.push_back(p);
  const std::string device=backend?ggml_backend_name(backend):"unassigned_metadata_node";
  nodes.push_back(json{{"name",n->name},{"op",static_cast<int>(n->op)},{"type",static_cast<int>(n->type)},{"flags",n->flags},
   {"ne",shape},{"nb",strides},{"src",sources},{"op_params",params},{"backend",device}});
  if(n->op==GGML_OP_MUL_MAT_ID){auto *w=n->src[0];auto *buffer=w&&w->view_src?w->view_src->buffer:w?w->buffer:nullptr;
   experts.push_back(json{{"name",n->name},{"backend",device},{"weight_buffer",buffer?ggml_backend_buffer_name(buffer):"unknown"},
    {"weight_type",w?json(ggml_type_name(w->type)):json(nullptr)}});}
 }
 const auto canonical=nodes.dump();return json{{"node_count",nodes.size()},{"scheduler_splits",ggml_backend_sched_get_n_splits(scheduler)},
  {"scheduler_copies",ggml_backend_sched_get_n_copies(scheduler)},{"graph_partition_fingerprint_sha256",tap_sha256(canonical.data(),canonical.size())},
  {"fingerprint_scope","actual graph op/type/flags/ne/nb/src names/op_params/backend; excludes pointers, timing and private CUDA fusion decisions"},{"expert_nodes",experts}};
}
static json tap_logit_summary(const float *values,int count,int greedy){
 require(count>0,"vocabulary size missing");std::vector<std::pair<float,int>> top;top.reserve(count);
 for(int i=0;i<count;++i){require(std::isfinite(values[i]),"nonfinite observed logit");top.emplace_back(values[i],i);}
 const int k=std::min(8,count);std::partial_sort(top.begin(),top.begin()+k,top.end(),[](const auto&a,const auto&b){return a.first>b.first||(a.first==b.first&&a.second<b.second);});
 json selected=json::array();for(int i=0;i<k;++i)selected.push_back(json{{"token",top[i].second},{"logit",top[i].first}});
 return json{{"greedy_token",greedy},{"full_f32_logits_sha256",tap_sha256(values,size_t(count)*sizeof(float))},{"vocabulary_size",count},
  {"top_logits",selected},{"top1_top2_margin",k>1?json(double(top[0].first)-top[1].first):json(nullptr)}};
}
