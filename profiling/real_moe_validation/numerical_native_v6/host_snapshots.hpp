// Persist the already copied host logits after the request's end tick.
#pragma once
static json diagnostic_environment(){
 const char *keys[]={"CUBLAS_WORKSPACE_CONFIG","CUDA_DEVICE_MAX_CONNECTIONS","CUDA_LAUNCH_BLOCKING","CUDA_MODULE_LOADING","CUDA_VISIBLE_DEVICES","NVIDIA_TF32_OVERRIDE","GGML_CUDA_DISABLE_GRAPHS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"};
 json out=json::object();for(auto key:keys){const auto *value=std::getenv(key);out[key]=value?json(value):json(nullptr);}return out;
}
static json save_host_logits(const std::filesystem::path &path,const float *values,size_t rows,size_t columns){
 static_assert(sizeof(float)==4&&std::numeric_limits<float>::is_iec559,"IEEE754 F32 host required");
 const uint32_t endian=1;require(*reinterpret_cast<const uint8_t*>(&endian)==1,"little-endian host required");
 require(path.is_absolute()&&std::filesystem::is_directory(path.parent_path()),"explicit existing artifact directory required");
 require(rows>0&&columns>0&&rows*columns<=20000000,"bounded host snapshot required");
 for(size_t i=0;i<rows*columns;++i)require(std::isfinite(values[i]),"nonfinite full logits refused");
 const size_t bytes=rows*columns*sizeof(float);const auto start=tick();
 HANDLE file=CreateFileW(path.c_str(),GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);require(file!=INVALID_HANDLE_VALUE,"exclusive host snapshot creation failed");
 size_t offset=0;bool ok=true;while(offset<bytes){DWORD done=0;const auto count=static_cast<DWORD>(std::min<size_t>(bytes-offset,1<<20));
  if(!WriteFile(file,reinterpret_cast<const uint8_t*>(values)+offset,count,&done,nullptr)||done!=count){ok=false;break;}offset+=done;}
 const bool flushed=ok&&FlushFileBuffers(file)!=0;CloseHandle(file);require(flushed,"host snapshot write/flush failed");
 return json{{"path",path.u8string()},{"sha256",tap_sha256(values,bytes)},{"bytes",bytes},{"rows",rows},{"columns",columns},{"format","little-endian IEEE754 F32 row-major"},{"finite",true},{"existing_host_copy_used",true},{"new_GPU_reads",0},{"export_wall_ns",elapsed(start,tick())},{"export_in_target_interval",false}};
}
struct ActualBatchLedger {
 struct Forward {int index=0;size_t offset=0,count=0;};
 std::vector<std::array<int32_t,5>> values;std::vector<Forward> forwards;size_t used=0;
 ActualBatchLedger(size_t rows,size_t count):values(rows),forwards(){forwards.reserve(count);}
 void record(const llama_batch &b,int forward){
  require(b.n_tokens>0&&used+size_t(b.n_tokens)<=values.size()&&forwards.size()<forwards.capacity(),"actual batch ledger envelope exceeded");
  const size_t offset=used;for(int i=0;i<b.n_tokens;++i){require(b.n_seq_id[i]==1&&b.seq_id[i],"single-sequence actual batch required");values[used++]={b.token[i],b.pos[i],b.n_seq_id[i],b.seq_id[i][0],int32_t(b.logits[i])};}
  forwards.push_back({forward,offset,size_t(b.n_tokens)});
 }
 json output()const{
  json out=json::array();for(const auto &f:forwards){json rows=json::array();for(size_t i=f.offset;i<f.offset+f.count;++i){const auto &x=values[i];rows.push_back(json{{"token",x[0]},{"position",x[1]},{"n_seq_id",x[2]},{"seq_id",x[3]},{"logits",bool(x[4])}});}
   out.push_back(json{{"forward_index",f.index},{"rows",rows},{"actual_input_rows_sha256",tap_sha256(values.data()+f.offset,f.count*sizeof(values[0]))},{"hash_format","five little-endian int32 fields: token,position,n_seq_id,seq_id,logits"}});}return out;
 }
};
