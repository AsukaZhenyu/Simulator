// Profiling-only input transport. No tensor values are supplied to Modeling.
#pragma once
#include <windows.h>
#include <bcrypt.h>
#include <filesystem>
#include <iomanip>
#include <sstream>

namespace depth_assets {
static size_t buffers_verified=0,total_bytes_verified=0;
static std::set<std::string> files_verified;
static std::string hash_bytes(const uint8_t *data,size_t size) {
    BCRYPT_ALG_HANDLE algorithm=nullptr;BCRYPT_HASH_HANDLE hash=nullptr;
    require(BCryptOpenAlgorithmProvider(&algorithm,BCRYPT_SHA256_ALGORITHM,nullptr,0)>=0,"SHA256 provider failed");
    DWORD object_size=0,used=0;
    require(BCryptGetProperty(algorithm,BCRYPT_OBJECT_LENGTH,reinterpret_cast<PUCHAR>(&object_size),sizeof(object_size),&used,0)>=0,"SHA256 object size failed");
    std::vector<uint8_t> object(object_size),digest(32);
    require(BCryptCreateHash(algorithm,&hash,object.data(),object_size,nullptr,0,0)>=0,"SHA256 create failed");
    for(size_t offset=0;offset<size;) {
        const auto n=static_cast<ULONG>(std::min<size_t>(size-offset,1u<<20));
        require(BCryptHashData(hash,const_cast<PUCHAR>(data+offset),n,0)>=0,"SHA256 data failed");offset+=n;
    }
    require(BCryptFinishHash(hash,digest.data(),32,0)>=0,"SHA256 finish failed");
    BCryptDestroyHash(hash);BCryptCloseAlgorithmProvider(algorithm,0);
    std::ostringstream result;for(auto b:digest)result<<std::hex<<std::setfill('0')<<std::setw(2)<<int(b);
    return result.str();
}
static void verify_file(const json &asset) {
    const auto path=str(asset,"path"),expected=str(asset,"sha256");
    const auto key=path+":"+expected;if(files_verified.count(key))return;
    const auto actual_size=std::filesystem::file_size(std::filesystem::u8path(path));
    require(actual_size==asset.at("bytes").get<uint64_t>(),"initial binary size mismatch");
    // Stream the whole-file digest without retaining a second complete weight copy.
    BCRYPT_ALG_HANDLE alg=nullptr;BCRYPT_HASH_HANDLE hash=nullptr;DWORD object_size=0,used=0;
    require(BCryptOpenAlgorithmProvider(&alg,BCRYPT_SHA256_ALGORITHM,nullptr,0)>=0,"asset SHA256 open failed");
    require(BCryptGetProperty(alg,BCRYPT_OBJECT_LENGTH,reinterpret_cast<PUCHAR>(&object_size),sizeof(object_size),&used,0)>=0,"asset SHA256 property failed");
    std::vector<uint8_t> object(object_size),chunk(1u<<20),digest(32);
    require(BCryptCreateHash(alg,&hash,object.data(),object_size,nullptr,0,0)>=0,"asset SHA256 create failed");
    std::ifstream input(std::filesystem::u8path(path),std::ios::binary);require(input.good(),"cannot open explicit initial binary");
    size_t count=0;while(input){input.read(reinterpret_cast<char*>(chunk.data()),chunk.size());const auto n=input.gcount();if(n>0){require(BCryptHashData(hash,chunk.data(),static_cast<ULONG>(n),0)>=0,"asset SHA256 read failed");count+=static_cast<size_t>(n);}}
    require(count==actual_size&&BCryptFinishHash(hash,digest.data(),32,0)>=0,"asset SHA256 finish failed");
    BCryptDestroyHash(hash);BCryptCloseAlgorithmProvider(alg,0);
    std::ostringstream observed;for(auto b:digest)observed<<std::hex<<std::setfill('0')<<std::setw(2)<<int(b);
    require(observed.str()==expected,"initial binary SHA256 mismatch");files_verified.insert(key);
}
static std::vector<uint8_t> load(const json &program,const std::string &id,size_t bytes,const std::string &dtype) {
    require(program.at("initial_values").empty(),"binary and text initial values cannot be mixed");
    const auto &assets=program.at("initial_assets");require(assets.at("schema")=="dense-initial-assets-1","unknown initial binary schema");
    const auto &file=assets.at("file");verify_file(file);
    const auto &entry=assets.at("tensors").at(id);
    const auto offset=entry.at("offset_bytes").get<uint64_t>();
    require(offset%4==0&&entry.at("bytes").get<uint64_t>()==bytes&&entry.at("elements").get<uint64_t>()==bytes/4&&entry.at("dtype")==dtype,"initial binary tensor geometry mismatch");
    require(offset<=file.at("bytes").get<uint64_t>()&&bytes<=file.at("bytes").get<uint64_t>()-offset,"initial binary range outside file");
    std::ifstream input(std::filesystem::u8path(str(file,"path")),std::ios::binary);require(input.good(),"cannot open initial binary range");
    input.seekg(static_cast<std::streamoff>(offset));std::vector<uint8_t> result(bytes);input.read(reinterpret_cast<char*>(result.data()),bytes);
    require(static_cast<size_t>(input.gcount())==bytes&&hash_bytes(result.data(),bytes)==str(entry,"sha256"),"initial binary tensor bytes differ");
    ++buffers_verified;total_bytes_verified+=bytes;return result;
}
}
