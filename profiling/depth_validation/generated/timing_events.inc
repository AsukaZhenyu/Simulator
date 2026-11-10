static void cuda_ok(cudaError_t status,const char *operation) {
    require(status==cudaSuccess,std::string(operation)+": "+cudaGetErrorString(status));
}
struct TimingEvents {
    cudaEvent_t first=nullptr,last=nullptr;ggml_backend_event first_wrapper{},last_wrapper{};
    explicit TimingEvents(ggml_backend_t backend,int device) {
        cuda_ok(cudaSetDevice(device),"event cudaSetDevice");
        cuda_ok(cudaEventCreateWithFlags(&first,cudaEventDefault),"first timing event create");
        const auto status=cudaEventCreateWithFlags(&last,cudaEventDefault);
        if(status!=cudaSuccess){cudaEventDestroy(first);first=nullptr;cuda_ok(status,"last timing event create");}
        first_wrapper.device=last_wrapper.device=ggml_backend_get_device(backend);
        first_wrapper.context=first;last_wrapper.context=last;
    }
    ~TimingEvents(){if(first)cudaEventDestroy(first);if(last)cudaEventDestroy(last);}
    void begin(ggml_backend_t backend){ggml_backend_event_record(&first_wrapper,backend);}
    void end(ggml_backend_t backend){ggml_backend_event_record(&last_wrapper,backend);}
    float elapsed_ms(){float value=0;cuda_ok(cudaEventElapsedTime(&value,first,last),"timing event elapsed");require(std::isfinite(value)&&value>=0,"invalid CUDA event duration");return value;}
};
