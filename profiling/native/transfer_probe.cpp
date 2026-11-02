// Isolated single-copy latency probe. Allocation and validation are outside timing.
#include <cuda_runtime_api.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cudaError_t status, const char * what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

struct Resources {
    void * device = nullptr;
    void * pinned = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;
    ~Resources() {
        if (stream) cudaStreamSynchronize(stream);
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
        if (stream) cudaStreamDestroy(stream);
        if (device) cudaFree(device);
        if (pinned) cudaFreeHost(pinned);
    }
};

static uint64_t unsigned_integer(const std::string & value) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
        throw std::runtime_error("invalid nonnegative integer: " + value);
    }
    return std::stoull(value);
}

static int bounded_integer(const std::string & value) {
    uint64_t parsed = unsigned_integer(value);
    if (parsed > 10000) throw std::runtime_error("integer option exceeds 10000");
    return static_cast<int>(parsed);
}

int main(int argc, char ** argv) {
    try {
        uint64_t size = 0;
        int warmup = 5, samples = 20, device_index = 0;
        std::string direction, host_memory;
        for (int i = 1; i < argc; ++i) {
            std::string key = argv[i];
            if (key == "--help") {
                std::cout << "cuda-transfer-probe --bytes N --direction H2D|D2H "
                             "--host-memory pinned|pageable [--warmup 5 --samples 20 --device 0]\n";
                return 0;
            }
            if (i + 1 == argc) throw std::runtime_error("missing value for " + key);
            std::string value = argv[++i];
            if (key == "--bytes") size = unsigned_integer(value);
            else if (key == "--direction") direction = value;
            else if (key == "--host-memory") host_memory = value;
            else if (key == "--warmup") warmup = bounded_integer(value);
            else if (key == "--samples") samples = bounded_integer(value);
            else if (key == "--device") device_index = bounded_integer(value);
            else throw std::runtime_error("unknown argument " + key);
        }
        if (!size || size > (1ULL << 30)) throw std::runtime_error("--bytes must be in 1..1073741824");
        if (warmup < 0 || samples < 3 || samples > 10000) throw std::runtime_error("invalid warmup/sample count");
        if (direction != "H2D" && direction != "D2H") throw std::runtime_error("invalid direction");
        if (host_memory != "pinned" && host_memory != "pageable") throw std::runtime_error("invalid host memory");

        check(cudaSetDevice(device_index), "cudaSetDevice");
        cudaDeviceProp props{};
        check(cudaGetDeviceProperties(&props, device_index), "cudaGetDeviceProperties");
        int runtime_version = 0, driver_version = 0;
        check(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
        check(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
        Resources r;
        check(cudaMalloc(&r.device, size), "cudaMalloc");
        check(cudaStreamCreateWithFlags(&r.stream, cudaStreamNonBlocking), "cudaStreamCreate");
        check(cudaEventCreate(&r.begin), "cudaEventCreate begin");
        check(cudaEventCreate(&r.end), "cudaEventCreate end");
        std::vector<unsigned char> pageable;
        void * host = nullptr;
        if (host_memory == "pinned") {
            check(cudaMallocHost(&r.pinned, size), "cudaMallocHost");
            host = r.pinned;
        } else {
            pageable.resize(size);
            host = pageable.data();
        }
        // All pages are touched before timing. Pageable staging inside CUDA remains included.
        std::memset(host, direction == "H2D" ? 0x5a : 0x00, size);
        check(cudaMemsetAsync(r.device, direction == "H2D" ? 0x00 : 0x5a, size, r.stream), "cudaMemsetAsync");
        check(cudaStreamSynchronize(r.stream), "initial synchronization");
        auto copy = [&]() {
            check(cudaMemcpyAsync(direction == "H2D" ? r.device : host,
                                  direction == "H2D" ? host : r.device,
                                  size, direction == "H2D" ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost,
                                  r.stream), "cudaMemcpyAsync");
        };
        for (int i = 0; i < warmup; ++i) {
            copy();
            check(cudaStreamSynchronize(r.stream), "warmup synchronization");
        }
        std::vector<int64_t> wall_ns, device_ns;
        for (int i = 0; i < samples; ++i) {
            auto start = std::chrono::steady_clock::now();
            check(cudaEventRecord(r.begin, r.stream), "cudaEventRecord begin");
            copy();
            check(cudaEventRecord(r.end, r.stream), "cudaEventRecord end");
            check(cudaEventSynchronize(r.end), "cudaEventSynchronize");
            auto stop = std::chrono::steady_clock::now();
            float elapsed_ms = 0;
            check(cudaEventElapsedTime(&elapsed_ms, r.begin, r.end), "cudaEventElapsedTime");
            wall_ns.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count());
            device_ns.push_back(static_cast<int64_t>(std::llround(elapsed_ms * 1e6)));
        }
        // Check every byte, including the untimed return copy needed by H2D validation.
        if (direction == "H2D") check(cudaMemcpy(host, r.device, size, cudaMemcpyDeviceToHost), "validation copy");
        const auto * bytes = static_cast<const unsigned char *>(host);
        for (uint64_t i = 0; i < size; ++i) {
            if (bytes[i] != 0x5a) throw std::runtime_error("copy validation failed at byte " + std::to_string(i));
        }
        auto array = [](const std::vector<int64_t> & data) {
            std::cout << '[';
            for (size_t i = 0; i < data.size(); ++i) std::cout << (i ? "," : "") << data[i];
            std::cout << ']';
        };
        std::cout << "{\"kind\":\"transfer\",\"status\":\"measured\",\"direction\":\"" << direction
                  << "\",\"host_memory\":\"" << host_memory << "\",\"size_bytes\":" << size
                  << ",\"device\":\"" << props.name << "\",\"device_index\":" << device_index
                  << ",\"cuda_runtime_version\":" << runtime_version << ",\"cuda_driver_version\":" << driver_version
                  << ",\"warmup\":" << warmup << ",\"samples\":" << samples
                  << ",\"timing_scope\":\"single_transfer_wall\",\"ops_per_sample\":1"
                  << ",\"timer\":\"steady_clock\",\"device_timer\":\"cuda_events\""
                  << ",\"synchronization\":\"cudaEventSynchronize after each isolated copy\""
                  << ",\"cache_policy\":\"reused buffers; host pages touched before warmup\""
                  << ",\"copy_api\":\"cudaMemcpyAsync\",\"validation\":\"all_bytes_exact\""
                  << ",\"includes\":[\"copy enqueue\",\"event bookkeeping\",\"completion wait\",\"implicit pageable staging if applicable\"]"
                  << ",\"excludes\":[\"allocation\",\"explicit host staging\",\"initialization\",\"validation\"]"
                  << ",\"device_interval_note\":\"CUDA event interval may include stream idle time during pageable staging; it is not a pure PCIe link measurement\""
                  << ",\"wall_ns\":";
        array(wall_ns);
        std::cout << ",\"device_ns\":";
        array(device_ns);
        std::cout << "}\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "cuda-transfer-probe: " << ex.what() << '\n';
        return 1;
    }
}
